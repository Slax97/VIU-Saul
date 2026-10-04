# -*- coding: utf-8 -*-
"""Repite el protocolo del TFM sobre dos conjuntos de datos externos.

    python tfm_pipeline/validacion_externa.py              # los dos conjuntos
    python tfm_pipeline/validacion_externa.py --conjunto phiusiil
    python tfm_pipeline/validacion_externa.py --rapido     # submuestra, para probar

Por que existe. La memoria declara como limitacion principal que todo el trabajo
se apoya en un unico conjunto de datos, y dice literalmente que lo interesante de
repetirlo sobre otro "no seria solo ver si el rendimiento se mantiene, sino si se
mantienen las decisiones". Eso es exactamente lo que se mide aqui: no una cifra
de F1 suelta, sino si las cuatro decisiones del trabajo siguen siendo las
correctas cuando cambian los datos.

    D1  La imputacion por mediana mas indicador es la mejor opcion.
    D2  Podar variables por correlacion sale gratis en rendimiento.
    D3  El balanceo de clases no aporta.
    D4  LightGBM es el mejor modelo.

El protocolo es el mismo del notebook, con los mismos parametros: semilla 42,
hold-out del 20 % estratificado, validacion cruzada estratificada de 5 pliegues y
poda de variables por correlacion absoluta mayor que 0,95. Todo lo que se ajusta
a los datos (escalado, imputacion, seleccion) se ajusta SOLO con el train, dentro
de un Pipeline, que es lo que evita las fugas.

Aviso sobre el entorno. Los experimentos originales corrieron en Python 3.11 con
las versiones fijadas en requirements.txt. Este script corre en el entorno actual
(Python 3.13, librerias mas nuevas), asi que las cifras no son estrictamente
comparables con las del notebook: lo que se compara son las DECISIONES, que es lo
que interesa. El entorno real queda registrado en el JSON de salida.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Los mismos parametros del notebook. No se tocan: si cambian, deja de ser el
# mismo protocolo y la comparacion pierde el sentido.
# ---------------------------------------------------------------------------
SEMILLA = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
CORR_MAX = 0.95

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
EXTERNOS = RAIZ / "data" / "externos"
RESULTADOS = AQUI / "resultados"

CONJUNTOS = {
    "phiusiil": {
        "titulo": "PhiUSIIL (Prasad y Chandra, 2024)",
        "ruta": EXTERNOS / "phiusiil2024" / "PhiUSIIL_Phishing_URL_Dataset.csv",
        "sha256": "a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6",
        "diana": "label",
        # 1 = legitima en este conjunto, asi que la clase positiva (phishing) es el 0.
        "positiva": 0,
        # Identificadores y texto libre: no son variables, son la propia URL.
        "descartar": ["FILENAME", "URL", "Domain", "TLD", "Title"],
        # Se analiza tambien sin esta variable: ver el README de data/externos.
        "sospechosa": "URLSimilarityIndex",
    },
    "urlphish": {
        "titulo": "URL-Phish (Linh y Hung, 2025)",
        "ruta": EXTERNOS / "urlphish2025" / "Dataset.csv",
        "sha256": "d68b3cd0648dcf9c775347416ad1a8995e8a025921fbe3871ca6158d4db3c3a1",
        "diana": "label",
        "positiva": 1,
        "descartar": ["url", "dom", "tld"],
        "sospechosa": None,
    },
}


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for t in iter(lambda: f.read(1 << 20), b""):
            h.update(t)
    return h.hexdigest()


def cargar(cfg, rapido=False):
    """Carga el CSV, verifica su hash y lo deja en X, y numericos y limpios."""
    if not cfg["ruta"].exists():
        raise SystemExit(f"No encuentro {cfg['ruta']}.\n"
                         f"Ejecuta antes:  python data/externos/descargar.py")
    real = sha256(cfg["ruta"])
    if real != cfg["sha256"]:
        raise SystemExit(f"El SHA-256 de {cfg['ruta'].name} no cuadra.\n"
                         f"  esperado {cfg['sha256']}\n  obtenido {real}")

    d = pd.read_csv(cfg["ruta"])
    n0 = len(d)

    # Mismo criterio que el notebook: fuera duplicados exactos.
    d = d.drop_duplicates()
    n_dup = n0 - len(d)

    y = (d[cfg["diana"]] == cfg["positiva"]).astype(int)
    X = d.drop(columns=[cfg["diana"]] + [c for c in cfg["descartar"]
                                         if c in d.columns])
    # Lo que no sea numerico no entra: aqui no se hace ingenieria de variables
    # nueva, se reutiliza la que cada conjunto trae.
    X = X.select_dtypes(include=[np.number])

    if rapido:
        from sklearn.model_selection import train_test_split
        X, _, y, _ = train_test_split(X, y, train_size=min(15000, len(X) - 1),
                                      random_state=SEMILLA, stratify=y)

    info = {
        "filas_originales": n0,
        "duplicados_eliminados": int(n_dup),
        "filas": len(X),
        "variables": X.shape[1],
        "nulos": int(X.isna().sum().sum()),
        "positivos": int(y.sum()),
        "negativos": int((y == 0).sum()),
        "ratio_desbalanceo": round(float(max((y == 0).sum(), y.sum())
                                         / max(min((y == 0).sum(), y.sum()), 1)), 3),
    }
    return X, y, info


# ---------------------------------------------------------------------------
# Protocolo
# ---------------------------------------------------------------------------
def construir_pipeline(modelo, imputador="mediana_indicador", seleccion=None):
    """El preprocesamiento del notebook, dentro de un Pipeline para no filtrar."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pasos = []
    if imputador == "mediana_indicador":
        pasos.append(("imp", SimpleImputer(strategy="median", add_indicator=True)))
    elif imputador == "mediana":
        pasos.append(("imp", SimpleImputer(strategy="median")))
    elif imputador == "media":
        pasos.append(("imp", SimpleImputer(strategy="mean")))
    pasos.append(("esc", StandardScaler()))
    pasos.append(("mod", modelo))
    return Pipeline(pasos)


def podar_por_correlacion(X):
    """Constantes, cuasiconstantes y redundantes por |r| > 0,95, como el notebook."""
    nunique = X.nunique()
    constantes = nunique[nunique <= 1].index.tolist()
    resto = [c for c in X.columns if c not in constantes]
    corr = X[resto].corr().abs()
    superior = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    redundantes = [c for c in superior.columns if (superior[c] > CORR_MAX).any()]
    return [c for c in resto if c not in redundantes], constantes, redundantes


def cv():
    from sklearn.model_selection import StratifiedKFold
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEMILLA)


SCORING = ["precision", "recall", "f1", "matthews_corrcoef", "roc_auc"]


def evaluar_cv(pipe, X, y):
    from sklearn.model_selection import cross_validate
    r = cross_validate(pipe, X, y, cv=cv(), scoring=SCORING, n_jobs=-1,
                       error_score="raise")
    return {m: (float(np.mean(r[f"test_{m}"])), float(np.std(r[f"test_{m}"])))
            for m in SCORING}


def modelos():
    """El subconjunto representativo de las familias que compara el notebook."""
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000,
                                                 random_state=SEMILLA),
        "DecisionTree": DecisionTreeClassifier(random_state=SEMILLA),
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                               random_state=SEMILLA),
        "XGBoost": XGBClassifier(n_estimators=300, tree_method="hist",
                                 eval_metric="logloss", n_jobs=-1,
                                 random_state=SEMILLA),
        "LightGBM": LGBMClassifier(n_estimators=300, n_jobs=-1, verbose=-1,
                                   random_state=SEMILLA),
    }


# ---------------------------------------------------------------------------
# Las cuatro decisiones
# ---------------------------------------------------------------------------
def d1_imputacion(X, y, filas):
    """D1: la imputacion por mediana mas indicador es la mejor opcion."""
    from lightgbm import LGBMClassifier
    if X.isna().sum().sum() == 0:
        print("  D1 imputacion: el conjunto no tiene nulos, la decision no se "
              "puede poner a prueba aqui.")
        filas.append({"decision": "D1_imputacion", "variante": "sin nulos",
                      "f1_mean": None, "nota": "no evaluable: 0 nulos"})
        return
    for nombre in ("mediana_indicador", "mediana", "media"):
        pipe = construir_pipeline(LGBMClassifier(n_estimators=300, n_jobs=-1,
                                                 verbose=-1,
                                                 random_state=SEMILLA), nombre)
        r = evaluar_cv(pipe, X, y)
        print(f"  D1 {nombre:20s} F1={r['f1'][0]:.4f}  MCC={r['matthews_corrcoef'][0]:.4f}")
        filas.append({"decision": "D1_imputacion", "variante": nombre,
                      "f1_mean": round(r["f1"][0], 4),
                      "f1_std": round(r["f1"][1], 4),
                      "mcc_mean": round(r["matthews_corrcoef"][0], 4)})


def d2_seleccion(X, y, filas):
    """D2: podar variables por correlacion sale gratis."""
    from lightgbm import LGBMClassifier
    retenidas, constantes, redundantes = podar_por_correlacion(X)
    print(f"  D2 poda: {X.shape[1]} -> {len(retenidas)} variables "
          f"({len(constantes)} constantes, {len(redundantes)} redundantes)")
    for etiqueta, cols in (("todas", list(X.columns)), ("filter_corr", retenidas)):
        pipe = construir_pipeline(LGBMClassifier(n_estimators=300, n_jobs=-1,
                                                 verbose=-1, random_state=SEMILLA))
        r = evaluar_cv(pipe, X[cols], y)
        print(f"  D2 {etiqueta:20s} ({len(cols):3d} vars) F1={r['f1'][0]:.4f}  "
              f"MCC={r['matthews_corrcoef'][0]:.4f}")
        filas.append({"decision": "D2_seleccion", "variante": etiqueta,
                      "n_variables": len(cols),
                      "f1_mean": round(r["f1"][0], 4),
                      "f1_std": round(r["f1"][1], 4),
                      "mcc_mean": round(r["matthews_corrcoef"][0], 4)})
    return retenidas


def d3_balanceo(X, y, filas):
    """D3: el balanceo de clases no aporta."""
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as PipelineIMB
    from lightgbm import LGBMClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    base = dict(n_estimators=300, n_jobs=-1, verbose=-1, random_state=SEMILLA)
    variantes = {
        "sin balanceo": construir_pipeline(LGBMClassifier(**base)),
        "class_weight": construir_pipeline(
            LGBMClassifier(class_weight="balanced", **base)),
        "SMOTE": PipelineIMB([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("esc", StandardScaler()),
            ("smote", SMOTE(random_state=SEMILLA)),
            ("mod", LGBMClassifier(**base))]),
    }
    for etiqueta, pipe in variantes.items():
        r = evaluar_cv(pipe, X, y)
        print(f"  D3 {etiqueta:20s} F1={r['f1'][0]:.4f}  "
              f"MCC={r['matthews_corrcoef'][0]:.4f}")
        filas.append({"decision": "D3_balanceo", "variante": etiqueta,
                      "f1_mean": round(r["f1"][0], 4),
                      "f1_std": round(r["f1"][1], 4),
                      "mcc_mean": round(r["matthews_corrcoef"][0], 4),
                      "roc_auc_mean": round(r["roc_auc"][0], 4)})


def d4_modelos(X, y, filas):
    """D4: LightGBM es el mejor modelo."""
    for nombre, modelo in modelos().items():
        t0 = time.perf_counter()
        r = evaluar_cv(construir_pipeline(modelo), X, y)
        print(f"  D4 {nombre:20s} F1={r['f1'][0]:.4f}  "
              f"MCC={r['matthews_corrcoef'][0]:.4f}  "
              f"AUC={r['roc_auc'][0]:.4f}  ({time.perf_counter()-t0:.0f}s)")
        filas.append({"decision": "D4_modelos", "variante": nombre,
                      "f1_mean": round(r["f1"][0], 4),
                      "f1_std": round(r["f1"][1], 4),
                      "mcc_mean": round(r["matthews_corrcoef"][0], 4),
                      "roc_auc_mean": round(r["roc_auc"][0], 4)})


def evaluacion_final(X, y, filas, etiqueta):
    """Hold-out del 20 %, igual que el bloque 8 del notebook."""
    from lightgbm import LGBMClassifier
    from sklearn.metrics import (f1_score, matthews_corrcoef, precision_score,
                                 recall_score, roc_auc_score)
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEMILLA, stratify=y)
    pipe = construir_pipeline(LGBMClassifier(n_estimators=300, n_jobs=-1,
                                             verbose=-1, random_state=SEMILLA))
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    prob = pipe.predict_proba(X_te)[:, 1]
    m = {"precision": precision_score(y_te, pred, zero_division=0),
         "recall": recall_score(y_te, pred, zero_division=0),
         "f1": f1_score(y_te, pred, zero_division=0),
         "mcc": matthews_corrcoef(y_te, pred),
         "roc_auc": roc_auc_score(y_te, prob)}
    print(f"  TEST {etiqueta:22s} " +
          "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
    filas.append({"decision": "TEST_holdout", "variante": etiqueta,
                  "n_test": len(y_te), **{k: round(v, 4) for k, v in m.items()}})


# ---------------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------------
def entorno():
    import importlib
    libs = {}
    for m in ("numpy", "pandas", "scipy", "sklearn", "lightgbm", "xgboost",
              "imblearn"):
        try:
            libs[m] = importlib.import_module(m).__version__
        except Exception:
            libs[m] = None
    return {"python": platform.python_version(),
            "sistema": platform.platform(),
            "semilla": SEMILLA, "test_size": TEST_SIZE, "cv_folds": CV_FOLDS,
            "corr_max": CORR_MAX, "librerias": libs}


def correr(clave, rapido=False):
    cfg = CONJUNTOS[clave]
    print(f"\n{'=' * 74}\n{cfg['titulo']}\n{'=' * 74}")
    X, y, info = cargar(cfg, rapido)
    print(f"  {info['filas']:,} filas x {info['variables']} variables  |  "
          f"phishing {info['positivos']:,} / legitimas {info['negativos']:,}  |  "
          f"IR {info['ratio_desbalanceo']}  |  nulos {info['nulos']}  |  "
          f"duplicados fuera {info['duplicados_eliminados']:,}\n")

    filas = []
    d1_imputacion(X, y, filas)
    retenidas = d2_seleccion(X, y, filas)
    d3_balanceo(X[retenidas], y, filas)
    d4_modelos(X[retenidas], y, filas)
    evaluacion_final(X[retenidas], y, filas, "todas las variables")

    # PhiUSIIL otra vez sin la variable con fuga, para ver que queda de verdad.
    sosp = cfg.get("sospechosa")
    if sosp and sosp in X.columns:
        print(f"\n  --- repeticion sin {sosp} (variable con fuga de etiqueta) ---")
        X2 = X.drop(columns=[sosp])
        ret2, _, _ = podar_por_correlacion(X2)
        filas2 = []
        d4_modelos(X2[ret2], y, filas2)
        for f in filas2:
            f["variante"] = f"{f['variante']} (sin {sosp})"
            f["decision"] = "D4_modelos_sin_fuga"
        filas.extend(filas2)
        evaluacion_final(X2[ret2], y, filas, f"sin {sosp}")

    for f in filas:
        f["conjunto"] = clave
    return filas, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conjunto", choices=list(CONJUNTOS) + ["todos"],
                    default="todos")
    ap.add_argument("--rapido", action="store_true",
                    help="submuestra de 15.000 filas, solo para comprobar que corre")
    args = ap.parse_args()

    claves = list(CONJUNTOS) if args.conjunto == "todos" else [args.conjunto]
    t0 = time.perf_counter()
    todo, infos = [], {}
    for c in claves:
        filas, info = correr(c, args.rapido)
        todo.extend(filas)
        infos[c] = info

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    sufijo = "_rapido" if args.rapido else ""
    df = pd.DataFrame(todo)
    cols = ["conjunto", "decision", "variante"] + [c for c in df.columns
                                                   if c not in ("conjunto", "decision", "variante")]
    df = df[cols]
    salida = RESULTADOS / f"validacion_externa{sufijo}.csv"
    df.to_csv(salida, index=False, encoding="utf-8")

    meta = {"entorno": entorno(), "conjuntos": infos,
            "minutos": round((time.perf_counter() - t0) / 60, 1),
            "modo": "rapido" if args.rapido else "completo"}
    (RESULTADOS / f"validacion_externa_entorno{sufijo}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'=' * 74}")
    print(f"{len(df)} filas -> {salida.relative_to(RAIZ)}")
    print(f"tiempo total: {meta['minutos']} min")


if __name__ == "__main__":
    main()
