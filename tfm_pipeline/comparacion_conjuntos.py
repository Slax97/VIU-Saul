# -*- coding: utf-8 -*-
"""Tres analisis que solo se pueden hacer teniendo mas de un conjunto de datos.

    python tfm_pipeline/comparacion_conjuntos.py
    python tfm_pipeline/comparacion_conjuntos.py --rapido

Por que existe. Repetir el mismo experimento sobre tres conjuntos y poner las
tres cifras en una tabla no aporta gran cosa: son tres trabajos sueltos, no uno
que aprenda algo de tener tres. Aqui se hacen las tres cosas que si necesitan los
tres a la vez.

    1. AUDITORIA. Una bateria de comprobaciones que se le pasa a un conjunto
       ANTES de entrenar nada, para saber si el problema que plantea es real o
       si viene resuelto de fabrica. Sale de haber encontrado a mano la fuga de
       PhiUSIIL; aqui se sistematiza y se le aplica a los tres, incluido el
       propio. Eso ultimo importa: despues de acusar a un conjunto ajeno de
       tener fuga, hay que poder demostrar que el de uno no la tiene.

    2. TRANSFERENCIA CRUZADA. Entrenar en un conjunto y evaluar en otro, las
       nueve combinaciones, sobre las variables lexicas que los tres comparten.
       Es la prueba dura de generalizacion, la que \\citet{mia2024features}
       senalan como ausente en la literatura. La diagonal dice lo bien que va
       cada uno en su casa; lo de fuera de la diagonal dice cuanto de eso era
       del problema y cuanto del conjunto.

    3. CURVAS DE APRENDIZAJE. Cuantos ejemplos necesita cada conjunto para
       saturar. Un conjunto que con dos mil filas ya da lo mismo que con
       doscientas mil no es un conjunto grande: es un problema facil.

Protocolo, el mismo del trabajo: semilla 42, hold-out del 20 % estratificado,
validacion cruzada estratificada de cinco pliegues y todo el preprocesamiento
ajustado solo con el train, dentro de un Pipeline.
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

SEMILLA = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
RESULTADOS = AQUI / "resultados"
FIGURAS = AQUI / "figuras"

# Okabe-Ito, la paleta del resto del trabajo.
COLOR = {"vrbancic": "#0072B2", "phiusiil": "#E69F00", "urlphish": "#009E73"}
NOMBRE = {"vrbancic": "Vrbančič (2020)",
          "phiusiil": "PhiUSIIL (2024)",
          "urlphish": "URL-Phish (2025)"}

def _dataset_original():
    """El conjunto original cambia de sitio segun como este montado el proyecto.

    En el repositorio vive en `data/raw/`, pero la entrega lo copia dentro de
    `tfm_pipeline/datos/` para que el notebook lo encuentre sin configurar nada.
    En vez de fijar una ruta se prueban las dos, que es lo mismo que hacen los
    scripts de la defensa.
    """
    for cand in (RAIZ / "data" / "raw" / "dataset_full.csv",
                 AQUI / "datos" / "dataset_full.csv",
                 RAIZ / "tfm_pipeline" / "datos" / "dataset_full.csv"):
        if cand.is_file():
            return cand
    return RAIZ / "data" / "raw" / "dataset_full.csv"   # para el mensaje de error


CONJUNTOS = {
    "vrbancic": {
        "ruta": _dataset_original(),
        "diana": "phishing", "positiva": 1,
        "centinela": -1,          # el `-1` es ausencia, como documenta el trabajo
        "descartar": [],
    },
    "phiusiil": {
        "ruta": RAIZ / "data" / "externos" / "phiusiil2024"
                / "PhiUSIIL_Phishing_URL_Dataset.csv",
        "diana": "label", "positiva": 0,   # aqui 1 = legitima
        "centinela": None,
        "descartar": ["FILENAME", "URL", "Domain", "TLD", "Title"],
    },
    "urlphish": {
        "ruta": RAIZ / "data" / "externos" / "urlphish2025" / "Dataset.csv",
        "diana": "label", "positiva": 1,
        "centinela": None,
        "descartar": ["url", "dom", "tld"],
    },
}

# Variables que los tres calculan y que significan lo mismo. Se alinean por
# significado y no por nombre, porque Vrbancic no publica la URL cruda y sus
# variables vienen ya calculadas.
#
# Queda FUERA a proposito el par tls_ssl_certificate / IsHTTPS / is_https: el
# primero dice si el certificado TLS es valido y los otros dos solo si el
# esquema de la URL es https, que no es lo mismo. Meterlo seria justo el error
# que este experimento pretende medir.
COMUNES = {
    "long_url":        ("length_url",           "URLLength",           "url_len"),
    "long_dominio":    ("domain_length",        "DomainLength",        "dom_len"),
    "dominio_es_ip":   ("domain_in_ip",         "IsDomainIP",          "is_ip"),
    "n_interrogacion": ("qty_questionmark_url", "NoOfQMarkInURL",      "qm_cnt"),
    "n_igual":         ("qty_equal_url",        "NoOfEqualsInURL",     "eq_cnt"),
    "n_ampersand":     ("qty_and_url",          "NoOfAmpersandInURL",  "amp_cnt"),
}
ORDEN = ["vrbancic", "phiusiil", "urlphish"]


# ---------------------------------------------------------------------------
def cargar(clave, rapido=False, solo_comunes=False):
    """Devuelve X, y ya limpios. Con solo_comunes=True, en el espacio compartido."""
    cfg = CONJUNTOS[clave]
    if not cfg["ruta"].exists():
        raise SystemExit(f"Falta {cfg['ruta']}. Ejecuta antes "
                         f"python data/externos/descargar.py")
    d = pd.read_csv(cfg["ruta"]).drop_duplicates()
    y = (d[cfg["diana"]] == cfg["positiva"]).astype(int)

    if solo_comunes:
        i = ORDEN.index(clave)
        X = pd.DataFrame({nom: d[cols[i]] for nom, cols in COMUNES.items()})
    else:
        X = d.drop(columns=[cfg["diana"]]
                   + [c for c in cfg["descartar"] if c in d.columns])
        X = X.select_dtypes(include=[np.number])

    if cfg["centinela"] is not None:
        X = X.replace(cfg["centinela"], np.nan)

    if rapido:
        from sklearn.model_selection import train_test_split
        X, _, y, _ = train_test_split(X, y, train_size=min(8000, len(X) - 1),
                                      random_state=SEMILLA, stratify=y)
    return X.reset_index(drop=True), y.reset_index(drop=True)


def pipeline(modelo):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("esc", StandardScaler()),
                     ("mod", modelo)])


def lgbm(**kw):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=300, n_jobs=-1, verbose=-1,
                          random_state=SEMILLA, **kw)


# ---------------------------------------------------------------------------
# 1. AUDITORIA: ¿el conjunto plantea un problema real?
# ---------------------------------------------------------------------------
def separabilidad_univariante(X, y):
    """Mejor accuracy alcanzable con UN corte en UNA sola variable."""
    mejor = ("", 0.0)
    for c in X.columns:
        v = X[c]
        if v.nunique(dropna=True) < 2:
            continue
        cortes = np.unique(np.nanquantile(v.dropna(), np.linspace(0.01, 0.99, 50)))
        for t in cortes:
            a = max(((v >= t) == (y == 1)).mean(), ((v < t) == (y == 1)).mean())
            if a > mejor[1]:
                mejor = (c, float(a))
    return mejor


def variables_degeneradas(X, y):
    """Variables constantes dentro de una clase: la senal de fuga mas clara."""
    fuera = []
    for c in X.columns:
        for clase in (0, 1):
            v = X.loc[y == clase, c].dropna()
            if len(v) > 100 and v.nunique() == 1:
                otra = X.loc[y != clase, c].dropna()
                # solo cuenta si en la otra clase no es casi siempre ese valor
                if len(otra) and (otra == v.iloc[0]).mean() < 0.5:
                    fuera.append((c, clase, float(v.iloc[0])))
    return fuera


def auditar(clave, X, y, filas):
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.tree import DecisionTreeClassifier

    base = float(max((y == 1).mean(), (y == 0).mean()))
    var, acc = separabilidad_univariante(X, y)
    degen = variables_degeneradas(X, y)
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEMILLA)

    prof = {}
    for d in (1, 2, 3):
        m = pipeline(DecisionTreeClassifier(max_depth=d, random_state=SEMILLA))
        prof[d] = float(cross_val_score(m, X, y, cv=cv, scoring="f1",
                                        n_jobs=-1).mean())

    print(f"  clase mayoritaria           {base*100:6.2f} %")
    print(f"  mejor variable sola         {acc*100:6.2f} %   ({var})")
    print(f"  ventaja sobre la mayoritaria{ (acc-base)*100:+6.2f} puntos")
    print(f"  variables degeneradas       {len(degen)}"
          + (f"   {[d[0] for d in degen][:3]}" if degen else ""))
    print(f"  F1 de un arbol de prof. 1/2/3: "
          f"{prof[1]:.4f} / {prof[2]:.4f} / {prof[3]:.4f}")

    filas.append({
        "conjunto": clave, "filas": len(X), "variables": X.shape[1],
        "clase_mayoritaria": round(base, 4),
        "mejor_variable_sola": var,
        "acc_una_variable": round(acc, 4),
        "ventaja_puntos": round((acc - base) * 100, 2),
        "n_degeneradas": len(degen),
        "degeneradas": "; ".join(d[0] for d in degen) or "",
        "f1_arbol_prof1": round(prof[1], 4),
        "f1_arbol_prof2": round(prof[2], 4),
        "f1_arbol_prof3": round(prof[3], 4),
    })


# ---------------------------------------------------------------------------
# 2. TRANSFERENCIA CRUZADA
# ---------------------------------------------------------------------------
def transferencia(datos, filas):
    from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
    from sklearn.model_selection import train_test_split

    partes = {}
    for k, (X, y) in datos.items():
        partes[k] = train_test_split(X, y, test_size=TEST_SIZE,
                                     random_state=SEMILLA, stratify=y)

    print(f"  espacio comun: {len(COMUNES)} variables "
          f"({', '.join(COMUNES)})")
    for origen in ORDEN:
        X_tr, _, y_tr, _ = partes[origen]
        modelo = pipeline(lgbm())
        modelo.fit(X_tr, y_tr)
        for destino in ORDEN:
            _, X_te, _, y_te = partes[destino]
            pred = modelo.predict(X_te)
            prob = modelo.predict_proba(X_te)[:, 1]
            m = {"f1": f1_score(y_te, pred, zero_division=0),
                 "mcc": matthews_corrcoef(y_te, pred),
                 "roc_auc": roc_auc_score(y_te, prob)}
            marca = "  (misma casa)" if origen == destino else ""
            print(f"  entrena {origen:9s} -> evalua {destino:9s} "
                  f"F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}{marca}")
            filas.append({"entrena_en": origen, "evalua_en": destino,
                          "n_test": len(y_te),
                          **{k: round(v, 4) for k, v in m.items()}})


# ---------------------------------------------------------------------------
# 3. CURVAS DE APRENDIZAJE
# ---------------------------------------------------------------------------
def curvas(datos, filas):
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    tamanos = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    for k, (X, y) in datos.items():
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=SEMILLA, stratify=y)
        print(f"  {NOMBRE[k]}")
        for n in tamanos:
            if n > len(X_tr):
                break
            Xs, _, ys, _ = train_test_split(X_tr, y_tr, train_size=n,
                                            random_state=SEMILLA, stratify=y_tr)
            m = pipeline(lgbm())
            m.fit(Xs, ys)
            f1 = f1_score(y_te, m.predict(X_te), zero_division=0)
            print(f"    n={n:>6,}  F1={f1:.4f}")
            filas.append({"conjunto": k, "n_entrenamiento": n,
                          "f1": round(float(f1), 4)})


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def figuras(trans, curv, sufijo=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from estilo_figuras import aplicar, coma_decimal, tamano
    aplicar(plt)

    # La matriz de transferencia tenia aqui su propio panel, pero repetia una a
    # una las nueve cifras de la tabla que va justo encima en la memoria. Queda
    # solo la curva de aprendizaje, que es lo que la tabla no cuenta.
    c = pd.DataFrame(curv)
    fig, ax = plt.subplots(figsize=tamano(0.52))
    for k in ORDEN:
        s = c[c.conjunto == k]
        ax.plot(s.n_entrenamiento, s.f1, marker="o", color=COLOR[k],
                label=NOMBRE[k])
    ax.set_xscale("log")
    ax.set_xlabel("Ejemplos de entrenamiento (escala logarítmica)")
    ax.set_ylabel("F1 sobre el test reservado")
    ax.set_title("Cuántos datos necesita cada conjunto para saturar")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.3)
    coma_decimal(ax, "y")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURAS / f"b12_comparacion_conjuntos{sufijo}.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigura -> figuras/b12_comparacion_conjuntos{sufijo}.png")


def entorno():
    import importlib
    libs = {}
    for m in ("numpy", "pandas", "scikit-learn", "sklearn", "lightgbm"):
        try:
            libs[m] = importlib.import_module(m).__version__
        except Exception:
            pass
    return {"python": platform.python_version(), "sistema": platform.platform(),
            "semilla": SEMILLA, "test_size": TEST_SIZE, "cv_folds": CV_FOLDS,
            "variables_comunes": list(COMUNES), "librerias": libs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    suf = "_rapido" if args.rapido else ""

    print("=" * 74)
    print("1. AUDITORIA DE LOS TRES CONJUNTOS (con todas sus variables)")
    print("=" * 74)
    aud = []
    for k in ORDEN:
        X, y = cargar(k, args.rapido)
        print(f"\n{NOMBRE[k]}  ({len(X):,} x {X.shape[1]})")
        auditar(k, X, y, aud)

    print("\n" + "=" * 74)
    print("2. TRANSFERENCIA CRUZADA (solo las variables comunes)")
    print("=" * 74)
    comunes = {k: cargar(k, args.rapido, solo_comunes=True) for k in ORDEN}
    tra = []
    transferencia(comunes, tra)

    print("\n" + "=" * 74)
    print("3. CURVAS DE APRENDIZAJE (cada uno con sus variables)")
    print("=" * 74)
    cur = []
    curvas({k: cargar(k, args.rapido) for k in ORDEN}, cur)

    pd.DataFrame(aud).to_csv(RESULTADOS / f"comparacion_auditoria{suf}.csv",
                             index=False, encoding="utf-8")
    pd.DataFrame(tra).to_csv(RESULTADOS / f"comparacion_transferencia{suf}.csv",
                             index=False, encoding="utf-8")
    pd.DataFrame(cur).to_csv(RESULTADOS / f"comparacion_curvas{suf}.csv",
                             index=False, encoding="utf-8")
    (RESULTADOS / f"comparacion_entorno{suf}.json").write_text(
        json.dumps({"entorno": entorno(),
                    "minutos": round((time.perf_counter() - t0) / 60, 1)},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    figuras(tra, cur, suf)
    print(f"\ntiempo total: {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
