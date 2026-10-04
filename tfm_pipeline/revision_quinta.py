# -*- coding: utf-8 -*-
"""Lo que la quinta revision del tutor pide medir en lugar de afirmar.

    python tfm_pipeline/revision_quinta.py --recalibrar
    python tfm_pipeline/revision_quinta.py --contraste
    python tfm_pipeline/revision_quinta.py --ataque
    python tfm_pipeline/revision_quinta.py --tasabase
    python tfm_pipeline/revision_quinta.py                # los cuatro

No forma parte del pipeline. Se apoya en el igual que `estabilidad.py`:
reconstruye el mismo preprocesamiento desde el fichero original y los puntos de
control, y comprueba la reconstruccion contra las cifras que el notebook dejo
guardadas, de modo que si se desviara aborta en lugar de dar otro numero.

1. --recalibrar  SI RECALIBRAR EL UMBRAL RECUPERA EL F1 FUERA DE DISTRIBUCION.
   La memoria concluye que la caida a 0,798 se arregla «recalibrando el umbral
   sobre una muestra reciente, que es mucho mas barato que reentrenar». Es la
   conclusion practica principal del trabajo y no tenia experimento detras: era
   una conjetura escrita como resultado. Aqui se mide.

   El protocolo evita el circulo obvio ---elegir el umbral sobre los mismos
   datos con los que luego se presume de haberlo arreglado---: la particion
   reciente se parte en dos, el umbral se elige en la mitad de calibracion y se
   mide en la otra, que no interviene en la eleccion.

   Se mide ademas algo que la memoria no decia y que el tutor pregunta: que pasa
   con los dominios de antiguedad desconocida. Son el 27,5 % de las filas, van
   codificados a -1 en el fichero original, y -1 queda por debajo del corte, de
   modo que caen TODOS en la particion de evaluacion y NINGUNO en la de
   entrenamiento. El modelo nunca ve ese valor al entrenar. Se repite por eso el
   experimento excluyendolos, para separar ese efecto de la antiguedad real.

2. --contraste  EL CONTRASTE ESTADISTICO QUE H2 NECESITA Y NO TENIA.
   H2 afirma que los ensamblados de arboles ganan a los modelos lineales, a SVM,
   a kNN y a las redes. El contraste que la memoria presenta ---Friedman y
   Wilcoxon-Holm--- solo compara seis modelos, y los seis son de arboles:
   justamente los que H2 no necesita distinguir. La desventaja de las otras
   familias solo aparece en la tabla de medias, sin contraste.

   Aqui se repite el contraste incluyendo un representante de cada familia,
   sobre el entrenamiento COMPLETO y con los mismos pliegues para todos, que es
   lo que exige un Wilcoxon pareado.

3. --ataque  UNA REFERENCIA CONTRA LA QUE LEER EL ATAQUE DE IMITACION.
   El experimento de la 6.3.1 no consulta al modelo: desplaza las variables
   manipulables hacia la mediana legitima. Eso no es un ataque adaptativo, es
   imitacion no adaptativa, y al 100 % de esfuerzo el resultado es tautologico.
   Sin una referencia no se sabe cuanto de la caida es del ataque y cuanto de
   haber movido las variables, asi que se anade perturbacion aleatoria de la
   misma magnitud. Se cuenta ademas cuantas URLs resultantes son imposibles.

4. --tasabase  QUE PASA CON LA PREVALENCIA REAL.
   La viabilidad operativa extrapola a un millon de URLs al dia «manteniendo la
   proporcion de clases», que en este conjunto es un 35 % de phishing. En
   trafico real el phishing es una fraccion muy pequena, y con un suceso raro
   una tasa de falsos positivos baja produce muchisimas falsas alarmas. Se
   tabula la precision y el volumen de avisos a prevalencias realistas.

Escribe en resultados/ para que `comprobar_cifras_todas.py` pueda respaldar
cualquier cifra que acabe citando la memoria.
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
DATOS = RAIZ / "data" / "raw" / "dataset_full.csv"
PUNTOS = AQUI / "checkpoints"
RESULTADOS = AQUI / "resultados"

DIANA = "phishing"
TEST_SIZE = 0.20
SEMILLA = 42
CORTE_ANTIGUEDAD = 0.70          # el mismo percentil que usa el notebook
FRACCION_CALIBRACION = 0.30      # de la particion reciente, para elegir el umbral


# ----------------------------------------------------------------------------
#  reconstruccion del preprocesamiento, igual que en estabilidad.py
# ----------------------------------------------------------------------------
def cargar_seleccion():
    d = json.load(io.open(PUNTOS / "seleccion.json", encoding="utf-8"))
    return list(d["variables"])


def cargar_hiperparametros():
    d = json.load(io.open(PUNTOS / "mejores_hiperparametros.json", encoding="utf-8"))
    return dict(d["LightGBM"])


def modelo(params, semilla=SEMILLA):
    return LGBMClassifier(n_jobs=-1, random_state=semilla, verbose=-1, **params)


def limpio():
    """v01 y v02 del notebook: sin conflictos de etiqueta, sin duplicados, -1 -> NaN."""
    crudo = pd.read_csv(DATOS)
    variables = [c for c in crudo.columns if c != DIANA]

    distintas = crudo.groupby(variables, dropna=False)[DIANA].transform("nunique")
    v01 = crudo.loc[distintas <= 1].copy().drop_duplicates(keep="first").reset_index(drop=True)

    esperado = json.load(io.open(PUNTOS / "eda_duplicados.json", encoding="utf-8"))
    quitadas = len(crudo) - len(v01)
    debian = esperado["filas_en_conflicto"] + (esperado["duplicados_exactos"] - 1)
    if quitadas != debian:
        sys.exit("ERROR: la deduplicacion no reproduce el notebook (%d vs %d)"
                 % (quitadas, debian))

    v02 = v01.copy()
    v02[variables] = v02[variables].replace(-1, np.nan)
    return v02, variables


def imputado_completo(v02, variables):
    """Train y test imputados y concatenados, como hace el bloque 10 del notebook.

    El imputador se ajusta SOLO con entrenamiento; despues se aplica a los dos y
    se concatenan, que es exactamente de donde sale `completo` en el cuaderno.
    """
    X, y = v02[variables], v02[DIANA]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEMILLA, stratify=y)

    imp = SimpleImputer(strategy="median", add_indicator=True).fit(X_tr)
    columnas = list(imp.get_feature_names_out(variables))
    tr = pd.DataFrame(imp.transform(X_tr), columns=columnas)
    te = pd.DataFrame(imp.transform(X_te), columns=columnas)
    completo = pd.concat([tr, te], ignore_index=True)
    diana = pd.concat([y_tr, y_te], ignore_index=True).reset_index(drop=True)
    return completo, diana


def metricas(y, pred, prob):
    return {"precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "mcc": matthews_corrcoef(y, pred),
            "roc_auc": roc_auc_score(y, prob)}


def mejor_umbral(y, prob):
    """El umbral que maximiza F1, buscado sobre una rejilla fina."""
    rejilla = np.unique(np.round(np.linspace(0.01, 0.99, 197), 4))
    puntuaciones = [f1_score(y, (prob >= u).astype(int), zero_division=0) for u in rejilla]
    i = int(np.argmax(puntuaciones))
    return float(rejilla[i]), float(puntuaciones[i])


# ----------------------------------------------------------------------------
#  1. recalibracion fuera de distribucion
# ----------------------------------------------------------------------------
def _una_recalibracion(completo, diana, seleccion, params, mascara, etiqueta, filas,
                       desconocidos_n=0):
    sub = completo.loc[mascara].reset_index(drop=True)
    y_sub = diana.loc[mascara].reset_index(drop=True)

    edad = sub["time_domain_activation"]
    corte = edad.quantile(CORTE_ANTIGUEDAD)
    es_veterano = edad >= corte

    X_vet, y_vet = sub.loc[es_veterano, seleccion], y_sub.loc[es_veterano]
    X_rec, y_rec = sub.loc[~es_veterano, seleccion], y_sub.loc[~es_veterano]

    m = modelo(params).fit(X_vet, y_vet)

    # la particion reciente se parte en dos: el umbral se elige en una mitad y
    # se mide en la otra, que no interviene en la eleccion
    idx_cal, idx_ev = train_test_split(
        np.arange(len(X_rec)), test_size=1 - FRACCION_CALIBRACION,
        random_state=SEMILLA, stratify=y_rec)

    prob = m.predict_proba(X_rec)[:, 1]
    y_rec = y_rec.to_numpy()

    u_nuevo, f1_cal = mejor_umbral(y_rec[idx_cal], prob[idx_cal])

    por_defecto = metricas(y_rec[idx_ev], (prob[idx_ev] >= 0.50).astype(int), prob[idx_ev])
    recalib = metricas(y_rec[idx_ev], (prob[idx_ev] >= u_nuevo).astype(int), prob[idx_ev])

    print("  %s" % etiqueta)
    print("    entrenamiento (30%% mas antiguo) %6d   evaluacion (70%% restante) %6d"
          % (len(X_vet), len(X_rec)))
    print("    de la evaluacion, para elegir umbral %5d   para medir %5d"
          % (len(idx_cal), len(idx_ev)))
    print("    umbral elegido en la mitad de calibracion: %.3f  (F1 alli %.4f)"
          % (u_nuevo, f1_cal))
    print("    sobre la mitad que no intervino:")
    print("      con 0,50          F1 %.4f  precision %.4f  recall %.4f  AUC %.4f"
          % (por_defecto["f1"], por_defecto["precision"], por_defecto["recall"],
             por_defecto["roc_auc"]))
    print("      con %.3f          F1 %.4f  precision %.4f  recall %.4f"
          % (u_nuevo, recalib["f1"], recalib["precision"], recalib["recall"]))
    print("      recuperacion de F1: %+.4f\n" % (recalib["f1"] - por_defecto["f1"]))

    for nombre, d, u in [("umbral 0,50", por_defecto, 0.50),
                         ("umbral recalibrado", recalib, u_nuevo)]:
        filas.append({"variante": etiqueta, "umbral": nombre, "valor_umbral": round(u, 4),
                      "n_total": len(sub), "n_desconocidos": int(desconocidos_n),
                      "n_train": len(X_vet), "n_calibracion": len(idx_cal),
                      "n_evaluacion": len(idx_ev),
                      **{k: round(float(v), 4) for k, v in d.items()}})
    return recalib["f1"] - por_defecto["f1"]


def experimento_recalibrar(v02, variables, seleccion, params):
    print("=" * 78)
    print("  RECALIBRAR EL UMBRAL FUERA DE DISTRIBUCION")
    print("=" * 78)
    completo, diana = imputado_completo(v02, variables)

    ind = "missingindicator_time_domain_activation"
    hay_indicador = ind in completo.columns
    if hay_indicador:
        desconocidos = completo[ind].astype(bool)
        print("  dominios de antiguedad desconocida: %d de %d  (%.1f %%)"
              % (desconocidos.sum(), len(completo), 100 * desconocidos.mean()))
        edad = completo["time_domain_activation"]
        corte = edad.quantile(CORTE_ANTIGUEDAD)
        en_recientes = (edad[desconocidos] < corte).mean()
        print("  de ellos, cuantos caen en la particion de evaluacion: %.1f %%"
              % (100 * en_recientes))
        print("  (la imputacion por mediana los deja por debajo del corte, asi que")
        print("   el modelo no ve ni uno al entrenar y luego se le evalua con ellos)\n")
    else:
        desconocidos = pd.Series(False, index=completo.index)
        print("  (no hay indicador de ausencia para la antiguedad)\n")

    filas = []
    n_desc = int(desconocidos.sum())
    todos = pd.Series(True, index=completo.index)
    _una_recalibracion(completo, diana, seleccion, params, todos,
                       "tal como lo hace el notebook", filas, n_desc)
    if hay_indicador and desconocidos.any():
        _una_recalibracion(completo, diana, seleccion, params, ~desconocidos,
                           "excluyendo los de antiguedad desconocida", filas, n_desc)

    df = pd.DataFrame(filas)
    destino = RESULTADOS / "revision_recalibracion_ood.csv"
    df.to_csv(destino, index=False)
    print("  -> %s" % destino.relative_to(RAIZ))
    return df


# ----------------------------------------------------------------------------
#  1b. el control de tamano: entrenar con lo mismo, elegido al azar
# ----------------------------------------------------------------------------
def experimento_control(v02, variables, seleccion, params, semillas=(42, 7, 13, 2024, 99)):
    """Separa «menos datos» de «otro regimen de datos».

    La particion por antiguedad entrena con el 30 % de las filas, de modo que
    parte de la caida podria venir del tamano y no del cambio de regimen. El
    control entrena con una muestra ALEATORIA del mismo tamano y evalua sobre
    las mismas 61.035 restantes: lo que se mantenga ahi no lo explica el tamano.

    Mide ademas la proporcion de phishing de cada lado del corte, que es el
    tercer efecto que la Seccion 6.3.2 declara.
    """
    print("=" * 78)
    print("  CONTROL DE TAMANO: MISMAS FILAS, ELEGIDAS AL AZAR")
    print("=" * 78)
    completo, diana = imputado_completo(v02, variables)
    edad = completo["time_domain_activation"]
    corte = edad.quantile(CORTE_ANTIGUEDAD)
    veteranos = edad >= corte
    n_train = int(veteranos.sum())
    print("  corte en el percentil %d de la antiguedad: %.0f"
          % (CORTE_ANTIGUEDAD * 100, corte))
    print("  entrenan %d filas y se evalua sobre %d\n"
          % (n_train, len(completo) - n_train))

    def una(mascara, etiqueta):
        X = completo[seleccion]
        X_tr, y_tr = X.loc[mascara], diana.loc[mascara]
        X_ev, y_ev = X.loc[~mascara], diana.loc[~mascara]
        m = modelo(params).fit(X_tr, y_tr)
        prob = m.predict_proba(X_ev)[:, 1]
        met = metricas(y_ev, (prob >= 0.5).astype(int), prob)
        print("  %-30s F1=%.4f  precision=%.4f  recall=%.4f  ROC-AUC=%.4f"
              % (etiqueta, met["f1"], met["precision"], met["recall"], met["roc_auc"]))
        return dict(particion=etiqueta, n_entrenamiento=len(y_tr),
                    n_evaluacion=len(y_ev),
                    phishing_entrenamiento=round(float(y_tr.mean()), 4),
                    phishing_evaluacion=round(float(y_ev.mean()), 4),
                    **{k: round(v, 4) for k, v in met.items()})

    filas = [una(veteranos, "por antiguedad")]
    for s in semillas:
        rng = np.random.RandomState(s)
        idx = rng.choice(len(completo), size=n_train, replace=False)
        mascara = pd.Series(False, index=completo.index)
        mascara.iloc[idx] = True
        filas.append(una(mascara, "aleatoria (semilla %d)" % s))

    df = pd.DataFrame(filas)
    azar = df.loc[df["particion"].str.startswith("aleatoria"), "f1"]
    print("\n  F1 por antiguedad %.4f frente a %.4f-%.4f al azar: la diferencia"
          % (df.loc[0, "f1"], azar.min(), azar.max()))
    print("  que NO explica el tamano es de %.4f" % (azar.mean() - df.loc[0, "f1"]))
    print("  phishing: %.1f %% entre los veteranos y %.1f %% entre los recientes"
          % (100 * df.loc[0, "phishing_entrenamiento"],
             100 * df.loc[0, "phishing_evaluacion"]))
    destino = RESULTADOS / "revision_control_tamano.csv"
    df.to_csv(destino, index=False)
    print("  -> %s" % destino.relative_to(RAIZ))
    return df


# ----------------------------------------------------------------------------
#  2. contraste estadistico con todas las familias
# ----------------------------------------------------------------------------
def experimento_contraste(v02, variables, seleccion, params):
    from scipy.stats import friedmanchisquare, wilcoxon
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    print("=" * 78)
    print("  CONTRASTE ESTADISTICO INCLUYENDO LAS FAMILIAS QUE H2 NOMBRA")
    print("=" * 78)

    X_tr, y_tr = _entrenamiento(v02, variables, seleccion)
    print("  entrenamiento completo: %d filas, %d variables\n" % X_tr.shape)

    escalado = lambda est: Pipeline([("sc", StandardScaler()), ("m", est)])
    candidatos = {
        "LightGBM(ajustado)": modelo(params),
        "XGBoost": _xgboost(),
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                               random_state=SEMILLA),
        "MLP(64,32)": escalado(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                                             random_state=SEMILLA)),
        "kNN(k=5)": escalado(KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        "SVM-RBF(Nystroem)": escalado(Pipeline([
            ("ny", Nystroem(n_components=300, random_state=SEMILLA)),
            ("sv", LinearSVC(C=1.0, dual="auto", random_state=SEMILLA))])),
        "LogisticRegression": escalado(LogisticRegression(max_iter=2000,
                                                          random_state=SEMILLA)),
    }

    # 5x3, los mismos que usa el notebook para su contraste. No es un detalle:
    # con 5 pliegues el Wilcoxon de dos colas no puede bajar de p = 0,0625
    # ---2/2^5---, de modo que ninguna diferencia podria salir significativa por
    # construccion, y el contraste no diria nada.
    from sklearn.model_selection import RepeatedStratifiedKFold
    particion = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEMILLA)
    pliegues = list(particion.split(X_tr, y_tr))
    por_modelo = {}
    for nombre, est in candidatos.items():
        t0 = time.perf_counter()
        puntuaciones = []
        for tr, te in pliegues:
            from sklearn.base import clone
            e = clone(est).fit(X_tr.iloc[tr], y_tr.iloc[tr])
            puntuaciones.append(f1_score(y_tr.iloc[te], e.predict(X_tr.iloc[te]),
                                         zero_division=0))
        por_modelo[nombre] = np.array(puntuaciones)
        print("    %-22s F1 = %.4f  (desv. %.4f)  %5.1f s"
              % (nombre, por_modelo[nombre].mean(), por_modelo[nombre].std(),
                 time.perf_counter() - t0))

    est_f, p_f = friedmanchisquare(*por_modelo.values())
    print("\n  Friedman sobre los %d modelos: chi2 = %.2f, p = %.2e"
          % (len(por_modelo), est_f, p_f))

    referencia = "LightGBM(ajustado)"
    crudos = []
    for nombre, v in por_modelo.items():
        if nombre == referencia:
            continue
        _, p = wilcoxon(por_modelo[referencia], v)
        crudos.append({"frente_a": nombre,
                       "dif_F1": round(float(por_modelo[referencia].mean() - v.mean()), 4),
                       "p": p})
    crudos.sort(key=lambda d: d["p"])
    m = len(crudos)
    anterior = 0.0
    for i, d in enumerate(crudos):                      # Holm-Bonferroni
        ajustado = max(anterior, min(1.0, (m - i) * d["p"]))
        d["p_holm"] = round(ajustado, 6)
        d["significativo"] = "si" if ajustado < 0.05 else "no"
        d["p"] = round(d["p"], 6)
        anterior = ajustado

    print("\n  Wilcoxon pareado frente a %s, con correccion de Holm:" % referencia)
    for d in crudos:
        print("    %-22s dif %+.4f   p = %.4f   p_Holm = %.4f   %s"
              % (d["frente_a"], d["dif_F1"], d["p"], d["p_holm"], d["significativo"]))

    df = pd.DataFrame(crudos)
    destino = RESULTADOS / "revision_contraste_familias.csv"
    df.to_csv(destino, index=False)
    pd.DataFrame({k: v for k, v in por_modelo.items()}).to_csv(
        RESULTADOS / "revision_contraste_pliegues.csv", index=False)
    print("  -> %s" % destino.relative_to(RAIZ))
    return df


def _entrenamiento(v02, variables, seleccion):
    X, y = v02[variables], v02[DIANA]
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=TEST_SIZE,
                                        random_state=SEMILLA, stratify=y)
    imp = SimpleImputer(strategy="median", add_indicator=True).fit(X_tr)
    columnas = list(imp.get_feature_names_out(variables))
    tr = pd.DataFrame(imp.transform(X_tr), columns=columnas)
    return tr[seleccion], y_tr.reset_index(drop=True)


def _xgboost():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=300, n_jobs=-1, random_state=SEMILLA,
                         eval_metric="logloss", tree_method="hist")


# ----------------------------------------------------------------------------
#  3. el ataque de imitacion, con referencia aleatoria
# ----------------------------------------------------------------------------
# Exactamente la misma lista que usa el bloque 10 del notebook, para que la
# columna de imitacion reproduzca sus cifras y la de azar sea comparable.
MANIPULABLES_PISTA = ("qty_", "length_", "directory_", "file_", "params_",
                      "domain_length")


def experimento_ataque(v02, variables, seleccion, params):
    print("=" * 78)
    print("  ATAQUE DE IMITACION, CONTRA UNA REFERENCIA ALEATORIA")
    print("=" * 78)

    X, y = v02[variables], v02[DIANA]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEMILLA, stratify=y)
    imp = SimpleImputer(strategy="median", add_indicator=True).fit(X_tr)
    columnas = list(imp.get_feature_names_out(variables))
    tr = pd.DataFrame(imp.transform(X_tr), columns=columnas)
    te = pd.DataFrame(imp.transform(X_te), columns=columnas)
    y_te = y_te.reset_index(drop=True)

    m = modelo(params).fit(tr[seleccion], y_tr.reset_index(drop=True))

    manipulables = [c for c in seleccion
                    if c.startswith(MANIPULABLES_PISTA)
                    and not c.startswith("missingindicator_")
                    and "domain_spf" not in c]
    print("  variables que el atacante reescribe sin coste (%d de %d):"
          % (len(manipulables), len(seleccion)))
    for c in manipulables:
        print("      %s" % c)
    print("\n  esfuerzo e: cada variable v pasa a (1-e)*v + e*mediana_legitima(v);")
    print("  e=0 deja la URL como esta y e=1 la pone exactamente en el perfil")
    print("  legitimo mediano, que por construccion el modelo clasifica como")
    print("  legitima. Por eso e=1 no mide robustez: mide el diseno.\n")

    perfil = te.loc[y_te == 0, manipulables].median()
    phishing = te.loc[y_te == 1].reset_index(drop=True)
    n = len(phishing)
    rng = np.random.default_rng(SEMILLA)

    filas = []
    for e in [0.0, 0.25, 0.50, 0.75, 1.0]:
        imita = phishing.copy()
        for v in manipulables:
            imita[v] = (1 - e) * phishing[v] + e * perfil[v]
        r_imita = m.predict(imita[seleccion]).sum()

        # referencia: mover lo mismo, pero en direccion aleatoria
        azar = phishing.copy()
        for v in manipulables:
            desplazamiento = np.abs(perfil[v] - phishing[v])
            signo = rng.choice([-1.0, 1.0], size=n)
            azar[v] = np.clip(phishing[v] + e * signo * desplazamiento, 0, None)
        r_azar = m.predict(azar[seleccion]).sum()

        # de las que la restriccion SI cumplia antes del ataque, cuantas la
        # rompe el ataque; en absoluto no significaria nada (ver _coherentes)
        antes, ahora = _coherentes(phishing), _coherentes(imita)
        if antes is None:
            rotas, base = 0, 1
        else:
            base = max(int(antes.sum()), 1)
            rotas = int((antes & ~ahora).sum())
        filas.append({"esfuerzo": e,
                      "recall_imitacion": round(float(r_imita / n), 4),
                      "recall_aleatorio": round(float(r_azar / n), 4),
                      "detectados_imitacion": int(r_imita),
                      "detectados_aleatorio": int(r_azar),
                      "coherentes_antes": base,
                      "rotas_por_el_ataque": rotas,
                      "pct_rotas": round(100.0 * rotas / base, 1)})
        print("    esfuerzo %3.0f %%   recall imitacion %.4f   recall aleatorio %.4f"
              "   URLs que el ataque vuelve imposibles %5.1f %%"
              % (100 * e, r_imita / n, r_azar / n, 100.0 * rotas / base))

    print("\n  Si la imitacion no baja el recall mas que el azar, el experimento")
    print("  mide haber movido las variables y no la estrategia del atacante.")
    df = pd.DataFrame(filas)
    destino = RESULTADOS / "revision_ataque_referencia.csv"
    df.to_csv(destino, index=False)
    print("  -> %s" % destino.relative_to(RAIZ))
    return df


def _coherentes(d):
    """Mascara de filas cuya longitud total cubre a la suma de sus partes.

    No se usa en absoluto, sino por diferencia: lo que interesa es cuantas
    filas CUMPLIAN la restriccion antes del ataque y dejan de cumplirla
    despues. Medido en absoluto no dice nada del ataque, porque estas
    variables no se definen de forma que la suma tenga que caber ---el 78 %
    de las URLs reales ya la incumple--- y confundir las dos cosas seria
    atribuir al ataque un defecto del dataset.
    """
    partes = [c for c in ("directory_length", "file_length", "params_length")
              if c in d.columns]
    if "length_url" not in d.columns or not partes:
        return None
    return d["length_url"] >= d[partes].sum(axis=1)


# ----------------------------------------------------------------------------
#  4. la tasa base
# ----------------------------------------------------------------------------
def experimento_tasabase():
    print("=" * 78)
    print("  QUE PASA CON LA PREVALENCIA REAL")
    print("=" * 78)
    recall, especificidad = 0.9618, 0.9793
    print("  Del test reservado: recall %.4f y especificidad %.4f.\n" % (recall, especificidad))
    print("  Sobre un millon de URLs analizadas al dia:\n")
    print("    %-14s %10s %10s %10s %12s" %
          ("phishing", "detectados", "perdidos", "falsas al.", "precision"))
    filas = []
    for p in [0.3497, 0.10, 0.01, 0.001]:
        n = 1_000_000
        mal, bien = n * p, n * (1 - p)
        tp, fn = mal * recall, mal * (1 - recall)
        fp = bien * (1 - especificidad)
        prec = tp / (tp + fp)
        filas.append({"prevalencia": p, "phishing_dia": int(round(mal)),
                      "detectados": int(round(tp)), "sin_detectar": int(round(fn)),
                      "falsas_alarmas": int(round(fp)), "precision": round(prec, 4)})
        print("    %-14s %10d %10d %10d %11.1f %%"
              % ("%.2f %%" % (100 * p), tp, fn, fp, 100 * prec))
    print("\n  La primera fila es la del conjunto (35 %% de phishing) y es la que")
    print("  usa la memoria. Las siguientes se acercan al trafico real, donde el")
    print("  phishing es raro: el recall no cambia, pero la precision se hunde")
    print("  porque hay muchisimas mas URLs legitimas que pueden fallar.")
    df = pd.DataFrame(filas)
    destino = RESULTADOS / "revision_tasa_base.csv"
    df.to_csv(destino, index=False)
    print("  -> %s" % destino.relative_to(RAIZ))
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recalibrar", action="store_true")
    ap.add_argument("--contraste", action="store_true")
    ap.add_argument("--ataque", action="store_true")
    ap.add_argument("--tasabase", action="store_true")
    ap.add_argument("--control", action="store_true")
    a = ap.parse_args()
    todos = not (a.recalibrar or a.contraste or a.ataque or a.tasabase
                 or a.control)

    RESULTADOS.mkdir(exist_ok=True)
    if a.tasabase or todos:
        experimento_tasabase()
        print()

    if a.recalibrar or a.contraste or a.ataque or a.control or todos:
        print("  reconstruyendo el preprocesamiento del notebook...")
        v02, variables = limpio()
        seleccion, params = cargar_seleccion(), cargar_hiperparametros()
        print("  %d filas limpias, %d variables seleccionadas\n" % (len(v02), len(seleccion)))

        if a.recalibrar or todos:
            experimento_recalibrar(v02, variables, seleccion, params)
            print()
        if a.control or todos:
            experimento_control(v02, variables, seleccion, params)
            print()
        if a.ataque or todos:
            experimento_ataque(v02, variables, seleccion, params)
            print()
        if a.contraste or todos:
            experimento_contraste(v02, variables, seleccion, params)


if __name__ == "__main__":
    main()
