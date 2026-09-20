# -*- coding: utf-8 -*-
"""Dos comprobaciones que el notebook no hace y que el tribunal puede pedir.

    python tfm_pipeline/estabilidad.py --semillas
    python tfm_pipeline/estabilidad.py --imputacion
    python tfm_pipeline/estabilidad.py            # las dos

No forma parte del pipeline: se apoya en el, reconstruyendo el mismo
preprocesamiento a partir del fichero original y de los puntos de control, y
mide dos cosas que quedaban declaradas como limitacion en lugar de medidas.

1. --semillas  CUANTO MUEVE LA SEMILLA AL RESULTADO.
   Las 900 filas del registro de experimentos se obtuvieron con la semilla 42,
   y las Limitaciones lo reconocen: «no he cuantificado cuanto varian los
   resultados entre semillas». Aqui se repite el modelo final con cinco
   semillas, y cada una rehace TODO lo que la semilla controla ---la particion
   80/20, los pliegues de la validacion cruzada y el propio modelo---, que es
   la unica forma de que la cifra signifique algo.

   Importa por algo concreto: `desempate.py` documenta que dos de las cinco
   decisiones del pipeline se separan CERO a los cuatro decimales que se
   publican. Si la variacion entre semillas es mayor que el margen de las otras
   tres, hay que saberlo antes de la defensa.

2. --imputacion  SI EL INDICADOR DE AUSENCIA LE SIRVE AL MODELO FINAL.
   La seleccion de variables y el balanceo se reevaluaron con LightGBM despues
   de haberse decidido con la regresion logistica de referencia. La imputacion
   no, y es justo la que sostiene el hallazgo principal del trabajo: que la
   ausencia de respuesta WHOIS ya es predictiva.

   La objecion es evidente para quien conozca LightGBM: enruta los NaN de forma
   nativa, asi que el indicador podria no aportarle nada y estar ayudando solo a
   los modelos lineales. Se comparan las tres variantes que zanjan la duda:

       a) las 54 variables con sus 9 indicadores   (lo que hace el trabajo)
       b) las 45 sin indicadores, imputando mediana
       c) las 45 sin imputar, dejando el NaN a LightGBM

   Si (a) no le gana a (c), el hallazgo se sostiene para los modelos lineales
   pero no para el modelo que se entrega, y eso habria que escribirlo.

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
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
DATOS = RAIZ / "data" / "raw" / "dataset_full.csv"
PUNTOS = AQUI / "checkpoints"
RESULTADOS = AQUI / "resultados"

DIANA = "phishing"
TEST_SIZE = 0.20
SEMILLA_TRABAJO = 42
# Cinco semillas fijas y escritas aqui, para que la comprobacion sea repetible.
# No se sortean: una semilla elegida al azar no se puede volver a usar.
SEMILLAS = [42, 7, 123, 2024, 31337]


def cargar_seleccion():
    d = json.load(io.open(PUNTOS / "seleccion.json", encoding="utf-8"))
    return list(d["variables"])


def cargar_hiperparametros():
    d = json.load(io.open(PUNTOS / "mejores_hiperparametros.json", encoding="utf-8"))
    return dict(d["LightGBM"])


def modelo(params, semilla):
    """El modelo final, igual que lo construye el notebook."""
    return LGBMClassifier(n_jobs=-1, random_state=semilla, verbose=-1, **params)


def limpio():
    """El dataset sin conflictos de etiqueta ni duplicados, con -1 como NaN.

    Reproduce los pasos v01 y v02 del notebook. Se comprueba contra las cifras
    que el propio notebook dejo en su punto de control, de modo que si la
    reconstruccion se desviara, esto aborta en lugar de dar un numero distinto
    sin avisar.
    """
    crudo = pd.read_csv(DATOS)
    variables = [c for c in crudo.columns if c != DIANA]

    distintas = crudo.groupby(variables, dropna=False)[DIANA].transform("nunique")
    v01 = crudo.loc[distintas <= 1].copy().drop_duplicates(keep="first")
    v01 = v01.reset_index(drop=True)

    esperado = json.load(io.open(PUNTOS / "eda_duplicados.json", encoding="utf-8"))
    quitadas = len(crudo) - len(v01)
    debian = esperado["filas_en_conflicto"] + (esperado["duplicados_exactos"] - 1)
    if quitadas != debian:
        sys.exit("ERROR: la deduplicacion no reproduce el notebook (%d vs %d)"
                 % (quitadas, debian))

    v02 = v01.copy()
    v02[variables] = v02[variables].replace(-1, np.nan)
    return v02, variables


def preparar(v02, variables, semilla, con_indicador=True, imputar=True):
    """Particion y imputacion, ajustando el imputador SOLO con entrenamiento."""
    X, y = v02[variables], v02[DIANA]
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=TEST_SIZE,
                                        random_state=semilla, stratify=y)
    if not imputar:
        tr = X_tr.reset_index(drop=True)
        tr.columns = list(variables)
        return tr, y_tr.reset_index(drop=True)

    imp = SimpleImputer(strategy="median", add_indicator=con_indicador)
    imp.fit(X_tr)
    columnas = list(imp.get_feature_names_out(variables))
    tr = pd.DataFrame(imp.transform(X_tr), columns=columnas)
    return tr, y_tr.reset_index(drop=True)


def f1_cv5(estimador, X, y, semilla):
    particion = StratifiedKFold(n_splits=5, shuffle=True, random_state=semilla)
    s = cross_val_score(estimador, X, y, scoring="f1", cv=particion, n_jobs=-1)
    return s


def experimento_semillas(v02, variables, seleccion, params):
    print("=" * 74)
    print("  CUANTO MUEVE LA SEMILLA AL MODELO FINAL")
    print("=" * 74)
    print("  Cada semilla rehace la particion 80/20, los pliegues de la CV y el")
    print("  modelo. Las 54 variables son las del trabajo.\n")
    filas = []
    for s in SEMILLAS:
        t0 = time.perf_counter()
        tr, y = preparar(v02, variables, s)
        faltan = [c for c in seleccion if c not in tr.columns]
        if faltan:
            sys.exit("ERROR: con la semilla %d faltan columnas: %s" % (s, faltan[:4]))
        r = f1_cv5(modelo(params, s), tr[seleccion], y, s)
        filas.append({"semilla": s, "f1_mean": round(float(r.mean()), 4),
                      "f1_std_pliegues": round(float(r.std()), 4),
                      "n_train": len(tr),
                      "tiempo_s": round(time.perf_counter() - t0, 1)})
        print("    semilla %-6d F1 = %.4f  (desv. entre pliegues %.4f)  %.1f s"
              % (s, r.mean(), r.std(), time.perf_counter() - t0))

    medias = np.array([f["f1_mean"] for f in filas])
    print("\n    media entre semillas   %.4f" % medias.mean())
    print("    desviacion             %.4f" % medias.std(ddof=1))
    print("    recorrido              %.4f  (de %.4f a %.4f)"
          % (medias.max() - medias.min(), medias.min(), medias.max()))
    print("\n    La cifra del trabajo, con la semilla 42, es %.4f."
          % [f["f1_mean"] for f in filas if f["semilla"] == SEMILLA_TRABAJO][0])

    df = pd.DataFrame(filas)
    destino = RESULTADOS / "estabilidad_semillas.csv"
    df.to_csv(destino, index=False)
    print("    -> %s" % destino.relative_to(RAIZ))
    return medias


def experimento_imputacion(v02, variables, seleccion, params):
    print("\n" + "=" * 74)
    print("  LE SIRVE EL INDICADOR DE AUSENCIA AL MODELO FINAL?")
    print("=" * 74)
    print("  LightGBM enruta los NaN de forma nativa, asi que la pregunta es si el")
    print("  indicador le aporta algo a EL o solo a los modelos lineales.\n")

    sin_indicador = [c for c in seleccion if not c.startswith("missingindicator_")]
    n_ind = len(seleccion) - len(sin_indicador)
    print("    de las %d variables, %d son indicadores de ausencia\n"
          % (len(seleccion), n_ind))

    filas = []
    casos = [
        ("con indicador (el del trabajo)", seleccion, True, True),
        ("sin indicador, mediana", sin_indicador, False, True),
        ("sin imputar, NaN nativo de LightGBM", sin_indicador, False, False),
    ]
    for nombre, cols, con_ind, imputa in casos:
        t0 = time.perf_counter()
        tr, y = preparar(v02, variables, SEMILLA_TRABAJO,
                         con_indicador=con_ind, imputar=imputa)
        r = f1_cv5(modelo(params, SEMILLA_TRABAJO), tr[cols], y, SEMILLA_TRABAJO)
        filas.append({"variante": nombre, "n_variables": len(cols),
                      "f1_mean": round(float(r.mean()), 4),
                      "f1_std": round(float(r.std()), 4),
                      "tiempo_s": round(time.perf_counter() - t0, 1)})
        print("    %-38s n=%2d  F1 = %.4f +- %.4f"
              % (nombre, len(cols), r.mean(), r.std()))

    df = pd.DataFrame(filas)
    destino = RESULTADOS / "estabilidad_imputacion.csv"
    df.to_csv(destino, index=False)
    print("\n    -> %s" % destino.relative_to(RAIZ))

    base = df.loc[0, "f1_mean"]
    nativo = df.loc[2, "f1_mean"]
    print("\n    El indicador le saca al NaN nativo %+.4f de F1." % (base - nativo))
    if base > nativo:
        print("    El hallazgo se sostiene con el modelo que se entrega, no solo")
        print("    con la regresion logistica de referencia.")
    else:
        print("    OJO: no le saca nada. El hallazgo valdria para los modelos")
        print("    lineales pero NO para el modelo final, y habria que escribirlo.")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--semillas", action="store_true")
    ap.add_argument("--imputacion", action="store_true")
    args = ap.parse_args()
    todo = not (args.semillas or args.imputacion)

    seleccion = cargar_seleccion()
    params = cargar_hiperparametros()
    print("  Reconstruyendo el preprocesamiento del notebook...")
    v02, variables = limpio()
    print("  %d filas, %d variables predictoras\n" % (len(v02), len(variables)))

    if todo or args.semillas:
        experimento_semillas(v02, variables, seleccion, params)
    if todo or args.imputacion:
        experimento_imputacion(v02, variables, seleccion, params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
