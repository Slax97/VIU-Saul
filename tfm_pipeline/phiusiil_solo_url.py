# -*- coding: utf-8 -*-
"""¿Se recupera PhiUSIIL quitandole las variables de contenido?

    python tfm_pipeline/phiusiil_solo_url.py

La Seccion 6.4.3 lo descarta como validacion porque sus paginas de phishing
estaban ya retiradas cuando se rastrearon: doce lineas de HTML y ningun recurso.
Eso vive en las variables de CONTENIDO, asi que la pregunta evidente ---y la que
el tribunal puede hacer, porque valdria como tercer contraste externo--- es si el
conjunto sirve quedandose solo con las veinte lexicas de la URL.

La respuesta es que no, y por un motivo distinto del primero: ninguna de sus URLs
legitimas tiene cadena de consulta. Se recogieron como paginas de inicio desnudas
y el phishing como URLs de ataque completas, asi que ver un `?` basta para
descartar que sea legitima. Son dos artefactos de recogida independientes.

Se le pasa la MISMA bateria de cuatro comprobaciones que la Seccion 6.4.3 aplica
a los tres conjuntos, para que las cifras sean comparables con las de su tabla, y
se guardan en resultados/ para que `comprobar_cifras_todas.py` respalde lo que
cita la memoria.
"""
import io
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
FUENTE = RAIZ / "data" / "externos" / "phiusiil2024" / "PhiUSIIL_Phishing_URL_Dataset.csv"
RESULTADOS = AQUI / "resultados"
SEMILLA = 42

# Las veinte que se calculan sobre la cadena de la URL. Fuera quedan las 28 de
# contenido HTML, los 5 identificadores y las 2 fugas ya documentadas
# (URLSimilarityIndex e IsHTTPS).
URL = ["URLLength", "DomainLength", "IsDomainIP", "CharContinuationRate",
       "TLDLegitimateProb", "URLCharProb", "TLDLength", "NoOfSubDomain",
       "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
       "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL",
       "DegitRatioInURL", "NoOfEqualsInURL", "NoOfQMarkInURL",
       "NoOfAmpersandInURL", "NoOfOtherSpecialCharsInURL",
       "SpacialCharRatioInURL"]
# En PhiUSIIL label = 1 es legitima, al contrario que en el conjunto de partida.
CONSULTA = ["NoOfQMarkInURL", "NoOfEqualsInURL", "NoOfAmpersandInURL"]


def cv3():
    return StratifiedKFold(3, shuffle=True, random_state=SEMILLA)


def cv5():
    return StratifiedKFold(5, shuffle=True, random_state=SEMILLA)


def main():
    if not FUENTE.exists():
        sys.exit("ERROR: falta %s. Lanza antes data/externos/descargar.py"
                 % FUENTE.relative_to(RAIZ))
    df = pd.read_csv(FUENTE, encoding="utf-8-sig")
    df = df.drop_duplicates(subset=URL + ["label"], keep="first").reset_index(drop=True)
    X, y = df[URL], df["label"]
    n_leg = int((y == 1).sum())
    n_phi = int((y == 0).sum())
    mayoritaria = float(y.value_counts(normalize=True).max())

    print("  %d filas tras quitar duplicados exactos sobre las 20 de URL" % len(df))
    print("  legitimas %d   phishing %d   clase mayoritaria %.4f\n"
          % (n_leg, n_phi, mayoritaria))

    # 1. la mejor variable sola, con un unico corte
    mejor, acc = None, 0.0
    for c in URL:
        a = cross_val_score(DecisionTreeClassifier(max_depth=1, random_state=SEMILLA),
                            X[[c]], y, scoring="accuracy", cv=cv3(), n_jobs=-1).mean()
        if a > acc:
            mejor, acc = c, a
    ventaja = (acc - mayoritaria) * 100
    print("  mejor variable sola : %s" % mejor)
    print("  su accuracy         : %.4f  (ventaja %+.2f puntos)" % (acc, ventaja))

    # 2. constantes dentro de una clase, y si eso separa o es rareza
    degeneradas = []
    for c in URL:
        for clase in (0, 1):
            v = X.loc[y == clase, c]
            if v.nunique(dropna=True) <= 1:
                unico = v.dropna().unique()[0]
                otra = X.loc[y != clase, c]
                fuera = int((otra != unico).sum())
                degeneradas.append({"variable": c, "constante_en_clase": clase,
                                    "valor": unico, "casos_otra_clase_distintos": fuera,
                                    "pct_otra_clase": round(100.0 * fuera / len(otra), 3)})
    print("  variables constantes dentro de una clase: %d" % len(degeneradas))

    # 3. el arbol de una sola pregunta, y dos y tres
    arboles = {}
    for d in (1, 2, 3):
        arboles[d] = float(cross_val_score(
            DecisionTreeClassifier(max_depth=d, random_state=SEMILLA),
            X, y, scoring="f1", cv=cv5(), n_jobs=-1).mean())
        print("  arbol de profundidad %d: F1 %.4f" % (d, arboles[d]))

    # 4. el segundo artefacto: las legitimas no llevan cadena de consulta
    print("\n  Cadena de consulta, por clase:")
    consulta = []
    for c in CONSULTA:
        leg = int((X.loc[y == 1, c] > 0).sum())
        phi = int((X.loc[y == 0, c] > 0).sum())
        consulta.append({"variable": c, "legitimas_con_valor": leg,
                         "phishing_con_valor": phi,
                         "pct_legitimas": round(100.0 * leg / n_leg, 3),
                         "pct_phishing": round(100.0 * phi / n_phi, 3)})
        print("    %-22s legitimas %6d (%.3f %%)   phishing %6d (%.2f %%)"
              % (c, leg, 100.0 * leg / n_leg, phi, 100.0 * phi / n_phi))

    RESULTADOS.mkdir(exist_ok=True)
    resumen = pd.DataFrame([{
        "filas": len(df), "variables_url": len(URL),
        "legitimas": n_leg, "phishing": n_phi,
        "clase_mayoritaria": round(mayoritaria, 4),
        "mejor_variable_sola": mejor,
        "acc_una_variable": round(acc, 4),
        "ventaja_puntos": round(ventaja, 2),
        "n_degeneradas": len(degeneradas),
        "f1_arbol_prof1": round(arboles[1], 4),
        "f1_arbol_prof2": round(arboles[2], 4),
        "f1_arbol_prof3": round(arboles[3], 4),
    }])
    resumen.to_csv(RESULTADOS / "validacion_externa_phiusiil_solo_url.csv", index=False)
    pd.DataFrame(consulta).to_csv(
        RESULTADOS / "validacion_externa_phiusiil_consulta.csv", index=False)
    pd.DataFrame(degeneradas).to_csv(
        RESULTADOS / "validacion_externa_phiusiil_degeneradas.csv", index=False)

    print("\n  -> validacion_externa_phiusiil_solo_url.csv")
    print("  -> validacion_externa_phiusiil_consulta.csv")
    print("  -> validacion_externa_phiusiil_degeneradas.csv")

    print("\n  LECTURA. Con solo la URL la bateria mejora mucho: la ventaja de la")
    print("  mejor variable baja de 42,48 a %.2f puntos y el arbol de una pregunta" % ventaja)
    print("  de 0,9961 a %.4f, que es la altura del conjunto de partida. Pero las" % arboles[1])
    print("  legitimas no llevan cadena de consulta y las de phishing si, asi que")
    print("  hay un segundo artefacto de recogida y el conjunto no sirve ni recortado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
