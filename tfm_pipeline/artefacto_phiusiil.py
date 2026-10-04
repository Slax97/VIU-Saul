# -*- coding: utf-8 -*-
"""Documenta por que PhiUSIIL se clasifica casi perfecto: un artefacto de recogida.

    python tfm_pipeline/artefacto_phiusiil.py

Escribe:
    resultados/validacion_externa_artefacto_phiusiil.csv
    resultados/validacion_externa_fuga_phiusiil.csv
    figuras/b12_artefacto_phiusiil.{png,pdf}

Que se comprueba aqui. Al aplicar el protocolo del TFM a PhiUSIIL sale un F1 de
practicamente 1, y sigue saliendo casi 1 despues de quitar `URLSimilarityIndex`,
que es una variable con fuga de etiqueta evidente (vale exactamente 100 en las
134.850 URLs legitimas, en todas). Que el rendimiento no baje al quitarla obliga
a preguntarse que mas esta separando las clases, y la respuesta no es agradable:
las variables de contenido HTML.

La pagina de phishing mediana del conjunto tiene doce lineas de codigo, cero
imagenes, cero hojas de estilo, cero JavaScript y cero enlaces. Eso no es una
pagina de phishing: es una pagina caida, un dominio aparcado o un aviso de
retirada. Un phishing de verdad IMITA a la legitima, asi que tiene imagenes,
estilos y un formulario. Lo que separa las clases aqui no es el phishing, es que
las URLs maliciosas se rastrearon cuando ya estaban dadas de baja.

Importa para el TFM porque el analisis de errores del trabajo original concluye
que los falsos negativos dificiles son los que imitan bien el aspecto de un sitio
legitimo. En PhiUSIIL esos casos dificiles practicamente no existen, de modo que
las cifras publicadas sobre este conjunto no se pueden comparar con las de un
conjunto donde si existen.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
CSV = RAIZ / "data" / "externos" / "phiusiil2024" / "PhiUSIIL_Phishing_URL_Dataset.csv"
RESULTADOS = AQUI / "resultados"
FIGURAS = AQUI / "figuras"

# Variables que solo se pueden calcular si la pagina se descargo con contenido.
CONTENIDO = ["LineOfCode", "LargestLineLength", "NoOfImage", "NoOfCSS", "NoOfJS",
             "NoOfSelfRef", "NoOfExternalRef", "HasTitle", "HasFavicon",
             "HasDescription", "HasCopyrightInfo", "HasSocialNet",
             "IsResponsive", "HasSubmitButton"]


def main():
    if not CSV.exists():
        raise SystemExit(f"No encuentro {CSV}. Ejecuta antes "
                         f"python data/externos/descargar.py")
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(CSV)
    d["clase"] = d["label"].map({1: "legitima", 0: "phishing"})

    # --- 1. La fuga de URLSimilarityIndex -----------------------------------
    es100 = d["URLSimilarityIndex"].eq(100.0)
    fuga = pd.crosstab(d["clase"], es100.map({True: "igual_100", False: "menor_100"}))
    fuga = fuga.reindex(columns=["igual_100", "menor_100"], fill_value=0).reset_index()
    n = len(d)
    aciertos = int(((es100) & (d["clase"] == "legitima")).sum()
                   + ((~es100) & (d["clase"] == "phishing")).sum())
    fuga.to_csv(RESULTADOS / "validacion_externa_fuga_phiusiil.csv",
                index=False, encoding="utf-8")
    print("Fuga de etiqueta en URLSimilarityIndex")
    print(fuga.to_string(index=False))
    print(f"  regla 'si vale 100 -> legitima': accuracy {aciertos / n * 100:.3f} % "
          f"sobre {n:,} filas, con una sola variable\n")

    # --- 2. El artefacto de recogida ----------------------------------------
    med = d.groupby("clase")[CONTENIDO].median().T
    med.columns = [f"mediana_{c}" for c in med.columns]
    med.index.name = "variable"
    med = med.reset_index()

    vacia = d["HasTitle"].eq(0) & d["HasFavicon"].eq(0) & d["HasDescription"].eq(0)
    sin_recursos = (d["NoOfImage"].eq(0) & d["NoOfCSS"].eq(0) & d["NoOfJS"].eq(0))
    resumen = pd.DataFrame({
        "indicador": ["sin titulo, favicon ni descripcion (%)",
                      "sin imagenes, CSS ni JS (%)",
                      "menos de 50 lineas de codigo (%)"],
        "legitimas": [
            round(vacia[d["clase"] == "legitima"].mean() * 100, 2),
            round(sin_recursos[d["clase"] == "legitima"].mean() * 100, 2),
            round(d.loc[d["clase"] == "legitima", "LineOfCode"].lt(50).mean() * 100, 2)],
        "phishing": [
            round(vacia[d["clase"] == "phishing"].mean() * 100, 2),
            round(sin_recursos[d["clase"] == "phishing"].mean() * 100, 2),
            round(d.loc[d["clase"] == "phishing", "LineOfCode"].lt(50).mean() * 100, 2)],
    })
    med.to_csv(RESULTADOS / "validacion_externa_artefacto_phiusiil.csv",
               index=False, encoding="utf-8")
    resumen.to_csv(RESULTADOS / "validacion_externa_artefacto_resumen.csv",
                   index=False, encoding="utf-8")
    print("Contenido HTML por clase (medianas)")
    print(med.to_string(index=False))
    print()
    print(resumen.to_string(index=False))

    # --- 3. Figura ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from estilo_figuras import aplicar, tamano
    aplicar(plt)

    NARANJA, AZUL = "#E69F00", "#0072B2"      # Okabe-Ito, como el resto del TFM
    # Dos paneles a lo ancho de la caja de texto: cada uno se lleva la mitad.
    # El alto deja sitio al titulo de dos lineas y a la leyenda comun.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=tamano(0.50))

    cortes = [0, 10, 50, 100, 500, 1000, 5000, d["LineOfCode"].max()]
    etiquetas = ["0-10", "10-50", "50-100", "100-500", "500-1k", "1k-5k", ">5k"]
    tramo = pd.cut(d["LineOfCode"], bins=cortes, labels=etiquetas,
                   include_lowest=True)
    tab = pd.crosstab(tramo, d["clase"], normalize="columns").mul(100)
    x = np.arange(len(etiquetas))
    a1.bar(x - 0.2, tab["legitima"], 0.4, label="Legítima", color=AZUL)
    a1.bar(x + 0.2, tab["phishing"], 0.4, label="Phishing", color=NARANJA)
    a1.set_xticks(x, etiquetas, rotation=30)
    a1.set_xlabel("Líneas de código HTML de la página")
    a1.set_ylabel("% de la clase")
    a1.set_title("Tamaño del HTML")

    ind = ["NoOfImage", "NoOfCSS", "NoOfJS", "NoOfExternalRef"]
    cero = pd.DataFrame({
        "Legítima": [d.loc[d["clase"] == "legitima", c].eq(0).mean() * 100 for c in ind],
        "Phishing": [d.loc[d["clase"] == "phishing", c].eq(0).mean() * 100 for c in ind],
    }, index=["Imágenes", "CSS", "JavaScript", "Enlaces ext."])
    y = np.arange(len(ind))
    a2.barh(y - 0.2, cero["Legítima"], 0.4, label="Legítima", color=AZUL)
    a2.barh(y + 0.2, cero["Phishing"], 0.4, label="Phishing", color=NARANJA)
    a2.set_yticks(y, cero.index)
    a2.set_xlabel("% de páginas con valor cero")
    a2.set_title("Recursos ausentes")

    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    # Una sola leyenda para los dos paneles, fuera de las areas de dibujo. Cada
    # panel tenia la suya, iguales las dos, y la del derecho se pintaba encima
    # de las barras de "Enlaces ext." y "JavaScript" y las tapaba.
    fig.legend(*a1.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.88), ncol=2, frameon=False)
    # El titulo iba en una sola linea y no cabia en los 15 cm de la caja de
    # texto: `bbox_inches='tight'` ensanchaba el PDF a 19 cm para que cupiera, y
    # entonces LaTeX lo encogia otra vez y la letra bajaba a 6,3 pt. Partido en
    # dos lineas, la figura sale ya al ancho al que se imprime.
    fig.suptitle("PhiUSIIL: lo que separa las clases es que el phishing\n"
                 "ya estaba dado de baja cuando se rastreó", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    for ext in ("png", "pdf"):
        fig.savefig(FIGURAS / f"b12_artefacto_phiusiil.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigura -> {(FIGURAS / 'b12_artefacto_phiusiil.png').relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
