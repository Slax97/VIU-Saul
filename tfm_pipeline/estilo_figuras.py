# -*- coding: utf-8 -*-
"""Estilo comun de las figuras que van a la memoria.

El problema que resuelve: matplotlib guardaba las figuras a 12 o 13 pulgadas de
ancho (30 cm) y LaTeX las metia en una caja de 11 a 15 cm. Esa reduccion encoge
tambien la tipografia, y una etiqueta de 9 pt acababa impresa a 3 pt, que no se
lee. Cambiar la fuente no arreglaba nada mientras la figura siguiera encogiendo.

La regla es generar la figura al ancho al que se va a imprimir. Si la escala es
1, el tamano de letra que se pide aqui es el que sale en el papel, y se acabo la
aritmetica. Por eso `tamano()` devuelve pulgadas a partir de la fraccion de la
caja de texto que ocupara en la memoria.

    from estilo_figuras import aplicar, tamano
    aplicar()
    fig, ax = plt.subplots(figsize=tamano(0.6))          # ancho completo
    fig, axes = plt.subplots(1, 2, figsize=tamano(0.42))  # dos paneles

Comprobar el resultado con `python memoria/revisar_figuras.py`.
"""

# Caja de texto de la memoria: A4 (21 cm) menos 3 cm de margen a cada lado.
ANCHO_TEXTO_CM = 15.0

# Tamanos tal y como se quieren ver impresos. El cuerpo de la memoria va a
# 11 pt; una figura puede ir algo por debajo, pero no a menos de 7.
TIPOGRAFIA = {
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
}


def aplicar(plt=None):
    """Fija el estilo. Devuelve el modulo pyplot por comodidad."""
    if plt is None:
        import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.3,
        **TIPOGRAFIA,
    })
    return plt


def coma_decimal(ax, ejes="xy"):
    """Pone coma decimal en los ejes, como el resto de la memoria.

    matplotlib escribe 0.944 y la memoria escribe 0,944. Mezclar las dos cosas
    dentro de una misma figura canta, sobre todo cuando los rotulos de los
    valores ya van con coma.
    """
    from matplotlib.ticker import ScalarFormatter

    # Se hereda del formateador de matplotlib y solo se le cambia el separador.
    # Con un formato propio ("%g") se pierden los ceros finales y la serie sale
    # descuadrada: 0,944  0,946  0,948  0,95  0,952.
    class Coma(ScalarFormatter):
        def __call__(self, x, pos=None):
            return super().__call__(x, pos).replace(".", ",")

    if "x" in ejes:
        ax.xaxis.set_major_formatter(Coma())
    if "y" in ejes:
        ax.yaxis.set_major_formatter(Coma())
    return ax


def tamano(alto_relativo=0.6, fraccion=1.0):
    """figsize en pulgadas para imprimirse sin encoger.

    Args:
        alto_relativo: alto como fraccion del ancho (0.6 = algo apaisada).
        fraccion: parte de la caja de texto que ocupara (1.0 = \\textwidth).

    Returns:
        (ancho, alto) en pulgadas.
    """
    ancho_in = ANCHO_TEXTO_CM * fraccion / 2.54
    return (ancho_in, ancho_in * alto_relativo)
