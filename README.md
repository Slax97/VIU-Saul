# Detección de sitios web de phishing mediante aprendizaje automático

Código y resultados del Trabajo Fin de Máster de **Saúl Lax Pérez** (Máster
Universitario en Big Data y Ciencia de Datos, Universidad Internacional de
Valencia).

Todo el trabajo experimental vive en **un único notebook documentado**,
[`tfm_pipeline/TFM_deteccion_phishing.ipynb`](tfm_pipeline/TFM_deteccion_phishing.ipynb),
que va del dato bruto al modelo final y del que salen todas las cifras, tablas y
figuras de la memoria. Se guarda **con sus salidas**, así que puede leerse entero
sin ejecutar nada.

## Qué hay aquí

| Carpeta | Contenido |
|---|---|
| `tfm_pipeline/` | El notebook, con todo el proceso en once bloques |
| `tfm_pipeline/datos/` | El dataset original, sin modificar |
| `tfm_pipeline/checkpoints/` | Puntos de control de cada etapa costosa |
| `tfm_pipeline/resultados/` | Las 48 tablas de resultados y el registro de experimentos |
| `tfm_pipeline/figuras/` | Las figuras, en PNG y PDF vectorial |
| `data/externos/` | Descarga y verificación de los dos conjuntos de validación |

Lo único que no está dentro del notebook es la **validación externa**, porque
trabaja sobre otros conjuntos de datos. Son tres *scripts* en `tfm_pipeline/`:

```bash
python data/externos/descargar.py        # baja los dos conjuntos y verifica su SHA-256
python tfm_pipeline/validacion_externa.py
python tfm_pipeline/artefacto_phiusiil.py
python tfm_pipeline/comparacion_conjuntos.py
```

Sus tablas quedan en `tfm_pipeline/resultados/` con los prefijos
`validacion_externa` y `comparacion`. Los conjuntos externos no se versionan
aquí: tienen su propia licencia y pesan, por lo que se bajan de su origen y se
comprueba el SHA-256 de cada uno.

## Reproducirlo

Hace falta **Python 3.11**: las versiones fijadas no publican paquete compilado
para 3.12 ni posterior.

```bash
pip install uv
uv venv .venv311 --python 3.11
uv pip install --python .venv311 -r requirements.lock.txt
```

Con los puntos de control, volver a ejecutar el notebook entero lleva segundos
en lugar de la media hora que cuesta calcularlo todo desde cero. Borrando
`tfm_pipeline/checkpoints/` se recalcula todo desde el dataset original, que se
verifica por SHA-256 y nunca se modifica.

Con la semilla fija (42) se recuperan las decisiones del proceso y las cifras que
la memoria concluye. Algunas comparaciones intermedias entre alternativas
descartadas se mueven en la cuarta cifra decimal según el número de núcleos de la
máquina.

## Resultado

Modelo final: **LightGBM ajustado** con optimización bayesiana. Sobre un test
reservado, evaluado una sola vez: **F1 = 0,962 · ROC-AUC = 0,996 · MCC = 0,941 ·
accuracy = 0,973**, con intervalos de confianza por *bootstrap*.

El protocolo repetido sobre otros dos conjuntos de datos muestra que las
decisiones del trabajo se sostienen fuera de los datos originales; uno de los dos
resultó tener dos fugas de etiqueta y un sesgo de recogida, y de ahí salió una
batería de comprobaciones previas al entrenamiento que se aplica a los tres.

## Datos

El conjunto procede de un artículo de datos publicado en *Data in Brief*:

> Vrbančič, G., Fister Jr., I., & Podgorelec, V. (2020). *Datasets for phishing
> websites detection.* **Data in Brief, 33**, 106438.
> <https://doi.org/10.1016/j.dib.2020.106438>

Repositorio oficial de los autores:
<https://github.com/GregaVrbancic/Phishing-Dataset>

Se redistribuye aquí bajo su licencia **Creative Commons CC BY 4.0**, que permite
el uso y la redistribución con atribución; la cita anterior constituye dicha
atribución.

## Licencia

El código de este repositorio se publica con fines académicos. El dataset
mantiene su licencia original (CC BY 4.0).
