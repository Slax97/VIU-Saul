# Detección de sitios web de phishing mediante aprendizaje automático

Código del Trabajo Fin de Máster de **Saúl Lax Pérez**, del Máster Universitario
en Big Data y Ciencia de Datos de la Universidad Internacional de Valencia.

Aquí está lo que produce las cifras: el cuaderno, los *scripts* de validación
externa, los puntos de control, las tablas de resultados, las figuras y el modelo
final. La memoria se deposita en la universidad y no se publica en este
repositorio.

## Qué hay

El grueso es **un único cuaderno documentado**,
[`tfm_pipeline/TFM_deteccion_phishing.ipynb`](tfm_pipeline/TFM_deteccion_phishing.ipynb),
que recorre el proceso entero del dato bruto al modelo final en doce bloques
numerados del 0 al 11, y del que salen todas las cifras, tablas y figuras de la
memoria. **Se guarda con sus salidas**, así que puede leerse sin ejecutar nada.

```
tfm_pipeline/
  TFM_deteccion_phishing.ipynb   el cuaderno, con sus salidas
  construir_notebook.py          lo genera y lo ejecuta de principio a fin
  validacion_externa.py          el protocolo sobre otros dos conjuntos
  artefacto_phiusiil.py          por qué PhiUSIIL no sirve como validación
  comparacion_conjuntos.py       auditoría y transferencia entre conjuntos
  estabilidad.py                 cuánto mueve la semilla; la imputación, repetida
  revision_quinta.py             recalibración fuera de distribución, tasa base,
                                 contraste entre familias y control de tamaño
  datos/                         el conjunto de partida, verificado por SHA-256
  checkpoints/                   el resultado de cada etapa costosa
  resultados/                    las tablas, en CSV
  figuras/                       las figuras, en PDF y PNG
  modelos/                       el modelo final y el estudio de Optuna
data/externos/descargar.py       baja los dos conjuntos de validación externa
                                 y verifica su SHA-256
```

## Cómo reproducirlo

Hace falta **Python 3.11**: las versiones con las que se obtuvieron estas cifras
no publican paquete compilado para 3.12 ni posterior.

```bash
python -m venv .venv311          # con un intérprete 3.11
.venv311/Scripts/pip install -r requirements.lock.txt
jupyter lab tfm_pipeline/TFM_deteccion_phishing.ipynb
```

Con los puntos de control ya calculados, reejecutar el cuaderno entero lleva unos
**quince segundos**. Borrando `tfm_pipeline/checkpoints/` se recalcula todo desde
el conjunto original, que nunca se modifica, en unos **veintitrés minutos**.

La semilla está fija en 42 y recupera las decisiones del proceso y las cifras de
la memoria. Algunas comparaciones intermedias entre alternativas descartadas se
mueven en la cuarta cifra decimal según el número de núcleos de la máquina; la
memoria explica por qué y qué criterio las resuelve.

La validación externa va aparte del cuaderno porque usa otros conjuntos de datos,
que no se versionan aquí:

```bash
python data/externos/descargar.py     # los baja y comprueba su SHA-256
python tfm_pipeline/validacion_externa.py
```

## Los datos

El conjunto de partida es el de **Vrbančič, Fister Jr. y Podgorelec (2020)**,
*Datasets for phishing websites detection*, Data in Brief 33, 106438
([10.1016/j.dib.2020.106438](https://doi.org/10.1016/j.dib.2020.106438)),
publicado con licencia CC BY 4.0. Son 88.647 sitios descritos por 111 variables
de la URL, del dominio, del DNS y de servicios externos.

Los dos conjuntos de validación externa son **PhiUSIIL** (Prasad y Chandra, 2024,
UCI Machine Learning Repository) y **URL-Phish 2025** (Linh et al., 2025). Cada
uno conserva su licencia y su cita; `data/externos/README.md` los documenta.
