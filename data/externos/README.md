# Conjuntos de datos externos (validación externa)

Estos conjuntos **no** son la fuente del trabajo: el TFM se apoya en
[`data/raw/dataset_full.csv`](../raw/README.md), de Vrbančič et al. (2020). Los de
aquí sirven para lo que la memoria declara como su limitación principal —«todo el
trabajo experimental se apoya en un único conjunto de datos, sin validación
externa»— y como primera línea futura: repetir el protocolo sobre un segundo
conjunto y comprobar si se mantienen **las decisiones**, no solo la cifra final.

Como el original, **no se modifican nunca**: se verifican por SHA-256 al cargarlos.

## 1. PhiUSIIL Phishing URL Dataset

`phiusiil2024/PhiUSIIL_Phishing_URL_Dataset.csv`

| | |
|---|---|
| Referencia | Prasad, A. & Chandra, S. (2024). *PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning.* **Computers & Security, 136**, 103545 |
| DOI | `10.1016/j.cose.2023.103545` |
| Repositorio | UCI ML Repository, dataset 967 · donado el 3 de marzo de 2024 |
| Licencia | CC BY 4.0 |
| Tamaño | 235.795 × 56 · 134.850 legítimas / 100.945 phishing · IR 1,34 |
| Nulos / duplicados | 0 / 0 |
| SHA-256 (CSV) | `a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6` |

**Por qué este.** Sus variables salen del **código HTML de la página** además de la
URL (`LineOfCode`, `NoOfiFrame`, `HasPasswordField`, `NoOfCSS`, `NoOfJS`,
`HasHiddenFields`…), que es precisamente la carencia que la memoria señala del conjunto
original: «no incluye contenido HTML ni señales visuales». Y no depende de
consultas WHOIS, la otra limitación declarada.

> ### ⚠ Fuga de etiqueta detectada
>
> La variable **`URLSimilarityIndex` vale exactamente 100,0 en las 134.850 URLs
> legítimas**, sin una sola excepción, y por debajo de 100 en 100.159 de las
> 100.945 phishing. La regla «si vale 100, es legítima» alcanza por sí sola un
> **99,667 % de accuracy con cero falsos negativos**.
>
> | | `== 100` | `< 100` |
> |---|---|---|
> | Legítimas | **134.850 (todas)** | 0 |
> | Phishing | 786 (0,78 %) | 100.159 |
>
> Cualquier resultado publicado sobre este conjunto que incluya esa variable está
> midiendo, en la práctica, esa única regla. No es motivo para descartar el
> conjunto: es un hallazgo, y el análisis se hace con y sin la variable para
> separar lo que aporta el resto.

## 2. URL-Phish

`urlphish2025/Dataset.csv`

| | |
|---|---|
| Referencia | Linh, D. M. & Hung, T. C. (2025). *A feature-engineered dataset of benign and phishing URLs for machine learning and large language models evaluation.* **Data in Brief, 63**, 112162 |
| DOI | `10.1016/j.dib.2025.112162` |
| Repositorio | Mendeley Data `65z9twcx3r` v1 |
| Licencia | CC BY 4.0 |
| Tamaño | 116.600 × 26 · 100.000 benignas / 16.600 phishing · IR 6,02 |
| Nulos / duplicados | 14 / 1.369 |
| Recogida | Phishing de PhishTank, **noviembre 2024 – septiembre 2025** |
| SHA-256 | `d68b3cd0648dcf9c775347416ad1a8995e8a025921fbe3871ca6158d4db3c3a1` |

**Por qué este.** Ya estaba citado en la memoria (`linh2025dataset`) como primera
línea futura, y se publica en *Data in Brief*, la misma revista que el conjunto
original, lo que facilita comparar cómo se documentan. Aporta dos elementos que el
original no puede dar:

- **Actualidad.** El phishing es de 2024-2025, así que permite medir deriva
  temporal real, y no el cambio de régimen por antigüedad del dominio que se
  midió en el bloque 10.
- **Desbalanceo real (6,02:1** frente al 1,89:1 del original). La conclusión de
  que «el balanceo no aporta» se tomó sobre un desbalanceo moderado; aquí se pone
  a prueba de forma exigente.

Se le aplicó la misma prueba de fuga que a PhiUSIIL y **está limpio**: la mejor
variable por sí sola llega al 90,4 %, frente al 85,8 % de predecir siempre la
clase mayoritaria. Un margen de 4,6 puntos es lo esperable.

> **Discrepancia con el artículo.** El resumen habla de 111.660 URLs (100.000
> benignas + 11.660 phishing). El fichero publicado en Mendeley v1 tiene **116.600
> filas (100.000 + 16.600)**. Se trabaja con lo que trae el fichero y se hace
> constar la diferencia.

## Cómo se descargaron

```bash
# PhiUSIIL
curl -L -o phiusiil.zip \
  "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
unzip phiusiil.zip

# URL-Phish
curl -L -o Dataset.csv \
  "https://data.mendeley.com/public-files/datasets/65z9twcx3r/files/0e9c55e4-9adb-43f5-8403-1bbd143ebdb6/file_downloaded"
```

Los SHA-256 de arriba son los de los ficheros efectivamente descargados; el de
URL-Phish coincide además con el que publica la API de Mendeley.
