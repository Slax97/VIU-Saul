# -*- coding: utf-8 -*-
"""Descarga los conjuntos externos de validacion y verifica su integridad.

    python data/externos/descargar.py            # descarga lo que falte
    python data/externos/descargar.py --verificar # solo comprueba los SHA-256

Los ficheros no se versionan: pesan 69 MB, son de terceros y estan publicados en
repositorios con DOI estable. Lo que se versiona es este script y el SHA-256 de
cada uno, que es lo que de verdad garantiza que se trabaja con el mismo fichero.
Si un hash no cuadra, el script se detiene: mejor eso que analizar otra cosa sin
enterarse.

Detalles y justificacion de cada conjunto en README.md.
"""
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent

CONJUNTOS = [
    {
        "nombre": "PhiUSIIL (Prasad & Chandra, 2024)",
        "destino": AQUI / "phiusiil2024" / "PhiUSIIL_Phishing_URL_Dataset.csv",
        "url": "https://archive.ics.uci.edu/static/public/967/"
               "phiusiil+phishing+url+dataset.zip",
        "zip_interno": "PhiUSIIL_Phishing_URL_Dataset.csv",
        "sha256": "a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6",
    },
    {
        "nombre": "URL-Phish (Linh & Hung, 2025)",
        "destino": AQUI / "urlphish2025" / "Dataset.csv",
        "url": "https://data.mendeley.com/public-files/datasets/65z9twcx3r/"
               "files/0e9c55e4-9adb-43f5-8403-1bbd143ebdb6/file_downloaded",
        "zip_interno": None,
        "sha256": "d68b3cd0648dcf9c775347416ad1a8995e8a025921fbe3871ca6158d4db3c3a1",
    },
]


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(1 << 20), b""):
            h.update(trozo)
    return h.hexdigest()


def descargar(c):
    print(f"  bajando {c['url'][:70]}...")
    with urllib.request.urlopen(c["url"], timeout=300) as r:
        datos = r.read()
    if c["zip_interno"]:
        with zipfile.ZipFile(io.BytesIO(datos)) as z:
            datos = z.read(c["zip_interno"])
    c["destino"].parent.mkdir(parents=True, exist_ok=True)
    c["destino"].write_bytes(datos)


def main():
    solo_verificar = "--verificar" in sys.argv
    fallos = 0

    for c in CONJUNTOS:
        print(f"\n{c['nombre']}")
        if not c["destino"].exists():
            if solo_verificar:
                print("  FALTA (ejecuta sin --verificar para bajarlo)")
                fallos += 1
                continue
            descargar(c)

        real = sha256(c["destino"])
        rel = c["destino"].relative_to(AQUI.parent.parent)
        if real == c["sha256"]:
            mb = c["destino"].stat().st_size / 1e6
            print(f"  OK  {rel}  ({mb:.1f} MB, SHA-256 verificado)")
        else:
            print(f"  ERROR de integridad en {rel}")
            print(f"      esperado: {c['sha256']}")
            print(f"      obtenido: {real}")
            fallos += 1

    if fallos:
        raise SystemExit(f"\n{fallos} conjunto(s) sin verificar: no sigas hasta "
                         f"resolverlo.")
    print("\nTodo verificado.")


if __name__ == "__main__":
    main()
