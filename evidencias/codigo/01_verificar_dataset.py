#!/usr/bin/env python3
"""
01_verificar_dataset.py — Comprueba los conteos del dataset ChestX-ray14.

Uso:
    python 01_verificar_dataset.py --entry-csv /ruta/Data_Entry_2017.csv

Salida:
    Reporte en consola. Código de salida 0 si OK, 1 si hay diferencias.

Dependencias: pandas
"""
import argparse
import sys

import pandas as pd

LABELS = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening",
    "Cardiomegaly", "Nodule", "Mass", "Hernia",
]
EXPECTED_TOTAL = 112120
EXPECTED_COUNTS = {
    "No Finding": 60361, "Infiltration": 19894, "Effusion": 13317, "Atelectasis": 11559,
    "Nodule": 6331, "Mass": 5782, "Pneumothorax": 5302, "Consolidation": 4667,
    "Pleural_Thickening": 3385, "Cardiomegaly": 2776, "Emphysema": 2516, "Edema": 2303,
    "Fibrosis": 1686, "Pneumonia": 1431, "Hernia": 227,
}


def add_multihot(df):
    findings = df["Finding Labels"].fillna("").str.split("|")
    for lab in LABELS:
        df[lab] = findings.apply(lambda xs, lab=lab: int(lab in xs))
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry-csv", required=True, help="Ruta a Data_Entry_2017.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.entry_csv)
    print(f"CSV: {a.entry_csv}")
    print(f"Filas leídas: {len(df)}")
    print(f"Pacientes únicos: {df['Patient ID'].nunique()}")
    print()

    df = add_multihot(df)

    ok = len(df) == EXPECTED_TOTAL
    print(f"Total imágenes: {len(df)} (esperado {EXPECTED_TOTAL}) -> "
          f"{'OK' if ok else 'DIFERENTE'}")
    print()

    got = {lab: int(df[lab].sum()) for lab in LABELS}
    got["No Finding"] = int((df[LABELS].sum(axis=1) == 0).sum())

    print(f"{'Etiqueta':<22s} {'Actual':>10s} {'Esperado':>10s} {'Estado':>8s}")
    print("-" * 55)
    for lab in ["No Finding"] + LABELS:
        actual = got[lab]
        esperado = EXPECTED_COUNTS[lab]
        estado = "OK" if actual == esperado else "DIF"
        if actual != esperado:
            ok = False
        print(f"{lab:<22s} {actual:>10d} {esperado:>10d} {estado:>8s}")

    print()
    if ok:
        print("Conteos coinciden con el dataset completo.")
    else:
        print("Hay diferencias. Verifique que el CSV sea el oficial.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()