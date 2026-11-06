#!/usr/bin/env python3
"""
03_calcular_pesos.py — Calcula pesos por clase para la pérdida ponderada.

Uso:
    python 03_calcular_pesos.py --splits-dir data/splits

Lee:  <splits-dir>/train.csv
Sale: <splits-dir>/class_weights.json

El JSON contiene:
    labels, n_train, n_pos, pos_weight, pos_weight_sqrt
El entrenamiento usa pos_weight_sqrt (raíz, más suave).

Dependencias: pandas numpy
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening",
    "Cardiomegaly", "Nodule", "Mass", "Hernia",
]
SPLITS_DIR = Path("data/splits")


# ============================================================
# NUEVA FUNCIÓN: class_weights (expuesta a nivel de módulo)
# ============================================================
def class_weights(train_df):
    """Calcula pesos balanceados por clase. Devuelve un dict con:

        {
            "labels": [14 strings],
            "n_train": int,
            "n_pos": [14 ints],
            "pos_weight": [14 floats],       # neg/pos
            "pos_weight_sqrt": [14 floats],  # sqrt(neg/pos), el que usa 05_entrenar
        }
    """
    n = len(train_df)
    pos = train_df[LABELS].sum().astype(float)
    pw = ((n - pos) / pos.clip(lower=1)).round(4)
    return {
        "labels": LABELS,
        "n_train": int(n),
        "n_pos": [int(pos[l]) for l in LABELS],
        "pos_weight": [float(pw[l]) for l in LABELS],
        "pos_weight_sqrt": [round(float(np.sqrt(pw[l])), 4) for l in LABELS],
    }


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-dir", default=SPLITS_DIR, type=Path)
    a = ap.parse_args()

    csv = a.splits_dir / "train.csv"
    if not csv.exists():
        raise SystemExit(f"Falta {csv}. Ejecute primero 02_split_dataset.py")

    train = pd.read_csv(csv)
    w = class_weights(train)

    dest = a.splits_dir / "class_weights.json"
    dest.write_text(json.dumps(w, indent=2), encoding="utf-8")

    print(f"Train: {w['n_train']} imágenes\n")
    print(f"{'Patología':<22s} {'Positivos':>10s} {'Frecuencia':>12s} {'Peso (raíz)':>12s}")
    print("-" * 60)
    for lab, p, s in zip(LABELS, w["n_pos"], w["pos_weight_sqrt"]):
        freq = p / w["n_train"] * 100
        print(f"{lab:<22s} {p:>10d} {freq:>11.2f}% {s:>12.4f}")

    print(f"\nGuardado en {dest}")


if __name__ == "__main__":
    main()
