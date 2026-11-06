#!/usr/bin/env python3
"""
04_crear_subconjunto.py — Crea un subconjunto estratificado para pruebas rápidas.

Uso:
    python 04_crear_subconjunto.py --splits-dir data/splits

Lee:  <splits-dir>/train.csv y <splits-dir>/val.csv
Sale: <splits-dir>/debug_train.csv  (~2000 imágenes con hasta N positivos por clase)
      <splits-dir>/debug_val.csv    (500 imágenes aleatorias de val)

Dependencias: pandas numpy
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening",
    "Cardiomegaly", "Nodule", "Mass", "Hernia",
]
SEED = 42


def stratified_subset(df, size, min_pos, seed):
    """Garantiza hasta min_pos imágenes positivas por patología (empezando por las
    más raras) y completa con aleatorias hasta llegar a `size`."""
    rng = np.random.default_rng(seed)
    chosen = set()

    # 1) Rellenar positivos de cada clase, de la más rara a la más común
    for lab in sorted(LABELS, key=lambda l: df[l].sum()):
        idx = df.index[df[lab] == 1].to_numpy()
        already = sum(1 for i in idx if i in chosen)
        need = max(0, min(min_pos, len(idx)) - already)
        pool = [i for i in idx if i not in chosen]
        if need and pool:
            chosen.update(rng.choice(pool, size=min(need, len(pool)),
                                       replace=False).tolist())

    # 2) Rellenar con imágenes al azar (mayoría serán sin hallazgos)
    rest = [i for i in df.index if i not in chosen]
    fill = max(0, size - len(chosen))
    chosen.update(rng.choice(rest, size=min(fill, len(rest)),
                               replace=False).tolist())

    return df.loc[sorted(chosen)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-dir", default=Path("data/splits"), type=Path)
    ap.add_argument("--size", default=2000, type=int,
                    help="Filas del subconjunto de entrenamiento")
    ap.add_argument("--val-size", default=500, type=int,
                    help="Filas del subconjunto de validación")
    ap.add_argument("--min-pos", default=60, type=int,
                    help="Positivos mínimos por clase en el subconjunto")
    ap.add_argument("--seed", default=SEED, type=int)
    a = ap.parse_args()

    tr = a.splits_dir / "train.csv"
    va = a.splits_dir / "val.csv"
    for p in (tr, va):
        if not p.exists():
            raise SystemExit(f"Falta {p}. Ejecute primero 02_split_dataset.py")

    train = pd.read_csv(tr)
    val = pd.read_csv(va)

    sub = stratified_subset(train, a.size, a.min_pos, a.seed)
    sub.to_csv(a.splits_dir / "debug_train.csv", index=False)

    vsub = val.sample(n=min(a.val_size, len(val)), random_state=a.seed).reset_index(drop=True)
    vsub.to_csv(a.splits_dir / "debug_val.csv", index=False)

    print(f"debug_train.csv: {len(sub)} imágenes")
    print(f"debug_val.csv  : {len(vsub)} imágenes\n")
    print("Positivos por etiqueta en debug_train:")
    print(f"{'Patología':<22s} {'Positivos':>10s}")
    print("-" * 34)
    for lab in LABELS:
        print(f"{lab:<22s} {int(sub[lab].sum()):>10d}")
    print(f"\nSin hallazgos: {int((sub[LABELS].sum(axis=1) == 0).sum())}")


if __name__ == "__main__":
    main()