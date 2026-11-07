#!/usr/bin/env python3
"""
02_split_dataset.py — Divide el dataset POR PACIENTE en train / val / test.

Uso:
    python 02_split_dataset.py --dataset-dir /ruta/dataset [--out-dir data/splits]

El directorio del dataset debe contener:
    Data_Entry_2017.csv
    train_val_list.txt
    test_list.txt

Salida en --out-dir (por defecto ./data/splits):
    train.csv, val.csv, test.csv
    Cada uno con columnas: Image Index, Patient ID, Patient Age, Patient Gender,
    View Position, Finding Labels, + 14 columnas multi-hot en orden fijo.

Dependencias: pandas scikit-learn
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

LABELS = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening",
    "Cardiomegaly", "Nodule", "Mass", "Hernia",
]
SEED = 42
VAL_FRACTION = 0.12
SPLITS_DIR = Path("data/splits")

KEEP = ["Image Index", "Patient ID", "Patient Age", "Patient Gender",
        "View Position", "Finding Labels"] + LABELS


# ============================================================
# HELPERS
# ============================================================
def read_list(path):
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def add_multihot(df):
    """Agrega 14 columnas binarias (0/1) a partir de 'Finding Labels'."""
    findings = df["Finding Labels"].fillna("").str.split("|")
    for lab in LABELS:
        df[lab] = findings.apply(lambda xs, lab=lab: int(lab in xs))
    return df


def check_no_overlap(*dfs, col="Patient ID"):
    """Lanza AssertionError si un paciente aparece en más de un conjunto."""
    sets = [set(d[col]) for d in dfs]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            common = sets[i] & sets[j]
            assert not common, f"{len(common)} pacientes repetidos entre conjuntos {i} y {j}"


# ============================================================
# NUEVA FUNCIÓN: make_splits (expuesta a nivel de módulo)
# ============================================================
def make_splits(entry_csv, train_val_list, test_list, out_dir=SPLITS_DIR,
                val_fraction=VAL_FRACTION, seed=SEED, verbose=True):
    """Divide por paciente. Devuelve (train_df, val_df, test_df).

    Parámetros:
        entry_csv        : Path al Data_Entry_2017.csv
        train_val_list   : Path al train_val_list.txt
        test_list        : Path al test_list.txt
        out_dir          : carpeta de salida para los 3 CSV
        val_fraction     : fracción de train_val que va a val (por paciente)
        seed             : semilla para el split reproducible
        verbose          : si True, imprime resumen en consola

    Devuelve:
        (train, val, test) como DataFrames con las columnas KEEP.
    """
    entry_csv = Path(entry_csv)
    train_val_list = Path(train_val_list)
    test_list = Path(test_list)
    out_dir = Path(out_dir)

    df = add_multihot(pd.read_csv(entry_csv))
    tv_names = set(read_list(train_val_list))
    te_names = set(read_list(test_list))
    overlap = tv_names & te_names
    assert not overlap, f"{len(overlap)} imágenes aparecen en train_val y test a la vez"

    train_val = df[df["Image Index"].isin(tv_names)].reset_index(drop=True)
    test = df[df["Image Index"].isin(te_names)].reset_index(drop=True)

    sin_asignar = len(df) - len(train_val) - len(test)
    if sin_asignar and verbose:
        print(f"Aviso: {sin_asignar} imágenes no están en ninguna lista oficial")

    # Split por paciente
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    tr_idx, va_idx = next(gss.split(train_val, groups=train_val["Patient ID"]))
    train = train_val.iloc[tr_idx].reset_index(drop=True)
    val = train_val.iloc[va_idx].reset_index(drop=True)

    check_no_overlap(train, val, test)

    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in KEEP if c in train.columns]
    for name, sub in (("train", train), ("val", val), ("test", test)):
        sub[cols].to_csv(out_dir / f"{name}.csv", index=False)
        if verbose:
            print(f"{name}: {len(sub)} imágenes, {sub['Patient ID'].nunique()} pacientes")

    return train, val, test


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--out-dir", default=SPLITS_DIR, type=Path)
    ap.add_argument("--val-fraction", default=VAL_FRACTION, type=float)
    ap.add_argument("--seed", default=SEED, type=int)
    a = ap.parse_args()

    d = a.dataset_dir
    make_splits(
        d / "Data_Entry_2017.csv",
        d / "train_val_list.txt",
        d / "test_list.txt",
        a.out_dir, a.val_fraction, a.seed,
    )


if __name__ == "__main__":
    main()
