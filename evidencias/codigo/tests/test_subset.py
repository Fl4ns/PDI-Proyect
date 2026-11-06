"""Tests para 04_crear_subconjunto.py."""
import pandas as pd


def test_pm1_subset_contiene_todas_las_clases(synthetic_dataset, subset_module):
    d, _ = synthetic_dataset
    # Cargar train_val (sin filtrar, así todas las clases están)
    tv = pd.read_csv(d / "Data_Entry_2017.csv")
    findings = tv["Finding Labels"].fillna("").str.split("|")
    for lab in subset_module.LABELS:
        tv[lab] = findings.apply(lambda xs, lab=lab: int(lab in xs))

    sub = subset_module.stratified_subset(tv, size=200, min_pos=5, seed=42)
    assert len(sub) == 200
    # Al menos una imagen positiva por clase
    for lab in subset_module.LABELS:
        assert sub[lab].sum() >= 1, f"{lab} sin positivos en el subset"


def test_pm1_subset_respeta_min_pos(synthetic_dataset, subset_module):
    d, _ = synthetic_dataset
    tv = pd.read_csv(d / "Data_Entry_2017.csv")
    findings = tv["Finding Labels"].fillna("").str.split("|")
    for lab in subset_module.LABELS:
        tv[lab] = findings.apply(lambda xs, lab=lab: int(lab in xs))

    sub = subset_module.stratified_subset(tv, size=2000, min_pos=5, seed=42)
    # Con min_pos=5, todas las clases deberían tener >= 5 (si hay >= 5 en el df)
    for lab in subset_module.LABELS:
        n_pos = int(tv[lab].sum())
        if n_pos >= 5:
            assert sub[lab].sum() >= 5, f"{lab} tiene solo {sub[lab].sum()} positivos"