"""Tests para 03_calcular_pesos.py."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def fake_train():
    """DataFrame con las 14 columnas, fuertemente desbalanceado."""
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({lab: rng.random(n) < 0.05 for lab in
                       ["Atelectasis", "Consolidation", "Infiltration",
                        "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                        "Effusion", "Pneumonia", "Pleural_Thickening",
                        "Cardiomegaly", "Nodule", "Mass", "Hernia"]})
    df["Hernia"] = False
    df.loc[0, "Hernia"] = True  # 1 solo positivo de Hernia
    return df.astype(int)


def test_pm1_weights_devuelve_14(fake_train, weights_module):
    w = weights_module.class_weights(fake_train)
    assert len(w["labels"]) == 14
    assert len(w["n_pos"]) == 14
    assert len(w["pos_weight"]) == 14
    assert len(w["pos_weight_sqrt"]) == 14


def test_pm1_weights_hernia_es_el_mayor(fake_train, weights_module):
    w = weights_module.class_weights(fake_train)
    idx_hernia = w["labels"].index("Hernia")
    # Hernia con 1 solo positivo → peso enorme
    assert w["n_pos"][idx_hernia] == 1
    # Es el mayor de todos
    assert w["pos_weight"][idx_hernia] == max(w["pos_weight"])
    # sqrt lo suaviza
    assert w["pos_weight_sqrt"][idx_hernia] < w["pos_weight"][idx_hernia]


def test_pm1_weights_todos_positivos(fake_train, weights_module):
    """Si una clase tiene todos positivos, pw sería 0 (neg=0)."""
    w = weights_module.class_weights(fake_train)
    # Ninguna clase debería tener peso 0 (porque todas tienen al menos un negativo)
    assert all(p >= 1 for p in w["pos_weight"])