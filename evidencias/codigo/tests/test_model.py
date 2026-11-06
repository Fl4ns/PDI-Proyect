"""Tests para 05_entrenar.py (build_model)."""
import numpy as np
import pytest

pytest.importorskip("tensorflow")


def test_pu4_salida_14_en_0_1(train_module):
    m = train_module.build_model(weights=None)
    x = np.random.rand(2, 224, 224, 3).astype("float32")
    p = m.predict(x, verbose=0)
    assert p.shape == (2, 14)
    assert p.min() >= 0 and p.max() <= 1


def test_pu4_capa_gradcam(train_module):
    m = train_module.build_model(weights=None)
    capa = m.get_layer(train_module.LAST_CONV_LAYER)
    assert capa.output.shape[-1] == 1024


def test_pu4_compila_con_pesos(train_module):
    m = train_module.build_model(weights=None)
    train_module.compile_model(m, pos_weight=np.ones(14))
    # Si no falla al compilar, pasa
    assert m.loss is not None