"""Tests para el preprocesamiento de 05_entrenar.py."""
import numpy as np
import pytest

pytest.importorskip("tensorflow")


def test_pu3_to_224_rgb_forma(train_module):
    img = np.random.randint(0, 255, (300, 400), dtype=np.uint8)
    out = train_module.to_224_rgb(img)
    arr = out.numpy() if hasattr(out, "numpy") else out
    assert arr.shape == (224, 224, 3)
    assert arr.dtype == np.uint8


def test_pu3_to_224_rgb_acepta_1_y_3_canales(train_module):
    a = np.random.randint(0, 255, (50, 60, 1), dtype=np.uint8)
    b = np.repeat(a, 3, axis=2)
    out_a = train_module.to_224_rgb(a).numpy()
    out_b = train_module.to_224_rgb(b).numpy()
    assert np.allclose(out_a, out_b, atol=2)


def test_pu3_normalize_tipo_y_rango(train_module):
    x = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    out = train_module.normalize(x)
    arr = out.numpy() if hasattr(out, "numpy") else out
    assert arr.shape == (224, 224, 3)
    assert arr.dtype == np.float32
    assert np.isfinite(arr).all()