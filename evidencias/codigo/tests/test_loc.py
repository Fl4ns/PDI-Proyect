"""Tests para to_boxes y heatmap de 06_gui.py."""
import numpy as np
import pytest


def test_pu5_to_boxes_dentro_de_limites(gui_module):
    hm = np.zeros((224, 224), dtype="float32")
    hm[40:100, 60:140] = 1.0
    hm[150:170, 10:30] = 0.9
    boxes = gui_module.to_boxes(hm, 0.5, (1024, 1024))
    assert len(boxes) == 2
    for x, y, w, h in boxes:
        assert 0 <= x and 0 <= y
        assert x + w <= 1024
        assert y + h <= 1024


def test_pu5_to_boxes_vacio(gui_module):
    assert gui_module.to_boxes(np.zeros((224, 224), dtype="float32")) == []


def test_pu5_to_boxes_escala(gui_module):
    hm = np.zeros((224, 224), dtype="float32")
    hm[56:112, 56:112] = 1.0  # región 56×56
    boxes = gui_module.to_boxes(hm, 0.5, (1024, 1024))
    # Esperamos ~4× escala: 56/224 × 1024 = 256
    assert len(boxes) == 1
    x, y, w, h = boxes[0]
    assert 200 < w < 320
    assert 200 < h < 320


@pytest.mark.requires_model
def test_pu5_heatmap_forma(gui_module):
    """Requiere modelo real (o stub) para grad-CAM."""
    import tensorflow as tf
    x = np.random.rand(224, 224, 3).astype("float32")
    if gui_module.USE_STUBS:
        hm = gui_module._heatmap_stub(x, 3)
    else:
        pytest.skip("Modelo real no disponible; requiere models/modelo_base.keras")
    assert hm.shape == (224, 224)