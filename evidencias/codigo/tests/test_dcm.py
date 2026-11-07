"""Tests para load_dicom de 06_gui.py."""
import numpy as np
import pytest


def test_pu1_lee_monochrome2(dicom_factory, gui_module):
    base = np.random.default_rng(0).integers(0, 255, (80, 96), dtype=np.uint8)
    f = dicom_factory(base, "a.dcm", photometric="MONOCHROME2")
    out = gui_module.load_dicom(f)
    assert out.dtype == np.uint8
    assert out.shape == base.shape
    # 8 bits sin ventana → pasa tal cual
    assert np.array_equal(out, base)


def test_pu1_monochrome1_invierte(dicom_factory, gui_module):
    base = np.random.default_rng(0).integers(0, 255, (80, 96), dtype=np.uint8)
    # MONOCHROME1 con valores invertidos → debe salir igual que base
    f = dicom_factory(255 - base, "b.dcm", photometric="MONOCHROME1")
    out = gui_module.load_dicom(f)
    assert np.array_equal(out, base)


def test_pu1_12bits_con_ventana(dicom_factory, gui_module):
    a = np.linspace(0, 4095, 64 * 64).reshape(64, 64).astype(np.uint16)
    f = dicom_factory(a, "c.dcm", bits_stored=12, window=(2048, 4096))
    out = gui_module.load_dicom(f)
    assert out.dtype == np.uint8
    assert out.min() == 0
    assert out.max() >= 250


def test_pu1_16bits_sin_ventana_usa_percentiles(dicom_factory, gui_module):
    a = (np.random.default_rng(1).random((64, 64)) * 60000).astype(np.uint16)
    out = gui_module.load_dicom(dicom_factory(a, "d.dcm"))
    assert out.dtype == np.uint8
    assert out.max() == 255
    assert out.min() == 0


def test_pu2_rechaza_no_dicom(tmp_path, gui_module):
    f = tmp_path / "no.dcm"
    f.write_bytes(b"esto no es un DICOM")
    with pytest.raises(ValueError):
        gui_module.load_dicom(f)


def test_pu2_archivo_inexistente(tmp_path, gui_module):
    with pytest.raises((FileNotFoundError, ValueError)):
        gui_module.load_dicom(tmp_path / "nada.dcm")