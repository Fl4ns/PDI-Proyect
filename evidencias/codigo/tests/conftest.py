"""
Fixtures compartidas para los tests de los scripts standalone 01–06.

Los scripts viven en el directorio padre. Los cargamos como módulos
sin ejecutar su __main__ (que arrancaría la GUI o el entrenamiento).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LABELS = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening",
    "Cardiomegaly", "Nodule", "Mass", "Hernia",
]


# ============================================================
# CARGA DE SCRIPTS STANDALONE
# ============================================================
def load_script(filename, module_name):
    """Importa un script del root como módulo, sin ejecutar __main__."""
    path = ROOT / filename
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def verify_module():
    """01_verificar_dataset.py"""
    return load_script("01_verificar_dataset.py", "verify01")


@pytest.fixture(scope="session")
def split_module():
    """02_split_dataset.py"""
    return load_script("02_split_dataset.py", "split02")


@pytest.fixture(scope="session")
def weights_module():
    """03_calcular_pesos.py"""
    return load_script("03_calcular_pesos.py", "weights03")


@pytest.fixture(scope="session")
def subset_module():
    """04_crear_subconjunto.py"""
    return load_script("04_crear_subconjunto.py", "subset04")


@pytest.fixture(scope="session")
def train_module():
    """05_entrenar.py — requiere TensorFlow instalado."""
    pytest.importorskip("tensorflow")
    return load_script("05_entrenar.py", "train05")


@pytest.fixture(scope="session")
def gui_module():
    """06_gui.py — no requiere TensorFlow al importar."""
    return load_script("06_gui.py", "gui06")


# ============================================================
# DATOS SINTÉTICOS
# ============================================================
def _make_entry(n_patients=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(1, n_patients + 1):
        for k in range(rng.integers(1, 5)):
            labs = [l for l in LABELS if rng.random() < 0.12]
            rows.append({
                "Image Index": f"{pid:08d}_{k:03d}.png",
                "Patient ID": pid,
                "Patient Age": 50,
                "Patient Gender": "M",
                "View Position": "PA",
                "Finding Labels": "|".join(labs) if labs else "No Finding",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Crea CSV oficial + listas train_val/test + PNGs sintéticos.

    Retorna (dataset_dir, df).
    """
    df = _make_entry()
    df.to_csv(tmp_path / "Data_Entry_2017.csv", index=False)

    patients = sorted(df["Patient ID"].unique())
    test_p = set(patients[: len(patients) // 5])

    (tmp_path / "test_list.txt").write_text(
        "\n".join(df[df["Patient ID"].isin(test_p)]["Image Index"]))
    (tmp_path / "train_val_list.txt").write_text(
        "\n".join(df[~df["Patient ID"].isin(test_p)]["Image Index"]))

    img_dir = tmp_path / "images_001" / "images"
    img_dir.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for name in df["Image Index"]:
        Image.fromarray(rng.integers(0, 255, (96, 128), dtype=np.uint8), "L").save(img_dir / name)

    return tmp_path, df


# ============================================================
# HELPER: crear DICOM sintético para los tests
# ============================================================
def make_dicom(arr, out_path, photometric="MONOCHROME2", bits_stored=None,
               window=None, slope=None, intercept=None):
    """Crea un DICOM válido de 8 o 16 bits a partir de un numpy array 2D."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    DX = "1.2.840.10008.5.1.4.1.1.1.1"
    arr = np.asarray(arr)
    assert arr.ndim == 2 and arr.dtype in (np.uint8, np.uint16)

    bits_alloc = 8 if arr.dtype == np.uint8 else 16
    bits_stored = bits_stored or bits_alloc

    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = DX
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(out_path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian, ds.is_implicit_VR = True, False
    ds.SOPClassUID = DX
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "DX"
    ds.PatientID = "TEST"
    ds.Rows, ds.Columns = arr.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = bits_alloc
    ds.BitsStored = bits_stored
    ds.HighBit = bits_stored - 1
    ds.PixelRepresentation = 0
    if window is not None:
        ds.WindowCenter, ds.WindowWidth = window
    if slope is not None:
        ds.RescaleSlope, ds.RescaleIntercept = slope, intercept or 0
    ds.PixelData = np.ascontiguousarray(arr).tobytes()
    ds.save_as(str(out_path))
    return str(out_path)


@pytest.fixture
def dicom_factory(tmp_path):
    """Fixture: dicom_factory(arr, name, **kwargs) → path del DICOM."""
    def _make(arr, name="x.dcm", **kw):
        return make_dicom(arr, tmp_path / name, **kw)
    return _make