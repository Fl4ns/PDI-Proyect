"""Tests para 02_split_dataset.py."""
import pandas as pd
import pytest


def test_pm1_split_sin_pacientes_repetidos(synthetic_dataset, tmp_path, split_module):
    d, _ = synthetic_dataset
    out = tmp_path / "splits"
    train, val, test = split_module.make_splits(
        d / "Data_Entry_2017.csv",
        d / "train_val_list.txt",
        d / "test_list.txt",
        out, verbose=False,
    )

    assert (out / "train.csv").exists()
    assert (out / "val.csv").exists()
    assert (out / "test.csv").exists()

    p_train = set(train["Patient ID"])
    p_val = set(val["Patient ID"])
    p_test = set(test["Patient ID"])
    assert not (p_train & p_val)
    assert not (p_train & p_test)
    assert not (p_val & p_test)


def test_pm1_check_no_overlap_detecta_repetidos(split_module):
    a = pd.DataFrame({"Patient ID": [1, 2]})
    b = pd.DataFrame({"Patient ID": [2, 3]})
    with pytest.raises(AssertionError):
        split_module.check_no_overlap(a, b)


def test_columnas_multihot_en_orden(synthetic_dataset, tmp_path, split_module):
    d, _ = synthetic_dataset
    train, _, _ = split_module.make_splits(
        d / "Data_Entry_2017.csv",
        d / "train_val_list.txt",
        d / "test_list.txt",
        tmp_path / "o", verbose=False,
    )
    cols = [c for c in train.columns if c in split_module.LABELS]
    assert cols == split_module.LABELS


def test_split_proporcion_val(synthetic_dataset, tmp_path, split_module):
    d, _ = synthetic_dataset
    train, val, test = split_module.make_splits(
        d / "Data_Entry_2017.csv",
        d / "train_val_list.txt",
        d / "test_list.txt",
        tmp_path / "o", verbose=False, val_fraction=0.15,
    )
    total_tv = len(train) + len(val)
    assert 0.10 <= len(val) / total_tv <= 0.20