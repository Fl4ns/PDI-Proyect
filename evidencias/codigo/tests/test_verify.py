"""Tests para 01_verificar_dataset.py."""
import pandas as pd


def test_pu1_conteos_ok(synthetic_dataset, verify_module, capsys):
    d, _ = synthetic_dataset
    df = pd.read_csv(d / "Data_Entry_2017.csv")
    df_multi = verify_module.add_multihot(df)

    # El dataset sintético no es 112.120, así que el verify global falla
    # pero el multihot debe funcionar:
    assert len(df_multi) == len(df)
    for lab in verify_module.LABELS:
        assert lab in df_multi.columns
        assert df_multi[lab].isin([0, 1]).all()


def test_pu1_add_multihot_con_multiples_etiquetas(verify_module):
    df = pd.DataFrame({
        "Finding Labels": ["Atelectasis|Effusion", "No Finding", "Hernia"],
    })
    out = verify_module.add_multihot(df)
    assert out.loc[0, "Atelectasis"] == 1
    assert out.loc[0, "Effusion"] == 1
    assert out.loc[0, "Hernia"] == 0
    assert out.loc[1].drop("Finding Labels").sum() == 0  # No Finding → todo 0
    assert out.loc[2, "Hernia"] == 1