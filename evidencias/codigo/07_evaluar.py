#!/usr/bin/env python3
"""
07_evaluar.py — Evalúa el modelo entrenado sobre el conjunto de prueba (T2.2.2).

Calcula AUC-ROC por patología y el promedio (RNF8 / prueba PM2) y guarda tablas y curvas ROC.

Uso:
    python 07_evaluar.py --images-dir /ruta/images
    python 07_evaluar.py --images-dir /ruta/images --limit 500     # prueba rápida

Requiere:
    data/splits/test.csv           (generado por 02_split_dataset.py)
    models/modelo_base.keras       (generado por 05_entrenar.py)

Produce (en --out-dir, por defecto reportes/):
    auc_test.csv   AUC-ROC por patología
    auc_test.md    tabla en Markdown + promedio
    roc_test.png   curvas ROC de las 14 patologías
"""
import argparse
import importlib.util
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

# Reutiliza el mismo preprocesamiento y orden de etiquetas del entrenamiento
_spec = importlib.util.spec_from_file_location(
    "entrenar", Path(__file__).with_name("05_entrenar.py"))
entrenar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(entrenar)
LABELS = entrenar.LABELS


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--splits-dir", default=Path("data/splits"), type=Path)
    ap.add_argument("--model", default=Path("models/modelo_base.keras"), type=Path)
    ap.add_argument("--out-dir", default=Path("reportes"), type=Path)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="usar solo las primeras N filas")
    a = ap.parse_args()

    test_csv = a.splits_dir / "test.csv"
    for p in (test_csv, a.model):
        if not p.exists():
            raise SystemExit(f"Falta {p}")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(test_csv)
    if a.limit:
        df = df.head(a.limit)
    y_true = df[LABELS].values.astype("float32")

    ds = entrenar.make_dataset(test_csv, a.images_dir, a.batch_size,
                               training=False, limit=a.limit)
    model = keras.models.load_model(a.model, compile=False)
    y_pred = model.predict(ds, verbose=1).astype("float32")

    rows = []
    plt.figure(figsize=(7, 6))
    for i, lab in enumerate(LABELS):
        n_pos = int(y_true[:, i].sum())
        if n_pos == 0 or n_pos == len(y_true):
            rows.append((lab, n_pos, np.nan))
            continue
        auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        plt.plot(fpr, tpr, lw=1, label=f"{lab} ({auc:.2f})")
        rows.append((lab, n_pos, auc))

    res = pd.DataFrame(rows, columns=["Patologia", "Positivos_test", "AUC_ROC"])
    mean_auc = res["AUC_ROC"].mean()
    res.to_csv(a.out_dir / "auc_test.csv", index=False)

    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("Tasa de falsos positivos"); plt.ylabel("Tasa de verdaderos positivos")
    plt.title(f"Curvas ROC – conjunto de prueba (AUC promedio {mean_auc:.3f})")
    plt.legend(fontsize=7, loc="lower right"); plt.tight_layout()
    plt.savefig(a.out_dir / "roc_test.png", dpi=150)

    with open(a.out_dir / "auc_test.md", "w", encoding="utf-8") as f:
        f.write("# AUC-ROC por patología (conjunto de prueba)\n\n")
        f.write(f"Imágenes evaluadas: {len(df)}  \nModelo: `{a.model}`\n\n")
        f.write("| Patología | Positivos | AUC-ROC |\n|---|---:|---:|\n")
        for lab, n, auc in rows:
            f.write(f"| {lab} | {n} | {'—' if np.isnan(auc) else f'{auc:.3f}'} |\n")
        f.write(f"| **Promedio** | | **{mean_auc:.3f}** |\n")
    print(res.to_string(index=False))
    print(f"\nAUC-ROC promedio: {mean_auc:.4f}")


if __name__ == "__main__":
    main()
