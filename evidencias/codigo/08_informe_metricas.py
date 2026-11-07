#!/usr/bin/env python3
"""
08_informe_metricas.py — Genera el "Informe de métricas" (entregable de T2.2.2).

Toma el resultado de 07_evaluar.py y lo compara con la literatura (Wang et al. 2017 y
CheXNet, Rajpurkar et al. 2017). Opcionalmente agrega las curvas de aprendizaje de los logs.

Uso:
    python 08_informe_metricas.py
    python 08_informe_metricas.py --auc reportes/auc_test.csv --logs models --out reportes/informe_metricas.md
"""
import argparse
from pathlib import Path
import pandas as pd

# AUC-ROC reportados en la literatura sobre ChestX-ray14 (CheXNet, tabla 2: incluye a Wang et al. 2017)
REF = {  # patología: (Wang 2017, CheXNet 2017)
    "Atelectasis": (0.716, 0.8094), "Cardiomegaly": (0.807, 0.9248), "Effusion": (0.784, 0.8638),
    "Infiltration": (0.609, 0.7345), "Mass": (0.706, 0.8676), "Nodule": (0.671, 0.7802),
    "Pneumonia": (0.633, 0.7680), "Pneumothorax": (0.806, 0.8887), "Consolidation": (0.708, 0.7901),
    "Edema": (0.835, 0.8878), "Emphysema": (0.815, 0.9371), "Fibrosis": (0.769, 0.8047),
    "Pleural_Thickening": (0.708, 0.8062), "Hernia": (0.767, 0.9164),
}
CHEXNET_MEAN = 0.8414


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--auc", default=Path("reportes/auc_test.csv"), type=Path)
    ap.add_argument("--logs", default=Path("models"), type=Path)
    ap.add_argument("--out", default=Path("reportes/informe_metricas.md"), type=Path)
    a = ap.parse_args()

    df = pd.read_csv(a.auc)
    df["Wang"] = df["Patologia"].map(lambda p: REF[p][0])
    df["CheXNet"] = df["Patologia"].map(lambda p: REF[p][1])
    df["Dif_CheXNet"] = df["AUC_ROC"] - df["CheXNet"]
    ok = df.dropna(subset=["AUC_ROC"])
    mean_auc = ok["AUC_ROC"].mean()
    best, worst = ok.loc[ok["AUC_ROC"].idxmax()], ok.loc[ok["AUC_ROC"].idxmin()]
    below07 = ok[ok["AUC_ROC"] < 0.7]["Patologia"].tolist()
    beat_wang = int((ok["AUC_ROC"] > ok["Wang"]).sum())
    rare = ok.nsmallest(3, "Positivos_test")

    L = ["# Informe de métricas — evaluación en el conjunto de prueba (T2.2.2)", ""]
    L += ["## 1. Resumen", "",
          f"- AUC-ROC promedio de las {len(ok)} patologías evaluadas: **{mean_auc:.3f}** "
          f"(CheXNet: {CHEXNET_MEAN:.3f}; diferencia {mean_auc - CHEXNET_MEAN:+.3f}).",
          f"- Mejor patología: **{best['Patologia']}** ({best['AUC_ROC']:.3f}). "
          f"Peor: **{worst['Patologia']}** ({worst['AUC_ROC']:.3f}).",
          f"- Patologías con AUC menor a 0,70: {', '.join(below07) if below07 else 'ninguna'}.",
          f"- Supera a Wang et al. (2017) en {beat_wang} de {len(ok)} patologías.", ""]
    L += ["## 2. AUC-ROC por patología", "",
          "| Patología | Positivos en test | AUC-ROC (este trabajo) | Wang 2017 | CheXNet 2017 | Dif. vs CheXNet |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in df.itertuples():
        mine = "—" if pd.isna(r.AUC_ROC) else f"{r.AUC_ROC:.3f}"
        dif = "—" if pd.isna(r.Dif_CheXNet) else f"{r.Dif_CheXNet:+.3f}"
        L.append(f"| {r.Patologia} | {r.Positivos_test} | {mine} | {r.Wang:.3f} | {r.CheXNet:.3f} | {dif} |")
    L += [f"| **Promedio** | | **{mean_auc:.3f}** | 0,745 | {CHEXNET_MEAN:.3f} | {mean_auc - CHEXNET_MEAN:+.3f} |", ""]
    L += ["Curvas ROC: `reportes/roc_test.png`.", ""]

    curves = []
    for n in (1, 2):
        p = a.logs / f"log_etapa{n}.csv"
        if p.exists():
            lg = pd.read_csv(p)
            curves.append((n, len(lg), lg["val_auc"].max(), lg["val_auc"].iloc[-1]))
    if curves:
        L += ["## 3. Entrenamiento (validación)", "", "| Etapa | Épocas | Mejor val_auc | val_auc final |", "|---|---:|---:|---:|"]
        L += [f"| {n} | {e} | {m:.3f} | {f:.3f} |" for n, e, m, f in curves]
        L.append("")

    L += ["## 4. Observaciones", "",
          f"- Las clases con menos positivos en test ({', '.join(f'{r.Patologia} ({r.Positivos_test})' for r in rare.itertuples())}) "
          "tienen un AUC menos estable: pocos casos positivos hacen variar mucho la métrica.",
          "- Se usó pérdida ponderada por clase para el desbalance, por eso se reporta el AUC por patología y no solo el promedio.",
          "- Las etiquetas del dataset se extrajeron de informes con NLP (exactitud estimada > 90 %), lo que pone un techo al desempeño.",
          "", "## 5. Limitaciones de la comparación", "",
          "- Las cifras de la literatura vienen de otros entrenamientos (más épocas, otro protocolo de evaluación); "
          "la comparación es referencial, no una prueba de que un modelo sea mejor que otro.",
          "- Evaluación solo sobre el conjunto de prueba del propio dataset, sin datos de otros hospitales ni radiólogos.",
          "", "## 6. Cumplimiento del requisito RNF8", "",
          f"Meta inicial: AUC-ROC promedio cercano al de la literatura. Resultado: {mean_auc:.3f} frente a "
          f"{CHEXNET_MEAN:.3f} (CheXNet) y 0,745 (Wang et al.).", "",
          "## Referencias", "",
          "- Wang et al., ChestX-ray8, CVPR 2017. - Rajpurkar et al., CheXNet, arXiv:1711.05225, 2017.", ""]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L), encoding="utf-8")
    print(f"Informe guardado en {a.out} (AUC promedio {mean_auc:.3f})")


if __name__ == "__main__":
    main()
