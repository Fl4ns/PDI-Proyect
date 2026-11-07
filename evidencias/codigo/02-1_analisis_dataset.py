#!/usr/bin/env python3
"""
02-1_analisis_dataset.py — Estadísticas y matriz de co-ocurrencia.

Uso:
    python 02-1_analisis_dataset.py --splits-dir data/splits --out-dir reportes

Produce:
    reportes/co_ocurrencia.png        Matriz 14×14 (lift)
    reportes/distribucion.png         Nº de hallazgos por imagen
    reportes/co_ocurrencia.csv        Matriz numérica de conteos
    reportes/pares_co_ocurrencia.csv  Ranking completo de pares por lift
    reportes/estadisticas.md          Informe consolidado para la bitácora

No necesita modelo entrenado. Solo requiere pandas, numpy, matplotlib.

Dependencias: pandas numpy matplotlib
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening",
    "Cardiomegaly", "Nodule", "Mass", "Hernia",
]


# ============================================================
# CARGA
# ============================================================
def load_all_splits(splits_dir):
    """Une train + val + test para que las estadísticas sean del dataset completo."""
    dfs = []
    for name in ("train", "val", "test"):
        p = Path(splits_dir) / f"{name}.csv"
        if not p.exists():
            raise SystemExit(f"Falta {p}. Ejecute primero 02_split_dataset.py")
        d = pd.read_csv(p)
        d["_split"] = name
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


# ============================================================
# GRÁFICOS
# ============================================================
def plot_coocurrencia(counts, marginal, n_total, out_png):
    """Heatmap del 'lift' = P(A,B) / (P(A)·P(B)).
    >1 → co-ocurren más de lo esperado por azar
    <1 → se repelen
    """
    lift = counts * n_total / np.outer(marginal, marginal)
    np.fill_diagonal(lift, np.nan)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(lift, cmap="RdBu_r", vmin=0, vmax=3)
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(LABELS, fontsize=9)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if i != j and not np.isnan(lift[i, j]):
                ax.text(j, i, f"{lift[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if lift[i, j] < 2 else "white")
    plt.colorbar(im, ax=ax, label="Lift (co-ocurrencia vs. azar)")
    ax.set_title("Matriz de co-ocurrencia — ChestX-ray14", fontsize=12, pad=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close(fig)
    return lift


def plot_distribucion(n_hallazgos, out_png):
    """Cuántas patologías tiene cada imagen (0 = No Finding)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    keys = sorted(n_hallazgos.keys())
    vals = [n_hallazgos[k] for k in keys]
    total = sum(vals)
    bars = ax.bar([str(k) for k in keys], vals,
                  color=["#2e7d32"] + ["#2e6fa3"] * (len(keys) - 1))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v}\n({v / total * 100:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Nº de patologías en la imagen")
    ax.set_ylabel("Imágenes")
    ax.set_title("Distribución de hallazgos por imagen")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close(fig)


# ============================================================
# INTERPRETACIÓN DINÁMICA
# ============================================================
# Mapa de interpretaciones clínicas conocidas por par.
# Si el par está aquí, se usa el texto. Si no, se genera uno genérico
# a partir de los valores de lift y casos absolutos.
KNOWN_PAIRS = {
    frozenset(["Infiltration", "Atelectasis"]):
        "Ambos producen opacidades; muchas veces es difícil distinguirlos en la placa.",
    frozenset(["Infiltration", "Effusion"]):
        "Ambos generan opacidades difusas difíciles de separar visualmente.",
    frozenset(["Consolidation", "Effusion"]):
        "Comparten signo radiológico: opacidad por ocupación del espacio.",
    frozenset(["Emphysema", "Fibrosis"]):
        "Enfermedad pulmonar crónica; procesos fibróticos y enfisematosos coexisten.",
    frozenset(["Edema", "Pneumonia"]):
        "Edema perilesional y complicación aspirativa de la neumonía.",
    frozenset(["Pneumothorax", "Emphysema"]):
        "Rotura de bullas enfisematosas → neumotórax espontáneo.",
    frozenset(["Fibrosis", "Pleural_Thickening"]):
        "Fibrosis subpleural con engrosamiento pleural periférico.",
    frozenset(["Effusion", "Cardiomegaly"]):
        "Insuficiencia cardíaca congestiva: corazón grande + derrame pleural.",
    frozenset(["Nodule", "Mass"]):
        "Espectro continuo: un nódulo es una masa < 3 cm.",
    frozenset(["Pleural_Thickening", "Mass"]):
        "Invasión pleural por tumor periférico o mesotelioma.",
    frozenset(["Atelectasis", "Consolidation"]):
        "Atelectasia y consolidación producen signos radiológicos superpuestos.",
    frozenset(["Consolidation", "Mass"]):
        "Consolidación peritumoral por obstrucción bronquial.",
    frozenset(["Infiltration", "Edema"]):
        "El edema pulmonar ES un tipo de infiltrado (ocupación alveolar).",
    frozenset(["Atelectasis", "Effusion"]):
        "Atelectasia compresiva: el líquido colapsa el pulmón adyacente.",
    frozenset(["Fibrosis", "Pneumonia"]):
        "Procesos opuestos: fibrosis (crónico, intersticial) vs. neumonía (agudo, alveolar).",
    frozenset(["Consolidation", "Hernia"]):
        "Hernia es tan rara que la co-ocurrencia es probablemente ruido estadístico.",
    frozenset(["Pneumothorax", "Cardiomegaly"]):
        "La cardiomegalia es central; el neumotórax suele originarse en la periferia.",
    frozenset(["Pneumothorax", "Edema"]):
        "Patrones incompatibles: edema bilateral y difuso vs. neumotórax unilateral.",
    frozenset(["Edema", "Fibrosis"]):
        "Procesos opuestos: edema agudo y alveolar vs. fibrosis crónica e intersticial.",
    frozenset(["Infiltration", "Nodule"]):
        "Los nódulos suelen acompañarse de infiltrado perilesional por "
        "inflamación o hemorragia local.",
}


def _interpretar_par(par, lift, casos):
    """Devuelve una interpretación clínica para un par dado."""
    fs = frozenset(par.replace(" + ", "|").split("|"))
    if fs in KNOWN_PAIRS:
        return KNOWN_PAIRS[fs]
    # Fallback dinámico según el valor de lift
    if lift > 3:
        return f"Asociación fuerte (lift {lift:.2f}), posible relación clínica."
    if lift > 1.5:
        return f"Asociación moderada (lift {lift:.2f}); podrían compartir signos radiológicos."
    if lift > 0.7:
        return f"Aproximadamente independientes (lift {lift:.2f}); no hay relación clara."
    return f"Exclusión parcial (lift {lift:.2f}); raramente coexisten."


def _build_interpretacion(pairs_df, n):
    """Construye la sección 4 del .md a partir de los datos reales."""
    lines = []
    lines.append("\n## 4. Interpretación clínica (generada desde los datos)\n")
    lines.append(
        "Los pares con mayor lift reflejan asociaciones fisiopatológicas "
        "documentadas en la literatura. Se listan los 5 principales con su "
        "interpretación clínica y los 3 pares con menor lift.\n"
    )

    lines.append("\n### Top 5 pares por lift\n")
    lines.append("| Par | Casos | Lift | Interpretación |")
    lines.append("|---|---:|---:|---|")
    for _, row in pairs_df.head(5).iterrows():
        interp = _interpretar_par(row["Par"], row["Lift"], row["Co-ocurrencias"])
        lines.append(f"| {row['Par']} | {row['Co-ocurrencias']:,} | {row['Lift']} | {interp} |")

    lines.append("\n### Top 5 pares por casos absolutos (mayor confiabilidad estadística)\n")
    lines.append("Ordenar por casos da una idea más robusta que ordenar solo por lift, "
                 "porque los lifts extremos con pocos casos son frágiles.\n")
    lines.append("\n| Par | Casos | Lift | Interpretación |")
    lines.append("|---|---:|---:|---|")
    for _, row in pairs_df.nlargest(5, "Co-ocurrencias").iterrows():
        interp = _interpretar_par(row["Par"], row["Lift"], row["Co-ocurrencias"])
        lines.append(f"| {row['Par']} | {row['Co-ocurrencias']:,} | {row['Lift']} | {interp} |")

    lines.append("\n### Pares con menor lift (exclusión parcial)\n")
    lines.append("| Par | Casos | Lift | Interpretación |")
    lines.append("|---|---:|---:|---|")
    for _, row in pairs_df.tail(3).iterrows():
        interp = _interpretar_par(row["Par"], row["Lift"], row["Co-ocurrencias"])
        lines.append(f"| {row['Par']} | {row['Co-ocurrencias']:,} | {row['Lift']} | {interp} |")

    lines.append(
        "\n**Observación sobre clases raras**: Hernia (227 casos) y Pneumonia "
        "(1.431 casos) tienen lifts extremos porque cualquier co-ocurrencia es "
        "estadísticamente significativa con tan pocos casos. Los pares que las "
        "involucran deben interpretarse con cautela.\n"
    )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-dir", default="data/splits", type=Path)
    ap.add_argument("--out-dir", default="reportes", type=Path)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Cargar datos ---
    df = load_all_splits(a.splits_dir)
    n = len(df)
    M = df[LABELS].values.astype(int)

    print(f"Dataset cargado: {n:,} imágenes, {df['Patient ID'].nunique():,} pacientes")

    # --- Conteos ---
    marginal = M.sum(axis=0)
    freqs = marginal / n

    n_hall = M.sum(axis=1)
    dist = {k: int((n_hall == k).sum()) for k in range(0, n_hall.max() + 1)}

    counts = M.T @ M

    # --- Guardar CSV de la matriz ---
    co_df = pd.DataFrame(counts, index=LABELS, columns=LABELS)
    co_df.to_csv(a.out_dir / "co_ocurrencia.csv")

    # --- Gráficos ---
    lift = plot_coocurrencia(counts, marginal, n, a.out_dir / "co_ocurrencia.png")
    plot_distribucion(dist, a.out_dir / "distribucion.png")

    # --- Ranking de pares ---
    pairs = []
    for i in range(len(LABELS)):
        for j in range(i + 1, len(LABELS)):
            pairs.append({
                "Par": f"{LABELS[i]} + {LABELS[j]}",
                "Co-ocurrencias": int(counts[i, j]),
                "Lift": round(float(lift[i, j]), 3),
            })
    pairs_df = pd.DataFrame(pairs).sort_values("Lift", ascending=False)
    pairs_df.to_csv(a.out_dir / "pares_co_ocurrencia.csv", index=False)

    # --- Informe Markdown ---
    lines = []
    lines.append("# Análisis del dataset ChestX-ray14\n")
    lines.append(f"**Total de imágenes analizadas**: {n:,}\n")
    lines.append(f"**Pacientes únicos**: {df['Patient ID'].nunique():,}\n")

    # Sección 1
    lines.append("\n## 1. Distribución de clases\n")
    lines.append("| Patología | Positivos | Frecuencia |\n|---|---:|---:|")
    for lab in LABELS:
        p = int(marginal[LABELS.index(lab)])
        lines.append(f"| {lab} | {p:,} | {p / n * 100:.2f}% |")
    lines.append(f"\n**Sin hallazgos (todas 0)**: {dist.get(0, 0):,} "
                 f"({dist.get(0, 0) / n * 100:.2f}%)\n")

    # Sección 2
    lines.append("\n## 2. Nº de hallazgos por imagen\n")
    lines.append("| Nº patologías | Imágenes | % |\n|---|---:|---:|")
    for k in sorted(dist.keys()):
        lines.append(f"| {k} | {dist[k]:,} | {dist[k] / n * 100:.2f}% |")

    # Sección 3
    lines.append("\n## 3. Pares con mayor co-ocurrencia (lift)\n")
    lines.append("Lift > 1 → aparecen juntas más de lo esperado por azar.  \n")
    lines.append("Lift < 1 → se repelen (una excluye a la otra).\n")
    lines.append("\n| Par | Co-ocurrencias | Lift |\n|---|---:|---:|")
    for _, row in pairs_df.head(10).iterrows():
        lines.append(f"| {row['Par']} | {row['Co-ocurrencias']:,} | {row['Lift']} |")
    lines.append("\n### Pares que menos co-ocurren\n")
    lines.append("| Par | Co-ocurrencias | Lift |\n|---|---:|---:|")
    for _, row in pairs_df.tail(5).iterrows():
        lines.append(f"| {row['Par']} | {row['Co-ocurrencias']:,} | {row['Lift']} |")

    # Sección 4 — DINÁMICA
    lines.append(_build_interpretacion(pairs_df, n))

    # Sección 5
    lines.append("\n## 5. Figuras generadas\n")
    lines.append("- `co_ocurrencia.png`: matriz 14×14 (lift)")
    lines.append("- `distribucion.png`: nº de hallazgos por imagen")
    lines.append("- `co_ocurrencia.csv`: matriz numérica de conteos")
    lines.append("- `pares_co_ocurrencia.csv`: ranking completo de los 91 pares\n")

    (a.out_dir / "estadisticas.md").write_text("\n".join(lines), encoding="utf-8")

    # --- Reporte consola ---
    print(f"\nReportes generados en {a.out_dir}/")
    print(f"  {a.out_dir}/estadisticas.md")
    print(f"  {a.out_dir}/co_ocurrencia.png")
    print(f"  {a.out_dir}/distribucion.png")
    print(f"  {a.out_dir}/co_ocurrencia.csv")
    print(f"  {a.out_dir}/pares_co_ocurrencia.csv")

    print("\n--- Top 5 pares por lift ---")
    for _, r in pairs_df.head(5).iterrows():
        print(f"  {r['Par']:<50s} lift={r['Lift']:>6.3f}  casos={r['Co-ocurrencias']}")

    print("\n--- Top 5 pares por casos absolutos ---")
    for _, r in pairs_df.nlargest(5, "Co-ocurrencias").iterrows():
        print(f"  {r['Par']:<50s} casos={r['Co-ocurrencias']:>6d}  lift={r['Lift']:.3f}")


if __name__ == "__main__":
    main()
