# Copyright (c) 2026 Hiba Arbel. Released under the MIT License (see the LICENSE file).
"""
Analysis of the CrossGated gates, class by class.

Reads results_gates/<dataset>_gates.csv (mean gate value per class and per fold)
and produces:
  - <dataset>_gates_par_classe.csv: table per class
  - <dataset>_gates_par_classe.pdf / .png: figure for the article

Measure used, for each class:
  difference = mean(AST gate) - mean(PANNs gate)
  > 0: the class lets more of the AST branch through
  < 0: the class lets more of the PANNs branch through
The means are weighted by the number of test clips of each fold.

Usage:
  python analyse_gates.py urban
  python analyse_gates.py esc50 --esc50-meta ESC-50/meta/esc50.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

URBAN_NAMES = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
               "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

# same colours as the architecture figure
COLOR_PANNS = "#3B7DD8"
COLOR_AST = "#D9741F"
INK, MUTED, GRID = "#1F2328", "#5B616B", "#D9DCE1"


def class_names(dataset, meta):
    """Dictionary {class number: class name} for the dataset (ESC-50 names come from its metadata file)."""
    if dataset == "urban":
        return dict(enumerate(URBAN_NAMES))
    if meta and Path(meta).exists():
        m = pd.read_csv(meta)
        return dict(m.drop_duplicates("target").set_index("target")["category"].sort_index())
    print("Noms des classes ESC-50 introuvables : utilise --esc50-meta ESC-50/meta/esc50.csv "
          "(les classes seront numérotées).")
    return {}


def summarize(df, names):
    """For each class: mean gate of each branch (weighted by the number of test clips) and their difference."""
    rows = []
    for c, d in df.groupby("class"):
        w = d["n"].to_numpy(float)
        gp = np.average(d["gate_panns"], weights=w)
        gs = np.average(d["gate_ast"], weights=w)
        diff_per_fold = (d["gate_ast"] - d["gate_panns"]).to_numpy()
        rows.append({"class": int(c), "name": names.get(int(c), str(int(c))),
                     "gate_panns": gp, "gate_ast": gs, "ecart_ast_moins_panns": gs - gp,
                     "ecart_min_plis": diff_per_fold.min(), "ecart_max_plis": diff_per_fold.max(),
                     "n_test": int(w.sum()), "n_plis": d["fold"].nunique()})
    return pd.DataFrame(rows).sort_values("ecart_ast_moins_panns").reset_index(drop=True)


def plot(tab, dataset, out):
    """Bar chart of the gate difference per class, with the range between folds; save as PDF and PNG."""
    n = len(tab)
    height = max(2.6, 0.24 * n + 1.0)
    fig, ax = plt.subplots(figsize=(6.4, height))
    y = np.arange(n)
    vals = tab["ecart_ast_moins_panns"].to_numpy()
    colors = [COLOR_AST if v > 0 else COLOR_PANNS for v in vals]
    ax.barh(y, vals, height=0.62, color=colors, edgecolor="white", linewidth=1.0, zorder=3)
    # range between folds: from the smallest to the largest difference observed
    ax.hlines(y, tab["ecart_min_plis"], tab["ecart_max_plis"], color=MUTED, linewidth=1.0, zorder=4)
    ax.axvline(0, color=INK, linewidth=0.9, zorder=5)
    ax.set_yticks(y, [s.replace("_", " ") for s in tab["name"]], fontsize=8.5, color=INK)
    ax.set_xlabel("Mean AST gate − mean PANNs gate", fontsize=9, color=INK)
    ax.tick_params(axis="x", labelsize=8, colors=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    lim = max(abs(tab["ecart_min_plis"].min()), abs(tab["ecart_max_plis"].max()), 1e-3) * 1.12
    ax.set_xlim(-lim, lim)
    ax.text(-lim, n - 0.2, "relies more on PANNs", ha="left", va="bottom", fontsize=8.5, color=COLOR_PANNS)
    ax.text(lim, n - 0.2, "relies more on AST", ha="right", va="bottom", fontsize=8.5, color=COLOR_AST)
    ax.set_ylim(-0.7, n + 0.6)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main():
    """Read the gate values written by crossgated.py, summarise them per class, write the table and the figure."""
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["esc50", "urban"])
    ap.add_argument("--gates", default=None, help="Gate file (default: results_gates/<dataset>_gates.csv)")
    ap.add_argument("--esc50-meta", default="ESC-50/meta/esc50.csv")
    ap.add_argument("--out", default="results_gates")
    args = ap.parse_args()

    path = Path(args.gates or f"results_gates/{args.dataset}_gates.csv")
    df = pd.read_csv(path, dtype=str)
    df = df[df["seed"] != "seed"]                               # possible repeated headers
    for c in ["seed", "fold", "class", "gate_panns", "gate_ast", "n"]:
        df[c] = pd.to_numeric(df[c])
    df = df.drop_duplicates(subset=["seed", "fold", "class"])

    tab = summarize(df, class_names(args.dataset, args.esc50_meta))
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{args.dataset}_gates_par_classe"
    tab.to_csv(stem.with_suffix(".csv"), index=False, float_format="%.4f")
    plot(tab, args.dataset, stem)

    pd.set_option("display.width", 120)
    print(f"{path} : {df['fold'].nunique()} plis, {len(tab)} classes, graine(s) {sorted(df['seed'].unique())}")
    print(f"Porte PANNs moyenne : {np.average(df['gate_panns'], weights=df['n']):.3f} | "
          f"porte AST moyenne : {np.average(df['gate_ast'], weights=df['n']):.3f}")
    print(tab[["name", "gate_panns", "gate_ast", "ecart_ast_moins_panns"]].round(3).to_string(index=False))
    print(f"\nFichiers : {stem.with_suffix('.csv')}, {stem.with_suffix('.pdf')}, {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()