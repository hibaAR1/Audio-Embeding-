# Copyright (c) 2026 Hiba Arbel. Released under the MIT License (see the LICENSE file).
"""
Figure 2: per-class accuracy of PANNs only, AST only and CrossGated (article style).

Reads results_preds/esc50_preds.csv and results_preds/urban_preds.csv
(created by crossgated.py --save-preds, columns: model, seed, fold, index, y_true, y_pred).
The classes are sorted by (AST accuracy - PANNs accuracy): on the left, the classes where
PANNs is better; on the right, those where AST is better.

Usage:
  python fig_classwise.py
  python fig_classwise.py --dir results_preds --out classwise
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (model name in the prediction file, legend label, line style, marker)
MODELS = [("panns_only", "PANNs only", "--", "s"),
          ("ast_only", "AST only", ":", "^"),
          ("crossgated", "CrossGated", "-", "o")]


def load(path):
    """Read a prediction file, drop damaged or duplicate rows, and return integer columns."""
    df = pd.read_csv(path, dtype=str, on_bad_lines="skip")
    df = df[df["model"] != "model"]
    for c in ["seed", "fold", "index", "y_true", "y_pred"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["seed", "fold", "index", "y_true", "y_pred"])
    for c in ["seed", "fold", "index", "y_true", "y_pred"]:
        df[c] = df[c].astype(int)
    return df.drop_duplicates(subset=["model", "seed", "fold", "index"], keep="last")


def per_class(df, model):
    """Accuracy (%) of one model for each class, over all folds and seeds."""
    d = df[df["model"] == model]
    if d.empty:
        raise SystemExit(f"Pas de prédictions pour « {model} ».")
    return d.assign(ok=(d["y_true"] == d["y_pred"])).groupby("y_true")["ok"].mean() * 100


def oracle_per_class(df):
    """Accuracy per class if the correct answer were taken whenever PANNs OR AST is right."""
    keys = ["seed", "fold", "index"]
    p = df[df["model"] == "panns_only"][keys + ["y_true", "y_pred"]]
    a = df[df["model"] == "ast_only"][keys + ["y_true", "y_pred"]]
    m = p.merge(a, on=keys, suffixes=("_p", "_a"))
    m["ok"] = (m["y_true_p"] == m["y_pred_p"]) | (m["y_true_a"] == m["y_pred_a"])
    return m.groupby("y_true_p")["ok"].mean() * 100


def panel(ax, df, title, ticklabels=None):
    """Draw one panel: per-class accuracy of the three models and the upper bound, classes sorted by AST minus PANNs."""
    acc = {m: per_class(df, m) for m, *_ in MODELS}
    # sort the classes from "PANNs better" (left) to "AST better" (right)
    order = (acc["ast_only"] - acc["panns_only"]).sort_values().index   # PANNs meilleur -> AST meilleur
    x = np.arange(len(order))
    for m, lab, ls, mk in MODELS:
        ax.plot(x, acc[m].loc[order].values, color="black", linestyle=ls, marker=mk,
                markersize=3 if len(order) > 20 else 4, linewidth=0.9, markerfacecolor="white",
                markeredgewidth=0.7, label=lab)
    orc = oracle_per_class(df)
    ax.plot(x, orc.loc[order].values, color="0.55", linestyle="-.", marker="v",
            markersize=3 if len(order) > 20 else 4, linewidth=0.9, markerfacecolor="white",
            markeredgewidth=0.7, label="Upper bound (either encoder correct)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(max(0, min(a.min() for a in acc.values()) - 5), 103)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.grid(axis="y", linewidth=0.3, color="0.8")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if ticklabels:
        ax.set_xticks(x)
        ax.set_xticklabels([ticklabels[int(c)] for c in order], rotation=40, ha="right", fontsize=6)
    else:
        ax.set_xticks([])
    ax.text(0.0, 1.04, title, transform=ax.transAxes, fontsize=8, va="bottom")


def main():
    """Draw Figure 2 of the article (two panels) from the saved predictions and save it as PDF and PNG."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_preds")
    ap.add_argument("--out", default="classwise")
    args = ap.parse_args()
    d = Path(args.dir)

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 7, "axes.linewidth": 0.6})
    urban = ["air conditioner", "car horn", "children playing", "dog bark", "drilling",
             "engine idling", "gun shot", "jackhammer", "siren", "street music"]
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.0), gridspec_kw={"height_ratios": [1, 1.25]})
    panel(axes[0], load(d / "esc50_preds.csv"), "(a) ESC-50")
    axes[0].set_xlabel("Classes, sorted from PANNs-favoured to AST-favoured", fontsize=6.5)
    panel(axes[1], load(d / "urban_preds.csv"), "(b) UrbanSound8K", dict(enumerate(urban)))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", fontsize=6.5, frameon=False, ncol=2, handlelength=2.4)
    fig.tight_layout(h_pad=1.2, rect=(0, 0, 1, 0.95))
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight")
    fig.savefig(f"{args.out}.png", dpi=300, bbox_inches="tight")
    print(f"Écrit : {args.out}.pdf et {args.out}.png")


if __name__ == "__main__":
    main()
