"""
Analyse des prédictions : POURQUOI la fusion aide sur UrbanSound8K et pas sur ESC-50.

Lit le fichier de prédictions écrit par :
  python crossgated.py <dataset> --models panns_only ast_only crossgated --save-preds --out results_preds

Produit deux analyses.

1) Complémentarité (réponse au « pourquoi ? »)
   Chaque extrait de test est rangé dans un des 4 cas, selon PANNs seul et AST seul :
     les deux justes | seul AST juste | seul PANNs juste | les deux faux
   « Seul PANNs juste » = ce que la fusion peut gagner par rapport à AST seul.
   Oracle = au moins un des deux juste = plafond de la fusion de ces deux modèles.
   On donne aussi ce que CrossGated fait de ces cas :
     récupérés = part des cas « seul PANNs juste » que CrossGated classe bien
     perdus    = part des cas « seul AST juste » que CrossGated classe mal
   Les pourcentages sont calculés par pli, moyennés sur les plis puis sur les graines
   (même calcul que le tableau principal de l'article).

2) Gain par classe
   Accuracy de chaque classe pour CrossGated moins celle d'AST seul
   (tous les plis réunis, moyenne des graines ; les traits vont de la plus petite
   à la plus grande valeur entre graines).
   Si results_gates/<dataset>_gates_par_classe.csv existe, la figure met le gain
   à côté de l'écart des portes, avec les classes dans le même ordre.

Usage :
  python analyse_predictions.py urban
  python analyse_predictions.py esc50 --esc50-meta ESC-50/meta/esc50.csv

Sorties dans results_preds/ :
  <dataset>_complementarite.csv
  <dataset>_gain_par_classe.csv
  <dataset>_gain_par_classe.pdf / .png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyse_gates import class_names, COLOR_PANNS, COLOR_AST, INK, MUTED, GRID

COLOR_GAIN, COLOR_LOSS = "#2E7D5B", "#B5475A"     # gain / perte de CrossGated par rapport à AST seul
NEEDED = ["panns_only", "ast_only", "crossgated"]


COLS = ["model", "seed", "fold", "index", "y_true", "y_pred"]


def backup_of(path):
    """Même règle que crossgated.py : copie de secours sur C:, dossier crossgated_backup."""
    return Path.home() / "crossgated_backup" / f"{path.parent.resolve().name}_{path.name}"


def read_raw_preds(path, verbose=True):
    """Toutes les prédictions connues : fichier du projet d'abord, puis la copie de secours sur C:.
    Les lignes coupées (disque plein ou débranché) sont ignorées."""
    frames = []
    for p in [path, backup_of(path)]:
        if not p.exists():
            continue
        try:
            d = pd.read_csv(p, dtype=str, on_bad_lines="skip")
            if not all(c in d.columns for c in COLS):          # fichier sans en-tête
                d = pd.read_csv(p, dtype=str, header=None, names=COLS, on_bad_lines="skip")
        except (pd.errors.EmptyDataError, OSError):
            continue
        frames.append(d[COLS])
        if verbose:
            print(f"Lu : {p} ({len(d)} lignes)")
    if not frames:
        raise SystemExit(f"Aucun fichier de prédictions trouvé ({path} ou {backup_of(path)}).")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["model"].isin(NEEDED)]                            # en-têtes répétés, lignes coupées
    for c in COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=COLS[1:])
    for c in COLS[1:]:
        df[c] = df[c].astype(int)
    return df.drop_duplicates(subset=["model", "seed", "fold", "index"], keep="first")


def load_preds(path):
    df = read_raw_preds(path)
    missing = [m for m in NEEDED if m not in set(df["model"])]
    if missing:
        raise SystemExit(f"Modèles absents de {path} : {missing}. Lance crossgated.py avec "
                         f"--models panns_only ast_only crossgated --save-preds.")
    # une colonne « correct » par modèle, une ligne par (graine, pli, extrait)
    df["ok"] = (df["y_true"] == df["y_pred"]).astype(float)
    wide = df.pivot_table(index=["seed", "fold", "index", "y_true"], columns="model",
                          values="ok", aggfunc="first").reset_index()
    complete = wide.dropna(subset=NEEDED)
    if len(complete) < len(wide):
        print(f"Attention : {len(wide) - len(complete)} lignes incomplètes ignorées "
              "(un des 3 modèles n'a pas de prédiction pour ce pli / cette graine). "
              "Lance : python complete_preds.py <dataset>")
    seeds = sorted(complete["seed"].unique())
    print(f"{path} : graine(s) {seeds}, {complete['fold'].nunique()} plis, "
          f"{complete['index'].nunique()} extraits")
    return complete


def complementarity(w):
    p, a, c = w["panns_only"] == 1, w["ast_only"] == 1, w["crossgated"] == 1
    only_p, only_a = p & ~a, a & ~p
    return pd.Series({
        "both_correct": (p & a).mean() * 100,
        "only_ast_correct": only_a.mean() * 100,
        "only_panns_correct": only_p.mean() * 100,
        "both_wrong": (~p & ~a).mean() * 100,
        "panns_only_acc": p.mean() * 100,
        "ast_only_acc": a.mean() * 100,
        "oracle": (p | a).mean() * 100,
        "crossgated_acc": c.mean() * 100,
        "n_only_panns": int(only_p.sum()),
        "n_only_ast": int(only_a.sum()),
        "recovered": (c & only_p).sum(),
        "lost": (~c & only_a).sum(),
    })


def complementarity_table(w):
    per_fold = w.groupby(["seed", "fold"]).apply(complementarity, include_groups=False)
    pct_cols = ["both_correct", "only_ast_correct", "only_panns_correct", "both_wrong",
                "panns_only_acc", "ast_only_acc", "oracle", "crossgated_acc"]
    per_seed = per_fold.groupby("seed")[pct_cols].mean()           # moyenne des plis
    out = per_seed.mean().to_frame("mean")                         # puis moyenne des graines
    out["std_seeds"] = per_seed.std(ddof=0)
    counts = per_fold.groupby("seed")[["n_only_panns", "n_only_ast", "recovered", "lost"]].sum()
    tot = counts.sum()
    extra = pd.DataFrame({
        "mean": [100 * tot["recovered"] / max(tot["n_only_panns"], 1),
                 100 * tot["lost"] / max(tot["n_only_ast"], 1)],
        "std_seeds": [(100 * counts["recovered"] / counts["n_only_panns"].clip(lower=1)).std(ddof=0),
                      (100 * counts["lost"] / counts["n_only_ast"].clip(lower=1)).std(ddof=0)],
    }, index=["recovered_of_only_panns", "lost_of_only_ast"])
    return pd.concat([out, extra])


def per_class_gain(w, names):
    rows = []
    for c, d in w.groupby("y_true"):
        by_seed = d.groupby("seed")[["crossgated", "ast_only", "panns_only"]].mean() * 100
        gain = by_seed["crossgated"] - by_seed["ast_only"]
        rows.append({"class": int(c), "name": names.get(int(c), str(int(c))),
                     "ast_only_acc": by_seed["ast_only"].mean(),
                     "panns_only_acc": by_seed["panns_only"].mean(),
                     "crossgated_acc": by_seed["crossgated"].mean(),
                     "gain_vs_ast": gain.mean(), "gain_min_seeds": gain.min(), "gain_max_seeds": gain.max(),
                     "panns_minus_ast": (by_seed["panns_only"] - by_seed["ast_only"]).mean(),
                     "n_test": int(d["index"].nunique())})
    return pd.DataFrame(rows)


def style(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(axis="x", labelsize=8, colors=MUTED)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.axvline(0, color=INK, linewidth=0.9, zorder=5)


def plot(gain, gates, out):
    if gates is not None:                     # même ordre que la figure des portes
        order = gates.sort_values("ecart_ast_moins_panns")["class"].tolist()
        gain = gain.set_index("class").loc[order].reset_index()
        gates = gates.set_index("class").loc[order].reset_index()
    else:
        gain = gain.sort_values("gain_vs_ast").reset_index(drop=True)
    n = len(gain)
    y = np.arange(n)
    ncols = 2 if gates is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols + 1.4, max(2.6, 0.24 * n + 1.1)),
                             sharey=True, squeeze=False)
    axes = axes[0]

    ax = axes[0]
    vals = gain["gain_vs_ast"].to_numpy()
    ax.barh(y, vals, height=0.62, color=[COLOR_GAIN if v >= 0 else COLOR_LOSS for v in vals],
            edgecolor="white", linewidth=1.0, zorder=3)
    if gain["gain_min_seeds"].ne(gain["gain_max_seeds"]).any():
        ax.hlines(y, gain["gain_min_seeds"], gain["gain_max_seeds"], color=MUTED, linewidth=1.0, zorder=4)
    lim = max(abs(gain["gain_min_seeds"].min()), abs(gain["gain_max_seeds"].max()), 1.0) * 1.12
    ax.set_xlim(-lim, lim)
    ax.set_yticks(y, [s.replace("_", " ") for s in gain["name"]], fontsize=8.5, color=INK)
    ax.set_xlabel("Accuracy gain over AST only (points)", fontsize=9, color=INK)
    ax.set_title("(a) CrossGated − AST only", fontsize=9, color=INK, loc="left")
    style(ax)

    if gates is not None:
        ax = axes[1]
        vals = gates["ecart_ast_moins_panns"].to_numpy()
        ax.barh(y, vals, height=0.62, color=[COLOR_AST if v > 0 else COLOR_PANNS for v in vals],
                edgecolor="white", linewidth=1.0, zorder=3)
        ax.hlines(y, gates["ecart_min_plis"], gates["ecart_max_plis"], color=MUTED, linewidth=1.0, zorder=4)
        lim = max(abs(gates["ecart_min_plis"].min()), abs(gates["ecart_max_plis"].max()), 1e-3) * 1.12
        ax.set_xlim(-lim, lim)
        ax.set_xlabel("Mean AST gate − mean PANNs gate", fontsize=9, color=INK)
        ax.set_title("(b) Gate difference", fontsize=9, color=INK, loc="left")
        style(ax)

    ax.set_ylim(-0.7, n - 0.3)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["esc50", "urban"])
    ap.add_argument("--preds", default=None, help="défaut : results_preds/<dataset>_preds.csv")
    ap.add_argument("--gates", default=None, help="défaut : results_gates/<dataset>_gates_par_classe.csv")
    ap.add_argument("--esc50-meta", default="ESC-50/meta/esc50.csv")
    ap.add_argument("--out", default="results_preds")
    args = ap.parse_args()

    w = load_preds(Path(args.preds or f"results_preds/{args.dataset}_preds.csv"))
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) complémentarité
    comp = complementarity_table(w)
    comp.to_csv(out_dir / f"{args.dataset}_complementarite.csv", float_format="%.2f")
    labels = {
        "both_correct": "Les deux justes", "only_ast_correct": "Seul AST juste",
        "only_panns_correct": "Seul PANNs juste  <- ce que la fusion peut gagner",
        "both_wrong": "Les deux faux", "panns_only_acc": "PANNs seul (accuracy)",
        "ast_only_acc": "AST seul (accuracy)", "oracle": "Oracle (au moins un juste) = plafond",
        "crossgated_acc": "CrossGated (accuracy)",
        "recovered_of_only_panns": "Cas « seul PANNs juste » récupérés par CrossGated",
        "lost_of_only_ast": "Cas « seul AST juste » perdus par CrossGated",
    }
    print(f"\n=== COMPLÉMENTARITÉ ({args.dataset}) : % des extraits de test ===")
    for k, lab in labels.items():
        print(f"  {lab:<52} {comp.loc[k, 'mean']:6.1f} %  (± {comp.loc[k, 'std_seeds']:.1f} entre graines)")

    # 2) gain par classe
    names = class_names(args.dataset, args.esc50_meta)
    gain = per_class_gain(w, names)
    gain.sort_values("gain_vs_ast").to_csv(out_dir / f"{args.dataset}_gain_par_classe.csv",
                                           index=False, float_format="%.2f")
    gates_path = Path(args.gates or f"results_gates/{args.dataset}_gates_par_classe.csv")
    gates = pd.read_csv(gates_path) if gates_path.exists() else None
    if gates is None:
        print(f"\n({gates_path} introuvable : la figure montre seulement le gain. "
              "Lance d'abord analyse_gates.py pour avoir les deux panneaux.)")
    stem = out_dir / f"{args.dataset}_gain_par_classe"
    plot(gain, gates, stem)

    print(f"\n=== GAIN PAR CLASSE ({args.dataset}) : CrossGated − AST seul, en points ===")
    show = gain.sort_values("gain_vs_ast", ascending=False)
    for r in show.itertuples():
        print(f"  {r.name:<28} AST {r.ast_only_acc:5.1f} | PANNs {r.panns_only_acc:5.1f} | "
              f"CrossGated {r.crossgated_acc:5.1f} | gain {r.gain_vs_ast:+5.1f}")

    if gates is not None:
        from scipy.stats import spearmanr
        m = gain.merge(gates[["class", "ecart_ast_moins_panns"]], on="class")
        rho, pval = spearmanr(m["gain_vs_ast"], -m["ecart_ast_moins_panns"])
        print(f"\nLien gain / porte PANNs plus ouverte : Spearman rho = {rho:+.2f} (p = {pval:.3f}, "
              f"{len(m)} classes). Avec peu de classes, à présenter comme une tendance, pas comme une preuve.")

    print(f"\nFichiers : {out_dir / (args.dataset + '_complementarite.csv')}, "
          f"{stem.with_suffix('.csv')}, {stem.with_suffix('.pdf')}, {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()