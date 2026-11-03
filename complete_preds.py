"""
Complète les prédictions manquantes (par exemple après un débranchement du disque F:).

Cherche, pour chaque graine, chaque pli et chaque modèle (PANNs seul, AST seul, CrossGated),
s'il manque des prédictions dans results_preds/<dataset>_preds.csv (et sa copie de secours sur C:).
Il réentraîne SEULEMENT ce qui manque, avec la même graine et les mêmes réglages,
et ajoute les prédictions au fichier. Le fichier des résultats n'est pas modifié.

Usage :
  python complete_preds.py urban
  python complete_preds.py esc50
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import crossgated as cg
from analyse_predictions import read_raw_preds, NEEDED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["esc50", "urban"])
    ap.add_argument("--features", default=None)
    ap.add_argument("--ast-file", default="ast.npy")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default="results_preds")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p, s, y, folds = cg.load_features(args.features or f"features_v2/{args.dataset}", args.ast_file)
    n_cls = len(np.unique(y))
    fold_ids = sorted(np.unique(folds))
    path = Path(args.out) / f"{args.dataset}_preds.csv"
    paths = [path, cg.backup_of(path)]

    df = read_raw_preds(path)
    counts = df.groupby(["model", "seed", "fold"]).size()
    res = cg.read_rows([Path(args.out) / f"{args.dataset}_results.csv",
                        cg.backup_of(Path(args.out) / f"{args.dataset}_results.csv")],
                       cg.RESULT_COLS, cg.KEY_COLS)
    acc_ref = {(r.model, int(r.seed), int(r.fold)): r.acc for r in res.itertuples()}

    todo = {}
    for seed in args.seeds:
        for fold in fold_ids:
            n = int((folds == fold).sum())
            for m in NEEDED:
                have = int(counts.get((m, seed, int(fold)), 0))
                if have < n:
                    todo.setdefault((seed, int(fold)), []).append(m)
                    print(f"  manque : {m:<11} graine {seed} pli {fold:>2} ({have}/{n} prédictions)")
    if not todo:
        print("Toutes les prédictions sont complètes. Lance : python analyse_predictions.py "
              f"{args.dataset}")
        return
    print(f"\n{sum(len(v) for v in todo.values())} entraînement(s) à refaire.\n", flush=True)

    for (seed, fold), models in todo.items():
        tr, te = folds != fold, folds == fold
        sc_p, sc_s = StandardScaler().fit(p[tr]), StandardScaler().fit(s[tr])
        Pt, Pe = sc_p.transform(p[tr]).astype(np.float32), sc_p.transform(p[te]).astype(np.float32)
        St, Se = sc_s.transform(s[tr]).astype(np.float32), sc_s.transform(s[te]).astype(np.float32)
        for m in models:
            t0 = time.time()
            model = cg.train_net(m, Pt, St, y[tr], n_cls, seed, device)
            pred = cg.predict(model, Pe, Se, device)
            acc = float((pred == y[te]).mean())
            cg.append_frame(paths, pd.DataFrame({
                "model": m, "seed": seed, "fold": fold, "index": np.flatnonzero(te),
                "y_true": y[te], "y_pred": np.asarray(pred).astype(int)}))
            ref = acc_ref.get((m, seed, fold))
            ref_txt = f" | résultat enregistré : {ref * 100:.1f} %" if ref is not None else ""
            print(f"graine {seed} | pli {fold:>2} | {m:<11} {acc * 100:5.1f} %{ref_txt}  "
                  f"({time.time() - t0:.0f} s)", flush=True)

    print(f"\nTerminé. Lance maintenant : python analyse_predictions.py {args.dataset}")


if __name__ == "__main__":
    main()
