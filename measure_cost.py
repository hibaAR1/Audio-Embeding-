"""
Coût du modèle CrossGated (point 6 : « notre modèle est léger »).

Mesure, sur TA machine :
  1) le nombre de paramètres des deux encodeurs gelés (PANNs CNN14 et AST) ;
  2) le nombre de paramètres ENTRAÎNÉS de chaque tête (PANNs seul, AST seul, CrossGated) ;
  3) le temps d'extraction d'un extrait de 5 s par chaque encodeur (médiane de 20 essais) ;
  4) le temps d'entraînement d'une tête sur un pli d'ESC-50 (60 époques, mêmes réglages que l'article) ;
  5) le temps de prédiction de la tête pour un extrait.

Les encodeurs sont comptés par le script lui-même : aucun chiffre n'est recopié d'un article.

Usage (dans le dossier du projet, .venv activé, rien d'autre ne tourne) :
  python measure_cost.py
  python measure_cost.py --skip-encoders     (seulement les têtes, si les encodeurs posent problème)

Sorties : results_cost/cost.csv et un tableau LaTeX affiché à l'écran.
"""
import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import crossgated as cg

N_REPEAT, N_WARMUP = 20, 3
HEADS = ["panns_only", "ast_only", "crossgated"]


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def median_ms(fn):
    for _ in range(N_WARMUP):
        fn()
    times = []
    for _ in range(N_REPEAT):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times))


def measure_encoders(device):
    from extract_features import ensure_panns_files, PANNS_CKPT, AST_MODEL, SR, AST_SR, N_SAMPLES
    from panns_inference import AudioTagging
    from transformers import ASTFeatureExtractor, ASTModel
    import librosa

    ensure_panns_files()
    panns = AudioTagging(checkpoint_path=str(PANNS_CKPT), device=device)
    ast_fe = ASTFeatureExtractor.from_pretrained(AST_MODEL)
    ast = ASTModel.from_pretrained(AST_MODEL).to(device).eval()

    wav32 = (np.random.default_rng(0).standard_normal(N_SAMPLES) * 0.1).astype(np.float32)
    wav16 = librosa.resample(wav32, orig_sr=SR, target_sr=AST_SR)

    def run_panns():
        panns.inference(wav32[None, :])

    def run_ast():
        inputs = {k: v.to(device) for k, v in ast_fe(wav16, sampling_rate=AST_SR, return_tensors="pt").items()}
        with torch.no_grad():
            ast(**inputs)

    return {
        "panns_params": n_params(panns.model),      # CNN14 complet (y compris sa couche de 527 classes AudioSet)
        "ast_params": n_params(ast),                # AST sans tête de classification
        "panns_ms": median_ms(run_panns),
        "ast_ms": median_ms(run_ast),
    }


def measure_heads(features_dir, ast_file, fold, device):
    p, s, y, folds = cg.load_features(features_dir, ast_file)
    n_cls = len(np.unique(y))
    tr, te = folds != fold, folds == fold
    sc_p, sc_s = StandardScaler().fit(p[tr]), StandardScaler().fit(s[tr])
    Pt, Pe = sc_p.transform(p[tr]).astype(np.float32), sc_p.transform(p[te]).astype(np.float32)
    St, Se = sc_s.transform(s[tr]).astype(np.float32), sc_s.transform(s[te]).astype(np.float32)
    rows = []
    for v in HEADS:
        t0 = time.perf_counter()
        model = cg.train_net(v, Pt, St, y[tr], n_cls, 0, device)
        train_s = time.perf_counter() - t0
        one_p, one_s = Pe[:1], Se[:1]
        head_ms = median_ms(lambda: cg.predict(model, one_p, one_s, device))
        acc = 100 * float((cg.predict(model, Pe, Se, device) == y[te]).mean())
        rows.append({"model": v, "trainable_params": cg.count_params(model),
                     "train_seconds_one_fold": train_s, "head_ms_per_clip": head_ms,
                     "n_train": int(tr.sum()), "check_acc_fold": acc})
        print(f"  {v:<11} {cg.count_params(model):>9,} paramètres entraînés | entraînement {train_s:6.1f} s "
              f"| prédiction {head_ms:.2f} ms/extrait | (contrôle : {acc:.1f} % sur le pli {fold})", flush=True)
    return pd.DataFrame(rows)


def fmt_m(n):
    return f"{n / 1e6:.1f}\\,M" if n >= 1e6 else f"{n / 1e6:.2f}\\,M"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features_v2/esc50")
    ap.add_argument("--ast-file", default="ast.npy")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--skip-encoders", action="store_true")
    ap.add_argument("--out", default="results_cost")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hw = (torch.cuda.get_device_name(0) if device.type == "cuda"
          else f"CPU {platform.processor() or platform.machine()}, {torch.get_num_threads()} threads")
    print(f"Machine : {hw} | {platform.system()} | torch {torch.__version__}")
    print("Ferme les autres programmes : les temps dépendent de la charge de la machine.\n")

    enc = None
    if not args.skip_encoders:
        print("1) Encodeurs gelés (extrait de 5 s) ...", flush=True)
        enc = measure_encoders(device)
        print(f"  PANNs CNN14 : {enc['panns_params']:>11,} paramètres | {enc['panns_ms']:7.1f} ms/extrait")
        print(f"  AST         : {enc['ast_params']:>11,} paramètres | {enc['ast_ms']:7.1f} ms/extrait\n")

    print(f"2) Têtes entraînées (ESC-50, pli {args.fold}, {cg.Config.EPOCHS} époques) ...", flush=True)
    heads = measure_heads(args.features, args.ast_file, args.fold, device)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    if enc:
        frozen = {"panns_only": enc["panns_params"], "ast_only": enc["ast_params"],
                  "crossgated": enc["panns_params"] + enc["ast_params"]}
        enc_ms = {"panns_only": enc["panns_ms"], "ast_only": enc["ast_ms"],
                  "crossgated": enc["panns_ms"] + enc["ast_ms"]}
        heads["frozen_params"] = heads["model"].map(frozen)
        heads["encoder_ms_per_clip"] = heads["model"].map(enc_ms)
        heads["total_ms_per_clip"] = heads["encoder_ms_per_clip"] + heads["head_ms_per_clip"]
    heads["hardware"] = hw
    heads.to_csv(out_dir / "cost.csv", index=False, float_format="%.4f")

    names = {"panns_only": "PANNs only", "ast_only": "AST only", "crossgated": "CrossGated (ours)"}
    print("\n" + "=" * 70 + "\nTABLEAU LaTeX (à vérifier, puis à coller dans main.tex)\n" + "=" * 70)
    print("\\begin{tabular}{lrrrr}\n\\toprule")
    print("Model & Trainable & Frozen & Training & Inference \\\\")
    print(" & parameters & parameters & time / fold (s) & time / clip (ms) \\\\\n\\midrule")
    for r in heads.itertuples():
        frozen_txt = fmt_m(r.frozen_params) if enc else "[ ]"
        inf_txt = f"{r.total_ms_per_clip:.0f}" if enc else f"[ ] + {r.head_ms_per_clip:.2f}"
        sep = "\\midrule\n" if r.model == "crossgated" else ""
        print(f"{sep}{names[r.model]} & {fmt_m(r.trainable_params)} & {frozen_txt} & "
              f"{r.train_seconds_one_fold:.0f} & {inf_txt} \\\\")
    if enc:
        print(f"\\midrule\nAST fine-tuned~\\cite{{gong2021ast}} & {fmt_m(enc['ast_params'])} & 0 & -- & "
              f"{enc['ast_ms']:.0f} \\\\")
    print("\\bottomrule\n\\end{tabular}")
    print(f"\nFichier : {out_dir / 'cost.csv'}")


if __name__ == "__main__":
    main()