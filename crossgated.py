"""
CrossGated : évaluation et ablation sur ESC-50 et UrbanSound8K.

Le modèle suit exactement l'article :
  Z-score par pli (estimé sur l'entraînement)
  -> projections linéaires 2048->128 (PANNs) et 768->128 (AST)
  -> contexte c = [p ; s] (256 d)
  -> portes gp = sigmoid(Wgp c + b), gs = sigmoid(Wgs c + b)
  -> modulation p*gp, s*gs puis concaténation (256 d)
  -> bloc de fusion : LayerNorm, Linear 256->256, ReLU, Dropout
  -> MLP : LayerNorm, Linear 256->128, ReLU, Dropout, Linear 128->64, ReLU  (= f)
  -> classifieur linéaire N classes
Perte : entropie croisée (lissage 0,1) + lambda(e) * Fisher, lambda(e) = min(0,3 ; 0,01 + 0,005 e)
Entraînement : AdamW (lr 5e-4, wd 1e-3), batch 16, 60 époques, dropout 0,3, écrêtage 1,0.
Évaluation à la fin de l'entraînement, SANS sélection de modèle sur le pli de test.

Variantes (--models) :
  svm         SVM RBF (C=10) sur la concaténation standardisée PANNs + AST
  panns_only  PANNs seul, même tête
  ast_only    AST seul, même tête
  concat      concaténation des projections, sans porte, même tête
  uncrossed   portes non croisées : gp dépend de p seul, gs de s seul
  nofisher    CrossGated sans perte de Fisher
  crossgated  méthode complète

Usage :
  python crossgated.py esc50
  python crossgated.py urban
  python crossgated.py esc50 --models crossgated svm --seeds 0
  python crossgated.py esc50 --ast-file ast_real.npy
  python crossgated.py summary results/esc50_results.csv
"""
import gc
import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

ALL_MODELS = ["svm", "panns_only", "ast_only", "concat", "uncrossed", "nofisher", "crossgated"]


class Config:
    PROJ = 128
    FUSION = 256
    HID = 128
    DROPOUT = 0.3
    LR = 5e-4
    WD = 1e-3
    BATCH = 16
    EPOCHS = 60
    LABEL_SMOOTHING = 0.1
    CLIP = 1.0
    SVM_C = 10


def fisher_weight(epoch):
    return min(0.3, 0.01 + 0.005 * epoch)


# ============================================================
# PERTE DE FISHER
# ============================================================
def fisher_loss(f, y, n_cls):
    """Rapport dispersion intra-classe / dispersion inter-classe sur le mini-batch.

    intra : moyenne, sur les exemples des classes présentes au moins 2 fois,
            de ||f_i - mu_k||^2 (une classe vue une seule fois n'a pas de dispersion mesurable)
    inter : moyenne pondérée par n_k de ||mu_k - mu||^2 sur les classes présentes
    """
    onehot = F.one_hot(y, n_cls).float()                  # (B, K)
    counts = onehot.sum(0)                                # (K,)
    present = counts > 0
    if present.sum() < 2:
        return f.new_zeros(())
    means = (onehot.t() @ f) / counts.clamp(min=1).unsqueeze(1)   # (K, D)
    mu = f.mean(0)
    inter = (counts[present] * ((means[present] - mu) ** 2).sum(1)).sum() / f.shape[0]

    valid = (counts >= 2)[y]                              # exemples d'une classe vue >= 2 fois
    if valid.sum() == 0:
        return f.new_zeros(())
    intra = ((f - means[y]) ** 2).sum(1)[valid].mean()
    return intra / (inter + 1e-6)


# ============================================================
# MODÈLES
# ============================================================
def make_head(in_dim, n_cls):
    fusion = nn.Sequential(
        nn.LayerNorm(in_dim), nn.Linear(in_dim, Config.FUSION), nn.ReLU(), nn.Dropout(Config.DROPOUT),
    )
    mlp = nn.Sequential(
        nn.LayerNorm(Config.FUSION),
        nn.Linear(Config.FUSION, Config.HID), nn.ReLU(), nn.Dropout(Config.DROPOUT),
        nn.Linear(Config.HID, Config.HID // 2), nn.ReLU(),
    )
    clf = nn.Linear(Config.HID // 2, n_cls)
    return fusion, mlp, clf


class FusionNet(nn.Module):
    """Une seule classe pour toutes les variantes, pour que la tête soit strictement identique."""

    def __init__(self, variant, panns_dim, ast_dim, n_cls):
        super().__init__()
        self.variant = variant
        P = Config.PROJ
        self.use_p = variant != "ast_only"
        self.use_s = variant != "panns_only"
        if self.use_p:
            self.proj_p = nn.Linear(panns_dim, P)
        if self.use_s:
            self.proj_s = nn.Linear(ast_dim, P)

        if variant in ("crossgated", "nofisher"):
            self.gate_p = nn.Linear(2 * P, P)            # porte calculée sur c = [p ; s]
            self.gate_s = nn.Linear(2 * P, P)
        elif variant == "uncrossed":
            self.gate_p = nn.Linear(P, P)                # porte calculée sur p seul
            self.gate_s = nn.Linear(P, P)

        in_dim = P * (int(self.use_p) + int(self.use_s))
        self.fusion, self.mlp, self.clf = make_head(in_dim, n_cls)

    def forward(self, xp, xs):
        gates = None
        if self.variant == "panns_only":
            f0 = self.proj_p(xp)
        elif self.variant == "ast_only":
            f0 = self.proj_s(xs)
        else:
            p, s = self.proj_p(xp), self.proj_s(xs)
            if self.variant == "concat":
                f0 = torch.cat([p, s], 1)
            else:
                if self.variant == "uncrossed":
                    gp, gs = torch.sigmoid(self.gate_p(p)), torch.sigmoid(self.gate_s(s))
                else:
                    c = torch.cat([p, s], 1)
                    gp, gs = torch.sigmoid(self.gate_p(c)), torch.sigmoid(self.gate_s(c))
                f0 = torch.cat([p * gp, s * gs], 1)
                gates = (gp, gs)
        f = self.mlp(self.fusion(f0))
        return self.clf(f), f, gates


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# ENTRAÎNEMENT
# ============================================================
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_net(variant, Xp, Xs, y, n_cls, seed, device):
    set_seed(seed)
    model = FusionNet(variant, Xp.shape[1], Xs.shape[1], n_cls).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WD)
    ce = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    use_fisher = variant != "nofisher"

    Xp_t = torch.as_tensor(Xp, dtype=torch.float32, device=device)
    Xs_t = torch.as_tensor(Xs, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.long, device=device)
    n = len(y)
    gen = torch.Generator().manual_seed(seed)

    for epoch in range(Config.EPOCHS):
        model.train()
        lam = fisher_weight(epoch)
        order = torch.randperm(n, generator=gen).to(device)
        for start in range(0, n, Config.BATCH):
            idx = order[start:start + Config.BATCH]
            logits, f, _ = model(Xp_t[idx], Xs_t[idx])
            loss = ce(logits, y_t[idx])
            if use_fisher:
                loss = loss + lam * fisher_loss(f, y_t[idx], n_cls)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP)
            opt.step()
    return model


@torch.no_grad()
def predict(model, Xp, Xs, device, return_gates=False):
    model.eval()
    Xp_t = torch.as_tensor(Xp, dtype=torch.float32, device=device)
    Xs_t = torch.as_tensor(Xs, dtype=torch.float32, device=device)
    preds, gps, gss = [], [], []
    for start in range(0, len(Xp_t), 512):
        logits, _, gates = model(Xp_t[start:start + 512], Xs_t[start:start + 512])
        preds.append(logits.argmax(1).cpu().numpy())
        if gates is not None:
            gps.append(gates[0].cpu().numpy())
            gss.append(gates[1].cpu().numpy())
    preds = np.concatenate(preds)
    if return_gates and gps:
        return preds, np.concatenate(gps), np.concatenate(gss)
    return preds


# ============================================================
# ÉVALUATION
# ============================================================
def load_features(features_dir, ast_file):
    d = Path(features_dir)
    p = np.load(d / "panns.npy").astype(np.float32)
    s = np.load(d / ast_file).astype(np.float32)
    y = np.load(d / "labels.npy").astype(np.int64)
    folds = np.load(d / "folds.npy").astype(np.int64)
    if p.shape[1] != 2048:
        print(f"ATTENTION : PANNs a {p.shape[1]} dimensions au lieu de 2048. "
              "Ce ne sont pas les embeddings CNN14 : réextrais avec extract_features.py.")
    return p, s, y, folds


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_dir = args.features or os.path.join("features_v2", args.dataset)
    p, s, y, folds = load_features(features_dir, args.ast_file)
    n_cls = len(np.unique(y))
    fold_ids = sorted(np.unique(folds))
    print(f"{args.dataset} | {len(y)} extraits | {n_cls} classes | {len(fold_ids)} plis | "
          f"PANNs {p.shape[1]} d | AST {s.shape[1]} d ({args.ast_file}) | device {device}")
    for v in args.models:
        if v != "svm":
            m = FusionNet(v, p.shape[1], s.shape[1], n_cls)
            print(f"  paramètres entraînés {v:<11}: {count_params(m):,}")

    out_dir = Path(args.out)
    tag = f"{args.dataset}" + ("" if args.ast_file == "ast.npy" else f"_{Path(args.ast_file).stem}")
    csv_path = out_dir / f"{tag}_results.csv"
    gates_path = out_dir / f"{tag}_gates.csv"
    res_paths = [csv_path, backup_of(csv_path)]               # copie de secours sur le disque C:
    preds_path = out_dir / f"{tag}_preds.csv"
    preds_paths = [preds_path, backup_of(preds_path)]
    gate_paths = [gates_path, backup_of(gates_path)]
    if args.restart:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for path in res_paths + gate_paths:
            if path.exists():
                path.rename(path.with_name(f"{path.stem}.old-{stamp}{path.suffix}"))
    old = read_rows(res_paths, RESULT_COLS, KEY_COLS)
    if len(old):
        sync_backup(old, res_paths[1])
    old_gates = read_rows(gate_paths, GATE_COLS, ["seed", "fold", "class"])
    if len(old_gates):
        sync_backup(old_gates, gate_paths[1])
    done = {(r.model, int(r.seed), int(r.fold)) for r in old.itertuples()}
    if done:
        print(f"Reprise : {len(done)} résultats déjà calculés ({csv_path} + copie {BACKUP_DIR})")

    for seed in args.seeds:
        for fold in fold_ids:
            todo = [v for v in args.models if (v, seed, fold) not in done
                    and not (v == "svm" and seed != args.seeds[0])]   # le SVM est déterministe : une seule exécution
            if not todo:
                continue
            tr, te = folds != fold, folds == fold
            # Z-score par branche, estimé sur l'entraînement du pli
            sc_p, sc_s = StandardScaler().fit(p[tr]), StandardScaler().fit(s[tr])
            Pt, Pe = sc_p.transform(p[tr]).astype(np.float32), sc_p.transform(p[te]).astype(np.float32)
            St, Se = sc_s.transform(s[tr]).astype(np.float32), sc_s.transform(s[te]).astype(np.float32)
            del sc_p, sc_s
            gc.collect()

            for v in todo:
                t0 = time.time()
                if v == "svm":
                    Xtr, Xte = np.hstack([Pt, St]), np.hstack([Pe, Se])
                    pred = SVC(kernel="rbf", C=Config.SVM_C).fit(Xtr, y[tr]).predict(Xte)
                else:
                    model = train_net(v, Pt, St, y[tr], n_cls, seed, device)
                    if v == "crossgated":
                        pred, gp, gs = predict(model, Pe, Se, device, return_gates=True)
                        for c in np.unique(y[te]):
                            m = y[te] == c
                            append_row(gate_paths, {"seed": seed, "fold": fold, "class": int(c),
                                                    "gate_panns": float(gp[m].mean()),
                                                    "gate_ast": float(gs[m].mean()),
                                                    "n": int(m.sum())}, GATE_COLS)
                    else:
                        pred = predict(model, Pe, Se, device)
                    del model
                gc.collect()
                acc = accuracy_score(y[te], pred)
                if args.save_preds:
                    idx_te = np.flatnonzero(te)
                    append_frame(preds_paths, pd.DataFrame({
                        "model": v, "seed": seed, "fold": fold, "index": idx_te,
                        "y_true": y[te], "y_pred": np.asarray(pred).astype(int)}))
                append_row(res_paths, {"model": v, "seed": seed, "fold": fold, "acc": acc}, RESULT_COLS)
                done.add((v, seed, fold))
                print(f"seed {seed} | pli {fold:>2} | {v:<11} {acc * 100:5.1f} %  ({time.time() - t0:.0f} s)",
                      flush=True)

    summarize(csv_path)


# ============================================================
# ENREGISTREMENT SÛR DES RÉSULTATS
# Chaque résultat est AJOUTÉ à la fin du fichier (jamais de réécriture complète),
# et recopié dans une sauvegarde sur le disque C: (dossier crossgated_backup).
# Si le disque du projet se déconnecte, aucun résultat déjà calculé n'est effacé.
# ============================================================
RESULT_COLS = ["model", "seed", "fold", "acc"]
KEY_COLS = ["model", "seed", "fold"]
GATE_COLS = ["seed", "fold", "class", "gate_panns", "gate_ast", "n"]
BACKUP_DIR = Path.home() / "crossgated_backup"


def backup_of(path):
    """Copie de secours sur C:, nommée d'après le dossier et le fichier (ex. results_urban_results.csv)."""
    path = Path(path)
    return BACKUP_DIR / f"{path.parent.resolve().name}_{path.name}"


def sync_backup(df, path):
    """Réécrit la sauvegarde avec l'union des résultats connus (elle ne peut que grandir)."""
    tmp = path.with_suffix(".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(tmp, index=False)
        tmp.replace(path)
    except OSError as e:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        _warn_backup(path, e)


def read_rows(paths, cols, key_cols):
    frames = []
    for path in paths:
        try:
            if not path.exists():
                continue
            # on_bad_lines="skip" : ignore une ligne coupée par un disque plein ou débranché
            df = pd.read_csv(path, dtype=str, on_bad_lines="skip")
        except OSError as e:
            sys.exit(f"ERREUR : impossible de lire {path} ({e}).\n"
                     "Le disque est-il bien branché ? Rien n'a été modifié : rebranche-le et relance.")
        except pd.errors.EmptyDataError:
            continue
        if any(c not in df.columns for c in cols):
            continue
        df = df[df[cols[0]] != cols[0]]                     # en-têtes répétés éventuels
        frames.append(df[cols])
    if not frames:
        return pd.DataFrame(columns=cols)
    df = pd.concat(frames, ignore_index=True)
    num = [c for c in cols if c != "model"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=num)
    if "model" in cols:
        df = df[df["model"].isin(ALL_MODELS)]
    return df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)


def _warn_backup(path, e):
    """La copie de secours sur C: a échoué mais le fichier principal est écrit : on continue."""
    global _BACKUP_WARNED
    if not _BACKUP_WARNED:
        print(f"ATTENTION : copie de secours impossible dans {path} ({e}). "
              "Le résultat est bien écrit dans le dossier du projet ; le calcul continue. "
              "Libère de la place sur C: pour retrouver la copie de secours.", flush=True)
        _BACKUP_WARNED = True


_BACKUP_WARNED = False


def append_frame(paths, df):
    """Ajoute un bloc de lignes (prédictions) à la fin de chaque fichier, sans réécrire l'existant."""
    written = False
    for i, path in enumerate(paths):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, mode="a", header=not path.exists(), index=False)
            written = True
        except OSError as e:
            if i == 0:
                print(f"ATTENTION : écriture impossible dans {path} ({e}). "
                      f"Les prédictions sont gardées dans la sauvegarde {BACKUP_DIR}.", flush=True)
            elif written:
                _warn_backup(path, e)
            else:
                raise


def append_row(paths, row, cols):
    line = ",".join(str(row[c]) for c in cols) + "\n"
    written = False
    for i, path in enumerate(paths):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            new = not path.exists()
            with open(path, "a", encoding="utf-8") as f:
                if new:
                    f.write(",".join(cols) + "\n")
                f.write(line)
            written = True
        except OSError as e:
            if i == 0:
                print(f"ATTENTION : écriture impossible dans {path} ({e}). "
                      f"Le résultat est gardé dans la sauvegarde {BACKUP_DIR}.", flush=True)
            elif written:
                _warn_backup(path, e)
            else:
                raise


def summarize(csv_path):
    csv_path = Path(csv_path)
    df = read_rows([csv_path, backup_of(csv_path)], RESULT_COLS, KEY_COLS)
    print(f"\n{'=' * 64}\nRÉSULTATS : {csv_path}\n{'=' * 64}")
    print(f"{'Modèle':<12} {'Moyenne':>8} {'± plis':>8} {'± graines':>10} {'graines':>8}")
    order = [m for m in ALL_MODELS if m in df["model"].unique()]
    fold_means = {}
    for m in order:
        d = df[df["model"] == m]
        per_seed = d.groupby("seed")["acc"].mean() * 100            # accuracy moyenne par graine
        per_fold = d.groupby("fold")["acc"].mean() * 100            # moyenne des graines par pli
        fold_means[m] = per_fold
        print(f"{m:<12} {per_seed.mean():7.1f} % {per_fold.std(ddof=0):7.1f} "
              f"{per_seed.std(ddof=0):9.1f} {d['seed'].nunique():>8}")

    if "crossgated" in fold_means:
        from scipy.stats import wilcoxon
        print("\nTest de Wilcoxon apparié sur les plis (crossgated contre chaque variante) :")
        for m in order:
            if m == "crossgated":
                continue
            a, b = fold_means["crossgated"], fold_means[m]
            common = a.index.intersection(b.index)
            diff = (a[common] - b[common]).mean()
            try:
                pval = wilcoxon(a[common], b[common]).pvalue
                ptxt = f"p = {pval:.3f}"
            except ValueError:
                ptxt = "p non calculable"
            print(f"  crossgated - {m:<11}: {diff:+5.1f} points  ({ptxt}, {len(common)} plis)")
        print("Avec 5 plis (ESC-50), le plus petit p possible est 0,0625 : "
              "rapporte aussi l'écart-type des graines.")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "summary":
        summarize(sys.argv[2])
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["esc50", "urban"])
    parser.add_argument("--features", default=None, help="Dossier des features (défaut : features_v2/<dataset>)")
    parser.add_argument("--ast-file", default="ast.npy", help="ast.npy (article) ou ast_real.npy")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
    parser.add_argument("--out", default="results")
    parser.add_argument("--restart", action="store_true", help="Ignore les résultats déjà enregistrés")
    parser.add_argument("--save-preds", action="store_true",
                        help="Enregistre aussi la prédiction de chaque extrait (<dataset>_preds.csv)")
    args = parser.parse_args()
    Config.EPOCHS = args.epochs
    run(args)


if __name__ == "__main__":
    main()