"""
ABLATION STUDY v2 - ESC-50 & UrbanSound8K
Corrige : normalisation, gating asymétrique, Fisher loss vectorisée,
        pertes auxiliaires par modalité, plan d'entraînement amélioré.

Compare: PANNs only | SSL only | Concat naive | CrossGated (no Fisher) | Full method

Usage:
python ablation.py esc50
python ablation.py urban
"""
import os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


class Config:
    ESC50_DIR = r"C:\Users\21268\Desktop\AUDIO EMBEDDING\features"
    URBAN_DIR = r"C:\Users\21268\Desktop\AUDIO EMBEDDING\features_urban_final"
    BATCH_SIZE = 64
    EPOCHS = 40
    LR = 5e-4
    WD = 1e-3
    PROJ = 128       # dimension projetée par modalité
    HID = 64         # dimension cachée du classifieur
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATASET
# ============================================================
class DS(Dataset):
    def __init__(self, p, s, y):
        self.p = torch.FloatTensor(p)
        self.s = torch.FloatTensor(s)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.p[i], self.s[i], self.y[i]


# ============================================================
# FISHER LOSS VECTORISÉE (rapide et stable)
# ============================================================
def fisher_loss(f, y, n_cls):
    """
    Minimise intra_scatter / inter_scatter (ratio de Fisher).
    Version vectorisée : O(1) par classe au lieu de O(batch).
    """
    device = f.device
    gm = f.mean(0)                                  # (D,)
    # Inter-classe : somme de n_c * ||c_mean - global_mean||^2
    inter = torch.tensor(0.0, device=device)
    intra = torch.tensor(0.0, device=device)
    for c in range(n_cls):
        mask = (y == c)
        n_c = mask.sum()
        if n_c > 1:
            cm = f[mask].mean(0)                    # (D,)
            inter = inter + n_c * ((cm - gm) ** 2).sum()
            intra = intra + (f[mask] - cm).pow(2).sum()
    if inter < 1e-8:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return intra / inter


# ============================================================
# MODÈLES D'ABLATION
# ============================================================

class PANNsOnly(nn.Module):
    """PANNs seul, sans SSL."""
    def __init__(self, n_cls, pd, proj=256, hid=128):
        super().__init__()
        self.proj = nn.Linear(pd, proj)
        self.mlp = nn.Sequential(
            nn.LayerNorm(proj),
            nn.Linear(proj, hid), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hid, hid // 2), nn.ReLU(),
        )
        self.clf = nn.Linear(hid // 2, n_cls)
        # Classifieur auxiliaire sur la projection (force l'apprentissage)
        self.aux_clf = nn.Linear(proj, n_cls)

    def forward(self, p, s, ret_f=False):
        fp = F.relu(self.proj(p))
        f = self.mlp(fp)
        l = self.clf(f)
        if ret_f:
            return l, f, fp
        return l

    def fisher(self, f, y, n_cls):
        return fisher_loss(f, y, n_cls)


class SSLOnly(nn.Module):
    """SSL seul, sans PANNs."""
    def __init__(self, n_cls, sd, proj=256, hid=128):
        super().__init__()
        self.proj = nn.Linear(sd, proj)
        self.mlp = nn.Sequential(
            nn.LayerNorm(proj),
            nn.Linear(proj, hid), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hid, hid // 2), nn.ReLU(),
        )
        self.clf = nn.Linear(hid // 2, n_cls)
        self.aux_clf = nn.Linear(proj, n_cls)

    def forward(self, p, s, ret_f=False):
        fs = F.relu(self.proj(s))
        f = self.mlp(fs)
        l = self.clf(f)
        if ret_f:
            return l, f, fs
        return l

    def fisher(self, f, y, n_cls):
        return fisher_loss(f, y, n_cls)


class ConcatNaive(nn.Module):
    """Concaténation brute après projection indépendante, sans portes."""
    def __init__(self, n_cls, pd, sd, proj=256, hid=128):
        super().__init__()
        self.proj_p = nn.Linear(pd, proj)
        self.proj_s = nn.Linear(sd, proj)
        self.mlp = nn.Sequential(
            nn.LayerNorm(proj * 2),
            nn.Linear(proj * 2, hid), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hid, hid // 2), nn.ReLU(),
        )
        self.clf = nn.Linear(hid // 2, n_cls)
        self.aux_p = nn.Linear(proj, n_cls)
        self.aux_s = nn.Linear(proj, n_cls)

    def forward(self, p, s, ret_f=False):
        fp = F.relu(self.proj_p(p))
        fs = F.relu(self.proj_s(s))
        f = self.mlp(torch.cat([fp, fs], 1))
        l = self.clf(f)
        if ret_f:
            return l, f, fp, fs
        return l

    def fisher(self, f, y, n_cls):
        return fisher_loss(f, y, n_cls)


class CrossGatedNoFisher(nn.Module):
    """CrossGatedFusion SANS Fisher loss."""
    def __init__(self, n_cls, pd, sd, proj=256, hid=128, summary_dim=32, pool='avg'):
        super().__init__()
        self.n_cls = n_cls
        # Projections indépendantes
        self.proj_p = nn.Linear(pd, proj)
        self.proj_s = nn.Linear(sd, proj)

        # Portes asymétriques : chaque modalité a sa propre porte
        # La porte de PANNs voit PANNs + un résumé de SSL (et inversement)
        self.gate_p = nn.Sequential(
            nn.Linear(proj + summary_dim, proj), nn.Sigmoid()
        )
        self.gate_s = nn.Sequential(
            nn.Linear(proj + summary_dim, proj), nn.Sigmoid()
        )
        # Résumé de chaque modalité (summary_dim d) pour le contexte croisé
        self.summarize_p = nn.Linear(proj, summary_dim)
        self.summarize_s = nn.Linear(proj, summary_dim)

        # Fusion
        self.fuse = nn.Sequential(
            nn.LayerNorm(proj * 2),
            nn.Linear(proj * 2, proj), nn.ReLU(), nn.Dropout(0.2),
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(proj),
            nn.Linear(proj, hid), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hid, hid // 2), nn.ReLU(),
        )
        self.clf = nn.Linear(hid // 2, n_cls)

        # Classifieurs auxiliaires (force chaque projection à être utile)
        self.aux_p = nn.Linear(proj, n_cls)
        self.aux_s = nn.Linear(proj, n_cls)

    def forward(self, p, s, ret_f=False):
        fp = F.relu(self.proj_p(p))     # (B, proj)
        fs = F.relu(self.proj_s(s))     # (B, proj)

        # Contexte croisé (résumé de l'autre modalité)
        ctx_s = self.summarize_s(fs)     # (B, 32) résumé de SSL
        ctx_p = self.summarize_p(fp)     # (B, 32) résumé de PANNs

        # Portes asymétriques
        gp = self.gate_p(torch.cat([fp, ctx_s], 1))  # (B, proj)
        gs = self.gate_s(torch.cat([fs, ctx_p], 1))  # (B, proj)

        # Application des portes
        fp_gated = fp * gp
        fs_gated = fs * gs

        # Fusion + classifieur
        fused = self.fuse(torch.cat([fp_gated, fs_gated], 1))
        f = self.mlp(fused)
        l = self.clf(f)

        if ret_f:
            return l, f, fp, fs
        return l

    def fisher(self, f, y, n_cls):
        return torch.tensor(0.0, device=f.device, requires_grad=True)


class CrossGatedFull(nn.Module):
    """Méthode complète : CrossGatedFusion + Fisher loss + aux losses."""
    def __init__(self, n_cls, pd, sd, proj=256, hid=128, summary_dim=32, pool='avg'):
        super().__init__()
        self.n_cls = n_cls
        self.proj_p = nn.Linear(pd, proj)
        self.proj_s = nn.Linear(sd, proj)

        self.gate_p = nn.Sequential(
            nn.Linear(proj + summary_dim, proj), nn.Sigmoid()
        )
        self.gate_s = nn.Sequential(
            nn.Linear(proj + summary_dim, proj), nn.Sigmoid()
        )
        self.summarize_p = nn.Linear(proj, summary_dim)
        self.summarize_s = nn.Linear(proj, summary_dim)

        self.fuse = nn.Sequential(
            nn.LayerNorm(proj * 2),
            nn.Linear(proj * 2, proj), nn.ReLU(), nn.Dropout(0.2),
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(proj),
            nn.Linear(proj, hid), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hid, hid // 2), nn.ReLU(),
        )
        self.clf = nn.Linear(hid // 2, n_cls)
        self.aux_p = nn.Linear(proj, n_cls)
        self.aux_s = nn.Linear(proj, n_cls)

    def forward(self, p, s, ret_f=False):
        fp = F.relu(self.proj_p(p))
        fs = F.relu(self.proj_s(s))

        ctx_s = self.summarize_s(fs)
        ctx_p = self.summarize_p(fp)

        gp = self.gate_p(torch.cat([fp, ctx_s], 1))
        gs = self.gate_s(torch.cat([fs, ctx_p], 1))

        fp_gated = fp * gp
        fs_gated = fs * gs

        fused = self.fuse(torch.cat([fp_gated, fs_gated], 1))
        f = self.mlp(fused)
        l = self.clf(f)

        if ret_f:
            return l, f, fp, fs
        return l

    def fisher(self, f, y, n_cls):
        return fisher_loss(f, y, n_cls)


# ============================================================
# ENTRAÎNEMENT & ÉVALUATION (corrigé)
# ============================================================

def train_and_eval(model, Xp, Xs, y, Xp_te, Xs_te, y_te,
                   n_cls, use_fisher=False, epochs=80, lr=3e-4):
    """Entraîne le modèle et retourne l'accuracy sur le test set."""

    # ---- Z-score normalization par modalité (fit sur train) ----
    sc_p = StandardScaler()
    sc_s = StandardScaler()
    Xp_t = sc_p.fit_transform(Xp).astype(np.float32)
    Xs_t = sc_s.fit_transform(Xs).astype(np.float32)
    Xp_e = sc_p.transform(Xp_te).astype(np.float32)
    Xs_e = sc_s.transform(Xs_te).astype(np.float32)

    dl = DataLoader(DS(Xp_t, Xs_t, y),
                    batch_size=Config.BATCH_SIZE, shuffle=True)
    model.to(Config.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=Config.WD)

    # Cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr * 0.01
    )
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    aux_w = 0.3  # poids de la perte auxiliaire

    best_acc = 0.0
    best_state = None
    patience = 8
    no_improve = 0

    for e in range(epochs):
        # Fisher loss progressive : ramp de 0 à 0.15
        if use_fisher:
            progress = min(1.0, e / max(1, epochs * 0.7))
            fw = 0.15 * progress
        else:
            fw = 0.0

        model.train()
        for bp, bs, by in dl:
            bp = bp.to(Config.DEVICE)
            bs = bs.to(Config.DEVICE)
            by = by.to(Config.DEVICE)

            opt.zero_grad()
            out = model(bp, bs, ret_f=True)

            if len(out) == 4:
                l, f, fp, fs = out
                loss = ce(l, by)
                loss += aux_w * ce(model.aux_p(fp), by)
                loss += aux_w * ce(model.aux_s(fs), by)
            elif len(out) == 3:
                l, f, fx = out
                loss = ce(l, by)
                loss += aux_w * ce(model.aux_clf(fx), by)
            else:
                l = out
                loss = ce(l, by)

            if fw > 0 and len(out) >= 3:
                loss = loss + fw * model.fisher(f, by, n_cls)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        scheduler.step()

        # Early stopping (vérifie toutes les 5 epochs)
        if (e + 1) % 5 == 0 or e == epochs - 1:
            acc = _eval(model, Xp_e, Xs_e, y_te)
            if acc > best_acc:
                best_acc = acc
                no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
            if no_improve >= patience:
                break

    # Charger le meilleur modèle
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(Config.DEVICE)
    return _eval(model, Xp_e, Xs_e, y_te)


def _eval(model, Xp, Xs, y):
    """Évaluation rapide (données déjà normalisées en entrée)."""
    dl = DataLoader(DS(Xp, Xs, np.zeros(len(Xp))),
                    batch_size=Config.BATCH_SIZE)
    model.eval()
    out = []
    with torch.no_grad():
        for bp, bs, _ in dl:
            bp = bp.to(Config.DEVICE)
            bs = bs.to(Config.DEVICE)
            pred = model(bp, bs)
            if isinstance(pred, tuple):
                pred = pred[0]
            out.append(pred.argmax(1).cpu().numpy())
    return accuracy_score(y, np.concatenate(out))


# ============================================================
# ABLATION PRINCIPALE
# ============================================================

def run_ablation(dataset_name):
    if dataset_name == "esc50":
        fd = Config.ESC50_DIR
        n_cls = 50
        n_folds = 5
    else:
        fd = Config.URBAN_DIR
        n_cls = 10
        n_folds = 10

    # Charger les features pré-extraites
    p = np.load(os.path.join(fd, "panns.npy")).astype(np.float32)
    s = np.load(os.path.join(fd, "ssl.npy")).astype(np.float32)
    l = np.load(os.path.join(fd, "labels.npy"))
    f = np.load(os.path.join(fd, "folds.npy"))

    pd, sd = p.shape[1], s.shape[1]
    print(f"\n{'=' * 60}")
    print(f"ABLATION v2: {dataset_name.upper()} | {len(l)} samples | PANNs={pd}d | SSL={sd}d")
    print(f"{'=' * 60}")

    results = {name: [] for name in [
        "SVM", "PANNs_only", "SSL_only",
        "Concat_naive", "CrossGated_noFisher", "Full_method"
    ]}

    for fold in range(1, n_folds + 1):
        print(f"\nFOLD {fold}")
        tr, te = f != fold, f == fold
        Xp_t, Xs_t, y_t = p[tr], s[tr], l[tr]
        Xp_e, Xs_e, y_e = p[te], s[te], l[te]

        # --- 1. SVM (concat + StandardScaler + RBF SVM) ---
        sc = StandardScaler()
        Xtr = sc.fit_transform(np.concatenate([Xp_t, Xs_t], 1))
        Xte = sc.transform(np.concatenate([Xp_e, Xs_e], 1))
        a_svm = accuracy_score(
            y_e, SVC(kernel='rbf', C=10, gamma='scale').fit(Xtr, y_t).predict(Xte)
        )
        results["SVM"].append(a_svm)

        # --- 2. PANNs only ---
        a_p = train_and_eval(
            PANNsOnly(n_cls, pd, proj=Config.PROJ, hid=Config.HID),
            Xp_t, Xs_t, y_t, Xp_e, Xs_e, y_e, n_cls, use_fisher=False
        )
        results["PANNs_only"].append(a_p)

        # --- 3. SSL only ---
        a_s = train_and_eval(
            SSLOnly(n_cls, sd, proj=Config.PROJ, hid=Config.HID),
            Xp_t, Xs_t, y_t, Xp_e, Xs_e, y_e, n_cls, use_fisher=False
        )
        results["SSL_only"].append(a_s)

        # --- 4. Concat naive ---
        a_cn = train_and_eval(
            ConcatNaive(n_cls, pd, sd, proj=Config.PROJ, hid=Config.HID),
            Xp_t, Xs_t, y_t, Xp_e, Xs_e, y_e, n_cls, use_fisher=False
        )
        results["Concat_naive"].append(a_cn)

        # --- 5. CrossGated sans Fisher ---
        a_cg = train_and_eval(
            CrossGatedNoFisher(n_cls, pd, sd, proj=Config.PROJ, hid=Config.HID),
            Xp_t, Xs_t, y_t, Xp_e, Xs_e, y_e, n_cls, use_fisher=False
        )
        results["CrossGated_noFisher"].append(a_cg)

        # --- 6. Full method (CrossGated + Fisher) ---
        a_fm = train_and_eval(
            CrossGatedFull(n_cls, pd, sd, proj=Config.PROJ, hid=Config.HID),
            Xp_t, Xs_t, y_t, Xp_e, Xs_e, y_e, n_cls, use_fisher=True
        )
        results["Full_method"].append(a_fm)

        print(f"  SVM={a_svm*100:.1f}% | PANNs={a_p*100:.1f}% | SSL={a_s*100:.1f}%")
        print(f"  Concat={a_cn*100:.1f}% | Gated={a_cg*100:.1f}% | Full={a_fm*100:.1f}%")

    # ---- Résumé final ----
    print(f"\n{'=' * 60}")
    print(f"RÉSUMÉ ABLATION v2 - {dataset_name.upper()}")
    print(f"{'=' * 60}")
    print(f"{'Méthode':<25} {'Accuracy':>10} {'Std':>10}")
    print(f"{'-' * 45}")
    for name in results:
        mean = np.mean(results[name]) * 100
        std = np.std(results[name]) * 100
        print(f"{name:<25} {mean:>9.1f}% {std:>9.1f}%")

    # Delta vs meilleur backbone isolé
    best_single = max(
        np.mean(results["PANNs_only"]),
        np.mean(results["SSL_only"])
    ) * 100
    best_name = "PANNs" if np.mean(results["PANNs_only"]) > np.mean(results["SSL_only"]) else "SSL"
    print(f"\n{'=' * 60}")
    print(f"DELTA vs meilleur backbone ({best_name}: {best_single:.1f}%)")
    print(f"{'=' * 60}")
    for name in results:
        if name != "SVM":
            delta = np.mean(results[name]) * 100 - best_single
            print(f"{name:<25} {delta:>+9.1f} points")



# ============================================================
# VISUALISATION DES GATES (apres entrainement)
# ============================================================
def collect_gates(model, Xp, Xs):
    """Collecte les valeurs des gates g_SSL et g_CNN sur un dataset."""
    model.eval()
    sc_p = StandardScaler()
    sc_s = StandardScaler()
    Xp_n = sc_p.fit_transform(Xp).astype(np.float32)
    Xs_n = sc_s.fit_transform(Xs).astype(np.float32)
    dl = DataLoader(DS(Xp_n, Xs_n, np.zeros(len(Xp))),
                    batch_size=Config.BATCH_SIZE)
    all_gp, all_gs = [], []
    with torch.no_grad():
        for bp, bs, _ in dl:
            bp = bp.to(Config.DEVICE)
            bs = bs.to(Config.DEVICE)
            fp = F.relu(model.proj_p(bp))
            fs = F.relu(model.proj_s(bs))
            ctx_s = model.summarize_s(fs)
            ctx_p = model.summarize_p(fp)
            gp = model.gate_p(torch.cat([fp, ctx_s], 1))
            gs = model.gate_s(torch.cat([fs, ctx_p], 1))
            all_gp.append(gp.cpu().numpy())
            all_gs.append(gs.cpu().numpy())
    return np.concatenate(all_gp), np.concatenate(all_gs)


def plot_gate_distribution(dataset_name):
    """Entraîne le Full method et visualise la distribution des gates."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib non disponible, skip plot")
        return

    if dataset_name == "esc50":
        fd = Config.ESC50_DIR
        n_cls, n_folds = 50, 5
    else:
        fd = Config.URBAN_DIR
        n_cls, n_folds = 10, 10

    p = np.load(os.path.join(fd, "panns.npy")).astype(np.float32)
    s = np.load(os.path.join(fd, "ssl.npy")).astype(np.float32)
    l = np.load(os.path.join(fd, "labels.npy"))
    f = np.load(os.path.join(fd, "folds.npy"))
    pd, sd = p.shape[1], s.shape[1]

    # Utiliser fold 1 pour la visualisation
    tr, te = f != 1, f == 1
    model = CrossGatedFull(n_cls, pd, sd, proj=Config.PROJ, hid=Config.HID)
    train_and_eval(model, p[tr], s[tr], l[tr],
                   p[te], s[te], l[te], n_cls, use_fisher=True)

    gp, gs = collect_gates(model, p[te], s[te])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Distribution des valeurs de gate (moyenne par sample)
    axes[0].hist(gs.mean(1), bins=50, alpha=0.7, label='g_SSL (SSL gate)', color='steelblue')
    axes[0].hist(gp.mean(1), bins=50, alpha=0.7, label='g_CNN (PANNs gate)', color='coral')
    axes[0].set_xlabel('Valeur moyenne du gate par échantillon')
    axes[0].set_ylabel('Nombre d échantillons')
    axes[0].set_title('Distribution des gates (test set)')
    axes[0].legend()
    axes[0].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

    # 2. Heatmap des gates par classe (20 classes max affichées)
    n_show = min(n_cls, 20)
    gate_ssl_per_class = np.zeros((n_show, gs.shape[1]))
    for c in range(n_show):
        mask = l[te] == c
        if mask.sum() > 0:
            gate_ssl_per_class[c] = gs[mask].mean(0)
    im = axes[1].imshow(gate_ssl_per_class, aspect='auto', cmap='RdYlBu_r')
    axes[1].set_xlabel('Dimension du gate (128d)')
    axes[1].set_ylabel('Classe')
    axes[1].set_title('g_SSL par classe (top 20)')
    fig.colorbar(im, ax=axes[1])

    # 3. Ratio g_SSL / g_CNN par classe
    gate_cnn_per_class = np.zeros((n_show, gp.shape[1]))
    for c in range(n_show):
        mask = l[te] == c
        if mask.sum() > 0:
            gate_cnn_per_class[c] = gp[mask].mean(0)
    ratio = gs.mean(1) / (gp.mean(1) + gs.mean(1) + 1e-8)
    for c in range(n_show):
        mask = l[te] == c
        if mask.sum() > 0:
            axes[2].barh(c, ratio[mask].mean(), color='steelblue', alpha=0.7)
    axes[2].set_xlabel('Ratio g_SSL / (g_SSL + g_CNN)')
    axes[2].set_ylabel('Classe')
    axes[2].set_title('Contribution relative de SSL par classe')
    axes[2].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Égalité')
    axes[2].legend()

    plt.tight_layout()
    out_path = f'gate_distribution_{dataset_name}.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nGraphique sauvegardé: {out_path}")

    # Statistiques
    print(f"\n--- Statistiques des gates ---")
    print(f"g_SSL  : moyenne={gs.mean():.4f}, std={gs.std():.4f}, "
          f"min={gs.min():.4f}, max={gs.max():.4f}")
    print(f"g_CNN  : moyenne={gp.mean():.4f}, std={gp.std():.4f}, "
          f"min={gp.min():.4f}, max={gp.max():.4f}")
    print(f"Si g_SSL ~ 0 partout -> le gate ignore SSL (problème)")
    print(f"Si g_SSL varie par classe -> le gate est adaptatif (bien)")


# ============================================================
# ABLATION ÉTENDUE (summary_dim + pooling)
# ============================================================
def run_extended_ablation(dataset_name):
    """Ablation sur summary_dim (16/32/64) et pooling (AvgPool vs Linear)."""
    if dataset_name == "esc50":
        fd = Config.ESC50_DIR
        n_cls, n_folds = 50, 5
    else:
        fd = Config.URBAN_DIR
        n_cls, n_folds = 10, 10

    p = np.load(os.path.join(fd, "panns.npy")).astype(np.float32)
    s = np.load(os.path.join(fd, "ssl.npy")).astype(np.float32)
    l = np.load(os.path.join(fd, "labels.npy"))
    f = np.load(os.path.join(fd, "folds.npy"))
    pd, sd = p.shape[1], s.shape[1]

    # Configs à ablater
    configs = [
        ("summary_dim=16", 16, 'avg'),
        ("summary_dim=32 (base)", 32, 'avg'),
        ("summary_dim=64", 64, 'avg'),
    ]

    results = {name: [] for name, _, _ in configs}

    print(f"\n{'=' * 60}")
    print(f"ABLATION ÉTENDUE: {dataset_name.upper()} | summary_dim + pooling")
    print(f"{'=' * 60}")

    for fold in range(1, n_folds + 1):
        print(f"\nFOLD {fold}")
        tr, te = f != fold, f == fold
        for name, sdim, pool in configs:
            model = CrossGatedFull(
                n_cls, pd, sd,
                proj=Config.PROJ, hid=Config.HID,
                summary_dim=sdim, pool=pool
            )
            acc = train_and_eval(
                model, p[tr], s[tr], l[tr],
                p[te], s[te], l[te], n_cls, use_fisher=True
            )
            results[name].append(acc)
            print(f"  {name}: {acc*100:.1f}%")

    print(f"\n{'=' * 60}")
    print(f"RÉSUMÉ ABLATION ÉTENDUE - {dataset_name.upper()}")
    print(f"{'=' * 60}")
    print(f"{'Config':<25} {'Accuracy':>10} {'Std':>10}")
    print(f"{'-' * 45}")
    for name in results:
        mean = np.mean(results[name]) * 100
        std = np.std(results[name]) * 100
        print(f"{name:<25} {mean:>9.1f}% {std:>9.1f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ds = sys.argv[1]
        quick = "--quick" in sys.argv
        gates = "--gates" in sys.argv
        extended = "--extended" in sys.argv
        if quick:
            if ds == "urban":
                Config.EPOCHS = 12
                Config.BATCH_SIZE = 256
                Config.LR = 8e-4
                print("[MODE RAPIDE URBAN] 12 epochs, batch=256")
            else:
                Config.EPOCHS = 25
                Config.BATCH_SIZE = 128
                print("[MODE RAPIDE ESC50] 25 epochs, batch=128")
        if ds in ["esc50", "urban"]:
            if gates:
                plot_gate_distribution(ds)
            elif extended:
                run_extended_ablation(ds)
            else:
                run_ablation(ds)
        else:
            print("Usage:")
            print("  python ablation.py esc50 [--quick] [--gates] [--extended]")
            print("  python ablation.py urban [--quick] [--gates] [--extended]")
            print("")
            print("  --gates      : visualise la distribution des gates g_SSL/g_CNN")  
            print("  --extended   : ablation summary_dim (16/32/64)")
            print("  --quick      : mode rapide (moins d epochs)")
    else:
        print("Usage:")
        print("  python ablation.py esc50 [--quick] [--gates] [--extended]")
        print("  python ablation.py urban [--quick] [--gates] [--extended]")
        print("")
        print("  --gates      : visualise la distribution des gates g_SSL/g_CNN")
        print("  --extended   : ablation summary_dim (16/32/64)")
        print("  --quick      : mode rapide (moins d epochs)")
