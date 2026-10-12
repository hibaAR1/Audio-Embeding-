"""
URBANSOUND8K - COMPLETE (quick + full + fine-tuning)
Usage:
  python urban_final.py --extract           (quick, 200 fichiers, ~10min)
  python urban_final.py --extract --full    (complet, 8732 fichiers, ~3-5h)
  python urban_final.py                     (evaluation)
"""
import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

class Config:
    US8K_PATH = r"C:\Users\21268\Downloads\UrbanSound8K\UrbanSound8K"
    FEATURES_DIR = r"C:\Users\21268\Desktop\AUDIO EMBEDDING\features_urban_final"
    SR = 32000
    AST_SR = 16000
    N_SAMPLES = SR * 5
    N_CLASSES = 10
    BATCH_SIZE = 16
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_audio(path):
    try:
        data, sr = sf.read(path, dtype='float32')
        if len(data.shape) > 1: data = np.mean(data, axis=1)
        if sr != Config.SR:
            from scipy.signal import resample
            data = resample(data, int(len(data) * Config.SR / sr))
        if len(data) < Config.N_SAMPLES:
            data = np.pad(data, (0, Config.N_SAMPLES - len(data)))
        else:
            data = data[:Config.N_SAMPLES]
        return data
    except:
        return np.zeros(Config.N_SAMPLES, dtype=np.float32)


class DS(Dataset):
    def __init__(self, p, s, y):
        self.p, self.s, self.y = torch.FloatTensor(p), torch.FloatTensor(s), torch.LongTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.p[i], self.s[i], self.y[i]


class Net(nn.Module):
    def __init__(self, n_cls, panns_dim, ssl_dim, proj=128, hid=128):
        super().__init__()
        self.n_cls = n_cls
        # Adapter les dimensions reelles (527 pour PANNs, pas 2048)
        self.proj_p = nn.Linear(panns_dim, proj)
        self.proj_s = nn.Linear(ssl_dim, proj)
        self.gp = nn.Sequential(nn.Linear(proj*2, proj), nn.Sigmoid())
        self.gs = nn.Sequential(nn.Linear(proj*2, proj), nn.Sigmoid())
        self.fproj = nn.Linear(proj*2, proj*2)
        self.mlp = nn.Sequential(nn.Linear(proj*2, hid), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(hid, hid//2), nn.ReLU())
        self.clf = nn.Linear(hid//2, n_cls)

    def forward(self, p, s, ret_f=False):
        p = F.relu(self.proj_p(p))
        s = F.relu(self.proj_s(s))
        c = torch.cat([p, s], 1)
        fused = self.fproj(torch.cat([p*self.gp(c), s*self.gs(c)], 1))
        f = self.mlp(fused)
        l = self.clf(f)
        return (l, f) if ret_f else l

    def fisher(self, f, y):
        gm = f.mean(0)
        inter = torch.tensor(0.0, device=f.device)
        intra = torch.tensor(0.0, device=f.device)
        for c in range(self.n_cls):
            m = y == c
            if m.sum() > 0:
                cm = f[m].mean(0)
                inter += ((cm - gm)**2).sum()
                intra += ((f[m] - cm)**2).sum()
        if inter < 1e-6: return torch.tensor(0.0, device=f.device, requires_grad=True)
        return torch.clamp(intra, 1e-6) / inter

    def train_pipe(self, Xp, Xs, y, ep=60, lr=5e-4):
        dl = DataLoader(DS(Xp, Xs, y), batch_size=Config.BATCH_SIZE, shuffle=True)
        self.to(Config.DEVICE).train()
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-3)
        ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        for e in range(ep):
            fw = min(0.3, 0.01 + e*0.005)
            for bp, bs, by in dl:
                bp, bs, by = bp.to(Config.DEVICE), bs.to(Config.DEVICE), by.to(Config.DEVICE)
                opt.zero_grad()
                l, f = self(bp, bs, ret_f=True)
                loss = ce(l, by) + fw * self.fisher(f, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                opt.step()

    def predict(self, Xp, Xs):
        self.eval()
        dl = DataLoader(DS(Xp, Xs, np.zeros(len(Xp))), batch_size=Config.BATCH_SIZE)
        out = []
        with torch.no_grad():
            for bp, bs, _ in dl:
                out.append(self(bp.to(Config.DEVICE), bs.to(Config.DEVICE)).argmax(1).cpu().numpy())
        return np.concatenate(out)


def extract_features(max_per_class=None):
    csv_path = os.path.join(Config.US8K_PATH, "metadata", "UrbanSound8K.csv")
    audio_root = os.path.join(Config.US8K_PATH, "audio")
    df = pd.read_csv(csv_path)

    if max_per_class:
        df = df.groupby('classID').apply(lambda x: x.head(max_per_class)).reset_index(drop=True)

    print(f"Extraction: {len(df)} fichiers")

    from panns_inference import AudioTagging
    pm = AudioTagging(checkpoint_path=None, device=str(Config.DEVICE))

    from transformers import ASTModel, ASTFeatureExtractor
    sm = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(Config.DEVICE).eval()
    se = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    from scipy.signal import resample as rs

    all_p, all_s, all_l, all_f = [], [], [], []

    for i in range(len(df)):
        row = df.iloc[i]
        path = os.path.join(audio_root, f"fold{int(row['fold'])}", row['slice_file_name'])
        wav = load_audio(path)
        bt = torch.FloatTensor(wav).unsqueeze(0).to(Config.DEVICE)

        with torch.no_grad():
            pf, _ = pm.inference(bt)
            pf = torch.FloatTensor(pf).to(Config.DEVICE)

            wav_16k = rs(wav, int(Config.N_SAMPLES * Config.AST_SR / Config.SR))
            inp = se(wav_16k, sampling_rate=Config.AST_SR, return_tensors="pt")
            inp = {k: v.to(Config.DEVICE) for k, v in inp.items()}
            sf_f = sm(**inp).last_hidden_state.mean(dim=1)

        all_p.append(pf.cpu().numpy())
        all_s.append(sf_f.cpu().numpy())
        all_l.append(int(row['classID']))
        all_f.append(int(row['fold']))

        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(df)}")

    all_p = np.concatenate(all_p)
    all_s = np.concatenate(all_s)

    os.makedirs(Config.FEATURES_DIR, exist_ok=True)
    np.save(os.path.join(Config.FEATURES_DIR, "panns.npy"), all_p)
    np.save(os.path.join(Config.FEATURES_DIR, "ssl.npy"), all_s)
    np.save(os.path.join(Config.FEATURES_DIR, "labels.npy"), np.array(all_l))
    np.save(os.path.join(Config.FEATURES_DIR, "folds.npy"), np.array(all_f))
    print(f"Termine! {len(df)} fichiers.")
    print(f"  PANNs shape: {all_p.shape}")
    print(f"  SSL shape:   {all_s.shape}")


def evaluate():
    fd = Config.FEATURES_DIR
    p = np.load(os.path.join(fd, "panns.npy"))
    s = np.load(os.path.join(fd, "ssl.npy"))
    l = np.load(os.path.join(fd, "labels.npy"))
    f = np.load(os.path.join(fd, "folds.npy"))

    # Detecter les dimensions reelles
    panns_dim = p.shape[1]
    ssl_dim = s.shape[1]
    print(f"{p.shape[0]} echantillons | PANNs={panns_dim}d | SSL={ssl_dim}d")

    n_folds = len(np.unique(f))
    if n_folds > 10: n_folds = 5

    res_svm, res_e2e = [], []

    for fold in range(1, n_folds + 1):
        print(f"\nFOLD {fold}")
        tr, te = f != fold, f == fold
        Xp_t, Xs_t, y_t = p[tr], s[tr], l[tr]
        Xp_e, Xs_e, y_e = p[te], s[te], l[te]
        print(f"  Train: {len(y_t)} | Test: {len(y_e)}")

        # SVM
        sc = StandardScaler()
        Xtr = sc.fit_transform(np.concatenate([Xp_t, Xs_t], 1))
        Xte = sc.transform(np.concatenate([Xp_e, Xs_e], 1))
        svc = SVC(kernel='rbf', C=10).fit(Xtr, y_t)
        a_svm = accuracy_score(y_e, svc.predict(Xte))
        res_svm.append(a_svm)

        # E2E avec les vraies dimensions
        m = Net(n_cls=Config.N_CLASSES, panns_dim=panns_dim, ssl_dim=ssl_dim)
        m.train_pipe(Xp_t, Xs_t, y_t, ep=60)
        a_e2e = accuracy_score(y_e, m.predict(Xp_e, Xs_e))
        res_e2e.append(a_e2e)

        print(f"  SVM={a_svm*100:.1f}% | E2E={a_e2e*100:.1f}%")

    print(f"\n{'='*50}")
    print(f"SVM:     {np.mean(res_svm)*100:.1f}% +/- {np.std(res_svm)*100:.1f}%")
    print(f"E2E:     {np.mean(res_e2e)*100:.1f}% +/- {np.std(res_e2e)*100:.1f}%")
    print(f"\nESC-50 (v8): 91.40%")
    print(f"US8K:        {np.mean(res_e2e)*100:.1f}% (E2E)")


if __name__ == "__main__":
    if "--extract" in sys.argv:
        if "--full" in sys.argv:
            extract_features(max_per_class=None)
        else:
            extract_features(max_per_class=20)
    else:
        if not os.path.exists(os.path.join(Config.FEATURES_DIR, "panns.npy")):
            print("Lance: python urban_final.py --extract")
        else:
            evaluate()
      
               