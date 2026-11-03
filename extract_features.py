"""
Extraction des représentations PANNs CNN14 et AST pour ESC-50 et UrbanSound8K.

Ce que fait ce script (conforme à l'article) :
  - charge chaque fichier en mono à 32 kHz, tronque ou complète à 5 s ;
  - PANNs CNN14 (gelé, 32 kHz) : embedding de 2048 dimensions
    (la 2e sortie de panns_inference, PAS les 527 scores de classes) ;
  - AST MIT/ast-finetuned-audioset-10-10-0.4593 (gelé, 16 kHz) :
      ast.npy      = moyenne de TOUS les tokens de la dernière couche (768 d)
      ast_real.npy = moyenne des seuls patches qui couvrent l'audio réel (768 d)
    (l'extracteur AST complète chaque entrée jusqu'à 1024 trames, soit ~10 s :
     ast.npy inclut donc ce remplissage, ast_real.npy l'exclut).

Usage :
  python extract_features.py esc50 --data "C:\\chemin\\vers\\ESC-50"
  python extract_features.py urban --data "C:\\chemin\\vers\\UrbanSound8K"

Les fichiers sont écrits dans features_v2/esc50/ ou features_v2/urban/.
Si l'extraction est interrompue, relance la même commande : elle reprend là où elle s'est arrêtée.
"""
import os
import sys
import math
import time
import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch

SR = 32000
AST_SR = 16000
DURATION = 5.0
N_SAMPLES = int(SR * DURATION)
AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
CHECKPOINT_EVERY = 200

PANNS_DIR = Path.home() / "panns_data"
PANNS_CKPT = PANNS_DIR / "Cnn14_mAP=0.431.pth"
PANNS_CKPT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
LABELS_CSV = PANNS_DIR / "class_labels_indices.csv"
LABELS_CSV_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"


def ensure_panns_files():
    """panns_inference télécharge ses fichiers avec wget, absent sous Windows : on le fait ici."""
    PANNS_DIR.mkdir(parents=True, exist_ok=True)
    if not LABELS_CSV.exists():
        print("Téléchargement de class_labels_indices.csv ...")
        urllib.request.urlretrieve(LABELS_CSV_URL, LABELS_CSV)
    if not PANNS_CKPT.exists() or PANNS_CKPT.stat().st_size < 300_000_000:
        print("Téléchargement du checkpoint PANNs CNN14 (~330 Mo) ...")
        urllib.request.urlretrieve(PANNS_CKPT_URL, PANNS_CKPT)


def list_files(dataset, data_dir):
    """Renvoie une table avec chemin, label, pli et nom de fichier."""
    data_dir = Path(data_dir)
    if dataset == "esc50":
        meta = pd.read_csv(data_dir / "meta" / "esc50.csv")
        return pd.DataFrame({
            "path": [str(data_dir / "audio" / f) for f in meta["filename"]],
            "label": meta["target"].astype(int),
            "fold": meta["fold"].astype(int),
            "name": meta["filename"],
        })
    # UrbanSound8K : accepte le dossier parent ou le dossier interne
    if not (data_dir / "metadata").exists() and (data_dir / "UrbanSound8K" / "metadata").exists():
        data_dir = data_dir / "UrbanSound8K"
    meta = pd.read_csv(data_dir / "metadata" / "UrbanSound8K.csv")
    return pd.DataFrame({
        "path": [str(data_dir / "audio" / f"fold{k}" / f)
                 for k, f in zip(meta["fold"], meta["slice_file_name"])],
        "label": meta["classID"].astype(int),
        "fold": meta["fold"].astype(int),
        "name": meta["slice_file_name"],
    })


def load_audio(path):
    """Mono, 32 kHz, tronqué ou complété par des zéros à 5 s. Renvoie aussi la durée réelle."""
    wav, _ = librosa.load(path, sr=SR, mono=True)
    real_len = min(len(wav), N_SAMPLES)
    if len(wav) < N_SAMPLES:
        wav = np.pad(wav, (0, N_SAMPLES - len(wav)))
    else:
        wav = wav[:N_SAMPLES]
    return wav.astype(np.float32), real_len / SR


def ast_real_time_patches(real_seconds, n_time_patches):
    """Nombre de colonnes temporelles de patches AST qui couvrent de l'audio réel.
    Trames fbank : fenêtre 25 ms, pas 10 ms ; patches de 16 trames, pas de 10 trames."""
    n_frames = max(1, int((real_seconds * AST_SR - 400) // 160) + 1)
    return int(min(n_time_patches, max(1, math.ceil(n_frames / 10))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["esc50", "urban"])
    parser.add_argument("--data", required=True, help="Dossier du jeu de données")
    parser.add_argument("--out", default="features_v2", help="Dossier de sortie")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / "_partial.npz"

    files = list_files(args.dataset, args.data)
    missing = [p for p in files["path"] if not os.path.exists(p)]
    if missing:
        print(f"ERREUR : {len(missing)} fichiers audio introuvables, par exemple :\n  {missing[0]}")
        print("Vérifie le chemin donné avec --data.")
        sys.exit(1)
    print(f"{args.dataset} : {len(files)} fichiers, {files['label'].nunique()} classes, "
          f"{files['fold'].nunique()} plis | device = {device}")

    ensure_panns_files()
    from panns_inference import AudioTagging
    from transformers import ASTFeatureExtractor, ASTModel

    panns = AudioTagging(checkpoint_path=str(PANNS_CKPT), device=device)
    ast_fe = ASTFeatureExtractor.from_pretrained(AST_MODEL)
    ast = ASTModel.from_pretrained(AST_MODEL).to(device).eval()

    n = len(files)
    P = np.zeros((n, 2048), np.float32)
    A = np.zeros((n, 768), np.float32)
    A_real = np.zeros((n, 768), np.float32)
    start = 0
    if partial.exists():
        z = np.load(partial)
        start = int(z["done"])
        P[:start], A[:start], A_real[:start] = z["P"][:start], z["A"][:start], z["A_real"][:start]
        print(f"Reprise à partir du fichier {start}/{n}")

    t0 = time.time()
    for i in range(start, n):
        wav32, real_sec = load_audio(files["path"].iloc[i])

        # PANNs : (scores 527, embedding 2048) -> on garde l'embedding
        _, emb = panns.inference(wav32[None, :])
        emb = np.asarray(emb).reshape(-1)
        if emb.shape[0] != 2048:
            raise ValueError(f"Embedding PANNs de taille {emb.shape[0]} au lieu de 2048")
        P[i] = emb

        # AST
        wav16 = librosa.resample(wav32, orig_sr=SR, target_sr=AST_SR)
        inputs = ast_fe(wav16, sampling_rate=AST_SR, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            hidden = ast(**inputs).last_hidden_state[0]          # (2 + 12*101, 768)
        A[i] = hidden.mean(0).cpu().numpy()
        patches = hidden[2:]                                     # sans les tokens CLS et distillation
        n_time = patches.shape[0] // 12                          # 101 colonnes temporelles
        grid = patches.reshape(12, n_time, -1)                   # (fréquence, temps, 768)
        k = ast_real_time_patches(real_sec, n_time)
        A_real[i] = grid[:, :k].mean((0, 1)).cpu().numpy()

        done = i + 1
        if done % CHECKPOINT_EVERY == 0 or done == n:
            np.savez(partial, P=P, A=A, A_real=A_real, done=done)
            speed = (done - start) / (time.time() - t0)
            remaining = (n - done) / max(speed, 1e-9) / 60
            print(f"  {done}/{n}  ({speed:.1f} fichiers/s, reste ~{remaining:.0f} min)", flush=True)

    np.save(out_dir / "panns.npy", P)
    np.save(out_dir / "ast.npy", A)
    np.save(out_dir / "ast_real.npy", A_real)
    np.save(out_dir / "labels.npy", files["label"].to_numpy(np.int64))
    np.save(out_dir / "folds.npy", files["fold"].to_numpy(np.int64))
    np.save(out_dir / "filenames.npy", files["name"].to_numpy(object))
    partial.unlink(missing_ok=True)

    print("\nTerminé.")
    print(f"  panns.npy    {P.shape}  (doit être (N, 2048))")
    print(f"  ast.npy      {A.shape}")
    print(f"  ast_real.npy {A_real.shape}")
    print(f"  Dossier : {out_dir.resolve()}")


if __name__ == "__main__":
    main()
