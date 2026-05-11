import os
import torch
import librosa
import numpy as np
import pandas as pd
import urllib.request
from panns_inference import AudioTagging
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

# ======================================
# 1. CONFIGURATION
# ======================================
PANN_PATH = "C:/Users/21268/panns_data"
os.makedirs(PANN_PATH, exist_ok=True)
os.environ['PANN_DATA'] = PANN_PATH

model_path = os.path.join(PANN_PATH, "Cnn14_mAP=0.431.pth")
if not os.path.exists(model_path):
    print("📥 Téléchargement PANNs...")
    url = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth"
    urllib.request.urlretrieve(url, model_path)

# ======================================
# 2. CHARGEMENT DES MODELES
# ======================================
print("🧠 Chargement des modèles...")
model_panns = AudioTagging(checkpoint_path=model_path, device='cpu')

ssl_model_name = "facebook/wav2vec2-base-960h"
extractor_ssl = Wav2Vec2FeatureExtractor.from_pretrained(ssl_model_name)
model_ssl = Wav2Vec2Model.from_pretrained(ssl_model_name)
model_ssl.eval()

# ======================================
# 3. DOSSIERS ET METADONNÉES
# ======================================
DATA_DIR = 'ESC-50'
audio_dir = os.path.join(DATA_DIR, 'audio')
os.makedirs('embeddings_panns', exist_ok=True)
os.makedirs('embeddings_ssl', exist_ok=True)

# CHARGER TOUS LES FICHIERS DEPUIS LE CSV
df = pd.read_csv(f'{DATA_DIR}/meta/esc50.csv')
all_audio_files = df['filename'].tolist() # On prend les 2000 noms ici

# ======================================
# 4. BOUCLE D'EXTRACTION TOTALE
# ======================================
print(f"🚀 Extraction lancée sur {len(all_audio_files)} fichiers...")
print("☕ C'est le moment de prendre un café, cela peut prendre 15-20 minutes.")

for i, filename in enumerate(all_audio_files, 1):
    audio_path = os.path.join(audio_dir, filename)
    base_name = filename.replace('.wav', '.npy')
    
    # ÉVITER DE REFAIRE LE TRAVAIL SI LE FICHIER EXISTE DÉJÀ
    if os.path.exists(os.path.join('embeddings_panns', base_name)):
        continue

    try:
        # A. Chargement Multi-SR
        # prq?le Sample Rate (SR) est la fréquence d'échantillonnage : c'est le nombre de "photos" du son que l'ordinateur prend par seconde. ---

            #: Le chargement multi-SR permet de donner à chaque "cerveau" 

            # d'IA le son exactement comme il a l'habitude de l'entendre. 

            # C'est ce qui garantit que ta comparaison est juste (fair).#
        y32, _ = librosa.load(audio_path, sr=32000)
        y16, _ = librosa.load(audio_path, sr=16000)

        # B. PANNs
        _, emb_panns = model_panns.inference(y32[None, :])
        
        # C. Wav2Vec
        inputs = extractor_ssl(y16, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = model_ssl(**inputs)
        emb_ssl = outputs.last_hidden_state.mean(dim=1).numpy()

        # D. Sauvegarde
        np.save(os.path.join('embeddings_panns', base_name), emb_panns)
        np.save(os.path.join('embeddings_ssl', base_name), emb_ssl)

        if i % 50 == 0: # Affiche un message tous les 50 fichiers
            print(f"✅ Progression : {i}/{len(all_audio_files)} fichiers traités...")

    except Exception as e:
        print(f"❌ Erreur sur {filename} : {e}")

print("\n✨ TERMINÉ ! Tes 2000 signatures sont prêtes.")