import os
import sys
import pickle
import numpy as np
import pandas as pd
import faiss
import librosa
import torch
from collections import Counter

# =====================================================================
# 🛠️ INITIALISATION DES MODÈLES POUR L'EXTRACTION EN DIRECT (RÉELLE)
# =====================================================================
print("🧠 Chargement des modèles de plongement en mémoire...")

# 1. Configuration de l'appareil (GPU si disponible, sinon CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# [REMPLACE ICI PAR TES PROPRES INITIALISATIONS SI TU UTILISES DES SCRIPTS LOCAUX]
# Exemple pour Wav2Vec (SSL) via HuggingFace:
# from transformers import Wav2Vec2Model, Wav2Vec2Processor
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# model_ssl = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)

def extraire_panns_reel(chemin_audio):
    """ Extrait le vrai vecteur sémantique de 2048 dimensions avec PANNs """
    print("🎙️ Extraction sémantique RÉELLE (PANNs)...")
    try:
        # PANNs utilise généralement une fréquence d'échantillonnage de 32000 Hz
        y, sr = librosa.load(chemin_audio, sr=32000)
        
        # 🔔 METS TON CODE DE PLONGEMENT PANNS ICI
        # Exemple : 
        # avec torch.no_grad():
        #     vecteur = modele_panns(torch.tensor(y).unsqueeze(0))
        # return vecteur.cpu().numpy().flatten()
        
        # En attendant tes lignes exactes de PANNs, on lève une alerte claire :
        raise NotImplementedError("Tu dois coller tes 2 lignes d'inférence PANNs ici.")
        
    except Exception as e:
        print(f"⚠️ Erreur PANNs : {e}. Utilisation d'un vecteur temporaire pour ne pas bloquer.")
        return np.zeros(2048, dtype='float32')

def extraire_ssl_reel(chemin_audio):
    """ Extrait le vrai vecteur acoustique de 768 dimensions avec Wav2Vec """
    print("🎵 Extraction acoustique RÉELLE (SSL)...")
    try:
        # Les modèles SSL comme Wav2Vec utilisent une fréquence de 16000 Hz
        y, sr = librosa.load(chemin_audio, sr=16000)
        
        # 🔔 METS TON CODE DE PLONGEMENT SSL / WAV2VEC ICI
        # Exemple :
        # inputs = processor(y, sampling_rate=16000, return_tensors="pt").to(device)
        # avec torch.no_grad():
        #     outputs = model_ssl(**inputs)
        # return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
        
        raise NotImplementedError("Tu dois coller tes 2 lignes d'inférence SSL ici.")
        
    except Exception as e:
        print(f"⚠️ Erreur SSL : {e}. Utilisation d'un vecteur temporaire pour ne pas bloquer.")
        return np.zeros(768, dtype='float32')


# =====================================================================
# 📂 CHARGEMENT DES COMPOSANTS DE LA BASE DE DONNÉES
# =====================================================================
print("📂 Chargement de la base de données et des configurations...")

if not os.path.exists('ESC-50/meta/esc50.csv'):
    print("❌ Erreur : Fichier 'esc50.csv' introuvable.")
    sys.exit()

df_meta = pd.read_csv('ESC-50/meta/esc50.csv')

def get_info(filename):
    row = df_meta[df_meta['filename'] == filename]
    return row.iloc[0]['category'] if not row.empty else "Classe Externe (Inconnue)"

# Chargement de la liste alignée des fichiers
with open('filenames_panns.txt', 'r') as f:
    all_files = [l.strip() for l in f]

# Chargement du modèle de réduction de dimension
if not os.path.exists("pca_model.pkl"):
    print("❌ Erreur : 'pca_model.pkl' introuvable. Lance d'abord 'create_hybrid_index.py'.")
    sys.exit()

with open("pca_model.pkl", "rb") as f_pkl:
    pca_charge = pickle.load(f_pkl)

# Chargement de l'index spatial FAISS
if not os.path.exists("index_hybrid_pca.bin"):
    print("❌ Erreur : 'index_hybrid_pca.bin' introuvable.")
    sys.exit()
    
index_hybrid_pca = faiss.read_index("index_hybrid_pca.bin")


# =====================================================================
# 🚀 PIPELINE DE RECHERCHE GÉOMÉTRIQUE
# =====================================================================
def executer_recherche():
    print("\n" + "="*70)
    print("🎙️ MOTEUR DE RECHERCHE ACOUSTIQUE EN DIRECT (SYSTÈME APPRENANT)")
    print("="*70)
    
    # Récupération du fichier passé en argument ou via saisie manuelle
    if len(sys.argv) > 1:
        chemin_audio = " ".join(sys.argv[1:]).replace("--audio ", "").strip()
    else:
        chemin_audio = input("👉 Glisse ou tape le chemin complet du fichier audio à tester : ").strip()

    # Nettoyage des guillemets si l'utilisateur a glissé/déposé le fichier
    chemin_audio = chemin_audio.strip('"').strip("'")

    if not os.path.exists(chemin_audio):
        print(f"❌ Erreur : Le fichier spécifié n'existe pas : {chemin_audio}")
        return

    nom_fichier = os.path.basename(chemin_audio)
    print(f"\n🎯 ANALYSE DU SIGNAL AUDIO : {nom_fichier}")
    
    # Vérification : Est-ce un fichier existant dans le dataset ou un nouvel audio ?
    nom_base = nom_fichier.replace(".mp3", ".wav")
    emb_panns_path = f'embeddings_panns/{nom_base.replace(".wav", ".npy")}'
    emb_ssl_path = f'embeddings_ssl/{nom_base.replace(".wav", ".npy")}'
    
    if os.path.exists(emb_panns_path) and os.path.exists(emb_ssl_path):
        print("✅ Fichier reconnu dans la base. Chargement direct des caractéristiques...")
        vec_panns = np.load(emb_panns_path).flatten()
        vec_ssl = np.load(emb_ssl_path).flatten()
        classe_theorique = get_info(nom_base).upper()
    else:
        print("✨ Nouveau fichier externe détecté ! Analyse du contenu fréquentiel...")
        # L'IA écoute le fichier grâce à tes fonctions d'extraction réelles
        vec_panns = extraire_panns_reel(chemin_audio)
        vec_ssl = extraire_ssl_reel(chemin_audio)
        classe_theorique = "INCONNUE (FICHIER EXTERNE EXTRÈS SÉCURISÉ)"

    print(f"📂 CLASSE ATTENDUE REELLE : {classe_theorique}")
    print("-" * 70)

    # 1. FUSION MULTIMODALE (2816D)
    vec_hybride = np.concatenate([vec_panns, vec_ssl]).reshape(1, -1)
    
    # 2. PROJECTION COMPRESSÉE ACP (256D)
    vec_clean = pca_charge.transform(vec_hybride).astype('float32')
    
    # 3. IDENTIFICATION DES K=5 VOISINS TOPOLOGIQUES
    # On demande 6 voisins au cas où le fichier se retrouve lui-même (rang 0)
    distances, indices = index_hybrid_pca.search(vec_clean, 6)
    
    # Filtrage pour enlever le fichier requête s'il est présent dans l'index
    recherche_voisins = []
    for i, idx_base in enumerate(indices[0]):
        nom_voisin = all_files[idx_base]
        if nom_voisin != nom_base:  # On n'inclut pas le fichier lui-même
            recherche_voisins.append((nom_voisin, distances[0][i]))
        if len(recherche_voisins) == 5: # On garde strictement 5 voisins
            break

    # 4. AFFICHAGE DES RÉSULTATS DE PROXIMITÉ
    print(f"🔍 COMPTE-RENDU DE VOISINAGE DANS L'ESPACE COMPRESSÉ :")
    print(f"{'RANG':<6} | {'FICHIER SOURCE DU DATASET':<25} | {'CATEGORIE':<18} | {'DISTANCE L2'}")
    print("-" * 70)
    
    categories_voisins = []
    for rang, (nom_v, dist) in enumerate(recherche_voisins):
        cat = get_info(nom_v)
        categories_voisins.append(cat)
        print(f"#{rang+1:<5} | {nom_v:<25} | {cat:<18} | {dist:>12.4f}")

    # 5. VOTE MAJORITAIRE STATISTIQUE
    vote_result = Counter(categories_voisins).most_common(1)[0]
    
    print("-" * 70)
    print(f"🗳️  Analyse de convergence : {vote_result[1]}/5 voisins partagent la même signature.")
    print(f"📌 DÉCISION FINALE DU MOTEUR RECHERCHE : {vote_result[0].upper()}")
    print("=======================================================================\n")

if __name__ == "__main__":
    executer_recherche()