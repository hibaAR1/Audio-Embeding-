import pandas as pd
import numpy as np
import faiss
from sklearn.decomposition import PCA
import pickle
import os

print("🔍 Étape 1 : Alignement et lecture des fichiers .npy...")

# Chargement de la liste officielle des fichiers
with open('filenames_panns.txt', 'r') as f:
    all_files = [l.strip() for l in f]

vecteurs_hybrides_list = []
fichiers_valides = []

# Fusion par alignement strict des noms de fichiers
for f in all_files:
    p_panns = f'embeddings_panns/{f.replace(".wav", ".npy")}'
    p_ssl = f'embeddings_ssl/{f.replace(".wav", ".npy")}'
    
    if os.path.exists(p_panns) and os.path.exists(p_ssl):
        v_panns = np.load(p_panns).flatten()
        v_ssl = np.load(p_ssl).flatten()
        
        # Concaténation propre sur le même fichier
        v_hybride = np.concatenate([v_panns, v_ssl])
        vecteurs_hybrides_list.append(v_hybride)
        fichiers_valides.append(f)

vecteurs_hybrides = np.array(vecteurs_hybrides_list, dtype='float32')
print(f"🤝 Étape 2 : Fusion alignée réussie. Matrice globale : {vecteurs_hybrides.shape}")

print("📉 Étape 3 : Compression par ACP...")
dimension_cible = 256 
pca = PCA(n_components=dimension_cible)
vecteurs_comprimes = pca.fit_transform(vecteurs_hybrides).astype('float32')

# Sauvegarde impérative de l'ACP pour le script de recherche
with open("pca_model.pkl", "wb") as f_pkl:
    pickle.dump(pca, f_pkl)
print("💾 Modèle 'pca_model.pkl' exporté avec succès !")

print("💾 Étape 4 : Création du nouvel Index FAISS Hybride...")
index_hybride = faiss.IndexFlatL2(dimension_cible)
index_hybride.add(vecteurs_comprimes)

faiss.write_index(index_hybride, "index_hybrid_pca.bin")
print("🎉 Index synchronisé généré avec succès !")