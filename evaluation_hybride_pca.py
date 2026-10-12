import numpy as np
import pandas as pd
import faiss
import pickle
from collections import Counter

print("📊 ÉVALUATION GLOBALE DU MOTEUR HYBRIDE PANNs + SSL...")

# 1. Chargement des métadonnées et de la liste officielle
df_meta = pd.read_csv('ESC-50/meta/esc50.csv')
with open('filenames_panns.txt', 'r') as f:
    all_files = [l.strip() for l in f]

def get_category(filename):
    row = df_meta[df_meta['filename'] == filename]
    return row.iloc[0]['category'] if not row.empty else None

# 2. Chargement de l'index FAISS et de l'ACP
index_hybrid_pca = faiss.read_index("index_hybrid_pca.bin")
with open("pca_model.pkl", "rb") as f:
    pca_charge = pickle.load(f)

# 3. Reconstruction de la matrice globale pour l'évaluation
print("🔄 Chargement et fusion des vecteurs pour les 2000 fichiers...")
all_panns = [np.load(f'embeddings_panns/{f.replace(".wav", ".npy")}').flatten() for f in all_files]
all_ssl = [np.load(f'embeddings_ssl/{f.replace(".wav", ".npy")}').flatten() for f in all_files]

X_hybrid = np.hstack([np.array(all_panns), np.array(all_ssl)])
X_compressed = pca_charge.transform(X_hybrid).astype('float32')

# 4. Boucle d'évaluation automatique
succes = 0
total = len(all_files)

print(f"🚀 Analyse géométrique des {total} fichiers en cours...")

for idx in range(total):
    nom_fichier = all_files[idx]
    classe_reelle = get_category(nom_fichier)
    
    # Extraction du vecteur compressé de ce fichier
    vecteur_requete = X_compressed[idx].reshape(1, -1)
    
    # Recherche des voisins (on demande 6 voisins pour exclure le fichier lui-même)
    distances, indices = index_hybrid_pca.search(vecteur_requete, 6)
    
    # Filtrage pour enlever le fichier lui-même du vote
    voisins_categories = []
    for i in indices[0]:
        if all_files[i] != nom_fichier:
            voisins_categories.append(get_category(all_files[i]))
        if len(voisins_categories) == 5: # On garde le Top 5 des "vrais" voisins
            break
            
    # Vote majoritaire
    decision_ia = Counter(voisins_categories).most_common(1)[0][0]
    
    # Vérification
    if decision_ia == classe_reelle:
        succes += 1

# 5. Calcul du pourcentage final
accuracy = (succes / total) * 100

print("\n" + "="*50)
print("🏆 SCORE SCIENTIFIQUE FINAL POUR TON ARTICLE 🏆")
print("="*50)
print(f"🎯 Fichiers correctement classés : {succes} / {total}")
print(f"📈 ACCURACY GLOBALE DE LA MÉTHODE : {accuracy:.2f} %")
print("="*50 + "\n")