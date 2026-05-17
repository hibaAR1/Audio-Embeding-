import pandas as pd
import numpy as np
import faiss
from collections import Counter

# 1. CHARGEMENT DES MÉTADONNÉES (Identique à ton code)
df_meta = pd.read_csv('ESC-50/meta/esc50.csv')
def get_info(filename):
    row = df_meta[df_meta['filename'] == filename]
    return row.iloc[0]['category'] if not row.empty else None

print("🧪 DÉBUT DE L'ÉVALUATION SCIENTIFIQUE SUR L'INDEX HYBRIDE (Fusion + ACP)\n")

# 2. CHARGEMENT DES FICHIERS ET DE TON SUPER INDEX
with open('filenames_panns.txt', 'r') as f:
    all_files = [l.strip() for l in f]

# On charge ton index hybride tout neuf à 256 dimensions
index_hybride = faiss.read_index("index_hybrid_pca.bin")
total = len(all_files)
num_vectors = index_hybride.ntotal

# 3. EXTRACTION DES VECTEURS COMPRIMÉS DEPUIS L'INDEX
# Vu que les .npy n'existent pas sur le disque, on reconstruit la matrice directement depuis FAISS
print("🧱 Extraction des requêtes hybrides depuis la mémoire de FAISS...")
vecteurs_requetes = np.zeros((num_vectors, index_hybride.d), dtype='float32')
for i in range(num_vectors):
    vecteurs_requetes[i] = index_hybride.reconstruct(i)

correct_top1 = 0
correct_top5_vote = 0

print("🔄 Analyse des performances du modèle HYBRIDE (256D)...")

# 4. RECHERCHE ET ÉVALUATION (La logique exacte de ton code d'origine)
for i, filename in enumerate(all_files):
    # On extrait le vecteur hybride compressé de l'audio actuel
    query_vec = vecteurs_requetes[i].reshape(1, -1)
    
    # Chercher les 6 plus proches (le #0 est lui-même, donc on prend les 5 suivants)
    distances, indices = index_hybride.search(query_vec, 6)
    
    classe_reelle = get_info(filename)
    
    # Voisins (on ignore l'indice 0 qui est le fichier lui-même)
    voisins_noms = [all_files[idx] for idx in indices[0][1:]]
    voisins_cats = [get_info(name) for name in voisins_noms]

    # --- CALCUL TOP-1 (Le plus proche est-il de la même catégorie ?) ---
    if voisins_cats[0] == classe_reelle:
        correct_top1 += 1
        
    # --- CALCUL VOTE MAJORITAIRE (TOP-5) ---
    vote_result = Counter(voisins_cats).most_common(1)[0][0]
    if vote_result == classe_reelle:
        correct_top5_vote += 1

# 5. AFFICHAGE DES RÉSULTATS SCIENTIFIQUES
acc_top1 = (correct_top1 / total) * 100
acc_top5 = (correct_top5_vote / total) * 100

print("\n" + "="*40)
print("📊 RÉSULTATS POUR L'INDEX HYBRIDE ACP :")
print(f"   ∟ Précision Top-1 (Le plus proche) : {acc_top1:.2f}%")
print(f"   ∟ Précision Top-5 (Vote majorité)  : {acc_top5:.2f}%")
print("="*40)
print("🎉 Évaluation terminée ! Tu as tes nouveaux scores comparatifs.")