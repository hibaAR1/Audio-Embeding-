import pandas as pd
import numpy as np
import faiss
from collections import Counter
import os

# 1. CHARGEMENT
df_meta = pd.read_csv('ESC-50/meta/esc50.csv')
def get_info(filename):
    row = df_meta[df_meta['filename'] == filename]
    return row.iloc[0]['category'] if not row.empty else None

models = ['panns', 'ssl']

print("🧪 DÉBUT DE L'ÉVALUATION SCIENTIFIQUE SUR 2000 FICHIERS\n")

for m_type in models:
    index = faiss.read_index(f'index_{m_type}.bin')
    with open(f'filenames_panns.txt', 'r') as f:
        all_files = [l.strip() for l in f]
    
    correct_top1 = 0
    correct_top5_vote = 0
    total = len(all_files)

    print(f"🔄 Analyse du modèle {m_type.upper()}...")

    for i, filename in enumerate(all_files):
        # Charger le vecteur du fichier actuel
        query_vec = np.load(f'embeddings_{m_type}/{filename.replace(".wav", ".npy")}').reshape(1, -1)
        
        # Chercher les 6 plus proches (le #0 est lui-même, donc on prend les 5 suivants)
        distances, indices = index.search(query_vec.astype('float32'), 6)
        
        classe_reelle = get_info(filename)
        
        # Voisins (on ignore l'indice 0 qui est le fichier lui-même)
        voisins_noms = [all_files[idx] for idx in indices[0][1:]]
        voisins_cats = [get_info(name) for name in voisins_noms]

        # --- CALCUL TOP-1 ---
        if voisins_cats[0] == classe_reelle:
            correct_top1 += 1
            
        # --- CALCUL VOTE MAJORITAIRE (TOP-5) ---
        vote_result = Counter(voisins_cats).most_common(1)[0][0]
        if vote_result == classe_reelle:
            correct_top5_vote += 1

    # AFFICHAGE DES RÉSULTATS SCIENTIFIQUES
    acc_top1 = (correct_top1 / total) * 100
    acc_top5 = (correct_top5_vote / total) * 100
    
    print(f"📊 RÉSULTATS POUR {m_type.upper()} :")
    print(f"   ∟ Précision Top-1 (Le plus proche) : {acc_top1:.2f}%")
    print(f"   ∟ Précision Top-5 (Vote majorité)  : {acc_top5:.2f}%")
    print("-" * 40)