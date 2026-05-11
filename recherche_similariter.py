import pandas as pd
import numpy as np
import faiss
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import random
from collections import Counter

# 1. CHARGEMENT DES RÉFÉRENCES
df_meta = pd.read_csv('ESC-50/meta/esc50.csv')
categories_disponibles = sorted(df_meta['category'].unique())

def get_info(filename):
    row = df_meta[df_meta['filename'] == filename]
    return row.iloc[0]['category'] if not row.empty else "Inconnue"

def run_interactive_session():
    # Nettoyage des fenêtres précédentes
    plt.close('all')

    print("\n" + "="*50)
    print("🎙️  SYSTEME DE CLASSIFICATION ACOUSTIQUE INTERACTIF")
    print("="*50)
    
    choix = input(f"\n👉 Entre une catégorie (ex: dog, rain, sea_waves) : ").strip().lower()
    
    if choix not in categories_disponibles:
        print(f"❌ Erreur : '{choix}' n'existe pas dans ESC-50.")
        return

    fichiers_categorie = df_meta[df_meta['category'] == choix]['filename'].tolist()
    target_file = random.choice(fichiers_categorie)
    
    print(f"\n🎯 FICHIER CIBLE : {target_file}")
    print(f"📂 CLASSE RÉELLE  : {choix.upper()}")
    print("-" * 50)

    models = ['panns', 'ssl']
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Chargement de la liste des noms de fichiers
    try:
        with open('filenames_panns.txt', 'r') as f:
            all_files = [l.strip() for l in f]
    except FileNotFoundError:
        print("❌ Erreur : filenames_panns.txt introuvable. Relance l'indexation.")
        return

    for ax_idx, m_type in enumerate(models):
        # --- A. RECHERCHE ---
        index_path = f'index_{m_type}.bin'
        if not os.path.exists(index_path):
            print(f"❌ Index {index_path} manquant.")
            continue
            
        index = faiss.read_index(index_path)
        query_vec = np.load(f'embeddings_{m_type}/{target_file.replace(".wav", ".npy")}').reshape(1, -1)
        distances, indices = index.search(query_vec.astype('float32'), 6)
        res_names = [all_files[i] for i in indices[0]]

        # --- B. AFFICHAGE TERMINAL ---
        print(f"\n🧠 ANALYSE DU MODÈLE : {m_type.upper()}")
        print(f"{'RANG':<8} | {'CATEGORIE':<15} | {'DISTANCE L2':<12} | {'SIMILARITÉ'}")
        print("-" * 55)

        for i, name in enumerate(res_names):
            cat = get_info(name)
            dist = distances[0][i]
            sim = (1 / (1 + dist)) * 100
            label = "CIBLE" if i == 0 else f"#{i}"
            print(f"{label:<8} | {cat:<15} | {dist:>12.4f} | {sim:>5.1f}%")

        # --- C. EXPLICATION MATHÉMATIQUE ---
        v1_name = res_names[1]
        v1_dist = distances[0][1]
        print(f"\n🔍 Preuve Mathématique ({m_type.upper()}) :")
        print(f"   Formule : d(A,B) = √Σ(ai - bi)²")
        print(f"   Distance entre Cible et Voisin #1 : {v1_dist:.4f}")

        # --- D. VOTE ET DÉDUCTION ---
        voisins_cats = [get_info(res_names[j]) for j in range(1, 6)]
        vote_result = Counter(voisins_cats).most_common(1)[0]
        
        couleur = "✅" if vote_result[0] == choix else "❌"
        print(f"🗳️  Vote (k=5) : {vote_result[1]}/5 pour '{vote_result[0]}'")
        print(f"📌 DÉCISION FINALE : {couleur} {vote_result[0].upper()}")
        print("-" * 50)

        # --- E. GRAPHIQUE PCA ---
        num_sample = min(len(all_files), 150)
        sample_files = random.sample(all_files, num_sample)
        sample_vecs = [np.load(f'embeddings_{m_type}/{f.replace(".wav", ".npy")}').flatten() for f in sample_files]
        
        pca = PCA(n_components=2)
        all_coords = pca.fit_transform(np.array(sample_vecs))
        query_coord = pca.transform(query_vec)
        
        axes[ax_idx].scatter(all_coords[:, 0], all_coords[:, 1], c='lightgray', alpha=0.5, label='Espace sonore')
        axes[ax_idx].scatter(query_coord[0, 0], query_coord[0, 1], c='red', s=250, marker='*', label='Requête', zorder=5)
        
        for i in range(1, 6):
            v_name = res_names[i]
            v_vec = np.load(f'embeddings_{m_type}/{v_name.replace(".wav", ".npy")}').reshape(1, -1)
            v_c = pca.transform(v_vec)
            color = 'green' if get_info(v_name) == choix else 'blue'
            axes[ax_idx].scatter(v_c[0, 0], v_c[0, 1], s=120, edgecolors=color, facecolors='none', linewidth=2)
            axes[ax_idx].text(v_c[0, 0]+0.1, v_c[0, 1]+0.1, f"#{i}", fontsize=9, fontweight='bold')

        axes[ax_idx].set_title(f"Espace Latent : {m_type.upper()}\nPrécision: {vote_result[1]}/5")
        axes[ax_idx].legend()

    plt.suptitle(f"COMPARAISON DE SIMILARITÉ : {choix.upper()}", fontsize=16)
    
    print("\n🖥️  Ouverture du graphique... (Fermez la fenêtre pour terminer)")
    # block=True permet d'éviter le crash KeyboardInterrupt
    plt.show(block=True) 

if __name__ == "__main__":
    try:
        run_interactive_session()
    except Exception as e:
        print(f"\n⚠️  Une erreur est survenue : {e}")