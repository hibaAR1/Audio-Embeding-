import os
import numpy as np
import faiss

# Configuration des deux mondes à indexer
tasks = [
    {'name': 'panns', 'folder': 'embeddings_panns', 'dim': 2048},
    {'name': 'ssl', 'folder': 'embeddings_ssl', 'dim': 768}
]

print("🛠️ DÉBUT DE L'INDEXATION FAISS\n")

for task in tasks:
    folder = task['folder']
    if not os.path.exists(folder):
        print(f"⚠️ Dossier {folder} introuvable, on passe...")
        continue

    # 1. Récupérer la liste des fichiers .npy (triée pour la correspondance)
    npy_files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
    
    if not npy_files:
        print(f"❌ Aucun fichier trouvé dans {folder}")
        continue

    # 2. Charger tous les vecteurs dans une liste
    embeddings_list = []
    for f in npy_files:
        data = np.load(os.path.join(folder, f))
        # On s'assure que le vecteur est "plat" (1 dimension)
        embeddings_list.append(data.flatten())

    # 3. Convertir en matrice NumPy pour FAISS
    matrix = np.array(embeddings_list).astype('float32')

    # 4. Créer l'index FAISS
    # IndexFlatL2 = Recherche exacte par distance euclidienne
    index = faiss.IndexFlatL2(task['dim'])
    index.add(matrix) # On "nourrit" l'index avec nos vecteurs

    # 5. Sauvegarder l'index (le cerveau) et les noms (la légende)
    faiss.write_index(index, f"index_{task['name']}.bin")
    
    with open(f"filenames_{task['name']}.txt", "w") as f:
        for name in npy_files:
            # On remplace .npy par .wav pour que ce soit plus joli plus tard
            f.write(name.replace('.npy', '.wav') + "\n")

    print(f"✅ Index {task['name'].upper()} créé !")
    print(f"   -> {index.ntotal} sons indexés")
    print(f"   -> Dimension : {task['dim']}\n")

print("✨ TERMINÉ ! Tu as maintenant tes moteurs de recherche prêts.")