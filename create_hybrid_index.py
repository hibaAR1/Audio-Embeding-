import faiss
import numpy as np
from sklearn.decomposition import PCA

print("🔍 Étape 1 : Lecture des index existants...")
# 1. On charge tes index FAISS actuels
index_panns = faiss.read_index("index_panns.bin")
index_ssl = faiss.read_index("index_ssl.bin")

# 2. Extraction des infos de base
num_vectors = index_panns.ntotal
dim_panns = index_panns.d
dim_ssl = index_ssl.d

print(f"📊 Fichiers trouvés : {num_vectors} audios.")
print(f"🧬 Dimension PANNs : {dim_panns} | Dimension SSL : {dim_ssl}")

print("🧱 Extraction des vecteurs depuis la mémoire de FAISS (Méthode robuste)...")
# On crée des tableaux vides pour recevoir les données
vecteurs_panns = np.zeros((num_vectors, dim_panns), dtype='float32')
vecteurs_ssl = np.zeros((num_vectors, dim_ssl), dtype='float32')

# On extrait les vecteurs un par un pour éviter le bug Windows
for i in range(num_vectors):
    vecteurs_panns[i] = index_panns.reconstruct(i)
    vecteurs_ssl[i] = index_ssl.reconstruct(i)

print("🤝 Étape 2 : Fusion (Concaténation) des deux mondes...")
# On colle le vecteur sémantique et le vecteur acoustique côte à côte
vecteurs_hybrides = np.hstack((vecteurs_panns, vecteurs_ssl))
print(f"🚀 Nouveau super-vecteur créé ! Dimension : {vecteurs_hybrides.shape[1]} (2816 attendu)")

print("📉 Étape 3 : Compression secrète par ACP (PCA)...")
# On choisit de compresser à 256 dimensions (ton paramètre secret)
dimension_cible = 256 
pca = PCA(n_components=dimension_cible)
vecteurs_comprimes = pca.fit_transform(vecteurs_hybrides).astype('float32')
print(f"🔒 Vecteurs compressés et chiffrés en {dimension_cible} dimensions !")

print("💾 Étape 4 : Création du nouvel Index FAISS Hybride...")
# Création de l'index FAISS avec la nouvelle dimension réduite
index_hybride = faiss.IndexFlatL2(dimension_cible)
index_hybride.add(vecteurs_comprimes)

# Sauvegarde du fichier final
faiss.write_index(index_hybride, "index_hybrid_pca.bin")
print("🎉 Terminé ! Le fichier 'index_hybrid_pca.bin' a été généré avec succès.")