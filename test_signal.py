import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. Charge un fichier audio de ton dataset (remplace par un vrai chemin de ton ESC-50)
audio_path = "ESC-50/audio/1-137-A-32.wav"
y, sr = librosa.load(audio_path, sr=None)

# 2. Configuration des fenêtres (Standard de 25 ms pour Wav2Vec)
taille_fenetre = int(0.025 * sr)  # 25 ms en nombre d'échantillons
pas_de_temps = int(0.010 * sr)    # On avance de 10 ms à chaque trame

print(f"🎵 Fréquence d'échantillonnage : {sr} Hz")
print(f"✂️ Taille d'une trame (25 ms) : {taille_fenetre} échantillons")

# 3. Affichage visuel
plt.figure(figsize=(12, 8))

# --- Graphique 1 : Zoom sur UNE seule trame (25 ms) ---
plt.subplot(2, 1, 1)
debut_trame = pas_de_temps * 50 # On prend la 50ème trame au pif
une_trame = y[debut_trame : debut_trame + taille_fenetre]
plt.plot(une_trame, color='red')
plt.title("🔬 VUE LOCALE : Une seule trame physique de 25 ms (Ce que Wav2Vec analyse)")
plt.xlabel("Échantillons")
plt.ylabel("Amplitude")

# --- Graphique 2 : Le Spectrogramme complet (Toutes les trames collées) ---
plt.subplot(2, 1, 2)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=taille_fenetre, hop_length=pas_de_temps)
S_db = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_db, sr=sr, hop_length=pas_de_temps, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("🗺️ VUE GLOBALE : Le Spectrogramme (Toutes les trames de 25 ms mises bout à bout)")

plt.tight_layout()
plt.show()