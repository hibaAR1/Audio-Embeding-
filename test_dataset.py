# test_dataset.py
import os
import pandas as pd
import soundfile as sf

ESC50_PATH = "./ESC-50"

# 1. Check CSV
csv_path = os.path.join(ESC50_PATH, "meta", "esc50.csv")
print(f"CSV exists: {os.path.exists(csv_path)}")

meta = pd.read_csv(csv_path)
print(f"Total rows in CSV: {len(meta)}")
print(f"Columns: {list(meta.columns)}")
print(f"Folds: {sorted(meta['fold'].unique())}")
print(f"\nFirst 3 rows:\n{meta.head(3)}")

# 2. Check audio dir
audio_dir = os.path.join(ESC50_PATH, "audio")
print(f"\nAudio dir exists: {os.path.exists(audio_dir)}")
if os.path.exists(audio_dir):
    files = os.listdir(audio_dir)
    print(f"Number of audio files: {len(files)}")
    print(f"First 3 files: {files[:3]}")

# 3. Try loading one file
if os.path.exists(audio_dir) and len(files) > 0:
    first_file = os.path.join(audio_dir, meta['filename'].values[0])
    print(f"\nTrying to load: {first_file}")
    print(f"File exists: {os.path.exists(first_file)}")
    try:
        data, sr = sf.read(first_file, dtype='float32')
        print(f"Loaded OK — shape={data.shape}, sr={sr}")
    except Exception as e:
        print(f"Load FAILED: {e}")

# 4. Test dataset directly
print("\n--- Testing ESC50Dataset ---")
import sys
sys.path.insert(0, ".")
from adaptive_audio_final import ESC50Dataset

ds = ESC50Dataset(ESC50_PATH, fold=1, train=True)
print(f"Dataset length: {len(ds)}")

if len(ds) > 0:
    print("Trying ds[0]...")
    try:
        w, label = ds[0]
        print(f"OK — waveform shape={w.shape}, label={label}")
    except Exception as e:
        import traceback
        traceback.print_exc()
else:
    print("Dataset is EMPTY — check CSV fold values match 1-5")