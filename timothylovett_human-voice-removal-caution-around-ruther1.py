import os
import pickle

# Path to the pickle in your Kaggle input (read‑only!)
pkl_path = "/kaggle/input/bc25-separation-voice-from-data/train_voice_data.pkl"

if not os.path.isfile(pkl_path):
    raise FileNotFoundError(f"No such file: {pkl_path}")

# Open in binary‑read mode and load
with open(pkl_path, "rb") as f:
    train_voice_data = pickle.load(f)

print(f"Loaded object of type {type(train_voice_data)}")
try:
    print(f"Contains {len(train_voice_data)} entries")
except Exception:
    pass


from IPython.display import Audio, display

ruther_records = [
    "/kaggle/input/birdclef-2025/train_audio/ruther1/XC512534.ogg",
    "/kaggle/input/birdclef-2025/train_audio/ruther1/XC257300.ogg",
    "/kaggle/input/birdclef-2025/train_audio/ruther1/XC681215.ogg",
    "/kaggle/input/birdclef-2025/train_audio/ruther1/XC257299.ogg",
    "/kaggle/input/birdclef-2025/train_audio/ruther1/XC257303.ogg",
    "/kaggle/input/birdclef-2025/train_audio/ruther1/XC657821.ogg",
    "/kaggle/input/birdclef-2025/train_audio/ruther1/XC178070.ogg",
    "/kaggle/input/birdclef-2025/train_audio/ruther1/XC504229.ogg"
]

for rec_path in ruther_records:
    metadata = train_voice_data.get(rec_path, {})
    print(f"{rec_path}: {metadata}")
    display(Audio(rec_path, autoplay=False))








