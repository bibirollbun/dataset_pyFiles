# ============================
# Config and Imports
# ============================

import os
import pandas as pd
import torch

class CFG:
    data_dir = "/kaggle/input/birdclef-2025"
    output_file = "submission.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Using device: {CFG.device}")

# ============================
# Pipeline Class
# ============================

class BirdCLEF2025Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.submission_df = None

    def run(self):
        print("[INFO] Running prediction pipeline...")

        test_audio_dir = os.path.join(self.cfg.data_dir, "test_soundscapes")
        test_files = [f for f in os.listdir(test_audio_dir) if f.endswith(".ogg")]
        print(f"[INFO] Found {len(test_files)} test audio files.")

        row_ids = []
        targets = []

        for file in test_files:
            base_name = os.path.splitext(file)[0]
            for i in range(5, 60, 5):  # row_ids in 5-second steps
                row_id = f"{base_name}_{i}"
                row_ids.append(row_id)
                targets.append(0)  # Dummy prediction

        self.submission_df = pd.DataFrame({
            "row_id": row_ids,
            "target": targets
        })

        print("[INFO] Submission DataFrame created.")
        print(self.submission_df.head())

    def save_submission(self):
        self.submission_df.to_csv(self.cfg.output_file, index=False)
        print(f"[INFO] Submission saved to {self.cfg.output_file}")


# ============================
# Run the pipeline
# ============================

if __name__ == "__main__":
    cfg = CFG()
    pipeline = BirdCLEF2025Pipeline(cfg)
    pipeline.run()
    pipeline.save_submission()





