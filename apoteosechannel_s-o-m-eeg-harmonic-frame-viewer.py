"""
S.O.M. – EEG: Harmonic Frame Viewer and Symbolic Submission Generator
Late Submission for HMS - Harmful Brain Activity Classification (Kaggle 2024)

Author: Emerson Italo Lima da Silva (Tiberius)

System: S.O.M. – EEG (System of Materarithmetric Operations)

Theoretical Foundation: T-Física – "The Language of Time and the Geometry of Perception"

DOI (T-Física): https://doi.org/10.34740/kaggle/ds/6969238
DOI (Dataset): https://doi.org/10.34740/kaggle/dsv/11455827

License: Apache 2.0 (per competition guidelines)
Data Attribution: CC BY-NC 4.0 – Harvard Medical School / CCEMRC (Kaggle)
"""

# === [0] Import Modules ===
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import spectrogram
import random

# === [1] Define CSV Folder Paths ===
CSV_EEG_FOLDER = Path("/kaggle/input/converted-csv-hms")
TRAIN_CSV_PATH = Path("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")
TEST_CSV_PATH = Path("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
SUBMISSION_OUTPUT_PATH = Path("/kaggle/working/submission.csv")

# === [2] Model Parameters ===
FREQ = 200
N_CHANNELS = 20
LAYER_SPACING = 40
COLOR_SCALE = 'Inferno'

# === [3] Read Metadata & Define ACNS Labels ===
metadata = pd.read_csv(TRAIN_CSV_PATH)
ACNS_VOTE_LABELS = {
    "seizure_vote": "Seizure",
    "lpd_vote": "LPD",
    "gpd_vote": "GPD",
    "lrda_vote": "LRDA",
    "grda_vote": "GRDA",
    "other_vote": "Other"
}
CLASSES = ["seizure", "lpd", "gpd", "grda", "lrda", "other"]

# === [4] Utility Functions ===
def get_label_from_metadata(file_id):
    subset = metadata[metadata["eeg_id"] == file_id]
    if subset.empty:
        return "Normal"
    row = subset.iloc[0]
    votes = {label: row.get(vote, 0) for vote, label in ACNS_VOTE_LABELS.items()}
    max_vote = max(votes.values())
    labels = [label for label, v in votes.items() if v == max_vote and v > 0]
    return labels[0] if labels else "Normal"

def build_3d_layers(data, fs=FREQ, offset=0):
    layers = []
    for idx, signal in enumerate(data):
        if len(signal) < 10:
            continue
        f, t, Sxx = spectrogram(signal, fs=fs)
        Sxx_log = 10 * np.log10(Sxx + 1e-10)
        z_layer = Sxx_log + (idx + offset) * LAYER_SPACING
        layers.append((t, f, z_layer, idx + offset))
    return layers

def plot_interactive_frame(layers, label="Normal"):
    fig = go.Figure()
    for t, f, z, idx in layers:
        fig.add_surface(
            x=t, y=f, z=z,
            colorscale=COLOR_SCALE,
            showscale=False,
            opacity=0.7 / (idx + 1)**0.3,
            name=f"Channel {idx + 1}",
            hoverinfo='name'
        )

    fig.update_layout(
        title={
            "text": f"<b>S.O.M. EEG Harmonic Model</b><br>Harmonic Frame Viewer – <b>{label}</b>",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.96,
            "yanchor": "top",
            "font": dict(size=24, family="Times New Roman")
        },
        scene=dict(
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            zaxis_title="Channels",
            xaxis=dict(title_font=dict(size=16, family="Times New Roman")),
            yaxis=dict(title_font=dict(size=16, family="Times New Roman")),
            zaxis=dict(title_font=dict(size=16, family="Times New Roman"))
        ),
        font=dict(family="Times New Roman", size=14),
        margin=dict(t=120, l=0, r=0, b=0),
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
    )
    fig.show()

# === [5] Viewer Runner ===
def run_viewer():
    eeg_files = sorted([f for f in CSV_EEG_FOLDER.iterdir() if f.suffix == ".csv"])
    if not eeg_files:
        print("[!] No EEG files found.")
        return
    selected_file = random.choice(eeg_files)
    file_id = int(selected_file.stem)
    label = get_label_from_metadata(file_id)
    data = pd.read_csv(selected_file).values.T
    layers = build_3d_layers(data)
    plot_interactive_frame(layers, label=label)

# === [6] Symbolic Submission Simulator ===
def simulate_prediction(file_id):
    probs = np.random.dirichlet([0.2, 0.3, 0.2, 5.0, 0.3, 0.3])
    return np.round(probs, 4)

def generate_submission():
    test_df = pd.read_csv(TEST_CSV_PATH)
    submission = []
    for _, row in test_df.iterrows():
        file_id = row["eeg_id"]
        probs = simulate_prediction(file_id)
        entry = {"id": file_id}
        entry.update(dict(zip(CLASSES, probs)))
        submission.append(entry)
    pd.DataFrame(submission).to_csv(SUBMISSION_OUTPUT_PATH, index=False)
    print(f"[\u2713] submission.csv saved to: {SUBMISSION_OUTPUT_PATH}")

# === [7] Execute ===
run_viewer()
generate_submission()


