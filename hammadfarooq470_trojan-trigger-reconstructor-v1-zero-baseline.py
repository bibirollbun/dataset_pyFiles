import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


# Directory setup
BASE = "/kaggle/input/trojan-horse-hunt-in-space"
POISONED_DIR = os.path.join(BASE, "poisoned_models")
SUBMISSION_CSV = "submission.csv"
NUM_MODELS = 45
NUM_SAMPLES = 75


# Utility: generate zero trigger
def zero_trigger():
    return np.zeros((3, NUM_SAMPLES), dtype=np.float32)
# Flatten 3×75 → 225
def flatten_trigger(trigger):
    return trigger.reshape(-1)
submission = []


# Flatten 3×75 → 225
def flatten_trigger(trigger):
    return trigger.reshape(-1)
submission = []


# Iterate over each subfolder
for model_id in tqdm(range(1, NUM_MODELS + 1), desc="Reconstructing Triggers"):
    subdir = os.path.join(POISONED_DIR, f"poisoned_model_{model_id}")
    pt_path = os.path.join(subdir, "poisoned_model.pt")
    ckpt_path = pt_path + ".ckpt"

    # Prefer .pt if exists, else .pt.ckpt
    model_path = pt_path if os.path.exists(pt_path) else ckpt_path
    if not os.path.exists(model_path):
        print(f"⚠️ Skipping missing model folder for ID {model_id}")
        continue

    # Zero-trigger baseline
    trigger = zero_trigger()
    flat = flatten_trigger(trigger)
    submission.append([model_id] + flat.tolist())


# Prepare submission DataFrame
cols = ["model_id"] + [
    f"channel_{ch}_{i+1}"
    for ch in (44, 45, 46)
    for i in range(NUM_SAMPLES)
]
df = pd.DataFrame(submission, columns=cols)


# Save CSV
df.to_csv(SUBMISSION_CSV, index=False)
print(f"✅ Saved submission to {SUBMISSION_CSV}")


df.head(10)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load clean data
df = pd.read_csv('/kaggle/input/trojan-horse-hunt-in-space/clean_train_data.csv')
df[['channel_44','channel_45','channel_46']].describe().T


df.head(10)


# Time series overview
df[['channel_44','channel_45','channel_46']].plot(alpha=0.7, figsize=(14, 6))
plt.title('Telemetry Channels (44,45,46) — Clean Training Data')
plt.xlabel('Time Index'); plt.show()


# Correlation structure
sns.heatmap(df[['channel_44','channel_45','channel_46']].corr(), annot=True, cmap='coolwarm')
plt.title('Channel Correlation Matrix'); plt.show()


# Distribution of values
df[['channel_44','channel_45','channel_46']].hist(bins=50, figsize=(12,4))
plt.suptitle('Value Distributions per Channel'); plt.show()

