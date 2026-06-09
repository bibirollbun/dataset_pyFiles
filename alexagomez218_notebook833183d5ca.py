!pip install py7zr
import py7zr
import os

if not os.path.exists('/kaggle/train/') :
    os.makedirs('/kaggle/train/')

if not os.path.exists('/kaggle/test/') :
    os.makedirs('/kaggle/test/')

with py7zr.SevenZipFile("/kaggle/input/statoil-iceberg-classifier-challenge/train.json.7z", 'r') as archive:
    archive.extractall(path="/kaggle/train")

with py7zr.SevenZipFile("/kaggle/input/statoil-iceberg-classifier-challenge/test.json.7z", 'r') as archive:
    archive.extractall(path="/kaggle/test")

for dirname, _, filenames in os.walk('/kaggle'): 
    for filename in filenames: 
        print(os.path.join(dirname, filename))



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load the dataset
# -----------------------------
df = pd.read_json('/kaggle/train/data/processed/train.json')

# -----------------------------
# 2. Find a ship and an iceberg example
# -----------------------------
ship_idx = df[df["is_iceberg"] == 0].index[0]
ice_idx  = df[df["is_iceberg"] == 1].index[0]

print("Ship index:", ship_idx)
print("Iceberg index:", ice_idx)

# -----------------------------
# 3. Function to plot HH + HV for a given index
# -----------------------------
def plot_hh_hv(idx, title):
    hh = np.array(df["band_1"][idx]).reshape(75, 75)
    hv = np.array(df["band_2"][idx]).reshape(75, 75)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    fig.suptitle(title, fontsize=14)

    axs[0].imshow(hh, cmap="gray")
    axs[0].set_title("HH polarization")
    axs[0].axis("off")

    axs[1].imshow(hv, cmap="gray")
    axs[1].set_title("HV polarization")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()

# -----------------------------
# 4. Plot both examples
# -----------------------------
plot_hh_hv(ship_idx, "Ship Example")
plot_hh_hv(ice_idx, "Iceberg Example")



# -----------------------------
# 5. False-color RGB composite generator
# -----------------------------
def false_color(idx):
    hh = np.array(df["band_1"][idx]).reshape(75, 75)
    hv = np.array(df["band_2"][idx]).reshape(75, 75)

    # normalize channels to 0-1
    hh_n = (hh - hh.min()) / (hh.max() - hh.min())
    hv_n = (hv - hv.min()) / (hv.max() - hv.min())

    # Build RGB image:
    # R = HH (strong surface backscatter)
    # G = HV (volume/depolarized scattering)
    # B = HH (low weight to add contrast)
    rgb = np.dstack([hh_n, hv_n, hh_n * 0.3])

    return rgb

# -----------------------------
# 6. Plot both false color composites
# -----------------------------
def plot_composite(idx, title):
    rgb = false_color(idx)

    plt.figure(figsize=(4, 4))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()

plot_composite(ship_idx, "Ship — False Color Composite")
plot_composite(ice_idx, "Iceberg — False Color Composite")

