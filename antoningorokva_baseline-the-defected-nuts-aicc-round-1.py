import random
import numpy as np
import torch

seed = 2025

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

train_dir = Path("/kaggle/input/the-defected-nuts-aicc-round-1-2/data/train")
train_paths = sorted(train_dir.glob("*.png"), key=lambda x: int(x.stem))
train_images = np.array([np.array(Image.open(p)) for p in train_paths])

plt.imshow(plt.imread(str(train_paths[0])))
plt.title("One nice hazelnut", fontsize=16, fontweight="bold")
plt.axis("off")
plt.tight_layout()
plt.show()


# Compute mean training image
mean_image = train_images.mean(axis=0)

# Load test images
test_dir = Path("/kaggle/input/the-defected-nuts-aicc-round-1-2/data/test")
test_paths = sorted(test_dir.glob("*.png"), key=lambda x: int(x.stem))
test_images = [np.array(Image.open(p)) for p in test_paths]

print(f"Processing {len(test_images)} test images...")

# Generate anomaly maps
anomaly_maps = []

for i, test_img in enumerate(test_images):
    # Compute absolute difference from mean
    diff = np.abs(test_img.astype(float) - mean_image)

    # Average across channels
    anomaly_score = diff.mean(axis=-1)

    # Normalize to [0, 255]
    anomaly_score = (anomaly_score / anomaly_score.max() * 255).astype(np.uint8)

    anomaly_maps.append(anomaly_score)

    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(test_images)}")

anomaly_maps = np.array(anomaly_maps)


import numpy as np
import base64
import pandas as pd
from tqdm.auto import tqdm

assert anomaly_maps.shape == (70, 1024, 1024), f"Wrong shape: {anomaly_maps.shape}"
assert anomaly_maps.dtype == np.uint8, f"Wrong dtype: {anomaly_maps.dtype}"

df = pd.DataFrame(columns=["data"])

for i, anomaly_map in enumerate(tqdm(anomaly_maps)):
    df.loc[i] = base64.b85encode(anomaly_map.tobytes()).decode("ascii")

df = df.reset_index()
df.columns = ["id", "data"]
df.to_csv("/kaggle/working/submission.csv", index=False)

print("✓ submission.csv written")

