import os
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np


# edit with your dataset path
root_path = "/kaggle/input/face-matching-aicc-round-2/face-matching"


ref_df = pd.read_csv(f"{root_path}/ref_img.csv", dtype={'ref_img': str})
ref_ids = ref_df["ref_img"].tolist()

img_dir = f"{root_path}/images"
all_images = os.listdir(img_dir)


features = {}
for img_path in tqdm(all_images):
    img_id = img_path[:-len(".jpg")]
    img_path = f"{img_dir}/{img_path}"

    img = cv2.imread(img_path)
    mean_color = img.reshape(-1, 3).mean(axis=0)
    features[img_id] = mean_color


results = []

for ref_id in ref_ids:
    ref_feature = features[ref_id]

    # compute distances to all images
    distances = {}
    for img_id, feature in features.items():
        dist = np.linalg.norm(ref_feature - feature)
        distances[img_id] = dist

    # sort by distance, exclude reference, take top 5
    sorted_ids = sorted(distances.items(), key=lambda x: x[1])
    top_5 = [img_id for img_id, _ in sorted_ids if img_id != ref_id][:5]

    results.append({"ref_img": ref_id, "photos": "|".join(top_5)})


submission = pd.DataFrame(results)
submission


submission.to_csv("submission.csv", index=False)

