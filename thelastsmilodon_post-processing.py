# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


sub = pd.read_csv('/kaggle/input/fork-of-final-e2c8ce/submission.csv')
sub


import pandas as pd
from typing import Union, Dict

def keep_topk_per_class(df: pd.DataFrame, k: Union[int, Dict[int, int]]) -> pd.DataFrame:
    """
    Keep only the top-k highest-confidence boxes *per class* for each image.
    - df must have columns: ['image_id', 'prediction_string']
    - prediction_string format: 'cls conf xc yc w h ...' (normalized coords)
    - k can be an int (same K for all classes) or a dict {class_id: K_for_that_class}
    - Rows with 'no boxes' remain unchanged.
    """
    def _per_image_topk(pred_str: str) -> str:
        if not isinstance(pred_str, str) or pred_str.strip().lower() == "no boxes":
            return "no boxes"

        toks = pred_str.strip().split()
        n = len(toks) // 6
        if n == 0:
            return "no boxes"

        # group by class -> list of (conf, cls, xc, yc, w, h)
        groups = {}
        for i in range(n):
            seg = toks[i*6:(i+1)*6]
            if len(seg) != 6:
                continue
            try:
                cls  = int(float(seg[0]))
                conf = float(seg[1])
                xc   = float(seg[2]); yc = float(seg[3])
                w    = float(seg[4]); h  = float(seg[5])
            except Exception:
                continue
            groups.setdefault(cls, []).append((conf, cls, xc, yc, w, h))

        kept = []
        for cls, items in groups.items():
            items.sort(key=lambda x: x[0], reverse=True)  # by confidence
            k_cls = k.get(cls, len(items)) if isinstance(k, dict) else int(k)
            kept.extend(items[:max(0, k_cls)])

        if not kept:
            return "no boxes"

        # (optional) sort final by confidence desc for readability
        kept.sort(key=lambda x: x[0], reverse=True)
        return " ".join(f"{cls} {conf:.6f} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                        for conf, cls, xc, yc, w, h in kept)

    out = df.copy()
    out["prediction_string"] = out["prediction_string"].apply(_per_image_topk)
    return out

# ---- usage ----
sub = pd.read_csv('/kaggle/input/final-inference/submission_2_1024.csv')

# Same K for every class:
sub_topk = keep_topk_per_class(sub, k=2)


sub_topk.to_csv('/kaggle/working/submission.csv', index=False)
print("saved -> /kaggle/working/submission.csv")



# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path
# from PIL import Image
# from matplotlib.patches import Rectangle

# def plot_submission_predictions(
#     submission_csv: str,
#     test_images_path: str,
#     K: int = 5,
#     min_conf: float = 0.0
# ):
#     """
#     Plot the first K images from submission.csv with their predicted boxes,
#     but only show boxes with confidence >= min_conf.
    
#     submission_csv:   path to your final submission.csv
#     test_images_path: folder containing the test images
#     K:                number of images to visualize
#     min_conf:         minimum confidence threshold for drawing a box
#     """
#     df = pd.read_csv(submission_csv)
#     img_folder = Path(test_images_path)
    
#     for _, row in df.head(K).iterrows():
#         image_id = row["image_id"]
#         pred_str = row["prediction_string"]
        
#         # locate the image file
#         matches = list(img_folder.glob(f"{image_id}.*"))
#         if not matches:
#             print(f"⚠️  Could not find file for {image_id}")
#             continue
#         img = Image.open(matches[0])
        
#         fig, ax = plt.subplots(figsize=(8, 6))
#         ax.imshow(img)
#         ax.axis("off")
        
#         if pred_str.lower() != "no boxes":
#             toks = pred_str.split()
#             for i in range(0, len(toks), 6):
#                 lbl   = int(toks[i])
#                 score = float(toks[i+1])
#                 if score < min_conf:
#                     continue
#                 x_c   = float(toks[i+2])
#                 y_c   = float(toks[i+3])
#                 w     = float(toks[i+4])
#                 h     = float(toks[i+5])
                
#                 # convert normalized center w,h to absolute top-left corner + size
#                 x1    = (x_c - w/2) * img.width
#                 y1    = (y_c - h/2) * img.height
#                 abs_w = w * img.width
#                 abs_h = h * img.height
                
#                 rect = Rectangle((x1, y1), abs_w, abs_h,
#                                  fill=False, edgecolor="red", lw=2)
#                 ax.add_patch(rect)
#                 ax.text(
#                     x1, y1 - 3,
#                     f"{lbl}:{score:.2f}",
#                     color="yellow", fontsize=10,
#                     backgroundcolor="black", alpha=0.7
#                 )
        
#         plt.show()
# plot_submission_predictions(
#     submission_csv="/kaggle/working/submission.csv",
#     test_images_path= "/kaggle/input/multi-class-object-detection-challenge/testImages/images",
#     K=280,
#     min_conf=0.005   # only show boxes with confidence ≥ min_conf
# )





