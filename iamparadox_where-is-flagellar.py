import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2


df = pd.read_csv("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv")


TOMO_ID = "tomo_02862f"


def plot_motors(tomo_id):
    tomo_motors = df[df["tomo_id"] == tomo_id]


    n = len(tomo_motors)

    for row in range(n):
        img = cv2.imread(f"/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train/{tomo_id}/slice_{int(tomo_motors['Motor axis 0'].iloc[row]):04d}.jpg", cv2.IMREAD_GRAYSCALE)
        img = cv2.circle(img, (int(tomo_motors["Motor axis 2"].iloc[row]),int(tomo_motors["Motor axis 1"].iloc[row])), 12, (255,0,0), thickness=5)
        plt.subplot(1, n, row + 1)
        plt.imshow(img)
    plt.tight_layout()
    plt.show()
    


plot_motors("tomo_02862f")


plot_motors("tomo_0de3ee")




