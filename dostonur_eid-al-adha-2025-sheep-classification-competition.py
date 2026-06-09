import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


dataset_folder = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images"
train_dir = os.path.join(dataset_folder, "train")
label_file = os.path.join(dataset_folder, "train_labels.csv")


df = pd.read_csv(label_file)
print("Sample:\n", df.head())


classes = df["label"].unique()
for c in classes:
    img_name = df[df["label"] == c].iloc[0]["filename"]
    img_path = os.path.join(train_dir, img_name)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(f"{c}")
    plt.axis('off')
    plt.show()


hsv_means = {label: [] for label in classes}


# Looping image and calculating mean HSV
for _, row in df.iterrows():
    img_path = os.path.join(train_dir, row['filename'])
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean = hsv.mean(axis=(0, 1))  # mean H, S, V
    hsv_means[row['label']].append(mean)


# Computing average HSV per class
hsv_avg_per_class = {}
for label in classes:
    values = np.array(hsv_means[label])
    avg = values.mean(axis=0)
    hsv_avg_per_class[label] = avg


# Showing results
print("\nðŸ“Š Average HSV values per class:")
for label, avg in hsv_avg_per_class.items():
    h, s, v = avg
    print(f"{label}: H={h:.1f}, S={s:.1f}, V={v:.1f}")


def classify_by_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv.mean(axis=(0, 1))  # get average values

    if v < 120:
        return "Najdi"
    elif v > 160 and s > 90:
        return "Sawakni"
    elif h > 38 and v < 150:
        return "Barbari"
    elif h > 36:
        return "Goat"
    elif h < 27 and v > 160:
        return "Harri"
    elif h < 27 and s > 85:
        return "Naeimi"
    else:
        return "Roman"


test_dir = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test"


test_files = sorted(os.listdir(test_dir))


# Storing predictions
predictions = []

for fname in test_files:
    img_path = os.path.join(test_dir, fname)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  # normalize size
    label = classify_by_hsv(img)
    predictions.append({"filename": fname, "label": label})


# Saving predictions to CSV
submission_df = pd.DataFrame(predictions)
submission_df.to_csv("submission.csv", index=False)


# Previewing first few rows
print("âœ… submission.csv is ready! Preview:")
print(submission_df.head())

