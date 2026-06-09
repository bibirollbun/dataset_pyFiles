import pandas as pd

INPUT_CSV = "/kaggle/input/grand-xray-slam-division-a/train1.csv"
OUTPUT_CSV = "/kaggle/working/Filtered"


df = pd.read_csv(INPUT_CSV)
df.head(5)


#All diseases in the dataset
ALL_DISEASE_COLUMNS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Lung Opacity",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
    "No Finding"
]


TARGET_CLASSES = [
    "Pneumothorax", #Hardest to detect
    "Cardiomegaly", #Easy to detect
    "Lung Opacity", #Most Common in Real World
    "Pleural Effusion", #Very Common in Real World
    "Support Devices" #Easy to Detect
]


META_DATA = [
    "Image_name",
    "Sex",
    "ViewCategory",
    "ViewPosition",
]


#FILTER: Frontal Only
df = df[
    (df["ViewCategory"] == "Frontal") &
    (df["ViewPosition"].isin(["AP", "PA"]))
].copy()

#FILTER: Target Diseases
df["target_positive_count"] = df[TARGET_CLASSES].sum(axis=1)
df = df[df["target_positive_count"] == 1].copy()

NON_TARGET_CLASSES = list(set(ALL_DISEASE_COLUMNS) - set(TARGET_CLASSES))
df["non_target_positive_count"] = df[NON_TARGET_CLASSES].sum(axis=1)
df = df[df["non_target_positive_count"] == 0].copy()

#Assigning Class Lables
df["label"] = df[TARGET_CLASSES].idxmax(axis=1)

#Final Cleanup
df = df[META_DATA + TARGET_CLASSES]
df.head(5)


#Images Per Class
df["label"] = df[TARGET_CLASSES].idxmax(axis=1)
df["label"].value_counts()


print("Total Images = ",len(df))
df.to_csv(OUTPUT_CSV, index=False)
print("The file is saved!!")

