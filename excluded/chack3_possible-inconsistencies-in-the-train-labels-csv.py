import pandas as pd

df = pd.read_csv("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv")
for tomo_id, gdf in df.groupby("tomo_id"):
    if gdf.iloc[0]["Number of motors"] == 0:
        continue
    if gdf.iloc[0]["Number of motors"] != len(gdf):
        print(gdf.to_string(index=False))
        print()




