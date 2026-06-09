import pandas as pd

# Skorları ve dosya yollarını tanımlayın
scores = {
    "cubmison": 0.05387,
    "geforcenowdankasyooorkasyorr": 0.05332,
    "submission": 0.05200,
    "tupmison": 0.05034,
    "zubmison": 0.05040,
}

files = {
    "cubmison": "/kaggle/input/essemble/cubmison.csv",
    "geforcenowdankasyooorkasyorr": "/kaggle/input/essemble/geforcenowdankasyooorkasyorr.csv",
    "submission": "/kaggle/input/essemble/submission.csv",
    "tupmison": "/kaggle/input/essemble/tupmison.csv",
    "zubmison": "/kaggle/input/essemble/zubmison.csv",
}

# Dosyaları yükleyin
dataframes = {name: pd.read_csv(path) for name, path in files.items()}

# Tahminleri birleştirin
merged_df = dataframes["cubmison"].copy()
merged_df.rename(columns={"num_sold": "cubmison"}, inplace=True)

for name, df in dataframes.items():
    if name != "cubmison":
        merged_df = merged_df.merge(df.rename(columns={"num_sold": name}), on="id")

# Skorlara göre ağırlıkları hesaplayın
inverse_scores = {key: 1 / score for key, score in scores.items()}
total_inverse_score = sum(inverse_scores.values())
weights = {key: value / total_inverse_score for key, value in inverse_scores.items()}

# Ağırlıklı ortalama tahminleri hesaplayın
merged_df["num_sold"] = sum(merged_df[col] * weights[col] for col in weights.keys())

# Sonuçları sadece id ve num_sold kolonları ile hazırlayın
final_predictions = merged_df[["id", "num_sold"]]

# Tahminleri CSV dosyasına kaydedin
final_predictions.to_csv("submission.csv", index=False)


