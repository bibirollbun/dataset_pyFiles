import pandas as pd

df = pd.read_parquet("hf://datasets/macabdul9/AIME2025/data/test-00000-of-00001.parquet")

df = df.drop(columns=["split"])

df = df[["id", "problem", "answer"]]

df.to_csv("aime_2025.csv", index=False)

df




