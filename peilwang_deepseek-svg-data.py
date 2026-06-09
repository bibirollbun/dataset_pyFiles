import pandas as pd

splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
train = pd.read_parquet("hf://datasets/thesantatitan/deepseek-svg-dataset/" + splits["train"])
test = pd.read_parquet("hf://datasets/thesantatitan/deepseek-svg-dataset/" + splits["test"])


train


test


train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)




