import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


test_df = pd.read_csv("/kaggle/input/playground-series-s3e24/test.csv")
train_df = pd.read_csv("/kaggle/input/playground-series-s3e24/train.csv")


ans = pd.DataFrame({'id': test_df["id"], 'smoking': np.arange(len(test_df))})
ans.to_csv('submission.csv', index=False)




