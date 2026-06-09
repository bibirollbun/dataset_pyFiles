import pandas as pd
import numpy as np


df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


pred = df_train.Calories.median()

result = pd.DataFrame({
    "id": df_test.id,
    "Calories": pred
})

result.to_csv('submission.csv', index=False)

