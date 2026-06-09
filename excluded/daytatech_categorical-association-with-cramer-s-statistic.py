import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv("/kaggle/input/playground-series-s4e6/train.csv", index_col="id")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(df.drop(columns=["Target"]), df.Target)

X = pd.read_csv("/kaggle/input/playground-series-s4e6/test.csv")
res = rf.predict(X.drop(columns=["id"]))
res = pd.DataFrame(res, index=X.id, columns=["Target"])
res.to_csv("submission.csv")


