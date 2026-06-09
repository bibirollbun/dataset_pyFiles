import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", index_col = "id")

for col in df.columns:
    if df[col].dtype == "O":
        df[col] = df[col].astype("category")

# Label encoding
le = LabelEncoder()
le.fit(["unknown", "primary", "secondary", "tertiary"])
df['education'] = le.fit_transform(df['education'])
le = LabelEncoder()
le.fit(["jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
df['month'] = le.fit_transform(df['month'])

X = pd.get_dummies(df.iloc[:,:-1])
y = df.y == 1
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size = .3, random_state = 0)

# Training the random forest
rf = RFC(max_depth=20, max_features="sqrt", n_estimators=50 , criterion="gini", bootstrap = True)
rf.fit(X_train, y_train)

# Plotting feature importance
#fig, ax = plt.subplots(figsize=(8,8))
#imp_ind = np.argsort(rf.feature_importances_)[::-1]
#ax.bar( rf.feature_names_in_[imp_ind] , rf.feature_importances_[imp_ind])
#plt.xticks(rotation=90)
#plt.tight_layout()

#Sumbission
rf = RFC(max_depth=20, max_features="sqrt", n_estimators=50 , criterion="gini", bootstrap = True)
rf.fit(X, y)

df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv", index_col = "id")
for col in df.columns:
    if df[col].dtype == "O":
        df[col] = df[col].astype("category")
le = LabelEncoder()
le.fit(["unknown", "primary", "secondary", "tertiary"])
df['education'] = le.fit_transform(df['education'])
le = LabelEncoder()
le.fit(["jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
df['month'] = le.fit_transform(df['month'])
X = pd.get_dummies(df)



results = rf.predict_proba(X)
results = pd.DataFrame(results[:,1], index=df.index, columns=["y"])
results.to_csv("submission.csv")
results.head(5)

