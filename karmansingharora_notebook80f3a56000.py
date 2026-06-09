import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("/kaggle/input/thapar-summer-school-2025-hack-ii/train.csv")
df.head()


df = df.drop(columns=["id", "Row#"], errors="ignore")
df.head()


q1, q3 = df["yield"].quantile([0.25, 0.75])
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
df = df[(df["yield"] >= lower) & (df["yield"] <= upper)]


correlations = df.corr(numeric_only=True)["yield"].abs().sort_values(ascending=False)
top_features = correlations[1:11].index.tolist()


X = df[top_features]
y = df["yield"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)


model = LinearRegression()
model.fit(X_pca, y)


y_pred = model.predict(X_pca)


mae = mean_absolute_error(y, y_pred)


mae


r2 = r2_score(y, y_pred)


r2


test_df = pd.read_csv("/kaggle/input/thapar-summer-school-2025-hack-ii/test.csv")
test_ids = test_df["id"]

test_features = test_df[top_features]

test_scaled = scaler.transform(test_features)

test_pca = pca.transform(test_scaled)

test_preds = model.predict(test_pca)

submission = pd.DataFrame({
    "id": test_ids,
    "target": test_preds
})

submission.to_csv("Name_LastName_RollNo.csv", index=False)




