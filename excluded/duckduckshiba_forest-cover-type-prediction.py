import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore") # すべてのWarningsを非表示

# ========================================
# 固有定義
# ========================================
RANDOM_SEED = 71
target_col = "Cover_Type"

# ========================================
# データの読み込み、前処理
# ========================================
train, test = (pd.read_csv(f"/kaggle/input/forest-cover-type-prediction/{name}.csv")
               for name in ["train", "test"])

train.drop("Id", axis=1, inplace=True)
test_ids = test["Id"].copy()
test.drop("Id", axis=1, inplace=True)

# ========================================
# 新規特徴量の追加
# ========================================
def add_new_features(df):
    df["Total_Distance_To_Hydrology"] = np.sqrt(df["Vertical_Distance_To_Hydrology"]**2 + df["Horizontal_Distance_To_Hydrology"]**2)
    df["Elevation_Plus_Vertical_Hydrology"] = df["Elevation"] + df["Vertical_Distance_To_Hydrology"]
    df["Elevation_Minus_Vertical_Hydrology"] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
    df["Hydrology_Plus_Fire_Points"] = df["Horizontal_Distance_To_Hydrology"] + df["Horizontal_Distance_To_Fire_Points"]
    df["Hydrology_Minus_Fire_Points"] = df["Horizontal_Distance_To_Hydrology"] - df["Horizontal_Distance_To_Fire_Points"]
    df["Hydrology_Plus_Roadways"] = df["Horizontal_Distance_To_Hydrology"] + df["Horizontal_Distance_To_Roadways"]
    df["Hydrology_Minus_Roadways"] = df["Horizontal_Distance_To_Hydrology"] - df["Horizontal_Distance_To_Roadways"]
    df["Fire_Points_Plus_Roadways"] = df["Horizontal_Distance_To_Fire_Points"] + df["Horizontal_Distance_To_Roadways"]
    df["Fire_Points_Minus_Roadways"] = df["Horizontal_Distance_To_Fire_Points"] - df["Horizontal_Distance_To_Roadways"]

add_new_features(train)
add_new_features(test)

# ========================================
# 学習・評価・提出処理
# ========================================
X = train.drop(target_col, axis=1)
y = train[target_col] - 1  # 0始まりにする

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=RANDOM_SEED)

model = RandomForestClassifier(
    random_state=RANDOM_SEED
) # モデルの定義
model.fit(X_train, y_train)

y_val_pred = model.predict(X_valid) + 1
print("Validation Accuracy: ", accuracy_score(y_valid + 1, y_val_pred))

test_pred = model.predict(test) + 1
pd.DataFrame({"Id": test_ids, "Cover_Type": test_pred}).to_csv("/kaggle/working/submission.csv", index=False)
print("output submission.csv")

# ========================================
# 特徴量重要度可視化
# ========================================
importances = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
})
importance_df["Contribution(%)"] = 100 * importance_df["Importance"] / importance_df["Importance"].sum()
importance_df = importance_df.sort_values(by="Importance", ascending=False)

print("\nFeature Contribution Rate (Top 30)\n")
print(importance_df.head(30).to_string(index=False))

plt.figure(figsize=(8, 12))
plt.barh(importance_df["Feature"][:40][::-1], importance_df["Importance"][:40][::-1])
plt.title("Feature Importance (Top 40)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


