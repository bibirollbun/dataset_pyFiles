import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Load data
train = pd.read_csv("/kaggle/input/exploring-predictive-health-factors/train.csv")
test = pd.read_csv("/kaggle/input/exploring-predictive-health-factors/test.csv")

print('Train set shape:', train.shape)
print('Test set shape:', test.shape)
train.head()


train.info()


# EDA: Basic information

print("Train Data Info:")
print(train.info())
print("\nMissing Values:")
print(train.isnull().sum())


# EDA: Basic statistics
print(train.describe())


# Visualization (Categorical variables)

categorical_cols = train.select_dtypes(include=['object']).columns
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=train[col])
    plt.xticks(rotation=45)
    plt.title(f"Distribution of {col}")
    plt.show()


# Handling missing values

for col in train.columns:
    if train[col].dtype == 'object':  # Categorical variables
        train[col] = train[col].fillna('missing')
        if col in test.columns:
            test[col] = test[col].fillna('missing')
    else:  # Numerical variables
        train[col] = train[col].fillna(train[col].median())
        if col in test.columns:
            test[col] = test[col].fillna(train[col].median())


# List of categorical features for CatBoost

cat_features = list(categorical_cols)
print("Categorical Features:", cat_features)


# Convert NaN to string for CatBoost
def convert_cat_features(df, cat_features):
    for col in cat_features:
        df[col] = df[col].astype(str)
    return df



import catboost
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split


# 特徴量と目的変数
X = train.drop(columns=["PCOS"])  # 目的変数 "target" を除く
y = train["PCOS"]

# 訓練データと検証データに分割
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# カテゴリカル特徴量の自動検出
cat_features = X.select_dtypes(include=['object']).columns.tolist()

# CatBoost モデルの定義
model = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, cat_features=cat_features, verbose=100)

# モデルの学習
model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=100, cat_features=cat_features)

# モデルの確率予測
pred_probs = model.predict_proba(test)[:, 1]  # YES（PCOS）の確率を取得


#　 Tree

model.plot_tree(tree_idx=0)


#特徴量の重要度を取得する
feature_importance = model.get_feature_importance()
feature_names = model.feature_names_  # 特徴量の名前を取得

# 重要度の昇順にソート
sorted_indices = np.argsort(feature_importance)
sorted_importance = feature_importance[sorted_indices]
sorted_feature_names = np.array(feature_names)[sorted_indices]

# 棒グラフとしてプロットする
plt.figure(figsize=(12, 6))
plt.barh(range(len(sorted_importance)), sorted_importance, align='center')

# y軸のラベルを設定
plt.yticks(range(len(sorted_importance)), sorted_feature_names)

plt.xlabel('Importance')
plt.ylabel('Features')
plt.grid()
plt.show()


# Submissionファイルの作成
submission = pd.DataFrame({
    'ID': test['ID'],  # IDのカラム名が異なる場合は適宜修正
    'PCOS': pred_probs  # YESの確率を格納
})

submission.to_csv('submission.csv', index=False)
submission.head()




