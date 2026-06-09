# 必要なライブラリをインポート（よく使うものだけに絞りました）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix


# データの読み込み
train = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/train.csv')
test = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/test.csv')

# 念のため形状を確認（データサイズを把握するのが大事）
print('train shape:', train.shape)
print('test shape:', test.shape)


# 直感的に重要そうな特徴を追加：家族人数や名前からの敬称など
for df in [train, test]:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)

# Titleのユニークな値を見てみる（意味があるか判断するため）
print(train['Title'].value_counts())


# 不要な列を削除し、特徴量と目的変数を定義
X = train.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y = train['Survived']
X_test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)


# データ型を使って分類（少し冗長でも自分で分けた方が安心）
cat_cols = [col for col in X.columns if X[col].dtype == 'object']
cat_cols += ['Pclass', 'Sex', 'Embarked', 'Title']  # 明示的に追加
cat_cols = list(set(cat_cols))
num_cols = [col for col in X.columns if col not in cat_cols]


# 前処理：欠損補完とエンコード
num_transformer = SimpleImputer(strategy='median')
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])


# 個人的に気になった3つのモデルだけ比較
models = {
    'RandomForest': RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=5, random_state=42),
    'LightGBM': LGBMClassifier(max_depth=7, random_state=42)
}


# モデル比較（精度と視覚化）
model_scores = {}
for name, model in models.items():
    pipe = Pipeline([('prep', preprocessor), ('model', model)])
    score = cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()
    model_scores[name] = score
    print(f'{name}: {score:.4f}')

# 視覚化
plt.figure(figsize=(8, 4))
plt.bar(model_scores.keys(), model_scores.values(), color='salmon')
plt.title('モデル比較（精度）')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()


# 一番良かったモデルで予測
best_model_name = max(model_scores, key=model_scores.get)
best_model = models[best_model_name]

final_pipeline = Pipeline([('prep', preprocessor), ('model', best_model)])
final_pipeline.fit(X, y)
pred = final_pipeline.predict(X_test)


# 混同行列で学習データの精度確認（過学習の有無もここで見る）
y_pred_train = final_pipeline.predict(X)
cm = confusion_matrix(y, y_pred_train)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# 最後に提出用ファイルを出力
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': pred
})
submission.to_csv('/kaggle/working/my_custom_submit.csv', index=False)
print('提出ファイルを書き出しました')

