＃ 1111
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc


train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')





train_df.head(10)


train_df.describe()


train_df['sunshine'] = train_df['sunshine'].replace(0, train_df['sunshine'].mean())
test_df['sunshine'] = test_df['sunshine'].replace(0, test_df['sunshine'].mean())


# id, day, rainfallを除いた特徴量カラムのリストを取得
feature_columns = train_df.drop(columns=['id','day','rainfall']).columns

# 各特徴量カラムについてループ処理
for column in feature_columns:
    plt.figure(figsize=(7, 5))  # グラフごとに新しい図を作成
    sns.boxplot(y=train_df[column])  # y軸にカラムのデータを指定して箱ひげ図を描画
    plt.title(f'Distribution and Outliers for "{column}"')  # 各グラフにタイトルを設定
    plt.ylabel(column)
    plt.grid(True)
    plt.show()  # グラフを一つずつ表示


# dropped_columns = ['id','day','pressure','maxtemp','temparature','mintemp','dewpoint','humidity','cloud','sunshine','winddirection','windspeed','rainfall']
# dropped_columns = ['id','day','pressure','maxtemp','temparature','mintemp','dewpoint','winddirection','windspeed','rainfall']
dropped_columns = ['id','day','rainfall']
X = train_df.drop(columns = dropped_columns)
y = train_df['rainfall']


corr_matrix = train_df.corr(numeric_only = True)

plt.figure(figsize = (10, 8))
sns.heatmap(data = corr_matrix, annot = True, fmt = ".2f", cmap = "coolwarm", square = True, linewidths = 0.5)
plt.title("Correlation Matrix")
plt.show()


transformer = Pipeline([
    ('impute', SimpleImputer(strategy = "most_frequent")),
    ('scaler', StandardScaler())
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


X_train_transformed = transformer.fit_transform(X_train)
X_test_transformed = transformer.fit_transform(X_test)


# # 基本となるランダムフォレストモデルを定義
# rand_forest = RandomForestClassifier(random_state = 42)

# # 探索するハイパーパラメータの範囲を拡大・詳細化
# param_grid = {
#     'n_estimators': [50, 100, 200, 300],  
#     'max_depth': [10, 20, None],
#     'max_features': ['sqrt', 'log2'],
#     'min_samples_split': [2, 5, 10],   
#     'min_samples_leaf': [1, 2, 4],       
#     'bootstrap': [True, False]            
# }

# grid_rand = GridSearchCV(rand_forest, param_grid, cv=5, verbose=1, n_jobs=-1)
# grid_rand.fit(X_train_transformed, y_train)

# print("Best params after tuning parameters:", grid_rand.best_params_)


print("Best params after tuning parameters: {'bootstrap': True, 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 200}")

rand_forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    max_features='sqrt',
    random_state=42,
    bootstrap=True,
    min_samples_split=10,
    min_samples_leaf=4
)

# モデルの学習
rand_forest.fit(X_train_transformed, y_train)


rand_forest_importances = pd.DataFrame(rand_forest.feature_importances_, index=X.columns, columns=['Importance'])
print("\nランダムフォレストの特徴量の重要度:")
print(rand_forest_importances.sort_values(by='Importance', ascending=False))


# 検証データで予測確率を計算
y_pred_proba = rand_forest.predict_proba(X_test_transformed)[:, 1]

# 確率を0か1のクラスに変換（閾値0.5）
y_pred = (y_pred_proba > 0.5).astype(int)

# 正解率 (Accuracy) の表示
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy (正解率): {accuracy:.4f}\n")

# 分類レポート (Classification Report) の表示
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 混同行列 (Confusion Matrix) の可視化
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Rain', 'Rain'], yticklabels=['No Rain', 'Rain'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC曲線とAUCスコアの可視化
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


test_df_altered = test_df.drop(columns = ['id', 'day'])
test_df_transformed = transformer.fit_transform(test_df_altered)

test_df_altered = test_df.drop(columns=['id', 'day'])
test_df_transformed = transformer.transform(test_df_altered)


probabilities = rand_forest.predict_proba(test_df_transformed)[:, 1]

submission = pd.DataFrame({
    'id': test_df['id'],
    'rainfall': probabilities
})

submission.to_csv('submission.csv', index = False)

