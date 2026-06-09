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


# ラグ特徴量を作成するカラムのリスト
lag_feature_cols = ['pressure', 'temparature', 'windspeed', 'humidity']

# 訓練データとテストデータの両方に処理を適用
for df in [train_df, test_df]:

    # 【修正点】変換前に欠損値を補完する
    # winddirectionなど、欠損値がある可能性のある列を中央値で埋める
    df.fillna(df.median(), inplace=True)
    
    # 処理前にdayでソートすることが重要
    df.sort_values(by='day', inplace=True)
    
    # 既存の特徴量
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['dewpoint_spread'] = df['temparature'] - df['dewpoint']
    
    # dayとwinddirectionの周期性特徴量
    df['day_sin'] = np.sin(2 * np.pi * df['day']/365)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/365)
    df['wind_sin'] = np.sin(2 * np.pi * df['winddirection']/360)
    df['wind_cos'] = np.cos(2 * np.pi * df['winddirection']/360)
    
    # 交互作用特徴量
    df['pressure_humidity_interaction'] = df['humidity'] / df['pressure']
    
    # 【新規追加】ラグ特徴量（前日からの変化量）
    for col in lag_feature_cols:
        # diff()メソッドで前日の行との差分を計算
        df[f'{col}_change'] = df[col].diff()
    
    # 差分計算で発生した最初の行のNaNを0で埋める
    df.fillna(0, inplace=True)

# 新しく追加された特徴量の一部を確認
print("前日からの変化量を示すラグ特徴量が追加されました:")
display(train_df.head(10))


# dropped_columns = ['id','day','pressure','maxtemp','temparature','mintemp','temp_range','dewpoint','dewpoint_spread','humidity','cloud','sunshine','winddirection','windspeed','rainfall']
# dropped_columns = ['id','day','rainfall','pressure','maxtemp','mintemp','temp_range','winddirection','windspeed','day_cos']
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


print("Best params after tuning parameters: {'bootstrap': True, 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 200}")

rand_forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    max_features='sqrt',
    random_state=42,
    bootstrap=True,
    min_samples_split=5,
    min_samples_leaf=1
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


# 予測結果と元のデータを結合して分析用のデータフレームを作成
results_df = X_test.copy()
results_df['actual_rainfall'] = y_test
results_df['predicted_rainfall'] = y_pred
results_df['prediction_probability'] = y_pred_proba

# 正解した予測と間違えた予測を分離
correct_predictions = results_df[results_df['actual_rainfall'] == results_df['predicted_rainfall']]
incorrect_predictions = results_df[results_df['actual_rainfall'] != results_df['predicted_rainfall']]

# 雨が降ると正しく予測したデータの中から、予測確率が高い上位5件を抽出
top_5_correct = correct_predictions[correct_predictions['actual_rainfall'] == 1].sort_values(by='prediction_probability', ascending=False).head(5)

print("--- 正解したデータの上位5件 (雨と予測した確率が高い順) ---")
display(top_5_correct)

print("\n--- 不正解だったデータすべて ---")
display(incorrect_predictions)


# train_dropped_columnsから'rainfall'を除外して、テストデータ用のリストを作成
test_dropped_columns = [col for col in dropped_columns if col != 'rainfall']

test_df_altered = test_df.drop(columns = test_dropped_columns)
test_df_transformed = transformer.fit_transform(test_df_altered)


probabilities = rand_forest.predict_proba(test_df_transformed)[:, 1]

submission = pd.DataFrame({
    'id': test_df['id'],
    'rainfall': probabilities
})

submission.to_csv('submission.csv', index = False)

