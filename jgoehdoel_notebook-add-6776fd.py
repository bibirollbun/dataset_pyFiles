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


# 訓練データとテストデータの両方に新しい特徴量を追加
for df in [train_df, test_df]:
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['dewpoint_spread'] = df['temparature'] - df['dewpoint']

# 月単位の情報を特徴量として追加
# 1年を約30日ごとの12ヶ月に分割
# train_df['month'] = (train_df['day'] - 1) // 30 + 1
# test_df['month'] = (test_df['day'] - 1) // 30 + 1

# 'month'をSin/Cosに変換 (12ヶ月周期)
# train_df['month_sin'] = np.sin(2 * np.pi * train_df['month']/12)
# train_df['month_cos'] = np.cos(2 * np.pi * train_df['month']/12)
# test_df['month_sin'] = np.sin(2 * np.pi * test_df['month']/12)
# test_df['month_cos'] = np.cos(2 * np.pi * test_df['month']/12)

# 'day'をSin/Cosに変換 (365日周期)
train_df['day_sin'] = np.sin(2 * np.pi * train_df['day']/365)
train_df['day_cos'] = np.cos(2 * np.pi * train_df['day']/365)
test_df['day_sin'] = np.sin(2 * np.pi * test_df['day']/365)
test_df['day_cos'] = np.cos(2 * np.pi * test_df['day']/365)

# 新しい特徴量が追加されたことを確認
print("新しい特徴量が追加されました:")
display(train_df[['maxtemp', 'mintemp', 'temp_range', 'temparature', 'dewpoint', 'dewpoint_spread']].head())


# dropped_columns = ['id','day','pressure','maxtemp','temparature','mintemp','temp_range','dewpoint','dewpoint_spread','humidity','cloud','sunshine','winddirection','windspeed','rainfall']
dropped_columns = ['id','day','rainfall']
# dropped_columns = ['id','day','rainfall']
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
    max_depth=20,
    max_features='sqrt',
    random_state=42,
    bootstrap=False,
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


pd.set_option('display.max_rows',900)


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



print("\n--- 不正解だったデータすべて（sunshine 昇順） ---")
incorrect_sorted = incorrect_predictions.sort_values(by='sunshine', ascending=True)
display(incorrect_sorted)


# sunshine が 3.6 以下の不正解データを抽出
low_sun_incorrect = incorrect_predictions[incorrect_predictions['sunshine'] <= 3.6].copy()

# 指定された15列
cols = [
   'pressure','maxtemp','temparature','mintemp','temp_range',
    'dewpoint','dewpoint_spread','humidity','cloud','sunshine',
    'winddirection','windspeed'
]

subset = low_sun_incorrect[cols]



import matplotlib.pyplot as plt
import seaborn as sns

# 相関行列
corr_matrix = subset.corr(numeric_only=True)

# ヒートマップで表示
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
plt.title("Correlation Matrix of Incorrect Predictions (sunshine ≤ 3.6)", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 特徴量の標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# モデル定義（max_iter 増加）
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 予測と評価
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC スコア: {auc:.4f}")

# ROC曲線
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})', color='darkorange', linewidth=2)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



# 提出用スコアファイルを作成
result_df = pd.DataFrame({
    'metric': ['roc_auc_score'],
    'value': [roc_auc]
})

# CSVに保存
result_df.to_csv("roc_auc_score_submission.csv", index=False)
print("✅ ROC-AUCスコアを 'roc_auc_score_submission.csv' に保存しました。")



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

