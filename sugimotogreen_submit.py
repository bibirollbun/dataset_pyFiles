import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc


train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train_df.head(10)


train_df.describe()


# ラグ特徴量を作成するカラムのリスト
lag_feature_cols = ['cloud','sunshine','mintemp','maxtemp']

# 訓練データとテストデータの両方に処理を適用
for df in [train_df, test_df]:

    # 変換前に欠損値を中央値で補完
    df.fillna(df.median(), inplace=True)
    
    # 処理前にdayでソートする
    df.sort_values(by='id', inplace=True)
    
    # 【新規追加】ラグ特徴量（前日からの変化量）
    for col in lag_feature_cols:
        # diff()メソッドで前日の行との差分を計算
        df[f'{col}_change'] = df[col].diff()
    
    # 差分計算で発生した最初の行のNaNを0で埋める
    df.fillna(0, inplace=True)


# 新しく追加された特徴量の一部を確認
print("前日からの変化量を示すラグ特徴量が追加されました:")
display(train_df.head(10))


# 特徴量 (X) とターゲット (y) を定義
# 'id' とターゲット列 'rainfall' を除いたすべての列を特徴量とする
X = train_df.drop(columns=['id', 'rainfall', 'day','winddirection','winddirection','dewpoint','temparature','mintemp','maxtemp'])
y = train_df['rainfall']

print("Full feature set shape (X):", X.shape)
print("Target variable shape (y):", y.shape)


# データを訓練用(80%)と検証用(20%)に分割
# この分割により、モデルの性能を未知のデータで公平に評価できるようになる
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


sns.set(style = "darkgrid")

sns.countplot(data=train_df, x="rainfall", hue="rainfall", palette="mako")
plt.tight_layout()
plt.show()


# --- ここからアンダーサンプリング処理を追加 ---

# 訓練データの特徴量と目的変数を一旦結合します
df_train_for_sampling = pd.concat([X_train, y_train], axis=1)

# 正例 (rainfall=1, 多数派) と負例 (rainfall=0, 少数派) に分割
majority_df = df_train_for_sampling[df_train_for_sampling['rainfall'] == 1]
minority_df = df_train_for_sampling[df_train_for_sampling['rainfall'] == 0]

# 少数派クラスのサンプル数を取得
minority_sample_count = len(minority_df)
print(f"少数派 (雨なし) のサンプル数: {minority_sample_count}")
print(f"多数派 (雨あり) のサンプル数: {len(majority_df)}")




# 多数派クラス（正例）を少数派クラスのサンプル数までアンダーサンプリングします
majority_downsampled = majority_df.sample(
    n=minority_sample_count,
    random_state=42  # 再現性のための乱数シード
)

# ダウンサンプリングした多数派データと、元の少数派データを結合します
df_resampled = pd.concat([minority_df, majority_downsampled])

# データをシャッフルしてインデックスをリセットします
df_resampled = df_resampled.sample(frac=1, random_state=42).reset_index(drop=True)

# 再度、特徴量 (X_train_resampled) と目的変数 (y_train_resampled) に分割します
X_train_resampled = df_resampled.drop('rainfall', axis=1)
y_train_resampled = df_resampled['rainfall']

print(f"\nアンダーサンプリング後の訓練データ数: {len(df_resampled)}")
print("サンプリング後のクラス分布:")
print(y_train_resampled.value_counts())

sns.set(style = "darkgrid")
sns.countplot(data=df_resampled, x="rainfall", hue="rainfall", palette="mako")
plt.tight_layout()
plt.show()


corr_matrix = train_df.corr(numeric_only = True)

plt.figure(figsize = (10, 8))
sns.heatmap(data = corr_matrix, annot = True, fmt = ".2f", cmap = "coolwarm", square = True, linewidths = 0.5)
plt.title("Correlation Matrix")
plt.show()


# パイプラインを定義 (欠損値補完 -> 標準化)
transformer = Pipeline([
    ('impute', SimpleImputer(strategy="most_frequent")),
    ('scaler', StandardScaler())
])

# パイプラインを訓練データで学習(fit)させ、変換(transform)する
X_train_transformed = transformer.fit_transform(X_train)

# 学習済みのパイプラインを使い、検証データを変換(transform)する
X_test_transformed = transformer.transform(X_test)

print("前処理後の訓練データの形状:", X_train_transformed.shape)
print("前処理後の検証データの形状:", X_test_transformed.shape)


# ランダムフォレストモデルを定義
rand_forest = RandomForestClassifier(random_state=42)

# モデルの学習
print("モデルの学習を開始します...")
rand_forest.fit(X_train_transformed, y_train)
print("モデルの学習が完了しました。")


# ランダムフォレストの特徴量の重要度を表示
rand_forest_importances = pd.DataFrame(rand_forest.feature_importances_, index=X.columns, columns=['Importance'])
print("ランダムフォレストの特徴量の重要度:")
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


# --- ここからファイル出力処理 ---

# 出力ファイル名を指定
output_filename = 'analysis_results.txt'

# ファイルを開いて書き込み（'w'は上書きモード, encoding='utf-8'は文字化け対策）
with open(output_filename, 'w', encoding='utf-8') as f:
    f.write("--- 正解したデータの上位5件 (雨と予測した確率が高い順) ---\n")
    # DataFrameを文字列に変換してファイルに書き込む
    f.write(top_5_correct.to_string() + '\n')

    f.write("\n--- 不正解だったデータすべて ---\n")
    # DataFrameを文字列に変換してファイルに書き込む
    f.write(incorrect_predictions.to_string() + '\n')

print(f"分析結果を {output_filename} に出力しました。")


# 特徴量エンジニアリング済みのテストデータを準備
#（これは以前のセルで作成済みの test_df_expanded を使います）
kaggle_test_df = test_df.copy()

# 訓練データから選択された特徴量と全く同じ特徴量をテストデータから抽出
# 'selected_features'は、RFECVで訓練データから学習した際に得られた特徴量のリストです
kaggle_test_selected = kaggle_test_df[X.columns]

print("Kaggle用テストデータから特徴量を選択しました。")
print("形状:", kaggle_test_selected.shape)

# 訓練データで学習済みの 'transformer' を使って、テストデータを変換（transform）のみ行う
# 【重要】ここでは .transform() を使います
kaggle_test_transformed = transformer.transform(kaggle_test_selected)

print("\n前処理後のKaggle用テストデータの形状:", kaggle_test_transformed.shape)



# 最終的なモデルを使って予測確率を計算
probabilities = rand_forest.predict_proba(kaggle_test_transformed)[:, 1]

# 提出用ファイルを作成
submission = pd.DataFrame({
    'id': kaggle_test_df['id'],
    'rainfall': probabilities
})

submission.to_csv('submission.csv', index=False)

print("\nsubmission.csv を作成しました。")
submission.head()

