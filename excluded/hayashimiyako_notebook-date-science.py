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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc


train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train_df.head(10)


train_df.describe()


# ラグ特徴量を作成するカラムのリスト
lag_feature_cols = ['pressure','maxtemp','temparature','mintemp','dewpoint','humidity','cloud','sunshine','winddirection','windspeed']

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
    # df['day_sin'] = np.sin(2 * np.pi * df['day']/365)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/365)
    # df['wind_sin'] = np.sin(2 * np.pi * df['winddirection']/360)
    df['wind_cos'] = np.cos(2 * np.pi * df['winddirection']/360)
    
    # 交互作用特徴量
    df['pressure_humidity_interaction'] = df['humidity'] / df['pressure']
    
    # 【新規追加】ラグ特徴量（前日からの変化量）
    for col in lag_feature_cols:
        # diff()メソッドで前日の行との差分を計算
        df[f'{col}_change'] = df[col].diff()
    
    # 差分計算で発生した最初の行のNaNを0で埋める
    df.fillna(0, inplace=True)


# # 多項式特徴量を生成するベースとなる特徴量を選択
# poly_features_base = ['pressure','maxtemp','temparature','mintemp','dewpoint','humidity','cloud','sunshine','winddirection','windspeed']

# # 訓練データとテストデータの両方で処理
# # 元のデータフレームを保持するためコピーを作成
# train_poly = train_df.copy()
# test_poly = test_df.copy()

# # PolynomialFeaturesのインスタンスを作成 (2次の項まで、バイアス項は含めない)
# poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)

# # 訓練データでフィットし、両方のデータを変換
# X_train_poly_base = train_poly[poly_features_base]
# X_test_poly_base = test_poly[poly_features_base]

# poly_features_train = poly.fit_transform(X_train_poly_base)
# poly_features_test = poly.transform(X_test_poly_base)

# # 新しい列名を作成
# poly_feature_names = poly.get_feature_names_out(poly_features_base)

# # NumPy配列をDataFrameに変換
# df_poly_train = pd.DataFrame(poly_features_train, columns=poly_feature_names, index=train_poly.index)
# df_poly_test = pd.DataFrame(poly_features_test, columns=poly_feature_names, index=test_poly.index)

# # 元のデータフレームからベースにした特徴量を削除し、新しい多項式特徴量を結合
# train_df_expanded = train_poly.drop(columns=poly_features_base).join(df_poly_train)
# test_df_expanded = test_poly.drop(columns=poly_features_base).join(df_poly_test)

# print(f"特徴量生成前: {train_df.shape[1]}列")
# print(f"特徴量生成後: {train_df_expanded.shape[1]}列")
# display(train_df_expanded.head())


# 新しく追加された特徴量の一部を確認
print("前日からの変化量を示すラグ特徴量が追加されました:")
display(train_df.head(10))


# 特徴量 (X) とターゲット (y) を定義
# 'id' とターゲット列 'rainfall' を除いたすべての列を特徴量とする
X = train_df.drop(columns=['day','id', 'rainfall','winddirection'])
y = train_df['rainfall']

print("Full feature set shape (X):", X.shape)
print("Target variable shape (y):", y.shape)


# # 元のデータセットのクラスごとのサンプル数を確認
# print("--- アンダーサンプリング前 ---")
# print(y.value_counts())

# # 特徴量Xと正解ラベルyを一度結合して処理しやすくする
# df_combined = pd.concat([X, y], axis=1)

# # 雨が降ったデータ（多数派）と降らなかったデータ（少数派）に分割
# df_majority = df_combined[df_combined.rainfall == 1] # 多数派 (雨)
# df_minority = df_combined[df_combined.rainfall == 0] # 少数派 (晴れ)

# # 多数派のデータを少数派のデータ数までランダムにサンプリング
# df_majority_undersampled = df_majority.sample(n=len(df_minority), random_state=42)

# # アンダーサンプリング後の多数派データと少数派データを結合
# df_undersampled = pd.concat([df_majority_undersampled, df_minority])

# # 結合したデータをシャッフル
# df_undersampled = df_undersampled.sample(frac=1, random_state=42).reset_index(drop=True)

# # 再度、説明変数と目的変数に分割
# X_resampled = df_undersampled.drop('rainfall', axis=1)
# y_resampled = df_undersampled['rainfall']

# # アンダーサンプリング後のデータセットでサンプル数を確認
# print("\n--- アンダーサンプリング後 ---")
# print(y_resampled.value_counts())


sns.set(style = "darkgrid")

sns.countplot(data=train_df, x="rainfall", hue="rainfall", palette="mako")
plt.tight_layout()
plt.show()


# データを訓練用(80%)と検証用(20%)に分割
# stratify=y_resampled を追加して、クラスの比率を維持したまま分割する
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print("--- 分割後のデータ確認 ---")
print("\n[訓練データ]のクラスごとのサンプル数:")
print(y_train.value_counts())

print("\n[検証データ]のクラスごとのサンプル数:")
print(y_test.value_counts())


# # RFEに使用するモデルを定義 (計算負荷を考慮)
# estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# # RFECVのインスタンスを作成
# # cv=5 (5-fold CV) を使用し、訓練データ内で最適な特徴量を探す
# # 【修正点】fitは訓練データ(X_train, y_train)のみで行う
# selector = RFECV(
#     estimator, 
#     step=1, 
#     cv=5, 
#     scoring='roc_auc', # 評価基準をAUCに変更
#     n_jobs=-1
# )

# print("RFECVによる特徴量選択を開始します（評価基準: roc_auc）...")
# selector.fit(X_train, y_train)
# print("特徴量選択が完了しました。")

# # 選択された特徴量の名前を取得
# selected_features = X_train.columns[selector.support_]
# print(f"\n最適な特徴量の数: {selector.n_features_}")
# print("選択された特徴量:")
# print(list(selected_features))

# # 訓練データと検証データの両方から、選択された特徴量のみを保持
# X_train_selected = X_train[selected_features]
# X_test_selected = X_test[selected_features]

# print("\n選択後の訓練データの形状:", X_train_selected.shape)
# print("選択後の検証データの形状:", X_test_selected.shape)


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

# 【修正点】パイプラインを訓練データで学習(fit)させ、変換(transform)する
X_train_transformed = transformer.fit_transform(X_train)

# 【修正点】学習済みのパイプラインを使い、検証データを変換(transform)する
X_test_transformed = transformer.transform(X_test)

print("前処理後の訓練データの形状:", X_train_transformed.shape)
print("前処理後の検証データの形状:", X_test_transformed.shape)


# # 基本となるランダムフォレストモデルを定義
# rand_forest = RandomForestClassifier(random_state = 42)

# # 探索するハイパーパラメータの範囲を拡大・詳細化
# param_grid = {
#     'n_estimators': [50, 100, 200, 300],  
#     'max_depth': [10, 20, None],
#     'max_features': ['sqrt'],
#     'min_samples_split': [2, 5, 10],   
#     'min_samples_leaf': [1, 2, 4],       
#     'bootstrap': [True, False]            
# }

# rand_forest = GridSearchCV(rand_forest, param_grid, cv=5, verbose=1, n_jobs=-1)
# rand_forest.fit(X_train_transformed, y_train)

# print("Best params after tuning parameters:", rand_forest.best_params_)


# 最適なパラメータ（GridSearchCVで見つけたと仮定）を使用してランダムフォレストモデルを定義
# このセルはあなたのコードをそのまま使用
rand_forest = RandomForestClassifier(n_estimators=300, random_state=42)

# 【修正点】モデルの学習は、前処理済みの訓練データで行う
print("モデルの学習を開始します...")
rand_forest.fit(X_train_transformed, y_train)
print("モデルの学習が完了しました。")


# indexを「X.columns」から「selected_features」に変更
rand_forest_importances = pd.DataFrame(rand_forest.feature_importances_, index=X.columns, columns=['Importance'])

# ランダムフォレストの特徴量の重要度を表示
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

print("--- 正解したデータの上位5件 (雨と予測した確率が高い順) ---")
display(top_5_correct)

print("\n--- 不正解だったデータすべて ---")
display(incorrect_predictions)

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

