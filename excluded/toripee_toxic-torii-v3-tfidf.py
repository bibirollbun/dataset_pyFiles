# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")
test_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")
sample_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip")


train_df


test_df


sample_df


# データ型を調べる
print(train_df.info())


# カラムの中に入っているデータを見る
train_df['toxic'].value_counts()


# 全カラムの中に入っているデータを見る

# 全カラムのリスト
label_columns = [
    'toxic',
    'severe_toxic',
    'obscene', 
    'threat', 
    'insult', 
    'identity_hate'
]

# 入っている値をカウント
train_df[label_columns].value_counts()


import numpy as np

# ランダムに10行を抽出
train_df_sample = train_df.sample(n=10)

# すべての行と列、そして列の最大幅を制限なしに設定
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

print(train_df_sample)


# リセットする
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
pd.reset_option('display.max_colwidth')


from sklearn.feature_extraction.text import TfidfVectorizer


# ベクトル化のパラメータ設定
tfidf_params = {
    # ------------------------------------------------------------------
    # 1. 頻度フィルタリング（語彙の制御）
    # ------------------------------------------------------------------
    'min_df': 3,                 # [int/float, デフォルト=1] この回数/割合未満しか出現しない単語は無視（ノイズ削減）。
    'max_df': 1.0,               # [int/float, デフォルト=1.0] この回数/割合を超えて出現する単語は無視（一般的すぎる単語の除去）。
    'max_features': 100000,      # [int, デフォルト=None] 特徴量（単語）の最大数。
    
    # ------------------------------------------------------------------
    # 2. N-gramと前処理
    # ------------------------------------------------------------------
    'ngram_range': (1, 2),       # [tuple, デフォルト=(1, 1)] 使用するN-gramの範囲 (min_n, max_n)。例: (1, 2)はユニグラムとバイグラムを使用。
    'stop_words': 'english',     # [str/'english'/list, デフォルト=None] ストップワードのリスト。'english'で組み込みリスト使用。
    'lowercase': True,           # [bool, デフォルト=True] テキストを小文字に変換するかどうか。
    'token_pattern': r'(?u)\b\w\w+\b', # [str, デフォルト=r'(?u)\b\w\w+\b'] トークンを抽出するための正規表現。
    'tokenizer': None,           # [callable, デフォルト=None] カスタムのトークン化関数（例: spaCyの活用）を指定。
    
    # ------------------------------------------------------------------
    # 3. TF-IDFの計算方法（重み付けの調整）
    # ------------------------------------------------------------------
    'use_idf': True,             # [bool, デフォルト=True] IDF（逆文書頻度）の重み付けを使用するかどうか。
    'smooth_idf': True,          # [bool, デフォルト=True] IDF値を平滑化（スムージング）するかどうか。
    'sublinear_tf': True,       # [bool, デフォルト=False] TF（単語頻度）を対数的に変換し、頻度が高い単語の影響を抑制するかどうか。
    'norm': 'l2',                # [str/'l1'/'l2', デフォルト='l2'] 出力ベクトルを正規化する方法。通常はL2正規化を使用。
}

"""
特にチューニングの重要度が高いパラメータは以下の通りです。
- ngram_range: (1, 2)（ユニグラムとバイグラム）に広げると、単語の組み合わせ（例: "not good"）を特徴量として捉えられ、性能が向上しやすいです。
- max_features: 使用可能なリソースに合わせて、10000〜100000の間で試行錯誤します。
- sublinear_tf: 頻度の高い単語の影響を抑えたい場合は True に設定を試みます。
"""


# 1. Vectorizerを初期化
vectorizer = TfidfVectorizer(**tfidf_params)

# 2. データをモデルに適合させ（fit）、同時に変換（transform）を実行
#    fit: 語彙リスト（単語辞書）とIDF値（逆文書頻度）を計算する
#    transform: 各文書のTF-IDF値を計算し、疎行列を生成する
X_tfidf = vectorizer.fit_transform(train_df['comment_text']) # train_dfを用いてモデルにfitさせる
X_test_tfidf = vectorizer.transform(test_df['comment_text']) # それをtest_dfに使う

# 疎行列（Sparse Matrix）の形状を確認
print(f"訓練データの特徴量形状: {X_tfidf.shape}") 
print(f"テストデータの特徴量形状: {X_test_tfidf.shape}")
# (行数, 単語の数) が出力されます


# 結果の確認
feature_names = vectorizer.get_feature_names_out()
print(f"最初の5つの特徴量（単語）: {feature_names[:5]}")


# ターゲットカラムを定義
label_columns = [
    'toxic', 
    'severe_toxic', 
    'obscene', 
    'threat', 
    'insult', 
    'identity_hate'
]

# ターゲット変数yを抽出
y = train_df[label_columns]


from sklearn.model_selection import train_test_split

# データを分割
X_train, X_valid, y_train, y_valid = train_test_split(
    X_tfidf, 
    y, 
    test_size=0.2, #20%にしておく
    random_state=42
)


# ロジスティクス回帰で予測・分類
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score

# 1. ベースラインモデルの定義（ロジスティック回帰）
#    C: 正則化の強さ（小さいほど強い正則化）
#    solver='sag': 大規模データセット向けの高速なソルバー
base_model = LogisticRegression(C=4.0, solver='sag', random_state=42, max_iter=1000)

# 2. MultiOutputClassifierで6つのモデルをラップ
multi_target_model = MultiOutputClassifier(base_model, n_jobs=-1)

# 3. モデルの訓練
print("モデル訓練を開始します...")
multi_target_model.fit(X_train, y_train)
print("モデル訓練が完了しました。")


# テストデータで確率を予測
y_pred_proba = multi_target_model.predict_proba(X_valid)

# y_pred_proba はリストのリスト（各ラベルの確率）なので、Pandas DataFrame形式に変換
y_pred_proba_df = pd.DataFrame(
    [proba[:, 1] for proba in y_pred_proba]
).T # 転置して (行数, ラベル数) の形状にする
y_pred_proba_df.columns = label_columns

# 各ラベルのROC-AUCスコアを計算
auc_scores = roc_auc_score(y_valid, y_pred_proba_df, average=None)

print("\n--- 各ラベルの ROC-AUC スコア ---")
for col, score in zip(label_columns, auc_scores):
    print(f"{col:<15}: {score:.4f}")

# 全体の平均ROC-AUCスコア（コンペティションの評価基準）
mean_auc = roc_auc_score(y_valid, y_pred_proba_df, average='macro')
print(f"\n平均 ROC-AUC スコア: {mean_auc:.4f}")


# 1. 確率の予測を実行
# predict_probaは各モデルの予測確率をリストで返す
y_sub_pred_proba = multi_target_model.predict_proba(X_test_tfidf)

# 2. 予測結果をDataFrameに整形
# y_submission_pred_proba は (ラベル数, 行数, 2) の構造になっているため、整形が必要です。
# 各要素の [:, 1] は、ラベルが 1 である確率を抽出しています。
sub_pred_df = pd.DataFrame(
    [proba[:, 1] for proba in y_sub_pred_proba]
).T # 転置して (行数, ラベル数) の正しい形状にする

# カラム名を設定
sub_pred_df.columns = label_columns

print("予測結果のDataFrame（確率）:")
print(sub_pred_df.head())


# 1. 提出用DataFrameを準備
# IDカラムは test_df から取得するか、sample_submission_df から取得します。
sub_df = pd.DataFrame({'id': test_df['id']})

# 2. 予測確率を結合
sub_df = pd.concat([sub_df, sub_pred_df], axis=1)

# 3. ファイルとして保存
# index=False は、Pandasのインデックス（行番号）をファイルに書き込まないように指定します。
sub_df.to_csv('submission.csv', index=False)

print("\n提出ファイル 'submission.csv' が作成されました。")
print(sub_df.head())




