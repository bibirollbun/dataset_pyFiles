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


# toxic フラグが立っているレコードの内訳を確認する

# 'toxic'が'1'のデータを抽出
toxic_only_df = train_df[train_df['toxic'] == 1].copy()

# toxic 以外のフラグカラムのリスト
toxic_other_label_columns = [
    'severe_toxic', 
    'obscene', 
    'threat', 
    'insult', 
    'identity_hate'
]

# toxic 以外のフラグの合計を新しいカラムとして作成
toxic_only_df['other_flag_sum'] = toxic_only_df[toxic_other_label_columns].sum(axis=1)

# --- 集計 ---
# dfが行数 N、列数 M の場合、df.shape は (N, M) を返す
toxic_count = toxic_only_df.shape[0] 
print(f"toxic フラグが立っているレコードの合計: {toxic_count:,} 件")

print("")
print("--- toxic フラグが立っているレコードの内訳 ---")




# A. toxic フラグ「だけ」立っているレコード数
#    (other_flag_sum が 0 のレコード)
toxic_only_count = (toxic_only_df['other_flag_sum'] == 0).sum()

print(f"1. toxic フラグ『だけ』立っているレコード数: {toxic_only_count:,} 件")

# B. toxic フラグ「以外」も立っているレコード数
#    (other_flag_sum が 1 以上のレコード)
toxic_and_other_count = (toxic_only_df['other_flag_sum'] >= 1).sum()

print(f"2. toxic フラグ『以外』も立っているレコード数: {toxic_and_other_count:,} 件")




# --- 集計 ---
# 他カラムの 1 の数を確認する
toxic_and_other_count_of_ones = toxic_only_df[toxic_other_label_columns].sum()

print("")
print("--- toxic フラグ以外の各カラムの 1 の数 ---")

# 結果を整形して表示
for col, count in toxic_and_other_count_of_ones.items(): # colにカラム名、countに値が代入される
    print(f"{col:<15}: {count:>6} 件") # colは左揃え(<)で15文字以内、countは右揃え(>)で6文字以内


# severe_toxic フラグが立っているレコードの内訳を確認する

# 'severe_toxic'が 1 のデータを抽出
severe_toxic_only_df = train_df[train_df['severe_toxic'] == 1].copy()

# severe_toxic 以外のフラグカラムのリスト
severe_toxic_other_label_columns = [
    'toxic', 
    'obscene', 
    'threat', 
    'insult', 
    'identity_hate'
]

# severe_toxic 以外のフラグの合計を新しいカラムとして作成
severe_toxic_only_df['other_flag_sum'] = severe_toxic_only_df[severe_toxic_other_label_columns].sum(axis=1)

# --- 集計 ---
# dfが行数 N、列数 M の場合、df.shape は (N, M) を返す
severe_toxic_count = severe_toxic_only_df.shape[0] 
print(f"severe_toxic フラグが立っているレコードの合計: {severe_toxic_count:,} 件")

print("")
print("--- severe_toxic フラグが立っているレコードの内訳 ---")




# A. severe_toxic フラグ「だけ」立っているレコード数
#    (other_flag_sum が 0 のレコード)
severe_toxic_only_count = (severe_toxic_only_df['other_flag_sum'] == 0).sum()

print(f"1. severe_toxic フラグ『だけ』立っているレコード数: {severe_toxic_only_count:,} 件")

# B. severe_toxic フラグ「以外」も立っているレコード数
#    (other_flag_sum が 1 以上のレコード)
severe_toxic_and_other_count = (severe_toxic_only_df['other_flag_sum'] >= 1).sum()

print(f"2. severe_toxic フラグ『以外』も立っているレコード数: {severe_toxic_and_other_count:,} 件")




# --- 集計 ---
# 他カラムの 1 の数を確認する
severe_toxic_and_other_count_of_ones = severe_toxic_only_df[severe_toxic_other_label_columns].sum()

print("")
print("--- severe_toxic フラグ以外の各カラムの 1 の数 ---")

# 結果を整形して表示
for col, count in severe_toxic_and_other_count_of_ones.items(): # colにカラム名、countに値が代入される
    print(f"{col:<15}: {count:>6} 件") # colは左揃え(<)で15文字以内、countは右揃え(>)で6文字以内


# obscene フラグが立っているレコードの内訳を確認する

# 'obscene'が 1 のデータを抽出
obscene_only_df = train_df[train_df['obscene'] == 1].copy()

# obscene 以外のフラグカラムのリスト
obscene_other_label_columns = [
    'toxic', 
    'severe_toxic', 
    'threat', 
    'insult', 
    'identity_hate'
]

# obscene 以外のフラグの合計を新しいカラムとして作成
obscene_only_df['other_flag_sum'] = obscene_only_df[obscene_other_label_columns].sum(axis=1)

# --- 集計 ---
# dfが行数 N、列数 M の場合、df.shape は (N, M) を返す
obscene_count = obscene_only_df.shape[0] 
print(f"obscene フラグが立っているレコードの合計: {obscene_count:,} 件")

print("")
print("--- obscene フラグが立っているレコードの内訳 ---")




# A. obscene フラグ「だけ」立っているレコード数
#    (other_flag_sum が 0 のレコード)
obscene_only_count = (obscene_only_df['other_flag_sum'] == 0).sum()

print(f"1. obscene フラグ『だけ』立っているレコード数: {obscene_only_count:,} 件")

# B. obscene フラグ「以外」も立っているレコード数
#    (other_flag_sum が 1 以上のレコード)
obscene_and_other_count = (obscene_only_df['other_flag_sum'] >= 1).sum()

print(f"2. obscene フラグ『以外』も立っているレコード数: {obscene_and_other_count:,} 件")




# --- 集計 ---
# 他カラムの 1 の数を確認する
obscene_and_other_count_of_ones = obscene_only_df[obscene_other_label_columns].sum()

print("")
print("--- obscene フラグ以外の各カラムの 1 の数 ---")

# 結果を整形して表示
for col, count in obscene_and_other_count_of_ones.items(): # colにカラム名、countに値が代入される
    print(f"{col:<15}: {count:>6} 件") # colは左揃え(<)で15文字以内、countは右揃え(>)で6文字以内


# threat フラグが立っているレコードの内訳を確認する

# 'threat'が 1 のデータを抽出
threat_only_df = train_df[train_df['threat'] == 1].copy()

# threat 以外のフラグカラムのリスト
threat_other_label_columns = [
    'toxic', 
    'severe_toxic', 
    'obscene', 
    'insult', 
    'identity_hate'
]

# threat 以外のフラグの合計を新しいカラムとして作成
threat_only_df['other_flag_sum'] = threat_only_df[threat_other_label_columns].sum(axis=1)

# --- 集計 ---
# dfが行数 N、列数 M の場合、df.shape は (N, M) を返す
threat_count = threat_only_df.shape[0] 
print(f"threat フラグが立っているレコードの合計: {threat_count:,} 件")

print("")
print("--- threat フラグが立っているレコードの内訳 ---")




# A. threat フラグ「だけ」立っているレコード数
#    (other_flag_sum が 0 のレコード)
threat_only_count = (threat_only_df['other_flag_sum'] == 0).sum()

print(f"1. threat フラグ『だけ』立っているレコード数: {threat_only_count:,} 件")

# B. threat フラグ「以外」も立っているレコード数
#    (other_flag_sum が 1 以上のレコード)
threat_and_other_count = (threat_only_df['other_flag_sum'] >= 1).sum()

print(f"2. threat フラグ『以外』も立っているレコード数: {threat_and_other_count:,} 件")




# --- 集計 ---
# 他カラムの 1 の数を確認する
threat_and_other_count_of_ones = threat_only_df[threat_other_label_columns].sum()

print("")
print("--- threat フラグ以外の各カラムの 1 の数 ---")

# 結果を整形して表示
for col, count in threat_and_other_count_of_ones.items(): # colにカラム名、countに値が代入される
    print(f"{col:<15}: {count:>6} 件") # colは左揃え(<)で15文字以内、countは右揃え(>)で6文字以内


# insult フラグが立っているレコードの内訳を確認する

# 'insult'が 1 のデータを抽出
insult_only_df = train_df[train_df['insult'] == 1].copy()

# insult 以外のフラグカラムのリスト
insult_other_label_columns = [
    'toxic', 
    'severe_toxic', 
    'obscene',
    'threat',  
    'identity_hate'
]

# insult 以外のフラグの合計を新しいカラムとして作成
insult_only_df['other_flag_sum'] = insult_only_df[insult_other_label_columns].sum(axis=1)

# --- 集計 ---
# dfが行数 N、列数 M の場合、df.shape は (N, M) を返す
insult_count = insult_only_df.shape[0] 
print(f"insult フラグが立っているレコードの合計: {insult_count:,} 件")

print("")
print("--- insult フラグが立っているレコードの内訳 ---")




# A. insult フラグ「だけ」立っているレコード数
#    (other_flag_sum が 0 のレコード)
insult_only_count = (insult_only_df['other_flag_sum'] == 0).sum()

print(f"1. insult フラグ『だけ』立っているレコード数: {insult_only_count:,} 件")

# B. insult フラグ「以外」も立っているレコード数
#    (other_flag_sum が 1 以上のレコード)
insult_and_other_count = (insult_only_df['other_flag_sum'] >= 1).sum()

print(f"2. insult フラグ『以外』も立っているレコード数: {insult_and_other_count:,} 件")




# --- 集計 ---
# 他カラムの 1 の数を確認する
insult_and_other_count_of_ones = insult_only_df[insult_other_label_columns].sum()

print("")
print("--- insult フラグ以外の各カラムの 1 の数 ---")

# 結果を整形して表示
for col, count in insult_and_other_count_of_ones.items(): # colにカラム名、countに値が代入される
    print(f"{col:<15}: {count:>6} 件") # colは左揃え(<)で15文字以内、countは右揃え(>)で6文字以内


# identity_hate フラグが立っているレコードの内訳を確認する

# 'identity_hate'が 1 のデータを抽出
identity_hate_only_df = train_df[train_df['identity_hate'] == 1].copy()

# identity_hate 以外のフラグカラムのリスト
identity_hate_other_label_columns = [
    'toxic', 
    'severe_toxic',
    'obscene', 
    'threat', 
    'insult'
]

# identity_hate 以外のフラグの合計を新しいカラムとして作成
identity_hate_only_df['other_flag_sum'] = identity_hate_only_df[identity_hate_other_label_columns].sum(axis=1)

# --- 集計 ---
# dfが行数 N、列数 M の場合、df.shape は (N, M) を返す
identity_hate_count = identity_hate_only_df.shape[0] 
print(f"identity_hate フラグが立っているレコードの合計: {identity_hate_count:,} 件")

print("")
print("--- identity_hate フラグが立っているレコードの内訳 ---")




# A. identity_hate フラグ「だけ」立っているレコード数
#    (other_flag_sum が 0 のレコード)
identity_hate_only_count = (identity_hate_only_df['other_flag_sum'] == 0).sum()

print(f"1. identity_hate フラグ『だけ』立っているレコード数: {identity_hate_only_count:,} 件")

# B. identity_hate フラグ「以外」も立っているレコード数
#    (other_flag_sum が 1 以上のレコード)
identity_hate_and_other_count = (identity_hate_only_df['other_flag_sum'] >= 1).sum()

print(f"2. identity_hate フラグ『以外』も立っているレコード数: {identity_hate_and_other_count:,} 件")




# --- 集計 ---
# 他カラムの 1 の数を確認する
identity_hate_and_other_count_of_ones = identity_hate_only_df[identity_hate_other_label_columns].sum()

print("")
print("--- identity_hate フラグ以外の各カラムの 1 の数 ---")

# 結果を整形して表示
for col, count in identity_hate_and_other_count_of_ones.items(): # colにカラム名、countに値が代入される
    print(f"{col:<15}: {count:>6} 件") # colは左揃え(<)で15文字以内、countは右揃え(>)で6文字以内


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
# 1. min_df: この回数未満しか出現しない単語は無視（ノイズ削減）
# 2. max_features: 特徴量（単語）の最大数を制限（計算量削減）
# 3. stop_words: 英語の一般的なストップワード（the, a, isなど）を除去
tfidf_params = {
    'min_df': 3,
    'max_features': 100000, # 例として最大100000語に制限
    'stop_words': 'english'
}


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




