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


# --- 必要なライブラリのインポート ---
import pandas as pd
import numpy as np
import re
import string
import gc
import zipfile
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# --- データ読み込みとZIP展開処理 ---
print("データファイルの読み込みとZIP解凍作業を開始")

# Kaggleデータセットのパス設定
data_source_path = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/'
output_extraction_path = '/kaggle/working/'

# zipファイルを展開する関数定義
def unzip_file_to_destination(zip_file_full_path, extraction_target_path):
    with zipfile.ZipFile(zip_file_full_path, 'r') as zip_archive:
        zip_archive.extractall(extraction_target_path)
    print(f"'{zip_file_full_path.split('/')[-1]}' の展開が完了")

try:
    # 各CSVファイルを展開
    unzip_file_to_destination(os.path.join(data_source_path, 'train.csv.zip'), output_extraction_path)
    unzip_file_to_destination(os.path.join(data_source_path, 'test.csv.zip'), output_extraction_path)
    unzip_file_to_destination(os.path.join(data_source_path, 'sample_submission.csv.zip'), output_extraction_path)

    # 展開されたCSVファイルをDataFrameとして読み込み
    train_dataframe = pd.read_csv(os.path.join(output_extraction_path, 'train.csv'))
    test_dataframe = pd.read_csv(os.path.join(output_extraction_path, 'test.csv'))
    sample_submission_df = pd.read_csv(os.path.join(output_extraction_path, 'sample_submission.csv'))

    print("全データセットの読み込みを確認")

except FileNotFoundError as fnf_error:
    print(f"エラー発生: 指定されたファイルが見つかりません - {fnf_error}")
    exit()
except Exception as general_error:
    print(f"データロードまたはZIP展開処理中に、問題が発生しました: {general_error}")
    exit()

# --- テキスト前処理（クレンジング） ---
TOXICITY_CATEGORIES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# コメント本文を整形する関数（小文字化・記号/数字/HTMLタグ除去など）
def normalize_comment_text(input_text):
    if not isinstance(input_text, str):
        return ""
    input_text = input_text.lower()
    input_text = re.sub(r'<.*?>', ' ', input_text)  # HTMLタグ除去
    input_text = re.sub(r'https?://\S+|www\.\S+', ' ', input_text)  # URL除去
    input_text = re.sub(f'[{re.escape(string.punctuation)}]', '', input_text)  # 記号除去
    input_text = re.sub(r'\d+', '', input_text)  # 数字除去
    input_text = re.sub(r'\s+', ' ', input_text).strip()  # 空白整形
    return input_text

print("コメントテキストの整形処理を実行中")
train_dataframe['comment_text'] = train_dataframe['comment_text'].apply(normalize_comment_text)
test_dataframe['comment_text'] = test_dataframe['comment_text'].apply(normalize_comment_text)
print("テキスト整形処理が完了")

# --- TF-IDFベクトルによる特徴量抽出 ---
print("TF-IDF変換の準備中")

# trainとtestのコメント本文を統合してTF-IDFベクトル化（語彙共有のため）
full_comment_corpus = pd.concat([train_dataframe['comment_text'], test_dataframe['comment_text']], axis=0)

# TF-IDFベクトライザの設定（n-gram強化、語彙数制限）
vectorizer_for_tfidf = TfidfVectorizer(
    min_df=3,               # 3文書未満でしか出ない単語は無視
    max_df=0.9,             # 90%以上の文書に出る単語は無視
    ngram_range=(1, 3),     # uni-gram～tri-gramまで使用
    stop_words='english',   # 英語ストップワード除去
    max_features=100000     # 最大特徴数の制限
)

# TF-IDFベクトルの作成（train + test）
transformed_features = vectorizer_for_tfidf.fit_transform(full_comment_corpus)

# trainとtestに再分割
train_features_set = transformed_features[:len(train_dataframe)]
test_features_set = transformed_features[len(train_dataframe):]
print(f"TF-IDFベクトル変換完了 特徴数: {train_features_set.shape[1]}")

# 不要なオブジェクト削除・メモリ解放
del full_comment_corpus, transformed_features
gc.collect()

# --- モデル学習と予測（ロジスティック回帰） ---
# OneVsRestClassifierを使用してマルチラベル分類を個別に処理
classifier = OneVsRestClassifier(
    LogisticRegression(
        solver='sag',        # L2正則化付き確率的平均勾配法
        C=4.0,               # 正則化パラメータ（大きめ＝過学習寄り）
        n_jobs=-1,           # 並列処理
        max_iter=1000,       # 最大反復回数
        random_state=43      # 再現性のための乱数シード
    )
)

print("モデル訓練を開始")

# 結果格納用のDataFrame（testのidを含む）
final_predictions_df = pd.DataFrame({'id': test_dataframe['id']})

# 6つのラベルごとに分類モデルを訓練・予測
for category in TOXICITY_CATEGORIES:
    print(f" '{category}' を訓練中")
    y = train_dataframe[category]
    classifier.fit(train_features_set, y)
    final_predictions_df[category] = classifier.predict_proba(test_features_set)[:, 1]
    print(f" '{category}' の予測完了")

# --- 結果ファイルの出力 ---
final_predictions_df.to_csv('submission.csv', index=False)
print("\n--- 結果ファイル 'submission.csv' を出力 ---")

