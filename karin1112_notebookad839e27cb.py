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


import os
import gc
import re
import string
import zipfile
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# データセットの初期ロードとZIPファイル解凍
print("データセットの初期ロードとZIPファイルの解凍プロセスを開始しています。")

# データが配置されているKaggleディレクトリパス
KAGGLE_DATA_DIR = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/'
# 展開および作業用ディレクトリパス
CURRENT_WORKING_DIR = '/kaggle/working/'

# ZIPファイルを指定のパスへ展開するユーティリティ関数
def safe_unzip_archive(zip_source_path, extraction_target_path):
    with zipfile.ZipFile(zip_source_path, 'r') as zip_handler:
        zip_handler.extractall(extraction_target_path)
    print(f"アーカイブ '{os.path.basename(zip_source_path)}' の展開を '{extraction_target_path}' に行いました。")

try:
    # 訓練用CSVの展開とPandas DataFrameへのロード
    safe_unzip_archive(os.path.join(KAGGLE_DATA_DIR, 'train.csv.zip'), CURRENT_WORKING_DIR)
    df_train_main = pd.read_csv(os.path.join(CURRENT_WORKING_DIR, 'train.csv'))
    
    # テスト用CSVの展開とPandas DataFrameへのロード
    safe_unzip_archive(os.path.join(KAGGLE_DATA_DIR, 'test.csv.zip'), CURRENT_WORKING_DIR)
    df_test_main = pd.read_csv(os.path.join(CURRENT_WORKING_DIR, 'test.csv'))
    
    # サブミッション用CSVの展開とPandas DataFrameへのロード
    safe_unzip_archive(os.path.join(KAGGLE_DATA_DIR, 'sample_submission.csv.zip'), CURRENT_WORKING_DIR)
    df_submission_base = pd.read_csv(os.path.join(CURRENT_WORKING_DIR, 'sample_submission.csv'))

    print("全てのデータセットが正常にロードされました。")

except FileNotFoundError as file_not_found_err:
    print(f"エラー: 必要なデータファイルが見つかりません。パスを確認してください - {file_not_found_err}")
    print(">>> Kaggle Notebookのデータソースが正しく設定されているかご確認ください。 <<<")
    exit()
except Exception as unexpected_err:
    print(f"データロードまたはZIP展開処理中に予期せぬエラーが発生しました: {unexpected_err}")
    exit()

if df_train_main is None or df_test_main is None or df_submission_base is None:
    print("データフレームのロードが失敗したため、プログラムの実行を停止します。")
else:
    # コメント分類用のターゲットラベル
    CLASSIFICATION_TARGETS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # --- 2. コメントテキストの前処理ロジック ---
    # テキストコメントを整形し、標準化する関数
    def normalize_comment_string_v2(raw_text):
        if not isinstance(raw_text, str):
            return "" # 入力が文字列でない場合のハンドリング
        normalized_text = raw_text.lower() # テキストを小文字化
        normalized_text = re.sub(f'[{re.escape(string.punctuation)}]', '', normalized_text) # 句読点を削除
        normalized_text = re.sub(r'\d+', '', normalized_text) # 数字を削除
        normalized_text = re.sub(r'\s+', ' ', normalized_text).strip() # 複数スペースを単一に、前後の空白も除去
        return normalized_text

    print("コメントテキストデータへの前処理適用を開始します...")
    df_train_main['comment_text'] = df_train_main['comment_text'].apply(normalize_comment_string_v2)
    df_test_main['comment_text'] = df_test_main['comment_text'].apply(normalize_comment_string_v2)
    print("テキスト前処理の適用が完了しました。")

# --- 3. 特徴量エンジニアリング (TF-IDFベクトル化) ---
print("TF-IDFベクトル化のために全コメントテキストを連結中...")
unified_comment_corpus = pd.concat([df_train_main['comment_text'], df_test_main['comment_text']], axis=0)

print("TF-IDF Vectorizerの訓練フェーズを開始中...")
tfidf_transformer_obj = TfidfVectorizer(
    min_df=3, max_df=0.9, ngram_range=(1, 3),
    stop_words='english', max_features=50000
)
sparse_features_matrix = tfidf_transformer_obj.fit_transform(unified_comment_corpus)

#  train/testのスライス
train_features_matrix = sparse_features_matrix[:len(df_train_main)]
test_features_matrix = sparse_features_matrix[len(df_train_main):]

print(f"TF-IDFベクトル化処理が完了。生成された最終特徴量次元数: {train_features_matrix.shape[1]}")

del unified_comment_corpus, sparse_features_matrix
gc.collect()

# --- 4. モデル訓練 ---
print("各ターゲットカテゴリに対するモデルの訓練プロセスを開始します...")
final_submission_dataframe = pd.DataFrame({'id': df_test_main['id']})

for target_cat in CLASSIFICATION_TARGETS:
    print(f"  ターゲットカテゴリ '{target_cat}' のモデル訓練中...")
    target_train_series = df_train_main[target_cat]

    #  各ターゲットごとに新しいモデルを作る
    model = OneVsRestClassifier(
        LogisticRegression(solver='sag', n_jobs=-1, max_iter=1000, random_state=47)
    )
    model.fit(train_features_matrix, target_train_series)

    final_submission_dataframe[target_cat] = model.predict_proba(test_features_matrix)[:, 1]
    print(f"  ターゲットカテゴリ '{target_cat}' の訓練と予測が完了しました。")

print("全ての予測モデルの訓練および予測結果の生成が完了しました。")

# --- 5. CSV保存 ---
final_submission_dataframe.to_csv('submission_F.csv', index=False)
print("\n--- 最終提出用CSVファイル 'submission_F.csv' が正常に作成されました！ ---")

