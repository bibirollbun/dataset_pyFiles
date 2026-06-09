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

# データセットのロードと準備フェーズ
print("データセットのロードとZIPファイル解凍プロセスを開始します。")

# データが格納されているパスと、展開先のパス
DATA_INPUT_DIR = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/'
EXTRACT_WORKING_DIR = '/kaggle/working/'




# ZIPファイルを安全に展開する関数
def unpack_zip_archive(source_zip_path, target_extract_path):
    with zipfile.ZipFile(source_zip_path, 'r') as zip_obj:
        zip_obj.extractall(target_extract_path)
    print(f"ZIPファイル '{os.path.basename(source_zip_path)}' を '{target_extract_path}' に展開しました。")

try:
    # 必要なデータファイルを展開し、Pandasデータフレームにロード
    unpack_zip_archive(os.path.join(DATA_INPUT_DIR, 'train.csv.zip'), EXTRACT_WORKING_DIR)
    train_dataframe_main = pd.read_csv(os.path.join(EXTRACT_WORKING_DIR, 'train.csv'))
    
    unpack_zip_archive(os.path.join(DATA_INPUT_DIR, 'test.csv.zip'), EXTRACT_WORKING_DIR)
    test_dataframe_main = pd.read_csv(os.path.join(EXTRACT_WORKING_DIR, 'test.csv'))
    
    unpack_zip_archive(os.path.join(DATA_INPUT_DIR, 'sample_submission.csv.zip'), EXTRACT_WORKING_DIR)
    submission_template_dataframe = pd.read_csv(os.path.join(EXTRACT_WORKING_DIR, 'sample_submission.csv'))

    print("全てのデータがロードされました。")

except FileNotFoundError as file_err:
    print(f"エラー: データファイルの読み込みに失敗しました - {file_err}")
    print(">>> Kaggle環境でデータセットが正しくリンクされているか、パスを確認してください。 <<<")
    exit()
except Exception as general_err:
    print(f"データロードまたはZIP展開中に、予期せぬエラーが発生しました: {general_err}")
    exit()
if train_dataframe_main is None or test_dataframe_main is None or submission_template_dataframe is None:
    print("データフレームのロードに失敗したため、スクリプトを終了します。")
else:
  # 分類対象となるコメントのカテゴリ名リスト
    COMMENT_CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


    # --- 2. コメントテキストの前処理ロジック ---
    # テキストコメントを整形し、クリーンな状態にする関数
    def preprocess_comment_string_final(input_string):
        if not isinstance(input_string, str):
            return "" # 文字列でない入力は空文字列として処理
        processed_string = input_string.lower() # 全体を小文字に変換
        processed_string = re.sub(f'[{re.escape(string.punctuation)}]', '', processed_string) # 句読点を全て除去
        processed_string = re.sub(r'\d+', '', processed_string) # 数字（数値）を全て削除
        processed_string = re.sub(r'\s+', ' ', processed_string).strip() # 複数の空白を単一の空白に、前後の空白も削除
        return processed_string

    print("コメントテキストデータの前処理を開始します...")
    train_dataframe_main['comment_text'] = train_dataframe_main['comment_text'].apply(preprocess_comment_string_final)
    test_dataframe_main['comment_text'] = test_dataframe_main['comment_text'].apply(preprocess_comment_string_final)
    print("テキストデータの前処理が完了しました。")



# --- 3. 特徴量抽出 (TF-IDFベクトル化) ---
print("TF-IDFベクトル化のため、全てのコメントテキストを連結しています...")
full_text_corpus_for_vectorizer = pd.concat([train_dataframe_main['comment_text'], test_dataframe_main['comment_text']], axis=0)

print("TF-IDF Vectorizerの訓練プロセスを開始中...")
tfidf_vectorizer_instance = TfidfVectorizer(
    min_df=3,
    max_df=0.9,
    ngram_range=(1, 2),
    stop_words='english',
    max_features=40000
)
transformed_sparse_features = tfidf_vectorizer_instance.fit_transform(full_text_corpus_for_vectorizer)

train_sparse_features = transformed_sparse_features[:len(train_dataframe_main)]
test_sparse_features = transformed_sparse_features[len(train_dataframe_main):]
print(f"TF-IDFベクトル化処理完了。最終的な特徴量の数: {train_sparse_features.shape[1]}")

del full_text_corpus_for_vectorizer, transformed_sparse_features
gc.collect()



# --- 4. モデル構築と訓練 (OneVsRest + ロジスティック回帰) ---
print("コメントの各クラスに対する予測モデルの訓練を開始します...")
submission_output_dataframe = pd.DataFrame({'id': test_dataframe_main['id']})

for class_label in COMMENT_CLASSES:
    print(f"クラス '{class_label}' の予測モデルを訓練中...")
    target_labels_for_training = train_dataframe_main[class_label]
    
    model = OneVsRestClassifier(
        LogisticRegression(solver='sag', n_jobs=-1, max_iter=1000, random_state=46)
    )
    model.fit(train_sparse_features, target_labels_for_training)
    
    submission_output_dataframe[class_label] = model.predict_proba(test_sparse_features)[:, 1]
    print(f"クラス '{class_label}' の訓練と予測が完了しました。")

print("全ての予測モデルの訓練と予測結果の生成が完了しました。")



# --- 5. 予測結果をCSV形式で保存 ---
submission_output_dataframe.to_csv('submission_Ogaki.csv', index=False)
print("\n--- 最終提出用CSVファイル 'submission_Ogaki.csv' が正常に作成されました！ ---")
print("このファイルをKaggleコンペティションに提出できます。")


