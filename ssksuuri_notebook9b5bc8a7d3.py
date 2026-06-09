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


# 入出力パス設定（Kaggle専用）
input_data_dir = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/'
working_output_dir = '/kaggle/working/'

# ZIP展開用の関数
def extract_zip_content(zip_filepath_in, extract_path_out):
    with zipfile.ZipFile(zip_filepath_in, 'r') as zip_object:
        zip_object.extractall(extract_path_out)



# 変数初期化
df_train_comments = None
df_test_comments = None
df_submission_template = None

data_loaded = True  # フラグで制御

# ZIP展開とCSV読み込み
try:
    extract_zip_content(os.path.join(input_data_dir, 'train.csv.zip'), working_output_dir)
    df_train_comments = pd.read_csv(os.path.join(working_output_dir, 'train.csv'))
    
    extract_zip_content(os.path.join(input_data_dir, 'test.csv.zip'), working_output_dir)
    df_test_comments = pd.read_csv(os.path.join(working_output_dir, 'test.csv'))
    
    extract_zip_content(os.path.join(input_data_dir, 'sample_submission.csv.zip'), working_output_dir)
    df_submission_template = pd.read_csv(os.path.join(working_output_dir, 'sample_submission.csv'))


except FileNotFoundError as err_nf:
    
    data_loaded = False
except Exception as general_err_loading:
    
    data_loaded = False


# データが正常に読み込めた場合のみ処理を継続
if not data_loaded or df_train_comments is None or df_test_comments is None or df_submission_template is None:
    print("データロード処理に失敗")
else:
    COMMENT_CATEGORIES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # コメントテキストの前処理関数
    def preprocess_comment_string(input_str):
        if not isinstance(input_str, str):
            return ""
        preprocessed_str = input_str.lower()
        preprocessed_str = re.sub(f'[{re.escape(string.punctuation)}]', '', preprocessed_str)
        preprocessed_str = re.sub(r'\d+', '', preprocessed_str)
        preprocessed_str = re.sub(r'\s+', ' ', preprocessed_str).strip()
        return preprocessed_str

    
    df_train_comments['comment_text'] = df_train_comments['comment_text'].apply(preprocess_comment_string)
    df_test_comments['comment_text'] = df_test_comments['comment_text'].apply(preprocess_comment_string)
 




    # TF-IDFベクトル化
    full_comment_collection = pd.concat([df_train_comments['comment_text'], df_test_comments['comment_text']], axis=0)

    tfidf_transformer = TfidfVectorizer(
        min_df=5,
        max_df=0.85,
        ngram_range=(1, 2),
        stop_words='english',
        max_features=50000
    )
    transformed_feature_vectors = tfidf_transformer.fit_transform(full_comment_collection)

    train_feature_vectors = transformed_feature_vectors[:len(df_train_comments)]
    test_feature_vectors = transformed_feature_vectors[len(df_train_comments):]
    

    del full_comment_collection, transformed_feature_vectors
    gc.collect()


    # モデルの定義（One-vs-Rest + Logistic Regression）
    multi_label_classifier_model = OneVsRestClassifier(
        LogisticRegression(solver='sag', n_jobs=-1, max_iter=1000, random_state=45)
    )

    output_submission_df = pd.DataFrame({'id': df_test_comments['id']})

    for category_name in COMMENT_CATEGORIES:
        current_target_series = df_train_comments[category_name]
        multi_label_classifier_model.fit(train_feature_vectors, current_target_series)
        output_submission_df[category_name] = multi_label_classifier_model.predict_proba(test_feature_vectors)[:, 1]
     




    # 提出用ファイルの保存
    output_submission_df.to_csv('submission.csv', index=False)
    print("\n--- 提出ファイル 'submission.csv' の作成 ---")

