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

#データの読み込みと解凍
#Kaggle環境に提供されているZIP形式のデータセットを読み込み、解凍してメモリに格納する
print("Loading and extracting data...")

# データセットが配置されているインプットディレクトリのベースパスを定義
data_base_path = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/'
# 解凍されたファイルを保存するディレクトリを指定
extract_path = '/kaggle/working/'

# ZIPファイルを指定されたパスに解凍するためのヘルパー関数を定義
def extract_zip_file(zip_file_path, extract_to_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref: 
        zip_ref.extractall(extract_to_path) 
    print(f"Extracted {zip_file_path.split('/')[-1]} to {extract_to_path}") 

try:
    extract_zip_file(os.path.join(data_base_path, 'train.csv.zip'), extract_path) # 訓練データZIPファイルを解凍
    extract_zip_file(os.path.join(data_base_path, 'test.csv.zip'), extract_path) # テストデータZIPファイルを解凍
    extract_zip_file(os.path.join(data_base_path, 'sample_submission.csv.zip'), extract_path) 
    # extract_zip_file(os.path.join(data_base_path, 'test_labels.csv.zip'), extract_path) 

    train_df = pd.read_csv(os.path.join(extract_path, 'train.csv')) # 解凍された訓練CSVファイルをDataFrameとして読み込み
    test_df = pd.read_csv(os.path.join(extract_path, 'test.csv')) # 解凍されたテストCSVファイルをDataFrameとして読み込み
    sample_submission = pd.read_csv(os.path.join(extract_path, 'sample_submission.csv')) # 解凍されたサンプル提出CSVファイルをDataFrameとして読み込み

    print("Data loaded successfully after extraction.")

except FileNotFoundError as e: # 指定されたファイルが見つからなかった場合に発生するエラーをキャッチ
    print(f"Error loading files: {e}") # エラーメッセージを出力
    print(">>> Please ensure the dataset is added to your Kaggle Notebook and the path is correct. <<<") 
    exit() # エラーが発生した場合はスクリプトの実行を中断

except Exception as e: # その他の予期せぬエラーをキャッチ
    print(f"An unexpected error occurred during data loading or extraction: {e}") # 予期せぬエラーメッセージを出力
    exit() # 予期せぬエラーが発生した場合もスクリプトの実行を中断

if train_df is None or test_df is None or sample_submission is None: # データが正常に読み込まれた場合にのみ、後続の処理を実行するための条件分岐
    print("Data loading failed. Exiting script.") # データロードが失敗した場合にメッセージを出力
else:
    CATEGORIES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'] # コメントが分類される毒性カテゴリのリストを定義

    # テキスト前処理関数
    # コメントテキストから不要な文字やノイズを除去し、テキストを標準化
    def clean_text(text):
        if not isinstance(text, str): return "" # 入力が文字列型でない場合に空文字列を返す
        text = text.lower() # 全ての文字を小文字に変換
        text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) # 句読点を正規表現で除去
        text = re.sub(r'\d+', '', text) # 文字列中の全ての数字を空文字列に置換し、除去
        text = re.sub(r'\s+', ' ', text).strip() # 複数の空白文字を単一のスペースに変換し、前後の空白を除去
        return text

    print("Applying text cleaning to comments...") 
    train_df['comment_text'] = train_df['comment_text'].apply(clean_text) # 訓練データのコメントテキストにclean_text関数を適用
    test_df['comment_text'] = test_df['comment_text'].apply(clean_text) # テストデータのコメントテキストにclean_text関数を適用
    print("Text cleaning complete.") 

    #特徴量エンジニアリング
    #クリーンアップされたコメントテキストデータを、機械学習モデルが学習できる数値ベクトル表現に変換
    print("Concatenating all comments for TF-IDF vectorization...") 
    all_comments = pd.concat([train_df['comment_text'], test_df['comment_text']], axis=0) # 訓練データとテストデータ両方のコメントテキスト列を結合

    print("Fitting TF-IDF Vectorizer...") 
    vectorizer = TfidfVectorizer(min_df=3, max_df=0.9, ngram_range=(1, 2), stop_words='english', max_features=50000) # TfidfVectorizerをパラメータ設定して初期化
    X_all = vectorizer.fit_transform(all_comments) 

    X_train = X_all[:len(train_df)] # 結合された特徴量行列を訓練データの部分に分割
    X_test = X_all[len(train_df):] # 結合された特徴量行列をテストデータの部分に分割
    print(f"TF-IDF Vectorization complete. Number of features: {X_train.shape[1]}")

    del all_comments
    gc.collect() 

    # モデル構築と学習
    # 各毒性カテゴリに対して個別の二項分類器を訓練し、テストデータに対する予測を生成
    classifier = OneVsRestClassifier(LogisticRegression(solver='sag', n_jobs=-1, max_iter=1000, random_state=42)) # OneVsRestClassifierとロジスティック回帰モデルを初期化

    print("Training models for each category...") 
    predictions = pd.DataFrame({'id': test_df['id']}) # 予測結果を格納するためのDataFrameを初期化

    for category in CATEGORIES: 
        print(f"  Training model for category: {category}...") 
        y_train_category = train_df[category] # 現在のカテゴリに対応する訓練データのターゲット変数を抽出
        classifier.fit(X_train, y_train_category) # 分類器を訓練データで学習
        predictions[category] = classifier.predict_proba(X_test)[:, 1] # テストデータに対する予測確率を計算し、結果をDataFrameに格納
        print(f"  Finished training for {category}.")

    print("All models trained and predictions generated.") 

    #予測結果の保存
    #Kaggleコンペティションに提出するためのCSVファイルを生成
    predictions.to_csv('result.csv', index=False) # 予測結果をCSVファイルとして保存
    print("\n--- result file 'result.csv' created ---") 

