import pandas as pd
import numpy as np
import re
from textblob import TextBlob  # 感情分析用
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# 1. データ読み込み
print("データを読み込んでいます...")
train_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")
test_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")
sub = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip")

# 欠損値埋め
train_df['comment_text'] = train_df['comment_text'].fillna("")
test_df['comment_text'] = test_df['comment_text'].fillna("")

# 2. 強化版：統計的特徴量の作成
def add_features(df):
    print("基本統計量を作成中...")
    # --- A. 基本セット（Andre流） ---
    df['total_length'] = df['comment_text'].apply(len)
    df['capitals'] = df['comment_text'].apply(lambda x: sum(1 for c in x if c.isupper()))
    df['caps_vs_length'] = df['capitals'] / df['total_length']
    df['num_exclamation_marks'] = df['comment_text'].apply(lambda x: x.count('!'))
    df['num_question_marks'] = df['comment_text'].apply(lambda x: x.count('?'))
    df['num_punctuation'] = df['comment_text'].apply(lambda x: sum(x.count(w) for w in '.,;:'))
    df['num_symbols'] = df['comment_text'].apply(lambda x: sum(x.count(w) for w in '*&$%'))
    df['num_words'] = df['comment_text'].apply(lambda x: len(str(x).split()))
    df['num_unique_words'] = df['comment_text'].apply(lambda x: len(set(str(x).split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
    
    # --- B. 感情分析（Sentiment Analysis）追加 ---
    print("感情スコアを計算中...")
    # polarity: -1.0(ネガティブ) 〜 1.0(ポジティブ)
    df['polarity'] = df['comment_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    
    # --- C. 攻撃ワードカウント（Bad Words）追加 ---
    print("NGワードをカウント中...")
    # 代表的な攻撃的単語リスト
    bad_words = ['fuck', 'shit', 'suck', 'stupid', 'idiot', 'bitch', 'ass', 'shut', 'kill', 'hate']
    
    for word in bad_words:
        # \bをつけることで "class" の中の "ass" を数えないようにする
        df[f'count_{word}'] = df['comment_text'].apply(lambda x: len(re.findall(rf'\b{word}\b', str(x).lower())))
    
    # NGワードの合計数
    df['total_bad_words'] = df[[f'count_{w}' for w in bad_words]].sum(axis=1)

    # --- D. 人称代名詞（You vs I）追加 ---
    # 相手を責める(You)か、自分を語る(I)か
    df['count_you'] = df['comment_text'].apply(lambda x: len(re.findall(r'\byou\b', str(x).lower())))
    df['count_i'] = df['comment_text'].apply(lambda x: len(re.findall(r'\bi\b', str(x).lower())))
    
    return df

print("Trainデータの特徴量作成...")
train_df = add_features(train_df)
print("Testデータの特徴量作成...")
test_df = add_features(test_df)

# 学習に使う特徴量のリスト（新しく作ったカラムを追加）
bad_words_cols = ['count_fuck', 'count_shit', 'count_suck', 'count_stupid', 'count_idiot', 
                  'count_bitch', 'count_ass', 'count_shut', 'count_kill', 'count_hate', 'total_bad_words']

features = [
    'total_length', 'capitals', 'caps_vs_length', 'num_exclamation_marks',
    'num_question_marks', 'num_punctuation', 'num_symbols', 
    'num_words', 'num_unique_words', 'words_vs_unique',
    'polarity', 'count_you', 'count_i'
] + bad_words_cols

# ターゲット列
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# 3. LightGBMによる学習と予測
preds = np.zeros((len(test_df), len(label_cols)))

# パラメータ設定
params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbose": -1
}

X = train_df[features]
X_test = test_df[features]

print(f"\n学習を開始します... (特徴量数: {len(features)})")

for i, col in enumerate(label_cols):
    y = train_df[col]
    
    # 訓練・検証分割
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    d_train = lgb.Dataset(X_train, label=y_train)
    d_val = lgb.Dataset(X_val, label=y_val)
    
    # 学習
    model = lgb.train(
        params,
        d_train,
        num_boost_round=1000,
        valid_sets=[d_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0) # ログを静かにする
        ]
    )
    
    # 予測
    preds[:, i] = model.predict(X_test)
    print(f"Done: {col} (Best Iteration: {model.best_iteration})")

# 4. 提出
print("提出ファイルを作成中...")
sub[label_cols] = preds
sub.to_csv('submission_andre_features_enhanced.csv', index=False)
print("完了！")




