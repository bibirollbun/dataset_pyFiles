import pandas as pd
import os

# ==========================================
# これまでの結果をインポートする
# ==========================================
# まずは右側のInput欄から、必要なデータを選択する
# 右側のInput欄を見て、それぞれのノートブックのフォルダ名をコピーしてください

# 1. BARTのファイルパス
path_bart = "/kaggle/input/toxic-torii-v5-bart-ver1-result/submission.csv"

# 2. TF-IDFのファイルパス
path_tfidf = "/kaggle/input/toxic-torii-v3-tfidf-ver1-result/submission.csv"

# 3. GRUのファイルパス
path_gru = "/kaggle/input/toxic-torii-v8-gru-ver1-result/submission_gru.csv"

# 4. LSTMのファイルパス
path_lstm = "/kaggle/input/toxic-torii-v9-lstm-ver1-result/submission_lstm.csv"

# 5. FetureEngeeringのファイルパス
path_feature = "/kaggle/input/toxic-torii-v7-featureengineering2-ver1-result/submission_andre_features_enhanced.csv"

# ==========================================
# 読み込みとブレンド実行
# ==========================================
print("ファイルを読み込んでいます...")

# 読み込み関数（ファイルがない場合のエラー回避付き）
def load_submission(path, name):
    if os.path.exists(path):
        print(f" - {name}: 読み込み成功 ({path})")
        return pd.read_csv(path)
    else:
        print(f" - {name}: ファイルが見つかりません。パスを確認してください: {path}")
        return None

df_bart = load_submission(path_bart, "BART")
df_tfidf = load_submission(path_tfidf, "TF-IDF")
df_gru = load_submission(path_gru, "GRU")
df_lstm = load_submission(path_lstm, "LSTM")
df_feature = load_submission(path_feature, "FeatureEngineering")

# ターゲット列
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
submission_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip")

# ブレンド計算 (ファイルが揃っている前提)
if df_bart is not None and df_tfidf is not None and df_gru is not None and df_lstm is not None and df_feature is not None:
    print("\n★5つのモデルをブレンドします")
    
    # シンプルな加重平均
    blend_preds = (df_bart[label_cols] * 0.55) + \
                  (df_tfidf[label_cols] * 0.20) + \
                  (df_gru[label_cols] * 0.10) + \
                  (df_lstm[label_cols] * 0.10) + \
                  (df_feature[label_cols] * 0.05)
    
    submission_df[label_cols] = blend_preds
    submission_df.to_csv('submission_final_ensemble.csv', index=False)
    print("完了！ 'submission_final_ensemble.csv' を作成しました。")
    
else:
    print("\nエラー: 全てのファイルが正しく読み込めませんでした。パスを修正してください。")

# 中身確認
print(submission_df.head())




