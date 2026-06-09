import pandas as pd
import os

# 正しいフォルダ名に更新
input_dir = '/kaggle/input/abcdefgh'

# sample_submission.csv を読み込んで row_id を取得
sample_df = pd.read_csv(f'/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv')


# 予測用 DataFrame を作成（row_id をそのまま使用）
submission = sample_df[['row_id']].copy()

# 各行に対して最大３つのダミー予測（空白区切り）
submission['Category:Misconception'] = "True_Correct:NA False_Neither:NA False_Misconception:Incomplete"

# 提出用CSVを出力
# submission.to_csv('/kaggle/working/submission.csv', index=False)

# 確認
print(submission.head())
print(f"出力行数: {len(submission)}, sample_submissionの行数: {len(sample_df)}")



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(f'/kaggle/input/map-charting-student-math-misunderstandings/train.csv')

df


df['Category'].value_counts()


df['Misconception'].value_counts()


# データ読み込み
df = pd.read_csv(f'/kaggle/input/map-charting-student-math-misunderstandings/train.csv')

# ① True% の行だけ抽出
true_df = df[df["Category"].str.startswith("True")]

# ② QuestionId と MC_Answer の組み合わせを取り出し、重複を除く
unique_correct_pairs = true_df[["QuestionId", "MC_Answer"]].drop_duplicates()

# 結果確認
print(unique_correct_pairs)


import pandas as pd

# 正解マッピング（trainで使ったものを使い回す）
correct_map = {
    31772: "\( \frac{1}{3} \)",
    31774: "\( \frac{1}{12} \)",
    31777: "\( 72 \)",
    31778: "\( 6 \)",
    32829: "\( 12 \)",
    32833: "\( 3 \frac{1}{3} \)",
    32835: "\( 6.2 \)",
    33471: "\( 15 \)",
    33472: "\( \frac{11}{15} \)",
    33474: "\( \frac{1}{3} \times \frac{2}{3} \)",
    76870: "\( 10 \)",
    89443: "\( -3 \)",
    91695: "\( 26 \)",
    104665: "\( 48 \) hours",
    109465: "Likely"

    
    # 必要に応じて追加
}

# ① test.csv 読み込み
test_df = pd.read_csv(f'/kaggle/input/map-charting-student-math-misunderstandings/test.csv')

# ② 正誤判定（ルールベース）
test_df["is_correct"] = test_df.apply(
    lambda row: row["MC_Answer"] == correct_map.get(row["QuestionId"], ""),
    axis=1
)

# ③ 提出ラベルの作成
test_df["Category:Misconception"] = test_df["is_correct"].map({
    True: "True_Correct:NA",
    False: "False_Neither:NA"
})

# ④ 提出用CSV作成
submission = test_df[["row_id", "Category:Misconception"]]
submission.to_csv("submission.csv", index=False)

submission

