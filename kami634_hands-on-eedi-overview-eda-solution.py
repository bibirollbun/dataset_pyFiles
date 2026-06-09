import pandas as pd

misconception_mapping_df = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv')
train_df = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv')
test_df = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv')
sample_submission_df = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/sample_submission.csv')


display(misconception_mapping_df.head())
display(train_df.head())
display(test_df.head())
display(sample_submission_df.head())


print(misconception_mapping_df.shape)
print(train_df.shape)
print(test_df.shape)
print(sample_submission_df.shape)


misconception_mapping_df[misconception_mapping_df['MisconceptionId']==1672].iloc[0]['MisconceptionName']


# 欠損値を-1.0で埋めて小数を整数に変換
letters = ["A", "B", "C", "D"]  # A, B, C, Dをループで処理するためのリスト
for letter in letters:
    col = f"Misconception{letter}Id"
    train_df[col] = train_df[col].fillna(-1.0).astype(int)

# misconception_mappingのMisconceptionIdを整数化
misconception_mapping_df["MisconceptionId"] = misconception_mapping_df["MisconceptionId"].astype(int)

# MisconceptionIdに対応するMisconceptionNameを結合
for letter in letters:
    id_col = f"Misconception{letter}Id"  # MisconceptionAId, MisconceptionBId, etc.
    name_col = f"Misconception{letter}Name"  # MisconceptionAName, MisconceptionBName, etc.

    train_df = train_df.merge(
        misconception_mapping_df,
        left_on=id_col,
        right_on="MisconceptionId",
        how="left"
    ).rename(columns={"MisconceptionName": name_col}).drop(columns=["MisconceptionId"])

train_df.head()


train_df.iloc[101].to_dict()


import numpy as np
import pandas as pd

sample_submission_df = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/sample_submission.csv")


def mapk(targets, predictions, k=25):
    """
    Compute Mean Average Precision at K (MAP@K).
    
    Parameters:
    - targets: List of ground truth values (correct misconceptions for each QuestionId_Answer).
    - predictions: List of predicted ranked lists for each QuestionId_Answer.
    - k: An integer, the cut-off for precision evaluation.
    
    Returns:
    - Mean Average Precision at K (MAP@K): Float.
    """
    def apk(actual, predicted, k):
        """
        Compute average precision at K for a single example.
        """
        if len(predicted) > k:
            predicted = predicted[:k]
        
        score = 0.0
        num_hits = 0.0
        
        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:  # Check if p is relevant and not duplicate
                num_hits += 1.0
                score += num_hits / (i + 1.0)  # Precision at i+1 (1-based index)
        
        return score / min(len(actual), k) if actual else 0.0
    
    return np.mean([apk(a, p, k) for a, p in zip(targets, predictions)])


predictions = [list(map(int, preds.split())) for preds in sample_submission_df["MisconceptionId"]]

# 仮定したターゲットラベルごとにMAP@25を計算
for target_value in [1, 2, 10, 25, 100]:
    targets = [[target_value] for _ in range(len(sample_submission_df))]
    map_at_25 = mapk(targets, predictions, k=25)
    print(f"MAP@25 when all targets are {target_value}: {map_at_25}")


import matplotlib.pyplot as plt

def calculate_ap(relevant_rank):
    """正解ランクからAPスコアを計算します。"""
    if relevant_rank > 25:
        return 0.0  # ランクが25を超えた場合はAPスコアは0
    
    precision_at_rank = 1.0 / relevant_rank
    return precision_at_rank

# ランクの範囲でAPスコアを計算
ranks = range(1, 26)
ap_scores = [calculate_ap(rank) for rank in ranks]

# プロット
plt.plot(ranks, ap_scores, marker='o')
# plt.title('AP Score vs. Relevant Rank (Top 25)')
plt.xlabel('Rank')
plt.ylabel('AP Score')
plt.grid(True)
plt.show()


data = {
 'QuestionId': 103,
 'ConstructId': 2566,
 'ConstructName': 'Calculate the area of a parallelogram where the dimensions are given in the same units',
 'SubjectId': 75,
 'SubjectName': 'Area of Simple Shapes',
 'CorrectAnswer': 'C',
 'QuestionText': 'What is the area of the parallelogram? ![A parallelogram with the length labelled 10cm, the slanted height labelled 4cm, and the perpendicular height (marked with a right angle) labelled 3cm]()',
 'IncorrectAnswerText': '\\( 40 \\mathrm{~cm}^{2} \\)',
}


from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

top_k = 20

misconception_mapping_df = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv')
misconception_texts = misconception_mapping_df['MisconceptionName']

vectorizer = TfidfVectorizer(
    ngram_range=(1, 1),  # 1‑gram
    max_df=0.4,
)

vectorizer.fit(misconception_texts)
corpus_vecs = vectorizer.transform(misconception_texts)
query_vec = vectorizer.transform([data['QuestionText']])
sims = cosine_similarity(corpus_vecs, query_vec).ravel()

# 類似度上位 K 件のインデックスを取得
top_idx = np.argsort(sims)[::-1][:top_k]
[(int(i), sims[i], misconception_texts[i]) for i in top_idx]

