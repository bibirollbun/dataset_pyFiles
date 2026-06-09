














data = {'QuestionId': 101,
 'ConstructId': 579,
 'ConstructName': 'Express one quantity as a percentage of another mentally',
 'SubjectId': 233,
 'SubjectName': 'Percentages of an Amount',
 'CorrectAnswer': 'B',
 'QuestionText': 'What is \\( 8 \\) out of \\( 40 \\) as a percentage?',
 'AnswerAText': '\\( 8.4 \\% \\)',
 'AnswerBText': '\\( 20 \\% \\)',
 'AnswerCText': '\\( 16 \\% \\)',
 'AnswerDText': '\\( 24 \\% \\)',
 'MisconceptionAId': 1786,
 'MisconceptionBId': -1,
 'MisconceptionCId': 658,
 'MisconceptionDId': -1,
 'MisconceptionAName': 'Converts a fraction to a percentage by writing the numerator followed by the denominator',
 'MisconceptionBName': None,
 'MisconceptionCName': 'Thinks they double the numerator to turn a fraction into a percentage',
 'MisconceptionDName': None}

# 演習データの準備
question_text = data['QuestionText']

# retrievalで本来得られる候補だが、簡単のために事前に定義しておく
misconceptions_candidates = [
    data['MisconceptionAName'],
    data['MisconceptionCName'],
    "Does not know that angles in a triangle sum to 180 degrees",
    "Uses dividing fractions method for multiplying fractions"
]

print("--- Step 1: 疑似 Retrieval 結果 (Re-ranking 対象候補) ---")
print(f"質問: {question_text}")
print("候補 Misconception:")
for i, mc in enumerate(misconceptions_candidates):
    print(f"  {i+1}. {mc}")
print("-" * 30)

















import pandas as pd
import numpy as np

df = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv')

# データ数を取得
data_num = len(df)

# train と test の分割比率を指定
train_rate = 0.8
train_num = int(data_num * train_rate)

# index を分割
np.random.seed(42)
train_index = np.random.choice(data_num, train_num, replace=False)
valid_index = list(set(range(data_num)) - set(train_index))

# train と test に分割
train_df = df.iloc[train_index]
valid_df = df.iloc[valid_index]

print("train_df の形状:", train_df.shape)
print("valid_df の形状:", valid_df.shape)


def preprocess(df):    
    result = []
    for i, row in df.iterrows():
        for option in ['A', 'B', 'C', 'D']:
            if pd.isnull(row[f'Misconception{option}Id']):
                continue
            result.append(
                {
                    'ConstructId': row['ConstructId'],
                    'ConstructName': row['ConstructName'],
                    'SubjectId': row['SubjectId'],
                    'SubjectName': row['SubjectName'],
                    'CorrectAnswer': row['CorrectAnswer'],
                    'IsCorrect': row['CorrectAnswer']==option,
                    'Option': option,
                    'AnswerText': row[f'Answer{option}Text'],
                    'MisconceptionId': int(row[f'Misconception{option}Id']),
                }
            )
    df = pd.DataFrame(result)

    misconception_mapping_df = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv')
    df = df.merge(
        misconception_mapping_df,
        on="MisconceptionId",
        how="left"
    )

    return df

train_df = preprocess(train_df)
valid_df = preprocess(valid_df)

print("train_df の形状:", train_df.shape)
display(train_df.head())


print("valid_df の形状:", valid_df.shape)
display(train_df.head())





training_data = [    
    # --- 分数 ---
    {
        'QuestionText': 'Calculate \\( \\frac{1}{2} + \\frac{1}{3} \\)',
        'AnswerText': '\\( \\frac{2}{5} \\)', # 分子同士、分母同士を足す間違い
        'MisconceptionName': 'Adds fractions by adding numerators and denominators separately',
        'IsCorrect': True
    },
    {
        'QuestionText': 'Calculate \\( \\frac{1}{2} + \\frac{1}{3} \\)',
        'AnswerText': '\\( \\frac{1}{5} \\)', # 何か別の計算間違い
        'MisconceptionName': 'Adds fractions by adding numerators and denominators separately',
        'IsCorrect': False # 上記の特定のMisconceptionとは異なる間違い
    },
    # --- 小数 ---
    {
        'QuestionText': 'What is \\( 0.5 \\times 0.2 \\)?',
        'AnswerText': '\\( 1.0 \\)', # 小数点の位置間違い
        'MisconceptionName': 'Misplaces the decimal point in multiplication of decimals',
        'IsCorrect': True
    },
    {
        'QuestionText': 'What is \\( 0.5 \\times 0.2 \\)?',
        'AnswerText': '\\( 0.7 \\)', # 掛け算を足し算と間違える
        'MisconceptionName': 'Confuses multiplication operation with addition',
        'IsCorrect': True
     },
    # --- 簡単な方程式 ---
    {
        'QuestionText': 'Solve for x: \\( 2x + 3 = 11 \\)',
        'AnswerText': '\\( x = 7 \\)', # 移項時に符号を変え忘れる (11+3)/2
        'MisconceptionName': 'Forgets to change the sign of a term when moving it across the equals sign',
        'IsCorrect': True
    },
    {
        'QuestionText': 'Solve for x: \\( 2x + 3 = 11 \\)',
        'AnswerText': '\\( x = 5.5 \\)', # 最初に2で割ってしまう 3を引くのを忘れる
        'MisconceptionName': 'Performs operations in the wrong order when solving equations',
        'IsCorrect': True # 操作順序の間違い
    },
    # --- 割合 ---
    {
        'QuestionText': 'What is 20% off a 500 yen item?',
        'AnswerText': '\\( 100 \\text{ yen} \\)', # 割引額を答えてしまう
        'MisconceptionName': 'Calculates the discount amount instead of the final price after discount',
        'IsCorrect': True
    },
    {
        'QuestionText': 'What is 20% off a 500 yen item?',
        'AnswerText': '\\( 480 \\text{ yen} \\)', # 20%を20円と勘違いして引く
        'MisconceptionName': 'Subtracts the percentage value directly as a fixed amount',
        'IsCorrect': True
    },
]

# {変数名} はあとで上書きすることができます
prompt_template = """Determine if the provided misconception is relevant to the student's answer for the given question. Respond with only 'Yes' or 'No'.

Question: {QuestionText}
Student's Answer: {AnswerText}
Misconception Candidate: {MisconceptionName}

Is the misconception relevant? (Yes/No)
Answer:{Answer}"""


# 必要なライブラリのインストール
!pip install trl -q




