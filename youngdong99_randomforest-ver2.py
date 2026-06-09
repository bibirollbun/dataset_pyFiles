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

# Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


import warnings

# 모든 경고 메시지 무시
warnings.filterwarnings("ignore")


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
train.info()


test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
test.info()


# Personality 값별 개수
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
counts = train['Personality'].value_counts()
bars = counts.plot(kind='bar', color=['skyblue', 'salmon'])

# 막대 위에 값 표시
for i, v in enumerate(counts):
    plt.text(i, v + (max(counts)*0.01), str(v), ha='center', va='bottom', fontsize=10)

plt.title('Extrovert vs Introvert Count')
plt.xlabel('Personality')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# Personality별 데이터 분리
extrovert = train[train['Personality'] == 'Extrovert']
introvert = train[train['Personality'] == 'Introvert']

# 분석 대상 컬럼 (id, Personality 제외)
cols = [c for c in train.columns if c not in ['id', 'Personality']]

# 결측치 비율 계산
extro_missing_ratio = (extrovert[cols].isnull().sum() / len(extrovert)) * 100
intro_missing_ratio = (introvert[cols].isnull().sum() / len(introvert)) * 100

# DataFrame으로 결합
missing_ratio_df = pd.DataFrame({
    'Extrovert (%)': extro_missing_ratio,
    'Introvert (%)': intro_missing_ratio
})

# 시각화 : 각 변수들이 결측치일때 Personality 비율
plt.figure(figsize=(14, 6))
missing_ratio_df.plot(kind='bar', figsize=(14, 6), color=['skyblue', 'salmon'])
plt.title('Missing Value Ratio by Personality (%)')
plt.xlabel('Columns')
plt.ylabel('Missing Ratio (%)')
plt.legend(title='Personality')
plt.tight_layout()
plt.show()


# 필요한 컬럼만 선택하고 결측치 제거
subset = train[['Stage_fear', 'Drained_after_socializing', 'Personality']].dropna()

# 4가지 경우로 분류
def categorize(row):
    if row['Stage_fear'] == 'Yes' and row['Drained_after_socializing'] == 'Yes':
        return 'Yes-Yes'
    elif row['Stage_fear'] == 'No' and row['Drained_after_socializing'] == 'No':
        return 'No-No'
    elif row['Stage_fear'] == 'Yes' and row['Drained_after_socializing'] == 'No':
        return 'Yes-No'
    else:
        return 'No-Yes'

subset['Category'] = subset.apply(categorize, axis=1)

# 각 Category별 Personality 비율 계산
category_counts = subset.groupby(['Category', 'Personality']).size().unstack(fill_value=0)
category_ratio = category_counts.div(category_counts.sum(axis=1), axis=0) * 100  # 비율 %

# 그래프 시각화
category_ratio.plot(kind='bar', figsize=(10, 6), color=['skyblue', 'salmon'])
plt.title('Personality Probability by Stage_fear & Drained_after_socializing (%)')
plt.xlabel('Category (Stage_fear & Drained_after_socializing)')
plt.ylabel('Probability (%)')
plt.xticks(rotation=0)
plt.legend(title='Personality')
plt.tight_layout()
plt.show()

# 결과 출력
print("Counts:")
print(category_counts)
print("\nProbability (%):")
print(category_ratio)


# 필요한 컬럼만 선택하고 결측치 제거
subset = train[['Stage_fear', 'Drained_after_socializing', 'Personality']].dropna()

# 4가지 카테고리 생성
def categorize(row):
    if row['Stage_fear'] == 'Yes' and row['Drained_after_socializing'] == 'Yes':
        return 'Yes-Yes'
    elif row['Stage_fear'] == 'No' and row['Drained_after_socializing'] == 'No':
        return 'No-No'
    elif row['Stage_fear'] == 'Yes' and row['Drained_after_socializing'] == 'No':
        return 'Yes-No'
    else:
        return 'No-Yes'

subset['Category'] = subset.apply(categorize, axis=1)

# Personality별 전체 개수 (클래스 불균형 보정용)
total_counts = train['Personality'].value_counts()

# 카테고리별 Personality 개수
category_counts = subset.groupby(['Category', 'Personality']).size().unstack(fill_value=0)

# 전체 Personality 개수로 나눠서 정규화 (비율 계산)
adjusted_ratio = category_counts.copy()
for col in adjusted_ratio.columns:
    adjusted_ratio[col] = adjusted_ratio[col] / total_counts[col] * 100  # 퍼센트화

# 그래프 시각화
fig, ax = plt.subplots(figsize=(10, 6))
bars = adjusted_ratio.plot(kind='bar', color=['skyblue', 'salmon'], ax=ax)

plt.title('Adjusted Personality Ratio by Stage_fear & Drained_after_socializing (%)')
plt.xlabel('Category (Stage_fear & Drained_after_socializing)')
plt.ylabel('Adjusted Ratio (%)')
plt.xticks(rotation=0)
plt.legend(title='Personality')

# 막대 위 값 표시
for container in bars.containers:
    bars.bar_label(container, fmt='%.2f%%', fontsize=9, label_type='edge', padding=2)

# y축 범위 약간 확대 (최대값의 1.2배)
max_val = adjusted_ratio.max().max()
plt.ylim(0, max_val * 1.2)

plt.tight_layout()
plt.show()

# 결과 출력
print("Original counts (per category):")
print(category_counts)
print("\nAdjusted ratio (class imbalance corrected):")
print(adjusted_ratio)


import pandas as pd
import matplotlib.pyplot as plt

# 전체 인원 수 (정규화 보정용)
extrovert_total = train[train['Personality'] == 'Extrovert'].shape[0]
introvert_total = train[train['Personality'] == 'Introvert'].shape[0]
extrovert_factor = 1 / extrovert_total
introvert_factor = 1 / introvert_total

# 결측치 조합 정의
nan_conditions = {
    'NaN-Yes': (train['Stage_fear'].isna()) & (train['Drained_after_socializing'] == 'Yes'),
    'NaN-No':  (train['Stage_fear'].isna()) & (train['Drained_after_socializing'] == 'No'),
    'Yes-NaN': (train['Stage_fear'] == 'Yes') & (train['Drained_after_socializing'].isna()),
    'No-NaN':  (train['Stage_fear'] == 'No') & (train['Drained_after_socializing'].isna()),
    'NaN-NaN': (train['Stage_fear'].isna()) & (train['Drained_after_socializing'].isna())
}

# 결과 저장용 데이터프레임 생성
result = pd.DataFrame(columns=['Extrovert', 'Introvert'])

for label, condition in nan_conditions.items():
    subset = train[condition]
    extro = (subset['Personality'] == 'Extrovert').sum() * extrovert_factor
    intro = (subset['Personality'] == 'Introvert').sum() * introvert_factor
    result.loc[label] = [extro, intro]

# 비율 계산
normalized_ratio = result.div(result.sum(axis=1), axis=0) * 100  # 퍼센트로 변환

# 시각화
fig, ax = plt.subplots(figsize=(10, 6))
bars = normalized_ratio.plot(kind='bar', stacked=False, ax=ax, color=['skyblue', 'salmon'])

plt.title('Adjusted Personality Ratio for NaN Combinations (%)')
plt.xlabel('Missing Data Combinations (stage_fear - drained_after_socializing)')
plt.ylabel('Adjusted Ratio (%)')
plt.xticks(rotation=0)
plt.legend(title='Personality')

# 막대 위에 비율 값 표시
for container in bars.containers:
    bars.bar_label(container, fmt='%.2f%%', fontsize=9, label_type='edge', padding=2)

# y축 범위 설정
max_val = normalized_ratio.max().max()
plt.ylim(0, max_val * 1.2)

plt.tight_layout()
plt.show()


# 결측치 처리 함수
def fill_missing_values(df):
    # 수치형 변수 처리
    df['Time_spent_Alone'] = df['Time_spent_Alone'].fillna(df['Time_spent_Alone'].median())
    
    numeric_cols = ['Social_event_attendance', 'Going_outside', 
                    'Friends_circle_size', 'Post_frequency']
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    # 범주형 변수 처리
    stage_col = 'Stage_fear'
    drain_col = 'Drained_after_socializing'

    for i in df.index:
        stage = df.at[i, stage_col]
        drain = df.at[i, drain_col]

        if pd.isna(stage) and pd.isna(drain):
            df.at[i, stage_col] = 'Yes'
            df.at[i, drain_col] = 'Yes'
        elif pd.isna(stage) and pd.notna(drain):
            df.at[i, stage_col] = drain
        elif pd.notna(stage) and pd.isna(drain):
            df.at[i, drain_col] = stage

    return df


# 함수 적용
train = fill_missing_values(train)
test = fill_missing_values(test)


# 범주형 변수 변환
for df in [train, test]:
    df['Stage_fear'] = df['Stage_fear'].map({'No': 0, 'Yes': 1})
    df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'No': 0, 'Yes': 1})


# 라벨 변환
train['Personality'] = train['Personality'].map({'Introvert': 0, 'Extrovert': 1})


# 데이터 분리
from sklearn.model_selection import train_test_split
X = train.drop(columns=['id', 'Personality'])
y = train['Personality']


# 랜덤 포레스트 모델 학습
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y)

# 랜덤 포레스트 정확도 확인 결과 : 0.9633


# 테스트 데이터 예측
X_test = test.drop(columns=['id'])
preds = rf_model.predict(X_test)
preds_labels = ['Extrovert' if x == 1 else 'Introvert' for x in preds]


# 제출 파일 생성
submission = pd.DataFrame({'id': test['id'], 'Personality': preds_labels})
submission.to_csv('/kaggle/working/submission.csv', index=False)

