import pandas as pd

data_path = '/kaggle/input/porto-seguro-safe-driver-prediction/'

train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(data_path + 'sample_submission.csv', index_col='id')


train.shape, test.shape, submission.shape


train.head()


test.head()


submission.head()


train.info()


# -1 값의 총 개수
num_missing = (train == -1).sum().sum()
print(f"train내의 -1 값의 총 개수: {num_missing}")

num_missing = (test == -1).sum().sum()
print(f"test내의 -1 값의 총 개수: {num_missing}")


import numpy as np
import missingno as msno

# 훈련데이터 사본을 만들어서 -1을 결측치 값으로 변환
train_copy = train.copy().replace(-1, np.NaN)

msno.bar(df=train_copy.iloc[:, 1:29], figsize=(13, 6))


msno.bar(df=train_copy.iloc[:, 29:], figsize=(13, 6))


msno.matrix(df=train_copy.iloc[:, 1:29], figsize=(13, 6))


msno.matrix(df=train_copy.iloc[:, 29:], figsize=(13, 6))


def resumetable(df):
    print(f'데이터 세트 형상: {df.shape}')
    summary = pd.DataFrame(df.dtypes, columns=['데이터 타입'])
    summary['결측값 개수'] = (df == -1).sum().values  # 피처별 -1 개수
    summary['고윳값 개수'] = df.nunique().values
    summary['데이터 종류'] = None
    for col in df.columns:
        if 'bin' in col or col == 'target':
            summary.loc[col, '데이터 종류'] = '이진형'
        elif 'cat' in col:
            summary.loc[col, '데이터 종류'] = '명목형'
        elif df[col].dtype == float:
            summary.loc[col, '데이터 종류'] = '연속형'
        elif df[col].dtype == int:
            summary.loc[col, '데이터 종류'] = '순서형'
    
    return summary


summary = resumetable(train)
summary


# 명목형만 추출
summary[summary['데이터 종류'] == '명목형'].index


# float64만 추출
summary[summary['데이터 타입'] == 'float64'].index


import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline


def write_percent(ax, total_size):
    '''도형 객체를 순회하며 막대 그래프 상단에 타깃값 비율 표시'''
    for patch in ax.patches:
        height = patch.get_height()      # 도형 높이(데이터 갯수)
        width = patch.get_width()        # 도형 너비
        left_coord = patch.get_x()       # 도형 왼쪽 테두리의 x축 위치
        percent = height/total_size*100  # 타깃값 비율

        # (x, y) 좌표에 텍스트 입력
        ax.text(left_coord + width/2.0,     # x축 위치
                height + total_size*0.001,  # y축 위치
                '{:1.1f}%'.format(percent), # 입력 텍스트
                ha='center')                # 가운데 정렬

mpl.rc('font', size=15)
plt.figure(figsize=(7, 6))

ax = sns.countplot(x='target', data=train)
write_percent(ax, len(train))                # 비율 표시
ax.set_title('Target Distribution')


import matplotlib.gridspec as gridspec

def plot_target_ratio_by_features(df, features, num_rows, num_cols, size=(12, 18)):
    mpl.rc('font', size=9)
    plt.figure(figsize=size)
    grid = gridspec.GridSpec(num_rows, num_cols)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    for idx, feature in enumerate(features):
        ax = plt.subplot(grid[idx])
        # ax축에 고윳값별 타깃값 1 비율을 막대그래프로 그리기
        sns.barplot(x=feature, y='target', data=df, palette='Set2', ax=ax)


bin_features = summary[summary['데이터 종류'] == '이진형'].index

plot_target_ratio_by_features(train, bin_features, 6, 3)


nom_features = summary[summary['데이터 종류'] == '명목형'].index

plot_target_ratio_by_features(train, nom_features, 7, 2)


import math

ord_features = summary[summary['데이터 종류'] == '순서형'].index

num_plots = len(ord_features)
num_cols = 2
num_rows = math.ceil(num_plots / num_cols)

plot_target_ratio_by_features(train, ord_features, num_rows, num_cols, (12, 20))

# plot_target_ratio_by_features(train, ord_features, 8, 2, (12, 20))




cont_features = summary[summary['데이터 종류'] == '연속형'].index # 연속형 피처

plt.figure(figsize=(12, 16))                # Figure 크기 설정
grid = gridspec.GridSpec(5, 2)              # GridSpec 객체 생성
plt.subplots_adjust(wspace=0.2, hspace=0.4) # 서브플롯 간 여백 설정

for idx, cont_feature in enumerate(cont_features):
    # 값을 5개 구간으로 나누기
    train[cont_feature] = pd.cut(train[cont_feature], 5)

    ax = plt.subplot(grid[idx])                # 분포도를 그릴 서브플롯 설정
    sns.barplot(x=cont_feature, y='target', data=train, palette='Set2', ax=ax)
    ax.tick_params(axis='x', labelrotation=10) # x축 라벨 회전


train_copy = train_copy.dropna() # np.NaN 값 삭제


plt.figure(figsize=(10, 8))
cont_corr = train_copy[cont_features].corr()     # 연속형 피처 간 상관관계 
sns.heatmap(cont_corr, annot=True, cmap='OrRd'); # 히트맵 그리기







