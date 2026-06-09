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


# 1) 7-zip 설치
!apt-get update && apt-get install -y p7zip-full

# 2) train.tsv.7z 압축 풀기
!7z x /kaggle/input/mercari-price-suggestion-challenge/train.tsv.7z -o/kaggle/working/train

# 3) test.tsv.7z 압축 풀기
!7z x /kaggle/input/mercari-price-suggestion-challenge/test.tsv.7z -o/kaggle/working/test


from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# 경로 지정
train_path = '/kaggle/working/train/train.tsv'
test_path  = '/kaggle/working/test/test.tsv'

# TSV 파일 읽기 (탭 구분자)
df_mercari = pd.read_csv(train_path, sep='\t')
# mercari_test  = pd.read_csv(test_path,  sep='\t')

print(df_mercari.shape)
df_mercari.head(5)


# 컬럼 정보 확인
print(df_mercari.info())


# null 값 체크
df_mercari.isnull().sum()


# Target 값인 price 칼럼의 데이터 분포도

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

y_train_df = df_mercari['price']
plt.figure(figsize=(6,4))
sns.distplot(y_train_df,kde=False)


# Price 값이 적은 가격의 데이터에 왜곡, Price 칼럼을 로그값으로 변환

y_train_df = np.log1p(y_train_df)
sns.distplot(y_train_df,kde=False)


# 데이터 세트의 price 칼럼을 원래 값에서 로그로 변환된 겂으로 변경

df_mercari['price'] = np.log1p(df_mercari['price'])
df_mercari['price'].head(3)


# shipping, item_condition_id 값의 유형

print('Shipping 값 유형:\n', df_mercari['shipping'].value_counts())
print('item_condition_id 값 유형:\n', df_mercari['item_condition_id'].value_counts())


# description의 'No description yet' 확인

boolean_cond = df_mercari['item_description']=='No description yet'
df_mercari[boolean_cond]['item_description'].count()


cats = df_mercari['category_name'].str.split('/', expand=True)
cats3 = cats.iloc[:, :3]


# cats3는 앞서 만든 DataFrame(첫 3개 컬럼만 포함)
df_mercari[['cat_dae', 'cat_jung', 'cat_so']] = cats3


# 기존 컬럼 삭제
df_mercari = df_mercari.drop(columns=['cat_dae','cat_jung','cat_so'], errors='ignore')

# join 실행
df_mercari = df_mercari.join(cats3)



def split_cat(x):
    if isinstance(x, str):
        parts = x.split('/')
    else:
        parts = []
    # 부족할 때는 None(또는 '')으로 채우고, 넘칠 땐 잘라내기
    parts = (parts + [None, None, None])[:3]
    return parts[0], parts[1], parts[2]

# apply 후 DataFrame으로 변환
df_mercari[['cat_dae', 'cat_jung', 'cat_so']] = df_mercari['category_name'] \
    .apply(split_cat) \
    .apply(pd.Series)

# 대분류만 값의 유형과 건수를 살펴보고, 중분류, 소분류는 값의 유형이 많으므로 분류 갯수만 추출
print('대분류 유형: \n', df_mercari['cat_dae'].value_counts())
print('중분류 갯수: \n', df_mercari['cat_jung'].nunique())
print('소분류 갯수: \n', df_mercari['cat_so'].nunique())


# brand_name, category_name, item_description 칼럼의 Null 값은 일괄적으로 'Other_Null'로 동일하게 변경
df_mercari['brand_name'] = df_mercari['brand_name'].fillna(value='Other_Null')
df_mercari['category_name'] = df_mercari['category_name'].fillna(value='Other_Null')
df_mercari['item_description'] = df_mercari['item_description'].fillna(value='Other_Null')

# 각 컬럼별로 Null값 건수 확인. 모두 0가 나와야 합니다
df_mercari.isnull().sum()


# 상품 브랜드명이 어떤 유형으로 돼 있는지 유형 건수와 대표적인 브랜드명을 5개만 확인
print('brand name의 유형 건수 :', df_mercari['brand_name'].nunique())
print('brand name sample 5건 : \n', df_mercari['brand_name'].value_counts()[:5])


# 상품명을 의미하는 name 속성 건수와 상품명을 7개
print('name의 종류 갯수 :', df_mercari['name'].nunique())
print('name sample 7건 : \n', df_mercari['name'][:7])


# item_description의 평균 문자열 크기

pd.set_option('max_colwidth', 200)

# item_description의 평균 문자열 개수
print('item_description 평균 문자열 개수:', df_mercari['item_description'].str.len().mean())

df_mercari['item_description'][:2]


# name 속성에 대한 feature vectorization 변환
cnt_vec = CountVectorizer(max_features=30000)
X_name = cnt_vec.fit_transform(df_mercari.name)

# item_description 에 대한 feature vectorization 변환 
tfidf_descp = TfidfVectorizer(max_features = 50000, ngram_range= (1,3) , stop_words='english')
X_descp = tfidf_descp.fit_transform(df_mercari['item_description'])

print('name vectorization shape:',X_name.shape)
print('item_description vectorization shape:',X_descp.shape)


from sklearn.preprocessing import OneHotEncoder

# 1) 결측 또는 None을 'Unknown'으로 대체
df_mercari['cat_jung'] = df_mercari['cat_jung'].fillna('Unknown')

# 2) OneHotEncoder 준비 (sparse output 원하면 sparse=True)
ohe = OneHotEncoder(sparse=True, handle_unknown='ignore')

# 3) fit_transform 으로 middle‐category를 행렬로 변환
X_cat_jung = ohe.fit_transform(df_mercari[['cat_jung']])

# 결과 확인
print("Encoded shape:", X_cat_jung.shape)
print("Categories:", ohe.categories_)


from sklearn.preprocessing import OneHotEncoder

# 1) 결측값을 'Unknown'으로 채웁니다
df_mercari['cat_so'] = df_mercari['cat_so'].fillna('Unknown')

# 2) OneHotEncoder 준비: 학습 시 보지 못한 카테고리는 무시하도록 설정
ohe_so = OneHotEncoder(sparse=True, handle_unknown='ignore')

# 3) fit_transform으로 인코딩
#    데이터는 2D여야 하므로 컬럼을 [['cat_so']] 형태로 전달
X_cat_so = ohe_so.fit_transform(df_mercari[['cat_so']])

print("Encoded shape (소카테고리):", X_cat_so.shape)
print("Categories:", ohe_so.categories_)


# 1) 결측값을 'Unknown'으로 채우기
df_mercari['cat_dae'] = df_mercari['cat_dae'].fillna('Unknown')

# 2) OneHotEncoder 준비 (알 수 없는 값은 무시)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=True, handle_unknown='ignore')

# 3) fit_transform으로 인코딩
X_cat_dae = ohe.fit_transform(df_mercari[['cat_dae']])

print("Encoded shape:", X_cat_dae.shape)
print("Categories:", ohe.categories_[0])



# 인코딩 대상 컬럼을 모두 LabelBinarizer로 원-핫 인코딩 변환

from sklearn.preprocessing import LabelBinarizer

# brand_name, item_condition_id, shipping 각 피처들을 희소 행렬 원-핫 인코딩 변환
lb_brand_name= LabelBinarizer(sparse_output=True)
X_brand = lb_brand_name.fit_transform(df_mercari['brand_name'])

lb_item_cond_id = LabelBinarizer(sparse_output=True)
X_item_cond_id = lb_item_cond_id.fit_transform(df_mercari['item_condition_id'])

lb_shipping= LabelBinarizer(sparse_output=True)
X_shipping = lb_shipping.fit_transform(df_mercari['shipping'])

# cat_dae, cat_jung, cat_so 각 피처들을 희소 행렬 원-핫 인코딩 변환
lb_cat_dae = LabelBinarizer(sparse_output=True)
X_cat_dae= lb_cat_dae.fit_transform(df_mercari['cat_dae'])

lb_cat_jung = LabelBinarizer(sparse_output=True)
X_cat_jung = lb_cat_jung.fit_transform(df_mercari['cat_jung'])

lb_cat_so = LabelBinarizer(sparse_output=True)
X_cat_so = lb_cat_so.fit_transform(df_mercari['cat_so'])


# 생성된 인코딩 데이터 세트의 타입과 shape 확인

print(type(X_brand),type(X_item_cond_id),type(X_shipping))
print('X_brand shape:{0}, X_item_cond_id shape:{1}'.format(X_brand.shape, X_item_cond_id.shape))
print('X_shipping shape:{0}, X_cat_dae shape:{1}'.format(X_shipping.shape, X_cat_dae.shape))
print('X_cat_jung shape:{0}, X_cat_so shape:{1}'.format(X_cat_jung.shape, X_cat_so.shape))


from scipy.sparse import hstack
import gc

sparse_matrix_list = (X_name, X_descp, X_brand, X_item_cond_id,
                     X_shipping, X_cat_dae, X_cat_jung, X_cat_so)

# 사이파이 sparse 모듈의 hstack 함수를 이용하여 앞에서 인코딩과 Vectorization을 수행한 데이터 셋을 모두 결합.
X_features_sparse = hstack(sparse_matrix_list).tocsr()
print(type(X_features_sparse), X_features_sparse.shape)

# 데이터 셋이 메모리를 많이 차지하므로 사용 용도가 끝났으면 바로 메모리에서 삭제. 
del X_features_sparse
gc.collect()


# RMSLE를 적용할 수 있도록 evaluate_org_price(y_test, preds) 함수 생성

def rmsle(y , y_pred):
    # underflow, overflow를 막기 위해 log가 아닌 log1p로 rmsle 계산 
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y_pred), 2)))

def evaluate_org_price(y_test , preds): 
    
    # 원본 데이터는 log1p로 변환되었으므로 exmpm1으로 원복 필요. 
    preds_exmpm = np.expm1(preds)
    y_test_exmpm = np.expm1(y_test)
    
    # rmsle로 RMSLE 값 추출
    rmsle_result = rmsle(y_test_exmpm, preds_exmpm)
    return rmsle_result


# 학습용 데이터를 생성하고, 모델을 학습/예측하는 로직을 별도 함수로 만들기

import gc 
from  scipy.sparse import hstack

def model_train_predict(model,matrix_list):
    # scipy.sparse 모듈의 hstack 을 이용하여 sparse matrix 결합
    X= hstack(matrix_list).tocsr()     
    
    X_train, X_test, y_train, y_test=train_test_split(X, df_mercari['price'], 
                                                      test_size=0.2, random_state=156)
    
    # 모델 학습 및 예측
    model.fit(X_train , y_train)
    preds = model.predict(X_test)
    
    del X , X_train , X_test , y_train 
    gc.collect()
    
    return preds , y_test


linear_model = Ridge(solver = "lsqr", fit_intercept=False)

sparse_matrix_list = (X_name, X_brand, X_item_cond_id,
                      X_shipping, X_cat_dae, X_cat_jung, X_cat_so)
linear_preds , y_test = model_train_predict(model=linear_model ,matrix_list=sparse_matrix_list)
print('Item Description을 제외했을 때 rmsle 값:', evaluate_org_price(y_test , linear_preds))

sparse_matrix_list = (X_descp, X_name, X_brand, X_item_cond_id,
                      X_shipping, X_cat_dae, X_cat_jung, X_cat_so)
linear_preds , y_test = model_train_predict(model=linear_model , matrix_list=sparse_matrix_list)
print('Item Description을 포함한 rmsle 값:',  evaluate_org_price(y_test ,linear_preds))

