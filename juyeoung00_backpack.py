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


train_df= pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df= pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


train_df.head(3)


train_df.info()


train_df.isnull().sum()


train_df.nunique()


train_df.describe()


nan_rows = train_df[train_df.notna().all(axis=1) == False]
nan_rows


compartments_group= train_df.groupby('Compartments')

compartments_group.get_group(10.0)


train_df.groupby(['Size','Brand','Material','Waterproof','Compartments'])['Price'].mean()



train_data= train_df

train_data['Size_numeric'] = train_data['Size'].map({'Small': 1, 'Medium': 2, 'Large': 3})
train_data['Material_numeric'] = train_data['Material'].map({'Leather': 0, 'Canvas': 1,'Nylon': 2, 'Polyester': 3 })
train_data['Waterproof_numeric'] = train_data['Waterproof'].map({'Yes': 1, 'No': 0})
train_data['Laptop Compartment_numeric']= train_data['Laptop Compartment'].map({'Yes':1, 'No':0})
train_data['Color_numeric']= train_data['Color'].map({'Black':1, 'Green':2, 'Red':3, 'Blue':4, 'Gray':5, 'Pink':6})

Size_compartments_price_corr = train_data[['Color_numeric','Laptop Compartment_numeric','Material_numeric', 'Waterproof_numeric', 'Size_numeric', 'Compartments', 'Price']].corr()

# 상관계수 출력
print("Price의 상관계수:\n", Size_compartments_price_corr)


train_data['Price'].describe()


# 가격 데이터 (예시로 데이터프레임을 만듦)
# train_data['Price']에 가격 정보가 있다고 가정
price_data = train_data['Price']

# 사분위수 계산
Q1 = price_data.quantile(0.25)
Q2 = price_data.quantile(0.50)  # 중앙값
Q3 = price_data.quantile(0.75)
min_price = price_data.min()
max_price = price_data.max()

# 가격을 범주형 데이터로 변환
bins = [min_price, Q1, Q2, Q3, max_price]  # 구간 설정
labels = ['Low', 'Medium', 'High', 'Very High']  # 각 구간에 해당하는 레이블

# 'Price_category' 컬럼에 범주형 데이터 추가
train_data['Price_category'] = pd.cut(price_data, bins=bins, labels=labels, include_lowest=True)

# 결과 확인
print(train_data[['Price', 'Price_category']].head())



from scipy.stats import chi2_contingency

# 교차 테이블 생성: Waterproof, Size, Material, Brand, Price_category 간의 교차 빈도 테이블
contingency_table = pd.crosstab([train_data['Waterproof'], train_data['Size'],train_data['Material'], train_data['Brand']], train_data['Price_category'])

# 각 셀의 빈도수가 충분히 큰지 확인 (빈도수가 너무 작은 셀을 확인하는 과정)
print(contingency_table)

# 카이제곱 검정 수행
chi2, p, dof, expected = chi2_contingency(contingency_table)

# 결과 출력
print("Chi-squared statistic:", chi2)
print("p-value:", p)
print("Degrees of freedom:", dof)


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 예시로 사용된 교차 테이블
# contigency_table는 이미 교차 테이블로 만들어졌다고 가정
contingency_table = pd.crosstab([train_data['Waterproof'], train_data['Size'], train_data['Material'], train_data['Brand']], train_data['Price_category'])

# 히트맵 생성
plt.figure(figsize=(20, 20))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap="YlGnBu", cbar=True)

# 제목 추가
plt.title('Heatmap of Price Category vs. Waterproof, Size, Material, Brand')

# 그래프 출력
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns


# 브랜드 별 가격
Brand_Price= train_df.groupby('Brand',dropna=False)['Price'].sum()

# 사이즈 별 가격
Size_Price= train_df.groupby('Size',dropna=False)['Price'].sum()

# 방수 기능 별 가격
Waterproof_Price= train_df.groupby('Waterproof',dropna=False)['Price'].sum()

# 컬러 별 가격
Color_Price= train_df.groupby('Color',dropna= False)['Price'].sum()

fig, axes= plt.subplots(2,2, figsize=(20,15))

Brand_Price.plot(kind='bar', ax=axes[0][0],color=sns.color_palette("crest"),width=0.9)
Size_Price.plot(kind='bar', ax=axes[0][1],color=sns.color_palette("crest"),width=0.9)

Waterproof_Price.plot(kind='bar', ax=axes[1][0], color=sns.color_palette("crest"),width=0.9)
Color_Price.plot(kind='bar', ax=axes[1][1], color=sns.color_palette("crest"),width=0.9)

axes[0][0].tick_params(axis='x', rotation=45)
axes[0][1].tick_params(axis='x', rotation=45)
axes[1][0].tick_params(axis='x', rotation=45)
axes[1][1].tick_params(axis='x', rotation=45)

plt.show()


# 브랜드와 재질 별로 가격
Brand_Material_price= train_df.groupby(['Brand','Material'])['Price'].sum().reset_index()

plt.figure(figsize=(16, 8))

# 국가별, 연도별 판매 수량을 막대 그래프로 표시
sns.barplot(data=Brand_Material_price, x='Brand', y='Price', hue='Material', palette="YlGnBu")

# 범례 표시
plt.legend(title='Brand')


# 그래프 출력
plt.show()


# 브랜드와 사이즈 별로 가격
Brand_Size_price= train_df.groupby(['Brand','Size'])['Price'].sum().reset_index()

plt.figure(figsize=(16, 8))

# 국가별, 연도별 판매 수량을 막대 그래프로 표시
sns.barplot(data=Brand_Size_price, x='Brand', y='Price', hue='Size', palette="YlGnBu")

# 범례 표시
plt.legend(title='Brand')


# 그래프 출력
plt.show()


from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


df= pd.get_dummies(train_df, columns=["Material","Style","Brand","Size","Waterproof","Laptop Compartment"],drop_first=True)


df


X= df.drop(["Price","Color"], axis=1)
y= df["Price"]

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=45)

xgb_model= XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)


y_pred= xgb_model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred): .2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")


import matplotlib.pyplot as plt
import xgboost as xgb

# 변수 중요도 시각화
xgb.plot_importance(xgb_model)
plt.show()


from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# 하이퍼파라미터 후보 설정
param_grid = {
    'n_estimators': [50, 100, 200],  # 트리 개수
    'learning_rate': [0.01, 0.1, 0.2],  # 학습률
    'max_depth': [3, 5, 7]  # 트리 깊이
}

# 그리드 서치 수행
xgb_model = XGBRegressor(random_state=42)
grid_search = GridSearchCV(xgb_model, param_grid, scoring='r2', cv=3)
grid_search.fit(X_train, y_train)

# 최적의 하이퍼파라미터 출력
print(grid_search.best_params_)


xgb_model= XGBRegressor(n_estimators=50, learning_rate=0.1,max_depth=3, random_state=42)
xgb_model.fit(X_train, y_train)


y_pred= xgb_model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred): .2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")


import matplotlib.pyplot as plt
import xgboost as xgb

# 변수 중요도 시각화
xgb.plot_importance(xgb_model)
plt.show()


X.info()


X=X.fillna(X["Weight Capacity (kg)"].mean())


X.info()


import statsmodels.api as sm

X = np.asarray(X, dtype=np.float64)  # 또는 dtype=np.int64 사용 가능
X= sm.add_constant(X)

model= sm.OLS(y,X).fit()


print(model.summary())




