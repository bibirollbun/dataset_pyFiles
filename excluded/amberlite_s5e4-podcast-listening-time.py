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


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer

from sklearn.metrics import make_scorer, mean_squared_error

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


train.head().T


train.shape


def null_and_types(df):
    df_null_and_types = pd.concat([df.isnull().sum(),df.dtypes], axis=1)
    return df_null_and_types
    
null_and_types(train)


null_and_types(test)


def percent_null(df):
    for col in df.columns:
        percent_null = (df[f'{col}'].isnull().sum()/len(df[f'{col}']))*100
        print(f'{col}: {percent_null:.2f}')

percent_null(train)


def num_cat_split(df, float=False, int=False, object=False):
    if float and not int:
        df_num = df.select_dtypes(include=['float'])
    elif int and not float:
        df_num = df.select_dtypes(include=['int'])
    elif float and int:
        df_num = pd.concat([df.select_dtypes(include=['float']), df.select_dtypes(include=['int'])], axis=1)
    if object:
        df_cat = df.select_dtypes(include=['object'])
    return df_num, df_cat


train_num, train_cat = num_cat_split(train, float=True, int=True, object=True)


test_num, test_cat = num_cat_split(test, float=True, int=True, object=True)


train_cat.head().T


train_cat.nunique()


train_cat = pd.concat([train_cat, train_num['Listening_Time_minutes']], axis=1)


qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(train_cat['Listening_Time_minutes']), 1000), random_state=0)
train_cat['Listening_Time_minutes_qt'] = qt.fit_transform(train_cat['Listening_Time_minutes'].values.reshape(-1, 1))
train_cat = train_cat.drop(columns=['Listening_Time_minutes'], axis=1)


train_cat['Genre'].value_counts()


import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

model = ols('Listening_Time_minutes_qt ~ C(Genre)', data=train_cat).fit()

anova_table = anova_lm(model, typ=2)

print("ANOVA Таблица:")
print(anova_table)

p_value_sm = anova_table['PR(>F)'][0] # P-значение для фактора 'группа'
alpha = 0.05

if p_value_sm < alpha:
    print("\nОтвергаем нулевую гипотезу: есть статистически значимые различия между средними групп.")
else:
    print("\nНе отвергаем нулевую гипотезу: нет статистически значимых различий между средними групп.")


from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey_result = pairwise_tukeyhsd(endog=train_cat['Listening_Time_minutes_qt'], groups=train_cat['Genre'], alpha=0.05)

print("\nРезультаты теста Тьюки HSD:")
print(tukey_result)


train_cat['Genre'] = train_cat['Genre'].replace('Business', 'BEHLT')
train_cat['Genre'] = train_cat['Genre'].replace('Education', 'BEHLT')
train_cat['Genre'] = train_cat['Genre'].replace('Health', 'BEHLT')
train_cat['Genre'] = train_cat['Genre'].replace('Lifestyle', 'BEHLT')
train_cat['Genre'] = train_cat['Genre'].replace('Technology', 'BEHLT')
train_cat['Genre'] = train_cat['Genre'].replace('Comedy', 'CN')
train_cat['Genre'] = train_cat['Genre'].replace('News', 'CN')


test_cat['Genre'] = test_cat['Genre'].replace('Business', 'BEHLT')
test_cat['Genre'] = test_cat['Genre'].replace('Education', 'BEHLT')
test_cat['Genre'] = test_cat['Genre'].replace('Health', 'BEHLT')
test_cat['Genre'] = test_cat['Genre'].replace('Lifestyle', 'BEHLT')
test_cat['Genre'] = test_cat['Genre'].replace('Technology', 'BEHLT')
test_cat['Genre'] = test_cat['Genre'].replace('Comedy', 'CN')
test_cat['Genre'] = test_cat['Genre'].replace('News', 'CN')


train_cat = train_cat.drop(columns=['Listening_Time_minutes_qt'], axis=1)


train_cat.head()


def EDA_hist_box(df):
    for col in df.columns:
        if df[f'{col}'].dtypes == float:
            plt.figure(figsize=(5,4))
            f, (ax_box, ax_kde) = plt.subplots(nrows=2, # из двух строк
                                    ncols=1, # и одного столбца
                                    figsize=(7,4),
                                    gridspec_kw={'height_ratios': (.15, .85)}) # зададим разную высоту строк

            # в первом подграфике построим boxplot
            sns.boxplot(df[f'{col}'], orient='h', ax=ax_box)
            ax_box.set(xlabel=None)
            # во втором - график плотности распределения
            sns.histplot(df[f'{col}'], kde=True)

            # зададим заголовок и подписи к осям
            ax_box.set_title((f'{col}'), fontsize = 10)
            plt.show()


# EDA_hist_box(train_num.drop(columns=['id'], axis=1))


train_num.head()


# sns.pairplot(data=train_num.drop(columns=['id'], axis=1))


def quantile_transform(df):
    for col in df.columns:
        qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(df[f'{col}']), 1000), random_state=0)
        df[f'{col}_qt'] = qt.fit_transform(df[f'{col}'].values.reshape(-1, 1))
        df = df.drop(columns=[col], axis=1)
    return df


train_num_qt = quantile_transform(train_num.drop(columns=['id'], axis=1))


test_num_qt = quantile_transform(test_num.drop(columns=['id'], axis=1))


train_num_qt.head()


test_num_qt.head()


percent_null(train_num_qt)


percent_null(test_num_qt)


train_num_qt['Episode_Length_minutes_qt'].fillna(train_num_qt['Episode_Length_minutes_qt'].mean(), inplace=True)
# train_num_qt['Guest_Popularity_percentage_qt'].fillna(train_num_qt['Guest_Popularity_percentage_qt'].median(), inplace=True)

train_num_qt = train_num_qt.drop(columns=['Guest_Popularity_percentage_qt'], axis=1)


test_num_qt['Episode_Length_minutes_qt'].fillna(test_num_qt['Episode_Length_minutes_qt'].mean(), inplace=True)
# test_num_qt['Guest_Popularity_percentage_qt'].fillna(test_num_qt['Guest_Popularity_percentage_qt'].mean(), inplace=True)

test_num_qt = test_num_qt.drop(columns=['Guest_Popularity_percentage_qt'], axis=1)


train_num_qt['Number_of_Ads_qt'].fillna(train_num_qt['Number_of_Ads_qt'].mean(), inplace=True)


train_num_qt.isnull().sum()


test_num_qt.isnull().sum()


# EDA_hist_box(train_num_qt)


# EDA_hist_box(test_num_qt)


corr_matrix = train_num_qt.corr()
plt.figure(figsize=(4, 3))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Матрица корреляции')
plt.show()


train_cat.head()


test_cat.head()


train_cat.nunique()


train_cat = pd.concat([train_cat, train_num_qt['Listening_Time_minutes_qt']], axis=1)


train_cat.head().T


train_cat['Episode_Title'] = train_cat['Episode_Title'].str.slice(8).astype('float')


test_cat['Episode_Title'] = test_cat['Episode_Title'].str.slice(8).astype('float')


train_cat['Episode_Title']


# plt.figure(figsize=(4,3))
# sns.histplot(train_cat['Episode_Title'])


qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(train_cat['Episode_Title']), 1000), random_state=0)
train_cat['Episode_Title_qt'] = qt.fit_transform(train_cat['Episode_Title'].values.reshape(-1, 1))
train_cat = train_cat.drop(columns=['Episode_Title'], axis=1)


qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(test_cat['Episode_Title']), 1000), random_state=0)
test_cat['Episode_Title_qt'] = qt.fit_transform(test_cat['Episode_Title'].values.reshape(-1, 1))
test_cat = test_cat.drop(columns=['Episode_Title'], axis=1)


# plt.figure(figsize=(4,3))
# sns.scatterplot(x=train_cat['Episode_Title_qt'], y=train_cat['Listening_Time_minutes_qt'])


train_num_qt.head().T


test_num_qt.head().T


train_cat.head().T


test_cat.head().T


target = train_num_qt['Listening_Time_minutes_qt']


train_num_qt = train_num_qt.drop(columns=['Listening_Time_minutes_qt'], axis=1)
train_cat = train_cat.drop(columns=['Listening_Time_minutes_qt', 'Episode_Title_qt'], axis=1)


test_cat = test_cat.drop(columns=['Episode_Title_qt'], axis=1)


train_prep = pd.concat([train_num_qt, train_cat], axis=1)


train_prep.dtypes


test_prep = pd.concat([test_num_qt, test_cat], axis=1)


def one_hot(df):
    loc_object = []
    for i in df.select_dtypes(include=['object']).columns:
        loc_object.append(df.columns.get_loc(i))

    loc_float = []
    for i in df.select_dtypes(include=['float']).columns:
        loc_float.append(df.columns.get_loc(i))

    c_transf = ColumnTransformer([
    ('onehot', OneHotEncoder(categories='auto', drop='first'), loc_object),
    ('nothing', 'passthrough', loc_float)
    ])
    return c_transf.fit_transform(df)


train_prep = one_hot(train_prep)


train_prep.shape


test_prep = one_hot(test_prep)


test_prep.shape


model = XGBRegressor()

scores = cross_val_score(model, train_prep, target.values, cv=5, scoring=make_scorer(mean_squared_error, squared=False))
print(f'CV scores: {scores}')
print(f'CV mean score: {scores.mean():.4} ± {scores.std():.4}')


model = LGBMRegressor(verbose=0)

scores = cross_val_score(model, train_prep, target.values, cv=5, scoring=make_scorer(mean_squared_error, squared=False))
print(f'CV scores: {scores}')
print(f'CV mean score: {scores.mean():.4} ± {scores.std():.4}')


from sklearn.linear_model import LinearRegression, Ridge

model = LinearRegression()

scores = cross_val_score(model, train_prep, target.values, cv=5, scoring=make_scorer(mean_squared_error, squared=False))
print(f'CV scores: {scores}')
print(f'CV mean score: {scores.mean():.4} ± {scores.std():.4}')


from sklearn.ensemble import VotingRegressor, StackingRegressor


models = [
    ('xgb', XGBRegressor()),
    ('lbbm', LGBMRegressor(verbose=0)),
    # ('lr', LinearRegression())
]

voting_model = VotingRegressor(estimators=models)

scores = cross_val_score(voting_model, train_prep, target.values, cv=5, scoring=make_scorer(mean_squared_error, squared=False))
print(f'CV scores: {scores}')
print(f'CV mean score: {scores.mean():.4} ± {scores.std():.4}')


models = [
    ('xgb', XGBRegressor()),
    ('lbbm', LGBMRegressor(verbose=0)),
    # ('lr', LinearRegression())
]

meta_model = Ridge(random_state=0)

stacking_model = StackingRegressor(
    estimators=models,
    final_estimator=meta_model)

scores = cross_val_score(stacking_model, train_prep, target.values, cv=5, scoring=make_scorer(mean_squared_error, squared=False))
print(f'CV scores: {scores}')
print(f'CV mean score: {scores.mean():.4} ± {scores.std():.4}')


voting_model.fit(train_prep, target.values)

predictions_qt = voting_model.predict(test_prep)

qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(train['Listening_Time_minutes']), 1000), random_state=0)
qt.fit(train['Listening_Time_minutes'].values.reshape(-1, 1))

predictions = qt.inverse_transform(predictions_qt.reshape(-1, 1))

results = pd.DataFrame({
    'id':test['id'].values,
    'Listening_Time_minutes':predictions.ravel()})
results.to_csv('sample_submission.csv', index=False)

