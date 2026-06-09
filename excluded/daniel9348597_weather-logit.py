import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

path = '/kaggle/input/playground-series-s5e3/train.csv'
path2 = '/kaggle/input/playground-series-s5e3/test.csv'

df= pd.read_csv(path)
df2= pd.read_csv(path2)


df.head()


# we isolate into separate dataframes the rain=1 and rain=0 observations
df_0 = df[df['rainfall'] == 0]
df_1 = df[df['rainfall'] == 1]


rainfall_count = df['rainfall'].value_counts()
print(rainfall_count)
print(f'No rain / rain : {rainfall_count[0] / rainfall_count[1]}')


# we define our functions to create a visual histogram with buckets, a histogram dictionary and a combined dictionary.
# the combined dictionary shows a list of top n values that are present in both 

def hist(df_0, df_1, col, n):
    # create a histogram of column "pressure" where rainfall is 0
    df_0[col].hist(bins=n)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {col} when Rainfall is 0 with {n} buckets')
    plt.show()
    # create a histogram of column "pressure" where rainfall is 1
    df_1[col].hist(bins=n)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {col} when Rainfall is 1 with {n} buckets')
    plt.show()

def dict_hist(df_0, df_1, col, n):
    global hist_0
    global hist_1
    hist_0 = df_0[col].value_counts().sort_index()
    hist_1 = df_1[col].value_counts().sort_index()
    hist_0 = hist_0.to_dict()
    hist_1 = hist_1.to_dict()
    hist_0 = {k: v for k, v in sorted(hist_0.items(), key=lambda item: item[1], reverse=True)}
    hist_1 = {k: v for k, v in sorted(hist_1.items(), key=lambda item: item[1], reverse=True)}
    hist_0n = dict(list(hist_0.items())[:n])
    hist_1n = dict(list(hist_1.items())[:n])
    return hist_0n, hist_1n

# order is important as hist_0 and hist_1 was designated global for use in the two subsequent functions.

def hist_combined(hist_0, hist_1, n):
    hist_combined = {}
    for k in set(hist_0.keys()).union(set(hist_1.keys())):
        hist_combined[k] = [hist_0.get(k, 0), hist_1.get(k, 0)]
    hist_combined = dict(sorted(hist_combined.items(), key=lambda item: sum(item[1]), reverse=True)[:n])
    print("Combined Histogram:")
    for k, v in hist_combined.items():
        print(f'{k}: {v}')

def create_buckets(hist_0, hist_1, num_buckets):
    global buckets_0
    global buckets_1
    min_val = min(hist_0.keys())
    max_val = max(hist_0.keys())
    bucket_size = (max_val - min_val) / num_buckets
    buckets_0 = {}
    buckets_1 = {}
    for i in range(num_buckets):
        lower_bound = min_val + i * bucket_size
        upper_bound = min_val + (i + 1) * bucket_size
        buckets_0[f'{lower_bound:.2f}-{upper_bound:.2f}'] = sum(v for k, v in hist_0.items() if lower_bound <= k < upper_bound)
        buckets_1[f'{lower_bound:.2f}-{upper_bound:.2f}'] = sum(v for k, v in hist_1.items() if lower_bound <= k < upper_bound)
    return buckets_0, buckets_1

def create_column_1(df, column, from_val, to_val):
    df[f'{column}_1'] = df[column].apply(lambda x: 1 if from_val <= x <= to_val else 0)

def create_column_1_amended(df, column, from_val, to_val):
    df[f'{column}_1'] = np.where((df[column] >= from_val) & (df[column] <= to_val),1,df.get(f'{column}_1', 0))

def log_transform_column(df, column):
    df[f'{column}_log'] = np.log1p(df[column])
    df.drop(columns=[column], inplace=True)


# following the assessments detailed later, we mark mean, individually non-categorically descriptive observations

create_column_1(df, 'pressure', 1007.90, 1009.68)
create_column_1(df, 'maxtemp', 20.56, 21.83)
create_column_1(df, 'temparature', 17.58, 18.74)
create_column_1(df, 'mintemp', 26.04, 27.29)
create_column_1(df, 'dewpoint', 23.73, 26.40)
create_column_1(df, 'winddirection', 39.00, 53.50)
create_column_1_amended(df, 'winddirection', 68.00, 82.50)
create_column_1(df, 'windspeed', 12.67, 15.42)

create_column_1(df2, 'pressure', 1007.90, 1009.68)
create_column_1(df2, 'maxtemp', 20.56, 21.83)
create_column_1(df2, 'temparature', 17.58, 18.74)
create_column_1(df2, 'mintemp', 26.04, 27.29)
create_column_1(df2, 'dewpoint', 23.73, 26.40)
create_column_1(df2, 'winddirection', 39.00, 53.50)
create_column_1_amended(df2, 'winddirection', 68.00, 82.50)
create_column_1(df2, 'windspeed', 12.67, 15.42)


df['factor_sum'] = df.filter(like='_1').sum(axis=1)
df2['factor_sum'] = df2.filter(like='_1').sum(axis=1)


print(df['factor_sum'].value_counts(), df2['factor_sum'].value_counts())


# log transformations

log_transform_column(df, 'maxtemp')
log_transform_column(df, 'sunshine')
log_transform_column(df, 'cloud')
log_transform_column(df, 'winddirection')
log_transform_column(df, 'humidity')
log_transform_column(df, 'windspeed')

log_transform_column(df2, 'maxtemp')
log_transform_column(df2, 'sunshine')
log_transform_column(df2, 'cloud')
log_transform_column(df2, 'winddirection')
log_transform_column(df2, 'humidity')
log_transform_column(df2, 'windspeed')


import statsmodels.api as sm
X = df.drop(columns=['rainfall'])
y = df['rainfall']
model = sm.Logit(y, X.astype(float))
results = model.fit()
print(results.summary())


# we drop values larger or equal to 3 from the factors which does increase the R2 marginally 
df.drop(df[df['factor_sum'] >= 3].index, inplace=True)
# df2.drop(df2[df2['factor_sum'] >= 3].index, inplace=True)


# we also drop the "_1" control variables used to asses uneventful observations
X = df.drop(columns=['rainfall', 'id', 'day', 'factor_sum'])
X2 = df2.drop(columns=['id', 'day', 'factor_sum'])
X2.dropna(inplace=True)
X = X.drop(columns=df.filter(like='_1').columns)
X2 = X2.drop(columns=df2.filter(like='_1').columns)
y = df['rainfall']
model = sm.Logit(y, X.astype(float))
results = model.fit()
print(results.summary())


# we drop the remaining variables that did not pass the significance test
X = X.drop(columns=['temparature', 'mintemp', 'maxtemp_log', 'winddirection_log'])
X2 = X2.drop(columns=['temparature', 'mintemp', 'maxtemp_log', 'winddirection_log'])


# as we do not know the exact interplays between the independent variables themselves, we will create interactions between these
# the interaction terms will have a degree of two, meaning at all independent variables will be multiplied by one other independent variable

from sklearn.preprocessing import PolynomialFeatures

original_feature_names_X = X.columns
original_feature_names_X2 = X2.columns
original_index_X = X.index
original_index_X2 = X2.index

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)
X2_poly = poly.fit_transform(X2)

feature_names_X = poly.get_feature_names_out(original_feature_names_X)
feature_names_X2 = poly.get_feature_names_out(original_feature_names_X2)

X = pd.DataFrame(X_poly, columns=feature_names_X, index=original_index_X)
X2 = pd.DataFrame(X2_poly, columns=feature_names_X2, index=original_index_X2)


# with creating interaction terms we have taken out a significance-test focus and instead shifted to joint significance of the variables

y = df['rainfall']
model = sm.Logit(y, X.astype(float))
results = model.fit()
print(results.summary())


y_pred_prob = results.predict(X)
y_pred_label = (y_pred_prob > 0.5).astype(int)


from sklearn.metrics import accuracy_score
accuracy_score(y, y_pred_label)


from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y, y_pred_prob)
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0,1], [0,1], '--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


y_test_pred_prob = results.predict(X2)


X2.head()


ids = '/kaggle/input/playground-series-s5e3/test.csv'

ids = pd.read_csv(ids)


type(y_test_pred_prob)


y_test_pred_prob


y_test_pred_prob.name = 'pred_prob'


df_f = pd.DataFrame({'id': ids['id'], 'y_test_pred_prob': y_test_pred_prob})


print(df_f.isnull().sum())


df_f.fillna(0.0001, inplace=True)


df_f.to_csv('/kaggle/working/submission.csv', index=False)
print(df_f)


hist(df_0, df_1, 'pressure', 20)
dict_hist(df_0, df_1, 'pressure', 20)
hist_combined(hist_0, hist_1, 20)
create_buckets(hist_0, hist_1, 20)


hist(df_0, df_1, 'maxtemp', 20)
dict_hist(df_0, df_1, 'maxtemp', 20)
hist_combined(hist_0, hist_1, 20)
create_buckets(hist_0, hist_1, 20)


hist(df_0, df_1, 'temparature', 20)
dict_hist(df_0, df_1, 'temparature', 20)
hist_combined(hist_0, hist_1, 20)
create_buckets(hist_0, hist_1, 20)


hist(df_0, df_1, 'mintemp', 20)
dict_hist(df_0, df_1, 'mintemp', 20)
hist_combined(hist_0, hist_1, 20)
create_buckets(hist_0, hist_1, 20)


hist(df_0, df_1, 'dewpoint', 20)
dict_hist(df_0, df_1, 'dewpoint', 20)
hist_combined(hist_0, hist_1, 20)
create_buckets(hist_0, hist_1, 20)


hist(df_0, df_1, 'humidity', 20)
dict_hist(df_0, df_1, 'humidity', 20)
hist_combined(hist_0, hist_1, 20)
create_buckets(hist_0, hist_1, 20)


hist(df_0, df_1, 'cloud', 20)
dict_hist(df_0, df_1, 'cloud', 20)
hist_combined(hist_0, hist_1, 20)
create_buckets(hist_0, hist_1, 20)


hist(df_0, df_1, 'sunshine', 20)
dict_hist(df_0, df_1, 'sunshine', 20)
hist_combined(hist_0, hist_1, 20)
create_buckets(hist_0, hist_1, 20)


hist(df_0, df_1, 'winddirection', 20)
dict_hist(df_0, df_1, 'winddirection', 20)
hist_combined(hist_0, hist_1, 20)
create_buckets(hist_0, hist_1, 20)


hist(df_0, df_1, 'windspeed', 20)
dict_hist(df_0, df_1, 'windspeed', 20)
hist_combined(hist_0, hist_1, 20)
create_buckets(hist_0, hist_1, 20)


hist(df_0, df_1, 'day', 20)
dict_hist(df_0, df_1, 'day', 20)
hist_combined(hist_0, hist_1, 20)
create_buckets(hist_0, hist_1, 20)




