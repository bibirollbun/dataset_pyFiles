import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import boxcox
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from scipy.stats import ks_2samp
warnings.filterwarnings('ignore')


df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
df_orig = pd.read_csv('/kaggle/input/calories-burnt-prediction/calories.csv')


df_orig.columns = df_train.columns


df_orig.drop(columns = ['id'], inplace = True)
df_train.drop(columns = ['id'], inplace = True)
df_test.drop(columns = ['id'], inplace = True)


df_orig['Sex'] = df_orig['Sex'].apply(lambda sex: 0 if sex == 'female' else 1)
df_train['Sex'] = df_train['Sex'].apply(lambda sex: 0 if sex == 'female' else 1)
df_test['Sex'] = df_test['Sex'].apply(lambda sex: 0 if sex == 'female' else 1)


df_res = pd.read_csv('/kaggle/input/new-res/res (1).csv')


plt.figure(figsize = (16, 12))
sns.heatmap(df_res.corr(), annot = True, vmin = -1, vmax = 1)


df_res.describe()


df_train.info()


df_orig.info()


for col in df_train.drop(columns = ['Calories']).columns:
    plt.figure(figsize = (12, 6))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)
    sns.histplot(x = df_train[col], ax = ax1, kde = True, bins = 20)
    sns.histplot(x = df_orig[col], ax = ax2, kde = True, bins = 20)
    sns.histplot(x = df_test[col], ax = ax3, kde = True, bins = 20)
    ax1.set_xlabel(xlabel = col + "_train")
    ax2.set_xlabel(xlabel = col + "_original")
    ax3.set_xlabel(xlabel = col + "_test")
    plt.tight_layout()
    


df_train.describe()


df_orig.describe()


for col in df_train.columns:
    stat, p_value = ks_2samp(df_train[col], df_orig[col])
    similarity = True if p_value >= 0.05 else False
    print(f'{col}: p_value = {p_value}, {similarity}')


for col in df_train.drop(columns = ['Calories']).columns:
    stat, p_value = ks_2samp(df_train[col], df_test[col])
    similarity = True if p_value >= 0.05 else False
    print(f'{col}: p_value = {p_value}, {similarity}')


for col in df_test.columns:
    stat, p_value = ks_2samp(df_test[col], df_orig[col])
    similarity = True if p_value >= 0.05 else False
    print(f'{col}: p_value = {p_value}, {similarity}')


def to_distribution(mask, sample):
    sorted_mask = sorted(mask.values)

    #mask_quantiles = np.argsort(np.argsort(mask)) / (len(mask) - 1)
    #sample_quantiles = np.linspace(0, 1, len(sample))

    mask_quantiles = np.linspace(0, 1, len(mask))
    sample_quantiles = np.argsort(np.argsort(sample)) / (len(sample) - 1)
    
    result = np.interp(sample_quantiles, mask_quantiles, sorted_mask)
    return result


res = match_template_to_source(df_orig['Age'], df_train['Age'])


df_age = df_orig[['Age']]
df_age['interp_Age'] = res
df_age['res'] = abs(df_age['Age'] - df_age['interp_Age'])
df_age


for col in df_train.select_dtypes(include = ['number']).columns:
    plt.figure(figsize = (12, 8))
    sns.lineplot(data = df_train, x = col, y = 'Calories')


for col in df_orig.select_dtypes(include = ['number']).columns:
    plt.figure(figsize = (12, 8))
    sns.lineplot(data = df_orig, x = col, y = 'Calories')


df_subm = pd.read_csv('/kaggle/input/new-subm/submission (3).csv')
df_subm


df_test_subm = pd.concat([df_test, df_subm['Calories']], axis = 1)
df_test_subm.head()


for col in df_test_subm.select_dtypes(include = ['number']).columns:
    plt.figure(figsize = (12, 8))
    sns.lineplot(data = df_test_subm, x = col, y = 'Calories')


bt = df_train[['Body_Temp', 'Calories']]


linregbt = LinearRegression()
poly = PolynomialFeatures(degree = 5)
X_bt = poly.fit_transform(bt[['Body_Temp']])
y_bt = bt['Calories']
linregbt.fit(X_bt, y_bt)


appr_bt = linregbt.predict(X_bt)


df_bt = pd.DataFrame({'appr_cal': appr_bt,
                     'body_temp': df_train['Body_Temp'],
                     'real_cal': df_train['Calories']})
df_bt


X_bt


plt.figure(figsize = (12, 8))
sns.lineplot(data = df_train, x = 'Body_Temp', y = 'Calories')
sns.lineplot(x = df_train['Body_Temp'], y = appr_bt)


age = df_train[['Age', 'Calories']]
linregage = LinearRegression()
polyage = PolynomialFeatures(degree = 3)
X_age = polyage.fit_transform(age[['Age']])
y_age = age['Calories']
linregage.fit(X_age, y_age)
appr_agecal = linregage.predict(X_age)


plt.figure(figsize = (12, 8))
sns.lineplot(data = df_train, x = 'Age', y = 'Calories')
sns.lineplot(x = df_train['Age'], y = appr_agecal)


hr = df_train[['Heart_Rate', 'Calories']]
linreghr = LinearRegression()
polyhr = PolynomialFeatures(degree = 5)
X_hr = polyhr.fit_transform(hr[['Heart_Rate']])
y_hr = hr['Calories']
linreghr.fit(X_hr, y_hr)
appr_hrcal = linreghr.predict(X_hr)


hr = df_train[['Heart_Rate', 'Calories']]
linreghr = LinearRegression()
polyhr = PolynomialFeatures(degree = 5)
X_hr = polyhr.fit_transform(hr[['Heart_Rate']])
y_hr = hr['Calories']
linreghr.fit(X_hr, y_hr)
appr_hrcal = linreghr.predict(X_hr)


appr_hrcal


plt.figure(figsize = (12, 8))
sns.lineplot(data = df_train, x = 'Heart_Rate', y = 'Calories')
sns.lineplot(x = df_train['Heart_Rate'], y = appr_hrcal)


df_res.head()


df_train.head()


for col in df_train.columns:
    plt.figure(figsize = (12, 8))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    sns.lineplot(data = df_train, x = col, y = 'Calories', ax = ax1)
    sns.lineplot(data = df_res, x = col, y = 'Abs_Residual', ax = ax2)


for col in df_train.columns:
    plt.figure(figsize = (12, 8))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    sns.lineplot(data = df_train, x = col, y = 'Calories', ax = ax1)
    sns.lineplot(data = df_res, x = col, y = 'Prediction', ax = ax2)


for col in df_train.columns:
    plt.figure(figsize = (16, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    sns.lineplot(data = df_train, x = col, y = 'Calories', ax = ax1)
    sns.lineplot(data = df_test_subm, x = col, y = 'Calories', ax = ax2)


df_subm = pd.read_csv('/kaggle/input/last-results/submission_last.csv')
df_test_res = pd.read_csv('/kaggle/input/last-results/test_res.csv')


df_test_res[['Calories', 'Predictions']] = np.expm1(df_test_res[['Calories', 'Predictions']])


df_test_res


df_test_subm = pd.concat([df_test, np.log1p(df_subm['Calories'])], axis = 1)
df_test_subm = df_test_subm.rename({"Calories": 'Predictions'}, axis = 1)
df_test_subm


def to_approx(column, target, degree, test = None):
    linreg = Ridge(alpha = 2)
    poly = PolynomialFeatures(degree = degree)
    X = poly.fit_transform(column)
    linreg.fit(X, target)
    predictions = linreg.predict(X)
    if test is not None:
        X_test = poly.transform(test)
        test_pred = linreg.predict(X_test)
        return predictions, test_pred
    return predictions


degrees = [3, 4, 2, 4, 10]


for num, col in enumerate(df_train.drop(columns = ['Sex', 'Duration', 'Calories']).columns):
    _, df_test_res[col + '_approx'] = to_approx(df_train[[col]], np.log1p(df_train['Calories']), degrees[num], df_test_res[[col]])
    _, df_test_subm[col + '_approx'] = to_approx(df_train[[col]], np.log1p(df_train['Calories']), degrees[num], df_test_subm[[col]])


df_test_res


df_test_subm


df_test_res


for col in df_train.drop(columns = ['Sex', 'Duration', 'Calories']).columns:
    plt.figure(figsize = (16, 8))
    sns.lineplot(data = df_test_res, x = col, y = 'Calories')
    sns.lineplot(data = df_test_res, x = col, y = col + '_approx')


for col in df_train.drop(columns = ['Sex', 'Duration', 'Calories']).columns:
    plt.figure(figsize = (16, 8))
    sns.lineplot(data = df_test_subm, x = col, y = 'Predictions')
    sns.lineplot(data = df_test_subm, x = col, y = col + '_approx')


df_test_res['id'] = df_test_res.index.to_list()


df_test_res['group_id'] = pd.cut(df_test_res['id'], bins = 50, labels = False)


df_test_res


for col in df_train.drop(columns = ['Sex', 'Duration', 'Calories']).columns:
    plt.figure(figsize = (16, 8))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    sns.lineplot(data = df_test_res, x = 'group_id', y = 'Calories', ax = ax1)
    sns.lineplot(data = df_test_res, x = 'group_id', y = col + '_approx', ax = ax1)
    sns.lineplot(data = df_test_res, x = 'group_id', y = df_test_res['Calories'] - df_test_res[col + '_approx'], ax = ax2)


df_subm = pd.read_csv('/kaggle/input/results-last/simple_orig_150.csv')
df_res = pd.read_csv('/kaggle/input/results-last/simple_res_orig_150.csv').drop(columns = ['Unnamed: 0'])


df_test['Calories'] = df_subm['Calories']


df_res['Resid'] = df_res['Calories'] - df_res['Prediction']
df_res['Abs_Res'] = df_res['Resid'].abs()
df_res['Rel_Res'] = df_res['Abs_Res'] / df_res['Calories'] * 100
df_res = df_res.sort_values(by = 'Abs_Res', ascending = False, ignore_index = True)


df_res.head(20)


df_train



for col in df_train.columns:
    plt.figure(figsize = (16, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    sns.lineplot(x = df_res[col], y = df_res['Abs_Res'], ax = ax1)
    sns.lineplot(x = df_res[col], y = df_res['Calories'], ax = ax2)


plt.figure(figsize = (16, 12))
sns.heatmap(df_res.corr(), annot = True, vmin = -1, vmax = 1)


from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline



pipeline_reg = Pipeline([('poly', PolynomialFeatures()),
                        ('linreg', LinearRegression())])


df_regres = df_res.copy()
df_res['Calories'] = np.log1p(df_res['Calories'])


df_regres.columns


params = {
    'poly__degree': np.arange(1, 11)
}


gridsearch = GridSearchCV(pipeline_reg, params, cv = 5, scoring = 'neg_root_mean_squared_error')


kf = KFold(n_splits = 5, shuffle = True, random_state = 42)


X = df_regres.drop(columns = ['Calories'])
y = df_regres['Calories']


reg_columns = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']


y


regresults = {col + '_reg': np.zeros(len(X)) for col in reg_columns}


regresults


for fold, (idx_train, idx_test) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_test, y_test = X.iloc[idx_test], y.iloc[idx_test]
    for col in reg_columns:
        gridsearch.fit(X_train[[col]], y_train)
        regresults[col + '_reg'][idx_test] = gridsearch.predict(X_test[[col]])


regresults


df_rg = pd.DataFrame(regresults)
df_regres = pd.concat([df_regres, df_rg], axis = 1)
df_regres.head(20)


sns.lineplot(x = df_res['Duration'] * df_res['Heart_Rate'] * df_res['Age'], y = df_res['Calories'])


sns.lineplot(x = abs(df_regres['Heart_Rate_reg'] - df_regres['Duration_reg']), y = df_regres['Abs_Res'])


sns.scatterplot(x = abs(df_regres['Heart_Rate_reg'] - df_regres['Duration_reg']), y = df_regres['Abs_Res'], hue = df_regres['Age_reg'], 
               palette = 'viridis', size = 1)


sns.lineplot(x = df_res['Duration'] + df_res['Heart_Rate'], y = df_res['Calories'])


sns.lineplot(x = df_res['Heart_Rate'], y = df_res['Calories'])


df_res['Age_reg'] = regresults['Age']
sns.lineplot(y = df_res['Calories'], x = df_res['Age'])
sns.lineplot(y = df_res['Age_reg'], x = df_res['Age'])


df_regres['Diff_HR_Duration'] = abs(df_regres['Heart_Rate_reg'] - df_regres['Duration_reg'])
df_regres['Intensity'] = df_regres['Heart_Rate'] / df_regres['Duration']
df_regres['Diff_HR_Age'] = abs(df_regres['Heart_Rate_reg'] - df_regres['Age_reg'])
df_regres['Diff_Duration_Age'] = abs(df_regres['Duration_reg'] - df_regres['Age_reg'])
df_regres['Diff_BT_HR'] = abs(df_regres['Body_Temp_reg'] - df_regres['Heart_Rate_reg'])
df_regres['Diff_Duration_BT'] = abs(df_regres['Duration_reg'] - df_regres['Body_Temp_reg'])
df_regres['Diff_Age_BT'] = abs(df_regres['Age_reg'] - df_regres['Body_Temp_reg'])
df_regres['Hard_Train_Index'] = abs(df_regres['Body_Temp'] / df_regres['Duration'])
df_regres['Hard_Train_Index2'] = abs(df_regres['Body_Temp'] / df_regres['Heart_Rate'])


df_posres = df_regres[df_regres['Resid'] > 0]
df_negres = df_regres[df_regres['Resid'] < 0]


df_posres.head(30)


df_negres.head(30)


plt.figure(figsize = (16, 8))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
sns.scatterplot(x = df_posres['Diff_BT_HR'], y = df_posres['Abs_Res'], hue = df_posres['Heart_Rate_reg'], ax = ax1,
               palette = 'viridis')
sns.scatterplot(x = df_negres['Diff_BT_HR'], y = df_negres['Abs_Res'], hue = df_negres['Heart_Rate_reg'], ax = ax2, palette = 'viridis')


plt.figure(figsize = (16, 8))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
sns.scatterplot(x = df_posres['Diff_Duration_BT'], y = df_posres['Abs_Res'], hue = df_posres['Heart_Rate_reg'], ax = ax1,
               palette = 'viridis')
sns.scatterplot(x = df_negres['Diff_Duration_BT'], y = df_negres['Abs_Res'], hue = df_negres['Heart_Rate_reg'], ax = ax2, palette = 'viridis')


plt.figure(figsize = (16, 8))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
sns.scatterplot(x = df_posres['Diff_Age_BT'], y = df_posres['Abs_Res'], hue = df_posres['Heart_Rate_reg'], ax = ax1,
               palette = 'viridis')
sns.scatterplot(x = df_negres['Diff_Age_BT'], y = df_negres['Abs_Res'], hue = df_negres['Heart_Rate_reg'], ax = ax2, palette = 'viridis')


plt.figure(figsize = (16, 8))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
sns.scatterplot(x = df_posres['Diff_HR_Duration'], y = df_posres['Abs_Res'], hue = df_posres['Heart_Rate_reg'], ax = ax1,
               palette = 'viridis')
sns.scatterplot(x = df_negres['Diff_HR_Duration'], y = df_negres['Abs_Res'], hue = df_negres['Heart_Rate_reg'], ax = ax2, palette = 'viridis')


plt.figure(figsize = (16, 8))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
sns.scatterplot(x = df_posres['Diff_Duration_Age'], y = df_posres['Abs_Res'], hue = df_posres['Diff_HR_Duration'], ax = ax1,
               palette = 'viridis')
sns.scatterplot(x = df_negres['Diff_Duration_Age'], y = df_negres['Abs_Res'], hue = df_negres['Diff_HR_Duration'], ax = ax2, palette = 'viridis')


plt.figure(figsize = (16, 8))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
sns.scatterplot(x = df_posres['Diff_HR_Age'], y = df_posres['Abs_Res'], hue = df_posres['Diff_HR_Duration'], ax = ax1,
               palette = 'viridis')
sns.scatterplot(x = df_negres['Diff_HR_Age'], y = df_negres['Abs_Res'], hue = df_negres['Diff_HR_Duration'], ax = ax2, palette = 'viridis')


plt.figure(figsize = (16, 8))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
sns.scatterplot(x = df_posres['Diff_HR_Duration'], y = df_posres['Abs_Res'], hue = df_posres['Duration_reg'], ax = ax1,
               palette = 'viridis')
sns.scatterplot(x = df_negres['Diff_HR_Duration'], y = df_negres['Abs_Res'], hue = df_negres['Duration_reg'], ax = ax2, palette = 'viridis')


plt.figure(figsize = (16, 8))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
sns.scatterplot(x = df_posres['Diff_HR_Duration'], y = df_posres['Abs_Res'], hue = df_posres['Intensity'], ax = ax1,
               palette = 'viridis')
sns.scatterplot(x = df_negres['Diff_HR_Duration'], y = df_negres['Abs_Res'], hue = df_negres['Intensity'], ax = ax2, palette = 'viridis')


df_regres[df_regres['Diff_HR_Duration'] < 50]


X_reg = pd.DataFrame()
X_reg['Intens'] = 1 / df_regres['Intensity']
X_reg['Intens2'] = 1 / df_regres['Intensity'] ** 2
X_reg['Hard'] = 1 / df_regres['Hard_Train_Index']
X_reg['Hard2'] = 1 / df_regres['Hard_Train_Index'] ** 2
y_reg = df_regres['Calories']
ridge = RidgeCV(alphas = np.linspace(0.01, 10, 20))
ridge.fit(X_reg, y_reg)


df_regres['Inverse_Predict'] = ridge.predict(X_reg)


sns.scatterplot(x = df_regres['Intensity'], y = df_regres['Calories'])
sns.scatterplot(x = df_regres['Intensity'], y = df_regres['Inverse_Predict'])


sns.scatterplot(x = df_regres['Hard_Train_Index'], y = df_regres['Calories'])
sns.scatterplot(x = df_regres['Hard_Train_Index'], y = df_regres['Inverse_Predict'])


sns.scatterplot(x = df_regres['Abs_Res'], y = abs(df_regres['Calories'] - df_regres['Inverse_Predict']))


sns.scatterplot(x = df_regres['Inverse_Predict'], y = abs(df_regres['Calories'] - df_regres['Inverse_Predict']), hue = df_regres['Abs_Res'], palette = 'viridis')


sns.scatterplot(x = df_regres['Prediction'], y = df_regres['Abs_Res'], hue = df_regres['Inverse_Predict'], palette = 'viridis')


sns.scatterplot(x = df_regres['Inverse_Predict'], y = df_regres['Abs_Res'])


sns.scatterplot(x = df_regres['Hard_Train_Index2'], y = df_regres['Calories'])

