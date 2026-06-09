# Importing in packages and preparing config dict

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os 

# plotting
from matplotlib import pyplot as plt
import seaborn as sns

# linear modeling
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# random forest models
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

# classification
from sklearn.linear_model import LogisticRegression

# config dict
c = {'dir' : '/kaggle/input/playground-series-s5e7', 
    'out_dir' : '/kaggle/working/', 
    'out' : 'submission.csv'}

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        c.update({filename.replace(".csv", "") : filename})

c


DATA = pd.read_csv(os.path.join(c['dir'], c['train']))
print(f"The data was successfully read and is shape: {DATA.shape}")


print(f"the columns of DATA are:\n{DATA.columns}")


data = DATA.copy() # where DATA is the backup version, data is the working copy


data.head()


data.info()


data.describe()


# d_ prefix to notate a categorical / dimensional column
data['d_stage_fear_yes'] = np.where(data['Stage_fear'] == 'Yes', 1, 0)
data['d_stage_fear_no'] = np.where(data['Stage_fear'] == 'No', 1, 0)
data['d_stage_fear_decline'] = np.where(data['Stage_fear'].isna(), 1, 0)

data['d_drained_after_socializing_yes'] = np.where(data['Drained_after_socializing'] == 'Yes', 1, 0)
data['d_drained_after_socializing_no'] = np.where(data['Drained_after_socializing'] == 'No', 1, 0)
data['d_drained_after_socializing_decline'] = np.where(data['Drained_after_socializing'].isna(), 1, 0)

# prepare measures with new column names f_ 
data['f_time_spent_alone'] = data['Time_spent_Alone']
data['f_social_event_attendance'] = data['Social_event_attendance']
data['f_going_outside'] = data['Going_outside']
data['f_friends_circle_size'] = data['Friends_circle_size']
data['f_post_freq'] = data['Post_frequency']

data.head()


d_test = data.sample(frac = .2, random_state=19) # testing data
d = data.drop(d_test.index).filter(regex="^f_|d_|Per") # working data

out = f"the shape of the training dataset is: {d.shape}\n"
out += f"the test + train rows equals the original dataset number of rows: {d.shape[0] + d_test.shape[0] == data.shape[0]}"
print(out)


import warnings
warnings.filterwarnings('ignore')

d_f = d.filter(regex="^f_|Per")
graph_cols = ['f_time_spent_alone', 'f_social_event_attendance',
       'f_going_outside', 'f_friends_circle_size', 'f_post_freq']
sns.pairplot(d_f, vars=graph_cols, hue='Personality')


# evaluating the percent of missing rows by personality to answer the following question:
# do introverts have more missing data than extroverts?

from pandasql import sqldf

d_mis_pers = d_f[d_f.isna().any(axis=1)]
d_mis = pd.DataFrame(d_mis_pers['Personality'].value_counts())
d_f_pers = pd.DataFrame(d_f['Personality'].value_counts())

q = 'select dm.Personality, dm.count as num_missing, df.count as num_rows from d_mis as dm left join d_f_pers as df on dm.Personality = df.Personality'

miss_eval = sqldf(q)
miss_eval['perc_missing'] = miss_eval['num_missing'] / miss_eval['num_rows']
miss_eval


# data for imputations
d_missing_prep = d.filter(regex="^d_|f_")
d_missing_prep = d_missing_prep.dropna()
d_missing_prep.columns


# time spent alone (tsa)
tsa_drop_cols = ['f_time_spent_alone', 'd_stage_fear_no',
       'd_drained_after_socializing_no']
d_tsa_x = d_missing_prep.drop(columns=tsa_drop_cols)
d_tsa_y = d_missing_prep['f_time_spent_alone']

out = f"the shape of predictors is: {d_tsa_x.shape}\nthe shape of predicted value is {d_tsa_y.shape}"
print(out)


# compute VIF to determine multicollinearity

vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(d_tsa_x.values, i) for i in range(d_tsa_x.shape[1])]
vif['variable'] = d_tsa_x.columns
vif


d_tsa_x = d_tsa_x.drop(columns=['f_going_outside'])
tsa_drop_cols.append('f_going_outside')

vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(d_tsa_x.values, i) for i in range(d_tsa_x.shape[1])]
vif['variable'] = d_tsa_x.columns
vif


d_tsa_x = d_tsa_x.drop(columns=['f_social_event_attendance'])
tsa_drop_cols.append('f_social_event_attendance')

vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(d_tsa_x.values, i) for i in range(d_tsa_x.shape[1])]
vif['variable'] = d_tsa_x.columns
vif


d_tsa_x = d_tsa_x.drop(columns=['f_friends_circle_size'])
tsa_drop_cols.append('f_friends_circle_size')

vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(d_tsa_x.values, i) for i in range(d_tsa_x.shape[1])]
vif['variable'] = d_tsa_x.columns
vif


# log f_post_freq to accomodate for exponential outcome variable

d_tsa_x['f_post_freq'] = np.log(d_tsa_x['f_post_freq'] + 1)


# creating linear regression model to predict TSA
d_tsa_ols_train_x = d_tsa_x.sample(frac=.8, random_state=19)
d_tsa_ols_test_x = d_tsa_x.drop(d_tsa_ols_train_x.index)

d_tsa_ols_train_x = d_tsa_ols_train_x.sort_index()

d_tsa_ols_test_y = d_tsa_y.drop(d_tsa_ols_train_x.index)
d_tsa_ols_train_y = d_tsa_y.drop(d_tsa_ols_test_y.index)

d_tsa_ols_train_x = sm.add_constant(d_tsa_ols_train_x)
d_tsa_ols_test_x = sm.add_constant(d_tsa_ols_test_x)

model_tsa = sm.OLS(d_tsa_ols_train_y, d_tsa_ols_train_x).fit()

print(model_tsa.summary())


print(f"RMSE: {np.sqrt(model_tsa.mse_model)}")


# imputing missing values and evaluating the fit

tsa_test = pd.DataFrame(round(model_tsa.predict(d_tsa_ols_test_x)), columns=['preds'])
tsa_test_results = pd.concat([tsa_test, d_tsa_ols_test_y], axis=1)
tsa_test_results['difference_preds'] = np.abs(tsa_test_results['f_time_spent_alone'] - tsa_test_results['preds'])
tsa_test_results['mean'] = round(np.mean(d_tsa_ols_train_y))
tsa_test_results['median'] = round(np.median(d_tsa_ols_train_y))
tsa_test_results['difference_mean'] = np.abs(tsa_test_results['f_time_spent_alone'] - tsa_test_results['mean'])
tsa_test_results['difference_median'] = np.abs(tsa_test_results['f_time_spent_alone'] - tsa_test_results['median'])


sns.histplot(tsa_test_results['difference_preds'])


out = f"Mean imputation: The difference between the mean and the true value is: {round(np.mean(tsa_test_results['difference_mean']), 2)}\n"
out += f"Median imputation: The difference between the median and the true value is: {round(np.mean(tsa_test_results['difference_median']), 2)}\n"
out += f"Log regression Imputation: The difference between the predicted and true value is: {round(np.mean(tsa_test_results['difference_preds']), 2)}\n"
print(out)





# removing columns with multiple missing values
# then log transforming post frequency
# lastly, creating predictions for rows with only missing TSA values
d_missing_tsa = d[d['f_time_spent_alone'].isna()]
tsa_temp = d_missing_tsa.drop(columns=['f_time_spent_alone'])
tsa_temp = tsa_temp.dropna()

d_missing_tsa = d_missing_tsa.loc[d_missing_tsa.index.intersection(tsa_temp.index)]
d_missing_tsa = d_missing_tsa.filter(regex="^d_|f_")

d_tsa_exdog = d_missing_tsa.drop(columns=tsa_drop_cols)
d_tsa_exdog['f_post_freq'] = np.log(d_tsa_exdog['f_post_freq'] + 1) 
d_tsa_exdog = sm.add_constant(d_tsa_exdog)

tsa_preds = pd.DataFrame(round(model_tsa.predict(d_tsa_exdog)), columns=['pred_tsa'])
d_missing_tsa['f_time_spent_alone'] = tsa_preds


# friends circle size (fcs)
fcs_drop_cols = ['f_friends_circle_size', 'd_stage_fear_no', 'd_drained_after_socializing_no']
d_fcs_x = d_missing_prep.drop(columns=fcs_drop_cols)
d_fcs_y = d_missing_prep['f_friends_circle_size']

out = f"the shape of predictors is: {d_fcs_x.shape}\nthe shape of predicted value is {d_fcs_y.shape}"
print(out)


vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(d_fcs_x.values, i) for i in range(d_fcs_x.shape[1])]
vif['variable'] = d_fcs_x.columns
vif


d_fcs_x = d_fcs_x.drop(columns=['f_going_outside'])
fcs_drop_cols.append('f_going_outside')

vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(d_fcs_x.values, i) for i in range(d_fcs_x.shape[1])]
vif['variable'] = d_fcs_x.columns
vif


d_fcs_x = d_fcs_x.drop(columns=['f_social_event_attendance'])
fcs_drop_cols.append('f_social_event_attendance')

vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(d_fcs_x.values, i) for i in range(d_fcs_x.shape[1])]
vif['variable'] = d_fcs_x.columns
vif


# sqrt f_ columns
d_fcs_x['f_time_spent_alone'] = np.sqrt(d_fcs_x['f_time_spent_alone'])
d_fcs_x['f_post_freq'] = np.sqrt(d_fcs_x['f_post_freq'])


d_fcs_ols_train_x = d_fcs_x.sample(frac=.8, random_state=19)
d_fcs_ols_test_x = d_fcs_x.drop(d_fcs_ols_train_x.index)

d_fcs_ols_train_x = d_fcs_ols_train_x.sort_index()

d_fcs_ols_test_y = d_fcs_y.drop(d_fcs_ols_train_x.index)
d_fcs_ols_train_y = d_fcs_y.drop(d_fcs_ols_test_y.index)

d_fcs_ols_train_x = sm.add_constant(d_fcs_ols_train_x)
d_fcs_ols_test_x = sm.add_constant(d_fcs_ols_test_x)

model_fcs = sm.OLS(d_fcs_ols_train_y, d_fcs_ols_train_x).fit()

print(model_fcs.summary())


print(f"RMSE: {np.sqrt(model_fcs.mse_model)}")


fcs_test = pd.DataFrame(round(model_fcs.predict(d_fcs_ols_test_x)), columns=['preds'])

fcs_test_results = pd.concat([fcs_test, d_fcs_ols_test_y], axis=1)
fcs_test_results['difference_preds'] = np.abs(fcs_test_results['f_friends_circle_size'] - fcs_test_results['preds'])
fcs_test_results['mean'] = round(np.mean(d_fcs_ols_train_y))
fcs_test_results['difference_mean'] = np.abs(fcs_test_results['f_friends_circle_size'] - fcs_test_results['mean'])
fcs_test_results['median'] = round(np.median(d_fcs_ols_train_y))
fcs_test_results['difference_median'] = np.abs(fcs_test_results['f_friends_circle_size'] - fcs_test_results['median'])

out = f"Mean imputation: The difference between the mean and the true value is: {round(np.mean(fcs_test_results['difference_mean']), 2)}\n"
out += f"Median imputation: The difference between the median and the true value is: {round(np.mean(fcs_test_results['difference_median']), 2)}\n"
out += f"Sqrt regression Imputation: The difference between the predicted and true value is: {round(np.mean(fcs_test_results['difference_preds']), 2)}\n"
print(out)


sns.histplot(fcs_test_results['difference_preds'])


# removing columns with multiple missing values
# then sqrt-ing transforming time spent alone and post frequency
# lastly, creating predictions for rows with only missing FCS values

d_missing_fcs = d[d['f_friends_circle_size'].isna()]
fcs_temp = d_missing_fcs.drop(columns=['f_friends_circle_size'])
fcs_temp = fcs_temp.dropna()

d_missing_fcs = d_missing_fcs.loc[d_missing_fcs.index.intersection(fcs_temp.index)]
d_missing_fcs = d_missing_fcs.filter(regex="^d_|f_")

d_fcs_exdog = d_missing_fcs.drop(columns=fcs_drop_cols)
d_fcs_exdog['f_time_spent_alone'] = np.sqrt(d_fcs_exdog['f_time_spent_alone'])
d_fcs_exdog['f_post_freq'] = np.sqrt(d_fcs_exdog['f_post_freq'])
d_fcs_exdog = sm.add_constant(d_fcs_exdog)

fcs_preds = pd.DataFrame(round(model_fcs.predict(d_fcs_exdog)), columns=['pred_fcs'])
d_missing_fcs['f_friends_circle_size'] = fcs_preds


pf_drop_cols = ['f_post_freq', 'd_stage_fear_no', 'd_drained_after_socializing_no']
d_pf_x = d_missing_prep.drop(columns=pf_drop_cols)
d_pf_y = d_missing_prep['f_post_freq']

out = f"the shape of predictors is: {d_pf_x.shape}\nthe shape of predicted value is {d_pf_y.shape}"
print(out)


vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(d_pf_x.values, i) for i in range(d_pf_x.shape[1])]
vif['variable'] = d_pf_x.columns
vif


d_pf_x = d_pf_x.drop(columns=['f_social_event_attendance'])
pf_drop_cols.append('f_social_event_attendance')

vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(d_pf_x.values, i) for i in range(d_pf_x.shape[1])]
vif['variable'] = d_pf_x.columns
vif


d_pf_x = d_pf_x.drop(columns=['f_friends_circle_size'])
pf_drop_cols.append('f_friends_circle_size')

vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(d_pf_x.values, i) for i in range(d_pf_x.shape[1])]
vif['variable'] = d_pf_x.columns
vif


# sqrt f_ columns
d_pf_x['f_time_spent_alone'] = np.sqrt(d_pf_x['f_time_spent_alone'])
d_pf_x['f_going_outside'] = np.sqrt(d_pf_x['f_going_outside'])


d_pf_ols_train_x = d_pf_x.sample(frac=.8, random_state=19)
d_pf_ols_test_x = d_pf_x.drop(d_pf_ols_train_x.index)

d_pf_ols_train_x = d_pf_ols_train_x.sort_index()

d_pf_ols_test_y = d_pf_y.drop(d_pf_ols_train_x.index)
d_pf_ols_train_y = d_pf_y.drop(d_pf_ols_test_y.index)

d_pf_ols_train_x = sm.add_constant(d_pf_ols_train_x)
d_pf_ols_test_x = sm.add_constant(d_pf_ols_test_x)

model_pf = sm.OLS(d_pf_ols_train_y, d_pf_ols_train_x).fit()

print(model_pf.summary())


print(f"RMSE: {np.sqrt(model_pf.mse_model)}")


pf_test = pd.DataFrame(round(model_pf.predict(d_pf_ols_test_x)), columns=['preds'])

pf_test_results = pd.concat([pf_test, d_pf_ols_test_y], axis=1)
pf_test_results['difference_preds'] = np.abs(pf_test_results['f_post_freq'] - pf_test_results['preds'])
pf_test_results['mean'] = round(np.mean(d_pf_ols_train_y))
pf_test_results['difference_mean'] = np.abs(pf_test_results['f_post_freq'] - pf_test_results['mean'])
pf_test_results['median'] = round(np.median(d_pf_ols_train_y))
pf_test_results['difference_median'] = np.abs(pf_test_results['f_post_freq'] - pf_test_results['median'])

out = f"Mean imputation: The difference between the mean and the true value is: {round(np.mean(pf_test_results['difference_mean']), 2)}\n"
out += f"Median imputation: The difference between the median and the true value is: {round(np.mean(pf_test_results['difference_median']), 2)}\n"
out += f"Sqrt regression Imputation: The difference between the predicted and true value is: {round(np.mean(pf_test_results['difference_preds']), 2)}\n"
print(out)


sns.histplot(pf_test_results['difference_preds'])


# removing columns with multiple missing values
# then sqrt-ing transforming time spent alone and post frequency
# lastly, creating predictions for rows with only missing PF values

d_missing_pf = d[d['f_post_freq'].isna()]
pf_temp = d_missing_pf.drop(columns=['f_post_freq'])
pf_temp = pf_temp.dropna()

d_missing_pf = d_missing_pf.loc[d_missing_pf.index.intersection(pf_temp.index)]
d_missing_pf = d_missing_pf.filter(regex="^d_|f_")

d_pf_exdog = d_missing_pf.drop(columns=pf_drop_cols)
d_pf_exdog['f_time_spent_alone'] = np.sqrt(d_pf_exdog['f_time_spent_alone'])
d_pf_exdog['f_going_outside'] = np.sqrt(d_pf_exdog['f_going_outside'])
d_pf_exdog = sm.add_constant(d_pf_exdog)

pf_preds = pd.DataFrame(round(model_pf.predict(d_pf_exdog)), columns=['pred_pf'])
d_missing_pf['f_post_freq'] = pf_preds


sea_drop_cols = ['f_social_event_attendance', 'd_stage_fear_no', 'd_drained_after_socializing_no']
d_sea_x = d_missing_prep.drop(columns=sea_drop_cols)
d_sea_y = d_missing_prep['f_social_event_attendance']

out = f"the shape of predictors is: {d_sea_x.shape}\nthe shape of predicted value is {d_sea_y.shape}"
print(out)


vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(d_sea_x.values, i) for i in range(d_sea_x.shape[1])]
vif['variable'] = d_sea_x.columns
vif


d_sea_x = d_sea_x.drop(columns=['f_friends_circle_size'])
sea_drop_cols.append('f_friends_circle_size')

vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(d_sea_x.values, i) for i in range(d_sea_x.shape[1])]
vif['variable'] = d_sea_x.columns
vif


d_sea_x = d_sea_x.drop(columns=['f_going_outside'])
sea_drop_cols.append('f_going_outside')

vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(d_sea_x.values, i) for i in range(d_sea_x.shape[1])]
vif['variable'] = d_sea_x.columns
vif


d_sea_ols_train_x = d_sea_x.sample(frac=.8, random_state=19)
d_sea_ols_test_x = d_sea_x.drop(d_sea_ols_train_x.index)

d_sea_ols_train_x = d_sea_ols_train_x.sort_index()

d_sea_ols_test_y = d_sea_y.drop(d_sea_ols_train_x.index)
d_sea_ols_train_y = d_sea_y.drop(d_sea_ols_test_y.index)

d_sea_ols_train_x = sm.add_constant(d_sea_ols_train_x)
d_sea_ols_test_x = sm.add_constant(d_sea_ols_test_x)

model_sea = sm.OLS(d_sea_ols_train_y, d_sea_ols_train_x).fit()

print(model_sea.summary())


print(f"RMSE: {np.sqrt(model_sea.mse_model)}")


sea_test = pd.DataFrame(round(model_sea.predict(d_sea_ols_test_x)), columns=['preds'])

sea_test_results = pd.concat([sea_test, d_sea_ols_test_y], axis=1)
sea_test_results['difference_preds'] = np.abs(sea_test_results['f_social_event_attendance'] - sea_test_results['preds'])
sea_test_results['mean'] = round(np.mean(d_sea_ols_train_y))
sea_test_results['difference_mean'] = np.abs(sea_test_results['f_social_event_attendance'] - sea_test_results['mean'])
sea_test_results['median'] = round(np.median(d_sea_ols_train_y))
sea_test_results['difference_median'] = np.abs(sea_test_results['f_social_event_attendance'] - sea_test_results['median'])

out = f"Mean imputation: The difference between the mean and the true value is: {round(np.mean(sea_test_results['difference_mean']), 2)}\n"
out += f"Median imputation: The difference between the median and the true value is: {round(np.mean(sea_test_results['difference_median']), 2)}\n"
out += f"Linear regression Imputation: The difference between the predicted and true value is: {round(np.mean(sea_test_results['difference_preds']), 2)}\n"
print(out)


sns.histplot(fcs_test_results['difference_preds'])


d_missing_sea = d[d['f_social_event_attendance'].isna()]
sea_temp = d_missing_sea.drop(columns=['f_social_event_attendance'])
sea_temp = sea_temp.dropna()

d_missing_sea = d_missing_sea.loc[d_missing_sea.index.intersection(sea_temp.index)]
d_missing_sea = d_missing_sea.filter(regex="^d_|f_")

d_sea_exdog = d_missing_sea.drop(columns=sea_drop_cols)
d_sea_exdog = sm.add_constant(d_sea_exdog)

sea_preds = pd.DataFrame(round(model_sea.predict(d_sea_exdog)), columns=['pred_sea'])
d_missing_sea['f_social_event_attendance'] = sea_preds


# going outside (go)
go_drop_cols = ['f_going_outside', 'd_stage_fear_no', 'd_drained_after_socializing_no']
d_go_x = d_missing_prep.drop(columns=go_drop_cols)
d_go_y = d_missing_prep['f_going_outside']

out = f"the shape of predictors is: {d_go_x.shape}\nthe shape of predicted value is {d_go_y.shape}"
print(out)


vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(d_go_x.values, i) for i in range(d_go_x.shape[1])]
vif['variable'] = d_go_x.columns
vif


d_go_x = d_go_x.drop(columns=['f_social_event_attendance'])
go_drop_cols.append('f_social_event_attendance')

vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(d_go_x.values, i) for i in range(d_go_x.shape[1])]
vif['variable'] = d_go_x.columns
vif


d_go_x = d_go_x.drop(columns=['f_friends_circle_size'])
go_drop_cols.append('f_friends_circle_size')

vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(d_go_x.values, i) for i in range(d_go_x.shape[1])]
vif['variable'] = d_go_x.columns
vif


d_go_ols_train_x = d_go_x.sample(frac=.8, random_state=19)
d_go_ols_test_x = d_go_x.drop(d_go_ols_train_x.index)

d_go_ols_train_x = d_go_ols_train_x.sort_index()

d_go_ols_test_y = d_go_y.drop(d_go_ols_train_x.index)
d_go_ols_train_y = d_go_y.drop(d_go_ols_test_y.index)

d_go_ols_train_x = sm.add_constant(d_go_ols_train_x)
d_go_ols_test_x = sm.add_constant(d_go_ols_test_x)

model_go = sm.OLS(d_go_ols_train_y, d_go_ols_train_x).fit()

print(model_go.summary())


print(f"RMSE: {np.sqrt(model_go.mse_model)}")


go_test = pd.DataFrame(round(model_go.predict(d_go_ols_test_x)), columns=['preds'])

go_test_results = pd.concat([go_test, d_go_ols_test_y], axis=1)
go_test_results['difference_preds'] = np.abs(go_test_results['f_going_outside'] - go_test_results['preds'])
go_test_results['mean'] = round(np.mean(d_go_ols_train_y))
go_test_results['difference_mean'] = np.abs(go_test_results['f_going_outside'] - go_test_results['mean'])
go_test_results['median'] = round(np.median(d_go_ols_train_y))
go_test_results['difference_median'] = np.abs(go_test_results['f_going_outside'] - go_test_results['median'])

out = f"Mean imputation: The difference between the mean and the true value is: {round(np.mean(go_test_results['difference_mean']), 2)}\n"
out += f"Median imputation: The difference between the median and the true value is: {round(np.mean(go_test_results['difference_median']), 2)}\n"
out += f"Linear regression Imputation: The difference between the predicted and true value is: {round(np.mean(go_test_results['difference_preds']), 2)}\n"
print(out)




d_missing_go = d[d['f_going_outside'].isna()]
go_temp = d_missing_go.drop(columns=['f_going_outside'])
go_temp = go_temp.dropna()

d_missing_go = d_missing_go.loc[d_missing_go.index.intersection(go_temp.index)]
d_missing_go = d_missing_go.filter(regex="^d_|f_")

d_go_exdog = d_missing_go.drop(columns=go_drop_cols)
d_go_exdog = sm.add_constant(d_go_exdog)

go_preds = pd.DataFrame(round(model_go.predict(d_go_exdog)), columns=['pred_go'])
d_missing_go['f_going_outside'] = go_preds



d_imputed = pd.concat(
    [d_missing_prep,
    d_missing_tsa,
    d_missing_fcs,
    d_missing_go,
    d_missing_pf, 
    d_missing_sea]
)

d_imputed = d_imputed.sort_index()
d_imputed


d_train_y = data['Personality'].loc[data.index.intersection(d_imputed.index)]
d_train_y = d_train_y.sort_index()

model_p_1 = LogisticRegression(random_state=19).fit(d_imputed, d_train_y)
model_p_1.score(d_imputed, d_train_y)


d_test_missing_prep = d_test.dropna().filter(regex="^d_|f_")


# impute tsa preds for test data

# removing columns with multiple missing values
d_test_missing_tsa = d_test[d_test['f_time_spent_alone'].isna()]
t_tsa_temp = d_test_missing_tsa.drop(columns=['f_time_spent_alone', 'Time_spent_Alone'])
t_tsa_temp = t_tsa_temp.dropna()

d_test_missing_tsa = d_test_missing_tsa.loc[d_test_missing_tsa.index.intersection(t_tsa_temp.index)]
d_test_missing_tsa = d_test_missing_tsa.filter(regex="^d_|f_")

# prep exdog for model

d_test_tsa_exdog = d_test_missing_tsa.drop(columns=['d_stage_fear_no','d_drained_after_socializing_no',
    'f_time_spent_alone','f_social_event_attendance', 
    'f_going_outside', 'f_friends_circle_size'])
d_test_tsa_exdog = sm.add_constant(d_test_tsa_exdog)

t_tsa_preds = pd.DataFrame(round(model_tsa.predict(d_test_tsa_exdog)), columns=['pred_tsa'])
d_test_missing_tsa['f_time_spent_alone'] = t_tsa_preds



# imputing sea preds
# preparing actual predictions
# removing columns with multiple missing values

d_test_missing_sea = d_test[d_test['f_social_event_attendance'].isna()]
t_sea_temp = d_test_missing_sea.drop(columns=['f_social_event_attendance', 'Social_event_attendance'])
t_sea_temp = t_sea_temp.dropna()

d_test_missing_sea = d_test_missing_sea.loc[d_test_missing_sea.index.intersection(t_sea_temp.index)]
d_test_missing_sea = d_test_missing_sea.filter(regex="^d_|f_")

# prep exdog for model

d_test_sea_exdog = d_test_missing_sea.drop(columns=sea_drop_cols)
d_test_sea_exdog = sm.add_constant(d_test_sea_exdog)

t_sea_preds = pd.DataFrame(round(model_sea.predict(d_test_sea_exdog)), columns=['pred_sea'])
d_test_missing_sea['f_social_event_attendance'] = t_sea_preds


# imputing pf
# preparing actual predictions
# removing columns with multiple missing values

d_test_missing_pf = d_test[d_test['f_post_freq'].isna()]
t_pf_temp = d_test_missing_pf.drop(columns=['f_post_freq', 'Post_frequency'])
t_pf_temp = t_pf_temp.dropna()

d_test_missing_pf = d_test_missing_pf.loc[d_test_missing_pf.index.intersection(t_pf_temp.index)]
d_test_missing_pf = d_test_missing_pf.filter(regex="^d_|f_")

# prep exdog for model

d_test_pf_exdog = d_test_missing_pf.drop(columns=pf_drop_cols)
d_test_pf_exdog = sm.add_constant(d_test_pf_exdog)

t_pf_preds = pd.DataFrame(round(model_pf.predict(d_test_pf_exdog)), columns=['pred_pf'])
d_test_missing_pf['f_post_freq'] = t_pf_preds


# imputing go
# preparing actual predictions
# removing columns with multiple missing values

d_test_missing_go = d_test[d_test['f_going_outside'].isna()]
t_go_temp = d_test_missing_go.drop(columns=['f_going_outside', 'Going_outside'])
t_go_temp = t_go_temp.dropna()

d_test_missing_go = d_test_missing_go.loc[d_test_missing_go.index.intersection(t_go_temp.index)]
d_test_missing_go = d_test_missing_go.filter(regex="^d_|f_")

# prep exdog for model

d_test_go_exdog = d_test_missing_go.drop(columns=go_drop_cols)
d_test_go_exdog = sm.add_constant(d_test_go_exdog)

t_go_preds = pd.DataFrame(round(model_go.predict(d_test_go_exdog)), columns=['pred_go'])
d_test_missing_go['f_going_outside'] = t_go_preds



# imputing fcs
# preparing actual predictions
# removing columns with multiple missing values

d_test_missing_fcs = d_test[d_test['f_friends_circle_size'].isna()]
t_fcs_temp = d_test_missing_fcs.drop(columns=['f_friends_circle_size', 'Friends_circle_size'])
t_fcs_temp = t_fcs_temp.dropna()

d_test_missing_fcs = d_test_missing_fcs.loc[d_test_missing_fcs.index.intersection(t_fcs_temp.index)]
d_test_missing_fcs = d_test_missing_fcs.filter(regex="^d_|f_")

# prep exdog for model

d_test_fcs_exdog = d_test_missing_fcs.drop(columns=fcs_drop_cols)
d_test_fcs_exdog = sm.add_constant(d_test_fcs_exdog)

t_fcs_preds = pd.DataFrame(round(model_fcs.predict(d_test_fcs_exdog)), columns=['pred_fcs'])
d_test_missing_fcs['f_friends_circle_size'] = t_fcs_preds


# impute test values
# make testing vector

d_test_imputed = pd.concat(
    [d_test_missing_prep,
    d_test_missing_tsa,
    d_test_missing_fcs,
    d_test_missing_go,
    d_test_missing_pf, 
    d_test_missing_sea]
)

d_test_imputed = d_test_imputed.sort_index()

d_test_y = d_test['Personality'].loc[d_test.index.intersection(d_test_imputed.index)]
d_test_y = d_test_y.sort_index()

model_p_1.score(d_test_imputed, d_test_y)


S_DATA = pd.read_csv(os.path.join(c['dir'], c['test']))
s_data = S_DATA.copy()


# then transform stage fear and drained after socializing into flags
# d_ prefix to notate a categorical / dimensional column
s_data['d_stage_fear_yes'] = np.where(s_data['Stage_fear'] == 'Yes', 1, 0)
s_data['d_stage_fear_no'] = np.where(s_data['Stage_fear'] == 'No', 1, 0)
s_data['d_stage_fear_decline'] = np.where(s_data['Stage_fear'].isna(), 1, 0)

s_data['d_drained_after_socializing_yes'] = np.where(s_data['Drained_after_socializing'] == 'Yes', 1, 0)
s_data['d_drained_after_socializing_no'] = np.where(s_data['Drained_after_socializing'] == 'No', 1, 0)
s_data['d_drained_after_socializing_decline'] = np.where(s_data['Drained_after_socializing'].isna(), 1, 0)

# prepare measures with new column names f_ 
s_data['f_time_spent_alone'] = s_data['Time_spent_Alone']
s_data['f_social_event_attendance'] = s_data['Social_event_attendance']
s_data['f_going_outside'] = s_data['Going_outside']
s_data['f_friends_circle_size'] = s_data['Friends_circle_size']
s_data['f_post_freq'] = s_data['Post_frequency']

# create working copy of dataset
s_d = s_data.filter(regex="^d_|f_")


s_missing_prep = s_d.dropna()


# impute tsa preds for test data

# removing columns with multiple missing values
s_missing_tsa = s_d[s_d['f_time_spent_alone'].isna()]
s_tsa_temp = s_missing_tsa.drop(columns=['f_time_spent_alone'])
s_tsa_temp = s_tsa_temp.dropna()

s_missing_tsa = s_missing_tsa.loc[s_missing_tsa.index.intersection(s_tsa_temp.index)]
s_missing_tsa = s_missing_tsa.filter(regex="^d_|f_")

# prep exdog for model
s_tsa_exdog = s_missing_tsa.drop(columns=tsa_drop_cols)
s_tsa_exdog = sm.add_constant(s_tsa_exdog)

s_tsa_preds = pd.DataFrame(round(model_tsa.predict(s_tsa_exdog)), columns=['pred_tsa'])
s_missing_tsa['f_time_spent_alone'] = s_tsa_preds


# imputing sea preds

s_missing_sea = s_d[s_d['f_social_event_attendance'].isna()]
s_sea_temp = s_missing_sea.drop(columns=['f_social_event_attendance'])
s_sea_temp = s_sea_temp.dropna()

s_missing_sea = s_missing_sea.loc[s_missing_sea.index.intersection(s_sea_temp.index)]
s_missing_sea = s_missing_sea.filter(regex="^d_|f_")

# prep exdog for model
s_sea_exdog = s_missing_sea.drop(columns=sea_drop_cols)
s_sea_exdog = sm.add_constant(s_sea_exdog)

s_sea_preds = pd.DataFrame(round(model_sea.predict(s_sea_exdog)), columns=['pred_sea'])
s_missing_sea['f_social_event_attendance'] = s_sea_preds


# imputing pf
s_missing_pf = s_d[s_d['f_post_freq'].isna()]
s_pf_temp = s_missing_pf.drop(columns=['f_post_freq'])
s_pf_temp = s_pf_temp.dropna()

s_missing_pf = s_missing_pf.loc[s_missing_pf.index.intersection(s_pf_temp.index)]
s_missing_pf = s_missing_pf.filter(regex="^d_|f_")

# prep exdog for model

s_pf_exdog = s_missing_pf.drop(columns=pf_drop_cols)
s_pf_exdog = sm.add_constant(s_pf_exdog)

s_pf_preds = pd.DataFrame(round(model_pf.predict(s_pf_exdog)), columns=['pred_pf'])
s_missing_pf['f_post_freq'] = s_pf_preds


# imputing go

s_missing_go = s_d[s_d['f_going_outside'].isna()]
s_go_temp = s_missing_go.drop(columns=['f_going_outside'])
s_go_temp = s_go_temp.dropna()

s_missing_go = s_missing_go.loc[s_missing_go.index.intersection(s_go_temp.index)]
s_missing_go = s_missing_go.filter(regex="^d_|f_")

# prep exdog for model

s_go_exdog = s_missing_go.drop(columns=go_drop_cols)
s_go_exdog = sm.add_constant(s_go_exdog)

s_go_preds = pd.DataFrame(round(model_go.predict(s_go_exdog)), columns=['pred_go'])
s_missing_go['f_going_outside'] = s_go_preds


# imputing fcs

s_missing_fcs = s_d[s_d['f_friends_circle_size'].isna()]
s_fcs_temp = s_missing_fcs.drop(columns=['f_friends_circle_size'])
s_fcs_temp = s_fcs_temp.dropna()

s_missing_fcs = s_missing_fcs.loc[s_missing_fcs.index.intersection(s_fcs_temp.index)]
s_missing_fcs = s_missing_fcs.filter(regex="^d_|f_")

# prep exdog for model

s_fcs_exdog = s_missing_fcs.drop(columns=fcs_drop_cols)
s_fcs_exdog = sm.add_constant(s_fcs_exdog)

s_fcs_preds = pd.DataFrame(round(model_fcs.predict(s_fcs_exdog)), columns=['pred_fcs'])
s_missing_fcs['f_friends_circle_size'] = s_fcs_preds


# impute for test values

s_imputed = pd.concat(
    [s_missing_prep,
    s_missing_tsa,
    s_missing_fcs,
    s_missing_go,
    s_missing_pf, 
    s_missing_sea]
)

s_imputed = s_imputed.sort_index()


# imputing values for rows with multiple missing values
s_multi_na = s_d.drop(index=s_imputed.index)

s_tsa_pf = s_multi_na[s_multi_na['f_time_spent_alone'].isna() & s_multi_na['f_post_freq'].isna()] 
s_tsa_pf_temp = s_multi_na.drop(index=s_tsa_pf.index)

s_tsa_pf['f_time_spent_alone'] = round(np.median(d_tsa_ols_test_x))
s_tsa_pf['f_post_freq'] = round(np.median(d_pf_ols_test_x))
s_tsa_pf['f_social_event_attendance'][s_tsa_pf['f_social_event_attendance'].isna()] = round(np.median(d_sea_ols_test_x))

# reunite the missing values dset
s_multi_na = pd.concat(
    [s_tsa_pf, 
     s_tsa_pf_temp]
)


# predicting tsa values without missing pf values

s_tsa_multi = s_multi_na[s_multi_na['f_time_spent_alone'].isna() & ~s_multi_na['f_post_freq'].isna()]
s_tsa_temp = s_multi_na.drop(s_tsa_multi.index)

s_tsa_exdog_2 = s_tsa_multi.drop(columns=['d_stage_fear_no','d_drained_after_socializing_no',
    'f_time_spent_alone','f_social_event_attendance', 
    'f_going_outside', 'f_friends_circle_size'])
s_tsa_exdog_2 = sm.add_constant(s_tsa_exdog_2)
# d_tsa_exdog

s_tsa_preds = pd.DataFrame(round(model_tsa.predict(s_tsa_exdog_2)), columns=['pred_tsa'])
s_tsa_multi['f_time_spent_alone'] = s_tsa_preds

# reunite the missing values dset
s_multi_na = pd.concat(
    [s_tsa_multi, 
     s_tsa_temp]
)


# now predict pf becasue all of tsa is imputed
# prep exdog for model

s_pf_multi = s_multi_na[s_multi_na['f_post_freq'].isna()]
s_pf_temp = s_multi_na.drop(s_pf_multi.index)

s_pf_exdog_2 = s_pf_multi.drop(columns=pf_drop_cols)
s_pf_exdog_2 = sm.add_constant(s_pf_exdog_2)

# since all pf values in this case are dependent on a predictor, going outside, I am going to perform median imputation
s_pf_multi['f_post_freq'] = round(np.median(d_pf_ols_test_x))

s_multi_na = pd.concat(
    [s_pf_multi,
     s_pf_temp]
)


# now predict go because go is dependent on tsa and pf

s_go_multi = s_multi_na[s_multi_na['f_going_outside'].isna()]
s_go_temp = s_multi_na.drop(s_go_multi.index)

# prep exdog for model

s_go_exdog_2 = s_go_multi.drop(columns=go_drop_cols)
s_go_exdog_2 = sm.add_constant(s_go_exdog_2)

s_go_preds = pd.DataFrame(round(model_go.predict(s_go_exdog_2)), columns=['pred_go'])
s_go_multi['f_going_outside'] = s_go_preds

# reunite dsets
s_multi_na = pd.concat(
    [s_go_multi,
     s_go_temp]
)


# now predict sea because tsa and pf have been imputed
s_sea_multi = s_multi_na[s_multi_na['f_social_event_attendance'].isna()]
s_sea_temp = s_multi_na.drop(s_sea_multi.index)

# prep exdog for model

s_sea_exdog_2 = s_sea_multi.drop(columns=sea_drop_cols)
s_sea_exdog_2 = sm.add_constant(s_sea_exdog_2)

s_sea_preds = pd.DataFrame(round(model_sea.predict(s_sea_exdog_2)), columns=['pred_sea'])
s_sea_multi['f_social_event_attendance'] = s_sea_preds

# reunite dsets
s_multi_na = pd.concat(
    [s_sea_multi,
     s_sea_temp]
)


# now predict fcs because tsa and pf have been imputed
s_fcs_multi = s_multi_na[s_multi_na['f_friends_circle_size'].isna()]
s_fcs_temp = s_multi_na.drop(s_fcs_multi.index)

# prep exdog for model

s_fcs_exdog_2 = s_fcs_multi.drop(columns=fcs_drop_cols)
s_fcs_exdog_2 = sm.add_constant(s_fcs_exdog_2)

s_fcs_preds = pd.DataFrame(round(model_fcs.predict(s_fcs_exdog_2)), columns=['pred_fcs'])
s_fcs_multi['f_friends_circle_size'] = s_fcs_preds

# reunite dsets
s_multi_na = pd.concat(
    [s_fcs_multi,
     s_fcs_temp]
)


# unite full dataset
s_full = pd.concat(
    [s_imputed,
    s_multi_na]
)

s_full = s_full.sort_index()

not_missing = s_full.dropna()
still_missing = s_full.drop(index=not_missing.index)

out = f"The shape of not missing data is: {not_missing.shape}"
out += f"\nThe num. rows that still have missing data: {still_missing.shape[0]}"
print(out)



final_results = pd.DataFrame(s_data["id"], columns=["id"])
final_results["Personality"] = model_p_1.predict(s_full)
out = f"The shape of the final output is: {final_results.shape}"
print(out)


final_results.head(10)


final_results.to_csv(os.path.join(c['out_dir'], c['out']), index=False)




