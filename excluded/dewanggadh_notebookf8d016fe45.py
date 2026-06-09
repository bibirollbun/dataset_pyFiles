import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_sub = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
df_train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

df_train['train'] = True
df_test['train'] = False

df_y = df_train['diagnosed_diabetes'].copy()
df_test_id = df_test['id'].copy()

df_train.drop('diagnosed_diabetes', axis = 1, inplace = True)

df_combine = pd.concat([df_train, df_test], axis = 0).reset_index().copy()
df_combine.drop(['id', 'index'], axis = 1, inplace = True)


df_y.value_counts()


df_combine.shape


pd.concat([df_combine.dtypes, df_combine.nunique(), df_combine.isna().sum()], axis = 1).rename({
    0:'datatypes', 1:'unique', 2:'missing'
}, axis = 1)


booling = df_combine.nunique()[df_combine.nunique() <= 2].index
df_num = df_combine.select_dtypes([int, float, bool]).drop(booling, axis = 1).copy()
df_cat = df_combine.select_dtypes(object).copy()
df_bool = df_combine[booling].copy()


df_num.head(5)


df_cat.head(5)


df_bool.head(5)


df_num.describe()


plt.figure(figsize = (20, 15))
count = 0
for i in df_num.columns:
    count += 1
    plt.subplot(5, 3, count)
    plt.title(i)
    data = df_num[i].copy()
    binning = pd.cut(data, bins = 10, include_lowest = True, retbins = True)
    plt.hist(data, bins = binning[1])
    plt.xticks(binning[1])
plt.tight_layout()
plt.show()


df_num['age_disc'] = pd.cut(df_num['age'], bins = [18, 40, 50, 60, 100], labels = [0, 1, 2, 3]).astype(int)
df_num['alcohol_disc'] = pd.cut(df_num['alcohol_consumption_per_week'], bins = [0, 1, 2, 10], labels = [0, 1, 2]).astype(int)
df_num['phy_act_disc'] = pd.cut(df_num['physical_activity_minutes_per_week'], bins = [-1, 75, 800], labels = [0, 1]).astype(int)
df_num.drop(['age', 'alcohol_consumption_per_week','physical_activity_minutes_per_week'], axis = 1, inplace = True)
df_num['map_bp'] = (2 * df_num['diastolic_bp'] + df_num['systolic_bp']) / 3
df_num['cholesterol_total'] = df_num['hdl_cholesterol'] + df_num['ldl_cholesterol'] + (df_num['triglycerides'] * 0.2)


df_num['phy_act_disc']


pd.concat([(round(df_cat[i].value_counts() / len(df_cat) * 100, 2)).reset_index() for i in df_cat.columns], axis = 1).fillna('')


from sklearn.preprocessing import OrdinalEncoder


order = [['No formal', 'Highschool', 'Graduate', 'Postgraduate']]
ordered = OrdinalEncoder(categories = order).fit_transform(df_cat[['education_level']])
df_cat['education_level'] = ordered[:,0]

order = [['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High']]
ordered = OrdinalEncoder(categories = order).fit_transform(df_cat[['income_level']])
df_cat['income_level'] = ordered[:,0]


from sklearn.preprocessing import OneHotEncoder


onehot = OneHotEncoder(drop = 'first', sparse_output = False).fit(df_cat.select_dtypes(object))
col_name = onehot.get_feature_names_out()
onehot_df = pd.DataFrame(onehot.transform(df_cat.select_dtypes(object)), columns = col_name)


df_cat_2 = pd.concat([df_cat[['education_level', 'income_level']], onehot_df], axis = 1).copy()
df_cat_2.head(5)


df_all = pd.concat([df_num, df_cat_2, df_bool], axis = 1).copy()
df_all.head(5)


df_corring = pd.concat([df_all[df_all['train'] == True].reset_index().drop('index', axis = 1),
                        df_y.reset_index().drop('index', axis = 1)], axis = 1)
df_corring.head(5)


corred = df_corring.drop('train', axis = 1).corr()
plt.figure(figsize = (20, 20))
plt.imshow(corred, cmap = 'RdYlGn_r')
plt.xticks(ticks = [i for i in range(len(corred.columns))], labels = corred.columns, rotation = 45, ha = 'right')
plt.yticks(ticks = [i for i in range(len(corred.columns))], labels = corred.columns)
plt.xticks(ticks = [0.5 + i for i in range(len(corred.columns) - 1)], minor = True)
plt.yticks(ticks = [0.5 + i for i in range(len(corred.columns) - 1)], minor = True)
plt.grid(axis = 'both', which = 'minor', color = 'black', linewidth = 3)
plt.colorbar()
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier


X_train, X_temp, y_train, y_temp = train_test_split(df_all[df_all['train'] == True].drop('train', axis = 1),
                                                    df_y, test_size = 0.2, random_state = 99)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = 99)


learn_rate = [0.01, 0.1, 1, 10]
xgb_param = {
    'max_depth' : [3, 6, 12],
    'min_child_weight' : [1, 2, 4, 8],
    'gamma' : learn_rate,
    'subsample' : [0.1, 0.5, 0.8],
    'colsample_bytree' : [0.6, 0.7, 0.8, 0.9],
    'reg_lambda' : [1, 2, 4, 8],
    'reg_alpha' : [0, 0.01, 0.1, 1, 10],
    'eta' : learn_rate
}


df_all.dtypes


choose_xgb = RandomizedSearchCV(XGBClassifier(random_state = 99,
                                 n_estimators = 1000).set_params(early_stopping_rounds = 5),
                   xgb_param, random_state =99,
                   cv = KFold(n_splits = 5),
                  scoring = 'roc_auc').fit(X_train, y_train, eval_set = [(X_val, y_val)], verbose = 0)


choose_xgb.best_estimator_.predict(X_test)


from sklearn.metrics import roc_auc_score


roc_auc_score(y_test, choose_xgb.best_estimator_.predict(X_test))


res = choose_xgb.best_estimator_.predict(df_all[df_all['train'] == False].drop('train', axis = 1))


df_test_id


#pd.DataFrame(res, index = df_test_id, columns = ['diagnosed_diabetes']).to_csv('sub.csv')


df_bool.head(5)

