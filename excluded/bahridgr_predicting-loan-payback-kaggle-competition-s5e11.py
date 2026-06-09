
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings


pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 500)
warnings.simplefilter(action='ignore', category=Warning)


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')



def check_data(dataframe):
    print("########################## HEAD ##########################")
    print(dataframe.head())
    print("########################## INFO ##########################")
    print(dataframe.info())
    print("########################## SHAPE ##########################")
    print(dataframe.shape)
    print("########################## ISNULL(?) ##########################")
    print(dataframe.isnull().sum())
    print("########################## DESCRIBE ##########################")
    print(dataframe.describe().T)
    print("####################################################")


print("################ Train Data İnfo  ###################")
check_data(train)


print("################Test data İnfo  ###################")
check_data(test)


target_col = 'loan_paid_back'
print(train[target_col].value_counts())
sns.countplot(x=target_col, data=train)
plt.show()


def grab_col_names(dataframe, cat_th=16, car_th=25):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


print("-----Train Data---------")
train_cat_cols, train_num_cols, train_cat_but_car = grab_col_names(train)


print("-----Test Data---------")
test_cat_cols, test_num_cols, test_cat_but_car = grab_col_names(test)


# Eğer hedef değişken de train_num_cols içinde ise çıkar
if target_col in train_cat_cols:
    train_cat_cols.remove(target_col)


print("Train Numerical cols:", train_num_cols)
print("Train Categorical cols:", train_cat_cols)


def outlier_thresholds(dataframe, variable, q1=0.01, q3=0.99):
    quartile1 = dataframe[variable].quantile(q1)
    quartile3 = dataframe[variable].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr
    return low_limit, up_limit

def check_outliers(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable] < low_limit) | (dataframe[variable] > up_limit)].any(axis=None):
        return True
    else:
        return False


print("----- Train------")
for col in train_num_cols:
    print(col, check_outliers(train,col))


print("----- Test------")
for col in test_num_cols:
    print(col, check_outliers(test,col))


def replace_with_threshold(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


print("-----Train-----")
for col in train_num_cols:
    if check_outliers(train,col):
        replace_with_threshold(train, col)

for col in train_num_cols:
    print(col, check_outliers(train,col))


print("-----Test-----")
for col in train_num_cols:
    if check_outliers(train,col):
        replace_with_threshold(train, col)

for col in test_num_cols:
    print(col, check_outliers(test,col))


train.isna().sum()


test.isna().sum()


train["NEW_loan_interest"] = train["loan_amount"] * train["interest_rate"] / 100

train["NEW_income_to_lo_ratio"] = train["annual_income"] / (train["NEW_loan_interest"] + 1)
train["NEW_loan_to_income_ratio"] = train["loan_amount"] / (train["annual_income"] + 1)

train["NEW_credit_risk"] = train["credit_score"] / (train["debt_to_income_ratio"] + 1)


train.head()


test["NEW_loan_interest"] = test["loan_amount"] * test["interest_rate"] / 100

test["NEW_income_to_lo_ratio"] = test["annual_income"] / (test["NEW_loan_interest"] + 1)
test["NEW_loan_to_income_ratio"] = test["loan_amount"] / (test["annual_income"] + 1)

test["NEW_credit_risk"] = test["credit_score"] / (test["debt_to_income_ratio"] + 1)


test.head()


print("------ Train------")
train_enc = pd.get_dummies(train[train_cat_cols], drop_first=True)
train_enc.head()


print("------ Test------")
test_enc = pd.get_dummies(test[test_cat_cols], drop_first=True)
test_enc.head()


# one-hot sonrası train/test’in sütun yapısını eşitleyelim:
train_enc, test_enc = train_enc.align(test_enc, join='left', axis=1, fill_value=0)


train_model = pd.concat([train[train_num_cols], train_enc], axis=1)
test_model = pd.concat([test[test_num_cols], test_enc], axis=1)


train_model.head()


y = train[target_col].copy()


scaler = StandardScaler()
train_model_scaled = scaler.fit_transform(train_model)
test_model_scaled = scaler.transform(test_model)


kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)  # eğer sınıflandırma ise
scores = cross_val_score(lgb_model, train_model_scaled, y, cv=kf, scoring='roc_auc')
print("LightGBM CV ROC AUC: ", np.mean(scores), np.std(scores))



from sklearn.model_selection import GridSearchCV

param_grid = {
    'num_leaves': [31, 50],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200]
}

grid = GridSearchCV(lgb_model, param_grid, cv=kf, scoring='roc_auc', verbose=1)
grid.fit(train_model_scaled, y)
print("Best params:", grid.best_params_)
print("Best CV score:", grid.best_score_)


# Best params: {'learning_rate': 0.1, 'n_estimators': 200, 'num_leaves': 50}
# Best CV score: 0.9211285418324794


best_model = grid.best_estimator_
preds = best_model.predict_proba(test_model_scaled)[:,1]


submission = pd.DataFrame({
    'id': test['id'],
    'target': preds
})
submission.to_csv('submission.csv', index=False)




