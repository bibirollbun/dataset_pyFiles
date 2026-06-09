import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import scipy.stats as stats      # Для автоматизации тестирвоания гипотез распределений
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import PrecisionRecallDisplay, f1_score
seed = 0
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
import ast
import os
pwd = "/kaggle/input/playground-series-s4e10"
path_train = os.path.join(pwd, "train.csv")
path_test = os.path.join(pwd, "test.csv")
path_submission = os.path.join(pwd, "sample_submission.csv")
from sklearn.model_selection import train_test_split


!pip install woe_iv_bin==0.1.2 -q -q -q


from woe_iv_bin import categorical_woe
from woe_iv_bin import continuous_woe


def interval_type(s):
    """Parse interval string to Interval"""
    table = str.maketrans({'[': '(', ']': ')'})
    left_closed = s.startswith('[')
    right_closed = s.endswith(']')
    left, right = ast.literal_eval(s.translate(table))
    t = 'neither'
    if left_closed and right_closed:
        t = 'both'
    elif left_closed:
        t = 'left'
    elif right_closed:
        t = 'right'
    return pd.Interval(left, right, closed=t)


def apply_bounds_open(series_interval):
    series_interval = list(series_interval)
    series_interval[0] = pd.Interval(left = -np.inf, right=series_interval[0].right, closed='right')
    series_interval[-1] = pd.Interval(left = series_interval[-1].left, right=np.inf)
    return series_interval


def woe_generate_col(series, binning_woe_results):
    result_bins = apply_bounds_open(binning_woe_results['optimized_bin'].apply(interval_type))
    series = pd.cut(series, result_bins)
    return series



class DataPrepTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        numeric_columns = [
            'person_age',
            'person_income',
            'person_emp_length',
            'loan_amnt',
            'loan_int_rate',
            'cb_person_cred_hist_length',
        ]
        cat_cols = [
            'person_home_ownership',
            'loan_intent',
            'cb_person_default_on_file',
        ]
        X['cred_hist_to_age_ratio'] = X.cb_person_cred_hist_length / X.person_age
        X['loan_amnt_to_cred_hist'] = X.loan_amnt / X.cb_person_cred_hist_length
        numeric_columns.append('loan_amnt_to_cred_hist')
        numeric_columns.append('cred_hist_to_age_ratio')
        X['person_home_ownership'] = (
            X['person_home_ownership']
            .map(
                {
                    'RENT': 'аренда',
                    'OWN': 'собственность',
                    'MORTGAGE': 'ипотека',
                    'OTHER':'другой'
                }
            )
        )
        X['loan_intent'] = (
            X['loan_intent']
            .map(
                {
                    'PERSONAL': 'личные цели',
                    'EDUCATION': 'образовательный',
                    'MEDICAL': 'медицинский',
                    'VENTURE': 'венчурный',
                    'HOMEIMPROVEMENT': 'ремонт жилья',
                    'DEBTCONSOLIDATION': 'консолидация задолжностей',
                }
            )
        )
        X['cb_person_default_on_file'] = (
            X['cb_person_default_on_file']
            .map(
                {
                    'Y': 'имеется',
                    'N': 'не имеется',
                }
            )
        )
        return X, numeric_columns, cat_cols

class BinningWOEDictTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_col):
        self.target_col = target_col
        self.binning_woe_dict = {}

    def fit(self, X, y=None):
        X, numeric_columns, cat_cols = X
        if self.target_col in X.columns:
            max_bins = 4
            min_samples_bin = 1
            for col in numeric_columns:
                _, _, binning_woe_results = continuous_woe(X,
                                                          feature=col,
                                                          target=self.target_col,
                                                          max_bins=max_bins,
                                                          min_samples_bin=min_samples_bin)
                self.binning_woe_dict[col] = binning_woe_results
                self.binning_woe_dict[col]['optimized_bin'] =\
                woe_generate_col(sorted(X.loc[:, col]), binning_woe_results).unique()
            for col in cat_cols:
                binning_woe_results = categorical_woe(X,
                                                      cat_variable_name=col,
                                                      y_df=X[self.target_col])
                self.binning_woe_dict[col] = binning_woe_results
        return self

    def transform(self, X):
        X, numeric_columns, cat_cols = X
        return X[numeric_columns + cat_cols], self.binning_woe_dict, numeric_columns, cat_cols

class WOEEncodingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, woe_dict=None):
        self.woe_dict = woe_dict

    def fit(self, X, y=None):
        X, woe_dict, numeric_columns, cat_cols = X
        self.woe_dict = woe_dict
        return self

    def transform(self, X):
        X, woe_dict, numeric_columns, cat_cols = X
        for col in list(woe_dict.keys()):
            if col in cat_cols:
                to_encode = woe_dict[col].sort_values(by='WoE', ascending=False)
                X[col] = (
                    X[col]
                    .map(
                        dict(zip(to_encode.index, list(range(len(to_encode.index), 0, -1))))
                    )
                )
            if col in numeric_columns:
                i = 4
                temp_map = {}
                to_encode = woe_dict[col].sort_values(by='optimized_WoE', ascending=False)
                for el in to_encode.optimized_bin:
                    temp_range = [el.left, el.right]
                    if temp_range[0] <= min(X[col]):
                        temp_map |= dict(zip(X[X[col].between(temp_range[0] - 1, temp_range[1], inclusive='right')][col],
                                             list(map(int, [i] * len(X[X[col].between(temp_range[0] - 1, temp_range[1], inclusive='right')][col])))))
                    else:
                        temp_map |= dict(zip(X[X[col].between(temp_range[0], temp_range[1], inclusive='right')][col],
                                             list(map(int, [i] * len(X[X[col].between(temp_range[0], temp_range[1], inclusive='right')][col])))))
                    i -= 1
                X[col] = X[col].map(temp_map)
        return X, woe_dict


df_train = pd.read_csv(path_train)
target_column = df_train.loan_status
df_test = pd.read_csv(path_test)


target_col = 'loan_status'
pipeline = Pipeline([
    ('data_prep', DataPrepTransformer()),
    ('binning_woe_dict', BinningWOEDictTransformer(target_col=target_col)),
    ('woe_encode', WOEEncodingTransformer())
])

df_train_encoded, woe_dict = pipeline.fit_transform(df_train)


df_train_encoded


X = df_train_encoded
y = target_column.map({0:1, 1:0})


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state = seed)


import lightgbm as lgb
from lightgbm import LGBMClassifier


clf_base = LGBMClassifier()
clf_base.fit(X_train, y_train)


y_test_proba = clf_base.predict_proba(np.array(X_test))


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_test_proba[:, 1])


plt.figure(figsize=(6,6))
display = PrecisionRecallDisplay.from_predictions(
    y_test
    ,y_test_proba[:, 1]
)
_ = display.ax_.set_title("2-class Precision-Recall curve")
plt.legend(loc='best')


from sklearn.metrics import precision_recall_curve


precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba[:, 1])


f1 = 2*precision*recall/(precision+recall)


thresholds[np.argmax(f1)], np.max(f1)


class CustomLGBMClassifier(LGBMClassifier):
    def fit(self, X, y, **kwargs):
        # Вызов оригинального метода fit с передачей всех аргументов
        super().fit(X, y, **kwargs)
        return self

    def predict(self, X, **kwargs):
        # Вызов оригинального метода predict с передачей всех аргументов
        predictions = np.where(super().predict_proba(X, **kwargs)[...,1]>=0.48818458785931473, 1, 0)
        # Здесь можно добавить дополнительную логику, если необходимо
        return predictions


model = CustomLGBMClassifier()


model.fit(X_train, y_train)


model.predict_proba(pipeline.transform(pd.DataFrame(dict(zip(df_test.iloc[0, 1:].index, df_test.iloc[0, 1:].values)),
             index = [0]))[0])


import joblib
joblib.dump(model, 'model.pkl')
joblib.dump(pipeline, 'feature_encode.pkl')


def process_data_and_predict(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data, index=[0])
    transformed_data, _ = pipeline.transform(df)
    score = model.predict_proba(transformed_data)
    prediction = model.predict(transformed_data)
    result = {
        'prediction': prediction.tolist(),
        'score': score.tolist()[0]
    }
    return result


import json


with open('data.json', 'w') as json_file:
    json_file.write(df_test.iloc[0, 1:].to_json(orient='columns'))


json_file_path = '/kaggle/working/data.json'
result = process_data_and_predict(json_file_path)


result

