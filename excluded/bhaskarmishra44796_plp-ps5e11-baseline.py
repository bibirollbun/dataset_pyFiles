import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings as w
w.filterwarnings('ignore')
from sklearn.model_selection import train_test_split,  StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder , OrdinalEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as  xgb
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import joblib


class pathing_init_dawg():
    def __init__(self):
        self.train_path = '/kaggle/input/playground-series-s5e11/train.csv'
        self.test_path = '/kaggle/input/playground-series-s5e11/test.csv'
        self.submission_path = '/kaggle/input/playground-series-s5e11/sample_submission.csv'
        self.target = 'loan_paid_back'
        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)
        self.submission_df = pd.read_csv(self.submission_path)
        print('Datasets loaded successfully!')
dawg_set = pathing_init_dawg()


'''class Data_Analyzer():
    
    def __init__(self, target = 'loan_paid_back' ):
        self.target = target
    def information(self, df: DataFrame, name: str):
        target = self.target
        print(f"NAME OF THE DATAFRAME-------> {name.upper()} <--------")
        print()
        print('-' * 79)
        print(F'---SHAPE OF THE {name.upper()}ING DATASET---')
        print(df.shape)
        print('-' * 79)
        print()
        print('--- NUMERICAL COLUMNS INFORMATION---')
        numerical_columns= df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        print(f'-NUMBERS OF NUMERCAL COLUMNS-- {len(numerical_columns)}')
        print()
        print('- NAMES OF NUMERICAL COLUMNS--')
        for i in numerical_columns:
            print(i)
        print('-' * 79)
        print()
        print('--- CATEGORICAL COLUMNS INFORMATION---')
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        print(f'-NUMBERS OF CATEGORICAL COLUMNS-- {len(categorical_columns)}')
        print()
        print('--NAMES OF CATEGORICAL COLUMNS--')
        for i in categorical_columns:
            print(i)
        print('-' * 79)
        print()
        print('---LABEL / TARGET / WHAT TO PREDICT / DA GOALLLLLL!')
        print()
        if target in df.columns:
            print(f'TARGET --- > {target}')
            print(f'DATA TYPES OF TARGET ---> {df[target].dtypes}')
    
        else:
            print('-OOPS WRONG DATASET, USE TRAINING SET')
            print("-THE DATASET DOESN'T HAVE ANY TARGET COLUMNS")
        print('-' * 79)
        print()
        print('--INFORMATION ABOUT DATA TYPES---')
        print()
        print(df.dtypes)
        print('-' * 79)
        print()
    
        print('---CHECKING OF THERE ARE ANY NULL VALUES IN THE DATASET---')
        print()
        print(df.isnull().sum())
        print()
        if df.isnull().sum().sum() > 0:
            print('-THE DATASET HAS NULL VALEUS')
        else:
            print('-THE DATASET HAS NO NULL VALUES')
            print('-THE DATASET IS CLEAN DAWG')
                
        print('-' * 79)
        
    def description(self, df: DataFrame, name: str):
        print(f'---DESCRIPTION OF {name.upper()}ING DATASET---')
        return df.describe()

    def unseless_line_function(self):
        print('<->' * 45)
        print()
        print('<->' * 45)
        print()
da = Data_Analyzer()
da.information(dawg_set.train_df, 'Train')
display(da.description(dawg_set.train_df, 'Train'))
da.unseless_line_function()
da.information(dawg_set.test_df, 'test')
display(da.description(dawg_set.test_df, 'test'))'''


dawg_set.train_df.info()


numerical_columns= dawg_set.train_df.select_dtypes(include=['number']).columns.tolist()
categorical_columns = dawg_set.train_df.select_dtypes(include=['object']).columns.tolist()
def numeric_subplots(df, numerical_columns):
    n = len(numerical_columns)
    plt.figure(figsize=(15, 5* (n//3)+1))  
    for i, col in enumerate(numerical_columns, 1):
        plt.subplot((n // 3) + 1, 3, i)
        sns.histplot(df[col], bins=30, edgecolor='black', kde= True)
        plt.title(col)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
numeric_subplots(dawg_set.train_df, numerical_columns)


def build_preprocessing(df):
    num = df.select_dtypes(include=['number']).columns.tolist()
    cat = df.select_dtypes(include=['object']).columns.tolist()
    cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                             ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                             ('scaler', StandardScaler())])
    preprocessing = ColumnTransformer([('pipe1', cat_pipeline, cat),
                                       ('pipe2', num_pipeline, num)])
    return preprocessing


lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 8000,
    'learning_rate': 0.008,
    'num_leaves': 127,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'reg_alpha': 2.5,
    'reg_lambda': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

xgb_params = {
    'n_estimators': 8000,
    'max_leaves': 127,
    'min_child_weight': 1.5,
    'max_depth': 0,
    'grow_policy': 'lossguide',
    'learning_rate': 0.008,
    'tree_method': 'hist',
    'subsample': 0.85,
    'colsample_bylevel': 0.7,
    'colsample_bytree': 0.75,
    'colsample_bynode': 0.85,
    'reg_alpha': 2.5,
    'reg_lambda': 0.8,
    'enable_categorical': True,
    'max_cat_to_onehot': 1,
    'device': 'cuda',
    'n_jobs': -1,
    'random_state': 42,
    'verbosity': 0,
    'objective': 'binary:logistic'
}

def train_lgb_xgb_with_auc_and_submission(
        X_train_split, y_train_split, test, 
        lgb_params, xgb_params, preprocessing,
        weight_lgb=0.5, weight_xgb=0.5):

    num = X_train_split.select_dtypes(include=['number']).columns.tolist()
    cat = X_train_split.select_dtypes(include=['object']).columns.tolist()

    models_lgb, models_xgb = [], []
    oof_pred_lgb = np.zeros(len(X_train_split))
    oof_pred_xgb = np.zeros(len(X_train_split))
    test_pred_lgb = np.zeros(len(test))
    test_pred_xgb = np.zeros(len(test))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (tr, val) in enumerate(skf.split(X_train_split, y_train_split)):
        X_tr, X_val = X_train_split.iloc[tr], X_train_split.iloc[val]
        y_tr, y_val = y_train_split.iloc[tr], y_train_split.iloc[val]

        preprocessing.fit(X_tr, y_tr)
        X_tr_pre = preprocessing.transform(X_tr)
        X_val_pre = preprocessing.transform(X_val)
        X_test_pre = preprocessing.transform(test)

        dtrain_lgb = lgb.Dataset(X_tr_pre, label=y_tr)
        dval_lgb = lgb.Dataset(X_val_pre, label=y_val)

        model_lgb = lgb.train(
            params=lgb_params,
            train_set=dtrain_lgb,
            valid_sets=[dtrain_lgb, dval_lgb]
        )

        pred_val_lgb = model_lgb.predict(X_val_pre)
        oof_pred_lgb[val] = pred_val_lgb
        test_pred_lgb += model_lgb.predict(X_test_pre) / skf.n_splits
        models_lgb.append(model_lgb)

        dtrain_xgb = xgb.DMatrix(X_tr_pre, label=y_tr)
        dval_xgb = xgb.DMatrix(X_val_pre, label=y_val)
        dtest_xgb = xgb.DMatrix(X_test_pre)

        model_xgb = xgb.train(
            params=xgb_params,
            dtrain=dtrain_xgb,
            evals=[(dtrain_xgb, 'train'), (dval_xgb, 'valid')]
        )

        pred_val_xgb = model_xgb.predict(dval_xgb)
        oof_pred_xgb[val] = pred_val_xgb
        test_pred_xgb += model_xgb.predict(dtest_xgb) / skf.n_splits
        models_xgb.append(model_xgb)

    auc_lgb = roc_auc_score(y_train_split, oof_pred_lgb)
    auc_xgb = roc_auc_score(y_train_split, oof_pred_xgb)

    y_pred_ensemble = weight_lgb * test_pred_lgb + weight_xgb * test_pred_xgb

    submission = pd.DataFrame({
        'id': test['id'],
        'loan_paid_back': y_pred_ensemble
    })

    submission.to_csv('submission.csv', index=False)

    return {
        'auc_lgb': auc_lgb,
        'auc_xgb': auc_xgb,
        'submission_path': 'submission.csv'
    }



df = dawg_set.train_df
X = df.drop(['loan_paid_back'], axis=1)
y = df['loan_paid_back']

X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessing = build_preprocessing(X_train_split)

result = train_lgb_xgb_with_auc_and_submission(
    X_train_split,
    y_train_split,
    dawg_set.test_df,
    lgb_params,
    xgb_params,
    preprocessing,
    weight_lgb=0.5,
    weight_xgb=0.5
)

print(result['auc_lgb'])
print(result['auc_xgb'])
print(result['submission_path'])





