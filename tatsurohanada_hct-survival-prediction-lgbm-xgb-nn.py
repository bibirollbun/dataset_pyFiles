import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder,OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
import optuna
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import re


# !pip install autogluon.tabular[all] --no-index --find-links=file:/kaggle/input/autogluon-1-20


# pip install /kaggle/input/lifelines-download/lifelines-0.30.0-py3-none-any.whl


train_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
data_dictionary = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')
sample_submission = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv')


train_data.head()


!pip install lifelines


from lifelines import KaplanMeierFitter

def transform_survival_probability(train_data, time_col='efs_time', event_col='efs'):
    """
    Transform using survival probability estimates
    """
    kmf = KaplanMeierFitter()
    kmf.fit(train_data[time_col], train_data[event_col])
    

    y = kmf.survival_function_at_times(train_data[time_col]).values
    
    # Adjust for censoring
    # censored_mask = df[event_col] == 0
    #y[censored_mask] = y[censored_mask] * 1.2  # Increase survival prob for censored
    
    return y

train_data["y"] = transform_survival_probability(train_data, time_col='efs_time', event_col='efs')


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

grouped_data = train_data.groupby('tce_match')['efs_time'].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(x='tce_match', y='efs_time', data=grouped_data, palette='coolwarm')
plt.xlabel("Category")
plt.ylabel("Average Target Value")
# plt.title("Average Target Value by Category")
plt.show()


plt.hist(train_data['age_at_hct'])


train_data_cutID = train_data.drop(columns=['ID'])
target = train_data['y']



X=train_data_cutID.drop(['efs','efs_time','y'], axis=1)
# X = X.drop(['gvhd_proph'],axis=1)
X.head()


print(X.columns.tolist())


X = pd.get_dummies(X,drop_first = False)
# Change columns names ([LightGBM] Do not support special JSON characters in feature name.)
new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in X.columns}
new_n_list = list(new_names.values())
# [LightGBM] Feature appears more than one time.
new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col for i, (col, new_col) in enumerate(new_names.items())}
X = X.rename(columns=new_names)
X.head()


categorical_cols = X.select_dtypes(include=["object"]).columns
cols_numerical = X.select_dtypes(include=["float64","int64"]).columns


from sklearn.impute import KNNImputer
from sklearn.preprocessing import PolynomialFeatures

numerical_cols = X[cols_numerical].columns
imputer = KNNImputer(n_neighbors=2)

numerical_data_after_imputation=imputer.fit_transform(X[numerical_cols])
X_imputed = []
X_imputed = pd.DataFrame(numerical_data_after_imputation,columns=numerical_cols, index=X.index)
X.loc[:, numerical_cols] = X_imputed





X.head()


poly = PolynomialFeatures(degree=2, include_bias=False)


poly_features = poly.fit_transform(X[['age_at_hct', 'donor_age', 'comorbidity_score']])
poly_columns = ['age_at_hct_poly',
                'donor_age_poly',
                'comorbidity_score_poly', 
                'age_at_hct^2',
                'donor_age^2',
                'comorbidity_score^2', 
                'age_at_hct * donor_age',
                'age_at_hct * comorbidity_score',
                'donor_age * comorbidity_score']

poly_df = pd.DataFrame(poly_features, columns=poly_columns, index=X.index)
X = pd.concat([X, poly_df], axis=1)

X.head()


X_int = pd.Series()
X_int['age_at_hct_int']=X['age_at_hct'].astype(int)
X['age_sum'] = X['donor_age'].fillna(0) + X_int['age_at_hct_int'].fillna(0)
X['age_diff'] = abs(X['donor_age'].fillna(0) - X_int['age_at_hct_int'].fillna(0))
X['karnof/age'] = X['karnofsky_score'] / X_int['age_at_hct_int']
X['karnof/age'].clip(-1e6, 1e6, inplace=True)
# X['comorbidity/age'] = X['comorbidity_score'] / X_int['age_at_hct_int']
# X['comorbidity/age'].clip(-1e6, 1e6, inplace=True)
# X['hla_match_c_high_x_drb1_high'] = X['hla_match_c_high'] * X['hla_match_drb1_high']
X.head()


X.info()


X_train,X_test,y_train,y_test=train_test_split(X,target,test_size=0.2,random_state=42)


best_lgbm_params = {'num_leaves': 12,
                    'max_depth': 10,
                    'learning_rate': 0.1,
                    'n_estimators': 128,
                    'subsample': 0.5,
                    'colsample_bytree': 0.6,
                    'reg_alpha': 0.4,
                    'reg_lambda': 0.3,
                    'min_child_samples': 20,
                    'min_split_gain': 0.5,
                    'max_bin': 128
                   }
best_xgb_params = {'xgb_max_depth': 12,
                   'xgb_learning_rate': 0.1,
                   'xgb_n_estimators': 240,
                   'xgb_subsample': 0.6,
                   'xgb_colsample_bytree': 0.9,
                   'xgb_reg_alpha': 0.4,
                   'xgb_reg_lambda': 0.3
                  }

lgbm_best = LGBMRegressor(**best_lgbm_params)
xgb_best = XGBRegressor(**best_xgb_params)

lgbm = LGBMRegressor(**best_lgbm_params)
lgbm.fit(X_train, y_train) 

xgb = XGBRegressor(**best_xgb_params)
xgb.fit(X_train, y_train)

final_voting_regressor = VotingRegressor(estimators=[
    ('lgbm', lgbm_best),
    ('xgb', xgb_best)
])

final_voting_regressor.fit(X_train, y_train)


from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
mse_scores = []
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(X_train):
    tr_x, va_x = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    tr_y, va_y = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    # final_voting_regressor = VotingRegressor(estimators=[
    # ('lgbm', lgbm_best),
    # ('xgb', xgb_best)
    # ])
    
    # final_voting_regressor.fit(tr_x, tr_y)
    y_pred = final_voting_regressor.predict(va_x)

    mse = mean_squared_error(va_y, y_pred)
    mse_scores.append(mse)

print(f"Average MSE: {np.mean(mse_scores):.4f}")



# sns.heatmap(X_train[['dri_score_High','comorbidity_score','donor_age','age_at_hct','cyto_score_detail_Poor','ethnicity_HispanicorLatino', 'ethnicity_NonresidentoftheUS', 'ethnicity_NotHispanicorLatino']].corr())


# import xgboost as xgbdm
# dtrain = xgbdm.DMatrix(X_train, label=y_train)
# params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
# num_round = 50
# model = xgbdm.train(params, dtrain, num_round)



# fscore = model.get_score(importance_type='total_gain')
# fscore = sorted([(k, v) for k, v in fscore.items()], key=lambda tpl: tpl[1], reverse=True)
# print('xgboost importance')
# print(fscore[:5])


test_data_cutID=test_data.drop(['ID'], axis=1)
test_df = pd.get_dummies(test_data_cutID,drop_first = False)
new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in test_df.columns}
new_n_list = list(new_names.values())
new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col for i, (col, new_col) in enumerate(new_names.items())}
test_df = test_df.rename(columns=new_names)



missing_columns = set(tr_x.columns) - set(test_df.columns)

for col in missing_columns:
    test_df[col] = np.nan

test_df = test_df[tr_x.columns]


# missing_columns


test_categorical_cols = test_df.select_dtypes(include=["object"]).columns
test_cols_numerical = test_df.select_dtypes(include=["float64","int64"]).columns


# test_numerical_cols = test_df[test_cols_numerical].columns
# test_imputer = KNNImputer(n_neighbors=2)

# test_numerical_data_after_imputation=test_imputer.fit_transform(test_df[test_numerical_cols])
# test_df_imputed = []
# test_df_imputed = pd.DataFrame(test_numerical_data_after_imputation,columns=test_numerical_cols, index=test_df.index)
# test_df.loc[:, numerical_cols] = test_df_imputed


test_int = pd.Series()
test_int['age_at_hct_int']=test_df['age_at_hct'].astype(int)
test_df['age_sum'] = test_df['donor_age'].fillna(0) + test_int['age_at_hct_int'].fillna(0)
test_df['age_diff'] = abs(test_df['donor_age'].fillna(0) - test_int['age_at_hct_int'].fillna(0))
test_df['karnof/age'] = test_df['karnofsky_score'] / test_int['age_at_hct_int']
# test_df['hla_match_c_high_x_drb1_high'] = test_df['hla_match_c_high'] * test_df['hla_match_drb1_high']
test_df['karnof/age'].clip(-1e6, 1e6, inplace=True)
# test_df['comorbidity/age'] = test_df['comorbidity_score'] / test_int['age_at_hct_int']
# test_df['comorbidity/age'].clip(-1e6, 1e6, inplace=True)


test_numerical_cols = test_df[cols_numerical].columns
imputer = KNNImputer(n_neighbors=2)

test_numerical_data_after_imputation=imputer.fit_transform(test_df[numerical_cols])
test_df_imputed = []
test_df_imputed = pd.DataFrame(test_numerical_data_after_imputation,columns=test_numerical_cols, index=test_df.index)
test_df.loc[:, numerical_cols] = test_df_imputed



test_imputed = []
test_imputed = pd.DataFrame(test_numerical_data_after_imputation,columns=test_numerical_cols, index=test_df.index)
test_poly = PolynomialFeatures(degree=2, include_bias=False)


test_poly_features = test_poly.fit_transform(test_imputed[['age_at_hct', 'donor_age', 'comorbidity_score']])
test_poly_columns = ['age_at_hct_poly', 'donor_age_poly', 'comorbidity_score_poly', 
                'age_at_hct^2', 'donor_age^2', 'comorbidity_score^2', 
                'age_at_hct * donor_age', 'age_at_hct * comorbidity_score', 'donor_age * comorbidity_score']
test_poly_df = pd.DataFrame(test_poly_features, columns=test_poly_columns, index=test_df.index)
test_df[test_poly_columns] = test_poly_df[test_poly_columns]


bool_cols = tr_x.select_dtypes(include=['bool']).columns
test_df[bool_cols] = test_df[bool_cols].fillna(False).astype(bool)


test_df.info()


submission_gb = pd.DataFrame()
submission_gb['ID'] = test_data['ID']

submission_gb['prediction'] = final_voting_regressor.predict(test_df)
file_name = 'submission.csv'
# submission_gb.to_csv(file_name, index=False)
submission_gb


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding,Flatten
from tensorflow.keras.layers import Concatenate, BatchNormalization
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,LearningRateScheduler
from tensorflow.keras.optimizers import Adam,SGD
# from metric import score

print('TF Version',tf.__version__)


def seed_everything(seed):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)



nn_target = pd.DataFrame(target)


x_tr,x_va,y_tr,y_va=train_test_split(X,nn_target,test_size=0.2,random_state=42)
print(x_tr.shape,x_va.shape,y_tr.shape,y_va.shape)


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout

def create_regression_model():
    input_num = Input(shape=(195,))
    
    x_num = Dense(10, activation='relu', kernel_initializer='glorot_uniform')(input_num)
    x_num = BatchNormalization()(x_num)
    x_num = Dropout(0.1)(x_num)

    x_num = Dense(10, activation='relu', kernel_initializer='glorot_uniform')(x_num)
    x_num = BatchNormalization()(x_num)
    x_num = Dropout(0.1)(x_num)

    x_num = Dense(5, activation='relu', kernel_initializer='glorot_uniform')(x_num)
    x_num = BatchNormalization()(x_num)
    x_num = Dropout(0.1)(x_num)

    x_num = BatchNormalization()(x_num)  # 追加
    out = Dense(1, activation=None)(x_num)  # 活性化関数なし

    nn_model = Model(inputs=input_num, outputs=out)
    
    nn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  
        loss='mean_squared_error',  # MSE に変更
        metrics=['mean_absolute_error'],  # MAE を評価指標に
    )

    return nn_model

# モデルの作成
nn_model = create_regression_model()
nn_model.summary()


seed_everything(seed=123)
nn_model = create_regression_model()
nn_model.fit(x=x_tr,
            y=y_tr,
            validation_data=(x_va,y_va),
            batch_size=8,
            epochs=20,
            callbacks=[
                ModelCheckpoint(filepath='model_keras.weights.h5',
                                monitor='val_loss',
                                mode='min',
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=True),
                EarlyStopping(monitor='val_loss',
                              mode='min',
                              min_delta=0,
                              patience=10,
                              verbose=1,
                              restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss',
                                mode='min',
                                factor=0.1,
                                patience=5,
                                verbose=1),
            ],
            verbose=1,
            )


y_va_pred = nn_model.predict(x_va,batch_size=8,verbose=1)
# print('accuracy:{:.4f}'.format(accuracy_score(y_va,y_va_pred)))
y_va_pred


x_tr.info()


# test_data_cutID=test_data.drop(['ID'], axis=1)
# test_df = pd.get_dummies(test_data_cutID,drop_first = False)
# new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in test_df.columns}
# new_n_list = list(new_names.values())
# new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col for i, (col, new_col) in enumerate(new_names.items())}
# test_df = test_df.rename(columns=new_names)



# test_df.info()


# missing_columns = set(x_tr.columns) - set(test_df.columns)

# for col in missing_columns:
#     test_df[col] = np.nan

# test_df = test_df[x_tr.columns]

# test_categorical_cols = test_df.select_dtypes(include=["object"]).columns
# test_cols_numerical = test_df.select_dtypes(include=["float64","int64"]).columns

# bool_cols = x_tr.select_dtypes(include=['bool']).columns
# test_df[bool_cols] = test_df[bool_cols].fillna(False).astype(bool)


# test_int = pd.Series()
# test_int['age_at_hct_int']=test_df['age_at_hct'].astype(int)
# test_df['age_sum'] = test_df['donor_age'].fillna(0) + test_int['age_at_hct_int'].fillna(0)
# test_df['age_diff'] = abs(test_df['donor_age'].fillna(0) - test_int['age_at_hct_int'].fillna(0))
# test_df['karnof/age'] = test_df['karnofsky_score'] / test_int['age_at_hct_int']
# # test_df['hla_match_c_high_x_drb1_high'] = test_df['hla_match_c_high'] * test_df['hla_match_drb1_high']
# test_df['karnof/age'].clip(-1e6, 1e6, inplace=True)
# # test_df['comorbidity/age'] = test_df['comorbidity_score'] / test_int['age_at_hct_int']
# # test_df['comorbidity/age'].clip(-1e6, 1e6, inplace=True)

# test_numerical_cols = test_df[cols_numerical].columns
# imputer = KNNImputer(n_neighbors=2)

# test_numerical_data_after_imputation=imputer.fit_transform(test_df[numerical_cols])
# test_df_imputed = []
# test_df_imputed = pd.DataFrame(test_numerical_data_after_imputation,columns=test_numerical_cols, index=test_df.index)
# test_df.loc[:, numerical_cols] = test_df_imputed

# test_imputed = []
# test_imputed = pd.DataFrame(test_numerical_data_after_imputation,columns=test_numerical_cols, index=test_df.index)
# test_poly = PolynomialFeatures(degree=2, include_bias=False)


# test_poly_features = test_poly.fit_transform(test_imputed[['age_at_hct', 'donor_age', 'comorbidity_score']])
# test_poly_columns = ['age_at_hct_poly', 'donor_age_poly', 'comorbidity_score_poly', 
#                 'age_at_hct^2', 'donor_age^2', 'comorbidity_score^2', 
#                 'age_at_hct * donor_age', 'age_at_hct * comorbidity_score', 'donor_age * comorbidity_score']
# test_poly_df = pd.DataFrame(test_poly_features, columns=test_poly_columns, index=test_df.index)
# test_df[test_poly_columns] = test_poly_df[test_poly_columns]



x_tr_dtypes = x_tr.dtypes
test_df_dtypes = test_df.dtypes


df_dtypes_comparison = pd.DataFrame({
    'x_tr_dtype': x_tr_dtypes,
    'test_df_dtype': test_df_dtypes
})

df_dtypes_comparison['match'] = df_dtypes_comparison['x_tr_dtype'] == df_dtypes_comparison['test_df_dtype']

pd.set_option('display.max_rows', None) 
pd.set_option('display.max_columns', None) 
pd.set_option('display.width', 1000) 
pd.set_option('display.colheader_justify', 'left') 

print(df_dtypes_comparison)


submission_nn = pd.DataFrame()
submission_nn['ID'] = test_data['ID']
nn_target = pd.DataFrame(nn_target)
submission_nn['prediction'] = nn_model.predict(test_df)
# file_name = 'submission.csv'
# submission.to_csv(file_name, index=False)
# submission


merged_df = submission_gb.merge(submission_nn, on='ID', suffixes=('_gb', '_nn'))

merged_df['prediction'] = 0.5 * merged_df['prediction_gb'] + 0.5 * merged_df['prediction_nn']

final_submission = merged_df[['ID', 'prediction']]

file_name = 'submission.csv'
final_submission.to_csv(file_name, index=False)
final_submission




