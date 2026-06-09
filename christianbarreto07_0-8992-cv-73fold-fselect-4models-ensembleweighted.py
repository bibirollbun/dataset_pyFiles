%load_ext cudf.pandas
import pandas as pd

train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv", index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv", index_col='id')


display(train.head())
display(train.info())
display(test.head())
display(test.info())


target = 'rainfall'
features = [f for f in train.columns.tolist() if f not in [target]]
display(features)


from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import warnings
warnings.simplefilter('ignore')

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        for col1 in features:
            for lag in range(1, 5):
                X[f'{col1}_diff_lag{lag}'] = X.loc[:, col1].diff(lag).fillna(0)
            
            # Trigonometric functions
            X[f"sin {col1}"] = np.sin(2 * np.pi * X.loc[:, col1] / 365)
            X[f"cos {col1}"] = np.cos(2 * np.pi * X.loc[:, col1] / 365)
            X[f"tanh {col1}"] = np.tanh(2 * np.pi * X.loc[:, col1] / 365)
            X[f"sin {col1}2"] = np.sin(X.loc[:, col1])
            X[f"cos {col1}2"] = np.cos(X.loc[:, col1])
            X[f"tanh {col1}2"] = np.tanh(X.loc[:, col1])
            X[f"rad {col1}"] = np.deg2rad(X.loc[:, col1])
            
            # Arithmetic operations
            X[f"mean {col1}"] = X.loc[:, col1].mean()
            X[f"median {col1}"] = X.loc[:, col1].median()
            X[f"mode {col1}"] = X.loc[:, col1].mode()[0] if not X.loc[:, col1].mode().empty else 0
            X[f"std {col1}"] = X.loc[:, col1].std()
            X[f"var {col1}"] = X.loc[:, col1].var()
            X[f"min {col1}"] = X.loc[:, col1].min()
            X[f"max {col1}"] = X.loc[:, col1].max()
            X[f"range {col1}"] = X.loc[:, col1].max() - X.loc[:, col1].min()
            X[f"iqr {col1}"] = X.loc[:, col1].quantile(0.75) - X.loc[:, col1].quantile(0.25)
            
            for col2 in features:
                if col1 != col2:
                    
                    X[f"{col1} x {col2}"] = X.loc[:, col1] * X.loc[:, col2]
                    X[f"{col1} / {col2}"] = X.loc[:, col1] / (X.loc[:, col2] + 1e-9)  # Evita divisão por zero
                    X[f"{col1} - {col2}"] = X.loc[:, col1] - X.loc[:, col2]
                    X[f"{col1} + {col2}"] = X.loc[:, col1] + X.loc[:, col2]
                    
                    # New combinations
                    X[f"sqrt_{col1} x {col2}"] = np.sqrt(abs(X.loc[:, col1] * X.loc[:, col2]))
                    X[f"log_{col1} + {col2}"] = np.log1p(abs(X.loc[:, col1] + X.loc[:, col2]))
                    X[f"inv_{col1} x {col2}"] = 1 / (X.loc[:, col1] * X.loc[:, col2] + 1e-9)
                    X[f"cos_{col1} x sin_{col2}"] = np.cos(X.loc[:, col1]) * np.sin(X.loc[:, col2])
                    X[f"tan_{col1} - tan_{col2}"] = np.tan(X.loc[:, col1]) - np.tan(X.loc[:, col2])
        
        return X


train_cop = CombinedAttributesAdder().transform(train)
test_cop = CombinedAttributesAdder().transform(test)


display(train_cop.head())
display(test_cop.head())


############### Valueless attributes ###############
IRRELEVANTS = ['day / pressure', 'pressure / day', 'day + pressure', 'day - pressure', 'sqrt_day x pressure', 'sqrt_pressure x day', 'day / maxtemp', 'maxtemp / day', 'day + maxtemp', 'day - maxtemp', 'sqrt_day x maxtemp', 'sqrt_maxtemp x day', 'day / temparature', 'temparature / day', 'day + temparature', 'day - temparature', 'sqrt_day x temparature', 'sqrt_temparature x day', 'day / mintemp', 'mintemp / day', 'day + mintemp', 'day - mintemp', 'sqrt_day x mintemp', 'sqrt_mintemp x day', 'day / dewpoint', 'dewpoint / day', 'day + dewpoint', 'day - dewpoint', 'sqrt_day x dewpoint', 'sqrt_dewpoint x day', 'day / humidity', 'humidity / day', 'day + humidity', 'day - humidity', 'sqrt_day x humidity', 'sqrt_humidity x day', 'day / cloud', 'cloud / day', 'day + cloud', 'day - cloud', 'sqrt_day x cloud', 'sqrt_cloud x day', 'day / sunshine', 'sunshine / day', 'day + sunshine', 'day - sunshine', 'sqrt_day x sunshine', 'sqrt_sunshine x day', 'day / winddirection', 'winddirection / day', 'day + winddirection', 'day - winddirection', 'sqrt_day x winddirection', 'sqrt_winddirection x day', 'day / windspeed', 'windspeed / day', 'day + windspeed', 'day - windspeed', 'sqrt_day x windspeed', 'sqrt_windspeed x day', 'pressure / maxtemp', 'maxtemp / pressure', 'pressure + maxtemp', 'pressure - maxtemp', 'sqrt_pressure x maxtemp', 'sqrt_maxtemp x pressure', 'pressure / temparature', 'temparature / pressure', 'pressure + temparature', 'pressure - temparature', 'sqrt_pressure x temparature', 'sqrt_temparature x pressure', 'pressure / mintemp', 'mintemp / pressure', 'pressure + mintemp', 'pressure - mintemp', 'sqrt_pressure x mintemp', 'sqrt_mintemp x pressure', 'pressure / dewpoint', 'dewpoint / pressure', 'pressure + dewpoint', 'pressure - dewpoint', 'sqrt_pressure x dewpoint', 'sqrt_dewpoint x pressure', 'pressure / humidity', 'humidity / pressure', 'pressure + humidity', 'pressure - humidity', 'sqrt_pressure x humidity', 'sqrt_humidity x pressure', 'pressure / cloud', 'cloud / pressure', 'pressure + cloud', 'pressure - cloud', 'sqrt_pressure x cloud', 'sqrt_cloud x pressure', 'pressure / sunshine', 'sunshine / pressure', 'pressure + sunshine', 'pressure - sunshine', 'sqrt_pressure x sunshine', 'sqrt_sunshine x pressure', 'pressure / winddirection', 'winddirection / pressure', 'pressure + winddirection', 'pressure - winddirection', 'sqrt_pressure x winddirection', 'sqrt_winddirection x pressure', 'pressure / windspeed', 'windspeed / pressure', 'pressure + windspeed', 'pressure - windspeed', 'sqrt_pressure x windspeed', 'sqrt_windspeed x pressure', 'maxtemp / temparature', 'temparature / maxtemp', 'maxtemp + temparature', 'maxtemp - temparature', 'sqrt_maxtemp x temparature', 'sqrt_temparature x maxtemp', 'maxtemp / mintemp', 'mintemp / maxtemp', 'maxtemp + mintemp', 'maxtemp - mintemp', 'sqrt_maxtemp x mintemp', 'sqrt_mintemp x maxtemp', 'maxtemp / dewpoint', 'dewpoint / maxtemp', 'maxtemp + dewpoint', 'maxtemp - dewpoint', 'sqrt_maxtemp x dewpoint', 'sqrt_dewpoint x maxtemp', 'maxtemp / humidity', 'humidity / maxtemp', 'maxtemp + humidity', 'maxtemp - humidity', 'sqrt_maxtemp x humidity', 'sqrt_humidity x maxtemp', 'maxtemp / cloud', 'cloud / maxtemp', 'maxtemp + cloud', 'maxtemp - cloud', 'sqrt_maxtemp x cloud', 'sqrt_cloud x maxtemp', 'maxtemp / sunshine', 'sunshine / maxtemp', 'maxtemp + sunshine', 'maxtemp - sunshine', 'sqrt_maxtemp x sunshine', 'sqrt_sunshine x maxtemp', 'maxtemp / winddirection', 'winddirection / maxtemp', 'maxtemp + winddirection', 'maxtemp - winddirection', 'sqrt_maxtemp x winddirection', 'sqrt_winddirection x maxtemp', 'maxtemp / windspeed', 'windspeed / maxtemp', 'maxtemp + windspeed', 'maxtemp - windspeed', 'sqrt_maxtemp x windspeed', 'sqrt_windspeed x maxtemp', 'temparature / mintemp', 'mintemp / temparature', 'temparature + mintemp', 'temparature - mintemp', 'sqrt_temparature x mintemp', 'sqrt_mintemp x temparature', 'temparature / dewpoint', 'dewpoint / temparature', 'temparature + dewpoint', 'temparature - dewpoint', 'sqrt_temparature x dewpoint', 'sqrt_dewpoint x temparature', 'temparature / humidity', 'humidity / temparature', 'temparature + humidity', 'temparature - humidity', 'sqrt_temparature x humidity', 'sqrt_humidity x temparature', 'temparature / cloud', 'cloud / temparature', 'temparature + cloud', 'temparature - cloud', 'sqrt_temparature x cloud', 'sqrt_cloud x temparature', 'temparature / sunshine', 'sunshine / temparature', 'temparature + sunshine', 'temparature - sunshine', 'sqrt_temparature x sunshine', 'sqrt_sunshine x temparature', 'temparature / winddirection', 'winddirection / temparature', 'temparature + winddirection', 'temparature - winddirection', 'sqrt_temparature x winddirection', 'sqrt_winddirection x temparature', 'temparature / windspeed', 'windspeed / temparature', 'temparature + windspeed', 'temparature - windspeed', 'sqrt_temparature x windspeed', 'sqrt_windspeed x temparature', 'mintemp / dewpoint', 'dewpoint / mintemp', 'mintemp + dewpoint', 'mintemp - dewpoint', 'sqrt_mintemp x dewpoint', 'sqrt_dewpoint x mintemp', 'mintemp / humidity', 'humidity / mintemp', 'mintemp + humidity', 'mintemp - humidity', 'sqrt_mintemp x humidity', 'sqrt_humidity x mintemp', 'mintemp / cloud', 'cloud / mintemp', 'mintemp + cloud', 'mintemp - cloud', 'sqrt_mintemp x cloud', 'sqrt_cloud x mintemp', 'mintemp / sunshine', 'sunshine / mintemp', 'mintemp + sunshine', 'mintemp - sunshine', 'sqrt_mintemp x sunshine', 'sqrt_sunshine x mintemp', 'mintemp / winddirection', 'winddirection / mintemp', 'mintemp + winddirection', 'mintemp - winddirection', 'sqrt_mintemp x winddirection', 'sqrt_winddirection x mintemp', 'mintemp / windspeed', 'windspeed / mintemp', 'mintemp + windspeed', 'mintemp - windspeed', 'sqrt_mintemp x windspeed', 'sqrt_windspeed x mintemp', 'dewpoint / humidity', 'humidity / dewpoint', 'dewpoint + humidity', 'dewpoint - humidity', 'sqrt_dewpoint x humidity', 'sqrt_humidity x dewpoint', 'dewpoint / cloud', 'cloud / dewpoint', 'dewpoint + cloud', 'dewpoint - cloud', 'sqrt_dewpoint x cloud', 'sqrt_cloud x dewpoint', 'dewpoint / sunshine', 'sunshine / dewpoint', 'dewpoint + sunshine', 'dewpoint - sunshine', 'sqrt_dewpoint x sunshine', 'sqrt_sunshine x dewpoint', 'dewpoint / winddirection', 'winddirection / dewpoint', 'dewpoint + winddirection', 'dewpoint - winddirection', 'sqrt_dewpoint x winddirection', 'sqrt_winddirection x dewpoint', 'dewpoint / windspeed', 'windspeed / dewpoint', 'dewpoint + windspeed', 'dewpoint - windspeed', 'sqrt_dewpoint x windspeed', 'sqrt_windspeed x dewpoint', 'humidity / cloud', 'cloud / humidity', 'humidity + cloud', 'humidity - cloud', 'sqrt_humidity x cloud', 'sqrt_cloud x humidity', 'humidity / sunshine', 'sunshine / humidity', 'humidity + sunshine', 'humidity - sunshine', 'sqrt_humidity x sunshine', 'sqrt_sunshine x humidity', 'humidity / winddirection', 'winddirection / humidity', 'humidity + winddirection', 'humidity - winddirection', 'sqrt_humidity x winddirection', 'sqrt_winddirection x humidity', 'humidity / windspeed', 'windspeed / humidity', 'humidity + windspeed', 'humidity - windspeed', 'sqrt_humidity x windspeed', 'sqrt_windspeed x humidity', 'cloud / sunshine', 'sunshine / cloud', 'cloud + sunshine', 'cloud - sunshine', 'sqrt_cloud x sunshine', 'sqrt_sunshine x cloud', 'cloud / winddirection', 'winddirection / cloud', 'cloud + winddirection', 'cloud - winddirection', 'sqrt_cloud x winddirection', 'sqrt_winddirection x cloud', 'cloud / windspeed', 'windspeed / cloud', 'cloud + windspeed', 'cloud - windspeed', 'sqrt_cloud x windspeed', 'sqrt_windspeed x cloud', 'sunshine / winddirection', 'winddirection / sunshine', 'sunshine + winddirection', 'sunshine - winddirection', 'sqrt_sunshine x winddirection', 'sqrt_winddirection x sunshine', 'sunshine / windspeed', 'windspeed / sunshine', 'sunshine + windspeed', 'sunshine - windspeed', 'sqrt_sunshine x windspeed', 'sqrt_windspeed x sunshine', 'winddirection / windspeed', 'windspeed / winddirection', 'winddirection + windspeed', 'winddirection - windspeed', 'sqrt_winddirection x windspeed', 'sqrt_windspeed x winddirection', 'rad pressure', 'rad humidity', 'log_pressure + day', 'inv_pressure x day', 'cos_pressure x sin_day', 'log_maxtemp + day', 'inv_maxtemp x day', 'cos_maxtemp x sin_day', 'log_temparature + day', 'inv_temparature x day', 'cos_temparature x sin_day', 'log_mintemp + day', 'inv_mintemp x day', 'cos_mintemp x sin_day', 'log_dewpoint + day', 'inv_dewpoint x day', 'cos_dewpoint x sin_day', 'log_humidity + day', 'inv_humidity x day', 'cos_humidity x sin_day', 'log_cloud + day', 'inv_cloud x day', 'cos_cloud x sin_day', 'log_sunshine + day', 'inv_sunshine x day', 'cos_sunshine x sin_day', 'log_winddirection + day', 'inv_winddirection x day', 'cos_winddirection x sin_day', 'log_windspeed + day', 'inv_windspeed x day', 'cos_windspeed x sin_day', 'log_maxtemp + pressure', 'inv_maxtemp x pressure', 'cos_maxtemp x sin_pressure', 'log_temparature + pressure', 'inv_temparature x pressure', 'cos_temparature x sin_pressure', 'log_mintemp + pressure', 'inv_mintemp x pressure', 'cos_mintemp x sin_pressure', 'log_dewpoint + pressure', 'inv_dewpoint x pressure', 'cos_dewpoint x sin_pressure', 'log_humidity + pressure', 'inv_humidity x pressure', 'cos_humidity x sin_pressure', 'log_cloud + pressure', 'inv_cloud x pressure', 'cos_cloud x sin_pressure', 'log_sunshine + pressure', 'inv_sunshine x pressure', 'cos_sunshine x sin_pressure', 'log_winddirection + pressure', 'inv_winddirection x pressure', 'cos_winddirection x sin_pressure', 'log_windspeed + pressure', 'inv_windspeed x pressure', 'cos_windspeed x sin_pressure', 'log_temparature + maxtemp', 'inv_temparature x maxtemp', 'cos_temparature x sin_maxtemp', 'log_mintemp + maxtemp', 'inv_mintemp x maxtemp', 'cos_mintemp x sin_maxtemp', 'log_dewpoint + maxtemp', 'inv_dewpoint x maxtemp', 'cos_dewpoint x sin_maxtemp', 'log_humidity + maxtemp', 'inv_humidity x maxtemp', 'cos_humidity x sin_maxtemp', 'log_cloud + maxtemp', 'inv_cloud x maxtemp', 'cos_cloud x sin_maxtemp', 'log_sunshine + maxtemp', 'inv_sunshine x maxtemp', 'cos_sunshine x sin_maxtemp', 'log_winddirection + maxtemp', 'inv_winddirection x maxtemp', 'cos_winddirection x sin_maxtemp', 'log_windspeed + maxtemp', 'inv_windspeed x maxtemp', 'cos_windspeed x sin_maxtemp', 'log_mintemp + temparature', 'inv_mintemp x temparature', 'cos_mintemp x sin_temparature', 'log_dewpoint + temparature', 'inv_dewpoint x temparature', 'cos_dewpoint x sin_temparature', 'log_humidity + temparature', 'inv_humidity x temparature', 'cos_humidity x sin_temparature', 'log_cloud + temparature', 'inv_cloud x temparature', 'cos_cloud x sin_temparature', 'log_sunshine + temparature', 'inv_sunshine x temparature', 'cos_sunshine x sin_temparature', 'log_winddirection + temparature', 'inv_winddirection x temparature', 'cos_winddirection x sin_temparature', 'log_windspeed + temparature', 'inv_windspeed x temparature', 'cos_windspeed x sin_temparature', 'log_dewpoint + mintemp', 'inv_dewpoint x mintemp', 'cos_dewpoint x sin_mintemp', 'log_humidity + mintemp', 'inv_humidity x mintemp', 'cos_humidity x sin_mintemp', 'log_cloud + mintemp', 'inv_cloud x mintemp', 'cos_cloud x sin_mintemp', 'log_sunshine + mintemp', 'inv_sunshine x mintemp', 'cos_sunshine x sin_mintemp', 'log_winddirection + mintemp', 'inv_winddirection x mintemp', 'cos_winddirection x sin_mintemp', 'log_windspeed + mintemp', 'inv_windspeed x mintemp', 'cos_windspeed x sin_mintemp', 'log_humidity + dewpoint', 'inv_humidity x dewpoint', 'cos_humidity x sin_dewpoint', 'log_cloud + dewpoint', 'inv_cloud x dewpoint', 'cos_cloud x sin_dewpoint', 'log_sunshine + dewpoint', 'inv_sunshine x dewpoint', 'cos_sunshine x sin_dewpoint', 'log_winddirection + dewpoint', 'inv_winddirection x dewpoint', 'cos_winddirection x sin_dewpoint', 'log_windspeed + dewpoint', 'inv_windspeed x dewpoint', 'cos_windspeed x sin_dewpoint', 'iqr cloud', 'log_cloud + humidity', 'inv_cloud x humidity', 'cos_cloud x sin_humidity', 'log_sunshine + humidity', 'inv_sunshine x humidity', 'cos_sunshine x sin_humidity', 'log_winddirection + humidity', 'inv_winddirection x humidity', 'cos_winddirection x sin_humidity', 'log_windspeed + humidity', 'inv_windspeed x humidity', 'cos_windspeed x sin_humidity', 'log_sunshine + cloud', 'inv_sunshine x cloud', 'cos_sunshine x sin_cloud', 'log_winddirection + cloud', 'inv_winddirection x cloud', 'cos_winddirection x sin_cloud', 'log_windspeed + cloud', 'inv_windspeed x cloud', 'cos_windspeed x sin_cloud', 'max sunshine', 'iqr sunshine', 'log_winddirection + sunshine', 'inv_winddirection x sunshine', 'cos_winddirection x sin_sunshine', 'log_windspeed + sunshine', 'inv_windspeed x sunshine', 'cos_windspeed x sin_sunshine', 'std winddirection', 'log_windspeed + winddirection', 'inv_windspeed x winddirection', 'cos_windspeed x sin_winddirection']

print(f"Shape before: ", train_cop.shape, test_cop.shape)                    
train_cop, test_cop = train_cop.drop(IRRELEVANTS, axis=1), test_cop.drop(IRRELEVANTS, axis=1)
print(f"Shape after: ", train_cop.shape, test_cop.shape)       


# Check for NaN and infinite values in both dataframes
def check_nan_infinite(df, df_name):
    print(f"Checking for NaN and infinite values in {df_name}...\n")
    
    # Check for NaN values
    nan_columns = df.columns[df.isna().any()].tolist()
    if nan_columns:
        print(f"Columns with NaN values in {df_name}: {nan_columns}")
    else:
        print(f"No NaN values found in {df_name}.")
    
    # Check for infinite values
    inf_columns = df.columns[(df == float('inf')).any() | (df == float('-inf')).any()].tolist()
    if inf_columns:
        print(f"Columns with infinite values in {df_name}: {inf_columns}")
    else:
        print(f"No infinite values found in {df_name}.\n")

check_nan_infinite(train_cop, "train_cop")
check_nan_infinite(test_cop, "test_cop")


train_cop['winddirection'] = train_cop['winddirection'].fillna(train_cop['winddirection'].mean())
test_cop['winddirection'] = test_cop['winddirection'].fillna(train_cop['winddirection'].mean())

############### NaN values ###############
print(f"Shape before: ", train_cop.shape, test_cop.shape) 

null_columns_train = set(train_cop.columns[train_cop.isnull().any()])
null_columns_test = set(test_cop.columns[test_cop.isnull().any()])

null_columns = null_columns_train.union(null_columns_test)

train_cop.drop(columns=null_columns, inplace=True)
test_cop.drop(columns=null_columns, inplace=True)

print(f"Shape after: ", train_cop.shape, test_cop.shape)   


check_nan_infinite(train_cop, "train_cop")
check_nan_infinite(test_cop, "test_cop")


def remove_zero_std_columns(train_df, test_df):
    print("Checking and removing columns with zero standard deviation...\n")
    
    # Calculate the standard deviation of each column for both DataFrames
    train_std_devs = train_df.std()
    test_std_devs = test_df.std()
    
    # Identify columns with standard deviation equal to 0 in both DataFrames
    zero_std_columns_train = train_std_devs[train_std_devs == 0].index.tolist()
    zero_std_columns_test = test_std_devs[test_std_devs == 0].index.tolist()
    
    # Combine the columns with zero standard deviation from both DataFrames
    zero_std_columns = list(set(zero_std_columns_train).union(zero_std_columns_test))
    
    if zero_std_columns:
        print(f"Columns with zero standard deviation: {zero_std_columns}")
        # Drop these columns from both DataFrames
        train_df = train_df.drop(columns=zero_std_columns)
        test_df = test_df.drop(columns=zero_std_columns)
        print("Removed columns with zero standard deviation from both DataFrames.")
    else:
        print("No columns with zero standard deviation found.\n")
    
    return train_df, test_df


train_cop, test_cop = remove_zero_std_columns(train_cop, test_cop)


new_attribs = [c for c in list( train_cop.columns ) if not c in [target] + features]
print("New attributes length: ", len(new_attribs))


from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

def cross_val_score(model, train_fuc, test_func, features_to_train, FOLDS, i=0, eval_set=False, return_models=False):
    oof_model_func = np.zeros(len(train_fuc))
    pred_model_func = np.zeros(len(test_func))
    mean_iterations = 0
    models = []
    
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=77+i)
    # GROUP K FOLD USING YEAR AS GROUP
    for i, (train_index, test_index) in enumerate(kf.split(train_fuc)):
        #print("#"*60)
        #print(f"#### FOLD {FOLDS} ####\n" )
        
        X_train = train_fuc.loc[train_index, features_to_train].copy()
        y_train = train_fuc.loc[train_index, target]
        X_valid = train_fuc.loc[test_index, features_to_train].copy()
        y_valid = train_fuc.loc[test_index, target]
        X_test = test_func[features_to_train].copy()
        for feat in features_to_train:
            # Standard
            mean = X_train[feat].mean()
            std = X_train[feat].std()
            X_train[feat] = (X_train[feat]-mean) / std
            X_valid[feat] = (X_valid[feat]-mean) / std
            X_test[feat] = (X_test[feat]-mean) / std

            ### NaN values ###
            X_train[feat] = X_train[feat].fillna(0)
            X_valid[feat] = X_valid[feat].fillna(0)
            X_test[feat] = X_test[feat].fillna(0)

        if isinstance(model, XGBClassifier):
            model.set_params(random_state=42+i)
            if eval_set:
                model.fit(
                    X_train, y_train, 
                    eval_set=[(X_valid, y_valid)],
                    verbose=5,
                )
                best_iteration = model.best_iteration
                mean_iterations += best_iteration / FOLDS
            else:
                model.fit(
                    X_train, y_train
                )
        else:
            ### TRAIN MODEL ###
            model.fit(X_train.values, y_train.values)
        
        if return_models:
            ### SAVING FITTED MODELS ###
            models.append(model)

        ### OOF VALID ###
        oof_model_func[test_index] = model.predict_proba(X_valid.values)[:,1]

        ### PRED OF TEST ###
        pred_model_func += model.predict_proba(X_test.values)[:,1]

    
    if return_models:
        ### RETURN PREDS AND ALL MODELS FITTED ###
        return oof_model_func, pred_model_func, models
    
    if mean_iterations > 0 and isinstance(model, XGBClassifier):
        print(f"Mean of iterations: {mean_iterations}")

    ### ONLY PREDS RETURN ###
    return oof_model_func, pred_model_func


from xgboost import XGBClassifier
from cuml.svm import LinearSVC
from sklearn.metrics import roc_auc_score
"""
linearSVC_FEATURES_TO_ADD  = []
best_auc = 0
best_oof = None
best_pred = None

# FORWARD FEATURE SELECTION 
for k, FEAT in enumerate(['baseline'] + new_attribs):
    
    oof_svc = np.zeros(len(train_cop))
    pred_svc = np.zeros(len(test_cop))

    if FEAT != 'baseline':
        linearSVC_FEATURES_TO_ADD.append(FEAT)

    FOLDS=10

    oof_svc, pred_svc = cross_val_score(
        LinearSVC(C=0.1, probability=True),
        train_cop,
        test_cop,
        features + linearSVC_FEATURES_TO_ADD,
        FOLDS
    )
    # COMPUTE AVERAGE TEST PREDS
    pred_svc /= FOLDS

    if (np.all(pred_svc == 0)) or (np.all(pred_svc == 1)):
        print("\nThe array is composed entirely of zeros or ones.")
        print(f"Worse with {FEAT} at {metric}\n")
        linearSVC_FEATURES_TO_ADD.remove(FEAT)
        
    else:
        # COMPUTE CV VALIDATION AUC SCORE
        true = train.rainfall.values
        metric = roc_auc_score(true, oof_svc)
        
        if metric>best_auc:
            print(f"NEW BEST with {FEAT} at {metric}")
            best_auc = metric
            best_oof = oof_svc.copy()
            best_pred = pred_svc.copy()
        else:
            print(f"Worse with {FEAT} at {metric}")
            linearSVC_FEATURES_TO_ADD.remove(FEAT)

print(f"We achieved CV SVC AUC = {best_auc:.4f} adding {len(linearSVC_FEATURES_TO_ADD)} interactions features:")
print( linearSVC_FEATURES_TO_ADD )

"""

# output:
# We achieved CV SVC AUC = 0.8966 adding 16 interactions features:
# ['day_diff_lag2', 'sin day', 'tan_day - tan_mintemp', 'tan_day - tan_windspeed', 'cos_maxtemp x sin_humidity', 'tan_temparature - tan_sunshine', 'dewpoint - temparature', 'dewpoint x humidity', 'log_dewpoint + cloud', 'cos_dewpoint x sin_cloud', 'log_dewpoint + sunshine', 'dewpoint x windspeed', 'humidity_diff_lag3', 'tanh cloud2', 'cos sunshine2', 'windspeed_diff_lag4']

linearSVC_FEATURES_TO_ADD = ['day_diff_lag2', 'sin day', 'tan_day - tan_mintemp', 'tan_day - tan_windspeed', 'cos_maxtemp x sin_humidity', 'tan_temparature - tan_sunshine', 'dewpoint - temparature', 'dewpoint x humidity', 'log_dewpoint + cloud', 'cos_dewpoint x sin_cloud', 'log_dewpoint + sunshine', 'dewpoint x windspeed', 'humidity_diff_lag3', 'tanh cloud2', 'cos sunshine2', 'windspeed_diff_lag4']



from cuml.neighbors import KNeighborsClassifier
"""
knn_FEATURES_TO_ADD  = []
best_auc = 0
best_oof = None
best_pred = None

# FORWARD FEATURE SELECTION 
for k, FEAT in enumerate(['baseline'] + new_attribs):
    
    oof_knn = np.zeros(len(train_cop))
    pred_knn = np.zeros(len(test_cop))

    if FEAT != 'baseline':
        knn_FEATURES_TO_ADD.append(FEAT)

    FOLDS=10

    oof_knn, pred_knn = cross_val_score(
        KNeighborsClassifier(n_neighbors=201, p=1),
        train_cop,
        test_cop,
        features + knn_FEATURES_TO_ADD,
        FOLDS
    )

    # COMPUTE AVERAGE TEST PREDS
    pred_knn /= FOLDS

    if (np.all(pred_knn == 0)) or (np.all(pred_knn == 1)):
        print("\nThe array is composed entirely of zeros or ones.")
        print(f"Worse with {FEAT} at {metric}\n")
        knn_FEATURES_TO_ADD.remove(FEAT)
    
    else:
        # COMPUTE CV VALIDATION AUC SCORE
        true = train.rainfall.values
        metric = roc_auc_score(true, oof_knn)
        
        if metric>best_auc:
            print(f"NEW BEST with {FEAT} at {metric}")
            best_auc = metric
            best_oof = oof_knn.copy()
            best_pred = pred_knn.copy()
        else:
            print(f"Worse with {FEAT} at {metric}")
            knn_FEATURES_TO_ADD.remove(FEAT)

print(f"We achieved CV KNN AUC = {best_auc:.4f} adding {len(knn_FEATURES_TO_ADD)} interactions features:")
print( knn_FEATURES_TO_ADD )
"""

# Output
# We achieved CV KNN AUC = 0.8963 adding 42 interactions features:
# ['sin day', 'sin day2', 'cos_day x sin_pressure', 'tan_day - tan_pressure', 'tan_day - tan_maxtemp', 'tan_day - tan_temparature', 'cos_day x sin_mintemp', 'tan_day - tan_mintemp', 'tan_day - tan_humidity', 'cos_day x sin_cloud', 'tan_day - tan_cloud', 'inv_day x sunshine', 'sin pressure2', 'pressure x maxtemp', 'pressure x cloud', 'log_pressure + cloud', 'inv_pressure x cloud', 'tanh maxtemp2', 'log_maxtemp + humidity', 'log_maxtemp + cloud', 'tan_maxtemp - tan_cloud', 'tanh temparature2', 'temparature x day', 'tan_temparature - tan_maxtemp', 'temparature x dewpoint', 'tan_temparature - tan_humidity', 'cos_temparature x sin_cloud', 'tan_temparature - tan_cloud', 'tanh mintemp2', 'dewpoint x humidity', 'log_dewpoint + humidity', 'log_dewpoint + cloud', 'tan_dewpoint - tan_cloud', 'cos_dewpoint x sin_sunshine', 'tan_humidity - tan_temparature', 'inv_humidity x cloud', 'tan_cloud - tan_day', 'cloud - pressure', 'tan_cloud - tan_pressure', 'tan_cloud - tan_mintemp', 'sunshine + mintemp', 'sunshine + cloud']

knn_FEATURES_TO_ADD = ['sin day', 'sin day2', 'cos_day x sin_pressure', 'tan_day - tan_pressure', 'tan_day - tan_maxtemp', 'tan_day - tan_temparature', 'cos_day x sin_mintemp', 'tan_day - tan_mintemp', 'tan_day - tan_humidity', 'cos_day x sin_cloud', 'tan_day - tan_cloud', 'inv_day x sunshine', 'sin pressure2', 'pressure x maxtemp', 'pressure x cloud', 'log_pressure + cloud', 'inv_pressure x cloud', 'tanh maxtemp2', 'log_maxtemp + humidity', 'log_maxtemp + cloud', 'tan_maxtemp - tan_cloud', 'tanh temparature2', 'temparature x day', 'tan_temparature - tan_maxtemp', 'temparature x dewpoint', 'tan_temparature - tan_humidity', 'cos_temparature x sin_cloud', 'tan_temparature - tan_cloud', 'tanh mintemp2', 'dewpoint x humidity', 'log_dewpoint + humidity', 'log_dewpoint + cloud', 'tan_dewpoint - tan_cloud', 'cos_dewpoint x sin_sunshine', 'tan_humidity - tan_temparature', 'inv_humidity x cloud', 'tan_cloud - tan_day', 'cloud - pressure', 'tan_cloud - tan_pressure', 'tan_cloud - tan_mintemp', 'sunshine + mintemp', 'sunshine + cloud']



params_xgb_1 = {
    'objective': 'binary',
    "device": "cpu",
    "random_state":42,
    'n_jobs': -1,
    'verbose': 0,
    'n_estimators': 100, 
}

"""
xgb_FEATURES_TO_ADD  = []
best_auc = 0
best_oof = None
best_pred = None

# FORWARD FEATURE SELECTION 
for k, FEAT in enumerate(['baseline'] + new_attribs):
    
    oof_xgb = np.zeros(len(train_cop))
    pred_xgb = np.zeros(len(test_cop))

    if FEAT != 'baseline':
        xgb_FEATURES_TO_ADD.append(FEAT)

    FOLDS=10

    oof_xgb, pred_xgb = cross_val_score(
        XGBClassifier(**params_xgb, n_estimators=50_000),
        train_cop,
        test_cop,
        features + xgb_FEATURES_TO_ADD,
        FOLDS
    )

    # COMPUTE AVERAGE TEST PREDS
    pred_xgb /= FOLDS
    
    if (np.all(pred_xgb == 0)) or (np.all(pred_xgb == 1)):
        print("\nThe array is composed entirely of zeros or ones.")
        print(f"Worse with {FEAT} at {metric}\n")
        xgb_FEATURES_TO_ADD.remove(FEAT)
    
    else:
        # COMPUTE CV VALIDATION AUC SCORE
        true = train.rainfall.values
        metric = roc_auc_score(true, oof_xgb)
        
        if metric>best_auc:
            print(f"NEW BEST with {FEAT} at {metric}")
            best_auc = metric
            best_oof = oof_xgb.copy()
            best_pred = pred_xgb.copy()
        else:
            print(f"Worse with {FEAT} at {metric}")
            xgb_FEATURES_TO_ADD.remove(FEAT)

print(f"We achieved CV XGBoost AUC = {best_auc:.4f} adding {len(xgb_FEATURES_TO_ADD)} interactions features:")
print( xgb_FEATURES_TO_ADD )
"""
# We achieved CV XGBoost AUC = 0.8958 adding 10 interactions features:
# ['day_diff_lag1', 'day_diff_lag2', 'log_day + pressure', 'inv_day x pressure', 'pressure x day', 'pressure + day', 'cos_pressure x sin_maxtemp', 'pressure x temparature', 'inv_pressure x temparature', 'inv_pressure x sunshine']

xgb_FEATURES_TO_ADD = ['day_diff_lag2', 'day_diff_lag3', 'cos day', 'cos day2', 'day x pressure', 'cos_day x sin_maxtemp', 'tan_day - tan_maxtemp', 'day x temparature', 'inv_pressure x maxtemp', 'inv_maxtemp x mintemp', 'inv_temparature x dewpoint', 'inv_temparature x sunshine', 'inv_mintemp x sunshine', 'tanh dewpoint2', 'cloud x windspeed']


from lightgbm import LGBMClassifier

params_lgb_1 = {
    'objective': 'binary',
    "device": "cpu",
    "random_state":42,
    'n_jobs': -1,
    'verbose': 0,
    'max_bins': 195,
    'n_estimators': 897, 
    'learning_rate': 0.0041830283250754935,
}

# ......

# We achieved CV Lightgbm AUC = 0.8908 adding 18 interactions features:
lgb_FEATURES_TO_ADD = ['day_diff_lag1', 'day_diff_lag2', 'cos day', 'tanh day2', 'cos_day x sin_pressure', 'cos_day x sin_maxtemp', 'cos_pressure x sin_maxtemp', 'tan_pressure - tan_humidity', 'sin temparature', 'tanh temparature', 'tan_temparature - tan_pressure', 'cos_temparature x sin_dewpoint', 'tan_temparature - tan_dewpoint', 'sin mintemp', 'mintemp x maxtemp', 'humidity - dewpoint', 'cloud_diff_lag3', 'winddirection_diff_lag3']


union_new_features = list(set(linearSVC_FEATURES_TO_ADD).union(knn_FEATURES_TO_ADD).union(xgb_FEATURES_TO_ADD).union(lgb_FEATURES_TO_ADD))
print("Union of new features:\n", union_new_features) 


"""
sample_submission = pd.read_csv("dataset/sample_submission.csv")

all_models = {
    'LinearSVC': LinearSVC(C=0.1, probability=True),
    'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=201, p=1),
    'XGBooster_1': XGBClassifier(**params_xgb_1),
    'LightGBM_1': LGBMClassifier(**params_lgb_1)
}

for model_name, model in all_models.items():
    print(f"### {model_name} ###")
    
    n_repeats = 6

    preds_oof = np.zeros(len(train_cop))
    preds_test = np.zeros(len(test_cop))
    
    for f in range(n_repeats):
        FOLDS = 73  # (73 folds) 30 DAYS to test
        
        if isinstance(model, LinearSVC):
            FEATURES_TO_ADD = linearSVC_FEATURES_TO_ADD
        elif isinstance(model, KNeighborsClassifier):
            FEATURES_TO_ADD = knn_FEATURES_TO_ADD
        elif isinstance(model, XGBClassifier):
            FEATURES_TO_ADD = xgb_FEATURES_TO_ADD
        elif isinstance(model, LGBMClassifier):
            FEATURES_TO_ADD = lgb_FEATURES_TO_ADD
        
        oof_model, sum_pred_model = cross_val_score(
            model=model,
            train_fuc=train_cop,
            test_func=test_cop,
            features_to_train=features + FEATURES_TO_ADD,
            FOLDS=FOLDS,
            i=f
        )
        
        pred_model = sum_pred_model / FOLDS 
        
        true = train_cop.rainfall.values
        metric = roc_auc_score(true, oof_model)
        
        preds_oof += oof_model / n_repeats
        preds_test += pred_model / n_repeats
        
        print(f"Repeat {f}: {metric:.6f}")
    print(f"Final - mean auc: {roc_auc_score(true, preds_oof):.5f}\n")

    ### OOF PRED SAVING ###
    df_oof = pd.DataFrame({"id": train.index})
    df_oof['rainfall'] = preds_oof
    df_oof.to_csv(f"best_models/{model_name}_oof_predictions.csv", index=False)

    ### PRED TEST SAVING ###
    sample_submission[target] = preds_test
    sample_submission.to_csv(f"best_models/{model_name}_test_predictions.csv", index=False)

print("All best models saved!")
"""


import os

path = "/kaggle/input/oof-test-svc-knn-xgb-lgb-rainfall/"

all_preds_oof = {}
all_preds_test = {}

for file in os.listdir(path):
    full_path = os.path.join(path, file)
    if file.endswith("_oof_predictions.csv"):
        model_name = file.replace("_oof_predictions.csv", "")
        all_preds_oof[model_name] = pd.read_csv(full_path)
    elif file.endswith("_test_predictions.csv"):
        model_name = file.replace("_test_predictions.csv", "")
        all_preds_test[model_name] = pd.read_csv(full_path)

print("Modelos OOF:", list(all_preds_oof.keys()))
print("Modelos Test:", list(all_preds_test.keys()))



import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

y_true = train_cop.rainfall.values

for model_name, df in all_preds_oof.items():
    y_prob = df.rainfall.values.flatten()
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plota a ROC Curve
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


oof_preds = [pred.rainfall.values for pred in all_preds_oof.values()]

test_preds = [pred.rainfall.values for pred in all_preds_test.values()]

print("Number of OOF predictions:", len(oof_preds))
print("Number of test predictions:", len(test_preds))


def find_best_weight(candidate_prediction, ensemble_prediction, true_values):
    all_aucs = []
    
    for w in [i / 1000 for i in range(1, 1000)]:
        ensemble_pred = w * candidate_prediction + (1 - w) * ensemble_prediction
        ensemble_auc = roc_auc_score(true_values, ensemble_pred)
        all_aucs.append(ensemble_auc)
    
    best_weight = (np.argmax(all_aucs) + 1) / 1000  # +1 porque começa em 0.001
    best_auc = max(all_aucs)

    return best_weight, best_auc


from itertools import combinations

true_values = train_cop.rainfall.values
best_overall_auc = 0
best_pair = None
best_final_weight = None

for i, j in combinations(range(len(oof_preds)), 2):
    print(f"Testing combination between prediction {i} and {j}...")

    best_w, best_auc = find_best_weight(oof_preds[i], oof_preds[j], true_values)

    if best_auc > best_overall_auc:
        best_overall_auc = best_auc
        best_pair = (i, j)
        best_final_weight = best_w

    print(f"Best weight for ({i}, {j}): {best_w}, AUC: {best_auc}\n")

# Final result
print(f"The best combination was between predictions {best_pair} with weight {best_final_weight}")
print(f"Best final AUC: {best_overall_auc}")



def build_ensemble(ensemble1, ensemble2, weight):
    return (ensemble1 * weight) + (ensemble2 * (1 - weight))


ensemble_23 = build_ensemble(oof_preds[2], oof_preds[3], 0.296)
print(f"AUC for ensemble_23: {roc_auc_score(true_values, ensemble_23):.4f}")


w1, auc1 = find_best_weight(oof_preds[1], ensemble_23, true_values)

ensemble_23_1 = build_ensemble(oof_preds[1], ensemble_23, w1)

print(f"AUC for ensemble_23_1: {roc_auc_score(true_values, ensemble_23_1):.4f}")


w2, auc2 = find_best_weight(oof_preds[0], ensemble_23_1, true_values)

ensemble_23_1_0 = build_ensemble(oof_preds[0], ensemble_23_1, w2)

print(f"AUC for ensemble_23_1_0: {roc_auc_score(true_values, ensemble_23_1_0):.4f}")


ensemble_test_23 = build_ensemble(test_preds[2], test_preds[3], 0.296)
ensemble_test_23_1 = build_ensemble(test_preds[1], ensemble_test_23, 0.327)


sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")

sample_submission[target] = ensemble_test_23_1
sample_submission.to_csv("sub_ensemble_23_1.csv", index=False)

display(sample_submission.head())
display(sample_submission.shape)

