import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


def split_numerical_categorical(df):
    """
    Splits the columns of a DataFrame into numerical and categorical features.

    Parameters:
    df (pandas.DataFrame): The DataFrame to split.

    Returns:
    tuple: A tuple containing two lists - numerical columns and categorical columns.
    """
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    return numerical_cols, categorical_cols


def nums_combinations(df_, numerical_cols):
    for col1 in numerical_cols:
        for col2 in numerical_cols:
            if col1 != col2:
                df_[f'{col1}__{col2}__dzielenie'] = df_[col1]/df_[col2]
                df_[f'{col1}__{col2}__mnozenie'] = df_[col1]*df_[col2]
                df_[f'{col1}__{col2}__dodawanie'] = df_[col1]+df_[col2]
                df_[f'{col1}__{col2}__odejmowanie'] = df_[col1]-df_[col2]
    return df_


path = '/kaggle/input/playground-series-s5e1/'


df_train = pd.read_csv(f'{path}train.csv')
df_test = pd.read_csv(f'{path}test.csv')


df_train = df_train[df_train['date']>="2014-01-01"]


df_train['date'].max()


df_train['date'].min()


df_test['date'].max()


df_test['date'].min()


df_train = df_train.dropna(subset=['num_sold'])


df_train.head()


df_test.head()


df_train.drop('id', axis=1, inplace=True)
df_test.drop('id', axis=1, inplace=True)


from datetime import datetime


def extract_date_features(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    #df['Year'] = df[column_name].dt.year
    df['Month'] = df[column_name].dt.month
    df['Day'] = df[column_name].dt.day
    df['Weekday'] = df[column_name].dt.weekday
    df['Week'] = df[column_name].dt.isocalendar().week
    df['Quarter'] = df[column_name].dt.quarter
    df['Day of Year'] = df[column_name].dt.dayofyear
    df['Is Month Start'] = df[column_name].dt.is_month_start
    df['Is Month End'] = df[column_name].dt.is_month_end
    df['Is Leap Year'] = df[column_name].dt.is_leap_year
    #df['Days Since Start'] = (datetime.now() - df[column_name]).dt.days
    df.drop(column_name, inplace=True, axis = 1)
    return df


column_name="date"


df_train = extract_date_features(df_train, column_name)


df_test = extract_date_features(df_test, column_name)


numerical_cols, categorical_cols = split_numerical_categorical(df_train)


numerical_cols, categorical_cols


target_column = 'num_sold'
numerical_cols.remove(target_column)


def transform_categoricals(df_, categorical_cols):
    return pd.get_dummies(df_, columns=categorical_cols) 


y_train = df_train[target_column]
df_train = df_train.drop([target_column], axis=1)
df_all = pd.concat([df_train, df_test], axis=0)


for c in categorical_cols:
    print(c, "n unique:",df_all[c].nunique())


df_all = transform_categoricals(df_all, categorical_cols)


from sklearn.model_selection import KFold
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor

# Assuming df_all, y_train, and df_train are already defined
X_train = df_all[:df_train.shape[0]]
X_test = df_all[df_train.shape[0]:]

# Prepare arrays to store out-of-fold predictions and test set predictions
oof_preds = np.zeros(X_train.shape[0])
test_preds = np.zeros(X_test.shape[0])

# Initialize 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Loop over each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Fold {fold + 1}")
    
    # Split data into train and validation sets
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Initialize and train the model
    model = LGBMRegressor()
    model.fit(X_tr, y_tr)
    
    # Predict on validation set and test set
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / kf.n_splits

# Final averaged predictions for the test set
y_pred_lgbm = test_preds


from xgboost import XGBRegressor
oof_preds = np.zeros(X_train.shape[0])
test_preds = np.zeros(X_test.shape[0])
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Fold {fold + 1}")
    
    # Split data into train and validation sets
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Initialize and train the model
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=100
    )
    
    # Predict on validation set and test set
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / kf.n_splits

# Final averaged predictions for the test set
y_pred_XGB = test_preds


from catboost import CatBoostRegressor 


oof_preds = np.zeros(X_train.shape[0])
test_preds = np.zeros(X_test.shape[0])


for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Fold {fold + 1}")
    
    # Split data into train and validation sets
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Initialize and train the model
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        subsample=0.8,
        colsample_bylevel=0.8,
        loss_function='RMSE',
        random_seed=42,
        eval_metric='RMSE',
        verbose=100,
        early_stopping_rounds=100
    )
    
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        #cat_features=categorical_features,
        use_best_model=True
    )
    
    # Predict on validation set and test set
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / kf.n_splits

# Final averaged predictions for the test set
y_pred_CATB = test_preds


y_pred_XGB.max()


y_pred_lgbm.max()


y_pred_CATB.max()


y_pred = (y_pred_XGB + y_pred_lgbm)/2


y_pred


y_pred.shape


ssub = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")


ssub['num_sold'] = y_pred


ssub.to_csv('submission.csv', index = False)


ssub







