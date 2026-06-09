import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


file_path = '/kaggle/input/exploring-predictive-health-factors/train.csv'
df = pd.read_csv(file_path)



df.info()


nan_rows_count = df.isnull().any(axis=1).sum()

print(f"Number of rows with at least one blank entry: {nan_rows_count}")


test = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/test.csv')


test.head()


sub_df = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/sample_submission.csv')
sub_df.head()


df['PCOS'].value_counts()


df['PCOS'] = df['PCOS'].map({'No': 0, 'Yes': 1})


def preprocess_data(df):
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else None
            if mode_value is not None:
                df[col] = df[col].fillna(mode_value)
            else:
                df[col] = df[col].fillna("Unknown")  
        elif df[col].dtype in ['int64', 'float64']:
            mean_value = df[col].mean()
            df[col] = df[col].fillna(mean_value)



y = df['PCOS']
df = df.drop(columns=['PCOS','ID'])
test = test.drop(columns=['ID'])


preprocess_data(df)
preprocess_data(test)


import matplotlib.pyplot as plt


unique_counts = {col: df[col].nunique() for col in df.columns if df[col].dtype == 'object' or df[col].dtype.name == 'category'}


plt.figure(figsize=(8, 6))
plt.bar(unique_counts.keys(), unique_counts.values(), color='skyblue')
plt.xlabel("Columns", fontsize=12)
plt.ylabel("Number of Unique Elements", fontsize=12)
plt.title("Number of Unique Elements in Object/Category Columns", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



def get_columns_to_encode(df):
    return [col for col in df.columns if (df[col].dtype == 'object' or df[col].dtype.name == 'category') and df[col].nunique() <= 10]


def apply_one_hot_encoding(df, columns_to_encode):
    return pd.get_dummies(df, columns=columns_to_encode, drop_first=False)


columns_to_encode_df = get_columns_to_encode(df)
columns_to_encode_test = get_columns_to_encode(test)


df_encoded = apply_one_hot_encoding(df, columns_to_encode_df)
test_encoded = apply_one_hot_encoding(test, columns_to_encode_test)


df_encoded, test_encoded = df_encoded.align(test_encoded, join='left', axis=1, fill_value=0)



num_columns = df_encoded.shape[1]
print(f"Number of columns: {num_columns}")



num_columns = test_encoded.shape[1]
print(f"Number of columns: {num_columns}")


import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X = df_encoded
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

cat_features = [col for col in X.columns if X[col].dtype == 'object']

def objective(trial):
    cat_params = dict(
        iterations=trial.suggest_int("iterations", 100, 1000),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        depth=trial.suggest_int("depth", 3, 12),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
        bagging_temperature=trial.suggest_float('bagging_temperature', 0, 2.5),
        random_strength=trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        task_type='GPU',
        early_stopping_rounds=200,
        verbose=False
    )
    
    model = CatBoostRegressor(**cat_params)
    X_train_pool = Pool(X_train, y_train, cat_features=cat_features)
    X_valid_pool = Pool(X_val, y_val, cat_features=cat_features)
    model.fit(X=X_train_pool, eval_set=X_valid_pool)
    
    y_pred = model.predict(X_val)
    score = mean_squared_error(y_val, y_pred, squared=False)
    
    return score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=150)

best_params = study.best_params

final_model = CatBoostRegressor(**best_params)
X_train_pool = Pool(X_train, y_train, cat_features=cat_features)
final_model.fit(X=X_train_pool)

test_predictions = final_model.predict(test_encoded)


test_predictions = np.clip(test_predictions, 0, 1)


sub_df['PCOS'] = test_predictions
sub_df.to_csv('submission.csv', index=False)


