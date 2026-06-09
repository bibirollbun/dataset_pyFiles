import os
from pathlib import Path

competition_data = "playground-series-s5e2"

iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
if iskaggle: path = Path('../input/' + competition_data)
else:
    path = Path('data')
    if not path.exists():
        import zipfile,kaggle
        kaggle.api.competition_download_cli(str(competition_data))
        zipfile.ZipFile(f'{competition_data}.zip').extractall(path)


from fastai.tabular.all import *

pd.options.display.float_format = '{:.2f}'.format


def concat_df(test_data, train_data, df_train_extra):
    # Returns a concatenated df of training and test set
    return pd.concat([test_data, train_data, df_train_extra], sort=True).reset_index(drop=True)

def divide_df(all_data):
    # Returns divided dfs of training and test set
    return all_data.loc[:199999], all_data.loc[200000:]

df_train = pd.read_csv(path/'train.csv', index_col='id')
df_train_extra = pd.read_csv(path/'training_extra.csv', index_col='id')
df_test = pd.read_csv(path/'test.csv', index_col='id')
df_all = concat_df(df_test, df_train, df_train_extra)
df_train


df_train["Brand"] = df_train["Brand"].fillna("unknown")


import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


# # Define categorical and continuous features
# cat_names = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
# cont_names = ['Weight Capacity (kg)']
# target = "Price"

# # Create a copy of the data to avoid modifying the original
# X = df_all[cat_names + cont_names].copy()
# y = df_all[target].loc[200000:].copy()

# # Handle missing values first
# cat_imputer = SimpleImputer(strategy="most_frequent")
# num_imputer = SimpleImputer(strategy="mean")

# # Impute missing values
# X[cat_names] = cat_imputer.fit_transform(X[cat_names])
# X[cont_names] = num_imputer.fit_transform(X[cont_names])

# # Encode categorical variables
# categorical_encoders = {}
# for col in cat_names:
#     categorical_encoders[col] = LabelEncoder()
#     X[col] = categorical_encoders[col].fit_transform(X[col])
#     X[col] = X[col].astype('int32')  # Ensure integer type

# filled_df_test, filled_df_train = divide_df(X)
# X_train, X_valid, y_train, y_valid = train_test_split(filled_df_train, y, test_size=0.2, random_state=42)

# # Convert data into LightGBM dataset
# lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat_names, free_raw_data=False)
# lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=cat_names, free_raw_data=False)

# # Define the objective function for Optuna
# def objective(trial):
#     params = {
#         'feature_pre_filter': False,
#         "objective": "regression",
#         "metric": "rmse",
#         "boosting_type": "gbdt",
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "num_leaves": trial.suggest_int("num_leaves", 20, 150),
#         "max_depth": trial.suggest_int("max_depth", -1, 15),
#         "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
#         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#         "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
#         "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
#         "random_state": 42
#     }

#     model = lgb.train(
#         params,
#         lgb_train,
#         valid_sets=[lgb_valid],
#         callbacks=[lgb.early_stopping(50, verbose=True)]
#     )

#     y_pred = model.predict(X_valid)
#     rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
#     return rmse

# # Run Optuna optimization
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=50)

# # Get the best parameters
# best_params = study.best_params
# print("Best Parameters:", best_params)

# # Train the final model with the best parameters
# final_model = lgb.train(
#     best_params,
#     lgb_train,
#     valid_sets=[lgb_valid],
#     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
# )

# # Predict and evaluate
# y_pred = final_model.predict(X_valid)
# rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
# print("Final Validation RMSE:", rmse)


# Define categorical and continuous features
cat_names = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
cont_names = ['Weight Capacity (kg)']
target = "Price"

# Create a copy of the data to avoid modifying the original
X = df_all[cat_names + cont_names].copy()
y = df_all[target].loc[200000:].copy()

# Handle missing values first
cat_imputer = SimpleImputer(strategy="most_frequent")
num_imputer = SimpleImputer(strategy="mean")

# Impute missing values
X[cat_names] = cat_imputer.fit_transform(X[cat_names])
X[cont_names] = num_imputer.fit_transform(X[cont_names])

# Encode categorical variables
categorical_encoders = {}
for col in cat_names:
    categorical_encoders[col] = LabelEncoder()
    X[col] = categorical_encoders[col].fit_transform(X[col])
    X[col] = X[col].astype('int32')  # Ensure integer type

filled_df_test, filled_df_train = divide_df(X)
X_train, X_valid, y_train, y_valid = train_test_split(filled_df_train, y, test_size=0.2)

# LightGBM dataset
lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat_names)
lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=cat_names)

# LightGBM parameters
params = {
    "random_state": 42,  # Ensures reproducibility
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.28783109603098567,
    "num_leaves": 112,
    "max_depth": 4,
    "min_child_samples": 31,
    "subsample": 0.6970193671311494,
    "colsample_bytree": 0.6520537605325276,
    "lambda_l1": 3.173406154551361e-06,
    "lambda_l2": 0.0019074721234898398
}

# Train LightGBM model
model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_valid],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# Predict and evaluate
y_pred = model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print("Validation RMSE:", rmse)


preds = model.predict(filled_df_test)


df_sub = pd.DataFrame({"id":df_test.index,"Price":preds})
df_sub.to_csv('submission.csv', index=False)
!head submission.csv

