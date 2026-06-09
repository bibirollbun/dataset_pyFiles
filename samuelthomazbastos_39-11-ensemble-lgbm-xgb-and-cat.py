import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import kaggle
from tqdm import tqdm
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import optuna
import catboost as cb
import lightgbm as lgb
from sklearn.model_selection import train_test_split


# !kaggle competitions download playground-series-s5e2
# import zipfile
# import os

# def unzip_file(zip_path, extract_to):
#     """
#     Unzips a zip file to the specified directory.
    
#     Args:
#         zip_path (str): The path to the zip file.
#         extract_to (str): The directory to extract the contents to.
#     """
#     try:
#         # Ensure the output directory exists
#         os.makedirs(extract_to, exist_ok=True)
        
#         # Open and extract the zip file
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             zip_ref.extractall(extract_to)
#         print(f"File unzipped successfully to {extract_to}")
    
#     except zipfile.BadZipFile:
#         print("Error: The file is not a valid zip file.")
#     except FileNotFoundError:
#         print("Error: The zip file does not exist.")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

# # Example usage
# zip_file_path = "./playground-series-s5e2.zip"  # Replace with your zip file path
# output_directory = "./"  # Replace with your desired output directory
# unzip_file(zip_file_path, output_directory)


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
training_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


train.drop('id', axis = 1, inplace=True)
test.drop('id', axis = 1, inplace=True)
training_extra.drop('id', axis = 1, inplace=True)



target = 'Price'


train.info()


train.head()


test.head()


test.info()


categorical_columns = train.select_dtypes(include=['object', 'category']).columns
numerical_columns = [col for col in train.columns if pd.api.types.is_numeric_dtype(train[col])]


for col in categorical_columns:
    print(col, train[col].unique())


for col in categorical_columns:
    # temp_serie = train[col ].copy()
    sns.boxplot(data=train, x = col, y = 'Price')
    # ax.bar_label(ax.containers[0])
    plt.title(col + ' Price Boxplot')
    plt.show()



for col in categorical_columns:
    # temp_serie = train[col ].copy()
    sns.boxplot(data=training_extra, x = col, y = 'Price')
    # ax.bar_label(ax.containers[0])
    plt.title(col + ' Price Boxplot')
    plt.show()



sns.pairplot(train[numerical_columns])


for col in train.columns:
    is_na_df = train[train[col].isna()].copy()
    is_not_na_df = train[~train[col].isna()].copy()

    print(col, round(100*is_na_df.shape[0]/train.shape[0], 2))



for col in train.columns:
    is_na_df = train[train[col].isna()].copy()
    is_not_na_df = train[~train[col].isna()].copy()
    
    if is_na_df.shape[0] == 0:
        print(col, 'is complete')

    else:
        print(
            f"""{col}
Is NA price: {is_na_df.Price.median()}
Is not NA prince: {is_not_na_df.Price.median()}
"""
        )

        sns.boxplot(x = train[col].isna(), y = train[target])
        plt.show()



combinations = {}

for index, row in tqdm(train.isna().iterrows()):
    temp_combination = []
    for index in row.index:    
        if row[index]:
            temp_combination.append(index)

    if len(temp_combination) <= 1:        
        continue
    elif str(temp_combination) not in combinations.keys():
        combinations[str(temp_combination)] = 1
    else:
        combinations[str(temp_combination)] +=1





combinations_df = pd.DataFrame(combinations, index= ['number']).T
combinations_df = combinations_df.reset_index()


combinations_df.sort_values(by = 'number', ascending=False).head(20)


final_df = train.copy()


import pandas as pd

def impute_missing_values(df: pd.DataFrame, group_by_cols: list, target_col: str, metric: str = "median") -> pd.DataFrame:
    """
    Imputes missing values in a numeric column using either the median or mean, grouping by categorical variables.
    
    :param df: Pandas DataFrame
    :param group_by_cols: List of categorical columns for grouping
    :param target_col: Numeric column with missing values to be imputed
    :param metric: Imputation method ("median" or "mean")
    :return: DataFrame with imputed values
    """
    if metric not in ["median", "mean"]:
        raise ValueError("Metric must be 'median' or 'mean'")
    
    agg_func = df.groupby(group_by_cols)[target_col].transform(metric)
    overall_value = df[target_col].agg(metric)
    
    df[target_col] = df[target_col].fillna(agg_func)
    df[target_col] = df[target_col].fillna(overall_value)
    
    return df

# # Exemplo de uso
# data = {
#     "Category": ["A", "A",  "A", "A", "C", "C"],
#     "Subcategory": ["X", "X", "X", "X", "X", "Y"],
#     "Value": [10, None, 15, None, 20, 25]
# }

# df = pd.DataFrame(data)
# df = impute_missing_values(df, ["Category", "Subcategory"], "Value", metric="median")
# print(df)


def feature_engineering(df, test = False):
    
    final_df = df.copy()
    final_df[categorical_columns] = final_df[categorical_columns].fillna('Missing')
    for num_col in numerical_columns:

        
        
        if (test) and (num_col == 'Price'):
            continue

        is_na_col = final_df[num_col].isna().copy()        

        if (num_col == 'Price') or ((is_na_col.sum()) == 0):
            continue
        else:
            final_df['is_missing_' + num_col] = is_na_col.astype(int)
            final_df = impute_missing_values(df=final_df, group_by_cols=list(categorical_columns), target_col=num_col, metric='median')

    
    final_df = pd.get_dummies(final_df, columns=categorical_columns, drop_first=True, dtype=int)


    return final_df


imputed_train_df = feature_engineering(df = train, test = False)


X = imputed_train_df.drop(target, axis=1)
y = imputed_train_df[target]


def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log = True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42
    }
    
    model = XGBRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
    return -cv_scores.mean()  # Minimize MSE


# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)
# xgb_best_params = study.best_params

# 39.00187811790006



xgb_best_params = {'n_estimators': 188,
 'learning_rate': 0.08624149976576163,
 'max_depth': 3,
 'subsample': 0.6724287534421187,
 'colsample_bytree': 0.5008144875002977,
 'random_state': 42
 }


# Objective function for Optuna
def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log = True),
        'depth': trial.suggest_int('depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'random_seed': 42,
        'loss_function': 'RMSE',
        'verbose': 0
    }
    
    model = cb.CatBoostRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
    return -cv_scores.mean()  # Minimize RMSE

# study_cat = optuna.create_study(direction='minimize')
# study_cat.optimize(objective_cat, n_trials=50)


# study_cat.best_value
# 38.999539197373295


# study_cat.best_params
cat_best_params = {
    'iterations': 262,
    'learning_rate': 0.09401835465217484,
    'depth': 4,
    'subsample': 0.6563415975882356,
    'colsample_bylevel': 0.6842356283311213,
    'random_state': 42
}


def objective_lgbm(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log = True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42,
        'metric': 'rmse'
    }
    
    model = lgb.LGBMRegressor(**params, verbosity = -1)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
    return -cv_scores.mean()  # Minimize RMSE

# study_lgbm = optuna.create_study(direction='minimize')
# study_lgbm.optimize(objective_lgbm, n_trials=50)


#39.00468469628485
lgbm_best_params = {
    'num_leaves': 27,
    'learning_rate': 0.021779565084541153,
    'n_estimators': 370,
    'subsample': 0.6502074174638701,
    'colsample_bytree': 0.522675202327236,
    'random_state': 42
}


regressors = {
    'LGBM': lgb.LGBMRegressor(**lgbm_best_params, verbosity = -1),
    'XGBRegressor': XGBRegressor(**xgb_best_params),
    'Catboost': cb.CatBoostRegressor(**cat_best_params, verbose=0)
}


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42
)


predictions_df = pd.DataFrame()
for model_name, model in tqdm(regressors.items()):
    model.fit(X_train, y_train)

    predictions_df[model_name] = model.predict(X_test)



def optimize_weights_with_optuna_cv(feature_df: pd.DataFrame, true_values: np.ndarray, n_splits: int = 5):
    """
    This function uses Optuna to find the optimal weights for combining the predictions
    of multiple models to minimize the RMSE using K-fold cross-validation (only on validation set).

    Parameters:
    - feature_df: DataFrame with columns as regressor names and rows as predictions.
    - true_values: Array of real values to compare the predictions with.
    - n_splits: Number of splits for K-fold cross-validation (default is 5).

    Returns:
    - optimal_weights: A dictionary where keys are regressor names and values are the optimal weights.
    """
    
    # Objective function for Optuna with cross-validation
    def objective(trial):
        # Number of models (regressors)
        num_models = feature_df.shape[1]
        
        # Sample weights between -1 and 1 for each model
        weights = [trial.suggest_float(f"weight_{i}", -1.0, 1.0) for i in range(num_models)]
        
        # K-fold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_rmse = []

        for train_idx, val_idx in kf.split(feature_df):
            # Split the data into training and validation sets
            X_train, X_val = feature_df.iloc[train_idx], feature_df.iloc[val_idx]
            y_train, y_val = true_values[train_idx], true_values[val_idx]

            # Compute the weighted sum of predictions for validation data
            weighted_predictions_val = np.dot(X_val, weights)
            # Compute RMSE for validation data
            rmse_val = np.sqrt(mean_squared_error(y_val, weighted_predictions_val))

            # Store RMSE for validation set
            cv_rmse.append(rmse_val)

        # Return the average RMSE across all folds for the validation sets
        return np.mean(cv_rmse)
    
    # Create an Optuna study to minimize the objective
    study = optuna.create_study(direction="minimize")
    
    # Optimize the objective function
    study.optimize(objective, n_trials=1000)
    
    # Get the best weights found by Optuna
    optimal_weights = study.best_trial.params

    print('The best metric was:', study.best_value)

    # Convert optimal weights into a dictionary with regressor names as keys
    optimal_weights_dict = {feature_df.columns[i]: optimal_weights[f"weight_{i}"] for i in range(feature_df.shape[1])}
    
    return optimal_weights_dict


# optimal_weights_dict = optimize_weights_with_optuna_cv(
#     feature_df=predictions_df,
#     true_values=y_test.values
# )

optimal_weights_dict = {'LGBM': 0.5076322274323838,
 'XGBRegressor': 0.5572622529216305,
 'Catboost': -0.06474159272608021}


imputed_test = feature_engineering(test, test=True)


def create_final_prediction(X, y, test_df, regressors, weights):

    predictions_df = pd.DataFrame()

    for regressor_name, regressor in tqdm(regressors.items()):
        
        regressor.fit(X, y)
        predictions_df[regressor_name] = regressor.predict(test_df)*weights[regressor_name]

    return predictions_df.sum(axis=1)


submission_pred = create_final_prediction(X, y, imputed_test, regressors, optimal_weights_dict)


final_submission = pd.DataFrame()
final_submission['id'] = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')['id'].copy()
final_submission['Price'] = submission_pred

final_submission.to_csv('submission1.csv', index = False)


inputed_extra_data = feature_engineering(training_extra)


X_extra = inputed_extra_data.drop(target, axis=1)
y_extra = inputed_extra_data[target]


submission_pred_extra = create_final_prediction(X_extra, y_extra, imputed_test, regressors, optimal_weights_dict)


final_submission_extra = pd.DataFrame()
final_submission_extra['id'] = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')['id'].copy()
final_submission_extra['Price'] = submission_pred_extra

final_submission_extra.head()

