# update libraries
!pip install --upgrade xgboost scikit-learn


%load_ext cudf.pandas


import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn-related
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder

# XGBoost
import xgboost as xgb

# feature engineering
from itertools import combinations

# QoL libraries
from tqdm.notebook import tqdm

# hyperparameters
SEED = 42

print(f"Used XGBoost version: {xgb.__version__}")


!ls /kaggle/input/


!ls /kaggle/input/bank-marketing-dataset-full


!ls /kaggle/input/playground-series-s5e8


train_path = "/kaggle/input/playground-series-s5e8/train.csv"
test_path = "/kaggle/input/playground-series-s5e8/test.csv"
original_data_path = "/kaggle/input/bank-marketing-dataset-full/bank-full.csv"
submission_data_path = "/kaggle/input/playground-series-s5e8/sample_submission.csv"


train_df = pd.read_csv(train_path, index_col="id")
test_df = pd.read_csv(test_path, index_col="id")
original_df = pd.read_csv(original_data_path, delimiter=";")

train_df.head()


original_df.head()


# prepare test dataframe
test_df["y"] = -1

# prepare original dataset
if (original_df["y"] == "no").any():
    original_df["y"] = original_df["y"].map({"yes": 1, "no": 0})

if original_df.index.name != "id":
    original_df["id"] = (np.arange(len(original_df)) + 1e6).astype(int)
    original_df = original_df.set_index("id")

# combine dataframes
combined_df = pd.concat([train_df, test_df, original_df])

combined_df.sample(5)


categorical_columns: list[str] = []
numerical_columns: list[str] = []

# iterate through all columns except the label
for column in combined_df.columns[:-1]:
    if combined_df[column].dtype == "object":
        categorical_columns.append(column)
    else:
        numerical_columns.append(column)

print(f"Categorical columns: {categorical_columns}")
print(f"Numerical columns: {numerical_columns}")


categorized_numerical_columns: list[str] = []
categorical_column_sizes: dict[str, int] = {}

# add numerical columns represented as categorical
for column in numerical_columns:
    new_column_name = f"{column}-categorized"
    categorized_numerical_columns.append(new_column_name)
    
    combined_df[new_column_name], _ = combined_df[column].factorize()
    combined_df[new_column_name] = combined_df[new_column_name].astype(np.int32)

    categorical_column_sizes[new_column_name] = combined_df[new_column_name].max() + 1

# encode the categorical columns
for column in categorical_columns:
    combined_df[column], _ = combined_df[column].factorize()
    combined_df[column] = combined_df[column].astype(np.int32)

    categorical_column_sizes[column] = combined_df[column].max() + 1

print(f"New categorical columns: {categorized_numerical_columns}\n")
print(f"All categorical column sizes: {categorical_column_sizes}")

combined_df.sample(5)


categorical_column_pairs = combinations(categorical_columns + categorized_numerical_columns, 2)
new_column_values: dict[str, pd.Series] = {}
combined_categorical_columns: list[str] = []

# create categorical column pairs with new unique values
for column1, column2 in categorical_column_pairs:
    new_column_name = '_'.join(sorted((column1, column2)))
    combined_categorical_columns.append(new_column_name)

    new_column_values[new_column_name] = combined_df[column1] * categorical_column_sizes[column2] + combined_df[column2]

# create a dataframe out of the new columns and merges it with the combined train dataframe
if new_column_values:
    new_df = pd.DataFrame(new_column_values)
    combined_df = pd.concat([combined_df, new_df], axis=1)

print(f"Created {len(combined_categorical_columns)} combined categorical columns.")

combined_df.sample(5)


all_categorical_columns: list[str] = categorical_columns + categorized_numerical_columns + combined_categorical_columns
count_encoded_columns: list[str] = []

print(f"Count-encoding {len(all_categorical_columns)} columns")
for column in tqdm(all_categorical_columns):
    # create count map of values in the column
    count_map: pd.Series = combined_df[column].value_counts().astype(np.int32)

    # create a name for the new column
    new_column_name = f"count-encoded_{column}"
    count_map.name = new_column_name
    count_encoded_columns.append(new_column_name)

    # merge the dataframe with the count series
    combined_df = combined_df.merge(count_map, on=column, how="left")

combined_df.sample(5)


train_df = combined_df[:len(train_df)]
test_df = combined_df[len(train_df) : len(train_df) + len(test_df)].drop(columns="y")
original_df = combined_df[-len(original_df):]

print(f"Train dataframe shape: {train_df.shape}")
print(f"Test dataframe shape: {test_df.shape}")
print(f"Original dataframe shape: {original_df.shape}")


def fit_xgb(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    original_df: pd.DataFrame
) -> tuple[xgb.Booster, list]:

    features = test_df.columns
    categorical_columns_to_encode: list[str] = categorical_columns + categorized_numerical_columns
    
    train: pd.DataFrame     = train_df.copy()
    original: pd.DataFrame  = original_df.copy()
    predictions: np.ndarray = np.zeros(len(test_df))
    
    num_folds = 7
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=SEED)

    params = {
        "objective": "binary:logistic",  
        "eval_metric": "auc",           
        "learning_rate": 0.059,
        "max_depth": 0,
        "subsample": 0.9,
        "colsample_bytree": 0.7,
        "seed": SEED,
        "device": "cuda",
        "grow_policy": "lossguide", 
        "max_leaves": 36,          
        "alpha": 3.5,
    }

    for fold_i, (train_index, val_index) in enumerate(kf.split(train), 1):
        print(f"Fold {fold_i:3d}/{num_folds}")
        
        x_train, x_val = train.loc[train_index, features], train.loc[val_index, features]
        y_train, y_val = train.loc[train_index, "y"],      train.loc[val_index, "y"]
        x_test = test_df.copy()

        x_train = pd.concat([x_train, original[features]], axis=0, ignore_index=True)
        y_train = pd.concat([y_train, original["y"]], axis=0, ignore_index=True)
        
        encoder = TargetEncoder(cv=5, smooth=3.5, random_state=SEED)
        x_train[categorical_columns_to_encode] = encoder.fit_transform(x_train[categorical_columns_to_encode], y_train).astype(np.float32)
        x_val[categorical_columns_to_encode] = encoder.transform(x_val[categorical_columns_to_encode]).astype(np.float32)
        x_test[categorical_columns_to_encode] = encoder.transform(x_test[categorical_columns_to_encode]).astype(np.float32)

        dtrain = xgb.QuantileDMatrix(x_train, label=y_train, enable_categorical=True, max_bin=256)
        dval   = xgb.DMatrix(x_val, label=y_val, enable_categorical=True)
        dtest  = xgb.DMatrix(x_test, enable_categorical=True)

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=5950,
            evals=[(dtrain, "train"), (dval, "validation")],
            early_stopping_rounds=200,
            verbose_eval=200
        )

        predictions += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / num_folds
        print()
    
    return model, predictions


model, predictions = fit_xgb(train_df, test_df, original_df)


submission_df = pd.read_csv(submission_data_path, index_col="id")

submission_df.sample(5)


submission_df["y"] = predictions
submission_df.sample(5)


submission_df.to_csv("submission.csv")

