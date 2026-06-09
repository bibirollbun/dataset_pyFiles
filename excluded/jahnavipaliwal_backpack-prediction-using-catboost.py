import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
import optuna


df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df.head(5)


df_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
df_extra.shape


df = pd.concat([df, df_extra])


print(df.columns)  # See all column names



df.info()


df["Brand"].unique(), df["Material"].unique(), df["Size"].unique(), df["Laptop Compartment"].unique(), 



df["Brand"].mode()[0], df["Material"].mode()[0], df["Size"].mode()[0], df["Laptop Compartment"].mode()[0]



df["Waterproof"].unique(), df["Style"].unique(), df["Color"].unique(),


(100*df.isnull().sum())/df.shape[0]


df["Brand"] = df["Brand"].fillna(df["Brand"].mode()[0]).astype("category")
df["Material"] = df["Material"].fillna(df["Material"].mode()[0]).astype("category")
df["Size"] = df["Size"].fillna(df["Size"].mode()[0]).astype("category")
df["Laptop Compartment"] = df["Laptop Compartment"].fillna("No").astype("category")
df["Waterproof"] = df["Waterproof"].fillna("No").astype("category")

df["Style"] = df["Style"].fillna(df["Style"].mode()[0]).astype("category")
df["Color"] = df["Color"].fillna(df["Color"].mode()[0]).astype("category")
df["Weight Capacity (kg)"] = df["Weight Capacity (kg)"].fillna(df["Weight Capacity (kg)"].mean())


# df_dummy = pd.get_dummies(df)
# df_dummy
# target_column = "Price"
# X = df_dummy.drop(columns=[target_column])  # Features
# y = df_dummy[target_column]  # Target
# X.head(3)


# rf = RandomForestRegressor(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)


# y_pred = rf.predict()
# r2 = r2_score(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)


# df_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
# df_extra.info()


df_test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")



df_test["Brand"] = df_test["Brand"].fillna(df["Brand"].mode()[0]).astype("category")
df_test["Material"] = df_test["Material"].fillna(df["Material"].mode()[0]).astype("category")
df_test["Size"] = df_test["Size"].fillna(df["Size"].mode()[0]).astype("category")
df_test["Laptop Compartment"] = df_test["Laptop Compartment"].fillna("No").astype("category")
df_test["Waterproof"] = df_test["Waterproof"].fillna("No").astype("category")
df_test["Style"] = df_test["Style"].fillna(df["Style"].mode()[0]).astype("category")
df_test["Color"] = df_test["Color"].fillna(df["Color"].mode()[0]).astype("category")
df_test["Weight Capacity (kg)"] = df_test["Weight Capacity (kg)"].fillna(df["Weight Capacity (kg)"].mean())


# df_test = pd.get_dummies(df_test)
# missing_cols = set(X.columns) - set(df_test.columns)  # Find missing columns
# missing_cols


# for col in missing_cols:
#     df_test[col] = 0


# y_pred = rf.predict(df_test)
# df_test["Price"] = y_pred
# result = df_test[["id", "Price"]]
# result.to_csv("submission.csv", index=False)


#result.head(3)


target_column = "Price" 
X = df.drop(columns=[target_column])  # Drop the 'price' column from X (features)
y = df[target_column]



df.isnull().sum()


cat_features = X.select_dtypes(include=["category"]).columns.tolist()
cat_features





# categorical_columns = X.select_dtypes(include=["category"]).columns.tolist()

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
#         ('num', SimpleImputer(strategy='mean'), X.select_dtypes(include=["number"]).columns)
#     ]
# )

# X = preprocessor.fit_transform(X)


# def objective(trial):
#     param = {
#         'objective': 'reg:squarederror',
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#         'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),  # Updated line
#         'max_depth': trial.suggest_int('max_depth', 3, 15),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'gamma': trial.suggest_float('gamma', 0, 0.5),
#         'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1e2, log=True),  # Updated line
#         'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1e2, log=True)  # Updated line
#     }
    
#     model = xgb.XGBRegressor(**param, random_state=42)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     return rmse


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# study = optuna.create_study(direction='minimize') 
# study.optimize(objective, n_trials=50)

# print(f"Best trial: {study.best_trial.value}")
# print(f"Best parameters: {study.best_trial.params}")

# best_params = study.best_trial.params
# best_xgb_model = xgb.XGBRegressor(**best_params, random_state=42)

# best_xgb_model.fit(X_train, y_train)

# y_pred = best_xgb_model.predict(X_test)

# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f"Final RMSE: {rmse}")



import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function for Optuna
def objective(trial):
    # Suggest values for hyperparameters
    param = {
        'iterations': trial.suggest_int('iterations', 300, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 0.1, log=True),
        'depth': trial.suggest_int('depth', 3, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 1e1, log=True),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10),
        'cat_features': cat_features,
        'loss_function': 'RMSE',
        'early_stopping_rounds': trial.suggest_int('iterations', 10, 200),
        'verbose': 0,
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        'random_seed': 1
    }
    
    # Create the CatBoost model with suggested hyperparameters
    model = CatBoostRegressor(**param)
    
    # Fit the model
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, cat_features=cat_features)
    
    # Predict and calculate RMSE
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    return rmse

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='minimize')  # Minimize RMSE
study.optimize(objective, n_trials=30)  # Try 50 different sets of hyperparameters

# Print the best trial and its parameters
print(f"Best trial RMSE: {study.best_trial.value}")
print(f"Best parameters: {study.best_trial.params}")



# categorical_columns = df_test.select_dtypes(include=["category"]).columns.tolist()

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
#         ('num', SimpleImputer(strategy='mean'), df_test.select_dtypes(include=["number"]).columns)
#     ]
# )

# df_test_transformed = preprocessor.fit_transform(df_test)


print(X_train.info())




# Extract the best parameters from the study
best_params = study.best_trial.params

# Train the best model with the best parameters
best_catboost_model = CatBoostRegressor(**best_params)
#best_catboost_model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
best_catboost_model.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features=cat_features)

# Make predictions with the best model
y_pred_best = best_catboost_model.predict(X_test)

# Calculate the RMSE for the best model
rmse_best = mean_squared_error(y_test, y_pred_best, squared=False)
print(f"Final RMSE with best parameters: {rmse_best}")



cat_features


X_test.shape, df_test.shape


y_pred = best_catboost_model.predict(df_test)
df_test["Price"] = y_pred
result = df_test[["id", "Price"]]
result.to_csv("submission.csv", index=False)


result.head(2)

