!pip install ngboost optuna --quiet


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


train_raw = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_raw = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


train_raw.describe(include = 'all')


train_raw.info()


train_raw.isna().sum()


train_raw.duplicated().sum()


test_raw.describe(include='all')


test_raw.info()


test_raw.isna().sum()


test_raw.duplicated().sum()


train_df = train_raw.copy()
test_df = test_raw.copy()


from sklearn.preprocessing import OneHotEncoder

# Select categorical/bool columns
columns = train_df.select_dtypes(include=['object']).columns

# Initialize OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # ignore unknown categories in test

# Fit on the combined data to avoid missing categories in test
combined = pd.concat([train_df[columns], test_df[columns]], axis=0)
ohe.fit(combined)

# Transform train and test
train_encoded = pd.DataFrame(
    ohe.transform(train_df[columns]),
    columns=ohe.get_feature_names_out(columns),
    index=train_df.index
)

test_encoded = pd.DataFrame(
    ohe.transform(test_df[columns]),
    columns=ohe.get_feature_names_out(columns),
    index=test_df.index
)

# Drop original columns and add one-hot columns
train_df = train_df.drop(columns=columns).join(train_encoded)
test_df = test_df.drop(columns=columns).join(test_encoded)


train_df.head()


import matplotlib.pyplot as plt
import seaborn as sns
import math

# Set aesthetic style
sns.set_style('whitegrid')

# --- 1 Visualize Target Variable Distribution ---
plt.figure(figsize=(12, 6))
sns.histplot(train_df['accident_risk'], kde=True, bins=50, color='skyblue')
plt.title('Distribution of accident_risk', fontsize=15)
plt.xlabel('accident_risk')
plt.ylabel('Frequency')
plt.show()

# --- 2 Visualize Feature Distributions ---
# Drop 'id' and target
features = train_df.drop(columns=['id', 'accident_risk']).columns
n_features = len(features)

# Determine subplot grid size
n_cols = 4
n_rows = math.ceil(n_features / n_cols)

plt.figure(figsize=(4*n_cols, 4*n_rows))
for i, feature in enumerate(features):
    plt.subplot(n_rows, n_cols, i+1)
    sns.histplot(train_df[feature], kde=True, bins=30, color='lightgreen')
    plt.title(f'Distribution of {feature}')
    plt.xlabel('')
    plt.ylabel('')
plt.tight_layout()
plt.show()



from sklearn.metrics import confusion_matrix

train_corr_matrix = train_df.corr()
plt.figure(figsize = (12, 8))
sns.heatmap(train_corr_matrix, fmt = ".2f", cmap = "coolwarm")
plt.title('Correlation Matrix - Train Dataset')
plt.show()



train_df = train_df.drop(columns = 'id')
test_df = test_df.drop(columns = 'id')


X = train_df.drop(columns = 'accident_risk')
y = train_df['accident_risk']


from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.13, random_state = 23)


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Fit XGBoost
xgb = XGBRegressor()
xgb.fit(x_train, y_train)

# Feature importance for loc trees
feature_importance_loc = xgb.feature_importances_[0]

# Feature importance for scale trees
feature_importance_scale = xgb.feature_importances_[1]

# dataframes for feature importance
df_loc = pd.DataFrame({'feature':x_train.columns,
                       'importance':feature_importance_loc})\
    .sort_values('importance',ascending=False)

df_scale = pd.DataFrame({'feature':x_train.columns,
                       'importance':feature_importance_scale})\
    .sort_values('importance',ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,6))
fig.suptitle("Feature importance plot for distribution parameters", fontsize=17)
sns.barplot(x='importance',y='feature',ax=ax1,data=df_loc, color="skyblue").set_title('loc param')
sns.barplot(x='importance',y='feature',ax=ax2,data=df_scale, color="skyblue").set_title('scale param')
plt.show()


xgb_model = XGBRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    random_state=23,
    eval_metric="rmse",
    n_jobs=-1,
    tree_method="gpu_hist"
)

xgb_model.fit(
    x_train.values, y_train.values,
    eval_set=[(x_val.values, y_val)],
    # early_stopping_rounds=50,
    verbose=100
)


train_raw.columns


train_raw.head(10)


columns = train_raw.select_dtypes(include = ['object', 'bool']).columns

for col in columns:
    print(f'{col} values: {np.unique(train_raw[col].values)}')


train_raw['low_visibility'] = train_raw['lighting'].isin(['dim', 'night']) | train_raw['weather'].isin(['foggy', 'rainy'])
train_raw['is_peak_hours'] = train_raw['time_of_day'].isin(['morning', 'evening'])
train_raw['is_narrow_road'] = train_raw['num_lanes'] <= 2


def categorize_curvature(curv):
    if curv < 0.2:
        return "straight"
    elif curv < 0.6:
        return "moderate"
    else:
        return "sharp turn"

# Apply it to the dataframe
train_raw["curvature_category"] = train_raw["curvature"].apply(categorize_curvature)



train_raw.head(10)


train_df = train_raw.copy()


from sklearn.preprocessing import OneHotEncoder

categorical_cols = train_df.select_dtypes(include=['object']).columns
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # ignores unseen categories

ohe.fit(train_df[categorical_cols])

train_encoded = pd.DataFrame(
    ohe.transform(train_df[categorical_cols]),
    columns=ohe.get_feature_names_out(categorical_cols),
    index=train_df.index
)

# Drop original categorical columns and add encoded columns
train_df = train_df.drop(columns=categorical_cols).join(train_encoded)



# Set aesthetic style
sns.set_style('whitegrid')

# --- 1 Visualize Target Variable Distribution ---
plt.figure(figsize=(12, 6))
sns.histplot(train_df['accident_risk'], kde=True, bins=50, color='skyblue')
plt.title('Distribution of accident_risk', fontsize=15)
plt.xlabel('accident_risk')
plt.ylabel('Frequency')
plt.show()

# --- 2 Visualize Feature Distributions ---
# Drop 'id' and target
features = train_df.drop(columns=['id', 'accident_risk']).columns
n_features = len(features)

# Determine subplot grid size
n_cols = 4
n_rows = math.ceil(n_features / n_cols)

plt.figure(figsize=(4*n_cols, 4*n_rows))
for i, feature in enumerate(features):
    plt.subplot(n_rows, n_cols, i+1)
    sns.histplot(train_df[feature], kde=True, bins=30, color='lightgreen')
    plt.title(f'Distribution of {feature}')
    plt.xlabel('')
    plt.ylabel('')
plt.tight_layout()
plt.show()



from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

train_corr_matrix = train_df.corr()
plt.figure(figsize = (12, 8))
sns.heatmap(train_corr_matrix, fmt = ".2f", cmap = "coolwarm")
plt.title('Correlation Matrix - Train Dataset')
plt.show()



train_df = train_df.drop(columns = 'id')


X = train_df.drop(columns = 'accident_risk')
y = train_df['accident_risk']


from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.13, random_state = 23)


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Fit XGBoost
xgb = XGBRegressor()
xgb.fit(x_train, y_train)

# Feature importance for loc trees
feature_importance_loc = xgb.feature_importances_[0]

# Feature importance for scale trees
feature_importance_scale = xgb.feature_importances_[1]

# dataframes for feature importance
df_loc = pd.DataFrame({'feature':x_train.columns,
                       'importance':feature_importance_loc})\
    .sort_values('importance',ascending=False)

df_scale = pd.DataFrame({'feature':x_train.columns,
                       'importance':feature_importance_scale})\
    .sort_values('importance',ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,6))
fig.suptitle("Feature importance plot for distribution parameters", fontsize=17)
sns.barplot(x='importance',y='feature',ax=ax1,data=df_loc, color="skyblue").set_title('loc param')
sns.barplot(x='importance',y='feature',ax=ax2,data=df_scale, color="skyblue").set_title('scale param')
plt.show()


xgb_model = XGBRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    random_state=23,
    eval_metric="rmse",
    n_jobs=-1,
    tree_method="gpu_hist"
)

xgb_model.fit(
    x_train.values, y_train.values,
    eval_set=[(x_val.values, y_val)],
    # early_stopping_rounds=50,
    verbose=100
)


!pip install mealpy --quiet


X = train_df.drop(columns=['accident_risk'])
y = train_df['accident_risk']
X_values = X.values
y_values = y.values
n_features = X.shape[1]


X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.13, random_state=23
)


import optuna

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_uniform("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10.0),
        "random_state": 23,
        "tree_method": "gpu_hist",
        "eval_metric":"rmse",
        "n_jobs": -1
    }

    model = XGBRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=500)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
print("\nBest trial:")
print(study.best_trial.params)
print(f"Best RMSE: {study.best_value:.6f}")



print("\n\n Best trial:")
best_params = study.best_trial.params
print(best_params)
print(f"Best RMSE: {study.best_value}")


from sklearn.model_selection import cross_val_score
from mealpy import GWO, FloatVar


def fitness_function(solution):
    # Convert continuous [0,1] → binary mask
    mask = solution > 0.5
    if np.sum(mask) == 0:
        return 1.0  # avoid empty feature subset
    
    # Subset features
    X_sub = X_values[:, mask]

    # Split data
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_sub, y_values, test_size=0.2, random_state=23
    )

    model = XGBRegressor(
        **best_params,
        verbose=100,
        random_state=23,
        n_jobs=-1,
        tree_method="gpu_hist"
    )

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)

    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse



problem = {
    "bounds": FloatVar(lb=[0.0]*n_features, ub=[1.0]*n_features, name="features"),
    "minmax": "min",
    "obj_func": fitness_function,
}

model = GWO.CG_GWO(epoch=30, pop_size=10, n_workers=-1)  # increase epochs for deeper search

print("Starting Grey Wolf Optimization for Feature Selection...")
g_best = model.solve(problem)
print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")


best_position = g_best.solution
best_fitness = g_best.target.fitness

# Convert best_position to selected feature names
selected_features = X.columns[best_position > 0.5]

# Summary
print(f"Best RMSE (validation): {best_fitness:.6f}")
print(f"Selected {len(selected_features)} features out of {X.shape[1]}:")
print(list(selected_features))


X_selected = X[selected_features]


# Train the final model with Optuna's best hyperparameters and GWO-selected features
model = XGBRegressor(
    **best_params,
    verbose=100
)

# Fit on the selected features
model.fit(X_selected.values, y_values)



import joblib
joblib.dump(model, "xgboost_gwo_optuna_final.pkl")


import joblib

save_dict = {
    "model": model,
    "selected_features": list(selected_features)
}

joblib.dump(save_dict, "xgboost_gwo_model.pkl")
print("Saved model and feature list together as 'xgboost_gwo_model.pkl'")



test_raw = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


columns = test_raw.select_dtypes(include = ['object', 'bool']).columns

for col in columns:
    print(f'{col} values: {np.unique(test_raw[col].values)}')


test_raw['low_visibility'] = test_raw['lighting'].isin(['dim', 'night']) | test_raw['weather'].isin(['foggy', 'rainy'])
test_raw['is_peak_hours'] = test_raw['time_of_day'].isin(['morning', 'evening'])
test_raw['is_narrow_road'] = test_raw['num_lanes'] <= 2


def categorize_curvature(curv):
    if curv < 0.2:
        return "straight"
    elif curv < 0.6:
        return "moderate"
    else:
        return "sharp turn"

# Apply it to the dataframe
test_raw["curvature_category"] = test_raw["curvature"].apply(categorize_curvature)



test_df = test_raw.copy()


from sklearn.preprocessing import OneHotEncoder

categorical_cols = test_df.select_dtypes(include=['object']).columns
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # ignores unseen categories

ohe.fit(test_df[categorical_cols])

test_encoded = pd.DataFrame(
    ohe.transform(test_df[categorical_cols]),
    columns=ohe.get_feature_names_out(categorical_cols),
    index=test_df.index
)

# Drop original categorical columns and add encoded columns
test_df = test_df.drop(columns=categorical_cols).join(test_encoded)



data = joblib.load("/kaggle/working/xgboost_gwo_model.pkl")
model = joblib.load("/kaggle/working/xgboost_gwo_optuna_final.pkl")
selected_features = data["selected_features"]



X = test_df
X_selected = X[selected_features]


preds = model.predict(X_selected.values)


ids = test_raw["id"].values 

pred_df = pd.DataFrame()
pred_df['id'] = ids
pred_df['accident_risk'] = preds
pred_df['accident_risk'] = pred_df['accident_risk'].abs()


pred_df.to_csv('predictions.csv', index = False)
len(pred_df)

