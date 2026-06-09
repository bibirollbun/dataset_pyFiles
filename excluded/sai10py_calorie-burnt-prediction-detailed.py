import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


# Train data overview
train.info()


train.head()


# sns.countplot(x="Sex", data=train) 


# print("Min Age: ", min(train["Age"]))
# print("Max Age: ", max(train["Age"]))
# print("Avg Age: ", sum(train["Age"])/len(train["Age"]))
# sns.histplot(train["Age"])


# # Height stats
# print("Min height: ", min(train["Height"]))
# print("Max height: ", max(train["Height"]))
# print("Avg height: ", sum(train["Height"])/len(train["Height"]))

# # Visualizing distribution
# sns.distplot(train["Height"], kde=True)


# # Weight stats
# print("Min weight: ", min(train["Weight"]))
# print("Max weight: ", max(train["Weight"]))
# print("Avg weight: ", sum(train["Weight"])/len(train["Weight"]))

# # Visualizing distribution
# sns.distplot(train["Weight"], kde=True)


# sns.scatterplot(x="Duration", y="Calories", data=train)
# sns.lineplot(x="Duration", y = "Calories", color = "red", data=train)
# plt.grid(True)
# plt.show()


# sns.scatterplot(x="Heart_Rate", y="Calories", data=train)
# sns.lineplot(x="Heart_Rate", y="Calories", data=train, color="Red")


# sns.scatterplot(x="Body_Temp", y="Calories", data=train)
# sns.lineplot(x="Body_Temp", y="Calories", data=train, color="Red")


# sns.scatterplot(x="Height", y="Calories", data=train)
# sns.lineplot(x="Height", y="Calories", data=train, color="Red")


# sns.scatterplot(x="Weight", y="Calories", data=train)
# sns.lineplot(x="Weight", y="Calories", data=train, color="Red")


# Creating BMI column
# train["BMI"] = (train["Weight"] * 100*100 )/(train["Height"] * train["Height"])


from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
encoder = LabelEncoder()
scaler = StandardScaler()

train["Sex"] = encoder.fit_transform(train["Sex"])
test["Sex"] = encoder.fit_transform(test["Sex"])


train.info()


plt.figure(figsize = (15, 15))
sns.heatmap(train.corr())


train["BMI"] = train["Weight"] / (train["Height"]/100)**2
train['MET'] = (train['Heart_Rate'] * 0.6309 + train['Body_Temp'] * 1.5 - 55) / 100
train['Weight_Height_Ratio'] = train['Weight'] / train['Height']
train["HR_Age_Ratio"] = train["Heart_Rate"] / train["Age"]
train["Body_Temp_Dev"] = train["Body_Temp"] - 37


# train['BMI_sq'] = train['BMI'] ** 2
# train['Weight_sq'] = train['Weight'] ** 2
# train['Height_sq'] = train['Height'] ** 2
# train['Duration_sq'] = train['Duration'] ** 2
# train['Heart_Rate_sq'] = train['Heart_Rate'] ** 2
# train['Body_Temp_sq'] = train['Body_Temp'] ** 2

# train['BMI_Duration'] = train['BMI'] * train['Duration']
# train['HR_Duration'] = train['Heart_Rate'] * train['Duration']
# train['BMI_HR'] = train['BMI'] * train['Heart_Rate']
# train['Weight_HR'] = train['Weight'] * train['Heart_Rate']
# train['Age_HR'] = train['Age'] * train['Heart_Rate']
# train['Age_Duration'] = train['Age'] * train['Duration']

# train['log_Weight'] = np.log(train['Weight'] + 1)
# train['log_Height'] = np.log(train['Height'] + 1)
# train['log_Duration'] = np.log(train['Duration'] + 1)
# train['log_Heart_Rate'] = np.log(train['Heart_Rate'] + 1)
# train['log_Body_Temp'] = np.log(train['Body_Temp'] + 1)
# train['log_Age'] = np.log(train['Age'] + 1)

# train['inv_Weight'] = 1 / (train['Weight'] + 1)
# train['inv_Height'] = 1 / (train['Height'] + 1)
# train['inv_Duration'] = 1 / (train['Duration'] + 1)
# train['inv_Heart_Rate'] = 1 / (train['Heart_Rate'] + 1)
# train['inv_Age'] = 1 / (train['Age'] + 1)


train.info()


from sklearn.model_selection import train_test_split

X = train.drop(columns = ["Calories", "id"])
y = train["Calories"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)


from sklearn.metrics import mean_squared_log_error


def rmsle(y_pred, y_true):
    y_true = np.maximum(0, y_true) 
    y_pred = np.maximum(0, y_pred) 
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


# Previous scores:
# RMSLE (Train Set): 
# CatBoost Model:  0.056022615008618375

# RMSLE (Validation Set): 
# CatBoost Model:  0.05925686847548469


from catboost import CatBoostRegressor, Pool

# Log-transform target to handle skewness
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

# Use Pool (CatBoost optimized structure)
train_pool = Pool(X_train, y_train_log)
val_pool = Pool(X_val, y_val_log)

# Define improved CatBoostRegressor with tuning
cat_model = CatBoostRegressor(
    iterations=4000,
    learning_rate=0.1,
    depth=8,
    # loss_function='RMSE',  # CatBoost does not support RMSLE directly
    # eval_metric='RMSE',
    # l2_leaf_reg=3.0,
    # subsample=0.8,
    # colsample_bylevel=0.8,
    random_seed=42,
    early_stopping_rounds=50,
    verbose=10000 
)

# Train with validation for early stopping
cat_model.fit(train_pool, eval_set=val_pool)

# Predict and reverse log
pred_train = np.expm1(cat_model.predict(X_train))
pred_val = np.expm1(cat_model.predict(X_val))

# Clamp predictions and evaluate
print("RMSLE (Train Set): ")
print("CatBoost Model: ", rmsle(np.clip(pred_train, 1, 314), y_train))

print("\n","-"*25,"\n")

print("RMSLE (Validation Set): ")
print("CatBoost Model: ", rmsle(np.clip(pred_val, 1, 314), y_val))


#Previous scores:
# XGBoost Model (Train): 0.055799815208138925
# XGBoost Model (Val): 0.059934856409019355

# XGBoost Model (Train): 0.053114360051216676
# XGBoost Model (Val): 0.0598140174559557


from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import numpy as np

# Log transform target
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

# Split training set for early stopping
X_tr, X_val_es, y_tr_log, y_val_es_log = train_test_split(X_train, y_train_log, test_size=0.1, random_state=42)

# Define and fit XGB model
xgb_model = XGBRegressor(
    n_estimators=5000,
    learning_rate=0.02,
    max_depth=10,
    subsample=0.9,
    colsample_bytree=0.8,
    gamma = 0.01,
    max_delta_step = 2,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="rmsle",  
    random_state=42
)

xgb_model.fit(
    X_tr, y_tr_log,
    eval_set=[(X_val_es, y_val_es_log)],
    early_stopping_rounds=20,
    verbose=False
)

# Predict and reverse log transformation
pred_train = np.expm1(xgb_model.predict(X_train))
pred_val = np.expm1(xgb_model.predict(X_val))

# Clamp predictions before computing RMSLE
print("XGBoost Model (Train):", rmsle(np.clip(pred_train, 1, 314), y_train))
print("XGBoost Model (Val):", rmsle(np.clip(pred_val, 1, 314), y_val))


# Previous Scores:
# LGBM Model(Train): 0.057879657282222566
# LGBM Model(Val): 0.06022574736560706

# LGBM Model(Train): 0.05713943784272827
# LGBM Model(Val): 0.06006561110996626

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
import matplotlib.pyplot as plt
import time

# Log transformation of target variable
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

# Feature engineering function - more targeted given your dataset size
def engineer_features(X):
    print(f"Starting feature engineering on shape: {X.shape}")
    start_time = time.time()
    
    X_new = X.copy()
    
    # Get numeric columns
    numeric_cols = X_new.select_dtypes(include=['float64', 'int64']).columns
    
    # With only 9 columns, we can create selective interactions without exploding dimensionality
    # Add interactions between most important pairs
    # Start with just a few important interactions to prevent explosion of features
    for i, col1 in enumerate(numeric_cols[:3]):  # Only use top 3 columns for interactions
        for col2 in numeric_cols[i+1:4]:  # Limit interaction pairs
            X_new[f'{col1}_x_{col2}'] = X_new[col1] * X_new[col2]
    
    # Add squared terms for numeric features
    for col in numeric_cols:
        X_new[f'{col}_squared'] = X_new[col] ** 2
    
    # Log transform of a few key features (avoid zeros or negative values)
    for col in numeric_cols[:3]:  # Apply only to most important features
        if (X_new[col] > 0).all():
            X_new[f'{col}_log'] = np.log1p(X_new[col])
    
    print(f"Feature engineering completed in {time.time() - start_time:.2f} seconds")
    print(f"New shape: {X_new.shape}")
    return X_new

# Apply feature engineering - but be careful with memory usage
print("Engineering features for training set...")
X_train_eng = engineer_features(X_train)
print("Engineering features for validation set...")
X_val_eng = engineer_features(X_val)

# Given dataset size, let's use a more focused hyperparameter approach rather than extensive search
print("Starting model training...")

# For a dataset of this size, we'll define a hyperparameter grid based on experience
# rather than doing extensive Optuna search which could be time-consuming
params = {
    'boosting_type': 'gbdt',
    'n_estimators': 3000,  # Reduced from 5000 to improve training speed
    'learning_rate': 0.01,
    'max_depth': 12,       # Slightly reduced from 15
    'num_leaves': 60,      # Control model complexity
    'subsample': 0.8,      # Use 80% of data for each tree
    'colsample_bytree': 0.8,  # Use 80% of features for each tree
    'min_child_samples': 20,  # Increased to reduce overfitting on large dataset
    'reg_alpha': 0.05,
    'reg_lambda': 0.5,
    'min_split_gain': 0.01,
    'random_state': 42,
    # Additional params that help with large datasets
    'n_jobs': -1,          # Use all available cores
    'device': 'cpu',       # 'gpu' if GPU available
    'verbose': -1          # Reduce output verbosity
}

# Create model with the parameters
lgbm_model = LGBMRegressor(**params)

# Create a validation set for early stopping
X_tr, X_val_es, y_tr_log, y_val_es_log = train_test_split(X_train_eng, y_train_log, test_size=0.1, random_state=42)

print(f"Training LGBM model on {X_tr.shape[0]} samples with {X_tr.shape[1]} features")
start_time = time.time()

# Train the model with early stopping
lgbm_model.fit(
    X_tr, y_tr_log,
    eval_set=[(X_val_es, y_val_es_log)],
    callbacks=[
        early_stopping(stopping_rounds=30),
        log_evaluation(100)  # Log every 100 iterations to reduce output
    ]
)

print(f"Model training completed in {time.time() - start_time:.2f} seconds")

# Predict and evaluate
pred_train = np.expm1(lgbm_model.predict(X_train_eng))
pred_val = np.expm1(lgbm_model.predict(X_val_eng))

print("LGBM Model(Train):", rmsle(np.clip(pred_train, 1, 314), y_train))
print("LGBM Model(Val):", rmsle(np.clip(pred_val, 1, 314), y_val))

# Feature importance analysis
feature_importance = pd.DataFrame({
    'Feature': X_train_eng.columns,
    'Importance': lgbm_model.feature_importances_
})

# Sort by importance
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print("\nTop Features by Importance:")
print(feature_importance.head(15))

# Plot feature importance for top features
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'][:15], feature_importance['Importance'][:15])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Let's evaluate with cross-validation for a more robust estimate
# Using just 3 folds to manage computational load
print("\nPerforming cross-validation...")
kf = KFold(n_splits=3, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_eng)):
    print(f"Training fold {fold+1}/3...")
    start_time = time.time()
    
    # Split data for this fold
    X_fold_train, X_fold_val = X_train_eng.iloc[train_idx], X_train_eng.iloc[val_idx]
    y_fold_train, y_fold_val = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
    
    # Create and train model for this fold
    fold_model = LGBMRegressor(**params)
    fold_model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        callbacks=[early_stopping(stopping_rounds=30)],
        verbose=100
    )
    
    # Predict and evaluate
    fold_preds = np.expm1(fold_model.predict(X_fold_val))
    fold_true = np.expm1(y_fold_val)
    fold_score = rmsle(np.clip(fold_preds, 1, 314), fold_true)
    cv_scores.append(fold_score)
    
    print(f"Fold {fold+1} RMSLE: {fold_score:.6f} (completed in {time.time() - start_time:.2f} seconds)")

print(f"\nCross-validation RMSLE: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")

# Let's try one more model with top features only
top_features = feature_importance['Feature'][:15].values  # Use top 15 features
X_train_top = X_train_eng[top_features]
X_val_top = X_val_eng[top_features]

print("\nTraining final model with top 15 features...")
final_model = LGBMRegressor(**params)

# Split for early stopping
X_tr_top, X_val_es_top, y_tr_log, y_val_es_log = train_test_split(X_train_top, y_train_log, test_size=0.1, random_state=42)

# Train final model
final_model.fit(
    X_tr_top, y_tr_log,
    eval_set=[(X_val_es_top, y_val_es_log)],
    callbacks=[
        early_stopping(stopping_rounds=30),
        log_evaluation(100)
    ]
)

# Predict and evaluate with top features
pred_train_final = np.expm1(final_model.predict(X_train_top))
pred_val_final = np.expm1(final_model.predict(X_val_top))

print("\nFinal Model with Top Features (Train):", rmsle(np.clip(pred_train_final, 1, 314), y_train))
print("Final Model with Top Features (Val):", rmsle(np.clip(pred_val_final, 1, 314), y_val))


# Previous scores
# RFR Model(Train):  0.02541545655860886
# RFR Model(Val):  0.06339603326021119

# RFR Model (Train): 0.04857129253961266
# RFR Model (Val): 0.06111281361648884

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

# Log-transform the target
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

# Define improved RFR model
rfr_model = HistGradientBoostingRegressor(
    max_iter=1000,                # equivalent to n_estimators
    learning_rate=0.05,           # slower learning for better generalization
    max_depth=10,                 # controls depth of trees
    max_leaf_nodes=31,            # default is 31, can reduce overfitting
    min_samples_leaf=20,          # controls complexity of trees
    l2_regularization=0.1,        # helps with regularization
    early_stopping=True,          # enables early stopping
    scoring='neg_root_mean_squared_error',
    validation_fraction=0.1,      # fraction of data for early stopping
    n_iter_no_change=20,          # rounds with no improvement to stop
    random_state=42,
    verbose=0
)

# Train on log-transformed target
rfr_model.fit(X_train, y_train_log)

# Predict and reverse log
pred_train = np.expm1(rfr_model.predict(X_train))
pred_val = np.expm1(rfr_model.predict(X_val))

# Clamp predictions and evaluate
print("RFR Model (Train):", rmsle(np.clip(pred_train, 1, 314), y_train))
print("RFR Model (Val):", rmsle(np.clip(pred_val, 1, 314), y_val))


pl, px, pc, pr = np.expm1(lgbm_model.predict(X_train)),np.expm1(xgb_model.predict(X_train)), np.expm1(cat_model.predict(X_train)), np.expm1(rfr_model.predict(X_train))

pl1, px1, pc1, pr1 = np.expm1(lgbm_model.predict(X_val)), np.expm1(xgb_model.predict(X_val)), np.expm1(cat_model.predict(X_val)), np.expm1(rfr_model.predict(X_val))


mini, best_l, best_c, best_x, best_r = 1, 0, 0, 0, 0

for i in range(0, 75):
    for j in range(0, 25):
        c, l, x, r = 75 - i, i, 25 - j, j
        rmsle_v = rmsle(np.clip(pl1*l*0.01+px1*x*0.01+pc1*c*0.01+pr1*r*0.01, 1, 314, None), y_val)
        if rmsle_v <= mini:
            mini = rmsle_v
            best_l = l
            best_c = c
            best_x = x
            best_r = r

print("Rmsle Val: ", mini)
print("best_l: ", best_l)
print("best_c: ", best_c)
print("best_x: ", best_x)
print("best_r: ", best_r)

# print("\n")
# print("Train RMSLE: ")
# print(rmsle(np.clip(pl*0.0+px*0.4+pc*0.5+pr*0.1, 1, 314, None), y_train))
# print("Validation RMSLE: ")
# print(rmsle(np.clip(pl1*0.0+px1*0.4+pc1*0.5+pr1*0.1, 1, 314, None), y_val))

# print("\n","*"*50, "\n")


# # Drop 'id' column if it exists
# test = test.drop(columns=["id"], errors='ignore')

# # Make predictions
# predictions = np.expm1(cat_model.predict(test))*0.61+np.expm1(rfr_model.predict(test))*0.25+np.expm1(xgb_model.predict(test))*0.09+np.expm1(lgbm_model.predict(test))*0.05

# # Using abs values predictions
# predictions = np.clip(predictions, 1, 314, None)

# # Create output DataFrame
# output = pd.DataFrame({
#     "id": range(750000, 750000 + len(predictions)),
#     "Calories": predictions
# })

# # Save to CSV without index
# output.to_csv("submission.csv", index=False)

# # Print first few rows without row numbers
# print(output.head().to_string(index=False))

