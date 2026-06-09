import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 



df = pd.read_csv('/kaggle/input/predicting-euphoria-in-the-streets/train.csv')
df_test = pd.read_csv('/kaggle/input/predicting-euphoria-in-the-streets/test.csv')


df.head()


df['Y'] = df['Y'].astype('int')
df



df.shape


df.info()


pd.concat([df.isnull().sum(), df.isnull().sum()/df.shape[0]*100], axis=1) #missing values


neg_inf_count = (df == -np.inf).sum()
pd.concat([neg_inf_count, neg_inf_count/df.shape[0]*100], axis=1) #infinite values



df.duplicated().sum() #duplicates


#no garbage values as no 'object' dtype


#descriptive statistics
df.describe().T





import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

for i in df.columns:
    col = df[i].dropna() 
    
    if col.max() > 1e6:  
        data = np.log1p(col)
        title = f"{i} (log1p transformed)"
    else:
        data = col
        title = i

    # Histogram
    plt.figure(figsize=(8,4))
    sns.histplot(data, bins=100, kde=True)
    plt.title(f"{title} - Histogram")
    plt.show()

    # Boxplot
    plt.figure(figsize=(8,2))
    sns.boxplot(x=data)
    plt.title(f"{title} - Boxplot")
    plt.show()






s = df.corr()
plt.figure(figsize=(25,25))
sns.heatmap(s, annot=True)


# By default, pandas computes Pearson correlation
corr_matrix = df.corr()



print(corr_matrix)




# df.replace([np.inf, -np.inf], np.nan, inplace=True)
# df.fillna(df.median(), inplace=True)

# features = [col for col in df.columns if col != "Y"]


# skewed_features = []
# for col in features:
   
#     q1 = df[col].quantile(0.25)
#     q3 = df[col].quantile(0.75)
#     if df[col].max() > 1e6 or (q1>0 and (q3/q1)>1000):
#         skewed_features.append(col)

# print("Highly skewed features (log-transform applied):", skewed_features)




# subset_features = features[:10]

# for i in range(1,len(subset_features)):
#     for j in range(i+1, len(subset_features)):
#         plt.figure(figsize=(5,2))
#         x_data = df[subset_features[i]]
#         y_data = df[subset_features[j]]
        
#         if subset_features[i] in skewed_features:
#             x_data = np.log1p(x_data)
#         if subset_features[j] in skewed_features:
#             y_data = np.log1p(y_data)
        
#         sns.scatterplot(x=x_data, y=y_data, hue=df["Y"], palette="coolwarm", alpha=0.7)
#         plt.xlabel(subset_features[i] + (" (log1p)" if subset_features[i] in skewed_features else ""))
#         plt.ylabel(subset_features[j] + (" (log1p)" if subset_features[j] in skewed_features else ""))
#         plt.title(f"{subset_features[i]} vs {subset_features[j]} colored by Y")
#         plt.show()






import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Replace inf values and define X, y
df.replace([np.inf, -np.inf], np.nan, inplace=True)

X = df.drop(columns=['id', 'Y'])
y = df['Y']

# Define model
params = {
    'n_estimators': 500,
    'learning_rate': 0.03,
    'max_depth': 6,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'verbose':-1
}

model = lgb.LGBMRegressor(**params)

# Initialize 20-Fold CV
kf = KFold(n_splits=20, shuffle=True, random_state=42)

rmse_scores = []

# Loop through folds
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Fit model
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse'
            )
    
    # Predict and evaluate
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    rmse_scores.append(rmse)
    
    print(f"Fold {fold + 1}: RMSE = {rmse:.4f}")

# Overall performance
print("\nAverage RMSE across 20 folds:", np.mean(rmse_scores))



import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split



import lightgbm as lgb
import optuna
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Clean data
df.replace([np.inf, -np.inf], np.nan, inplace=True)

X = df.drop(columns=['id', 'Y'])
y = df['Y']

# Split train–validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define objective function for Optuna
def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': 42,
    }

    model = lgb.LGBMRegressor(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='rmse'
        
    )

    y_pred = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, y_pred, squared=False)

    return rmse

# Run Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

# Show best result
print("\nBest RMSE:", study.best_value)
print("Best Parameters:", study.best_params)

# Train final model on full data using best params
best_params = study.best_params
best_params.update({
    'n_estimators': 1000,
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity':-1
})


model = lgb.LGBMRegressor(**best_params)

# Initialize 20-Fold CV
kf = KFold(n_splits=20, shuffle=True, random_state=42)

rmse_scores = []

# Loop through folds
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Fit model
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse'
            )
    
    # Predict and evaluate
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    rmse_scores.append(rmse)
    
    print(f"Fold {fold + 1}: RMSE = {rmse:.4f}")

# Overall performance
print("\nAverage RMSE across 20 folds:", np.mean(rmse_scores))


from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
y_pred = model.predict(X_valid)
# cutoff = 0.5
# y_pred = (y_pred > cutoff).astype(int)
# acc = accuracy_score(y_valid, y_pred)
# print("Accuracy:", acc)
mse = mean_squared_error(y_valid, y_pred)
rmse = np.sqrt(mse)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")



import matplotlib.pyplot as plt
import numpy as np

# --- Replace with your data ---
# y_true should be your test labels (containing 0s and 1s)
# y_pred_continuous should be the continuous predictions from your model
y_true = y_valid
y_pred_continuous = y_pred
# -----------------------------


# Create an index for the x-axis
ids = np.arange(len(y_true))

plt.figure(figsize=(12, 7))

# Scatter plot of predicted values, colored by their true class
plt.scatter(ids[y_true == 0], y_pred_continuous[y_true == 0], color='cornflowerblue', alpha=0.7, label='True Class: 0')
plt.scatter(ids[y_true == 1], y_pred_continuous[y_true == 1], color='darkorange', alpha=0.7, label='True Class: 1')

# Add a horizontal line for the cut-off threshold (you can change the y value)
plt.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Cut-off Threshold at 0.5')

# Add labels and title
plt.xlabel("Data Point ID")
plt.ylabel("Predicted Continuous Value")
plt.title("Regression Output vs. True Class to Determine Cut-off")
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.show()


df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test_features = df_test.drop(columns=['id'])
y_pred = model.predict(X_test_features)
cutoff = 0.5
y_pred = (y_pred > cutoff).astype(int)

output = pd.DataFrame({
    'id': df_test['id'],
    'prediction': y_pred
})

# Save to CSV
output.to_csv('test_predictions.csv', index=False)

print("CSV with test predictions saved successfully!")


y_train.value_counts(normalize=True)



lgb.plot_importance(model, max_num_features=20)


import pandas as pd
import numpy as np


feature_importance = {
    "x_1": 912, "x_15": 553, "x_8": 402, "x_6": 316, "x_18": 226,
    "x_17": 192, "x_19": 167, "x_16": 144, "x_9": 120, "x_20": 110,
    "x_2": 107, "x_7": 102, "x_10": 97, "x_12": 85, "x_3": 76,
    "x_4": 75, "x_21": 72, "x_5": 61, "x_14": 54, "x_13": 48
}

high_corr_groups = [
    ['x_5', 'x_6', 'x_18', 'x_19', 'x_20'],  # very high correlation group
    ['x_1', 'x_2', 'x_5', 'x_6'],           # moderate-high correlation
    ['x_7', 'x_8'],                         # strong negative/positive correlation
]


# Keep highest importance feature from each high correlation group
keep_features = []
for group in high_corr_groups:
    # Sort group by importance descending
    group_sorted = sorted(group, key=lambda x: feature_importance.get(x, 0), reverse=True)
    # Keep the top one
    keep_features.append(group_sorted[0])

# Also keep other top important features not in these groups
top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
for f, _ in top_features:
    if f not in sum(high_corr_groups, []):  # flatten groups
        keep_features.append(f)

# Ensure no duplicates
keep_features = list(set(keep_features))


drop_features = [f for f in df.columns if f not in keep_features and f not in ['id', 'Y']]
df_fe = df.drop(columns=drop_features)
df_test_fe = df_test.drop(columns=[f for f in df_test.columns if f not in keep_features and f != 'id'])


interaction_features = keep_features[:5]  # top 5 for interactions
for i in range(len(interaction_features)):
    for j in range(i+1, len(interaction_features)):
        f1, f2 = interaction_features[i], interaction_features[j]
        df_fe[f'{f1}_{f2}_sum'] = df_fe[f1] + df_fe[f2]
        df_fe[f'{f1}_{f2}_diff'] = df_fe[f1] - df_fe[f2]
        df_fe[f'{f1}_{f2}_ratio'] = df_fe[f1] / (df_fe[f2] + 1e-5)
        
        df_test_fe[f'{f1}_{f2}_sum'] = df_test_fe[f1] + df_test_fe[f2]
        df_test_fe[f'{f1}_{f2}_diff'] = df_test_fe[f1] - df_test_fe[f2]
        df_test_fe[f'{f1}_{f2}_ratio'] = df_test_fe[f1] / (df_test_fe[f2] + 1e-5)


for group in high_corr_groups:
    # Keep only features that exist in df_fe
    group_feats = [f for f in group if f in df_fe.columns]
    if len(group_feats) > 1:
        df_fe[f"{'_'.join(group_feats)}_mean"] = df_fe[group_feats].mean(axis=1)
        df_fe[f"{'_'.join(group_feats)}_std"] = df_fe[group_feats].std(axis=1)
        
        df_test_fe[f"{'_'.join(group_feats)}_mean"] = df_test_fe[group_feats].mean(axis=1)
        df_test_fe[f"{'_'.join(group_feats)}_std"] = df_test_fe[group_feats].std(axis=1)


print("Final engineered train shape:", df_fe.shape)
print("Final engineered test shape:", df_test_fe.shape)



import lightgbm as lgb
import optuna
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Clean data
df_fe.replace([np.inf, -np.inf], np.nan, inplace=True)

X = df_fe.drop(columns=['id', 'Y'])
y = df_fe['Y']

# Split train–validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define objective function for Optuna
def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': 42,
    }

    model = lgb.LGBMRegressor(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='rmse'
        
    )

    y_pred = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, y_pred, squared=False)

    return rmse

# Run Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

# Show best result
print("\nBest RMSE:", study.best_value)
print("Best Parameters:", study.best_params)

# Train final model on full data using best params
best_params = study.best_params
best_params.update({
    'n_estimators': 1000,
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity':-1
})


model = lgb.LGBMRegressor(**best_params)

# Initialize 20-Fold CV
kf = KFold(n_splits=20, shuffle=True, random_state=42)

rmse_scores = []

# Loop through folds
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Fit model
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse'
            )
    
    # Predict and evaluate
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    rmse_scores.append(rmse)
    
    print(f"Fold {fold + 1}: RMSE = {rmse:.4f}")

# Overall performance
print("\nAverage RMSE across 20 folds:", np.mean(rmse_scores))


from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
y_pred = model.predict(X_valid)
# cutoff = 0.5
# y_pred = (y_pred > cutoff).astype(int)
# acc = accuracy_score(y_valid, y_pred)
# print("Accuracy:", acc)
mse = mean_squared_error(y_valid, y_pred)
rmse = np.sqrt(mse)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")



import matplotlib.pyplot as plt
import numpy as np

# --- Replace with your data ---
# y_true should be your test labels (containing 0s and 1s)
# y_pred_continuous should be the continuous predictions from your model
y_true = y_test
y_pred_continuous = y_pred
# -----------------------------


# Create an index for the x-axis
ids = np.arange(len(y_true))

plt.figure(figsize=(12, 7))

# Scatter plot of predicted values, colored by their true class
plt.scatter(ids[y_true == 0], y_pred_continuous[y_true == 0], color='cornflowerblue', alpha=0.7, label='True Class: 0')
plt.scatter(ids[y_true == 1], y_pred_continuous[y_true == 1], color='darkorange', alpha=0.7, label='True Class: 1')

# Add a horizontal line for the cut-off threshold (you can change the y value)
plt.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Cut-off Threshold at 0.5')

# Add labels and title
plt.xlabel("Data Point ID")
plt.ylabel("Predicted Continuous Value")
plt.title("Regression Output vs. True Class to Determine Cut-off")
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.show()





df_test_fe.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test_features = df_test_fe.drop(columns=['id'])
y_pred = model.predict(X_test_features)
cutoff = 0.5
y_pred = (y_pred > cutoff).astype(int)


output = pd.DataFrame({
    'id': df_test['id'],
    'prediction': y_pred
})

# Save to CSV
output.to_csv('test_predictions.csv', index=False)

print("CSV with test predictions saved successfully!")


df.iloc[:,1:-1] = df.iloc[:,1:-1].replace(0, np.nan)
df.iloc[:,1:-1] = df.iloc[:,1:-1].replace(-np.inf, np.nan)




df


#removing outlier 

# For training data
for col in df.columns[1:-1]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Replace outliers with NaN
    df.loc[(df[col] < lower) | (df[col] > upper), col] = np.nan






df.isna().sum()/df.shape[0]*100



'''inferences -
x_5,x_16,x_3,x_14 - >50% missing (Should be dropped)
x_9,x_2,x_4,x_15 - 15-25% missing (could use a advance method like KNNImputer)
x_19,x_20,x_18,x_17,x_10,x_12,x_13 - 7% missing (replace with median)
x_11 - eslg drop
'''

from sklearn.impute import KNNImputer
# from sklearn.impute import SimpleImputer


drop_cols = ['x_5', 'x_16', 'x_3', 'x_14','x_11']
df.drop(columns=drop_cols, inplace=True, errors='ignore')
df_test.drop(columns=drop_cols, inplace=True, errors='ignore')


# knn_cols = []

# if knn_cols:
#     imputer = KNNImputer(n_neighbors=5)
#     df[knn_cols] = imputer.fit_transform(df[knn_cols])


median_cols = ['x_1','x_19', 'x_20', 'x_18', 'x_17', 'x_10', 'x_12', 'x_13','x_2', 'x_4', 'x_15','x_9','x_6','x_7','x_8', 'x_21']

median_cols = [col for col in median_cols if col in df.columns]

for col in median_cols:
    median_value = df[col].median()
    std_dev = df[col].std()
    
    
    na_mask = df[col].isna()
    jitter_strength = 0.01
    # Fill NaN with median + jitter
    df.loc[na_mask, col] = median_value + np.random.normal(0, jitter_strength * std_dev, size=na_mask.sum())






for i in df.columns:
    col = df[i].dropna()  # drop NaN for plotting
    
    if col.max() > 1e6:   # skewed columns
        data = np.log1p(col)
        title = f"{i} (log1p transformed)"
    else:
        data = col
        title = i

    # Histogram
    plt.figure(figsize=(8,4))
    sns.histplot(data, bins=100, kde=True)
    plt.title(f"{title} - Histogram")
    plt.show()

    # Boxplot
    plt.figure(figsize=(8,2))
    sns.boxplot(x=data)
    plt.title(f"{title} - Boxplot")
    plt.show()



df.isna().sum()


df.isna().sum()



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



X = df.drop(columns=['id', 'Y'])
X.replace([0], np.nan, inplace=True)
y = df['Y']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# Step 7: Evaluate performance
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", acc)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)



import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

df.replace([np.inf, -np.inf], np.nan, inplace=True)


X = df.drop(columns=['id', 'Y'])
X.replace([0], np.nan, inplace=True)
y = df['Y']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = lgb.LGBMClassifier(
    n_estimators=267,
    learning_rate=0.01,
    max_depth=4,
    random_state=42
)


model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
# # Evaluate
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# print(f"Mean Squared Error: {mse:.4f}")
# print(f"Root Mean Squared Error: {rmse:.4f}")



df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test_features = df_test.drop(columns=['id'])
y_pred = model.predict(X_test_features)


output = pd.DataFrame({
    'id': df_test['id'],
    'prediction': y_pred
})

# Save to CSV
output.to_csv('test_predictions.csv', index=False)

print("CSV with test predictions saved successfully!")

