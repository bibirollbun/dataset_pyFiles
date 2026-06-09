import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import matplotlib as plt


import os
for dirname, _, filenames in os.walk('/kaggle/input/playground-series-s5e4'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')


print(df_train.info())


df_train.head()


df_train.isna().sum()


df_train.describe()


df1 = df_train.copy()


df1['Episode_Length_minutes'] = df1['Episode_Length_minutes'].fillna(df1['Episode_Length_minutes'].median())
df1['Guest_Popularity_percentage'] = df1['Guest_Popularity_percentage'].fillna(df1['Guest_Popularity_percentage'].median())


df1.isna().sum()


ep_len_cap = df1['Episode_Length_minutes'].quantile(0.99)
ads_cap = df1['Number_of_Ads'].quantile(0.99)


df1['Episode_Length_minutes'] = np.clip(df1['Episode_Length_minutes'], None, ep_len_cap)
df1['Number_of_Ads'] = np.clip(df1['Number_of_Ads'], None, ads_cap)


df1[['Episode_Length_minutes', 'Number_of_Ads']].max()


columns=df1.columns.tolist()
print(columns)


for dtype in df1.dtypes.unique():
    cat_columns = list(df1.select_dtypes(include = object).columns)

print(cat_columns)


df_encoded = pd.get_dummies(df1, columns=cat_columns, drop_first=True)



print("Shape after encoding:", df_encoded.shape)
df_encoded.head()


df_encoded = df_encoded.astype({col: int for col in df_encoded.select_dtypes('bool').columns})


episode_number = df_train['Episode_Title'].str.extract(r'(\d+)').astype(float).rename(columns={0: 'Episode_Number'})
total_popularity = (df_encoded['Host_Popularity_percentage'] + df_encoded['Guest_Popularity_percentage']).rename('Total_Popularity')


df_encoded = pd.concat([df_encoded, episode_number, total_popularity], axis=1)


df_encoded.head()


df_full = pd.concat([df1.reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)



df_full.head()


df_full.shape



df_full.drop(columns=cat_columns, inplace=True, errors='ignore')


df_full.head()


# Remove duplicate columns by name, keeping the first occurrence
df_full = df_full.loc[:, ~df_full.columns.duplicated()]


df_full.head()


X = df_full.drop('Listening_Time_minutes', axis=1)
y = df_full['Listening_Time_minutes']

# Train-test split



# Check missing values in your features
print(X.isna().sum().sort_values(ascending=False).head(10))



X = X.fillna(X.median())



from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)





from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

from xgboost import XGBRegressor

model = XGBRegressor(
    tree_method="hist",
    device = "cuda",
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
print("R² Score:", r2_score(y_val, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_val, y_pred)))


df_test.head()


df_test.info()


df_test.isna().sum()


df_test.describe()


df_test_clean = df_test.copy()


median_episode_length = df1['Episode_Length_minutes'].median()
median_guest_pop = df1['Guest_Popularity_percentage'].median()

ep_len_cap = df1['Episode_Length_minutes'].quantile(0.99)
ads_cap = df1['Number_of_Ads'].quantile(0.99)


# Fill missing values
df_test_clean['Episode_Length_minutes'] = df_test_clean['Episode_Length_minutes'].fillna(median_episode_length)
df_test_clean['Guest_Popularity_percentage'] = df_test_clean['Guest_Popularity_percentage'].fillna(median_guest_pop)


# Cap outliers like train
df_test_clean['Episode_Length_minutes'] = np.clip(df_test_clean['Episode_Length_minutes'], None, ep_len_cap)
df_test_clean['Number_of_Ads'] = np.clip(df_test_clean['Number_of_Ads'], None, ads_cap)


for dtype in df_test_clean.dtypes.unique():
    cat_cols = list(df_test_clean.select_dtypes(include = object).columns)
print(cat_cols)

# One-hot encode categorical features
df_test_encoded = pd.get_dummies(df_test_clean, columns=cat_cols, drop_first=True)


# Align test with train (add missing columns, drop extras)
df_test_encoded = df_test_encoded.reindex(columns=X.columns, fill_value=0)

# Feature engineering
df_test_encoded['Episode_Number'] = df_test_clean['Episode_Title'].str.extract(r'(\d+)').astype(float)
df_test_encoded['Total_Popularity'] = df_test_encoded['Host_Popularity_percentage'] + df_test_encoded['Guest_Popularity_percentage']

# Convert bools to int (if any)
df_test_encoded = df_test_encoded.astype({col: int for col in df_test_encoded.select_dtypes('bool').columns})






# Predict
test_preds = model.predict(df_test_encoded)


# Load sample submission & write predictions
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
sample_submission['Listening_Time_minutes'] = test_preds

# Save the CSV for submission
sample_submission.to_csv("submission.csv", index=False)



# Step 1: Predict again (use a different model if needed)
# If you've already fitted a model like RandomForest or XGBoost:
new_preds = model.predict(df_test_encoded)

#  Step 2: Copy the sample submission
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
sample_submission['Listening_Time_minutes'] = new_preds

# Step 3: Save with a new name
sample_submission.to_csv("submission_v2.csv", index=False)



import matplotlib.pyplot as plt
import seaborn as sns

importances = model.feature_importances_
features = X.columns

# Create DataFrame
feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_imp.sort_values(by='Importance', ascending=False, inplace=True)

# Plot top 20
plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp.head(20), x='Importance', y='Feature')
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()


low_imp_features = feat_imp[feat_imp['Importance'] < 0.001]['Feature'].tolist()
print(low_imp_features)


X_reduced = X.drop(columns=low_imp_features)


X_train, X_val, y_train, y_val = train_test_split(X_reduced, y, test_size=0.2, random_state=42)






from xgboost import XGBRegressor

model = XGBRegressor(
    tree_method="hist",
    device = "cuda",
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    verbosity=1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
print("R² Score:", r2_score(y_val, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_val, y_pred)))





from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

# Base model with GPU
xgb = XGBRegressor(tree_method="hist", device="cuda", random_state=42)

# Hyperparameter grid
param_dist = {
    'n_estimators': [100, 300, 500],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Randomized Search
rs_xgb = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=10,
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit on reduced features
rs_xgb.fit(X_reduced, y)

# Best params and score
print("Best RMSE score:", -rs_xgb.best_score_)
print("Best parameters:", rs_xgb.best_params_)



# Retrain best model on full train data
from xgboost import XGBRegressor

best_model = XGBRegressor(
    tree_method="hist",
    device="cuda",
    n_estimators=500,
    max_depth=8,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=42
)

best_model.fit(X_reduced, y)



# Predict on test set
final_preds = best_model.predict(df_test_encoded[X_reduced.columns])

# Load sample submission
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
sample_submission['Listening_Time_minutes'] = final_preds

# Save submission file
sample_submission.to_csv("submission_tuned_xgb.csv", index=False)



df1['Total_Popularity'] = df1['Host_Popularity_percentage'] + df1['Guest_Popularity_percentage']


df1['Host_Ads_Interaction'] = df1['Host_Popularity_percentage'] * df1['Number_of_Ads']
df1['Guest_Ads_Interaction'] = df1['Guest_Popularity_percentage'] * df1['Number_of_Ads']


day_map = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
    'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
}
df1['Day_Ordinal'] = df1['Publication_Day'].map(day_map)


time_map = {
    'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4
}
df1['Time_Ordinal'] = df1['Publication_Time'].map(time_map)


# Add to X and rerun model
X_new = pd.concat([X_reduced, df1[['Total_Popularity', 'Host_Ads_Interaction', 'Guest_Ads_Interaction', 'Day_Ordinal', 'Time_Ordinal']]], axis=1)

# (Optional) Check correlation to target
df1[['Listening_Time_minutes', 'Total_Popularity', 'Host_Ads_Interaction', 'Guest_Ads_Interaction', 'Day_Ordinal', 'Time_Ordinal']].corr()



import matplotlib.pyplot as plt
import seaborn as sns

# Subset of features + target
corr_features = ['Listening_Time_minutes', 'Total_Popularity',
                 'Host_Ads_Interaction', 'Guest_Ads_Interaction',
                 'Day_Ordinal', 'Time_Ordinal']

# Compute correlation
corr_matrix = df1[corr_features].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.tight_layout()
plt.show()


# Add engineered features to your training feature set
engineered_features = ['Total_Popularity', 'Host_Ads_Interaction', 'Guest_Ads_Interaction', 'Day_Ordinal', 'Time_Ordinal']
X_final = pd.concat([X_reduced, df1[engineered_features]], axis=1)


# Apply the same feature engineering to df_test_clean
df_test_clean['Total_Popularity'] = df_test_clean['Host_Popularity_percentage'] + df_test_clean['Guest_Popularity_percentage']
df_test_clean['Host_Ads_Interaction'] = df_test_clean['Host_Popularity_percentage'] * df_test_clean['Number_of_Ads']
df_test_clean['Guest_Ads_Interaction'] = df_test_clean['Guest_Popularity_percentage'] * df_test_clean['Number_of_Ads']

df_test_clean['Day_Ordinal'] = df_test_clean['Publication_Day'].map(day_map)
df_test_clean['Time_Ordinal'] = df_test_clean['Publication_Time'].map(time_map)


# Add these to encoded test set
# Remove duplicate column names if any
df_test_encoded = df_test_encoded.loc[:, ~df_test_encoded.columns.duplicated()]



df_test_encoded = df_test_encoded.reindex(columns=X_final.columns, fill_value=0)


df_test_encoded.head()


X_final = X_final.loc[:, ~X_final.columns.duplicated()]


from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

model_final = XGBRegressor(
    tree_method="hist",
    device="cuda",
    n_estimators=500,
    max_depth=8,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=42
)

model_final.fit(X_final, y)

y_pred = model_final.predict(X_final)
print("Final R² Score:", r2_score(y, y_pred))
print("Final RMSE:", np.sqrt(mean_squared_error(y, y_pred)))



# Remove duplicate columns from test data
df_test_encoded = df_test_encoded.loc[:, ~df_test_encoded.columns.duplicated()]


# Predict on the test set using your final trained model
final_preds = model_final.predict(df_test_encoded)

# Load sample submission template
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')  # replace with correct path

# Assign predictions
sample_submission['Listening_Time_minutes'] = final_preds

# Save to CSV
sample_submission.to_csv("submissionv4.csv", index=False)



from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import numpy as np

kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_final))
test_preds = np.zeros(len(df_test_encoded))
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_final)):
    X_tr, X_val = X_final.iloc[train_idx], X_final.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBRegressor(
        tree_method="hist",
        device="cuda",
        n_estimators=500,
        max_depth=8,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=fold
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)
    oof_preds[val_idx] = preds
    scores.append(np.sqrt(mean_squared_error(y_val, preds)))

    # Predict on test set and accumulate
    test_preds += model.predict(df_test_encoded) / kf.n_splits

# Final RMSE across folds
print("CV RMSE per fold:", scores)
print("Mean CV RMSE:", np.mean(scores))



from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from scipy.stats import uniform, randint


X_train, X_val, y_train, y_val = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

xgb_model = XGBRegressor(tree_method="hist", device="cuda", random_state=42)

# Define hyperparameter search space
param_dist = {
    "n_estimators": randint(100, 1000),
    "learning_rate": uniform(0.01, 0.3),
    "max_depth": randint(3, 12),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.5, 0.5),
    "reg_alpha": uniform(0, 1),
    "reg_lambda": uniform(1, 5)
}

# Randomized Search
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=50,
    scoring="neg_root_mean_squared_error",
    cv=3,
    verbose=1,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# Best model
best_xgb = random_search.best_estimator_

# Evaluate
from sklearn.metrics import mean_squared_error
y_pred = best_xgb.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Tuned RMSE: {rmse:.4f}")



df_test_aligned = df_test_encoded[X_reduced.columns]

final_test_preds = best_xgb.predict(df_test_aligned)

sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
sample_submission['Listening_Time_minutes'] = final_test_preds
sample_submission.to_csv("submission_tuned_v5.csv", index=False)





