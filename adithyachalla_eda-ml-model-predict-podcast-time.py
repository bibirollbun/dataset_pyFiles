# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Core Libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Model Selection
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Models: Linear
from sklearn.linear_model import Ridge, LinearRegression
from catboost import CatBoostRegressor, Pool

# Models: Tree-based
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    StackingRegressor
)

# Models: Boosting Libraries
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Evaluation
from sklearn.metrics import mean_squared_error, r2_score

# Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



train_df=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
train_df


test_df=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
test_df


train_df.info()


train_df['Genre'].value_counts()


test_df['Genre'].value_counts()


train_df['Number_of_Ads'].describe()


train_df.isnull().sum()


train_df['Publication_Time'].value_counts()


train_df['Guest_Popularity_percentage'].describe()


sns.boxplot(x=train_df['Number_of_Ads'])
plt.title("Boxplot of Number of Ads")



train_df['Number_of_Ads'].fillna(train_df['Number_of_Ads'].mode()[0], inplace=True)


train_df.loc[train_df['Number_of_Ads'] > 10, 'Number_of_Ads'] = train_df['Number_of_Ads'].mode()[0]


train_df[train_df['Guest_Popularity_percentage'] > 100]['Guest_Popularity_percentage'].value_counts()


sns.boxplot(x=train_df["Episode_Length_minutes"])
plt.title("Boxplot of Episode Lengths")





# --------- Handle Missing Values in TRAIN ---------
impute_cols = ["Episode_Length_minutes", "Guest_Popularity_percentage", "Number_of_Ads"]

# Fit imputer on train data
imputer = SimpleImputer(strategy='median')
train_df[impute_cols] = imputer.fit_transform(train_df[impute_cols])

# --------- Clip Guest Popularity in TRAIN ---------
train_df["Guest_Popularity_percentage"] = train_df["Guest_Popularity_percentage"].clip(upper=100)

# --------- Compute IQR thresholds on TRAIN ---------
Q1_ep = train_df["Episode_Length_minutes"].quantile(0.25)
Q3_ep = train_df["Episode_Length_minutes"].quantile(0.75)
IQR_ep = Q3_ep - Q1_ep
upper_ep_len = Q3_ep + 1.5 * IQR_ep

Q1_ads = train_df["Number_of_Ads"].quantile(0.25)
Q3_ads = train_df["Number_of_Ads"].quantile(0.75)
IQR_ads = Q3_ads - Q1_ads
upper_ads = Q3_ads + 1.5 * IQR_ads

# Clip train values
train_df["Episode_Length_minutes"] = train_df["Episode_Length_minutes"].clip(upper=upper_ep_len)
train_df["Number_of_Ads"] = train_df["Number_of_Ads"].clip(upper=upper_ads)

# --------- Apply SAME logic to TEST ---------

# Impute using train-fitted imputer
test_df[impute_cols] = imputer.transform(test_df[impute_cols])

# Clip Guest Popularity
test_df["Guest_Popularity_percentage"] = test_df["Guest_Popularity_percentage"].clip(upper=100)

# Clip using train IQR thresholds
test_df["Episode_Length_minutes"] = test_df["Episode_Length_minutes"].clip(upper=upper_ep_len)
test_df["Number_of_Ads"] = test_df["Number_of_Ads"].clip(upper=upper_ads)

# Clean up any residual infs or NaNs
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.fillna(0, inplace=True)



test_df.isnull().sum()


# Get value counts
genre_counts = train_df['Genre'].value_counts()

# Create barplot
sns.barplot(x=genre_counts.index, y=genre_counts.values)
plt.xticks(rotation=45)  # Rotate x-axis labels if needed
plt.xlabel("Genre")
plt.ylabel("Count")
plt.title("Genre Distribution")
plt.tight_layout()
plt.show()




sns.boxplot(x=train_df['Episode_Length_minutes'])
plt.title("Boxplot of Episode Lengths")
plt.xlabel("Minutes")
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot(data=train_df, x='Genre', order=train_df['Genre'].value_counts().index, palette='Set2')
plt.title('Number of Episodes per Genre')
plt.xticks(rotation=45)
plt.xlabel('Genre')
plt.ylabel('Number of Episodes')
plt.tight_layout()
plt.show()




numeric_cols = train_df[["Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Number_of_Ads", "Listening_Time_minutes"]]
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()



sns.boxplot(x="Episode_Sentiment", y="Listening_Time_minutes", data=train_df)
plt.title("Listening Time by Episode Sentiment")
plt.show()



genre_stats = train_df.groupby("Genre")["Listening_Time_minutes"].mean().sort_values()
genre_stats.plot(kind='barh', title="Average Listening Time by Genre")
plt.xlabel("Avg Listening Time (minutes)")
plt.show()



train_df["Completion_Ratio"] = train_df["Listening_Time_minutes"] / train_df["Episode_Length_minutes"]



sns.boxplot(x="Number_of_Ads", y="Completion_Ratio", data=train_df)
plt.title("Completion Ratio vs Number of Ads")
plt.show()



train_df["Pub_Window"] = train_df["Publication_Day"] + " - " + train_df["Publication_Time"]
top_windows = train_df.groupby("Pub_Window")["Listening_Time_minutes"].mean().sort_values(ascending=False)
top_windows.plot(kind='bar', figsize=(12, 5), title="Avg Listening Time by Day & Time")
plt.ylabel("Avg Listening Time")
plt.xticks(rotation=45)
plt.show()



sns.scatterplot(x="Host_Popularity_percentage", y="Listening_Time_minutes", data=train_df, label="Host", alpha=0.5)
sns.scatterplot(x="Guest_Popularity_percentage", y="Listening_Time_minutes", data=train_df, label="Guest", alpha=0.5)
plt.legend()
plt.title("Popularity vs Listening Time")
plt.show()



pivot = train_df.pivot_table(index="Genre", columns="Episode_Sentiment", values="Listening_Time_minutes", aggfunc="mean")
pivot.plot(kind="bar", figsize=(12, 5))
plt.title("Avg Listening Time by Genre and Sentiment")
plt.ylabel("Listening Time")
plt.xticks(rotation=45)
plt.show()



common_cols = train_df.columns.intersection(test_df.columns)
categorical_cols = train_df[common_cols].select_dtypes(include=['object']).columns
numerical_cols = train_df[common_cols].select_dtypes(include=['int64', 'float64']).columns


X = train_df[common_cols]
y = train_df['Listening_Time_minutes']



# Define common columns
common_cols = train_df.columns.intersection(test_df.columns)
categorical_cols = train_df[common_cols].select_dtypes(include=['object']).columns
numerical_cols = train_df[common_cols].select_dtypes(include=['int64', 'float64']).columns

X = train_df[common_cols].copy()
y = train_df['Listening_Time_minutes'].copy()
X_test = test_df[common_cols].copy()

# Label Encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# Impute missing values
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])
X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

# Clean test set residuals
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.fillna(0, inplace=True)

# Log-transform the target variable
# Avoid log(0) by clipping very small values (if needed)
y = y.clip(lower=1e-5)  # or any small epsilon
y_log = np.log(y)



# Split training data for evaluation
X_train, X_val, y_train_log, y_val_log = train_test_split(X, y_log, test_size=0.2, random_state=42)



# Initialize models
models = {
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Train, predict, and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train_log)
    preds_log = model.predict(X_val)
    preds = np.exp(preds_log)  # inverse log-transform

    y_val_true = np.exp(y_val_log)  # inverse log-transform
    rmse = mean_squared_error(y_val_true, preds, squared=False)
    r2 = r2_score(y_val_true, preds)

    print(f"\n{name}")
    print(f"RMSE: {rmse:.4f}")
    print(f"RÂ² Score: {r2:.4f}")



# Use CatBoost's native categorical support
cat_features = [X.columns.get_loc(col) for col in categorical_cols]
train_pool = Pool(X_train, label=y_train_log, cat_features=cat_features)
val_pool = Pool(X_val, cat_features=cat_features)

# Train model
model = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, loss_function='RMSE', verbose=100)
model.fit(train_pool)

# Evaluate
y_pred_log = model.predict(val_pool)
y_pred = np.exp(y_pred_log)
y_true = np.exp(y_val_log)

print("RMSE:", mean_squared_error(y_true, y_pred, squared=False))
print("RÂ²:", r2_score(y_true, y_pred))


# Final features (numerical + label-encoded categoricals)
final_features = numerical_cols.tolist() + categorical_cols.tolist()
X_train_final = X_train[final_features]
X_val_final = X_val[final_features]

# Define base models including CatBoost
base_models = [
    ('ridge', Ridge(alpha=1.0)),
    ('gbr', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),
    ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),
    ('lgbm', LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),
    ('cat', CatBoostRegressor(iterations=100, learning_rate=0.1, depth=5, verbose=0, random_state=42))
]

# Meta-model
meta_model = LinearRegression()

# Create and train stacking regressor
stacking_reg = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    passthrough=True,
    n_jobs=-1
)

# Fit on log-transformed target
stacking_reg.fit(X_train_final, y_train_log)

# Predict and inverse-transform
y_val_pred_log = stacking_reg.predict(X_val_final)
y_val_pred = np.exp(y_val_pred_log)
y_val_true = np.exp(y_val_log)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))
print(f"Stacking Regressor Validation RMSE (with CatBoost): {rmse:.4f}")



# Re-applying the same preprocessing steps to test_df

# Step 1: Extract test IDs
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
test_ids = test_df['id']
X_test = test_df[common_cols].copy()

# Step 2: Label Encoding using previously fitted encoders
for col in categorical_cols:
    le = label_encoders[col]
    X_test[col] = le.transform(X_test[col].astype(str))

# Step 3: Imputation using previously fitted imputers
X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])
X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

# Step 4: Cleanup
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.fillna(0, inplace=True)

# Step 5: Predict and inverse-transform
preds_log = model.predict(X_test)
preds = np.exp(preds_log)  # Inverse of log-transform

# Step 6: Create submission
submission = pd.DataFrame({
    'id': test_ids,
    'Listening_Time_minutes': preds
})

submission.to_csv("submission.csv", index=False)
submission.head()



# Ensure X_test_final has the same final features as training
X_test_final = X_test[final_features]  # numerical + label-encoded categorical

# Predict on test set (log scale)
y_test_pred_log = stacking_reg.predict(X_test_final)

# Inverse log transform
y_test_pred = np.exp(y_test_pred_log)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],
    'Listening_Time_minutes': y_test_pred
})

# Save to CSV
submission.to_csv("submission.csv", index=False)

# Show preview
submission.head()


