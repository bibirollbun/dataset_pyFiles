import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
import warnings
warnings.filterwarnings('ignore')
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample submission shape:", sample_submission.shape)

#  Preview datasets
print("\n--- Train Head ---")
display(train.head())

print("\n--- Test Head ---")
display(test.head())

print("\n--- Sample Submission Head ---")
display(sample_submission.head())


#  Check data types, nulls, unique counts
def dataset_overview(df, name="Dataset"):
    print(f"\n{name} Overview")
    print("="*40)
    print(df.info())
    
    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        print("\nMissing Values:")
        print(missing)
    else:
        print("\nNo missing values detected.")

    # Unique counts for first few cols
    print("\nUnique values per column (top 10):")
    print(df.nunique().sort_values(ascending=False).head(10))
    
   

dataset_overview(train, "Train")
dataset_overview(test, "Test")


target = "accident_risk"

print("Target Variable Analysis")
print("="*40)
print(train[target].describe())
print("\nSkewness:", train[target].skew())
print("Kurtosis:", train[target].kurtosis())

plt.figure(figsize=(10,5))
sns.histplot(train[target], kde=True, bins=50, color="skyblue")
plt.title("Distribution of Accident Risk")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x=train[target], color="lightcoral")
plt.title("Boxplot of Accident Risk")
plt.show()


num_features = ["curvature", "num_lanes", "speed_limit", "num_reported_accidents"]

# Summary stats
print("\nNumerical Features Summary")
print("="*40)
print(train[num_features].describe())

# Distribution plots
for col in num_features:
    plt.figure(figsize=(10,4))
    sns.histplot(train[col], kde=True, bins=30, color="lightgreen")
    plt.title(f"Distribution of {col}")
    plt.show()

# Correlation with target
corrs = train[num_features + [target]].corr()[target].sort_values(ascending=False)
print("\nCorrelation with target:")
print(corrs)

plt.figure(figsize=(6,4))
sns.heatmap(train[num_features + [target]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Numeric Features + Target)")
plt.show()

# Scatterplots with target
for col in num_features:
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=train[col], y=train[target], alpha=0.3)
    plt.title(f"{col} vs Accident Risk")
    plt.show()





cat_features = ["road_type", "lighting", "weather", "time_of_day"]
target = "accident_risk"

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", palette="Set2", font_scale=1.1)

# Countplots (horizontal layout)
fig, axes = plt.subplots(1, len(cat_features), figsize=(20,5))
for i, col in enumerate(cat_features):
    sns.countplot(data=train, x=col, ax=axes[i])
    axes[i].set_title(f"{col} counts")
    axes[i].tick_params(axis='x', rotation=30)
plt.suptitle("Categorical Feature Distributions", fontsize=14)
plt.tight_layout()
plt.show()

# Boxplots of target by category (horizontal layout)
fig, axes = plt.subplots(1, len(cat_features), figsize=(22,6))
for i, col in enumerate(cat_features):
    sns.boxplot(data=train, x=col, y=target, ax=axes[i], palette="Set3")
    axes[i].set_title(f"Accident Risk by {col}")
    axes[i].tick_params(axis='x', rotation=30)
plt.suptitle("Accident Risk Distribution by Categorical Features", fontsize=14)
plt.tight_layout()
plt.show()

# Mean target per category (tabular summary)
for col in cat_features:
    mean_risk = train.groupby(col)[target].mean().sort_values(ascending=False)
    print(f"\nAverage Accident Risk by {col}:")
    print(mean_risk)



bool_features = ["road_signs_present", "public_road", "holiday", "school_season"]

# Countplots 
fig, axes = plt.subplots(1, len(bool_features), figsize=(20,4))
for i, col in enumerate(bool_features):
    sns.countplot(data=train, x=col, ax=axes[i], palette="pastel")
    axes[i].set_title(f"{col} counts")
plt.suptitle("Boolean Feature Distributions", fontsize=14)
plt.tight_layout()
plt.show()

# Boxplots: target distribution per boolean feature
fig, axes = plt.subplots(1, len(bool_features), figsize=(20,4))
for i, col in enumerate(bool_features):
    sns.boxplot(data=train, x=col, y=target, ax=axes[i], palette="Set2")
    axes[i].set_title(f"Accident Risk by {col}")
plt.suptitle("Accident Risk by Boolean Features", fontsize=14)
plt.tight_layout()
plt.show()

# Mean target per boolean feature
for col in bool_features:
    mean_risk = train.groupby(col)[target].mean().sort_values(ascending=False)
    print(f"\nAverage Accident Risk by {col}:")
    print(mean_risk)


# bin curvature into 3 categories
train['curvature_bin'] = pd.qcut(train['curvature'], q=3, labels=['Low','Medium','High'])

plt.figure(figsize=(8,4))
sns.boxplot(x='road_type', y=target, hue='curvature_bin', data=train, palette="Set2")
plt.title("Accident Risk by Road Type & Curvature Bin")
plt.show()





from scipy import stats

# Z-score method for numeric features
z_scores = np.abs(stats.zscore(train[num_features]))
outlier_rows = (z_scores > 3).any(axis=1)
print(f"Number of outlier rows (z-score > 3): {outlier_rows.sum()}")

# Boxplots highlighting outliers
fig, axes = plt.subplots(1, len(num_features), figsize=(20,4))
for i, col in enumerate(num_features):
    sns.boxplot(data=train, x=train[col], ax=axes[i], color="lightcoral")
    axes[i].set_title(f"Outlier check: {col}")
plt.suptitle("Boxplots for Numeric Features (Outliers)", fontsize=14)
plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np


target = "accident_risk"
id_col = "id"

num_features = ["curvature", "speed_limit", "num_reported_accidents", "num_lanes"]
cat_features = ["road_type", "lighting", "weather", "time_of_day"]
bool_features = ["road_signs_present", "public_road", "holiday", "school_season"]


X = train.drop(columns=[target, id_col])
y = train[target]



# Numeric pipeline
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Boolean pipeline
bool_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', 'passthrough')  # Already True/False â†’ handled as 1/0 later
])




# 4. Combine all pipelines
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features),
    ('bool', bool_pipeline, bool_features)
])


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training shape:", X_train.shape)
print("Validation shape:", X_valid.shape)


# Convert boolean columns to integers (0 and 1)
bool_cols = X_train.select_dtypes(include='bool').columns
X_train[bool_cols] = X_train[bool_cols].astype(int)
X_valid[bool_cols] = X_valid[bool_cols].astype(int)

# Now  fit and transform
X_train_prep = preprocessor.fit_transform(X_train)
X_valid_prep = preprocessor.transform(X_valid)

print("Transformed Training shape:", X_train_prep.shape)
print("Transformed Validation shape:", X_valid_prep.shape)




from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


models = {
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
}


results = []

for name, model in models.items():
    # Train
    model.fit(X_train_prep, y_train)
    
    # Predict on validation set
    y_pred = model.predict(X_valid_prep)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    mae = mean_absolute_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)
    
    results.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "RÂ²": r2
    })

# --------------------------
# 3. Display results

results_df = pd.DataFrame(results).sort_values(by="RMSE")
results_df





X_test = test.drop(columns=["id"])

# Convert boolean columns to int 
bool_cols = X_test.select_dtypes(include='bool').columns
X_test[bool_cols] = X_test[bool_cols].astype(int)

# Apply the same preprocessing as training data
X_test_prep = preprocessor.transform(X_test)


best_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
best_model.fit(X_train_prep, y_train)  
predictions = best_model.predict(X_test_prep)

# --------------------------
# 3. Prepare submission DataFrame

submission = pd.DataFrame({
    "id": test["id"],
    "accident_risk": predictions
})

# Round predictions to 3 decimals (as in sample submission)
submission["accident_risk"] = submission["accident_risk"].round(3)


submission.to_csv("submission.csv", index=False)
submission.head()




import joblib


model_filename = "gradient_boosting_model.pkl"
joblib.dump(best_model, model_filename)

print(f"Model saved as {model_filename}")





import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np

# --------------------------
# 0. Define feature groups
# --------------------------
numeric_features = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# --------------------------
# 1. Feature Importance (Gradient Boosting)
# --------------------------
# Get transformed feature names from preprocessor
# If using OneHotEncoder in ColumnTransformer
cat_ohe = preprocessor.named_transformers_['cat']['onehot']
cat_feature_names = cat_ohe.get_feature_names_out(categorical_features)

feature_names = np.concatenate([numeric_features, cat_feature_names])

# Extract feature importances
importances = best_model.feature_importances_

# Create DataFrame
feat_imp = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot top 20 features
plt.figure(figsize=(12,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp.head(20))
plt.title("Top 20 Feature Importances - Gradient Boosting Regressor")
plt.show()

# --------------------------
# 2. Optional: SHAP values (slower, for explanation)
# --------------------------
explainer = shap.Explainer(best_model, X_train_prep)
shap_values = explainer(X_valid_prep[:5000])  # sample for speed
shap.summary_plot(shap_values, features=X_valid_prep[:5000], feature_names=feature_names)





