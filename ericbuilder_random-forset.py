# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')


## Loading the data
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')  

print("Train shape:", train.shape)
print("Test shape:", test.shape)


# Display first few rows of the training data
train.head()


# remove "id" column from train and test datasets
train.drop(columns=["id"], inplace=True)
test.drop(columns=["id"], inplace=True)


# Check missing values
missing_train = train.isnull().sum().sum()
missing_test = test.isnull().sum().sum()
print(f"Missing values in training set: {missing_train}")
print(f"Missing values in test set: {missing_test}")


# Check duplicate rows
train.duplicated().sum()
#remove duplicate rows
train.drop_duplicates(inplace=True)


# Encoding categorical variables

num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = train.select_dtypes(exclude=[np.number]).columns.tolist()
num_cols.remove("accident_risk")

print(f"categorical columns : {cat_cols}")
print(f"numerical columns : {num_cols}")



#categorical columns unique values
for col in cat_cols:
    print(f"{col} : {train[col].unique()}")


# duplicate X for feature engineering
X = train.drop(columns=["accident_risk"])
y = train["accident_risk"]

X_orig = X.copy()
test_orig = test.copy()



# Non-linear transformation features (handling long-tailed distributions and non-linear relationships)
# Capture non-linear risk of speed limits (risk increases faster at high speeds)
X_orig["speed_limit_sq"] = X_orig["speed_limit"] **2  
# Smooth the impact of extreme curvatures 
X_orig["curvature_sqrt"] = np.sqrt(X_orig["curvature"] + 1)

#The test dataset undergoes identical changes
test_orig["speed_limit_sq"] = test_orig["speed_limit"]** 2
test_orig["curvature_sqrt"] = np.sqrt(test_orig["curvature"] + 1)

#A lower number of lanes might together wtih a higher number of accidents means the road might be a little bit higher than others.
X_orig["accident_density"] = X_orig["num_reported_accidents"] / (X_orig["num_lanes"] + 1)
test_orig["accident_density"] = test_orig["num_reported_accidents"] / (test_orig["num_lanes"] + 1)
#We believe that the number of lanes together with speed limit could cause some change of the risk.
#And the random forest model(axis-aligned split function) might not be that good at dealing with liner features.
X_orig["speed_limit_x_lanes"] = X_orig["speed_limit_sq"] * X_orig["num_lanes"]
X_orig["curvature_x_accidents"] = X_orig["curvature_sqrt"] * X_orig["num_reported_accidents"]
test_orig["speed_limit_x_lanes"] = test_orig["speed_limit_sq"] * test_orig["num_lanes"]
test_orig["curvature_x_accidents"] = test_orig["curvature_sqrt"] * test_orig["num_reported_accidents"]

# Custom feature: stress score based on speed limit, number of lanes, road type, and accident history
def stress_score(X):
    return (
        0.3*(X["speed_limit"]/100) +
        0.3*(X["num_lanes"]/5) +
        0.2*(X["public_road"] == 1).astype(int) +
        0.2*(X["num_reported_accidents"] > 1).astype(int)
    )
X_orig["stress_score"] = stress_score(X_orig)
test_orig["stress_score"] = stress_score(test_orig)



def visibility_score(X):
    return (
        0.3*(X["lighting"] == "night").astype(int) +
        0.4*(X["weather"].isin(["fog", "rain", "snow"])).astype(int) +
        0.3*(X["time_of_day"].isin(["dawn", "dusk"])).astype(int)
    )
X_orig["visibility_score"] = visibility_score(X_orig)
test_orig["visibility_score"] = visibility_score(test_orig)


# Convert boolean columns to integers
bool_cols = ["road_signs_present", "public_road","holiday", "school_season"]
for col in bool_cols :
    X_orig[col]= X_orig[col].astype(int)
    test_orig[col]=test_orig[col].astype(int)
# label encoding for categorical variables
cate_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
for col in cate_cols:
    le = LabelEncoder()
    combined = pd.concat([X_orig[col], test_orig[col]], axis=0).astype(str)
    le.fit(combined)
    X_orig[col] = le.transform(X_orig[col].astype(str))
    test_orig[col] = le.transform(test_orig[col].astype(str))

# Group statistical features (involve group patterns in categorical variables)
# Statistical features grouped by road type
# We believe the different type can be better measured in the model by changing them into the average of accidents happened.
road_type_stats = X_orig.groupby("road_type")["num_reported_accidents"].agg(["mean"]).reset_index()
road_type_stats.columns = ["road_type", "roadtype_avg_accident"]
X_orig = X_orig.merge(road_type_stats, on="road_type", how="left")
test_orig = test_orig.merge(road_type_stats, on="road_type", how="left")


# drop features that have high correlation with other features
# This step actually means we throw some basic features we use in the feature engineering above.
X_orig.drop(columns=["speed_limit", "curvature","num_lanes","num_reported_accidents"], inplace=True)
test_orig.drop(columns=["speed_limit", "curvature","num_lanes","num_reported_accidents"], inplace=True)



# heatmap of correlation matrix and show numbers
plt.figure(figsize=(12,10))
sns.heatmap(X_orig.corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f")
plt.title("Feature Correlation Heatmap (Numerical Features)")
plt.tight_layout()
plt.show()


# Hyperparameter tuning with RandomizedSearchCV
param_grid = {
    'n_estimators': [300, 600, 900],
    'max_depth': [10,14,18],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [2, 3, 4],
    'max_features': ['sqrt', 0.5]
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_search = RandomizedSearchCV(
    rf,
    param_distributions=param_grid,
    n_iter=8,
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

rf_search.fit(X_orig, y)
best_params = rf_search.best_params_
print("Best Hyperparameters：", best_params)
print("Best Cross-Validation RMSE：", -rf_search.best_score_)



# Ensemble model with Voting Regressor and Repeated K-Fold Cross-Validation
rf1 = RandomForestRegressor(**best_params, random_state=123, n_jobs=-1)
rf2 = RandomForestRegressor(**best_params, random_state=213, n_jobs=-1)
rf3 = RandomForestRegressor(**best_params, random_state=321, n_jobs=-1)
voting_rf = VotingRegressor(estimators=[('rf1', rf1), ('rf2', rf2), ('rf3', rf3)])

cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
rmse_scores, r2_scores = [], []
df_val = pd.DataFrame()

for fold, (train_idx, val_idx) in enumerate(cv.split(X_orig, y), 1):
    X_train, X_val = X_orig.iloc[train_idx], X_orig.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    voting_rf.fit(X_train, y_train)
    preds = voting_rf.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    rmse_scores.append(rmse)
    r2_scores.append(r2)

    if fold == 1:
        df_val = pd.DataFrame({'y_true': y_val, 'y_pred': preds})

    print(f"Fold {fold}: RMSE={rmse:.5f}, R2={r2:.5f}")

print(f"\n Mean RMSE: {np.mean(rmse_scores):.5f}")
print(f" Mean R²: {np.mean(r2_scores):.5f}")


# Visualizing RMSE scores across folds
plt.figure(figsize=(8,5))
sns.barplot(x=list(range(1, len(rmse_scores)+1)), y=rmse_scores, palette="viridis")
plt.title("RMSE per Fold (Repeated K-Fold)")
plt.xlabel("Fold")
plt.ylabel("RMSE")
plt.tight_layout()
plt.show()


# Boxplot of RMSE scores
plt.figure(figsize=(5,5))
sns.boxplot(y=rmse_scores, color="lightblue")
plt.title("RMSE Distribution across Folds")
plt.ylabel("RMSE")
plt.tight_layout()
plt.show()




# Predicted vs Actual Scatter Plot for Fold 1
plt.figure(figsize=(6,6))
sns.scatterplot(x='y_true', y='y_pred', data=df_val, alpha=0.4)
plt.plot([0,1],[0,1],'--',color='red')
plt.title("Predicted vs Actual (Fold 1 Example)")
plt.xlabel("Actual accident_risk")
plt.ylabel("Predicted accident_risk")
plt.tight_layout()
plt.show()




# Prediction Error Distribution for Fold 1
df_val["error"] = df_val["y_pred"] - df_val["y_true"]
plt.figure(figsize=(7,4))
sns.histplot(df_val["error"], bins=40, color="salmon", kde=True)
plt.title("Prediction Error Distribution (Fold 1)")
plt.xlabel("Prediction Error")
plt.tight_layout()
plt.show()


# Final training on full data and prediction on test set
voting_rf.fit(X_orig, y)
test_pred = np.clip(voting_rf.predict(test_orig), 0, 1)
sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
sub["accident_risk"] = test_pred
sub.to_csv("submission.csv", index=False)
sub.head()

plt.figure(figsize=(7,4))
sns.histplot(test_pred, bins=40, color="orange", kde=True)
plt.title("Distribution of Test Predictions")
plt.xlabel("Predicted accident_risk")
plt.tight_layout()
plt.show()


# Feature importance from the optimized Random Forest
final_rf = rf_search.best_estimator_
final_rf.fit(X_orig, y)
importances = pd.DataFrame({
    'Feature': X_orig.columns,
    'Importance': final_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(data=importances.head(6), x='Importance', y='Feature', palette='viridis')
plt.title('Top 6 Feature Importances (Optimized RF)')
plt.tight_layout()
plt.show()

