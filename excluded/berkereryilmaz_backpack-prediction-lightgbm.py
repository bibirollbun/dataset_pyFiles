# Import necessary modules
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve, validation_curve, cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')


# Load the data
train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
train2 = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")

# Combine train and train2 datasets into one
df = pd.concat([train, train2], axis=0,ignore_index=True)

# rows, cols
print("Train dataset: ", df.shape)
print("Test dataset: ", test.shape)


df.head()


test.head()


print(df.info())
print("=====================================")
print(test.info())


df.describe()


df.isnull().sum()


test.isnull().sum()


df.nunique()


df["Style"].unique()


# Set figure size
plt.figure(figsize=(12, 6))

# Boxplot for Price Distribution
plt.subplot(1, 2, 1)
sns.boxplot(y=df["Price"], color="lightblue")
plt.title("Price Distribution (Boxplot)")

# Violin plot for Weight Capacity Comparison
plt.subplot(1, 2, 2)
sns.violinplot(data=[df["Weight Capacity (kg)"], test["Weight Capacity (kg)"]], 
               palette=["blue", "orange"])
plt.xticks([0, 1], ["Train", "Test"])
plt.title("Weight Capacity Distribution (Violin Plot)")

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
sns.kdeplot(df['Weight Capacity (kg)'], label="Train", fill=True, alpha=0.5, color="blue")
sns.kdeplot(test['Weight Capacity (kg)'], label="Test", fill=True, alpha=0.5, color="red")
plt.title("Weight Capacity Density Comparison")
plt.legend()
plt.show()


# Plot price distributions
plt.figure(figsize=(12, 5))
sns.histplot(df["Price"], bins=60, kde=True, color="blue")
plt.title("Price Distribution")
plt.show()


# 'id' column copied 
test_ids = test['id']

# 'id' column dropped
df.drop(columns=["id"], inplace=True)
test.drop(columns=["id"], inplace=True)


# Encode the categorical features
categorical_cols = ['Size', 'Brand', 'Material', 'Style', 'Color', 'Laptop Compartment', 'Waterproof']

# fill data with "unknownn"
df[categorical_cols] = df[categorical_cols].fillna("Unknown")
test[categorical_cols] = test[categorical_cols].fillna('Unknown')

# Filling 'Weight Capacity (kg)' missing values using Mean
df['Weight Capacity (kg)'].fillna(df['Weight Capacity (kg)'].mean(), inplace=True)
test['Weight Capacity (kg)'].fillna(test['Weight Capacity (kg)'].mean(), inplace=True)


# New feature: Size and Weight Capacity multiply
df['Size_Weight'] = df['Size'].astype(str) + "_" + df['Weight Capacity (kg)'].astype(str)
test['Size_Weight'] = test['Size'].astype(str) + "_" + test['Weight Capacity (kg)'].astype(str)


# Target Encoder ile Size_Weight özelliğini encode et
target_encoder_sw = TargetEncoder(cols=['Size_Weight'], smoothing=25, min_samples_leaf=15)
df['Size_Weight'] = target_encoder_sw.fit_transform(df['Size_Weight'], df['Price'])
test['Size_Weight'] = target_encoder_sw.transform(test['Size_Weight'])


# Standartization (Normalizing big values such as Weight Capacity)
scaler = StandardScaler()
df['Weight Capacity (kg)'] = scaler.fit_transform(df[['Weight Capacity (kg)']])
test['Weight Capacity (kg)'] = scaler.transform(test[['Weight Capacity (kg)']])


df.isnull().sum()


# train test split
X = df.drop(columns=["Price"])
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Target Encoding (Ensuring Consistency)
target_encoder = TargetEncoder(cols=categorical_cols, smoothing=25, min_samples_leaf=15)
X_train_encoded = target_encoder.fit_transform(X_train, y_train)
X_test_encoded = target_encoder.transform(X_test)  # Validation set
Test_encoded = target_encoder.transform(test)  # Competition test set for final prediction


# Function to find best hyperparameters to tune the LGBMRegressor using RandomizedSearchCV
def randomized_search_lgbm(X_train, y_train):
    lgbm = LGBMRegressor(objective='regression')
    
    params = {
        'n_estimators': [400, 500, 750],
        'learning_rate': [0.01, 0.1, 0.05],
        'max_depth': [5, 7, 10],
        'num_leaves': [20, 40],
        'min_child_samples': [20, 30, 40],
        'subsample': [0.5, 1.0],
        'colsample_bytree': [0.5, 1.0],
        'reg_lambda': [0, 5, 10],
        'reg_alpha': [0, 0.1, 0.5],
    }
    
    randomized_search_lgbm = RandomizedSearchCV(lgbm, params, n_iter=50, cv=5, scoring='neg_mean_squared_error', 
                                           verbose=2, n_jobs=-1)
    
    randomized_search_lgbm.fit(X_train, y_train)
    print("Best parameters for LGBM:", randomized_search_lgbm.best_params_)
    print("Best Score for LGBM:", randomized_search_lgbm.best_score_)

    return randomized_search_lgbm.best_params_, randomized_search_lgbm.best_score_


# Find best hyperparameters for LGBMRegressor 
best_params_lgbm, best_score_lgbm = randomized_search_lgbm(X_train_encoded, y_train)

# Build the LGBM model with the best parameters
best_lgbm = LGBMRegressor(**best_params_lgbm)

# Fit the Model
best_lgbm.fit(X_train_encoded, y_train)

# Make predict for test
y_pred_lgbm = best_lgbm.predict(X_test_encoded)

# Evaluate performance
mse_lgbm = mean_squared_error(y_test, y_pred_lgbm)
rmse_lgbm = np.sqrt(mse_lgbm)
print(f"Test RMSE for LGBM: {rmse_lgbm:}")

# Make final predictions using the Competition Test Set (Test_encoded)
final_predictions_lgbm = best_lgbm.predict(Test_encoded)
print("Final Predictions (LightGBM):", final_predictions_lgbm)


# Plot feature importances
importance_xgb = best_lgbm.feature_importances_
sorted_idx = np.argsort(importance_xgb)[::-1]
features = X_train_encoded.columns

plt.figure(figsize=(10, 6))
plt.barh([features[i] for i in sorted_idx], importance_xgb[sorted_idx])
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Light GBM Regression Feature Importance')
plt.gca().invert_yaxis()  
plt.show()


# Create submission csv 
test_pred = best_lgbm.predict(Test_encoded, num_iteration=best_lgbm.best_iteration_)

# Submission dosyasına kaydet
submission = pd.DataFrame({'id': test_ids, 'Price': test_pred})
submission.to_csv('/kaggle/working/submission.csv', index=False)
display(submission)

