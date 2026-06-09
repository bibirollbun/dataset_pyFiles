# Load Lab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



# Load datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


# info Size and Columns 
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nTrain columns:\n", train.columns)
print("\nTest columns:\n", test.columns)


# First 5 rows
train.head()


test.head()


# Check data types and missing values
print("\nTrain info:")
train.info()
print("\nMissing values in train:\n", train.isnull().sum())

print("\nTest info:")
test.info()
print("\nMissing values in test:\n", test.isnull().sum())



# Describe numeric features
print("\nStatistical summary (train):\n", train.describe().T)


# Encode categorical variable we have just sex column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])



# Check target distribution and target variable
target = 'Calories'  #target 
plt.figure(figsize=(10, 6))
sns.histplot(train[target], kde=True, color='teal', bins=30) 
plt.title('Distribution of Target Variable: {}'.format(target), fontsize=14)
plt.xlabel(target, fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True)
plt.show()



#  Correlation Heatmap of Features"
plt.figure(figsize=(12, 10))
corr = train.corr()  

sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5, linecolor='teal', cbar_kws={'shrink': 0.8})
plt.title("Correlation Heatmap of Features", fontsize=16)
plt.xticks(fontsize=12, rotation=45)  
plt.yticks(fontsize=12, rotation=0)   
plt.tight_layout() 
plt.show()


# Compute absolute correlation of all features with the target
target_corr = corr[target].abs().sort_values(ascending=False)
print (target_corr)

target_corr = target_corr.drop(target)

plt.figure(figsize=(10, 6))
sns.barplot(x=target_corr.values, y=target_corr.index, color='teal')
plt.title(f"Feature Correlation with Target: {target}", fontsize=16)
plt.xlabel("Absolute Correlation", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



# correlation with the target variable
top_features = target_corr.index[1:9]  

# Visualizing correlated features
for feature in top_features:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=train, x=feature, y=target, color='teal', s=50) 
    plt.title(f"{feature} vs {target}", fontsize=14)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel(target, fontsize=12)
    plt.grid(True)  
    plt.tight_layout()  
    plt.show()


# Selecting features
features = [col for col in train.columns if col not in ['id', target]]

# Plotting the distribution of each feature
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.histplot(train[feature], kde=True, color='teal', bins=30)
    plt.title(f"Distribution of Feature: {feature}", fontsize=14)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Outlier detection (IQR)
for feature in features:
    Q1 = train[feature].quantile(0.25)
    Q3 = train[feature].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((train[feature] < (Q1 - 1.5 * IQR)) | (train[feature] > (Q3 + 1.5 * IQR))).sum()
    print(f"{feature}: {outliers} outliers")


# Split features and target (Calories)
X = train.drop(['id', 'Calories'], axis=1)
y = train['Calories']
X_test = test.drop('id', axis=1)


# Train,Validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# RandomForestRegressor and hyperparameter grid
rf = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 40],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 10]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=1)

# Fit model
grid_search.fit(X_train_scaled, y_train)

# Best model
best_rf = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)


# Extract feature importances Random Forest model
importances = best_rf.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Importance', y='Feature', data=feature_importance, color='teal')
plt.title("Feature Importance from Random Forest", fontsize=16)

# Annotate bars with numerical importance values
for i, (importance, feature) in enumerate(zip(feature_importance['Importance'], feature_importance['Feature'])):
    ax.text(importance + 0.001, i, f'{importance:.4f}', va='center', ha='left', fontsize=8)

plt.xlabel("Importance", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



# Predict on validation set
val_preds = grid_search.predict(X_val_scaled)
val_preds


# Evaluation Metrics
mse = mean_squared_error(y_val, val_preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val, val_preds)
r2 = r2_score(y_val, val_preds)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.4f}")


# Predict test
test_preds = grid_search.predict(X_test_scaled)
test_preds


# Prepare submission
submission = pd.DataFrame({
    'id': test['id'],
    'Calories': test_preds
})
submission.to_csv("submission.csv", index=False)
submission

