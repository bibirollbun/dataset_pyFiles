import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


train = pd.concat([train, train_extra], ignore_index=True)


##Since id is not important to removing id column

train = train.drop(columns=['id'])
test = test.drop(columns=['id'])


# Checking dataset shape
print(f"Train Data Shape: {train.shape}")
print(f"Test Data Shape: {test.shape}")

train.head()


import missingno as msno

# Visualizing the missing values
msno.bar(train, color='limegreen')


# Handling missing values
# Fill missing values in numerical columns using median
numerical_cols = train.select_dtypes(include=['number']).columns
train[numerical_cols] = train[numerical_cols].fillna(train[numerical_cols].median())

# Fill missing values in categorical columns using mode
categorical_cols = train.select_dtypes(exclude=['number']).columns
train[categorical_cols] = train[categorical_cols].fillna(train[categorical_cols].mode().iloc[0])


import matplotlib.pyplot as plt
import seaborn as sns

# Histograms for numerical columns
train.hist(figsize=(12, 8), bins=30, edgecolor='black')
plt.show()


# Count plots for all categorical columns
categorical_cols = train.select_dtypes(include=['object']).columns

plt.figure(figsize=(12, 6))
for col in categorical_cols:
    sns.countplot(y=train[col], order=train[col].value_counts().index)
    plt.title(f"Distribution of {col}")
    plt.show()


##Checking price disrtibution with all categorical features using boxplots

cat_feature = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
plt.figure(figsize=(12, 12))

for i, col in enumerate(cat_feature, 1):
    plt.subplot(4, 2, i)
    sns.boxplot(x=train[col], y=train["Price"], hue=train[col], palette="Dark2")
    plt.xticks(rotation=90)
    plt.ylabel("Price")
    plt.title(f"Price Distribution by {col}")

plt.tight_layout()
plt.show()


# Select numerical columns
numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove 'Price' if it's already in numeric_cols to avoid duplication
if 'Price' in numeric_cols:
    numeric_cols.remove('Price')

# Pairplot
sns.pairplot(train[[*numeric_cols, 'Price']])
plt.show()


# Select only numerical features for correlation analysis
numerical_features = train.select_dtypes(include=['number'])

# Calculate correlation matrix
correlation_matrix = numerical_features.corr()

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="Dark2", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


X = train.drop(columns=['Price'])  # Features
y = train['Price']  # Target


# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()


# Handling Missing Values
num_imputer = SimpleImputer(strategy='median')  # Fill missing numerical values with median
cat_imputer = SimpleImputer(strategy='most_frequent')  # Fill missing categorical values with most frequent


preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', num_imputer), ('scaler', StandardScaler())]), numeric_cols),
    ('cat', Pipeline([('imputer', cat_imputer), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)
])


# Transform training data, then spliting into training and validation sets and then reducing training size for faster processing.
X_processed = preprocessor.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, test_size=0.7, random_state=42)



# Initializing models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=50, learning_rate=0.1, n_jobs=-1, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.1, n_jobs=-1)
}


# Train and evaluate models. Execution of models on sequential basis and not on parallel basis.
rmse_scores = {}

for name, model in models.items():
    model.fit(X_train_sample, y_train_sample)
    y_pred = model.predict(X_val)  
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))  
    rmse_scores[name] = rmse  # Store results
    print(f"{name} RMSE: {rmse:.4f}")


# Selecting the best model basis minimum RMSE score
best_model_name = min(rmse_scores, key=rmse_scores.get)
best_model = models[best_model_name]

print(f"\nBest Model: {best_model_name} with RMSE: {rmse_scores[best_model_name]:.4f}")


# Train best model on full dataset and then preprocess test data and finally predict the prices using the best model.
best_model.fit(X_processed, y)

X_test_processed = preprocessor.transform(test)

test_predictions = best_model.predict(X_test_processed)


# Save predictions in submission format
submission = sample_submission.copy()
submission['Price'] = test_predictions
submission.to_csv('/kaggle/working/final_submission.csv', index=False)

print("Predictions saved as final_submission.csv")




