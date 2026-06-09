# Load libraries
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
warnings.filterwarnings('ignore')

# Ignore warnings from dataset
msgs = [
    'invalid value encountered in greater',
    'invalid value encountered in less'
]
for msg in msgs:
    warnings.filterwarnings('ignore', category=RuntimeWarning, message=msg)


# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv', index_col='id')


y = train['Listening_Time_minutes']
X = train.drop('Listening_Time_minutes', axis=1)


train.head()


test.head()


# Check train data types and missing balues
print("== Train: Info ==")
print(train.info())
print("\n-- Train: Missing data --")
print(train.isna().sum()[train.isna().sum() > 0])

# Check test data types and missing balues
print("\n\n== Test: Info ==")
print(test.info())
print("\n-- Test: Missing data --")
print(test.isna().sum()[test.isna().sum() > 0])



target_col = 'Listening_Time_minutes'
print(train[target_col].describe())

# 1) Target variable histogram
plt.figure(figsize=(8, 5))
sns.histplot(train[target_col], bins=30, kde=True)
plt.title('Distribution of Listening Time (minutes)')
plt.xlabel('Listening Time (minutes)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# 2) Target variable boxplot
plt.figure(figsize=(8, 4))
sns.boxplot(x=train[target_col])
plt.title(f'Distribution of {target_col}')
plt.xlabel(target_col)
plt.tight_layout()
plt.show()


numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != 'id']
print(f"4 Numeric columns: {numeric_cols}")


# 1) Correlation heatmap
train_numeric = train[numeric_cols + [target_col]]

plt.figure(figsize=(6, 4))
correlation = train_numeric.corr()[target_col].drop(target_col).sort_values(ascending=False)
print("\nCorrelation with Listening_Time_minutes:\n", correlation)

sns.barplot(x=correlation.values, y=correlation.index, palette='YlGnBu')
plt.title('Feature Correlation with Target (Listening_Time_minutes)')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(train_numeric.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix (Numeric Features)')
plt.tight_layout()
plt.show()


# 2) Histogram (Distribution)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

for i, feature in enumerate(numeric_cols):
    row = i // 2
    col = i % 2
    sns.histplot(data=train.sample(10000), x=feature, kde=True, ax=axs[row][col])
    axs[row][col].set_title(f"Distribution of {feature}")

plt.tight_layout()
plt.show()


# 3) Scatter plots between each numeric feature and target in a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
sample_data = train.sample(5000, random_state=42)

for i, col in enumerate(numeric_cols):
    row = i // 2
    col_idx = i % 2
    ax = axs[row][col_idx]
    
    sns.scatterplot(data=sample_data, x=col, y=target_col, alpha=0.6, ax=ax)
    ax.set_title(f'{col} vs {target_col}')
    ax.set_xlabel(col)
    ax.set_ylabel(target_col)
    ax.grid(True)

plt.tight_layout()
plt.show()


categorical_cols = X.select_dtypes(include='object').columns.tolist()
print(f"6 Categorical columns: {categorical_cols}")


# 1) Distribution
import math

n_cols = 3
n_rows = 2

cols_to_plot = categorical_cols[:n_cols * n_rows]

# Create subplots
fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

for i, col in enumerate(cols_to_plot):
    row = i // n_cols
    col_idx = i % n_cols
    ax = axs[row][col_idx]

    # Plot countplot for each categorical column
    sns.countplot(data=train, x=col, order=train[col].value_counts().index, ax=ax)
    ax.set_title(f'Distribution of {col}')
    ax.tick_params(axis='x', rotation=45)
    
# Adjust layout
plt.tight_layout()
plt.show()


# 2) Box plots
import math

sample_data = train.sample(5000, random_state=42)

n_cols = 3
n_rows = math.ceil(len(categorical_cols) / n_cols)

fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

for i, col in enumerate(categorical_cols):
    row = i // n_cols
    col_idx = i % n_cols
    ax = axs[row][col_idx]

    sns.boxplot(x=col, y=target_col, data=sample_data, ax=ax)
    ax.set_title(f'Listening Time by {col}')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


columns_to_drop = ['Episode_Title', 'Podcast_Name'] # drop two from categorical_features
numeric_features = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage']
categorical_features = ['Genre', 'Publication_Time', 'Publication_Day', 'Episode_Sentiment','Number_of_Ads']



# 1) Drop unhelpful columns
X = X.drop(columns=columns_to_drop) # drop from train dataset
test_simple = test.drop(columns=columns_to_drop) # drop from test dataset


# 2) Create pre-processor pipelines

# Numeric pipeline
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')), # handle missing value
    ('scaler', StandardScaler()) 
])

# Categorical pipeline (OneHot)
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')), # handle missing value
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Full preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
])


# Split data into train and validation(test) using an 80/20 ratio.
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    'LinearRegression': LinearRegression(), # Baseline
    'CatBoost': CatBoostRegressor(
        iterations=1000,  # # of trees
        depth=7,          # deeper trees
        learning_rate=0.03,
        l2_leaf_reg=5,  
        random_seed=42,
        verbose=0,
    ),
    'XGBoost': XGBRegressor(n_estimators=100, max_depth=4, subsample=0.8,
                            colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0),
    "Ridge Regression": Ridge(),
    # 'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    # "Lasso Regression": Lasso(),                  # Less precision
    # "ElasticNet Regression": ElasticNet(),        # Less precision
    # "Decision Tree": DecisionTreeRegressor(),     # Less precision, Slow
    # "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),     # Too slow
    # "KNN Regressor": KNeighborsRegressor(),       # Too slow
}


model_scores = {}
model_preds = {}

for name, regressor in models.items():
    print(f"Training {name}...")
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', regressor)
    ])

    model_pipeline.fit(X_train, y_train)

    y_val_pred = model_pipeline.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, y_val_pred))

    model_scores[name] = rmse
    model_preds[name] = model_pipeline

    print(f"{name} Validation RMSE: {rmse:.4f}")



# Select the best model
best_model_name = min(model_scores, key=model_scores.get)
best_model = model_preds[best_model_name]
print(f"Best Model: {best_model_name} (RMSE: {model_scores[best_model_name]:.4f})")

# Make final prediction
test_preds = best_model.predict(test_simple)


# Submit
submission = pd.DataFrame({
    'id': test['id'],
    'Listening_Time_minutes': test_preds
})
submission.to_csv("submission.csv", index=False)
print("\nsubmission.csv saved")
print(submission.head())


import matplotlib.pyplot as plt

# Scatter plot:
y_val_pred_best = best_model.predict(X_valid)

plt.figure(figsize=(7, 6))
plt.scatter(y_valid, y_val_pred_best, alpha=0.4)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  
plt.xlabel("Actual Listening Time")
plt.ylabel("Predicted Listening Time")
plt.title(f"{best_model_name} Prediction vs Actual")
plt.grid(True)
plt.tight_layout()
plt.show()


# Residual Distribution:
residuals = y_valid - y_val_pred_best

plt.figure(figsize=(7, 5))
sns.histplot(residuals, bins=30, kde=True)
plt.title("Distribution of Residuals (Actual - Predicted)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()


