import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# File paths
train_path = '/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv'
test_path = '/kaggle/input/prediction-interval-competition-ii-house-price/test.csv'

# Load datasets
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Basic info
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Drop rows in train with >50% missing values
train = train.dropna(thresh=train.shape[1] * 0.5)

# Identify numerical and categorical columns
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = train.select_dtypes(include=['object']).columns.tolist()

# Remove target from numerical columns
if 'sale_price' in num_cols:
    num_cols.remove('sale_price')

# Initialize imputers
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

# Impute train set
train[num_cols] = num_imputer.fit_transform(train[num_cols])
train[cat_cols] = cat_imputer.fit_transform(train[cat_cols])

# Match test columns with train columns
num_cols_test = [col for col in num_cols if col in test.columns]
cat_cols_test = [col for col in cat_cols if col in test.columns]

# Impute test set using train imputers
test[num_cols_test] = num_imputer.transform(test[num_cols_test])
test[cat_cols_test] = cat_imputer.transform(test[cat_cols_test])

# Final missing value check
print("\nRemaining missing values:")
print(f"Train: {train.isnull().sum().sum()}, Test: {test.isnull().sum().sum()}")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
train = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')

# Convert 'sale_date' to datetime
train['sale_date'] = pd.to_datetime(train['sale_date'], errors='coerce')

# Drop ID for analysis
train.drop(columns=['id'], inplace=True)

# Identify column types
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = train.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical features: {len(num_cols)} | Categorical features: {len(cat_cols)}")

# ========================
# 1. Correlation Heatmap
# ========================
print("\nğŸ”� Plotting correlation heatmap...")

plt.figure(figsize=(16, 12))
corr_matrix = train[num_cols].corr()
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# ===============================
# 2. Count Plots for Categorical
# ===============================
print("\nğŸ“Š Plotting count plots for categorical features...")

n_cols = 3
n_rows = (len(cat_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    sns.countplot(data=train, x=col, ax=axes[i], order=train[col].value_counts().index)
    axes[i].tick_params(axis='x', rotation=90)
    axes[i].set_title(f'{col}')

# Remove empty plots
for j in range(len(cat_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle("Categorical Feature Distributions", y=1.02)
plt.show()

# ===========================================
# 3. Box Plots: Sale Price vs Categorical Var
# ===========================================
print("\nğŸ“¦ Box plots: Sale Price vs Important Categorical Features...")

important_cats = ['city', 'zoning', 'condition', 'grade', 'submarket']
for col in important_cats:
    if col in train.columns:
        plt.figure(figsize=(14, 6))
        sns.boxplot(data=train, x=col, y='sale_price')
        plt.xticks(rotation=45)
        plt.title(f'Sale Price vs {col}')
        plt.tight_layout()
        plt.show()

# ===============================
# 4. Trend: Sale Price Over Time
# ===============================
print("\nğŸ“ˆ Plotting median sale price trend over time...")

if 'sale_date' in train.columns:
    train['year'] = train['sale_date'].dt.year
    yearly_price = train.groupby('year')['sale_price'].median().dropna()

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=yearly_price.index, y=yearly_price.values, marker='o')
    plt.title("Median Sale Price Over Years")
    plt.xlabel("Year")
    plt.ylabel("Median Sale Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# ======================
# 1. Load and Parse Data
# ======================
train = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')
test = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')

train['sale_date'] = pd.to_datetime(train['sale_date'], errors='coerce')
test['sale_date'] = pd.to_datetime(test['sale_date'], errors='coerce')

# Drop rows with too many missing values
train.dropna(thresh=train.shape[1] * 0.5, inplace=True)

# Identify column types
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = train.select_dtypes(include=['object']).columns.tolist()
if 'sale_price' in num_cols:
    num_cols.remove('sale_price')

# =====================
# 2. Data Imputation
# =====================
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

train[num_cols] = num_imputer.fit_transform(train[num_cols])
train[cat_cols] = cat_imputer.fit_transform(train[cat_cols])

test[num_cols] = num_imputer.transform(test[num_cols])
test[cat_cols] = cat_imputer.transform(test[cat_cols])

# ========================
# 3. Feature Engineering
# ========================
def feature_engineering(df):
    df = df.copy()
    df['house_age'] = pd.to_datetime(df['sale_date']).dt.year - df['year_built']
    df['total_sqft'] = df['sqft'] + df['sqft_fbsmt'] + df['gara_sqft']
    df['total_bath'] = df['bath_full'] + 0.75 * df['bath_3qtr'] + 0.5 * df['bath_half']
    df['log_sqft'] = np.log1p(df['sqft'])
    df['log_lot'] = np.log1p(df['sqft_lot'])
    return df

train = feature_engineering(train)
test = feature_engineering(test)

# =====================
# 4. Feature Selection
# =====================
features = ['sqft', 'sqft_lot', 'grade', 'condition', 'beds', 'bath_full',
            'bath_3qtr', 'bath_half', 'year_built', 'latitude', 'longitude',
            'house_age', 'total_sqft', 'total_bath', 'log_sqft', 'log_lot']
categorical_features = ['city', 'zoning', 'submarket']

X = train[features + categorical_features]
y = train['sale_price']
X_test = test[features + categorical_features]

# =============================
# 5. Preprocessing & Pipelines
# =============================
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, features),
    ('cat', categorical_transformer, categorical_features)
])

# =====================
# 6. Quantile Models
# =====================
def train_quantile_model(alpha):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            loss='quantile',
            alpha=alpha,
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ))
    ])
    pipeline.fit(X, y)
    return pipeline

# Train lower and upper quantile models
model_lower = train_quantile_model(alpha=0.05)
model_upper = train_quantile_model(alpha=0.95)

# Predict on test set
pi_lower = model_lower.predict(X_test)
pi_upper = model_upper.predict(X_test)

# =====================
# 7. Save Submission
# =====================
submission = pd.DataFrame({
    'id': test['id'],
    'pi_lower': pi_lower,
    'pi_upper': pi_upper
})
submission.to_csv('submission.csv', index=False)
print("âœ… Quantile interval submission file saved as 'submission.csv'")


