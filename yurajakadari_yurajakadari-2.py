import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Load datasets
train_path = '/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv'  
test_path = '/kaggle/input/prediction-interval-competition-ii-house-price/test.csv'    
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Display info
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Check missing values
print("\nMissing in Train:")
print(train.isnull().sum()[train.isnull().sum() > 0])

print("\nMissing in Test:")
print(test.isnull().sum()[test.isnull().sum() > 0])

# Drop rows in train with too many missing values (optional)
train = train.dropna(thresh=train.shape[1]*0.5)

# Separate columns by type
num_cols_train = train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_train = train.select_dtypes(include=['object']).columns.tolist()

# Don't try to impute 'sale_price' in test
if 'sale_price' in num_cols_train:
    num_cols_train.remove('sale_price')

# Apply imputers
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

# Impute train
train[num_cols_train] = num_imputer.fit_transform(train[num_cols_train])
train[cat_cols_train] = cat_imputer.fit_transform(train[cat_cols_train])

# Align test columns with train
num_cols_test = [col for col in num_cols_train if col in test.columns]
cat_cols_test = [col for col in cat_cols_train if col in test.columns]

# Impute test
test[num_cols_test] = num_imputer.transform(test[num_cols_test])
test[cat_cols_test] = cat_imputer.transform(test[cat_cols_test])

# Final check
print("\nTrain missing values:", train.isnull().sum().sum())
print("Test missing values:", test.isnull().sum().sum())

# Save cleaned data (optional)
# train.to_csv("cleaned_train.csv", index=False)
# test.to_csv("cleaned_test.csv", index=False)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load train dataset
train = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')  # Update path if needed

# Convert date column to datetime
train['sale_date'] = pd.to_datetime(train['sale_date'], errors='coerce')

# Basic info
print("Train shape:", train.shape)
print("Columns:", train.columns.tolist())

# Drop id for analysis
train = train.drop(columns=['id'])

# Identify numerical and categorical features
numerical_cols = train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()


# ---------- CORRELATION HEATMAP ----------
print("\nPlotting correlation heatmap...")

plt.figure(figsize=(16, 12))
corr = train[numerical_cols].corr()
sns.heatmap(corr, cmap='coolwarm', annot=False, fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# ---------- CATEGORICAL FEATURES ----------
print("\nPlotting count plots of categorical features...")

cat_cols = len(categorical_cols)
cat_rows = (cat_cols + 2) // 3
fig, axes = plt.subplots(cat_rows, 3, figsize=(18, 5*cat_rows))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.countplot(data=train, x=col, ax=axes[i], order=train[col].value_counts().index)
    axes[i].tick_params(axis='x', rotation=90)
    axes[i].set_title(f'Count of {col}')

for i in range(len(categorical_cols), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.suptitle("Count Plots of Categorical Features", y=1.02)
plt.show()

# ---------- SALE PRICE vs CATEGORICAL ----------
important_cats = ['city', 'zoning', 'condition', 'grade', 'submarket']

print("\nBox plots of Sale Price vs Categorical Features...")

for col in important_cats:
    if col in train.columns:
        plt.figure(figsize=(14, 6))
        sns.boxplot(data=train, x=col, y='sale_price')
        plt.xticks(rotation=45)
        plt.title(f'Sale Price vs {col}')
        plt.show()

# ---------- TIME TREND ----------
if 'sale_date' in train.columns:
    train['year'] = train['sale_date'].dt.year
    yearly_price = train.groupby('year')['sale_price'].median()

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=yearly_price.index, y=yearly_price.values)
    plt.title("Median Sale Price Over Time")
    plt.xlabel("Year")
    plt.ylabel("Median Sale Price")
    plt.show()



import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load datasets
train = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')  
test = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')

# Convert date
train['sale_date'] = pd.to_datetime(train['sale_date'], errors='coerce')
test['sale_date'] = pd.to_datetime(test['sale_date'], errors='coerce')

# Drop rows with too many missing values
train = train.dropna(thresh=train.shape[1]*0.5)

# Separate column types
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = train.select_dtypes(include=['object']).columns.tolist()

if 'sale_price' in num_cols:
    num_cols.remove('sale_price')

# Impute missing values
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

train[num_cols] = num_imputer.fit_transform(train[num_cols])
train[cat_cols] = cat_imputer.fit_transform(train[cat_cols])
test[num_cols] = num_imputer.transform(test[num_cols])
test[cat_cols] = cat_imputer.transform(test[cat_cols])

# Feature Engineering
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

# Features
features = ['sqft', 'sqft_lot', 'grade', 'condition', 'beds', 'bath_full',
            'bath_3qtr', 'bath_half', 'year_built', 'latitude', 'longitude',
            'house_age', 'total_sqft', 'total_bath', 'log_sqft', 'log_lot']
categorical_features = ['city', 'zoning', 'submarket']

# Prepare final sets
X = train[features + categorical_features]
y = train['sale_price']
X_test = test[features + categorical_features]

# Preprocessing
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

# Transform data
X_transformed = preprocessor.fit_transform(X)
X_test_transformed = preprocessor.transform(X_test)

# LightGBM Quantile Model Trainer
def train_lgb_quantile_model(X, y, alpha):
    params = {
        "objective": "quantile",
        "alpha": alpha,
        "metric": "quantile",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "random_state": 42
    }
    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_data, num_boost_round=1000)
    return model

# Train quantile models
model_lower = train_lgb_quantile_model(X_transformed, y, alpha=0.05)
model_upper = train_lgb_quantile_model(X_transformed, y, alpha=0.95)

# Predict
pi_lower = model_lower.predict(X_test_transformed)
pi_upper = model_upper.predict(X_test_transformed)

# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'pi_lower': pi_lower,
    'pi_upper': pi_upper
})

submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")


