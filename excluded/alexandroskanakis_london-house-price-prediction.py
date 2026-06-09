%pip install xgboost


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyRegressor

from xgboost import XGBRegressor as xgb

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv('/kaggle/input/london-house-price-prediction-advanced-techniques/train.csv')
test_df = pd.read_csv('/kaggle/input/london-house-price-prediction-advanced-techniques/test.csv')


def data_info(df, name):
    """
    Function to display basic information about the DataFrame including:
    - Shape (rows and columns)
    - Column names
    - Detailed info (including dtypes and non-null counts)
    - First 5 rows
    - Descriptive statistics for all columns
    - Count of missing values per column
    - Count of duplicated rows
    """
    print(name)
    print(f"DataFrame shape:\n{df.shape}\n")  # e.g., (266325, 17)
    print(f"DataFrame columns:\n{df.columns.tolist()}\n")
    print(f"DataFrame info:\n{df.info()}\n")
    print(f"DataFrame head:\n{df.head()}\n")
    print(f"DataFrame description:\n{df.describe()}\n") # Summary for numeric columns
    print(f"DataFrame description:\n{df.describe(include=['object'])}\n") # Count of unique values in categorical variables
    print(f"DataFrame null values:\n{df.isnull().sum()}\n")  # Check for missing data
    print(f"DataFrame duplicates:\n{df.duplicated().sum()}\n")  # Check for duplicate rows


data_info(train_df, "Training Dataset")

'''
Training DataFrame shape:
(266325, 17)

Training DataFrame null values:
bathrooms              48479
bedrooms               24843
floorAreaSqM           13806
livingRooms            37040
tenure                  5721
propertyType             508
currentEnergyRating    56814

Missing Values in Rows:
96603
'''

print(f"\nRows with Missing Values: {train_df.isna().any(axis=1).sum()}")


data_info(test_df, "Testing Dataset")

'''
Testing DataFrame shape:
(16547, 16) -1 Price

Testing DataFrame null values:
bathrooms              2624
bedrooms               1375
floorAreaSqM           2006
livingRooms            2095
tenure                  590
propertyType            167
currentEnergyRating    1497

Missing Values in Rows:
5100
'''

print(f"\nRows with Missing Values: {test_df.isna().any(axis=1).sum()}")


summary_cols = [
    'bathrooms', 'bedrooms', 'livingRooms',
    'tenure', 'propertyType', 'currentEnergyRating',
    'sale_month', 'sale_year']

for col in summary_cols:
    summary_table = train_df.groupby(col)['price'].agg(
        mean='mean', median='median', max='max', min='min'
    ).round(2)
    print(f"\nPrice Summary by {col.capitalize()}:\n")
    print(summary_table.reset_index().to_string(index=False))
    print("-" * 50)


categorical_cols = [
    'bathrooms', 'bedrooms', 'livingRooms',
    'tenure', 'propertyType', 'currentEnergyRating']

for col in categorical_cols:
    print(f"\nCategory Proportion for {col.capitalize()}:")
    print(train_df[col].value_counts(normalize=True).round(4))
    print("-" * 50)


numeric_cols = [
    'latitude', 'longitude',
    'floorAreaSqM', 'bathrooms', 'bedrooms', 'livingRooms',
    'sale_month', 'sale_year',
    'price']

train_df[numeric_cols].hist(bins=30, figsize=(12, 6))
plt.tight_layout()
plt.suptitle('Distributions of Numeric Features in Training Data', fontsize=16, y=1.02)
plt.show()


test_df[[
    'latitude', 'longitude',
    'floorAreaSqM', 'bathrooms', 'bedrooms', 'livingRooms',
    'sale_month', 'sale_year']].hist(bins=30, figsize=(12, 6))
plt.tight_layout()
plt.suptitle('Distributions of Numeric Features in Testing Data', fontsize=16, y=1.02)
plt.show()


cols = 3
n = len(categorical_cols)
rows = n // cols + int(n % cols != 0)

fig, axes = plt.subplots(rows, cols, figsize=(24, 8))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    counts = train_df[col].value_counts(normalize=True)
    ax = axes[i]
    bars = counts.plot(kind='bar', ax=ax, color=["#001191FF", "#001191CA", "#001191A3", "#00119168",
                                                 "#0011916A", "#0011913C", "#0011912B", "#0011913E",
                                                 "#0011911D", "#00119100"], width=0.7)
    ax.set_title(f"Training: {col.capitalize()} Frequency")
    ax.set_ylabel('Proportion')
    ax.set_xlabel(col)
    plt.setp(ax.get_xticklabels(), rotation=90)
    for p in bars.patches:
        percent = f'{100 * p.get_height():.2f}%'
        ax.annotate(percent, (p.get_x() + p.get_width()/2, p.get_height()), ha='center', va='bottom', fontsize=8)

for j in range(len(categorical_cols), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('Category Frequency Distribution - Training Set', size=20)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(rows, cols, figsize=(24, 8))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    counts = test_df[col].value_counts(normalize=True)
    ax = axes[i]
    bars = counts.plot(kind='bar', ax=ax, color=["#001191FF", "#001191CA", "#001191A3", "#00119168",
                                                 "#0011916A", "#0011913C", "#0011912B", "#0011913E",
                                                 "#0011911D", "#00119100"], width=0.7)
    ax.set_title(f"Testing: {col.capitalize()} Frequency")
    ax.set_ylabel('Proportion')
    ax.set_xlabel(col)
    plt.setp(ax.get_xticklabels(), rotation=90)
    for p in bars.patches:
        percent = f'{100 * p.get_height():.2f}%'
        ax.annotate(percent, (p.get_x() + p.get_width()/2, p.get_height()), ha='center', va='bottom', fontsize=8)

for j in range(len(categorical_cols), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('Category Frequency Distribution - Testing Set', size=20)
plt.tight_layout()
plt.show()



corr = train_df[numeric_cols].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix: Training Data')
plt.show()

target_corr = corr['price'].drop('price').sort_values(key=np.abs, ascending=False)
print("Top 10 features most correlated with price:\n", target_corr.head(10))



corr = test_df[
    ['latitude', 'longitude',
     'floorAreaSqM', 'bathrooms', 'bedrooms', 'livingRooms',
     'sale_month', 'sale_year']].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix: Testing Data')
plt.show()



# Add flag before merging datasets
train_df['is_train'] = 1
test_df['is_train'] = 0

print(f"Shape of Training Dataset: {train_df.shape}")
print(f"Shape of Test Dataset: {test_df.shape}")


house_df = pd.concat([train_df, test_df], axis=0)


### Fill missing categorical with "Unknown"
cat_fill_cols = ["tenure", "propertyType", "currentEnergyRating"]
for col in cat_fill_cols:
    house_df[col] = house_df[col].fillna("Unknown")

### Fill missing numerical with 0
numerical_list = ["bathrooms", "bedrooms", "floorAreaSqM", "livingRooms"]
for col in numerical_list:
    house_df[col] = house_df[col].fillna(0)


house_df[["street", "city", "postcode"]] = house_df["fullAddress"].str.rsplit(", ", n=2, expand=True)
house_df = house_df.drop(columns="fullAddress", axis=1)


if house_df["country"].nunique() == 1:
    house_df = house_df.drop(columns="country", axis=1)


house_df[['price', 'floorAreaSqM']] = house_df[['price', 'floorAreaSqM']].clip(lower=0)
house_df[['price', 'floorAreaSqM']] = np.log1p(house_df[['price', 'floorAreaSqM']])


house_df['sale_date'] = pd.to_datetime(house_df['sale_year'].astype(str) + '-' + house_df['sale_month'].astype(str) + '-01')
house_df['days_since_first_sale'] = (house_df['sale_date'] - house_df['sale_date'].min()).dt.days
house_df['sale_quarter'] = house_df['sale_date'].dt.quarter
house_df['sale_month_sin'] = np.sin(2 * np.pi * house_df['sale_month'] / 12)
house_df['sale_month_cos'] = np.cos(2 * np.pi * house_df['sale_month'] / 12)


house_df['total_rooms'] = house_df['bedrooms'] + house_df['bathrooms'] + house_df['livingRooms']
house_df['room_density'] = house_df['floorAreaSqM'] / (house_df['total_rooms'] + 1)


for col in [
    "street", "city", "postcode", "outcode", "tenure",
    "propertyType", "currentEnergyRating"]:
    freq = house_df[col].value_counts(normalize=True)
    house_df[col + "_freq"] = house_df[col].map(freq)


house_df["outcode_encoded"] = house_df["outcode"].astype("category").cat.codes


cat_cols = ["tenure", "propertyType", "currentEnergyRating", "outcode", "city"]
house_df = pd.get_dummies(house_df, columns=cat_cols, drop_first=True)


house_df = house_df.drop(columns=['street','postcode'], axis=1, errors='ignore')


house_df.columns = house_df.columns.str.replace(' ', '_')


train_df = house_df[house_df['is_train'] == 1].drop(columns='is_train').reset_index(drop=True)
test_df = house_df[house_df['is_train'] == 0].drop(columns=['is_train','price']).reset_index(drop=True)

print(f"Shape of Training Dataset: {train_df.shape}")
print(f"Shape of Test Dataset: {test_df.shape}")


geo_features = ['latitude', 'longitude']


scaler = StandardScaler()
X_geo_train_scaled = pd.DataFrame(scaler.fit_transform(train_df[geo_features]), columns=geo_features, index=train_df.index)
X_geo_test_scaled = pd.DataFrame(scaler.transform(test_df[geo_features]), columns=geo_features, index=test_df.index)


inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeans.fit(X_geo_train_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), inertia, '-o')
plt.xlabel('Number of clusters k')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.xticks(range(1,11))
plt.grid(True)
plt.show()


kmeans_geo = KMeans(n_clusters=4, n_init='auto', random_state=42)
train_df['geo_cluster'] = kmeans_geo.fit_predict(X_geo_train_scaled)
test_df['geo_cluster']  = kmeans_geo.predict(X_geo_test_scaled)


kmeans_geo = KMeans(n_clusters=4, n_init='auto', random_state=42)
train_df['geo_cluster'] = kmeans_geo.fit_predict(X_geo_train_scaled)
test_df['geo_cluster']  = kmeans_geo.predict(X_geo_test_scaled)


cluster_stats = train_df.groupby('geo_cluster').agg(
    mean_price_geo_cluster=('price', 'mean'),
    median_price_geo_cluster=('price', 'median'),
).reset_index()

train_df = train_df.merge(cluster_stats, on='geo_cluster', how='left')
test_df  = test_df.merge(cluster_stats, on='geo_cluster', how='left')


features = train_df.drop(columns=['ID','price','sale_date']).columns.tolist()
X = train_df[features]
y = train_df['price']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


def evaluate_model(model, X, Y):
    y_pred = model.predict(X)
    y_pred = np.exp(y_pred)     # inverse log-transform
    y_val = np.exp(Y)
    return {
        "R^2 Score": r2_score(y_val, y_pred),
        "Mean Absolute Error": mean_absolute_error(y_val, y_pred),
        "Mean Squared Error": mean_squared_error(y_val, y_pred),
        "Root Mean Squared Error": np.sqrt(mean_squared_error(y_val, y_pred))
    }


models = {
    'Mean Baseline': DummyRegressor(strategy= "mean"),
    'Median Baseline': DummyRegressor(strategy= "median"),
    'Quantile Baseline': DummyRegressor(strategy= "quantile", quantile=0.75),
    'Constant Baseline': DummyRegressor(strategy= "constant", constant=0)
}

results = {}

### Train + Evaluate Baselines
for name, model in models.items():
    model.fit(X_train, y_train)
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    results[name] = {"Train": train_metrics, "Validation": val_metrics}
    
    print(f"Model: {name}")
    print("Training set evaluation:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("-" * 50)
    print("Validation set evaluation:")
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("=" * 50, "\n")


models = {
    "XGBoost Regression": xgb(
        n_estimators=1500,
        max_depth=10,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        gamma=0.05,
        min_child_weight=6,
        reg_alpha=0.5,
        reg_lambda=5,
        objective='reg:squarederror',
        random_state=42,
        tree_method='hist',
        device="cuda"
    )
}

results = {}

### Train + Evaluate XGBoost
for name, model in models.items():
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    results[name] = {"Train": train_metrics, "Validation": val_metrics}
    
    print(f"Model: {name}")
    print("Training set evaluation:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("-" * 50)
    print("Validation set evaluation:")
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("=" * 50, "\n")


best_model_name = min(results, key=lambda name: results[name]['Validation']['Mean Absolute Error'])
best_model = models[best_model_name]
print(f"Best model selected: {best_model_name}")


def create_submission(best_model, test_df, features, id_col='ID', filename='London_Price_Predictions.csv'):
    """ Create submission file from best model predictions. """
    # Copy test set
    submission_df = test_df.copy()
    
    # Predictions (inverse log transform applied)
    submission_df['price'] = best_model.predict(submission_df[features])
    submission_df['price'] = np.exp(submission_df['price'])
    
    # Keep only ID + Price
    London_Price_Predictions = submission_df[[id_col, 'price']]
    
    # Save CSV
    London_Price_Predictions.to_csv(os.path.join(filename), index=False)
    
    print(f"Submission file saved as '{filename}'")

# Generate Final Submission
create_submission(best_model, test_df, features, id_col='ID', filename='London_Price_Predictions.csv')

