import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import plotly.express as px


df_tr   = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
df_ts = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
df_tr.head()


df_tr.info()


# missing values
print(df_tr.isnull().sum())
sns.heatmap(df_tr.isnull(),cbar=False)


# missing values by country
# Calculate missing sales data
missing_sales = df_tr[df_tr['num_sold'].isnull()].groupby('country').size()

# Visualization
plt.figure(figsize=(10, 6))
bars = plt.bar(missing_sales.index, missing_sales.values, color='salmon')
plt.title("Missing num_sold Data by Country")
plt.ylabel("Count of Missing Values")
plt.xlabel("Country")

# Adding values on the bars
for bar in bars:
    plt.annotate(f'{bar.get_height()}', 
                 (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                 ha='center', va='bottom')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Convert to datetime and extract key features
df_tr['date'] = pd.to_datetime(df_tr['date'])
df_tr['year'] = df_tr['date'].dt.year
df_tr['month'] = df_tr['date'].dt.month
df_tr['day'] = df_tr['date'].dt.day
df_tr['dayofweek'] = df_tr['date'].dt.dayofweek
df_tr['quarter'] = df_tr['date'].dt.quarter

# cyclical encoding for month and day
df_tr['month_sin'] = np.sin(2 * np.pi * df_tr['month']/12)
df_tr['month_cos'] = np.cos(2 * np.pi * df_tr['month']/12)
# remove date column
df_tr = df_tr.drop(columns=['date'])


df_tr.head()


df_tr['country'].unique()



# Visualization
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=country_sales.index, y=country_sales.values, palette='magma')
plt.title("Total Sales by Country")
plt.xlabel("Country")
plt.ylabel("Total Sales")

# Adding values on the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():,.0f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Grouping the data
country_sales = df_tr.groupby(['year', 'country'])['num_sold'].sum().reset_index()

# Set the style for the plot
sns.set(style="whitegrid")

# Create the line plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=country_sales, 
             x='year', 
             y='num_sold', 
             hue='country', 
             marker='o')

# Add titles and labels
plt.title("Sales Trend Over Time by Country", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Total Sales", fontsize=12)
plt.legend(title="Country", fontsize=10)

# Display the plot
plt.show()


country_avg_sales = df_tr.groupby('country')['num_sold'].mean()

# Plotting the simplified pie chart
plt.figure(figsize=(8, 8))
country_avg_sales.plot(
    kind='pie', 
    autopct='%1.1f%%', 
    startangle=90, 
    colors=sns.color_palette('pastel')
)
plt.title("Average Sales by Country", fontsize=14)
plt.ylabel("") 
plt.show()



df_tr['store'].unique()



total_sales_by_store = df_tr.groupby('store')['num_sold'].sum()

# Create a figure
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=total_sales_by_store.index, y=total_sales_by_store.values, palette='viridis')

plt.title("Total Sales by Store", fontsize=16)
plt.xlabel("Store Type", fontsize=12)
plt.ylabel("Total Sales", fontsize=12)
plt.xticks(rotation=45)

# Adding values on top of the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():,.0f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom')

plt.tight_layout()
plt.show()


# Assuming 'store_trend' DataFrame is already created
store_trend = df_tr.groupby(['year', 'store'])['num_sold'].sum().reset_index()

# Create a seaborn line plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=store_trend, 
             x='year', 
             y='num_sold', 
             hue='store', 
             marker='o')

# Add titles and labels
plt.title('Sales Trends Over Time by Store', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Total Sales', fontsize=12)
plt.legend(title='Store Type')

# Show the plot
plt.tight_layout()
plt.show()


# Total Sales by Store and Country
store_country_sales = df_tr.groupby(['store', 'country'])['num_sold'].sum().unstack()

plt.figure(figsize=(14, 8))
sns.heatmap(store_country_sales, annot=True, fmt=".0f", cmap="coolwarm")
plt.title("Total Sales by Store and Country", fontsize=16)
plt.xlabel("Country", fontsize=12)
plt.ylabel("Store Type", fontsize=12)
plt.show()


df_tr['product'].unique()


total_sales_by_product = df_tr.groupby('product')['num_sold'].sum()

plt.figure(figsize=(12, 6))

# Create a bar plot with a color palette
colors = plt.cm.magma(np.linspace(0, 1, len(total_sales_by_product)))
bars = total_sales_by_product.plot(kind='bar', color=colors, edgecolor='black')

plt.title("Total Sales by Product", fontsize=16)
plt.xlabel("Product", fontsize=12)
plt.ylabel("Total Sales", fontsize=12)
plt.xticks(rotation=90)

# Adding values on top of the bars
for bar in bars.patches:
    bars.annotate(f'{bar.get_height():,.0f}', 
                  (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                  ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# Heatmap for sales by product and store
total_sales_product_store = df_tr.groupby(['product', 'store'])['num_sold'].sum().unstack()
plt.figure(figsize=(10, 8))
sns.heatmap(total_sales_product_store, annot=True, fmt='.0f', cmap='YlGnBu', cbar_kws={'label': 'Total Sales'})
plt.title("Total Sales by Product and Store")
plt.xlabel("Store")
plt.ylabel("Product")
plt.show()


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GroupKFold
from lightgbm import LGBMRegressor
import optuna
from sklearn.preprocessing import LabelEncoder



# Read the data
df_tr = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
df_ts = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
# Check columns
print("Training columns:", df_tr.columns.tolist())
print("Test columns:", df_ts.columns.tolist())

# Remove missing values
df_tr = df_tr.dropna()


def preprocess_data(df, label_encoders=None, is_train=True):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter

    year_min = 2013
    year_max = 2017
    year_range = year_max - year_min + 1

    df['year_sin'] = np.sin(2 * np.pi * (df['year'] - year_min) / year_range)
    df['year_cos'] = np.cos(2 * np.pi * (df['year'] - year_min) / year_range)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

    df['group'] = (df['year'] - year_min) * 48 + df['month'] * 4 + df['day'] // 7
    df = df.drop(columns=['date'])

    categorical_cols = ['country', 'store', 'product']
    if is_train:
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        return df, label_encoders
    else:
        for col in categorical_cols:
            df[col] = label_encoders[col].transform(df[col])
        return df



# Preprocess training and test datasets
df_tr_processed, label_encoders = preprocess_data(df_tr, is_train=True)
df_ts_processed = preprocess_data(df_ts, label_encoders=label_encoders, is_train=False)

# Prepare features and target
X = df_tr_processed.drop(columns=['num_sold', 'id'])
y = df_tr_processed['num_sold']
groups = df_tr_processed['group']

# Test features
X_test = df_ts_processed.drop(columns=['id'])



def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 6, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
        # 'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'boosting_type': 'dart',
        'random_state': 42
    }

    group_kfold = GroupKFold(n_splits=5)
    rmse_scores = []

    for train_idx, val_idx in group_kfold.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, val_preds)))

    return np.mean(rmse_scores)



# Optimize hyperparameters using Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

# Display best parameters
print("Best parameters from Optuna optimization:")
print(study.best_params)


optuna.visualization.plot_optimization_history(study)


optuna.visualization.plot_param_importances(study)


optuna.visualization.plot_slice(study)


optuna.visualization.plot_parallel_coordinate(study)


optuna.visualization.plot_contour(study)


# Get the best parameters from Optuna
best_params = study.best_params

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the final model
final_model = LGBMRegressor(**best_params)
final_model.fit(X_train, y_train)


# Predictions and evaluation
val_preds = final_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
mae = mean_absolute_error(y_val, val_preds)
r2 = r2_score(y_val, val_preds)

print("\nEvaluation Metrics on Validation Set:")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (RÂ²): {r2:.4f}")


# Plot feature importance directly from the trained model
plt.figure(figsize=(12, 8))
importance = final_model.feature_importances_  # Feature importance from the model
features = X.columns  # Feature names

# Sorting features by importance
indices = np.argsort(importance)[::-1]
sorted_features = [features[i] for i in indices]

plt.barh(sorted_features, importance[indices], color='skyblue')
plt.title('Feature Importance from Model', fontsize=16)
plt.xlabel('Feature Importance', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
plt.show()


# Train on the full dataset
final_model.fit(X, y)
final_preds = final_model.predict(X_test)

# Save submission file
submission = pd.DataFrame({
    'id': df_ts['id'],
    'num_sold': final_preds
})
submission.to_csv("submission.csv", index=False)

print("\nModel training complete. Submission file saved as 'submission.csv'.")


