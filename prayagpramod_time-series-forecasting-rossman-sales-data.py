# Import libraries
!pip install xgboost seaborn scikit-learn --quiet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')


# Data Loading
# Source: https://www.kaggle.com/competitions/rossmann-store-sales/data
train = pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv', parse_dates=['Date'])
store = pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')
df = pd.merge(train, store, on='Store', how='left')


# Sample 20% of stores
sample_stores = np.random.choice(df['Store'].unique(), size=int(0.2 * len(df['Store'].unique())), replace=False)
df = df[df['Store'].isin(sample_stores)]
print(f"Sampled dataset shape: {df.shape}")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

df.info()
df.head()


# Data Cleaning
df = df[df['Open'] == 1].reset_index(drop=True)
df.drop(['Open', 'PromoInterval'], axis=1, inplace=True, errors='ignore')
df = df[df['Sales'] > 0]
df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
df['CompetitionOpenMissing'] = df['CompetitionOpenSinceMonth'].isna().astype('int8')
df['CompetitionOpenSinceMonth'].fillna(df['CompetitionOpenSinceMonth'].median(), inplace=True)
df['CompetitionOpenSinceYear'].fillna(df['CompetitionOpenSinceYear'].median(), inplace=True)
df.fillna(0, inplace=True)

print(f"Cleaned dataset shape: {df.shape}")
print("\nSummary statistics:")
print(df.describe())
print("\nCategorical feature value counts:")
for col in ['StoreType', 'Assortment']:
    print(f"\n{col}:\n{df[col].value_counts()}")


# Outlier Handling
Q1 = df['Sales'].quantile(0.25)
Q3 = df['Sales'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df['Sales'] = np.where(df['Sales'] > upper_bound, upper_bound,
                       np.where(df['Sales'] < lower_bound, lower_bound, df['Sales']))


# Feature Engineering
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
df['IsHoliday'] = df['StateHoliday'].apply(lambda x: 1 if x in ['a', 'b', 'c'] else 0)
df['lag_7'] = df.groupby('Store')['Sales'].shift(7)
df['lag_7'] = df['lag_7'].fillna(df['lag_7'].mean())

# One-hot encode categorical columns
df_encoded = pd.get_dummies(df, columns=['StoreType', 'Assortment', 'StateHoliday'], drop_first=True)


#  Exploratory Data Analysis (EDA)
#  Sales distribution histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['Sales'], bins=30, kde=True)
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()


# Sales by categorical features
for col in ['StoreType', 'Assortment']:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=col, y='Sales', data=df)
    plt.title(f'Sales by {col}')
    plt.xlabel(col)
    plt.ylabel('Sales')
    plt.show()


#Plot Sales by DayOfWeek
plt.figure(figsize=(8, 5))
sns.boxplot(x='DayOfWeek', y='Sales', data=df)
plt.title('Sales by Day of Week')
# Map numeric days (1-7) to weekday names
day_names = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
plt.xticks(ticks=range(7), labels=[day_names[i+1] for i in range(7)], rotation=45)
plt.xlabel('Day of Week')
plt.ylabel('Sales')
plt.show()


#how promotions affect daily sales
plt.figure(figsize=(10, 5))
sns.boxplot(x='DayOfWeek', y='Sales', hue='Promo', data=df)
plt.title('Sales by Day of Week with Promotions')
day_names = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
plt.xticks(ticks=range(7), labels=[day_names[i+1] for i in range(7)], rotation=45)
plt.xlabel('Day of Week')
plt.ylabel('Sales')
plt.show()


#Sales Over Time
plt.figure(figsize=(12, 6))
# Compute daily average sales
daily_sales = df.groupby('Date')['Sales'].mean().reset_index()
plt.plot(daily_sales['Date'], daily_sales['Sales'], color='blue', label='Daily Average Sales')
plt.title('Average Daily Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sales')
plt.legend()
plt.grid(True)
plt.show()


# Correlation analysis
numeric_cols = df_encoded.select_dtypes(include=np.number).columns
df_numeric = df_encoded[numeric_cols]
corr_matrix = df_numeric.corr()
sales_corr = corr_matrix['Sales'].drop('Sales')
top_12_features = sales_corr.abs().sort_values(ascending=False).head(12)
top_12_df = pd.DataFrame({
    'Feature': top_12_features.index,
    'Correlation with Sales': sales_corr[top_12_features.index]
}).reset_index(drop=True)
top_12_df.index += 1
print("Top 12 Features Correlated with Sales:\n", top_12_df)

cols_to_plot = ['Sales'] + top_12_features.index.tolist()
cols_to_plot = [col for col in cols_to_plot if col in corr_matrix.columns]
filtered_corr_matrix = corr_matrix.loc[cols_to_plot, cols_to_plot]
plt.figure(figsize=(12, 8))
sns.heatmap(filtered_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, center=0, square=True)
plt.title("Correlation Matrix of Sales and Top 12 Correlated Features", pad=20)
plt.savefig('corr_matrix_top12.png', dpi=300, bbox_inches='tight')
plt.show()


# Prepare data
X = df_encoded.drop(columns=['Sales', 'Date'], errors='ignore')
y = df_encoded['Sales']
feature_names = X.columns  # Store feature names for importance plots


# Time-based split
train_size = int(len(df) * 0.6)
val_size = int(len(df) * 0.2)
X_train = X.iloc[:train_size]
X_val = X.iloc[train_size:train_size + val_size]
X_test = X.iloc[train_size + val_size:]
y_train = y.iloc[:train_size]
y_val = y.iloc[train_size:train_size + val_size]
y_test = y.iloc[train_size + val_size:]

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Initialize models
models = {
    "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1)
}

# Hyperparameter grids
param_grids = {
    "XGBoost": {
        'n_estimators': [100, 300],
        'max_depth': [4],
        'learning_rate': [0.05]
    },
      "Random Forest": {
    'n_estimators': [100],               # Keep fixed for now
    'max_depth': [4, 6],                 # Shallower trees = better generalization
    'min_samples_split': [20, 50],       # Force splits to have more samples
    'min_samples_leaf': [10, 20]         # Each leaf must have more examples
}
}



# Train and evaluate
results = {}
for name, model in models.items():
    try:
        print(f"\nTuning {name}...")
        grid_search = GridSearchCV(model, param_grids[name], cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best parameters for {name}:", grid_search.best_params_)

        # Evaluate on training set
        y_train_pred = best_model.predict(X_train)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        print(f"{name} Training Performance:")
        print(f"MAE: €{train_mae:.2f}, R²: {train_r2:.4f}, MSE: {train_mse:.2f}")

        # Evaluate on validation set
        y_val_pred = best_model.predict(X_val)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)
        print(f"{name} Validation Performance:")
        print(f"MAE: €{val_mae:.2f}, R²: {val_r2:.4f}, MSE: {val_mse:.2f}")

        # Evaluate on test set
        y_test_pred = best_model.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        results[name] = {
            "Train_MAE": train_mae,
            "Train_R2": train_r2,
            "Train_MSE": train_mse,
            "Val_MAE": val_mae,
            "Val_R2": val_r2,
            "Val_MSE": val_mse,
            "Test_MAE": test_mae,
            "Test_R2": test_r2,
            "Test_MSE": test_mse,
            "model": best_model,
            "predictions": y_test_pred,
            "val_predictions": y_val_pred
        }
        print(f"{name} Test Performance:")
        print(f"MAE: €{test_mae:.2f}, R²: {test_r2:.4f}, MSE: {test_mse:.2f}")

    except Exception as e:
        print(f"Error training {name}: {e}")
        continue


# Model Performance Summary
avg_daily_sales = df_encoded['Sales'].mean()
print(f"Average Daily Sales: €{avg_daily_sales:.2f}")

metrics_df = pd.DataFrame({
    'Model': ['XGBoost', 'Random Forest'],
    'Train_MAE': [results['XGBoost']['Train_MAE'], results['Random Forest']['Train_MAE']],
    'Val_MAE': [results['XGBoost']['Val_MAE'], results['Random Forest']['Val_MAE']],
    'Test_MAE': [results['XGBoost']['Test_MAE'], results['Random Forest']['Test_MAE']],
    'Train_R2': [results['XGBoost']['Train_R2'], results['Random Forest']['Train_R2']],
    'Val_R2': [results['XGBoost']['Val_R2'], results['Random Forest']['Val_R2']],
    'Test_R2': [results['XGBoost']['Test_R2'], results['Random Forest']['Test_R2']]
})
print("\nModel Performance Summary:")
print(metrics_df)


# MAE as percentage of average daily sales
for model in results.keys():
    test_mae = results[model]['Test_MAE']
    mae_to_avg_sales_ratio = (test_mae / avg_daily_sales) * 100
    print(f"\n{model} Test MAE as Percentage of Average Daily Sales: {mae_to_avg_sales_ratio:.2f}%")



# Bar plot for metrics
Comparison_Both = pd.DataFrame({
    'Model': list(results.keys()) * 2,
    'Type': ['Test'] * len(results) + ['Validation'] * len(results),
    'MAE': [results[name]['Test_MAE'] for name in results.keys()] + [results[name]['Val_MAE'] for name in results.keys()],
    'R2': [results[name]['Test_R2'] for name in results.keys()] + [results[name]['Val_R2'] for name in results.keys()],
    'MSE': [results[name]['Test_MSE'] for name in results.keys()] + [results[name]['Val_MSE'] for name in results.keys()]
})
metrics = ['MAE', 'R2', 'MSE']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, metric in enumerate(metrics):
    sns.barplot(x='Model', y=metric, hue='Type', data=Comparison_Both, ax=axes[idx], palette='viridis')
    axes[idx].set_title(f'{metric} Comparison (Test vs Validation)')
    axes[idx].set_xlabel('Model')
    axes[idx].set_ylabel(metric)
    if metric == 'R2':
        axes[idx].set_ylim(0, 1)
    axes[idx].legend(title='Data Set')
    for container in axes[idx].containers:
        axes[idx].bar_label(container, fmt='%.2f', fontsize=8)
plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# Actual vs Predicted Sales
plt.figure(figsize=(15, 6))
sample_indices = np.random.choice(len(y_test), size=100, replace=False)
actual = y_test.iloc[sample_indices].values
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True, sharex=True)
colors = {'XGBoost': 'blue', 'Random Forest': 'green'}
for idx, (name, res) in enumerate(results.items()):
    predicted = res["predictions"][sample_indices]
    axes[idx].scatter(actual, predicted, alpha=0.5, color=colors[name], label=f'{name} Predictions')
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    axes[idx].set_title(f'{name}: Actual vs Predicted Sales')
    axes[idx].set_xlabel('Actual Sales')
    axes[idx].set_ylabel('Predicted Sales')
    axes[idx].grid(True)
    axes[idx].legend()
    axes[idx].text(0.05, 0.95, f'MAE: €{res["Test_MAE"]:.2f}\nR²: {res["Test_R2"]:.4f}',
                   transform=axes[idx].transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()


# Feature Importance
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# XGBoost
xgb.plot_importance(results["XGBoost"]["model"], ax=axes[0], max_num_features=10)
axes[0].set_title('XGBoost Feature Importance')
# Random Forest
sorted_idx = results["Random Forest"]["model"].feature_importances_.argsort()[-10:]
axes[1].barh(feature_names[sorted_idx], results["Random Forest"]["model"].feature_importances_[sorted_idx], color='green')
axes[1].set_title('Random Forest Feature Importance')
axes[1].set_xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

