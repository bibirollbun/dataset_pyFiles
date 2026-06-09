# Import required liraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import stats
from datetime import datetime, date, timedelta

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DateType
from pyspark.sql.types import IntegerType

from pyspark.sql.functions import col, when, avg, sum, count, countDistinct
from pyspark.sql.functions import expr, corr, desc
from pyspark.sql.functions import dayofweek, month, lag
from pyspark.sql.functions import sin, cos, lit
from pyspark.sql.functions import year, when, dayofmonth

from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer


# Create Spark Session
spark = SparkSession.builder.appName('StickerSale').getOrCreate()
spark


df_train = spark.read.csv('/kaggle/input/kaggle-sticker-sale-competition-train',
                          header = True, inferSchema = True)
df_train.show(5)


df_train.printSchema()


print(f"Dataset Shape: {df_train.count()}, {len(df_train.columns)}")


df_train.describe('num_sold').show()


# Count missing values in each column
df_train.select([count(when(col(c).isNull(), 1)).alias(c) for c in df_train.columns]).show()


# Check unique categories
df_train.select(countDistinct('country').alias('Country #'),
                countDistinct('store').alias(' Unique Store #'),
                countDistinct('product').alias('Unique Product #')).show()


df_train.groupby('country').count().show(),
df_train.groupby('store').count().show(),
df_train.groupby('product').count().show()


# Parse date into columns
df_train = df_train.withColumn('year', year('date'))
df_train = df_train.withColumn('month', month('date'))
df_train = df_train.withColumn('weekday', dayofweek('date'))
df_train = df_train.withColumn('is_weekend', when((col('weekday') == 6) | (col('weekday') == 7), 1).otherwise(0))
df_train.show(10)


df_train.groupby('year').count().show(),
df_train.groupby('month').count().show(),
df_train.groupby('is_weekend').count().show()


# Define the holiday dates
holiday_dates = {
    'Singapore': [('01-01', 'New Year'), ('02-10', 'Chinese New Year'), ('11-11', 'Singles’ Day'), ('12-25', 'Christmas')],
    'Finland': [('12-25', 'Christmas'), ('11-24', 'Black Friday'), ('12-06', 'Independence Day')],
    'Italy': [('12-25', 'Christmas'), ('04-09', 'Easter'), ('01-06', 'Epiphany')],
    'Norway': [('12-25', 'Christmas'), ('11-24', 'Black Friday'), ('05-17', 'Constitution Day')],
    'Canada': [('12-25', 'Christmas'), ('12-26', 'Boxing Day'), ('10-09', 'Thanksgiving')],
    'Kenya': [('12-25', 'Christmas'), ('12-12', 'Jamhuri Day'), ('11-24', 'Black Friday')]
}

# Create a function to determine if a date is within the holiday interval
def is_holiday(country, date_value):
    # Ensure date_value is a string
    if isinstance(date_value, date):  # Check if it's a date object
        date_value = date_value.strftime('%Y-%m-%d')

    # Parse holidays and date
    country_holidays = [h for h in holiday_dates.get(country, [])]
    date_obj = datetime.strptime(date_value, '%Y-%m-%d')

    for month_day, _ in country_holidays:
        holiday_date = datetime.strptime(f"{date_obj.year}-{month_day}", '%Y-%m-%d')
        if holiday_date - timedelta(days=7) <= date_obj <= holiday_date:
            return 1
    return 0

# Register the function as a UDF
is_holiday_udf = F.udf(is_holiday, IntegerType())

# Add the is_holiday column to the DataFrame
df_train_with_holiday = df_train.withColumn(
    'is_holiday',
    is_holiday_udf(F.col('country'), F.col('date')))

# Show the updated DataFrame
df_train_with_holiday.show()


df_train_with_holiday.groupBy('country', 'is_holiday').count().show()


# Verify holiday for Singapore
df_train_with_holiday.filter(
    (F.col('country') == 'Singapore') & (F.col('date') == '2010-12-25')).show(5)

# Verify a non-holiday date
df_train_with_holiday.filter(
    (F.col('country') == 'Singapore') & (F.col('date') == '2010-12-10')).show(5)


# Impute missing values in num_column

# Calculate the mean of `num_sold` grouped by `country` and `store`
mean_values = df_train_with_holiday.groupBy('country', 'store').agg(
    F.mean('num_sold').alias('mean_num_sold'))

# Join the mean values back to the original DataFrame
df_with_means = df_train_with_holiday.join(
    mean_values,
    on = ['country', 'store'],
    how = 'left')

# Impute missing values in `num_sold` with the calculated mean
df_imputed = df_with_means.withColumn(
    'num_sold',
    F.when(F.col('num_sold').isNull(), F.col('mean_num_sold')).otherwise(F.col('num_sold'))
).drop('mean_num_sold')  # Drop the intermediate column since not needed

df_imputed.show(5)


# Verify the results --> should show no rows with NULL
df_imputed.filter(F.col('num_sold').isNull()).show()


print(f"Dataset Shape: {df_imputed.count()}, {len(df_imputed.columns)}")


df_imputed.dtypes


# Separate plots
fig, axes = plt.subplots(1, 2, figsize = (8, 4))

# Weekend Impact
weekend_sales = df_imputed.groupBy('is_weekend').sum('num_sold').collect()
labels = ['Not Weekend', 'Weekend']
sales = [row['sum(num_sold)'] for row in weekend_sales]
axes[0].bar(labels, sales, color = ['indianred', 'darkcyan'], width = 0.4)
axes[0].set_xlabel('Weekend Status')
axes[0].set_ylabel('Total Sales Volume')
axes[0].set_title('Impact of Weekends on Sales')

# Holiday Impact
holiday_sales = df_imputed.groupBy('is_holiday').sum('num_sold').collect()
labels = ['No Holiday', 'Holiday']
sales = [row['sum(num_sold)'] for row in holiday_sales]
axes[1].bar(labels, sales, color = ['red', 'royalblue'], width = 0.4)
axes[1].set_xlabel('Holiday Status')
axes[1].set_ylabel('Total Sales Volume')
axes[1].set_title('Impact of Holidays on Sales')

plt.tight_layout()
sns.despine()
plt.show()


# Separate plots
fig, axes = plt.subplots(1, 3, figsize = (12, 4))

# Sales by Day of Week
sales_by_day = df_imputed.groupBy('weekday').sum('num_sold').collect()
days = [row['weekday'] for row in sales_by_day]
sales = [row['sum(num_sold)'] for row in sales_by_day]
axes[0].bar(days, sales, color = 'seagreen')
axes[0].set_xlabel('Day of Week')
axes[0].set_ylabel('Total Sales Volume')
axes[0].set_title('Sales Volume by Day of Week')

# Sales by Month
sales_by_month = df_imputed.groupBy('month').sum('num_sold').collect()
months = [row['month'] for row in sales_by_month]
sales = [row['sum(num_sold)'] for row in sales_by_month]
axes[1].bar(months, sales, color = 'orchid')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Total Sales Volume')
axes[1].set_title('Sales Volume by Month')

# Sales by Year
sales_by_year = df_imputed.groupBy('year').sum('num_sold').collect()
years = [row['year'] for row in sales_by_year]
sales = [row['sum(num_sold)'] for row in sales_by_year]
axes[2].bar(years, sales, color = 'goldenrod')
axes[2].set_xlabel('Year')
axes[2].set_ylabel('Total Sales Volume')
axes[2].set_title('Sales Volume by Year')

plt.tight_layout()
sns.despine()
plt.show()


# Separate plots
fig, axes = plt.subplots(1, 3, figsize = (11, 5))

# Sales by Product
sales_by_product = df_imputed.groupBy('product').sum('num_sold').collect()
products = [row['product'] for row in sales_by_product]
sales = [row['sum(num_sold)'] for row in sales_by_product]
axes[0].bar(products, sales, color = 'skyblue')
axes[0].set_xlabel('Product')
axes[0].set_ylabel('Total Sales Volume')
axes[0].set_title('Sales Volume by Product')
axes[0].set_xticks(np.arange(len(products)))
axes[0].set_xticklabels(products, rotation = 45)

# Sales by Store
sales_by_store = df_imputed.groupBy('store').sum('num_sold').collect()
stores = [row['store'] for row in sales_by_store]
sales = [row['sum(num_sold)'] for row in sales_by_store]
axes[1].bar(stores, sales, color = 'hotpink')
axes[1].set_xlabel('Store')
axes[1].set_ylabel('Total Sales Volume')
axes[1].set_title('Sales Volume by Store')
axes[1].set_xticks(np.arange(len(stores)))
axes[1].set_xticklabels(stores, rotation = 45)

# Sales by Country
sales_by_country = df_imputed.groupBy('country').sum('num_sold').collect()
countries = [row['country'] for row in sales_by_country]
sales = [row['sum(num_sold)'] for row in sales_by_country]
axes[2].bar(countries, sales, color = 'limegreen')
axes[2].set_xlabel('Country')
axes[2].set_ylabel('Total Sales Volume')
axes[2].set_title('Sales Volume by Country')
axes[2].set_xticks(np.arange(len(countries)))
axes[2].set_xticklabels(countries, rotation = 45)

plt.tight_layout()
sns.despine()
plt.show()


# Sales Trends Over Time
plt.figure(figsize = (12, 4))
sales_trend = df_imputed.groupby('date').sum('num_sold').orderBy('date').collect()
dates = [row['date'] for row in sales_trend]
sales = [row['sum(num_sold)'] for row in sales_trend]
plt.plot(dates, sales, linestyle = '-', color = 'c')
plt.title('Sales Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Sales Volume')
plt.grid(True)
sns.despine()
plt.show()


# Category Columns Indexing
indexer_store = StringIndexer(inputCol = 'store', outputCol = 'store_index')
indexer_country = StringIndexer(inputCol = 'country', outputCol = 'country_index')
indexer_product = StringIndexer(inputCol = 'product', outputCol = 'product_index')
df_imputed = indexer_store.fit(df_imputed).transform(df_imputed)
df_imputed = indexer_country.fit(df_imputed).transform(df_imputed)
df_imputed = indexer_product.fit(df_imputed).transform(df_imputed)


# Create new column as 'holiday_store_interaction'
df_imputed = df_imputed.withColumn('holiday_store_interaction', col('is_holiday') * col('store_index'))
df_imputed.show(5)


# Cyclical Encoding
df_imputed = df_imputed.withColumn("day_sin", sin(col("weekday") * (2 * 3.141592653589793 / lit(31))))
df_imputed = df_imputed.withColumn("day_cos", cos(col("weekday") * (2 * 3.141592653589793 / lit(31))))
df_imputed = df_imputed.withColumn("month_sin", sin(col("month") * (2 * 3.141592653589793 / lit(12))))
df_imputed = df_imputed.withColumn("month_cos", cos(col("month") * (2 * 3.141592653589793 / lit(12))))
df_imputed = df_imputed.withColumn("year_sin", sin(col("year") * (2 * 3.141592653589793 / lit(1))))
df_imputed = df_imputed.withColumn("year_cos", cos(col("year") * (2 * 3.141592653589793 / lit(1))))

df_imputed.show()


# Convert the data from PySpark frame to Pandas DF
df = df_imputed.toPandas()
df['date'] = pd.to_datetime(df['date'])
df.info()


# Plot with corrected estimator
plt.figure(figsize = (10, 6))
sns.barplot(data = df, x = 'country', y = 'num_sold',
            hue = 'product', estimator = np.sum, palette = 'inferno')
plt.title('Sales Distribution by Country and Product')
plt.xlabel('Country')
plt.ylabel('Total Sales Volume')
plt.xticks(rotation=45)
plt.legend(title = 'Product', bbox_to_anchor = (1, 1), loc = 'upper left')

sns.despine()
plt.tight_layout()
plt.show()


# Plot with corrected estimator
plt.figure(figsize = (10, 6))
sns.barplot(data = df, x = 'country', y = 'num_sold',
            hue = 'year', estimator = np.sum, palette = 'viridis')
plt.title('Sales Distribution by Country and Year')
plt.xlabel('Country')
plt.ylabel('Total Sales Volume')
plt.xticks(rotation = 45)
plt.legend(title = 'Year', bbox_to_anchor = (1, 1), loc = 'upper left')

sns.despine()
plt.tight_layout()
plt.show()


# Heatmap of Sales Volume by Day of the Week and Month
heatmap_data = df.groupby(['weekday', 'month'])['num_sold'].sum().unstack()
plt.figure(figsize = (10, 5))
sns.heatmap(heatmap_data, annot = False, cmap = 'YlGnBu',
            cbar_kws = {'label': 'Sales Volume'})
plt.title('Heatmap of Sales Volume by Day of the Week and Month')
plt.xlabel('Month')
plt.ylabel('Day of Week')
plt.show()


# Correlation Heatmap
plt.figure(figsize = (10, 6))
correlation_matrix = df[['num_sold', 'year', 'month', 'weekday', 'is_weekend',
                         'is_holiday', 'holiday_store_interaction',
                         'store_index', 'country_index', 'product_index',
                         'day_sin', 'day_cos', 'month_sin', 'month_cos',
                         'year_sin', 'year_cos']].corr()
sns.heatmap(correlation_matrix, annot = True, fmt = '.2f',
            cmap = 'coolwarm', cbar_kws = {'label': 'Correlation'})
plt.title('Correlation Matrix of Numerical Features')
plt.show()


pip install optuna


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb
import optuna


# Define features and target variable
features = ['store_index', 'product_index', 'is_holiday', 'weekday', 'month',
            'year', 'is_weekend', 'day_sin', 'day_cos', 'month_sin',
            'month_cos', 'year_sin', 'year_cos']
target = 'num_sold'

# Step 1: Split data by country for separate modeling
countries = df['country'].unique()  # Extract unique country names

# Store results for each country
results = {}

# Loop through each country to train a separate model
for country in countries:
    print(f"Processing country: {country}")

    # Filter data for the specific country
    country_data = df[df['country'] == country]
    X = country_data[features]
    y = country_data[target]

    # Split the country-specific data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.2,
                                                        random_state = 42)

    # Step 2: Define an Optuna objective function for hyperparameter tuning
    def objective(trial):
        # Define the hyperparameter search space
        param = {
            'objective': 'reg:squarederror',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1400),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': 42}

        # Train an XGBoost model with the trial parameters
        model = xgb.XGBRegressor(**param)
        model.fit(X_train, y_train)

        # Evaluate using the MAPE on the test set
        y_pred = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, y_pred) # Calculate MAPE
        return mape

    # Step 3: Run Optuna optimization
    print("Starting Optuna optimization...")
    study = optuna.create_study(direction = 'minimize')  # Minimize MAPE
    study.optimize(objective, n_trials = 50, show_progress_bar = True)

    # Best hyperparameters
    best_params = study.best_params
    print(f"Best params for {country}: {best_params}")

    # Step 4: Train the final model using the best parameters
    print("Training final model with best parameters...")
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_train, y_train)

    # Step 5: Make predictions and evaluate the final model
    y_pred = final_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Evaluation for {country} - MSE: {mse}, RMSE: {rmse}, MAE: {mae}, MAPE: {mape}, R2: {r2}")

    # Step 6: Store results for each country
    results[country] = {'model': final_model,
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'mape': mape,
                        'r2': r2,
                        'best_params': best_params}

# Step 7: Summarize results for all countries
print("\nFinal Results:")
for country, metrics in results.items():
    print(f"{country} - MSE: {metrics['mse']}, RMSE: {metrics['rmse']}, MAE: {metrics['mae']}, MAPE: {metrics['mape']}, R2: {metrics['r2']}")


import pickle
# Save model as dictionary
# 'results' contains the trained models for each country
file_name = "xgb_models_dict.pkl"

# Extract only the models from results and store them in a dictionary
models_dict = {country: results[country]['model'] for country in results}

# Save the dictionary of models
with open(file_name, "wb") as f:
    pickle.dump(models_dict, f)

print("✅ Models saved as dictionary!")


# Separate plots
fig, axes = plt.subplots(3, 2, figsize = (12, 15))

# Access feature importances from the trained model for the last processed country
canada_importances = results['Canada']['model'].feature_importances_
finland_importances = results['Finland']['model'].feature_importances_
italy_importances = results['Italy']['model'].feature_importances_
kenya_importances = results['Kenya']['model'].feature_importances_
norway_importances = results['Norway']['model'].feature_importances_
singapore_importances = results['Singapore']['model'].feature_importances_

# Create a Pandas Series for plotting
xgb_canada_importances = pd.Series(canada_importances, index=X_test.columns)
xgb_finland_importances = pd.Series(finland_importances, index=X_test.columns)
xgb_italy_importances = pd.Series(italy_importances, index=X_test.columns)
xgb_kenya_importances = pd.Series(kenya_importances, index=X_test.columns)
xgb_norway_importances = pd.Series(norway_importances, index=X_test.columns)
xgb_singapore_importances = pd.Series(singapore_importances, index=X_test.columns)

# Plot subplots
xgb_canada_importances.plot.bar(ax=axes[0, 0], color = 'firebrick')
axes[0, 0].set_title('Canada - XGBoost Feature Importances')
axes[0, 0].set_xlabel('Features')
axes[0, 0].set_ylabel('Mean Decrease in Impurity')

xgb_finland_importances.plot.bar(ax=axes[0, 1], color = 'mediumblue')
axes[0, 1].set_title('Finland - XGBoost Feature Importances')
axes[0, 1].set_xlabel('Features')
axes[0, 1].set_ylabel('Mean Decrease in Impurity')

xgb_italy_importances.plot.bar(ax=axes[1, 0], color = 'crimson')
axes[1, 0].set_title('Italy - XGBoost Feature Importances')
axes[1, 0].set_xlabel('Features')
axes[1, 0].set_ylabel('Mean Decrease in Impurity')

xgb_kenya_importances.plot.bar(ax=axes[1, 1], color = 'darkorange')
axes[1, 1].set_title('Kenya - XGBoost Feature Importances')
axes[1, 1].set_xlabel('Features')
axes[1, 1].set_ylabel('Mean Decrease in Impurity')

xgb_norway_importances.plot.bar(ax=axes[2, 0], color = 'forestgreen')
axes[2, 0].set_title('Norway - XGBoost Feature Importances')
axes[2, 0].set_xlabel('Features')
axes[2, 0].set_ylabel('Mean Decrease in Impurity')

xgb_singapore_importances.plot.bar(ax=axes[2, 1], color = 'silver')
axes[2, 1].set_title('Singapore - XGBoost Feature Importances')
axes[2, 1].set_xlabel('Features')
axes[2, 1].set_ylabel('Mean Decrease in Impurity')

plt.tight_layout()
plt.show()


# Plot for trend analysis
plt.figure(figsize = (9, 6))
plt.scatter(y_test, y_pred, alpha = 0.4, color = "crimson")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
         linestyle = "--", color = "black", label = "Perfect Fit")

plt.xlabel("Actual Num Sold")
plt.ylabel("Predicted Num Sold")
plt.title("Predicted vs. Actual Scatter Plot")

plt.legend()
plt.grid(True)
plt.show()


# Initialize an empty list to store results
results_list = []

# Store results for the 3rd model
for country, metrics in results.items():
    results_list.append({"Country": country,
                         "RMSE": round(metrics["rmse"], 3),
                         "MAE": round(metrics["mae"], 3),
                         "MAPE(%)": round(metrics["mape"], 3),
                         "R² Score": round(metrics["r2"], 3)})

# Convert to DataFrame
results_df = pd.DataFrame(results_list)

# Calculate Accuracy (%) as 100 - MAPE
results_df["Accuracy(%)"] = round(100 - results_df["MAPE(%)"], 3)

# Sort by MAPE
results_df = results_df.sort_values(by = ["MAPE(%)"]).reset_index(drop = True)
results_df


# Define features and target variable
features_2 = ['store_index', 'product_index', 'weekday', 'month', 'year',
              'is_weekend', 'day_sin', 'month_sin', 'month_cos', 'year_sin']
target = 'num_sold'

# Step 1: Split data by country for separate modeling
countries = df['country'].unique()  # Extract unique country names

# Store results for each country
results_2 = {}

# Loop through each country to train a separate model
for country in countries:
    print(f"Processing country: {country}")

    # Filter data for the specific country
    country_data = df[df['country'] == country]
    X = country_data[features_2]
    y = country_data[target]

    # Split the country-specific data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.2,
                                                        random_state = 42)

    # Step 2: Define an Optuna objective function for hyperparameter tuning
    def objective_2(trial):
        # Define the hyperparameter search space
        param = {
            'objective': 'reg:squarederror',
            'n_estimators': trial.suggest_int('n_estimators', 90, 1400),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': 42}

        # Train an XGBoost model with the trial parameters
        model_2 = xgb.XGBRegressor(**param)
        model_2.fit(X_train, y_train)

        # Evaluate using the MAPE on the test set
        y_pred2 = model_2.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, y_pred2)
        return mape

    # Step 3: Run Optuna optimization
    print("Starting Optuna optimization...")
    study_2 = optuna.create_study(direction = 'minimize')  # Minimize MAPE
    study_2.optimize(objective_2, n_trials = 50, show_progress_bar = True)

    # Best hyperparameters
    best_params_2 = study_2.best_params
    print(f"Best params for {country}: {best_params_2}")

    # Step 4: Train the final model using the best parameters
    print("Training final model with best parameters...")
    final_model_2 = xgb.XGBRegressor(**best_params_2)
    final_model_2.fit(X_train, y_train)

    # Step 5: Make predictions and evaluate the final model
    y_pred2 = final_model_2.predict(X_test)
    mse_2 = mean_squared_error(y_test, y_pred2)
    rmse_2 = np.sqrt(mse_2)
    mae_2 = mean_absolute_error(y_test, y_pred2)
    mape_2 = mean_absolute_percentage_error(y_test, y_pred2)
    r2_2 = r2_score(y_test, y_pred2)

    print(f"Evaluation for {country} - MSE: {mse_2}, RMSE: {rmse_2}, MAE: {mae_2}, MAPE: {mape_2}, R2: {r2_2}")

    # Step 6: Store results for each country
    results_2[country] = {'model': final_model_2,
                          'mse': mse_2,
                          'rmse': rmse_2,
                          'mae': mae_2,
                          'mape': mape_2,
                          'r2': r2_2,
                          'best_params': best_params_2}

# Step 7: Summarize results for all countries
print("\nFinal Results:")
for country, metrics in results_2.items():
    print(f"{country} - MSE: {metrics['mse']}, RMSE: {metrics['rmse']}, MAE: {metrics['mae']}, MAPE: {metrics['mape']}, R2: {metrics['r2']}")


# Initialize an empty list to store results
results_list = []
# Store results for the second model
for country, metrics in results_2.items():
    results_list.append({
        "Country": country,
        "RMSE": round(metrics["rmse"], 3),
        "MAE": round(metrics["mae"], 3),
        "MAPE(%)": round(metrics["mape"], 3),
        "R² Score": round(metrics["r2"], 3)})

# Convert to DataFrame
results_df = pd.DataFrame(results_list)

# Calculate Accuracy (%) as 100 - MAPE
results_df["Accuracy(%)"] = round(100 - results_df["MAPE(%)"], 3)

# Sort
results_df = results_df.sort_values(by = ["MAPE(%)"]).reset_index(drop = True)

from IPython.display import display
display(results_df)


# Save model as dictionary
# 'results' contains the trained models for each country
file_name = "2nd_xgb_models_dict.pkl"

# Extract only the models from results and store them in a dictionary
models_dict_2 = {country: results_2[country]['model'] for country in results_2}

# Save the dictionary of models
with open(file_name, "wb") as f:
    pickle.dump(models_dict_2, f)

print("✅ Models saved as dictionary!")


# Plot for trend analysis
plt.figure(figsize = (9, 6))
plt.scatter(y_test, y_pred2, alpha = 0.5, color = "violet")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
         linestyle = "--", color = "black", label = "Perfect Fit")

plt.xlabel("Actual Num Sold")
plt.ylabel("Predicted Num Sold")
plt.title("Predicted vs. Actual Scatter Plot")

plt.legend()
plt.grid(True)
plt.show()


# Define features and target variable
features_3 = ['store_index', 'country_index', 'is_holiday', 'weekday', 'month',
              'year', 'is_weekend', 'day_sin', 'day_cos', 'month_sin',
              'month_cos', 'year_sin', 'year_cos']
target = 'num_sold'

# Step 1: Split data by product for separate modeling
products = df['product'].unique()  # Extract unique product names

# Store results for each product
results_3 = {}

# Loop through each product to train a separate model
for product in products:
    print(f"Processing product: {product}")

    # Filter data for the specific product
    product_data = df[df['product'] == product]
    X = product_data[features_3]
    y = product_data[target]

    # Split the product-specific data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.2,
                                                        random_state = 42)

    # Step 2: Define an Optuna objective function for hyperparameter tuning
    def objective_3(trial):
        # Define the hyperparameter search space
        param = {
            'objective': 'reg:squarederror',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': 42}

        # Train an XGBoost model with the trial parameters
        model_3 = xgb.XGBRegressor(**param)
        model_3.fit(X_train, y_train)

        # Evaluate using the MAPE on the test set
        y_pred3 = model_3.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, y_pred3)
        return mape

    # Step 3: Run Optuna optimization
    print("Starting Optuna optimization...")
    study_3 = optuna.create_study(direction = 'minimize')
    study_3.optimize(objective_3, n_trials = 50, show_progress_bar = True)

    # Best hyperparameters
    best_params_3 = study_3.best_params
    print(f"Best params for {product}: {best_params_3}")

    # Step 4: Train the final model using the best parameters
    print("Training final model with best parameters...")
    final_model_3 = xgb.XGBRegressor(**best_params_3)
    final_model_3.fit(X_train, y_train)

    # Step 5: Make predictions and evaluate the final model
    y_pred3 = final_model_3.predict(X_test)
    mse_3 = mean_squared_error(y_test, y_pred3)
    rmse_3 = np.sqrt(mse_3)
    mae_3 = mean_absolute_error(y_test, y_pred3)
    mape_3 = mean_absolute_percentage_error(y_test, y_pred3)
    r2_3 = r2_score(y_test, y_pred3)

    print(f"Evaluation for {product} - MSE: {mse_3}, RMSE: {rmse_3}, MAE: {mae_3}, MAPE: {mape_3}, R2: {r2_3}")

    # Step 6: Store results for each product
    results_3[product] = {'model': final_model_3,
                          'mse': mse_3,
                          'rmse': rmse_3,
                          'mae': mae_3,
                          'mape': mape_3,
                          'r2': r2_3,
                          'best_params': best_params_3}

# Step 7: Summarize results for all products
print("\nFinal Results:")
for product, metrics in results_3.items():
    print(f"{product} - MSE: {metrics['mse']}, RMSE: {metrics['rmse']}, MAE: {metrics['mae']}, MAPE: {metrics['mape']}, R2: {metrics['r2']}")


# Initialize an empty list to store results
results_list = []

# Store results for the 3rd model
for product, metrics in results_3.items():
    results_list.append({
        "Product": product,
        "RMSE": round(metrics["rmse"], 3),
        "MAE": round(metrics["mae"], 3),
        "MAPE(%)": round(metrics["mape"], 3),
        "R² Score": round(metrics["r2"], 3)})

# Convert to DataFrame
results_df = pd.DataFrame(results_list)

# Calculate Accuracy (%) as 100 - MAPE
results_df["Accuracy(%)"] = round(100 - results_df["MAPE(%)"], 3)

# Sort
results_df = results_df.sort_values(by = ["MAPE(%)"]).reset_index(drop = True)
results_df


# Plot for trend analysis
plt.figure(figsize = (9, 6))
plt.scatter(y_test, y_pred3, alpha = 0.4, color = "teal")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
         linestyle = "--", color = "black", label = "Perfect Fit")

plt.xlabel("Actual Num Sold")
plt.ylabel("Predicted Num Sold")
plt.title("Predicted vs. Actual Scatter Plot")

plt.legend()
plt.grid(True)
plt.show()


# Separate plots
fig, axes = plt.subplots(3, 2, figsize = (12, 15))

# Access feature importances from the trained model for the last processed country
holographic_goose_importances = results_3['Holographic Goose']['model'].feature_importances_
kaggle_importances = results_3['Kaggle']['model'].feature_importances_
kaggle_tiers_importances = results_3['Kaggle Tiers']['model'].feature_importances_
kerneler_importances = results_3['Kerneler']['model'].feature_importances_
kerneler_dark_mode_importances = results_3['Kerneler Dark Mode']['model'].feature_importances_

# Create a Pandas Series for plotting
xgb_holographic_goose_importances = pd.Series(holographic_goose_importances, index=X_test.columns)
xgb_kaggle_importances = pd.Series(kaggle_importances, index=X_test.columns)
xgb_kaggle_tiers_importances = pd.Series(kaggle_tiers_importances, index=X_test.columns)
xgb_kerneler_importances = pd.Series(kerneler_importances, index=X_test.columns)
xgb_kerneler_dark_mode_importances = pd.Series(kerneler_dark_mode_importances, index=X_test.columns)

# Plot subplots
xgb_holographic_goose_importances.plot.bar(ax=axes[0, 0], color = 'firebrick')
axes[0, 0].set_title('holographic_goose - XGBoost Feature Importances')
axes[0, 0].set_xlabel('Features')
axes[0, 0].set_ylabel('Mean Decrease in Impurity')

xgb_kaggle_importances.plot.bar(ax=axes[0, 1], color = 'mediumblue')
axes[0, 1].set_title('kaggle - XGBoost Feature Importances')
axes[0, 1].set_xlabel('Features')
axes[0, 1].set_ylabel('Mean Decrease in Impurity')

xgb_kaggle_tiers_importances.plot.bar(ax=axes[1, 0], color = 'crimson')
axes[1, 0].set_title('kaggle_tiers - XGBoost Feature Importances')
axes[1, 0].set_xlabel('Features')
axes[1, 0].set_ylabel('Mean Decrease in Impurity')

xgb_kerneler_importances.plot.bar(ax=axes[1, 1], color = 'darkorange')
axes[1, 1].set_title('kerneler - XGBoost Feature Importances')
axes[1, 1].set_xlabel('Features')
axes[1, 1].set_ylabel('Mean Decrease in Impurity')

xgb_kerneler_dark_mode_importances.plot.bar(ax=axes[2, 0], color = 'forestgreen')
axes[2, 0].set_title('kerneler_dark_mode - XGBoost Feature Importances')
axes[2, 0].set_xlabel('Features')
axes[2, 0].set_ylabel('Mean Decrease in Impurity')

plt.tight_layout()
plt.show()

