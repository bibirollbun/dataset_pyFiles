# Load libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error


# Load the necessary data
train_data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


# Summarize the data for a better overview
def summarize_data(df):
    summary = {
        "Statistical description": df.describe(include="all"),  
        "First rows": df.head(), 
        "Null values": df.isnull().sum()  
    }
    for key, value in summary.items():
        print(f"\n=== {key} ===\n")
        print(value)

summarize_data(train_data)


# Mostrar ejemplos de filas con `NA` en num_sold
train_data[train_data['num_sold'].isna()].sample(20)



# Contar los NA en num_sold por producto
na_by_product = train_data[train_data['num_sold'].isna()]['product'].value_counts()

# Mostrar los resultados
print(na_by_product)


# Total de registros por producto
total_by_country = train_data['country'].value_counts()

# Porcentaje de NA por producto
percentage_na_by_country = (na_by_country / total_by_country) * 100

# Mostrar los resultados ordenados por porcentaje
print(percentage_na_by_country.sort_values(ascending=False))


# Total de registros por producto
total_by_product = train_data['product'].value_counts()

# Porcentaje de NA por producto
percentage_na_by_product = (na_by_product / total_by_product) * 100

# Mostrar los resultados ordenados por porcentaje
print(percentage_na_by_product.sort_values(ascending=False))

# Gráfico de barras
plt.figure(figsize=(10, 6))
na_by_product.plot(kind='bar', color='skyblue')
plt.xlabel('Producto', fontsize=12)
plt.ylabel('Cantidad de NA en num_sold', fontsize=12)
plt.title('Valores NA en num_sold por producto', fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()
plt.show()


# Replace NaN  with 0
na_cols = train_data.select_dtypes(include=['float'])  
train_data[na_cols.columns] = na_cols.fillna(0)


print("NaN values replaced with 0")
print(train_data.isna().sum())


# Transform date column to datetime variable
train_data['date'] = pd.to_datetime(train_data['date'])

# Create new features from date 
train_data['year'] = train_data['date'].dt.year  
train_data['month'] = train_data['date'].dt.month  
train_data['day'] = train_data['date'].dt.day
train_data['day_of_week'] = train_data['date'].dt.dayofweek


# Filtrar los datos con NA y producto "Holographic Goose"
na_holo = train_data[(train_data['product'] == 'Holographic Goose') & (train_data['num_sold'].isna())]



# Contar NA por año
na_by_year = na_holo.groupby('year').size()

# Mostrar resultados
print(na_by_year)


plt.figure(figsize=(10, 6))
na_by_year.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribución de NA en Holographic Goose por Año', fontsize=14)
plt.xlabel('Año', fontsize=12)
plt.ylabel('Cantidad de NA', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# Contar NA por mes
na_by_month = na_holo.groupby('month').size()

# Mostrar resultados
print(na_by_month)

plt.figure(figsize=(10, 6))
na_by_month.plot(kind='bar', color='orange', edgecolor='black')
plt.title('Distribución de NA en Holographic Goose por Mes', fontsize=14)
plt.xlabel('Mes', fontsize=12)
plt.ylabel('Cantidad de NA', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# Contar NA por año y mes
na_by_year_month = na_holo.groupby(['year', 'month']).size().unstack()

# Graficar un mapa de calor para año y mes
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.heatmap(na_by_year_month, annot=True, fmt="d", cmap="YlGnBu")
plt.title('Distribución de NA en Holographic Goose por Año y Mes', fontsize=14)
plt.xlabel('Mes', fontsize=12)
plt.ylabel('Año', fontsize=12)
plt.show()



# Calcular la mediana por producto y mes
median_sales = train_data.groupby(['product', 'month', 'year'])['num_sold'].transform('median')

# Rellenar los NA con la mediana
train_data['num_sold'] = train_data['num_sold'].fillna(median_sales)



# Transform date column to datetime variable
test_data['date'] = pd.to_datetime(test_data['date'])

# Create new features from date 
test_data['year'] = test_data['date'].dt.year  
test_data['month'] = test_data['date'].dt.month  
test_data['day'] = test_data['date'].dt.day
test_data['day_of_week'] = test_data['date'].dt.dayofweek


# Remove outliers 
def remove_outliers_iqr(df, columns, factor=1.5):
    df_cleaned = df.copy()  
    
    for col in columns:
        # Calculate Q1, Q3, and IQR
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define limits
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        # Filter data within the limits
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    
    return df_cleaned

columns_to_check = ['num_sold']
cleaned_data = remove_outliers_iqr(train_data, columns=columns_to_check)
print("\nData after removing outliers:")
print(cleaned_data.describe())


variables = ['year', 'month', 'day']

for var in variables:
    # Group by each unique value and calculate the average
    data_grouped = train_data.groupby(var)['num_sold'].mean().reset_index()
    
    # Plot grouped data
    plt.plot(data_grouped[var], data_grouped['num_sold'], marker='o')
    plt.title(f'Relationship between {var} and num_sold')
    plt.xlabel(var)
    plt.ylabel('Average num_sold')
    plt.grid(True)
    plt.show()


variables = ["product", "store", "country"] 

for var in variables: 
    # Group by the current variable and calculate the average num_sold
    grouped_data = train_data.groupby(var)['num_sold'].mean().reset_index()

    # Create a bar chart
    plt.figure(figsize=(15, 8))
    sns.barplot(data=grouped_data, x=var, y='num_sold')
    plt.title(f'Promedio de ventas por {var}')
    plt.xlabel(var)
    plt.ylabel('Promedio de ventas')
    plt.xticks(rotation=45)
    plt.show()


# Encode categorical columns
columns = ["product", "store", "country"]

def encode_categorical_columns(dataframe, columns):
    for col in columns:
        dataframe[col] = pd.Categorical(dataframe[col]).codes  
    return dataframe

train_v2 = encode_categorical_columns(train_data, columns)
test_v2 = encode_categorical_columns(test_data, columns)


# Correlation of independent variables with the main variable
correlaciones = train_v2.corr()  
correlation_target = correlaciones['num_sold'].sort_values(ascending=False)
print(correlation_target)


# Combine train_data and test_data
combined_data = pd.concat([train_v2, test_v2], axis=0, ignore_index=True)

# Calcular promedios para 'store', 'country' y 'product' en un solo paso
for column, new_col_name in [('store', 'avg_sales_bystore'), 
                             ('country', 'avg_sales_bycountry'), 
                             ('product', 'avg_sales_byproduct')]:
    averages = combined_data.groupby(column)[['num_sold']].mean().reset_index()
    averages.rename(columns={'num_sold': new_col_name}, inplace=True)
    combined_data = combined_data.merge(averages, on=column, how='left')

print(combined_data.head())



# Prepare for model training
train_v3 = combined_data[~combined_data['num_sold'].isna()].copy()
test_v3 = combined_data[combined_data['num_sold'].isna()].copy()
train_v3 = train_v3.drop(columns=['date'])
test_v3 = test_v3.drop(columns=['date'])

# Transform 'num_sold' to logarithmic scale
train_v3['sold_log'] = np.log1p(train_v3['num_sold'])  

# Separate independent and dependent variables
X = train_v3.drop(columns=['num_sold', 'sold_log'])  
y = train_v3['sold_log']  

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Hyperparameter selection for CatBoost
catboost_model = CatBoostRegressor(
    eval_metric='RMSE',
    verbose=False,
    random_seed=42
)

param_distributions = {
    'iterations': [500, 1000, 2000],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 7],
    'bagging_temperature': [0.2, 0.5, 0.7, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=catboost_model,
    param_distributions=param_distributions,
    n_iter=10,                   
    scoring='neg_mean_squared_error',  
    cv=3,                        
    random_state=42,
    verbose=1
)


random_search.fit(X_train, y_train)

print("Best hyperparameters:", random_search.best_params_)


# Define the MAPE metric
def mean_absolute_percentage_error(y_true, y_pred):
    non_zero_indices = y_true != 0
    y_true_filtered = y_true[non_zero_indices]
    y_pred_filtered = y_pred[non_zero_indices]
    return np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100

# Train a CatBoost model with the previously obtained hyperparameters
catboost_model = CatBoostRegressor(
    iterations=500,           
    learning_rate=0.05,        
    depth=10,                   
    eval_metric='RMSE',        
    verbose=False,             
    random_seed=42,
    l2_leaf_reg=5,
    bagging_temperature=0.5
)
catboost_model.fit(X_train, y_train)

# Make predictions on the logarithmic scale and transform them back to the original scale
catboost_predictions_log = catboost_model.predict(X_test)
catboost_predictions_original = np.expm1(catboost_predictions_log)  

# Transform y_test back to the original scale
y_test_original = np.expm1(y_test)

# Calculate MAPE
mape = mean_absolute_percentage_error(y_test_original, catboost_predictions_original)
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

output_df = pd.DataFrame({
    'Actual Sales (Original)': y_test_original,
    'Predicted Sales (Original)': catboost_predictions_original
})
print("\nFirst predictions of the model:")
print(output_df.head())



 # Preprocess test_data
test_data_preprocessed = test_v3[X_train.columns] 

# Make predictions on the new dataset
test_predictions_log = catboost_model.predict(test_data_preprocessed)
test_predictions_original = np.expm1(test_predictions_log)  

# Create a DataFrame to store the predictions
output_test = pd.DataFrame({
    'Predicted Sales (Original)': test_predictions_original
})

# Display the first rows of the predictions
print("\nPredictions for the test dataset:")
print(output_test.head())




# Create the submission DataFrame with the columns 'id' and 'sales'
submission = test_data[['id']].copy()  

# Assign predictions to the 'submission' DataFrame
submission['num_sold'] = output_test['Predicted Sales (Original)'].values  
print(submission.head())

# Save the CSV file
submission.to_csv('/kaggle/working/final_submission.csv', index=False)
print("Predictions file created.")

