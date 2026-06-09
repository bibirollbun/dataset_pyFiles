# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
%matplotlib inline 


from time import time

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


train.head(4)


train.info()


train = train.dropna() # drop missing values


def add_date_features(df):
    # Ensure 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Extract date components
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['year'] = df['date'].dt.year
    df['week'] = df['date'].dt.isocalendar().week
    df['dayofweek'] = df['date'].dt.dayofweek

    # Weekend flag: 1 for weekend (Saturday and Sunday), 0 for weekday (Monday to Friday)
    df.loc[df['dayofweek'] > 4, 'weekend'] = 1
    df.loc[df['dayofweek'] <= 4, 'weekend'] = 0

    # Sunday flag: 1 for Sunday, 0 otherwise
    df.loc[df['dayofweek'] == 6, 'sunday'] = 1
    df.loc[df['dayofweek'] != 6, 'sunday'] = 0

    # Fourier features for cyclic patterns (month, day, day of the week, week)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 6)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 6)

    df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)

    return df



train = add_date_features(train)


plt.figure(figsize=(20, 10), dpi = 300)
sns.lineplot(data=train, x='date', y='num_sold', errorbar=None)


sales_by_year = train.groupby('year')['num_sold'].mean()

sales_by_year.plot(kind='line')

plt.title('Average Sales by Year')
plt.xlabel('Year')
plt.ylabel('Average Number of Units Sold')
plt.xticks(sales_by_year.index, rotation=45)
plt.show()


sales_by_day = train.groupby('day')['num_sold'].mean()

sales_by_day.plot(kind='line')

plt.title('Average Sales by Day')
plt.xlabel('days')
plt.ylabel('Average Number of Units Sold')
plt.xticks(sales_by_day.index, rotation=45)
plt.show()


import calendar

# Group by month and calculate the average number of units sold
sales_by_month = train.groupby('month')['num_sold'].mean()

# Convert month numbers to month names
sales_by_month.index = sales_by_month.index.map(lambda x: calendar.month_name[x])

# Set Seaborn style for better aesthetics
sns.set(style="whitegrid")

# Create the plot
plt.figure(figsize=(10, 6))

# Plotting the line plot
plt.plot(sales_by_month.index, sales_by_month.values, marker='o', color='dodgerblue', linewidth=2, markersize=6)

# Adding title and labels with better font sizes
plt.title('Average Sales by Month', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=14)
plt.ylabel('Average Number of Units Sold', fontsize=14)

# Customize x-axis ticks and rotate for better readability
plt.xticks(rotation=45, fontsize=12)

# Add gridlines for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Display the plot
plt.tight_layout()
plt.show()


avg_stroke = train["num_sold"].astype("float").mean(axis = 0)
print("Averge of number sold:", avg_stroke)


import matplotlib.pyplot as plt
import seaborn as sns

def visualize_distribution(data, feature, figsize=(12, 7), kde=False, bins=None):
    # Set the style for better readability and aesthetics
    sns.set(style="whitegrid", palette="muted")
    
    # Create subplots with better height ratio for boxplot and histogram
    f2, (ax_box2, ax_hist2) = plt.subplots(nrows=2,  
                                           sharex=True, 
                                           gridspec_kw={"height_ratios": (0.25, 0.75)},
                                           figsize=figsize)  

    # Boxplot customization
    sns.boxplot(data=data, x=feature, ax=ax_box2, showmeans=True, 
                palette="coolwarm", fliersize=5, linewidth=2)
    ax_box2.set_title(f"Boxplot of {feature}", fontsize=14)
    ax_box2.set_ylabel('Values', fontsize=12)

    # Histogram customization
    sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2, 
                 bins=bins, color="dodgerblue", edgecolor="black", linewidth=1.5) if bins else \
        sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2, 
                     color="dodgerblue", edgecolor="black", linewidth=1.5)

    # Vertical lines for mean and median
    ax_hist2.axvline(data[feature].mean(), color="green", linestyle='--', linewidth=2, label="Mean")
    ax_hist2.axvline(data[feature].median(), color="black", linestyle='-', linewidth=2, label="Median")

    # Add labels and title to histogram
    ax_hist2.set_title(f"Histogram of {feature} with KDE", fontsize=14)
    ax_hist2.set_xlabel(f'{feature} Values', fontsize=12)
    ax_hist2.set_ylabel('Frequency', fontsize=12)
    
    # Adding a legend for mean and median lines
    ax_hist2.legend()

    # Customize the gridlines
    ax_hist2.grid(True, linestyle='--', alpha=0.6)

    # Adjust layout for better visual space
    plt.tight_layout()
    plt.show()



visualize_distribution(train, "num_sold", kde=True)


def visualize_line_plot(train, x_col='date', y_col='num_sold', hue_col='country', figsize=(20, 10), dpi=300):
    # Set seaborn style for better aesthetics
    sns.set(style="whitegrid", palette="Set1")
    
    # Create the line plot
    plt.figure(figsize=figsize, dpi=dpi)
    sns.lineplot(data=train, x=x_col, y=y_col, hue=hue_col, errorbar=None, marker="o", linewidth=2, markersize=6)
    
    # Add a title and axis labels
    plt.title(f"Number of Units Sold Over Time by Country", fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Number of Units Sold', fontsize=14)
    
    # Customize the legend
    plt.legend(title='Country', title_fontsize=12, loc='upper left', fontsize=12, frameon=False)
    
    # Improve the grid lines
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Rotate the x-axis labels for better readability if needed
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    # Adjust layout to avoid overlap
    plt.tight_layout()
    
    # Show the plot
    plt.show()



visualize_line_plot(train)


def visualize_sales(df_train,col='country',title="Yearly Sales by Country",legend_title="Country"):
    # Get unique countries
    countries = df_train[col].unique()

    # Set seaborn style and color palette
    sns.set(style="whitegrid", palette="tab10")
    
    # Create the figure and set its size and resolution
    plt.figure(figsize=(15, 7), dpi=300)

    # Loop through each country to plot the sales over time
    for country in countries:
        # Filter the data for the country and group by date
        country_sold = df_train.loc[df_train[col] == country].groupby(['date'])['num_sold'].sum()
        
        # Convert index to datetime and resample monthly
        country_sold.index = pd.to_datetime(country_sold.index)
        country_sold_monthly = country_sold.resample('M').sum()
        
        # Plot the line for each country
        sns.lineplot(data=country_sold_monthly, label=country, linewidth=2)

    # Add a title and labels
    plt.title(title, fontsize=18)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Total Number of Units Sold", fontsize=14)

    # Customize the legend
    plt.legend(title=legend_title, title_fontsize=12, loc='upper left', fontsize=12, frameon=False)

    # Add gridlines and adjust their appearance
    plt.grid(True, linestyle='--', alpha=0.6)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=12)

    # Adjust layout for a clean and spacious look
    plt.tight_layout()

    # Show the plot
    plt.show()



visualize_sales(train)


visualize_line_plot(train,hue_col='store')


visualize_sales(train,col='store',title="Yearly Sales by Store",legend_title="Store")


visualize_line_plot(train,hue_col='product')


visualize_sales(train,col='product',title="Yearly Sales by Product",legend_title="Product")


train.drop(columns=['date', 'day', 'dayofweek', 'week', 'month'], inplace=True)


cat_features = np.array([i for i in train.columns.tolist() if train[i].dtype == 'object'])
num_features = np.array([i for i in train.columns.tolist() if train[i].dtype != 'object'])

print("Number features column =" , len(num_features))
print("Categorial features column =" , len(cat_features))


X = train.drop('num_sold', axis=1)
y = train['num_sold']


from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Create Pipline
column_transformer = make_column_transformer(
    (StandardScaler(),['year', 'day_sin', 
                       'day_cos', 'dayofweek_sin', 
                       'dayofweek_cos', 'week_sin', 
                       'week_cos', 'month_sin', 'month_cos']),
    
    (OneHotEncoder(handle_unknown='ignore', drop='first'),
     make_column_selector(dtype_include='object')),
    remainder='passthrough',
    verbose_feature_names_out=False)


X = column_transformer.fit_transform(X)


column_transformer.get_feature_names_out().tolist()

X = pd.DataFrame(data=X, 
                columns=column_transformer.get_feature_names_out().tolist())
X.sample(5)


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.95, random_state=2137)


import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

def train_and_predict_with_cv(X, y, X_test,model, n_splits=5):
    
    # Initialize variables
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_predictions = np.zeros(len(X))  # Placeholder for training predictions
    mape_scores = []  # List to store MAPE scores for each fold
    test_predictions_list = []  # List to store test predictions from each fold

    # Train and validate the model using 5-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        print("fold: ",fold)
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fit the model
        model.fit(X_train, y_train,eval_set=[(X_val,y_val)],verbose=False)

        # Make predictions on validation set
        y_val_pred = model.predict(X_val)
        train_predictions[val_idx] = y_val_pred

        # Calculate MAPE for this fold
        mape = mean_absolute_percentage_error(y_val, y_val_pred)
        mape_scores.append(mape)
        print(f"Fold {fold}: MAPE = {mape:.4f}")

        # Predict on test data for this fold
        test_pred_fold = model.predict(X_test)
        test_predictions_list.append(test_pred_fold)

    # Average test predictions across folds
    test_predictions_avg = np.mean(test_predictions_list, axis=0)

    # # Apply inverse transformation if using log-transformed data (e.g., expm1)
    # test_predictions_avg = np.expm1(test_predictions_avg)  # If needed (depends on your transformation)

    # Calculate the average MAPE score across all folds
    avg_mape = np.mean(mape_scores)
    
    # Print the final training MAPE score
    print(f"Training MAPE score (5-fold average): {avg_mape:.4f}")
    
    return test_predictions_avg, mape_scores, avg_mape


def train_and_predict(X, y, X_test,y_test, model):

    # Start timer
    start_time = time()

    # Fit the model on the entire training data
    model.fit(X, y,eval_set=[(X_test, y_test)],verbose=False)


    # cat_boost.fit(X_train, y_train, eval_set=[( X_train, y_train), ( X_test, y_test)],verbose=False)

    # Make predictions on the training data (for evaluating the model)
    y_train_pred = model.predict(X)

    # Calculate MAPE for the training data
    mape = mean_absolute_percentage_error(y, y_train_pred)
    
    # Make predictions on the test data
    test_predictions = model.predict(X_test)
    
    # Calculate time taken to train the model
    train_time = np.round(time() - start_time, 3)

    # Print the results
    print(f"Training score: {mape:.4f}")
    print(f"Time taken to train the model: {train_time} seconds")
    
    return test_predictions, mape, train_time



# CatBoost Hyperparameters
# cat_params = {
#           'iterations': 500, 
#           'learning_rate': 0.01, 
#           'depth': 6, 
#          'l2_leaf_reg': 6.888098846099011, 
#           'early_stopping_rounds': 12,
#     'loss_function': 'MAPE',
#     }

cat_params = {'learning_rate': 0.1173420053590065,
 'depth': 10,
 'l2_leaf_reg': 7.845518851375985,
 'random_seed': 42,
 'loss_function': 'MAPE', 
 'iterations': 1000}

catboost_model = CatBoostRegressor(**cat_params)

cat_test_predictions, cat_mape_scores, cat_avg_mape = train_and_predict_with_cv(X_train, y_train, X_test,catboost_model)


print(f"Mean MAPE score: {cat_avg_mape}")
print("Test Predictions: ",cat_test_predictions[:5])


def plot_residuals(y_pred_test, y_test):
    # Calculate residuals
    residuals = y_pred_test - y_test

    # Set seaborn style for a clean look
    sns.set(style="whitegrid")

    # Create the figure
    plt.figure(figsize=(10, 6), dpi=120)

    # Scatter plot for residuals
    plt.scatter(range(len(y_test)), residuals, label='Residuals', color='blue', alpha=0.6)

    # Red line at y = 0 for reference
    plt.axhline(0, color='red', linestyle='--', label='Zero Line')

    # Labels and Title
    plt.xlabel('Fitted Points', fontsize=14)
    plt.ylabel('Residuals', fontsize=14)
    plt.title('Residual Plot', fontsize=16, fontweight='bold')

    # Add gridlines for better visual alignment
    plt.grid(True, linestyle='--', alpha=0.6)

    # Add legend
    plt.legend(fontsize=12)

    # Tight layout to ensure everything fits well
    plt.tight_layout()

    # Show the plot
    plt.show()

# 


plot_residuals(cat_test_predictions, y_test)


def plot_feature_importance(cat_boost, X):
    # Extract feature importances from the model
    importances = cat_boost.feature_importances_

    # Create a DataFrame for feature importances
    importance_dict = {'Feature': list(X.columns), 'Feature Importance': importances}
    importance_df = pd.DataFrame(importance_dict)

    # Sort features by importance in descending order
    importance_df.sort_values(by='Feature Importance', ascending=False, inplace=True)

    # Set Seaborn style for better aesthetics
    sns.set(style="whitegrid")

    # Create the plot
    plt.figure(figsize=(15, 8))

    # Plot top 10 most important features
    sns.barplot(x='Feature Importance', y='Feature', data=importance_df[:10], palette='viridis')

    # Add a title and labels
    plt.title('Top 10 Features Based on Feature Importance', fontsize=18, fontweight='bold')
    plt.xlabel('Feature Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)

    # Add the importance values on top of the bars for clarity
    for index, value in enumerate(importance_df['Feature Importance'][:10]):
        plt.text(value + 0.02, index, f'{value:.4f}', va='center', fontsize=12)

    # Show the plot
    plt.tight_layout()
    plt.show()



plot_feature_importance(catboost_model, X)


test = add_date_features(test)
test.head(4)


# Drop original date-related columns as they are no longer needed
test.drop(columns=['date', 'day', 'dayofweek', 'week', 'month'], inplace=True)


# Apply the column transformer (assuming it's already defined)
test = column_transformer.transform(test)

# Create a DataFrame from the transformed data with the correct column names
test = pd.DataFrame(data=test, columns=column_transformer.get_feature_names_out().tolist())


def get_predictions(model, test):
    
    # Prediction using the trained model (cat_boost)
    y_pred_test = model.predict(test)
    
    return y_pred_test



y_pred_test=get_predictions(catboost_model, test)

print("Sample Predictions:")
print(y_pred_test[:5])





# Create submission file
submission = pd.DataFrame({
    'id': test['id'].astype(int),
    'num_sold': y_pred_test.astype(int)
})


submission.to_csv('submission.csv', index=False)

submission.head(4)




