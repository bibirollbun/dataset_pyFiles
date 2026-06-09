import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
import category_encoders as ce
from sklearn.preprocessing import normalize
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import List
from sklearn.preprocessing import PowerTransformer

SEED=42


def hist_box_chart(df, hist_col):
    # Select numeric columns
    hist_plot = df.select_dtypes(include=['int', 'float'])
    
    # Define number of rows and columns (adjust based on the number of numeric columns)
    ncols = 2
    nrows = len(hist_col)
    
    # Create a figure with subplots for both histograms and boxplots
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*3))  # Adjust height based on the number of rows
    
    # Loop through columns and plot histograms and boxplots
    for i, col in enumerate(hist_col):
        sns.histplot(hist_plot[col], ax=ax[i, 0])  # Histogram on the left
        ax[i, 0].set_title(f'Histogram of {col} with skewness {round(hist_plot[col].skew(),3)}')
        ax[i, 0].set_xlabel(col)
        
        sns.boxplot(x=hist_plot[col], ax=ax[i, 1])  # Boxplot on the right
        ax[i, 1].set_title(f'Boxplot of {col}')
        ax[i, 1].set_xlabel(col)
    
    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()


def col_info(path_list, file_number=1):
    #Header
    print(f"Avaiable file:\n {[path.split('/')[-1] for path in path_list]}")
    print("\nYou are reading:", path_list[file_number-1],end='\n\n')
    
    #Read data
    data = pd.read_csv(path_list[file_number-1])
    value_df = data.copy(deep=True)
    
    for col in value_df.columns:
        value_df[col] = value_df[col].astype('str')
        print(f"There are {value_df[col].nunique()} Unique valuse in Column '{col}'.")
        if data[col].nunique() < 150:
            print(f"\nUnique values in '{col}' column: {sorted(value_df[col].unique().tolist())}")
            df = pd.DataFrame(value_df[col].value_counts().values, columns=['Value'], index=value_df[col].value_counts().index)
            df['Percentage'] = value_df[col].value_counts(normalize=True)*100
            print(df)
        print()
        print("--"*50)

def table_info(path_list):
    for path in path_list:
        df = pd.read_csv(path)
        print(" "*30,"-"*10, path, "-"*10)
        display(df.info(), df.describe(), df.head())


pd.set_option('display.max_columns',500) 

path_list = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        path_list.append(os.path.join(dirname, filename))


table_info(path_list)


col_info(path_list, 3)


calendar_df = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv',)
test_weights_df = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv')
inventory_df = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv')
sales_train_df = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv')
sales_test_df = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv')
solution_df = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv')

train = pd.merge(sales_train_df, inventory_df, on=['unique_id', 'warehouse'], 
                 how='inner', suffixes=['', '_inven'])
train = pd.merge(train, calendar_df, on=['date', 'warehouse'], how='inner', suffixes=['', '_cal'])

merged_df = train.copy()


display(train.info(), train.head())


def date_fun(df):
    # Removing data before year 2021
    df = df[df['date']>='2021-01-01']
    
    # `Date` is dtype is wrong
    df['date'] = pd.to_datetime(df['date'])
    # Sort by date
    df.sort_values(by='date',axis=0, ascending=True, inplace=True)
    #periodic features
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['day_of_year'] = df['date'].dt.dayofyear
    # cyclical features
    df['cos_day'] = np.cos(df['day_of_year']*2*np.pi/365)
    df['sin_day'] = np.sin(df['day_of_year']*2*np.pi/365)

    return df
train = date_fun(train)


def line_chart(df, x='month'):
    date_df = df.copy()
    date_df['date'] = pd.to_datetime(date_df['date'])
    
    # Extract the year from the 'date' column
    date_df['year_month'] = date_df['date'].dt.year.astype(str) + '-' + date_df['date'].dt.month.astype(str).str.zfill(2)
    
    fig = plt.figure(figsize=(15, 5))
    sns.lineplot(data=date_df.sort_values('year_month'), x=x, y='total_orders')
    
    # Add title and labels
    plt.title('Total Orders per Year')
    plt.xlabel(x)
    plt.xticks(rotation=45)
    plt.ylabel('Total Orders')
    
    # Show the plot
    plt.show()

line_chart(train, 'year_month')


# Since only highest discount is apply, Calculate the maximum discount for each row across all discount types
train['discount'] = train[['type_0_discount', 'type_1_discount', 'type_2_discount', 
                           'type_3_discount', 'type_4_discount', 'type_5_discount', 
                           'type_6_discount']].max(axis=1)
# Set negative discounts to 0
train.loc[train['discount'] < 0, 'discount'] = 0


# Drop the unnecessary columns
train.drop(columns=['unique_id', 'type_0_discount', 'type_1_discount', 'type_2_discount', 'type_3_discount', 
                    'type_4_discount', 'type_5_discount', 'type_6_discount', 
                    'L2_category_name_en', 'L3_category_name_en', 'L4_category_name_en',
                    'holiday_name'], inplace=True)

# Rename some columns for Readability
train.rename(columns={'L1_category_name_en': 'category', 'name': 'product_name_id'}, inplace=True)

# There are 'NA' values in total_orders, sales.
display("NA Values:",train.isna().sum())
train.dropna(axis=0,inplace=True)


x = train.groupby(by='category')['total_orders'].sum()
labels = x.index  # Use index for labels

plt.figure(figsize=(6, 4))
plt.pie(x, labels=labels, autopct='%1.1f%%', startangle=140)  # Start angle for better layout
plt.title('Total Orders by Category')  # Add a title
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


train['Product Name'] = train['product_name_id'].str.split('_', expand=True)[0]
top_10_hot = train['Product Name'].value_counts()[:10]

plt.figure(figsize=(15, 5))
plt.bar(data=top_10_hot, x=top_10_hot.index, height=top_10_hot.values,)
# Set title
plt.title("Top 10 Hot Products")
# Display the pie chart
plt.show()


# Price of each Product over time
train.groupby(['Product Name', 'year'])['sales'].max()


# one-hot encoding for top 10
for col in top_10_hot.keys():
    train[col] = train['Product Name'].apply(lambda x: 1 if x==col else 0)


hist_box_chart(train, ['total_orders','sales','sell_price_main'])


from typing import List

def outliers(df, cols: List = [], op='replace', qtr=1.5):
    """
    Handles outliers in the specified columns using the IQR method.
    Depending on the 'op' argument, it either replaces outliers with the median
    or removes the rows containing outliers.
    
    Parameters:
    df: DataFrame - The input data where outliers should be handled.
    col: List - List of column names where outliers should be handled.
    op: str - Operation to perform: 'replace' (replace outliers with median) or 'remove' (remove outliers).
    qtr: float - The multiplier for the IQR (default is 1.5).
    
    Returns:
    DataFrame - The data with outliers either replaced or removed for the specified columns.
    """
    
    for col in cols:
        print("-"*10, f"Performing {op} on column: {col}", "-"*10)
        Q1 = df[col].quantile(0.25)  # 25th percentile (first quartile)
        Q3 = df[col].quantile(0.75)  # 75th percentile (third quartile)
        IQR = Q3 - Q1  # Interquartile range
        
        # Define the lower and upper bounds
        lower_bound = Q1 - qtr * IQR
        upper_bound = Q3 + qtr * IQR
        
        # Calculate the median of the column
        median = df[col].median()
        print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}, Median: {median}")
        
        # Operation Conditions
        if op == 'replace':
            print(f"Performing replacement with median: {median}")
            # Replace outliers with the median
            df.loc[df[col] < lower_bound, col] = median
            df.loc[df[col] > upper_bound, col] = median
            
        elif op == 'remove':
            print(f"Removing values outside {lower_bound} and {upper_bound}")
            # Remove outliers
            df = df.loc[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        else:
            raise ValueError("Invalid operation. Use 'replace' or 'remove'.")
    
    return df


# Feature is left skewed, so i am using 1.5 bound to replace with median
train = outliers(train, ['total_orders'], 'replace')
# Target Column: Sales cannot be altered, Changing the values in the sales column would be incorrect.
train = outliers(train, ['sales', 'sell_price_main'], 'remove')


#train[train['sales']!=0]

# log sales: log transformation
train['log_sales'] = np.log1p(train['sales'])

# PowerTransformer with yeo-johnson method(negtive and zero values present)
box_cox_transformer = PowerTransformer(method='yeo-johnson')
train['sales_yeo_johnson'] = box_cox_transformer.fit_transform(train[['sales']])


def timely_sales(df):
    """
    Aggregates the given dataframe by month, year, and quarter,
    applying 'sum', 'mean' and other operations to the specified sales-related columns.
    Quantile Bins and lags features for sales features.

    Parameters:
    df (DataFrame): Input dataframe with date-related and sales-related columns.
    Returns:
    DataFrame: The dataframe with aggregated features merged.
    """
    # List of time-based features for grouping
    time_features = ['month', 'year', 'quarter']
    # List of sales-related features to aggregate
    sales_features = ['sales', 'total_orders', 'sell_price_main']
    
    # Ensure necessary columns are present
    missing_time_columns = set(time_features) - set(df.columns)
    if missing_time_columns:
        raise ValueError(f"Missing required time columns: {missing_time_columns}")
    # Filter out missing sales features
    sales_features = [col for col in sales_features if col in df.columns]
    if not sales_features:
        raise ValueError("No valid sales-related columns found in the DataFrame.")
    
    # Perform aggregation for each operation
    for operation in ['sum', 'mean', 'median', 'min', 'max']:
        # Group by the time features and aggregate the sales features
        aggregated_sales = df.groupby(by=time_features)[sales_features].agg(operation).reset_index()
        
        # Merge the aggregated results back to the original DataFrame
        df = df.merge(
            aggregated_sales, 
            how='left', 
            on=time_features, 
            suffixes=(None, f'_{operation}')
        )
    
    # Preform quantile by time
    for q in [0.25, 0.75]:
        quantile = df.groupby(by=time_features)[sales_features].quantile(q).reset_index()
        df = df.merge(
            quantile,
            how='left', 
            on=time_features, 
            suffixes=(None, f'_{q}')
        )
    
    # Preform quartile for features
    for col in sales_features:
        df[col+'_quartile'] = pd.qcut(df[col], q=4, labels=[1,2,3,4]).astype(int)
        # Lagged feature
        df[col+'_lag'] = df.sort_values(by='date',axis=0, ascending=True, inplace=False).groupby(['month', 'year'])[col].shift(1,fill_value=0)
        df[col+'_lag_diff'] = df[col] - df[col+'_lag']
    
    return df

train = timely_sales(train)


# Separate features and target after all cleaning is done
X = train.drop(['date','sales'], axis=1)
y = train['sales']
#print(f"X shape: {X.shape}, y shape: {y.shape}")

# apply encoding on X
encoder = ce.TargetEncoder(cols=X.select_dtypes(include='object').columns)
X = encoder.fit_transform(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


_col = X_train.columns
X_train.head()


corr_df = X_train.copy()
corr_df['sales'] = y_train
corr = corr_df.corr()
pd.DataFrame(abs(corr['sales']).sort_values())


# Initialize the XGBoost regressor Base Model
model = XGBRegressor()
# Fit the model
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance (using mean squared error)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")


importance = model.get_booster().get_score(importance_type='weight')
importance_df = pd.DataFrame(importance.items(), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)
importance_df


lr = .1
es = 10
n_est = round(5000/lr)
base_params = {
    'n_estimators':n_est
    ,'learning_rate':lr
    ,'verbosity':0
    ,'enable_categorical':True
    #,'early_stopping_rounds':es
    ,'objective':'reg:squarederror'
    ,'eval_metric':'rmse'
    ,'device':'cuda'
    ,'reg_lambda':0
    ,'min_child_weight':1,
    "random_state": SEED
}

# Initialize the XGBoost regressor with Hypermeter
regressor = XGBRegressor(**base_params)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Evaluate the model's performance (using mean squared error)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

