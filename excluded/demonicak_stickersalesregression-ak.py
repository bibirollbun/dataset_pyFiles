import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
%matplotlib inline


pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.expand_frame_repr", False)  # Prevent line wrapping
pd.set_option("display.float_format", "{:.2f}".format)  # Format floats to 2 decimal places


data=pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
df=data.copy()
df.head()


df.tail()


target="num_sold"


# EDA Function
def perform_eda(df):
    print("----- General Information -----")
    print(df.info())  # Basic information about the dataframe
    
    print("\nSummary Statistics (Numerical Features):")
    print(df.describe().T)
    print("\nSummary Statistics (Categorical Features):")
    print(df.describe(include=["O"]).T)
    
    print("\n----- Unique Value Counts -----")
    print(df.nunique())  # Number of unique values for each column
    
    
    print("\n----- Missing Values -----")
    missing_values = df.isnull().sum()

    print("\nMissing Values Count by Column:")
    print(missing_values[missing_values > 0])
    
    
    # Visualization of Missing Values
    print("\n----- Visualization of Missing Values -----")
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Visualization of Missing Values")
    plt.show()
    

    
    print("\nEDA Completed!")
    
perform_eda(df)


df['date']=pd.to_datetime(df['date'])


def numerical_plots(df):
    # Histograms of Numerical Variables
    print("\n----- Distribution of Numerical Variables -----")
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_columns].hist(bins=10, figsize=(15, 5), edgecolor="black")
    plt.suptitle("Distribution of Numerical Variables")
    plt.show()
    
    # Visualization of Categorical Variables
numerical_plots(df)


def categorical_plot(df, target_col):
    """
    Visualizes categorical variables and their relationship with the target variable.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_col (str): The name of the target column.

    Returns:
    None
    """
    print("\n----- Visualization of Categorical Variables -----")
    
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    
    # Loop through categorical columns
    for column in categorical_columns:
        grouped = df.groupby(column)[target_col].agg(['mean', 'sum']).reset_index()

        # Create a single figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

        # Plot 1: Countplot
        sns.countplot(
            x=column, 
            data=df, 
            palette="viridis", 
            order=df[column].value_counts().index, 
            ax=axes[0]
        )
        axes[0].set_title(f"Count of {column}")
        axes[0].set_xlabel(column)
        axes[0].set_ylabel("Count")
        axes[0].tick_params(axis='x', rotation=45)

        # Plot 2: Mean of target
        sns.barplot(
            x=column, 
            y='mean', 
            data=grouped, 
            palette="viridis", 
            order=grouped[column], 
            ax=axes[1]
        )
        axes[1].set_title(f"Average of {target_col} by {column}")
        axes[1].set_xlabel(column)
        axes[1].set_ylabel(f"Average {target_col}")
        axes[1].tick_params(axis='x', rotation=45)

        # Plot 3: Sum of target
        sns.barplot(
            x=column, 
            y='sum', 
            data=grouped, 
            palette="viridis", 
            order=grouped[column], 
            ax=axes[2]
        )
        axes[2].set_title(f"Sum of {target_col} by {column}")
        axes[2].set_xlabel(column)
        axes[2].set_ylabel(f"Sum {target_col}")
        axes[2].tick_params(axis='x', rotation=45)

        # Show the combined plot
        plt.suptitle(f"Visualization of {column}", fontsize=16, y=1.05)
        plt.show()


categorical_plot(df,target)


df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month


def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

df['season'] = df['month'].apply(get_season)

monthly_sales = df.groupby(['year', 'month'])['num_sold'].sum().reset_index()

# Create a new column for a "Month-Year" string for plotting
monthly_sales['month_year'] = pd.to_datetime(
    monthly_sales[['year', 'month']].assign(day=1)
)

# Plot the data
plt.figure(figsize=(12, 6))
for year in sorted(monthly_sales['year'].unique()):
    year_data = monthly_sales[monthly_sales['year'] == year]
    plt.plot(
        year_data['month_year'],
        year_data['num_sold'],
        marker='o',
        label=f'{year}'
    )

# Customize the plot
plt.xlabel('Month-Year')
plt.ylabel('Total Stickers Sold')
plt.title('Monthly Stickers Sold Over Years')
plt.xticks(rotation=45)
plt.legend(title='Year')
plt.grid()

# Show the plot
plt.show()


# Group by month and calculate average stickers sold
monthly_pattern = df.groupby(df['date'].dt.month)['num_sold'].mean()

# Plot monthly sales
plt.figure(figsize=(8, 5))
monthly_pattern.plot(kind='bar', color='lightgreen')

# Customize the plot
plt.title('Average Stickers Sold by Month')
plt.ylabel('Average Stickers Sold')
plt.xlabel('Month')
plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# Add a column for day of the week
df['day_of_week'] = df['date'].dt.day_name()

# Group by day of the week and calculate average stickers sold
weekly_pattern = df.groupby('day_of_week')['num_sold'].mean()

# Plot the weekly pattern
plt.figure(figsize=(8, 5))
weekly_pattern = weekly_pattern.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])  # Ensure order
weekly_pattern.plot(kind='bar', color='skyblue')

# Customize the plot
plt.title('Average Stickers Sold by Day of the Week')
plt.ylabel('Average Stickers Sold')
plt.xlabel('Day of the Week')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



holidays = [
    # New Year (Global)
    '2010-01-01', '2011-01-01', '2012-01-01', '2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01',
    
    # Christmas (Global)
    '2010-12-25', '2011-12-25', '2012-12-25', '2013-12-25', '2014-12-25', '2015-12-25', '2016-12-25',
    
    # Valentine's Day (Global, February 14)
    '2010-02-14', '2011-02-14', '2012-02-14', '2013-02-14', '2014-02-14', '2015-02-14', '2016-02-14',

    # Halloween (Canada, Norway, etc.)
    '2010-10-31', '2011-10-31', '2012-10-31', '2013-10-31', '2014-10-31', '2015-10-31', '2016-10-31',

    # Easter Sunday (approximation, varies per year)
    '2010-04-04', '2011-04-24', '2012-04-08', '2013-03-31', '2014-04-20', '2015-04-05', '2016-03-27',

    # Canada Day (Canada, July 1)
    '2010-07-01', '2011-07-01', '2012-07-01', '2013-07-01', '2014-07-01', '2015-07-01', '2016-07-01',

    # National Day of Norway (Norway, May 17)
    '2010-05-17', '2011-05-17', '2012-05-17', '2013-05-17', '2014-05-17', '2015-05-17', '2016-05-17',

    # Republic Day of Italy (Italy, June 2)
    '2010-06-02', '2011-06-02', '2012-06-02', '2013-06-02', '2014-06-02', '2015-06-02', '2016-06-02',

    # Independence Day (Kenya, December 12)
    '2010-12-12', '2011-12-12', '2012-12-12', '2013-12-12', '2014-12-12', '2015-12-12', '2016-12-12',

    # Deepavali (Singapore, varies annually, example dates included)
    '2010-11-05', '2011-10-26', '2012-11-13', '2013-11-02', '2014-10-22', '2015-11-10', '2016-10-29',

    # Boxing Day (Canada, Finland, UK, December 26)
    '2010-12-26', '2011-12-26', '2012-12-26', '2013-12-26', '2014-12-26', '2015-12-26', '2016-12-26'
]




df['is_holiday'] = df['date'].isin(pd.to_datetime(holidays))


# Group by month and holiday status
monthly_holiday_effect = df.groupby(['month', 'is_holiday'])['num_sold'].mean().unstack()

# Plot holiday vs non-holiday sales
monthly_holiday_effect.plot(kind='bar', figsize=(12, 6), stacked=True, color=['gray', 'gold'])

# Customize the plot
plt.title('Monthly Stickers Sold: Holiday vs Non-Holiday')
plt.xlabel('Month')
plt.ylabel('Average Stickers Sold')
plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(['Non-Holiday', 'Holiday'], title='Holiday Status')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Group by month and calculate rolling average
df['month_year'] = df['date'].dt.to_period('M')
monthly_sales = df.groupby('month_year')['num_sold'].sum().reset_index()
monthly_sales['month_year'] = monthly_sales['month_year'].dt.to_timestamp()
monthly_sales['3_month_avg'] = monthly_sales['num_sold'].rolling(window=3).mean()

# Plot seasonal trends with rolling averages
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['month_year'], monthly_sales['num_sold'], label='Monthly Stickers Sold', color='lightblue')
plt.plot(monthly_sales['month_year'], monthly_sales['3_month_avg'], label='3-Month Rolling Avg', color='red')

# Customize the plot
plt.title('Monthly Stickers Sold with Seasonal Rolling Average')
plt.xlabel('Month-Year')
plt.ylabel('Total Stickers Sold')
plt.legend()
plt.grid()
plt.show()


# Calculate sales 7 days before and after holidays
df['is_holiday_week'] = df['date'].isin(
    [holiday + pd.Timedelta(days=offset) for holiday in pd.to_datetime(holidays) for offset in range(-7, 8)]
)

# Compare holiday week vs non-holiday week
holiday_week_effect = df.groupby('is_holiday_week')['num_sold'].mean()

# Plot holiday week effect
plt.figure(figsize=(6, 4))
holiday_week_effect.plot(kind='bar', color=['gray', 'green'])

# Customize the plot
plt.title('Holiday Week vs Non-Holiday Week Sales')
plt.xlabel('Holiday Week')
plt.ylabel('Average Stickers Sold')
plt.xticks([0, 1], labels=['Non-Holiday Week', 'Holiday Week'], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()





df.head()


def analyse_target_variable_numeric(df, target):
    print(len(df))
    print("----- Target Statistics -----")
    print(df[target].describe())

    print("\n----- Missing Value Count -----")
    print(df[target].isnull().sum())

    # Distribution plot (Histogram)
    plt.figure(figsize=(10, 5))
    sns.histplot(df[target], bins=20, kde=True, color='blue')
    plt.title(f'{target} Distribution')
    plt.xlabel(target)
    plt.ylabel("Frequency")
    plt.show()


    # Outlier Detection using IQR
    Q1 = df[target].quantile(0.25)  # First Quartile (25%)
    Q3 = df[target].quantile(0.75)  # Third Quartile (75%)
    IQR = Q3 - Q1                  # Interquartile Range

    lower_bound = Q1 - 1.5 * IQR   # Lower Bound
    upper_bound = Q3 + 1.5 * IQR   # Upper Bound

    print("\n----- Outlier Analysis -----")
    print(f"IQR: {IQR}")
    print(f"Lower Bound: {lower_bound}")
    print(f"Upper Bound: {upper_bound}")

    # Number of outliers below and above bounds
    outliers_below = df[target][df[target] < lower_bound].count()
    outliers_above = df[target][df[target] > upper_bound].count()
    total_outliers = outliers_below + outliers_above

    print(f"Number of outliers below lower bound: {outliers_below}")
    print(f"Number of outliers above upper bound: {outliers_above}")
    print(f"Total number of outliers: {total_outliers}")

    # Highlighting outliers in the boxplot
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[target], color='skyblue', flierprops=dict(markerfacecolor='red', marker='o'))
    plt.title(f'{target} Boxplot with Outliers Highlighted')
    plt.xlabel(target)
    plt.show()

    # Visualize outliers (if any)
    if total_outliers > 0:
        outliers = df[(df[target] < lower_bound) | (df[target] > upper_bound)]
        print("\n----- Outlier Details -----")
        print(outliers)
    else:
        print("\nNo significant outliers detected.")

    return lower_bound,upper_bound

# Call the function
lower_bound,upper_bound=analyse_target_variable_numeric(df, "num_sold")



#     # Correlation Matrix
# def correlation_matrix(df):
#     numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
#     print("\n----- Correlation Matrix -----")
#     if len(numeric_columns) > 1:f
#         correlation_matrix = df[numeric_columns].corr()
#         sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
#         plt.title("Correlation Matrix")
#         plt.show()
#         strong_corr = correlation_matrix[(correlation_matrix >= 0.6) | (correlation_matrix <= -0.6)]
#         sns.heatmap(strong_corr, annot=True, cmap="coolwarm", fmt=".2f")
#         plt.title("Strong Correlations Only")
#         plt.show()
#     else:
#         print("Correlation analysis cannot be performed as the number of numerical variables is less than two.")

# correlation_matrix(df)


# Split the data into 80% training and 20% testing
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Print the sizes of the resulting sets
print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")


perform_eda(train_df)


# # Fill missing values in numeric columns
# numeric_columns = train.select_dtypes(include=['number']).columns
# for col in numeric_columns:
#     if col in test.columns:
#         median_value = train[col].median()  # Calculate the median
#         train[col].fillna(median_value, inplace=True)
#         test[col].fillna(median_value, inplace=True)

# # Fill missing values in object columns
# object_columns = train.select_dtypes(include=['object']).columns
# for col in object_columns:
#     if col in test.columns:
#         train[col].fillna("Unknown", inplace=True)
#         test[col].fillna("Unknown", inplace=True)

train_df=train_df.dropna()



lower_bound,upper_bound= analyse_target_variable_numeric(train_df,target)


train_df_better = train_df[(train_df[target] >= lower_bound) | (train_df[target] <= upper_bound)]





train_df_better.head()


train_df_better.columns


features=['country', 'store', 'product',  'is_holiday',
       'year', 'month', 'day_of_week', 'is_holiday_week','season']


# Check available columns in the DataFrame
print("Columns in train_df_better:", train_df_better.columns)

# Check for missing columns
missing_features = [col for col in features if col not in train_df_better.columns]
if missing_features:
    print("Missing features:", missing_features)
else:
    print("All features are present.")



X_train=train_df_better[features]
y_train=train_df_better[target]
X_train.head()


def encode_categorical_variables(df, features):
    """
    Encodes categorical and boolean variables in the given DataFrame for the specified features.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        features (list): List of columns to encode.

    Returns:
        df (pd.DataFrame): The DataFrame with encoded features.
        label_encoders (dict): A dictionary of LabelEncoder objects for categorical features.
    """
    label_encoders = {}  # Store LabelEncoder objects for categorical features

    for feature in features:
        if df[feature].dtype == 'object':  # Handle object/string columns
            print(f"Encoding categorical column: {feature}")
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature].astype(str))  # Convert to numerical values
            label_encoders[feature] = le
        
        elif df[feature].dtype == 'bool':  # Handle boolean columns
            print(f"Encoding boolean column: {feature}")
            df[feature] = df[feature].astype(int)  # Convert True/False to 1/0
        
        elif df[feature].dtype in ['int64', 'int32']:  # Handle integer columns
            print(f"Integer column '{feature}' does not require encoding.")
            # No encoding needed for integers
            
        else:
            print(f"Column '{feature}' has unsupported dtype: {df[feature].dtype}. Skipping.")
    
    print("Encoding complete!")
    return df, label_encoders

X_train_encoded,label_encoders=encode_categorical_variables(X_train,features)


X_train_encoded.head()


from sklearn.ensemble import RandomForestRegressor


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_encoded, y_train)



# Predictions
y_pred_train=model.predict(X_train_encoded)
mape_train=mean_absolute_percentage_error(y_train,y_pred_train)
print(f"mape of train: { mape_train}")


# from sklearn.model_selection import GridSearchCV

# # Define hyperparameter grid
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# # Initialize GridSearchCV
# grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
#                            param_grid=param_grid,
#                            cv=3,
#                            n_jobs=-1,
#                            verbose=2)

# # Fit GridSearchCV
# grid_search.fit(X_train_encoded, y_train)

# # Best parameters
# print("Best Parameters:", grid_search.best_params_)



# # Use the best parameters from GridSearchCV
# best_params = grid_search.best_params_
# optimized_rf = RandomForestRegressor(**best_params, random_state=42)

# # Retrain the model
# optimized_rf.fit(X_train, y_train)

# # Re-evaluate on test set
# y_pred_optimized = optimized_rf.predict(X_test)
# mape_optimized = mean_absolute_percentage_error(y_test, y_pred_optimized)

# print(f"Optimized MAPE: {mape_optimized}")



test_df.head()
test_df=test_df.dropna()


X_test=test_df[features]
y_test=test_df[target]
X_test.head()


def encode_test_data_with_label_encoders(df, label_encoders):
    """
    Encodes the test DataFrame using pre-trained LabelEncoder objects.

    Parameters:
        df (pd.DataFrame): The test DataFrame to encode.
        label_encoders (dict): A dictionary of LabelEncoder objects used for encoding.

    Returns:
        df (pd.DataFrame): The encoded test DataFrame.
    """
    
    # print(11)
    for feature, le in label_encoders.items():
        # print(12)
        if df[feature].dtype == 'object':  # Handle object/string columns
            print(f"Encoding categorical column: {feature}")
            df[feature] = le.transform(df[feature].astype(str))  # Convert to numerical values
            label_encoders[feature] = le
        
        elif df[feature].dtype == 'bool':  # Handle boolean columns
            print(f"Encoding boolean column: {feature}")
            df[feature] = df[feature].astype(int)  # Convert True/False to 1/0
        
        elif df[feature].dtype in ['int64', 'int32']:  # Handle integer columns
            print(f"Integer column '{feature}' does not require encoding.")
            # No encoding needed for integers
            
        else:
            print(f"Column '{feature}' has unsupported dtype: {df[feature].dtype}. Skipping.")
            
    return df
    

# Encode test data using the pre-trained LabelEncoders
X_test_encoded = encode_test_data_with_label_encoders(X_test, label_encoders)



X_test_encoded.head()


y_pred = model.predict(X_test_encoded)
# Evaluate
mape_test = mean_absolute_percentage_error(y_test, y_pred)
print(f"mean_absolute_percentage_error for test: {mape_test}")


final_df=pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
final_df.head()


features


final_df['date']=pd.to_datetime(df['date'])
final_df['year'] = df['date'].dt.year
final_df['month'] = df['date'].dt.month


def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

final_df['season'] = final_df['month'].apply(get_season)


final_df['day_of_week'] = final_df['date'].dt.day_name()
final_df['is_holiday'] = final_df['date'].isin(pd.to_datetime(holidays))
# Calculate sales 7 days before and after holidays
final_df['is_holiday_week'] = final_df['date'].isin(
    [holiday + pd.Timedelta(days=offset) for holiday in pd.to_datetime(holidays) for offset in range(-7, 8)]
)
final_df.head()


id=final_df['id']


final_X=final_df[features]


final_X.head()


final_X_test_encoded = encode_test_data_with_label_encoders(final_X, label_encoders)


y_pred = model.predict(final_X_test_encoded)



# Create the DataFrame
result_df = pd.DataFrame({
    'id': id,
    'num_sold': y_pred
})

# Display the resulting DataFrame
print(result_df)


result_df.to_csv('submission.csv', index=False)


result=pd.read_csv("/kaggle/working/submission.csv")
result.head()

