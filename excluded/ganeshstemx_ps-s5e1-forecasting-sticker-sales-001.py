print('Begin')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold, GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder


train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
gdp_per_capita_df = pd.read_csv('/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv')


train_df.head()


test_df.head()


submission_df.head()


gdp_per_capita_df.head()


print(f'Shape of the train set: {train_df.shape}')
print(f'Shape of the test set: {test_df.shape}')
print(f'Shape of the sample submission: {submission_df.shape}')


train_df.isnull().sum()


test_df.isnull().sum()


def prepare_gdp_ratios(gdp_df, train_df, years):
    """Prepare GDP ratios dataframe."""
    # Filter and calculate ratios
    gdp_filtered = gdp_df.loc[
        gdp_df["Country Name"].isin(train_df["country"].unique()),
        ["Country Name"] + years
    ].set_index("Country Name")
    
    # Calculate ratios for all years at once
    gdp_ratios = gdp_filtered.div(gdp_filtered.sum())
    
    # Reshape to long format
    gdp_ratios = (
        gdp_ratios
        .rename_axis(index="country")
        .reset_index()
        .melt(
            id_vars="country",
            value_vars=years,
            var_name="year",
            value_name="ratio"
        )
    )
    gdp_ratios["year"] = gdp_ratios["year"].astype(int)
    
    return gdp_ratios

def impute_sales(train_df, gdp_ratios, reference_country="Norway"):
    """Impute missing sales data using GDP ratios."""
    # Create copy and extract year
    df_imputed = train_df.copy()
    df_imputed['date'] = pd.to_datetime(df_imputed['date'])
    df_imputed["year"] = df_imputed["date"].dt.year
    
    # Define imputation configurations
    configs = [
        # (country, store, product, impute_all)
        ("Canada", "Discount Stickers", "Holographic Goose", True),
        ("Canada", "Premium Sticker Mart", "Holographic Goose", False),
        ("Canada", "Stickers for Less", "Holographic Goose", False),
        ("Kenya", "Discount Stickers", "Holographic Goose", True),
        ("Kenya", "Premium Sticker Mart", "Holographic Goose", False),
        ("Kenya", "Stickers for Less", "Holographic Goose", False),
        ("Kenya", "Discount Stickers", "Kerneler", False)
    ]
    
    def get_ratio(year, country):
        """Get GDP ratio for a specific country and year."""
        mask = (gdp_ratios["year"] == year) & (gdp_ratios["country"] == country)
        return gdp_ratios.loc[mask, "ratio"].iloc[0]
    
    def create_mask(df, country, store, product, year, dates=None):
        """Create boolean mask for filtering dataframe."""
        mask = (
            (df["country"] == country) &
            (df["store"] == store) &
            (df["product"] == product) &
            (df["year"] == year)
        )
        if dates is not None:
            mask &= df["date"].isin(dates)
        return mask
    
    # Perform imputation
    for year in df_imputed["year"].unique():
        target_ratio = get_ratio(year, reference_country)
        
        for country, store, product, impute_all in configs:
            # Calculate country's ratio relative to reference
            current_ratio = get_ratio(year, country)
            ratio = current_ratio / target_ratio
            
            # Get missing dates if not imputing all
            missing_dates = None
            if not impute_all:
                current_data = df_imputed[
                    create_mask(df_imputed, country, store, product, year)
                ]
                missing_dates = current_data.loc[
                    current_data["num_sold"].isna(), 
                    "date"
                ]
            
            # Create masks for target and reference data
            target_mask = create_mask(
                df_imputed, country, store, product, year, missing_dates
            )
            ref_mask = create_mask(
                df_imputed, reference_country, store, product, year, missing_dates
            )
            
            # Impute values
            df_imputed.loc[target_mask, "num_sold"] = (
                df_imputed.loc[ref_mask, "num_sold"] * ratio
            ).values
    
    # Handle remaining missing values
    specific_imputations = {
        23719: 4,
        207003: 195
    }
    for id_val, value in specific_imputations.items():
        df_imputed.loc[df_imputed["id"] == id_val, "num_sold"] = value
    
    return df_imputed

# Define Years
years = [str(year) for year in range(2010, 2021)]

# Process GDP ratios
gdp_ratios = prepare_gdp_ratios(gdp_per_capita_df, train_df, years)

# Perform imputation
train_df = impute_sales(train_df, gdp_ratios)

print(f"Missing values remaining: {train_df['num_sold'].isna().sum()}")


train_df.head()


def get_string_columns(df):
    # Get the string column names
    string_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Display the string column names
    print("String column names:")
    print(string_columns)
    return string_columns

string_columns = get_string_columns(train_df)


# Correct the data type for date
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

def get_season(row):
    """
    Determine season based on country and date
    
    Returns:
    str: Season name
    """
    month = row['month']
    country = row['country']
    day = row['date'].day
    
    # Countries in Northern Hemisphere with four distinct seasons
    northern_four_seasons = ['Canada', 'Finland', 'Norway']
    
    # Countries in Northern Hemisphere with Mediterranean climate
    mediterranean = ['Italy']
    
    # Countries near the equator
    tropical = ['Singapore', 'Kenya']
    
    # For countries with four distinct seasons in Northern Hemisphere
    if country in northern_four_seasons:
        # Meteorological seasons
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:  # month in [9, 10, 11]
            return 'Autumn'
            
    # For Mediterranean countries
    elif country in mediterranean:
        if month in [12, 1, 2]:
            return 'Mild Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Hot Summer'
        else:  # month in [9, 10, 11]
            return 'Autumn'
            
    # For tropical countries
    elif country in tropical:
        if country == 'Singapore':
            if month in [11, 12, 1]:
                return 'Northeast Monsoon'
            elif month in [2, 3, 4]:
                return 'Inter-monsoon'
            elif month in [5, 6, 7, 8, 9]:
                return 'Southwest Monsoon'
            else:  # month in [10]
                return 'Inter-monsoon'
        
        # Kenya seasonal patterns
        elif country == 'Kenya':
            if month in [12, 1, 2]:
                return 'Hot and Dry'
            elif month in [3, 4, 5]:
                return 'Long Rains'
            elif month in [6, 7, 8, 9]:
                return 'Cool and Dry'
            else:  # month in [10, 11]
                return 'Short Rains'
    
    return 'Unknown'

# Apply Seasons to the DataFrame
def assign_seasons(df):
    """
    Assign seasons to a DataFrame containing country and date columns
    """
    df['season'] = df.apply(get_season, axis=1)
    return df
    
def generate_new_features(df):
    # Extract month, year, and weekday (0 for Monday, 6 for Sunday)
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['weekday'] = df['date'].dt.weekday
    df['weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    
    df['month_sin'] = np.sin(2 * np.pi * df[f'month'] / 12) 
    df['month_cos'] = np.cos(2 * np.pi * df[f'month'] / 12)
    
    df['product_store'] = df['product'] + "_" + df['store']
    df['country_product'] = df['country'] + "_" + df['product']
    df['country_store'] = df['country'] + "_" + df['store']
    df['product_weekday'] = df['product'] + "_" + str(df['weekday'])
    df = assign_seasons(df)
    
    ###############################################################################################
    #### These features are taken from this notebook : 
    #### https://www.kaggle.com/code/cabaxiom/s5e1-eda-and-linear-regression-baseline#Modeling
    ###############################################################################################
    df["day_of_year"] = df['date'].apply(
        lambda x: x.timetuple().tm_yday if not (x.is_leap_year and x.month > 2) else x.timetuple().tm_yday - 1
    )

    df["important_dates"] = df["day_of_year"].apply(lambda x: x if x in [1,2,3,4,5,6,7,8,9,10,99, 100, 101, 125,126,355,256,357,358,359,360,361,362,363,364,365] else 0)
    df["year"] = df["date"].dt.year - 2010
    ################################################################################################

    df = df.drop(columns=["date", "day_of_year"])

    return df

train_df = generate_new_features(train_df)
test_df = generate_new_features(test_df)

categorical_columns = get_string_columns(train_df)


# Change all the values to lowercase.
train_df[categorical_columns] = train_df[categorical_columns].map(lambda x: x.lower() if isinstance(x, str) else x)
test_df[categorical_columns] = test_df[categorical_columns].map(lambda x: x.lower() if isinstance(x, str) else x)


def update(df):
    
    global categorical_columns

    for c in categorical_columns:
        df[c] = df[c].fillna('None')

    j_ch=',[]{}:"\\<'
    for ch in j_ch:
        for c in categorical_columns:
            df[c] = df[c].apply(lambda x:str(x).replace(ch,''))
                
    return df

train_df = update(train_df)
test_df = update(test_df)


# Convert all object type columns to category in train set
train_df[categorical_columns] = train_df[categorical_columns].astype('category')

# Convert all object type columns to category in test set
test_df[categorical_columns] = test_df[categorical_columns].astype('category')


# Count unique categories in each categorical column
unique_categories_counts = {col: train_df[col].nunique() for col in categorical_columns}

# Convert the counts to a pandas Series for easy plotting
unique_categories_series = pd.Series(unique_categories_counts)

# Plotting the unique category counts as a horizontal bar plot
plt.figure(figsize=(12, 8))
ax = sns.barplot(x=unique_categories_series.values, y=unique_categories_series.index, palette='viridis')

# Adding labels on the side of each bar
for i in ax.containers:
    ax.bar_label(i, label_type='edge')

# Displaying grid lines
plt.grid(True, which='both', axis='x', linestyle='--', linewidth=0.7)

plt.title('Number of Unique Categories in Categorical Columns')
plt.xlabel('Number of Unique Categories')
plt.ylabel('Categorical Columns')
plt.show()


train_df.head()


# Drop rows with null values in train_df
train_df = train_df.dropna().reset_index(drop = True)

print(train_df.shape)


columns_to_remove = ['id', 'num_sold']
X = train_df.drop(columns_to_remove, axis = 1)
y = train_df['num_sold']
group_by = train_df['year']
X_test = test_df.drop(columns_to_remove[:-1], axis = 1)


# Define parameters
num_estimators = 10000
params = {
    'n_estimators': num_estimators,
    'metric': 'mape',
    'boosting_type': 'gbdt',
    'max_depth': 8,
    'learning_rate': 0.03,
    'lambda_l1': 0.001,
    'lambda_l2': 0.01,
    'random_state': 42,
    'verbose': -1
}


def run_lightgbm_cv(X, y, test_df=None, n_splits=5, group_by = None, random_state=42):
    """
    Run cross-validation for LightGBM regression
    """
    # Initialize KFold
    kf = GroupKFold(n_splits=n_splits)
    
    # Initialize lists to hold scores
    mape_scores = []
    
    # Initialize test predictions if test set is provided
    if test_df is not None:
        test_preds = np.zeros(len(test_df))
    
    # Run cross-validation
    for fold, (train_index, val_index) in enumerate(kf.split(X, y, groups=group_by), start=1):
        print(f"\nFold {fold}/{n_splits}")
        print("=" * 50)
        
        # Split data for this fold
        X_train_fold = X.iloc[train_index]
        X_val_fold = X.iloc[val_index]
        y_train_fold = y.iloc[train_index]
        y_val_fold = y.iloc[val_index]
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['Train', 'Valid'],
            callbacks=[
                lgb.log_evaluation(250),
                lgb.early_stopping(500)
            ]
        )
        
        # Predict on validation set
        val_preds = model.predict(X_val_fold, num_iteration = model.best_iteration)
        
        # Calculate MAPE for this fold
        fold_mape = mean_absolute_percentage_error(y_val_fold, val_preds)
        mape_scores.append(fold_mape)
        
        # Predict on test set if provided
        if test_df is not None:
            test_preds += model.predict(test_df, num_iteration = model.best_iteration) / n_splits
    
    # Print overall results
    print("\n" + "=" * 50)
    print(f"Average MAPE across folds: {np.mean(mape_scores):.4f}")
    print(f"Std of MAPE across folds: {np.std(mape_scores):.4f}")
    
    # Return results
    results = {
        'fold_mape': mape_scores,
        'mean_mape': np.mean(mape_scores),
        'std_mape': np.std(mape_scores)
    }
    
    if test_df is not None:
        results['test_predictions'] = test_preds
    
    return results

# Run cross-validation
results = run_lightgbm_cv(X, y, test_df=X_test, group_by = group_by)


submission_df['num_sold'] = results['test_predictions']
submission_df.to_csv('submission.csv', index = False)
submission_df.head()

