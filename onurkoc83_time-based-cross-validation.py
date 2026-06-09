import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Read the data
df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
df['date'] = pd.to_datetime(df['date'])

# Sort by date
df = df.sort_values('date')

# Categorical features
cat_features = ['country', 'product', 'store']



print(f"Shape before dropping NaN categories: {len(df)}")
# Calculate NaN percentage for each category combination
nan_percentages = df.groupby(cat_features)['num_sold'].apply(
    lambda x: (x.isna().sum() / len(x)) * 100
).reset_index(name='nan_percentage')

# Get category combinations to drop (those with NaN values)
categories_to_drop = nan_percentages[nan_percentages['nan_percentage'] > 0]


# Drop rows with these category combinations from the original dataframe
for _, row in categories_to_drop.iterrows():
    mask = True
    for feat in cat_features:
        mask &= (df[feat] == row[feat])
    df = df[~mask]


print(f"Shape after dropping NaN categories: {len(df)}")
categories_to_drop.sort_values('nan_percentage')



def create_date_features(data):

   # 1. Quarter (Ã‡eyrek)
   data['quarter'] = data['date'].dt.quarter

   # 2. Month (Ay)
   data['month'] = data['date'].dt.month 

   # 3. Day (GÃ¼n)
   data['day'] = data['date'].dt.day

   # 4. Day of week (HaftanÄ±n gÃ¼nÃ¼)
   data['day_of_week'] = data['date'].dt.dayofweek

   # 5. Day of year (YÄ±lÄ±n gÃ¼nÃ¼)
   data['day_of_year'] = data['date'].dt.dayofyear

   # 6. Week of month (AyÄ±n haftasÄ±)
   data['week_of_month'] = data['date'].dt.day.apply(lambda x: (x-1)//7 + 1)

   # 7. Week of year (YÄ±lÄ±n haftasÄ±)
   data['week_of_year'] = data['date'].dt.isocalendar().week

   # 8. Is weekend (Hafta sonu mu?)
   data['is_weekend'] = data['date'].dt.dayofweek.isin([5,6]).astype(int)

   # 9. Is month end (AyÄ±n son gÃ¼nÃ¼ mÃ¼?)
   data['is_month_end'] = data['date'].dt.is_month_end.astype(int)

   # 10. Year (YÄ±l)
   data['year'] = data['date'].dt.year
   
   date_feats = [ 
                 'quarter',
                 'month',
                 'day',
                 'day_of_week',
                 'day_of_year',
                 'week_of_month',
                 'week_of_year',
                 'is_weekend',
                 'is_month_end',
                 'year']
   
   return data, date_feats


from IPython.display import Image, display
display(Image(filename='/kaggle/input/expanding-window/expanding_window.png'))


# Label encode categorical features
label_encoders = {}
for feature in cat_features:
    label_encoders[feature] = LabelEncoder()
    df[f'{feature}_encoded'] = label_encoders[feature].fit_transform(df[feature])

# Update cat_features to use encoded versions
encoded_cat_features = [f'{feature}_encoded' for feature in cat_features]

# ANSI color codes
BLUE = '\033[94m'
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'  # Reset color formatting

# Function to create splits for a single category combination
def create_splits(category_data, n_splits=5):
    splits = []
    dates = category_data['date'].sort_values().unique()
    n_dates = len(dates)
    
    # Calculate validation size (approximately 20% of total data)
    val_size = int(n_dates * 0.2)
    
    # Calculate initial training size
    initial_train_size = int(n_dates * 0.4)  # Start with 40% of data
    
    # Calculate step size for sliding
    remaining_points = n_dates - initial_train_size - val_size
    if remaining_points < n_splits - 1:
        return []
        
    step_size = remaining_points // (n_splits - 1)
    
    for i in range(n_splits):
        train_end_idx = initial_train_size + (i * step_size)
        val_end_idx = train_end_idx + val_size
        
        if val_end_idx > n_dates:
            break
            
        train_dates = dates[:train_end_idx]
        val_dates = dates[train_end_idx:val_end_idx]
        
        splits.append((train_dates, val_dates))
    
    return splits

# Initialize lists to store results
all_mapes = []

# Get unique category combinations
category_combinations = df.groupby(cat_features).size().reset_index()[cat_features]

# Create 5 folds
for fold in range(5):
    print(f"\nFold {fold + 1}:")
    print()
    # Initialize lists to store train and validation indices for this fold
    train_indices = []
    val_indices = []
    
    # Process each category combination
    for idx, cat_combo in category_combinations.iterrows():
        # Get data for this category combination
        mask = True
        for feat, value in cat_combo.items():
            mask = mask & (df[feat] == value)
        category_data = df[mask].copy()
            
        # Create splits for this category
        splits = create_splits(category_data, n_splits=5)
        
        if splits and fold < len(splits):  # Only proceed if we have enough data for this fold
            train_dates, val_dates = splits[fold]
            
            # Add indices to our lists
            train_indices.extend(category_data[category_data['date'].isin(train_dates)].index)
            val_indices.extend(category_data[category_data['date'].isin(val_dates)].index)
    
    # Create train and validation sets
    train_data = df.loc[train_indices]
    val_data = df.loc[val_indices]
    
    # Print date ranges and shapes with colors
    print(f"{BLUE}Train date range: {train_data['date'].dt.date.min()} to {train_data['date'].dt.date.max()}")
    print(f"Val date range: {val_data['date'].dt.date.min()} to {val_data['date'].dt.date.max()}{RESET}")
    print()  # Add empty line between date ranges and shapes
    print(f"{RED}Train shape: {train_data.shape}")
    print(f"Val shape: {val_data.shape}{RESET}")
    print()  # Add empty line between shapes and MAPE
    
    # Apply the function to train and validation data
    train_data, date_feats = create_date_features(train_data)
    val_data, _ = create_date_features(val_data)
    
    # Update feature columns
    feature_cols = encoded_cat_features + date_feats

    X_train = train_data[feature_cols]
    X_val = val_data[feature_cols]
    y_train = train_data['num_sold']
    y_val = val_data['num_sold']
    
    # Train model
    model = LGBMRegressor(n_estimators=1000,
                         learning_rate=0.1,
                         boosting_type='gbdt',
                          verbosity=-1,
                         categorical_feature=encoded_cat_features,
                         random_state=42)
    
    model.fit(
        X_train, 
        y_train,
        categorical_feature=encoded_cat_features
    )
    
    # Make predictions
    val_preds = model.predict(X_val)
    
    # Calculate and print MAPE with color
    mape = mean_absolute_percentage_error(y_val, val_preds)
    all_mapes.append(mape)
    print(f"{GREEN}Fold {fold + 1} MAPE: {mape:.4f}{RESET}")
    print()  # Add empty line after MAPE

# Print average MAPE
valid_mapes = [m for m in all_mapes if not np.isnan(m)]
if valid_mapes:
    print(f"\n{GREEN}Average MAPE across all folds: {np.mean(valid_mapes):.4f}{RESET}")
else:
    print("\nNo valid MAPE scores calculated")

