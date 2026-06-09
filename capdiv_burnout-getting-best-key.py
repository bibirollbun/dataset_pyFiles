import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


train_df=pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/train.csv")
test_df=pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/test.csv")
val_df=pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/val.csv")


train_df = pd.concat([train_df, val_df], ignore_index=True)


key_cols = ['Rider_ID', 'Bike', 'Team', 'Circuit_name']


train_df.dtypes


# Convert key columns to string
key_cols = ['rider', 'bike', 'team', 'circuit_name']

for col in key_cols:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

# Create composite key
train_df['key'] = train_df[key_cols].agg('_'.join, axis=1)
test_df['key'] = test_df[key_cols].agg('_'.join, axis=1)



# Step 1: Compute average Lap_Time_Seconds for each key from train_df
key_to_avg_lap_time = train_df.groupby('key')['Lap_Time_Seconds'].mean()

# Step 2: Map this to test_df based on the key
test_df['calc_lap_time'] = test_df['key'].map(key_to_avg_lap_time)

# Step 3: Count NaN values
num_nan = test_df['calc_lap_time'].isna().sum()

print(f"â�Œ Number of NaN values in test_df['calc_lap_time']: {num_nan}")



train_df['calc_lap_time'] = train_df['key'].map(key_to_avg_lap_time)

# Step 3: Compute RMSE between predicted and actual
rmse_key_mean = mean_squared_error(train_df['Lap_Time_Seconds'], train_df['calc_lap_time'], squared=False)

# Step 4: Print result
print(f"ğŸ“� RMSE using group mean by key in train_df: {rmse_key_mean:.4f} seconds")


# Convert key columns to string
key_cols = ['rider', 'bike', 'team', 'circuit_name','sequence']

for col in key_cols:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

# Create composite key
train_df['key'] = train_df[key_cols].agg('_'.join, axis=1)
test_df['key'] = test_df[key_cols].agg('_'.join, axis=1)



# Step 1: Compute average Lap_Time_Seconds for each key from train_df
key_to_avg_lap_time = train_df.groupby('key')['Lap_Time_Seconds'].mean()

# Step 2: Map this to test_df based on the key
test_df['calc_lap_time'] = test_df['key'].map(key_to_avg_lap_time)

# Step 3: Count NaN values
num_nan = test_df['calc_lap_time'].isna().sum()

print(f"â�Œ Number of NaN values in test_df['calc_lap_time']: {num_nan}")



train_df['calc_lap_time'] = train_df['key'].map(key_to_avg_lap_time)

# Step 3: Compute RMSE between predicted and actual
rmse_key_mean = mean_squared_error(train_df['Lap_Time_Seconds'], train_df['calc_lap_time'], squared=False)

# Step 4: Print result
print(f"ğŸ“� RMSE using group mean by key in train_df: {rmse_key_mean:.4f} seconds")


# Convert key columns to string
key_cols = ['rider', 'bike', 'team', 'circuit_name','sequence','year_x']

for col in key_cols:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

# Create composite key
train_df['key'] = train_df[key_cols].agg('_'.join, axis=1)
test_df['key'] = test_df[key_cols].agg('_'.join, axis=1)



# Step 1: Compute average Lap_Time_Seconds for each key from train_df
key_to_avg_lap_time = train_df.groupby('key')['Lap_Time_Seconds'].mean()

# Step 2: Map this to test_df based on the key
test_df['calc_lap_time'] = test_df['key'].map(key_to_avg_lap_time)

# Step 3: Count NaN values
num_nan = test_df['calc_lap_time'].isna().sum()

print(f"â�Œ Number of NaN values in test_df['calc_lap_time']: {num_nan}")



train_df['calc_lap_time'] = train_df['key'].map(key_to_avg_lap_time)

# Step 3: Compute RMSE between predicted and actual
rmse_key_mean = mean_squared_error(train_df['Lap_Time_Seconds'], train_df['calc_lap_time'], squared=False)

# Step 4: Print result
print(f"ğŸ“� RMSE using group mean by key in train_df: {rmse_key_mean:.4f} seconds")


# Convert key columns to string
key_cols = ['rider', 'bike', 'team', 'circuit_name','sequence','year_x','category_x']

for col in key_cols:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

# Create composite key
train_df['key'] = train_df[key_cols].agg('_'.join, axis=1)
test_df['key'] = test_df[key_cols].agg('_'.join, axis=1)



# Step 1: Compute average Lap_Time_Seconds for each key from train_df
key_to_avg_lap_time = train_df.groupby('key')['Lap_Time_Seconds'].mean()

# Step 2: Map this to test_df based on the key
test_df['calc_lap_time'] = test_df['key'].map(key_to_avg_lap_time)

# Step 3: Count NaN values
num_nan = test_df['calc_lap_time'].isna().sum()

print(f"â�Œ Number of NaN values in test_df['calc_lap_time']: {num_nan}")



train_df['calc_lap_time'] = train_df['key'].map(key_to_avg_lap_time)

# Step 3: Compute RMSE between predicted and actual
rmse_key_mean = mean_squared_error(train_df['Lap_Time_Seconds'], train_df['calc_lap_time'], squared=False)

# Step 4: Print result
print(f"ğŸ“� RMSE using group mean by key in train_df: {rmse_key_mean:.4f} seconds")


# Convert key columns to string
key_cols = ['rider', 'bike', 'team', 'circuit_name','sequence','year_x','category_x','Tire_Compound_Front','Tire_Compound_Rear']

for col in key_cols:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

# Create composite key
train_df['key'] = train_df[key_cols].agg('_'.join, axis=1)
test_df['key'] = test_df[key_cols].agg('_'.join, axis=1)



# Step 1: Compute average Lap_Time_Seconds for each key from train_df
key_to_avg_lap_time = train_df.groupby('key')['Lap_Time_Seconds'].mean()

# Step 2: Map this to test_df based on the key
test_df['calc_lap_time'] = test_df['key'].map(key_to_avg_lap_time)

# Step 3: Count NaN values
num_nan = test_df['calc_lap_time'].isna().sum()

print(f"â�Œ Number of NaN values in test_df['calc_lap_time']: {num_nan}")



train_df['calc_lap_time'] = train_df['key'].map(key_to_avg_lap_time)

# Step 3: Compute RMSE between predicted and actual
rmse_key_mean = mean_squared_error(train_df['Lap_Time_Seconds'], train_df['calc_lap_time'], squared=False)

# Step 4: Print result
print(f"ğŸ“� RMSE using group mean by key in train_df: {rmse_key_mean:.4f} seconds")


# Convert key columns to string
key_cols = ['rider', 'bike', 'team', 'circuit_name','sequence','year_x','category_x','position']

for col in key_cols:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

# Create composite key
train_df['key'] = train_df[key_cols].agg('_'.join, axis=1)
test_df['key'] = test_df[key_cols].agg('_'.join, axis=1)



# Step 1: Compute average Lap_Time_Seconds for each key from train_df
key_to_avg_lap_time = train_df.groupby('key')['Lap_Time_Seconds'].mean()

# Step 2: Map this to test_df based on the key
test_df['calc_lap_time'] = test_df['key'].map(key_to_avg_lap_time)

# Step 3: Count NaN values
num_nan = test_df['calc_lap_time'].isna().sum()

print(f"â�Œ Number of NaN values in test_df['calc_lap_time']: {num_nan}")



train_df['calc_lap_time'] = train_df['key'].map(key_to_avg_lap_time)

# Step 3: Compute RMSE between predicted and actual
rmse_key_mean = mean_squared_error(train_df['Lap_Time_Seconds'], train_df['calc_lap_time'], squared=False)

# Step 4: Print result
print(f"ğŸ“� RMSE using group mean by key in train_df: {rmse_key_mean:.4f} seconds")


# Convert key columns to string
key_cols = ['rider', 'bike', 'team', 'circuit_name','sequence','year_x','category_x','position','Grid_Position']

for col in key_cols:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

# Create composite key
train_df['key'] = train_df[key_cols].agg('_'.join, axis=1)
test_df['key'] = test_df[key_cols].agg('_'.join, axis=1)



# Step 1: Compute average Lap_Time_Seconds for each key from train_df
key_to_avg_lap_time = train_df.groupby('key')['Lap_Time_Seconds'].mean()

# Step 2: Map this to test_df based on the key
test_df['calc_lap_time'] = test_df['key'].map(key_to_avg_lap_time)

# Step 3: Count NaN values
num_nan = test_df['calc_lap_time'].isna().sum()

print(f"â�Œ Number of NaN values in test_df['calc_lap_time']: {num_nan}")



train_df['calc_lap_time'] = train_df['key'].map(key_to_avg_lap_time)

# Step 3: Compute RMSE between predicted and actual
rmse_key_mean = mean_squared_error(train_df['Lap_Time_Seconds'], train_df['calc_lap_time'], squared=False)

# Step 4: Print result
print(f"ğŸ“� RMSE using group mean by key in train_df: {rmse_key_mean:.4f} seconds")


# Convert key columns to string
key_cols = ['rider', 'bike', 'team', 'circuit_name','sequence','year_x','category_x','position','Track_Condition']

for col in key_cols:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

# Create composite key
train_df['key'] = train_df[key_cols].agg('_'.join, axis=1)
test_df['key'] = test_df[key_cols].agg('_'.join, axis=1)



# Step 1: Compute average Lap_Time_Seconds for each key from train_df
key_to_avg_lap_time = train_df.groupby('key')['Lap_Time_Seconds'].mean()

# Step 2: Map this to test_df based on the key
test_df['calc_lap_time'] = test_df['key'].map(key_to_avg_lap_time)

# Step 3: Count NaN values
num_nan = test_df['calc_lap_time'].isna().sum()

print(f"â�Œ Number of NaN values in test_df['calc_lap_time']: {num_nan}")



train_df['calc_lap_time'] = train_df['key'].map(key_to_avg_lap_time)

# Step 3: Compute RMSE between predicted and actual
rmse_key_mean = mean_squared_error(train_df['Lap_Time_Seconds'], train_df['calc_lap_time'], squared=False)

# Step 4: Print result
print(f"ğŸ“� RMSE using group mean by key in train_df: {rmse_key_mean:.4f} seconds")


# Convert key columns to string
key_cols = ['rider', 'bike', 'team', 'circuit_name','sequence','year_x','category_x','position','weather']

for col in key_cols:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

# Create composite key
train_df['key'] = train_df[key_cols].agg('_'.join, axis=1)
test_df['key'] = test_df[key_cols].agg('_'.join, axis=1)



# Step 1: Compute average Lap_Time_Seconds for each key from train_df
key_to_avg_lap_time = train_df.groupby('key')['Lap_Time_Seconds'].mean()

# Step 2: Map this to test_df based on the key
test_df['calc_lap_time'] = test_df['key'].map(key_to_avg_lap_time)

# Step 3: Count NaN values
num_nan = test_df['calc_lap_time'].isna().sum()

print(f"â�Œ Number of NaN values in test_df['calc_lap_time']: {num_nan}")



train_df['calc_lap_time'] = train_df['key'].map(key_to_avg_lap_time)

# Step 3: Compute RMSE between predicted and actual
rmse_key_mean = mean_squared_error(train_df['Lap_Time_Seconds'], train_df['calc_lap_time'], squared=False)

# Step 4: Print result
print(f"ğŸ“� RMSE using group mean by key in train_df: {rmse_key_mean:.4f} seconds")


# Convert key columns to string
key_cols = ['rider', 'bike', 'team', 'circuit_name','sequence','year_x','category_x','position','weather','track']

for col in key_cols:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

# Create composite key
train_df['key'] = train_df[key_cols].agg('_'.join, axis=1)
test_df['key'] = test_df[key_cols].agg('_'.join, axis=1)



# Step 1: Compute average Lap_Time_Seconds for each key from train_df
key_to_avg_lap_time = train_df.groupby('key')['Lap_Time_Seconds'].mean()

# Step 2: Map this to test_df based on the key
test_df['calc_lap_time'] = test_df['key'].map(key_to_avg_lap_time)

# Step 3: Count NaN values
num_nan = test_df['calc_lap_time'].isna().sum()

print(f"â�Œ Number of NaN values in test_df['calc_lap_time']: {num_nan}")



train_df['calc_lap_time'] = train_df['key'].map(key_to_avg_lap_time)

# Step 3: Compute RMSE between predicted and actual
rmse_key_mean = mean_squared_error(train_df['Lap_Time_Seconds'], train_df['calc_lap_time'], squared=False)

# Step 4: Print result
print(f"ğŸ“� RMSE using group mean by key in train_df: {rmse_key_mean:.4f} seconds")


train_df=pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/train.csv")
test_df=pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/test.csv")
val_df=pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/val.csv")


train_df = pd.concat([train_df, val_df], ignore_index=True)


# Convert key columns to string
key_cols = ['rider','circuit_name', 'year_x',
     'Corners_per_Lap', 'Pit_Stop_Duration_Seconds']

for col in key_cols:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

# Create composite key
train_df['key'] = train_df[key_cols].agg('_'.join, axis=1)
test_df['key'] = test_df[key_cols].agg('_'.join, axis=1)



# Step 1: Compute average Lap_Time_Seconds for each key from train_df
key_to_avg_lap_time = train_df.groupby('key')['Lap_Time_Seconds'].mean()

# Step 2: Map this to test_df based on the key
test_df['calc_lap_time'] = test_df['key'].map(key_to_avg_lap_time)

# Step 3: Count NaN values
num_nan = test_df['calc_lap_time'].isna().sum()

print(f"â�Œ Number of NaN values in test_df['calc_lap_time']: {num_nan}")



from sklearn.metrics import mean_squared_error

train_df['calc_lap_time'] = train_df['key'].map(key_to_avg_lap_time)

# Step 3: Compute RMSE between predicted and actual
rmse_key_mean = mean_squared_error(train_df['Lap_Time_Seconds'], train_df['calc_lap_time'], squared=False)

# Step 4: Print result
print(f"ğŸ“� RMSE using group mean by key in train_df: {rmse_key_mean:.4f} seconds")


# Step 1: Compute mean Lap_Time_Seconds for each key
key_to_avg_lap_time = train_df.groupby('key')['Lap_Time_Seconds'].mean()

# Step 2: Map to test_df
test_df['Lap_Time_Seconds'] = test_df['key'].map(key_to_avg_lap_time)

# Step 3: Handle missing keys (optional: fill with global mean or median)
global_mean = train_df['Lap_Time_Seconds'].mean()
test_df['Lap_Time_Seconds'] = test_df['Lap_Time_Seconds'].fillna(global_mean)

# Step 4: Create submission DataFrame
submission = test_df[['Unique ID', 'Lap_Time_Seconds']]

# Step 5: Save to CSV
submission.to_csv('submission.csv', index=False)

print("âœ… Submission file 'submission.csv' created.")



submission.isna().sum()


train_df=pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/train.csv")
test_df=pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/test.csv")
val_df=pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/val.csv")


train_df = pd.concat([train_df, val_df], ignore_index=True)


# Convert key columns to string
key_cols = ['rider','bike','shortname','circuit_name','year_x','Session','sequence','position','points','Corners_per_Lap','Pit_Stop_Duration_Seconds']

for col in key_cols:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

# Create composite key
train_df['key'] = train_df[key_cols].agg('_'.join, axis=1)
test_df['key'] = test_df[key_cols].agg('_'.join, axis=1)



# Step 1: Compute average Lap_Time_Seconds for each key from train_df
key_to_avg_lap_time = train_df.groupby('key')['Lap_Time_Seconds'].mean()

# Step 2: Map this to test_df based on the key
test_df['calc_lap_time'] = test_df['key'].map(key_to_avg_lap_time)

# Step 3: Count NaN values
num_nan = test_df['calc_lap_time'].isna().sum()

print(f"â�Œ Number of NaN values in test_df['calc_lap_time']: {num_nan}")



from sklearn.metrics import mean_squared_error


train_df['calc_lap_time'] = train_df['key'].map(key_to_avg_lap_time)

# Step 3: Compute RMSE between predicted and actual
rmse_key_mean = mean_squared_error(train_df['Lap_Time_Seconds'], train_df['calc_lap_time'], squared=False)

# Step 4: Print result
print(f"ğŸ“� RMSE using group mean by key in train_df: {rmse_key_mean:.4f} seconds")


train_df=pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/train.csv")
test_df=pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/test.csv")
val_df=pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/val.csv")


train_df = pd.concat([train_df, val_df], ignore_index=True)


import pandas as pd
from sklearn.metrics import mean_squared_error

# Define the key columns
original_key_cols = [
    'rider','circuit_name', 'year_x',
     'Corners_per_Lap', 'Pit_Stop_Duration_Seconds'
]

# Make sure all columns are strings
for col in original_key_cols:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

# Store results
results = []

# Loop through each column, remove one at a time
for col_to_remove in original_key_cols:
    reduced_key_cols = [col for col in original_key_cols if col != col_to_remove]

    # Create composite key
    train_df['key'] = train_df[reduced_key_cols].agg('_'.join, axis=1)
    test_df['key'] = test_df[reduced_key_cols].agg('_'.join, axis=1)

    # Group mean mapping
    key_to_avg = train_df.groupby('key')['Lap_Time_Seconds'].mean()

    # Map predictions
    train_df['calc_lap_time'] = train_df['key'].map(key_to_avg)
    test_df['calc_lap_time'] = test_df['key'].map(key_to_avg)

    # Metrics
    rmse = mean_squared_error(train_df['Lap_Time_Seconds'], train_df['calc_lap_time'], squared=False)
    nan_count = test_df['calc_lap_time'].isna().sum()

    results.append({
        'Removed_Column': col_to_remove,
        'NaN_Count_in_Test': nan_count,
        'Train_RMSE': rmse
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Show result
print(results_df.sort_values(by='Train_RMSE'))





