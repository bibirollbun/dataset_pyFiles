import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


train_df=pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/train.csv")
test_df=pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/test.csv")
val_df=pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/val.csv")


print(train_df.shape,test_df.shape,val_df.shape)


train_df = pd.concat([train_df, val_df], ignore_index=True)


train_df.head()


train_df .isna().sum()


train_df=train_df.fillna("Unknown")


train_df.describe()


train_df.dtypes


train_df.nunique()


# Select numeric columns
numeric_cols = train_df.select_dtypes(include=['number']).columns

# Compute correlation with target
correlations = train_df[numeric_cols].corr()['Lap_Time_Seconds'].drop('Lap_Time_Seconds').sort_values(ascending=False)

# Display correlations
print("ğŸ“Š Correlation of numeric features with Lap_Time_Seconds:\n")
print(correlations)


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





