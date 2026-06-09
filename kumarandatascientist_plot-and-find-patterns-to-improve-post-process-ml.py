import pandas as pd
no_model = pd.read_csv('/kaggle/input/private-1/submission (19).csv') # 0.05415
linear_reg=pd.read_csv('/kaggle/input/blending-forecast/linear_reg_predict.csv') 
lgbm_reg = pd.read_csv('/kaggle/input/blending-forecast/lgbm_predict.csv') 


blended = no_model.copy()

blended['num_sold'] = (
    (0.11) * lgbm_reg['num_sold'] +
    (0.09) * linear_reg['num_sold'] +
    (0.80) * no_model['num_sold'] 
)
# Save the blended results
blended.to_csv('submission.csv', index=False)

blended.head(10)

blended.shape


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the submission file
submission_file_path = 'submission.csv'
df = pd.read_csv(submission_file_path)

# Take a few sample data points
sample_df = df.sample(98550, random_state=42)  # Adjust the sample size as needed

# Plot the sample data
plt.figure(figsize=(10, 6))
sns.lineplot(x='id', y='num_sold', data=sample_df)
plt.xticks(rotation=45)
plt.xlabel('ID')
plt.ylabel('Number Sold')
plt.title('Sample Data: ID vs Number Sold')
plt.show()

# Identify potential improvements based on the plots
# For example: Look for patterns, outliers, or inconsistencies



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define a threshold for outliers (e.g., 95th percentile)
threshold = np.percentile(df['num_sold'], 95)

# Identify and filter outliers
outliers = df[df['num_sold'] > threshold]
non_outliers = df[df['num_sold'] <= threshold]

# Apply Min-Max Normalization
df['num_sold_normalized'] = (df['num_sold'] - df['num_sold'].min()) / (df['num_sold'].max() - df['num_sold'].min())

# Plot the original and normalized data
plt.figure(figsize=(14, 6))
sns.lineplot(x='id', y='num_sold', data=df.sample(2000, random_state=42), label='Original Data')
sns.lineplot(x='id', y='num_sold_normalized', data=df.sample(2000, random_state=42), label='Normalized Data')
plt.axhline(threshold, color='red', linestyle='--', label=f'Outlier Threshold ({threshold:.2f})')
plt.xticks(rotation=45)
plt.xlabel('ID')
plt.ylabel('Number Sold')
plt.title('Original and Normalized Data: ID vs Number Sold')
plt.legend()
plt.show()

# Display the outliers for further inspection
outliers.head()



# Replace outliers in the `num_sold` column with the mean value
mean_value = df['num_sold'].mean()
df.loc[df['num_sold'] > threshold, 'num_sold'] = mean_value

# Normalize the `num_sold` column after replacement
df['num_sold_normalized'] = (df['num_sold'] - df['num_sold'].min()) / (df['num_sold'].max() - df['num_sold'].min())

# Save the updated DataFrame to a new CSV file
final_submission_path = 'final_submission.csv'
df.to_csv(final_submission_path, index=False)

final_submission_path



# Plot the data after replacing outliers and normalization
plt.figure(figsize=(14, 6))
sns.lineplot(x='id', y='num_sold', data=df.sample(2000, random_state=42), label='After Outlier Replacement')
sns.lineplot(x='id', y='num_sold_normalized', data=df.sample(2000, random_state=42), label='Normalized Data')
plt.xlabel('ID')
plt.ylabel('Number Sold')
plt.title('Data After Outlier Replacement and Normalization: ID vs Number Sold')
plt.legend()
plt.show()



df.shape


df.head()


final_df = df[['id', 'num_sold']]

# Save the updated DataFrame to a new CSV file
final_submission_reduced_path = 'submission.csv'
final_df.to_csv(final_submission_reduced_path, index=False)

final_submission_reduced_path


final_df.shape


final_df.head()




