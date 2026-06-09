# Data manipulation & visualization libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for clean output
import warnings
warnings.filterwarnings("ignore")


# Load the transfer plate CSV file
transfer_df = pd.read_csv('/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/transfer_plate.csv')

# Display the first few rows
transfer_df.head()


# Select the relevant target columns
target_cols = ['Glucose (g/L)', 'Sodium Acetate (g/L)', 'Magnesium Acetate (g/L)']
targets = transfer_df[target_cols]

# Check for any missing values
print("Missing values per column:")
print(targets.isnull().sum())


# Calculate the mean value for each compound
means = targets.mean()

# Display the results
print(" Mean Concentration Values:")
print(means)


# Create a bar plot of the mean values
plt.figure(figsize=(8, 5))
bar_colors = ['#5DADE2', '#F5B041', '#58D68D']

bars = plt.bar(means.index, means.values, color=bar_colors)

# Add labels and annotations
plt.title('Average Concentrations of Compounds', fontsize=14)
plt.ylabel('Mean Value (g/L)', fontsize=12)
plt.xticks(rotation=15)

# Annotate each bar with the value
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}',
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()


# Load the sample submission file
submission = pd.read_csv('/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/sample_submission.csv')

# Preview before filling
submission.head()


# Use the mean values for all rows in the submission
submission['Glucose'] = means['Glucose (g/L)']
submission['Sodium Acetate'] = means['Sodium Acetate (g/L)']
submission['Magnesium Sulfate'] = means['Magnesium Acetate (g/L)']  # Assumes acetate → sulfate match

# Preview the updated submission
submission.head()


# Export the DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)

# Final confirmation
print(" Submission file 'submission.csv' has been saved successfully!")

