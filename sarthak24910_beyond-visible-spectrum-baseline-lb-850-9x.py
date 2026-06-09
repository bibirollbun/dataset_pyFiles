import pandas as pd


# Load train and test CSVs
train_df = pd.read_csv("/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/train.csv")
test_df = pd.read_csv("/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/test.csv")


# Compute the mean of the target column
mean_target = train_df['label'].mean()

# Create a prediction column with the mean value for all test samples
test_df['TARGET'] = mean_target

# Prepare submission with renamed ID column
submission = test_df[['id', 'TARGET']].rename(columns={'id': 'ID'})
submission.to_csv('submission.csv', index=False)

print("submission.csv created successfully!")

