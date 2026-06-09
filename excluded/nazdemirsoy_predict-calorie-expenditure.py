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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')

train_df.head()


train_df.info()


train_df.shape


#Converting Sex coulmn to Binary
train_df = pd.get_dummies(train_df, columns=['Sex'], drop_first=True)

train_df.head()


# Exploring relationship with Target and Features with the help of coefficient matrix
# Target = Calories

import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix
corr_matrix = train_df.corr(numeric_only=True)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()




#Preprocess for both train and test

#train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

X_full = train_df.drop(columns=["Calories"])
y_full = train_df["Calories"]

print(f"Training features shape: {X_full.shape}")
print(f"Training target shape: {y_full.shape}")

X_test_new = test_df.copy()
X_test_new = pd.get_dummies(X_test_new, columns=["Sex"], drop_first=True)

# Ensure test data has same columns as training data
X_test_new = X_test_new.reindex(columns=X_full.columns, fill_value=0)

print(f"Test features shape: {X_test_new.shape}")



# Split data for validation
from sklearn.model_selection import train_test_split

print("\nSplitting training data...")
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

print(f"Train set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test_new.shape[0]} samples")




from sklearn.preprocessing import StandardScaler

# CRITICAL - Fit scaler ONLY on training data
print("\nFitting scaler on TRAINING data only...")
scaler = StandardScaler()
scaler.fit(X_train) 

print("Scaler fitted. Feature means and stds calculated from training data only.")



# Transform ALL datasets using the SAME fitted scaler
print("\n Transforming all datasets with the SAME scaler...")
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_new)

print("âœ… All datasets transformed with consistent scaling")

# Let's verify the scaling is reasonable
print(f"\nScaling verification:")
print(f"Train scaled - mean: {X_train_scaled.mean():.6f}, std: {X_train_scaled.std():.6f}")
print(f"Val scaled - mean: {X_val_scaled.mean():.6f}, std: {X_val_scaled.std():.6f}")
print(f"Test scaled - mean: {X_test_scaled.mean():.6f}, std: {X_test_scaled.std():.6f}")



# Train the model
from sklearn.linear_model import LinearRegression

print("\nTraining Linear Regression model...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)



print("\nMaking predictions...")
y_train_pred = lr_model.predict(X_train_scaled)
y_val_pred = lr_model.predict(X_val_scaled)
y_test_pred = lr_model.predict(X_test_scaled)



# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print(f"\nğŸ“Š MODEL PERFORMANCE:")
print("="*50)

def evaluate_model(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"{dataset_name} Set:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  RÂ² Score: {r2:.4f}")
    return rmse, mae, r2

train_rmse, train_mae, train_r2 = evaluate_model(y_train, y_train_pred, "Training")
val_rmse, val_mae, val_r2 = evaluate_model(y_val, y_val_pred, "Validation")



plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(y_train, y_train_pred, alpha=0.6, s=1)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.title(f'Training: Actual vs Predicted\nRÂ² = {train_r2:.4f}')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.scatter(y_val, y_val_pred, alpha=0.6, s=1, color='orange')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.title(f'Validation: Actual vs Predicted\nRÂ² = {val_r2:.4f}')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.hist(y_test_pred, bins=50, alpha=0.7, color='green')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero line')
plt.xlabel('Predicted Calories')
plt.ylabel('Frequency')
plt.title('Test Predictions Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
import os
import zipfile
from IPython.display import FileLink, display

print(f"\nğŸ“� CREATING FINAL SUBMISSION:")
print("="*50)

# Apply safety clipping just in case
y_test_pred_safe = np.clip(y_test_pred, 0.1, None)

submission_final = pd.DataFrame({
    'id': test_df['id'],
    'Calories': y_test_pred_safe
})

print(f"Final submission stats:")
print(f"Shape: {submission_final.shape}")
print(f"Min calories: {submission_final['Calories'].min():.2f}")
print(f"Max calories: {submission_final['Calories'].max():.2f}")
print(f"Mean calories: {submission_final['Calories'].mean():.2f}")
print(f"Negative values: {np.sum(submission_final['Calories'] < 0)}")

# Save the submission FIRST
submission_filename = 'submission_final_corrected.csv'
submission_final.to_csv(submission_filename, index=False)
print(f"\nâœ… Saved final submission as '{submission_filename}'")

# Check what submission files exist AFTER creating the new one
print(f"\nğŸ“� Available submission files:")
submission_files = []
for filename in ['submission.csv', 'submission_final_corrected.csv', 'submission_corrected.csv']:
    if os.path.exists(filename):
        submission_files.append(filename)
        file_size = os.path.getsize(filename)
        print(f"âœ… Found: {filename} ({file_size:,} bytes)")

# Use the file we just created for ZIP
current_submission_file = 'submission_final_corrected.csv'

print(f"\nğŸ“¦ Creating ZIP file for easier download")
zip_filename = current_submission_file.replace('.csv', '.zip')

try:
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(current_submission_file)
    print(f"âœ… Created {zip_filename}")
    
    # Display download link for zip
    try:
        print(f"ğŸ“¥ Download links:")
        display(FileLink(current_submission_file))
        display(FileLink(zip_filename))
        print("ğŸ‘† Click the links above to download")
    except:
        print("Download links not available in this environment")
        
except Exception as e:
    print(f"â�Œ Error creating ZIP: {e}")

# Verify the file was created correctly
print(f"\nâœ… VERIFICATION:")
print("="*30)

if os.path.exists(current_submission_file):
    # Read and verify the saved file
    verification_df = pd.read_csv(current_submission_file)
    print(f"âœ… File exists: {current_submission_file}")
    print(f"âœ… File size: {os.path.getsize(current_submission_file):,} bytes")
    print(f"âœ… Shape: {verification_df.shape}")
    print(f"âœ… Columns: {list(verification_df.columns)}")
    print(f"âœ… Min calories: {verification_df['Calories'].min():.2f}")
    print(f"âœ… Max calories: {verification_df['Calories'].max():.2f}")
    print(f"âœ… No negative values: {sum(verification_df['Calories'] < 0) == 0}")
    
    print(f"\nğŸ“‹ Sample of final submission:")
    print(verification_df.head(10))
    
else:
    print(f"â�Œ File was not created: {current_submission_file}")

# Alternative download methods
print(f"\nğŸ”½ DOWNLOAD METHODS:")
print("="*50)

print(f"Method 1: Direct file access")
print(f"File location: {os.path.abspath(current_submission_file)}")

print(f"\nMethod 2: Google Colab (if applicable)")
try:
    from google.colab import files
    print("Downloading file in Google Colab...")
    files.download(current_submission_file)
    print("âœ… File downloaded in Colab")
except ImportError:
    print("Not in Google Colab environment")

print(f"\nMethod 3: Manual copy (backup)")
print("If download doesn't work, here's the file content structure:")
if os.path.exists(current_submission_file):
    with open(current_submission_file, 'r') as f:
        lines = f.readlines()
    print(f"Header: {lines[0].strip()}")
    print("Sample rows:")
    for i in range(1, min(6, len(lines))):
        print(f"  {lines[i].strip()}")
    print(f"... ({len(lines)-1} total data rows)")

print(f"\nğŸ�¯ READY FOR KAGGLE SUBMISSION!")
print(f"Upload file: {current_submission_file}")
print(f"Expected score: High (RÂ² â‰ˆ 0.968)")
print(f"No negative calories: âœ…")
print(f"Correct format: âœ…")




