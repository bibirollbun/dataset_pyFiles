import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


# Load the datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


# Overview of the data
print("Train Data Head:")
train_df.head().style.background_gradient(cmap='plasma')


print("\nTest Data Head:")
test_df.head().style.background_gradient(cmap='plasma')


print("\nSubmission Data Head:")
sub.head().style.background_gradient(cmap='plasma')


print("\nTrain Data Info:")
train_df.info()


print("\nTest Data Info:")
test_df.info()


print("\nTrain Data Describe:")
train_df.describe().style.background_gradient(cmap='tab20c')



print("\nTest Data Describe:")
test_df.describe().style.background_gradient(cmap='tab20c')


# Rainfall distribution (Target variable)
plt.figure(figsize=(8, 6))
sns.countplot(x='rainfall', data=train_df)
plt.title('Rainfall Distribution')
plt.show()

# Correlation heatmap of features
plt.figure(figsize=(12, 10))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Time series plot of rainfall (if 'day' is an actual time component - requires assumption)
plt.figure(figsize=(15, 6))
plt.plot(train_df['day'], train_df['rainfall'])
plt.xlabel('Day')
plt.ylabel('Rainfall')
plt.title('Rainfall Time Series')
plt.show()


# --- "Lazy Prediction" Model:  Using Average Rainfall ---

# Method: Predict the average rainfall from the training data for all test days.
# This is a baseline model and will likely perform poorly, but it's quick.

# Calculate average rainfall
avg_rainfall = train_df['rainfall'].mean()

# Predict average rainfall for all test data points
test_df['rainfall'] = avg_rainfall
sub['rainfall'] = avg_rainfall



# --- Evaluation (ROC AUC) and Visualization ---

#Create a validation split
X = train_df.drop('rainfall', axis=1)
y = train_df['rainfall']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Predict rainfall on validation set
y_pred_val = np.full(len(X_val), avg_rainfall) #Predict with our "model"

# Calculate ROC AUC
roc_auc = roc_auc_score(y_val, y_pred_val)
print(f"ROC AUC: {roc_auc}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_val, y_pred_val)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line (random guessing)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# --- Enhanced "Lazy Prediction" (considering day of year) ----

# Try to make it *slightly* less lazy:  Calculate the average rainfall *for each day*

daily_avg = train_df.groupby('day')['rainfall'].mean()

# Handle missing days in the test set (if any)
# This is critical.  If a day exists in test but not train, it will cause errors.
for day in test_df['day'].unique():
    if day not in daily_avg.index:
        daily_avg[day] = train_df['rainfall'].mean()  # Use overall average if day is missing


# Predict based on the daily average
def predict_rainfall(day):
    return daily_avg[day]

test_df['rainfall'] = test_df['day'].apply(predict_rainfall)
sub['rainfall'] = test_df['rainfall']


# --- Re-Evaluation (ROC AUC) and Visualization with Enhanced Prediction ---

# Predict rainfall on validation set using the enhanced prediction
def predict_rainfall_val(day):
    if day in daily_avg.index:
        return daily_avg[day]
    else:
        return train_df['rainfall'].mean()  #Use overall average if day is missing

y_pred_val = X_val['day'].apply(predict_rainfall_val)

# Calculate ROC AUC
roc_auc = roc_auc_score(y_val, y_pred_val)
print(f"Enhanced ROC AUC: {roc_auc}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_val, y_pred_val)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line (random guessing)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Enhanced Prediction)')
plt.legend()
plt.show()



# --- Submission ---
submission = sub.copy()  # Create a copy to avoid modifying the original

submission.to_csv('submission.csv', index=False)

# Show submission head
print("\nSubmission Head:")
submission.head()




