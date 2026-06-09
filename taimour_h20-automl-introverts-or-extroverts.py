import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from h2o.estimators.gbm import H2OGradientBoostingEstimator
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')

# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
train.head()


plt.figure(figsize=(10,6))
sns.heatmap(train.isnull().T, cmap='viridis', cbar=False)
plt.title('Missing Values Heatmap')
plt.show()


# Drop ID column
train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)

# Separate features and target
X = train.drop(columns=['Personality'])
y = train['Personality']

print("Feature-Target Split Complete")


sns.countplot(x=y)
plt.title('Class Distribution (Personality)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# Initialize H2O
h2o.init()

# Convert to H2O Frames
train_data = h2o.H2OFrame(train)
test_data = h2o.H2OFrame(test)

# Configure AutoML
aml = H2OAutoML(
    max_runtime_secs=130,
    seed=5,
    project_name="personality_prediction"
)

# Train models
aml.train(y='Personality', training_frame=train_data)


# Display top models
lb = aml.leaderboard
lb.head(10)


# Get leaderboard as pandas DataFrame
lb = aml.leaderboard.as_data_frame()

# Plot model performance comparison
plt.figure(figsize=(12, 6))
sns.barplot(x='mse', y='model_id', data=lb.head(10))
plt.title('Top 10 Models by Performance')
plt.xlabel('Validation Metric')
plt.ylabel('Model')
plt.show()


# Get best model
best_model = aml.leader

# Model performance metrics
perf = best_model.model_performance(train_data)
print(f"Model Accuracy: {perf.accuracy()[0][1]:.4f}")


gbm = H2OGradientBoostingEstimator()
gbm.train(y='Personality', training_frame=train_data)
gbm.varimp_plot(num_of_features=10)


cm = perf.confusion_matrix()

# 2. Print the confusion matrix
# Printing the object gives a well-formatted table.
print("Confusion Matrix on Training Data:")
print(cm)


cm_df = cm.table.as_data_frame()
cm_df.set_index(cm_df.columns[0], inplace=True)

# Iterate through each column and convert to numeric, coercing errors to NaN
for col in cm_df.columns:
    cm_df[col] = pd.to_numeric(cm_df[col], errors='coerce')
# Fill any NaN values that resulted from coercion with -1 or an appropriate value
# cm_df.fillna(-1, inplace=True)

# Create the heatmap plot
plt.figure(figsize=(10, 7)) # Set the figure size for better readability
heatmap = sns.heatmap(
    cm_df,
    annot=True,      # Display the numbers in each cell
    fmt='g',         # Use general format for numbers (no scientific notation)
    cmap='Blues',    # Use a blue color map
    linewidths=.5,   # Add lines between cells
    linecolor='black' # Set line color to black
)

# Add titles and labels for clarity
heatmap.set_title('Confusion Matrix Heatmap', fontdict={'size':18}, pad=12)
heatmap.set_xlabel('Predicted Label', fontsize=14)
heatmap.set_ylabel('Actual Label', fontsize=14)

# 4. Display the plot
plt.show()


# Make predictions
predictions = best_model.predict(test_data)
predictions_df = predictions.as_data_frame()

# Create submission
sub['Personality'] = predictions_df['predict'].values
sub.to_csv('submission.csv', index=False)
sub.head()


sns.countplot(x=predictions_df['predict'])
plt.title('Prediction Distribution')
plt.show()

