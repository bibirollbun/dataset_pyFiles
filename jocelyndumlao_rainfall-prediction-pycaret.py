pip install pycaret pandas scikit-learn matplotlib seaborn


import pandas as pd
from pycaret.classification import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# Load the datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


# --- Data Exploration and Visualization ---

# 1. Overview of Data
print("Train Data Info:")
train_df.info()
print("\nTrain Data Describe:")
train_df.describe()


print("\nTest Data Info:")
test_df.info()
print("\nTest Data Describe:")
test_df.describe()


# 2. Visualizing Target Variable Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='rainfall', data=train_df)
plt.title('Distribution of Rainfall (Target Variable)')
plt.show()

# 3. Time Series Analysis (Day vs. Rainfall)
plt.figure(figsize=(15, 6))
sns.lineplot(x='day', y='rainfall', data=train_df)
plt.title('Rainfall Over Days (Time Series)')
plt.xlabel('Day')
plt.ylabel('Rainfall')
plt.show()

# 4. Climate Analysis (Temparature, Pressure, Humidity)

# Temparature Distribution
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(train_df['temparature'], kde=True)
plt.title('Temparature Distribution')

# Pressure Distribution
plt.subplot(1, 3, 2)
sns.histplot(train_df['pressure'], kde=True)
plt.title('Pressure Distribution')

# Humidity Distribution
plt.subplot(1, 3, 3)
sns.histplot(train_df['humidity'], kde=True)
plt.title('Humidity Distribution')
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

# 5. Weather Variable Correlation Analysis

correlation_matrix = train_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Matrix of Weather Variables')
plt.show()



!pip install --upgrade pandas
import pandas as pd
print(pd.__version__)



!pip install --upgrade pycaret


!pip install pandas==1.5.3
import pandas as pd
print(pd.__version__)



# --- PyCaret Setup ---

# Initialize PyCaret classification setup
try:
    s = setup(data=train_df, target='rainfall', session_id=42,
              imputation_type='simple', # Use simple imputation
              numeric_imputation='mean', # Impute numeric features with the mean
              normalize=True, # Apply normalization
              normalize_method='zscore', # Use zscore normalization
              remove_multicollinearity=True, # Remove highly correlated features
              multicollinearity_threshold=0.9, # Threshold for multicollinearity
              feature_selection=True # Enable feature selection
              )
except NameError as e:
    print(f"Error during setup: {e}")
    print("This likely means that PyCaret was not installed correctly or is missing dependencies.")
    raise
except TypeError as e:
        print(f"TypeError during setup: {e}")
        print("Check the arguments passed to setup() for compatibility with your PyCaret version.")
        raise


# --- Model Training and Selection ---

# Compare different models and select the best one based on AUC
try:
    best_model = compare_models(sort='AUC', n_select=1)

    # Print the best model
    print(best_model)
except NameError:
    print("PyCaret compare_models function not found. Check your PyCaret installation.")
    raise

# --- Model Evaluation and Visualization ---

# Evaluate the best model
try:
    evaluate_model(best_model)

    # Plot ROC curve
    # plot_model(best_model, plot='auc', display_format='streamlit') # set display format so there's no errors
    plot_model(best_model, plot='auc')
except NameError:
    print("PyCaret evaluation functions not found. Check your PyCaret installation.")
    raise
except ModuleNotFoundError as e:
    print(f"ModuleNotFoundError during plot_model: {e}")
    print("It seems a required module is not installed. Try installing it.")
    raise
except TypeError as e:
    print(f"TypeError during plot_model: {e}")
    print("Check the arguments passed to plot_model for compatibility with your PyCaret version and model type.")
    # Optionally, skip plotting the model to continue execution
    # If continuing, it's crucial to understand why plot_model failed
    pass


import pandas as pd

# --- Prediction and Submission ---
try:
    predictions = predict_model(best_model, data=test_df)
    print("Predictions Head:\n", predictions.head())
    print("Prediction Score Describe:\n", predictions['prediction_score'].describe())

    # Ensure 'sub' DataFrame is properly initialized
    sub = pd.DataFrame()
    sub['id'] = test_df['id'].values  # Ensure 'id' column is included
    sub['rainfall'] = predictions['prediction_score'].values  # Use the raw prediction score

    # Reset index and ensure columns are correctly formatted
    sub = sub.reset_index(drop=True)  # Reset index completely
    sub.columns = sub.columns.astype(str)  # Ensure column names are strings
    sub = sub.copy()  # Ensure no memory issues

    # Convert DataFrame to CSV safely
    sub.to_csv('submission.csv', index=False)

    # Display submission head
    print("\nSubmission File Head:")
    print(sub.head())

except NameError:
    print("PyCaret prediction functions not found. Check your PyCaret installation.")
    raise

except KeyError as e:
    print(f"KeyError: {e}")
    print("Ensure the test DataFrame has an 'id' column and predictions contain 'prediction_label'.")
    raise

except AttributeError as e:
    print(f"AttributeError: {e}")
    print("Resetting index and forcing reformatting.")
    sub = sub.copy()  # Force a new memory allocation
    sub.index = pd.RangeIndex(start=0, stop=len(sub), step=1)  # Reset index manually
    sub.columns = [str(col) for col in sub.columns]  # Ensure all column names are strings
    sub.to_csv('submission.csv', index=False)  # Retry saving

except Exception as e:
    print(f"An unexpected error occurred: {e}")
    raise




