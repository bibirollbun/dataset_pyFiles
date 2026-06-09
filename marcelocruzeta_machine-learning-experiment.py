# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
from sklearn.experimental import enable_iterative_imputer  # Needed to use IterativeImputer
from sklearn.impute import IterativeImputer  # For iterative imputation of missing values
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and validation sets
from sklearn.ensemble import RandomForestClassifier  # Importing the Random Forest classifier for model training
from sklearn.metrics import classification_report, accuracy_score  # For evaluating model performance metrics
from catboost import CatBoostClassifier  # Importing the CatBoost classifier for gradient boosting on decision trees
import xgboost as xgb  # Importing the XGBoost classifier
from collections import Counter  # Importing Counter for counting hashable objects, useful for tallying predictions
from tabulate import tabulate  # For displaying the DataFrame with borders
from colorama import Fore, Style  # Import Colorama for coloring text
from sklearn.tree import DecisionTreeClassifier  # Using a decision tree model
from IPython.display import display  # For displaying DataFrames nicely in Jupyter
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.decomposition import TruncatedSVD  # For dimensionality reduction
import matplotlib.pyplot as plt  # Ensure to import matplotlib for plotting



def preprocess(df):
    # Make a copy of the DataFrame to avoid modifying the original data
    df = df.copy()
    
    # Map categorical variables to numerical values
    df['Stage_fear'] = df['Stage_fear'].fillna('Unknown').map({'Yes': 1, 'No': 0, 'Unknown': -1})
    df['Drained_after_socializing'] = df['Drained_after_socializing'].fillna('Unknown').map({'Yes': 1, 'No': 0, 'Unknown': -1})
    
    # Select numeric columns for imputation
    num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    
    # Use IterativeImputer to fill missing numeric values
    imputer = IterativeImputer(random_state=42)
    df[num_cols] = imputer.fit_transform(df[num_cols])
    
    return df

# Load the datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# Preprocess the training dataset keeping the 'Personality' column
X = preprocess(train.drop(['id', 'Personality'], axis=1))  # Remove 'Personality' for features
y = train['Personality'].map({'Introvert': 0, 'Extrovert': 1})  # Define target variable

# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)  # Fit the model with features X and target y

# Preprocess the test dataset
X_test = preprocess(test.drop('id', axis=1))  # Similar preprocessing for the test data

# Predict the Personality values for the test dataset
predicted_personality = model.predict(X_test)

# Add the predictions to the X_test DataFrame
X_test['Personality'] = predicted_personality

# Display the first 10 rows of X with styled formatting
styled_X = X.head(10).style.set_table_attributes('style="border: 1px solid black; border-collapse: collapse;"') \
                    .set_table_styles([{
                        'selector': 'th',
                        'props': [('border', '1px solid black'), ('padding', '5px'), ('background-color', '#87CEFA'), ('color', 'black')]
                    }, {
                        'selector': 'td',
                        'props': [('border', '1px solid black'), ('padding', '5px')]
                    }])

# Display the first 10 rows of X_test with styled formatting
styled_X_test = X_test.head(10).style.set_table_attributes('style="border: 1px solid black; border-collapse: collapse;"') \
                              .set_table_styles([{
                                  'selector': 'th',
                                  'props': [('border', '1px solid black'), ('padding', '5px'), ('background-color', '#87CEFA'), ('color', 'black')]
                              }, {
                                  'selector': 'td',
                                  'props': [('border', '1px solid black'), ('padding', '5px')]
                              }])

# Display the styled DataFrames
display(styled_X)
display(styled_X_test)


def preprocess(df):
    # Make a copy of the DataFrame
    df = df.copy()
    
    # Fill categorical variables and map them to numeric values
    df['Stage_fear'] = df['Stage_fear'].fillna('Unknown').map({'Yes': 1, 'No': 0, 'Unknown': -1})
    df['Drained_after_socializing'] = df['Drained_after_socializing'].fillna('Unknown').map({'Yes': 1, 'No': 0, 'Unknown': -1})
    
    # Select numeric columns for imputation
    num_cols = ['Time_spent_Alone', 'Social_event_attendance',
                'Going_outside', 'Friends_circle_size',
                'Post_frequency']
    
    # Use IterativeImputer to fill missing numeric values
    imputer = IterativeImputer(random_state=42)
    df[num_cols] = imputer.fit_transform(df[num_cols])
    
    return df

# Load the datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# Preprocess training data
X = preprocess(train.drop(['id', 'Personality'], axis=1))  # Remove 'Personality'
y = train['Personality'].map({'Introvert': 0, 'Extrovert': 1})  # Target variable

# Preprocess test data
X_test = preprocess(test.drop('id', axis=1))

# Scale the data using MinMaxScaler or RobustScaler
# Option 1: MinMaxScaler
minmax_scaler = MinMaxScaler()
X_scaled = minmax_scaler.fit_transform(X)

# Scale test data based on training data scaling
X_test_scaled = minmax_scaler.transform(X_test)  

# Display the first 10 rows of scaled training and test data with styled formatting
styled_X_scaled = pd.DataFrame(X_scaled, columns=X.columns).head(10).style.set_table_attributes('style="border: 1px solid black; border-collapse: collapse;"') \
                    .set_table_styles([{
                        'selector': 'th',
                        'props': [('border', '1px solid black'), ('padding', '5px'), ('background-color', '#87CEFA'), ('color', 'black')]
                    }, {
                        'selector': 'td',
                        'props': [('border', '1px solid black'), ('padding', '5px')]
                    }])

styled_X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns).head(10).style.set_table_attributes('style="border: 1px solid black; border-collapse: collapse;"') \
                              .set_table_styles([{
                                  'selector': 'th',
                                  'props': [('border', '1px solid black'), ('padding', '5px'), ('background-color', '#87CEFA'), ('color', 'black')]
                              }, {
                                  'selector': 'td',
                                  'props': [('border', '1px solid black'), ('padding', '5px')]
                              }])

# Display the styled scaled DataFrames
display(styled_X_scaled)
display(styled_X_test_scaled)


def preprocess(df):
    # Make a copy of the DataFrame
    df = df.copy()
    
    # Fill categorical variables and map them to numeric values
    df['Stage_fear'] = df['Stage_fear'].fillna('Unknown').map({'Yes': 1, 'No': 0, 'Unknown': -1})
    df['Drained_after_socializing'] = df['Drained_after_socializing'].fillna('Unknown').map({'Yes': 1, 'No': 0, 'Unknown': -1})
    
    # Select numeric columns for imputation
    num_cols = ['Time_spent_Alone', 'Social_event_attendance',
                'Going_outside', 'Friends_circle_size',
                'Post_frequency']
    
    # Use IterativeImputer to fill missing numeric values
    imputer = IterativeImputer(random_state=42)
    df[num_cols] = imputer.fit_transform(df[num_cols])
    
    return df

# Load the datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# Preprocess training data
X = preprocess(train.drop(['id', 'Personality'], axis=1))  # Removing 'Personality'
y = train['Personality'].map({'Introvert': 0, 'Extrovert': 1})  # Target variable

# Preprocess test data
X_test = preprocess(test.drop('id', axis=1))

# Scale the data using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Implement Truncated SVD for dimensionality reduction
svd = TruncatedSVD(n_components=2, random_state=42)  # Set the number of components
X_svd = svd.fit_transform(X_scaled)  # Fit and transform the training scaled data
X_test_svd = svd.transform(X_test_scaled)  # Transform the test scaled data

# Display the first 10 rows of SVD results with styled formatting
styled_X_svd = pd.DataFrame(X_svd, columns=['tSVD1', 'tSVD2']).head(10).style.set_table_attributes('style="border: 1px solid black; border-collapse: collapse;"') \
                    .set_table_styles([{
                        'selector': 'th',
                        'props': [('border', '1px solid black'), ('padding', '5px'), ('background-color', '#87CEFA'), ('color', 'black')]
                    }, {
                        'selector': 'td',
                        'props': [('border', '1px solid black'), ('padding', '5px')]
                    }])

styled_X_test_svd = pd.DataFrame(X_test_svd, columns=['tSVD1', 'tSVD2']).head(10).style.set_table_attributes('style="border: 1px solid black; border-collapse: collapse;"') \
                              .set_table_styles([{
                                  'selector': 'th',
                                  'props': [('border', '1px solid black'), ('padding', '5px'), ('background-color', '#87CEFA'), ('color', 'black')]
                              }, {
                                  'selector': 'td',
                                  'props': [('border', '1px solid black'), ('padding', '5px')]
                              }])

# Display the styled SVD DataFrames
display(styled_X_svd)
display(styled_X_test_svd)


# Create a DataFrame to include the tSVD components and the target variable
svd_df = pd.DataFrame(data=X_svd, columns=['tSVD1', 'tSVD2'])
svd_df['Personality'] = y  # Add the target variable (0: Introvert, 1: Extrovert)

# Adding small random noise to enhance visualization
noise_scale = 0.1  # Change this value for more or less spacing
tSVD1_noisy = svd_df['tSVD1'] + np.random.normal(0, noise_scale, size=len(svd_df))
tSVD2_noisy = svd_df['tSVD2'] + np.random.normal(0, noise_scale, size=len(svd_df))

# Plotting the SVD results
plt.figure(figsize=(10, 10))
scatter = plt.scatter(tSVD1_noisy, tSVD2_noisy,
                      c=svd_df['Personality'], cmap='coolwarm', alpha=0.8, s=50, marker='o')

plt.title("tSVD of Training Data", fontsize=16)
plt.xlabel("tSVD Component 1 Explains %.1f %% of Variance" % (svd.explained_variance_ratio_[0] * 100.0), fontsize=14)
plt.ylabel("tSVD Component 2 Explains %.1f %% of Variance" % (svd.explained_variance_ratio_[1] * 100.0), fontsize=14)

# Create a custom legend
# Map numbered labels to text labels
handles, labels = scatter.legend_elements() 
new_labels = ['Introvert', 'Extrovert']
plt.legend(handles, new_labels, title="Personality", loc='best', markerscale=1.5, prop={'size': 14})

plt.xlim(-0.5, 2.5)  # Adjust x limit as needed to accommodate noise
plt.ylim(-1.5, 2)    # Adjust y limit as needed to accommodate noise

plt.tight_layout()
plt.show()


# Preprocessing function
def preprocess(df):
    df = df.copy()
    df['Stage_fear'] = df['Stage_fear'].fillna('Unknown').map({'Yes': 1, 'No': 0, 'Unknown': -1})
    df['Drained_after_socializing'] = df['Drained_after_socializing'].fillna('Unknown').map({'Yes': 1, 'No': 0, 'Unknown': -1})
    num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    imputer = IterativeImputer(random_state=42)
    df[num_cols] = imputer.fit_transform(df[num_cols])
    return df

# Load the datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# Preprocess training and test data
X = preprocess(train.drop(['id', 'Personality'], axis=1))  # Remove 'Personality'
y = train['Personality'].map({'Introvert': 0, 'Extrovert': 1})  # Map target variable

X_test = preprocess(test.drop('id', axis=1))

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the validation set
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)
print(classification_report(y_val, y_pred))

# Make predictions on the test data
test_predictions = model.predict(X_test)

# Map back to original labels
test_predictions_labels = np.where(test_predictions == 0, 'Introvert', 'Extrovert')

# Create a DataFrame to save the results with descriptive labels
results = pd.DataFrame({'id': test['id'], 'Predicted_Personality': test_predictions_labels})

# Save the results to CSV
results.to_csv('result_r_forest.csv', index=False)  # Random Forest results


# Visualizing the Predictions
plt.figure(figsize=(10, 10))

# Adding noise to the features to disperse the points
noise_scale = 0.1  # Adjust the noise scale as needed
plt.scatter(X_test['Time_spent_Alone'] + np.random.normal(0, noise_scale, size=len(X_test)),
            X_test['Social_event_attendance'] + np.random.normal(0, noise_scale, size=len(X_test)),
            c=[1 if label == 'Extrovert' else 0 for label in test_predictions_labels],
            cmap='coolwarm', alpha=0.8, s=50, marker='o')

plt.title("Random Forest Predictions of Personality (Introverts and Extroverts)", fontsize=16)
plt.xlabel("Feature: Time Spent Alone + Noise", fontsize=14)
plt.ylabel("Feature: Social Event Attendance + Noise", fontsize=14)

# Custom legend for Introverts and Extroverts
introvert_patch = plt.Line2D([0], [0], marker='o', color='w', label='Introvert',
                              markerfacecolor='blue', markersize=10)  # Adjust color as necessary
extrovert_patch = plt.Line2D([0], [0], marker='o', color='w', label='Extrovert',
                              markerfacecolor='red', markersize=10)  # Adjust color as necessary

# Add the custom legend
plt.legend(handles=[introvert_patch, extrovert_patch], title="Personality")

plt.tight_layout()
plt.show()


# Preprocessing function
def preprocess(df):
    df = df.copy()
    df['Stage_fear'] = df['Stage_fear'].fillna('Unknown').map({'Yes': 1, 'No': 0, 'Unknown': -1})
    df['Drained_after_socializing'] = df['Drained_after_socializing'].fillna('Unknown').map({'Yes': 1, 'No': 0, 'Unknown': -1})
    num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    imputer = IterativeImputer(random_state=42)
    df[num_cols] = imputer.fit_transform(df[num_cols])
    return df

# Load the datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# Preprocess training and test data
X = preprocess(train.drop(['id', 'Personality'], axis=1))  # Remove 'Personality'
y = train['Personality'].map({'Introvert': 0, 'Extrovert': 1})  # Map target variable

X_test = preprocess(test.drop('id', axis=1))

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the CatBoost model with adjustable parameters
catboost_model = CatBoostClassifier(
    iterations=1000,                  # Increase the number of iterations
    learning_rate=0.05,               # Decrease the learning rate
    depth=6,                          # Adjust the depth
    l2_leaf_reg=3,                    # Use L2 regularization to prevent overfitting
    bagging_temperature=0.75,         # Adjust the bagging temperature
    random_state=42,
    verbose=100                       # Keep verbose for monitoring progress
)

# Fit the model on the training data
catboost_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = catboost_model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")
print(classification_report(y_val, y_pred))

# Make predictions on the test set
catboost_predictions = catboost_model.predict(X_test)

# Map back to original labels
catboost_predictions_labels = np.where(catboost_predictions == 0, 'Introvert', 'Extrovert')

# Create a DataFrame to save the results with descriptive labels
catboost_results = pd.DataFrame({'id': test['id'], 'Predicted_Personality': catboost_predictions_labels})

# Save the results to CSV
catboost_results.to_csv('/kaggle/working/result_c_boost.csv', index=False)  # Save results


# Vamos usar as duas primeiras colunas do X_test, se disponíveis, ou características que você deseja usar para visualização
plt.figure(figsize=(10, 10))

# Use as características originais de teste para a visualização
# Se tiver apenas uma dimensão, isso precisa ser ajustado
if X_test.shape[1] > 1:
    feature1 = X_test_scaled[:, 0] + np.random.normal(0, 0.1, size=len(X_test_scaled))
    feature2 = X_test_scaled[:, 1] + np.random.normal(0, 0.1, size=len(X_test_scaled))
else:
    feature1 = X_test_scaled[:, 0] + np.random.normal(0, 0.1, size=len(X_test_scaled))
    feature2 = np.zeros(len(X_test_scaled))  # Se o X_test não tiver uma segunda coluna

# Scatter plot
scatter = plt.scatter(feature1, feature2,
                      c=[1 if label == 'Extrovert' else 0 for label in results['Predicted_Personality']],
                      cmap='coolwarm', alpha=0.8, s=50, marker='o')

plt.title("Predictions of Personality (Introverts and Extroverts)", fontsize=16)
plt.xlabel("Feature 1", fontsize=14)  # Você pode rotular de acordo com as características que está usando
plt.ylabel("Feature 2", fontsize=14)

# Custom legend labels
introvert_patch = plt.Line2D([0], [0], marker='o', color='w', label='Introvert',
                              markerfacecolor='blue', markersize=10)  # Adjust color as necessary
extrovert_patch = plt.Line2D([0], [0], marker='o', color='w', label='Extrovert',
                              markerfacecolor='red', markersize=10)  # Adjust color as necessary

# Add the custom legend
plt.legend(handles=[introvert_patch, extrovert_patch], title="Personality")

plt.tight_layout()
plt.show()


# Preprocessing function
def preprocess(df):
    df = df.copy()
    df['Stage_fear'] = df['Stage_fear'].fillna('Unknown').map({'Yes': 1, 'No': 0, 'Unknown': -1})
    df['Drained_after_socializing'] = df['Drained_after_socializing'].fillna('Unknown').map({'Yes': 1, 'No': 0, 'Unknown': -1})
    num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    imputer = IterativeImputer(random_state=42)
    df[num_cols] = imputer.fit_transform(df[num_cols])
    return df

# Load the datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# Preprocess training and test data
X = preprocess(train.drop(['id', 'Personality'], axis=1))  # Remove 'Personality'
y = train['Personality'].map({'Introvert': 0, 'Extrovert': 1})  # Map target variable

X_test = preprocess(test.drop('id', axis=1))

# Scale the data using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the XGBoost model
xgboost_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Fit the model on the training data
xgboost_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = xgboost_model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")
print(classification_report(y_val, y_pred))

# Make predictions on the test data
xgboost_predictions = xgboost_model.predict(X_test_scaled)

# Map back to original labels
xgboost_predictions_labels = np.where(xgboost_predictions == 0, 'Introvert', 'Extrovert')

# Create a DataFrame to save the results with descriptive labels
xgboost_results = pd.DataFrame({'id': test['id'], 'Predicted_Personality': xgboost_predictions_labels})

# Save the results to CSV
xgboost_results.to_csv('result_xg_boost.csv', index=False)  # Save results


# Plotting the Predictions from XGBoost
plt.figure(figsize=(10, 10))

# Adding noise to the features for better visualization
noise_scale = 0.1  # Adjust the noise scale as needed
plt.scatter(X_test_scaled[:, 0] + np.random.normal(0, noise_scale, size=len(X_test_scaled)), 
            X_test_scaled[:, 1] + np.random.normal(0, noise_scale, size=len(X_test_scaled)),
            c=[1 if label == 'Extrovert' else 0 for label in xgboost_predictions_labels],
            cmap='coolwarm', alpha=0.8, s=50, marker='o')

plt.title("XGBoost Predictions of Personality (Introverts and Extroverts)", fontsize=16)
plt.xlabel("Feature 1 (Scaled) + Noise", fontsize=14)
plt.ylabel("Feature 2 (Scaled) + Noise", fontsize=14)

# Custom legend for Introverts and Extroverts
introvert_patch = plt.Line2D([0], [0], marker='o', color='w', label='Introvert',
                              markerfacecolor='blue', markersize=10)  # Adjust color as necessary
extrovert_patch = plt.Line2D([0], [0], marker='o', color='w', label='Extrovert',
                              markerfacecolor='red', markersize=10)  # Adjust color as necessary

# Add the custom legend
plt.legend(handles=[introvert_patch, extrovert_patch], title="Personality")

plt.tight_layout()
plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import statistics
from IPython.display import display  # To display DataFrames nicely

# Load the results from previous models
rf_results = pd.read_csv('/kaggle/working/result_r_forest.csv')  # Random Forest results
catboost_results = pd.read_csv('/kaggle/working/result_c_boost.csv')  # CatBoost results
xgboost_results = pd.read_csv('/kaggle/working/result_xg_boost.csv')  # XGBoost results

# Load the original feature data (test set)
test_data = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# Prepare 'id' columns
for df in [rf_results, catboost_results, xgboost_results, test_data]:
    df['id'] = df['id'].astype(str)  # Ensure all 'id' columns are of the same type

# Rename the prediction columns for clarity
rf_results.rename(columns={'Predicted_Personality': 'Random Forest'}, inplace=True)
catboost_results.rename(columns={'Predicted_Personality': 'CatBoost'}, inplace=True)
xgboost_results.rename(columns={'Predicted_Personality': 'XGBoost'}, inplace=True)

# Merge results on 'id'
comparison_df = rf_results.merge(catboost_results, on='id')
comparison_df = comparison_df.merge(xgboost_results, on='id')

# Define a function to determine majority voting
def majority_vote(row):
    predictions = [row['Random Forest'], row['CatBoost'], row['XGBoost']]
    return statistics.mode(predictions)  # Get the most common prediction

# Apply the majority vote function to each row
comparison_df['Final Prediction'] = comparison_df.apply(majority_vote, axis=1)

# Create a new column to check if predictions differ
comparison_df['Divergence'] = (comparison_df['Random Forest'] != comparison_df['Final Prediction']) | \
                              (comparison_df['CatBoost'] != comparison_df['Final Prediction']) | \
                              (comparison_df['XGBoost'] != comparison_df['Final Prediction'])

# Filter to keep only those rows where there is a divergence
divergent_predictions_df = comparison_df[comparison_df['Divergence']]

# Prepare features DataFrame from test data
feature_columns = ['Time_spent_Alone', 'Stage_fear', 
                   'Social_event_attendance', 'Going_outside', 
                   'Drained_after_socializing', 'Friends_circle_size', 
                   'Post_frequency']  # Original feature names

features_df = test_data[feature_columns].copy()
features_df['id'] = test_data['id'].astype(str)  # Add the id for merging

# Merge the feature columns with the divergent predictions DataFrame
divergent_predictions_df = divergent_predictions_df.merge(features_df, on='id', how='left')

# Prepare the DataFrame for displaying with styled formatting
divergent_predictions_to_print = divergent_predictions_df[['id', 'Random Forest', 'CatBoost', 'XGBoost', 
                                                           'Time_spent_Alone', 'Stage_fear', 
                                                           'Social_event_attendance', 'Going_outside', 
                                                           'Drained_after_socializing', 'Friends_circle_size', 
                                                           'Post_frequency']].copy()

# Display the DataFrame with divergent predictions formatted with borders
styled_divergent_predictions = divergent_predictions_to_print.style.set_table_attributes('style="border: 1px solid black; border-collapse: collapse;"') \
                    .set_table_styles([{
                        'selector': 'th',
                        'props': [('border', '1px solid black'), ('padding', '5px'), ('background-color', '#87CEFA'), ('color', 'black')]
                    }, {
                        'selector': 'td',
                        'props': [('border', '1px solid black'), ('padding', '5px')]
                    }])

# Display the styled DataFrame including the id column
print("\nDivergent Predictions (with Feature Columns):")
display(styled_divergent_predictions)  # Display the styled DataFrame (index will not be shown automatically)


import pandas as pd
from collections import Counter

# Assuming you already have a DataFrame 'comparison_df' filled with predictions from the models
# Define a function to determine the final prediction using majority voting
def final_prediction(row):
    # Record the predictions of each model using full names
    predictions = [row['Random Forest'], row['CatBoost'], row['XGBoost']]
    # Calculate the majority vote
    majority_vote = Counter(predictions).most_common(1)[0][0]  # Get the most common prediction
    return majority_vote

# Add the final prediction column to the DataFrame
comparison_df['Final Prediction'] = comparison_df.apply(final_prediction, axis=1)

# Create the submission DataFrame
submission_df = comparison_df[['id', 'Final Prediction']].copy()
submission_df.rename(columns={'Final Prediction': 'Predicted_Personality'}, inplace=True)

# Optionally check for any null values in predicted personalities
if submission_df['Predicted_Personality'].isnull().any():
    print("Warning: There are null values in the predictions.")

# Save the DataFrame as a CSV
submission_df.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' has been created.")


import matplotlib.pyplot as plt

# Plotting the Predictions from Random Forest
plt.figure(figsize=(10, 10))

# Adding noise to the features to disperse the points
noise_scale = 0.1  # Adjust the noise scale as needed
plt.scatter(X_test['Time_spent_Alone'] + np.random.normal(0, noise_scale, size=len(X_test)), 
            X_test['Social_event_attendance'] + np.random.normal(0, noise_scale, size=len(X_test)),
            c=[1 if label == 'Extrovert' else 0 for label in submission_df['Predicted_Personality']],
            cmap='coolwarm', alpha=0.8, s=50, marker='o')

plt.title("Random Forest Predictions of Personality (Introverts and Extroverts)", fontsize=16)
plt.xlabel("Time Spent Alone + Noise", fontsize=14)
plt.ylabel("Social Event Attendance + Noise", fontsize=14)

# Custom legend for Introverts and Extroverts
introvert_patch = plt.Line2D([0], [0], marker='o', color='w', label='Introvert',
                              markerfacecolor='blue', markersize=10)  
extrovert_patch = plt.Line2D([0], [0], marker='o', color='w', label='Extrovert',
                              markerfacecolor='red', markersize=10)  

# Add the custom legend
plt.legend(handles=[introvert_patch, extrovert_patch], title="Personality")

plt.tight_layout()
plt.show()

