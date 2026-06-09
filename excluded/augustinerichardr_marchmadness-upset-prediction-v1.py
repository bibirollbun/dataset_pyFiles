import gc

# Run garbage collection to clear unreferenced objects from memory
gc.collect()

# Optionally, restart the kernel (if allowed on the platform)
# This can be done manually if you want a fresh environment in some cases.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout  


#file paths for Kaggle environment
base_path = "/kaggle/input/march-machine-learning-mania-2024/"

# Loading datasets using the updated base path
teams_df = pd.read_csv(base_path + "MTeams.csv")
seasons_df = pd.read_csv(base_path + "MSeasons.csv")
results_df = pd.read_csv(base_path + "MRegularSeasonCompactResults.csv")
tourney_results_df = pd.read_csv(base_path + "MNCAATourneyCompactResults.csv")
seeds_df = pd.read_csv(base_path + "MNCAATourneySeeds.csv")
massey_ordinals_df = pd.read_csv(base_path + "MMasseyOrdinals_thruSeason2024_day128.csv")
detailed_results_df = pd.read_csv(base_path + "MRegularSeasonDetailedResults.csv")


teams_df.head()


results_df.head()


tourney_results_df.head()


seeds_df.head()


massey_ordinals_df.head()


detailed_results_df.head()


for df_name, df in zip(["Detailed Results", "Teams", "Seasons", "Results", "Tourney Results", "Seeds", "Massey Ordinals"], 
                       [detailed_results_df, teams_df, seasons_df, results_df, tourney_results_df, seeds_df, massey_ordinals_df]):
    print(f"\n{df_name} Missing Values:")
    print(df.isnull().sum())
    print(f"{df_name} Data Types:")
    print(df.dtypes)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create the 'upset' column
results_df['upset'] = (results_df['WScore'] < results_df['LScore']).astype(int)

# Define features (X) and target (y)
X = results_df[['WScore', 'LScore', 'WTeamID', 'LTeamID', 'NumOT']]  # Selecting relevant features
y = results_df['upset']  # Target variable

# Optional: Drop rows with missing values, if necessary
X = X.dropna()
y = y.loc[X.index]  # Ensure that y matches the rows in X after dropping missing values

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Optionally scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Check the shape of the train and test sets
print("X_train shape:", X_train_scaled.shape)
print("y_train shape:", y_train.shape)


# Check class distribution in y_train
print("Class distribution in y_train:")
print(y_train.value_counts())


# Check the column names of the dataframe
print(df.columns)

# Check a few rows to inspect the data
print(df.head())


results_df.head()


# Merge detailed_results_df with massey_ordinals_df based on common columns
merged_df = pd.merge(results_df, massey_ordinals_df, left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"], how="inner")

# Check the result of the merge
print(merged_df.head())


merged_df.shape


import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
correlation_matrix = X.corr()

# Plot the heatmap for visualization with correlation numbers
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix with Correlation Values")
plt.show()

# Define the threshold for high correlation
correlation_threshold = 0.6

# Identify highly correlated pairs and print their correlation values
highly_correlated_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            highly_correlated_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

# Print the highly correlated pairs with their correlation values
print("Highly Correlated Pairs (Threshold > 0.6):")
for pair in highly_correlated_pairs:
    print(f"Feature 1: {pair[0]}, Feature 2: {pair[1]}, Correlation: {pair[2]:.2f}")

# Also print the full correlation matrix for reference
print("\nFull Correlation Matrix (with correlation values):")
print(correlation_matrix)


# Identify non-numeric columns
non_numeric_cols = merged_df.select_dtypes(include=['object']).columns
print("Non-Numeric Columns:", non_numeric_cols)


merged_df['WLoc'] = merged_df['WLoc'].map({'H': 1, 'A': -1, 'N': 0})
merged_df = merged_df.drop(columns=['SystemName'])


print(merged_df.isnull().sum())
merged_df.fillna(merged_df.median(), inplace=True)


merged_df["upset"] = (merged_df["LTeamID"] < merged_df["WTeamID"]).astype(int)


print(merged_df["upset"].value_counts())


correlation_matrix = merged_df.corr()
top_features = correlation_matrix["upset"].abs().sort_values(ascending=False).head(6)  # Include 'upset' itself
print(top_features)


# Sampling the dataset to reduce size (2% of the original data)
sampled_df = merged_df.sample(frac=0.02, random_state=42)

# Selecting relevant features and target
selected_features = ['TeamID', 'WTeamID', 'LTeamID', 'LScore', 'OrdinalRank']
X = sampled_df[selected_features]  # Features
y = sampled_df['upset']  # Target

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print the sample size to confirm the subset
print(f"Sampled Data Size: {sampled_df.shape}")


y_train.value_counts()


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x=y)
plt.title("Class Distribution of Upsets (1) vs. Non-Upsets (0)")
plt.xlabel("Upset (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.show()


# # Selecting relevant features and target
# selected_features = ['TeamID', 'WTeamID', 'LTeamID', 'LScore', 'OrdinalRank']
# X = merged_df[selected_features]  # Features
# y = merged_df['upset']  # Target

# # Split the data into training and testing sets
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


!pip install xgboost


!pip install --upgrade pip


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Logistic Regression Model
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
logreg_accuracy = accuracy_score(y_test, y_pred_logreg)
logreg_precision = precision_score(y_test, y_pred_logreg)
logreg_recall = recall_score(y_test, y_pred_logreg)
logreg_f1 = f1_score(y_test, y_pred_logreg)


# Random Forest Model
import gc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Convert data to low-memory format
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

# Optimized Random Forest
rf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=1, random_state=42)

print("ğŸ”„ Training Random Forest (Memory Optimized)...")
rf.fit(X_train, y_train)

print("ğŸ”� Making Predictions...")
y_pred_rf = rf.predict(X_test)

# Compute all metrics in one step
rf_accuracy, rf_precision, rf_recall, rf_f1 = map(
    lambda metric: metric(y_test, y_pred_rf),
    [accuracy_score, precision_score, recall_score, f1_score]
)

# Print results
print(f"âœ… Accuracy: {rf_accuracy:.4f}, Precision: {rf_precision:.4f}, Recall: {rf_recall:.4f}, F1-score: {rf_f1:.4f}")

# Free memory
# del rf, y_pred_rf
# gc.collect()


# XGBoost Model
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
xgb_precision = precision_score(y_test, y_pred_xgb)
xgb_recall = recall_score(y_test, y_pred_xgb)
xgb_f1 = f1_score(y_test, y_pred_xgb)


# import gc
# import numpy as np
# from sklearn.experimental import enable_halving_search_cv  
# from sklearn.model_selection import HalvingRandomSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# # Convert dataset to lower memory types
# X_train = X_train.astype(np.float32)  
# X_test = X_test.astype(np.float32)

# # Parameter grids (reduced for efficiency)
# logreg_params = {
#     'C': np.logspace(-3, 3, 5),
#     'solver': ['liblinear'],  
#     'class_weight': ['balanced', None]
# }

# rf_params = {
#     'n_estimators': [50, 100],  
#     'max_depth': [5, 10],  
#     'min_samples_split': [2, 5],  
#     'min_samples_leaf': [1, 2],  
#     'max_features': ['sqrt'],  
#     'class_weight': ['balanced']
# }

# xgb_params = {
#     'n_estimators': [100, 200],  
#     'learning_rate': [0.05, 0.1],  
#     'max_depth': [3, 6],  
#     'subsample': [0.8],  
#     'colsample_bytree': [0.8],  
#     'gamma': [0, 0.1],  
#     'scale_pos_weight': [1, 5]
# }

# # Function to run optimized tuning with memory clearing
# def run_optimized_tuning(model_name, model, param_grid, X_train, y_train):
#     print(f"\nğŸ”„ Tuning {model_name} (Optimized)...")

#     try:
#         search = HalvingRandomSearchCV(
#             model, param_grid, factor=2, cv=2, scoring='recall',
#             n_jobs=1, verbose=1, random_state=42
#         )

#         search.fit(X_train, y_train)
        
#         print(f"âœ… Best parameters for {model_name}: {search.best_params_}\n")

#         # **Memory cleanup** after each tuning
#         del search
#         gc.collect()
        
#         return model.set_params(**search.best_params_)

#     except Exception as e:
#         print(f"â�Œ Error during tuning {model_name}: {e}")
#         return model  # Return the original model if tuning fails

# # Run memory-optimized tuning
# best_logreg = run_optimized_tuning("Logistic Regression", LogisticRegression(max_iter=500, random_state=42), logreg_params, X_train, y_train)
# gc.collect()

# best_rf = run_optimized_tuning("Random Forest", RandomForestClassifier(random_state=42), rf_params, X_train, y_train)
# gc.collect()

# best_xgb = run_optimized_tuning("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), xgb_params, X_train, y_train)
# gc.collect()

# # Evaluate models
# models = {'Logistic Regression': best_logreg, 'Random Forest': best_rf, 'XGBoost': best_xgb}

# for name, model in models.items():
#     print(f"\nğŸ”� Evaluating {name}...")
    
#     y_pred = model.predict(X_test)

#     print(f"ğŸ“Š {name} - Tuned Model Results:")
#     print(f"âœ… Accuracy: {accuracy_score(y_test, y_pred):.4f}")
#     print(f"âœ… Precision: {precision_score(y_test, y_pred):.4f}")
#     print(f"âœ… Recall: {recall_score(y_test, y_pred):.4f}  <-- Key metric for upsets!")
#     print(f"âœ… F1-score: {f1_score(y_test, y_pred):.4f}")
#     print("-" * 50)

#     # **Clear memory after model evaluation**
#     del model, y_pred
#     gc.collect()


# Display the performance of all models
models = ['Logistic Regression', 'Random Forest', 'XGBoost']
accuracies = [logreg_accuracy, rf_accuracy, xgb_accuracy]
precisions = [logreg_precision, rf_precision, xgb_precision]
recalls = [logreg_recall, rf_recall, xgb_recall]
f1_scores = [logreg_f1, rf_f1, xgb_f1]

results_df = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracies,
    'Precision': precisions,
    'Recall': recalls,
    'F1 Score': f1_scores
})
print(results_df)


# Import necessary models and evaluation metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# # Train the model (using Random Forest as the best performing model)
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model performance (optional, as we already know it performs well)
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# # Print evaluation metrics
# print(f"Model Accuracy: {accuracy}")
# print(f"Model Precision: {precision}")
# print(f"Model Recall: {recall}")
# print(f"Model F1 Score: {f1}")

# Create a DataFrame to store the test set features along with predictions
test_predictions_df = X_test.copy()
test_predictions_df['upset_pred'] = y_pred_rf


# Correct the path for saving the file in Kaggle's environment
output_path = '/kaggle/working/upset_predictions.csv'  

# Save the predictions DataFrame as a CSV file
test_predictions_df.to_csv(output_path, index=False)

# Optionally, display the first few rows of the DataFrame
print(test_predictions_df.head())


def simulate_upset_prediction(model, X_sample, num_simulations=10, output_path='/kaggle/working/simulated_upset_predictions.csv'):
    """
    Simulates upset predictions and saves the results to a CSV file in a Kaggle Notebook.
    
    Args:
    model: The trained ML model (e.g., XGBoost, RandomForest).
    X_sample: DataFrame containing input features.
    num_simulations (int): Number of simulations.
    output_path (str): File path to save the predictions.
    
    Returns:
    None
    """
    simulated_results = []

    for _ in range(num_simulations):
        # Generate random feature values by sampling from each column independently
        random_values = {col: np.random.choice(X_sample[col].values) for col in X_sample.columns}
        
        # Convert to DataFrame to maintain feature names
        random_features = pd.DataFrame([random_values])
        
        # Make prediction using the trained model
        simulated_upset = model.predict(random_features)[0]
        
        # Store result
        simulated_results.append(random_features.iloc[0].tolist() + [simulated_upset])

    # Create DataFrame with results
    simulated_df = pd.DataFrame(simulated_results, columns=X_sample.columns.tolist() + ['SimulatedUpset'])
    
    # Save to CSV in Kaggle working directory
    simulated_df.to_csv(output_path, index=False)
    
    print(f"Simulated predictions saved to {output_path}")
    display(simulated_df.head())  # Display first few rows in Kaggle output

# Example usage (assuming `model` and `X_test` are defined)
simulate_upset_prediction(rf, X_test, num_simulations=10)


import numpy as np
import pandas as pd

def simulate_upset_prediction(model, X_sample, y_sample, num_simulations=10, output_path='/kaggle/working/simulated_upset_predictions.csv'):
    """
    Simulates upset predictions and returns a DataFrame for evaluation.
    
    Args:
    model: Trained ML model.
    X_sample: DataFrame with input features.
    y_sample: Series or DataFrame with actual outcomes.
    num_simulations (int): Number of simulations.
    output_path (str): File path to save results.
    
    Returns:
    comparison_df (DataFrame): DataFrame containing real vs simulated upsets.
    """
    simulated_results = []

    for _ in range(num_simulations):
        # Generate random values by sampling from existing data
        random_values = {col: np.random.choice(X_sample[col].values) for col in X_sample.columns}
        
        # Convert to DataFrame
        random_features = pd.DataFrame([random_values])
        
        # Ensure correct data types
        random_features = random_features.astype(X_sample.dtypes)

        try:
            simulated_upset = model.predict(random_features)[0]
        except Exception as e:
            print(f"Prediction failed: {e}")
            simulated_upset = np.nan

        simulated_results.append(random_features.iloc[0].tolist() + [simulated_upset])

    # Create DataFrame for simulated results
    simulated_df = pd.DataFrame(simulated_results, columns=X_sample.columns.tolist() + ['SimulatedUpset'])

    # Merge real results for comparison
    real_results = X_sample.copy()
    real_results['RealUpset'] = y_sample.values

    # Merge simulated and real results
    comparison_df = real_results.reset_index(drop=True).join(simulated_df[['SimulatedUpset']], how='left')

    # Apply dynamic thresholding (e.g., 75th percentile)
    threshold = np.percentile(comparison_df['SimulatedUpset'].dropna(), 75)
    comparison_df["SimulatedUpset"] = (comparison_df["SimulatedUpset"] > threshold).astype(int)

    # Save to CSV
    comparison_df.to_csv(output_path, index=False)
    
    print(f"Simulated predictions saved to {output_path}")
    display(comparison_df.head())  # Show preview

    return comparison_df  # Return DataFrame for further analysis

# Run simulation and store results
comparison_df = simulate_upset_prediction(rf, X_test, y_test, num_simulations=500)

# Compute accuracy
comparison_df["CorrectPrediction"] = (comparison_df["RealUpset"] == comparison_df["SimulatedUpset"]).astype(int)
accuracy = comparison_df["CorrectPrediction"].mean()
print(f"Simulation Accuracy: {accuracy * 100:.2f}%")

