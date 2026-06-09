import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

import os
import random
import time
import warnings
warnings.filterwarnings('ignore')


data_train=pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
data_test=pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
data_train.info()


data_train.info()
data_train.isnull().sum()
#we have 8k missing in the target value so imputing them may not sound as a bad idea but after experimenting we found dropping them does way better for our score 


# Step 1: Check which rows have null values in the 'num_sold' column
null_num_sold = data_train[data_train['num_sold'].isnull()]
print("Rows with null values in 'num_sold':")
print(null_num_sold)

# Step 2: Calculate the mean of 'num_sold' for each country
country_mean_num_sold = data_train.groupby('country')['num_sold'].transform('mean')

# Step 3: Fill null values in 'num_sold' with the mean of the corresponding country
data_train['num_sold'] = data_train['num_sold'].fillna(country_mean_num_sold)

# Verify that there are no more null values in 'num_sold'
print("\nAfter filling null values:")
print(data_train['num_sold'].isnull().sum())  # Should print 0


# Check for duplicate rows
duplicates = data_train.duplicated()
# Count the number of duplicate rows
num_duplicates = duplicates.sum()
print(f"Number of duplicate rows: {num_duplicates}")

duplicates = data_test.duplicated()
num_duplicates = duplicates.sum()
print(f"Number of duplicate rows: {num_duplicates}")






import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_Categorical_dataset(data):
    
    # Step 1: Identify categorical columns
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    
    # Step 2: Perform label encoding for categorical columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le  # Store the encoder for later use
    
    # Step 3: Handle the 'date' column
    # if 'date' in data.columns:
    #     # Convert to datetime
    #     data['date'] = pd.to_datetime(data['date'])
        
    #     # Extract useful features
    #     data['year'] = data['date'].dt.year
    #     data['month'] = data['date'].dt.month
    #     data['day_of_week'] = data['date'].dt.dayofweek  # Monday=0, Sunday=6
    #     data['quarter'] = data['date'].dt.quarter
        
    #     # Drop the original 'date' column
    data.drop(columns=['date'], inplace=True)
    
    # Return the processed dataset and the encoders
    return data, label_encoders
data_train_og_with_categories=data_train
# Example usage
data_train, label_encoders_train = preprocess_Categorical_dataset(data_train)
print(data_train.head())




#for train data
# Select numeric columns
numeric_columns = data_train.select_dtypes(include=['int64', 'float64']).columns
print("Numeric columns:", numeric_columns)

import seaborn as sns
import matplotlib.pyplot as plt

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Create box plots for each numeric column
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(2, 3, i)  # Adjust the subplot grid as needed
    sns.boxplot(y=data_train[column])
    plt.title(column)

plt.tight_layout()
plt.show()


#for test data
# Select numeric columns
numeric_columns = data_test.select_dtypes(include=['int64', 'float64']).columns
print("Numeric columns:", numeric_columns)

import seaborn as sns
import matplotlib.pyplot as plt

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Create box plots for each numeric column
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(2, 3, i)  # Adjust the subplot grid as needed
    sns.boxplot(y=data_test[column])
    plt.title(column)

plt.tight_layout()
plt.show()


data_train['num_sold'] = np.log(data_train['num_sold'])


#for train data after capping
# Select numeric columns
numeric_columns = data_train.select_dtypes(include=['int64', 'float64']).columns
print("Numeric columns:", numeric_columns)

import seaborn as sns
import matplotlib.pyplot as plt

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Create box plots for each numeric column
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(2, 3, i)  # Adjust the subplot grid as needed
    sns.boxplot(y=data_train[column])
    plt.title(column)

plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split

# Drop the 'id' column from data_train
data_train = data_train.drop(columns=['id'])  # This will return a modified DataFrame

# Split into features (X) and target (y) for data_train
X_train = data_train.drop(columns=['num_sold'])
y_train = data_train['num_sold']

# Split the data into train and test sets
X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Verify the shapes of the resulting sets
print(f"Training data (X_train_set): {X_train_set.shape}")
print(f"Test data (X_test_set): {X_test_set.shape}")
print(f"Training labels (y_train_set): {y_train_set.shape}")
print(f"Test labels (y_test_set): {y_test_set.shape}")

# For data_train_og_with_categories, remove 'id' column
data_train_og_with_categories = data_train_og_with_categories.drop(columns=['id'])  # This will return a modified DataFrame

# Split into features (X) and target (y) for data_train_og_with_categories
X_train_with_categories = data_train_og_with_categories.drop(columns=['num_sold'])
y_train_with_categories = data_train_og_with_categories['num_sold']

# Split the data into train and test sets for data_train_og_with_categories
X_train_set_with_categories, X_test_set_with_categories, y_train_set_with_categories, y_test_set_with_categories = train_test_split(
    X_train_with_categories, y_train_with_categories, test_size=0.2, random_state=42
)

# Verify the shapes of the resulting sets
print(f"Training data with categories (X_train_set_with_categories): {X_train_set_with_categories.shape}")
print(f"Test data with categories (X_test_set_with_categories): {X_test_set_with_categories.shape}")
print(f"Training labels with categories (y_train_set_with_categories): {y_train_set_with_categories.shape}")
print(f"Test labels with categories (y_test_set_with_categories): {y_test_set_with_categories.shape}")






import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_decision_tree(X_train, X_test, y_train, y_test, max_depth=3, feature_names=None):
    """
    Analyze and visualize a decision tree with detailed metrics and explanations
    """
    # Convert feature_names to list if it's a pandas Index
    if feature_names is not None:
        feature_names = list(feature_names)
    
    # Initialize and train the decision tree
    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=42
    )
    tree.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = tree.predict(X_train)
    y_pred_test = tree.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'Training': {
            'MSE': mean_squared_error(y_train, y_pred_train),
            'RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'MAE': mean_absolute_error(y_train, y_pred_train),
            'MAPE': mean_absolute_percentage_error(y_train, y_pred_train)
        },
        'Test': {
            'MSE': mean_squared_error(y_test, y_pred_test),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'MAE': mean_absolute_error(y_test, y_pred_test),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred_test)
        }
    }
    
    # Perform cross-validation
    cv_scores = cross_val_score(tree, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    
    # Print detailed analysis
    print("="*50)
    print("DECISION TREE ANALYSIS")
    print("="*50)
    
    print("\n1. Model Parameters:")
    print(f"Max Depth: {max_depth}")
    print(f"Number of leaves: {tree.get_n_leaves()}")
    print(f"Tree depth: {tree.get_depth()}")
    
    print("\n2. Performance Metrics:")
    print("\nTraining Metrics:")
    for metric, value in metrics['Training'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\nTest Metrics:")
    for metric, value in metrics['Test'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\n3. Cross-validation Results (RMSE):")
    print(f"Mean RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")
    
    # Visualizations
    plt.figure(figsize=(20, 10))
    
    # 1. Tree Visualization
    plot_tree(tree, 
             feature_names=feature_names,
             filled=True,
             rounded=True,
             fontsize=10)
    plt.title("Decision Tree Visualization")
    plt.show()
    
    # 2. Feature Importance
    importance_data = pd.DataFrame({
        'feature': feature_names if feature_names is not None else [f'Feature {i}' for i in range(X_train.shape[1])],
        'importance': tree.feature_importances_
    })
    importance_data = importance_data.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_data, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()
    
    # 3. Residuals Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    residuals_train = y_train - y_pred_train
    plt.scatter(y_pred_train, residuals_train, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Training Residuals vs Predicted')
    
    plt.subplot(1, 2, 2)
    residuals_test = y_test - y_pred_test
    plt.scatter(y_pred_test, residuals_test, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Test Residuals vs Predicted')
    
    plt.tight_layout()
    plt.show()
    
    # Print tree structure as text
    print("\n4. Tree Structure:")
    print(export_text(tree, feature_names=feature_names))
    
    # Additional decision path analysis
    print("\n5. Decision Path Analysis:")
    # Get decision paths for a few sample instances
    n_samples = min(5, len(X_test))
    paths = tree.decision_path(X_test[:n_samples])
    
    for i in range(n_samples):
        print(f"\nSample {i+1} decision path:")
        path = paths[i].indices
        for node_id in path:
            if node_id == tree.tree_.children_left[path[0]]:  # If it's a decision node
                threshold = tree.tree_.threshold[node_id]
                feature = feature_names[tree.tree_.feature[node_id]] if feature_names else f"feature_{tree.tree_.feature[node_id]}"
                print(f"Split on {feature} <= {threshold:.2f}")
    
    return tree, metrics, importance_data

# Example usage:
print("\nAnalyzing model without categories:")
tree1, metrics1, importance1 = analyze_decision_tree(
    X_train_set, 
    X_test_set, 
    y_train_set, 
    y_test_set,
    max_depth=3,
    feature_names=X_train_set.columns
)

print("\nAnalyzing model with categories:")
tree2, metrics2, importance2 = analyze_decision_tree(
    X_train_set_with_categories, 
    X_test_set_with_categories, 
    y_train_set_with_categories, 
    y_test_set_with_categories,
    max_depth=3,
    feature_names=X_train_set_with_categories.columns
)


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd

# Define the parameter grid for RandomForestRegressor
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
    'max_features': ['auto', 'sqrt']  # Number of features to consider at each split
}

# Initialize the RandomForestRegressor
rf = RandomForestRegressor(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # Use negative MSE for scoring
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # Use all available CPU cores
    verbose=2  # Print progress
)

# Perform Grid Search on data_train (without categories)
print("Performing Grid Search for Model 1 (data_train)...")
grid_search.fit(X_train_set, y_train_set)

# Get the best model from Grid Search
best_model_1 = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred_1 = best_model_1.predict(X_test_set)
mae_1 = mean_absolute_error(y_test_set, y_pred_1)
rmse_1 = np.sqrt(mean_squared_error(y_test_set, y_pred_1))
mape_1 = mean_absolute_percentage_error(y_test_set, y_pred_1)

# Display metrics for the best model
print("\nMetrics for Best Model 1 (Trained on data_train):")
print("MAE:", mae_1)
print("RMSE:", rmse_1)
print("MAPE:", mape_1)

# Feature importance for the best model
importances_1 = best_model_1.feature_importances_
feature_importance_df_1 = pd.DataFrame({'Feature': X_train_set.columns, 'Importance': importances_1})
feature_importance_df_1 = feature_importance_df_1.sort_values(by='Importance', ascending=False)
print("\nFeature Importance for Best Model 1 (data_train):")
print(feature_importance_df_1)

# Perform Grid Search on data_train_og_with_categories (with categories)
print("\nPerforming Grid Search for Model 2 (data_train_og_with_categories)...")
grid_search.fit(X_train_set_with_categories, y_train_set_with_categories)

# Get the best model from Grid Search
best_model_2 = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred_2 = best_model_2.predict(X_test_set_with_categories)
mae_2 = mean_absolute_error(y_test_set_with_categories, y_pred_2)
rmse_2 = np.sqrt(mean_squared_error(y_test_set_with_categories, y_pred_2))
mape_2 = mean_absolute_percentage_error(y_test_set_with_categories, y_pred_2)

# Display metrics for the best model
print("\nMetrics for Best Model 2 (Trained on data_train_og_with_categories):")
print("MAE:", mae_2)
print("RMSE:", rmse_2)
print("MAPE:", mape_2)

# Feature importance for the best model
importances_2 = best_model_2.feature_importances_
feature_importance_df_2 = pd.DataFrame({'Feature': X_train_set_with_categories.columns, 'Importance': importances_2})
feature_importance_df_2 = feature_importance_df_2.sort_values(by='Importance', ascending=False)
print("\nFeature Importance for Best Model 2 (data_train_og_with_categories):")
print(feature_importance_df_2)



import matplotlib.pyplot as plt

# Function to calculate MSE
def calculate_mse(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    return train_mse, test_mse

# Calculate MSE for both models
train_mse_1, test_mse_1 = calculate_mse(best_model_1, X_train_set, y_train_set, X_test_set, y_test_set)
train_mse_2, test_mse_2 = calculate_mse(best_model_2, X_train_set_with_categories, y_train_set_with_categories, X_test_set_with_categories, y_test_set_with_categories)

# Data for plotting
models = ['Model 1 (data_train)', 'Model 2 (data_train_og_with_categories)']
train_loss = [train_mse_1, train_mse_2]
test_loss = [test_mse_1, test_mse_2]

# Plotting
x = np.arange(len(models))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, train_loss, width, label='Train Loss (MSE)', color='blue')
rects2 = ax.bar(x + width/2, test_loss, width, label='Test Loss (MSE)', color='orange')

# Add labels, title, and custom x-axis tick labels
ax.set_xlabel('Models')
ax.set_ylabel('Mean Squared Error (MSE)')
ax.set_title('Training and Test Loss for Best Models')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Add value labels on top of the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()


from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

# Train the model on data_train (without categories)
model_1 = AdaBoostRegressor(n_estimators=100, random_state=42)
model_1.fit(X_train_set, y_train_set)

# Evaluate the model on data_train test set
y_pred_1 = model_1.predict(X_test_set)
mae_1 = mean_absolute_error(y_test_set, y_pred_1)
rmse_1 = np.sqrt(mean_squared_error(y_test_set, y_pred_1))
mape_1 = mean_absolute_percentage_error(y_test_set, y_pred_1)

# Display metrics for model trained on data_train
print("Metrics for Model 1 (Trained on data_train):")
print("MAE:", mae_1)
print("RMSE:", rmse_1)
print("MAPE:", mape_1)

# Feature importance for the model trained on data_train
importances_1 = model_1.feature_importances_
feature_importance_df_1 = pd.DataFrame({'Feature': X_train_set.columns, 'Importance': importances_1})
feature_importance_df_1 = feature_importance_df_1.sort_values(by='Importance', ascending=False)
print("\nFeature Importance for Model 1 (data_train):")
print(feature_importance_df_1)

# Train the model on data_train_og_with_categories (with categories)
model_2 = AdaBoostRegressor(n_estimators=100, random_state=42)
model_2.fit(X_train_set_with_categories, y_train_set_with_categories)

# Evaluate the model on data_train_og_with_categories test set
y_pred_2 = model_2.predict(X_test_set_with_categories)
mae_2 = mean_absolute_error(y_test_set_with_categories, y_pred_2)
rmse_2 = np.sqrt(mean_squared_error(y_test_set_with_categories, y_pred_2))
mape_2 = mean_absolute_percentage_error(y_test_set_with_categories, y_pred_2)

# Display metrics for model trained on data_train_og_with_categories
print("\nMetrics for Model 2 (Trained on data_train_og_with_categories):")
print("MAE:", mae_2)
print("RMSE:", rmse_2)
print("MAPE:", mape_2)

# Feature importance for the model trained on data_train_og_with_categories
importances_2 = model_2.feature_importances_
feature_importance_df_2 = pd.DataFrame({'Feature': X_train_set_with_categories.columns, 'Importance': importances_2})
feature_importance_df_2 = feature_importance_df_2.sort_values(by='Importance', ascending=False)
print("\nFeature Importance for Model 2 (data_train_og_with_categories):")
print(feature_importance_df_2)



from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

# Initialize base models
model_1_ensemble = AdaBoostRegressor(n_estimators=100, random_state=42)
model_2_ensemble = RandomForestRegressor(n_estimators=100, random_state=42)
model_3_ensemble = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Create the ensemble model
ensemble_model_1 = VotingRegressor(estimators=[('ada', model_1_ensemble), ('rf', model_2_ensemble), ('gb', model_3_ensemble)])

# Train the ensemble model on data_train (without categories)
ensemble_model_1.fit(X_train_set, y_train_set)

# Evaluate the ensemble model on data_train test set
y_pred_ensemble_1 = ensemble_model_1.predict(X_test_set)
mae_ensemble_1 = mean_absolute_error(y_test_set, y_pred_ensemble_1)
rmse_ensemble_1 = np.sqrt(mean_squared_error(y_test_set, y_pred_ensemble_1))
mape_ensemble_1 = mean_absolute_percentage_error(y_test_set, y_pred_ensemble_1)

# Display metrics for the ensemble model trained on data_train
print("Metrics for Ensemble Model 1 (Trained on data_train):")
print("MAE:", mae_ensemble_1)
print("RMSE:", rmse_ensemble_1)
print("MAPE:", mape_ensemble_1)

# Feature importance for the ensemble model trained on data_train
# To get feature importance, we can use the RandomForestRegressor (as an estimator in the ensemble)
importances_ensemble_1 = ensemble_model_1.named_estimators_['rf'].feature_importances_
feature_importance_df_1 = pd.DataFrame({'Feature': X_train_set.columns, 'Importance': importances_ensemble_1})
feature_importance_df_1 = feature_importance_df_1.sort_values(by='Importance', ascending=False)
print("\nFeature Importance for Ensemble Model 1 (data_train):")
print(feature_importance_df_1)

# Train the ensemble model on data_train_og_with_categories (with categories)
ensemble_model_2 = VotingRegressor(estimators=[('ada', model_1_ensemble), ('rf', model_2_ensemble), ('gb', model_3_ensemble)])
ensemble_model_2.fit(X_train_set_with_categories, y_train_set_with_categories)

# Evaluate the ensemble model on data_train_og_with_categories test set
y_pred_ensemble_2 = ensemble_model_2.predict(X_test_set_with_categories)
mae_ensemble_2 = mean_absolute_error(y_test_set_with_categories, y_pred_ensemble_2)
rmse_ensemble_2 = np.sqrt(mean_squared_error(y_test_set_with_categories, y_pred_ensemble_2))
mape_ensemble_2 = mean_absolute_percentage_error(y_test_set_with_categories, y_pred_ensemble_2)

# Display metrics for the ensemble model trained on data_train_og_with_categories
print("\nMetrics for Ensemble Model 2 (Trained on data_train_og_with_categories):")
print("MAE:", mae_ensemble_2)
print("RMSE:", rmse_ensemble_2)
print("MAPE:", mape_ensemble_2)

# Feature importance for the ensemble model trained on data_train_og_with_categories
importances_ensemble_2 = ensemble_model_2.named_estimators_['rf'].feature_importances_
feature_importance_df_2 = pd.DataFrame({'Feature': X_train_set_with_categories.columns, 'Importance': importances_ensemble_2})
feature_importance_df_2 = feature_importance_df_2.sort_values(by='Importance', ascending=False)
print("\nFeature Importance for Ensemble Model 2 (data_train_og_with_categories):")
print(feature_importance_df_2)




import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import r2_score, explained_variance_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def train_and_evaluate_model(X_train, y_train, X_test, y_test, categorical_features=None, model_name="Model", 
                           plot_results=True, feature_importance_top_n=20):
    """
    Enhanced training and evaluation function with comprehensive metrics and visualizations
    
    Parameters:
    -----------
    X_train, y_train : training data and labels
    X_test, y_test : test data and labels
    categorical_features : list of categorical feature names
    model_name : string, name of the model for plotting
    plot_results : boolean, whether to show plots
    feature_importance_top_n : int, number of top features to show in importance plot
    """
    print(f"\n{'='*50}\nTraining {model_name}...\n{'='*50}")
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, categorical_feature=categorical_features)
    
    # Training history for plotting
    training_history = {}
    
    def callback_recorder(env):
        """Record training history for plotting"""
        training_history[env.iteration] = {
            'valid_0': env.evaluation_result_list[0][2],
            'training': env.evaluation_result_list[1][2] if len(env.evaluation_result_list) > 1 else None
        }
    
    # Train the model with enhanced monitoring
    model = lgb.train(
        params=lgb_params,
        train_set=train_data,
        num_boost_round=1000,  # Increased number of rounds
        valid_sets=[test_data, train_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),  # More patience
            lgb.log_evaluation(period=100),
            callback_recorder
        ]
    )
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate comprehensive metrics
    metrics = {
        'Training': {
            'MAE': mean_absolute_error(y_train, y_pred_train),
            'RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'MAPE': mean_absolute_percentage_error(y_train, y_pred_train),
            'R2': r2_score(y_train, y_pred_train),
            'EVS': explained_variance_score(y_train, y_pred_train)
        },
        'Test': {
            'MAE': mean_absolute_error(y_test, y_pred_test),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred_test),
            'R2': r2_score(y_test, y_pred_test),
            'EVS': explained_variance_score(y_test, y_pred_test)
        }
    }
    
    # Print metrics
    print("\nModel Performance Metrics:")
    print(f"{'Metric':<20} {'Training':<15} {'Test':<15}")
    print("-" * 50)
    for metric in metrics['Training'].keys():
        print(f"{metric:<20} {metrics['Training'][metric]:<15.4f} {metrics['Test'][metric]:<15.4f}")
    
    if plot_results:
        # 1. Training History Plot
        plt.figure(figsize=(12, 6))
        iterations = list(training_history.keys())
        valid_scores = [v['valid_0'] for v in training_history.values()]
        train_scores = [v['training'] for v in training_history.values()]
        
        plt.plot(iterations, valid_scores, label='Validation')
        if train_scores[0] is not None:
            plt.plot(iterations, train_scores, label='Training')
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.title(f'{model_name} Training History')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # 2. Feature Importance Plot (Enhanced)
        importances = model.feature_importance(importance_type='gain')
        feature_importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importances,
            'Importance_Normalized': importances / np.sum(importances) * 100
        })
        feature_importance_df = feature_importance_df.sort_values(
            by='Importance_Normalized', ascending=True
        ).tail(feature_importance_top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=feature_importance_df,
            x='Importance_Normalized',
            y='Feature',
            palette='viridis'
        )
        plt.title(f'Top {feature_importance_top_n} Feature Importance ({model_name})')
        plt.xlabel('Importance (%)')
        plt.tight_layout()
        plt.show()
        
        # 3. Residuals Analysis
        residuals_test = y_test - y_pred_test
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Residuals Analysis for {model_name}')
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred_test, residuals_test, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        
        # Residuals Distribution
        sns.histplot(residuals_test, kde=True, ax=axes[0, 1])
        axes[0, 1].set_title('Residuals Distribution')
        
        # Q-Q Plot
        stats.probplot(residuals_test, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        # Actual vs Predicted
        axes[1, 1].scatter(y_test, y_pred_test, alpha=0.5)
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title('Actual vs Predicted')
        
        plt.tight_layout()
        plt.show()
    
    # Save feature importance to DataFrame
    feature_importance_df = feature_importance_df.sort_values(by='Importance_Normalized', ascending=False)
    print("\nDetailed Feature Importance:")
    print(feature_importance_df)
    
    return model, metrics, feature_importance_df

# Enhanced parameters with randomized search results
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 127,
    'learning_rate': 0.0989627523617693,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42,
    'n_estimators': 400,
    'subsample': 0.8615034945856372,
    'colsample_bytree': 0.8050238118519274,
    'reg_alpha': 0.03,  # L1 regularization
    'reg_lambda': 0.07,  # L2 regularization
    'min_child_samples': 20,  # Minimum number of samples needed in a child
    'max_depth': -1,  # Unlimited depth
    'min_split_gain': 0.0,  # Minimum loss reduction required to make a split
    'max_bin': 255  # Maximum number of bins that feature values will be bucketed in
}

# Train and evaluate the model
model_1, metrics_1, importance_df_1 = train_and_evaluate_model(
    X_train_set, 
    y_train_set, 
    X_test_set, 
    y_test_set, 
    model_name="Enhanced Model"
)


import pandas as pd
import lightgbm as lgb

# Step 1: Preprocess the test data (assuming categorical columns are already preprocessed)
# Replace `preprocess_Categorical_dataset` with your actual preprocessing function
data_test, label_encoders_test = preprocess_Categorical_dataset(data_test)

# Step 2: Drop the 'id' column from data_test to make predictions
X_test_final = data_test.drop(columns=['id'])

# Step 3: Ensure the test data has the same feature columns as the training data
# This is important to avoid mismatches in feature names or order
X_test_final = X_test_final[X_train_set.columns]  # Use the columns from the training data

# Step 4: Use the trained LightGBM model to predict the target 'num_sold'
# Ensure `model_1` (or `best_model_1`) is your trained LightGBM model
y_pred_final = model_1.predict(X_test_final)

# Step 5: Create a DataFrame to hold the 'id' and predicted 'num_sold' values
output_df = pd.DataFrame({'id': data_test['id'], 'num_sold': y_pred_final})

# Step 6: Save the predictions to a CSV file
output_df.to_csv('predictions_LGBM.csv', index=False)

print("Predictions saved to 'predictions.csv'")

