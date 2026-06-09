import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')



train_df.info()


train_df.describe()


train_df.isnull().sum()


X = train_df.drop('Listening_Time_minutes', axis=1)
y = train_df['Listening_Time_minutes']


X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.8, random_state=42)


X_train, X_val, y_train, y_val = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

print(f"Sample training size: {X_train.shape}")


X_train['Number_of_Ads'] = X_train['Number_of_Ads'].fillna(X_train['Number_of_Ads'].median())
X_val['Number_of_Ads'] = X_val['Number_of_Ads'].fillna(X_val['Number_of_Ads'].median())


categorical_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']


X_train['Guest_Popularity_Missing'] = X_train['Guest_Popularity_percentage'].isna().astype(int)
X_val['Guest_Popularity_Missing'] = X_val['Guest_Popularity_percentage'].isna().astype(int)
X_train['Episode_Length_Missing'] = X_train['Episode_Length_minutes'].isna().astype(int)
X_val['Episode_Length_Missing'] = X_val['Episode_Length_minutes'].isna().astype(int)


numerical_cols.extend(['Guest_Popularity_Missing', 'Episode_Length_Missing'])


drop_cols = ['id', 'Episode_Title', 'Podcast_Name']
X_train = X_train.drop(columns=drop_cols)
X_val = X_val.drop(columns=drop_cols)


numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42))
])


print("Training Random Forest model...")
rf_model.fit(X_train, y_train)
print("Random Forest model trained!")


y_pred_rf = rf_model.predict(X_val)


rf_rmse = np.sqrt(mean_squared_error(y_val, y_pred_rf))
rf_r2 = r2_score(y_val, y_pred_rf)

print(f"Random Forest RMSE: {rf_rmse:.4f}")
print(f"Random Forest R²: {rf_r2:.4f}")


lr_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])


print("Training Linear Regression model...")
lr_model.fit(X_train, y_train)
print("Linear Regression model trained!")


y_pred_lr = lr_model.predict(X_val)


lr_rmse = np.sqrt(mean_squared_error(y_val, y_pred_lr))
lr_r2 = r2_score(y_val, y_pred_lr)

print(f"Linear Regression RMSE: {lr_rmse:.4f}")
print(f"Linear Regression R²: {lr_r2:.4f}")


plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred_rf if rf_r2 > lr_r2 else y_pred_lr, alpha=0.3)
plt.plot([0, 120], [0, 120], 'r--')
plt.xlabel('Actual Listening Time')
plt.ylabel('Predicted Listening Time')
plt.title('Actual vs Predicted Listening Time')
plt.tight_layout()
plt.show()


# Get feature names after preprocessing
preprocessor = rf_model.named_steps['preprocessor']
feature_names = []


# Get numerical feature names
num_features = numerical_cols
for feature in num_features:
    feature_names.append(feature)


#Analyzing feature importance from the Random Forest model
X_train_processed = rf_model.named_steps['preprocessor'].transform(X_train)


# Get feature importances
importances = rf_model.named_steps['regressor'].feature_importances_


plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances)
plt.title('Feature Importances')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()


#trying Gradient Boosting as another model
from sklearn.ensemble import GradientBoostingRegressor


gb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
])


print("Training Gradient Boosting model...")
gb_model.fit(X_train, y_train)
print("Gradient Boosting model trained!")


y_pred_gb = gb_model.predict(X_val)


gb_rmse = np.sqrt(mean_squared_error(y_val, y_pred_gb))
gb_r2 = r2_score(y_val, y_pred_gb)
print(f"Gradient Boosting RMSE: {gb_rmse:.4f}")
print(f"Gradient Boosting R²: {gb_r2:.4f}")


# Let's also analyze the relationship between actual and predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred_gb, alpha=0.3)
plt.plot([0, 120], [0, 120], 'r--')
plt.xlabel('Actual Listening Time')
plt.ylabel('Predicted Listening Time')
plt.title('Actual vs Predicted Listening Time (Gradient Boosting)')
plt.tight_layout()
plt.show()


# Let's create a function to prepare new data for prediction
def prepare_data_for_prediction(df):
    """Prepares new data for prediction using our best model"""
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Handle missing values in Number_of_Ads
    df_copy['Number_of_Ads'] = df_copy['Number_of_Ads'].fillna(df_copy['Number_of_Ads'].median())
    
    # Create missing flags
    df_copy['Guest_Popularity_Missing'] = df_copy['Guest_Popularity_percentage'].isna().astype(int)
    df_copy['Episode_Length_Missing'] = df_copy['Episode_Length_minutes'].isna().astype(int)
    
    # Drop unnecessary columns
    if 'id' in df_copy.columns:
        df_copy = df_copy.drop(columns=['id'])
    if 'Episode_Title' in df_copy.columns:
        df_copy = df_copy.drop(columns=['Episode_Title'])
    if 'Podcast_Name' in df_copy.columns:
        df_copy = df_copy.drop(columns=['Podcast_Name'])
    
    return df_copy

# Choose the best model based on R²
best_model = rf_model if rf_r2 > gb_r2 else gb_model
best_model_name = "Random Forest" if rf_r2 > gb_r2 else "Gradient Boosting"
print(f"Best model: {best_model_name}")

# Function to make predictions on new data
def predict_listening_time(df, model):
    """Makes predictions on new data using the trained model"""
    # Prepare the data
    prepared_df = prepare_data_for_prediction(df)
    
    # Make predictions
    predictions = model.predict(prepared_df)
    
    return predictions

# Example of how to use the prediction function
print("\nExample of how to predict on new data:")
print("test_df_prepared = prepare_data_for_prediction(test_df)")
print("predictions = best_model.predict(test_df_prepared)")


X_train_processed = preprocessor.fit_transform(X_train)


# Create a simpler Random Forest model directly on processed features
direct_rf = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42)
direct_rf.fit(X_train_processed, y_train)


# Get feature importances
importances = direct_rf.feature_importances_


# Sort and get top 10 feature indices
top_indices = importances.argsort()[-10:][::-1]
top_importances = importances[top_indices]

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_indices)), top_importances)
plt.yticks(range(len(top_indices)), [f"Feature {i}" for i in top_indices])
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.show()
print(f"Top 10 most important feature indices: {top_indices}")


X = train_df.drop('Listening_Time_minutes', axis=1)
y = train_df['Listening_Time_minutes']


X['Number_of_Ads'] = X['Number_of_Ads'].fillna(X['Number_of_Ads'].median())


categorical_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']


X['Guest_Popularity_Missing'] = X['Guest_Popularity_percentage'].isna().astype(int)
X['Episode_Length_Missing'] = X['Episode_Length_minutes'].isna().astype(int)


numerical_cols.extend(['Guest_Popularity_Missing', 'Episode_Length_Missing'])


drop_cols = ['id', 'Episode_Title', 'Podcast_Name']
X = X.drop(columns=drop_cols)


# Create preprocessing steps
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# Create the final Gradient Boosting model (since it performed best)
final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
])


# Fit the model on the entire training dataset
print("Training final model on full dataset...")
final_model.fit(X, y)
print("Final model trained!")


# Define a function to prepare new data for prediction
def prepare_new_data(df):
    """Prepares new data for prediction"""
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Handle missing values in Number_of_Ads
    df_copy['Number_of_Ads'] = df_copy['Number_of_Ads'].fillna(df_copy['Number_of_Ads'].median())
    
    # Create missing flags
    df_copy['Guest_Popularity_Missing'] = df_copy['Guest_Popularity_percentage'].isna().astype(int)
    df_copy['Episode_Length_Missing'] = df_copy['Episode_Length_minutes'].isna().astype(int)
    
    # Drop unnecessary columns if they exist
    cols_to_drop = ['id', 'Episode_Title', 'Podcast_Name']
    for col in cols_to_drop:
        if col in df_copy.columns:
            df_copy = df_copy.drop(columns=[col])
    
    return df_copy



# Function to make predictions on new data
def predict_listening_time(new_data, model=final_model):
    """Makes predictions on new data using the trained model"""
    # Prepare the data
    prepared_data = prepare_new_data(new_data)
    
    # Make predictions
    predictions = model.predict(prepared_data)
    
    return predictions


prepared_test_df = prepare_new_data(test_df)

# Now we can make predictions using our trained model
predictions = final_model.predict(prepared_test_df)

# Add these predictions to your test dataframe
test_df['Predicted_Listening_Time'] = predictions

# Display the first few rows to verify the predictions were added
print(test_df[['Episode_Length_minutes', 'Genre', 'Host_Popularity_percentage', 'Predicted_Listening_Time']].head())




test_df


submission_df = test_df[['id', 'Predicted_Listening_Time']].copy()
submission_df.rename(columns={'Predicted_Listening_Time': 'Listening_Time_minutes'}, inplace=True)



submission_df

