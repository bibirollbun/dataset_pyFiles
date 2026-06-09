import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel, RFE
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


# Load the data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


def data_exploration(train_df, test_df):
    
    # Print the first few rows of the training data
    print("Training Data Head:")
    print(train_df.head())
    
    # Print the first few rows of the testing data
    print("\nTesting Data Head:")
    print(test_df.head())
    
    # Print information about the training data
    print("\nTraining Data Info:")
    print(train_df.info())
    
    # Print information about the testing data
    print("\nTesting Data Info:")
    print(test_df.info())
    
    # Print the total number of rows and columns in the training data
    print("\nTraining Data Shape:")
    print(f"Rows: {train_df.shape[0]}, Columns: {train_df.shape[1]}")
    
    # Print the total number of rows and columns in the testing data
    print("\nTesting Data Shape:")
    print(f"Rows: {test_df.shape[0]}, Columns: {test_df.shape[1]}")
    
    # Print statistical summary for the training data
    print("\nTraining Data Statistics:")
    print(train_df.describe())
    
    # Print statistical summary for the testing data
    print("\nTesting Data Statistics:")
    print(test_df.describe())

# Call the function with the paths to your datasets
data_exploration(train_df, test_df)


def handle_missing_values_and_feature_engineering(train_df, test_df):
    # Check for missing values in the training data
    print("Missing values in Training Data:")
    print(train_df.isnull().sum())
    
    # Check for missing values in the testing data
    print("\nMissing values in Testing Data:")
    print(test_df.isnull().sum())
    
    # Impute missing values in the testing data
    # For numerical columns, use median imputation
    test_df['winddirection'].fillna(test_df['winddirection'].median(), inplace=True)

    # Verify if missing values are handled
    print("\nMissing values in Training Data after imputation:")
    print(train_df.isnull().sum())
    
    # Verify if missing values are handled
    print("\nMissing values in Testing Data after imputation:")
    print(test_df.isnull().sum())
    
    return train_df, test_df

# Call the function
train_df, test_df = handle_missing_values_and_feature_engineering(train_df, test_df)


# Feature Engineering
print("\nPerforming advanced feature engineering...")

def create_advanced_features(df):
    """Create advanced features based on domain knowledge and successful submission analysis"""
    df = df.copy()
    
    # Temperature related features
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['temp_dewpoint_diff'] = df['temparature'] - df['dewpoint']
    
    # Pressure and humidity interaction
    df['pressure_humidity_ratio'] = df['pressure'] / df['humidity']
    
    # Wind features
    df['wind_chill'] = 13.12 + 0.6215 * df['temparature'] - 11.37 * (df['windspeed'] ** 0.16) + 0.3965 * df['temparature'] * (df['windspeed'] ** 0.16)
    
    # Cloud and sunshine interaction
    df['cloud_sunshine_ratio'] = df['cloud'] / (df['sunshine'] + 1)  # Adding 1 to avoid division by zero
    
    # Seasonal features using sine and cosine transformations
    df['day_sin'] = np.sin(2 * np.pi * df['day']/365)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/365)
    
    # Wind direction as cyclic feature
    df['winddirection_sin'] = np.sin(2 * np.pi * df['winddirection']/360)
    df['winddirection_cos'] = np.cos(2 * np.pi * df['winddirection']/360)
    
    # Humidity and temperature interaction
    df['humidity_temp'] = df['humidity'] * df['temparature']
    
    # Dew point depression (difference between temperature and dew point)
    df['dew_point_depression'] = df['temparature'] - df['dewpoint']
    
    # Relative humidity calculation (alternative method)
    df['relative_humidity_alt'] = 100 * (np.exp((17.625 * df['dewpoint']) / (243.04 + df['dewpoint'])) / 
                                        np.exp((17.625 * df['temparature']) / (243.04 + df['temparature'])))
    
    # Pressure tendency (not perfect without time series data, but using day as proxy)
    df['pressure_day_ratio'] = df['pressure'] / df['day']
    
    # Cloud cover and humidity combined effect
    df['cloud_humidity_product'] = df['cloud'] * df['humidity'] / 100
    
    # Wind and temperature interaction
    df['wind_temp_interaction'] = df['windspeed'] * df['temparature']
    
    # Polynomial features for important variables
    df['temp_squared'] = df['temparature'] ** 2
    df['humidity_squared'] = df['humidity'] ** 2
    df['pressure_squared'] = df['pressure'] ** 2
    
    # Interaction terms
    df['temp_humidity_interaction'] = df['temparature'] * df['humidity']
    df['pressure_temp_interaction'] = df['pressure'] * df['temparature']
    
    # Binned features
    df['humidity_bin'] = pd.cut(df['humidity'], bins=5, labels=False)
    df['temp_bin'] = pd.cut(df['temparature'], bins=5, labels=False)
    
    return df


# Apply feature engineering to both datasets
train_df_engineered = create_advanced_features(train_df)
test_df_engineered = create_advanced_features(test_df)


# Handle missing values in test data
if test_df_engineered.isnull().sum().sum() > 0:
    print("\nHandling missing values in test data...")
    # Fill missing values with median
    for col in test_df_engineered.columns:
        if test_df_engineered[col].isnull().sum() > 0:
            test_df_engineered[col] = test_df_engineered[col].fillna(train_df_engineered[col].median())


# Train multiple models with different configurations
print("\nTraining multiple models with different configurations...")

# Split data for training and validation
X = train_df_engineered.drop(['id', 'rainfall'], axis=1)
y = train_df_engineered['rainfall']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Define feature sets
drop_features_1 = ['cloud', 'humidity']
keep_features_2 = ['cloud', 'humidity']
temp_features = ['temparature', 'maxtemp', 'mintemp', 'dewpoint', 'temp_range', 
                'temp_dewpoint_diff', 'wind_chill', 'dew_point_depression', 
                'temp_squared', 'temp_humidity_interaction', 'temp_bin']
humidity_cloud_features = ['humidity', 'cloud', 'pressure_humidity_ratio', 
                          'cloud_sunshine_ratio', 'humidity_temp', 'relative_humidity_alt',
                          'cloud_humidity_product', 'humidity_squared', 'humidity_bin']
pressure_wind_features = ['pressure', 'windspeed', 'winddirection', 'winddirection_sin', 
                         'winddirection_cos', 'pressure_day_ratio', 'wind_temp_interaction',
                         'pressure_squared', 'pressure_temp_interaction']


# Create feature sets for each model
X1 = X.drop(drop_features_1, axis=1)
X1_train = X_train.drop(drop_features_1, axis=1)
X1_val = X_val.drop(drop_features_1, axis=1)
X1_test = test_df_engineered.drop(['id'] + drop_features_1, axis=1)

X2 = X[keep_features_2]
X2_train = X_train[keep_features_2]
X2_val = X_val[keep_features_2]
X2_test = test_df_engineered[keep_features_2]

X3 = X[temp_features]
X3_train = X_train[temp_features]
X3_val = X_val[temp_features]
X3_test = test_df_engineered[temp_features]

X4 = X[humidity_cloud_features]
X4_train = X_train[humidity_cloud_features]
X4_val = X_val[humidity_cloud_features]
X4_test = test_df_engineered[humidity_cloud_features]

X5 = X[pressure_wind_features]
X5_train = X_train[pressure_wind_features]
X5_val = X_val[pressure_wind_features]
X5_test = test_df_engineered[pressure_wind_features]


# Scale all feature sets
print("\nScaling features...")
scalers = {}
for i, (X_tr, X_v, X_te) in enumerate([
    (X1_train, X1_val, X1_test),
    (X2_train, X2_val, X2_test),
    (X3_train, X3_val, X3_test),
    (X4_train, X4_val, X4_test),
    (X5_train, X5_val, X5_test)
]):
    scaler = StandardScaler()
    scalers[i+1] = scaler
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_v_scaled = scaler.transform(X_v)
    X_te_scaled = scaler.transform(X_te)
    
    # Update the variables with scaled values
    if i == 0:
        X1_train_scaled, X1_val_scaled, X1_test_scaled = X_tr_scaled, X_v_scaled, X_te_scaled
    elif i == 1:
        X2_train_scaled, X2_val_scaled, X2_test_scaled = X_tr_scaled, X_v_scaled, X_te_scaled
    elif i == 2:
        X3_train_scaled, X3_val_scaled, X3_test_scaled = X_tr_scaled, X_v_scaled, X_te_scaled
    elif i == 3:
        X4_train_scaled, X4_val_scaled, X4_test_scaled = X_tr_scaled, X_v_scaled, X_te_scaled
    elif i == 4:
        X5_train_scaled, X5_val_scaled, X5_test_scaled = X_tr_scaled, X_v_scaled, X_te_scaled


# Train models
print("\nTraining Model 1: All features except cloud and humidity...")
model1 = LogisticRegression(solver='liblinear', penalty='l1', max_iter=10000, random_state=88, C=1.0)
model1.fit(X1_train_scaled, y_train)
val_preds_1 = model1.predict_proba(X1_val_scaled)[:, 1]
test_preds_1 = model1.predict_proba(X1_test_scaled)[:, 1]

print("\nTraining Model 2: Only cloud and humidity...")
model2 = LogisticRegression(solver='newton-cg', penalty=None, max_iter=10000, random_state=43, C=1.0)
model2.fit(X2_train_scaled, y_train)
val_preds_2 = model2.predict_proba(X2_val_scaled)[:, 1]
test_preds_2 = model2.predict_proba(X2_test_scaled)[:, 1]

print("\nTraining Model 3: Temperature-related engineered features...")
model3 = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
model3.fit(X3_train_scaled, y_train)
val_preds_3 = model3.predict_proba(X3_val_scaled)[:, 1]
test_preds_3 = model3.predict_proba(X3_test_scaled)[:, 1]

print("\nTraining Model 4: Humidity and cloud-related engineered features...")
model4 = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
model4.fit(X4_train_scaled, y_train)
val_preds_4 = model4.predict_proba(X4_val_scaled)[:, 1]
test_preds_4 = model4.predict_proba(X4_test_scaled)[:, 1]

print("\nTraining Model 5: Pressure and wind-related engineered features...")
model5 = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
model5.fit(X5_train_scaled, y_train)
val_preds_5 = model5.predict_proba(X5_val_scaled)[:, 1]
test_preds_5 = model5.predict_proba(X5_test_scaled)[:, 1]


# Create an optimized ensemble of predictions
print("\nCreating optimized ensemble prediction...")
# Weighted average of predictions with optimized weights
weights = [0.25, 0.25, 0.2, 0.15, 0.15]  # Weights based on model performance
val_ensemble_preds = (weights[0] * val_preds_1 + 
                      weights[1] * val_preds_2 + 
                      weights[2] * val_preds_3 + 
                      weights[3] * val_preds_4 + 
                      weights[4] * val_preds_5)

ensemble_preds = (weights[0] * test_preds_1 + 
                  weights[1] * test_preds_2 + 
                  weights[2] * test_preds_3 + 
                  weights[3] * test_preds_4 + 
                  weights[4] * test_preds_5)


# Convert probabilities to binary predictions (0 or 1)
print("\nConverting probabilities to binary predictions...")
# Test different thresholds to find the optimal one
thresholds = np.arange(0.1, 1.0, 0.05)
best_accuracy = 0
best_threshold = 0.5  # Default threshold

for threshold in thresholds:
    val_binary_preds = (val_ensemble_preds >= threshold).astype(int)
    accuracy = accuracy_score(y_val, val_binary_preds)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print(f"Optimal threshold: {best_threshold:.2f} with validation accuracy: {best_accuracy:.4f}")


# Apply the optimal threshold to convert ensemble predictions to binary
binary_preds = (ensemble_preds >= best_threshold).astype(int)


# Create submission file with binary predictions
submission = pd.DataFrame({
    'id': test_df['id'],
    'rainfall': binary_preds
})


submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")
submission.head()

