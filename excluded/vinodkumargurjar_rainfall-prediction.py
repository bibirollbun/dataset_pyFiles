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


df_train=pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
sample_submission=pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


df_train.head(5)


df_train.drop(columns=["id","day"], axis=1, inplace=True)
df_test.drop(columns=["id","day"], axis=1, inplace=True)


from ydata_profiling import ProfileReport
profile = ProfileReport(df_train, title="Pandas Profiling Report")


# profile


import seaborn as sns
import matplotlib.pyplot as plt
target_variable="rainfall"
def eda_pipeline(df_train, df_test):
    
    # Display first few rows
    print("\n--- First few rows of train data ---")
    display(df_train.head())
    
    print("\n--- First few rows of test data ---")
    display(df_test.head())
    
    # Dataset info
    print("\n--- Train Data Info ---")
    print(df_train.info())
    
    print("\n--- Test Data Info ---")
    print(df_test.info())
    
    # Missing values
    print("\n--- Missing Values in Train Data ---")
    print(df_train.isnull().sum())
    
    print("\n--- Missing Values in Test Data ---")
    print(df_test.isnull().sum())
    
    print("\n--- Percentage of Missing Values in Train Data ---")
    print((df_train.isnull().sum() / len(df_train)) * 100)
    
    print("\n--- Percentage of Missing Values in Test Data ---")
    print((df_test.isnull().sum() / len(df_test)) * 100)
    
    # Summary statistics
    print("\n--- Train Data Summary Statistics ---")
    print(df_train.describe())
    
    print("\n--- Test Data Summary Statistics ---")
    print(df_test.describe())
    
    # Identify categorical columns
    train_cat_columns = [col for col in df_train.columns if df_train[col].dtype == 'O']
    test_cat_columns = [col for col in df_test.columns if df_test[col].dtype == 'O']
    
    print("\n--- Categorical Columns in Train Data ---")
    print(train_cat_columns)
    
    print("\n--- Unique Values in Categorical Columns (Train) ---")
    print(df_train[train_cat_columns].nunique())
    
    print("\n--- Categorical Columns in Test Data ---")
    print(test_cat_columns)
    
    print("\n--- Unique Values in Categorical Columns (Test) ---")
    print(df_test[test_cat_columns].nunique())
    
    # Identify numerical columns
    train_num_columns = [col for col in df_train.columns if df_train[col].dtype in ['int64', 'float64']]
    test_num_columns = [col for col in df_test.columns if df_test[col].dtype in ['int64', 'float64']]
    
    print("\n--- Numerical Columns in Train Data ---")
    print(train_num_columns)
    
    print("\n--- Numerical Columns in Test Data ---")
    print(test_num_columns)
    
    # Check for duplicate rows
    print("\n--- Duplicate Rows in Train Data ---")
    print(df_train.duplicated().sum())
    
    print("\n--- Duplicate Rows in Test Data ---")
    print(df_test.duplicated().sum())
    
    # Correlation matrix (excluding non-numeric columns)
    print("\n--- Correlation Matrix ---")
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_train[train_num_columns].corr(), annot=True, cmap='coolwarm')
    plt.show()
    
    # Correlation with Target Variable
    print("\n--- Correlation with Target Variable ---")
    target_corr = df_train[train_num_columns].corr()[target_variable].sort_values(ascending=False)
    print(target_corr)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=target_corr.index, y=target_corr.values, palette='coolwarm')
    plt.xticks(rotation=90)
    plt.title(f'Feature Correlation with {target_variable}')
    plt.show()   
    
    # Distribution plots for numerical features
    print("\n--- Distribution of Numerical Features ---")
    df_train[train_num_columns].hist(figsize=(12, 10), bins=30)
    plt.show()
    
    # Box plots for outlier detection
    print("\n--- Box Plots for Outlier Detection ---")
    for col in train_num_columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df_train[col])
        plt.title(f'Box plot of {col}')
        plt.show()
    
    # Value counts for categorical features
    print("\n--- Value Counts for Categorical Columns ---")
    for col in train_cat_columns:
        print(f"\nValue counts for {col}:")
        print(df_train[col].value_counts())


eda_pipeline(df_train, df_test)


from sklearn.preprocessing import LabelEncoder

def data_preprocessing_pipeline(df_train, df_test):
    """
    Preprocess the dataset by handling missing values and encoding categorical variables.
    """
    # Fill missing values
    for column in df_train.columns:
        if df_train[column].dtype == 'object':
            mode_value = df_train[column].mode()[0]  # Fill categorical with mode
            df_train[column].fillna(mode_value, inplace=True)
        elif df_train[column].dtype in ['int64', 'float64']:
            mean_value = df_train[column].mean()  # Fill numerical with mean
            df_train[column].fillna(mean_value, inplace=True)
    
    for column in df_test.columns:
        if df_test[column].dtype == 'object':
            mode_value = df_test[column].mode()[0]
            df_test[column].fillna(mode_value, inplace=True)
        elif df_test[column].dtype in ['int64', 'float64']:
            mean_value = df_test[column].mean()
            df_test[column].fillna(mean_value, inplace=True)
    
    # Encode categorical features
    label_encoders = {}
    for column in df_train.columns:
        if df_train[column].dtype == 'object':
            le = LabelEncoder()
            df_train[column] = le.fit_transform(df_train[column].astype(str))
            label_encoders[column] = le  # Store encoder for consistency
    
    for column in df_test.columns:
        if df_test[column].dtype == 'object':
            if column in label_encoders:
                le = label_encoders[column]
                # Handle unseen labels by assigning -1
                df_test[column] = df_test[column].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            else:
                df_test[column] = -1  # Assign -1 if encoder was not created in df_train
    
    return df_train, df_test


df_train, df_test = data_preprocessing_pipeline(df_train, df_test)


from sklearn.preprocessing import StandardScaler

def standardize_data(df_train, df_test):
    """
    Standardize all numerical features using StandardScaler,
    ensuring both train and test have the same columns, while preserving the target variable.
    """
    # Separate target column from train data
    target_values = df_train[target_variable]
    df_train = df_train.drop(columns=[target_variable])
    
    # Ensure both datasets have the same feature columns
    common_columns = df_train.columns.intersection(df_test.columns)
    df_train = df_train[common_columns]
    df_test = df_test[common_columns]
    
    # Initialize StandardScaler
    scaler = StandardScaler()
    
    # Fit on train data and transform both train and test data
    df_train_scaled = pd.DataFrame(scaler.fit_transform(df_train), columns=common_columns)
    df_test_scaled = pd.DataFrame(scaler.transform(df_test), columns=common_columns)
    
    # Reattach the target column to the scaled train data
    df_train_scaled[target_variable] = target_values.reset_index(drop=True)
    
    return df_train_scaled, df_test_scaled


df_train_scaled, df_test_scaled = standardize_data(df_train, df_test)


df_train_scaled.head(5)


df_test_scaled.head(5)


df_train.head(5)


df_test.head(5)


X = df_train.drop(columns=[target_variable])
y = df_train[target_variable]


X1 = df_train_scaled.drop(columns=[target_variable])
y1 = df_train_scaled[target_variable]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)


import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
import numpy as np

# Split data (ensure stratification)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define objective function for Optuna
def objective(trial):
    # Hyperparameters to tune
    n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=50)
    max_depth = trial.suggest_int('max_depth', 3, 30, step=3)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    
    # Create model with suggested hyperparameters
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    # Perform 5-fold cross-validation and compute AUC
    auc_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)

    return np.mean(auc_scores)  # Optuna maximizes mean AUC score

# Run Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)  # Increase trials for better tuning

# Best hyperparameters
print("Best hyperparameters:", study.best_params)
best_params = study.best_params


best_params_rf={'n_estimators': 800, 'max_depth': 15, 
                'min_samples_split': 3, 'min_samples_leaf': 10,
                'max_features': 'log2'}


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize models
rf_model = RandomForestClassifier(n_estimators=500, random_state=42)
xgb_model = XGBClassifier(n_estimators=500, use_label_encoder=False, eval_metric='logloss', random_state=42)
cat_model = CatBoostClassifier(n_estimators=500, verbose=0, random_state=42)
lgbm_model = LGBMClassifier(n_estimators=500, random_state=42,verbose=-1)

# Train all models
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
cat_model.fit(X_train, y_train)
lgbm_model.fit(X_train, y_train)

# Make predictions on X_test
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)
cat_pred = cat_model.predict(X_test)
lgbm_pred = lgbm_model.predict(X_test)

# Accuracy Score
print("Accuracy Score rf is ",accuracy_score(y_test,rf_pred))
print("Accuracy Score xgb is ",accuracy_score(y_test,xgb_pred))
print("Accuracy Score catboost is ",accuracy_score(y_test,cat_pred))
print("Accuracy Score lgbm is ",accuracy_score(y_test,lgbm_pred))


# Train final model using best parameters
best_rf_model = RandomForestClassifier(**best_params_rf, random_state=42, n_jobs=-1)
best_rf_model.fit(X_train, y_train)

# Predict and evaluate
rf_pred = best_rf_model.predict(X_test)
print("Optimized RF Accuracy:", accuracy_score(y_test, rf_pred))


# Initialize models
rf_model1 = RandomForestClassifier(n_estimators=500, random_state=42)
xgb_model1 = XGBClassifier(n_estimators=500, use_label_encoder=False, eval_metric='logloss', random_state=42)
cat_model1= CatBoostClassifier(n_estimators=500, verbose=0, random_state=42)
lgbm_model1 = LGBMClassifier(n_estimators=500, random_state=42,verbose=-1)

# Train all models
rf_model1.fit(X_train1, y_train1)
xgb_model1.fit(X_train1, y_train1)
cat_model1.fit(X_train1, y_train1)
lgbm_model1.fit(X_train1, y_train1)

# Make predictions on X_test1
rf_pred1 = rf_model1.predict(X_test1)
xgb_pred1 = xgb_model1.predict(X_test1)
cat_pred1 = cat_model1.predict(X_test1)
lgbm_pred1 = lgbm_model1.predict(X_test1)

# Accuracy Score
print("Accuracy Score rf is ",accuracy_score(y_test1,rf_pred1))
print("Accuracy Score xgb is ",accuracy_score(y_test1,xgb_pred1))
print("Accuracy Score catboost is ",accuracy_score(y_test1,cat_pred1))
print("Accuracy Score lgbm is ",accuracy_score(y_test1,lgbm_pred1))



# Tune Class Weights for Imbalanced Data
from sklearn.utils.class_weight import compute_class_weight
# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train1), y=y_train1)
class_weights = dict(enumerate(class_weights))
print(class_weights)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC
from tensorflow.keras.regularizers import l2

early_stopping = EarlyStopping(monitor='val_auc', patience=20, restore_best_weights=True, mode='max')

model = Sequential([
    Input(shape=(X_train1.shape[1],)),

    Dense(256, kernel_initializer='he_normal', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.4),

    Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),

    Dense(64, kernel_initializer='he_normal', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),

    Dense(32, kernel_initializer='he_normal', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.2),

    Dense(16, kernel_initializer='he_normal', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Activation('relu'),

    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=0.0005)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[AUC(name='auc')])

history = model.fit(X_train1, y_train1, epochs=200, batch_size=64,
                    validation_data=(X_test1, y_test1), callbacks=[early_stopping],
                     verbose=1)


y_pred_keras = model.predict(X_test1)


y_pred_keras_binary = (y_pred_keras > 0.5).astype(int)  # Convert probabilities to 0 or 1
print("Accuracy Score keras DL model is ", accuracy_score(y_test1, y_pred_keras_binary))



# final prediction on test data 
final_prediction=best_rf_model.predict(df_test)
final_prediction_probability=best_rf_model.predict_proba(df_test)[:, 1] 


sample_submission.head(5)


sample_submission["rainfall"]=final_prediction_probability
sample_submission.to_csv('submission.csv',index=False)


sample_submission.head(5)




