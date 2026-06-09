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


train=pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
k=pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
sub=pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


train=train.drop("id",axis=1)
train=pd.concat([train,k],axis=0)


train=train.rename(columns={'Temparature': 'Temperature'})
test=test.rename(columns={'Temparature': 'Temperature'})


train


test=test.drop("id",axis=1)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

# PREPROCESSING FUNCTION
def preprocess_data(train_df, test_df=None):
    # Process features
    X_train = train_df.drop('Fertilizer Name', axis=1)
    y_train = train_df['Fertilizer Name']
    
    # Initialize encoders and scaler
    label_encoders = {}
    categorical_cols = ['Soil Type', 'Crop Type']
    numerical_cols = ['Temperature', 'Humidity', 'Moisture', 
                    'Nitrogen', 'Potassium', 'Phosphorous']
    
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        label_encoders[col] = le
    
    
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    
    
    le_target = LabelEncoder()
    y_train_encoded = le_target.fit_transform(y_train)
    
    # Process test data if provided
    if test_df is not None:
        X_test = test_df.copy()
        for col in categorical_cols:
            X_test[col] = label_encoders[col].transform(X_test[col])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
        return X_train, y_train_encoded, X_test, label_encoders, scaler, le_target
    
    return X_train, y_train_encoded, label_encoders, scaler, le_target

# OPTUNA OPTIMIZATION
def optimize_xgb(X_train, y_train, X_val, y_val, n_classes):
    def objective(trial):
        params = {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'num_class': n_classes,
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'alpha': trial.suggest_float('alpha', 0, 1),
            'lambda': trial.suggest_float('lambda', 0, 1),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'random_state': 42
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                 early_stopping_rounds=20, verbose=False)
        return accuracy_score(y_val, model.predict(X_val))
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=5, show_progress_bar=True)
    return study.best_params

# MAIN TRAINING FUNCTION
def train_and_predict(train, test):
    # Preprocess data
    X_train, y_train, X_test, label_encoders, scaler, le_target = preprocess_data(train, test)
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    
    print("Running Optuna optimization...")
    best_params = optimize_xgb(X_train, y_train, X_val, y_val, len(le_target.classes_))
    
    # Train final model with best params
    final_params = {
        **best_params,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': len(le_target.classes_),
        'random_state': 42
    }
    
    print("\nTraining final model with best parameters...")
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
    
    # Evaluate on validation set
    val_pred = final_model.predict(X_val)
    print("\nValidation Performance:")
    print(classification_report(y_val, val_pred, target_names=le_target.classes_))
    print(f"Validation Accuracy: {accuracy_score(y_val, val_pred):.4f}")
    
    # Predict on test data
    test_pred = final_model.predict(X_test)
    test_proba = final_model.predict_proba(X_test)
    test_pred_labels = le_target.inverse_transform(test_pred)
    
    return final_model, test_pred_labels, test_proba

# EXAMPLE USAGE:
# Assuming you have train and test DataFrames loaded
final_model, predictions, probabilities = train_and_predict(train, test)
print("Test Predictions:", predictions)
print("Prediction Probabilities:", probabilities)


X_train, y_train, label_encoders, scaler, le_target = preprocess_data(train)



class_names = le_target.classes_  # Get fertilizer names from label encoder

# Get indices of top 3 predictions for each row
top3_indices = np.argsort(-probabilities, axis=1)[:, :3]

# Create DataFrame with ID and top 3 predictions
results = pd.DataFrame({
    'id': range(len(test)),  # Or use range(len(test)) if you want sequential IDs
    'pred1': [class_names[i[0]] for i in top3_indices],
    'pred2': [class_names[i[1]] for i in top3_indices],
    'pred3': [class_names[i[2]] for i in top3_indices],
    'prob1': [probabilities[row, i[0]] for row, i in enumerate(top3_indices)],
    'prob2': [probabilities[row, i[1]] for row, i in enumerate(top3_indices)],
    'prob3': [probabilities[row, i[2]] for row, i in enumerate(top3_indices)]
})

print("Top 3 Predictions for Each Sample:")
print(results.head())  


results['Fertilizer Name'] = results['pred1'] + ' ' + results['pred2'] + ' ' + results['pred3']


results


sub.drop("Fertilizer Name",axis=1)


sub["Fertilizer Name"]=results["Fertilizer Name"]


sub


sub.to_csv('submission.csv', index=False)


































