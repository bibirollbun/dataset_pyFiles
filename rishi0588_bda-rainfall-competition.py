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


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

test_data = test_data.dropna()

# Preprocess the data
# Assuming 'RainTomorrow' is the target variable and the rest are features
X = train_data.drop(columns=['rainfall'])
y = train_data['rainfall']

# Handle categorical variables if any
X = pd.get_dummies(X)

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_data = scaler.transform(pd.get_dummies(test_data))

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validate the model
y_pred = model.predict(X_val)
print(f'Validation Accuracy: {accuracy_score(y_val, y_pred)}')

# Predict on the test set
test_predictions = model.predict(test_data)

# Prepare submission
submission = pd.DataFrame({
    'Id': range(len(test_predictions)),  # Creates an index from 0 to len(test_predictions) - 1
    'RainTomorrow': test_predictions
})
submission.to_csv('submission2.csv', index=False)



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Store test IDs before handling missing values
test_ids = test_data['Id'].values if 'Id' in test_data.columns else range(1, len(test_data) + 1)

# Fill NaN values instead of dropping them
test_data = test_data.fillna(test_data.median(numeric_only=True))  # Fill missing numerical values with median

# Preprocess the data
X = train_data.drop(columns=['rainfall'])  # Features
y = train_data['rainfall']  # Target variable

# Handle categorical variables if any
X = pd.get_dummies(X)
test_data = pd.get_dummies(test_data)

# Align test_data with X to ensure feature consistency
X, test_data = X.align(test_data, join="left", axis=1, fill_value=0)

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_data = scaler.transform(test_data)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validate the model
y_pred = model.predict(X_val)
print(f'Validation Accuracy: {accuracy_score(y_val, y_pred)}')

# Predict on the test set
test_predictions = model.predict(test_data)

# Ensure ID and Predictions have the same length
assert len(test_ids) == len(test_predictions), f"Mismatch: {len(test_ids)} IDs vs {len(test_predictions)} predictions"

# Create submission file with lowercase 'id' column
submission = pd.DataFrame({
    'id': test_ids,  # Change column name to lowercase
    'RainTomorrow': test_predictions
})
submission.to_csv('submission5.csv', index=False)

print("Submission file saved as 'submission5.csv'")



print(f"Length of test_ids: {len(test_ids)}")
print(f"Length of test_predictions: {len(test_predictions)}")



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Store test IDs before modifying test_data
test_ids = test_data['id']  # Correct column name

# Handle missing values - Fill NaN with median (for numerical) or mode (for categorical)
train_data.fillna(train_data.median(numeric_only=True), inplace=True)
test_data.fillna(test_data.median(numeric_only=True), inplace=True)

# Preprocess the data
X = train_data.drop(columns=['rainfall'])  # Features
y = train_data['rainfall']  # Target variable

# One-hot encoding for categorical variables
X = pd.get_dummies(X)
test_data = pd.get_dummies(test_data)

# Align test_data with X to ensure feature consistency
X, test_data = X.align(test_data, join="left", axis=1, fill_value=0)

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_data = scaler.transform(test_data)

# Train the model with optimized parameters
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Validate the model
y_pred = model.predict(X_val)
print(f'Validation Accuracy: {accuracy_score(y_val, y_pred)}')

# Predict on the test set
test_predictions = model.predict(test_data)

# Ensure ID and Predictions have the same length
assert len(test_ids) == len(test_predictions), f"Mismatch: {len(test_ids)} IDs vs {len(test_predictions)} predictions"

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,  # Ensure 'id' column matches expected format
    'RainTomorrow': test_predictions
})
submission.to_csv('submission_final.csv', index=False)

print("Submission file saved as 'submission_final.csv'")



print(test_data.columns)



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer

# Load the dataset
train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Store test IDs
test_ids = test_data['id']  # Correct column name

# Drop ID column before processing
train_data.drop(columns=['id'], inplace=True)
test_data.drop(columns=['id'], inplace=True)

# Separate target variable before imputation
y = train_data['rainfall']
X = train_data.drop(columns=['rainfall'])

# Handle missing values using KNN Imputer
imputer = KNNImputer(n_neighbors=5)
X.iloc[:, :] = imputer.fit_transform(X)  # Fit on train_data without target column
test_data.iloc[:, :] = imputer.transform(test_data)  # Apply to test_data

# Feature Engineering: Create new features
X['temp_range'] = X['maxtemp'] - X['mintemp']
test_data['temp_range'] = test_data['maxtemp'] - test_data['mintemp']

X['humidity_sun_ratio'] = X['humidity'] / (X['sunshine'] + 1)
test_data['humidity_sun_ratio'] = test_data['humidity'] / (test_data['sunshine'] + 1)

# One-hot encoding for categorical variables
X = pd.get_dummies(X)
test_data = pd.get_dummies(test_data)

# Align test_data with X to ensure feature consistency
X, test_data = X.align(test_data, join="left", axis=1, fill_value=0)

# Feature Selection: Drop low-importance features using RandomForest importance
temp_model = RandomForestClassifier(n_estimators=100, random_state=42)
temp_model.fit(X, y)
feature_importances = pd.Series(temp_model.feature_importances_, index=X.columns)
important_features = feature_importances[feature_importances > 0.005].index  # Keeping only relevant features

X = X[important_features]
test_data = test_data[important_features]

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_data = scaler.transform(test_data)

# Train a more optimized RandomForest model
rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_split=5,
    random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Train Gradient Boosting model (can outperform RF)
gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)

# Validation
rf_pred = rf_model.predict(X_val)
gb_pred = gb_model.predict(X_val)

# Ensemble Model: Averaging predictions
final_pred = (rf_pred + gb_pred) / 2
final_pred = np.round(final_pred).astype(int)  # Convert to binary

print(f'RandomForest Accuracy: {accuracy_score(y_val, rf_pred)}')
print(f'GradientBoosting Accuracy: {accuracy_score(y_val, gb_pred)}')
print(f'Final Ensemble Accuracy: {accuracy_score(y_val, final_pred)}')

# Predict on the test set using ensemble
test_rf_pred = rf_model.predict(test_data)
test_gb_pred = gb_model.predict(test_data)

test_final_pred = (test_rf_pred + test_gb_pred) / 2
test_final_pred = np.round(test_final_pred).astype(int)

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'RainTomorrow': test_final_pred
})
submission.to_csv('submission_final_optimized.csv', index=False)

print("Optimized submission file saved as 'submission_final_optimized.csv'")



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import xgboost as xgb
import catboost as cb

# Load the dataset
train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Store test IDs
test_ids = test_data['id']
train_data.drop(columns=['id'], inplace=True)
test_data.drop(columns=['id'], inplace=True)

# Separate target variable
y = train_data['rainfall']
X = train_data.drop(columns=['rainfall'])

# Handle missing values (better method than KNNImputer)
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

X.iloc[:, :] = num_imputer.fit_transform(X)
test_data.iloc[:, :] = num_imputer.transform(test_data)

# Feature Engineering: Adding New Features
X['temp_range'] = X['maxtemp'] - X['mintemp']
test_data['temp_range'] = test_data['maxtemp'] - test_data['mintemp']

X['humidity_sun_ratio'] = X['humidity'] / (X['sunshine'] + 1)
test_data['humidity_sun_ratio'] = test_data['humidity'] / (test_data['sunshine'] + 1)

# Align test_data with X to ensure consistency
X, test_data = X.align(test_data, join="left", axis=1, fill_value=0)

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_data_scaled = scaler.transform(test_data)

# Cross-validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_preds = np.zeros(len(test_data))
cb_preds = np.zeros(len(test_data))

# Train models with cross-validation
for train_idx, val_idx in skf.split(X_scaled, y):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # XGBoost Model
    xgb_model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    
    # CatBoost Model
    cb_model = cb.CatBoostClassifier(n_estimators=300, learning_rate=0.03, depth=6, random_state=42, verbose=0)
    cb_model.fit(X_train, y_train)

    # Make predictions
    xgb_preds += xgb_model.predict(test_data_scaled) / skf.n_splits
    cb_preds += cb_model.predict(test_data_scaled) / skf.n_splits

# Ensemble Predictions
final_test_preds = (xgb_preds + cb_preds) / 2
final_test_preds = np.round(final_test_preds).astype(int)

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'RainTomorrow': final_test_preds
})
submission.to_csv('submission_optimized.csv', index=False)

print("Optimized submission file saved as 'submission_optimized.csv'")



import pandas as pd, numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
print("Train shape", train.shape )
train.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
print("Test shape:", test.shape )
test.head()


RMV = ['rainfall','id']
FEATURES = [c for c in train.columns if not c in RMV]
print("Our features are:")
print( FEATURES )


from sklearn.model_selection import KFold
from cuml.neighbors import KNeighborsClassifier


# WEIGHTS TO ADJUST IMPORTANCE OF FEATURES DURING KNN
WGT = {'day': 24, 'pressure': 1, 'maxtemp': 1, 'temparature': 1, 'mintemp': 1, 'dewpoint': 1, 'humidity': 1, 
       'cloud': 1, 'sunshine': 1, 'winddirection': 1, 'windspeed': 1}


%%time
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=777)
    
oof_knn = np.zeros(len(train))
pred_knn = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"rainfall"]    
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"rainfall"]
    x_test = test[FEATURES].copy()

    for c in FEATURES:
        m = x_train[c].mean()
        s = x_train[c].std()
        x_train[c] = WGT[c] * (x_train[c]-m)/s
        x_valid[c] = WGT[c] * (x_valid[c]-m)/s
        x_test[c] = WGT[c] * (x_test[c]-m)/s
        x_test[c] = x_test[c].fillna(0)
        x_train[c] = x_train[c].fillna(0)

    model = KNeighborsClassifier(n_neighbors=201, p=1)
    model.fit(x_train.values, y_train.values)

    # INFER OOF
    oof_knn[test_index] = model.predict_proba(x_valid.values)[:,1]
    # INFER TEST
    pred_knn += model.predict_proba(x_test.values)[:,1]

# COMPUTE AVERAGE TEST PREDS
pred_knn /= FOLDS



from sklearn.metrics import roc_auc_score
true = train.rainfall.values
m = roc_auc_score(true, oof_knn)
print(f"KNN CV Score AUC = {m:.3f}")


print("Best Public Notebook achieves LB = 0.954!")
best_public = pd.read_csv("/kaggle/input/lb-915-public-notebook/submission95427.csv")
display( best_public.head() )
best_public = best_public.rainfall.values


from scipy.stats import rankdata

print("Ensemble achieves LB = 0.961! Hooray!")
sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
sub.rainfall = -0.25 * rankdata( pred_knn ) + 1.25 * rankdata( best_public )
sub.rainfall = rankdata( sub.rainfall ) / len(sub)
print( sub.shape )
sub.to_csv(f"submission_ensemble.csv",index=False)
sub.head()




