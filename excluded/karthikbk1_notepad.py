import pandas as pd
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

# Placeholder function for exec_fit_and_proba
def exec_fit_and_proba(data, qheight=13):
    """
    Simulates model fitting and probability prediction.
    :param data: Input dataset (assumed to be a Pandas DataFrame or NumPy array)
    :param qheight: An optional parameter (not used in this example)
    :return: Simulated probability predictions
    """
    np.random.seed(42)  # For reproducibility
    return np.random.rand(len(data))  # Return random probabilities

# Simulated input dataset (Replace this with your actual 'fen_2__' data)
fen_2__ = pd.DataFrame({'feature1': np.random.rand(100), 'feature2': np.random.rand(100)})

# Calling the function without error
hyper_space_2__ = exec_fit_and_proba(fen_2__, qheight=13)

# Display the first 5 values
print(hyper_space_2__[:5])



import pandas as pd
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

# Define exec_fit_and_proba function
def exec_fit_and_proba(data, qheight=13):
    """
    Simulates model fitting and probability prediction.
    :param data: Input dataset (assumed to be a Pandas DataFrame or NumPy array)
    :param qheight: An optional parameter
    :return: Simulated probability predictions
    """
    np.random.seed(42)  # For reproducibility
    return np.random.rand(len(data))  # Return random probabilities

# Check if 'fen_2_h' exists; otherwise, create a placeholder dataset
try:
    fen_2_h
except NameError:
    print("Warning: 'fen_2_h' was not defined. Creating a placeholder dataset.")
    fen_2_h = pd.DataFrame({'feature1': np.random.rand(100), 'feature2': np.random.rand(100)})

# Now we can call exec_fit_and_proba without error
hyper_space_2_h = exec_fit_and_proba(fen_2_h, qheight=100)

# Display the first 5 values
print(hyper_space_2_h[:5])


import pandas as pd
import numpy as np

# Define exec_fit_and_proba function
def exec_fit_and_proba(data, qheight=13):
    """
    Simulates model fitting and probability prediction.
    :param data: Input dataset (assumed to be a Pandas DataFrame or NumPy array)
    :param qheight: An optional parameter
    :return: Simulated probability predictions
    """
    np.random.seed(42)
    return np.random.rand(len(data))

# Define exec_KNN__fit_and_proba function
def exec_KNN__fit_and_proba(data):
    """
    Simulates KNN model fitting and probability prediction.
    :param data: Input dataset (assumed to be a Pandas DataFrame or NumPy array)
    :return: Simulated probability predictions
    """
    np.random.seed(42)
    return np.random.rand(len(data))

# Create placeholder datasets if missing
try:
    fen_4__
except NameError:
    print("Warning: 'fen_4__' was not defined. Creating a placeholder dataset.")
    fen_4__ = pd.DataFrame({'feature1': np.random.rand(100), 'feature2': np.random.rand(100)})

try:
    fen_2__
except NameError:
    print("Warning: 'fen_2__' was not defined. Creating a placeholder dataset.")
    fen_2__ = pd.DataFrame({'feature1': np.random.rand(100), 'feature2': np.random.rand(100)})

try:
    fen_4_h
except NameError:
    print("Warning: 'fen_4_h' was not defined. Creating a placeholder dataset.")
    fen_4_h = pd.DataFrame({'feature1': np.random.rand(100), 'feature2': np.random.rand(100)})

try:
    fen_2_h
except NameError:
    print("Warning: 'fen_2_h' was not defined. Creating a placeholder dataset.")
    fen_2_h = pd.DataFrame({'feature1': np.random.rand(100), 'feature2': np.random.rand(100)})

try:
    fen_3__
except NameError:
    print("Warning: 'fen_3__' was not defined. Creating a placeholder dataset.")
    fen_3__ = pd.DataFrame({'feature1': np.random.rand(100), 'feature2': np.random.rand(100)})

# Execute functions
hyper_space_4__ = exec_fit_and_proba(fen_4__, qheight=50)
knn_hyper_space_2__ = exec_KNN__fit_and_proba(fen_2__)
hyper_space_4_h = exec_fit_and_proba(fen_4_h, qheight=150)
knn_hyper_space_2_h = exec_KNN__fit_and_proba(fen_2_h)
knn_hyper_space_3__ = exec_KNN__fit_and_proba(fen_3__)
knn_hyper_space_4__ = exec_KNN__fit_and_proba(fen_4__)
knn_hyper_space_4_h = exec_KNN__fit_and_proba(fen_4_h)

# Display results
print("hyper_space_4__:", hyper_space_4__[:5])
print("knn_hyper_space_2__:", knn_hyper_space_2__[:5])
print("hyper_space_4_h:", hyper_space_4_h[:5])
print("knn_hyper_space_2_h:", knn_hyper_space_2_h[:5])
print("knn_hyper_space_3__:", knn_hyper_space_3__[:5])
print("knn_hyper_space_4__:", knn_hyper_space_4__[:5])
print("knn_hyper_space_4_h:", knn_hyper_space_4_h[:5])



import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Suppress warnings
warnings.simplefilter('ignore')

# Load datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv", index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv", index_col='id')
train_extra = pd.read_csv("//kaggle/input/rainfall/Rainfall.csv")

# Preprocess train_extra
train_extra.columns = train_extra.columns.str.replace(' ', '')
train_extra['rainfall'] = train_extra['rainfall'].map({'no': 0, 'yes': 1})
train_extra['humidity'] = train_extra['humidity'].astype(float)
train_extra['cloud'] = train_extra['cloud'].astype(float)
train_features = list(train)
train_extra = train_extra[train_features]

# Combine datasets
train = pd.concat([train, train_extra], axis=0, ignore_index=True)
train = train.drop_duplicates()

# Align train and test features
features = list(test)
features.append('rainfall')
train = train[features]

# Fill missing values
train['winddirection'] = train['winddirection'].fillna(value=train['winddirection'].mean())
test['winddirection'] = test['winddirection'].fillna(value=test['winddirection'].mean())
train['windspeed'] = train['windspeed'].fillna(value=train['windspeed'].mean())
test['windspeed'] = test['windspeed'].fillna(value=test['windspeed'].mean())

# Prepare target and features
target = "rainfall"
X = train.drop(columns=[target])
y = train[target]
test_data = test.copy()

# Scaling and encoding
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_data)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Feature engineering with LDA
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X_scaled, y_encoded)
test_lda = lda.transform(test_scaled)

train['lda'] = X_lda
test['lda'] = test_lda

# Split data for training and validation
X_train, X_val, y_train, y_val = train_test_split(train.drop(columns=['rainfall']), y_encoded, test_size=0.2, random_state=42)

# Model training with Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_preds = log_reg.predict_proba(X_val)[:, 1]

# Evaluate Logistic Regression
log_reg_auc = roc_auc_score(y_val, log_reg_preds)

# Ensemble with Gradient Boosting
gbm = GradientBoostingClassifier(random_state=42)
param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1, 0.2], 'max_depth': [3, 5]}
grid_search = GridSearchCV(gbm, param_grid, scoring='roc_auc', cv=3)
grid_search.fit(X_train, y_train)

best_gbm = grid_search.best_estimator_
gbm_preds = best_gbm.predict_proba(X_val)[:, 1]

# Evaluate Gradient Boosting
gbm_auc = roc_auc_score(y_val, gbm_preds)

# Combine predictions for ensembling
final_preds = (log_reg_preds + gbm_preds) / 2
final_auc = roc_auc_score(y_val, final_preds)

print("Logistic Regression AUC:", log_reg_auc)
print("Gradient Boosting AUC:", gbm_auc)
print("Ensembled Model AUC:", final_auc)

# Make predictions on test data
test_preds = best_gbm.predict_proba(test)[:, 1]

# Create submission file
submission = pd.DataFrame({"id": test.index, "rainfall": test_preds})
submission.to_csv("submission_ensembled.csv", index=False)
submission.head(10)


