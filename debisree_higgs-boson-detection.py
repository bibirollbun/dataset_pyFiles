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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.feature_selection import mutual_info_classif


import shap
from shap import  summary_plot

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer
from sklearn.preprocessing import LabelEncoder


import xgboost as xgb

from xgboost import XGBClassifier


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import optuna

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor

from tqdm import tqdm 

from sklearn.metrics import mean_squared_error, r2_score,  mean_squared_log_error, roc_curve, roc_auc_score

from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',None)


train  = pd.read_csv(f"/kaggle/input/higgs-boson-detection-2025/train.csv")
test   = pd.read_csv(f"/kaggle/input/higgs-boson-detection-2025/test.csv")

train.head()


print(train.shape)
print(test.shape)


train.isnull().sum()


test.isnull().sum()


def detect_outlier_percentages(df):
    numeric_cols = df.select_dtypes(include='number').columns
    outlier_percentages = {}

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outlier_mask = (df[col] < lower) | (df[col] > upper)
        outlier_percentage = 100 * outlier_mask.sum() / len(df)
        outlier_percentages[col] = round(outlier_percentage, 2)

    return pd.Series(outlier_percentages).sort_values(ascending=False)

detect_outlier_percentages(train)


# Target == label

sns.countplot(data = train, x= 'label')
plt.show()



def plot_feature_distributions(df, label_col='label'):
    features = [col for col in df.columns if col != label_col and df[col].dtype in ['float64', 'int64']]

    for feature in features:
        plt.figure(figsize=(8, 4))

        sns.kdeplot(
            data=df[df[label_col] == True],
            x=feature,
            label='Label = 1',
            fill=True,
            alpha=0.5
        )

        sns.kdeplot(
            data=df[df[label_col] == False],
            x=feature,
            label='Label = 0',
            fill=True,
            alpha=0.5
        )

        plt.title(f'Distribution of {feature} by Label')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return



 plot_feature_distributions(train, label_col='label')


X = train.drop('label', axis=1)  
y = train['label']

# Compute Mutual Information scores
mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)

# Create DataFrame for easy plotting
mi_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
mi_df = mi_df.sort_values(by='MI Score', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(mi_df['Feature'], mi_df['MI Score'], color='skyblue')
plt.xlabel("Mutual Information Score")
plt.title("Feature Importance via Mutual Information")
plt.gca().invert_yaxis()  # most important on top
plt.tight_layout()
plt.show()


# Dropping columns:

# train = train.drop(columns=['f18', 'f6'], axis =1)
# train.head()



X = train.drop(columns=['label']).values
y = train['label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# model = Sequential([
#     Dense(64, input_dim=X_train.shape[1], activation='relu'),
#     Dropout(0.3),
#     Dense(32, activation='relu'),
#     Dropout(0.2),
#     Dense(1, activation='sigmoid')  # For binary classification
# ])



# model.compile(
#     loss='binary_crossentropy',
#     optimizer=Adam(learning_rate=0.001),
#     metrics=['accuracy']
# )


# early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# history = model.fit(
#     X_train, y_train,
#     validation_split=0.2,
#     epochs=100,
#     batch_size=32,
#     callbacks=[early_stop],
#     verbose=1
# )


# loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
# print(f'Test Accuracy: {accuracy:.4f}')





def objective(trial):
    # Define hyperparameters to tune
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-5, 1e1),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 1e1),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 1e1)
    }

    # Create the XGBoost model
    model = xgb.XGBClassifier(**param)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict_proba(X_test)[:, 1]
    
    # Calculate the AUC score
    auc = roc_auc_score(y_test, y_pred)
    return auc
    
# Create the Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Print the best hyperparameters and the corresponding AUC score
print('Best hyperparameters:', study.best_params)
print('Best AUC score:', study.best_value)


best_params = study.best_params
final_model = xgb.XGBClassifier(**best_params, random_state=42)

final_model.fit(X_train, y_train)



y_probs = final_model.predict_proba(X_test)[:, 1]



# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = roc_auc_score(y_test, y_probs)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
plt.show()



print(f"ROC AUC Score: {roc_auc:.4f}")



explainer = shap.TreeExplainer(final_model)
shap_values = explainer(X_test) 


shap.summary_plot(shap_values.values, X_test)




# Transform final test data
X_final_test_scaled = scaler.transform(test)




# Predict probabilities for the positive class (class = 1)
y_final_probs = final_model.predict_proba(X_final_test_scaled)[:, 1]

# Convert probabilities to binary predictions using threshold = 0.5
y_final_pred = (y_final_probs >= 0.5).astype(int)


submission = pd.DataFrame({
    'Id': [f"{float(i):.18e}" for i in range(len(y_final_pred))],
    'Predicted': y_final_pred
})

submission.to_csv('submission.csv', index=False, float_format='%.6f')





