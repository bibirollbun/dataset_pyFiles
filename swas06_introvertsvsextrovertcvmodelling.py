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


from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import optuna
import xgboost as xgb
from sklearn.metrics import classification_report,accuracy_score
import warnings
warnings.filterwarnings('ignore')




df_train =pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


df_train.columns = df_train.columns.str.replace(' ', '_').str.lower()
df_test.columns = df_test.columns.str.replace(' ', '_').str.lower()


df_train.info(),df_test.info()


df_train.head(3),df_test.head(3)


df_train.isnull().sum(),df_test.isnull().sum()


 for col in df_train.columns:
        if df_train[col].dtype == 'object':  # Categorical column
            df_train[col].fillna(df_train[col].mode()[0], inplace=True)
        else:  # Numeric column
            df_train[col].fillna(df_train[col].mean(), inplace=True)


 for col in df_test.columns:
        if df_test[col].dtype == 'object':  # Categorical column
            df_test[col].fillna(df_test[col].mode()[0], inplace=True)
        else:  # Numeric column
            df_test[col].fillna(df_test[col].mean(), inplace=True)


df_train.isnull().sum(),df_test.isnull().sum()


df_train['social_score'] = (df_train['social_event_attendance'] + df_train['going_outside'] + df_train['friends_circle_size'])
# Introvert-Tendency Proxy
df_train['introversion_tendency'] = (df_train['time_spent_alone'] - df_train['social_score'])
df_train['social_imbalance'] = df_train['social_event_attendance'] - df_train['time_spent_alone']



df_test['social_score'] = (df_test['social_event_attendance'] + df_test['going_outside'] + df_test['friends_circle_size'])
# Introvert-Tendency Proxy
df_test['introversion_tendency'] = (df_test['time_spent_alone'] - df_test['social_score'])
df_test['social_imbalance'] = df_test['social_event_attendance'] - df_test['time_spent_alone']




df_train.head(10)


df_train = df_train.drop(['social_event_attendance'], axis=1)
df_test = df_test.drop(['social_event_attendance'], axis=1)




encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Fit on train only
encoder.fit(df_train[['stage_fear', 'drained_after_socializing']])

# Transform both
df_train[['stage_fear', 'drained_after_socializing']] = encoder.transform(df_train[['stage_fear', 'drained_after_socializing']])
df_test[['stage_fear', 'drained_after_socializing']] = encoder.transform(df_test[['stage_fear', 'drained_after_socializing']])




le = LabelEncoder()
df_train['personality'] = le.fit_transform(df_train['personality'])


df_train.head(3)


col_to_scale=['time_spent_alone','social_score','going_outside','friends_circle_size','post_frequency']




scaler = StandardScaler()

# Fit the scaler on train data and transform both train and test
df_train[col_to_scale] = scaler.fit_transform(df_train[col_to_scale])
df_test[col_to_scale] = scaler.transform(df_test[col_to_scale])


df_train.head(3)


X=df_train.drop(columns=['personality'])
y=df_train['personality']


from xgboost import XGBClassifier
import optuna
from sklearn.model_selection import cross_val_score
def objective(trial):
    param = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        'use_label_encoder': False,
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }

    model = XGBClassifier(**param)
    score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    return score


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)


best_params = study.best_params


final_model = xgb.XGBClassifier(
    **best_params,
    objective='binary:logistic',
    use_label_encoder=False,
    verbosity=0
)




# Initialize Stratified K-Fold
from sklearn.model_selection import StratifiedKFold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# List to store the accuracies for each fold
accuracies = []

# Stratified K-Fold Cross-Validation
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Train the Voting Classifier model
    final_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = final_model.predict(X_test)  # 
    
    # Evaluate the accuracy for this fold
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

print(f"Average Accuracy across {n_splits} folds: {np.mean(accuracies):.4f}")


final_model.fit(X, y)  # Fit on the whole data for final model
final_predictions = final_model.predict(X)
print("Final Classification Report on Entire Dataset:\n", classification_report(y, final_predictions))


final_model.fit(X,y)


importances = pd.Series(final_model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh')


test_probs = final_model.predict(df_test)


test_labels = le.inverse_transform(test_probs)


test_labels


submission = pd.DataFrame({
    'id': df_test['id'],
    'Personality': test_labels
})
submission.to_csv('submission.csv', index=False)



submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

