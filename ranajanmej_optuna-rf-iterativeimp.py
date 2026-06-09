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


import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train_df.head()


train_df['Time_spent_Alone'].value_counts()


train_df['Stage_fear'].value_counts()


train_df['Social_event_attendance'].value_counts()


train_df['Post_frequency'].value_counts()


train_df['Going_outside'].value_counts()


train_df['Friends_circle_size'].value_counts()


train_df.head()


outliers_extr = train_df[(train_df['Time_spent_Alone'] == 11) & (train_df['Friends_circle_size'] == 0) & (train_df['Personality'] == 'Extrovert')]


train_df[(train_df['Time_spent_Alone'] == 10) & (train_df['Friends_circle_size'] == 0) & (train_df['Personality'] == 'Extrovert')]


outliers_1 = train_df[(train_df['Time_spent_Alone'] >= 9) & (train_df['Friends_circle_size'] <= 2) & (train_df['Personality'] == 'Extrovert')]


train_df[(train_df['Time_spent_Alone'] >= 9) & (train_df['Friends_circle_size'] <= 2) & (train_df['Personality'] == 'Extrovert')].shape


outliers_2 = train_df[(train_df['Time_spent_Alone'] <= 1) & (train_df['Friends_circle_size'] > 10) & (train_df['Personality'] == 'Introvert')]


train_df.columns


column = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
       'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
       'Post_frequency', 'Personality']


for col in column:
    print(f"{col} : {outliers_2[col].value_counts()}")


outlier_indexes = outliers_1.index.union(outliers_2.index)
outlier_indexes.shape


train_df_cleaned = train_df.drop(index=outlier_indexes)


train_df_cleaned.head()


train_df_cleaned[(train_df_cleaned['Time_spent_Alone'] >= 9) & (train_df_cleaned['Friends_circle_size'] <= 2) & (train_df_cleaned['Personality'] == 'Extrovert')]


X = train_df_cleaned.drop(columns = 'Personality',axis = 1)
y = train_df_cleaned['Personality']


X.head()


categorical_col = ['Stage_fear','Drained_after_socializing']
X_encoded = pd.get_dummies(X,columns = categorical_col,drop_first = True)


test_df_encoded = pd.get_dummies(test_df,columns = categorical_col,drop_first = True)


test_df_encoded


X_encoded.columns


# from sklearn.impute import KNNImputer
# # Create the imputer
# imputer = KNNImputer(n_neighbors=3)

# # Fit and transform the data
# train_df_imputed = pd.DataFrame(imputer.fit_transform(X_encoded), columns=X_encoded.columns)
# test_df_imputed = pd.DataFrame(imputer.transform(test_df_encoded), columns=test_df_encoded.columns)


from sklearn.experimental import enable_iterative_imputer  # ğŸ‘ˆ required to enable it
from sklearn.impute import IterativeImputer
import pandas as pd

# Create the imputer
imputer = IterativeImputer(max_iter=10, random_state=42)

# Fit on train and transform both train and test
train_df_imputed = pd.DataFrame(imputer.fit_transform(X_encoded), columns=X_encoded.columns)
test_df_imputed = pd.DataFrame(imputer.transform(test_df_encoded), columns=test_df_encoded.columns)


train_df_imputed.isnull().sum()


train_df_imputed.head()


cols_to_scale = ['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size',
                'Post_frequency']


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_df_imputed[cols_to_scale] = scaler.fit_transform(train_df_imputed[cols_to_scale])
test_df_imputed[cols_to_scale] = scaler.transform(test_df_imputed[cols_to_scale])


train_df_imputed.head()


test_df_imputed.head()


y.head()


y.value_counts()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)


y_encoded


target_label = pd.DataFrame({
    'Personality' : y_encoded
})


target_label['Personality'].value_counts()


target_label.head()


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train_df_imputed,target_label['Personality'],test_size = 0.2,random_state = 42)


y_train


X_train.head()


import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

# You already have:
# X_train, X_test, y_train, y_test
# test_df_imputed â€“ the final data to predict on

# Step 1: Compare models with cross-validation
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVC": SVC(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier()
}

print("ğŸ”� Cross-validating base models...")
cv_scores = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    avg_score = scores.mean()
    cv_scores[name] = avg_score
    print(f"{name}: {avg_score:.4f}")

# Step 2: Select best model
best_model_name = max(cv_scores, key=cv_scores.get)
print(f"\nâœ… Best model based on CV accuracy: {best_model_name}")

# Step 3: Hyperparameter tuning with Optuna (cross-validation based)
def objective(trial):
    if best_model_name == "LogisticRegression":
        C = trial.suggest_float("C", 1e-4, 10.0, log=True)
        solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs", "saga"])
        model = LogisticRegression(C=C, solver=solver, max_iter=1000)

    elif best_model_name == "SVC":
        C = trial.suggest_float("C", 0.1, 10)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        model = SVC(C=C, kernel=kernel, gamma=gamma)

    elif best_model_name == "DecisionTree":
        max_depth = trial.suggest_int("max_depth", 2, 32)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)

    elif best_model_name == "RandomForest":
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 2, 32)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    elif best_model_name == "XGBoost":
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 12)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
        model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, use_label_encoder=False, eval_metric='logloss')

    elif best_model_name == "LightGBM":
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 12)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
        model = LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)

    # Cross-validation score
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return score.mean()

# Step 4: Optimize
print("\nğŸ”§ Tuning best model with Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print(f"\nğŸ�† Best hyperparameters for {best_model_name}:")
print(study.best_params)
print(f"Best CV accuracy after tuning: {study.best_value:.4f}")

# Step 5: Retrain on full training set with best hyperparams
if best_model_name == "LogisticRegression":
    best_model = LogisticRegression(**study.best_params, max_iter=1000)

elif best_model_name == "SVC":
    best_model = SVC(**study.best_params)

elif best_model_name == "DecisionTree":
    best_model = DecisionTreeClassifier(**study.best_params)

elif best_model_name == "RandomForest":
    best_model = RandomForestClassifier(**study.best_params)

elif best_model_name == "XGBoost":
    best_model = XGBClassifier(**study.best_params, use_label_encoder=False, eval_metric='logloss')

elif best_model_name == "LightGBM":
    best_model = LGBMClassifier(**study.best_params)

# Final training
best_model.fit(X_train, y_train)

# Optional: evaluate on test set
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nğŸ“Š Final Test Accuracy: {test_accuracy:.4f}")

# Step 6: Predict on test_df_imputed
final_predictions = best_model.predict(test_df_imputed)

print("\nâœ… Prediction on test_df_imputed complete.")


fin_df = pd.DataFrame({
    'id' :test_df_imputed['id'].astype(int),
    'Personality' : final_predictions
})


fin_df['Personality'].value_counts()


fin_df['Personality'] = fin_df['Personality'].replace({0:'Extrovert',1:'Introvert'})


fin_df.head()


fin_df.to_csv("bes_model2.csv",index = False)




