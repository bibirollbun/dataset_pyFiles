#library installations
!pip install catboost


import gdown
from google.colab import drive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, make_scorer, roc_curve, auc, roc_auc_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import pickle


#drive.mount('/content/drive')

train_path = "/kaggle/input/playground-series-s5e7/train.csv"
test_path = "/kaggle/input/playground-series-s5e7/test.csv"
submission_file_path = "/kaggle/input/playground-series-s5e7/sample_submission.csv"


#file_path_training = '/content/drive/MyDrive/Projects/Predicting_Introverts_Extroverts/Intro_Extro_train.csv'
#file_path_testing = '/content/drive/MyDrive/Projects/Predicting_Introverts_Extroverts/Intro_Extro_test.csv'


training_data = pd.read_csv(train_path)


training_data.head()


training_data = training_data.iloc[:,1:]
training_data.head()


# summary of numerical attributes
training_data.describe()


print(training_data.isna().sum(),'\n')
print(training_data.isnull().sum())


# more useful when target variable is numeric
corr_matrix = training_data.corr(numeric_only=True)
corr_matrix


# Dividing the training data into independent and dependent features
X_train = training_data.iloc[:,:-1]


X_train.head()


y_train = training_data.iloc[:,-1]


y_train.head()


numeric_features = training_data.select_dtypes(include=['number']).columns.tolist()
numeric_features


categorical_features = training_data.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_features


training_data[numeric_features].hist(bins=50, figsize=(12, 8))
plt.show()


for categorical_feature in categorical_features:
  plt.figure(figsize=(8, 4))
  training_data[categorical_feature].value_counts().plot(kind='bar', color='blue', alpha=0.7)
  plt.title(f'Categorization of {categorical_feature}')
  plt.xlabel(categorical_feature)
  plt.ylabel('Frequency')
  plt.show()


# Models to train - Logistic Regression, Random Forest, XGBoost, SVM


# Preprocessing steps
numeric_full = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

numeric_no_scaling = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_encoded = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])


# Column transformers

numeric_features_present = [col for col in numeric_features if col in X_train.columns]
categorical_features_present = [col for col in categorical_features if col in X_train.columns]

full_preprocessor = ColumnTransformer([
    ('num', numeric_full, numeric_features_present),
    ('cat', categorical_encoded, categorical_features_present)
])

no_scaling_preprocessor = ColumnTransformer([
    ('num', numeric_no_scaling, numeric_features_present),
    ('cat', categorical_encoded, categorical_features_present)
])



# 4. Define model + pipeline combinations
model_pipelines = {
    'Logistic Regression': Pipeline([
        ('preprocessor', full_preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ]),

    'SVM': Pipeline([
        ('preprocessor', full_preprocessor),
        ('classifier', SVC(probability=True))
    ]),

    'Random Forest': Pipeline([
        ('preprocessor', no_scaling_preprocessor),
        ('classifier', RandomForestClassifier())
    ])
}


# 5. Define param grids (customize as needed)
param_grids = {
    'Logistic Regression': {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__penalty': ['l2']
    },
    'SVM': {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['rbf', 'linear']
    },
    'Random Forest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None]
    }
}


X_train.head()


y_train.head()


y_train.unique()


def train_and_select_model(name, pipeline, param_grid, X_train, y_train, scoring):
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring=scoring,
    refit='accuracy',
    n_jobs=-1,
    verbose=1
    )
    grid.fit(X_train, y_train)

    print(f"\n{name}")
    print(f"Best parameters: {grid.best_params_}")
    print(f"Accuracy: {grid.best_score_:.4f}")
    print(f"Precision: {grid.cv_results_['mean_test_precision'][grid.best_index_]:.4f}")
    print(f"Recall: {grid.cv_results_['mean_test_recall'][grid.best_index_]:.4f}")

    return grid.best_estimator_


trained_models = {}

scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='weighted', zero_division=0),
    'recall': make_scorer(recall_score, average='weighted')
}

for name, pipeline in model_pipelines.items():
    best_model = train_and_select_model(name, pipeline, param_grids[name], X_train, y_train, scoring)
    trained_models[name] = best_model


testing_data = pd.read_csv(test_path)


testing_data.head()


X_test = testing_data.iloc[:,1:]


X_test.head()


# Prediction using Logistic Regression
y_pred = trained_models['Logistic Regression'].predict(X_test)


y_pred


ids = testing_data['id']

prediction_results = pd.DataFrame({
        'id': ids,
        'loan_status': y_pred
    })

prediction_results.to_csv('Predictions_Introvert_Extrovert.csv', index=False)


y_pred_Logistic_prob = trained_models['Logistic Regression'].predict_proba(X_test)


y_pred_Logistic_prob


Logistic_df = pd.DataFrame(y_pred_Logistic_prob, columns=["Class Extrovert", "Class Introvert"])


Logistic_predictions_Probabilities = pd.concat([prediction_results,Logistic_df], axis=1)

Logistic_predictions_Probabilities.to_csv("Logistic_predictions_Probabilities.csv", index=False)


voting_clf = VotingClassifier(
    estimators=[
        ('lr', trained_models['Logistic Regression']),
        ('rf', trained_models['Random Forest']),
        ('svm', trained_models['SVM'])
    ],
    voting='soft'
)


voting_clf.fit(X_train, y_train)


y_pred = voting_clf.predict(X_test)
y_proba = voting_clf.predict_proba(X_test)


y_proba


ids = testing_data['id']

prediction_results = pd.DataFrame({
        'id': ids,
        'loan_status': y_pred
    })

prediction_results.to_csv('Predictions_Ensemble.csv', index=False)


y_pred


### XGBoost model


le = LabelEncoder()


y_train_encoded = le.fit_transform(y_train)


y_train_encoded


X_train.head()


combined_df = pd.concat([X_train, X_test], axis=0)
cat_cols = combined_df.select_dtypes(include="object").columns.tolist()

encoder = OrdinalEncoder()
combined_df[cat_cols] = encoder.fit_transform(combined_df[cat_cols])

X_train_XG = combined_df.iloc[:len(X_train)].reset_index(drop=True)
X_test_XG = combined_df.iloc[len(X_train):].reset_index(drop=True)


X_train_XG.head()


X_test_XG.head()


param_grid_XG = {
    'max_depth': [3,4, 5],
    'learning_rate': [0.1, 0.01],
    'subsample': [0.8, 1.0],
    "colsample_bytree": [0.8],
    'n_estimators': [100],
    "eta": [0.1]
}


xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)


cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


grid_search_XG = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid_XG,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=1
)


grid_search_XG.fit(X_train_XG, y_train_encoded)


y_pred_XG = grid_search_XG.predict(X_test_XG)


y_pred_XG


submission_data = pd.read_csv(submission_file_path)


submission_data["Personality"] = le.inverse_transform(y_pred_XG)
submission_data.to_csv("XGBoost_predictions.csv", index=False)


submission_data.head()


y_proba_XG = grid_search_XG.predict_proba(X_test_XG)


y_proba_XG


XGBoost_df = pd.DataFrame(y_proba_XG, columns=["Class Extrovert", "Class Introvert"])


XGBoost_predictions_Probabilities = pd.concat([submission_data,XGBoost_df], axis=1)

XGBoost_predictions_Probabilities.to_csv("XGBoost_predictions_Probabilities.csv", index=False)


XGBoost_predictions_Probabilities.head()

