!pip install -r /kaggle/input/requirement/requirements.txt

# Data manipulation
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Modelling
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from mlxtend.classifier import EnsembleVoteClassifier

from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, History
from tensorflow.keras import regularizers

# Evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, make_scorer
from yellowbrick.classifier import ROCAUC

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Warnings
import warnings

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)


# Load data
train_dataset = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
test_dataset = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')

# Display the dataset
print("Train data:")
train_dataset.head()


print("Test data:")
test_dataset


# Display the data types of the features
train_dataset.info()


# Check for missing values
train_dataset.isnull().sum()


# Check for duplicate records
train_dataset.duplicated().sum()


# Summary dataset
train_dataset.describe()


# Drop the 'id' column
train_dataset.drop(columns=['id'], inplace=True)
train_dataset.info()


sns.countplot(x='loan_status', data=train_dataset)
plt.title("Distribution of loan approvals in the dataset")
plt.xlabel("Loan Status")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1], labels=['Denied (0)', 'Approved (1)'])
total = len(train_dataset['loan_status'])
approved = train_dataset['loan_status'].sum()
denied = total - approved
plt.text(0, denied, f"{denied} ({round(denied/total*100, 1)}%)", ha='center')
plt.text(1, approved, f"{approved} ({round(approved/total*100, 1)}%)", ha='center')
plt.show()


# Get the numerical columns
cols = train_dataset.select_dtypes(include=['number']).columns.tolist()
target_variable = 'loan_status'

# Remove the target variable from the list
if target_variable in cols:
    cols.remove(target_variable)
    
# Grid for the plots
n_rows = len(cols)
fig, axes = plt.subplots(n_rows, 2, figsize=(12, n_rows * 4))

for i, col in enumerate(cols):
    # Distribution of the feature with respect to the target variable
    sns.boxplot(data=train_dataset, y=col, x='loan_status', ax=axes[i, 0])
    axes[i, 0].set_title(f'{col}')
    
    # Histogram of the feature with respect to the target variable
    sns.histplot(data=train_dataset, x=col, hue='loan_status', ax=axes[i, 1], kde=True)
    axes[i, 1].set_title(f'{col}')
    
plt.tight_layout()
plt.show()


numerical_features = [
    'person_age',
    'person_income',
    'person_emp_length',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length',
    'loan_status'
]

correlation_matrix = train_dataset[numerical_features].corr()

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation matrix of numerical features", fontsize=14)
plt.show()


# Plot the relationship between categorical features and the target variable
fig, axes = plt.subplots(2, 2, figsize=(20, 15))
fig.suptitle("Relentionship between categorical features and loan status")

sns.countplot(x='person_home_ownership', hue='loan_status', data=train_dataset, ax=axes[0, 0])
sns.countplot(x='loan_intent', hue='loan_status', data=train_dataset, ax=axes[0, 1])
sns.countplot(x='loan_grade', hue='loan_status', data=train_dataset, ax=axes[1, 0])
sns.countplot(x='cb_person_default_on_file', hue='loan_status', data=train_dataset, ax=axes[1, 1])

plt.show()


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1)*(r - 1)) / (n - 1))
    rcorr = r - ((r - 1)**2) / (n - 1)
    kcorr = k - ((k - 1)**2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

# Define the categorical features
categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

cramers_v_matrix = pd.DataFrame(index=categorical_features, columns=categorical_features)

for col1 in categorical_features:
    for col2 in categorical_features:
        cramers_v_matrix.loc[col1, col2] = cramers_v(train_dataset[col1], train_dataset[col2])

plt.figure(figsize=(8, 6))
sns.heatmap(cramers_v_matrix.astype(float), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("CramÃ©r's V - Correlation between categorical features")
plt.show()



# Count the number of before removing outliers
before_count = train_dataset.shape[0]

# Remove outliers
train_dataset = train_dataset[
    (train_dataset['person_age'] <= 100) &
    (train_dataset['person_emp_length'] <= 60) &
    (train_dataset['person_age'] > train_dataset['person_emp_length'])
]
after_count = train_dataset.shape[0]
removed_outliers = before_count - after_count
print(f"Number of removed outliers: {removed_outliers}")


features = train_dataset.drop(columns=['loan_status'])
target = train_dataset['loan_status']

X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

# Divide the features
numerical_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
           'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',]
categorical_features = ['person_home_ownership', 'loan_intent']
ordinal_features = ['loan_grade', 'cb_person_default_on_file']
ordinal_mappings = [['A', 'B', 'C', 'D', 'E', 'F', 'G'], ['N', 'Y']]


preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
    ]), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) ,
    ('ordinal', OrdinalEncoder(categories=ordinal_mappings), ordinal_features),
])


def build_base_model(meta):
    input_dim = meta["X_shape_"][1] 
    base_model = Sequential([
        Input(shape=(input_dim,)),
        Dense(1, activation='sigmoid')
    ])
    base_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return base_model

model = KerasClassifier(
    model=build_base_model,
    epochs=20,
    batch_size=32,
    verbose=0
)

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', model) 
])

pipeline.fit(X_train, y_train)


accuracy = pipeline.score(X_val, y_val)
print(f": {accuracy:.4f}")

# Predict the target variable

y_train_pred = pipeline.predict(X_train)
y_val_pred = pipeline.predict(X_val)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Train Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")

# Classification report
print("Classification Report:")
print(classification_report(y_val, y_val_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

# ROC
visualizer = ROCAUC(pipeline, classes=[0, 1])
visualizer.fit(X_train, y_train)
visualizer.score(X_val, y_val)
visualizer.show()



model = KerasClassifier(
    model=build_base_model,
    epochs=30,
    batch_size=32,
    verbose=0,
)

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('model', model)
])

pipeline.fit(X_train, y_train)
# Predict the target variable
y_train_pred = pipeline.predict(X_train)
y_val_pred = pipeline.predict(X_val)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Train Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")

# Classification report
print("Classification Report:")
print(classification_report(y_val, y_val_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

# ROC
visualizer = ROCAUC(pipeline, classes=[0, 1])
visualizer.fit(X_train, y_train)
visualizer.score(X_val, y_val)
visualizer.show()


def build_optimized_model(meta):
    input_dim = meta["X_shape_"][1]
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.3), 
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

optimized_model = KerasClassifier(
    model=build_optimized_model,
    epochs=50,
    batch_size=64,
    verbose=0
)

# Callbacks for regularization
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1)

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('smote', SMOTE(sampling_strategy=0.6, random_state=42)),
    ('model', optimized_model),
])

pipeline.fit(X_train, y_train, model__callbacks=[early_stop, reduce_lr])
# Predict the target variable
y_train_pred = pipeline.predict(X_train)
y_val_pred = pipeline.predict(X_val)
y_val_pred_proba = pipeline.predict_proba(X_val)[:, 1]

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
roc_auc = roc_auc_score(y_val, y_val_pred_proba)

print(f"Train Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Classification report
print("Classification Report:")
print(classification_report(y_val, y_val_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

# ROC
visualizer = ROCAUC(pipeline, classes=[0, 1])
visualizer.fit(X_train, y_train)
visualizer.score(X_val, y_val)
visualizer.show()


def build_optimized_model_with_params(meta, dropout_rate=0.3, learning_rate=0.001, neurons=128):
    input_dim = meta["X_shape_"][1]
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(neurons, activation='relu', kernel_regularizer=regularizers.l2(learning_rate)),
        BatchNormalization(),
        Dropout(dropout_rate), 
        Dense(neurons // 2, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate / 2),
        Dense(neurons // 4, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

#optimized_model_with_params = KerasClassifier(
#    model=build_optimized_model_with_params,
#    dropout_rate=0.3,
#    learning_rate=0.001,
#    neurons=128,
#    epochs=50,
#    batch_size=64,
#    verbose=0
#)

#pipeline = Pipeline([
#    ('preprocessing', preprocessor),
#    ('smote', SMOTE(sampling_strategy=0.6, random_state=42)),
#    ('model', optimized_model_with_params),
#])


#param_dist = {
#    'model__dropout_rate': [0.2, 0.3, 0.4],
#    'model__learning_rate': [0.001, 0.0005, 0.0001],
#    'model__neurons': [64, 128, 256],
#    'model__batch_size': [32, 64, 128],
#    'model__epochs': [50, 100]
#}
#grid_search = GridSearchCV(
#    pipeline,
#    param_grid=param_dist,
#    scoring='f1',
#    cv=3,
#    verbose=2,
#    n_jobs=-1,
#)

#grid_search.fit(X_train, y_train, model__callbacks=[early_stop, reduce_lr])
#print(f"Best hyperparameter: {grid_search.best_params_}")
#print(f"Best F1-score: {grid_search.best_score_:.4f}")

# Evaluation
#best_model = grid_search.best_estimator_

#y_train_pred = best_model.predict(X_train)
#y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]
#y_val_pred = (y_val_pred_proba >= 0.5).astype(int)

#train_accuracy = accuracy_score(y_train, y_train_pred)
#val_accuracy = accuracy_score(y_val, y_val_pred)
#roc_auc = roc_auc_score(y_val, y_val_pred_proba)


#print(f"Train Accuracy: {train_accuracy}")
#print(f"Validation Accuracy: {val_accuracy}")
#print(f"ROC-AUC Score: {roc_auc:.4f}")
#print("Classification Report:")
#print(classification_report(y_val, y_val_pred))
#print("Confusion Matrix:")
#print(confusion_matrix(y_val, y_val_pred))


optimized_model_with_params = KerasClassifier(
    model=build_optimized_model_with_params,
    dropout_rate=0.4,
    learning_rate=0.0001,
    neurons=64,
    epochs=100,
    batch_size=128,
    verbose=0
)

class PolynomialFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        return X_poly
    
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('poly_feature_engineer', PolynomialFeatureEngineer()),
    ('model', optimized_model_with_params) 
])

pipeline.fit(X_train, y_train, model__callbacks=[early_stop, reduce_lr])

# Predict the target variable
y_train_pred = pipeline.predict(X_train)
y_val_pred = pipeline.predict(X_val)
y_val_pred_proba = pipeline.predict_proba(X_val)[:, 1]

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
roc_auc = roc_auc_score(y_val, y_val_pred_proba)

print(f"Train Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Classification report
print("Classification Report:")
print(classification_report(y_val, y_val_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

# ROC
visualizer = ROCAUC(pipeline, classes=[0, 1])
visualizer.fit(X_train, y_train)
visualizer.score(X_val, y_val)
visualizer.show()


class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['loan_burden_score'] = X['loan_amnt'] * (X['person_income'] + 1)
        X['risk_amplifier'] = X['loan_amnt'] * X['loan_int_rate']
        X['income_stability'] = X['person_income'] / (X['person_emp_length'] + 1)
        X['historical_trust'] = X['cb_person_cred_hist_length'] * X['loan_amnt']
        return X
    
custom_numerical_features = list(numerical_features) + ['loan_burden_score', 'risk_amplifier', 'income_stability', 'historical_trust']

custom_preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
    ]), custom_numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('ordinal', OrdinalEncoder(categories=ordinal_mappings), ordinal_features),
])

pipeline = Pipeline([
    ('custom_feature_engineer', CustomFeatureEngineer()),
    ('preprocessing', custom_preprocessor),
    ('model', optimized_model_with_params) 
])

pipeline.fit(X_train, y_train, model__callbacks=[early_stop, reduce_lr])

# Predict the target variable
y_train_pred = pipeline.predict(X_train)
y_val_pred = pipeline.predict(X_val)
y_val_pred_proba = pipeline.predict_proba(X_val)[:, 1]

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
roc_auc = roc_auc_score(y_val, y_val_pred_proba)

print(f"Train Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Classification report
print("Classification Report:")
print(classification_report(y_val, y_val_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

# ROC
visualizer = ROCAUC(pipeline, classes=[0, 1])
visualizer.fit(X_train, y_train)
visualizer.score(X_val, y_val)
visualizer.show()


def optimize_threshold(y_true, y_pred_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    valid_indices = np.where((thresholds >= 0.4) & (thresholds <= 0.8))
    precision, recall, thresholds = precision[valid_indices], recall[valid_indices], thresholds[valid_indices]
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]
    return best_threshold


# Predict the target variable
y_train_pred_proba = pipeline.predict_proba(X_train)[:, 1]
y_val_pred_proba = pipeline.predict_proba(X_val)[:, 1]

# Optimize the threshold
optimal_threshold = optimize_threshold(y_val, y_val_pred_proba)
print(f"Best threshold: {optimal_threshold:.4f}")

# Apply the threshold
y_train_pred = (y_train_pred_proba >= optimal_threshold).astype(int)
y_val_pred = (y_val_pred_proba >= optimal_threshold).astype(int)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
roc_auc = roc_auc_score(y_val, y_val_pred_proba)


print(f"Train Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Classification report
print("Classification Report:")
print(classification_report(y_val, y_val_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

# ROC
visualizer = ROCAUC(pipeline, classes=[0, 1])
visualizer.fit(X_train, y_train)
visualizer.score(X_val, y_val)
visualizer.show()


# NN (MLP) â€“ Keras Pipeline
nn_pipeline = Pipeline([
    ('custom_feature_engineer', CustomFeatureEngineer()),
    ('custom_preprocessing', custom_preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', KerasClassifier(
        model=build_optimized_model_with_params,
        dropout_rate=0.4,
        learning_rate=0.0001,
        neurons=64,
        epochs=100,
        batch_size=128,
        verbose=0,
        callbacks=[early_stop, reduce_lr]
    ))
])

# Random Forest Pipeline
rf_pipeline = Pipeline([
    ('preprocessing', ColumnTransformer(
        transformers=[
            ('numerical', StandardScaler(), X_train.select_dtypes(include=['number']).columns),
            ('categorical', OneHotEncoder(), X_train.select_dtypes(include=['object', 'category']).columns)
        ])),
    ('feature_generation', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('classifier', RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=1,
        bootstrap=True,
        random_state=42,
        class_weight='balanced'
    ))
])

# XGBoost Pipeline
xgb_pipeline = Pipeline([
    ('preprocessing', ColumnTransformer(
        transformers=[
            ('numerical', StandardScaler(), X_train.select_dtypes(include=['number']).columns),
            ('categorical', OneHotEncoder(), X_train.select_dtypes(include=['object', 'category']).columns)
        ])),
    ('feature_generation', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('classifier', XGBClassifier(n_estimators=200, learning_rate=0.01, max_depth=7, random_state=42, scale_pos_weight=3))
])

voting_clf = EnsembleVoteClassifier(
    clfs=[rf_pipeline, xgb_pipeline, nn_pipeline],
    voting='soft'
)

voting_clf.fit(X_train, y_train)

# Predict the target variable
y_train_pred = voting_clf.predict(X_train)
y_val_pred = voting_clf.predict(X_val)
y_val_pred_proba = voting_clf.predict_proba(X_val)[:, 1]

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
roc_auc = roc_auc_score(y_val, y_val_pred_proba)


print(f"Train Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Classification report
print("Classification Report:")
print(classification_report(y_val, y_val_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

# ROC
visualizer = ROCAUC(voting_clf, classes=[0, 1])
visualizer.fit(X_train, y_train)
visualizer.score(X_val, y_val)
visualizer.show()


cv_scores = cross_val_score(voting_clf, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")



test_features = test_dataset.drop(columns=['id'])
test_pred = voting_clf.predict_proba(test_features)[:, 1]

test_dataset['loan_status'] = test_pred
test_dataset[['id', 'loan_status']].to_csv('submission.csv', index=False)

# Display the submission file
submission_file = pd.read_csv('submission.csv')
submission_file




