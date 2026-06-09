import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


pip install imblearn


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, auc
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE




print(f"scikit-learn version: {sklearn.__version__}")


df = pd.read_csv(r'/kaggle/input/customer-churn-prediction-2020/train.csv')


df.head()


df.shape


df.info()


df.describe()


df.isnull().sum()


df.duplicated().sum()


ax = sns.countplot(x='churn', hue='churn', data=df)
for container in ax.containers:
    ax.bar_label(container)
plt.title('Distribution of Churn')
plt.xlabel('Churn (0: No, 1: Yes)')
plt.ylabel('Count')
plt.show()


state = df['state'].value_counts()
state


# Numerical Distribution of First 10 States
ax = state[:10].plot(kind='bar', color='violet')
ax.bar_label(ax.containers[0], fontsize=10);


area_code = df['area_code'].value_counts()
area_code


ax = area_code.plot(kind='bar', color='indigo')
ax.bar_label(ax.containers[0], fontsize=10);


international_plan = df['international_plan'].value_counts()
international_plan


ax = international_plan.plot(kind='bar', color='skyblue')
ax.bar_label(ax.containers[0], fontsize=10);


list(set(df.dtypes.tolist()))


train_num = df.select_dtypes(include = ['float64', 'int64'])
train_num.head()


train_num.hist(figsize=(16, 20), bins=10, xlabelsize=8, ylabelsize=8)


n = 0
for col in train_num:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    low_limit = Q1 - (IQR * 1.5)
    high_limit = Q3 + (IQR * 1.5)
    total = len(df[col][(((df[col] < low_limit)) | (df[col] > high_limit))])
    n = n + total
    print('There are {} row data outlier in {}'.format(total, col))



colour_list = ['orange','goldenrod', 'lightseagreen', 'rebeccapurple', 'crimson', 'violet', 'brown', 'peru', 'burlywood', 'indigo',
           'firebrick', 'gray', 'forestgreen', 'cyan', 'forestgreen']
fig = plt.figure(figsize=[20,50])
for i, col in enumerate(train_num):
    ax = fig.add_subplot(8,4, i+1)
    ax = sns.boxplot(data = df, x=col, color=colour_list[i])
    title = re.sub('_', ' ', col.title())
    ax.set_title(title)
    ax.set_xlabel(col)


n = 0
for col in train_num:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    low_limit = Q1 - (IQR * 1.5)
    high_limit = Q3 + (IQR * 1.5)
    df[col] = df[col].apply(lambda x: high_limit if x > high_limit else (low_limit if x < low_limit else x))
    total = len(df[col][(((df[col] < low_limit)) | (df[col] > high_limit))])
    n = n + total
    print('There are {} row data outlier in columns (variables) {}'.format(total, col))


fig = plt.figure(figsize=[20,50])
for i ,col in enumerate(train_num):
    ax = fig.add_subplot(7,4, i+1)
    ax = sns.boxplot(data = df, x=col, color=colour_list[i])
    title = re.sub('_', ' ', col.title())
    ax.set_title(title)
    ax.set_xlabel(col)


df_cat = df[['international_plan', 'voice_mail_plan', 'churn']].replace({'yes': 1, 'no': 0})


total_minutes = train_num['total_day_minutes'] + train_num['total_eve_minutes'] + train_num['total_night_minutes']
total_charge = train_num['total_day_charge'] + train_num['total_eve_charge'] + train_num['total_night_charge']
total_calls = train_num['total_day_calls'] + train_num['total_eve_calls'] + train_num['total_night_calls']
train_num['total_minutes'] = total_minutes
train_num['total_charge'] = total_charge
train_num['total_calls'] = total_calls


train_num.drop(['total_day_minutes', 'total_eve_minutes', 'total_night_minutes',
                'total_day_charge', 'total_eve_charge', 'total_night_charge',
                'total_day_calls', 'total_eve_calls', 'total_night_calls'], axis=1, inplace=True)


train_num.head()


# One-hot encode categorical columns
categorical_cols = ['state', 'area_code']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_train = encoder.fit_transform(df[categorical_cols])
encoded_df_train = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)


df = pd.concat([train_num, encoded_df_train, df_cat], axis=1)


X = df.drop(['churn'], axis=1)
y = df['churn']


# Train/Validation/Test Split - Split X and y directly
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, shuffle=True)


# Scale the data
scaler = StandardScaler()
# Fit the scaler only on the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the validation and test data using the scaler fitted on the training data
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE only to the training data
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train_scaled, y_train)

X_train = X_train_resampled
y_train = y_train_resampled
X_valid = X_valid_scaled
X_test = X_test_scaled


# --- k nearest neighbor ---
knn_grid = {
    'n_neighbors': list(range(1, 30, 2)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
grid_knn = GridSearchCV(KNeighborsClassifier(), knn_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_knn.fit(X_train, y_train)

# --- Logistic regression ---
logreg_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.001, 0.01, 0.1, 1, 10],
    'solver': ['saga'],
    'max_iter': [1000]
}
grid_logreg = GridSearchCV(LogisticRegression(), logreg_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_logreg.fit(X_train, y_train)

# --- Naive bayes ---
nb_grid = {
    'var_smoothing': np.logspace(-9, -2, 8)
}
grid_nb = GridSearchCV(GaussianNB(), nb_grid, cv=5, scoring='f1', n_jobs=-1)
grid_nb.fit(X_train, y_train)

# --- Decision Tree ---
tree_grid = {
    'max_depth': [3, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_tree.fit(X_train, y_train)

# --- Random Forest ---
rf_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
    'criterion': ['gini', 'entropy']
}
grid_rf = GridSearchCV(RandomForestClassifier(), rf_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train, y_train)

# --- XGBoost ---
# xgb_grid = {
#     'n_estimators': [50, 100],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 6, 10]
# }
# grid_xgb = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgb_grid, cv=5, scoring='accuracy', n_jobs=-1)
# grid_xgb.fit(X_train, y_train)



results = []

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = None

    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f"{name} - ROC Curve")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.grid(True)
        plt.show()

    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC AUC': roc_auc_score(y_test, y_proba) if y_proba is not None else None
    })


evaluate_model("KNN (Tuned)", grid_knn.best_estimator_, X_test, y_test)
evaluate_model("Logistic Regression (Tuned)", grid_logreg.best_estimator_, X_test, y_test)
evaluate_model("Naive Bayes (Tuned)", grid_nb.best_estimator_, X_test, y_test)
evaluate_model("Decision Tree (Tuned)", grid_tree.best_estimator_, X_test, y_test)
evaluate_model("Random Forest (Tuned)", grid_rf.best_estimator_, X_test, y_test)
# evaluate_model("XGBoost (Tuned)", grid_xgb.best_estimator_, X_test, y_test)


results_df = pd.DataFrame(results)
results_df.sort_values(by='Accuracy', ascending=False)


plt.figure(figsize=(8, 6))
for name, model in [
    ("KNN (Tuned)", grid_knn.best_estimator_),
    ("Logistic Regression (Tuned)", grid_logreg.best_estimator_),
    ("Naive Bayes (Tuned)", grid_nb.best_estimator_),
    ("Decision Tree (Tuned)", grid_tree.best_estimator_),
    ("Random Forest (Tuned)", grid_rf.best_estimator_),
    # ("XGBoost (Tuned)", grid_xgb.best_estimator_)
]:
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        continue

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()


df_test = pd.read_csv(r'/kaggle/input/customer-churn-prediction-2020/test.csv')


df_test.head()


df_test[['international_plan', 'voice_mail_plan']] = df_test[['international_plan', 'voice_mail_plan']].replace({'yes': 1, 'no': 0})


test_total_minutes = df_test['total_day_minutes'] + df_test['total_eve_minutes'] + df_test['total_night_minutes']
test_total_charge = df_test['total_day_charge'] + df_test['total_eve_charge'] + df_test['total_night_charge']
test_total_calls = df_test['total_day_calls'] + df_test['total_eve_calls'] + df_test['total_night_calls']
df_test['total_minutes'] = test_total_minutes
df_test['total_charge'] = test_total_charge
df_test['total_calls'] = test_total_calls


df_test.drop(['id', 'total_day_minutes', 'total_eve_minutes', 'total_night_minutes',
                'total_day_charge', 'total_eve_charge', 'total_night_charge',
                'total_day_calls', 'total_eve_calls', 'total_night_calls'], axis=1, inplace=True)


df_test.head()


df_test.shape


# One-hot encode categorical columns using the encoder fitted on the training data
categorical_cols = ['state', 'area_code'] # Use the same columns
test_encoded = encoder.transform(df_test[categorical_cols]) # Use the fitted 'encoder'
test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df_test.index)



test_encoded_df.shape


# Drop the original categorical columns from df_test
df_test_processed = df_test.drop(columns=categorical_cols)


df_test_processed = pd.concat([df_test_processed, test_encoded_df], axis=1)


df_test_processed.shape


feature_cols_train = X.columns # X is the dataframe before scaling/smote


# Reindex the test data to match the training data columns and fill missing columns with 0
df_test_processed = df_test_processed.reindex(columns=feature_cols_train, fill_value=0)


# Scale the processed test data using the scaler fitted on the training data
X_submission = scaler.transform(df_test_processed)


df_test_raw = pd.read_csv(r'/kaggle/input/customer-churn-prediction-2020/sampleSubmission.csv')
ids = df_test_raw['id']


# Use the best performing model, which was grid_rf.best_estimator_
preds = grid_rf.best_estimator_.predict(X_submission)


output = pd.DataFrame({'Id': ids,'churn': preds.squeeze()})


output.churn.value_counts()


# Convert churn predictions back to 'yes'/'no' string labels as per sampleSubmission.csv
output['churn'] = output['churn'].replace({1: 'yes', 0: 'no'})


# Save the submission file
output.to_csv('/kaggle/working/submission.csv', index=False)


ax = sns.countplot(x='churn', data=output)
ax.bar_label(ax.containers[0], fontsize=10);


print("Submission file created successfully!")
print(output.head())

