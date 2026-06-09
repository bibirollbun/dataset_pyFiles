import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
# from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV


df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
display(df.head())


print("Shape of the DataFrame:")
print(df.shape)

print("\nData types and non-null values:")
df.info()


print("\nDescriptive statistics for numerical columns:")
display(df.describe())


print("\nMissing values per column:")
print(df.isnull().sum())


print("Distribution of the target variable 'y':")
print(df['y'].value_counts())
print("\nProportion of the target variable 'y':")
print(df['y'].value_counts(normalize=True))


print("\nDistribution of categorical variables:")
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    print(f"\nUnique values and counts for '{col}':")
    print(df[col].value_counts())


print("Distribution of numerical variables:")
numerical_cols = df.select_dtypes(include=np.number).columns
print(df[numerical_cols].agg(['mean', 'median', 'std']))


print("\nRelationship between 'y' and categorical variables (proportion of 'y'=1):")
for col in categorical_cols:
    if col != 'y':
        print(f"\nProportion of 'y'=1 for different categories of '{col}':")
        print(df.groupby(col)['y'].mean().sort_values(ascending=False))


print("\nRelationship between 'y' and numerical variables (mean of 'y' for different ranges):")
numerical_cols_for_analysis = numerical_cols.drop(['id', 'y'])
for col in numerical_cols_for_analysis:
    print(f"\nMean of 'y' for different ranges of '{col}':")
    # Create bins for numerical variables to analyze relationship with 'y'
    if df[col].nunique() > 20: # Only bin if there are many unique values
        df[f'{col}_bin'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')
        print(df.groupby(f'{col}_bin')['y'].mean())
    else:
        print(df.groupby(col)['y'].mean())


# 1. Countplot of the target variable 'y'
plt.figure(figsize=(6, 4))
sns.countplot(x='y', data=df, palette='viridis')
plt.title('Distribution of Target Variable (y)')
plt.xlabel('Target Variable')
plt.ylabel('Count')
plt.show()


# 2. Countplots for key categorical variables with 'y' as hue
categorical_vars_for_plot = ['job', 'marital', 'education', 'contact', 'poutcome']
for col in categorical_vars_for_plot:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col, hue='y', data=df, palette='viridis')
    plt.title(f'Distribution of {col} with Target Variable (y)')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# 3. Histograms for key numerical variables
numerical_vars_for_hist = ['age', 'balance', 'duration', 'campaign']
for col in numerical_vars_for_hist:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=col, kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


# 4. Box plots for key numerical variables against the target variable 'y'
numerical_vars_for_box = ['age', 'balance', 'duration', 'campaign']
for col in numerical_vars_for_box:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='y', y=col, data=df, palette='viridis')
    plt.title(f'{col} vs. Target Variable (y)')
    plt.xlabel('Target Variable')
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()


# 5. Bar plots showing the proportion of 'y=1' for different categories of categorical variables
categorical_vars_for_proportion_plot = ['job', 'marital', 'education', 'contact', 'poutcome']
for col in categorical_vars_for_proportion_plot:
    plt.figure(figsize=(10, 6))
    proportion_df = df.groupby(col)['y'].value_counts(normalize=True).unstack().fillna(0)
    # The column names are integers 0 and 1, not strings '0' and '1'
    proportion_df[1].sort_values(ascending=False).plot(kind='bar', color='mediumseagreen')
    plt.title(f'Proportion of y=1 for different categories of {col}')
    plt.xlabel(col)
    plt.ylabel('Proportion of y=1')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


sns.heatmap(df[numerical_cols_for_analysis].corr(method="pearson"), annot=True)
plt.show()


# 1. Identify categorical columns
categorical_cols = df.select_dtypes(include='object').columns.tolist()
# Exclude 'y' from the categorical columns if it's included
if 'y' in categorical_cols:
    categorical_cols.remove('y')

# 2. Apply one-hot encoding to the identified categorical columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Drop original categorical columns (already handled by get_dummies with drop_first=True)
# and the original 'y' column before separating
# df_processed = df_encoded.drop(columns=categorical_cols) # No need to drop again

# 3. Separate the target variable 'y' from the features
y = df_encoded['y']
X = df_encoded.drop(columns=['y', 'id']) # Drop 'id' as it's not a feature

# 4. Identify numerical columns in the DataFrame that are not the target variable or the 'id' column
# These are the columns remaining in X that are not the newly created dummy variables
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

# 5. Apply StandardScaler to the identified numerical columns
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# X now contains both scaled numerical features and one-hot encoded categorical features
# No need to concatenate as get_dummies and scaling were applied to parts of the same DataFrame

# 6. X already contains the concatenated features

# 7. Display the first few rows and the shape of the resulting preprocessed DataFrame
print("Shape of the preprocessed DataFrame (X):")
print(X.shape)
print("\nFirst few rows of the preprocessed DataFrame (X):")
display(X.head())


# 1. Split the preprocessed data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Shape of training data (X_train, y_train):")
print(X_train.shape, y_train.shape)
print("\nShape of testing data (X_test, y_test):")
print(X_test.shape, y_test.shape)


# 2. Instantiate a Logistic Regression model
model = LogisticRegression(random_state=42, solver='liblinear') # Using liblinear solver for potentially large dataset

# 3. Train the chosen model on the training data
print("\nTraining the Logistic Regression model...")
model.fit(X_train, y_train)
print("Model training complete.")

# 4. Make predictions on the test data
y_pred = model.predict(X_test)

# 5. Evaluate the model's performance using appropriate metrics
print("\nEvaluating the model performance:")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 1. Define a parameter distribution for the Decision Tree model
param_dist = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'criterion': ['gini', 'entropy']
}

# 2. Instantiate a RandomizedSearchCV object
dt_classifier = DecisionTreeClassifier(random_state=42)
random_search = RandomizedSearchCV(estimator=dt_classifier, param_distributions=param_dist,
                                   n_iter=10, cv=3, random_state=42, n_jobs=-1, scoring='recall') # Use recall for imbalanced dataset

# 3. Fit the RandomizedSearchCV object to the training data
print("Performing Randomized Search for Decision Tree...")
random_search.fit(X_train, y_train)
print("Randomized Search complete.")


# 4. Get the best Decision Tree model from the RandomizedSearchCV object
best_dt_model = random_search.best_estimator_

# 5. Print the best parameters found by Randomized Search
print("\nBest parameters found by Randomized Search:")
print(random_search.best_params_)

# 6. Make predictions on the test data using the best model
y_pred_dt = best_dt_model.predict(X_test)

# 7. Evaluate the best Decision Tree model's performance using classification_report
print("\nClassification Report for Best Decision Tree Model:")
print(classification_report(y_test, y_pred_dt))


param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [5, 10, 15, 20],
    'min_samples_leaf': [1, 3, 5, 7],
    'max_features': ['sqrt', 'log2']
}


# 1. Import the RandomForestClassifier class from sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


# 2. Using RandomForestClassifier as the base estimator
bagging_classifier = RandomForestClassifier(class_weight="balanced", random_state=42, oob_score=True)

# 3. Instantiate a RandomizedSearch object with a base estimator
random_search_bagging = RandomizedSearchCV(estimator=bagging_classifier, param_distributions=param_dist,
                                       n_iter=10, cv=3,
                                           random_state=42, n_jobs=-1, scoring='recall') # Use recall for imbalanced dataset

# 4. Fit the RandomizedSearchCV object to the training data
print("Performing Randomized Search for Bagging Classifier...")
random_search_bagging.fit(X_train, y_train)
print("Randomized Search complete.")


# 5. Get the best RandomForestClassifier model from the RandomizedSearchCV object
best_bagging_model = random_search_bagging.best_estimator_

# 6. Print the best parameters found by Randomized Search
print("\nBest parameters found by Randomized Search:")
print(random_search_bagging.best_params_)

# 7. Make predictions on the test data using the best model
y_pred_bagging = best_bagging_model.predict(X_test)

# 8. Evaluate the best RandomForestClassifier model's performance using classification_report
print("\nClassification Report for Best Bagging Model:")
print(classification_report(y_test, y_pred_bagging))


# 1. Import the GradientBoostingClassifier class from sklearn.ensemble
from sklearn.ensemble import GradientBoostingClassifier

# 2. Instantiate a GradientBoostingClassifier object
gbm_classifier = GradientBoostingClassifier(random_state=42)

# 3. Instantiate a RandomizedSearchCV object
random_search_gbm = RandomizedSearchCV(estimator=gbm_classifier, param_distributions=param_dist,
                                       n_iter=10, cv=3,
                                       random_state=42, n_jobs=-1, scoring='recall') # Use recall for imbalanced dataset

# 4. Fit the RandomizedSearchCV object to the training data
print("Performing Randomized Search for Gradient Boosting Classifier...")
random_search_gbm.fit(X_train, y_train)
print("Randomized Search complete.")


# 5. Get the best GradientBoostingClassifier model from the RandomizedSearchCV object
best_gb_model = random_search_gbm.best_estimator_

# 6. Print the best parameters found by Randomized Search
print("\nBest parameters found by Randomized Search:")
print(random_search_gbm.best_params_)

# 7. Make predictions on the test data using the best model
y_pred_gbdt = best_gb_model.predict(X_test)

# 8. Evaluate the best GradientBoostingClassifier model's performance using classification_report
print("\nClassification Report for Best Gradient Boosting Model:")
print(classification_report(y_test, y_pred_gbdt))


# 1. Import the XGBClassifier class from xgboost
from xgboost import XGBClassifier

# 2. Define a parameter distribution for the XGBClassifier
param_dist_xgb = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'min_split_loss': [5, 10, 15, 20],
    'min_child_weight': [1, 3, 5, 7]
}

# 3. Instantiate an XGBClassifier object
xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# 4. Instantiate a RandomizedSearchCV object
random_search_xgb = RandomizedSearchCV(estimator=xgb_classifier, param_distributions=param_dist_xgb,
                                       n_iter=10, cv=3, random_state=42, n_jobs=-1, scoring='recall') # Use recall for imbalanced dataset

# 5. Fit the RandomizedSearchCV object to the training data
print("Performing Randomized Search for XGBoost Classifier...")
random_search_xgb.fit(X_train, y_train)
print("Randomized Search complete.")


# 6. Get the best XGBoost model from the RandomizedSearchCV object
best_xgb_model = random_search_xgb.best_estimator_

# 7. Print the best parameters found by Randomized Search
print("\nBest parameters found by Randomized Search:")
print(random_search_xgb.best_params_)

# 8. Make predictions on the test data using the best model
y_pred_xgb = best_xgb_model.predict(X_test)

# 9. Evaluate the best XGBoost model's performance using classification_report
print("\nClassification Report for Best XGBoost Model:")
print(classification_report(y_test, y_pred_xgb))


# 1. Import the StackingClassifier class from sklearn.ensemble
from sklearn.ensemble import StackingClassifier

# 2. Define a list of estimators
estimators = [
    ('lr', model), # Logistic Regression model from previous steps
    ('dt', best_dt_model), # Best Decision Tree model from previous steps
    ('bag', best_bagging_model), # Best Bagging Classifier model from previous steps
    ('gbm', best_gb_model), # Best Gradient Boosting model from previous steps
    ('xgb', best_xgb_model) # Best XGBoost model from previous steps
]

# 3. Instantiate a StackingClassifier object
stacking_classifier = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(random_state=42), cv=3, n_jobs=-1)

# 4. Train the StackingClassifier model on the training data
print("Training the Stacking Classifier model...")
stacking_classifier.fit(X_train, y_train)
print("Stacking Classifier training complete.")


# Make predictions on the test data using the Stacking Classifier
y_pred_stacking = stacking_classifier.predict(X_test)

# Evaluate the Stacking Classifier model's performance using classification_report
print("\nClassification Report for Stacking Classifier Model:")
print(classification_report(y_test, y_pred_stacking))


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Confusion Matrix for Stacking Classifier
cm_stacking = confusion_matrix(y_test, y_pred_stacking)
print("\nConfusion Matrix for Stacking Classifier:")
print(cm_stacking)

# Display Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm_stacking, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Stacking Classifier')
plt.show()


# 1. Import necessary libraries
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 2. Get the predicted probabilities for the positive class (class 1)
y_pred_proba_stacking = stacking_classifier.predict_proba(X_test)[:, 1]

# 3. Calculate the False Positive Rate (FPR) and True Positive Rate (TPR)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_stacking)

# 4. Calculate the Area Under the ROC Curve (AUC)
roc_auc = auc(fpr, tpr)

# 5. Print the calculated AUC value
print(f"\nAUC for Stacking Classifier: {roc_auc:.4f}")

# 6. Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Stacking Classifier')
plt.legend(loc="lower right")
plt.show()


df_test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
display(df_test.head())


# 1. Identify categorical columns
categorical_cols = df_test.select_dtypes(include='object').columns.tolist()
# Exclude 'y' from the categorical columns if it's included
if 'y' in categorical_cols:
    categorical_cols.remove('y')
    
# 2. Apply one-hot encoding to the identified categorical columns
df_encoded_test = pd.get_dummies(df_test, columns=categorical_cols, drop_first=True)

# 3. Separate the target variable 'y' from the features
# ydf_test = df_encoded_test['y']
Xdf_test = df_encoded_test.drop(columns=['id']) # Drop 'id' as it's not a feature

# 4. Identify numerical columns in the DataFrame that are not the target variable or the 'id' column
# These are the columns remaining in X that are not the newly created dummy variables
test_numerical_cols = Xdf_test.select_dtypes(include=np.number).columns.tolist()

missing_cols = set(X.columns) - set(Xdf_test.columns)
for c in missing_cols:
    Xdf_test[c] = 0
# Ensure the order of columns is the same
Xdf_test = Xdf_test[X.columns]

# 5. Apply StandardScaler to the identified numerical columns
scaler = StandardScaler()
Xdf_test[test_numerical_cols] = scaler.fit_transform(Xdf_test[test_numerical_cols])

# 6. Display the first few rows and the shape of the resulting preprocessed DataFrame
print("Shape of the preprocessed DataFrame (X):")
print(Xdf_test.shape)
print("\nFirst few rows of the preprocessed DataFrame (X):")
display(Xdf_test.head())


ydf_test = stacking_classifier.predict_proba(Xdf_test)[:, 1]


pd.concat([df_test['id'], pd.Series(ydf_test.round(5), name='y')], axis=1).to_csv('submission.csv', index=False)

