pip install scikeras


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


# Create new features from 'day'
def create_season(day):
    if day <= 91:
        return 1  # Winter
    elif day <= 182:
        return 2  # Spring
    elif day <= 273:
        return 3  # Summer
    else:
        return 4  # Fall

df['season'] = df['day'].apply(create_season)
test['season'] = test['day'].apply(create_season)


df['humidity_cloud_interaction'] = df['humidity'] * df['cloud']
# df['cloud + humidity'] =  df['humidity'] + df['cloud']
df['cloud + humidity + sunshine'] = df['humidity'] + df['cloud'] + df['sunshine']
df['humidity * sunshine'] = df['humidity'] * df['sunshine']


test['humidity_cloud_interaction'] = test['humidity'] * test['cloud']
# test['cloud + humidity'] =  test['humidity'] + test['cloud']
test['cloud + humidity + sunshine'] = test['humidity'] + test['cloud'] + test['sunshine']
test['humidity * sunshine'] = test['humidity'] * test['sunshine']


f=['id']
df.drop(columns=f,inplace=True)
test.drop(columns=f,inplace=True)


# Set the figure size
plt.figure(figsize=(15, 10)) 
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()


# Separate features and target
X = df.drop('rainfall', axis=1)
y = df['rainfall']


from sklearn.feature_selection import mutual_info_classif
# Calculate MI scores
mi_scores = mutual_info_classif(X, y, random_state=42)

# Create a DataFrame to store feature names and their MI scores
mi_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})

# Sort the DataFrame by MI scores in descending order
mi_df = mi_df.sort_values(by='MI Score', ascending=False)

# Display the top features
print(mi_df)


from sklearn.impute import KNNImputer
# Handle missing values using KNN Imputer
imputer = KNNImputer(n_neighbors=5)
train = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
test = pd.DataFrame(imputer.transform(test), columns=test.columns)


# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X1, y1 = smote.fit_resample(X, y)


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X1, y1, test_size=0.2, random_state=42)


# Define the refined hyperparameter grid for ExtraTreesClassifier
param_dist = {
    'n_estimators': [768, 600, 2000],  # Number of trees in the forest
    'max_depth': [15, 20, 25, None],  # Maximum depth of the tree (None means no limit)
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],    # Minimum number of samples required at each leaf node
    'max_features': ['sqrt', 'log2', None],  # Number of features to consider for the best split
    'bootstrap': [True, False],       # Whether to bootstrap samples when building trees
}

# Initialize the ExtraTreesClassifier
model = ExtraTreesClassifier()

# RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=100,           # Number of parameter settings sampled
    cv=3,               # Number of cross-validation folds
    scoring='roc_auc',   # Use AUC-ROC as the scoring metric
    n_jobs=-1,           # Use all available cores
    verbose=2,
    random_state=42
)

# Fit the RandomizedSearchCV
random_search.fit(X_train, y_train)

# Best parameters and model
best_params = random_search.best_params_
print(f"\nBest Parameters: {best_params}")
best_model = random_search.best_estimator_


# Evaluate the model on the validation set
y_pred = best_model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
print(f"\nValidation AUC: {auc:.4f}")

# Cross-validation score
cv_scores = cross_val_score(best_model, X1, y1, cv=10, scoring='roc_auc')
print(f"\nCross-Validation AUC: {np.mean(cv_scores):.4f}")


# Generate predictions for the test set
test_preds = best_model.predict_proba(test)


# Prepare the submission file
submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
submission['rainfall'] = test_preds
submission.to_csv("submission.csv", index=False)
print("\nSubmission file saved as 'submission.csv'.")




