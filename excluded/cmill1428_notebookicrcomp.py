import pandas as pd
import os
import numpy as np
from sklearn.metrics import make_scorer



# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report, precision_recall_fscore_support
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.model_selection import train_test_split 
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier


PATH = "icr-identify-age-related-conditions"
greeks = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/greeks.csv")
sample_submission = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/sample_submission.csv")
test = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/test.csv")
train = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/train.csv")
train.set_index('Id', inplace=True)
display(greeks)
display(sample_submission)
display(test)
display(train)


import numpy as np
from sklearn.metrics import make_scorer

def balanced_log_loss(y_true, y_pred_proba):
    # Ensure predicted probabilities are clipped to avoid log(0)
    eps = 1e-15
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)

    # If it's 1D, convert to 2D (only happens in some edge cases)
    if y_pred_proba.ndim == 1:
        y_pred_proba = np.vstack([1 - y_pred_proba, y_pred_proba]).T

    if y_pred_proba.shape[1] != 2:
        raise ValueError("y_pred_proba should be a 2D array with 2 columns for binary classification.")

    # Separate predicted probabilities for each class
    p0 = y_pred_proba[:, 0]
    p1 = y_pred_proba[:, 1]

    # Identify samples belonging to each class
    idx_0 = (y_true == 0)
    idx_1 = (y_true == 1)

    # Compute balanced log loss
    loss_0 = -np.mean(np.log(p0[idx_0])) if np.any(idx_0) else 0
    loss_1 = -np.mean(np.log(p1[idx_1])) if np.any(idx_1) else 0

    return (loss_0 + loss_1) / 2

# Wrap it for use with GridSearchCV
balanced_log_loss_scorer = make_scorer(balanced_log_loss, greater_is_better=False, needs_proba=True)





def createSubmission(model, name):
    predictions = model.predict_proba(test)
    df = pd.DataFrame({"Id": test["Id"]})
    df['class_0'], df['class_1'] = predictions[:, 0], predictions[:, 1]
    #df = pd.DataFrame(predictions, columns=['class_0', 'class_1'])
    display(df)
    output_dir = "./kaggle/working/"
    if not os.path.exists(output_dir):
        output_dir = "./" #if kaggle/working does not exist, save to local directory.
        print(f"Kaggle working directory not found, saving to current directory: {output_dir}")

    df.to_csv(os.path.join(output_dir, name), index=False)


train.describe()


X = train.drop(['Class'], axis=1) # Drop id column

y = train['Class'] # Get the labels

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
display(X_train)
display(X_test)
display(y_train)
display(y_test)

ct = make_column_transformer(
    (
        make_pipeline(SimpleImputer(strategy='mean')),  # explicitly set strategy
        make_column_selector(dtype_include=['int64', 'float64']),
    ),
    (
        OneHotEncoder(handle_unknown="ignore"),
        ["EJ"],
    ),
)



from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)

y_pred = dummy.predict(X_test)


accuracy = accuracy_score(y_test, y_pred) # Get the accuracy score
print("Accuracy:", accuracy)




model_pipeline = Pipeline([
    ('preprocessor', ct),
    ('classifier', DecisionTreeClassifier())
])

# Train the model
model_pipeline.fit(X_train, y_train)




y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)


accuracy = accuracy_score(y_test, y_pred) # Get the accuracy score
print("Accuracy:", accuracy)
balanced_log = balanced_log_loss(y_test, y_pred_proba)
print("Balanced Log Loss:", balanced_log)



cm = confusion_matrix(y_test, y_pred) # Make confusion matrix

ConfusionMatrixDisplay(confusion_matrix=cm).plot()


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

display("Accuracy:", accuracy)
display("Precision:", precision)
display("Recall:", recall)


p,r,f,s = precision_recall_fscore_support(y_test, y_pred,labels=[0,1], zero_division=1)
print(p, r, f, s)


model_pipeline = Pipeline([
    ('preprocessor', ct),
    ('classifier', RandomForestClassifier())
])

# Train the model
model_pipeline.fit(X_train, y_train)



y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)



accuracy = accuracy_score(y_test, y_pred) # Get the accuracy score
print("Accuracy:", accuracy)
balanced_log = balanced_log_loss(y_test, y_pred_proba)
print("Balanced Log Loss:", balanced_log)



cm = confusion_matrix(y_test, y_pred) # Make confusion matrix

ConfusionMatrixDisplay(confusion_matrix=cm).plot()


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

display("Accuracy:", accuracy)
display("Precision:", precision)
display("Recall:", recall)


p,r,f,s = precision_recall_fscore_support(y_test, y_pred,labels=[0,1], zero_division=1)
print(p, r, f, s)


# Split the data into training and testing sets
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X, y, test_size=0.2)
display(X_train_lr)
display(X_test_lr)
display(y_train_lr)
display(y_test_lr)


# Train the Logistic Regression model
logistic_model_pipeline = Pipeline([
    ('preprocessor', ct),
    ('classifier', LogisticRegression())
])

logistic_model_pipeline.fit(X_train_lr, y_train_lr)


# Evaluate the model
y_pred_lr = logistic_model_pipeline.predict(X_test_lr)
accuracy_lr = accuracy_score(y_test_lr, y_pred_lr)
print("Accuracy: {:.2f}%".format(accuracy * 100))
y_pred_proba = logistic_model_pipeline.predict_proba(X_test_lr)
balanced_log = balanced_log_loss(y_test, y_pred_proba)
print("Balanced Log Loss:", balanced_log)




p,r,f,s = precision_recall_fscore_support(y_test_lr, y_pred_lr,labels=[0,1], zero_division=1)
print(p, r, f, s)
accuracy_lr = accuracy_score(y_test_lr, y_pred_lr)
precision_lr = precision_score(y_test_lr, y_pred_lr)
recall_lr = recall_score(y_test_lr, y_pred_lr)

display("Accuracy:", accuracy_lr)
display("Precision:", precision_lr)
display("Recall:", recall_lr)


cm_lr = confusion_matrix(y_test_lr, y_pred_lr) # Make confusion matrix

ConfusionMatrixDisplay(confusion_matrix=cm_lr).plot()


from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier

# Define the CatBoost model
catboost_model_pipeline = Pipeline([
    ('preprocessor', ct),  # Keep this if preprocessing is needed
    ('classifier', CatBoostClassifier(verbose=0, random_seed=42))  # Suppress output for clean logs
])

# Train the model
catboost_model_pipeline.fit(X_train, y_train)

# createSubmission(catboost_model_pipeline, "submission.csv")



from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

print(X_train[["BQ", "CB", "CC"]].dtypes)


pipe_Tree = Pipeline([
    ('transform', ct),
    ('model', DecisionTreeClassifier())
])

# Define the hyperparameter grid
param_grid = {
    'model__criterion': ['gini', 'entropy'],          # Model parameters prefixed with 'model'
    'model__max_depth': [5, 6, 7, 8, 9, 10, 11, 12, 13],           # Tree depth
    'model__min_samples_split': [2, 5, 10],           # Minimum samples to split an internal node
    'model__min_samples_leaf': [1, 2, 4, 6],          # Minimum samples at a leaf node
    'model__max_features': ['sqrt', 'log2', None]  # Features considered for a split
}

# Initialize the Decision Tree Classifier
model = make_pipeline(ct, DecisionTreeClassifier())

# Set up GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=pipe_Tree,
    param_grid=param_grid,
    scoring=balanced_log_loss_scorer, 
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # Use all processors
    verbose=1   # Display progress messages
)

# # Fit the GridSearchCV on the training data
# grid_search.fit(X_train, y_train)

# # Print the best parameters and the best score
# print(f"Best Parameters: {grid_search.best_params_}")
# print(f"Best Score: {grid_search.best_score_:.2f}")



from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Define pipeline with Random Forest
pipe_RF = Pipeline([
    ('transform', ct),
    ('model', RandomForestClassifier(random_state=42))
])


param_grid_random_forest = {
    'model__bootstrap': [True],
    'model__ccp_alpha': [0.0],  # Usually left at default unless pruning
    'model__class_weight': [None, 'balanced'],  # Try automatic balancing
    'model__criterion': ['gini', 'entropy'],  # Split quality metric
    'model__max_depth': [5, 10, 15, 20, None],  # Control overfitting
    'model__max_features': ['sqrt', 'log2', None],  # Features at each split
    'model__max_leaf_nodes': [None, 10, 20, 30],  # Limit tree complexity
    'model__max_samples': [None],  # Subsampling — keep off for now
    'model__min_impurity_decrease': [0.0],  # Leave default
    'model__min_samples_leaf': [1, 2, 4],  # Control leaf size
    'model__min_samples_split': [2, 5, 10],  # Min samples for split
    'model__min_weight_fraction_leaf': [0.0],  # Usually left at 0.0
    'model__monotonic_cst': [None],  # Only used for monotonic constraints
    'model__n_estimators': [100, 200, 300],  # Number of trees
    'model__n_jobs': [-1],  # Use all cores
    'model__oob_score': [False],  # Can turn on if bootstrap=True
    'model__random_state': [42],  # For reproducibility
    'model__verbose': [0],
    'model__warm_start': [False]
}

# GridSearchCV setup with balanced log loss (neg_log_loss)
grid_search = GridSearchCV(
    estimator=pipe_RF,
    param_grid=param_grid_random_forest,
    scoring=balanced_log_loss_scorer,  # Use negative log loss
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # Use all processors
    verbose=1  # Display progress
)

# # Fit GridSearchCV
# grid_search.fit(X_train, y_train)

# # Results
# print(f"Best Parameters: {grid_search.best_params_}")
# print(f"Best Score: {grid_search.best_score_:.2f}")



from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define the pipeline
pipe_RF = Pipeline([
    ('transform', ct),
    ('model', RandomForestClassifier())
])

param_grid_random_forest = {
    'model__bootstrap': [True],
    'model__ccp_alpha': [0.0],  # Usually left at default unless pruning
    'model__class_weight': [None, 'balanced'],  # Try automatic balancing
    'model__criterion': ['gini', 'entropy'],  # Split quality metric
    'model__max_depth': [5, 10, 15, 20, None],  # Control overfitting
    'model__max_features': ['sqrt', 'log2', None],  # Features at each split
    'model__max_leaf_nodes': [None, 10, 20, 30],  # Limit tree complexity
    'model__max_samples': [None],  # Subsampling — keep off for now
    'model__min_impurity_decrease': [0.0],  # Leave default
    'model__min_samples_leaf': [1, 2, 4],  # Control leaf size
    'model__min_samples_split': [2, 5, 10],  # Min samples for split
    'model__min_weight_fraction_leaf': [0.0],  # Usually left at 0.0
    'model__n_estimators': [100, 200, 300],  # Number of trees
    'model__n_jobs': [-1],  # Use all cores
    'model__oob_score': [False],  # Can turn on if bootstrap=True
    'model__random_state': [42],  # For reproducibility
    'model__verbose': [0],
    'model__warm_start': [False]
}

# RandomizedSearchCV with the same param grid
random_search = RandomizedSearchCV(
    estimator=pipe_RF,
    param_distributions=param_grid_random_forest,  # Use the same grid
    n_iter=50,  # Number of random combinations to try
    scoring=balanced_log_loss_scorer,
    cv=5,  # 5-fold cross-validation
    verbose=1,  # Display progress
    n_jobs=-1,  # Use all processors
    random_state=42,  # For reproducibility
    pre_dispatch='2*n_jobs'  # Parallelize job dispatch
)

# Define the pipeline with the best parameters
pipe_RF_best = Pipeline([
    ('transform', ct),
    ('model', RandomForestClassifier(
        warm_start=False,
        verbose=0,
        random_state=42,
        oob_score=False,
        n_jobs=-1,
        n_estimators=200,
        min_weight_fraction_leaf=0.0,
        min_samples_split=2,
        min_samples_leaf=4,
        min_impurity_decrease=0.0,
        max_samples=None,
        max_leaf_nodes=30,
        max_features=None,
        max_depth=5,
        criterion='gini',
        class_weight='balanced',
        ccp_alpha=0.0,
        bootstrap=True
    ))
])

pipe_RF_best.fit(X_train, y_train)

# Predict
y_pred_proba = pipe_RF_best.predict_proba(X_test)
y_pred = pipe_RF_best.predict(X_test)

#scores
accuracy = accuracy_score(y_test, y_pred) # Get the accuracy score
print("Accuracy:", accuracy)
balanced_log = balanced_log_loss(y_test, y_pred_proba)
print("Balanced Log Loss:", balanced_log)



# # Fit RandomizedSearchCV on the training data
# random_search.fit(X_train, y_train)

# # Print the best parameters and the best score
# print(f"Best Parameters: {random_search.best_params_}")
# print(f"Best Score: {random_search.best_score_:.2f}")

# createSubmission(random_search.best_estimator_, "submission.csv")



from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier

# Define the pipeline
catboost_model_pipeline = Pipeline([
    ('preprocessor', ct),
    ('classifier', CatBoostClassifier(verbose=0, random_seed=42))
])

# Define parameter distributions for RandomizedSearchCV
param_distributions_bayesian = {
    'classifier__iterations': [100, 200, 300, 500],
    'classifier__depth': [4, 6, 8, 10],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__l2_leaf_reg': [1, 3, 5, 7, 10],
    'classifier__bootstrap_type': ['Bayesian'],
    'classifier__bagging_temperature': [0, 0.5, 1, 2],
    'classifier__border_count': [32, 64, 128],
    'classifier__random_strength': [1, 5, 10],
}

param_distributions = {
    'classifier__iterations': [100, 200, 300, 500],
    'classifier__depth': [4, 6, 8, 10],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__l2_leaf_reg': [1, 3, 5, 7, 10],
    'classifier__bootstrap_type': ['Bernoulli', 'MVS'],
    'classifier__border_count': [32, 64, 128],
    'classifier__random_strength': [1, 5, 10],
}

# RandomizedSearchCV setup
random_search = RandomizedSearchCV(
    estimator=catboost_model_pipeline,
    param_distributions=param_distributions,
    n_iter=30,  # Number of combinations to try
    scoring=balanced_log_loss_scorer,  # or 'neg_log_loss' if you prefer
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search_bayesian = RandomizedSearchCV(
    estimator=catboost_model_pipeline,
    param_distributions=param_distributions_bayesian,
    n_iter=30,  # Number of combinations to try
    scoring=balanced_log_loss_scorer,  # or 'neg_log_loss' if you prefer
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

cb_model = CatBoostClassifier(
    random_strength=10,
    learning_rate=0.2,
    l2_leaf_reg=3,
    iterations=100,
    depth=4,
    border_count=128,
    bootstrap_type='Bayesian',
    bagging_temperature=2,
    verbose=0,               # silence training output
    random_state=42
)

# --- Pipeline ---
pipeline = make_pipeline(
    ct,
    cb_model
)

# --- Train ---
pipeline.fit(X_train, y_train)

# --- Predict & evaluate ---
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)

#scores
accuracy = accuracy_score(y_test, y_pred) # Get the accuracy score
print("Accuracy:", accuracy)
balanced_log = balanced_log_loss(y_test, y_pred_proba)
print("Balanced Log Loss:", balanced_log)

createSubmission(pipeline, "submission.csv")



# # Fit the randomized search
# random_search.fit(X_train, y_train)
# random_search_bayesian.fit(X_train, y_train)

# Show results
# print(f"Best Parameters: {random_search.best_params_}")
# print(f"Best Score: {random_search.best_score_:.4f}")

# print(f"Best Parameters bay: {random_search_bayesian.best_params_}")
# print(f"Best Score bay: {random_search_bayesian.best_score_:.4f}")

# createSubmission(random_search.best_estimator_, "submission.csv")



from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Full pipeline with MLPClassifier
clf = make_pipeline(
    ct,
    MLPClassifier(random_state=42)
)

# Train
clf.fit(X_train, y_train)

# Predict
y_pred_proba = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

#scores
accuracy = accuracy_score(y_test, y_pred) # Get the accuracy score
print("Accuracy:", accuracy)
balanced_log = balanced_log_loss(y_test, y_pred_proba)
print("Balanced Log Loss:", balanced_log)






# Create pipeline with placeholder MLPClassifier
pipeline = make_pipeline(
    ct,
    MLPClassifier(random_state=42)
)

# Define parameter space for RandomizedSearchCV
param_distributions = {
    'mlpclassifier__hidden_layer_sizes': [
        (50,), (100,), (100, 50), (50, 25), (100, 100), (150, 100, 50)
    ],
    'mlpclassifier__activation': ['tanh', 'relu'],
    'mlpclassifier__solver': ['adam', 'sgd'],
    'mlpclassifier__alpha': [1e-4, 1e-3, 1e-2],
    'mlpclassifier__learning_rate': ['constant', 'adaptive'],
    'mlpclassifier__max_iter': [200, 300, 500, 1000, 2000],
}

mlp = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=300,
    cv=5,
    scoring=balanced_log_loss_scorer,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

best_model = make_pipeline(
    ct,
    MLPClassifier(
        solver='sgd',
        max_iter=500,
        learning_rate='constant',
        hidden_layer_sizes=(100,),
        alpha=0.01,
        activation='tanh',
        random_state=42
    )
)

# # fit
# best_model.fit(X_train, y_train)

# # Predict
# y_pred_proba = best_model.predict_proba(X_test)
# y_pred = best_model.predict(X_test)

# #scores
# accuracy = accuracy_score(y_test, y_pred) # Get the accuracy score
# print("Accuracy:", accuracy)
# balanced_log = balanced_log_loss(y_test, y_pred_proba)
# print("Balanced Log Loss:", balanced_log)



# Fit the search
mlp.fit(X_train, y_train)

# Evaluate
print("Best parameters:", mlp.best_params_)
print("Best score:", mlp.best_score_)


createSubmission(mlp.best_estimator_, "submission.csv")


