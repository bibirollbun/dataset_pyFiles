import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn import metrics

train_df = pd.read_csv('/kaggle/input/aviakompaniya/train_dataset.csv')
test_df = pd.read_csv('/kaggle/input/aviakompaniya/test_dataset.csv')
sample_solution_df = pd.read_csv('/kaggle/input/aviakompaniya/sample_submission.csv')


train_df.head()


train_df.info()


train_df["Arrival Delay in Minutes"] = train_df["Arrival Delay in Minutes"].fillna(train_df["Arrival Delay in Minutes"].mean())


X = train_df.drop(["id", "satisfaction"], axis=1)
y = train_df['satisfaction']

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=12)


categorical_features = X.select_dtypes(include=['category', 'object']).columns.tolist()
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

categorical_features


numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
])


X_prepared = full_pipeline.fit_transform(X_train)
X_prepared


param_grid = {
    'n_estimators': [10, 20, 30],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the RandomForestClassifier and GridSearchCV
model = RandomForestClassifier()
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)

grid_search.fit(X_prepared, y_train)

# Best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy found: ", grid_search.best_score_)

# Retrieve the best model
best_model = grid_search.best_estimator_

# Optionally: Evaluate on the test set
X_test_prepared = full_pipeline.transform(X_test)
predictions =  best_model.predict(X_test_prepared)
print("Test set accuracy: ", metrics.accuracy_score(y_test, predictions))


# confusion matrix
conf_mat = metrics.confusion_matrix(y_test, predictions)
sns.heatmap(conf_mat, annot=True,fmt="g")
plt.show()

## ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC curve')
display.plot()
plt.show()


test_df.head()


test_df["Arrival Delay in Minutes"] = test_df["Arrival Delay in Minutes"].fillna(test_df["Arrival Delay in Minutes"].mean())


# Prepare test dataset
X_final_test = test_df.drop(['id'], axis=1)
X_final_test_prepared = full_pipeline.transform(X_final_test)

# Make predictions
y_final_predicted = best_model.predict(X_final_test_prepared)

# Create submission file
submission_df = pd.DataFrame({
    'id': test_df['id'],  # Keep the original IDs
    'satisfaction': y_final_predicted  # Predicted values
})

# Save to CSV
submission_df.to_csv("submission.csv", index=False)

print("Submission file saved as submission.csv!")


submission = pd.read_csv("/kaggle/working/submission.csv")

submission.info()


submission['satisfaction'].value_counts()




