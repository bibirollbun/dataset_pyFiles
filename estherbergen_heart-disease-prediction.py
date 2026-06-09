import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Imports
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    roc_auc_score, precision_score, recall_score, f1_score
    )
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

# Bring in the data
df = pd.read_csv("/kaggle/input/heart-disease-prediction-dataquest/heart_train.csv")

# Set random seed
SEED = 99


df.info()
# Note: No null values to handle
# Note: Data types include int64, float64, and object


df.head(10)


df.describe()


df['HeartDisease'].value_counts().plot(kind="bar")



# Create X list of features and and y list of target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=SEED, stratify = y)

# Separate features into categorical and numeric columns
# Knowing the data types from df.info() above lets us shortcut this instead of referring to each column by name
cat_features = X.select_dtypes(include=["object"]).columns
num_features = X.select_dtypes(include=["int64", "float64"]).columns


# Normalize the numeric features so they are all scaled from 0 to 1
num_transformer = StandardScaler()

# One-hot encode the categorical features to turn them into numeric binary categories for ML model
cat_transformer = OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False)

# Create a column transformer to run the features through the appropriate transformer
preprocessor = ColumnTransformer(
    transformers = [
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])


# Optional: Process data for heatmap
encoded_cats = cat_transformer.fit_transform(X[cat_features])
encoded_cat_columns = cat_transformer.get_feature_names_out(cat_features)
encoded_df = pd.DataFrame(encoded_cats, columns=encoded_cat_columns)
full_encoded_X = pd.concat([X[num_features], encoded_df], axis=1)
full_encoded_X['HeartDisease'] = y
corr_matrix = full_encoded_X.corr()

# Display heatmap
plt.figure(figsize=(9, 6))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0, fmt=".2f", annot=False)
plt.tight_layout()
plt.show()
# Note that since the categorical transformer used drop='first', this is not a full representation of every possible value


# Create a pipeline to apply the transformers and initiate the classifier
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(random_state=SEED, solver="liblinear"))
])


# Do a grid search to check for different values of max_iter
param_grid = {'classifier__max_iter': [100, 400, 1000]}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Best max_iter:", grid_search.best_params_)


best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))


y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]


# Calculate metrics for classification model
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "ROC AUC": roc_auc_score(y_test, y_pred_proba),
}

for name, score in metrics.items():
    print(f"{name}: {round(score, 3)}")




conf_matrix = confusion_matrix(y_test, y_pred),
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)


# Import the test CSV
test_df = pd.read_csv('/kaggle/input/heart-disease-prediction-dataquest/heart_test.csv')
test_df.head()


test_features = test_df[X.columns]
test_predictions = best_model.predict(test_features)
test_predictions


submission_df = pd.DataFrame({"HeartDisease": test_predictions})
submission_df.head()
# sub_sample = pd.read_csv("/kaggle/input/heart-disease-prediction-dataquest/sample_submission.csv")
# sub_sample.head()


sub = pd.DataFrame({"id": submission_df.index.values, "HeartDisease": test_predictions})
sub.head()
sub.to_csv("submission.csv")

