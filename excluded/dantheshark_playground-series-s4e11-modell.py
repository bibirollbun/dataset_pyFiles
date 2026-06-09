from IPython.display import display, Markdown
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
import math
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import pandas as pd 
import scipy.stats as ss
import seaborn as sns
import os
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Decide between local or kaggle cloud storage         
KAGGLE_ENV = 'kaggle' in os.listdir('/')
data_path = '/kaggle/input' if KAGGLE_ENV else '../kaggle/input'

# This is a good idea to work only locally. But If you wanna ran your NB also at kaggle... this is not working.
# # Pull the dataset from kaggle, it is concat dataset train + original dataset
# dataset_name = 'dantheshark/s4-e11-train-concat'
# if KAGGLE_ENV:
#     kaggle.api.dataset_download_files(dataset_name, path="../kaggle/input/", unzip=True)


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
    
for dirname, _, filenames in os.walk(data_path):
    for filename in filenames:
        print(os.path.join(dirname, filename)) 


# Load the data
train_original = pd.read_csv(data_path + '/playground-series-s4e11/train.csv')
test_original = pd.read_csv(data_path + '/playground-series-s4e11/test.csv')
sample_submission = pd.read_csv(data_path + '/playground-series-s4e11/sample_submission.csv')

original_data = pd.read_csv(data_path + '/depression-surveydataset-for-analysis/final_depression_dataset_1.csv')

train_concat_data = pd.read_csv(data_path + '/s4-e11-train-concat/s4-e11-train-concat.csv')
train_final_data = pd.read_csv(data_path + '/s4-e11-train-concat-final/s4-e11-train-concat-final.csv')

test_concat_data = pd.read_csv(data_path + '/s4-e11-test-concat/s4-e11-test-concat.csv')
test_final_data = pd.read_csv(data_path + '/s4-e11-test-concat-final/s4-e11-test-concat-final.csv')
submission_template = pd.read_csv(data_path + '/submission.csv')


train_final_data.head(100)


test_final_data.head(100)


# Load preprocessed data
df = train_final_data.copy()

# Define features and target variable
target_column = "Depression"  # Update with your actual target column
X = df.drop(columns=[target_column])
y = df[target_column]

# Split into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)




# Define classifiers
models = {
    "XGBoost": xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method = "hist", device = "cuda"  
    ),
    # "XGBoost": xgb.XGBClassifier(
    #     enable_categorical=True, 
    #     eval_metric='logloss',
    #     random_state=42,
    #     tree_method = "hist", device = "cuda"  
    # ),  
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        device="cpu"
        #tree_method = "hist", device = "cuda"
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )
}

# Train and evaluate each model
best_model = None
best_accuracy = 0
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.2f}")
    
    # Save best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# Save the best model
MODEL_PATH = f"best_model_{best_model_name}.pkl"
joblib.dump(best_model, MODEL_PATH)
print(f"Best model ({best_model_name}) saved to {MODEL_PATH}")


####

param_grid = {
    #, 5, 7, 10],  # Controls the depth of the trees
        'learning_rate': [0.1],#0.01, 0.05, 0.1, 0.2],  # Step size for updates
        'n_estimators': [300],#100, 200, 300],  # Number of trees
        'subsample': [0.8],#, 1.0],  # Fraction of data per tree
        'colsample_bytree': [0.8],#, 1.0],  # Fraction of features per tree
        'gamma': [0],#, 0.1, 0.2],  # Minimum loss reduction to make a split
        'tree_method': ['hist']  # GPU-friendly method
}
grid_search = GridSearchCV(
    estimator=XGBClassifier(enable_categorical=True, random_state=42, eval_metric='logloss'),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=1,
    n_jobs=-1  # all CPU cores
)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy (CV):", grid_search.best_score_)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_temp)
accuracy = accuracy_score(y_temp, y_pred)
print(f"Accuracy on the validation set with optimized parameters: {accuracy:.5f}")

y_pred_test = best_model.predict(test_final_data)






predicitions = best_model.predict(test_final_data)
submission_template['Depression'] = predicitions
submission_template.to_csv('submission_boy.csv', index=False)
submission_template.head(100)


import pandas as pd
import matplotlib.pyplot as plt

# Predictions in DataFrame speichern
df_predictions = pd.DataFrame({"Prediction": predictions})

# Verteilung der Vorhersagen anzeigen
df_predictions["Prediction"].value_counts().plot(kind="bar")
plt.title("Distribution of Predictions")
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.show()



importances = best_model.feature_importances_
feature_names = test_final_data.columns

# DataFrame mit Feature-Wichtigkeit
df_importances = pd.DataFrame({"Feature": feature_names, "Importance": importances})
df_importances = df_importances.sort_values(by="Importance", ascending=False)

# Plot
plt.figure(figsize=(10,5))
sns.barplot(x=df_importances["Importance"], y=df_importances["Feature"])
plt.title("Feature Importance")
plt.show()



# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import seaborn as sns

# # Echte Labels (falls vorhanden)
# y_true = test_labels  # Falls y_test existiert

# # Accuracy Score berechnen
# accuracy = accuracy_score(y_true, predictions)
# print(f"Accuracy Score: {accuracy:.2f}")

# # Confusion Matrix
# cm = confusion_matrix(y_true, predictions)

# # Visualisierung der Confusion Matrix
# plt.figure(figsize=(6,4))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix")
# plt.show()

# # Classification Report
# print(classification_report(y_true, predictions))



# Wahrscheinlichkeiten der Predictions abrufen
y_probs = best_model.predict_proba(test_final_data)

# In DataFrame speichern
df_probs = pd.DataFrame(y_probs, columns=["Prob_0", "Prob_1"])

# Histogramm der Wahrscheinlichkeiten
df_probs["Prob_1"].hist(bins=30, alpha=0.7)
plt.title("Prediction Probabilities for Class 1")
plt.xlabel("Probability")
plt.ylabel("Count")
plt.show()


