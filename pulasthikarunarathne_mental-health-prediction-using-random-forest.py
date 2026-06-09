import os
os.listdir('/kaggle/input')


#1 Loading dataset by importing pandas 
import pandas as pd
df_train = pd.read_csv('/kaggle/input/depressed-people/train.csv')


#Dropping the Index Column
df_train.drop(columns = ['index'], inplace=True)


#Converting / Encoding categorical data into numerical data
from sklearn.preprocessing import LabelEncoder 

#encoding categorical columns
binary_columns = ['Gender', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness', 'Depression']

encoders = {}


for col in binary_columns: 
    le = LabelEncoder()
    df_train[col]= le.fit_transform(df_train[col])
    encoders[col] = le #for future use 

print(f"\n{col} Mapping:")
for idx, label in enumerate(le.classes_):
    print(f"{label} → {idx} ")


#Mapping for 'Dietary Habits'
diet_mapping = {'Unhealthy': 0, 
                'Moderate': 1, 
                'Healthy': 2}

df_train['Dietary Habits'] = df_train['Dietary Habits'].map(diet_mapping)


# Training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


sleep_mapping = {'Less than 5 hours': 4.5,
                'More than 8 hours': 9,
                '5-6 hours': 5.5,
                '7-8 hours': 7.5}

df_train['Sleep Duration'] = df_train['Sleep Duration'].map(sleep_mapping)


# Target Column
target = 'Depression'

#Features all the other columns except target
x = df_train.drop(columns=[target])
y = df_train[target]

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.2, #20% test set
                                                    random_state = 42, # for reprodicibility
                                                    stratify = y #preserve class  balance
                                                   )


# Training the model
model = RandomForestClassifier (random_state=42)
model.fit(x_train, y_train)


y_pred = model.predict(x_test)


print("Accuracy: ", accuracy_score(y_test, y_pred) * 100) # accuracy

print("\nConfusion Matrix: ")
print(confusion_matrix(y_test, y_pred))   # Confusion matrix


report = classification_report(y_test, y_pred, target_names=['No', 'Yes'], output_dict=True)
report_df = pd.DataFrame(report).T * 100
print("\nClassification Report (%):")
print(report_df.round(2))


# Fine tuning (Hyperparameter Tunning)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


param_grid = {
    'n_estimators': [100, 200],           # number of trees
    'max_depth': [None, 10, 20],          # tree depth
    'min_samples_split': [2, 5],          # minimum samples to split
    'min_samples_leaf': [1, 2],           # minimum samples at leaf
    'criterion': ['gini', 'entropy']      # splitting criteria
}



rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,                         # 5-fold cross-validation
    scoring='accuracy',
    verbose=1,
    n_jobs=-1                     # use all CPU cores
)


grid_search.fit(x_train, y_train)


print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy: {:.2f}%".format(grid_search.best_score_ * 100))


#Evaluation on test
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Use the best model to predict test data
best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)

# Accuracy
print("Test Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))


# Loading the test file
df_test = pd.read_csv('/kaggle/input/depressed-people/test.csv')


df_test.isna().any().any()


df_test.head(10)


# 1️⃣ Drop index column if present
if 'index' in df_test.columns:
    df_test.drop(columns=['index'], inplace=True)

# Normalize all string columns to strip spaces and match case
for col in df_test.select_dtypes(include='object').columns:
    df_test[col] = df_test[col].str.strip().str.title()

# 2️⃣ Sleep Duration mapping (case/space safe)
sleep_mapping = {
    'Less Than 5 Hours': 4.5,
    'More Than 8 Hours': 9,
    '5-6 Hours': 5.5,
    '7-8 Hours': 7.5
}
df_test['Sleep Duration'] = df_test['Sleep Duration'].map(sleep_mapping)

# 3️⃣ Encode binary columns using encoders from training
for col in binary_columns:
    if col in df_test.columns:
        le = encoders[col]
        df_test[col] = df_test[col].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else le.transform([le.classes_[0]])[0]
        )

# 4️⃣ Dietary Habits mapping (case/space safe)
diet_mapping = {'Unhealthy': 0, 'Moderate': 1, 'Healthy': 2}
df_test['Dietary Habits'] = df_test['Dietary Habits'].map(diet_mapping)



import numpy as np
import pandas as pd

# 1) Clean NaNs just in case
if df_test.isnull().sum().sum() > 0:
    print("⚠️ Warning: NaNs found in test data. Filling with 0.")
    df_test = df_test.fillna(0)

# 2) Predict (classes, not probabilities)
test_predictions = best_model.predict(df_test)

# 3) Convert predictions to 'Yes'/'No'
if 'Depression' in globals() and 'encoders' in globals() and 'Depression' in encoders:
    # Use your target encoder if you encoded the target earlier
    labels = encoders['Depression'].inverse_transform(test_predictions)
else:
    # Robust mapping for common cases: 1/0, "1"/"0", True/False, "Yes"/"No"
    pred_series = pd.Series(test_predictions).astype(object)
    mapping = {1: "Yes", 0: "No", "1": "Yes", "0": "No", True: "Yes", False: "No", "Yes": "Yes", "No": "No"}
    labels = pred_series.map(mapping).fillna(pred_series.astype(str)).to_numpy()

# 4) Choose the ID column for 'index'
id_col = None
for c in ["index", "Id", "ID", "id", "row_id", "RowId"]:
    if c in df_test.columns:
        id_col = c
        break

if id_col is None:
    # Fallback: 1-based row numbers if your test file doesn't carry an id column
    idx_values = (pd.RangeIndex(start=0, stop=len(df_test)) + 1).astype(int)
else:
    idx_values = df_test[id_col].astype(int).to_numpy()

# 5) Build submission exactly like your template: ['index', 'Depression']
submission = pd.DataFrame({"index": idx_values, "Depression": labels}, columns=["index", "Depression"])
submission["index"] = submission["index"].astype(int)
submission["Depression"] = submission["Depression"].astype(str)

# 6) (Optional) If you have a template CSV, align to its indices
#    Set template_path to your sample submission CSV if you want to enforce ordering
template_path = "submission.csv"  # change or comment out if not using a template file
try:
    tpl = pd.read_csv(template_path)
    if {"index", "Depression"}.issubset(tpl.columns):
        pred_map = dict(zip(submission["index"], submission["Depression"]))
        tpl["Depression"] = tpl["index"].map(pred_map)
        # If any indices in template were missing predictions, default to 'No' (or choose your default)
        tpl["Depression"] = tpl["Depression"].fillna("No")
        submission = tpl[["index", "Depression"]]
except Exception:
    pass

# 7) Save
submission.to_csv("submission_ready.csv", index=False)
print("✅ Saved: submission_ready.csv")
print(submission.head())






