! pip install catboost missingno
from catboost import CatBoostClassifier
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno 
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OrdinalEncoder


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv', index_col = "id")
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')



# Print shape of the data
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Display data types and missing values
display(train.info())
display(train.describe())


# Visualize target distribution

plt.figure(figsize=(10,5))
sns.countplot(data=train, x="Personality", palette="pastel")
plt.title("Target Distribution: Introverts vs Extroverts", fontsize=14)
plt.xlabel("Personality Traits")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


numerical_columns = train.select_dtypes(include="number").columns.tolist()
num_cols_grid = 3
num_rows = (len(numerical_columns) + num_cols_grid-1)//num_cols_grid

fig, axes = plt.subplots(num_rows, num_cols_grid, figsize=(15, 5 * num_rows), constrained_layout=False)
axes = axes.flatten()

for i, col in enumerate(numerical_columns):
    sns.kdeplot(train[col], ax=axes[i], color = 'blue', fill = True)
    if col in train.columns:
        sns.kdeplot(test[col], ax=axes[i], color = 'red', fill = True) 
        axes[i].set_title(col)

    ax_box = axes[i].inset_axes([0.2, -0.3, 0.6, 0.2])  # [x, y, width, height]
    sns.boxplot(x=train[col], ax=ax_box, orient='h')
    ax_box.set(xlabel='')
for j in range(len(numerical_columns), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
corr = train[numerical_columns].corr()

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    center=0,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    square=True
)

plt.title("Correlation Heatmap: Features", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



categorical_columns = train.select_dtypes(include=["category", 'object']).columns.tolist()

for col in categorical_columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=train, x=col, order=train[col].value_counts().index, palette='Set3')
    plt.title(f'Categorical Distribution: {col}')
    plt.ylabel("Count")
    plt.xlabel(col)
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()



# Check and visualize missing values
missing_cols = train.columns[train.isnull().any()]
if not missing_cols.empty:
    print("Missing Value Summary:")
    display(train[missing_cols].isnull().sum().sort_values(ascending=False))

    # Bar chart of missing data
    msno.bar(train[missing_cols], color='skyblue', figsize=(8, 4), fontsize=12)
    plt.title("Missing Values per Column")
    plt.show()

    # Matrix view for missingness pattern
    msno.matrix(train[missing_cols], figsize=(10, 4), sparkline=False)
    plt.title("Matrix showing missing values")
    plt.show()
else:
    print("No missing values in the dataset!")



# Not a very sophisticated way of checking the number of missing values

missing_per_row = train.isnull().sum(axis=1)
rows_with_multiple_missing_1 = (missing_per_row > 1).sum()
rows_with_multiple_missing_2 = (missing_per_row > 2).sum()
rows_with_multiple_missing_3 = (missing_per_row > 3).sum()

print(f"Rows with more than 1 missing value: {rows_with_multiple_missing_1}")
print(f"Rows with more than 2 missing values: {rows_with_multiple_missing_2}")
print(f"Rows with more than 3 missing values: {rows_with_multiple_missing_3}")



train_cleaned = train[train.isnull().sum(axis=1) <3]
msno.matrix(train_cleaned)


# Repeated for the test set 
missing_per_row = test.isnull().sum(axis=1)
rows_with_multiple_missing_1 = (missing_per_row > 1).sum()
rows_with_multiple_missing_2 = (missing_per_row > 2).sum()
rows_with_multiple_missing_3 = (missing_per_row > 3).sum()

print(f"Rows with more than 1 missing value: {rows_with_multiple_missing_1}")
print(f"Rows with more than 2 missing values: {rows_with_multiple_missing_2}")
print(f"Rows with more than 3 missing values: {rows_with_multiple_missing_3}")



# Split into X, y subsets 

target_col = "Personality"

X = train.drop(columns=[target_col])
y = train[target_col]


# Create preprocessing transformers - starting with some easy preprocessing techniques 
categorical_columns.remove("Personality")

cat_pipeline = Pipeline(steps=[
                       ("imputer", SimpleImputer(strategy = "most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown = "use_encoded_value", unknown_value=-1))
])

num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))
])
preprocessor = ColumnTransformer(transformers = [
    ("cat", cat_pipeline, categorical_columns),
    ("num", num_pipeline, numerical_columns)
])

preprocessor.set_output(transform="pandas")

X_prepared = preprocessor.fit_transform(X)
test_prepared = preprocessor.transform(test)


print("Columns available in X_prepared:", X_prepared.columns.tolist())

new_categorical_columns = ['num__Social_event_attendance', 'num__Going_outside', 'num__Post_frequency', "cat__Stage_fear", "cat__Drained_after_socializing"]

for col in new_categorical_columns:
    if col in X_prepared.columns:
        X_prepared[col] = X_prepared[col].astype(str).astype("category")
    if col in test_prepared.columns:
        test_prepared[col] = test_prepared[col].astype(str).astype("category")




X_prepared.info()


!pip install tabpfn torch


from tabpfn import TabPFNClassifier
import torch

clf = TabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu", ignore_pretraining_limits=True)
clf.fit(X_prepared.values, y.values)



predicted_values = clf.predict(test_prepared.values)
predicted_values
#predicted_values.to_csv("submission.csv")


submission = pd.DataFrame({
    "id": test["id"],
    "Personality": predicted_values
})


submission.to_csv("submission.csv", index=False)



prediction_probabilities = clf.predict_proba(test_prepared)


# Look into the entries where TABpfn is the least confident about 
confidence_val = np.max(prediction_probabilities, axis=1)
val_results = pd.DataFrame({
    "predicted_label": np.argmax(prediction_probabilities, axis=1),
    "confidence": confidence_val,
    "kinda_uncertainty": 1-confidence_val
})


val_results_sorted = val_results.sort_values(by="kinda_uncertainty", ascending=False)
val_results_sorted.head(10)


plt.figure(figsize=(6, 3.5))
sns.histplot(val_results["kinda_uncertainty"], bins=50, color="salmon", kde=True)
plt.title("Distribution of Model Uncertainty (Validation Set)")
plt.xlabel("Uncertainty (1 - Confidence)")
plt.ylabel("Count")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


confidence = np.max(prediction_probabilities, axis=1)
predicted_class = np.argmax(prediction_probabilities, axis=1)

plt.figure(figsize=(8, 4))
sns.histplot(confidence, bins=100, kde=True, color="cornflowerblue", edgecolor="black")
plt.axvline(0.6, color="red", linestyle="--", label="Low-confidence threshold (0.6)")

plt.title("Model Prediction Confidence (Validation Set)", fontsize=14)
plt.xlabel("Model Confidence", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xlim(0.5, 1.01)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()


cat_cols = X_prepared.select_dtypes(include=['object', 'category']).columns.tolist()


catboost_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric='Accuracy',
    random_seed=42,
    verbose=100
)
catboost_model.fit(X_prepared, y, cat_features = cat_cols)


catboost_test = catboost_model.predict(test_prepared)
submission_catboost = pd.DataFrame({
    "id": test["id"],
    "Personality": catboost_test
})
submission_catboost.to_csv("submission_catboost_2.csv",index=False)


catboost_test_proba = catboost_model.predict_proba(test_prepared)


confidence_cb = np.max(catboost_test_proba, axis=1)
val_results = pd.DataFrame({
    "predicted_label": np.argmax(catboost_test_proba, axis=1),
    "confidence": confidence_cb,
    "kinda_uncertainty": 1-confidence_cb
})


val_results_sorted = val_results.sort_values(by="kinda_uncertainty", ascending=False)
val_results_sorted.head(10)


# One of the results where both models output different outcomes 
val_results.loc[3276]


plt.figure(figsize=(6, 3.5))
sns.histplot(val_results["kinda_uncertainty"], bins=50, color="salmon", kde=True)
plt.title("Catboost: Distribution of Model Uncertainty (Validation Set)")
plt.xlabel("Uncertainty (1 - Confidence)")
plt.ylabel("Count")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


prediction_probabilities


# Average predicted probabilities (equal weight)
ensemble_test_proba = 0.5 * prediction_probabilities + 0.5 * catboost_test_proba
ensemble_test_pred = np.argmax(ensemble_test_proba, axis=1)
personality_map = {0: "Extrovert", 1: "Introvert"}
final_labels = pd.Series(ensemble_test_pred).map(personality_map)
submission_ensemble = pd.DataFrame({
    "id": test["id"],
    "Personality": final_labels
})


#submission.to_csv("submission_ensemble_weighted.csv", index=False)
#submission.head()

