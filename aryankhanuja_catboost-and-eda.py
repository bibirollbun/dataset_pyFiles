# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings("ignore")
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
train_df=train_df.replace('NaN', np.nan)
test_df=test_df.replace('NaN', np.nan)
submission_df=submission_df.replace('NaN', np.nan)


print(train_df.shape)
print(test_df.shape)
print(submission_df.shape)


train_df.head(2)


test_df.head(2)


submission_df.head(2)


for x in train_df.columns:
    print(x)
    print(train_df[x].value_counts(dropna=False).sort_index())


display(train_df.describe(include="all").T)
print(f"Duplicate rows (train): {test_df.duplicated().sum()}  |  (test): {train_df.duplicated().sum()}")


test_df.info()


categorical_cols = ['Stage_fear', 'Drained_after_socializing']
for col in categorical_cols:
    print(f"\n{col} value counts:\n", test_df[col].value_counts(dropna=False))


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


corr= test_df.corr(numeric_only=True)
plt.figure(figsize=(5, 3)) # Adjust figure size as needed
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np


df_impute_train=train_df.copy()


# 1. Separate column types
num_cols = df_impute_train.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df_impute_train.select_dtypes(include=['object', 'category', 'bool']).columns


#Median
num_imputer = SimpleImputer(strategy='median')
df_impute_train[num_cols] = num_imputer.fit_transform(df_impute_train[num_cols])


#MODE
cat_imputer = SimpleImputer(strategy='most_frequent')
df_impute_train[cat_cols] = cat_imputer.fit_transform(df_impute_train[cat_cols])


df_impute_train.isnull().sum().sort_values(ascending=False)


from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Define features and target
X = df_impute_train.drop(columns=['Personality'])  # Assuming this is your target
y = df_impute_train['Personality']

# Step 2: Identify categorical columns (by index)
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


cat_features = X_train.select_dtypes(include=['object', 'category','bool']).columns.tolist()


cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric='Accuracy',
    use_best_model=True,
    random_seed=42,
    verbose=100
)
# model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
model.fit(X_train, y_train, eval_set=(X_val, y_val),use_best_model=True, cat_features=cat_features)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_pred = model.predict(X_val)
cm = confusion_matrix(y_val, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.grid(False)
plt.show()


from sklearn.model_selection import GridSearchCV

# Define base model (no fitting yet)
cat_model_gscv = CatBoostClassifier(cat_features=cat_features, 
                                    verbose=100,
                                    task_type="CPU",
                                    loss_function="MultiClass",
                                    eval_metric="Accuracy", 
                                    random_seed=42)

# Define parameter grid
param_grid = {
    'depth': [8],
    'learning_rate': [0.01, 0.05],
    'l2_leaf_reg': [5, 7],
    'iterations': [500, 1000]
}

# Wrap in GridSearchCV
grid_search = GridSearchCV(estimator=cat_model_gscv,
                           param_grid=param_grid,
                           cv=5,  # 5-fold cross-validation
                           scoring='accuracy',  # or 'f1_macro', etc.
                           n_jobs=-1)

# Fit on training data
grid_search.fit(X_train, y_train)



print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

# Best model
best_model = grid_search.best_estimator_


import optuna
from sklearn.metrics import f1_score

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 300, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'loss_function': 'MultiClass',
        'eval_metric': 'Accuracy',
        'cat_features': cat_features,
        'verbose': 0,
        'task_type': 'CPU',
        'random_seed': 42
    }

    catboost_optuna_model = CatBoostClassifier(**params)
    catboost_optuna_model.fit(X_train, y_train)

    y_pred = catboost_optuna_model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='macro')
    return f1



study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20, show_progress_bar=True)


print("Best F1 Score:", study.best_value)
print("Best Params:", study.best_params)
optuna_params=study.best_params


final_model_optuna = CatBoostClassifier(
    **optuna_params,                         # unpack tuned params
    loss_function='MultiClass',             # still required
    eval_metric='Accuracy',
    cat_features=cat_features,
    task_type='CPU',
    verbose=100,
    random_seed=42
)

final_model_optuna.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)



# Step 1: Copy test data
X_test = test_df.copy()

# Step 2: Identify categorical columns
cat_features = X_test.select_dtypes(include=['object']).columns.tolist()

# Step 3: Fix categorical NaNs and convert to string
for col in cat_features:
    X_test[col] = X_test[col].fillna("missing").astype(str)

# Step 4: Convert integer columns with NaN to float
for col in X_test.columns:
    if pd.api.types.is_integer_dtype(X_test[col]) and X_test[col].isna().sum() > 0:
        X_test[col] = X_test[col].astype(float)

# Step 5: Predict using trained CatBoost model
y_test_pred = final_model_optuna.predict(X_test)

# Step 6: If you need class probabilities instead
# y_test_proba = final_model_optuna.predict_proba(X_test)



y_test_pred = y_test_pred.ravel()  # or .flatten() or .squeeze()
print(y_test_pred)


submission_df = pd.DataFrame({
    'id': test_df['id'],
    'Personality': y_test_pred
})


submission_df.head(5)


submission_df.to_csv("submission_1.1.csv", index=False)

