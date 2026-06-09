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

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings
warnings.filterwarnings("ignore")


file_path="/kaggle/input/playground-series-s5e12/train.csv"


df=pd.read_csv(file_path)


df.shape


df.head()


df.info()


df=df.drop(columns=['id'])


#after delet id column
df.info()


print(df.duplicated().sum())


df.describe()


for col in df.columns:
    unique_values = df[col].unique() 
    print(f"Column '{col}' has {len(unique_values)} unique values:")
    print(unique_values)
    print("-"*50)


# List of categorical columns
categorical_columns = df.select_dtypes(include=['object', 'category']).columns

for col in categorical_columns:
    print(f"Column '{col}' frequency count:")
    print(df[col].value_counts())  # Count occurrences of each unique value
    print('-'*50)


import matplotlib.pyplot as plt
for col in categorical_columns:
   
    counts = df[col].value_counts()
    labels = counts.index.tolist()
    sizes = counts.values.tolist()

   
    colors = plt.cm.tab20.colors[:len(labels)] 
    explode = [0.05 if size < max(sizes)*0.1 else 0 for size in sizes] 
   
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        explode=explode,
        pctdistance=0.85,
        textprops={'fontsize': 10},
    )

    ax.axis('equal')  
    plt.title(f'Distribution of {col}', fontsize=14, weight='bold')
    ax.legend(wedges, labels, title=col, loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.tight_layout()
    plt.show()


numeric_cols = df.select_dtypes(include=['int64', 'float64'])


# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Compute correlation matrix
corr_matrix = numeric_cols.corr()

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap (Numeric Features)")
plt.show()


!pip install optuna



import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score



X = df.drop(columns=["diagnosed_diabetes"])
y = df["diagnosed_diabetes"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

categorical_features = X.select_dtypes(include=['object']).columns.tolist()



def objective(trial):

    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 2),
        "random_strength": trial.suggest_float("random_strength", 0, 2),
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "task_type": "GPU",          #GPU
        "devices": "0",
        "random_seed": 42,
        "verbose": 0,
        "use_best_model": True
    }

    model = CatBoostClassifier(**params)

    model.fit(
        X_train, y_train,
        cat_features=categorical_features,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50
    )

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)

    return auc



study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(objective, n_trials=30)



from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

# Best hyperparameters obtained from Optuna
best_params = study.best_params

# Create the final model using the best hyperparameters
final_model = CatBoostClassifier(
    iterations=1000,                     # You can adjust this based on dataset size
    learning_rate=best_params['learning_rate'],
    depth=best_params['depth'],
    l2_leaf_reg=best_params['l2_leaf_reg'],
    bagging_temperature=best_params['bagging_temperature'],
    random_strength=best_params['random_strength'],
    eval_metric='AUC',
    task_type='GPU',                     # Use GPU for training
    devices='0',                         # Specify GPU device (single GPU)
    random_seed=42,
    verbose=100,                         # Show training progress every 100 iterations
    use_best_model=True                  # Save the best model during training
)

# Train the model on the full training dataset
final_model.fit(
    X, y,                               # Full training data
    cat_features=categorical_features,
    eval_set=(X_val, y_val),            # Validation set for early stopping
    early_stopping_rounds=50
)






# Load the test data
X_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")



categorical_features_test = X_test.select_dtypes(include=['object']).columns.tolist()



from catboost import Pool

# Create a Pool object and specify categorical feature columns
test_pool = Pool(
    data=X_test,
    cat_features=categorical_features_test
)

# predict
y_test_pred = final_model.predict_proba(test_pool)[:, 1]




# create Submission File
submission = pd.DataFrame({
    "id": X_test["id"],
    "diagnosed_diabetes": y_test_pred
})


submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")







