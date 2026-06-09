import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import label_ranking_average_precision_score
import warnings
warnings.filterwarnings('ignore')



df_train=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
original= pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')


df_train.shape, df_test.shape


df_train.head()


df_train.describe()



df_train.isnull().sum()


# Visuallize Distribution of Numerical Features
num_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']


for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df_train[col], kde=True,color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


cat_cols = ['Soil Type', 'Crop Type']
# Visualize Distribution of Categorical Features
for col in cat_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df_train, x=col, palette='viridis')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()


#Target Variable Distribution
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df_train,
    x='Fertilizer Name',
    palette='coolwarm',
    edgecolor='black',
    order=df_train['Fertilizer Name'].value_counts().index
)
plt.title('Distribution of Fertilizer Name (All 7 Classes)', fontsize=14)
plt.xlabel('Fertilizer Name', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


#Categorical Feature Distributions by Fertilizer Name
for col in cat_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_train, x=col, hue='Fertilizer Name', palette='Set2')
    plt.title(f'Distribution of {col} by Fertilizer Name')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Fertilizer Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# We will also used original data in training
df_train.drop(columns=['id'], inplace=True)

df_train = pd.concat([df_train, original], ignore_index=True)

#Label Encode Target Variable
le = LabelEncoder()

df_train['Fertilizer Name'] = le.fit_transform(df_train['Fertilizer Name'])

# One Hot Encoding Categorical Features
df_train=pd.get_dummies(df_train, columns=['Soil Type', 'Crop Type'], drop_first=True,dtype='int64')


X=df_train.drop(columns=['Fertilizer Name'])
y=df_train['Fertilizer Name']


#XGB Hyperparameter Tuning
# MAP@3 scorer
def mapk(actual, predicted, k=3):
    score = 0.0
    for a, p in zip(actual, predicted):
        try:
            score += 1 / (p[:k].index(a) + 1)
        except ValueError:
            pass
    return score / len(actual)

# Optuna objective function
def objective(trial):
    params = {
        "objective": "multi:softprob",
        "num_class": len(np.unique(y)),
        "eval_metric": "mlogloss",
        "use_label_encoder": False,
        "n_estimators": trial.suggest_int("n_estimators", 50, 150),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "tree_method": "hist",
        "random_state": 42,
        "verbosity": 0
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    map3_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_val)
        top_3 = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
        map3 = mapk(y_val.tolist(), top_3.tolist(), k=3)
        map3_scores.append(map3)

    return np.mean(map3_scores)

# Run optimization
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=15)  # Try 50+ for better results

# print("Best score (MAP@3):", study.best_value)
# print("Best hyperparameters:", study.best_params)



# CatBoost Hyperparameter Tuning
# #  Train-validation split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
train_pool = Pool(X_train, y_train)
valid_pool = Pool(X_valid, y_valid)

# Define objective for Optuna
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 500),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 0.1, 2.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "loss_function": "MultiClass",
        "eval_metric": "TotalF1",
        "verbose": 0,
        "random_seed": 42,
    }

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool)

    probs = model.predict_proba(X_valid)
    top_3 = np.argsort(probs, axis=1)[:, -3:][:, ::-1]

    # Convert true labels to one-hot for MAP@3
    y_true_binary = np.zeros_like(probs)
    for i, label in enumerate(y_valid):
        y_true_binary[i, label] = 1

    return label_ranking_average_precision_score(y_true_binary, probs)

# # Run Optuna tuning
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=20)  

# print("Best params:", study.best_params)
# print("Best MAP@3:", study.best_value)



## I already tried and tested multiple best params but got high accuracy on these ones
xgbmodel = XGBClassifier(
    n_estimators=223,
    max_depth=9,
    learning_rate=0.06545810287407865,
    subsample=0.6785548075439453,
    colsample_bytree=0.7058152625560293,
    objective="multi:softprob",
    num_class=len(np.unique(y)),
    eval_metric="mlogloss",
    use_label_encoder=False,
    tree_method="hist",
    random_state=42
)

xgbmodel.fit(X, y)




df_test=pd.get_dummies(df_test, columns=cat_cols, drop_first=True, dtype='int64')
test_ids= df_test['id']
df_test.drop(columns=['id'], inplace=True)
predictions = xgbmodel.predict_proba(df_test)


top_3 = np.argsort(predictions, axis=1)[:, -3:][:, ::-1]
top_3_labels=le.inverse_transform(top_3.flatten()).reshape(top_3.shape)



submission = pd.DataFrame({
    "id": test_ids,
    "Fertilizer Name": [" ".join(row) for row in top_3_labels]
})

submission.to_csv("submission_file.csv", index=False)




