


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from google.colab import drive
# drive.mount('/content/drive')


path = "/content/drive/MyDrive/kaggle_1/data/"
train_data = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


X = train_data.drop(['y'], axis=1)
y = train_data['y']

print(X.shape)
print(y.shape)


train_data.head()


train_data.info()


train_data.isna().sum()


cat_cols = X.select_dtypes(include='object').columns
num_cols = X.select_dtypes(exclude='object').columns

for col in cat_cols:
    print(f"{col}: {X[col].unique()}")


X.replace("unknown", np.nan, inplace=True)

print((X[cat_cols].isna().sum()/X.size)*100)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# On force à garder les noms
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

X_train.head()


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Categorical steps for pipeline
cat_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ('scaler', StandardScaler())
])

# --- Pipeline pour colonnes numériques ---
num_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# --- Preprocessor complet ---
preprocessor = ColumnTransformer(transformers=[
    ('cat', cat_pipe, cat_cols),
    ('num', num_pipe, num_cols)
], remainder='drop')





from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


models = {
    'XGBoost': XGBClassifier(tree_method="hist", device="cuda", use_label_encoder=False, eval_metric="logloss"),
    'LightGBM': LGBMClassifier(),
}


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Exemple d'hyperparamètres à tester pour XGBoost

params = {
    'XGBoost': {
        'classifier__learning_rate': uniform(0.01, 0.2),  # valeurs continues entre 0.01 et 0.21
        'classifier__max_depth': randint(3, 10),          # entiers de 3 à 9
        'classifier__n_estimators': randint(50, 300),
        'classifier__subsample': uniform(0.6, 0.4),
        'classifier__colsample_bytree': uniform(0.6, 0.4)
    },
    'LightGBM': {
        'classifier__learning_rate': uniform(0.01, 0.2),
        'classifier__num_leaves': randint(20, 150),
        'classifier__n_estimators': randint(50, 300),
        'classifier__subsample': uniform(0.6, 0.4),
        'classifier__colsample_bytree': uniform(0.6, 0.4)
    }
}


n_iter_search = 20

best_models = {}

for model_name, model in models.items():
    pipeline_final = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', model)
    ])
    random_search = RandomizedSearchCV(
        pipeline_final,
        param_distributions=params[model_name],
        n_iter=n_iter_search,
        cv=3,
        scoring='roc_auc',
        verbose=2,
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train, y_train)
    best_models[model_name] = random_search.best_estimator_
    print(f"Best {model_name} params:", random_search.best_params_)
    print(f"Best {model_name} score:", random_search.best_score_)



from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

XGBModel = best_models['XGBoost']
LGBMModel = best_models['LightGBM']

estimators = [
    ('XGBosst', XGBModel),
    ('LightGBM', LGBMModel)
]

meta_model = RandomForestClassifier()
meta_model1 = LogisticRegression()

# Stacking
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model,
    cv=5,
    stack_method='predict_proba',  # on utilise les probabilités comme entrée
    n_jobs=-1
)

stacking_clf1 = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model1,
    cv=5,
    stack_method='predict_proba',  # on utilise les probabilités comme entrée
    n_jobs=-1
)



y_train


from sklearn.model_selection import cross_val_score


results = []

models = {
    'XGBoost': XGBModel,
    'LightGBM': LGBMModel,
    'Stacking_RF': stacking_clf,
    'Stacking_LR': stacking_clf1,
}

print(type(X_train))
for model_name, model in models.items():
    pipeline_final = Pipeline([
        ('classifier', model)
    ])
    cv_results = cross_val_score(pipeline_final, X_train, y_train, cv=3, scoring='roc_auc', error_score='raise')
    results.append(cv_results)

print(f"Scores: {results}")
print(f"Mean: {np.mean(results)}")
print(f"Std: {np.std(results)}")





plt.figure(figsize=(10, 10))
plt.boxplot(results, labels=models.keys())
plt.xlabel('Modèle')
plt.ylabel('Score ROC AUC')
plt.title('Comparaison des modèles')
plt.savefig('comparaison_modeles.png')
plt.show()


stacking_clf1.fit(X, y)
y_pred_proba = stacking_clf1.predict_proba(test_data)[:, 1]


submission = pd.DataFrame({'id': test_data['id'], 'y': y_pred_proba})
submission.to_csv('submission.csv', index=False)













