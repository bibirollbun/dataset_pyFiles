import numpy as np 
import pandas as pd
pd.set_option('display.max_columns',220)
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import time

import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv", sep=",")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv", sep=",")


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid, cross_validate
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression


# select the features for each of the `ColumnTransformer`
# in our case, all are numericals but one may 1hot encode `day` 

num_cols: list[str] = ['pressure','maxtemp','temparature','mintemp','dewpoint','humidity','cloud','sunshine','winddirection','windspeed']
num_cols_untouched: list[str] = ['day']
# cat_cols: list[str] = ['day'] TO DO
target: str = "rainfall"

X = df_train[num_cols + num_cols_untouched]
y = df_train[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    stratify=y,
    shuffle=True,
    random_state=1234)

print(f"train: {X_train.shape}, test: {X_test.shape}")


# Create a base pipeline
# numerical features will be median imputed, then standardized
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), 
           ("scaler", StandardScaler())]
)

# in case of categorical feature(s), the imputation is the most_frequent
categorical_onehot_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# here all features but `day` will go through the numerical pipeline
# only `day` is not touched by the pipeline. 
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
    ],
    remainder='passthrough'
)

# baseline algorithms
rf = RandomForestClassifier(random_state=1234)
lr = LogisticRegression()

# the votingclassifier will combine the prediction on soft voting:
# soft’, predicts the class label based on the argmax of the sums of the predicted probabilities, 
# which is recommended for an ensemble of well-calibrated classifiers.
ens = VotingClassifier(estimators=[("rf", rf), ("logisticregression", lr)], voting='soft')

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor), 
    ('selectkbest', SelectKBest()),
    ('voting_ensemble', ens)
])


pipeline


# to get access to the individual parameters of the pipeline (classfiers, selectKBest), one can type:
# pipeline.get_params()


# the parameters grid below include the numbers of features to be selected, as well as individuals parameters for RF and LR
# note the naming convention:
# voting_ensemble__logisticregression__C: voting --> classifier --> parameter
param_grid = {
'selectkbest__k': [2, 6, 10],
'voting_ensemble__logisticregression__C':[1e-1, 1],
'voting_ensemble__logisticregression__penalty': ['l1', 'l2', 'elasticnet'],
'voting_ensemble__logisticregression__l1_ratio': np.arange(0, 1.1, 0.2),
'voting_ensemble__rf__max_depth': [2, 4, 6],
'voting_ensemble__rf__n_estimators':[50],
'voting_ensemble__rf__max_features': ['sqrt','log2'],
'voting_ensemble__rf__criterion':['gini', 'entropy', 'log_loss'],
}


# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', verbose=1)
start = time.time()
grid_search.fit(X_train, y_train)
end = time.time()


# Print the best hyperparameters and score
print(f"Best hyperparameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
print(f"timing: {end-start}")

# Evaluate the best model on the train set
best_model = grid_search.best_estimator_
train_score = best_model.score(X_train, y_train)
print("Train score:", train_score)

# Evaluate the best model on the test set
test_score = best_model.score(X_test, y_test)
print("Test score:", test_score)


from sklearn.metrics import roc_curve, RocCurveDisplay, auc
fig, ax = plt.subplots(1,1, figsize=(4,4))
RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=ax, linewidth=1)
plt.show()


preds = best_model.predict(df_test)
preds_probs = best_model.predict_proba(df_test)[:,1]
#sample = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv", sep=",")
#sample["rainfall"] = preds_probs
#sample.to_csv("rain_2025_competition_sample_solution.csv", index=False)




