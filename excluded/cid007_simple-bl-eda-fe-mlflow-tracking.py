!pip install google-api-core>=2.10.2,<3.0.0dev

!pip install mlflow


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


sns.set_theme(style="whitegrid")
sns.set_palette('pastel')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


# Function to load data
base_path="/kaggle/input/playground-series-s5e7/"

"""Load and return train, test and sample submission dataframes"""
train = pd.read_csv(f"{base_path}train.csv").set_index("id")
test = pd.read_csv(f"{base_path}test.csv").set_index("id")
submission = pd.read_csv(f"{base_path}sample_submission.csv").set_index("id")


# Columns for submission data
TARGET = 'Personality'
ID_COL = 'id'

train["Personality"] = train["Personality"].map({"Extrovert": 1, "Introvert": 0})
train.head()


train.shape


train.info()


train.describe()


train.isnull().sum()


# Identify feature columns
numeric_cols = train.drop(columns=[ "Personality"]).select_dtypes(include=np.number).columns.tolist()
categorical_cols = train.drop(columns=[ "Personality"]).select_dtypes(include=['object', 'category'])\
                      .columns.tolist()

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Numeric cols ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical cols ({len(categorical_cols)}): {categorical_cols}")



class EDA:
    def __init__(self, df):
        self.df = df

    def plot_numeric(self, cols):
        for c in cols:
            plt.figure(figsize=(6,4))
            sns.histplot(self.df[c].dropna(), kde=True)
            plt.title(f'Distribution of {c}')
            plt.show()

    def plot_categorical(self, cols):
        for c in cols:
            plt.figure(figsize=(6,4))
            sns.countplot(x=self.df[c])
            plt.title(f'Countplot of {c}')
            plt.xticks(rotation=45)
            plt.show()



eda = EDA(train)
eda.plot_numeric(numeric_cols)
eda.plot_categorical(categorical_cols)



numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler',  StandardScaler())
])
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
])
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols),
])



train.head()


X = train.drop(columns=[ "Personality"])
y = train[TARGET]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc   = preprocessor.transform(X_val)



mlflow.set_experiment('personality_classification')

models = {
    'XGBoost':    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'CatBoost':   CatBoostClassifier(verbose=0, random_state=42),
    'LightGBM':   lgb.LGBMClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'Logistic':   LogisticRegression(max_iter=1000, random_state=42),
}

trained_models = {}
metrics = {}
for name, mdl in models.items():
    with mlflow.start_run(run_name=name):
        mdl.fit(X_train_proc, y_train)
        train_acc = mdl.score(X_train_proc, y_train)
        val_acc   = mdl.score(X_val_proc, y_val)
        mlflow.log_params(mdl.get_params())
        mlflow.log_metric('train_accuracy', train_acc)
        mlflow.log_metric('val_accuracy', val_acc)
        mlflow.sklearn.log_model(mdl, name)
        trained_models[name] = mdl
        metrics[name] = {'train_accuracy': train_acc, 'val_accuracy': val_acc}

# Ensemble voting classifier
ensemble = VotingClassifier(
    estimators=[(n, m) for n, m in trained_models.items()],
    voting='soft'
)
with mlflow.start_run(run_name='Ensemble'):
    ensemble.fit(X_train_proc, y_train)
    train_acc = ensemble.score(X_train_proc, y_train)
    val_acc   = ensemble.score(X_val_proc, y_val)
    mlflow.log_metric('train_accuracy', train_acc)
    mlflow.log_metric('val_accuracy', val_acc)
    mlflow.sklearn.log_model(ensemble, 'Ensemble')
    metrics['Ensemble'] = {'train_accuracy': train_acc, 'val_accuracy': val_acc}

print("Metrics:", metrics)



import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
exp = client.get_experiment_by_name('personality_classification')
runs = client.search_runs(exp.experiment_id)


rows = []
for run in runs:
    rows.append({
        'run_name': run.info.run_name,
        'train_acc': run.data.metrics.get('train_accuracy'),
        'val_acc':   run.data.metrics.get('val_accuracy')
    })
df_runs = pd.DataFrame(rows)

# Plot train vs val accuracy
plt.figure(figsize=(8,5))
sns.barplot(x='run_name', y='train_acc', data=df_runs, alpha=0.7, label='train')
sns.barplot(x='run_name', y='val_acc', data=df_runs, alpha=0.7, label='val')
plt.xticks(rotation=45)
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train vs Validation Accuracy per Model')
plt.show()



test.head()


X_full = preprocessor.fit_transform(X)  
ensemble.fit(X_full, y)                  

X_test = preprocessor.transform(test)
y_pred = ensemble.predict(X_test)




submission = pd.DataFrame({ID_COL: test.reset_index()[ID_COL], TARGET: y_pred})
submission["Personality"] = submission["Personality"].map({1:"Extrovert", 0:"Introvert"})
submission.to_csv('submission.csv', index=False)
print("Saved submission.csv")



submission.head()


submission.Personality.value_counts()

