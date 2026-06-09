
import pandas as pd
import numpy as np 

#Eda
import matplotlib.pyplot as plt
import seaborn as sns

# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif

# Model
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from skopt.space import Real, Integer, Categorical


import warnings 
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
test_subID = test['id']


test.head(8)


df.head(8)


df.describe(include='all')


df.info()


df = df.drop('id',axis=1)
test = test.drop('id',axis=1)


# EDA values
x_values = df.select_dtypes(include=['number'])



corr_matrix = x_values.corr()
plt.figure(figsize=(20, 12))
sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt='.2f')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()



plt.figure(figsize=(18, 8))
sns.countplot(data=df, x='Personality')
plt.title('Personality Distribution')
plt.show()



fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

for ax, x_value in zip(axis.flat, x_values):
    sns.kdeplot(data=df, x=x_value, hue='Personality', fill=True, common_norm=False, alpha=0.5, ax=ax)
    ax.set_title(f'{x_value.capitalize()}')
plt.tight_layout()
plt.show()



fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

for ax, x_value in zip(axis.flat, x_values):
    sns.histplot(data=df, x=x_value, hue="Personality", kde=True, ax=ax, bins=20, alpha=0.6)
    ax.set_title(f'Histogram of {x_value.capitalize()} by Fuel Type')
plt.tight_layout()
plt.show()


# Separation of features and target
X = df.drop('Personality',axis=1)
y = df['Personality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.fit_transform(y_test)

numerical_columns = list(X_train.select_dtypes(include=['float64', 'int64']).columns)
categorical_columns = list(X_train.select_dtypes(include=['object', 'category']).columns)


#OutlierClipper 
class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, quantile=0.01):
        self.quantile = quantile

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns
            X_df = X
        else:
            self.columns_ = [f'x{i}' for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=self.columns_)
        self.lower_ = X_df.quantile(self.quantile)
        self.upper_ = X_df.quantile(1 - self.quantile)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.columns_)
        X_clipped = X_df.clip(self.lower_, self.upper_, axis=1)
        return X_clipped.values 


search_space = {
    'feature_selection__k': Integer(5, X_train.shape[1]),  # busca o melhor nÃºmero de features
    'classifier__learning_rate': Real(0.001, 0.3, prior='log-uniform'),
    'classifier__max_leaf_nodes': Integer(10, 100),
    'classifier__max_depth': Integer(3, 30),
    'classifier__l2_regularization': Real(0.0, 1.0)
}


#Pipeline
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('outlier_clipper', OutlierClipper(quantile=0.01)),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, numerical_columns),
    ('cat', cat_pipeline, categorical_columns)
])


#Optuna 
def objective_xgb(trial):
    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('feature_selection', SelectKBest(score_func=f_classif,k = trial.suggest_categorical('k', ['all', 5, 7, 9]) )),
        ('classifier', XGBClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 200),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1),
            subsample=trial.suggest_float('subsample', 0.8, 1.0),
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        ))
    ])
    score = cross_val_score(pipeline, X_train, y_encoded, cv=5, scoring='accuracy', n_jobs=-1)
    return score.mean()


XGBoost = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.08301980801070748,
        subsample=0.8534115834322589,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ))
])


XGBoost.fit(X_train, y_train_encoded)
y_pred = XGBoost.predict(X_test)
print(classification_report(y_test_encoded, y_pred))



cm = confusion_matrix(y_test_encoded, y_pred) 

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()


y_pred_subm = XGBoost.predict(test)

label_mapping = {0: 'Extrovert', 1: 'Introvert'}
y_pred_labels = pd.Series(y_pred_subm).map(label_mapping)
submission = pd.DataFrame({
    'id': test_subID,
    'Personality': y_pred_labels
})

submission.to_csv('submission4.csv', index=False)


submission

