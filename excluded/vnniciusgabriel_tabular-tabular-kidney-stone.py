import numpy as np
import pandas as pd
import seaborn as sns
from functools import wraps
from time import perf_counter
from matplotlib import pyplot as plt
from scipy.stats import skew, kurtosis, pearsonr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

__import__('warnings').filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv('/kaggle/input/playground-series-s3e12/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s3e12/test.csv')


df_train.head()


df_train.info()


f, ax = plt.subplots(figsize=(8, 4))
ax = sns.countplot(y='target', data=df_train, palette='viridis')
plt.show()


features = ['gravity', 'ph', 'osmo', 'cond', 'urea', 'calc']

for feature in features:
    plt.figure(figsize=(10, 4))
    
    sns.histplot(df_train[feature], color='blue', kde=True, stat="density", label='Treino', alpha=0.6)
    
    sns.histplot(df_test[feature], color='orange', kde=True, stat="density", label='Teste', alpha=0.6)
    
    plt.title(f'feature distribution: {feature}')
    plt.xlabel(feature)
    plt.legend()
    plt.show()


fig, axs = plt.subplots(nrows=len(features), figsize=(10, 5 * len(features)))
for i, var in enumerate(features):
    sns.histplot(df_train[var], kde=True, ax=axs[i])
    axs[i].set_title(f'distribution of {var}')
plt.tight_layout()
plt.show()


sns.pairplot(df_train, vars=features, hue='target', palette='viridis', diag_kind='kde')
plt.suptitle("scatterplot matrix das features por target", y=1.02)
plt.show()


def treat_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for var in features:
    
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.boxplot(y=df[var], color='skyblue')
        plt.title(f'before treatment: {var}')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        Q1 = df[var].quantile(0.25)
        Q3 = df[var].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    
        outliers = df[(df[var] < lower) | (df[var] > upper)]
        print(f"{var}: {len(outliers)} outliers detected")
    
        df[var] = np.where(df[var] < lower, lower, np.where(df[var] > upper, upper, df[var]))
    
        plt.subplot(1, 2, 2)
        sns.boxplot(y=df[var], color='lightgreen')
        plt.title(f"after treatment: {var}")
        plt.grid(True, linestyle='--', alpha=0.7)
    
        plt.tight_layout()
        plt.show()

    return df


df_train_without_outliers = treat_outliers(df_train)


df_test_without_outliers = treat_outliers(df_test)


pipeline = Pipeline(
    steps=[
        ('scaler', MinMaxScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False))
    ]
)

X, y = pipeline.fit_transform(df_train_without_outliers[features]), df_train['target']


feature_names = pipeline.get_feature_names_out()

selector = SelectKBest(chi2, k=8)
selector.fit(X, y)

features_scores = pd.DataFrame({
    'Feature': feature_names,
    'Chi2_Score': selector.scores_,
    'P_value': selector.pvalues_
})

selected_features = feature_names[selector.get_support()]
selected_features_scores = features_scores.loc[selector.get_support()].sort_values('Chi2_Score', ascending=False)

print("top features com scores chi^2")
display(selected_features_scores.style.background_gradient(cmap='Blues', subset=['Chi2_Score']))

plt.figure(figsize=(12, 6))
sns.barplot(x='Chi2_Score', y='Feature', data=selected_features_scores, palette='viridis')
plt.title('Feature importance using Chi² test')
plt.xlabel('Score Chi²')
plt.ylabel('Features')
plt.tight_layout()
plt.show()


X = pd.DataFrame(X, columns=feature_names)[selected_features]
corr_matrix = X.corrwith(pd.Series(y))

plt.figure(figsize=(15, 8))
sns.heatmap(corr_matrix.to_frame('correlation'), annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('feature correlations with the target')
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"train size: {len(X_train)}, test_size: {len(X_test)}")


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {perf_counter() - start:.2f} seconds")
        return result
    return wrapper


@timeit
def train_model_and_evaluate(model):

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    cm = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
    cm.plot()
    plt.grid(False)
    plt.show()


train_model_and_evaluate(LogisticRegression(class_weight='balanced'))


param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

logreg = LogisticRegression(class_weight='balanced', max_iter=1000)

grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid, scoring='roc_auc', cv=5)
grid_search.fit(X_train, y_train)

print("best parameters:", grid_search.best_params_)
print("best ROC AUC:", round(grid_search.best_score_, 4))

y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))

cm = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
cm.plot()
plt.grid(False)
plt.show()


X = pipeline.transform(df_test_without_outliers[features])
X = pd.DataFrame(X, columns=feature_names)[selected_features]

y_pred = grid_search.predict_proba(X)[:, 1]

df_test_without_outliers['target'] = y_pred
df_test_without_outliers[['id', 'target']].to_csv('/kaggle/working/submission.csv', index=False)

