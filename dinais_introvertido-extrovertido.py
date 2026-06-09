

import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
train_df


test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
test_df


train_df.dtypes


train_df.info()


train_df.describe()


train_df.isna().sum()


train_df = train_df.fillna(train_df.median(numeric_only=True))


train_df.isna().sum()


target = "Personality"
id_col = "id"

X = train_df.drop(columns=[target, id_col])
y = train_df[target]

# Determinar categóricas y numéricas
cat_cols = [col for col in X.columns if X[col].dtype == "object"]
num_cols = [col for col in X.columns if col not in cat_cols]

cat_cols, num_cols



import matplotlib.pyplot as plt
import seaborn as sns

train_df[num_cols].hist(bins=20, figsize=(12, 8))
plt.tight_layout()
plt.show()



for c in cat_cols:
    print(c, train_df[c].value_counts())
    sns.countplot(data=train_df, x=c, hue=target)
    plt.show()



from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaled = StandardScaler().fit_transform(X[num_cols])

pca = PCA(n_components=2, random_state=42)
pca_components = pca.fit_transform(scaled)

pca_df = pd.DataFrame({
    "PC1": pca_components[:,0],
    "PC2": pca_components[:,1],
    "Personality": y
})

sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Personality", alpha=0.6)
plt.title("PCA - 2 Componentes")
plt.show()

pca.explained_variance_ratio_



from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_comp = tsne.fit_transform(scaled)

tsne_df = pd.DataFrame({
    "TSNE1": tsne_comp[:,0],
    "TSNE2": tsne_comp[:,1],
    "Personality": y
})

sns.scatterplot(data=tsne_df, x="TSNE1", y="TSNE2", hue="Personality", alpha=0.6)
plt.title("t-SNE")
plt.show()



from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_cols),
    ("cat", categorical_pipeline, cat_cols)
])



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

baseline_model = Pipeline([
    ("preprocess", preprocessor),
    ("clf", LogisticRegression(max_iter=500))
])

baseline_model.fit(X_train, y_train)
preds = baseline_model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, preds))
print(classification_report(y_val, preds))



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

rf = RandomForestClassifier(random_state=42)

pipeline_rf = Pipeline([
    ("preprocess", preprocessor),
    ("clf", rf)
])

param_grid = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [None, 8, 15],
    "clf__min_samples_split": [2, 5],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipeline_rf,
    param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid.fit(X, y)
print("Mejor accuracy:", grid.best_score_)
print("Mejores parámetros:", grid.best_params_)

best_model = grid.best_estimator_



from sklearn.metrics import confusion_matrix, roc_auc_score, RocCurveDisplay

y_pred = best_model.predict(X_val)
y_proba = best_model.predict_proba(X_val)[:,1]

print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Matriz de Confusión")
plt.show()




best_model.fit(X, y)

test_processed = test_df.drop(columns=[id_col])
test_preds = best_model.predict(test_processed)

submission = pd.DataFrame({
    "id": test_df["id"],
    "Personality": test_preds
})

submission.to_csv("submission.csv", index=False)

submission.head()





