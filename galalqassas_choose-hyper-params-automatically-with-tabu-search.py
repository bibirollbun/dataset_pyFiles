import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score,\
     recall_score, f1_score, ConfusionMatrixDisplay # this creates a confusion matrix, and draw it
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, scale
from xgboost import XGBClassifier
from matplotlib.pylab import rcParams


df = pd.read_csv("/kaggle/input/playground-series-s4e2/train.csv")
df


df.isna().sum()


df.duplicated().sum()


num_cols = df.select_dtypes(include=['number']).columns
cat_cols = df.select_dtypes(include='object').columns


df[num_cols].drop('id', axis=1).hist(figsize=(30, 20))


df[num_cols].drop('id', axis=1).boxplot(figsize=(5, 5))
len(num_cols)


fig, axs = plt.subplots(2, 4, figsize=(8, 12))
axs= axs.flatten()
# Plotting each boxplot in a separate subplot
for i, col in enumerate(num_cols[1:]):
    sns.boxplot(data=df, y=col, ax=axs[i])
    axs[i].set_title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()


fig, axs = plt.subplots(3, 3, figsize=(20, 15))
axs = axs.flatten()  

for i, col in enumerate(cat_cols):
    sns.countplot(data=df, x=col, ax=axs[i], palette=sns.color_palette("pastel"))
    axs[i].set_title(f'Count of {col}')
    if i==8:
        axs[i].tick_params(axis='x', rotation=20)


for j in range(i + 1, len(axs)):
    axs[j].set_visible(False)

plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 10))
data = df[num_cols].drop('id', axis=1)
sns.heatmap(data.corr(), cmap='coolwarm', annot=True)


cat_cols_ = ['Gender', "SMOKE", 'SCC', 'family_history_with_overweight', 'FAVC', 'CAEC', 'CALC']

for col in cat_cols_:
    df[col] = df[col].astype('category').cat.codes

df = pd.get_dummies(df, columns=['MTRANS'], dtype=int).drop(["MTRANS_Motorbike", "MTRANS_Walking", "MTRANS_Bike"], axis=1)

print(df.shape)
list(df)


X = df.drop(['id', 'NObeyesdad', 'SMOKE'], axis=1).copy()
y = df.NObeyesdad.copy()
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42, shuffle=True, stratify=y)


clf = HistGradientBoostingClassifier().fit(X_train, y_train)
print(f"Random Forest Train Accuracy: {clf.score(X_train, y_train)}")
print(f"Random Forest: {clf.score(X_val, y_val)}\n")


from itertools import product
from sklearn.metrics import balanced_accuracy_score

param_space = {
    "learning_rate": [0.03, 0.05, 0.08, 0.1],
    "max_depth": [None, 3, 5, 7, 8],
    "max_leaf_nodes": [15, 31, 63, 127]
}

def score_model(params):
    model = HistGradientBoostingClassifier(
        **params,
        random_state=42,
        categorical_features=None
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return balanced_accuracy_score(y_val, preds)

def save(params):
    return tuple(sorted(params.items()))
    


iters, tenure = 5, 2
def tabu_search(param_space, init):
    current = init.copy()
    best_params = current.copy()
    best_score = score_model(current)
    tabu = {save(current)}
    
    for _ in range(iters):
        neighbors = []
        for key in param_space:
            for val in param_space[key]:
                if val == current[key]:
                    continue
                candidate = current.copy()
                candidate[key] = val
                saved = save(candidate)
                if saved not in tabu:
                    neighbors.append(candidate)
        if not neighbors:
            break
        scored = [(score_model(nb), nb) for nb in neighbors]
        scored.sort(reverse=True, key=lambda x: x[0])
        current_score, current = scored[0]
        tabu.add(save(current))
        if len(tabu) > tenure:
            tabu = set(list(tabu)[-tenure:])
        if current_score > best_score:
            best_score = current_score
            best_params = current.copy()
    return best_params, best_score

init = {
    "learning_rate": 0.1,
    "max_depth": None,
    "max_leaf_nodes": 31
}

best_params, best_score = tabu_search(param_space, init)
print("Best params:", best_params)
print("Best balanced accuracy:", round(best_score, 4))


pca = PCA(n_components=2)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_val_pca = pca.transform(X_val)

clf_pca = HistGradientBoostingClassifier().fit(X_train_pca, y_train)

x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                     np.arange(y_min, y_max, 1))

Z_input = np.c_[xx.ravel(), yy.ravel()]

Z = clf_pca.predict(Z_input)
le = LabelEncoder()
le.fit(y_train)
Z = le.transform(Z)
Z = Z.reshape(xx.shape)

y_train_encoded = le.transform(y_train)


plt.figure(figsize=(20, 10))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_train_pca[:500, 0], X_train_pca[:500, 1], c=y_train_encoded[:500], s=20, edgecolor='k', alpha=0.7, cmap='coolwarm')
plt.title("Decision surface of a HistGradientBoostingClassifier using PCA")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.show()

