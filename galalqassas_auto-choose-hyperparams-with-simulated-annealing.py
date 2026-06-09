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


import numpy as np
from tabulate import tabulate
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)

param_space = {
    'learning_rate':     np.round(np.linspace(0.001, 0.3, 15), 3),
    'l2_regularization': np.round(np.linspace(0.0, 1.0, 21), 2),
    'max_depth':         np.arange(3, 13),
    'max_leaf_nodes':    np.arange(20, 130, 10),
}

def random_choice():
    return {k: np.random.choice(v) for k, v in param_space.items()}

def neighbor(params):
    k = np.random.choice(list(param_space))
    options = [v for v in param_space[k] if v != params[k]]
    params = params.copy()
    params[k] = np.random.choice(options) if options else params[k]
    return params

def score(params):
    model = HistGradientBoostingClassifier(random_state=42, **params)
    model.fit(X_train, y_train)
    return accuracy_score(y_val, model.predict(X_val))

def fmt(params):
    return ", ".join(f"{k}={params[k]}" for k in params)

current = random_choice()
current_score = score(current)
best, best_score = current.copy(), current_score
temp, final_temp, alpha = 1.0, 0.01, 0.5
rows = []

while temp > final_temp:
    candidate = neighbor(current)
    candidate_score = score(candidate)
    gain = candidate_score - current_score

    if gain > 0 or np.exp(gain / temp) > np.random.rand():
        current, current_score = candidate, candidate_score

    if current_score > best_score:
        best, best_score = current.copy(), current_score

    rows.append([
        round(temp, 4),
        fmt(current),
        round(current_score, 4),
    ])
    temp *= alpha

print(tabulate(rows, headers=["Temp", "Current Params", "Current Acc"]))
print("Best settings:", best)
print(f"Validation accuracy: {best_score:.4f}")


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




