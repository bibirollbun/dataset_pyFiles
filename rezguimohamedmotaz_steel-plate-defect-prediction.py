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


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



df_train = pd.read_csv('/kaggle/input/playground-series-s4e3/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s4e3/test.csv')


pd.set_option('display.max_columns', None)
df_train.head()


print('Le nombre de valeurs nulles our train par colonne :\n', df_train.isnull().sum())


df_train.duplicated().sum()


target_columns = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

X_train = df_train.drop(columns=['id', 'Outside_Global_Index'] + target_columns)
y_train = df_train[target_columns]
X_test = df_test.drop(columns=['id','Outside_Global_Index'])


print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)


for i in X_train.columns:
    print(i,df_train[i].nunique())


x,y =df_train['TypeOfSteel_A300'].value_counts(),df_train['TypeOfSteel_A400'].value_counts()
print(x,y)


cat_col=[]
num_col=[]
for i in X_train.columns:
    if df_train[i].nunique()<=2:
        cat_col.append(i)
    else:
        num_col.append(i)


print(cat_col)
print(num_col)
print(target_columns)


df_train[cat_col].columns


import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Plot categorical features
for col in cat_col:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df_train, x=col)
    plt.title(f"Distribution of {col}")
    plt.show()


df_train[num_col].columns


plt.figure(figsize=(15, 20))
for i, col in enumerate(num_col):
    plt.subplot(7, 4, i+1)
    sns.histplot(data=df_train, x=col, kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
plt.show()


df_train[target_columns].columns


# Prepare data
target_stats = df_train[target_columns].apply(lambda x: x.value_counts()).T

# Plot
plt.figure(figsize=(12, 6))
width = 0.35
x = range(len(target_stats))

plt.bar(x, target_stats[0], width, label='0 (No Fault)', color='skyblue')
plt.bar(x, target_stats[1], width, bottom=target_stats[0], label='1 (Fault)', color='salmon')

plt.title("Distribution of 0s and 1s for Each Fault Type")
plt.ylabel("Count")
plt.xlabel("Fault Type")
plt.xticks(x, target_stats.index, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

corr = X_train.corr(numeric_only=True)

plt.figure(figsize=(18, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', center=0,
            annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



plt.figure(figsize=(15, 20))
for i, col in enumerate(num_col):
    plt.subplot(7, 4, i+1)
    sns.boxplot(x=X_train[col])
    plt.title(col)
plt.tight_layout()
plt.show()


# Calculate Q1, Q3, and IQR
Q1 = X_train[num_col].quantile(0.25)
Q3 = X_train[num_col].quantile(0.75)
IQR = Q3 - Q1

# Detect outliers
is_outlier = (X_train[num_col] < (Q1 - 1.5 * IQR)) | (X_train[num_col] > (Q3 + 1.5 * IQR))

# Count and percentage
outlier_counts = is_outlier.sum()
outlier_percentages = (outlier_counts / X_train.shape[0]) * 100

# Print results
print("Initial Outliers (IQR method):\n")
for col in num_col:
    print(f"{col}")
    print(f"Outliers count: {outlier_counts[col]}")
    print(f"Outliers percentage: {outlier_percentages[col]:.2f}%")
    print("-------------------------")



df_train.shape


# Recalculate Q1, Q3, IQR and apply capping
Q1 = X_train[num_col].quantile(0.25)
Q3 = X_train[num_col].quantile(0.75)
IQR = Q3 - Q1
low = Q1 - 1.5 * IQR
up = Q3 + 1.5 * IQR

#Capping : on remplace les outliers par les valeurs limites (borne basse ou haute).
# Apply capping to X_train and X_test
for col in num_col:
    X_train.loc[X_train[col] < low[col], col] = low[col]
    X_train.loc[X_train[col] > up[col], col] = up[col]
    X_test.loc[X_test[col] < low[col], col] = low[col]
    X_test.loc[X_test[col] > up[col], col] = up[col]

# Recalculate outliers after capping
is_outlier_after = (X_train[num_col] < low) | (X_train[num_col] > up)
outlier_counts_after = is_outlier_after.sum()
outlier_percentages_after = (outlier_counts_after / X_train.shape[0]) * 100

# Print results
print("Outliers After Capping:\n")
for col in num_col:
    print(f"{col}")
    print(f"Remaining outliers: {outlier_counts_after[col]}")
    print(f"Outliers percentage: {outlier_percentages_after[col]:.2f}%")
    print("-------------------------")



plt.figure(figsize=(15, 20))
for i, col in enumerate(num_col):
    plt.subplot(7, 4, i+1)
    sns.boxplot(x=X_train[col])
    plt.title(col)

plt.tight_layout()
plt.show()


X_train,y_train=df_train[X_train.columns],df_train[target_columns]
X_test=df_test


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)


#perturber les algorithmes

import pandas as pd

df_train = pd.read_csv('/kaggle/input/playground-series-s4e3/train.csv')

threshold = 0.5

skewed_columns = []

# Calculer l'asymétrie pour chaque colonne et vérifier si elle dépasse le seuil
for column in num_col:
    skewness = df_train[column].skew()
    if abs(skewness) > threshold:
        skewed_columns.append(column)


print("Colonnes skewed :", skewed_columns)



from sklearn.preprocessing import StandardScaler
import numpy as np


skewed_cols = [
    'Y_Minimum', 'Y_Maximum', 'Pixels_Areas', 'X_Perimeter', 'Y_Perimeter',
    'Sum_of_Luminosity', 'Maximum_of_Luminosity', 'Length_of_Conveyer',
    'Steel_Plate_Thickness', 'Edges_Index', 'Outside_X_Index',
    'Edges_Y_Index', 'LogOfAreas', 'Log_X_Index', 'Luminosity_Index'
]

#  trans log 0 ou negatif(log1p)
for col in skewed_cols:
    X_train[col] = np.log1p(X_train[col])
    X_test[col] = np.log1p(X_test[col])

# Standardisation val-moy/equart mm echl
scaler = StandardScaler()
X_train[num_col] = scaler.fit_transform(X_train[num_col])
X_test[num_col] = scaler.transform(X_test[num_col])




X_ft_train,X_ft_valid,y_ft_train,y_ft_valid=train_test_split(X_train,y_train,test_size=0.3,random_state=42)



import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                             GradientBoostingClassifier,
                             ExtraTreesClassifier,
                             AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Model evaluation
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score
from tqdm import tqdm

# Define defect categories
DEFECT_CATEGORIES = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains',
                    'Dirtiness', 'Bumps', 'Other_Faults']

# Initialize models
models = [
    DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    GradientBoostingClassifier(random_state=42),
    ExtraTreesClassifier(random_state=42, class_weight='balanced'),
    AdaBoostClassifier(random_state=42),
    LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    XGBClassifier(random_state=42, scale_pos_weight=1),  #manual
]

# Initialize result storage
results = []

# Model evaluation loop
for model in models:
    model_scores = []
    model_name = model.__class__.__name__

    for defect in DEFECT_CATEGORIES:
        if model_name == 'XGBClassifier':
            # Calculate imbalance ratio for this defect
              neg, pos = np.bincount(y_ft_train[defect])
              ratio = neg / max(pos, 1)  # avoid division by zero
              model.set_params(scale_pos_weight=ratio)
        # Train and predict
        model.fit(X_ft_train, y_ft_train[defect])
        y_proba = model.predict_proba(X_ft_valid)[:, 1]

        # Calculate and store ROC score
        roc_score = roc_auc_score(y_ft_valid[defect], y_proba) * 100
        model_scores.append(roc_score)

    # Store results
    mean_score = np.mean(model_scores)
    results.append({
        'Model': model.__class__.__name__,
        'Mean ROC Score': mean_score,
        'All Scores': model_scores
    })

    # Print immediate result
    print(f"{model.__class__.__name__:25} | Mean ROC: {mean_score:.2f}% | All Scores: {model_scores}")

import pandas as pd
results_df = pd.DataFrame(results)



model_names = [result['Model'] for result in results]
roc_auc_scores = [result['Mean ROC Score'] for result in results]

plt.figure(figsize=(12,6))
sns.barplot(x=model_names, y=roc_auc_scores, hue=model_names,
           palette="viridis", dodge=False)
plt.xticks(rotation=45, ha='right')
plt.title('Mean ROC AUC Scores by Model', pad=20)
plt.ylabel('Mean ROC AUC Score (%)')
plt.xlabel('')

for i, score in enumerate(roc_auc_scores):
    plt.text(i, score+1, f'{score:.1f}%', ha='center')
plt.tight_layout()
plt.show()


# Étendre les scores dans un tableau long
detailed_results = []

for result in results:
    model_name = result['Model']
    for defect_name, score in zip(DEFECT_CATEGORIES, result['All Scores']):
        detailed_results.append({
            'Model': model_name,
            'Defect': defect_name,
            'ROC_AUC': score
        })

detailed_df = pd.DataFrame(detailed_results)



import seaborn as sns
import matplotlib.pyplot as plt

pivot_df = detailed_df.pivot(index='Defect', columns='Model', values='ROC_AUC')

plt.figure(figsize=(12, 6))
sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("ROC AUC Scores par Défaut et Modèle")
plt.xlabel("Modèle")
plt.ylabel("Défaut")
plt.tight_layout()
plt.show()



from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

# Catégories de défauts
DEFECT_CATEGORIES = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

# Initialiser le modèle
final_model = GradientBoostingClassifier(random_state=42)

# Créer le DataFrame de soumission
submission_df = pd.DataFrame()
submission_df['id'] = range(len(X_test))

# Supprimer les colonnes non vues à l'entraînement
X_test_clean = X_test.drop(columns=['id', 'Outside_Global_Index'], errors='ignore')

# Boucle de prédiction
for defect in DEFECT_CATEGORIES:
    final_model.fit(X_ft_train, y_ft_train[defect])
    y_proba = final_model.predict_proba(X_test_clean)[:, 1]
    submission_df[defect] = y_proba

# Sauvegarder
submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())



submission_df.to_csv("/kaggle/working/predictions.csv", index=False)

