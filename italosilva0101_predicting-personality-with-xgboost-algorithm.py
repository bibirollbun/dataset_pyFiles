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


!pip uninstall -y imblearn scikit-learn


!pip install scikit-learn==1.2.2 imblearn


import sklearn
import imblearn

print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Imblearn version: {imblearn.__version__}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, auc, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV


df_train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_test= pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


style_color = 'Set2'


df_train.sample(5)


df_test.sample(5)


df_train.shape, df_test.shape


df_train.info()


df_test.info()


df_train.isna().sum()


df_test.isna().sum()


df_train.fillna({
    'Time_spent_Alone': df_train['Time_spent_Alone'].mean(),
    'Social_event_attendance': df_train['Social_event_attendance'].mean(),
    'Going_outside': df_train['Going_outside'].mean(),
    'Friends_circle_size': df_train['Friends_circle_size'].mean(),
    'Post_frequency': df_train['Post_frequency'].mean()
}, inplace=True)


df_test.fillna({
    'Time_spent_Alone': df_test['Time_spent_Alone'].mean(),
    'Social_event_attendance': df_test['Social_event_attendance'].mean(),
    'Going_outside': df_test['Going_outside'].mean(),
    'Friends_circle_size': df_test['Friends_circle_size'].mean(),
    'Post_frequency': df_test['Post_frequency'].mean()
}, inplace=True)


df_train.dropna(inplace=True)


df_test['Stage_fear'] = df_test['Stage_fear'].fillna(df_test['Stage_fear'].mode()[0])
df_test['Drained_after_socializing'] = df_test['Drained_after_socializing'].fillna(df_test['Drained_after_socializing'].mode()[0])


df_test.shape


df_test.isna().sum()


metadados = df_train.dtypes


object_var = metadados[metadados == 'object'].index
numeric_var = metadados[metadados != 'object'].drop('id').index


df_train_corr = df_train[numeric_var].corr()


sns.heatmap(df_train_corr, annot=True, cmap=style_color)


gdf_train_rouped_means = df_train.groupby('Personality')[numeric_var].mean().round(2)


plt.figure(figsize=(12, 6))
sns.heatmap(gdf_train_rouped_means, annot=True, cmap=style_color, fmt='.2f')
plt.title('Média das Variáveis Numéricas por Tipo de Personalidade')
plt.ylabel('Personalidade')
plt.xlabel('Variáveis Numéricas')
plt.tight_layout()
plt.show()


for i in numeric_var:
  sns.histplot(df_train[i], kde=True, color='skyblue')
  plt.title(f'Histograma com {i}')
  plt.show()


df_train['Personality'].value_counts()


df_train['Personality'] = df_train['Personality'].replace({
    'Extrovert': 1,
    'Introvert': 0
})


df_train.sample()


def descritiva(df_, var, vresp='Personality', max_classes=5):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df_copy = df_.copy()

    if df_copy[var].nunique() > max_classes:
        df_copy[var] = pd.qcut(df_copy[var], max_classes, duplicates='drop')

    fig, ax1 = plt.subplots(figsize=(10, 6))

    sns.pointplot(data=df_copy, y=vresp, x=var, ax=ax1, color='black')

    ax1.set_ylabel('Personalidade')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Introvertido', 'Extrovertido'])

    ax2 = ax1.twinx()
    sns.countplot(data=df_copy, x=var, palette='viridis', alpha=0.5, ax=ax2)
    ax2.set_ylabel('Frequência', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    ax1.set_zorder(2)
    ax1.patch.set_visible(False)

    plt.title(f'Análise Descritiva: {var} vs Personality')
    plt.tight_layout()
    plt.show()


columns = df_train.columns

for col in df_train.select_dtypes(include=['int64', 'float64']).columns:
    if col != 'Personality':
        descritiva(df_train, col)


df_train['Personality'].value_counts()


object_var = object_var.drop('Personality')


object_var


df_train_copy = df_train.copy()


df_train = pd.get_dummies(df_train, columns=object_var, drop_first=True)
df_test = pd.get_dummies(df_test, columns=object_var, drop_first=True)


df_train.sample(3)


df_test.sample(3)


X = df_train.drop(columns=(['Personality', 'id']))
y = df_train['Personality']


X.shape, y.shape


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)


y_res.value_counts()


xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)


param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
    'scale_pos_weight': [1, (y_train == 0).sum() / (y_train == 1).sum()]
}


grid = GridSearchCV(
    estimator=xgb,
    param_grid = param_grid,
    scoring='recall',
    cv=5,
    verbose=1,
    n_jobs=-1
)


grid.fit(X_res, y_res)


grid.best_params_


best_model = grid.best_estimator_


train_predict = best_model.predict(X_res)
train_predict_proba = best_model.predict_proba(X_res)


acc_tree_train = accuracy_score(y_res, train_predict)
sens_tree_train = recall_score(y_res, train_predict, pos_label=1)
espec_tree_train = recall_score(y_res, train_predict, pos_label=0)
prec_tree_train = precision_score(y_res, train_predict)

print("Avaliação da Árvore (Base de Treino)")
print(f"Acurácia: {acc_tree_train:.1%}")
print(f"Sensibilidade: {sens_tree_train:.1%}")
print(f"Especificidade: {espec_tree_train:.1%}")
print(f"Precision: {prec_tree_train:.1%}")


teste_predict = best_model.predict(X_test)
test_predict_proba = best_model.predict_proba(X_test)


acc_tree_test = accuracy_score(y_test, teste_predict)
sens_tree_test = recall_score(y_test, teste_predict, pos_label=1)
espec_tree_test = recall_score(y_test, teste_predict, pos_label=0)
prec_tree_test = precision_score(y_test, teste_predict)

print("Avaliação da Árvore (Base de Treino)")
print(f"Acurácia: {acc_tree_test:.1%}")
print(f"Sensibilidade: {sens_tree_test:.1%}")
print(f"Especificidade: {espec_tree_test:.1%}")
print(f"Precision: {prec_tree_test:.1%}")


cm = confusion_matrix(y_test, teste_predict)


plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Extroverted', 'Introverted'], yticklabels=['Extroverted', 'Introverted'])
plt.title("Matriz de Confusão - Teste")
plt.xlabel("Previsão")
plt.ylabel("Valor Real")
plt.show()


fpr_tree, tpr_tree, thresholds_tree = roc_curve(y_test, test_predict_proba[:,1])


roc_auc_tree = auc(fpr_tree, tpr_tree)


plt.figure(figsize=(6,3), dpi=200)
plt.plot(fpr_tree, tpr_tree, color='blue', linewidth=4)
plt.plot(fpr_tree, tpr_tree, color='gray', linestyle='dashed')
plt.title('AUC-ROC RF: %g' % round(roc_auc_tree, 3), fontsize=22)
plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensibilidade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show()


df_final = pd.DataFrame({'id': df_test['id'],
                         'Personality': best_model.predict(df_test.drop('id', axis=1))})


df_final_copy = df_final.copy()


df_final['Personality'] = df_final['Personality'].replace({
    1: 'Extrovert',
    0: 'Introvert'
})


df_final.to_csv('sample_submission_final.csv', index=False)

