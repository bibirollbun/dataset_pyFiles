import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme()
sns.set_palette('Set2')


data_url = 'https://raw.githubusercontent.com/caalvaro/machine-learning/refs/heads/main/Classification%20-%20Multi-Class%20Prediction%20of%20Obesity%20Risk/data/train.csv'

raw_data = pd.read_csv(data_url, index_col='id')
raw_data.head()


raw_data.shape


raw_data.isna().sum()


raw_data.describe()


fig, axs = plt.subplots(2, 4, figsize=(15, 10))

sns.histplot(data=raw_data, x='Age', ax=axs[0, 0], bins=20)
sns.histplot(data=raw_data, x='Height', ax=axs[0, 1], bins=20)
sns.histplot(data=raw_data, x='Weight', ax=axs[0, 2], bins=20)
sns.histplot(data=raw_data, x='FCVC', ax=axs[0, 3], bins=20)
sns.histplot(data=raw_data, x='NCP', ax=axs[1, 0], bins=8)
sns.histplot(data=raw_data, x='CH2O', ax=axs[1, 1], bins=10)
sns.histplot(data=raw_data, x='FAF', ax=axs[1, 2], bins=10)
sns.histplot(data=raw_data, x='TUE', ax=axs[1, 3], bins=10)

plt.tight_layout()
plt.show()


fig, axs = plt.subplots(3, 3, figsize=(20, 15))

# Gráfico Gender
ax = sns.countplot(data=raw_data, x='Gender', ax=axs[0, 0])
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# Gráfico family_history_with_overweight
ax = sns.countplot(data=raw_data, x='family_history_with_overweight', ax=axs[0, 1])
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# Gráfico FAVC
ax = sns.countplot(data=raw_data, x='FAVC', ax=axs[0, 2])
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# Gráfico CAEC
ax = sns.countplot(data=raw_data, x='CAEC', ax=axs[1, 0])
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# Gráfico SMOKE
ax = sns.countplot(data=raw_data, x='SMOKE', ax=axs[1, 1])
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# Gráfico SCC
ax = sns.countplot(data=raw_data, x='SCC', ax=axs[1, 2])
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# Gráfico CALC
ax = sns.countplot(data=raw_data, x='CALC', ax=axs[2, 0])
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# Gráfico MTRANS
ax = sns.countplot(data=raw_data, x='MTRANS', ax=axs[2, 1])
ax.tick_params(axis='x', rotation=45) # Mantendo a rotação se necessário
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# Gráfico NObeyesdad
ax = sns.countplot(data=raw_data, x='NObeyesdad', ax=axs[2, 2])
ax.tick_params(axis='x', rotation=45) # Mantendo a rotação se necessário
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')


plt.tight_layout()
plt.show()


features = pd.get_dummies(raw_data.drop(columns='NObeyesdad'), dtype=int)


labels_map = {'Insufficient_Weight': 0,
              'Normal_Weight': 1,
              'Overweight_Level_I': 2,
              'Overweight_Level_II': 3,
              'Obesity_Type_I': 4,
              'Obesity_Type_II': 5,
              'Obesity_Type_III': 6}

target = raw_data['NObeyesdad'].map(labels_map)


features.head()


from sklearn.model_selection import train_test_split

n_splits = 5 # parâmetro para a valdação cruzada

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)


# função auxiliar para os resultados da validação cruzada
def print_results(results):
  mean = results['test_score'].mean()*100
  std = results['test_score'].std()*100
  lim_inf = mean - 2 * std
  lim_sup = mean + 2 * std
  print(f"Accuracy com cross validation de 5 splits: entre [{lim_inf:.2f}, {lim_sup:.2f}]. \n\
        Média = {mean:.2f}% \n\
        Desvio Padrão = {std:.2f}%")


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate

model_nb = GaussianNB()

results_nb = cross_validate(model_nb, features, target, cv = n_splits, return_train_score=False, n_jobs=-1)

print_results(results_nb)


from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

results_rf = cross_validate(model_rf, features, target, cv = n_splits, return_train_score=False, n_jobs=-1)

print_results(results_rf)


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

model_svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))

results_svm = cross_validate(model_svm, features, target, cv = n_splits, return_train_score=False, n_jobs=-1)

print_results(results_svm)


import xgboost as xgb

model_xgb = xgb.XGBClassifier(objective='multi:softmax', num_class=len(labels_map), eval_metric='mlogloss', random_state=42, device='cuda')

results_xgb = cross_validate(model_xgb, features, target, cv = n_splits, return_train_score=False, n_jobs=-1)

print_results(results_xgb)


from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold # Usar StratifiedKFold para manter a distribuição das classes

# 1. Definir o espaço de hiperparâmetros para otimização
param_grid = {
    'n_estimators': [100, 200, 300], # Número de árvores
    'learning_rate': [0.01, 0.1, 0.2], # Taxa de aprendizado
    'max_depth': [3, 5, 7], # Profundidade máxima das árvores
    'subsample': [0.7, 0.8, 1.0], # Fração de amostras para treinar cada árvore
    'colsample_bytree': [0.7, 0.8, 1.0], # Fração de features para treinar cada árvore
    'gamma': [0, 0.1, 0.2] # Redução mínima de perda necessária para fazer uma partição adicional em um nó folha da árvore.
}

# 2. Criar um objeto do modelo XGBoost
xgb_model = xgb.XGBClassifier(objective='multi:softmax',
                              num_class=len(labels_map),
                              eval_metric='mlogloss',
                              random_state=42,
                              device='cuda')

# 3. Configurar a validação cruzada
cv_method = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 4. Usar GridSearchCV
grid_search = RandomizedSearchCV(estimator=xgb_model,
                           param_distributions=param_grid,
                           n_iter=40,
                           scoring='accuracy',
                           cv=cv_method,
                           verbose=2,
                           n_jobs=-1)

# 5. Treinar o GridSearchCV
print("Iniciando otimização de hiperparâmetros com Grid Search...")
grid_search.fit(x_train, y_train)

# 6. Obter os melhores hiperparâmetros e o melhor modelo
print("Melhores hiperparâmetros encontrados:")
print(grid_search.best_params_)

print("\nMelhor score de cross-validation:")
print(grid_search.best_score_)

# O melhor modelo treinado está disponível em grid_search.best_estimator_
best_xgb_model = grid_search.best_estimator_


pd.DataFrame(grid_search.cv_results_).sort_values(by='mean_test_score', ascending=False)


from sklearn.metrics import classification_report

xgb_model_best = xgb.XGBClassifier(objective='multi:softmax',
                              subsample=0.8, n_estimators=200, max_depth=5, learning_rate=0.1, gamma=0.1, colsample_bytree=0.8,
                              num_class=len(labels_map),
                              eval_metric='mlogloss',
                              random_state=42,
                              device='cuda')

xgb_model_best.fit(x_train, y_train)

y_pred_xgb_best = xgb_model_best.predict(x_test)

print(classification_report(y_test, y_pred_xgb_best))


xgb_model_best.fit(features, target)


url_test = 'https://raw.githubusercontent.com/caalvaro/machine-learning/refs/heads/main/Classification%20-%20Multi-Class%20Prediction%20of%20Obesity%20Risk/data/test.csv'
df_test = pd.read_csv(url_test, index_col='id')

df_test.head()


features_test = pd.get_dummies(df_test, dtype=int)
features_test.head()


y_pred_test = xgb_model_best.predict(features_test.drop(columns='CALC_Always'))


labels_map = {0 : 'Insufficient_Weight',
              1 : 'Normal_Weight',
              2 : 'Overweight_Level_I',
              3 : 'Overweight_Level_II',
              4 : 'Obesity_Type_I',
              5 : 'Obesity_Type_II',
              6 : 'Obesity_Type_III'}

submission = pd.DataFrame({'id': features_test.index, 'NObeyesdad': y_pred_test}).replace(labels_map)
submission.head()


submission.to_csv('submission.csv', index=False)

