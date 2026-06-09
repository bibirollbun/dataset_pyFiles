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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from xgboost import XGBClassifier, plot_importance


# Carregar dados
df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')

# VisÃ£o geral
df.head()

# InformaÃ§Ãµes
df.info()

# Checar valores nulos
print(df.isnull().sum())

# Analisar categorias
for col in ['Soil Type', 'Crop Type', 'Fertilizer Name']:
    print(f"\nValue counts for {col}:\n", df[col].value_counts(normalize=True))

# Resumo estatÃ­stico
df.describe()

# Plot distribuiÃ§Ã£o do target
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Fertilizer Name', order=df['Fertilizer Name'].value_counts().index)
plt.title('Target Distribution (Fertilizer Name)')
plt.xticks(rotation=45)
plt.show()


# Features e target
X = df.drop(columns=['id', 'Fertilizer Name'])
y = df['Fertilizer Name']

# Encode das categÃ³ricas
cat_features = ['Soil Type', 'Crop Type']
encoder_cat = OrdinalEncoder()
X[cat_features] = encoder_cat.fit_transform(X[cat_features])

# Encode do target
encoder_target = LabelEncoder()
y_encoded = encoder_target.fit_transform(y)

# Split dos dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")



from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# Instanciar o modelo
model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(encoder_target.classes_),
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

# Treinar
model.fit(X_train, y_train)

# ğŸ”¥ Probabilidades para todas as classes
y_pred_proba = model.predict_proba(X_test)

# ğŸ”¥ Pegando as Top 3 prediÃ§Ãµes ordenadas
y_pred_top3 = np.argsort(y_pred_proba, axis=1)[:, -3:][:, ::-1]  # Top 3, ordenado decrescente

# ğŸ”¥ PrediÃ§Ã£o da classe mais provÃ¡vel (para accuracy, confusion matrix etc.)
y_pred = np.argmax(y_pred_proba, axis=1)

# ===========================
# âœ”ï¸� MÃ©trica Accuracy
# ===========================
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc:.4f}')

# ===========================
# âœ”ï¸� MÃ©trica MAP@3
# ===========================

# FunÃ§Ã£o MAP@K
def mapk(actual, predicted, k=3):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def apk(actual, predicted, k=3):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p == actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score

# Calcular MAP@3
map3 = mapk(actual=y_test, predicted=y_pred_top3, k=3)
print(f'MAP@3: {map3:.4f}')

# ===========================
# âœ”ï¸� Classification report
# ===========================
print(classification_report(y_test, y_pred, target_names=encoder_target.classes_))

# ===========================
# âœ”ï¸� Confusion Matrix
# ===========================
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder_target.classes_, yticklabels=encoder_target.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()




plt.figure(figsize=(10, 6))
plot_importance(model, max_num_features=10)
plt.title('Top 10 Feature Importances')
plt.show()



model.save_model('xgboost_model.json')


import seaborn as sns
import matplotlib.pyplot as plt

for col in ['Nitrogen', 'Phosphorous', 'Potassium', 'Temparature', 'Humidity', 'Moisture']:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='Fertilizer Name', y=col, data=df)
    plt.title(f'{col} Distribution by Fertilizer')
    plt.xticks(rotation=45)
    plt.show()


## LetÂ´s check the categorical features now ! 


import matplotlib.pyplot as plt
import seaborn as sns

# ğŸ”¥ VariÃ¡veis numÃ©ricas
num_features = ['Nitrogen', 'Phosphorous', 'Potassium', 'Temparature', 'Humidity', 'Moisture']

# ğŸ”¥ VariÃ¡veis categÃ³ricas
cat_features = ['Soil Type', 'Crop Type']

# ğŸ”¥ Loop para plotar boxplots de cada numÃ©rica versus cada categÃ³rica
for cat in cat_features:
    for num in num_features:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=cat, y=num, data=df)
        plt.title(f'{num} Distribution by {cat}')
        plt.xticks(rotation=45)
        plt.show()



# ğŸ”¥ FrequÃªncia entre Soil Type e Fertilizer Name
soil_vs_fertilizer = pd.crosstab(df['Soil Type'], df['Fertilizer Name'], normalize='index')
plt.figure(figsize=(8,6))
sns.heatmap(soil_vs_fertilizer, annot=True, cmap="Blues", fmt=".2f")
plt.title('Fertilizer Distribution by Soil Type')
plt.ylabel('Soil Type')
plt.xlabel('Fertilizer Name')
plt.show()

# ğŸ”¥ FrequÃªncia entre Crop Type e Fertilizer Name
crop_vs_fertilizer = pd.crosstab(df['Crop Type'], df['Fertilizer Name'], normalize='index')
plt.figure(figsize=(10,6))
sns.heatmap(crop_vs_fertilizer, annot=True, cmap="Greens", fmt=".2f")
plt.title('Fertilizer Distribution by Crop Type')
plt.ylabel('Crop Type')
plt.xlabel('Fertilizer Name')
plt.show()



plt.figure(figsize=(12,6))
sns.countplot(data=df, x='Soil Type', hue='Fertilizer Name')
plt.title('Fertilizer Distribution by Soil Type')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12,6))
sns.countplot(data=df, x='Crop Type', hue='Fertilizer Name')
plt.title('Fertilizer Distribution by Crop Type')
plt.xticks(rotation=45)
plt.show()



#  Ver relaÃ§Ã£o entre Soil Type e Crop Type
soil_vs_crop = pd.crosstab(df['Soil Type'], df['Crop Type'], normalize='index')
plt.figure(figsize=(10,6))
sns.heatmap(soil_vs_crop, annot=True, cmap="Purples", fmt=".2f")
plt.title('Crop Distribution by Soil Type')
plt.ylabel('Soil Type')
plt.xlabel('Crop Type')
plt.show()



# âœ… Instalar Optuna se necessÃ¡rio
!pip install optuna --quiet

# ğŸ“¦ Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import optuna

# âœ… Encoder para target
from sklearn.preprocessing import LabelEncoder

# ğŸ”¥ Definir features e target
X = df.drop(columns=['id', 'Fertilizer Name'])
y = df['Fertilizer Name']

# ğŸ”§ Encoding das categÃ³ricas
from sklearn.preprocessing import OrdinalEncoder
cat_features = ['Soil Type', 'Crop Type']
encoder_cat = OrdinalEncoder()
X[cat_features] = encoder_cat.fit_transform(X[cat_features])

# ğŸ”§ Encoding do target
encoder_target = LabelEncoder()
y_encoded = encoder_target.fit_transform(y)

# ğŸ”¥ Split dos dados
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# âœ… FunÃ§Ã£o de avaliaÃ§Ã£o MAP@3
def mapk(actual, predicted, k=3):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def apk(actual, predicted, k=3):
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p == actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score


# âœ… FunÃ§Ã£o objetivo para o Optuna
def objective(trial):
    params = {
        'objective': 'multi:softprob',
        'num_class': len(encoder_target.classes_),
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'tree_method': 'gpu_hist',  
        'random_state': 42,

        # HiperparÃ¢metros a otimizar
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000)
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)
    y_pred_top3 = np.argsort(y_pred_proba, axis=1)[:, -3:][:, ::-1]
    map3 = mapk(actual=y_test, predicted=y_pred_top3, k=3)

    return map3


# âœ… Rodar o tuning com Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("ğŸ”¥ Best Hyperparameters:", study.best_params)
print("ğŸ”¥ Best MAP@3 from tuning:", study.best_value)

# âœ… Treinar modelo final com os melhores hiperparÃ¢metros
best_params = study.best_params
best_params.update({
    'objective': 'multi:softprob',
    'num_class': len(encoder_target.classes_),
    'eval_metric': 'mlogloss',
    'use_label_encoder': False,
    'tree_method': 'hist',
    'random_state': 42
})

model = XGBClassifier(**best_params)
model.fit(X_train, y_train)

model.save_model('xgboost_model_with_optuna.json')

# âœ… Fazer prediÃ§Ãµes
y_pred_proba = model.predict_proba(X_test)
y_pred_top3 = np.argsort(y_pred_proba, axis=1)[:, -3:][:, ::-1]
y_pred = np.argmax(y_pred_proba, axis=1)

# âœ… Calcular mÃ©tricas
acc = accuracy_score(y_test, y_pred)
map3 = mapk(actual=y_test, predicted=y_pred_top3, k=3)

print(f'âœ… Accuracy: {acc:.4f}')
print(f'âœ… MAP@3: {map3:.4f}')

# âœ… Classification report
print(classification_report(y_test, y_pred, target_names=encoder_target.classes_))

# âœ… Confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder_target.classes_, yticklabels=encoder_target.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# âœ… Feature importance plot
plt.figure(figsize=(10,6))
plot_importance(model, max_num_features=10)
plt.title('Top 10 Feature Importances')
plt.show()



import joblib

# ğŸ”¥ Salvar encoder das features categÃ³ricas (Soil Type, Crop Type)
joblib.dump(encoder_cat, 'encoder_cat.pkl')

# ğŸ”¥ Salvar encoder do target (Fertilizer Name)
joblib.dump(encoder_target, 'encoder_target.pkl')

model = XGBClassifier()
model.load_model('xgboost_model_with_optuna.json')

# âœ… Carregar os encoders
encoder_cat = joblib.load('encoder_cat.pkl')
encoder_target = joblib.load('encoder_target.pkl')

# âœ… Carregar o dataset de teste do Kaggle
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# âœ… Preprocessamento â€” aplicar o mesmo encoding do treino
X_test = test_df.drop(columns=['id'])
X_test[['Soil Type', 'Crop Type']] = encoder_cat.transform(X_test[['Soil Type', 'Crop Type']])

# ğŸ”® Fazer prediÃ§Ãµes â€” gerar probabilidades
y_pred_proba = model.predict_proba(X_test)

# ğŸ”¥ Selecionar os Top 3 fertilizantes mais provÃ¡veis
y_pred_top3 = np.argsort(y_pred_proba, axis=1)[:, -3:][:, ::-1]  # Ordenado do mais provÃ¡vel pro menos

# ğŸ”§ Converter os Ã­ndices numÃ©ricos de volta para os nomes dos fertilizantes
pred_labels = [
    ' '.join(encoder_target.inverse_transform(row)) for row in y_pred_top3
]

# âœ… Gerar o DataFrame de submissÃ£o
submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': pred_labels
})

# âœ… Salvar o CSV
submission.to_csv('submission.csv', index=False)

print("ğŸ”¥ SubmissÃ£o salva como 'submission.csv'. Pronta para enviar no Kaggle!")

