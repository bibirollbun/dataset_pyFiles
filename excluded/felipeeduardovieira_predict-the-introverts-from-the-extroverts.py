import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import catboost
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')



# Carregar os dados
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# Exibir as primeiras linhas para análise
print('Train shape:', train.shape)
print('Test shape:', test.shape)
display(train.head())
display(test.head())



# Distribuição da variável target
sns.countplot(x='Personality', data=train)
plt.title('Distribuição da variável target (Personality)')
plt.show()

print(train['Personality'].value_counts(normalize=True))



# Tipos de dados e valores ausentes
print(train.info())
print(train.isnull().sum())

# Verificando possíveis valores ausentes também no teste
print(test.isnull().sum())



# Estatísticas descritivas das features numéricas
display(train.describe())

# Se houver colunas categóricas (além da target)
cat_cols = train.select_dtypes(include='object').columns
print("Colunas categóricas:", cat_cols.tolist())



# Visualizando distribuições das features numéricas
num_cols = train.select_dtypes(include=np.number).columns.tolist()
# Drop 'id' coluna
if 'id' in num_cols:
    num_cols.remove('id')

train[num_cols].hist(figsize=(14, 8), bins=20)
plt.suptitle('Distribuição das variáveis numéricas')
plt.show()


# Matriz de correlação
corr = train[num_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlação')
plt.show()



# Comparando estatísticas entre treino e teste
display(train[num_cols].describe().T)
display(test[num_cols].describe().T)



for col in num_cols:
    plt.figure()
    sns.boxplot(x='Personality', y=col, data=train)
    plt.title(f'{col} vs Personality')
    plt.show()



for col in cat_cols:
    print(pd.crosstab(train[col], train['Personality'], normalize='index'))



# Guardar os IDs para a submissão final
train_id = train['id']
test_id = test['id']
train_df = train.drop('id', axis=1)
test_df = test.drop('id', axis=1)

# Listando colunas numéricas e categóricas automaticamente
num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])




# Codificar a variável alvo (Personality)
le = LabelEncoder()
train_df['Personality'] = le.fit_transform(train_df['Personality'])

# Unir treino e teste para pré-processamento consistente
all_data = pd.concat([train_df.drop('Personality', axis=1), test_df], axis=0)

# Tratar as colunas categóricas com One-Hot Encoding
# Esta abordagem cria novas colunas para cada valor possível, o que funciona bem com muitos modelos
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
all_data = pd.get_dummies(all_data, columns=categorical_cols, dummy_na=True, drop_first=True)


# Separar os dados de volta em treino e teste
X = all_data[:len(train_df)].copy()
y = train_df['Personality'].copy()
X_test = all_data[len(train_df):].copy()

# Garantir que todas as colunas sejam numéricas (float para consistência)
X = X.astype(float)
X_test = X_test.astype(float)


print("Formato dos dados de treino (X):", X.shape)
print("Formato dos dados de teste (X_test):", X_test.shape)


# Definir a estratégia de validação cruzada
NFOLDS = 5
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# Arrays para guardar as previsões
# Out-of-Fold (OOF): previsões sobre os dados de treino para treinar o meta-modelo
oof_preds = np.zeros((len(X), 3))
# Previsões no conjunto de teste
test_preds = np.zeros((len(X_test), 3))

# --- Modelos de Base ---
# Hiperparâmetros otimizados, inspirados nos notebooks de sucesso
lgbm = LGBMClassifier(random_state=42, colsample_bytree=0.7, learning_rate=0.05, max_depth=10, n_estimators=500, subsample=0.8)
xgb = XGBClassifier(random_state=42, colsample_bytree=0.8, eta=0.05, max_depth=5, n_estimators=500, subsample=0.8)
cat = CatBoostClassifier(random_state=42, iterations=500, depth=6, learning_rate=0.05, verbose=0)

models = [('lgbm', lgbm), ('xgb', xgb), ('cat', cat)]

# --- Loop de Validação Cruzada ---
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"========== FOLD {fold+1} ==========")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    for i, (name, model) in enumerate(models):
        print(f"--- Treinando modelo: {name} ---")
        model.fit(X_train, y_train)

        # Previsões de probabilidade para a classe '1' (Extrovert)
        oof_preds[val_idx, i] = model.predict_proba(X_val)[:, 1]
        test_preds[:, i] += model.predict_proba(X_test)[:, 1] / NFOLDS

print("\nTreinamento dos modelos de base concluído!")

# Avaliar a acurácia de cada modelo de base usando as previsões OOF
for i, (name, _) in enumerate(models):
    acc = accuracy_score(y, (oof_preds[:, i] > 0.5).astype(int))
    print(f"Acurácia OOF do modelo {name}: {acc:.4f}")


print("--- Treinando o Meta-Modelo ---")

# As previsões OOF são as novas features
X_meta_train = oof_preds

# A regressão logística é um ótimo e simples meta-modelo
meta_model = LogisticRegression(random_state=42)
meta_model.fit(X_meta_train, y)

print("Meta-Modelo treinado com sucesso!")

# Fazer a previsão final nos dados de teste
final_test_preds_proba = meta_model.predict_proba(test_preds)[:, 1]

# Aplicar um limiar de 0.5 para decidir a classe final
final_predictions = (final_test_preds_proba > 0.5).astype(int)


# Criar o DataFrame de submissão
submission_df = pd.DataFrame({'id': test_id, 'Personality': final_predictions})

# Converter as previsões numéricas de volta para os rótulos originais
submission_df['Personality'] = le.inverse_transform(submission_df['Personality'])

# Salvar para o arquivo de submissão
submission_df.to_csv('submission.csv', index=False)

print("Arquivo 'submission.csv' criado com sucesso!")
print("\n--- Visualização da Submissão ---")
display(submission_df.head())


import matplotlib.pyplot as plt
# Calcular as métricas no conjunto de validação OOF
y_true_oof = y
y_pred_oof_meta = meta_model.predict(oof_preds) # Previsões do meta-modelo no OOF

print("\n--- Métricas de Avaliação no conjunto OOF (Meta-Modelo) ---")
print(classification_report(y_true_oof, y_pred_oof_meta, target_names=le.classes_))

# Matriz de Confusão no conjunto OOF
cm = confusion_matrix(y_true_oof, y_pred_oof_meta)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Previsto')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão (Conjunto OOF)')
plt.show()

# Calcular a acurácia no conjunto de teste (usando as previsões finais)
# Note: Não temos o 'y' real para o conjunto de teste, então calculamos apenas
# a distribuição das previsões finais, não a acurácia real.
print("\n--- Distribuição das Previsões Finais no Conjunto de Teste ---")
print(submission_df['Personality'].value_counts(normalize=True))


from sklearn.metrics import roc_curve, roc_auc_score

# Calcular a curva ROC
fpr, tpr, thresholds = roc_curve(y_true_oof, meta_model.predict_proba(oof_preds)[:, 1])

# Calcular a área sob a curva ROC (AUC)
auc = roc_auc_score(y_true_oof, meta_model.predict_proba(oof_preds)[:, 1])

# Plotar a curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

