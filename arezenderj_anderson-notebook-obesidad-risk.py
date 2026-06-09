import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
from pathlib import Path
import os

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

pd.set_option('display.max_columns', None)
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

import warnings
warnings.filterwarnings('ignore')

print("Bibliotecas importadas com sucesso!")


print("\n--- Configurando Caminhos e Constantes ---")
INPUT_DIR = Path('/kaggle/input/playground-series-s4e2')
OUTPUT_DIR = Path('/kaggle/working/')

TRAIN_FILE_PATH = INPUT_DIR / "train.csv"
TEST_FILE_PATH = INPUT_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = INPUT_DIR / "sample_submission.csv"

MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PREPROCESSOR_PATH = MODELS_DIR / "rf_preprocessor_age_group.joblib"
TARGET_LABEL_ENCODER_PATH = MODELS_DIR / "rf_target_label_encoder.joblib"
FINAL_RF_MODEL_PATH = MODELS_DIR / "random_forest_final_model.joblib"

SUBMISSIONS_DIR = OUTPUT_DIR
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "NObeyesdad"
ID_COLUMN = "id"
NEW_AGE_GROUP_COLUMN = "Age_Group"

RANDOM_STATE = 42
TEST_SPLIT_SIZE = 0.2
CV_FOLDS_OPTIMIZATION = 3
N_ITER_RANDOM_SEARCH_RF = 10 # Reduzido para execuções mais rápidas no exemplo. Aumente para busca real.

print("Configurações definidas.")


print("\n--- Carregando Dados ---")
try:
    df_train_raw = pd.read_csv(TRAIN_FILE_PATH)
    df_test_raw = pd.read_csv(TEST_FILE_PATH)
    df_submission_sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    print(f"Dataset de Treino: {df_train_raw.shape}")
    print(f"Dataset de Teste: {df_test_raw.shape}")
    print(f"Exemplo de Submissão: {df_submission_sample.shape}")

    print("\nPrimeiras 5 linhas do df_train_raw:")
    display(df_train_raw.head())
    print("\nPrimeiras 5 linhas do df_test_raw:")
    display(df_test_raw.head())
    print("\nPrimeiras 5 linhas do df_submission_sample (formato esperado):")
    display(df_submission_sample.head())

except FileNotFoundError as e:
    print(f"ERRO: Arquivo de dados não encontrado: {e}")
    df_train_raw, df_test_raw, df_submission_sample = None, None, None
except Exception as e:
    print(f"ERRO ao carregar dados: {e}")
    df_train_raw, df_test_raw, df_submission_sample = None, None, None


if df_train_raw is not None and df_test_raw is not None:
    print("\n--- Engenharia de Features: Criando Faixas Etárias ---")
    bins = [0, 12.9, 19.9, 29.9, 44.9, 59.9, np.inf]
    labels = ['Criança', 'Adolescente', 'Jovem Adulto', 'Adulto', 'Adulto Meia-Idade', 'Idoso']
    df_train_raw[NEW_AGE_GROUP_COLUMN] = pd.cut(df_train_raw['Age'], bins=bins, labels=labels, right=True)
    df_test_raw[NEW_AGE_GROUP_COLUMN] = pd.cut(df_test_raw['Age'], bins=bins, labels=labels, right=True)
    print(f"Coluna '{NEW_AGE_GROUP_COLUMN}' criada e adicionada aos DataFrames.")
    print(df_train_raw[[ID_COLUMN, 'Age', NEW_AGE_GROUP_COLUMN]].head()) # Mostrar exemplo
else:
    print("Dados brutos não carregados, pulando engenharia de features de idade.")


# ## 6. Análise Exploratória de Dados (EDA) - Focada
# (Após a Seção 5: Engenharia de Features: Criação de Faixas Etárias)

if df_train_raw is not None: # Garante que os dados foram carregados
    print("\n--- Iniciando Análise Exploratória de Dados (EDA) com Age_Group ---")

    # ### 6.1 Visão Geral e Nulos
    print("\nInformações básicas do df_train_raw (após adicionar Age_Group):")
    df_train_raw.info(verbose=False)
    print(f"\nTotal de valores ausentes no treino: {df_train_raw.isnull().sum().sum()}") # Deve ser 0

    # ### 6.2 Análise da Variável Alvo
    print(f"\nDistribuição da Variável Alvo: {TARGET_COLUMN}")
    target_counts = df_train_raw[TARGET_COLUMN].value_counts() # Lembre-se que TARGET_COLUMN ainda é string aqui
    plt.figure(figsize=(10,6))
    sns.barplot(x=target_counts.index, y=target_counts.values, palette="crest")
    plt.xlabel("Nível de Obesidade")
    plt.ylabel("Contagem")
    plt.title("Distribuição da Variável Alvo")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    # for i, count in target_counts.items(): # Descomente para ver as porcentagens
    #     print(f"{i}: {count} ({count/len(df_train_raw)*100:.2f}%)")


    # ### 6.3 Identificação de Features para Análise e Pré-processamento
    # Excluir 'Age' original, pois usaremos 'Age_Group'
    # Guardar a lista de features que serão efetivamente usadas no pré-processamento
    features_for_processing = [col for col in df_train_raw.columns if col not in [ID_COLUMN, TARGET_COLUMN, 'Age']]

    numerical_cols_eda = df_train_raw[features_for_processing].select_dtypes(include=np.number).columns.tolist()
    categorical_cols_eda = df_train_raw[features_for_processing].select_dtypes(include=['object', 'category']).columns.tolist()

    # Garantir que NEW_AGE_GROUP_COLUMN seja tratada como categórica se existir e não tiver sido pega
    if NEW_AGE_GROUP_COLUMN in df_train_raw[features_for_processing].columns and \
       df_train_raw[NEW_AGE_GROUP_COLUMN].dtype.name == 'category' and \
       NEW_AGE_GROUP_COLUMN not in categorical_cols_eda:
        categorical_cols_eda.append(NEW_AGE_GROUP_COLUMN)
    categorical_cols_eda = list(dict.fromkeys(categorical_cols_eda)) # Remover duplicatas

    print(f"\nFeatures Numéricas para Análise/Processamento: {numerical_cols_eda}")
    print(f"Features Categóricas para Análise/Processamento: {categorical_cols_eda}")


    # ### 6.4 Visualização de Features Numéricas
    print("\nHistogramas das Features Numéricas:")
    if numerical_cols_eda: # Apenas plotar se houver features numéricas
        df_train_raw[numerical_cols_eda].hist(bins=25, figsize=(18,max(12, len(numerical_cols_eda)*3)), layout=(-1,4), color='skyblue', edgecolor='black')
        plt.suptitle("Distribuição das Features Numéricas", fontsize=16)
        plt.tight_layout(rect=[0,0,1,0.97])
        plt.show()
    else:
        print("Nenhuma feature numérica para plotar histogramas.")

    print("\nBoxplots das Features Numéricas vs Alvo:")
    if numerical_cols_eda:
        for col in numerical_cols_eda:
            plt.figure(figsize=(12,7))
            sns.boxplot(data=df_train_raw, x=TARGET_COLUMN, y=col, palette="pastel") # TARGET_COLUMN aqui é string
            plt.title(f"{col} vs {TARGET_COLUMN}", fontsize=15)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()
    else:
        print("Nenhuma feature numérica para plotar boxplots.")


    # ### 6.5 Visualização de Features Categóricas (incluindo Age_Group)
    print("\nCountplots das Features Categóricas vs Alvo:")
    if categorical_cols_eda:
        for col in categorical_cols_eda:
            plt.figure(figsize=(12,7)) # Aumentado para melhor visualização com hue
            order_values = labels if col == NEW_AGE_GROUP_COLUMN else df_train_raw[col].value_counts().index
            sns.countplot(data=df_train_raw, y=col, hue=TARGET_COLUMN, order=order_values, palette="Spectral", dodge=True)
            plt.title(f"Distribuição de {col} por {TARGET_COLUMN}", fontsize=15)
            plt.legend(title=TARGET_COLUMN, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
    else:
        print("Nenhuma feature categórica para plotar countplots.")


    # ### 6.6 Matriz de Correlação (Features Numéricas)
    print("\nMatriz de Correlação das Features Numéricas:")
    if numerical_cols_eda:
        # Para a matriz de correlação, é útil se a variável alvo também for numérica,
        # mas aqui vamos focar apenas nas correlações entre as features numéricas.
        # Se quiséssemos incluir o alvo, teríamos que codificá-lo primeiro para esta visualização específica.
        correlation_matrix = df_train_raw[numerical_cols_eda].corr()
        plt.figure(figsize=(max(10, len(numerical_cols_eda)*0.8), max(8, len(numerical_cols_eda)*0.6))) # Ajuste dinâmico do tamanho
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, annot_kws={"size": 8})
        plt.title("Matriz de Correlação das Features Numéricas", fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    else:
        print("Nenhuma feature numérica para calcular a matriz de correlação.")

    print("EDA concluída.")
else:
    print("df_train_raw não foi carregado. Pulando EDA.")


print("\n--- Pré-processamento para Modelagem com RandomForest")
X_processed = None
y = None
X_submission_test_processed = None
preprocessor_fitted_global = None
target_encoder_global = None

if df_train_raw is not None and df_test_raw is not None:
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(df_train_raw[TARGET_COLUMN])
    joblib.dump(target_encoder, TARGET_LABEL_ENCODER_PATH)
    target_encoder_global = target_encoder
    print(f"LabelEncoder da variável alvo salvo. Classes: {list(target_encoder_global.classes_)}")

    X = df_train_raw.drop([ID_COLUMN, TARGET_COLUMN, 'Age'], axis=1)
    y = pd.Series(y_encoded, name=TARGET_COLUMN, index=X.index)
    X_test_for_submission = df_test_raw.drop([ID_COLUMN, 'Age'], axis=1)

    numerical_cols_for_proc = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_for_proc = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if NEW_AGE_GROUP_COLUMN in X.columns and X[NEW_AGE_GROUP_COLUMN].dtype.name == 'category' and NEW_AGE_GROUP_COLUMN not in categorical_cols_for_proc:
        categorical_cols_for_proc.append(NEW_AGE_GROUP_COLUMN)
    categorical_cols_for_proc = list(dict.fromkeys(categorical_cols_for_proc))

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categoric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numerical_cols_for_proc), ('cat', categoric_transformer, categorical_cols_for_proc)], remainder='drop')

    X_processed = preprocessor.fit_transform(X)
    preprocessor_fitted_global = preprocessor
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"Pré-processador ajustado e salvo. Shape de X_processed: {X_processed.shape}")

    X_submission_test_processed = preprocessor.transform(X_test_for_submission)
    print(f"Shape de X_test_final_processed: {X_submission_test_processed.shape}")

    X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f"Dados divididos em Treino ({X_train.shape}) e Validação ({X_val.shape})")
    print("Pré-processamento concluído.")
else:
    print("Dados brutos não carregados. Pulando pré-processamento.")


print("\n--- Treinamento e Avaliação do RandomForest Baseline ---")
rf_baseline_model = None
if 'X_train' in globals() and X_train is not None:
    rf_baseline = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, n_estimators=100)
    start_time = time.time()
    rf_baseline.fit(X_train, y_train)
    training_time = time.time() - start_time
    rf_baseline_model = rf_baseline # Salvar modelo treinado

    y_pred_val_rf_baseline = rf_baseline.predict(X_val)
    accuracy_rf_baseline = accuracy_score(y_val, y_pred_val_rf_baseline)

    print(f"Acurácia do RandomForest Baseline na validação: {accuracy_rf_baseline:.4f}")
    print(f"Tempo de treinamento do RandomForest Baseline: {training_time:.2f}s")

    if 'target_encoder_global' in globals() and target_encoder_global is not None:
        print("\nRelatório de Classificação (RandomForest Baseline - Validação):")
        print(classification_report(y_val, y_pred_val_rf_baseline, target_names=target_encoder_global.classes_))

        cm_rf_baseline = confusion_matrix(y_val, y_pred_val_rf_baseline)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm_rf_baseline, annot=True, fmt='d', cmap='Greens', xticklabels=target_encoder_global.classes_, yticklabels=target_encoder_global.classes_)
        plt.xlabel('Previsto')
        plt.ylabel('Verdadeiro')
        plt.title('Matriz de Confusão - RandomForest Baseline (Validação)')
        plt.show()
    else:
        print("target_encoder_global não definido, relatório de classificação detalhado não pode ser gerado com nomes de classes.")
else:
    print("Dados de treino não disponíveis. Pulando treinamento do RandomForest baseline.")


print("\n--- Otimização de Hiperparâmetros para RandomForest ---")
rf_tuned_model = None # Inicializar
best_rf_params = None
best_rf_cv_score = 0.0

if 'rf_baseline_model' in globals() and rf_baseline_model is not None and X_processed is not None and y is not None:
    param_dist_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'], # Removido None para evitar problemas potenciais com grande número de features
        'class_weight': [None, 'balanced', 'balanced_subsample']
    }
    rf_for_tuning = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    random_search_rf = RandomizedSearchCV(
        estimator=rf_for_tuning, param_distributions=param_dist_rf,
        n_iter=N_ITER_RANDOM_SEARCH_RF, cv=StratifiedKFold(n_splits=CV_FOLDS_OPTIMIZATION, shuffle=True, random_state=RANDOM_STATE),
        scoring='accuracy', random_state=RANDOM_STATE, n_jobs=-1, verbose=1
    )
    print("Iniciando RandomizedSearchCV para RandomForest...")
    start_time_tuning = time.time()
    random_search_rf.fit(X_processed, y) # Otimizar usando todos os dados de treino processados
    tuning_time = time.time() - start_time_tuning

    print(f"\nTempo de otimização: {tuning_time:.2f}s")
    best_rf_params = random_search_rf.best_params_
    best_rf_cv_score = random_search_rf.best_score_
    print(f"Melhores hiperparâmetros para RandomForest: {best_rf_params}")
    print(f"Melhor acurácia (CV com RandomizedSearch): {best_rf_cv_score:.4f}")

    rf_tuned_model = random_search_rf.best_estimator_ # Modelo já treinado com os melhores params em todos os dados de X_processed

    y_pred_val_rf_tuned = rf_tuned_model.predict(X_val)
    accuracy_rf_tuned_val = accuracy_score(y_val, y_pred_val_rf_tuned)
    print(f"\nAcurácia do RandomForest Otimizado na validação local (X_val): {accuracy_rf_tuned_val:.4f}")

    if 'target_encoder_global' in globals() and target_encoder_global is not None:
        print("\nRelatório de Classificação (RandomForest Otimizado - Validação):")
        print(classification_report(y_val, y_pred_val_rf_tuned, target_names=target_encoder_global.classes_))
        # ... (plot da matriz de confusão como antes) ...
    else:
        print("target_encoder_global não definido.")
else:
    print("Modelo baseline RandomForest ou dados processados não disponíveis. Pulando otimização.")


print("\n--- Preparando Submissão Final com RandomForest ---")
model_to_submit = None

if 'rf_tuned_model' in globals() and rf_tuned_model is not None:
    # Verificar se o modelo tunado é melhor que o baseline NA VALIDAÇÃO LOCAL
    # (best_rf_cv_score é do CV da otimização, accuracy_rf_tuned_val é na validação local do modelo tunado)
    if 'accuracy_rf_tuned_val' in globals() and 'accuracy_rf_baseline' in globals() and accuracy_rf_tuned_val > accuracy_rf_baseline:
        print(f"Usando RandomForest Otimizado (Val Acc: {accuracy_rf_tuned_val:.4f}) para submissão.")
        model_to_submit = rf_tuned_model # Já está treinado em X_processed
    elif 'rf_baseline_model' in globals() and rf_baseline_model is not None:
        print(f"Usando RandomForest Baseline (Val Acc: {accuracy_rf_baseline:.4f}) para submissão, pois o otimizado não foi melhor na validação local.")
        model_to_submit = rf_baseline_model
        # Re-treinar o baseline em todos os dados X_processed se ele foi treinado apenas em X_train antes
        print("Re-treinando modelo baseline em todos os dados de treino (X_processed)...")
        model_to_submit.fit(X_processed, y)
    else:
        print("Não foi possível determinar o melhor modelo para submissão a partir da otimização ou baseline.")
elif 'rf_baseline_model' in globals() and rf_baseline_model is not None:
    print("Otimização não foi executada ou falhou. Usando RandomForest Baseline para submissão.")
    model_to_submit = rf_baseline_model
    print("Re-treinando modelo baseline em todos os dados de treino (X_processed)...")
    model_to_submit.fit(X_processed, y)
else:
    print("ERRO: Nenhum modelo RandomForest treinado disponível para submissão.")


if model_to_submit is not None and X_submission_test_processed is not None and df_test_raw is not None:
    joblib.dump(model_to_submit, FINAL_RF_MODEL_PATH)
    print(f"Modelo RandomForest final salvo em: {FINAL_RF_MODEL_PATH}")

    print("\nFazendo previsões no conjunto de teste final...")
    test_predictions_encoded = model_to_submit.predict(X_submission_test_processed)

    try:
        target_encoder_loaded = joblib.load(TARGET_LABEL_ENCODER_PATH)
        test_predictions_original_labels = target_encoder_loaded.inverse_transform(test_predictions_encoded)
        print("Previsões decodificadas.")

        submission_df_final = pd.DataFrame({
            ID_COLUMN: df_test_raw[ID_COLUMN],
            TARGET_COLUMN: test_predictions_original_labels
        })
        submission_file_final_path = SUBMISSIONS_DIR / "submission_rf_final.csv" # Nome específico
        submission_df_final.to_csv(submission_file_final_path, index=False)

        print(f"\nArquivo de submissão '{submission_file_final_path}' criado.")
        display(submission_df_final.head())
    except FileNotFoundError:
        print(f"ERRO: Arquivo do LabelEncoder do alvo não encontrado em {TARGET_LABEL_ENCODER_PATH}.")
    except Exception as e:
        print(f"ERRO durante a decodificação ou criação da submissão: {e}")
else:
    print("ERRO: Não foi possível gerar a submissão. Verifique as etapas anteriores.")


print("\n--- Conclusão Final ---")
if 'best_overall_model_name' in globals() and 'accuracy_rf_tuned_val' in globals() and 'best_rf_cv_score' in globals():
    print(f"Pipeline de classificação de risco de obesidade utilizando RandomForest ({best_overall_model_name}) concluído.")
    print(f"A feature '{NEW_AGE_GROUP_COLUMN}' foi criada e incorporada ao modelo.")
    if "Tuned" in best_overall_model_name:
        print(f"O modelo RandomForest foi otimizado, resultando em uma acurácia CV de {best_rf_cv_score:.4f} e uma acurácia na validação local de {accuracy_rf_tuned_val:.4f}.")
    elif 'accuracy_rf_baseline' in globals() :
         print(f"O modelo RandomForest baseline obteve uma acurácia na validação local de {accuracy_rf_baseline:.4f}.")
    if 'submission_file_final_path' in globals() and submission_file_final_path.exists():
        print(f"O arquivo de submissão foi gerado em: {submission_file_final_path}")
    else:
        print("Arquivo de submissão não foi gerado devido a erros anteriores.")
else:
    print("Conclusão não pode ser totalmente gerada devido a variáveis ausentes.")

print("\nPróximos passos podem incluir exploração de outras features, algoritmos mais avançados ou técnicas de ensembling.")

