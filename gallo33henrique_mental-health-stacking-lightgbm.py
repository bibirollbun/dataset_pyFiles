import random
import pandas as pd
import numpy as np

#
%matplotlib inline
import squarify
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#
from scipy import stats

#
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer

#
from sklearn.model_selection import train_test_split

#
from sklearn.model_selection import GridSearchCV, cross_val_score

#
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, HistGradientBoostingClassifier

#
from sklearn.neural_network import MLPClassifier

#
from sklearn.metrics import make_scorer, accuracy_score

# Set random seed
rs = 42

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# -------------------------------
# 1. Carregar os Dados
# -------------------------------
# Ajuste os caminhos conforme sua necessidade ou ambiente (por exemplo, Kaggle)
treino_caminho = "/kaggle/input/analyze-the-insights-over-mental-health-data/train.csv"
teste_caminho  = "/kaggle/input/analyze-the-insights-over-mental-health-data/test.csv"

df_treino = pd.read_csv(treino_caminho)
df_teste  = pd.read_csv(teste_caminho)


#
df_treino.head()


#
df_treino.head()


df_treino.tail()


df_treino.info()


df_treino.dtypes


# -------------------------------
# 2. Remover a coluna de identificação (id)
# -------------------------------
# Guardar os IDs do conjunto de teste para eventual submissão
ids_teste = df_teste['id'].copy()
df_treino.drop('id', axis=1, inplace=True)
df_teste.drop('id', axis=1, inplace=True)


# 3. Preparação dos dados para modelagem

# Definir a coluna alvo
target_column = 'Depression'

# Selecionar as colunas categóricas e numéricas (inicialmente)
categorical_columns = df_treino.select_dtypes(include=['object']).columns.tolist()
numerical_columns   = df_treino.select_dtypes(exclude=['object']).columns.drop(target_column).tolist()

print("Coluna Alvo:", target_column)
print("Colunas Categóricas:", categorical_columns)
print("Colunas Numéricas:", numerical_columns)


# -------------------------------
# 3. Criar Novas Variáveis com Nomes Diferentes
# -------------------------------

def criar_novas_features(df):
    """
    Cria novas features a partir do DataFrame fornecido e retorna uma cópia com as transformações.

    As novas features criadas são:
      - ip_idade_pressao: interação entre "Age" e "Work Pressure"
      - r_satisfacao: razão entre "Study Satisfaction" e "Job Satisfaction"
      - r_pressao: razão entre "Academic Pressure" e "Work Pressure"
      - total_estresse: soma de "Academic Pressure", "Work Pressure" e "Financial Stress"
      - r_horas_estresse: razão entre "Work/Study Hours" e "Financial Stress"
      - freq_prof: frequência de ocorrência da "Profession" (calculada com value_counts)
    
    Parâmetros:
      df (DataFrame): DataFrame de entrada com as colunas originais.
    
    Retorna:
      DataFrame: Uma cópia do DataFrame com as novas features adicionadas.
    """
    # Fazer uma cópia para não alterar o DataFrame original
    df_mod = df.copy()
    
    # 3.1. Interação entre 'Age' e 'Work Pressure'
    df_mod["ip_idade_pressao"] = df_mod["Age"] * df_mod["Work Pressure"]
    
    # 3.2. Razão entre 'Study Satisfaction' e 'Job Satisfaction'
    df_mod["r_satisfacao"] = df_mod["Study Satisfaction"] / (df_mod["Job Satisfaction"] + 1e-5)
    
    # 3.3. Razão entre 'Academic Pressure' e 'Work Pressure'
    df_mod["r_pressao"] = df_mod["Academic Pressure"] / (df_mod["Work Pressure"] + 1e-5)
    
    # 3.4. Score Total de Estresse: soma de 'Academic Pressure', 'Work Pressure' e 'Financial Stress'
    df_mod["total_estresse"] = df_mod["Academic Pressure"] + df_mod["Work Pressure"] + df_mod["Financial Stress"]
    
    # 3.5. Razão entre 'Work/Study Hours' e 'Financial Stress'
    df_mod["r_horas_estresse"] = df_mod["Work/Study Hours"] / (df_mod["Financial Stress"] + 1e-5)
    
    # 3.6. Frequência da Profissão
    contagem_prof = df_mod["Profession"].value_counts().to_dict()
    df_mod["freq_prof"] = df_mod["Profession"].map(contagem_prof)
    
    return df_mod

# Supondo que df_treino e df_teste sejam os DataFrames originais
df_treino_mod = criar_novas_features(df_treino)
df_teste_mod  = criar_novas_features(df_teste)


def converter_sono(df, coluna_origem="Sleep Duration", nova_coluna="sono_valor"):
    """
    Converte os valores da coluna de duração de sono para valores numéricos
    utilizando um mapeamento customizado e cria uma nova coluna com os valores convertidos.
    
    Parâmetros:
        df (DataFrame): O DataFrame contendo a coluna de duração de sono.
        coluna_origem (str): O nome da coluna que contém a duração do sono (padrão: "Sleep Duration").
        nova_coluna (str): O nome da nova coluna a ser criada com os valores numéricos (padrão: "sono_valor").
    
    Retorna:
        DataFrame: O DataFrame com a nova coluna adicionada.
    """
    mapa_sono = {
        "Less than 5 hours": 4,
        "5-6 hours": 5.5,
        "6-7 hours": 6.5,
        "7-8 hours": 7.5,
        "More than 8 hours": 9,
        "3-4 hours": 3.5,
        "4-5 hours": 4.5,
        "4-6 hours": 5,
        "2-3 hours": 2.5,
        "6-8 hours": 7
    }
    
    df[nova_coluna] = df[coluna_origem].map(mapa_sono)
    return df

# DataFrames de treino e teste:
df_treino = converter_sono(df_treino)
df_teste  = converter_sono(df_teste)


# -------------------------------
# 4. Visualizar as Novas Features
# -------------------------------

# Salvando dataset
df_treino.to_csv("dataset.csv")

print("Preview do DataFrame de Treino com Novas Variáveis:")
print(df_treino.head())

print("\nColunas disponíveis após as alterações:")
print(df_treino.columns.tolist())


# Dataset para visualização dados
df = pd.read_csv("/kaggle/working/dataset.csv")


# Análise de valores ausentes
print("\nValores ausentes por coluna:")
print(df.isnull().sum())


# Visualizar valores ausentes com um heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Mapa de Valores Ausentes")
plt.show()


# Estatísticas descritivas para variáveis numéricas
print("\nEstatísticas descritivas (numéricas):")
df.describe().T


# Estatísticas descritivas para variáveis categóricas
print("\nContagem de valores para variáveis categóricas:")

#
for col in ['Name', 'Gender', 'City', 'Working Professional or Student', 'Profession', 
            'Sleep Duration', 'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?', 
            'Family History of Mental Illness']:

    #
    print(f"\nColuna: {col}")
    print(df[col].value_counts())


# -----------------------
# Análise da variável alvo "Depression"
# -----------------------

# Distribuição da variável alvo
plt.figure()
sns.countplot(x='Depression', data=df, palette="Set2")
plt.title("Distribuição da Variável Alvo: Depression")
plt.xlabel("Depression (0 = Não, 1 = Sim)")
plt.ylabel("Contagem")
plt.show()


# -----------------------
# Distribuição de variáveis numéricas
# -----------------------

# Lista de variáveis numéricas (ajuste conforme seu dataset)
numerical_cols = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
                  'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours', 'Financial Stress', 'sono_valor']

# Gráfico histrograma
for col in numerical_cols:
    plt.figure()
    sns.histplot(df[col].dropna(), kde=True, bins=25)
    plt.title(f"Distribuição de {col}")
    plt.xlabel(col)
    plt.ylabel("Frequência")
    plt.show()


# Análise da relação entre Depression e algumas features numéricas
selected_features = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'sono_valor']
for col in selected_features:
    plt.figure()
    sns.boxplot(x='Depression', y=col, data=df, palette="Set2")
    plt.title(f"Distribuição de {col} por Depression")
    plt.xlabel("Depression (0 = Não, 1 = Sim)")
    plt.ylabel(col)
    plt.show()


plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.violinplot(x='Depression', y='Academic Pressure', data=df, palette="Set2")
plt.title("Academic Pressure por Depression")
plt.xlabel("Depression (0 = Não, 1 = Sim)")

plt.subplot(1, 2, 2)
sns.violinplot(x='Depression', y='Work Pressure', data=df, palette="Set2")
plt.title("Work Pressure por Depression")
plt.xlabel("Depression (0 = Não, 1 = Sim)")

plt.tight_layout()
plt.show()


# -----------------------
# Análise de correlação
# -----------------------

# Calcular a matriz de correlação apenas para as variáveis numéricas
corr_matrix = df[numerical_cols].corr()

# Exibir a matriz de correlação com seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlação entre Variáveis Numéricas")
plt.show()


import seaborn as sns
from scipy.stats import pointbiserialr, chi2_contingency

# Converter 'Depression' para numérico (já está em 0 e 1)
df['Depression'] = df['Depression'].astype(int)

# Remover valores nulos das colunas de interesse para análise de correlação
df_corr = df[['Academic Pressure', 'Work Pressure', 'Depression']].dropna()

# Contar valores não nulos nas colunas de interesse
valid_counts = df[['Academic Pressure', 'Work Pressure', 'Depression']].notnull().sum()

# Remover valores nulos apenas para Work Pressure e Depression
df_work_corr = df[['Work Pressure', 'Depression']].dropna()

# Calcular a correlação entre Work Pressure e Depression
corr_work, p_work = pointbiserialr(df_work_corr['Depression'], df_work_corr['Work Pressure'])

# Exibir resultado da correlação
corr_work, p_work


# Remover valores nulos para Study Satisfaction e Job Satisfaction
df_satisfaction = df[['Study Satisfaction', 'Job Satisfaction', 'Depression']].dropna()

# Calcular médias de satisfação para cada grupo de depressão
study_satisfaction_mean = df_satisfaction.groupby('Depression')['Study Satisfaction'].mean()
job_satisfaction_mean = df_satisfaction.groupby('Depression')['Job Satisfaction'].mean()

# Criar gráficos para visualizar a relação entre satisfação e depressão
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Gráfico para Study Satisfaction
sns.boxplot(x=df_satisfaction['Depression'], y=df_satisfaction['Study Satisfaction'], ax=axes[0])
axes[0].set_title("Satisfação com os Estudos vs Depressão")
axes[0].set_xlabel("Depressão (0 = Não, 1 = Sim)")
axes[0].set_ylabel("Satisfação com os Estudos")

# Gráfico para Job Satisfaction
sns.boxplot(x=df_satisfaction['Depression'], y=df_satisfaction['Job Satisfaction'], ax=axes[1])
axes[1].set_title("Satisfação no Trabalho vs Depressão")
axes[1].set_xlabel("Depressão (0 = Não, 1 = Sim)")
axes[1].set_ylabel("Satisfação no Trabalho")

plt.tight_layout()
plt.show()

# Aplicar teste estatístico (teste t) para verificar diferenças significativas
from scipy.stats import ttest_ind

study_ttest = ttest_ind(
    df_satisfaction[df_satisfaction['Depression'] == 0]['Study Satisfaction'],
    df_satisfaction[df_satisfaction['Depression'] == 1]['Study Satisfaction'],
    equal_var=False
)

job_ttest = ttest_ind(
    df_satisfaction[df_satisfaction['Depression'] == 0]['Job Satisfaction'],
    df_satisfaction[df_satisfaction['Depression'] == 1]['Job Satisfaction'],
    equal_var=False
)

# Exibir resultados das médias e testes estatísticos
satisfaction_results = {
    "Média de Study Satisfaction (Sem Depressão)": study_satisfaction_mean[0],
    "Média de Study Satisfaction (Com Depressão)": study_satisfaction_mean[1],
    "p-valor Study Satisfaction": study_ttest.pvalue,
    "Média de Job Satisfaction (Sem Depressão)": job_satisfaction_mean[0],
    "Média de Job Satisfaction (Com Depressão)": job_satisfaction_mean[1],
    "p-valor Job Satisfaction": job_ttest.pvalue,
}

satisfaction_results


# Remover valores nulos para Work/Study Hours e Depression
df_work_hours = df[['Work/Study Hours', 'Depression']].dropna()

# Criar gráfico de distribuição das horas de trabalho/estudo para cada grupo de depressão
plt.figure(figsize=(8, 5))
sns.boxplot(x=df_work_hours['Depression'], y=df_work_hours['Work/Study Hours'])
plt.title("Relação entre Horas de Trabalho/Estudo e Depressão")
plt.xlabel("Depressão (0 = Não, 1 = Sim)")
plt.ylabel("Horas de Trabalho/Estudo")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Calcular médias de horas de trabalho/estudo para cada grupo de depressão
work_hours_mean = df_work_hours.groupby('Depression')['Work/Study Hours'].mean()

# Aplicar teste estatístico (teste t) para verificar diferenças significativas
work_hours_ttest = ttest_ind(df_work_hours[df_work_hours['Depression'] == 0]['Work/Study Hours'],
                             df_work_hours[df_work_hours['Depression'] == 1]['Work/Study Hours'],
                             equal_var=False)

# Exibir resultados das médias e testes estatísticos
work_hours_results = {"Média de Work/Study Hours (Sem Depressão)": work_hours_mean[0],
                      "Média de Work/Study Hours (Com Depressão)": work_hours_mean[1],
                      "p-valor Work/Study Hours": work_hours_ttest.pvalue,}

# Visualizando gráfico
work_hours_results


# Contar a distribuição de hábitos alimentares por depressão
diet_counts = df.groupby(['Dietary Habits', 'Depression']).size().unstack()

# Criar gráfico de barras para visualizar a relação entre hábitos alimentares e depressão
plt.figure(figsize=(20.5, 10))
diet_counts.plot(kind='bar', stacked=True, figsize=(30.5, 10), colormap="coolwarm")
plt.title("Relação entre Hábitos Alimentares e Depressão")
plt.xlabel("Hábitos Alimentares")
plt.ylabel("Número de Pessoas")
plt.legend(["Sem Depressão", "Com Depressão"], title="Depressão")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Aplicar teste qui-quadrado para verificar associação entre hábitos alimentares e depressão
chi2, p_diet, dof, expected = chi2_contingency(diet_counts.fillna(0))

# Exibir resultados do teste estatístico
diet_results = {"p-valor Hábitos Alimentares vs Depressão": p_diet,
                "Estatística Qui-Quadrado": chi2,}

diet_results


# Contar a distribuição de histórico familiar por depressão
family_history_counts = df.groupby(['Family History of Mental Illness', 'Depression']).size().unstack()

# Criar gráfico de barras para visualizar a relação entre histórico familiar e depressão
plt.figure(figsize=(6, 4))
family_history_counts.plot(kind='bar', stacked=True, figsize=(8, 5), colormap="coolwarm")
plt.title("Histórico Familiar de Doenças Mentais vs Depressão")
plt.xlabel("Histórico Familiar de Doenças Mentais")
plt.ylabel("Número de Pessoas")
plt.legend(["Sem Depressão", "Com Depressão"], title="Depressão")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Aplicar teste qui-quadrado para verificar associação entre histórico familiar e depressão
chi2_family, p_family, dof, expected = chi2_contingency(family_history_counts.fillna(0))

# Exibir resultados do teste estatístico
family_history_results = {"p-valor Histórico Familiar vs Depressão": p_family,
                          "Estatística Qui-Quadrado": chi2_family,}

# Visualizando teste
family_history_results


# Verificar valores únicos na coluna 'Sleep Duration' para entender seu formato
df['Sleep Duration'].unique()

# Criar um mapeamento para categorizar os valores corretos
sleep_mapping = {'Less than 5 hours': '<5h',
                 '1-2 hours': '<5h',
                 '2-3 hours': '<5h',
                 '3-4 hours': '<5h',
                 '4-5 hours': '<5h',
                 '4-6 hours': '5-6h',
                 '5-6 hours': '5-6h',
                 '6-7 hours': '6-8h',
                 '6-8 hours': '6-8h',
                 '7-8 hours': '6-8h',
                 '8-9 hours': '>8h',
                 '9-11 hours': '>8h',
                 '10-11 hours': '>8h',
                 'More than 8 hours': '>8h'}

# Aplicar a limpeza, mantendo NaN para valores inconsistentes
df['Sleep Duration Cleaned'] = df['Sleep Duration'].map(sleep_mapping)


# Contar a distribuição de duração do sono por depressão
sleep_counts = df.groupby(['Sleep Duration Cleaned', 'Depression']).size().unstack()

# Criar gráfico de barras para visualizar a relação entre sono e depressão
plt.figure(figsize=(6, 4))
sleep_counts.plot(kind='bar', stacked=True, figsize=(20.5, 10), colormap="coolwarm")
plt.title("Duração do Sono vs Depressão")
plt.xlabel("Duração do Sono")
plt.ylabel("Número de Pessoas")
plt.legend(["Sem Depressão", "Com Depressão"], title="Depressão")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Aplicar teste qui-quadrado para verificar associação entre duração do sono e depressão
chi2_sleep, p_sleep, dof, expected = chi2_contingency(sleep_counts.fillna(0))

# Exibir resultados do teste estatístico
sleep_results = {"p-valor Duração do Sono vs Depressão": p_sleep,
                 "Estatística Qui-Quadrado": chi2_sleep,}

sleep_results


# Remover valores nulos para Financial Stress e Depression
df_financial_stress = df[['Financial Stress', 'Depression']].dropna()

# Criar gráfico de barras para visualizar a relação entre estresse financeiro e depressão
plt.figure(figsize=(6, 4))
sns.boxplot(x=df_financial_stress['Depression'], y=df_financial_stress['Financial Stress'])
plt.title("Relação entre Estresse Financeiro e Depressão")
plt.xlabel("Depressão (0 = Não, 1 = Sim)")
plt.ylabel("Estresse Financeiro (1 = Baixo, 5 = Alto)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Calcular médias de estresse financeiro para cada grupo de depressão
financial_stress_mean = df_financial_stress.groupby('Depression')['Financial Stress'].mean()

# Aplicar teste estatístico (teste t) para verificar diferenças significativas
financial_stress_ttest = ttest_ind(
    df_financial_stress[df_financial_stress['Depression'] == 0]['Financial Stress'],
    df_financial_stress[df_financial_stress['Depression'] == 1]['Financial Stress'],
    equal_var=False
)

# Exibir resultados das médias e testes estatísticos
financial_stress_results = {
    "Média de Financial Stress (Sem Depressão)": financial_stress_mean[0],
    "Média de Financial Stress (Com Depressão)": financial_stress_mean[1],
    "p-valor Financial Stress": financial_stress_ttest.pvalue,
}

financial_stress_results


# Contar a distribuição de depressão entre profissionais e estudantes
occupation_depression_counts = df.groupby(['Working Professional or Student', 'Depression']).size().unstack()

# Criar gráfico de barras para visualizar a relação entre profissão/estudante e depressão
plt.figure(figsize=(6, 4))
occupation_depression_counts.plot(kind='bar', stacked=True, figsize=(6, 4), colormap="coolwarm")
plt.title("Profissionais vs Estudantes: Incidência de Depressão")
plt.xlabel("Ocupação")
plt.ylabel("Número de Pessoas")
plt.legend(["Sem Depressão", "Com Depressão"], title="Depressão")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Aplicar teste qui-quadrado para verificar associação entre ocupação e depressão
chi2_occupation, p_occupation, dof, expected = chi2_contingency(occupation_depression_counts.fillna(0))

# Exibir resultados do teste estatístico
occupation_results = {
    "p-valor Ocupação vs Depressão": p_occupation,
    "Estatística Qui-Quadrado": chi2_occupation,
}

occupation_results


# Criar faixas etárias para análise
bins = [18, 25, 35, 45, 55, 65]
labels = ["18-25", "26-35", "36-45", "46-55", "56-65"]
df["Age Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)

# Contar a distribuição de depressão por faixa etária
age_depression_counts = df.groupby(["Age Group", "Depression"]).size().unstack()

# Criar gráfico de barras para visualizar a relação entre idade e depressão
plt.figure(figsize=(8, 5))
age_depression_counts.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="coolwarm")
plt.title("Prevalência da Depressão por Faixa Etária")
plt.xlabel("Faixa Etária")
plt.ylabel("Número de Pessoas")
plt.legend(["Sem Depressão", "Com Depressão"], title="Depressão")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Aplicar teste qui-quadrado para verificar associação entre idade e depressão
chi2_age, p_age, dof, expected = chi2_contingency(age_depression_counts.fillna(0))

# Exibir resultados do teste estatístico
age_results = {
    "p-valor Idade vs Depressão": p_age,
    "Estatística Qui-Quadrado": chi2_age,
}

age_results



# Contar a distribuição de depressão por gênero
gender_depression_counts = df.groupby(["Gender", "Depression"]).size().unstack()

# Criar gráfico de barras para visualizar a relação entre gênero e depressão
plt.figure(figsize=(6, 4))
gender_depression_counts.plot(kind="bar", stacked=True, figsize=(6, 4), colormap="coolwarm")
plt.title("Prevalência da Depressão por Gênero")
plt.xlabel("Gênero")
plt.ylabel("Número de Pessoas")
plt.legend(["Sem Depressão", "Com Depressão"], title="Depressão")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Aplicar teste qui-quadrado para verificar associação entre gênero e depressão
chi2_gender, p_gender, dof, expected = chi2_contingency(gender_depression_counts.fillna(0))

# Exibir resultados do teste estatístico
gender_results = {
    "p-valor Gênero vs Depressão": p_gender,
    "Estatística Qui-Quadrado": chi2_gender,
}

gender_results


# Converter a variável 'Have you ever had suicidal thoughts ?' para numérico (0 = No, 1 = Yes)
df['Suicidal Thoughts'] = df['Have you ever had suicidal thoughts ?'].map({'No': 0, 'Yes': 1})

# Selecionar variáveis numéricas para análise de correlação
numerical_cols = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 
                  'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours', 
                  'Financial Stress', 'Depression']

# Remover valores nulos
df_corr_suicidal = df[numerical_cols + ['Suicidal Thoughts']].dropna()

# Calcular correlação entre fatores e pensamentos suicidas
correlation_suicidal = df_corr_suicidal.corr()['Suicidal Thoughts'].sort_values(ascending=False)

# Verificar quantos valores não nulos existem na coluna 'Suicidal Thoughts'
suicidal_thoughts_valid = df['Suicidal Thoughts'].notnull().sum()

# Verificar valores únicos na coluna 'Suicidal Thoughts'
unique_suicidal_values = df['Suicidal Thoughts'].unique()

# Verificar quantos valores não nulos existem nas colunas numéricas selecionadas
valid_counts_suicidal = df[numerical_cols].notnull().sum()

# Remover linhas com valores ausentes apenas das colunas relevantes
df_corr_suicidal = df[numerical_cols + ['Suicidal Thoughts']].dropna()

# Calcular correlação entre fatores e pensamentos suicidas
correlation_suicidal = df_corr_suicidal.corr()['Suicidal Thoughts'].sort_values(ascending=False)

# Contar a distribuição de pensamentos suicidas
suicidal_thoughts_counts = df['Suicidal Thoughts'].value_counts()

# Exibir a distribuição
suicidal_thoughts_counts


# Gráfico de barras para visualizar a relação entre depressão e pensamentos suicidas
plt.figure(figsize=(12, 5))
sns.countplot(x=df['Suicidal Thoughts'], hue=df['Depression'], palette="coolwarm")
plt.title("Depressão vs Pensamentos Suicidas")
plt.xlabel("Pensamentos Suicidas (0 = Não, 1 = Sim)")
plt.ylabel("Número de Pessoas")
plt.legend(["Sem Depressão", "Com Depressão"], title="Depressão")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


# 1. Indicador de prevalência de pensamentos suicidas entre pessoas com depressão
total_depressed = df[df['Depression'] == 1].shape[0]
total_suicidal_depressed = df[(df['Depression'] == 1) & (df['Suicidal Thoughts'] == 1)].shape[0]

suicidal_rate_depressed = (total_suicidal_depressed / total_depressed) * 100

fig1 = go.Figure(go.Indicator(
    mode="gauge+number",
    value=suicidal_rate_depressed,
    title={"text": "Pessoas com Depressão que Tiveram Pensamentos Suicidas (%)"},
    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "red"}}
))

# 2. Indicador de relação entre estresse financeiro e pensamentos suicidas
high_stress_suicidal = df[(df['Financial Stress'] >= 4) & (df['Suicidal Thoughts'] == 1)].shape[0]
total_high_stress = df[df['Financial Stress'] >= 4].shape[0]

suicidal_rate_high_stress = (high_stress_suicidal / total_high_stress) * 100

fig2 = go.Figure(go.Indicator(
    mode="gauge+number",
    value=suicidal_rate_high_stress,
    title={"text": "Pessoas com Alto Estresse Financeiro que Tiveram Pensamentos Suicidas (%)"},
    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "blue"}}
))

# Exibir gráficos
fig1.show()
fig2.show()


# Recalcular as taxas antes de recriar os gráficos

# Pensamentos suicidas entre pessoas com privação de sono (<5h)
low_sleep_suicidal = df[(df['Sleep Duration Cleaned'] == '<5h') & (df['Suicidal Thoughts'] == 1)].shape[0]
total_low_sleep = df[df['Sleep Duration Cleaned'] == '<5h'].shape[0]
suicidal_rate_low_sleep = (low_sleep_suicidal / total_low_sleep) * 100 if total_low_sleep > 0 else 0

# Pensamentos suicidas entre pessoas com histórico familiar de doenças mentais
family_history_suicidal = df[(df['Family History of Mental Illness'] == 'Yes') & (df['Suicidal Thoughts'] == 1)].shape[0]
total_family_history = df[df['Family History of Mental Illness'] == 'Yes'].shape[0]
suicidal_rate_family_history = (family_history_suicidal / total_family_history) * 100 if total_family_history > 0 else 0

# Criar gráficos de pizza
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Indicador 1: Pensamentos Suicidas entre pessoas com privação de sono
labels_sleep = ['Com Pensamentos Suicidas', 'Sem Pensamentos Suicidas']
sizes_sleep = [suicidal_rate_low_sleep, 100 - suicidal_rate_low_sleep]
colors_sleep = ['purple', 'lightgray']
axes[0].pie(sizes_sleep, labels=labels_sleep, autopct='%1.1f%%', colors=colors_sleep, startangle=90)
axes[0].set_title("Pensamentos Suicidas entre Pessoas com Poucas Horas de Sono")

# Indicador 2: Pensamentos Suicidas entre pessoas com histórico familiar de doenças mentais
labels_family = ['Com Pensamentos Suicidas', 'Sem Pensamentos Suicidas']
sizes_family = [suicidal_rate_family_history, 100 - suicidal_rate_family_history]
colors_family = ['green', 'lightgray']
axes[1].pie(sizes_family, labels=labels_family, autopct='%1.1f%%', colors=colors_family, startangle=90)
axes[1].set_title("Pensamentos Suicidas entre Pessoas com Histórico Familiar de Doenças Mentais")

plt.tight_layout()
plt.show()


# 5. Separar features (X) e alvo (y)
X_train = df_treino.drop(target_column, axis=1)
y_train = df_treino[target_column]


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer

def aplicar_preprocessamento(X_treino, X_teste, colunas_numericas, colunas_categoricas):
    """
    Aplica pré-processamento aos conjuntos de treino e teste utilizando pipelines para
    features numéricas e categóricas.
    
    Parâmetros:
        X_treino (DataFrame ou array): Conjunto de treino.
        X_teste (DataFrame ou array): Conjunto de teste.
        colunas_numericas (list): Lista com os nomes das colunas numéricas.
        colunas_categoricas (list): Lista com os nomes das colunas categóricas.
    
    Retorna:
        X_treino_proc, X_teste_proc: Dados de treino e teste após o pré-processamento.
    """
    
    # Pipeline para features numéricas
    pipeline_numerico = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('to_float32', FunctionTransformer(lambda x: x.astype(np.float32)))
    ])
    
    # Pipeline para features categóricas
    pipeline_categorico = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OrdinalEncoder(dtype=np.int32, handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    # Combinar os pipelines utilizando ColumnTransformer
    processador = ColumnTransformer(transformers=[
        ('num', pipeline_numerico, colunas_numericas),
        ('cat', pipeline_categorico, colunas_categoricas)
    ])
    
    # Aplicar transformações
    X_treino_proc = processador.fit_transform(X_treino)
    X_teste_proc = processador.transform(X_teste)
    
    return X_treino_proc, X_teste_proc


# Supondo que df_treino seja o DataFrame completo com a coluna 'Depression'
# e df_teste seja o DataFrame de teste (que não possui a coluna 'Depression')
target = "Depression"

# Separe as features e o alvo no treino
X_train = df_treino.drop(columns=[target])
y_train = df_treino[target]


# Defina as listas de colunas com base em X_train
numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()


# (Opcional) Remova 'Depression' se estiver presente (não deve ocorrer se você já removeu)
if target in numerical_cols:
    numerical_cols.remove(target)
if target in categorical_cols:
    categorical_cols.remove(target)


# Agora, aplique o pré-processamento
X_train_preprocessed, X_test_preprocessed = aplicar_preprocessamento(X_train, df_teste, numerical_cols, categorical_cols)


# 7. Detecção de Outliers com Isolation Forest
from sklearn.ensemble import IsolationForest

def remover_outliers(X, y, contamination=0.04, random_state=42):
    """
    Aplica Isolation Forest para detectar e remover outliers.
    
    Parâmetros:
      - X: Dados de treino (features), geralmente um array ou DataFrame.
      - y: Rótulos correspondentes aos dados de treino.
      - contamination: Proporção estimada de outliers no conjunto de dados.
      - random_state: Semente para reprodutibilidade.
      
    Retorna:
      - X_filtrado: Dados de treino sem os outliers.
      - y_filtrado: Rótulos correspondentes aos dados filtrados.
      - mask: Um array booleano indicando quais registros foram mantidos (True para dados não-outliers).
    """
    # Cria o modelo IsolationForest com os parâmetros fornecidos
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    
    # Ajusta o modelo e obtém as previsões (-1 para outliers, 1 para pontos normais)
    pred_labels = iso_forest.fit_predict(X)
    
    # Cria uma máscara para selecionar apenas os registros que NÃO são outliers
    mask = pred_labels != -1
    
    # Filtra X e y usando a máscara
    X_filtrado = X[mask]
    y_filtrado = y[mask]
    
    return X_filtrado, y_filtrado, mask


# Exemplo de uso:
# Suponha que X_train_preprocessed seja o conjunto de features do treino e y_train os rótulos
X_train_clean, y_train_clean, indices_validos = remover_outliers(X_train_preprocessed, y_train, contamination=0.04, random_state=rs)


# Visualizando linha colunas antes 
print("Tamanho original:", X_train_preprocessed.shape)
print("Tamanho após remoção de outliers:", X_train_clean.shape)


# Remover outliers (rótulo -1 indica outlier)

def filtrar_outliers_preprocess(X, y, contamination=0.04, random_state=42):
    """
    Remove outliers dos dados utilizando o Isolation Forest.
    
    Parâmetros:
      - X: Conjunto de features (pode ser um array ou DataFrame).
      - y: Conjunto de rótulos correspondentes.
      - contamination: Proporção estimada de outliers no conjunto.
      - random_state: Semente para reprodutibilidade.
    
    Retorna:
      - X_filtrado: Conjunto X sem os outliers.
      - y_filtrado: Conjunto y correspondente sem os outliers.
    """
    # Cria e ajusta o modelo Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    outlier_labels = iso_forest.fit_predict(X)
    
    # Cria a máscara: True para registros que não são outliers (rótulo != -1)
    mask = outlier_labels != -1
    
    # Filtra X e y com a máscara
    X_filtrado = X[mask]
    y_filtrado = y[mask]
    
    return X_filtrado, y_filtrado


# Exemplo de uso:
# Suponha que X_train_preprocessed seja seu conjunto de features pré-processado e y_train os rótulos.
X_train_filtrado, y_train_filtrado = filtrar_outliers_preprocess(X_train_preprocessed, y_train, contamination=0.04, random_state=rs)

# Cópia dados
df_treino_data = df_treino.to_csv("dataset_limpo2.csv")

print("Tamanho original de X_train:", X_train_preprocessed.shape)
print("Tamanho após remoção de outliers:", X_train_filtrado.shape)


# 8. Configuração e Treinamento dos Modelos

# Parâmetros para XGBoost
xgb_params = {'learning_rate': 0.298913248058474, 
              'max_depth': 9, 
              'min_child_weight': 3, 
              'n_estimators': 673, 
              'subsample': 0.5933970249700855, 
              'gamma': 2.597137534750985, 
              'reg_lambda': 0.11328048420927406, 
              'colsample_bytree': 0.1381203919800721}

# Parâmetros para CatBoost
catboost_params = {'iterations': 145, 
                   'depth': 7, 
                   'learning_rate': 0.29930179265937246, 
                   'l2_leaf_reg': 1.242352421942431, 
                   'random_strength': 8.325681754379957, 
                   'bagging_temperature': 0.7869848919618048, 
                   'border_count': 139}

# Parâmetros para HistGradientBoosting
hgb_params = {'learning_rate': 0.16299202834206894, 
              'max_iter': 250, 
              'max_depth': 4, 
              'l2_regularization': 7.1826466833939895,
              'early_stopping': True}

# Modelo machine learning XGBoost
xgb_model = XGBClassifier(**xgb_params, 
                          use_label_encoder=False, 
                          random_state=rs)

# Modelo machine learning CatBoost
catboost_model = CatBoostClassifier(**catboost_params, 
                                    task_type="GPU", 
                                    random_state=rs, 
                                    verbose=0)

# Modelo machine learning Hist Gradient Boosting
hgb_model = HistGradientBoostingClassifier(**hgb_params, 
                                           random_state=rs)

# 6. Treinamento dos Modelos e Ensemble via Stacking

# Criar ensemble via Stacking
stacking_ensemble = StackingClassifier(estimators=[('catboost', catboost_model),

                                                   #
                                                   ('xgb', xgb_model),

                                                   #
                                                   ('hgb', hgb_model)],

                                       #
                                       final_estimator=LogisticRegression(),
                                       passthrough=False,
                                       cv=5)

# Validação Cruzada usando Accuracy Score
scoring = make_scorer(accuracy_score)
cv_scores = cross_val_score(stacking_ensemble, X_train_preprocessed, y_train, cv=5, scoring=scoring)

print("\nScores da Validação Cruzada:")
print(cv_scores)
print(f"Acurácia Média: {cv_scores.mean():.4f}")
print(f"Desvio Padrão: {cv_scores.std():.4f}")


# Treinar o ensemble com todo o conjunto de treino (após remoção dos outliers)
stacking_ensemble.fit(X_train_preprocessed, y_train)


from catboost import Pool

def plot_xgb_feature_importance(xgb_model, feature_names, model_name="XGBoost", importance_type="gain", figsize=(12,6)):
    """
    Plota a importância das features de um modelo XGBoost utilizando o método do booster.
    
    Parâmetros:
      - xgb_model: Modelo XGBoost treinado.
      - feature_names (list): Lista com os nomes das features (na ordem em que foram usadas no treinamento).
      - model_name (str): Nome do modelo, usado no título do gráfico.
      - importance_type (str): Tipo de importância a ser extraída ('gain', 'weight', 'cover', etc.).
      - figsize (tuple): Tamanho da figura.
    """
    # Obter o booster do modelo XGBoost
    booster = xgb_model.get_booster()
    
    # Obter as importâncias (retorna um dicionário com chaves como "f0", "f1", etc.)
    importance_dict = booster.get_score(importance_type=importance_type)
    
    # Para cada feature (assumindo que elas foram transformadas na ordem original), 
    # extraímos a importância ou atribuímos 0 caso não apareça no dicionário.
    importances = np.array([importance_dict.get(f"f{i}", 0.0) for i in range(len(feature_names))])
    
    # Ordenar as importâncias do maior para o menor
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=figsize)
    plt.title(f"Importância das Features - {model_name} ({importance_type})")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Importância")
    plt.tight_layout()
    plt.show()

#
def plot_catboost_feature_importance(cat_model, feature_names, X_data, model_name="CatBoost", figsize=(12,6)):
    """
    Plota a importância das features para um modelo CatBoost.
    
    Parâmetros:
      - cat_model: Modelo CatBoost treinado (CatBoostClassifier ou CatBoostRegressor).
      - feature_names (list): Lista com os nomes das features (na mesma ordem utilizada no treinamento).
      - X_data: Dados de treinamento (ou conjunto de features) a serem usados para extrair a importância.
                Esses dados serão convertidos para um Pool.
      - model_name (str): Nome do modelo, usado no título do gráfico.
      - figsize (tuple): Tamanho da figura do gráfico.
    """
    # Converter os dados para um objeto Pool do CatBoost
    pool_data = Pool(X_data)
    
    # Extração das importâncias utilizando o Pool
    importancias = cat_model.get_feature_importance(data=pool_data)
    
    # Ordena as importâncias (do maior para o menor)
    indices = np.argsort(importancias)[::-1]
    
    plt.figure(figsize=figsize)
    plt.title(f"Importância das Features - {model_name}")
    plt.bar(range(len(importancias)), importancias[indices], align="center")
    plt.xticks(range(len(importancias)), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Importância")
    plt.tight_layout()
    plt.show()

# Exemplo de uso:
# Suponha que 'numerical_cols' e 'categorical_cols' foram definidos previamente e
# que o pré-processamento não alterou a ordem das features.
feature_names = numerical_cols + categorical_cols

# Treine o modelo XGBoost
xgb_model = xgb_model.fit(X_train_preprocessed, y_train)

# Supondo que 'stacking_ensemble' seja seu ensemble já treinado:
stacking_ensemble.fit(X_train_preprocessed, y_train)

# Extraia o modelo CatBoost ajustado
catboost_fitted = stacking_ensemble.named_estimators_["catboost"]


# Exemplo de uso:
# Supondo que 'xgb_model' seja o seu modelo XGBoost treinado
# e que 'feature_names' seja uma lista com os nomes das features (por exemplo, numerical_cols + categorical_cols).

# Ajuste conforme seu pipeline
feature_names = numerical_cols + categorical_cols  
feature_names


# Plot para XGBoost (xgb_model)
xgb_fitted = stacking_ensemble.named_estimators_["xgb"]
plot_xgb_feature_importance(xgb_fitted, feature_names, model_name="XGBoost", importance_type="gain")


# Plote as importâncias usando o modelo ajustado e os dados de treinamento (convertidos para Pool)
plot_catboost_feature_importance(catboost_fitted, feature_names, X_train_preprocessed, model_name="CatBoost")


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc

# Crie um dicionário com os modelos que deseja avaliar
# Se você treinou o ensemble (stacking_ensemble) e deseja avaliar também os modelos base,
# Modelos ajustados do ensemble, se necessário.
modelos = {"HistGradientBoosting": hgb_model,
           "XGBoost": xgb_model,
           "CatBoost": catboost_model,
           "Ensemble": stacking_ensemble  # opcional, se desejar plotar o ensemble também
          }

plt.figure(figsize=(10,8))

# Para cada modelo, usamos cross_val_predict para obter as probabilidades preditas de forma out-of-sample
for nome, modelo in modelos.items():
    
    # cross_val_predict com cv=5 e método 'predict_proba'
    # A saída terá duas colunas (para classe 0 e 1); 
    # selecionamos a coluna da classe positiva (1)
    y_proba = cross_val_predict(modelo, X_train_preprocessed, y_train, cv=5, method="predict_proba")[:, 1]
    
    # Calcula os pontos da curva ROC e a AUC
    fpr, tpr, _ = roc_curve(y_train, y_proba)
    area = auc(fpr, tpr)
    
    # Plota a curva ROC
    plt.plot(fpr, tpr, lw=2, label=f"{nome} (AUC = {area:.2f})")

# Linha de referência (chance)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Chance")

plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.title("Curvas ROC dos Modelos")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

# Gerar as previsões out-of-sample utilizando validação cruzada
y_pred = cross_val_predict(stacking_ensemble, X_train_preprocessed, y_train, cv=5)

# Calcular a matriz de confusão
cm = confusion_matrix(y_train, y_pred)

# Definir os rótulos desejados (ordem: classe 0 = "Não Depressão", classe 1 = "Depressão")
rotulos = ["Não Depressão", "Depressão"]

# Plotar a matriz de confusão utilizando seaborn
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=rotulos, yticklabels=rotulos)
plt.xlabel("Previsões")
plt.ylabel("Valores Reais")
plt.title("Matriz de Confusão")
plt.show()


from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict

# Gerar as previsões out-of-sample utilizando validação cruzada
y_pred = cross_val_predict(stacking_ensemble, X_train_preprocessed, y_train, cv=5)

# Gerar o classification report, definindo os nomes das classes
report = classification_report(y_train, y_pred, target_names=["Não Depressão", "Depressão"])

print("Classification Report:\n")
print(report)


# Fazer previsões no conjunto de teste
test_preds = stacking_ensemble.predict(X_test_preprocessed)


# 7. Geração do arquivo de submissão

# Use os índices do DataFrame como identificadores
submission = pd.DataFrame({'id': df_teste.index, 'Depression': test_preds})
submission.head(n=8)


submission.to_csv('submission1.csv', index=False)
print("\nArquivo de submissão salvo como 'submission.csv'.")
print(submission.head(n=10))


import lightgbm as lgb

# Defina os parâmetros para o modelo LightGBM com suporte para GPU
lgb_params = {'learning_rate': 0.1,         # Taxa de aprendizado
              'n_estimators': 200,          # Número de iterações/árvores
              'num_leaves': 31,             # Número máximo de folhas em cada árvore
              'max_depth': -1,              # -1 significa sem limite de profundidade
              'subsample': 0.8,             # Proporção de amostras para cada árvore (bootstrap)
              'colsample_bytree': 0.8,      # Proporção de features para cada árvore
              'random_state': rs,           # Semente para reprodutibilidade
              'device': 'gpu'               # Usa GPU (certifique-se de que seu LightGBM tem suporte para GPU)
             }

# Instancie o modelo LightGBM utilizando os parâmetros definidos
lgb_model = lgb.LGBMClassifier(**lgb_params)

# Treine o modelo com os dados pré-processados
lgb_model.fit(X_train_preprocessed,
              y_train,
              eval_set=[(X_train_preprocessed, y_train)],  # Idealmente, use um conjunto de validação separado
              eval_metric='logloss',                         # Métrica a ser monitorada
              callbacks=[lgb.log_evaluation(10)]              # Imprime logs a cada 10 iterações
             )



# Faça previsões no conjunto de teste (ou em outro conjunto de interesse)
y_pred_lgb = lgb_model.predict(X_test_preprocessed)


# Obter as probabilidades preditas para a classe positiva (índice 1) usando validação cruzada (cv=5)
y_proba_lgb = cross_val_predict(lgb_model, X_train_preprocessed, y_train, cv=5, method="predict_proba")[:, 1]

# Calcular os pontos da curva ROC e a área sob a curva (AUC)
fpr, tpr, thresholds = roc_curve(y_train, y_proba_lgb)
roc_auc = auc(fpr, tpr)

# Plotar a curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'LightGBM (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.title("Curva ROC - LightGBM")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# Obter as previsões out-of-sample utilizando validação cruzada (cv=5)
y_pred_lgb = cross_val_predict(lgb_model, X_train_preprocessed, y_train, cv=5)

# Calcular a matriz de confusão
cm_lgb = confusion_matrix(y_train, y_pred_lgb)

# Definir os rótulos desejados: classe 0 = "Não Depressão", classe 1 = "Depressão"
rotulos = ["Não Depressão", "Depressão"]

# Plotar a matriz de confusão utilizando o seaborn
plt.figure(figsize=(10, 5))
sns.heatmap(cm_lgb, annot=True, fmt="d", cmap="Blues",
            xticklabels=rotulos, yticklabels=rotulos)
plt.xlabel("Previsões")
plt.ylabel("Valores Reais")
plt.title("Matriz de Confusão - LightGBM")
plt.tight_layout()
plt.show()


# Caso queira avaliar o desempenho no conjunto de treino:
y_pred_train = lgb_model.predict(X_train_preprocessed)
print("Acurácia no conjunto de treino:", accuracy_score(y_train, y_pred_train))
print("\nClassification Report no conjunto de treino:")
print(classification_report(y_train, y_pred_train, target_names=["Não Depressão", "Depressão"]))


# Leia o arquivo de teste original (ou uma cópia) para obter os ids
df_teste_original = pd.read_csv("/kaggle/input/analyze-the-insights-over-mental-health-data/test.csv")

# Extraia os IDs exatamente como estão
test_ids = df_teste_original['id']

# Deve imprimir 93800
print("Número de IDs:", test_ids.shape[0])  

test_ids = test_ids.astype(int)


# Supondo que 'test_preds' seja o array (ou Series) com as previsões para o conjunto de teste:
submission2 = pd.DataFrame({'id': test_ids,           # Use os IDs originais do arquivo de teste
                            'Depression': test_preds  # Previsões geradas pelo seu modelo
                           })

# Visualize as primeiras linhas do arquivo de submissão
submission2.head(n=10)


# Salve o arquivo CSV para submissão (sem índice)
submission2.to_csv('submission_LightGBM.csv', index=False)
print("\nArquivo de submissão salvo como 'submission.csv'.")

print(submission2.head(n=10))


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# Definir o número de features de entrada (input_dim)
input_dim = X_train_preprocessed.shape[1]


# Construindo um modelo com mais camadas e mais neurônios
model = Sequential([
    
                    # Camada de entrada e primeira camada oculta com 256 neurônios
                    Dense(256, activation='relu', input_shape=(input_dim,)), 
                    Dropout(0.3),  # 30% de dropout
                    
                    # Segunda camada oculta com 128 neurônios
                    Dense(128, activation='relu'),
                    Dropout(0.3),
                    
                    # Terceira camada oculta com 128 neurônios
                    Dense(128, activation='relu'),
                    Dropout(0.3),
                    
                    # Quarta camada oculta com 64 neurônios
                    Dense(64, activation='relu'),
                    Dropout(0.2),
                    
                    # Quinta camada oculta com 32 neurônios
                    Dense(32, activation='relu'),
                    Dropout(0.2),
                    
                    # Camada de saída para classificação binária
                    Dense(1, activation='sigmoid')])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Configurar Early Stopping para evitar overfitting (monitora a perda no conjunto de validação)
#early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Summary model
model.summary()


from tensorflow.keras.utils import plot_model
import matplotlib.image as mpimg

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

img = mpimg.imread('model.png')
plt.figure(figsize=(20, 20))
plt.imshow(img)
plt.axis('off')
plt.show()


# Treinar o modelo (aqui, usamos 20% dos dados para validação)
history = model.fit(X_train_preprocessed,
                    y_train,
                    epochs=100,             # Número máximo de épocas (pode ser ajustado)
                    batch_size=32,          # Tamanho do batch (pode ser ajustado)
                    validation_split=0.2,   # 20% dos dados usados para validação
                    verbose=1)              # callbacks=[early_stop]


# Plot da Loss (treino e validação)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss Treino')
plt.plot(history.history['val_loss'], label='Loss Validação')
plt.title('Evolução da Loss')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.grid(False)

# Plot da Acurácia (treino e validação)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Acurácia Treino')
plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
plt.title('Evolução da Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(False)

plt.tight_layout()
plt.show()


# Obter as probabilidades preditas para a classe positiva a partir do modelo Keras.
# model.predict retorna um array com shape (n_amostras, 1) para classificação binária.
y_pred_prob_nn = model.predict(X_train_preprocessed)

# Converte para vetor unidimensional
y_pred_prob_nn = y_pred_prob_nn.flatten()  

# Calcular a curva ROC e a área sob a curva (AUC)
fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_train, y_pred_prob_nn)
roc_auc_nn = auc(fpr_nn, tpr_nn)

# Plotar a curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr_nn, tpr_nn, color='darkorange', lw=2, label=f'Rede Neural (AUC = {roc_auc_nn:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.title("Curva ROC - Rede Neural")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# Fazer previsões no conjunto de treino (ou use um conjunto de teste/validação para avaliação)
y_pred_prob = model.predict(X_train_preprocessed)

# Converter as probabilidades para classes utilizando limiar de 0.5
y_pred = (y_pred_prob > 0.5).astype(int).flatten()


# As previsões são probabilidades; convertê-las para 0 ou 1 usando o limiar de 0.5:
y_pred_nn = (y_pred_prob > 0.5).astype(int).flatten()


# Calcular a acurácia 
acc = accuracy_score(y_train, y_pred)
print("Acurácia no conjunto de treino:", acc)


# Calcular exibir classification report
report = classification_report(y_train, y_pred, target_names=["Não Depressão", "Depressão"])
print("\nClassification Report:\n", report)


# Calcular a matriz de confusão comparando os rótulos reais (y_train) com as previsões
cm_nn = confusion_matrix(y_train, y_pred_nn)

# Definir os rótulos desejados (assumindo que 0 = "Não Depressão" e 1 = "Depressão")
rotulos = ["Não Depressão", "Depressão"]

# Plotar a matriz de confusão com Seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(cm_nn, annot=True, fmt="d", cmap="Blues",
            xticklabels=rotulos, yticklabels=rotulos)
plt.xlabel("Previsões")
plt.ylabel("Valores Reais")
plt.title("Matriz de Confusão - Rede Neural")
plt.tight_layout()
plt.show()


# Obter as probabilidades preditas para o conjunto de teste
y_test_prob = model.predict(X_test_preprocessed)

# Converter as probabilidades para classes (utilizando limiar 0.5)
y_test_pred = (y_test_prob > 0.5).astype(int).flatten()

# Verifique o número de previsões (deve ser igual ao número de IDs, por exemplo, 93.800)
print("Número de previsões:", len(y_test_pred))


# Crie o DataFrame de submissão utilizando os IDs originais e as previsões

# test_ids deve conter os IDs originais do arquivo de teste
submission_nn = pd.DataFrame({'id': test_ids, 'Depression': y_test_pred})

# Visualize as primeiras linhas do arquivo de submissão
submission_nn.head(n=10)


# Salve o arquivo CSV para submissão (sem incluir o índice)
submission_nn.to_csv('submission_nn.csv', index=False)
print("\nArquivo de submissão salvo como 'submission_nn.csv'.")

print(submission_nn.head(n=15))


import joblib

# Salvar cada modelo individualmente
joblib.dump(xgb_model, 'xgb_model.pkl')
print("Modelo XGBoost salvo como 'xgb_model.pkl'.")

joblib.dump(catboost_model, 'catboost_model.pkl')
print("Modelo CatBoost salvo como 'catboost_model.pkl'.")

joblib.dump(hgb_model, 'hgb_model.pkl')
print("Modelo HistGradientBoosting salvo como 'hgb_model.pkl'.")

# Salvar o ensemble via Stacking
joblib.dump(stacking_ensemble, 'stacking_ensemble.pkl')
print("Modelo Ensemble (Stacking) salvo como 'stacking_ensemble.pkl'.")

# Salvar o modelo LightGBM
joblib.dump(lgb_model, 'lightgbm_model.pkl')
print("Modelo LightGBM salvo como 'lightgbm_model.pkl'.")


# Modelo rede neural
model.save("modelo_rede_neural.h5")


import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Suponha que você já tenha treinado o seu Tokenizer:
tokenizer = Tokenizer(num_words=10000)
# ... ajuste o tokenizer com tokenizer.fit_on_texts(textos)

# Para salvar o tokenizer:
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Tokenizer salvo como 'tokenizer.pickle'")

# Carregar o tokenizer salvo
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
print("Tokenizer carregado com sucesso!")


# Carregar o modelo de classificação de texto salvo (formato H5)
text_model = load_model('/kaggle/working/modelo_rede_neural.h5')


# Defina o tamanho máximo da sequência (deve ser o mesmo usado durante o treinamento)
max_length = 23  # ajuste para o valor que seu modelo espera

def prever_texto(texto):
    sequencia = tokenizer.texts_to_sequences([texto])
    sequencia_pad = pad_sequences(sequencia, maxlen=max_length, padding='post', truncating='post')
    
    # Obter a probabilidade predita
    proba = text_model.predict(sequencia_pad)[0, 0]
    
    # Se os rótulos estiverem invertidos, por exemplo, 0 = Depressão, 1 = Não Depressão,
    # você pode inverter a lógica. Exemplo:

    # Agora, se proba for menor que 0.5, preveja "Depressão"
    prediction = int(proba < 0.5)  
    
    if prediction == 1:
        return f"O Rede Neural diz: Depressão (probabilidade = {1-proba:.2f})"
    else:
        return f"O Rede Neural diz: Não Depressão (probabilidade = {proba:.2f})"

def interativo_prever_texto():
    texto_input = input("Digite sua frase: \n")
    resultado = prever_texto(texto_input)
    print("\nResultado:", resultado)


# Teste a função interativa
#interativo_prever_texto()


# Teste a função interativa
#interativo_prever_texto()




