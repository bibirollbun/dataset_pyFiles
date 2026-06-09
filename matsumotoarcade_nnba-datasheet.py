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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
import warnings
warnings.filterwarnings('ignore')

# --- 1. Funções Auxiliares ---

def load_and_preprocess(file_path, is_target=False):
    """Carrega e pré-processa os dados, criando a variável alvo, se necessário."""
    df = pd.read_csv(file_path)
    if is_target:
        # Cria variável alvo (1 se WTeamID ganhou, 0 caso contrário)
        df['Target'] = (df['WScore'] > df['LScore']).astype(int)
    return df

def create_id(row):
    """Cria o ID da submissão no formato correto."""
    return f"{row['Season']}_{row['Team1ID']}_{row['Team2ID']}"

# --- 2. Carregamento dos Dados ---
DATA_PATH = '/kaggle/input/march-machine-learning-mania-2025/'

tourney_results = load_and_preprocess(DATA_PATH + 'MNCAATourneyCompactResults.csv', is_target=True)
regular_results = load_and_preprocess(DATA_PATH + 'MRegularSeasonCompactResults.csv', is_target=False)
teams = pd.read_csv(DATA_PATH + 'MTeams.csv')
seeds = pd.read_csv(DATA_PATH + 'MNCAATourneySeeds.csv')

# --- 3. Pré-processamento e Engenharia de Variáveis ---

# 3.1. Tratamento do arquivo de Seeds
seeds['SeedNum'] = seeds['Seed'].apply(lambda x: int(x[1:3]))
seeds = seeds.rename(columns={'TeamID': 'Team1ID', 'SeedNum': 'Team1Seed'})  # Rename *AFTER* loading and extracting SeedNum

# ***CRITICAL: Create a SEPARATE copy for the second merge***
seeds2 = seeds.rename(columns={'Team1ID': 'Team2ID', 'Team1Seed': 'Team2Seed'})


# 3.2. Criação do Dataset de Treino (Invertendo os jogos)
df_wins = pd.DataFrame()
df_wins[['Season', 'Team1ID', 'Team2ID', 'Target']] = tourney_results[['Season', 'WTeamID', 'LTeamID', 'Target']]

df_losses = pd.DataFrame()
df_losses[['Season', 'Team1ID', 'Team2ID']] = tourney_results[['Season', 'LTeamID', 'WTeamID']]  # Inverte
df_losses['Target'] = 0  # Time 1 perdeu (LTeamID)

df_games = pd.concat((df_wins, df_losses)).reset_index(drop=True)

# 3.3. Merge com Seeds  -- ***USE SEPARATE DATAFRAMES***
df_games = pd.merge(df_games, seeds, how='left', on=['Season', 'Team1ID'])  # First merge
df_games = pd.merge(df_games, seeds2, how='left', on=['Season', 'Team2ID']) # Second merge, using the *COPY*


# 3.4. Engenharia de Variáveis (Temporada Regular)
# Exemplo: Média de Pontos Marcados e Sofridos na Temporada Regular

# Pontos marcados
win_pts = regular_results.groupby(['Season', 'WTeamID'])['WScore'].mean().reset_index()
lose_pts = regular_results.groupby(['Season', 'LTeamID'])['LScore'].mean().reset_index()
team_season_pts = pd.concat([win_pts.rename(columns={'WTeamID':'TeamID','WScore':'Pts'}),
                            lose_pts.rename(columns={'LTeamID':'TeamID','LScore':'Pts'})])
team_season_pts = team_season_pts.groupby(['Season', 'TeamID'])['Pts'].mean().reset_index()

#Pontos sofridos
win_pts_a = regular_results.groupby(['Season', 'WTeamID'])['LScore'].mean().reset_index()
lose_pts_a = regular_results.groupby(['Season', 'LTeamID'])['WScore'].mean().reset_index()
team_season_pts_against = pd.concat([win_pts_a.rename(columns={'WTeamID':'TeamID','LScore':'PtsAgainst'}),
                                    lose_pts_a.rename(columns={'LTeamID':'TeamID','WScore':'PtsAgainst'})])
team_season_pts_against = team_season_pts_against.groupby(['Season', 'TeamID'])['PtsAgainst'].mean().reset_index()


# Merge das estatísticas da temporada regular
df_games = pd.merge(df_games, team_season_pts.rename(columns={'TeamID':'Team1ID'}), how='left', on=['Season', 'Team1ID'])
df_games = pd.merge(df_games, team_season_pts.rename(columns={'TeamID':'Team2ID'}), how='left', on=['Season', 'Team2ID'])

df_games = pd.merge(df_games, team_season_pts_against.rename(columns={'TeamID':'Team1ID'}), how='left', on=['Season', 'Team1ID'])
df_games = pd.merge(df_games, team_season_pts_against.rename(columns={'TeamID':'Team2ID'}), how='left', on=['Season', 'Team2ID'])


# 3.5. Feature Engineering Adicional (Exemplos)
df_games['SeedDiff'] = df_games['Team1Seed'] - df_games['Team2Seed']
df_games['PtsDiff'] = df_games['Pts_x'] - df_games['Pts_y']
df_games['PtsAgainstDiff'] = df_games['PtsAgainst_x'] - df_games['PtsAgainst_y']


# 3.6. Tratamento de NAs (preenche com a média - poderia ser algo mais sofisticado)
for col in ['Team1Seed', 'Team2Seed', 'Pts_x', 'Pts_y', 'PtsAgainst_x', 'PtsAgainst_y', 'SeedDiff','PtsDiff', 'PtsAgainstDiff']:
    df_games[col].fillna(df_games[col].mean(), inplace=True)


# --- 4. Preparação para Modelagem ---
features = ['Season', 'Team1Seed', 'Team2Seed', 'SeedDiff', 'Pts_x', 'Pts_y','PtsAgainst_x','PtsAgainst_y','PtsDiff','PtsAgainstDiff']
X = df_games[features]
y = df_games['Target']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# --- 5. Validação Cruzada e Treinamento ---

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
brier_scores = []

for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}/{n_splits}")
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model = LogisticRegression(random_state=42, solver='liblinear', C=0.1)
    model.fit(X_train, y_train)

    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_index] = val_preds
    brier_score = brier_score_loss(y_val, val_preds)
    brier_scores.append(brier_score)
    print(f"Brier Score (Fold {fold+1}): {brier_score:.4f}")

print(f'\nBrier Score Médio: {np.mean(brier_scores):.4f}')

# --- 6. Preparação da Submissão ---

# 6.1. Get ALL 2025 Teams (from MNCAATourneySeeds.csv)
teams_2025 = seeds[seeds['Season'] == 2025]['Team1ID'].unique() # Now uses the correct 'Team1ID'

# 6.2. Generate ALL Possible Matchups
submission_data = []
for team1_id in teams_2025:
    for team2_id in teams_2025:
        if team1_id < team2_id:  # Ensure Team1ID < Team2ID
            submission_data.append({
                'Season': 2025,
                'Team1ID': team1_id,
                'Team2ID': team2_id,
                'ID': f"2025_{team1_id}_{team2_id}"
            })

submission_df = pd.DataFrame(submission_data)

# 6.3.  Merge with the features (Seeds, etc.)  -- ***USE SEPARATE DATAFRAMES***
submission_df = pd.merge(submission_df, seeds, how='left', on=['Season', 'Team1ID'])  # First merge
submission_df = pd.merge(submission_df, seeds2, how='left', on=['Season', 'Team2ID']) # Second merge, using the *COPY*

#Merge das estatísticas da temporada regular
submission_df = pd.merge(submission_df, team_season_pts.rename(columns={'TeamID':'Team1ID'}), how='left', on=['Season', 'Team1ID'])
submission_df = pd.merge(submission_df, team_season_pts.rename(columns={'TeamID':'Team2ID'}), how='left', on=['Season', 'Team2ID'])
submission_df = pd.merge(submission_df, team_season_pts_against.rename(columns={'TeamID':'Team1ID'}), how='left', on=['Season', 'Team1ID'])
submission_df = pd.merge(submission_df, team_season_pts_against.rename(columns={'TeamID':'Team2ID'}), how='left', on=['Season', 'Team2ID'])

# Feature Engineering
submission_df['SeedDiff'] = submission_df['Team1Seed'] - submission_df['Team2Seed']
submission_df['PtsDiff'] = submission_df['Pts_x'] - submission_df['Pts_y']
submission_df['PtsAgainstDiff'] = submission_df['PtsAgainst_x'] - submission_df['PtsAgainst_y']

#Tratamento de NAs
for col in ['Team1Seed', 'Team2Seed', 'Pts_x', 'Pts_y', 'PtsAgainst_x', 'PtsAgainst_y', 'SeedDiff','PtsDiff', 'PtsAgainstDiff']:
    submission_df[col].fillna(submission_df[col].mean(), inplace=True)


# 6.4.  Previsões no conjunto de submissão
X_submission = submission_df[features]
X_submission = scaler.transform(X_submission)

# Train the final model on ALL available training data
final_model = LogisticRegression(random_state=42, solver='liblinear', C=0.1)
final_model.fit(X, y)

submission_preds = final_model.predict_proba(X_submission)[:, 1]
submission_df['Pred'] = submission_preds

# 6.5.  Salvar o arquivo de submissão
final_submission_df = submission_df[['ID', 'Pred']]
final_submission_df.to_csv('submission.csv', index=False)
print("Arquivo submission.csv criado com sucesso!")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Carregar o arquivo de submissão ---
try:
    submission_df = pd.read_csv('submission.csv')
except FileNotFoundError:
    print("Erro: O arquivo 'submission.csv' não foi encontrado. Certifique-se de que o código principal foi executado e o arquivo foi criado.")
    exit()

# --- 1. Ordenar por Predição (DECRESCENTE para "maior pontuação") ---
# A competição usa Brier Score, que é um *loss*, então queremos MINIMIZAR.
# No entanto, o prompt pede "maior pontuação".  Isso é um pouco confuso,
# mas, no contexto de *previsões*, "maior pontuação" geralmente significa
# maior *confiança* na previsão.
#
# Portanto, para previsões *próximas de 1*, uma "maior pontuação" significa
# um valor de 'Pred' *mais alto*.
# Para previsões *próximas de 0*, uma "maior pontuação" significa
# um valor de 'Pred' *mais baixo*.
#
# Como não temos como saber *a priori* quais jogos serão "fáceis" (perto de 0 ou 1),
# a melhor interpretação do pedido é mostrar os 55 jogos com as predições
# *mais extremas*, ou seja, mais distantes de 0.5.  Isso é equivalente a
# ordenar por abs(Pred - 0.5) em ordem *decrescente*.

submission_df['Abs_Diff_From_0.5'] = abs(submission_df['Pred'] - 0.5)
top_55_scores = submission_df.sort_values(by='Abs_Diff_From_0.5', ascending=False).head(55)

# --- 2. Criação da Tabela Dinâmica ---
print("Tabela dos 55 Melhores Resultados (Predições Mais Extremas):")
display(top_55_scores[['ID', 'Pred', 'Abs_Diff_From_0.5']])


# --- 3. Visualização Gráfica ---

# 3.1.  Predições vs. ID (Ordenado por Abs_Diff_From_0.5)
plt.figure(figsize=(12, 6))
plt.bar(top_55_scores['ID'], top_55_scores['Pred'], color='skyblue')
plt.axhline(y=0.5, color='r', linestyle='--', label='0.5 (Incerteza Máxima)')  # Linha de referência
plt.title('Top 55 Predições Mais Extremas')
plt.xlabel('ID do Jogo')
plt.ylabel('Predição (Probabilidade)')
plt.xticks(rotation=90, ha='right')  # Rotação para melhor legibilidade
plt.legend()
plt.tight_layout()
plt.show()

# 3.2.  Histograma das Predições dos Top 55
plt.figure(figsize=(10, 6))
sns.histplot(top_55_scores['Pred'], bins=20, kde=True, color='skyblue') #Usar kde
plt.title('Distribuição das Predições (Top 55)')
plt.xlabel('Predição (Probabilidade)')
plt.ylabel('Frequência')
plt.grid(axis='y', alpha=0.75)
plt.show()

# 3.3. Scatter Plot (Abs_Diff_From_0.5 vs. Pred)
plt.figure(figsize=(8, 6))
plt.scatter(top_55_scores['Pred'], top_55_scores['Abs_Diff_From_0.5'], alpha=0.7)
plt.title('Distância de 0.5 vs. Predição (Top 55)')
plt.xlabel('Predição (Probabilidade)')
plt.ylabel('Distância Absoluta de 0.5')
plt.grid(True)
plt.show()

# --- 4. Explicação e Fórmulas ---

print("\n--- Explicação e Fórmulas ---")

print("\n**1. Predição (Pred):**")
print("   - A coluna 'Pred' representa a probabilidade prevista de que o time com o menor ID vença o jogo.")
print("   - Essa probabilidade é gerada pelo modelo de Regressão Logística treinado.")
print("   - Um valor de Pred próximo a 1 indica alta confiança de que o time com menor ID vencerá.")
print("   - Um valor de Pred próximo a 0 indica alta confiança de que o time com maior ID vencerá.")
print("   - Um valor de Pred próximo a 0.5 indica incerteza máxima (o modelo considera que ambos os times têm chances iguais).")

print("\n**2. Distância Absoluta de 0.5 (Abs_Diff_From_0.5):**")
print("   - Esta coluna mede o quão 'extrema' é a previsão, ou seja, o quão distante ela está da incerteza máxima (0.5).")
print("   - Fórmula:  `Abs_Diff_From_0.5 = abs(Pred - 0.5)`")
print("   - Um valor alto de Abs_Diff_From_0.5 indica uma previsão mais confiante (seja para vitória do time com menor ID ou maior ID).")
print("   - Um valor baixo indica uma previsão próxima a 0.5, ou seja, alta incerteza.")

print("\n**3. Ordenação e Seleção dos Top 55:**")
print("   - O DataFrame foi ordenado em ordem *decrescente* de `Abs_Diff_From_0.5`. Isso significa que os jogos com as predições")
print("     mais confiantes (mais distantes de 0.5) aparecem primeiro.")
print("   - Os 55 primeiros jogos (os com maiores `Abs_Diff_From_0.5`) foram selecionados para exibição na tabela e nos gráficos.")

print("\n**4. Interpretação dos Gráficos:**")
print("   - **Predições vs. ID:**  Mostra as predições (probabilidades) para cada um dos 55 jogos selecionados, ordenados por")
print("     sua 'confiança' (distância de 0.5).  A linha tracejada em 0.5 representa o ponto de incerteza máxima.")
print("   - **Histograma:**  Mostra a distribuição das probabilidades previstas para os 55 jogos.  Permite visualizar se há")
print("     mais previsões próximas a 0, a 1, ou se estão mais distribuídas.")
print("   - **Scatter Plot:**  Relaciona a predição ('Pred') com a sua distância absoluta de 0.5 ('Abs_Diff_From_0.5').")
print("     Este gráfico ajuda a visualizar a relação entre a confiança da previsão e o valor da probabilidade.")

print("\n**5. Brier Score (Não Usado Diretamente Aqui, Mas É a Métrica da Competição):**")
print("  - O Brier Score mede a precisão das previsões probabilísticas.")
print("  - Fórmula: `Brier Score = (1/N) * Σ (Pred_i - Actual_i)^2`")
print("      - `N` é o número de jogos.")
print("      - `Pred_i` é a probabilidade prevista para o jogo i.")
print("      - `Actual_i` é o resultado real do jogo i (1 se o time com menor ID venceu, 0 caso contrário).")
print("  - O Brier Score é uma *loss function*, o que significa que um valor *menor* é melhor (mais preciso).")
print("  - Embora não calculemos o Brier Score aqui (porque não temos os resultados reais dos jogos de 2025),")
print("    a ordenação por `Abs_Diff_From_0.5` é uma forma de *aproximar* a ideia de 'melhores resultados',")
print("    dado que previsões mais extremas (mais distantes de 0.5) *tendem* a ter um Brier Score melhor, *se estiverem corretas*. ")

