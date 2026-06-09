# ===================================================================
# 1. MANIPULAÃ‡ÃƒO DE DADOS E OPERAÃ‡Ã•ES NUMÃ‰RICAS
# ===================================================================
import pandas as pd  # Para manipulaÃ§Ã£o de dataframes
import numpy as np   # Para operaÃ§Ãµes numÃ©ricas e arrays

# ===================================================================
# 2. SISTEMA DE ARQUIVOS E UTILITÃ�RIOS
# ===================================================================
import os          # Para interagir com o sistema operacional
import glob        # Para encontrar caminhos de arquivo que correspondem a um padrÃ£o
import gc          # Para gerenciamento de memÃ³ria (coletor de lixo)
import warnings    # Para controlar mensagens de aviso
import re          # Para operaÃ§Ãµes com expressÃµes regulares
import pyarrow.parquet as pq  # Para trabalhar com arquivos parquet

# ConfiguraÃ§Ã£o de warnings
warnings.filterwarnings('ignore') # Oculta avisos para um output mais limpo

# ===================================================================
# 3. VISUALIZAÃ‡ÃƒO DE DADOS
# ===================================================================
import matplotlib.pyplot as plt # Biblioteca base para visualizaÃ§Ãµes
import seaborn as sns           # Biblioteca de alto nÃ­vel para grÃ¡ficos estatÃ­sticos
from matplotlib.ticker import FuncFormatter  # Para formataÃ§Ã£o de eixos
from matplotlib.patches import Patch  # Para personalizaÃ§Ã£o de legendas

# ===================================================================
# 4. PRÃ‰-PROCESSAMENTO E MODELAGEM DE MACHINE LEARNING
# ===================================================================
from sklearn.model_selection import (StratifiedKFold, TimeSeriesSplit, 
                                    train_test_split) # Para validaÃ§Ã£o cruzada e divisÃ£o de dados
from sklearn.metrics import (roc_auc_score, classification_report, 
                            ConfusionMatrixDisplay, confusion_matrix)  # MÃ©tricas de avaliaÃ§Ã£o
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # Para codificaÃ§Ã£o de variÃ¡veis categÃ³ricas
from sklearn.linear_model import (LinearRegression, 
                                 LogisticRegression) # Modelos lineares

# LightGBM para Gradient Boosting
import lightgbm as lgb 

# ===================================================================
# 5. OTIMIZAÃ‡ÃƒO DE HIPERPARÃ‚METROS
# ===================================================================
# Optuna para otimizaÃ§Ã£o Bayesiana 
import optuna

# ===================================================================
# 6. INTERPRETABILIDADE DO MODELO (XAI - IA ExplicÃ¡vel)
# ===================================================================
# SHAP para explicar as previsÃµes do modelo 
import shap

# ===================================================================
# 7. BARRA DE PROGRESSO (Opcional, mas Ãºtil)
# ===================================================================
from tqdm.notebook import tqdm # Para visualizar o progresso de loops longos

# --- ConfiguraÃ§Ãµes Iniciais ---
# ConfiguraÃ§Ã£o para exibir todas as colunas do pandas
pd.set_option('display.max_columns', None)

print("Todas as bibliotecas foram importadas com sucesso.")
print("Estamos prontos para iniciar a modelagem preditiva, incluindo o modelo Baseline.")


# 1. Definir o caminho para o diretÃ³rio com os arquivos CSV de treino
CSV_TRAIN_PATH = '/kaggle/input/home-credit-credit-risk-model-stability/csv_files/train'

# 2. DicionÃ¡rio de termos de busca e suas descriÃ§Ãµes
search_terms = {
    # --- Identificadores e Alvo ---
    'id': 'identificador Ãºnico do caso (case_id)',
    'target': 'variÃ¡vel alvo (target)',
    'week': 'semana da aplicaÃ§Ã£o (WEEK_NUM)',
    
    # --- Dados da AplicaÃ§Ã£o e Cliente ---
    'inc': 'renda principal (mainoccupationinc)',
    'amount': 'valor do crÃ©dito (credamount)',
    'pmt': 'valor da entrada/pagamento inicial (downpmt)',
    'type': 'tipo de moradia (housetype)',
    
    # --- HistÃ³rico de CrÃ©dito ---
    'dpd': 'dias em atraso (days past due)',
    'overdue': 'valor em atraso (overdue amount)'
}

print("="*80)
print(f"Iniciando varredura individual dos cabeÃ§alhos em: {CSV_TRAIN_PATH}")
print("="*80)

# 3. Obter a lista de todos os arquivos .csv no diretÃ³rio
try:
    csv_files = glob.glob(os.path.join(CSV_TRAIN_PATH, "*.csv"))
    if not csv_files:
        print(f"AVISO: Nenhum arquivo .csv foi encontrado no diretÃ³rio especificado.")
except Exception as e:
    print(f"Erro ao acessar o diretÃ³rio: {e}")
    csv_files = []

# 4. Loop principal para iterar sobre cada arquivo encontrado
for file_path in sorted(csv_files):
    """
    Processa cada arquivo CSV encontrado no diretÃ³rio especificado, buscando colunas
    que correspondam aos termos definidos no dicionÃ¡rio search_terms.
    
    ParÃ¢metros implÃ­citos:
    ---------------------
    file_path : str
        Caminho completo para o arquivo CSV sendo processado
        
    Comportamento:
    --------------
    - Extrai o nome base do arquivo para exibiÃ§Ã£o amigÃ¡vel
    - LÃª apenas o cabeÃ§alho do CSV (sem carregar dados) para eficiÃªncia
    - Verifica cada termo de busca contra os nomes das colunas
    - Exibe resultados encontrados ou mensagem informativa
    - Captura e trata possÃ­veis erros de leitura de arquivo
    """
    file_name = os.path.basename(file_path)
    print(f"\n--- Verificando o arquivo: [{file_name}] ---")

    try:
        # Leitura eficiente: apenas cabeÃ§alhos (0 linhas de dados)
        df_header = pd.read_csv(file_path, nrows=0)
        columns = df_header.columns.tolist()

        found_something_in_file = False

        # Busca inteligente: verifica cada termo no dicionÃ¡rio
        for term, description in search_terms.items():
            # Busca case-insensitive por termos nas colunas
            found_columns = [col for col in columns if term.lower() in col.lower()]

            if found_columns:
                print(f"  > Termo '{term}' ({description}) encontrado nas colunas: {found_columns}")
                found_something_in_file = True

        if not found_something_in_file:
            print("  > Nenhum dos termos de busca foi encontrado neste arquivo.")

    except Exception as e:
        print(f"  > Ocorreu um erro ao processar o arquivo {file_name}: {e}")

print("\n" + "="*80)
print("Varredura de arquivos concluÃ­da.")
print("="*80)

# ==============================================================================
# 5. CLÃ�USULA DE LIMPEZA DE DADOS (WORKSPACE CLEANUP)
# ==============================================================================
# ApÃ³s a varredura, removemos as variÃ¡veis temporÃ¡rias usadas no loop
# para garantir um ambiente de trabalho limpo e liberar memÃ³ria.

try:
    # Deleta as variÃ¡veis que foram usadas dentro do loop
    del file_path, file_name, df_header, columns
    del found_something_in_file, term, description, found_columns
    
    # ForÃ§a o "coletor de lixo" do Python a limpar a memÃ³ria
    gc.collect()
    
    print("\nClÃ¡usula de limpeza executada: VariÃ¡veis temporÃ¡rias removidas da memÃ³ria.")

except NameError:
    # Este bloco Ã© executado caso o loop nÃ£o tenha sido iniciado (nenhum arquivo encontrado)
    print("\nNenhuma variÃ¡vel temporÃ¡ria para limpar.")



# FunÃ§Ãµes de ajuda
def find_col(df, keyword):
    """Encontra o nome completo de uma coluna em um dataframe a partir de uma palavra-chave."""
    for col in df.columns:
        if keyword in col: return col
    return None

def reduce_mem_usage(df, name=""):
    """Reduz o uso de memÃ³ria de um dataframe."""
    df.name = name
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Uso de memÃ³ria de "{name}" antes: {start_mem:.2f} MB')
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type.name != 'category':
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max: df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32)
                else: df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max: df[col] = df[col].astype(np.float32)
                else: df[col] = df[col].astype(np.float64)
        elif col_type.name != 'category': df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Uso de memÃ³ria de "{name}" depois: {end_mem:.2f} MB. ReduÃ§Ã£o de {100 * (start_mem - end_mem) / start_mem:.1f}%')
    return df


# ==============================================================================
# SPLIT 1: CARREGAMENTO OTIMIZADO DOS ARQUIVOS GERAIS
# ==============================================================================
print("--- INICIANDO SPLIT 1: Carregamento OTIMIZADO ---")

DATA_PATH = "/kaggle/input/home-credit-credit-risk-model-stability/parquet_files/train"

# --- MAPA DE COLUNAS INDISPENSÃ�VEIS ---
# Mapeia o termo de busca para um conceito de negÃ³cio. Usaremos as chaves para buscar.
INDISPENSABLE_KEYWORDS = {
    'case_id': 'ID', 'target': 'Alvo', 'WEEK_NUM': 'Semana',
    'credamount': 'Valor do CrÃ©dito', 'downpmt': 'Entrada',
    'mainoccupationinc': 'Renda', 'housetype': 'Moradia'
    # NÃ£o incluÃ­mos 'pmts_dpd' e 'pmts_overdue' aqui, pois eles serÃ£o tratados no Split 2
}

def get_relevant_columns(file_path, keywords_map):
    """LÃª o esquema de um arquivo Parquet e retorna uma lista de colunas relevantes."""
    try:
        # Leitura eficiente apenas do esquema, sem carregar os dados
        schema = pq.read_schema(file_path)
        all_columns = schema.names
        
        columns_to_load = []
        for keyword in keywords_map.keys():
            for col in all_columns:
                if keyword in col and col not in columns_to_load:
                    columns_to_load.append(col)
        return columns_to_load
    except Exception:
        return []

dataframes = {}

try:
    all_parquet_files = glob.glob(os.path.join(DATA_PATH, "*.parquet"))
    files_to_load = [path for path in all_parquet_files if not re.search(r'train_credit_bureau_a_2_\d+\.parquet$', path)]
    
    print(f"Encontrados {len(files_to_load)} arquivos para carregar nesta etapa.")

    for file_path in sorted(files_to_load):
        file_name = os.path.basename(file_path)
        base_name = "df_" + re.sub(r'_\d+\.parquet$', '.parquet', file_name).replace('.parquet', '')
        
        # Identifica as colunas a serem carregadas ANTES de ler o arquivo todo
        columns_to_load = get_relevant_columns(file_path, INDISPENSABLE_KEYWORDS)

        if columns_to_load: # SÃ³ processa se encontrar colunas relevantes
            print(f"\nCarregando colunas de {file_name} -> {base_name}")
            try:
                temp_df = pd.read_parquet(file_path, columns=columns_to_load)
                
                if base_name not in dataframes:
                    dataframes[base_name] = temp_df
                else:
                    dataframes[base_name] = pd.concat([dataframes[base_name], temp_df], ignore_index=True)
                
                print(f"  > Sucesso! Shape atual de {base_name}: {dataframes[base_name].shape}")

            except Exception as e:
                print(f"  > ERRO ao carregar dados de {file_name}: {e}")

except Exception as e:
    print(f"Erro geral ao tentar listar os arquivos: {e}")

globals().update(dataframes)

print("\n" + "="*80)
print("SPLIT 1 (OTIMIZADO) CONCLUÃ�DO!")
for name in sorted(dataframes.keys()):
    if name in locals() or name in globals():
        print(f"- {name} (Shape: {dataframes[name].shape})")
print("="*80)


# ==============================================================================
# SPLIT 2: CARREGAMENTO OTIMIZADO DOS ARQUIVOS DE HISTÃ“RICO DE PAGAMENTO
# ==============================================================================
print("\n--- INICIANDO SPLIT 2: Carregamento OTIMIZADO do Bureau ---")

DATA_PATH = "/kaggle/input/home-credit-credit-risk-model-stability/parquet_files/train"
FILE_PATTERN_ESSENCIAL = "train_credit_bureau_a_2_*.parquet"

# --- MAPA DE COLUNAS INDISPENSÃ�VEIS PARA O BUREAU ---
BUREAU_KEYWORDS = {'case_id': 'ID', 'pmts_dpd': 'Dias em Atraso', 'pmts_overdue': 'Valor em Atraso'}

# Reutilizando a funÃ§Ã£o 'get_relevant_columns'
# def get_relevant_columns(file_path, keywords_map): ...

dfs_list = []
search_pattern = os.path.join(DATA_PATH, FILE_PATTERN_ESSENCIAL)
file_paths = sorted(glob.glob(search_pattern))

print(f"\nEncontrados {len(file_paths)} arquivos essenciais do bureau para processar...")

for file_path in file_paths:
    file_name = os.path.basename(file_path)
    
    # Identifica as colunas ANTES de carregar
    columns_to_load = get_relevant_columns(file_path, BUREAU_KEYWORDS)

    if len(columns_to_load) > 1: # Se encontrou mais do que apenas case_id
        print(f"Carregando {columns_to_load} de {file_name}")
        try:
            temp_df = pd.read_parquet(file_path, columns=columns_to_load)
            dfs_list.append(temp_df)
        except Exception as e:
            print(f"  > ERRO ao carregar dados de {file_name}: {e}")
    else:
        print(f"Nenhuma coluna relevante encontrada em {file_name}, pulando.")

if dfs_list:
    df_credit_bureau_a_2 = pd.concat(dfs_list, ignore_index=True)
    print(f"\n> Grupo de dados do bureau combinado com sucesso. Shape final: {df_credit_bureau_a_2.shape}")
else:
    df_credit_bureau_a_2 = pd.DataFrame()
    print("\nNenhum dado do bureau foi carregado.")
    
print("\n" + "="*80)
print("SPLIT 2 (OTIMIZADO) CONCLUÃ�DO!")
if 'df_credit_bureau_a_2' in locals() and not df_credit_bureau_a_2.empty:
    print("O DataFrame 'df_credit_bureau_a_2' foi criado e estÃ¡ pronto para uso.")
else:
    print("O DataFrame 'df_credit_bureau_a_2' nÃ£o pÃ´de ser criado.")
print("="*80)


# Supondo que o DataFrame 'df_credit_bureau_a_2' foi carregado pelo Split 2

print("--- INICIANDO ETAPA DE AGREGAÃ‡ÃƒO ---")
print(f"Shape do df_credit_bureau_a_2 ANTES da agregaÃ§Ã£o: {df_credit_bureau_a_2.shape}")

# --- Selecionando as colunas que vamos agregar ---
# Usando a funÃ§Ã£o find_col que definimos anteriormente para pegar os nomes reais
col_dpd = find_col(df_credit_bureau_a_2, 'pmts_dpd')
col_overdue = find_col(df_credit_bureau_a_2, 'pmts_overdue')

# Lista de colunas para a agregaÃ§Ã£o
agg_cols = ['case_id']
if col_dpd:
    agg_cols.append(col_dpd)
if col_overdue:
    agg_cols.append(col_overdue)

# --- Agrupando por case_id e aplicando a funÃ§Ã£o de agregaÃ§Ã£o ---
if len(agg_cols) > 1:
    # Agrupamos por 'case_id' e calculamos o valor MÃ�XIMO para cada coluna numÃ©rica
    bureau_agg = df_credit_bureau_a_2[agg_cols].groupby('case_id').max().reset_index()

    # Renomeando as colunas para refletir a agregaÃ§Ã£o (boa prÃ¡tica)
    bureau_agg.rename(columns={
        col_dpd: 'dias_atraso_max',
        col_overdue: 'valor_atraso_max'
    }, inplace=True)
    
    print(f"\nShape do bureau_agg DEPOIS da agregaÃ§Ã£o: {bureau_agg.shape}")
    print("AgregaÃ§Ã£o do histÃ³rico de pagamentos concluÃ­da com sucesso.")
    display(bureau_agg.head())
else:
    print("\nAVISO: Nenhuma das colunas de agregaÃ§Ã£o foi encontrada. 'bureau_agg' nÃ£o foi criado.")


# (Opcional) Limpeza de memÃ³ria, jÃ¡ que nÃ£o precisaremos mais do dataframe gigante
# import gc
# del df_credit_bureau_a_2
# gc.collect()



import pandas as pd
import numpy as np
import gc

# Supondo que todos os DataFrames do Split 1 e 2 estÃ£o carregados.

print("--- INICIANDO ETAPA DE JUNÃ‡ÃƒO (MERGE) - VERSÃƒO COM AGREGAÃ‡ÃƒO EXPLÃ�CITA ---")

# ==============================================================================
# 1. DEFINIÃ‡ÃƒO DA BASE
# ==============================================================================
df_analise = df_train_base.copy()
print(f"Shape inicial (df_train_base): {df_analise.shape}")


# ==============================================================================
# 2. FUNÃ‡ÃƒO DE AJUDA PARA AGREGAÃ‡ÃƒO INTELIGENTE
# ==============================================================================
def aggregate_dataframe(df, df_name):
    """
    Agrega um dataframe, aplicando 'max' a colunas numÃ©ricas e 'first' a colunas categÃ³ricas.
    """
    print(f"\nAgregando {df_name}...")
    
    # Identifica os tipos de colunas
    numeric_cols = df.select_dtypes(include=np.number).columns.drop('case_id', errors='ignore')
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Cria o dicionÃ¡rio de agregaÃ§Ã£o
    agg_spec = {}
    for col in numeric_cols:
        agg_spec[col] = 'max'
    for col in categorical_cols:
        agg_spec[col] = 'first'
        
    if not agg_spec:
        print(f"  - AVISO: Nenhuma coluna para agregar encontrada em {df_name}.")
        return pd.DataFrame(columns=['case_id'])

    # Aplica a agregaÃ§Ã£o
    df_agg = df.groupby('case_id').agg(agg_spec).reset_index()
    print(f"  - {df_name} agregado. Shape: {df_agg.shape}")
    return df_agg

# ==============================================================================
# 3. AGREGAÃ‡ÃƒO E JUNÃ‡ÃƒO EM SEQUÃŠNCIA
# ==============================================================================
# Agregando cada dataframe antes do merge
static_agg = aggregate_dataframe(df_train_static_0, "df_train_static_0")
person_agg = aggregate_dataframe(df_train_person, "df_train_person")
bureau_agg = aggregate_dataframe(df_credit_bureau_a_2, "df_credit_bureau_a_2")
# Adicione aqui a agregaÃ§Ã£o de outros dataframes se necessÃ¡rio (ex: df_applprev_1)

print("\nJuntando os dataframes agregados...")

# Agora, cada merge Ã© seguro pois os dataframes agregados tÃªm case_id's Ãºnicos
df_analise = df_analise.merge(static_agg, on='case_id', how='left')
print(f"Shape apÃ³s merge com static_agg: {df_analise.shape}")

df_analise = df_analise.merge(person_agg, on='case_id', how='left')
print(f"Shape apÃ³s merge com person_agg: {df_analise.shape}")

df_analise = df_analise.merge(bureau_agg, on='case_id', how='left')
print(f"Shape apÃ³s merge com bureau_agg: {df_analise.shape}")

# --- Limpeza de MemÃ³ria ---
del df_train_base, df_train_person, df_train_static_0, df_credit_bureau_a_2
del static_agg, person_agg, bureau_agg
gc.collect()

# ==============================================================================
# 4. VERIFICAÃ‡ÃƒO FINAL
# ==============================================================================
print("\n--- JUNÃ‡ÃƒO DE DADOS CONCLUÃ�DA ---")
print(f"Shape final de df_analise: {df_analise.shape}")

duplicatas = df_analise['case_id'].duplicated().sum()
if duplicatas == 0:
    print("SUCESSO: NÃ£o hÃ¡ case_id's duplicados no DataFrame final.")
else:
    print(f"AVISO: Ainda existem {duplicatas} case_id's duplicados. A lÃ³gica precisa ser revista.")

display(df_analise.head())



# Supondo que o seu DataFrame 'df_analise' jÃ¡ estÃ¡ carregado com as 14 colunas.

print("--- INICIANDO ETAPA DE RENOMEAÃ‡ÃƒO DE COLUNAS ---")

# --- MAPA DE RENOMEAÃ‡ÃƒO ---
# Criamos um dicionÃ¡rio que mapeia o nome antigo para o nome novo.
# A seleÃ§Ã£o dos nomes foi baseada na sua lista e no feature_definitions.csv.
mapa_nomes = {
    # Colunas Base (mantidas ou ajustadas)
    'WEEK_NUM': 'semana_num',
    
    # Colunas da AplicaÃ§Ã£o Atual
    'credamount_770A': 'valor_credito_atual',
    'disbursedcredamount_1113A': 'valor_desembolsado_atual',
    'downpmt_116A': 'valor_entrada_atual',
    
    # Colunas HistÃ³ricas da AplicaÃ§Ã£o
    'lastapprcredamount_781A': 'valor_ult_credito_aprovado',
    'lastrejectcredamount_222A': 'valor_ult_credito_rejeitado',
    
    # Colunas Pessoais
    'mainoccupationinc_384A': 'renda_principal',
    'housetype_905L': 'tipo_moradia',
    
    # Colunas de HistÃ³rico de Pagamento (Bureau)
    # Temos duas fontes para DPD e Overdue, vamos nomeÃ¡-las para diferenciÃ¡-las.
    'pmts_dpd_1073P': 'hist_dias_atraso_1',
    'pmts_dpd_303P': 'hist_dias_atraso_2',
    'pmts_overdue_1140A': 'hist_valor_atraso_1',
    'pmts_overdue_1152A': 'hist_valor_atraso_2',
}

# --- APLICANDO A RENOMEAÃ‡ÃƒO ---
# O inplace=True modifica o DataFrame diretamente
df_analise.rename(columns=mapa_nomes, inplace=True)


# --- VERIFICAÃ‡ÃƒO FINAL ---
print("\nColunas renomeadas com sucesso!")
print("\nNovas colunas do DataFrame 'df_analise':")
print(df_analise.columns.tolist())

# Exibir as primeiras linhas com os novos nomes para confirmaÃ§Ã£o visual
print("\nAmostra do DataFrame com colunas renomeadas:")
display(df_analise.head())


# --- ANÃ�LISE INICIAL DE DADOS FALTANTES ---
print("--- 1. Tratando Valores Faltantes ---")
print("Contagem de valores nulos ANTES do tratamento (Top 10):")
print(df_analise.isnull().sum().sort_values(ascending=False).head(10))

# --- ESTRATÃ‰GIA DE IMPUTAÃ‡ÃƒO ---
# Para colunas de histÃ³rico, um valor nulo (NaN) geralmente significa "nÃ£o hÃ¡ registro".
# Preencher com 0 Ã© uma suposiÃ§Ã£o de negÃ³cio segura e comum.
cols_to_fill_zero = [
    'valor_ult_credito_aprovado', 
    'valor_ult_credito_rejeitado',
    'hist_dias_atraso_1',
    'hist_dias_atraso_2',
    'hist_valor_atraso_1',
    'hist_valor_atraso_2'
]

for col in cols_to_fill_zero:
    if col in df_analise.columns:
        df_analise[col].fillna(0, inplace=True)

print("\nValores nulos em colunas de histÃ³rico e de crÃ©dito anterior foram preenchidos com 0.")
# A coluna 'tipo_moradia' serÃ¡ tratada na etapa de codificaÃ§Ã£o.


# --- CRIANDO FEATURES DE RÃ�CIO ---
print("\n--- 2. Criando Novas Features (Engenharia de Features) ---")

# Adicionamos 1 para evitar qualquer divisÃ£o por zero
df_analise['relacao_credito_renda'] = df_analise['valor_credito_atual'] / (df_analise['renda_principal'] + 1)
df_analise['percentual_entrada'] = df_analise['valor_entrada_atual'] / (df_analise['valor_credito_atual'] + 1)

# ApÃ³s a criaÃ§Ã£o, preenchemos quaisquer NaNs ou Infs que possam ter sido gerados
df_analise.replace([np.inf, -np.inf], np.nan, inplace=True) # Garante que nÃ£o hÃ¡ infinitos
df_analise['relacao_credito_renda'].fillna(0, inplace=True)
df_analise['percentual_entrada'].fillna(0, inplace=True)

print("Features de rÃ¡cio ('relacao_credito_renda', 'percentual_entrada') criadas e limpas.")


# ---  CODIFICAÃ‡ÃƒO DE VARIÃ�VEIS CATEGÃ“RICAS (TARGET ENCODING)

print("\n--- 4. Iniciando PrÃ©-processamento Final para Modelagem ---")

# --- DivisÃ£o prÃ©via dos dados para evitar Data Leakage ---
# Esta etapa garante que todas as operaÃ§Ãµes de prÃ©-processamento (imputaÃ§Ã£o, encoding)
# sejam calculadas apenas com os dados de treino.
X_temp = df_analise.drop(columns=['target', 'case_id'])
y_temp = df_analise['target']

X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)


# --- Tratamento e CodificaÃ§Ã£o da VariÃ¡vel CategÃ³rica (Target Encoding) ---
if 'tipo_moradia' in X_train.columns:
    print("\nCodificando 'tipo_moradia' com Target Encoding...")
    
    # Passo 1: Trata os valores nulos, convertendo-os em uma categoria prÃ³pria.
    # Fazemos isso em ambos os conjuntos de treino e teste.
    X_train['tipo_moradia'].fillna('NÃ£o Informado', inplace=True)
    X_test['tipo_moradia'].fillna('NÃ£o Informado', inplace=True)
    
    # Passo 2: Calcula o mapa de risco APENAS com os dados de TREINO.
    # A categoria 'NÃ£o Informado' agora serÃ¡ incluÃ­da no cÃ¡lculo.
    train_data_for_encoding = pd.concat([X_train, y_train], axis=1)
    risk_map = train_data_for_encoding.groupby('tipo_moradia')['target'].mean().to_dict()
    
    print("Mapa de Risco (Target Encoding) criado a partir dos dados de treino:")
    print(risk_map)
    
    # Passo 3: Aplica o mapa para transformar a coluna em numÃ©rica.
    # A mÃ©dia global Ã© usada como seguranÃ§a para categorias que possam aparecer no futuro.
    global_mean = y_train.mean()
    X_train['tipo_moradia_encoded'] = X_train['tipo_moradia'].map(risk_map).fillna(global_mean)
    X_test['tipo_moradia_encoded'] = X_test['tipo_moradia'].map(risk_map).fillna(global_mean)
    
    # Passo 4: Remove a coluna de texto original.
    X_train.drop(columns=['tipo_moradia'], inplace=True)
    X_test.drop(columns=['tipo_moradia'], inplace=True)
    
    print("\nColuna 'tipo_moradia' tratada e codificada com sucesso.")


# ---- LIMPEZA FINAL E VERIFICAÃ‡ÃƒO

print("\n--- 5. Realizando Limpeza Final e VerificaÃ§Ã£o ---")

# VerificaÃ§Ã£o final de nulos: preenche qualquer valor faltante com a mediana do TREINO
for col in X_train.select_dtypes(include=np.number).columns:
    if X_train[col].isnull().any():
        # Calcula a mediana APENAS no conjunto de treino
        median_val = X_train[col].median()
        # Aplica a mesma mediana em ambos os conjuntos
        X_train[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)
        print(f"Valores nulos na coluna '{col}' foram preenchidos com a mediana do treino ({median_val:.2f}).")

print("\n--- PREPARAÃ‡ÃƒO FINAL CONCLUÃ�DA ---")
print("Dados divididos e prÃ©-processados, prontos para a Fase de Modelagem.")
print(f"Shape de X_train (features de treino): {X_train.shape}")
print(f"Shape de X_test (features de teste): {X_test.shape}")


def apply_standard_style(ax, title, xlabel, ylabel, subtitle="", 
                        highlight_color='#1f77b4', grid_style=None, 
                        high_contrast=False, **kwargs):
    """
    Aplica estilo padronizado com boas prÃ¡ticas de DataViz.
    
    ParÃ¢metros:
    - highlight_color: Cor para elementos de destaque
    - grid_style: DicionÃ¡rio com configuraÃ§Ãµes customizadas do grid
    - high_contrast: Modo de alto contraste para acessibilidade
    - **kwargs: ParÃ¢metros para tick_params
    """
    # ConfiguraÃ§Ãµes de cor
    TEXT_COLOR = '#000000' if high_contrast else '#333333'
    SUBTITLE_COLOR = '#555555' if high_contrast else '#777777'
    GRID_COLOR = '#AAAAAA' if high_contrast else '#DDDDDD'
    
    # TÃ­tulo e subtÃ­tulo
    ax.set_title(title, fontsize=20, weight='bold', pad=25, 
                color=TEXT_COLOR, loc='left')
    if subtitle:
        ax.text(0, 1.05, subtitle, transform=ax.transAxes, 
                fontsize=14, style='italic', color=SUBTITLE_COLOR)
    
    # Eixos
    ax.set_xlabel(xlabel, fontsize=14, color=TEXT_COLOR, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=14, color=TEXT_COLOR, labelpad=10)
    
    # Ticks
    ax.tick_params(axis='x', labelsize=12, color=TEXT_COLOR, **kwargs)
    ax.tick_params(axis='y', labelsize=12, color=TEXT_COLOR)
    
    # Grid
    default_grid = {
        'visible': True,
        'which': 'both',
        'linestyle': '--',
        'linewidth': 0.5,
        'alpha': 0.7,
        'color': GRID_COLOR
    }
    if grid_style:
        default_grid.update(grid_style)
    ax.grid(**default_grid)
    
    # Bordas
    sns.despine(ax=ax, top=True, right=True, left=True, bottom=False)
    
    # Retorna o eixo para method chaining
    return ax


# ==============================================================================
# ANÃ�LISE EXPLORATÃ“RIA (CORRETA) - USANDO APENAS O CONJUNTO DE TREINO
# ==============================================================================
print("--- INICIANDO ANÃ�LISE EXPLORATÃ“RIA (CONJUNTO DE TREINO) ---")

# --- Criando um DataFrame temporÃ¡rio para a EDA ---
df_train_eda = pd.concat([X_train, y_train], axis=1)
print(f"DataFrame de treino para EDA criado com shape: {df_train_eda.shape}")

# ==============================================================================
# 1. AnÃ¡lise da VariÃ¡vel Alvo no Conjunto de Treino
# ==============================================================================
print("\n--- 1. AnÃ¡lise da VariÃ¡vel Alvo (target) no Conjunto de Treino ---")

# Criando a figura com tamanho adequado
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=df_train_eda, x='target', palette=['#4CAF50', '#F44336'])

# Aplicando o estilo padrÃ£o
apply_standard_style(
    ax=ax,
    title="DistribuiÃ§Ã£o da VariÃ¡vel Alvo",
    subtitle="Amostra de Treino | 0: Adimplente | 1: Inadimplente",
    xlabel="Status do CrÃ©dito",
    ylabel="Contagem de Clientes",
    rotation=0
)

# Adiciona anotaÃ§Ãµes de porcentagem formatadas
total = len(df_train_eda['target'])
for p in ax.patches:
    height = p.get_height()
    percentage = f'{100 * height / total:.1f}%'
    x = p.get_x() + p.get_width() / 2
    y = height
    ax.annotate(
        percentage, 
        (x, y), 
        ha='center', 
        va='bottom', 
        fontsize=12,
        weight='bold',
        color='#333333',
        xytext=(0, 10),
        textcoords='offset points'
    )

# Ajuste final do layout
plt.tight_layout()
plt.show()

# Verificando a estratificaÃ§Ã£o
print("\nProporÃ§Ã£o exata das classes na amostra de treino:")
print(df_train_eda['target'].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))


display(df_train_eda.tail())


display(df_train_eda.info())


df_train_eda.describe().round(2)


# Contagem de valores ausentes por coluna
df_train_eda.isnull().sum().sort_values(ascending=False)


# --- Lista das variÃ¡veis numÃ©ricas que queremos investigar ---
# Incluindo as features de rÃ¡cio que criamos, pois sÃ£o muito importantes.
variaveis_numericas_analise = [
    'renda_principal',
    'valor_credito_atual',
    'hist_dias_atraso_1',
    'hist_valor_atraso_1',
    'hist_dias_atraso_2',
    'hist_valor_atraso_2',
    'relacao_credito_renda',
    'percentual_entrada'
]

print("--- 2. AnÃ¡lise de VariÃ¡veis NumÃ©ricas vs. InadimplÃªncia (Conjunto de Treino) ---")

for var in variaveis_numericas_analise:
    # Verifica se a coluna realmente existe no dataframe
    if var in df_train_eda.columns:
        plt.figure(figsize=(12, 7))
        
        # Cria o boxplot comparando target 0 e 1
        sns.boxplot(data=df_train_eda, x='target', y=var, palette=['#4CAF50', '#F44336'])
        
        plt.title(f'DistribuiÃ§Ã£o de "{var}" vs. InadimplÃªncia (Treino)', fontsize=16, weight='bold')
        plt.ylabel(f'Valor de {var}')
        plt.xlabel('Status do CrÃ©dito (0: Adimplente | 1: Inadimplente)')
        
        # Limita o eixo Y para melhor visualizaÃ§Ã£o da "caixa", cortando outliers extremos.
        # Calculamos o limite com base na prÃ³pria variÃ¡vel para tornar o grÃ¡fico mais legÃ­vel.
        limite_superior = df_train_eda[var].quantile(0.95) # Usamos 95% para cortar menos dados
        limite_inferior = df_train_eda[var].quantile(0.01)

        # Apenas aplicamos o limite se o quantil superior for maior que o inferior
        if limite_superior > limite_inferior:
          plt.ylim(limite_inferior, limite_superior)
        
        plt.show()
    else:
        print(f"AVISO: Coluna '{var}' nÃ£o encontrada no df_train_eda.")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Supondo que 'df_train_eda' (junÃ§Ã£o de X_train e y_train) jÃ¡ existe.

# ==============================================================================
# FUNÃ‡ÃƒO PADRÃƒO DE ESTILIZAÃ‡ÃƒO (NOSSO "MANUAL DE ESTILO")
# ==============================================================================
def apply_standard_style(ax, title, xlabel, ylabel):
    """
    Aplica um conjunto de regras de estilizaÃ§Ã£o padrÃ£o a um eixo do Matplotlib.
    """
    ax.set_title(title, fontsize=18, weight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='x', labelsize=12, rotation=0) # RotaÃ§Ã£o 0 para melhor leitura
    ax.tick_params(axis='y', labelsize=12, rotation=0)
    ax.grid(False) # Remove a grade para um visual de heatmap mais limpo
    sns.despine(ax=ax, top=False, right=False, left=False, bottom=False)

# ==============================================================================
# ANÃ�LISE DE SEVERIDADE: DIAS DE ATRASO vs. VALOR EM ATRASO
# ==============================================================================
print("--- Gerando Mapa de Calor de Severidade do Risco ---")

# --- VariÃ¡veis a serem analisadas ---
col_dias = 'hist_dias_atraso_1'
col_valor = 'hist_valor_atraso_1'

if col_dias in df_train_eda.columns and col_valor in df_train_eda.columns:

    # --- 1. Filtragem: Foco nos Inadimplentes e RemoÃ§Ã£o de Outliers ---
    df_inadimplentes = df_train_eda[df_train_eda['target'] == 1].copy()

    # Filtramos outliers para focar na Ã¡rea de maior densidade de dados
    limite_dias = df_inadimplentes[col_dias].quantile(0.95)
    limite_valor = df_inadimplentes[col_valor].quantile(0.95)
    
    df_plot = df_inadimplentes[
        (df_inadimplentes[col_dias] <= limite_dias) &
        (df_inadimplentes[col_valor] <= limite_valor)
    ]
    print(f"Analisando {len(df_plot)} inadimplentes (95% dos dados, sem outliers extremos).")

    # --- 2. Binning (CriaÃ§Ã£o de Faixas) para as duas variÃ¡veis ---
    try:
        # Criando 8 faixas para cada variÃ¡vel
        df_plot['faixa_dias'] = pd.qcut(df_plot[col_dias], q=8, duplicates='drop', labels=False)
        df_plot['faixa_valor'] = pd.qcut(df_plot[col_valor], q=8, duplicates='drop', labels=False)

        # --- 3. CÃ¡lculo Cruzado: Contagem de Clientes ---
        # Agrupamos pelas duas faixas e contamos o nÃºmero de clientes em cada cÃ©lula
        contagem_cruzada = df_plot.groupby(['faixa_valor', 'faixa_dias']).size().unstack(fill_value=0)

        # Mapeando os labels para texto legÃ­vel
        labels_dias = {i: f"{int(row['min'])}-{int(row['max'])}" for i, row in df_plot.groupby('faixa_dias')[col_dias].agg(['min', 'max']).iterrows()}
        labels_valor = {i: f"{int(row['min'])}-{int(row['max'])}" for i, row in df_plot.groupby('faixa_valor')[col_valor].agg(['min', 'max']).iterrows()}
        
        contagem_cruzada.rename(index=labels_valor, columns=labels_dias, inplace=True)
    
        # --- 4. VisualizaÃ§Ã£o com Heatmap ---
        fig, ax = plt.subplots(figsize=(14, 10))
        
        sns.heatmap(contagem_cruzada,
                    annot=True,        # Mostra os nÃºmeros dentro de cada cÃ©lula
                    fmt='g',           # Formato geral para os nÃºmeros
                    cmap='YlOrRd',     # Paleta de Amarelo para Vermelho, ideal para "calor"
                    linewidths=.5,
                    linecolor='lightgray',
                    ax=ax)

        # --- 5. AplicaÃ§Ã£o do nosso PadrÃ£o de Estilo ---
        apply_standard_style(ax, 
                             title='ConcentraÃ§Ã£o de Inadimplentes por Severidade do Atraso',
                             xlabel='Faixas de Dias em Atraso',
                             ylabel='Faixas de Valor em Atraso (R$)')
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"NÃ£o foi possÃ­vel criar o grÃ¡fico. Erro: {e}")

else:
    print(f"AVISO: As colunas necessÃ¡rias nÃ£o foram encontradas no df_train_eda.")




# Supondo que 'df_train_eda' Ã© o seu DataFrame final, jÃ¡ com a coluna 'tipo_moradia_encoded'.

# ==============================================================================
# ANÃ�LISE DE INTERAÃ‡ÃƒO: RENDA vs. RISCO DE MORADIA vs. INADIMPLÃŠNCIA
# ==============================================================================
print("--- Gerando Mapa de Calor de Risco (Renda vs. Risco de Moradia) ---")

# --- VariÃ¡veis a serem analisadas ---
col_renda = 'renda_principal'
col_risco_moradia = 'tipo_moradia_encoded'

if col_renda in df_train_eda.columns and col_risco_moradia in df_train_eda.columns:

    # --- 1. PreparaÃ§Ã£o dos Dados ---
    # Criamos uma cÃ³pia para nÃ£o alterar o dataframe original da EDA
    df_plot = df_train_eda.copy()
    
    try:
        # --- 2. Binning (CriaÃ§Ã£o de Faixas) para ambas as variÃ¡veis ---
        # Criando 8 faixas de renda
        df_plot['faixa_renda'] = pd.qcut(df_plot[col_renda], q=8, duplicates='drop', labels=False)
        
        # Criando 4 nÃ­veis de risco de moradia com base nos scores da coluna encodada
        df_plot['nivel_risco_moradia'] = pd.qcut(df_plot[col_risco_moradia], q=4, duplicates='drop', labels=False)

        # --- 3. CÃ¡lculo Cruzado da Taxa de InadimplÃªncia ---
        taxa_inadimplencia_cruzada = df_plot.groupby(['faixa_renda', 'nivel_risco_moradia'])['target'].mean().unstack()

        # Mapeando os labels numÃ©ricos para texto legÃ­vel
        faixas_renda_labels = {i: f"{int(row['min'])}-{int(row['max'])}" for i, row in df_plot.groupby('faixa_renda')[col_renda].agg(['min', 'max']).iterrows()}
        niveis_risco_labels = {i: f"NÃ­vel {i+1}" for i in range(len(taxa_inadimplencia_cruzada.columns))}
        
        taxa_inadimplencia_cruzada.rename(index=faixas_renda_labels, columns=niveis_risco_labels, inplace=True)
    
        # --- 4. VisualizaÃ§Ã£o com Heatmap ---
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(taxa_inadimplencia_cruzada,
                    annot=True,
                    fmt='.1%', 
                    cmap='Reds',
                    linewidths=.5,
                    linecolor='lightgray')

        plt.title('Mapa de Calor: InadimplÃªncia por Renda e NÃ­vel de Risco da Moradia', fontsize=16, weight='bold', pad=20)
        plt.xlabel('NÃ­vel de Risco da Moradia (NÃ­vel 1 = Menor Risco)', fontsize=12)
        plt.ylabel('Faixa de Renda Principal', fontsize=12)
        plt.yticks(rotation=0)
        
        plt.show()

    except Exception as e:
        print(f"NÃ£o foi possÃ­vel criar o grÃ¡fico. Erro: {e}")

else:
    print(f"AVISO: As colunas '{col_renda}' ou '{col_risco_moradia}' nÃ£o foram encontradas no df_train_eda.")


# --- Bloco de DiagnÃ³stico ---
# Vamos verificar os nomes exatos das colunas em df_train_eda
print("Colunas disponÃ­veis no seu DataFrame 'df_train_eda':")
print(df_train_eda.columns.tolist())


print("--- AnÃ¡lise de InteraÃ§Ã£o: Renda vs. Valor do CrÃ©dito ---")


plt.figure(figsize=(14, 8))

# O scatter plot mostra cada cliente como um ponto
# 'hue' colore os pontos com base na inadimplÃªncia
# 'alpha' controla a transparÃªncia, Ãºtil para ver a densidade dos pontos
sns.scatterplot(data=df_train_eda, 
                x='renda_principal', 
                y='valor_credito_atual', 
                hue='target', 
                palette=['#4CAF50', '#F44336'],
                alpha=0.6)

plt.title('InteraÃ§Ã£o entre Renda Principal e Valor do CrÃ©dito', fontsize=16, weight='bold')
plt.xlabel('Renda Principal')
plt.ylabel('Valor do CrÃ©dito Atual')
plt.legend(title='Status do CrÃ©dito', labels=['1: Inadimplente', '0: Adimplente'])

# Usando escala de log para melhor visualizaÃ§Ã£o de dados financeiros
plt.xscale('log')
plt.yscale('log')

plt.show()


# ==============================================================================
# GRÃ�FICO DE DENSIDADE 2D (VERSÃƒO FINAL COM PRÃ‰-TRANSFORMAÃ‡ÃƒO)
# ==============================================================================
print("--- Gerando GrÃ¡fico de Densidade 2D com Eixos Corrigidos ---")

# --- VariÃ¡veis a serem analisadas ---
col_renda = 'renda_principal'
col_credito = 'valor_credito_atual'

if col_renda in df_train_eda.columns and col_credito in df_train_eda.columns:

    # --- 1. PreparaÃ§Ã£o e PrÃ©-transformaÃ§Ã£o dos Dados ---
    # Criamos uma cÃ³pia de uma amostra para a plotagem
    df_plot = df_train_eda.copy()
    
    # <<< CORREÃ‡ÃƒO: APLICAÃ‡ÃƒO MANUAL DA TRANSFORMAÃ‡ÃƒO LOGARÃ�TMICA >>>
    # Usamos np.log1p que Ã© log(1+x), uma funÃ§Ã£o segura para valores que podem ser zero.
    df_plot['log_renda'] = np.log1p(df_plot[col_renda])
    df_plot['log_credito'] = np.log1p(df_plot[col_credito])
    
    # Separamos os dataframes para plotar um de cada vez
    # df_adimplentes = df_plot[df_plot['target'] == 0]
    df_inadimplentes = df_plot[df_plot['target'] == 1]

    # --- 2. VisualizaÃ§Ã£o com o KDE Plot em Escala Linear ---
    fig, ax = plt.subplots(figsize=(14, 9))

    # Plotamos as densidades usando as NOVAS colunas transformadas, SEM o parÃ¢metro log_scale
    # sns.kdeplot(
    #     data=df_adimplentes, x='log_renda', y='log_credito',
    #     fill=True, cmap="Greens", alpha=0.6, ax=ax
    # )
    sns.kdeplot(
        data=df_inadimplentes, x='log_renda', y='log_credito',
        fill=True, cmap="Reds", alpha=0.7, ax=ax
    )

    # ==========================================================================
    # <<< CORREÃ‡ÃƒO: NOVA FUNÃ‡ÃƒO FORMATADORA PARA EIXOS >>>
    # ==========================================================================
    # Esta funÃ§Ã£o faz a operaÃ§Ã£o inversa de np.log1p, que Ã© np.expm1 (e^x - 1)
    def log_to_linear_formatter(x, pos):
        # x Ã© o valor do tick na escala log-transformada
        # np.expm1(x) converte de volta para a escala linear
        return f'R$ {np.expm1(x):,.0f}'.replace(',', '.')

    # Aplica o formatador customizado e robusto aos eixos X e Y
    ax.xaxis.set_major_formatter(FuncFormatter(log_to_linear_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(log_to_linear_formatter))
    
    # Rotaciona os labels do eixo X para nÃ£o sobrepor
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # --- 3. APLICAÃ‡ÃƒO DO NOSSO PADRÃƒO DE ESTILO ---
    # Agora os labels nÃ£o precisam mais mencionar a escala log
    apply_standard_style(ax,
                         title='ConcentraÃ§Ã£o de Risco: Renda vs. Valor do CrÃ©dito',
                         xlabel='Renda Principal',
                         ylabel='Valor do CrÃ©dito Atual')
    
    # --- 4. Legenda e Contexto ---
    legend_elements = [Patch(facecolor='#F44336', alpha=0.7, label='ConcentraÃ§Ã£o de Inadimplentes')]
    ax.legend(handles=legend_elements, title='Perfil do Cliente', loc='upper left')

    plt.tight_layout()
    plt.show()

else:
    print(f"AVISO: As colunas necessÃ¡rias nÃ£o foram encontradas no df_train_eda.")




# ==============================================================================
# ANÃ�LISE DE DENSIDADE COM VIOLIN PLOT (VERSÃƒO CORRIGIDA)
# ==============================================================================
print("--- AnÃ¡lise de Densidade com Violin Plot ---")

# --- Colunas que usaremos (os nomes finais e corretos) ---
coluna_categorica = 'tipo_moradia'
coluna_numerica = 'hist_dias_atraso_1' 

# Verifica se as colunas necessÃ¡rias existem no DataFrame
if coluna_categorica in df_analise.columns and coluna_numerica in df_analise.columns:

    # Usaremos uma amostra para o grÃ¡fico ser gerado mais rapidamente
    df_sample_violin = df_analise.sample(n=100000, random_state=42)

    # Criando uma coluna para visualizaÃ§Ã£o para nÃ£o cortar os dados originais
    # O .clip() limita os valores extremos para que o grÃ¡fico fique mais legÃ­vel
    limite_superior = df_sample_violin[coluna_numerica].quantile(0.95)
    df_sample_violin['viz_col_numerica'] = df_sample_violin[coluna_numerica].clip(upper=limite_superior)

    plt.figure(figsize=(16, 8))
    
    # Ordenando as categorias do eixo X para uma visualizaÃ§Ã£o consistente
    ordem_categorias = sorted(df_sample_violin[coluna_categorica].dropna().unique())
    
    # --- O Violin Plot ---
    # split=True divide cada violino ao meio para comparar as duas classes do 'target'
    sns.violinplot(data=df_sample_violin, 
                   x=coluna_categorica, 
                   y='viz_col_numerica', 
                   hue='target', 
                   split=True, 
                   palette=['#4CAF50', '#F44336'], # Verde para Adimplente, Vermelho para Inadimplente
                   order=ordem_categorias)

    # --- TÃ­tulos e RÃ³tulos ---
    plt.title(f'DistribuiÃ§Ã£o de "{coluna_numerica}" por "{coluna_categorica}"', fontsize=16, weight='bold')
    plt.xlabel(coluna_categorica)
    plt.ylabel(f'{coluna_numerica} (limitado para visualizaÃ§Ã£o)')
    plt.xticks(rotation=45, ha='right')
    
    # Criando a legenda manualmente para garantir os rÃ³tulos corretos
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#4CAF50', label='Adimplente (0)'),
                       Patch(facecolor='#F44336', label='Inadimplente (1)')]
    plt.legend(title='Status do CrÃ©dito', handles=legend_elements)
    
    plt.show()

else:
    print(f"AVISO: As colunas '{coluna_categorica}' ou '{coluna_numerica}' nÃ£o foram encontradas no df_analise.")


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mticker

# Supondo que 'df_analise' Ã© o seu DataFrame final e renomeado

# ==============================================================================
# GRÃ�FICO DE PIRÃ‚MIDE DE RISCO (VERSÃƒO FINAL COM LABELS)
# ==============================================================================
print("--- Gerando GrÃ¡fico de PirÃ¢mide de Risco Aprimorado ---")

# --- VariÃ¡vel a ser analisada ---
coluna_numerica = 'hist_dias_atraso_1'

if coluna_numerica in df_analise.columns:

    # --- 1. PreparaÃ§Ã£o dos Dados (sem alteraÃ§Ãµes) ---
    limite_superior = df_analise[coluna_numerica].quantile(0.95)
    df_plot = df_analise[df_analise[coluna_numerica] <= limite_superior]

    df_adimplentes = df_plot[df_plot['target'] == 0]
    df_inadimplentes = df_plot[df_plot['target'] == 1]
    
    bins = np.linspace(df_plot[coluna_numerica].min(), df_plot[coluna_numerica].max(), 30)
    counts_adimplentes, bin_edges = np.histogram(df_adimplentes[coluna_numerica].dropna(), bins=bins)
    counts_inadimplentes, _ = np.histogram(df_inadimplentes[coluna_numerica].dropna(), bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # --- 2. CriaÃ§Ã£o do GrÃ¡fico ---
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plotando os dados
    ax.barh(bin_centers, -counts_inadimplentes, height=(bin_edges[1]-bin_edges[0]), 
            color='#F44336', label='Inadimplentes (1)', alpha=0.8)
    ax.barh(bin_centers, counts_adimplentes, height=(bin_edges[1]-bin_edges[0]), 
            color='#4CAF50', label='Adimplentes (0)', alpha=0.8)

    # ==========================================================================
    # <<< MELHORIA: LABELS DESCRITIVOS E LEGENDA EXPLICATIVA >>>
    # ==========================================================================
    
    # --- RÃ³tulos dos Eixos e TÃ­tulo Aprimorados ---
    ax.set_title(f'PirÃ¢mide de Risco: Perfil de "{coluna_numerica}"', fontsize=18, weight='bold', pad=20)
    ax.set_ylabel(f"Faixas de Valor para '{coluna_numerica}'", fontsize=14)
    ax.set_xlabel("Contagem de Clientes em Cada Faixa", fontsize=14)
    
    # --- Legenda Principal (para as cores) ---
    ax.legend(title="Perfil do Cliente", loc="upper right", fontsize=11)
    
    # --- FormataÃ§Ã£o do Eixo X ---
    # Garante que os nÃºmeros negativos no eixo sejam exibidos como positivos
    formatter = mticker.FuncFormatter(lambda x, pos: f'{abs(x):,.0f}')
    ax.xaxis.set_major_formatter(formatter)
    
    # Adiciona a linha central de referÃªncia
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    # --- Caixa de Texto Explicativa ---
    explanation_text = (
        "Como Ler o GrÃ¡fico:\n\n"
        "â€¢ Eixo Vertical: Mostra as faixas de valor para a variÃ¡vel.\n"
        "â€¢ Eixo Horizontal: A largura da barra indica o nÃºmero\n"
        "  de clientes naquela faixa de valor.\n\n"
        "â€¢ Lado Esquerdo (Vermelho): Perfil dos INADIMPLENTES.\n"
        "â€¢ Lado Direito (Verde): Perfil dos ADIMPLENTES.\n\n"
        "Insight Chave: Compare o 'peso' das barras. Barras\n"
        "vermelhas mais longas em valores mais altos (acima no eixo Y)\n"
        "indicam um forte sinal de risco."
    )
    
    # Posiciona a caixa de texto no canto superior esquerdo do grÃ¡fico
    ax.text(0.03, 0.97, explanation_text,
            transform=ax.transAxes, # Coordenadas relativas ao eixo
            fontsize=11,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', fc='ghostwhite', alpha=0.85))

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Ajusta para o tÃ­tulo nÃ£o sobrepor
    plt.show()

else:
    print(f"AVISO: A coluna '{coluna_numerica}' nÃ£o foi encontrada no df_analise.")


# ==============================================================================
# ANÃ�LISE APROFUNDADA DO PERFIL INADIMPLENTE (COM TRADUÃ‡ÃƒO)
# ==============================================================================
print("--- Gerando anÃ¡lise do perfil de dias em atraso para clientes INADIMPLENTES ---")

# --- VariÃ¡veis a serem analisadas ---
coluna_numerica = 'hist_dias_atraso_1'
coluna_categorica = 'tipo_moradia'

if coluna_numerica in df_analise.columns and coluna_categorica in df_analise.columns:

    # --- 1. FILTRO: Foco exclusivo nos Inadimplentes ---
    df_inadimplentes = df_analise[df_analise['target'] == 1].copy()
    print(f"Analisando um total de {len(df_inadimplentes)} clientes inadimplentes.")
    
    # ==========================================================================
    # <<< MELHORIA: DICIONÃ�RIO DE TRADUÃ‡ÃƒO >>>
    # ==========================================================================
    mapa_traducao_moradia = {
        'FLAT': 'Apartamento',
        'OWNED': 'Casa PrÃ³pria',
        'PARENTAL': 'Com os Pais',
        'STATE_FLAT': 'Apto. Social',
        'COOP_FLAT': 'Apto. Cooperativa',
        'COMPANY_FLAT': 'Apto. da Empresa',
        'NÃ£o Informado': 'NÃ£o Informado' # MantÃ©m a categoria que criamos
    }
    
    # Aplica a traduÃ§Ã£o, criando uma nova coluna para o grÃ¡fico
    df_inadimplentes['tipo_moradia_pt'] = df_inadimplentes[coluna_categorica].map(mapa_traducao_moradia)
    
    # --- 2. PreparaÃ§Ã£o dos Dados para VisualizaÃ§Ã£o ---
    limite_superior = df_inadimplentes[coluna_numerica].quantile(0.95)
    df_inadimplentes['viz_col_numerica'] = df_inadimplentes[coluna_numerica].clip(upper=limite_superior)
    
    # Ordena as categorias pela mediana de dias de atraso
    ordem_plotagem = df_inadimplentes.groupby('tipo_moradia_pt')['viz_col_numerica'].median().sort_values().index
    
    # --- 3. CriaÃ§Ã£o do GrÃ¡fico de Violino ---
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # AQUI USAMOS A COLUNA TRADUZIDA NO EIXO X
    sns.violinplot(data=df_inadimplentes,
                   x='tipo_moradia_pt', # <-- MUDANÃ‡A AQUI
                   y='viz_col_numerica',
                   order=ordem_plotagem,
                   palette='Reds',
                   inner='quartile',
                   ax=ax)

    # --- 4. APLICAÃ‡ÃƒO DO NOSSO PADRÃƒO DE ESTILO ---
    apply_standard_style(ax, 
                         title=f'Perfil de Atraso dos Clientes INADIMPLENTES por Tipo de Moradia',
                         xlabel='Tipo de Moradia',
                         ylabel=f'DistribuiÃ§Ã£o de "{coluna_numerica}"')
    
    plt.tight_layout()
    plt.show()

else:
    print(f"AVISO: As colunas necessÃ¡rias nÃ£o foram encontradas no df_analise.")




import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Supondo que 'df_train_eda' (junÃ§Ã£o de X_train e y_train) jÃ¡ existe.

# ==============================================================================
# ANÃ�LISE APROFUNDADA: O IMPACTO DE TER UM HISTÃ“RICO DE ATRASO
# ==============================================================================
print("--- AnÃ¡lise Aprofundada: O Impacto de Ter um HistÃ³rico de Atraso ---")

# --- 1. Engenharia da Feature BinÃ¡ria ---
# Criamos a nova coluna baseada na nossa observaÃ§Ã£o do grÃ¡fico
# O .astype(int) converte True/False para 1/0
df_train_eda['possui_historico_atraso'] = (df_train_eda['hist_dias_atraso_1'] > 0).astype(int)

print("Nova feature 'possui_historico_atraso' criada com sucesso.")


# --- 2. AnÃ¡lise de Impacto Direto ---
# Calculamos a taxa de inadimplÃªncia para os dois novos grupos
taxa_inadimplencia_por_historico = df_train_eda.groupby('possui_historico_atraso')['target'].mean() * 100

# --- 3. VisualizaÃ§Ã£o do Impacto ---
plt.figure(figsize=(10, 7))
ax = sns.barplot(x=taxa_inadimplencia_por_historico.index, 
                 y=taxa_inadimplencia_por_historico.values, 
                 palette=['#4CAF50', '#F44336'])

# Adiciona os rÃ³tulos de dados
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.2f}%', (p.get_x() + p.get_width() / 2, height),
                ha='center', va='bottom', fontsize=13, color='black', weight='bold', 
                xytext=(0, 5), textcoords='offset points')

# TÃ­tulos e RÃ³tulos
ax.set_title('Taxa de InadimplÃªncia vs. ExistÃªncia de HistÃ³rico de Atraso', fontsize=16, weight='bold', pad=20)
ax.set_ylabel('Taxa de InadimplÃªncia (%)', fontsize=12)
ax.set_xlabel('O Cliente Possui HistÃ³rico de Atraso?', fontsize=12)
ax.set_xticklabels(['NÃ£o (Nunca Atrasou)', 'Sim (JÃ¡ Atrasou)'], fontsize=12)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Supondo que 'df_train_eda' (junÃ§Ã£o de X_train e y_train) jÃ¡ existe.

# ==============================================================================
# ANÃ�LISE 3D: INTERAÃ‡ÃƒO ENTRE RENDA, DIAS DE ATRASO E INADIMPLÃŠNCIA
# ==============================================================================
print("--- Gerando Mapa de Calor de Risco (Renda vs. Atraso vs. InadimplÃªncia) ---")

# --- VariÃ¡veis a serem analisadas ---
col_renda = 'renda_principal'
col_atraso = 'hist_dias_atraso_1'

if col_renda in df_train_eda.columns and col_atraso in df_train_eda.columns:

    # --- 1. Filtragem 90/10 da Renda ---
    limite_inferior = df_train_eda[col_renda].quantile(0.10)
    limite_superior = df_train_eda[col_renda].quantile(0.90)
    df_filtrado = df_train_eda[
        (df_train_eda[col_renda] >= limite_inferior) &
        (df_train_eda[col_renda] <= limite_superior)
    ].copy() # Usamos .copy() para evitar SettingWithCopyWarning

    # --- 2. Binning (CriaÃ§Ã£o de Faixas) para ambas as variÃ¡veis ---
    # Criando 8 faixas de renda com base nos dados filtrados
    df_filtrado['faixa_renda'] = pd.qcut(df_filtrado[col_renda], q=8, duplicates='drop', labels=False)
    
    # Criando 3 nÃ­veis de atraso com base em regras de negÃ³cio
    bins_atraso = [-np.inf, 0, 30, np.inf]
    labels_atraso = ['Sem Atraso', 'Atraso Leve (1-30d)', 'Atraso Grave (>30d)']
    df_filtrado['nivel_atraso'] = pd.cut(df_filtrado[col_atraso], bins=bins_atraso, labels=labels_atraso)

    # --- 3. CÃ¡lculo Cruzado da Taxa de InadimplÃªncia ---
    # Agrupamos pelas duas novas categorias e calculamos a taxa de inadimplÃªncia
    taxa_inadimplencia_cruzada = df_filtrado.groupby(['faixa_renda', 'nivel_atraso'])['target'].mean().unstack()

    # Mapeando os labels numÃ©ricos da faixa de renda para texto legÃ­vel
    faixas_labels_map = {i: f"{int(row['min'])}-{int(row['max'])}" for i, row in df_filtrado.groupby('faixa_renda')[col_renda].agg(['min', 'max']).iterrows()}
    taxa_inadimplencia_cruzada.rename(index=faixas_labels_map, inplace=True)
    
    # --- 4. VisualizaÃ§Ã£o com Heatmap ---
    plt.figure(figsize=(14, 10))
    
    # O heatmap usa a intensidade da cor para representar o valor
    sns.heatmap(taxa_inadimplencia_cruzada,
                annot=True,        # Mostra os nÃºmeros dentro de cada cÃ©lula
                fmt='.1%',         # Formata os nÃºmeros como porcentagem
                cmap='Reds',       # Paleta de cores "quente" para risco
                linewidths=.5,
                linecolor='lightgray')

    plt.title('Mapa de Calor de Risco: InadimplÃªncia por Faixa de Renda e HistÃ³rico de Atraso', fontsize=16, weight='bold', pad=20)
    plt.xlabel('NÃ­vel do HistÃ³rico de Atraso', fontsize=12)
    plt.ylabel('Faixa de Renda Principal (80% Centrais)', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.show()

else:
    print(f"AVISO: As colunas necessÃ¡rias nÃ£o foram encontradas no df_train_eda.")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Supondo que 'df_train_eda' jÃ¡ existe e o dicionÃ¡rio 'risk_map' foi salvo.
# Se o risk_map nÃ£o estiver na memÃ³ria, vocÃª precisarÃ¡ recriÃ¡-lo a partir dos dados de treino.
# Exemplo de como recriar o risk_map caso necessÃ¡rio:
# if 'risk_map' not in locals():
#     print("Recriando 'risk_map'...")
#     train_data_for_encoding = pd.concat([X_train, y_train], axis=1)
#     train_data_for_encoding['tipo_moradia'].fillna('NÃ£o Informado', inplace=True)
#     risk_map = train_data_for_encoding.groupby('tipo_moradia')['target'].mean().to_dict()

# ==============================================================================
# ANÃ�LISE DE INTERAÃ‡ÃƒO COM LABELS DECODIFICADOS
# ==============================================================================
print("--- Gerando grÃ¡fico de interaÃ§Ã£o com labels originais de moradia ---")

# --- VariÃ¡veis a serem analisadas ---
col_renda = 'renda_principal'
col_moradia_encoded = 'tipo_moradia_encoded' # Usamos a coluna numÃ©rica para os cÃ¡lculos

if col_renda in df_train_eda.columns and col_moradia_encoded in df_train_eda.columns:

    # --- 1. PreparaÃ§Ã£o dos Dados ---
    df_plot = df_train_eda.copy()
    try:
        df_plot['faixa_renda_num'] = pd.qcut(df_plot[col_renda], q=10, duplicates='drop', labels=False)
        
        # Agrupamos usando a coluna numÃ©rica ENCODED
        taxa_inadimplencia = df_plot.groupby(['faixa_renda_num', col_moradia_encoded])['target'].mean().reset_index()
        taxa_inadimplencia.rename(columns={'target': 'taxa_de_inadimplencia'}, inplace=True)
        taxa_inadimplencia['taxa_de_inadimplencia'] *= 100
    
        # Criamos os labels da faixa de renda para o eixo X
        faixas_labels = df_plot.groupby('faixa_renda_num')[col_renda].agg(['min', 'max'])
        faixas_labels_map = {i: f"R$ {int(row['min'])}-{int(row['max'])}" for i, row in faixas_labels.iterrows()}
        taxa_inadimplencia['faixa_renda_label'] = taxa_inadimplencia['faixa_renda_num'].map(faixas_labels_map)
        
        # ==========================================================================
        # <<< MELHORIA: DECODIFICAÃ‡ÃƒO DOS LABELS PARA A PLOTAGEM >>>
        # ==========================================================================
        # Criamos um "mapa reverso" a partir do nosso dicionÃ¡rio de risco
        reverse_risk_map = {v: k for k, v in risk_map.items()}
        
        # Criamos uma nova coluna com os nomes originais para usar na legenda do grÃ¡fico
        taxa_inadimplencia['tipo_moradia_original'] = taxa_inadimplencia[col_moradia_encoded].map(reverse_risk_map)
        print("\nLabels de 'tipo_moradia' decodificados para a visualizaÃ§Ã£o.")

        # Paleta de cores customizada
        cores_customizadas = {
            "OWNED": "#1f77b4", "NÃ£o Informado": "#ff7f0e", "COMPANY_FLAT": "#d62728",
            "FLAT": "#2ca02c", "STATE_FLAT": "#9467bd", "PARENTAL": "#8c564b", "COOP_FLAT": "#e377c2"
        }

        # --- 2. VisualizaÃ§Ã£o com GrÃ¡fico de Linhas ---
        fig, ax = plt.subplots(figsize=(16, 9))
        
        # AQUI ESTÃ� A MUDANÃ‡A: usamos a nova coluna 'tipo_moradia_original' para o HUE
        sns.lineplot(data=taxa_inadimplencia, 
                     x='faixa_renda_label', 
                     y='taxa_de_inadimplencia', 
                     hue='tipo_moradia_original', # <-- HUE com os nomes originais
                     marker='o',
                     markersize=8,
                     linewidth=2.5,
                     palette=cores_customizadas,
                     ax=ax)

        # --- 3. AplicaÃ§Ã£o do Estilo PadrÃ£o ---
        ax.set_title('Taxa de InadimplÃªncia por Faixa de Renda e Tipo de Moradia', fontsize=18, weight='bold', pad=20)
        ax.set_xlabel('Faixas de Renda', fontsize=14)
        ax.set_ylabel('Taxa de InadimplÃªncia (%)', fontsize=14)
        ax.tick_params(axis='x', labelsize=12, rotation=45)
        ax.legend(title='Tipo de Moradia', fontsize=11, title_fontsize=13)
        ax.grid(True, linestyle='--', alpha=0.7)
        sns.despine(ax=ax)
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"NÃ£o foi possÃ­vel criar o grÃ¡fico. Erro: {e}")

else:
    print(f"AVISO: As colunas necessÃ¡rias nÃ£o foram encontradas no df_train_eda.")


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Supondo que 'df_train_eda' jÃ¡ existe.

# ==============================================================================
# ANÃ�LISE APROFUNDADA (VERSÃƒO FINAL COM ALINHAMENTO CORRIGIDO)
# ==============================================================================
print("--- Gerando GrÃ¡fico Final com Alinhamento de Legenda Corrigido ---")

# --- VariÃ¡vel a ser analisada ---
var_analise = 'relacao_credito_renda'

if var_analise in df_train_eda.columns:

    # --- 1. CÃ¡lculo dos Limites e 2. Filtragem dos Dados ---
    limite_inferior = df_train_eda[var_analise].quantile(0.10)
    limite_superior = df_train_eda[var_analise].quantile(0.90)
    df_filtrado = df_train_eda[
        (df_train_eda[var_analise] >= limite_inferior) &
        (df_train_eda[var_analise] <= limite_superior)
    ]

    # --- 3. VisualizaÃ§Ã£o Aprofundada ---
    fig, ax = plt.subplots(figsize=(15, 9))
    
    sns.histplot(data=df_filtrado, x=var_analise, hue='target',
                 palette=['#4CAF50', '#F44336'], kde=True, 
                 stat="density", common_norm=False, ax=ax)
    
    media_adimplente = df_filtrado[df_filtrado['target'] == 0][var_analise].mean()
    media_inadimplente = df_filtrado[df_filtrado['target'] == 1][var_analise].mean()
    
    ax.axvline(media_adimplente, color='#4CAF50', linestyle='--', linewidth=2.5, label=f'MÃ©dia Adimplentes: {media_adimplente:.2f}')
    ax.axvline(media_inadimplente, color='#F44336', linestyle='--', linewidth=2.5, label=f'MÃ©dia Inadimplentes: {media_inadimplente:.2f}')
    
    # ==========================================================================
    # <<< AJUSTE DE ALINHAMENTO DAS LEGENDAS >>>
    # ==========================================================================

    # --- TÃ­tulo e RÃ³tulos dos Eixos ---
    ax.set_title('Perfil de Endividamento: Adimplentes vs. Inadimplentes', fontsize=18, weight='bold', pad=20)
    ax.set_xlabel("NÃ­vel de Endividamento (RelaÃ§Ã£o CrÃ©dito / Renda)", fontsize=14)
    ax.set_ylabel("ConcentraÃ§Ã£o de Clientes (Densidade)", fontsize=14)
    
    # --- Legenda Principal (para as cores e linhas) ---
    # Mantida na melhor posiÃ§Ã£o automÃ¡tica ('best' ou 'upper right')
    ax.legend(title='Status do CrÃ©dito', fontsize=11, loc='upper right')

    # --- Caixa de Texto Explicativa para os Eixos ---
    explanation_text = (
        "Como Ler o GrÃ¡fico:\n\n"
        "â€¢ Eixo X (Horizontal): Mostra o quÃ£o 'pesado' o\n"
        "  emprÃ©stimo Ã© para a renda do cliente.\n"
        "  Valores mais altos = maior endividamento.\n\n"
        "â€¢ Eixo Y (Vertical): Mostra a 'popularidade' de\n"
        "  cada nÃ­vel de endividamento. Picos altos\n"
        "  significam que muitos clientes se concentram\n"
        "  naquele valor."
    )
    
    # --- ALTERAÃ‡ÃƒO AQUI ---
    # Posiciona a caixa de texto no canto inferior esquerdo para melhor balanÃ§o visual
    ax.text(0.03, 0.03, explanation_text,
            transform=ax.transAxes, # Coordenadas relativas ao eixo do grÃ¡fico
            fontsize=11,
            verticalalignment='bottom', # Alinha pela base da caixa
            horizontalalignment='left', # Alinha pela esquerda da caixa
            bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.9))

    plt.tight_layout()
    plt.show()

else:
    print(f"AVISO: Coluna '{var_analise}' nÃ£o encontrada no df_train_eda.")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Supondo que 'df_train_eda' (junÃ§Ã£o de X_train e y_train) jÃ¡ existe.

# ==============================================================================
# ANÃ�LISE DE TAXA DE INADIMPLÃŠNCIA POR FAIXA DE RENDA
# ==============================================================================
print("--- AnÃ¡lise da Taxa de InadimplÃªncia por Faixa de Renda ---")

# --- VariÃ¡vel a ser analisada ---
coluna_renda = 'renda_principal'

if coluna_renda in df_train_eda.columns:

    # --- 1. CriaÃ§Ã£o das Faixas de Renda (Binning) ---
    # Usamos pd.qcut para dividir os clientes em 10 grupos (decis) com aproximadamente
    # o mesmo nÃºmero de pessoas em cada grupo. Isso Ã© mais robusto para dados com outliers.
    try:
        df_train_eda['faixa_renda'] = pd.qcut(df_train_eda[coluna_renda], 
                                             q=10, 
                                             duplicates='drop', # Remove limites duplicados se houver
                                             labels=False) # Retorna nÃºmeros para as faixas (0 a 9)
        
        # --- 2. CÃ¡lculo da Taxa de InadimplÃªncia por Faixa ---
        # Agrupamos pela faixa de renda e calculamos a mÃ©dia do 'target'.
        # Como target Ã© 0 ou 1, a mÃ©dia Ã© exatamente a taxa de inadimplÃªncia.
        taxa_inadimplencia_por_faixa = df_train_eda.groupby('faixa_renda')['target'].mean() * 100
        taxa_inadimplencia_por_faixa = taxa_inadimplencia_por_faixa.sort_index()

        # Para tornar o label do eixo X mais legÃ­vel, vamos pegar os limites de cada faixa
        faixas_labels = df_train_eda.groupby('faixa_renda')[coluna_renda].agg(['min', 'max'])
        faixas_labels_str = [f"{int(row['min'])}-{int(row['max'])}" for index, row in faixas_labels.iterrows()]
        
        # --- 3. VisualizaÃ§Ã£o com GrÃ¡fico de Barras ---
        plt.figure(figsize=(16, 8))
        
        ax = sns.barplot(x=taxa_inadimplencia_por_faixa.index, 
                         y=taxa_inadimplencia_por_faixa.values, 
                         palette='Reds_r') # Paleta de vermelhos, do mais claro ao mais escuro

        # Adiciona os rÃ³tulos de dados (data labels) em cada barra
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.2f}%', (p.get_x() + p.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=12, color='black', weight='bold', 
                        xytext=(0, 5), textcoords='offset points')

        # --- TÃ­tulos e RÃ³tulos ---
        ax.set_title('Taxa de InadimplÃªncia por Faixa de Renda Principal', fontsize=18, weight='bold', pad=20)
        ax.set_ylabel('Taxa de InadimplÃªncia (%)', fontsize=14)
        ax.set_xlabel('Faixas de Renda (R$)', fontsize=14)
        
        # Usando os labels legÃ­veis que criamos
        ax.set_xticklabels(faixas_labels_str, rotation=45, ha='right')
        
        # Adiciona uma linha com a taxa de inadimplÃªncia mÃ©dia geral como referÃªncia
        media_geral = df_train_eda['target'].mean() * 100
        ax.axhline(media_geral, color='black', linestyle='--', label=f'MÃ©dia Geral: {media_geral:.2f}%')
        
        ax.legend()
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"NÃ£o foi possÃ­vel criar as faixas de renda. Erro: {e}")
        print("Isso pode acontecer se houver poucos valores Ãºnicos na coluna de renda.")

else:
    print(f"AVISO: Coluna '{coluna_renda}' nÃ£o encontrada no df_train_eda.")


# Verifica se a coluna 'tipo_moradia' existe
if 'tipo_moradia' in df_analise.columns:
    # Calcula a taxa de inadimplÃªncia (mÃ©dia do target) para cada categoria
    taxa_inadimplencia_moradia = df_analise.groupby('tipo_moradia')['target'].mean().sort_values(ascending=False) * 100

    plt.figure(figsize=(12, 7))

    # Cria o grÃ¡fico de barras
    ax = sns.barplot(x=taxa_inadimplencia_moradia.index, y=taxa_inadimplencia_moradia.values, palette='viridis')

    # Adiciona anotaÃ§Ãµes de porcentagem
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.2f}%', (p.get_x() + p.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=12, color='black', weight='bold', xytext=(0, 5),
                    textcoords='offset points')

    # TÃ­tulos e rÃ³tulos
    plt.title('Taxa de InadimplÃªncia por Tipo de Moradia', fontsize=16, weight='bold')
    plt.ylabel('Taxa de InadimplÃªncia (%)')
    plt.xlabel('Tipo de Moradia')
    plt.xticks(rotation=45, ha='right') # Rotaciona os labels para nÃ£o sobrepor

    plt.show()


# ==============================================================================
# ANÃ�LISE DA RELAÃ‡ÃƒO: TIPO DE MORADIA vs. VARIÃ�VEIS NUMÃ‰RICAS
# ==============================================================================
print("--- Gerando anÃ¡lise do perfil mÃ©dio por Tipo de Moradia ---")

# --- 1. DecodificaÃ§Ã£o da VariÃ¡vel 'tipo_moradia' ---
# Para esta anÃ¡lise especÃ­fica, vamos criar uma cÃ³pia e adicionar os nomes originais de volta.

# Se o risk_map nÃ£o estiver na memÃ³ria, vocÃª precisarÃ¡ recriÃ¡-lo
if 'risk_map' not in locals():
    print("AVISO: 'risk_map' nÃ£o encontrado. Recriando para a visualizaÃ§Ã£o...")
    # Esta parte assume que X_train e y_train existem
    train_data_for_encoding = pd.concat([X_train, y_train], axis=1)
    train_data_for_encoding['tipo_moradia'].fillna('NÃ£o Informado', inplace=True)
    risk_map = train_data_for_encoding.groupby('tipo_moradia')['target'].mean().to_dict()

# Criamos um mapa reverso para traduzir o score de volta para o nome
reverse_risk_map = {v: k for k, v in risk_map.items()}

# Criamos uma cÃ³pia do dataframe da EDA para nÃ£o alterar o original
df_plot = df_train_eda.copy()
df_plot['tipo_moradia_original'] = df_plot['tipo_moradia_encoded'].map(reverse_risk_map)


# --- 2. AnÃ¡lise e VisualizaÃ§Ã£o ---
# Lista de variÃ¡veis numÃ©ricas que queremos comparar por tipo de moradia
variaveis_para_comparar = ['renda_principal', 'valor_credito_atual', 'relacao_credito_renda', 'dias_atraso']

for var in variaveis_para_comparar:
    if var in df_plot.columns:
        
        # Calcula a mÃ©dia da variÃ¡vel numÃ©rica para cada tipo de moradia
        media_por_moradia = df_plot.groupby('tipo_moradia_original')[var].mean().sort_values(ascending=False)
        
        # CriaÃ§Ã£o do GrÃ¡fico de Barras
        plt.figure(figsize=(12, 7))
        ax = sns.barplot(x=media_por_moradia.index, y=media_por_moradia.values, palette='viridis')
        
        # Adiciona rÃ³tulos de dados
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:,.0f}', (p.get_x() + p.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=12, color='black', weight='bold', 
                        xytext=(0, 5), textcoords='offset points')
        
        # TÃ­tulos e RÃ³tulos (seguindo nosso padrÃ£o)
        ax.set_title(f'MÃ©dia de "{var}" por Tipo de Moradia', fontsize=16, weight='bold', pad=20)
        ax.set_ylabel(f'Valor MÃ©dio de {var}', fontsize=12)
        ax.set_xlabel('Tipo de Moradia', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()


# ==============================================================================
# ANÃ�LISE 3D: PERFIL MÃ‰DIO POR TIPO DE MORADIA E RISCO (COM FILTRO 80/20)
# ==============================================================================
print("--- Gerando anÃ¡lise de perfil com GrÃ¡fico de Barras Agrupado e Filtro de Outliers ---")

# --- 1. DecodificaÃ§Ã£o da VariÃ¡vel 'tipo_moradia' ---
if 'risk_map' not in locals():
    print("AVISO: 'risk_map' nÃ£o encontrado. Recriando para a visualizaÃ§Ã£o...")
    train_data_for_encoding = pd.concat([X_train, y_train], axis=1)
    train_data_for_encoding['tipo_moradia'].fillna('NÃ£o Informado', inplace=True)
    risk_map = train_data_for_encoding.groupby('tipo_moradia')['target'].mean().to_dict()

reverse_risk_map = {v: k for k, v in risk_map.items()}
df_plot_base = df_train_eda.copy()
df_plot_base['tipo_moradia_original'] = df_plot_base['tipo_moradia_encoded'].map(reverse_risk_map)


# --- 2. AnÃ¡lise e VisualizaÃ§Ã£o em Loop ---
variaveis_para_comparar = ['renda_principal', 'valor_credito_atual', 'relacao_credito_renda', 'hist_dias_atraso_1']

for var in variaveis_para_comparar:
    if var in df_plot_base.columns:
        
        # --- Filtragem de Outliers (80/20) ---
        limite_inferior = df_plot_base[var].quantile(0.10)
        limite_superior = df_plot_base[var].quantile(0.90)
        df_filtrado = df_plot_base[(df_plot_base[var] >= limite_inferior) & (df_plot_base[var] <= limite_superior)]
        
        print(f"\nAnalisando '{var}' para os 80% centrais dos dados...")
        
        # --- CÃ¡lculo para o GrÃ¡fico Agrupado ---
        media_agrupada = df_filtrado.groupby(['tipo_moradia_original', 'target'])[var].mean().reset_index()
        
        # --- CriaÃ§Ã£o do GrÃ¡fico de Barras Agrupado ---
        fig, ax = plt.subplots(figsize=(16, 9))
        
        # --- AJUSTE AQUI ---
        # Adicionamos errorbar=None para um visual mais limpo, focando apenas na mÃ©dia.
        sns.barplot(data=media_agrupada, 
                    x='tipo_moradia_original', 
                    y=var, 
                    hue='target',
                    palette=['#4CAF50', '#F44336'],
                    errorbar=None, # Remove as barras de erro para um visual mais limpo
                    ax=ax)
        
        # --- AplicaÃ§Ã£o do nosso PadrÃ£o de Estilo Corrigido ---
        apply_standard_style(ax, 
                             title=f'Perfil MÃ©dio de "{var}" (80% Centrais) por Moradia e Risco',
                             xlabel='Tipo de Moradia',
                             ylabel=f'Valor MÃ©dio de {var}')
        
        # --- Melhorando a Legenda ---
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=['Adimplente (0)', 'Inadimplente (1)'], 
                  title='Status do Cliente', fontsize=11, title_fontsize=13)
        
        plt.tight_layout()
        plt.show()


# ==============================================================================
# ANÃ�LISE DE CORRELAÃ‡ÃƒO (COM ESTILIZAÃ‡ÃƒO PADRÃƒO)
# ==============================================================================
print("--- Gerando Mapa de Calor de CorrelaÃ§Ã£o Aprimorado ---")

# --- 1. SeleÃ§Ã£o das VariÃ¡veis Mais Relevantes ---
# Para um heatmap legÃ­vel, focamos nas colunas mais importantes que jÃ¡ analisamos.
colunas_relevantes = [
    'target',
    'renda_principal',
    'valor_credito_atual',
    'valor_entrada_atual',
    'hist_dias_atraso_1',
    'hist_valor_atraso_1',
    'hist_dias_atraso_2',
    'hist_valor_atraso_2',
    'relacao_credito_renda',
    'percentual_entrada',
    'valor_desembolsado_atual',
    'tipo_moradia_encoded' # Incluindo nossa feature de engenharia categÃ³rica
]

# Filtramos o DataFrame para conter apenas estas colunas
df_corr = df_train_eda[colunas_relevantes]

# --- 2. CÃ¡lculo da Matriz de CorrelaÃ§Ã£o ---
matriz_correlacao = df_corr.corr()

# --- 3. VisualizaÃ§Ã£o com o Mapa de Calor Estilizado ---
fig, ax = plt.subplots(figsize=(12, 10))

# Desenhamos o heatmap
sns.heatmap(matriz_correlacao,
            annot=True,          # Adiciona os nÃºmeros dentro das cÃ©lulas
            cmap='coolwarm',     # Paleta de cores ideal para correlaÃ§Ã£o (negativo-neutro-positivo)
            fmt=".2f",           # Formata os nÃºmeros para duas casas decimais
            linewidths=.5,
            linecolor='lightgray',
            ax=ax)

# --- 4. APLICAÃ‡ÃƒO DO NOSSO PADRÃƒO DE ESTILO ---
# TÃ­tulo claro e informativo
ax.set_title('CorrelaÃ§Ã£o entre as Principais VariÃ¡veis do Modelo', fontsize=18, weight='bold', pad=20)

# Melhora a legibilidade dos ticks
ax.tick_params(axis='x', labelsize=11, rotation=45)
ax.tick_params(axis='y', labelsize=11, rotation=0)

plt.tight_layout()
plt.show()




# ==============================================================================
# FASE 2: MODELAGEM PREDITIVA
# ETAPA 1: MODELO BASELINE (REGRESSÃƒO LOGÃ�STICA)
# ==============================================================================
# Objetivo: Treinar um modelo simples e rÃ¡pido que servirÃ¡ como nosso ponto de
# referÃªncia para comparar com modelos mais avanÃ§ados.

# Supondo que X_train, X_test, y_train, y_test jÃ¡ existem da fase de preparaÃ§Ã£o.

print("--- 2. Treinando e Avaliando o Modelo Baseline (RegressÃ£o LogÃ­stica) ---")

# --- Treinamento ---
# Instanciamos o modelo. Usamos:
# - random_state=42: Para garantir que os resultados sejam reprodutÃ­veis.
# - max_iter=1000: Para dar ao algoritmo iteraÃ§Ãµes suficientes para convergir (encontrar a melhor soluÃ§Ã£o).
baseline_model = LogisticRegression(random_state=42, max_iter=1000)

# O comando .fit() Ã© onde o modelo "aprende" os padrÃµes dos dados de treino.
baseline_model.fit(X_train, y_train)
print("Modelo Baseline treinado com sucesso.")

# --- AvaliaÃ§Ã£o ---
print("\nAvaliando o modelo no conjunto de teste...")

# Prevemos as probabilidades para a classe positiva (1 = Inadimplente).
# A mÃ©trica AUC necessita das probabilidades, nÃ£o da classe final (0 ou 1).
y_prob_baseline = baseline_model.predict_proba(X_test)[:, 1]

# Calculamos e imprimimos o AUC Score, nossa principal mÃ©trica de performance.
auc_baseline = roc_auc_score(y_test, y_prob_baseline)
print(f'\nAUC Score (Baseline): {auc_baseline:.4f}')

# Para uma anÃ¡lise mais detalhada do desempenho em cada classe, geramos o relatÃ³rio de classificaÃ§Ã£o.
# Para isso, precisamos das prediÃ§Ãµes de classe (0 ou 1).
y_pred_baseline = baseline_model.predict(X_test)

print("\nRelatÃ³rio de ClassificaÃ§Ã£o (Baseline):")
print(classification_report(y_test, y_pred_baseline, target_names=['Adimplente', 'Inadimplente']))

print("\n--- Etapa do Modelo Baseline ConcluÃ­da ---")
  


# ==============================================================================
# FASE 2: MODELAGEM PREDITIVA
# ETAPA 2: MODELO AVANÃ‡ADO (LIGHTGBM)
# ==============================================================================
# Objetivo: Treinar um modelo robusto (LightGBM) utilizando uma tÃ©cnica para
# lidar com o desbalanceamento de classes e superar a performance do baseline.

# Supondo que X_train, X_test, y_train, y_test jÃ¡ existem da fase de preparaÃ§Ã£o.

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np

print("--- 3. Treinando e Avaliando o Modelo AvanÃ§ado (LightGBM) ---")

# --- Tratamento do Desbalanceamento: CÃ¡lculo do Peso ---
# Esta Ã© a tÃ©cnica mais importante para o sucesso do nosso modelo.
# Calculamos um peso para dizer ao modelo que errar um 'Inadimplente' Ã© muito mais grave.
try:
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"Fator de Peso (scale_pos_weight) para a Classe Inadimplente: {scale_pos_weight:.2f}")
except ZeroDivisionError:
    print("AVISO: NÃ£o hÃ¡ amostras da classe minoritÃ¡ria no conjunto de treino. O peso nÃ£o pode ser calculado.")
    scale_pos_weight = 1 # Define um peso neutro

# --- Treinamento ---
# Instanciamos o modelo LightGBM. Passamos:
# - scale_pos_weight: O peso calculado para a classe de inadimplentes.
# - n_estimators: O nÃºmero de "Ã¡rvores de decisÃ£o" que o modelo irÃ¡ construir.
# - random_state=42: Para garantir a reprodutibilidade dos resultados.
lgbm_model = lgb.LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight, n_estimators=300)

# O comando .fit() treina o modelo com os dados de treino.
lgbm_model.fit(X_train, y_train)
print("Modelo AvanÃ§ado treinado com sucesso.")

# --- AvaliaÃ§Ã£o ---
print("\nAvaliando o modelo avanÃ§ado no conjunto de teste...")

# Prevemos as probabilidades para a classe positiva (1 = Inadimplente).
y_prob_lgbm = lgbm_model.predict_proba(X_test)[:, 1]

# Calculamos e imprimimos o AUC Score.
auc_lgbm = roc_auc_score(y_test, y_prob_lgbm)
print(f'\nAUC Score (LightGBM): {auc_lgbm:.4f}')

# Geramos o relatÃ³rio de classificaÃ§Ã£o completo para uma anÃ¡lise detalhada.
y_pred_lgbm = lgbm_model.predict(X_test)
print("\nRelatÃ³rio de ClassificaÃ§Ã£o (LightGBM):")
print(classification_report(y_test, y_pred_lgbm, target_names=['Adimplente', 'Inadimplente']))

print("\n--- Etapa do Modelo AvanÃ§ado ConcluÃ­da ---")



# Supondo que os seguintes objetos jÃ¡ existem na memÃ³ria:
# baseline_model, lgbm_model, X_test, y_test

# ==============================================================================
# FUNÃ‡ÃƒO CUSTOMIZADA PARA PLOTAR A MATRIZ DE CONFUSÃƒO LÃšDICA
# ==============================================================================
def plot_ludic_confusion_matrix(model, X, y, ax, title, cmap):
    """
    Plota uma Matriz de ConfusÃ£o com formataÃ§Ã£o numÃ©rica clara e rÃ³tulos lÃºdicos.
    """
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    vn, fp, fn, vp = cm.ravel()
    
    # --- MELHORIA: RÃ“TULOS LÃšDICOS COM Ã�CONES ---
    labels = np.array([
        f" Cliente ConfiÃ¡vel\n(Verdadeiro Negativo)\n\n{vn:,.0f}",
        f" Alarme Falso\n(Falso Positivo)\n\n{fp:,.0f}",
        f" Risco Ignorado\n(Falso Negativo)\n\n{fn:,.0f}",
        f" Caloteiro Identificado\n(Verdadeiro Positivo)\n\n{vp:,.0f}"
    ]).reshape(2, 2)
    
    # Desenha o heatmap com as novas anotaÃ§Ãµes
    sns.heatmap(cm, annot=labels, fmt="", cmap=cmap, ax=ax, cbar=False,
                xticklabels=['Adimplente', 'Inadimplente'], 
                yticklabels=['Adimplente', 'Inadimplente'],
                annot_kws={"size": 11, "va": "center", "ha": "center", "weight": "bold"})
    
    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_ylabel('Valor Real (Verdade)', fontsize=12)
    ax.set_xlabel('Valor Previsto pelo Modelo', fontsize=12)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11, rotation=0)

# ==============================================================================
# ETAPA 4: COMPARAÃ‡ÃƒO VISUAL FINAL (COM GRÃ�FICO LÃšDICO)
# ==============================================================================
print("--- Gerando a ComparaÃ§Ã£o Visual LÃºdica ---")

# --- CriaÃ§Ã£o da Figura ---
fig, axes = plt.subplots(1, 2, figsize=(22, 9))
fig.suptitle('ComparaÃ§Ã£o de Matrizes de ConfusÃ£o', fontsize=20, weight='bold')

# --- GrÃ¡fico 1: Modelo Baseline ---
plot_ludic_confusion_matrix(baseline_model, X_test, y_test, 
                                ax=axes[0], 
                                title='Baseline (RegressÃ£o LogÃ­stica)', 
                                cmap='Reds')

# --- GrÃ¡fico 2: Modelo AvanÃ§ado (LightGBM) ---
plot_ludic_confusion_matrix(lgbm_model, X_test, y_test, 
                                ax=axes[1], 
                                title='AvanÃ§ado (LightGBM)', 
                                cmap='Greens')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

print("\n--- ComparaÃ§Ã£o Visual ConcluÃ­da ---")



# ==============================================================================
# FASE 4: INTERPRETABILIDADE DO MODELO COM SHAP
# ==============================================================================
print("--- Iniciando Fase 4: Interpretabilidade do Modelo ---")

# --- 1. PreparaÃ§Ã£o para a AnÃ¡lise SHAP ---
X_test_sample = X_test.sample(n=5000, random_state=42)

print("\nCalculando os valores SHAP para uma amostra de 5000 clientes...")
explainer = shap.TreeExplainer(lgbm_model)
shap_values_list = explainer.shap_values(X_test_sample)
print("CÃ¡lculo dos valores SHAP concluÃ­do.")

# ==========================================================================
# <<< MELHORIA: RENOMEAÃ‡ÃƒO DE COLUNAS PARA A VISUALIZAÃ‡ÃƒO >>>
# ==========================================================================
# Criamos um dicionÃ¡rio de renomeaÃ§Ã£o para os labels do grÃ¡fico
mapa_nomes_grafico = {
    'semana_num': 'Semana da AplicaÃ§Ã£o',
    'valor_credito_atual': 'Valor do CrÃ©dito',
    'valor_desembolsado_atual': 'Valor Desembolsado',
    'valor_ult_credito_aprovado': 'Valor Ãšlt. CrÃ©dito Aprovado',
    'valor_ult_credito_rejeitado': 'Valor Ãšlt. CrÃ©dito Rejeitado',
    'valor_entrada_atual': 'Valor da Entrada',
    'renda_principal': 'Renda Principal',
    'hist_dias_atraso_1': 'Hist. Dias Atraso (Fonte 1)',
    'hist_dias_atraso_2': 'Hist. Dias Atraso (Fonte 2)',
    'hist_valor_atraso_1': 'Hist. Valor Atraso (Fonte 1)',
    'hist_valor_atraso_2': 'Hist. Valor Atraso (Fonte 2)',
    'relacao_credito_renda': 'RelaÃ§Ã£o CrÃ©dito/Renda',
    'percentual_entrada': 'Percentual de Entrada',
    'tipo_moradia_encoded': 'Risco por Moradia (Encoded)'
}

# Criamos uma cÃ³pia da amostra para renomear, sem alterar o X_test_sample original
X_test_sample_renamed = X_test_sample.copy()
X_test_sample_renamed.rename(columns=mapa_nomes_grafico, inplace=True)


# --- 2. VisualizaÃ§Ã£o da ImportÃ¢ncia Global das Features ---
print("\nGerando o GrÃ¡fico de ImportÃ¢ncia das Features com nomes aprimorados...")

# Passamos a amostra RENOMEADA para a plotagem
shap.summary_plot(shap_values_list[1], X_test_sample_renamed,
                  plot_type="dot",
                  show=False)

# --- Melhorias Visuais ---
fig = plt.gcf()
fig.set_figheight(8)
fig.set_figwidth(14)
plt.title("ImportÃ¢ncia das Features para o Risco de InadimplÃªncia", fontsize=16, weight='bold')
plt.xlabel("Impacto no Score do Modelo (Valor SHAP)")
# Ajusta a margem esquerda para garantir que os nomes longos nÃ£o sejam cortados
plt.subplots_adjust(left=0.3)
plt.show()

print("\n--- AnÃ¡lise de ImportÃ¢ncia de Features ConcluÃ­da ---")


