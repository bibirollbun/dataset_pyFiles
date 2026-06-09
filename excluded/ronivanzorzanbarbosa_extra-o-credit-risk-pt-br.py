from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay, roc_auc_score  
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
import os # Para listar os arquivos
from tqdm import tqdm # Barra de Progresso
import gc # Garbage Collector para liberar memÃ³ria
import warnings
warnings.filterwarnings("ignore")


def carregar_dados_merge_filtrado(
    diretorio_base: str,
    threshold_null: float = 0.5,
    min_variancia: float = 1e-5,
    relatorio: bool = True
) -> pd.DataFrame:
    """
    Carrega 'train_base' e agrega arquivos auxiliares com merge, 
    filtrando colunas com muitos nulos antes do carregamento completo,
    e incluindo colunas categÃ³ricas.

    ParÃ¢metros:
    ------------
    diretorio_base : str
        Caminho dos arquivos .parquet.
    threshold_null : float
        MÃ¡x. % de valores nulos permitidos para incluir coluna no merge.
    min_variancia : float
        VariÃ¢ncia mÃ­nima para manter coluna numÃ©rica no dataset final.
    relatorio : bool
        Se True, imprime relatÃ³rios de integraÃ§Ã£o e filtragem.

    Retorna:
    --------
    pd.DataFrame consolidado e enxuto para modelagem.
    """

    # Carrega base principal
    base = pd.read_parquet(os.path.join(diretorio_base, "train_base.parquet"))
    base.set_index("case_id", inplace=True)

    arquivos = sorted([
        f for f in os.listdir(diretorio_base) if "train" in f and f.endswith(".parquet")
    ]) # Retorna os nomes dos arquivos na pasta de treino

    resumo = {}
    
    for arq in tqdm(arquivos, desc="ğŸ”— Integrando arquivos via merge filtrado"):
        try:
            path = os.path.join(diretorio_base, arq)
            df_aux = pd.read_parquet(path)

            # Identifica colunas numÃ©ricas e categÃ³ricas (excluindo 'case_id')
            col_num = df_aux.select_dtypes(include="number").columns.difference(["case_id"]).tolist()
            col_cat = df_aux.select_dtypes(include=['object', 'category']).columns.difference(["case_id"]).tolist()

            # Calcula % de nulos para todas as colunas relevantes
            # Consideramos numÃ©ricas e categÃ³ricas para o filtro de nulos
            cols_to_check = col_num + col_cat
            if not cols_to_check: # Se nÃ£o houver colunas para verificar (alÃ©m de case_id), pula
                resumo[arq] = 0
                continue

            pct_null = df_aux[cols_to_check].isnull().mean() #MÃ©dia de Nulos por coluna
            col_validas = pct_null[pct_null <= threshold_null].index.tolist()

            # Separa as colunas vÃ¡lidas em numÃ©ricas e categÃ³ricas novamente
            col_num_validas = [col for col in col_validas if col in col_num]
            col_cat_validas = [col for col in col_validas if col in col_cat]

            if not col_num_validas and not col_cat_validas:
                resumo[arq] = 0
                continue

            prefixo = arq.replace(".parquet", "").replace("train_credit_bureau_", "")
            
            # DataFrame para agregaÃ§Ã£o
            df_ag = None

            # AgregaÃ§Ã£o para colunas numÃ©ricas
            if col_num_validas:
                df_num_ag = df_aux.groupby("case_id")[col_num_validas].agg(['mean', 'sum'])
                df_num_ag.columns = [f"{prefixo}_{col}_{agg}" for col, agg in df_num_ag.columns]
                df_ag = df_num_ag
            
            # AgregaÃ§Ã£o para colunas categÃ³ricas (usando a moda)
            # Se uma coluna categÃ³rica tiver vÃ¡rias modas, 'mode()' retorna um Series,
            # entÃ£o pegamos o primeiro valor [0].
            if col_cat_validas:
                df_cat_ag = df_aux.groupby("case_id")[col_cat_validas].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
                df_cat_ag.columns = [f"{prefixo}_{col}_mode" for col in df_cat_ag.columns]
                
                if df_ag is not None:
                    df_ag = df_ag.merge(df_cat_ag, on='case_id', how='left')
                else:
                    df_ag = df_cat_ag

            if df_ag is not None:
                df_ag.reset_index(inplace=True)
                base = base.merge(df_ag, how='left', on='case_id')
                resumo[arq] = df_ag.shape[1] - 1 # removendo case_id
            else:
                resumo[arq] = 0 # Nenhuma coluna vÃ¡lida para merge

            gc.collect()

        except Exception as e:
            print(f"âš ï¸� Erro em {arq}: {e}")
            continue

    base.reset_index(inplace=True)

    # Separa colunas numÃ©ricas e categÃ³ricas na base final para filtro de variÃ¢ncia
    col_numericas_finais = base.select_dtypes(include='number').columns.tolist()
    col_categoricas_finais = base.select_dtypes(include=['object', 'category']).columns.tolist()

    # Aplica filtro de variÃ¢ncia SOMENTE Ã s colunas numÃ©ricas
    col_numericas_para_filtrar = [c for c in col_numericas_finais if c not in ['case_id', 'target']]
    if col_numericas_para_filtrar: # Garante que hÃ¡ colunas para calcular variÃ¢ncia
        variancia = base[col_numericas_para_filtrar].var() # Calcula variÃ¢ncia
        col_num_pos_variancia = variancia[variancia > min_variancia].index.tolist() # MantÃ©m as colunas com variÃ¢ncia mÃ­nima
    else:
        col_num_pos_variancia = []

    col_essenciais = ['case_id', 'target'] # Estas sempre serÃ£o mantidas
    
    # Combina todas as colunas que devem ser mantidas
    # Colunas essenciais + colunas numÃ©ricas apÃ³s filtro de variÃ¢ncia + todas as categÃ³ricas
    cols_a_manter = list(set(col_essenciais + col_num_pos_variancia + col_categoricas_finais))

    base = base[cols_a_manter] # Filtra o DataFrame final

    # Se RelatÃ³rio = True mostra um resumo final
    if relatorio:
        print(f"\nâœ… Dataset final: {base.shape}")
        print("\nğŸ“‹ Colunas agregadas por arquivo (jÃ¡ filtradas por nulos):")
        for nome, qtd in resumo.items():
            print(f"â€¢ {nome}: {qtd} colunas")
        
        print(f"\nğŸ“‰ Colunas numÃ©ricas mantidas apÃ³s filtro de variÃ¢ncia: {len(col_num_pos_variancia)}")
        print(f"ğŸ—„ï¸� Colunas categÃ³ricas finais mantidas (sem filtro de variÃ¢ncia): {len(col_categoricas_finais)}")
        print(f"ğŸ“Š Total de colunas no dataset final: {base.shape[1]}")

    return base


diretorio = "/kaggle/input/home-credit-credit-risk-model-stability/parquet_files/train" # Caminho do diretÃ³rio
df_modelagem = carregar_dados_merge_filtrado(diretorio, threshold_null=0.05, min_variancia=0.5) # Chamada da funÃ§Ã£o com exibiÃ§Ã£o de relatÃ³rio final


if "index" in df_modelagem.columns:
    df_modelagem.drop("index", axis=1, inplace=True) #Se houver, remove coluna de Ã­ndice sem valor para o modelo


# 5 primeiros registros
df_modelagem.head(5)


#VerificaÃ§Ã£o de duplicatas
df_modelagem.drop_duplicates(subset="case_id", inplace=True)
df_modelagem.head()
gc.collect()


# Estrutura dos dados
df_modelagem.shape


#VerificaÃ§Ã£o de nulos das 20 primeiras colunas
print(df_modelagem.isnull().sum()[:20])


gc.collect()


#PersistÃªncia do arquivo parcialmente tratado
path_saida = "df_modelagem_parcial.parquet"
df_modelagem.to_parquet(path_saida, index=False)
print(f"ğŸ’¾ Dataset salvo em: {path_saida}")
gc.collect()


train_base = df_modelagem.copy()
train_base.tail()


# InformaÃ§Ãµes Ãºteis sobre o conjunto de dados final
gc.collect()
train_base.info()


# Balanceamento da classe
inadimp = train_base["target"].value_counts()
inadimp


inadimp[0]


taxa_inad = inadimp[1] / (inadimp[1] + inadimp[0])
print(str(round(taxa_inad, 4) * 100) + "% dos clientes da empresa se tornaram inadimplentes")


#Resumo EstatÃ­stico
train_base.describe()


# Colunas com maior quantidade de nulos
train_base.isnull().sum().sort_values(ascending=False).head(20)


def filtrar_por_nulos(df: pd.DataFrame, limite: float = 0.15) -> pd.DataFrame:
    """
    Remove colunas com proporÃ§Ã£o de nulos maior que o limite especificado.

    ParÃ¢metros:
    ------------
    df : DataFrame
        Conjunto de dados original.
    limite : float
        ProporÃ§Ã£o mÃ¡xima de nulos (entre 0 e 1) permitida por coluna.

    Retorna:
    --------
    DataFrame apenas com colunas com poucos nulos.
    """
    filtro = df.isnull().mean() <= limite
    return df.loc[:, filtro]


train_base = filtrar_por_nulos(train_base, 0.5) # MantÃ©m colunas com atÃ© 50% de nulos
print("Total de colunas apÃ³s filtro: %d " % train_base.shape[1])


train_base.head()


path_saida = "df_modelagem_pronto.parquet"
train_base.to_parquet(path_saida, index=False)
print(f"ğŸ’¾ Dataset salvo em: {path_saida}")


def codificar_categorias_labelencoder(df: pd.DataFrame, armazenar_codificadores: bool = False):
    """
    Codifica colunas categÃ³ricas com LabelEncoder.

    ParÃ¢metros:
    ------------
    df : DataFrame
        Base de dados original.
    armazenar_codificadores : bool
        Se True, retorna tambÃ©m um dicionÃ¡rio com os LabelEncoders usados.

    Retorna:
    --------
    df_codificado : DataFrame
        Dados com colunas categÃ³ricas codificadas como nÃºmeros inteiros.
    encoders : dict (opcional)
        DicionÃ¡rio {coluna: LabelEncoder} para reutilizaÃ§Ã£o posterior.
    """
    df_codificado = df.copy()
    encoders = {}

    col_categoricas = df_codificado.select_dtypes(include=["object", "category"]).columns

    for col in col_categoricas:
        le = LabelEncoder()
        df_codificado[col] = le.fit_transform(df_codificado[col].astype(str))
        gc.collect()
        if armazenar_codificadores:
            encoders[col] = le
            

    return (df_codificado, encoders) if armazenar_codificadores else df_codificado


def treinar_modelo_lightgbm(df: pd.DataFrame, n_folds: int = 5):

    features = [col for col in df.columns if col not in ['case_id', 'target', 'MONTH', 'WEEK_NUM']]
    X = df[features]                                    # Colunas a serem ignoradas no treinamento do modelo
    y = df['target']

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    importancias = []

    print("ğŸš€ Iniciando treino com LightGBM...")
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        model = LGBMClassifier(n_estimators=50, learning_rate=0.01, 
                               num_leaves=10, random_state=fold)
        model.fit(X_train, y_train,
                  eval_set=[(X_valid, y_valid)],
                  eval_metric='auc')
        
                
        oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]
        importancias.append(pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_,
            'fold': fold
        }))
        
        gc.collect()

    score = roc_auc_score(y, oof_preds)
    print(f"\nğŸ“ˆ AUC final (CV mÃ©dia): {score:.4f}")

    # ImportÃ¢ncia mÃ©dia
    df_imp = pd.concat(importancias).groupby('feature')['importance'].mean().sort_values(ascending=False)
    print("\nğŸ”� Top 10 features mais importantes:")
    print(df_imp.head(10))

    return df_imp


# Libera MemÃ³ria RAM
gc.collect()


#CodificaÃ§Ã£o com LabelEncoder
train_base = codificar_categorias_labelencoder(train_base)
train_base.head()


# Libera MemÃ³ria RAM
gc.collect()


train_base.shape


# Amostragem aleatÃ³ria dos dados para reduzir o uso de memÃ³ria
train_base_final = train_base.sample(frac=0.6, random_state=33)
# Treinar modelo com validaÃ§Ã£o cruzada
treinar_modelo_lightgbm(train_base, n_folds=10)




