import pandas as pd
import os
import subprocess
import zipfile
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np

from itertools import product
from sklearn.model_selection import GroupShuffleSplit
from itertools import chain
from sklearn.model_selection import GroupKFold


# resetando as configuraÃ§Ãµes 
pd.reset_option('display.max_columns')

# setando as configuraÃ§Ãµes de coluna maxima
pd.set_option('display.max_columns', None)


def download_files():
    # Define caminhos
    zip_path = "data/aeroclub-recsys-2025.zip"
    extract_path = "data/aeroclub"
    
    # Cria a pasta base se necessÃ¡rio
    os.makedirs("data", exist_ok=True)

    # Verifica se o arquivo .zip jÃ¡ foi baixado
    if not os.path.exists(zip_path):
        print("ğŸ”½ Baixando arquivos da competiÃ§Ã£o...")
        subprocess.run([
            "kaggle", "competitions", "download",
            "-c", "aeroclub-recsys-2025",
            "-p", "data"
        ])
    else:
        print("âœ… Arquivo ZIP jÃ¡ existe. Pulando download.")

    # Verifica se os arquivos jÃ¡ foram extraÃ­dos
    if not os.path.exists(extract_path) or not os.listdir(extract_path):
        print("ğŸ“¦ Extraindo arquivos...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    else:
        print("âœ… Arquivos jÃ¡ extraÃ­dos. Pulando extraÃ§Ã£o.")


# Executa
download_files()


train = pd.read_parquet("data/aeroclub/train.parquet")


def reduce_memory_usage(df):
    for col in df.columns:
        col_type = df[col].dtypes
        
        if col_type == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif col_type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif col_type == 'object':
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')
    
    return df
train = reduce_memory_usage(train)


df_train_raw = train.copy()


# Define as colunas que vocÃª quer manter
columns_to_keep = [
    # Identifiers
    'Id',  # num
    'ranker_id', 
    'profileId', 
    'companyID',
    
    # User info
    'sex', 'nationality', 'frequentFlyer', 'isVip', 'bySelf', 'isAccess3D',

    # Company info
    'corporateTariffCode',

    # Search & route
    'searchRoute', 'requestDate',

    # Pricing
    'totalPrice', 'taxes',

    # Flight timing
    'legs0_departureAt', 'legs0_arrivalAt', 'legs0_duration',
    'legs1_departureAt', 'legs1_arrivalAt', 'legs1_duration',

    # Segment-level info (sÃ³ do segmento 0 da ida para simplificar no baseline)
    'legs0_segments0_departureFrom_airport_iata',
    'legs0_segments0_arrivalTo_airport_iata',
    'legs0_segments0_arrivalTo_airport_city_iata',
    'legs0_segments0_marketingCarrier_code',
    'legs0_segments0_operatingCarrier_code',
    'legs0_segments0_aircraft_code',
    'legs0_segments0_flightNumber',
    'legs0_segments0_duration',
    'legs0_segments0_baggageAllowance_quantity',
    'legs0_segments0_baggageAllowance_weightMeasurementType',
    'legs0_segments0_cabinClass',
    'legs0_segments0_seatsAvailable', 
    'legs0_segments1_departureFrom_airport_iata',
    'legs0_segments2_departureFrom_airport_iata',
    'legs0_segments3_departureFrom_airport_iata',

    # Cancellation & exchange rules
    'miniRules0_monetaryAmount', 'miniRules0_percentage', 'miniRules0_statusInfos',
    'miniRules1_monetaryAmount', 'miniRules1_percentage', 'miniRules1_statusInfos',

    # Pricing policy
    'pricingInfo_isAccessTP', 'pricingInfo_passengerCount',

    # Target
    'selected'
]

# Filtra os dados para o baseline
def load_subset(df, columns,  max_rows=None):
    if max_rows:
        return df[columns].iloc[:max_rows].copy()
    else:
        return df[columns].copy()

# Exemplo de uso
df_train = load_subset(df_train_raw, columns_to_keep, max_rows=1_000_000) # ONLY 1M

#############################          IMPORTANT      ########################################
#############################          IMPORTANT      ########################################
#############################          IMPORTANT      ########################################
#############################          IMPORTANT      ########################################
#df_train = load_subset(df_train_raw, columns_to_keep) # ALL REGISTERS


def fix_column_types(df):
    df_fixed = df.copy()
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            # Tenta converter para tipo numÃ©rico
            try:
                df_fixed[col] = pd.to_numeric(df[col])
            except:
                # Se nÃ£o for numÃ©rico, tenta bool
                unique_vals = df[col].dropna().unique()
                if set(unique_vals) <= {True, False}:
                    df_fixed[col] = df[col].astype(bool)
                else:
                    df_fixed[col] = df[col].astype(str)
    return df_fixed
df_train = fix_column_types(df_train)

# Ajusta a nacionalidade (estÃ¡ em Int)
df_train["nationality"] = df_train["nationality"].astype("str")
df_train['companyID'] = df_train['companyID'].astype('category')

df_train.dtypes  # Checar resultado


# 1. Cria a feature 'n_segments_ida' baseada na presenÃ§a de segmentos 1, 2, 3
segments_ida_cols = {
    1: 'legs0_segments1_departureFrom_airport_iata',
    2: 'legs0_segments2_departureFrom_airport_iata',
    3: 'legs0_segments3_departureFrom_airport_iata'
}

# ComeÃ§a com 1 porque o segmento 0 estÃ¡ sempre presente
df_train['n_segments_ida'] = 1

for seg, col in segments_ida_cols.items():
    df_train['n_segments_ida'] += df_train[col].notnull().astype(int)

# 2. Cria a flag booleana se hÃ¡ conexÃµes (mais de 1 segmento na ida)
df_train['has_connections_ida'] = (df_train['n_segments_ida'] > 1).astype('boolean')

df_train.drop('legs0_segments1_departureFrom_airport_iata', inplace=True, axis=1)
df_train.drop('legs0_segments2_departureFrom_airport_iata', inplace=True, axis=1)
df_train.drop('legs0_segments3_departureFrom_airport_iata', inplace=True, axis=1)


def count_frequent_flyers(value):
    if pd.isna(value):
        return 0
    return len(str(value).split('/'))

df_train['frequentFlyer_count'] = df_train['frequentFlyer'].apply(count_frequent_flyers)

# Cria flag binÃ¡ria para frequent flyer
df_train['hasFrequentFlyer'] = df_train['frequentFlyer'].notnull().astype(int)

# Substituir valores NaN por string vazia
ff_series = df_train['frequentFlyer'].fillna('').astype(str)

# Dividir por '/' para obter lista
ff_lists = ff_series.str.split('/')

all_programs = set(chain.from_iterable(ff_lists))
print(f"Total de companhias Ãºnicas: {len(all_programs)}")

df_train.drop('frequentFlyer', axis=1, inplace=True)


# ğŸ—“ï¸� Colunas de datas e horÃ¡rios
cols_datetime = [
    'requestDate',
    'legs0_departureAt', 'legs0_arrivalAt',
    'legs1_departureAt', 'legs1_arrivalAt'
]
def process_datetime_and_duration(df):
    df_processed = df.copy()

    # Datas para datetime
    for col in cols_datetime:
        df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')

    # Features de hora e dia da semana
    df_processed['legs0_dep_hour'] = df_processed['legs0_departureAt'].dt.hour
    df_processed['legs0_dep_dayofweek'] = df_processed['legs0_departureAt'].dt.dayofweek
    df_processed['legs1_dep_hour'] = df_processed['legs1_departureAt'].dt.hour
    df_processed['legs1_dep_dayofweek'] = df_processed['legs1_departureAt'].dt.dayofweek

    # Dias entre ida e volta (duraÃ§Ã£o da viagem)
    df_processed['trip_days'] = (df_processed['legs1_departureAt'] - df_processed['legs0_departureAt']).dt.days

    # Dias de antecedÃªncia (request â†’ ida)
    df_processed['booking_to_trip_days'] = (df_processed['legs0_departureAt'] - df_processed['requestDate']).dt.days

    # Final de semana (ida/volta)
    df_processed['ida_fds'] = df_processed['legs0_dep_dayofweek'].isin([5, 6]).astype(int)
    df_processed['volta_fds'] = df_processed['legs1_dep_dayofweek'].isin([5, 6]).astype(int)

    # HorÃ¡rio comercial (7h Ã s 19h)
    def is_business_hour(hour):
        return int(7 <= hour <= 19)

    df_processed['ida_comercial'] = df_processed['legs0_dep_hour'].apply(is_business_hour)
    df_processed['volta_comercial'] = df_processed['legs1_dep_hour'].apply(is_business_hour)

    # â�±ï¸� Converter colunas de duraÃ§Ã£o para minutos
    def clean_and_convert_duration(col):
        return (
            col
            .fillna("00:00:00")
            .astype(str)
            .str.strip()
            .str.replace("nan", "00:00:00")
            .pipe(pd.to_timedelta, errors='coerce')
            .dt.total_seconds() / 60  # minutos
        )

    cols_duration = ['legs0_duration', 'legs1_duration']
    for col in cols_duration:
        df_processed[col] = clean_and_convert_duration(df_processed[col])

    return df_processed


df_train['legs0_duration_minutes'] = (
    pd.to_timedelta(
        df_train['legs0_segments0_duration'].fillna("00:00:00").astype(str).str.strip(),
        errors='coerce'
    ).dt.total_seconds() / 60  # em minutos
)

df_train.drop('legs0_segments0_duration', axis=1, inplace=True)


# âœ… ApplicaÃ§Ã£o
df_train = process_datetime_and_duration(df_train)
df_train.drop(columns=cols_datetime, inplace=True)


df_train['searchRoute'] = df_train['searchRoute'].astype(str)
df_train['searchRoute_count'] = df_train['searchRoute'].apply(lambda x: x.split("/"))
df_train['searchRoute_count'] = df_train['searchRoute_count'].apply(lambda x: len(x))
print(f" min {min(df_train['searchRoute_count'])}")
print(f" max {max(df_train['searchRoute_count'])}")
df_train.drop('searchRoute_count', axis=1, inplace=True)


# Garante que searchRoute estÃ¡ como string
df_train['searchRoute'] = df_train['searchRoute'].astype(str)

# Separa ida e volta
df_train[['route_ida', 'route_volta']] = df_train['searchRoute'].str.split('/', expand=True)

# Extrai origem e destino da ida
df_train['ida_from'] = df_train['route_ida'].str[:3]
df_train['ida_to'] = df_train['route_ida'].str[3:]

# Extrai origem e destino da volta (se existir)
df_train['volta_from'] = df_train['route_volta'].str[:3]
df_train['volta_to'] = df_train['route_volta'].str[3:]

df_train.drop('searchRoute', axis=1, inplace=True)


df_train.head()


# --- Target e grupo
target_col = "selected"
group_col = "ranker_id"

# --- CategÃ³ricas para LightGBM
categorical_cols = [
    'companyID',
    'nationality',
    'legs0_segments0_departureFrom_airport_iata',
    'legs0_segments0_arrivalTo_airport_iata',
    'legs0_segments0_arrivalTo_airport_city_iata',
    'legs0_segments0_marketingCarrier_code',
    'legs0_segments0_operatingCarrier_code',
    'legs0_segments0_aircraft_code',
    'corporateTariffCode',
    
    # novas features categÃ³ricas da searchRoute
    'route_ida',
    'route_volta',
    'ida_from',
    'ida_to',
    'volta_from',
    'volta_to'
]

# --- Booleanas e numÃ©ricas
boolean_cols = [
    'sex', 'isVip', 'bySelf',
    'pricingInfo_isAccessTP', 'hasFrequentFlyer',
    'ida_fds', 'volta_fds',
    'ida_comercial', 'volta_comercial',
    'isAccess3D',
    'has_connections_ida'
] + [col for col in df_train.columns if col.startswith("ff_")]

numeric_cols = [
    'totalPrice', 'taxes',
    'legs0_duration', 'legs1_duration',
    'legs0_segments0_baggageAllowance_quantity',
    'legs0_segments0_baggageAllowance_weightMeasurementType',
    'legs0_segments0_cabinClass',
    'legs0_segments0_seatsAvailable',
    'miniRules0_monetaryAmount', 'miniRules0_percentage',
    'miniRules1_monetaryAmount', 'miniRules1_percentage',
    'booking_to_trip_days', 'trip_days',
    'legs0_dep_hour', 'legs0_dep_dayofweek',
    'legs1_dep_hour', 'legs1_dep_dayofweek',
    'frequentFlyer_count', 'legs0_duration_minutes'
]
features = numeric_cols + categorical_cols + boolean_cols

# --- Converte categÃ³ricas para category
for col in categorical_cols:
    df_train[col] = df_train[col].astype("category")


# --- SeparaÃ§Ã£o por grupo (ranker_id)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(df_train, groups=df_train["ranker_id"]))

df_train_split = df_train.iloc[train_idx].copy()
df_val = df_train.iloc[val_idx].copy()

# --- Features e targets
X_train = df_train_split[features]
y_train = df_train_split[target_col]
groups_train = df_train_split[group_col].value_counts().sort_index().values

X_val = df_val[features]
y_val = df_val[target_col]
groups_val = df_val[group_col].value_counts().sort_index().values

dataset_params = {
    "max_bin": 63 
}

# --- CriaÃ§Ã£o dos Datasets
train_dataset = lgb.Dataset(
    X_train,
    label=y_train,
    group=groups_train,
    categorical_feature=categorical_cols,
    params=dataset_params  # ğŸ’¡ AQUI Ã© onde max_bin deve ir tambÃ©m!
)

val_dataset = lgb.Dataset(
    X_val,
    label=y_val,
    group=groups_val,
    categorical_feature=categorical_cols,
    reference=train_dataset,
    params=dataset_params
)




param_grid = {
    'learning_rate': [0.05],
    'num_leaves': [63, 127],        
    'min_data_in_leaf': [50, 70, 100]     
}
# Gera todas combinaÃ§Ãµes de parÃ¢metros
param_combinations = list(product(*param_grid.values()))
param_keys = list(param_grid.keys())

best_score = -1
best_model = None
best_params = None

for combo in param_combinations:
    param_set = dict(zip(param_keys, combo))
    print(f"Treinando com: {param_set}")

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [3],
        "boosting_type": "gbdt",
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": 42,
        "verbosity": -1,
        "num_threads": 8,
        **param_set
    }

    model = lgb.train(
        params,
        train_dataset,
        valid_sets=[val_dataset],
        valid_names=["valid"],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50)],
    )

    score = model.best_score["valid"]["ndcg@3"]

    if score > best_score:
        best_score = score
        best_model = model
        best_params = param_set

print("\nâœ… Melhor combinaÃ§Ã£o:")
print(best_params)
print(f"NDCG@3: {best_score:.5f}")



# --- PrediÃ§Ã£o
y_pred = model.predict(X_val)

# --- AvaliaÃ§Ã£o Top-1
df_pred = df_val.copy()
df_pred['y_true'] = y_val
df_pred['y_pred'] = y_pred

df_pred_sorted = df_pred.sort_values(['ranker_id', 'y_pred'], ascending=[True, False])
df_top1 = df_pred_sorted.groupby('ranker_id').head(1)

acertos = df_top1['y_true'].sum()
total = df_top1.shape[0]

print(f"Voos escolhidos corretamente (top1): {acertos} de {total} sessÃµes")
print(f"AcurÃ¡cia top1: {acertos / total:.4f}")


params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [3],
    "learning_rate": best_params['learning_rate'],
    "num_leaves": best_params['num_leaves'],
    "min_data_in_leaf": best_params['min_data_in_leaf'],
    "boosting_type": "gbdt",
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
    "verbosity": -1,
    "num_threads": 8  # âœ… Aqui estÃ¡ correto
}

model = lgb.train(
    params,
    train_dataset,
    valid_sets=[train_dataset, val_dataset],
    valid_names=["train", "valid"],
    num_boost_round=1000, 
    callbacks=[lgb.early_stopping(stopping_rounds=120)],
)



# --- PrediÃ§Ã£o
y_pred = model.predict(X_val)

# --- AvaliaÃ§Ã£o Top-1
df_pred = df_val.copy()
df_pred['y_true'] = y_val
df_pred['y_pred'] = y_pred

df_pred_sorted = df_pred.sort_values(['ranker_id', 'y_pred'], ascending=[True, False])
df_top1 = df_pred_sorted.groupby('ranker_id').head(1)

acertos = df_top1['y_true'].sum()
total = df_top1.shape[0]

print(f"Voos escolhidos corretamente (top1): {acertos} de {total} sessÃµes")
print(f"AcurÃ¡cia top1: {acertos / total:.4f}")




# âœ… Treinamento final com TODO o dataset de treino
#     usando best_iteration encontrado na validaÃ§Ã£o
# ============================================================

X_full = df_train[features]
y_full = df_train[target_col]
groups_full = df_train[group_col].value_counts().sort_index().values

full_dataset = lgb.Dataset(X_full, y_full, group=groups_full, categorical_feature=categorical_cols)

# âš ï¸� Usa o nÃºmero ideal de iteraÃ§Ãµes do treino anterior
final_model = lgb.train(
    params,
    full_dataset,
    num_boost_round=model.best_iteration 
)


final_model.save_model('modelo_final.txt')


model = lgb.Booster(model_file='modelo_final.txt')


# ============================================================
# ## 6. GeraÃ§Ã£o de SubmissÃ£o
# ============================================================

# 1. Ler test.parquet
df_test = pd.read_parquet("data/aeroclub/test.parquet")

# 2. Aplicar transformaÃ§Ãµes mÃ­nimas necessÃ¡rias
df_test['ranker_id'] = df_test['ranker_id'].astype(str)
df_test['nationality'] = df_test['nationality'].astype(str)
df_test['searchRoute'] = df_test['searchRoute'].astype(str)

# --- Frequent Flyer (mesmos one-hot do treino)
df_test['frequentFlyer'] = df_test['frequentFlyer'].fillna('').astype(str)
ff_lists_test = df_test['frequentFlyer'].str.split('/')


all_programs = set(chain.from_iterable(ff_lists))
print(f"Total de companhias Ãºnicas: {len(all_programs)}")


for program in all_programs:
    if program == '':
        continue
    df_test[f'ff_{program}'] = ff_lists_test.apply(lambda x: int(program in x))

for col in [col for col in df_test.columns if col.startswith("ff_")]:
    df_test[col] = df_test[col].astype(pd.BooleanDtype())

df_test['frequentFlyer_count'] = df_test['frequentFlyer'].apply(count_frequent_flyers)
df_test['hasFrequentFlyer'] = df_test['frequentFlyer'].notnull().astype(int)
df_test.drop(columns=['frequentFlyer'], inplace=True)




# --- Datas
cols_datetime = [
    'requestDate',
    'legs0_departureAt', 'legs0_arrivalAt',
    'legs1_departureAt', 'legs1_arrivalAt'
]
for col in cols_datetime:
    df_test[col] = pd.to_datetime(df_test[col], errors='coerce')

df_test['legs0_dep_hour'] = df_test['legs0_departureAt'].dt.hour
df_test['legs0_dep_dayofweek'] = df_test['legs0_departureAt'].dt.dayofweek
df_test['legs1_dep_hour'] = df_test['legs1_departureAt'].dt.hour
df_test['legs1_dep_dayofweek'] = df_test['legs1_departureAt'].dt.dayofweek
df_test['trip_days'] = (df_test['legs1_departureAt'] - df_test['legs0_departureAt']).dt.days
df_test['booking_to_trip_days'] = (df_test['legs0_departureAt'] - df_test['requestDate']).dt.days
df_test['ida_fds'] = df_test['legs0_dep_dayofweek'].isin([5, 6]).astype(int)
df_test['volta_fds'] = df_test['legs1_dep_dayofweek'].isin([5, 6]).astype(int)

df_test['ida_comercial'] = df_test['legs0_dep_hour'].apply(lambda x: int(7 <= x <= 19))
df_test['volta_comercial'] = df_test['legs1_dep_hour'].apply(lambda x: int(7 <= x <= 19))

df_test.drop(columns=cols_datetime, inplace=True)


######################################
# Cria a feature 'n_segments_ida' baseada na presenÃ§a de segmentos 1, 2, 3
segments_ida_cols = {
    1: 'legs0_segments1_departureFrom_airport_iata',
    2: 'legs0_segments2_departureFrom_airport_iata',
    3: 'legs0_segments3_departureFrom_airport_iata'
}

# ComeÃ§a com 1 porque o segmento 0 estÃ¡ sempre presente
df_test['n_segments_ida'] = 1

for seg, col in segments_ida_cols.items():
    df_test['n_segments_ida'] += df_test[col].notnull().astype(int)

# Cria a flag booleana se hÃ¡ conexÃµes (mais de 1 segmento na ida)
df_test['has_connections_ida'] = (df_test['n_segments_ida'] > 1).astype('boolean')

# companyID como categoria
df_test['companyID'] = df_test['companyID'].astype('category')

# isAccess3D como boolean
df_test['isAccess3D'] = df_test['isAccess3D'].astype('boolean')

df_test.drop('legs0_segments1_departureFrom_airport_iata', inplace=True, axis=1)
df_test.drop('legs0_segments2_departureFrom_airport_iata', inplace=True, axis=1)
df_test.drop('legs0_segments3_departureFrom_airport_iata', inplace=True, axis=1)
########################


# --- DuraÃ§Ã£o
def clean_and_convert_duration(col):
    return (
        col
        .fillna("00:00:00")
        .astype(str)
        .str.strip()
        .str.replace("nan", "00:00:00")
        .pipe(pd.to_timedelta, errors='coerce')
        .dt.total_seconds() / 60
    )

df_test['legs0_duration'] = clean_and_convert_duration(df_test['legs0_duration'])
df_test['legs1_duration'] = clean_and_convert_duration(df_test['legs1_duration'])
df_test['legs0_segments0_duration'] = clean_and_convert_duration(df_test['legs0_segments0_duration'])
df_test['legs0_duration_minutes'] = df_test['legs0_duration']
df_test.drop(columns=['legs0_segments0_duration'], inplace=True)

# --- SearchRoute features
df_test[['route_ida', 'route_volta']] = df_test['searchRoute'].str.split('/', expand=True)
df_test['ida_from'] = df_test['route_ida'].str[:3]
df_test['ida_to'] = df_test['route_ida'].str[3:]
df_test['volta_from'] = df_test['route_volta'].str[:3]
df_test['volta_to'] = df_test['route_volta'].str[3:]
df_test.drop('searchRoute', axis=1, inplace=True)



df_train['companyID'] = df_train['companyID'].astype('category')
# Converte para booleano (caso ainda nÃ£o esteja)
df_train['isAccess3D'] = df_train['isAccess3D'].astype('boolean')
# Cria flag indicando se hÃ¡ conexÃµes na ida
df_train['has_connections_ida'] = (df_train['n_segments_ida'] > 1).astype('boolean')

# --- Tipagem
for col in categorical_cols:
    df_test[col] = df_test[col].astype("category")

for col in boolean_cols:
    if col in df_test.columns:
        df_test[col] = df_test[col].astype('boolean')


# Prever com o modelo
X_test = df_test[features]
df_test['y_pred'] = model.predict(X_test)

# 4. Gerar submissÃ£o
df_test_sorted = df_test.sort_values(['ranker_id', 'y_pred'], ascending=[True, False])
df_test_sorted['selected'] = df_test_sorted.groupby('ranker_id').cumcount() + 1

submission = df_test_sorted[['Id', 'ranker_id', 'selected']]
submission.to_csv("submission.csv", index=False)
print("âœ… Arquivo de submissÃ£o salvo como 'submission.csv'")


