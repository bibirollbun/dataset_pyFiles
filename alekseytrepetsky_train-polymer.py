from IPython.display import clear_output as clr
!pip install --no-index --no-deps /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl /kaggle/input/torch-geometric-2-6-1/torch_geometric-2.6.1-py3-none-any.whl
!pip install --no-index --no-deps /kaggle/input/psmiles-whl/psmiles-0.6.10-py3-none-any.whl /kaggle/input/psmiles-whl/canonicalize_psmiles-0.1.2-py3-none-any.whl
clr()


import os
import math
import random
import logging
from tqdm import tqdm
from joblib import Parallel, delayed
from itertools import combinations

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans, DBSCAN, OPTICS, SpectralClustering, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.inspection import permutation_importance
from scipy.spatial.distance import cdist
from scipy.stats import trim_mean, wasserstein_distance
import umap
import hdbscan

import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, GraphDescriptors, rdMolDescriptors

import psmiles
from psmiles import PolymerSmiles as PS

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.nn import (GCNConv, SAGEConv, GATv2Conv, BatchNorm, InstanceNorm, 
                                aggr, global_mean_pool, global_add_pool, global_max_pool)
from torch_geometric.data import Data, Batch, DataLoader, InMemoryDataset
from torch_geometric.utils import from_smiles

from transformers import AutoModelForMaskedLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

is_comp = os.getenv('KAGGLE_IS_COMPETITION_RERUN')
# Отключение логов RDKit
RDLogger.DisableLog('rdApp.*')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clr()


import random
from sklearn.utils import check_random_state
from transformers import set_seed
seed = 0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
check_random_state(seed)
set_seed(seed)

clr()


extrapolation_val = False


chemberta = AutoModelForMaskedLM.from_pretrained("/kaggle/input/transformer-models/model_folder_chemberta")
tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/transformer-models/model_folder_chemberta")
chemberta.eval()
def featurize_ChemBERTa(smiles_list, padding=False):
    embeddings_cls = torch.zeros(len(smiles_list), 600)
    embeddings_mean = torch.zeros(len(smiles_list), 600)

    with torch.no_grad():
        for i, smiles in enumerate(smiles_list):
            encoded_input = tokenizer(smiles, return_tensors="pt",padding=padding,truncation=True)
            model_output = chemberta(**encoded_input)
            
            embedding = model_output[0][::,0,::]
            embeddings_cls[i] = embedding
            
            # embedding = torch.mean(model_output[0],1)
            # embeddings_mean[i] = embedding
            
    return embeddings_cls.numpy()[0].reshape(-1).tolist()#, embeddings_mean.numpy()[0]

clr()

siamese_smole = SentenceTransformer('/kaggle/input/transformer-models/model_folder_siamese')
siamese_smole.eval()
def featurize_siamese_smole(smiles_list):
    embeddings = np.zeros((len(smiles_list), 512))

    with torch.no_grad():
        for i, smiles in enumerate(smiles_list):
            embedding = siamese_smole.encode([smiles],show_progress_bar=False)
            embeddings[i] = embedding
            
    return embeddings.reshape(-1).tolist()
clr()

polyBERT = SentenceTransformer('/kaggle/input/transformer-models/model_folder_polybert')
polyBERT.eval()
def featurize_polyBERT(smiles_list):
    embeddings = np.zeros((len(smiles_list), 600))

    with torch.no_grad():
        for i, smiles in enumerate(smiles_list):
            embedding = polyBERT.encode([smiles],show_progress_bar=False)
            embeddings[i] = embedding
            
    return embeddings.reshape(-1).tolist()
clr()

# Функция для вычисления дескрипторов
def compute_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles.replace("*","C"))
    #num_all_atoms = mol.GetNumAtoms(onlyExplicit=False)
    #try:
        #ps = PS(smiles)
        #n_copy = math.floor(300/num_all_atoms)
        #if n_copy == 0:
            #n_copy = 1
        #extend_smiles = str(ps.random_copolymer(ps, units=n_copy, ratio=0.5)).replace('[*]','C')
        #mol = Chem.MolFromSmiles(extend_smiles)
    #except:
        #pass
    mol = Chem.rdmolops.AddHs(mol)
    if mol is None:
        return [np.nan] * 1585
    desc1 = [desc[1](mol) for desc in Descriptors.descList]
    fp = rdMolDescriptors.GetHashedMorganFingerprint(mol, 3, nBits=256)
    desc1.extend(fp.ToList())
    desc1.extend(featurize_ChemBERTa([smiles]))
    desc1.extend(featurize_siamese_smole([smiles]))
    desc1.extend(featurize_polyBERT([smiles.replace("*","[*]")]))
    return desc1


# Загрузка данных
train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')#[:900]
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

def is_valid_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None

train_df = train_df[train_df['SMILES'].apply(is_valid_smiles)].reset_index(drop=True)

def make_smile_canonical(smile): # To avoid duplicates, for example: canonical '*C=C(*)C' == '*C(=C*)C'
    try:
        mol = Chem.MolFromSmiles(smile)
        canon_smile = Chem.MolToSmiles(mol, canonical=True)
        return canon_smile
    except:
        return None

train_df['SMILES'] = train_df['SMILES'].apply(lambda s: make_smile_canonical(s))
test_df['SMILES'] = test_df['SMILES'].apply(lambda s: make_smile_canonical(s))


# https://www.kaggle.com/datasets/minatoyukinaxlisa/tc-smiles
data_tc = pd.read_csv('/kaggle/input/tc-smiles/Tc_SMILES.csv')
data_tc = data_tc.rename(columns={'TC_mean': 'Tc'})
data_tc['SMILES'] = data_tc['SMILES'].apply(lambda s: make_smile_canonical(s))

# https://springernature.figshare.com/articles/dataset/dataset_with_glass_transition_temperature/24219958?file=42507037
data_tg2 = pd.read_csv('/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv', usecols=['SMILES', 'Tg (C)'])
data_tg2 = data_tg2.rename(columns={'Tg (C)': 'Tg'})
data_tg2['SMILES'] = data_tg2['SMILES'].apply(lambda s: make_smile_canonical(s))

# https://www.sciencedirect.com/science/article/pii/S2590159123000377#ec0005
data_tg3 = pd.read_excel('/kaggle/input/smiles-extra-data/data_tg3.xlsx')
data_tg3 = data_tg3.rename(columns={'Tg [K]': 'Tg'})
data_tg3['Tg'] = data_tg3['Tg'] - 273.15
data_tg3['SMILES'] = data_tg3['SMILES'].apply(lambda s: make_smile_canonical(s))

# https://github.com/Duke-MatSci/ChemProps
data_dnst = pd.read_excel('/kaggle/input/smiles-extra-data/data_dnst1.xlsx')
data_dnst = data_dnst.rename(columns={'density(g/cm3)': 'Density'})[['SMILES', 'Density']]
data_dnst['SMILES'] = data_dnst['SMILES'].apply(lambda s: make_smile_canonical(s))
data_dnst = data_dnst[(data_dnst['SMILES'].notnull())&(data_dnst['Density'].notnull())&(data_dnst['Density'] != 'nylon')]
data_dnst['Density'] = data_dnst['Density'].astype('float64')
data_dnst['Density'] -= 0.118

def add_extra_data(df_train, df_extra, target):
    n_samples_before = len(df_train[df_train[target].notnull()])
    
    df_extra['SMILES'] = df_extra['SMILES'].apply(lambda s: make_smile_canonical(s))
    df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()
    cross_smiles = set(df_extra['SMILES']) & set(df_train['SMILES'])
    unique_smiles_extra = set(df_extra['SMILES']) - set(df_train['SMILES'])

    # Make priority target value from competition's df
    for smile in df_train[df_train[target].notnull()]['SMILES'].tolist():
        if smile in cross_smiles:
            cross_smiles.remove(smile)

    # Imput missing values for competition's SMILES
    for smile in cross_smiles:
        df_train.loc[df_train['SMILES']==smile, target] = df_extra[df_extra['SMILES']==smile][target].values[0]
    
    df_train = pd.concat([df_train, df_extra[df_extra['SMILES'].isin(unique_smiles_extra)]], axis=0).reset_index(drop=True)

    n_samples_after = len(df_train[df_train[target].notnull()])
    print(f'\nFor target "{target}" added {n_samples_after-n_samples_before} new samples!')
    print(f'New unique SMILES: {len(unique_smiles_extra)}')
    return df_train

train_df = add_extra_data(train_df, data_tc, 'Tc')
train_df = add_extra_data(train_df, data_tg2, 'Tg')
# train_df = add_extra_data(train_df, data_tg3, 'Tg')
train_df = add_extra_data(train_df, data_dnst, 'Density')#[:1000]


add1 = pd.read_csv("/kaggle/input/add-data/add_data1.csv")
add2 = pd.read_csv("/kaggle/input/add-data/add_data2.csv")

# 1. Выбор нужных столбцов
add1 = add1[["SMILES", "Tg", "TC_mean"]]

# 2. Переименование TC_mean → Tc
add1 = add1.rename(columns={'TC_mean': 'Tc'})
add2 = add2.rename(columns={'smiles': 'SMILES'})

# 3. Добавление недостающих столбцов с NaN
target_columns = ['SMILES', "Tg", "Tc", 'Eat', 'Eea', 'Egb', 'Egc', 'Ei', 'Xc', 'eps', 'nc']

for column in target_columns:
    if column not in add1.columns:
        add1[column] = np.nan

for column in target_columns:
    if column not in add2.columns:
        add2[column] = np.nan

# 5. Приведение к целевому порядку столбцов
add1 = add1[target_columns]
add2 = add2[target_columns]

add = pd.concat([add1, add2], axis=0)
add = add.dropna(subset=['SMILES'])
add = add[add['SMILES'].apply(is_valid_smiles)].reset_index(drop=True)#[7100:8000]


add.info()


%%time
feats = [f'desc_{i}' for i in range(2185)]

load_feats = False
if not load_feats:
    add['descriptors'] = add['SMILES'].apply(compute_all_descriptors)
    train_df['descriptors'] = train_df['SMILES'].apply(compute_all_descriptors)
    test_df['descriptors'] = test_df['SMILES'].apply(compute_all_descriptors)
    
    # Разворачиваем дескрипторы в столбцы
    add = pd.concat([add, pd.DataFrame(add['descriptors'].tolist(), columns=[f'desc_{i}' for i in range(2185)])], axis=1)
    train_df = pd.concat([train_df, pd.DataFrame(train_df['descriptors'].tolist(), columns=[f'desc_{i}' for i in range(2185)])], axis=1)
    test_df = pd.concat([test_df, pd.DataFrame(test_df['descriptors'].tolist(), columns=[f'desc_{i}' for i in range(2185)])], axis=1)

    # !pip install fancyimpute
    # clr()
    # from fancyimpute import KNN
    
    # add[feats] = add[feats].replace([np.inf, -np.inf], np.nan)
    # train_df[feats] = train_df[feats].replace([np.inf, -np.inf], np.nan)
    # test_df[feats] = test_df[feats].replace([np.inf, -np.inf], np.nan)
    
    # combined_df = pd.concat([train_df[feats], test_df[feats], add[feats]], axis=0)
    
    # imputed_data = KNN(k=5).fit_transform(combined_df)
    
    # train_df[feats] = imputed_data[:len(train_df)]
    # test_df[feats] = imputed_data[len(train_df):len(train_df) + len(test_df)]
    # add[feats] = imputed_data[len(train_df) + len(test_df):]
    
    add[feats] = add[feats].replace([np.inf, -np.inf], np.nan)
    train_df[feats] = train_df[feats].replace([np.inf, -np.inf], np.nan)
    test_df[feats] = test_df[feats].replace([np.inf, -np.inf], np.nan)
    
    # Находим столбцы, где процент пропусков <= 50%
    threshold = 0.5
    feats1 = [col for col in feats if train_df[col].isna().mean() <= threshold]
    feats2 = [col for col in feats if add[col].isna().mean() <= threshold]
    feats3 = [col for col in feats if test_df[col].isna().mean() <= threshold]
    feats = list(set(feats1) & set(feats2) & set(feats3))
    
    # Заменяем NaN на среднее значение по столбцу
    for col in feats:
        mean_val = add[col].mean()
        add[col] = add[col].fillna(mean_val)
        
        mean_val = train_df[col].mean()
        train_df[col] = train_df[col].fillna(mean_val)
        
        mean_val = test_df[col].mean()
        test_df[col] = test_df[col].fillna(mean_val)
    
    # run time CPU 2:18:00
elif not is_comp:
    train_df = pd.read_csv("/kaggle/input/polyfeats/train_df_feats.csv", low_memory=False)
    test_df = pd.read_csv("/kaggle/input/polyfeats/test_df_feats.csv", low_memory=False)
    add = pd.read_csv("/kaggle/input/polyfeats/add_df_feats.csv", low_memory=False)

    feats = [col for col in feats if not train_df[col].isna().any()]
else:
    train_df = pd.read_csv("/kaggle/input/polyfeats/train_df_feats.csv", low_memory=False)
    add = pd.read_csv("/kaggle/input/polyfeats/add_df_feats.csv", low_memory=False)
    
    test_df['descriptors'] = test_df['SMILES'].apply(compute_all_descriptors)
    test_df = pd.concat([test_df, pd.DataFrame(test_df['descriptors'].tolist(), columns=[f'desc_{i}' for i in range(2185)])], axis=1)
    test_df[feats] = test_df[feats].replace([np.nan, np.inf, -np.inf], np.nan)

    feats = [col for col in feats if not train_df[col].isna().any()]
    test_df[feats] = test_df[feats].replace([np.nan, np.inf, -np.inf], 0)


feats = [feats[i] for i in range(len(feats)) if train_df[feats].iloc[i].std() > 0]


# # Определяем признаки
# feats = [f'desc_{i}' for i in range(2185)]
# add[feats] = add[feats].replace([np.nan, np.inf, -np.inf], 0)
# train_df[feats] = train_df[feats].replace([np.nan, np.inf, -np.inf], 0)
# test_df[feats] = test_df[feats].replace([np.nan, np.inf, -np.inf], 0)


# Разделение на train и val с учетом пропусков
target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Вычисляем количество пропущенных свойств
missing_count = train_df[target_columns].isnull().sum(axis=1)

# Создаем группы для стратификации
stratify_groups = pd.cut(
    missing_count,
    bins=[-1, 0, 1, 2, 3, 5],
    labels=['0', '1', '2', '3', '4+']
)

# Разделение данных
train_idx, val_idx = train_test_split(
    train_df.index,
    test_size=0.05,
    stratify=stratify_groups,
    random_state=seed
)

train_split = train_df.loc[train_idx].reset_index(drop=True)
val_split = train_df.loc[val_idx].reset_index(drop=True)

def create_extrapolation_split_single_df(train_df, target_columns, extrapolation_frac=0.10):
    """
    Создание train и val выборок для экстраполяции с единым валидационным датафреймом.
    Для молекул, выбранных как экстремальные по одному таргету, остальные таргеты заменяются на NaN.
    Args:
        train_split: pandas DataFrame - тренировочные данные
        val_split: pandas DataFrame - валидационные данные
        target_columns: list - список целевых свойств ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        extrapolation_frac: float - доля экстремальных значений для валидации
    Returns:
        train_split_extra: pandas DataFrame - тренировочная выборка
        val_split_extra: pandas DataFrame - единая валидационная выборка
    """
    
    # Список для хранения масок экстраполяции по каждому таргету
    masks = []
    target_assignments = []  # Для отслеживания, по какому таргету выбрана молекула
    
    for target_name in target_columns:
        y = train_df[target_name].values
        target_mask = ~np.isnan(y)
        
        # Пороги для экстремальных значений
        low_threshold = np.percentile(y[target_mask], 100 * extrapolation_frac)
        high_threshold = np.percentile(y[target_mask], 100 * (1 - extrapolation_frac))
        
        # Маски для нижних и верхних экстремальных значений
        low_extrapolation = (y <= low_threshold) & target_mask
        high_extrapolation = (y >= high_threshold) & target_mask
        
        # Комбинированная маска для текущего таргета
        extrapolation_mask = low_extrapolation | high_extrapolation
        masks.append(extrapolation_mask)
        target_assignments.append(np.full_like(y, fill_value=target_name, dtype=object) * extrapolation_mask)
    
    # Комбинированная маска для валидационной выборки
    combined_val_mask = np.any(masks, axis=0)
    
    # Определяем, по какому таргету каждая молекула попала в валидацию
    target_assignment = np.full(len(train_df), '', dtype=object)
    for mask, target in zip(masks, target_columns):
        target_assignment[mask] = target
    
    # Создаем валидационную выборку
    val_split_extra = train_df[combined_val_mask].copy()
    
    # Заменяем значения соседних таргетов на NaN
    for idx in val_split_extra.index:
        selected_target = target_assignment[idx]
        for other_target in target_columns:
            if other_target != selected_target:
                val_split_extra.loc[idx, other_target] = np.nan
    
    # Тренировочная выборка: исключаем все молекулы, попавшие в валидацию
    train_split_extra = train_df[~combined_val_mask].copy()
    
    # Проверка размеров
    print("Train split size:", len(train_split_extra))
    print("Validation split size:", len(val_split_extra))
    
    # Проверка пропусков и состава валидационной выборки
    print("\nValidation split non-NaN counts:")
    print(val_split_extra[target_columns].notnull().sum())
    print("\nValidation split target assignments:")
    print(pd.Series(target_assignment[combined_val_mask]).value_counts())
    
    return train_split_extra, val_split_extra

# Применение
target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
if extrapolation_val:
    train_split, val_split = create_extrapolation_split_single_df(train_df, target_columns, extrapolation_frac=0.20)

# Применение PLS для каждого таргета
pls_models = {}
n_components = 10  # Количество компонент для каждого таргета
pls_features_add = []
pls_features_train = []
pls_features_val = []
pls_features_test = []

for target in target_columns:
    # Инициализация PLS
    pls = PLSRegression(n_components=n_components)
    
    # Маска для непропущенных значений таргета
    mask = train_split[target].notnull()
    
    # Обучение PLS только на непропущенных данных
    if mask.sum() > 0:  # Проверяем, что есть данные для обучения
        X_train = train_split.loc[mask, feats].values
        y_train = train_split.loc[mask, target].values
        pls.fit(X_train, y_train)
        pls_models[target] = pls
        
        # Трансформация данных
        pls_add = pls.transform(add[feats].values)
        pls_train = pls.transform(train_split[feats].values)
        pls_val = pls.transform(val_split[feats].values)
        pls_test = pls.transform(test_df[feats].values)
        
        # Сохранение сжатых признаков
        pls_features_add.append(pls_add)
        pls_features_train.append(pls_train)
        pls_features_val.append(pls_val)
        pls_features_test.append(pls_test)

# Конкатенация сжатых признаков
add_features_pls = np.concatenate(pls_features_add, axis=1)
train_features_pls = np.concatenate(pls_features_train, axis=1)
val_features_pls = np.concatenate(pls_features_val, axis=1)
test_features_pls = np.concatenate(pls_features_test, axis=1)

add_features_pls = np.concatenate([add_features_pls, add[feats].values], axis=1)
train_features_pls = np.concatenate([train_features_pls, train_split[feats].values], axis=1)
val_features_pls = np.concatenate([val_features_pls, val_split[feats].values], axis=1)
test_features_pls = np.concatenate([test_features_pls, test_df[feats].values], axis=1)

# Обновление списка признаков
feats = [f'pls_{i}' for i in range(train_features_pls.shape[1])]

# Масштабирование признаков
feature_scaler = StandardScaler()
train_features_scaled = feature_scaler.fit_transform(train_features_pls)
val_features_scaled = feature_scaler.transform(val_features_pls)
test_features_scaled = feature_scaler.transform(test_features_pls)

add_features_scaled = feature_scaler.transform(add_features_pls)

clr()


features_scaled = np.vstack([train_features_scaled,val_features_scaled,
                             test_features_scaled,add_features_scaled])

umap_model = umap.UMAP(n_components=200, n_neighbors=20, min_dist=0.1, 
                       metric='euclidean', random_state=42)
X_umap = umap_model.fit_transform(features_scaled)

features_scaled2 = np.hstack([features_scaled, X_umap])

k_range = range(1, 31)
inertia = []
for k in tqdm(k_range):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for K-Means')
plt.show()


kmeans = KMeans(n_clusters=8, random_state=0)
labels = kmeans.fit_predict(X_umap)

# PCA 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_umap)

# Визуализация PCA 2D с K-Means
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.title('PCA 2D with K-Means Clustering')
plt.tight_layout()
plt.show()

# PCA 3D
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_umap)

# Визуализация PCA 3D с K-Means
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap='viridis')
plt.title('PCA 3D with K-Means Clustering')
plt.tight_layout()
plt.show()

# UMAP 2D
umap_model = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, 
                       metric='euclidean', random_state=42)
X_umap = umap_model.fit_transform(features_scaled)

# Визуализация UMAP 2D с K-Means
plt.figure(figsize=(10, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap='viridis')
plt.title('UMAP 2D with K-Means Clustering')
plt.tight_layout()
plt.show()

# UMAP 3D
umap_model = umap.UMAP(n_components=3, n_neighbors=20, min_dist=0.1, 
                       metric='euclidean', random_state=42)
X_umap = umap_model.fit_transform(features_scaled)

# Визуализация UMAP 3D с K-Means
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_umap[:, 0], X_umap[:, 1], X_umap[:, 2], c=labels, cmap='viridis')
plt.title('UMAP 3D with K-Means Clustering')
plt.tight_layout()
plt.show()


l_train = len(train_features_scaled)
l_val = len(val_features_scaled)
l_test = len(test_features_scaled)
l_add = len(add_features_scaled)
train_features_scaled = features_scaled2[:l_train]
val_features_scaled = features_scaled2[l_train:l_train+l_val]
test_features_scaled = features_scaled2[l_train+l_val:l_train+l_val+l_test]
add_features_scaled = features_scaled2[l_train+l_val+l_test:]


target_columns = ["Tg", "Tc", 'Eat', 'Eea', 'Egb', 'Egc', 'Ei', 'Xc', 'eps', 'nc']
for target in target_columns:
    scaler = MinMaxScaler()
    mask = add[target].notnull()
    values = add.loc[mask, target].values.reshape(-1, 1)
    scaler.fit(values)
    add[target + '_scaled'] = np.nan
    add.loc[mask, target + '_scaled'] = scaler.transform(values).flatten()



target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
# Масштабирование целевых переменных
target_scalers = {}
for target in target_columns:
    scaler = MinMaxScaler()
    mask = train_split[target].notnull()
    values = train_split.loc[mask, target].values.reshape(-1, 1)
    scaler.fit(values)
    train_split[target + '_scaled'] = np.nan
    val_split[target + '_scaled'] = np.nan
    train_split.loc[mask, target + '_scaled'] = scaler.transform(values).flatten()
    mask_val = val_split[target].notnull()
    val_split.loc[mask_val, target + '_scaled'] = scaler.transform(val_split.loc[mask_val, target].values.reshape(-1, 1)).flatten()
    target_scalers[target] = scaler

# Подготовка тензоров
target_columns = ["Tg", "Tc", 'Eat', 'Eea', 'Egb', 'Egc', 'Ei', 'Xc', 'eps', 'nc']
add_features_tensor = torch.tensor(add_features_scaled, dtype=torch.float32).to(device)
add_targets_tensor = torch.tensor(add[[t + '_scaled' for t in target_columns]].values, dtype=torch.float32).to(device)
target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
train_features_tensor = torch.tensor(train_features_scaled, dtype=torch.float32).to(device)
train_targets_tensor = torch.tensor(train_split[[t + '_scaled' for t in target_columns]].values, dtype=torch.float32).to(device)
with torch.no_grad():
    val_features_tensor = torch.tensor(val_features_scaled, dtype=torch.float32).to(device)
    val_targets_tensor = torch.tensor(val_split[[t + '_scaled' for t in target_columns]].values, dtype=torch.float32).to(device)
    test_features_tensor = torch.tensor(test_features_scaled, dtype=torch.float32).to(device)

clr()
# Проверка распределения
print("Размер тренировочного набора:", len(train_split))
print("Размер валидационного набора:", len(val_split))
print("\nРаспределение пропусков в тренировочном наборе:")
print(train_split[target_columns].isnull().sum())
print("\nРаспределение пропусков в валидационном наборе:")
print(val_split[target_columns].isnull().sum())


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Получаем атомы (узлы) с набором признаков
    atom_features = []
    for atom in mol.GetAtoms():
        # 1. Атомный номер как индекс для one-hot (1-54, например)
        atomic_num = atom.GetAtomicNum() - 1  # Сдвигаем, чтобы H=0, He=1 и т.д.
        
        # 2. Степень атома (0-5)
        degree_idx = min(atom.GetDegree(), 5)
        
        # 3. Гибридизация (sp, sp2, sp3, другое)
        hybr = atom.GetHybridization()
        hybr_idx = {Chem.HybridizationType.SP: 0, 
                    Chem.HybridizationType.SP2: 1, 
                    Chem.HybridizationType.SP3: 2}.get(hybr, 3)  # 3 - "другое"
        
        # 4. Ароматичность (0 или 1)
        aromatic_idx = 1 if atom.GetIsAromatic() else 0
        
        # 5. Число водородов (0-4)
        h_count_idx = min(atom.GetTotalNumHs(), 4)
        
        # 6. Формальный заряд (-1, 0, 1)
        charge_idx = atom.GetFormalCharge() + 1  # Сдвиг: -1 -> 0, 0 -> 1, 1 -> 2
        
        # Собираем индексы для one-hot кодирования
        atom_features.append([atomic_num, degree_idx, hybr_idx, aromatic_idx, h_count_idx, charge_idx])
    
    # Преобразуем в тензор
    indices = torch.tensor(atom_features, dtype=torch.long)
    num_nodes = len(indices)
    
    # Размеры признаков: атомный номер (54), степень (6), гибридизация (4), ароматичность (2), водороды (5), заряд (3)
    feature_dims = [54, 6, 4, 2, 5, 3]  # Атомный номер до 54, как в твоём примере
    total_dim = sum(feature_dims)  # 74
    
    # Инициализируем x и заполняем one-hot для каждого признака
    x = torch.zeros((num_nodes, total_dim), dtype=torch.float)
    offset = 0
    for i, dim in enumerate(feature_dims):
        x.scatter_(1, indices[:, i].unsqueeze(1) + offset, 1)
        offset += dim
    
    # Получаем связи (ребра)
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  # Ненаправленный граф
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)


def process_row(data):
    try:
        # graph = from_smiles(data[0], with_hydrogen=True)
        graph, n_seq = smiles_to_graph(data[0])
        if graph is not None:
            graph.x = graph.x.to(torch.float32)
            graph.smiles_features = data[1].reshape(1,-1)
            if data[2] is not None:
                graph.y = data[2].reshape(1,-1)
            graph.n_seq = torch.tensor(n_seq).reshape(1,-1)
            return graph
    except Exception as e:
        pass
    return None

def prepare_data(list_smles, smiles_feats, targets=[None]*10000):
    dataset = []
    for i in tqdm(range(len(list_smles))):
        graph = process_row([list_smles[i], smiles_feats[i], targets[i]])
        if graph is not None:
            dataset.append(graph)
    
    return dataset


# Настройка логирования для отладки
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def smiles_to_graph(smiles, return_descriptors=False):
    """
    Преобразует SMILES-строку в графовое представление молекулы для PyTorch Geometric.
    Опционально возвращает RDKit-подобные дескрипторы для Bilinear Transduction.
    
    Args:
        smiles (str): SMILES-строка молекулы.
        return_descriptors (bool): Если True, возвращает также дескрипторы молекулы.
    
    Returns:
        Data: Объект PyTorch Geometric с x (признаки узлов), edge_index (ребра), edge_attr (признаки ребер).
        np.ndarray (optional): Вектор дескрипторов молекулы (если return_descriptors=True).
    """
    try:
        # Проверка SMILES
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return None, None if return_descriptors else None
        
        # Получаем атомы (узлы) с набором признаков
        atom_features = []
        for atom in mol.GetAtoms():
            # 1. Атомный номер (1-100, H=1 -> idx=0)
            atomic_num = atom.GetAtomicNum() - 1
            if atomic_num < 0 or atomic_num >= 100:
                logger.warning(f"Invalid atomic number {atomic_num + 1} in SMILES: {smiles}")
                atomic_num = 0  # По умолчанию H
            atomic_num = min(atomic_num, 99)
            
            # 2. Степень атома (0-6)
            degree_idx = min(atom.GetDegree(), 6)
            
            # 3. Гибридизация (sp, sp2, sp3, sp3d, sp3d2, другое)
            hybr = atom.GetHybridization()
            hybr_idx = {
                Chem.HybridizationType.SP: 0,
                Chem.HybridizationType.SP2: 1,
                Chem.HybridizationType.SP3: 2,
                Chem.HybridizationType.SP3D: 3,
                Chem.HybridizationType.SP3D2: 4
            }.get(hybr, 5)  # 5 - "другое"
            
            # 4. Ароматичность (0 или 1)
            aromatic_idx = 1 if atom.GetIsAromatic() else 0
            
            # 5. Число водородов (0-5)
            h_count_idx = min(atom.GetTotalNumHs(), 5)
            
            # 6. Формальный заряд (-2, -1, 0, 1, 2 -> 0, 1, 2, 3, 4)
            charge = atom.GetFormalCharge()
            charge_idx = charge + 2  # Сдвиг: -2 -> 0, ..., 2 -> 4
            if charge_idx < 0 or charge_idx >= 5:
                logger.warning(f"Invalid charge {charge} in SMILES: {smiles}")
                charge_idx = 2  # По умолчанию 0
            
            # 7. Входит ли атом в кольцо (0 или 1)
            ring_idx = 1 if atom.IsInRing() else 0
            
            atom_features.append([atomic_num, degree_idx, hybr_idx, aromatic_idx, h_count_idx, charge_idx, ring_idx])
        
        # Преобразуем в тензор
        indices = torch.tensor(atom_features, dtype=torch.long)
        num_nodes = len(indices)
        
        # Размеры признаков: атомный номер (100), степень (7), гибридизация (6), ароматичность (2), водороды (6), заряд (5), кольцо (2)
        feature_dims = [80, 7, 6, 2, 6, 5, 2]
        total_dim = sum(feature_dims)  # 128
        
        # Проверка индексов
        for i, dim in enumerate(feature_dims):
            if (indices[:, i] < 0).any() or (indices[:, i] >= dim).any():
                logger.error(f"Invalid index in feature {i} for SMILES: {smiles}, indices: {indices[:, i]}")
                return None, None if return_descriptors else None
        
        # One-hot кодирование
        x = []
        for i, dim in enumerate(feature_dims):
            one_hot = F.one_hot(indices[:, i], num_classes=dim).float()
            x.append(one_hot)
        x = torch.cat(x, dim=1)  # Shape: (num_nodes, total_dim)
        
        # Получаем связи (ребра) и их признаки
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.extend([[i, j], [j, i]])  # Ненаправленный граф
            # Тип связи: одинарная (1), двойная (2), тройная (3), ароматическая (4)
            bond_type = bond.GetBondType()
            bond_type_idx = {
                Chem.BondType.SINGLE: 1,
                Chem.BondType.DOUBLE: 2,
                Chem.BondType.TRIPLE: 3,
                Chem.BondType.AROMATIC: 4
            }.get(bond_type, 1)  # По умолчанию одинарная
            edge_attr.extend([bond_type_idx, bond_type_idx])
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = F.one_hot(torch.tensor(edge_attr, dtype=torch.long) - 1, num_classes=4).float()
        else:
            # Для молекул без связей (например, один атом)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 4), dtype=torch.float)
        
        # Создаем объект Data
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
        
        
        return data, 600/mol.GetNumAtoms()
    
    except Exception as e:
        # logger.error(f"Error processing SMILES {smiles}: {str(e)}")
        return None, None if return_descriptors else None


add_dataset = prepare_data(add['SMILES'].tolist(), add_features_tensor, add_targets_tensor)
train_dataset = prepare_data(train_split['SMILES'].tolist(), train_features_tensor, train_targets_tensor)
val_dataset = prepare_data(val_split['SMILES'].tolist(), val_features_tensor, val_targets_tensor)
test_dataset = prepare_data(test_df['SMILES'].tolist(), test_features_tensor)


train_dataset[0]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

batch_size = 128
add_loader = DataLoader(add_dataset, batch_size=batch_size, shuffle=True,
    worker_init_fn=seed_worker,  # Для воспроизводимости в многопоточности
    generator=g,  # Генератор для управления перемешиванием
                       )
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
    worker_init_fn=seed_worker,  # Для воспроизводимости в многопоточности
    generator=g,  # Генератор для управления перемешиванием
                         )
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
    worker_init_fn=seed_worker,  # Для воспроизводимости в многопоточности
    generator=g,  # Генератор для управления перемешиванием
                       )
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
    worker_init_fn=seed_worker,  # Для воспроизводимости в многопоточности
    generator=g,  # Генератор для управления перемешиванием
                        )


target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
weights = torch.tensor([0.24578393, 0.03106061, 0.23833992, 0.24242424, 0.2423913], device=device)
weights = train_df[target_columns].isnull().sum(axis=0).values
weights = torch.tensor(weights/weights.sum(), device=device)

target_columns = ["Tg", "Tc", 'Eat', 'Eea', 'Egb', 'Egc', 'Ei', 'Xc', 'eps', 'nc']
weights_add = add[target_columns].isnull().sum(axis=0).values
weights_add = torch.tensor(weights_add/weights_add.sum(), device=device)
target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']


class GraphEmb(nn.Module):
    def __init__(self, input_dim=108, hidden_dim=256, heads=8) -> None:
        super().__init__()
        self.pre_norm = BatchNorm(input_dim)
        self.emb = torch.nn.Linear(input_dim, hidden_dim)
        self.emb_norm = BatchNorm(hidden_dim)
        
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim//heads, heads=heads)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim//heads, heads=heads)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim//heads, heads=heads)
        self.conv4 = GATv2Conv(hidden_dim, hidden_dim//heads, heads=heads)
        self.g_norm1 = BatchNorm(hidden_dim)
        self.g_norm2 = BatchNorm(hidden_dim)
        self.g_norm3 = BatchNorm(hidden_dim)
        self.g_norm4 = BatchNorm(hidden_dim)
        
        self.global_min_pool = aggr.MinAggregation()

    def forward(self, x, edge_index, smiles_features, batch):
        
        x = self.pre_norm(x)
        x = self.emb(x)
        x = self.emb_norm(x)
        x = F.relu(x)
        
        # Слои GCN
        x = self.conv1(x, edge_index)
        x = self.g_norm1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.g_norm2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = self.g_norm3(x)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = self.g_norm4(x)
        x = F.relu(x)
        
        # Агрегация по графу (пулинг)
        x_mean = global_mean_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_min = self.global_min_pool(x, batch)

        x = torch.cat([x_mean, x_sum, x_max, x_min, smiles_features], dim=1)
        
        return x

class PolymerNet(nn.Module):
    def __init__(self, input_dim=1224+len(feats), hidden_dim1=2048, hidden_dim2=1024, dropout_rate=0.1):
        super(PolymerNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.graph_emb = GraphEmb()
        self.heads = nn.ModuleList([nn.Linear(hidden_dim2, 512) for _ in range(5)])
        self.out = nn.Linear(512*5, 10)
    
    def forward(self, batch_g):
        x = self.graph_emb(batch_g.x, batch_g.edge_index, batch_g.smiles_features, batch_g.batch) 
        x = self.backbone(x)
        x = [head(x) for head in self.heads]
        x = torch.cat(x, dim=1)
        x = torch.relu(x)
        out = self.out(x)
        # print(out[:,-1].shape, batch_g.n_seq.shape)
        out[:,-1:] = out[:,-1:] * batch_g.n_seq
        return out


def masked_l1_loss(predictions, targets, weights=None):
    # Создаем маску валидных значений (1 - валидное, 0 - NaN)
    mask = ~torch.isnan(targets)
    
    # Заменяем NaN в таргетах на 0 (для безопасных вычислений)
    safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
    
    # Вычисляем абсолютные ошибки
    losses = torch.abs(predictions - safe_targets)
    
    # Обнуляем потери для невалидных элементов
    masked_losses = losses * mask.float()
    
    # Применяем веса к каждому столбцу, если weights задан
    if weights is not None:
        # Проверяем, что размерность весов совпадает с количеством столбцов
        assert weights.shape[0] == masked_losses.shape[1], \
            f"Размерность весов ({weights.shape[0]}) не совпадает с количеством столбцов ({masked_losses.shape[1]})"
        # Расширяем веса до размерности [1, num_columns] для поэлементного умножения
        weights = weights.view(1, -1)
        masked_losses = masked_losses * weights
    
    # Суммируем все валидные потери
    total_loss = masked_losses.sum()
    
    # Считаем общее количество валидных элементов
    valid_count = mask.sum()
    
    return total_loss / valid_count

def masked_r2_score(predictions, targets, weights=None):
    # Создаем маску валидных значений (1 - валидное, 0 - NaN)
    mask = ~torch.isnan(targets)
    
    # Заменяем NaN в таргетах на 0 (для безопасных вычислений)
    safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
    safe_predictions = torch.where(mask, predictions, torch.zeros_like(predictions))
    
    # Вычисляем среднее значение таргетов для валидных элементов
    valid_count = mask.sum().float()
    mean_targets = (safe_targets * mask.float()).sum() / valid_count
    
    # Вычисляем SS_tot (total sum of squares)
    ss_tot = ((safe_targets - mean_targets) ** 2 * mask.float())
    
    # Вычисляем SS_res (residual sum of squares)
    ss_res = ((safe_targets - safe_predictions) ** 2 * mask.float())
    
    # Применяем веса к каждому столбцу, если weights задан
    if weights is not None:
        # Проверяем, что размерность весов совпадает с количеством столбцов
        assert weights.shape[0] == ss_res.shape[1], \
            f"Размерность весов ({weights.shape[0]}) не совпадает с количеством столбцов ({ss_res.shape[1]})"
        # Расширяем веса до размерности [1, num_columns] для поэлементного умножения
        weights = weights.view(1, -1)
        ss_tot = ss_tot * weights
        ss_res = ss_res * weights
    
    # Суммируем все значения
    ss_tot_sum = ss_tot.sum()
    ss_res_sum = ss_res.sum()
    
    # Вычисляем R² = 1 - SS_res/SS_tot
    # Добавляем небольшую константу в знаменатель для избежания деления на 0
    r2 = 1 - ss_res_sum / (ss_tot_sum + 1e-10)
    
    return r2
    
def masked_r2_score_per_column(predictions, targets, weights=None):
    # Создаем маску валидных значений (1 - валидное, 0 - NaN)
    mask = ~torch.isnan(targets)
    
    # Заменяем NaN в таргетах и предсказаниях на 0 (для безопасных вычислений)
    safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
    safe_predictions = torch.where(mask, predictions, torch.zeros_like(predictions))
    
    # Вычисляем среднее значение таргетов для валидных элементов по каждому столбцу
    valid_count = mask.sum(dim=0).float()  # [num_columns]
    mean_targets = (safe_targets * mask.float()).sum(dim=0) / valid_count  # [num_columns]
    
    # Расширяем mean_targets до размерности [batch_size, num_columns]
    mean_targets = mean_targets.unsqueeze(0).expand_as(safe_targets)
    
    # Вычисляем SS_tot (total sum of squares) по каждому столбцу
    ss_tot = ((safe_targets - mean_targets) ** 2 * mask.float()).sum(dim=0)  # [num_columns]
    
    # Вычисляем SS_res (residual sum of squares) по каждому столбцу
    ss_res = ((safe_targets - safe_predictions) ** 2 * mask.float()).sum(dim=0)  # [num_columns]
    
    # Применяем веса к каждому столбцу, если weights задан
    if weights is not None:
        # Проверяем, что размерность весов совпадает с количеством столбцов
        assert weights.shape[0] == ss_res.shape[0], \
            f"Размерность весов ({weights.shape[0]}) не совпадает с количеством столбцов ({ss_res.shape[0]})"
        # Применяем веса
        weights = weights.view(-1)
        ss_tot = ss_tot * weights
        ss_res = ss_res * weights
    
    # Вычисляем R² = 1 - SS_res/SS_tot для каждого столбца
    # Добавляем небольшую константу в знаменатель для избежания деления на 0
    r2 = 1 - ss_res / (ss_tot + 1e-10)  # [num_columns]
    return r2


model = PolymerNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-6)

if is_comp:
    num_epochs = 100
else:
    num_epochs = 100

add_losses = []

for epoch in range(num_epochs):
    
    epoch_seed = seed + epoch
    g.manual_seed(epoch_seed)
    
    # Обучение
    model.train()
    add_loss = 0
    for batch in add_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        predictions = model(batch)
        loss = masked_l1_loss(predictions, batch.y, weights_add)
        loss.backward()
        optimizer.step()
        add_loss += loss.item()
    add_loss /= len(add_loader)
    add_losses.append(add_loss)
    
    # Обновление шедулера
    # scheduler.step(epoch + 1)
    
    print(f'Epoch {epoch+1}, Add Loss: {add_loss:.6f}')

# Построение графика потерь
plt.figure(figsize=(10, 5))
plt.plot(add_losses, label='Add Loss')
plt.xlabel('Epoch')
plt.ylabel('Masked L1 Loss')
plt.title('Add Loss')
plt.ylim(0, 0.1)
plt.legend()
plt.grid(True)
plt.show()


model.out = nn.Linear(512 * 5, 5)
model = model.to(device)
# Оптимизатор и шедулер
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-6)

train_losses = []
val_losses = []
best_val_loss = float('inf')
best_model_path = 'best_model.pth'

for epoch in range(num_epochs):
    
    epoch_seed = seed + epoch
    g.manual_seed(epoch_seed)
    
    # Обучение
    model.train()
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        predictions = model(batch)
        loss = masked_l1_loss(predictions, batch.y, weights)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Валидация
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            predictions = model(batch)
            loss = masked_l1_loss(predictions, batch.y, weights)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    # Обновление шедулера
    # scheduler.step(epoch + 1)
    
    # Сохранение лучшей модели
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
    
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

# Построение графика потерь
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Masked L1 Loss')
plt.title('Training and Validation Losses')
plt.ylim(0, 0.05)
plt.legend()
plt.grid(True)
plt.show()


#model.load_state_dict(torch.load(best_model_path))

# Предсказание на тестовом наборе
model.eval()
val_predictions_scaled = []
with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(device)
        predictions = model(batch)
        val_predictions_scaled.append(predictions)

val_predictions_scaled = torch.cat(val_predictions_scaled, axis=0)
masked_l1_loss(val_predictions_scaled, val_targets_tensor), masked_r2_score(val_predictions_scaled, val_targets_tensor), masked_r2_score_per_column(val_predictions_scaled, val_targets_tensor)


# (tensor(0.0243, device='cuda:0'),
# tensor(0.8971, device='cuda:0'),
# tensor([0.8014, 0.8120, 0.7354, 0.9457, 0.7003], device='cuda:0'))

# (tensor(0.0229, device='cuda:0'),
# tensor(0.8083, device='cuda:0'),
# tensor([0.6605, 0.8485, 0.9054, 0.8812, 0.5903], device='cuda:0'))

# (tensor(0.0257, device='cuda:0'),
# tensor(0.7921, device='cuda:0'),
# tensor([0.6401, 0.6844, 0.9290, 0.8610, 0.6433], device='cuda:0'))

# (tensor(0.0260, device='cuda:0'),
#  tensor(0.7471, device='cuda:0'),
#  tensor([0.5507, 0.4351, 0.9176, 0.8693, 0.6886], device='cuda:0'))

# (tensor(0.0247, device='cuda:0'),
#  tensor(0.7961, device='cuda:0'),
#  tensor([0.6546, 0.6628, 0.8756, 0.8696, 0.7077], device='cuda:0'))

# (tensor(0.0289, device='cuda:0'),
#  tensor(0.5874, device='cuda:0'),
#  tensor([ 0.6376, -0.6430,  0.8857,  0.8356,  0.7238], device='cuda:0'))

# (tensor(0.5353, device='cuda:0'),
#  tensor(0.4934, device='cuda:0'),
#  tensor([0.3720, 0.5047, 0.7177, 0.2330, 0.5458], device='cuda:0'))

# (tensor(0.0230, device='cuda:0'),
#  tensor(0.8093, device='cuda:0'),
#  tensor([0.6248, 0.8067, 0.9298, 0.8563, 0.6459], device='cuda:0'))

# (tensor(0.5238, device='cuda:0'),
#  tensor(0.5053, device='cuda:0'),
#  tensor([0.4088, 0.5206, 0.7330, 0.1838, 0.5003], device='cuda:0'))


# Загрузка лучшей модели
# model.load_state_dict(torch.load(best_model_path))

# Предсказание на тестовом наборе
model.eval()
test_predictions_scaled = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        predictions = model(batch).cpu().numpy()
        test_predictions_scaled.append(predictions)

test_predictions_scaled = np.concatenate(test_predictions_scaled, axis=0)

# Обратное масштабирование предсказаний
submission = pd.DataFrame({'id': test_df['id']})
for i, target in enumerate(target_columns):
    scaler = target_scalers[target]
    preds = test_predictions_scaled[:, i].reshape(-1, 1)
    submission[target] = scaler.inverse_transform(preds).flatten()

# Сохранение результатов
submission.to_csv('submission.csv', index=False)


submission


!find /kaggle/working -mindepth 1 ! -name 'submission.csv' -exec rm -rf {} +

