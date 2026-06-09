# EDA, pré-processamento e engenharia — dados do MVP com automação e reprodutibilidade
# EDA, preprocesamiento e ingeniería — datos del MVP con automatización y reproducibilidad
# EDA, предварительная обработка и инженерия — данные MVP с автоматизацией и воспроизводимостью
# EDA, preprocessing and engineering — MVP data with automation and reproducibility

import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pandas.api.types import is_numeric_dtype
import warnings

# Ignorar warnings RuntimeWarning | Ignorar advertencias RuntimeWarning | Игнорировать предупреждения RuntimeWarning | Ignore RuntimeWarning warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Listar arquivos em /kaggle/input | Listar archivos en /kaggle/input | Список файлов в /kaggle/input | List files in /kaggle/input
print("Arquivos disponíveis em /kaggle/input:")
for root, _, files in os.walk('/kaggle/input'):
    for f in files:
        print(os.path.join(root, f))

# Ler CSV treino | Leer CSV entrenamiento | Чтение CSV обучения | Read train CSV
train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')

# Ler CSV teste | Leer CSV prueba | Чтение CSV теста | Read test CSV
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

# Ler CSV submissão | Leer CSV submission | Чтение CSV сабмишн | Read submission CSV
submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')

# Mostrar primeiras linhas treino | Mostrar primeras filas entrenamiento | Показать первые строки обучения | Show first train rows
print("\nPrimeiras linhas do dataset de treino:\n", train.head())

# Info geral dataframe | Info general del dataframe | Общая информация о датафрейме | General dataframe info
print("\nInformações gerais do dataset:")
train.info()

# Contar valores ausentes | Contar valores faltantes | Подсчет пропущенных значений | Count missing values
print("\nValores ausentes por coluna:\n", train.isnull().sum())

# Estatísticas descritivas | Estadísticas descriptivas | Описательная статистика | Descriptive statistics
print("\nEstatísticas descritivas:\n", train.describe())

target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Estatísticas variáveis alvo | Estadísticas variables objetivo | Статистика целевых переменных | Target variables statistics
print("\nEstatísticas individuais das variáveis alvo:")
for c in target_cols:
    print(f"\n{c}:\n{train[c].describe()}")

# Matriz correlação alvo | Matriz correlación objetivo | Корреляционная матрица целей | Target correlation matrix
print("\nMatriz de correlação entre as variáveis alvo:\n", train[target_cols].corr())

# Contar SMILES únicos | Contar SMILES únicos | Подсчет уникальных SMILES | Count unique SMILES
print("\nTotal de SMILES únicos:", train['SMILES'].nunique())

# Encontrar SMILES duplicados | Encontrar duplicados SMILES | Поиск дубликатов SMILES | Find duplicate SMILES
duplicates = train[train.duplicated('SMILES')]
print(f"\nNúmero de SMILES duplicados: {len(duplicates)}")
if not duplicates.empty:
    print(duplicates.head())

# Criar coluna comprimento SMILES | Crear columna longitud SMILES | Создать колонку длины SMILES | Create SMILES length column
train['smiles_length'] = train['SMILES'].str.len()

# Estatísticas comprimento SMILES | Estadísticas longitud SMILES | Статистика длины SMILES | SMILES length statistics
print("\nEstatísticas do comprimento dos SMILES:\n", train['smiles_length'].describe())

# Mostrar 5 menores SMILES | Mostrar 5 SMILES más cortos | Показать 5 самых коротких SMILES | Show 5 shortest SMILES
print("\n5 menores SMILES:\n", train.nsmallest(5, 'smiles_length')[['SMILES','smiles_length']])

# Mostrar 5 maiores SMILES | Mostrar 5 SMILES más largos | Показать 5 самых длинных SMILES | Show 5 longest SMILES
print("\n5 maiores SMILES:\n", train.nlargest(5, 'smiles_length')[['SMILES','smiles_length']])

# Extrair features SMILES | Extraer features SMILES | Извлечение признаков SMILES | Extract SMILES features
def smiles_robust_features(smiles):
    atom_counts = {
        'C': len(re.findall(r'C(?![a-z])', smiles)),
        'O': len(re.findall(r'O', smiles)),
        'N': len(re.findall(r'N(?![a-z])', smiles)),
        'S': len(re.findall(r'S(?![a-z])', smiles)),
        'F': len(re.findall(r'F', smiles)),
        'Cl': len(re.findall(r'Cl', smiles)),
        'Br': len(re.findall(r'Br', smiles)),
        'I': len(re.findall(r'I', smiles)),
        'P': len(re.findall(r'P', smiles)),
    }
    features = {
        'smiles_length': len(smiles),
        'num_branches': smiles.count('(') + smiles.count(')'),
        'num_double_bonds': smiles.count('='),
        'num_triple_bonds': smiles.count('#'),
        'num_ring_closures': len(re.findall(r'\d', smiles)),
        'num_aromatic_atoms': len(re.findall(r'[bcnops]', smiles)),
        'num_aliphatic_atoms': len(re.findall(r'[BCNOPSFHI]', smiles)),
        **atom_counts,
    }
    features['num_atoms_total'] = sum(atom_counts.values())
    features['num_atoms_unique'] = sum(v > 0 for v in atom_counts.values())
    return pd.Series(features)

# Aplicar features SMILES no treino | Aplicar features SMILES en entrenamiento | Применить признаки SMILES к обучению | Apply SMILES features to train
train = pd.concat([train, train['SMILES'].apply(smiles_robust_features)], axis=1)

# Preencher NA variáveis alvo | Rellenar NA variables objetivo | Заполнить NA целевые переменные | Fill NA target variables
train[target_cols] = train[target_cols].fillna(train[target_cols].mean())

exclude_cols = ['id', 'SMILES'] + target_cols

# Selecionar colunas numéricas | Seleccionar columnas numéricas | Выбор числовых колонок | Select numeric columns
numerical_cols = [col for col in train.columns if col not in exclude_cols and is_numeric_dtype(train[col])]
print("\nColunas numéricas detectadas automaticamente (ordem preservada):\n", numerical_cols)

# Inicializar scaler | Inicializar scaler | Инициализировать scaler | Initialize scaler
scaler = StandardScaler()

# Normalizar colunas numéricas | Normalizar columnas numéricas | Нормализовать числовые колонки | Normalize numeric columns
train[numerical_cols] = scaler.fit_transform(train[numerical_cols])

# Salvar scaler | Guardar scaler | Сохранить scaler | Save scaler
joblib.dump(scaler, "scaler.pkl")

# Extrair features SMILES teste | Extraer features SMILES test | Извлечь признаки SMILES теста | Extract SMILES features test
X_test_raw = test['SMILES'].apply(smiles_robust_features)

# Ajustar ordem colunas teste | Ajustar orden columnas test | Согласовать порядок колонок теста | Align test columns order
X_test_raw = X_test_raw[numerical_cols]

# Carregar scaler salvo | Cargar scaler guardado | Загрузить сохранённый scaler | Load saved scaler
scaler = joblib.load("scaler.pkl")

# Aplicar scaler no teste | Aplicar scaler en test | Применить scaler к тесту | Apply scaler to test
X_test = scaler.transform(X_test_raw)

# Extrair ids teste | Extraer ids test | Извлечь id теста | Extract test ids
y_test = test['id'].values

# Conferir shape X_test | Revisar forma X_test | Проверить размер X_test | Check X_test shape
print("\nX_test shape:", X_test.shape)

# Conferir shape y_test | Revisar forma y_test | Проверить размер y_test | Check y_test shape
print("y_test shape:", y_test.shape)


# Modelo | Modelo | Модель | Model
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Caminho de dados | Ruta de datos | Путь к данным | Data path
train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')

# Alvos | Objetivos | Цели | Targets
target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Extração simples de features dos SMILES | Extracción de características | Извлечение признаков | Feature extraction
def smiles_robust_features(smiles):
    import re
    atom_counts = {
        'C': len(re.findall(r'C(?![a-z])', smiles)),
        'O': len(re.findall(r'O', smiles)),
        'N': len(re.findall(r'N(?![a-z])', smiles)),
        'S': len(re.findall(r'S(?![a-z])', smiles)),
        'F': len(re.findall(r'F', smiles)),
        'Cl': len(re.findall(r'Cl', smiles)),
        'Br': len(re.findall(r'Br', smiles)),
        'I': len(re.findall(r'I', smiles)),
        'P': len(re.findall(r'P', smiles)),
    }
    features = {
        'smiles_length': len(smiles),
        'num_branches': smiles.count('(') + smiles.count(')'),
        'num_double_bonds': smiles.count('='),
        'num_triple_bonds': smiles.count('#'),
        'num_ring_closures': len(re.findall(r'\d', smiles)),
        'num_aromatic_atoms': len(re.findall(r'[bcnops]', smiles)),
        'num_aliphatic_atoms': len(re.findall(r'[BCNOPSFHI]', smiles)),
        **atom_counts,
    }
    features['num_atoms_total'] = sum(atom_counts.values())
    features['num_atoms_unique'] = sum(v > 0 for v in atom_counts.values())
    return pd.Series(features)

# Aplicar features | Aplicar características | Применить признаки | Apply features
train_feats = train['SMILES'].apply(smiles_robust_features)
test_feats = test['SMILES'].apply(smiles_robust_features)

# Preenchimento de alvos ausentes | Relleno de objetivos | Заполнение целей | Fill missing targets
train[target_cols] = train[target_cols].fillna(train[target_cols].mean())

# Colunas numéricas | Columnas numéricas | Числовые признаки | Numerical columns
numerical_cols = train_feats.columns.tolist()

# Normalização | Normalización | Нормализация | Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(train_feats[numerical_cols])
X_test = scaler.transform(test_feats[numerical_cols])
X_all = np.vstack([X_train, X_test])

# Similaridade por cosseno | Similitud coseno | Косинусное сходство | Cosine similarity
threshold = 0.8
sim_matrix = cosine_similarity(X_all)
edges = np.array(np.nonzero(np.triu(sim_matrix, k=1) > threshold))
edges = np.hstack([edges, edges[::-1]])  # bidirecional
edge_index = torch.tensor(edges, dtype=torch.long)

# GCN manual | GCN manual | Ручной GCN | Manual GCN
class ManualGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, x[col] * norm.unsqueeze(1))
        return self.linear(agg)

# Modelo simples GNN | GNN simple | Простая GNN | Simple GNN
class SimpleGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.gcn1 = ManualGCNLayer(in_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.gcn2 = ManualGCNLayer(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = self.relu(x)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

# Dispositivo | Dispositivo | Устройство | Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tensores | Tensores | Тензоры | Tensors
x_all = torch.tensor(X_all, dtype=torch.float32).to(device)
y_train = train[target_cols].values
y_test = np.zeros((X_test.shape[0], len(target_cols)))
y_all = torch.tensor(np.vstack([y_train, y_test]), dtype=torch.float32).to(device)

# Máscaras | Máscaras | Маски | Masks
n_train = X_train.shape[0]
train_mask = torch.zeros(x_all.size(0), dtype=torch.bool).to(device)
train_mask[:n_train] = True
test_mask = ~train_mask
edge_index = edge_index.to(device)

# Instanciar modelo | Instanciar modelo | Инициализация | Instantiate model
model = SimpleGNN(
    in_channels=x_all.shape[1],
    hidden_channels=64,
    out_channels=len(target_cols)
).to(device)

# Otimizador | Optimizador | Оптимизатор | Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Treinamento | Entrenamiento | Обучение | Training
print("Iniciando treinamento da GNN...")  # PT

for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
    out = model(x_all, edge_index)
    loss = criterion(out[train_mask], y_all[train_mask])
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

# wMAE | wMAE | wMAE | wMAE
def weighted_mae(y_true, y_pred):
    n_samples, n_targets = y_true.shape
    n_j = np.sum(~np.isnan(y_true), axis=0)
    R_j = np.nanmax(y_true, axis=0) - np.nanmin(y_true, axis=0)
    w_j = 1 / (np.sqrt(n_j) * R_j)
    w_j /= w_j.sum()
    mae_j = np.nanmean(np.abs(y_true - y_pred), axis=0)
    wmae = np.sum(w_j * mae_j)
    return wmae

# Avaliação | Evaluación | Оценка | Evaluation
model.eval()
with torch.no_grad():
    train_preds = model(x_all, edge_index)[train_mask].cpu().numpy()
    y_train_true = y_all[train_mask].cpu().numpy()

wmae_train = weighted_mae(y_train_true, train_preds)
print(f"\nWeighted MAE (wMAE) no conjunto de treino: {wmae_train:.6f}")

# Inferência | Inferencia | Предсказание | Inference
with torch.no_grad():
    predictions = model(x_all, edge_index)[test_mask].cpu().numpy()

# Submissão | Envío | Сабмишн | Submission
submission = pd.DataFrame({'id': test['id']})
for i, col in enumerate(target_cols):
    submission[col] = predictions[:, i]

submission.to_csv("submission.csv", index=False)
print("\nArquivo de submissão salvo: submission.csv")

# Prévia | Vista previa | Предпросмотр | Preview
print("\nPrévia da submissão:")
print(submission.head())

