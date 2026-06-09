# Instalar pacotes necessários
!pip -q install lifelines
!pip -q install scikit-survival


# Imports
import pandas as pd
import numpy as np
import re
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Carregar dados
train_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
data_dict = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')


# Checar o cabeçalho do CSV
train_data.head()


# Checar a descrição de todas variaveis
data_dict[['variable','description']]


# Separar features e target
X = train_data.drop(['efs', 'efs_time', 'ID'], axis=1)
y = train_data[['efs', 'efs_time']]
y_structured = np.array(
    list(zip(y['efs'], y['efs_time'])),
    dtype=[('event', 'bool'), ('time', 'float64')]
)


# Extrair valores permitidos do dicionário de dados
data_dict['values_parsed'] = data_dict['values'].apply(lambda x: re.findall(r"'([^']*)'|(\bnan\b)", str(x)))
data_dict['allowed_nan'] = data_dict['values_parsed'].apply(lambda lst: any('nan' in item for tuple_ in lst for item in tuple_))

# Printar as únicas colunas que aceitam valores NaN
data_dict[data_dict.allowed_nan == False]


# Identificar features categóricas
categorical_features = data_dict[data_dict['type'] == 'Categorical']['variable'].tolist()
categorical_features.remove("efs")
pd.DataFrame(categorical_features)


# Identificar features numéricas
numerical_features = data_dict[data_dict['type'] == 'Numerical']['variable'].tolist()
numerical_features.remove("efs_time")
pd.DataFrame(numerical_features)


# Pré-processamento

threshold = 0.05 * len(X)
rare_categories = X['prim_disease_hct'].value_counts()[X['prim_disease_hct'].value_counts() < threshold].index
X['prim_disease_hct'] = X['prim_disease_hct'].replace(rare_categories, 'Outros')

# Aplicar a todas as variáveis categóricas (ajuste conforme necessário)
for col in categorical_features:
    counts = X[col].value_counts()
    rare_cats = counts[counts < threshold].index
    X[col] = X[col].replace(rare_cats, 'Outros')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='nan')),  # Assume 'nan' como categoria válida
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ],
    sparse_threshold=0.0
)


# Pipeline do modelo
rsf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomSurvivalForest(random_state=42))
])

# Ajuste de hiperparâmetros para Random Survival Forest (RSF)
param_grid = {
    'classifier__n_estimators': [1, 5],
    'classifier__max_depth': [1, 3]
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(rsf_pipeline, param_grid, cv=cv)
grid_search.fit(X, y_structured) 

# Melhor modelo RSF
best_rsf = grid_search.best_estimator_

# Avaliar com modelo CoxPH
cph_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', CoxPHFitter(penalizer=0.1))
])

# Converter dados para CoxPH
X_processed = preprocessor.fit_transform(X)
X_processed_df = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())
X_processed_df['efs'] = y['efs']
X_processed_df['efs_time'] = y['efs_time']

cph = CoxPHFitter(penalizer=0.1)
cph.fit(X_processed_df, duration_col='efs_time', event_col='efs')


# Validação cruzada para C-index
kf = KFold(n_splits=10, shuffle=True, random_state=42)
c_indices = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Pré-processar dados
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    y_train_structured = np.array(
    list(zip(y_train['efs'], y_train['efs_time'])),
    dtype=[('event', 'bool'), ('time', 'float64')]
    )
    y_test_structured = np.array(
    list(zip(y_test['efs'], y_test['efs_time'])),
    dtype=[('event', 'bool'), ('time', 'float64')]
    )
    
    # Treinar RSF
    rsf = RandomSurvivalForest()
    rsf.fit(X_train_processed, y_train_structured)
    
    # Prever
    risk_scores = rsf.predict(X_test_processed)
    c_index = concordance_index_censored(y_test_structured['efs'], y_test_structured['efs_time'], risk_scores)[0]
    c_indices.append(c_index)

print(f"C-index médio (validação cruzada 10-fold): {np.mean(c_indices):.3f}")


# Gerar previsões nos dados de teste
test_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
test_processed = preprocessor.transform(test_data)
test_preds = best_rsf.predict(test_processed)

# Preparar submissão
submission = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv')
submission['efs'] = test_preds
submission.to_csv('submission.csv', index=False)

