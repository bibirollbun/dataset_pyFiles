import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
plt.rcParams["figure.figsize"] = (16, 6)


import random
import re

import optuna
from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler


train = pd.read_csv('/kaggle/input/chocolate-rating-prediction-ai-edu/chocolate_train.csv')
test = pd.read_csv('/kaggle/input/chocolate-rating-prediction-ai-edu/chocolate_test_new.csv')


train.head()


train.info()


def expand_regions(df, split_mode='duplicate'): 
    """
    Обрабатывает колонку region, разделяя ее на отдельные регионы.
    
    Параметры:
    - df: DataFrame с колонкой 'region'
    - split_mode: 'duplicate' (разбивать на строки) или 'first' (оставлять только первый регион)
    
    Возвращает:
    - Обновленный DataFrame
    """
    if split_mode == 'duplicate':
        df_expanded = df.assign(region=df['Broad Bean Origin'].str.split(r'\s*[(,&/]\s*', regex=True)).explode('region')
        df_expanded['region'] = df_expanded['region'].str.strip()
    elif split_mode == 'first':
        df_expanded = df.copy()
        df_expanded['region'] = df_expanded['Broad Bean Origin'].str.split(r'\s*[(,&/]\s*', regex=True).str[0].str.strip()
    else:
        raise ValueError("split_mode должен быть 'duplicate' или 'first'")
    return df_expanded.reset_index(drop=True)


# Словарь соответствий (дочерние -> основное). Названия регионов
region_mapping = {
    'Venezuela': ['Ven.', 'Ven', 'Venez', 'Venezuela/ Ghana', ],
    'Dominican Republic': ['Domincan Republic', 'Dom. Rep', 'D.R.', 'Domin. Rep', 'Dom. Rep.', 'Dominican Rep.', 'DR'],
    'Costa Rica': ['Cost Rica'],
    'Ecuador': ['Ecuad.', 'Ecu.'],
    'Nicaragua': ['Nic.'],
    'Madagascar': ['Mad.', 'Mad'],
    'Brazil': ['Brasil'],
    'Mexico': ['Mex'],
    'Guatemala': ['Guat.'],
    'Sao Tome & Principe': ['Sao Tome', 'Principe'],
    'Papua New Guinea': ['PNG'],
    'Carribean': ['Carribean(DR/Jam/Tri)'],
    'Indonesia': ['Java', 'Bali'],
    'West Africa': ['Africa'],
    'Trinidad': ['Trinidad-Tobago', 'Trinidad']
}

# Функция для замены названий
def standardize_region(region):
    for main_region, aliases in region_mapping.items():
        if region in aliases:
            return main_region
    return region # Если нет в словаре, оставить как есть



# Словарь маппинга для разновидностей какао сорта
cacao_map = {
'Criollo': ['Criollo', 'Criollo (Porcelana)', 'Criollo (Amarru)', 'Criollo (Ocumare 77)', 'Criollo (Ocumare 67)',
'Criollo (Ocumare 61)', 'Criollo (Ocumare)', 'Criollo (Wild)', 'Criollo, +'],
'Trinitario': ['Trinitario', 'Trinitario (Amelonado)', 'Trinitario, TCGA', 'Trinitario (85% Criollo)'],
'Forastero': ['Forastero', 'Forastero (Arriba)', 'Forastero (Arriba) ASS', 'Forastero (Arriba) ASSS',
'Forastero (Nacional)', 'Forastero (Parazinho)', 'Forastero (Catongo)', 'Forastero (Amelonado)'],
'Nacional': ['Nacional', 'Nacional (Arriba)'],
'Amazonian': ['Amazon', 'Amazon mix', 'Amazon, ICS'],
'Blend': ['Criollo, Trinitario', 'Trinitario, Criollo', 'Trinitario, Forastero', 'Forastero, Trinitario',
'Trinitario, Nacional', 'Criollo, Forastero', 'Blend-Forastero,Criollo', 'Blend'],
'Unknown': ['\xa0', 'EET', np.nan]
}

# Функция для маппинга типов какао
def map_cacao_type(value):
    for category, variants in cacao_map.items():
        if value in variants:
            return category
    return 'Unknown' # Если нет в словаре


known_cocoa_varieties = [
    "Forastero", "Criollo", "Trinitario", "Nacional", "Arriba", "Amelonado",
    "Venezuelan Criollo", "Porcelana", "Chuncho", "Madagascar", "Ecuadorian Nacional", "Heirloom",
    "Wild Bolivian", "Chuao", "Porcelana"
]

# Функция для поиска сорта какао в строке
def find_cocoa_variety(text):
    text = str(text)
    for variety in known_cocoa_varieties:
        if re.search(rf"(^|[\s,.;:]){variety}([\s,.;:]|$)", text, re.IGNORECASE):
            return variety
    return "Unknown" # Если нет в словаре


def fill_unknown_values(df, region_col, value_col, coefficient=0.7):
    for region in df[region_col].unique():
        # Выбираем строки для текущего региона
        region_data = df[df[region_col] == region]
        
        # Находим наиболее часто встречающиеся значения в регионе
        top_values = region_data[value_col].value_counts().index.tolist()
        
        if 'Unknown' in top_values:
            top_values.remove('Unknown')
        
        if top_values:
            # Если только одно значение, вероятность его выбора равна 1
            if len(top_values) == 1:
                fill_value = top_values[0]
            else:
                # Выбираем случайное значение из наиболее часто встречающихся
                probabilities = [coefficient] + [(1 - coefficient) / (len(top_values) - 1)] * (len(top_values) - 1)
                fill_value = np.random.choice(top_values, p=probabilities)
            
            # Заполняем пропуски
            df.loc[(df[region_col] == region) & (df[value_col] == 'Unknown'), value_col] = fill_value
        else:
            top_values = df[value_col].value_counts().index.tolist()
            if 'Unknown' in top_values:
                top_values.remove('Unknown')
            probabilities = [coefficient] + [(1 - coefficient) / (len(top_values) - 1)] * (len(top_values) - 1)
            fill_value = np.random.choice(top_values, p=probabilities)
            df.loc[df[value_col] == 'Unknown', value_col] = fill_value
    return df


cleaned_train = expand_regions(train)
cleaned_train['region'] = cleaned_train['region'].apply(standardize_region)
# Применяем маппинг
cleaned_train['cacao_category'] = cleaned_train['Bean Type'].apply(map_cacao_type)


cleaned_train.cacao_category.value_counts()


cleaned_train["cacao_category_2"] = cleaned_train["Specific Bean Origin"].apply(find_cocoa_variety)
cleaned_train.loc[(cleaned_train.cacao_category == 'Unknown') & (cleaned_train.cacao_category_2 != 'Unknown'), 'cacao_category'] = cleaned_train.cacao_category_2
cleaned_train = fill_unknown_values(cleaned_train, 'region', 'cacao_category')


cleaned_test = expand_regions(test, 'first')
cleaned_test['region'] = cleaned_test['region'].apply(standardize_region)
cleaned_test['region'] = cleaned_test['region'].apply(lambda x: 'Venezuela' if x not in cleaned_train['region'].unique() else x)


cleaned_test['region'].unique()


cleaned_test['cacao_category'] = cleaned_test['Bean Type'].apply(map_cacao_type)
cleaned_test["cacao_category_2"] = cleaned_test["Specific Bean Origin"].apply(find_cocoa_variety)
cleaned_test.loc[(cleaned_test.cacao_category == 'Unknown') & (cleaned_test.cacao_category_2 != 'Unknown'), 'cacao_category'] = cleaned_test.cacao_category_2
cleaned_test = fill_unknown_values(cleaned_test, 'region', 'cacao_category')


cleaned_test.cacao_category.value_counts()


cleaned_test.region.unique()


print('Train shape: {}, {}'.format(*cleaned_train.shape))
print('Test shape: {}, {}'.format(*cleaned_test.shape))


cleaned_train.isna().sum()


for column in cleaned_train.columns:
    most_frequent_value = cleaned_train[column].mode()[0]
    cleaned_train[column].fillna(most_frequent_value, inplace=True)


for column in cleaned_test.columns:
    most_frequent_value = cleaned_test[column].mode()[0]
    cleaned_test[column].fillna(most_frequent_value, inplace=True)


for col in cleaned_train.columns:
    print(f'Nunique {col}: {cleaned_train[col].nunique()}')


cleaned_train["Cocoa Percent"] = cleaned_train["Cocoa Percent"].str[:-1].astype('float64')
cleaned_test["Cocoa Percent"] = cleaned_test["Cocoa Percent"].str[:-1].astype('float64')


train_shape = cleaned_train.shape
cleaned_train.drop_duplicates(subset=cleaned_train.columns.values[1:], keep='last', inplace=True)
if (train_shape == cleaned_train.shape):
    print("# No duplicates")
else:
    print(f"# Duplicates found, {train_shape[0] - cleaned_train.shape[0]} num.")


sns.color_palette()


cat_param = ['region', 'cacao_category', 'Company Location', 'Review']
num_param = ['REF', 'Cocoa Percent', 'Rating']


cleaned_train['dataset'] = 'train'
cleaned_test['dataset'] = 'test'

visualisation_df = pd.concat([cleaned_train, cleaned_test], axis=0)

cleaned_train.drop(columns='dataset', inplace=True)
cleaned_test.drop(columns='dataset', inplace=True)


for feature in cat_param:
    sns.countplot(data = visualisation_df, x=feature, hue='dataset', palette='summer')
    plt.xticks(rotation=60)
    plt.title(f'Distribution of {feature}')
    plt.show()


for features in num_param:
    fig, ax = plt.subplots(nrows=1, ncols=2)
    sns.boxplot(data=visualisation_df, y='dataset', x=features, ax=ax[0], orient='h', palette='hot')
    sns.violinplot(data=visualisation_df, y='dataset', x=features, ax=ax[1], palette='summer')
    plt.show()


for features in num_param:
    sns.histplot(data=visualisation_df, hue='dataset', x=features, kde=True, bins=20)
    plt.show()


correlation_matrix = cleaned_train.corr(numeric_only=True) 
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot = True, fmt = '.2f', cmap = 'coolwarm', mask=mask)
plt.title('Corr matrix')
plt.show()


cols = [
    'Company',
    'Specific Bean Origin',
    'REF',
    'Review',
    'Cocoa Percent',
    'Company Location',
    'region',
    'cacao_category'
]


X = cleaned_train.loc[:, cols]
target = cleaned_train['Rating']
test = cleaned_test.loc[:, cols]


mm_scaler = MinMaxScaler()
# scaler = StandardScaler()

X['REF'] = mm_scaler.fit_transform(X[['REF']])  
test['REF'] = mm_scaler.transform(cleaned_test[['REF']])


mm_scaler = MinMaxScaler()

X['Cocoa Percent'] = mm_scaler.fit_transform(X[['Cocoa Percent']])  
test['Cocoa Percent'] = mm_scaler.transform(test[['Cocoa Percent']])


X


def objective_catboost(trial):
    max_depth = 4 #trial.suggest_int("max_depth", 3, 5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    n_estimators = trial.suggest_int("n_estimators", 1200, 2000)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1, 10)
    random_strength = trial.suggest_float("random_strength", 0, 10)
    bagging_temperature = trial.suggest_float("bagging_temperature", 0, 1)
    # border_count = trial.suggest_int("border_count", 32, 255)

    model = CatBoostRegressor(
        cat_features=[0, 1, 3, 5, 6, 7],
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        l2_leaf_reg=l2_leaf_reg,
        random_strength=random_strength,
        bagging_temperature=bagging_temperature,
        # border_count=border_count,
        silent=True
    )

    score = cross_val_score(model, X, target, cv=3, scoring="r2", n_jobs=-1).mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective_catboost, n_trials=100)


!pip install plotly -q


from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour, plot_slice
plot_optimization_history(study).show()
plot_param_importances(study).show()
plot_contour(study, params=["learning_rate", "n_estimators"]).show()
plot_contour(study, params=["l2_leaf_reg", "random_strength"]).show()
plot_slice(study, params=["learning_rate", "n_estimators", "l2_leaf_reg"]).show()


params = study.best_params
model = CatBoostRegressor(
        cat_features=[0, 1, 5, 6, 7],
        max_depth=4,
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        l2_leaf_reg=params['l2_leaf_reg'],
        random_strength=params['random_strength'],
        bagging_temperature=params['bagging_temperature'],
        # border_count=params['border_count'],
        silent=True
    )
model.fit(X, target)


pred = model.predict(test)


test = test.copy()
result = pd.DataFrame()
result['id'] = np.arange(len(test))
result['Rating'] = pred

result[['id','Rating']].to_csv("submission.csv", index=False)

