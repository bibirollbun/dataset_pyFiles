pip install symspellpy


pip install pymorphy3


# ======= ПОДКЛЮЧЕНИЕ БИБЛИОТЕК =======

import random
import numpy as np
import pandas as pd
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import chi2_contingency
import scipy.sparse as sp

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import RobustScaler, QuantileTransformer

from datetime import datetime
from collections import Counter
from functools import lru_cache

import re
import pymorphy3
from symspellpy import SymSpell, Verbosity

from sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings('ignore')


# ======= КОНФИГУРАЦИЯ ПАЙПЛАЙНА =======

class Config:
    SEED = 42
    IS_VALIDATION = False
    
    DATA_PATHS = {
        'train': '/kaggle/input/21-vek-by-searched-products-ranking/train.csv',
        'test': '/kaggle/input/21-vek-by-searched-products-ranking/test.csv', 
        'products': '/kaggle/input/21-vek-by-searched-products-ranking/products.csv'
    }
    
    SAMPLING_CONFIG = {
        'min_group_size': 5,
        'validation_group_size': 20,
        'max_group_size': 20,
        'validation_months': 2
    }
    
    TEXT_PROCESSING = {
        'lemmatize_min_length': 3,
        'batch_size': 100000,
        'chunk_size': 100000
    }
    
    SPELL_CHECKER_CONFIG = {
        'max_edit_distance': 6,
        'prefix_length': 7,
        'count_threshold': 1,
    }
    
    BM25_CONFIG = {
        'k1': 1.5,
        'b': 0.75,
        'vocab_size': 50000,
        'batch_size': 100000,
        'query_column': 'query_model',
        'document_column': 'model'
    }
    
    EMBEDDING_CONFIG = {
        'model_name': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        'chunk_size': 100000,
        'device': 'cpu',
        'normalize_embeddings': True,
        'show_progress_bar': True,
        'convert_to_tensor': False,
        'text_configs': [
            {
                'name': 'query_text',
                'source': 'single_column',
                'column': 'query_general',
            },
            {
                'name': 'category_text',
                'source': 'combined_columns',
                'columns': ['type', 'category_name_1', 'category_name_2', 
                           'category_name_3', 'category_name_4', 'country'],
            }
        ]
    }
    
    LGBM_PARAMS = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'boosting_type': 'gbdt',
        'lambdarank_norm': True,
        'lambdarank_truncation_level': 20,
        'ndcg_eval_at': [20],
        
        'num_leaves': 127,
        'min_data_in_leaf': 50,
        'min_child_samples': 20,
        'reg_alpha': 0.2,
        'reg_lambda': 0.2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'subsample_freq': 1,

        'learning_rate': 0.05,
        'n_estimators': 500,
        'n_jobs': 8,
        
        'max_depth': -1,
        'max_bin': 255,
        'verbosity': -1,
        'force_row_wise': True
    }
    
    OUTPUT_CONFIG = {
        'submission_file': 'submission.csv',
        'feature_importance_file': 'lgb_feature_importance.csv',
    }


# ======= УТИЛИТЫ ДЛЯ РАБОТЫ С КОНФИГОМ ======= 

def print_complete_config_summary(config):
    print("# ==== КОНФИГУРАЦИЯ ПАЙПЛАЙНА ====")
    
    sections = {
        "ДАННЫЕ": config.DATA_PATHS,
        "СЭМПЛИРОВАНИЕ": config.SAMPLING_CONFIG,
        "ТЕКСТОВАЯ ОБРАБОТКА": config.TEXT_PROCESSING,
        "BM25": config.BM25_CONFIG,
        "ЭМБЕДДИНГИ": config.EMBEDDING_CONFIG,
        "LIGHTGBM": {k: v for k, v in config.LGBM_PARAMS.items()},
        "ВЫВОД": config.OUTPUT_CONFIG
    }
    
    for section_name, section_data in sections.items():
        print(f"\n{section_name}")
        print("-" * 40)
        for key, value in section_data.items():
            print(f"  {key:.<30} {value}")

config = Config()
np.random.seed(config.SEED)
random.seed(config.SEED)


# ===== КОНСТАНТЫ ДЛЯ ТЕКСТОВОЙ ПРЕДОБРАБОТКИ =====

WORD_PATTERNS = {
    'word': re.compile(r'^[a-zA-Z0-9\-\+]+$'),
    'percent': re.compile(r'(\d+)%'),
    'size': re.compile(r'(\d+)[xх](\d+)'),
    'plus': re.compile(r'([a-zA-Zа-яА-Я]+)[+]'),
    'non_word': re.compile(r'[^\w\s\.\+-]'),
    'multi_space': re.compile(r'\s+'),
    'latin': re.compile('[a-zA-Z]'),
    'letters_digits': re.compile(r'([a-zA-Zа-яА-Я]+)(\d+)'),
    'digits_letters': re.compile(r'(\d+)([a-zA-Zа-яА-Я]+)'),
    'comma_numbers': re.compile(r'(\d+),(\d+)')
}

# ===== СЛОВАРЬ ДЛЯ ТРАНСЛИТЕРАЦИИ =====

SPECIAL_CASES = {
    # Бренды и модели
    'iphone': 'айфон', 'apple': 'эпл', 'ipad': 'айпад', 'macbook': 'макбук',
    'galaxy': 'галакси', 'thinkpad': 'тинкпад', 'xbox': 'ксбокс',
    'windows': 'виндоус', 'android': 'андроид', 'samsung': 'самсунг',
    'nokia': 'нокиа', 'lenovo': 'леново', 'huawei': 'хуавей',
    'xiaomi': 'сяоми', 'oppo': 'оппо', 'vivo': 'виво', 'realme': 'рилми',
    'oneplus': 'ванплас', 'google': 'гугл', 'sony': 'сони',
    'lg': 'элджи', 'asus': 'асус', 'acer': 'асер', 'dell': 'делл',
    'hp': 'хп', 'canon': 'канон', 'nikon': 'никон', 'band': 'бенд',
    'playstation': 'плейстейшн', 'nintendo': 'нинтендо',
    'watch': 'вотч', 'nfc': 'нфс', 'plus': 'плюс', 'synergetic': 'синергетик', 
    
    # Основные цвета
    'red': 'красный', 'blue': 'синий', 'green': 'зеленый', 'yellow': 'желтый', 'black': 'черный', 'white': 'белый',
    
    # Фиолетовые
    'purple': 'фиолетовый', 'violet': 'фиолетовый', 'lavender': 'лавандовый', 'lilac': 'сиреневый', 'plum': 'сливовый',
    'magenta': 'пурпурный', 'orchid': 'орхидея', 'mauve': 'розовато-лиловый', 'indigo': 'индиго', 
    
    # Розовые
    'pink': 'розовый', 'rose': 'розовый', 'coral': 'коралловый', 'salmon': 'лососевый', 'fuchsia': 'фуксия',
    
    # Оранжевые и коричневые
    'orange': 'оранжевый', 'brown': 'коричневый', 'tan': 'желто-коричневый', 'beige': 'бежевый', 'cream': 'кремовый',
    'peach': 'персиковый', 'apricot': 'абрикосовый', 'caramel': 'карамельный', 'chocolate': 'шоколадный',
    
    # Серые и металлические
    'gray': 'серый', 'silver': 'серебристый', 'charcoal': 'угольный', 'graphite': 'графитовый', 'slate': 'аспидный',
    'steel': 'стальной', 'gunmetal': 'оружейный металл', 'pewter': 'оловянный', 'ash': 'пепельный',
    
    # Синие оттенки
    'navy': 'темно-синий', 'cyan': 'голубой', 'turquoise': 'бирюзовый', 'teal': 'сине-зеленый', 'aqua': 'аква',
    'sky': 'небесный', 'royal': 'королевский', 'cobalt': 'кобальтовый', 'cerulean': 'лазоревый',
    
    # Золотые и желтые
    'gold': 'золотой', 'bronze': 'бронзовый', 'copper': 'медный', 'amber': 'янтарный', 'mustard': 'горчичный',
    'lemon': 'лимонный', 'honey': 'медовый', 'vanilla': 'ванильный',
    
    # Темные
    'midnight': 'полуночный', 'ebony': 'эбеновый', 'espresso': 'эспрессо', 'mahogany': 'красное дерево',
    'maroon': 'бордовый', 'burgundy': 'бургундский', 'chestnut': 'каштановый',
    
    # Светлые и пастельные
    'ivory': 'слоновая кость', 'pearl': 'жемчужный', 'alabaster': 'алебастровый', 'porcelain': 'фарфоровый',
    'champagne': 'шампань', 'butter': 'сливочный', 'mist': 'туманный', 'frost': 'морозный',
    
    # Яркие и неоновые
    'neon': 'неоновый', 'electric': 'электрический', 'fluorescent': 'флуоресцентный', 'crimson': 'малиновый',
    'scarlet': 'алый', 'ruby': 'рубиновый', 'vermillion': 'киноварь',
    
    # Природные
    'olive': 'оливковый', 'khaki': 'хаки', 'mint': 'мятный', 'lime': 'лаймовый', 'forest': 'лесной',
    'jade': 'нефритовый', 'emerald': 'изумрудный', 'malachite': 'малахитовый', 'pine': 'сосновый',
    'sea': 'морской', 'ocean': 'океанский', 'lagoon': 'лагуна', 'arctic': 'арктический', 'glacier': 'ледниковый'
}


# ===== ПРАВИЛА ТРАНСЛИТЕРАЦИИ =====

TRANSLITERATION_RULES = [
    ('sch', 'щ'), ('ch', 'ч'), ('sh', 'ш'), ('zh', 'ж'), 
    ('th', 'т'), ('ph', 'ф'), ('kh', 'х'),
    ('you', 'ю'), ('ya', 'я'), ('ye', 'ье'), ('yo', 'йо'),
    ('ia', 'ия'), ('io', 'ио'), ('iu', 'ю'), ('yu', 'ю'),
    ('a', 'а'), ('b', 'б'), ('c', 'к'), ('d', 'д'), 
    ('e', 'е'), ('f', 'ф'), ('g', 'г'), ('h', 'х'),
    ('i', 'и'), ('j', 'дж'), ('k', 'к'), ('l', 'л'),
    ('m', 'м'), ('n', 'н'), ('o', 'о'), ('p', 'п'),
    ('q', 'к'), ('r', 'р'), ('s', 'с'), ('t', 'т'),
    ('u', 'у'), ('v', 'в'), ('w', 'в'), ('x', 'кс'),
    ('y', 'й'), ('z', 'з')
]


# ===== РАСКЛАДКИ КЛАВИАТУРЫ =====

KEYBOARD_LAYOUTS = {
    'ru_to_en': {
        'й': 'q', 'ц': 'w', 'у': 'e', 'к': 'r', 'е': 't', 'н': 'y', 'г': 'u', 
        'ш': 'i', 'щ': 'o', 'з': 'p', 'х': '[', 'ъ': ']', 'ф': 'a', 'ы': 's', 
        'в': 'd', 'а': 'f', 'п': 'g', 'р': 'h', 'о': 'j', 'л': 'k', 'д': 'l', 
        'ж': ';', 'э': "'", 'я': 'z', 'ч': 'x', 'с': 'c', 'м': 'v', 'и': 'b', 
        'т': 'n', 'ь': 'm', 'б': ',', 'ю': '.', 'ё': '`'
    },
    'en_to_ru': {
        'q': 'й', 'w': 'ц', 'e': 'у', 'r': 'к', 't': 'е', 'y': 'н', 'u': 'г',
        'i': 'ш', 'o': 'щ', 'p': 'з', '[': 'х', ']': 'ъ', 'a': 'ф', 's': 'ы',
        'd': 'в', 'f': 'а', 'g': 'п', 'h': 'р', 'j': 'о', 'k': 'л', 'l': 'д',
        ';': 'ж', "'": 'э', 'z': 'я', 'x': 'ч', 'c': 'с', 'v': 'м', 'b': 'и',
        'n': 'т', 'm': 'ь', ',': 'б', '.': 'ю', '`': 'ё'
    }
}

CYRILLIC_CHARS = set(KEYBOARD_LAYOUTS['ru_to_en'].keys())
LATIN_CHARS = set(KEYBOARD_LAYOUTS['en_to_ru'].keys())


# ===== СПИСОК КОЛОНОК ДЛЯ УДАЛЕНИЯ =====

COLUMNS_TO_DROP = [
    # Исходные числовые колонки
    'price', 'price_discount', 'browse_time', 'response_time', 'product_position',
    
    # Временные колонки
    'date_of_create', 'date', 
    
    # Идентификаторы
    'selected_category_id', 'user_number', 'search_number', 'session_product_id', 'actual_product_id',
    
    # Текстовые колонки
    'query', 'category_name_1', 'category_name_2', 'category_name_3', 'category_name_4',
    'query_model', 'query_general', 'found_brand',
    'brand', 'model', 'country', 'type', 'device',
    'available', 'price_is_discounted',
    
    # Временные метрики продуктов
    'cumulative_views', 'cumulative_clicks', 'cumulative_carts', 
    'cumulative_purchases', 'cumulative_relevance_sum',
    'cumulative_user_clicks', 'cumulative_user_purchases',
    'cumulative_user_sessions', 'cumulative_filters_sum',
    
    # Целевая переменная
    'relevance',
    
    # Временные фичи
    'price_is_discounted', 'brand_category_count',
    'query_global_frequency', 'query_word_count', 'query_avg_word_length',
    'cumulative_sessions_count', 'cumulative_complexity_sum',
    'prev_relevance', 'prev_brand'
]


train_df = pd.read_csv(config.DATA_PATHS['train'])
test_df = pd.read_csv(config.DATA_PATHS['test'])
products_df = pd.read_csv(config.DATA_PATHS['products'])


train = train_df.copy()
test = test_df.copy()
products = products_df.copy()
train = train.merge(products, on='product_id', how='left')
train['date'] = pd.to_datetime(train['date'])


print("Размеры данных:")
print(f"Train: {train.shape}, Test: {test.shape}, Products: {products.shape}")


train.info()


test.info()


products.info()


train.head(10)


products.head(10)


def analyze_missing_and_outliers(train, products):
    print("\n=== АНАЛИЗ ПРОПУСКОВ И ВЫБРОСОВ ===")
    
    print("Пропуски в train:")
    for col in train.columns:
        if train[col].isna().sum() > 0:
            print(f"  {col}: {train[col].isna().sum():,} ({train[col].isna().mean():.1%})")
    
    print("\nПропуски в products:")
    for col in products.columns:
        if products[col].isna().sum() > 0:
            print(f"  {col}: {products[col].isna().sum():,} ({products[col].isna().mean():.1%})")

    browse_outliers = train[train['browse_time'] > train['browse_time'].quantile(0.99)]
    print(f"\nВыбросы browse_time (>99%): {len(browse_outliers):,} записей")
    print(f"Конверсия выбросов: {(browse_outliers['relevance'] >= 1).mean():.3f}")

    product_position_outliers = train[train['product_position'] > train['product_position'].quantile(0.99)]
    print(f"\nВыбросы product_position (>99%): {len(product_position_outliers):,} записей")
    print(f"Конверсия выбросов: {(product_position_outliers['relevance'] >= 1).mean():.3f}")

    price_outliers = train[train['price'] > train['price'].quantile(0.99)]
    print(f"\nВыбросы price (>99%): {len(price_outliers):,} записей")
    print(f"Конверсия выбросов: {(price_outliers['relevance'] >= 1).mean():.3f}")

    response_time_outliers = train[train['response_time'] > train['response_time'].quantile(0.99)]
    print(f"\nВыбросы response_time (>99%): {len(response_time_outliers):,} записей")
    print(f"Конверсия выбросов: {(response_time_outliers['relevance'] >= 1).mean():.3f}")

analyze_missing_and_outliers(train, products)


print("=== БАЗОВОЕ СРАВНЕНИЕ TRAIN/TEST ===")

print(f"\n--- Основные метрики ---")
print(f"Уникальные пользователи: train={train['user_number'].nunique():,}, test={test['user_number'].nunique():,}")
print(f"Уникальные сессии: train={train['search_number'].nunique():,}, test={test['search_number'].nunique():,}")
print(f"Уникальные товары: train={train['product_id'].nunique():,}, test={test['product_id'].nunique():,}")



print("=== СРАВНЕНИЕ ДАТ ===")

train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

print(f"Диапазон дат train: {train['date'].min()} - {train['date'].max()}")
print(f"Диапазон дат test: {test['date'].min()} - {test['date'].max()}")
print(f"Разрыв между выборками: {(test['date'].min() - train['date'].max()).days} день")


fig, ax = plt.subplots(figsize=(16, 6))

train_daily = pd.to_datetime(train['date']).dt.date.value_counts().sort_index() / train['date'].nunique()
test_daily = pd.to_datetime(test['date']).dt.date.value_counts().sort_index() / test['date'].nunique()

ax.plot(train_daily.index, train_daily.values, label='Train', linewidth=2)
ax.plot(test_daily.index, test_daily.values, label='Test', linewidth=2)

ax.set_title('Средняя активность по дням', fontsize=16)
ax.set_ylabel('Среднее количество записей в день', fontsize=12)
ax.legend(fontsize=12)
ax.tick_params(labelsize=10)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


numeric_cols = ['product_position', 'response_products', 'response_time', 'browse_time', 'price', 'price_discount']
    

def detailed_distribution_comparison(train, test):
    print("=== СРАВНЕНИЕ РАСПРЕДЕЛЕНИЙ ===")
    
    comparison_results = []
    
    for col in numeric_cols:
        if col in train.columns and col in test.columns:
            train_stats = {
                'mean': train[col].mean(),
                'std': train[col].std(),
                'min': train[col].min(),
                'max': train[col].max(),
                'median': train[col].median(),
                'q95': train[col].quantile(0.95)
            }
            
            test_stats = {
                'mean': test[col].mean(),
                'std': test[col].std(),
                'min': test[col].min(),
                'max': test[col].max(),
                'median': test[col].median(),
                'q95': test[col].quantile(0.95)
            }
            
            mean_shift = (test_stats['mean'] - train_stats['mean']) / train_stats['mean'] * 100
            median_shift = (test_stats['median'] - train_stats['median']) / train_stats['median'] * 100
            
            comparison_results.append({
                'feature': col,
                'train_mean': train_stats['mean'],
                'test_mean': test_stats['mean'],
                'mean_shift_pct': mean_shift,
                'train_median': train_stats['median'],
                'test_median': test_stats['median'],
                'median_shift_pct': median_shift,
                'train_std': train_stats['std'],
                'test_std': test_stats['std']
            })

    comp_df = pd.DataFrame(comparison_results)
    
    print("\n--- Сравнение числовых признаков ---")
    for _, row in comp_df.iterrows():
        print(f"\n{row['feature']}:")
        print(f"  Среднее: train={row['train_mean']:.2f}, test={row['test_mean']:.2f} (смещение: {row['mean_shift_pct']:+.1f}%)")
        print(f"  Медиана: train={row['train_median']:.2f}, test={row['test_median']:.2f} (смещение: {row['median_shift_pct']:+.1f}%)")
        print(f"  Std: train={row['train_std']:.2f}, test={row['test_std']:.2f}")
    
    return comp_df

distribution_comparison = detailed_distribution_comparison(train, test)


def statistical_tests(train, test):
    print("\n=== СТАТИСТИЧЕСКИЕ ТЕСТЫ РАЗЛИЧИЙ ===")

    for col in numeric_cols:
        t_stat, p_value = stats.ttest_ind(train[col].dropna(), test[col].dropna(), equal_var=False)
        ks_stat, ks_pvalue = stats.ks_2samp(train[col].dropna(), test[col].dropna())
        
        print(f"\n{col}:")
        print(f"  t-тест: p-value={p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'NS'}")
        print(f"  KS-тест: p-value={ks_pvalue:.6f} {'***' if ks_pvalue < 0.001 else '**' if ks_pvalue < 0.01 else '*' if ks_pvalue < 0.05 else 'NS'}")

statistical_tests(train, test)


def plot_distributions(train, test):

    fig, axes = plt.subplots(6, 1, figsize=(12, 30))
    
    for i, col in enumerate(numeric_cols):
        train_clean = train[col].dropna()
        test_clean = test[col].dropna()
        
        axes[i].hist(train_clean, alpha=0.7, label='Train', bins=50, density=True)
        axes[i].hist(test_clean, alpha=0.7, label='Test', bins=50, density=True)
        axes[i].set_title(f'{col}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

print("\n=== ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЙ НА TRAIN/TEST ===")
plot_distributions(train, test)


def analyze_user_overlap(train, test):
    print("\n=== АНАЛИЗ ПЕРЕСЕЧЕНИЯ ПОЛЬЗОВАТЕЛЕЙ TRAIN/TEST ===")
    
    train_users = set(train['user_number'].unique())
    test_users = set(test['user_number'].unique())
    
    common_users = train_users & test_users
    unique_train_users = train_users - test_users
    unique_test_users = test_users - train_users
    
    print(f"Уникальных пользователей в train: {len(train_users):,}")
    print(f"Уникальных пользователей в test: {len(test_users):,}")
    print(f"Общих пользователей: {len(common_users):,}")
    print(f"Уникальных в train: {len(unique_train_users):,}")
    print(f"Уникальных в test: {len(unique_test_users):,}")
    print(f"Покрытие test пользователей: {len(common_users)/len(test_users)*100:.1f}%")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    overlap_sizes = [len(common_users), len(unique_train_users), len(unique_test_users)]
    overlap_labels = ['Общие пользователи', 'Только в train', 'Только в test']
    
    ax1.pie(overlap_sizes, labels=overlap_labels, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Пересечение пользователей между train и test')

    train_user_activity = train.groupby('user_number').size()
    test_user_activity = test.groupby('user_number').size()
    
    activity_bins = [1, 2, 5, 10, 20, 50, 100, 1000]
    train_activity_dist = pd.cut(train_user_activity, bins=activity_bins).value_counts().sort_index()
    test_activity_dist = pd.cut(test_user_activity, bins=activity_bins).value_counts().sort_index()
    
    x = np.arange(len(train_activity_dist))
    width = 0.35
    
    ax2.bar(x - width/2, train_activity_dist.values, width, label='Train', alpha=0.7)
    ax2.bar(x + width/2, test_activity_dist.values, width, label='Test', alpha=0.7)
    ax2.set_xlabel('Количество сессий на пользователя')
    ax2.set_ylabel('Количество пользователей')
    ax2.set_title('Распределение пользователей по активности')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(bin_range) for bin_range in train_activity_dist.index])
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

    if len(common_users) > 0:
        common_users_train_activity = train[train['user_number'].isin(common_users)].groupby('user_number').size()
        common_users_test_activity = test[test['user_number'].isin(common_users)].groupby('user_number').size()
        
        print(f"\nАктивность общих пользователей:")
        print(f"  В train: mean={common_users_train_activity.mean():.1f}, max={common_users_train_activity.max()}")
        print(f"  В test: mean={common_users_test_activity.mean():.1f}, max={common_users_test_activity.max()}")

analyze_user_overlap(train, test)


def analyze_product_overlap(train, test):
    print("\n=== АНАЛИЗ ПЕРЕСЕЧЕНИЯ ТОВАРОВ TRAIN/TEST ===")
    
    train_products = set(train['product_id'].unique())
    test_products = set(test['product_id'].unique())
    
    common_products = train_products & test_products
    unique_train_products = train_products - test_products
    unique_test_products = test_products - train_products
    
    print(f"Уникальных товаров в train: {len(train_products):,}")
    print(f"Уникальных товаров в test: {len(test_products):,}")
    print(f"Общих товаров: {len(common_products):,}")
    print(f"Уникальных в train: {len(unique_train_products):,}")
    print(f"Уникальных в test: {len(unique_test_products):,}")
    print(f"Покрытие test товаров: {len(common_products)/len(test_products)*100:.1f}%")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    overlap_sizes = [len(common_products), len(unique_train_products), len(unique_test_products)]
    overlap_labels = ['Общие товары', 'Только в train', 'Только в test']
    
    ax1.pie(overlap_sizes, labels=overlap_labels, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Пересечение товаров между train и test')

    train_product_popularity = train.groupby('product_id').size()
    test_product_popularity = test.groupby('product_id').size()
    
    popularity_bins = [1, 2, 5, 10, 20, 50, 100, 1000, 10000]
    train_popularity_dist = pd.cut(train_product_popularity, bins=popularity_bins).value_counts().sort_index()
    test_popularity_dist = pd.cut(test_product_popularity, bins=popularity_bins).value_counts().sort_index()
    
    x = np.arange(len(train_popularity_dist))
    width = 0.35
    
    ax2.bar(x - width/2, train_popularity_dist.values, width, label='Train', alpha=0.7)
    ax2.bar(x + width/2, test_popularity_dist.values, width, label='Test', alpha=0.7)
    ax2.set_xlabel('Количество взаимодействий с товаром')
    ax2.set_ylabel('Количество товаров')
    ax2.set_title('Распределение товаров по популярности')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(bin_range) for bin_range in train_popularity_dist.index])
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

    if len(common_products) > 0:
        common_products_train_popularity = train[train['product_id'].isin(common_products)].groupby('product_id').size()
        common_products_test_popularity = test[test['product_id'].isin(common_products)].groupby('product_id').size()
        
        print(f"\nПопулярность общих товаров:")
        print(f"  В train: mean={common_products_train_popularity.mean():.1f}, max={common_products_train_popularity.max()}")
        print(f"  В test: mean={common_products_test_popularity.mean():.1f}, max={common_products_test_popularity.max()}")

# Вызов функции
analyze_product_overlap(train, test)


def analyze_product_positions_with_plot(train, test):
    print("\n=== АНАЛИЗ ПОЗИЦИЙ ТОВАРОВ И ГЛУБИНЫ ВЫДАЧИ ===")
    
    position_stats_train = train.groupby('search_number')['product_position'].max().describe()
    position_stats_test = test.groupby('search_number')['product_position'].max().describe()
    
    print("Максимальные позиции в сессиях:")
    print(f"Train - mean: {position_stats_train['mean']:.1f}, std: {position_stats_train['std']:.1f}, max: {position_stats_train['max']}")
    print(f"Test - mean: {position_stats_test['mean']:.1f}, std: {position_stats_test['std']:.1f}, max: {position_stats_test['max']}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    position_bins = np.arange(0, 251, 5)
    ax1.hist(train['product_position'], bins=position_bins, alpha=0.7, label='Train', density=True)
    ax1.hist(test['product_position'], bins=position_bins, alpha=0.7, label='Test', density=True)
    ax1.set_xlabel('Позиция товара')
    ax1.set_ylabel('Плотность распределения')
    ax1.set_title('Распределение позиций товаров')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    train_sorted = np.sort(train['product_position'])
    test_sorted = np.sort(test['product_position'])
    train_cdf = np.arange(1, len(train_sorted)+1) / len(train_sorted)
    test_cdf = np.arange(1, len(test_sorted)+1) / len(test_sorted)
    
    ax2.plot(train_sorted, train_cdf, label='Train', linewidth=2)
    ax2.plot(test_sorted, test_cdf, label='Test', linewidth=2)
    ax2.set_xlabel('Позиция товара')
    ax2.set_ylabel('CDF')
    ax2.set_title('CDF распределения позиций товаров')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 250)
    
    plt.tight_layout()
    plt.show()

    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    train_quantiles = train['product_position'].quantile(quantiles)
    test_quantiles = test['product_position'].quantile(quantiles)
    
    print("\nКвантили распределения позиций:")
    for q in quantiles:
        print(f"  {q:.0%}: train={train_quantiles[q]:.1f}, test={test_quantiles[q]:.1f}")

analyze_product_positions_with_plot(train, test)


sessions_with_engagement = train.groupby('search_number')['relevance'].max()
print(f"\nСессии с взаимодействиями: {(sessions_with_engagement >= 1).mean():.1%}")
print(f"Сессии с корзинами: {(sessions_with_engagement >= 2).mean():.1%}")
print(f"Сессии с покупками: {(sessions_with_engagement == 3).mean():.1%}")

plt.figure(figsize=(10, 6))
sns.countplot(data=train, x='relevance')
plt.title('Распределение релевантности', fontsize=16)
plt.xlabel('Релевантность')
plt.ylabel('Количество')
plt.show()


pos_relevance = train.groupby('product_position')['relevance'].mean()
plt.figure(figsize=(12, 6))
pos_relevance.plot()
plt.title('Средняя релевантность по позициям товаров')
plt.xlabel('Позиция товара')
plt.ylabel('Средняя relevance')
plt.show()


relevant_items = train[train['relevance'] > 0]
print(f"\nПозиции релевантных товаров:")
print(relevant_items['product_position'].describe().round(1).to_frame().T)
print('\n')

first_ten_positions = len(relevant_items[relevant_items['product_position'] <= 10])
ten_to_twelve_positions = len(relevant_items[relevant_items['product_position'] <= 20]) - first_ten_positions
later_positions = len(relevant_items[relevant_items['product_position'] > 20])


print(f"Релевантных товаров в первых 10 позициях: {first_ten_positions} ({first_ten_positions/len(relevant_items):.1%})")
print(f"Релевантных товаров с 10 до 20 позиции: {ten_to_twelve_positions} ({ten_to_twelve_positions/len(relevant_items):.1%})")
print(f"Релевантных товаров после 20 позиции: {later_positions} ({later_positions/len(relevant_items):.1%})")


train['day_of_week'] = train['date'].dt.dayofweek
train['day_of_month'] = train['date'].dt.day

def analyze_user_weekday_patterns(df, top_users=10000):
    top_users_list = df['user_number'].value_counts().head(top_users).index
    results = []
    
    for user in top_users_list:
        user_data = df[df['user_number'] == user]
        weekday_relevance = user_data.groupby(['day_of_week', 'relevance']).size().unstack(fill_value=0)
        
        for rel in [0,1,2,3]:
            if rel not in weekday_relevance.columns:
                weekday_relevance[rel] = 0
        
        for day in range(7):
            if day in weekday_relevance.index:
                day_row = weekday_relevance.loc[day]
                total = day_row.sum()
                if total > 0:
                    results.append({
                        'user_number': user,
                        'day_of_week': day,
                        'purchase_rate': day_row.get(3, 0) / total,
                        'click_rate': day_row.get(1, 0) / total,
                        'cart_rate': day_row.get(2, 0) / total,
                        'view_rate': day_row.get(0, 0) / total,
                        'total_actions': total,
                        'day_name': ['Пн','Вт','Ср','Чт','Пт','Сб','Вс'][day]
                    })
    
    return pd.DataFrame(results)

def analyze_user_monthday_patterns(df, top_users=10000):
    top_users_list = df['user_number'].value_counts().head(top_users).index
    results = []
    
    for user in top_users_list:
        user_data = df[df['user_number'] == user]
        monthday_relevance = user_data.groupby(['day_of_month', 'relevance']).size().unstack(fill_value=0)
        
        for rel in [0,1,2,3]:
            if rel not in monthday_relevance.columns:
                monthday_relevance[rel] = 0
        
        for day in range(1, 32):
            if day in monthday_relevance.index:
                day_row = monthday_relevance.loc[day]
                total = day_row.sum()
                if total > 0:
                    results.append({
                        'user_number': user,
                        'day_of_month': day,
                        'purchase_rate': day_row.get(3, 0) / total,
                        'click_rate': day_row.get(1, 0) / total,
                        'cart_rate': day_row.get(2, 0) / total,
                        'view_rate': day_row.get(0, 0) / total,
                        'total_actions': total
                    })
    
    return pd.DataFrame(results)

weekday_analysis = analyze_user_weekday_patterns(train)
monthday_analysis = analyze_user_monthday_patterns(train)

weekday_order = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
metrics = ['purchase_rate', 'cart_rate', 'click_rate']
titles = ['Доля покупок', 'Доля корзин', 'Доля кликов']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for i, metric in enumerate(metrics):
    data = weekday_analysis.groupby('day_name')[metric].mean().reindex(weekday_order)
    axes[0,i].bar(data.index, data.values, alpha=0.7)
    axes[0,i].set_title(f'{titles[i]} по дням недели')
    axes[0,i].set_ylabel('Доля')
    axes[0,i].grid(axis='y', alpha=0.3)

for i, metric in enumerate(metrics):
    data = monthday_analysis.groupby('day_of_month')[metric].mean()
    axes[1,i].bar(data.index, data.values, alpha=0.7)
    axes[1,i].set_title(f'{titles[i]} по дням месяца')
    axes[1,i].set_xlabel('День месяца')
    axes[1,i].set_ylabel('Доля')
    axes[1,i].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

train = train.drop(columns=['day_of_week', 'day_of_month'])


user_actions = train['user_number'].value_counts()

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(user_actions, bins=50, alpha=0.7, edgecolor='black')
plt.title('Распределение количества действий по пользователям')
plt.xlabel('Количество действий')
plt.ylabel('Количество пользователей')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(user_actions, vert=False)
plt.title('Боксплот распределения действий')
plt.xlabel('Количество действий')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Всего пользователей: {len(user_actions)}")
print(f"Медиана действий на пользователя: {user_actions.median():.1f}")
print(f"Среднее действий на пользователя: {user_actions.mean():.1f}")
print(f"Максимум действий: {user_actions.max()}")
print(f"Минимум действий: {user_actions.min()}")
print(f"Пользователей с >100 действиями: {len(user_actions[user_actions > 100])}")
print(f"Пользователей с >30 действиями: {len(user_actions[user_actions > 30])}")


def analyze_queries_overlap(train_sessions, test_sessions):
    """Анализ пересечения уникальных запросов между train и test"""

    train_queries = set(train_sessions['query'].dropna().unique())
    test_queries = set(test_sessions['query'].dropna().unique())

    print("=== АНАЛИЗ ПЕРЕСЕЧЕНИЯ ЗАПРОСОВ ===")
    print(f"Уникальных запросов в train: {len(train_queries):,}")
    print(f"Уникальных запросов в test: {len(test_queries):,}")
    print(f"Всего уникальных запросов всего: {len(train_queries | test_queries):,}")

    common_queries = train_queries & test_queries
    unique_test_queries = test_queries - train_queries
    unique_train_queries = train_queries - test_queries
    
    print(f"\n--- Пересечение запросов ---")
    print(f"Общих запросов (train ∩ test): {len(common_queries):,}")
    print(f"Уникальных запросов в test (не в train): {len(unique_test_queries):,}")

    plt.figure(figsize=(5, 4))
    overlap_sizes = [len(common_queries), len(train_queries - test_queries), len(unique_test_queries)]
    overlap_labels = ['Общие запросы', 'Только в train', 'Только в test']
    
    plt.pie(overlap_sizes, autopct=lambda p: f'{p:.1f}%' if p > 5 else '', 
            startangle=90, textprops={'fontsize': 12})
    plt.title('Пересечение запросов между train и test', fontsize=12)
    plt.legend(overlap_labels, title="Категории", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.tight_layout()
    plt.show()

    if len(test_queries) > 0:
        coverage_percent = len(common_queries) / len(test_queries) * 100
        novelty_percent = len(unique_test_queries) / len(test_queries) * 100
        print(f"\n--- Процент покрытия ---")
        print(f"Запросов из test, которые есть в train: {coverage_percent:.2f}%")
        print(f"Новых запросов в test (неизвестных из train): {novelty_percent:.2f}%")
    
    if len(unique_test_queries) > 0:
        print(f"Примеры новых запросов в test (первые 10): {list(unique_test_queries)[:10]}")
    
    train_query_lengths = [len(str(q)) for q in train_queries]
    test_query_lengths = [len(str(q)) for q in test_queries]
    
    print(f"\n--- Статистика длины запросов ---")
    print(f"Train - Средняя длина: {np.mean(train_query_lengths):.1f}, "
          f"Мин: {np.min(train_query_lengths)}, Макс: {np.max(train_query_lengths)}")
    print(f"Test - Средняя длина: {np.mean(test_query_lengths):.1f}, "
          f"Мин: {np.min(test_query_lengths)}, Макс: {np.max(test_query_lengths)}")
    
    return {
        'train_queries': train_queries,
        'test_queries': test_queries,
        'common_queries': common_queries,
        'unique_test_queries': unique_test_queries,
        'coverage_percent': coverage_percent if len(test_queries) > 0 else 0,
        'novelty_percent': novelty_percent if len(test_queries) > 0 else 0
    }

results = analyze_queries_overlap(train, test)


def analyze_query_quality(train):
    print("\n=== АНАЛИЗ КАЧЕСТВА ЗАПРОСОВ И ОШИБОК ===")

    all_words = []
    for query in train['query'].dropna():
        words = query.lower().split()
        all_words.extend(words)
    
    word_freq = Counter(all_words)

    low_freq_words = [word for word, count in word_freq.items() if count <= 3 and len(word) > 4]
    
    print(f"Потенциальных опечаток (слово встречается <= 3 раз): {len(low_freq_words):,}")
    print(f"Примеры потенциальных опечаток: {low_freq_words[:20]}")

    query_stats = train.groupby('query').agg({
        'relevance': ['count', lambda x: (x >= 2).mean()]
    }).round(4)
    query_stats.columns = ['count', 'conversion_rate']
    
    low_conversion_queries = query_stats[(query_stats['count'] >= 10) & (query_stats['conversion_rate'] == 0)]
    high_conversion_queries = query_stats[(query_stats['count'] >= 10) & (query_stats['conversion_rate'] >= 0.1)]
    
    print(f"\nЗапросы с нулевой конверсией (>=10 показов): {len(low_conversion_queries):,}")
    print(f"Запросы с высокой конверсией (>=10%): {len(high_conversion_queries):,}")
    
    short_queries = train[train['query'].str.len() <= 2]
    long_queries = train[train['query'].str.len() >= 50]
    
    print(f"\nЭкстремальные запросы:")
    print(f"  Очень короткие (<=2 символа): {len(short_queries):,}")
    print(f"  Очень длинные (>=50 символов): {len(long_queries):,}")
    
    if len(short_queries) > 0:
        print(f"  Примеры коротких: {short_queries['query'].unique()[:10]}")
    if len(long_queries) > 0:
        print(f"  Примеры длинных: {[q[:50] + '...' for q in long_queries['query'].unique()[:5]]}")

analyze_query_quality(train)


def analyze_query_semantics(train):
    all_queries = train['query'].dropna().str.lower()
    stop_words = ['купить', 'цена', 'отзывы', 'характеристики', 'фото']
    
    print(f"Запросы со стоп-словами:")
    for stop_word in stop_words:
        matches = all_queries[all_queries.str.contains(stop_word, regex=False)]
        print(f"  '{stop_word}': {len(matches):,}")

analyze_query_semantics(train)


def get_all_special_chars(queries):
    pattern = r'[^\w\s]'
    
    all_chars = []
    for query in queries:
        special_chars = re.findall(pattern, str(query))
        all_chars.extend(special_chars)
    
    return Counter(all_chars)

char_counter = get_all_special_chars(train['query'])

print("=== СПЕЦСИМВОЛЫ В ЗАПРОСАХ === ")
for char, count in char_counter.most_common():
    print(f"'{char}': {count} раз")

print(f"\nВСЕГО УНИКАЛЬНЫХ СПЕЦСИМВОЛОВ: {len(char_counter)}")


def analyze_products_and_replacements(products):
    print("\n=== АНАЛИЗ ТОВАРНЫХ ХАРАКТЕРИСТИК ===")
    
    print(f"Уникальные бренды: {products['brand'].nunique()}")
    print(f"Уникальные типы товаров: {products['type'].nunique()}")
    print(f"Уникальные модели: {products['model'].nunique()}")
    
    replacements = products[products['replaced_by'] > 0]
    print(f"\nТоваров с заменой: {len(replacements)} ({len(replacements)/len(products)*100:.1f}%)")

analyze_products_and_replacements(products)


def analyze_products_per_search(df):
    products_per_search = df.groupby('search_number').size()
    total_searches = len(products_per_search)
    
    ranges = [
        (1, 5),
        (5, 10),
        (10, 15),
        (15, 19),
        (20, 20),
        (21, 40),
        (40, 100),
        (100, float('inf'))
    ]
    
    labels = [
        '1-5 товаров в выдаче',
        '5-10 товаров в выдаче', 
        '10-15 товаров в выдаче',
        '15-19 товаров',
        'ровно 20 товаров в выдаче',
        '21-40 товаров',
        '40+ товаров',
        '100+ товаров'
    ]
    
    distribution_stats = []
    
    for i, ((start, end), label) in enumerate(zip(ranges, labels)):
        if start == end:
            count = (products_per_search == start).sum()
        else:
            count = ((products_per_search >= start) & (products_per_search < end)).sum()
        
        percentage = (count / total_searches) * 100
        distribution_stats.append({
            'range_label': label,
            'count': count,
            'percentage': percentage
        })
    
    return products_per_search, distribution_stats, total_searches

products_per_search, distribution_stats, total_searches = analyze_products_per_search(train)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(products_per_search, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Распределение товаров по поисковым выдачам')
plt.xlabel('Количество товаров в search_number')
plt.ylabel('Количество выдачей')
plt.grid(alpha=0.3)

plt.axvline(products_per_search.mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Среднее: {products_per_search.mean():.1f}')
plt.axvline(products_per_search.median(), color='green', linestyle='--', linewidth=2, 
           label=f'Медиана: {products_per_search.median()}')
plt.legend()

plt.subplot(1, 2, 2)
plt.boxplot(products_per_search, vert=False)
plt.title('Боксплот распределения товаров по выдачам')
plt.xlabel('Количество товаров')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== СТАТИСТИКА РАСПРЕДЕЛЕНИЯ ТОВАРОВ ПО ВЫДАЧАМ ===")

for stat in distribution_stats:
    print(f"{stat['range_label']}: {stat['percentage']:.2f}% ({stat['count']} шт.)")

print("=" * 50)
print(f"Всего уникальных сессий: {total_searches}")
print(f"Медиана товаров: {products_per_search.median():.1f}")
print(f"Среднее товаров: {products_per_search.mean():.1f}")
print(f"Максимум товаров: {products_per_search.max()}")
print(f"Минимум товаров: {products_per_search.min()}")
print(f"Выдачей с >20 товарами: {len(products_per_search[products_per_search > 20])}")
print(f"Выдачей с >40 товарами: {len(products_per_search[products_per_search > 40])}")


def analyze_replacement_graph(products):
    print("\n=== АНАЛИЗ ТОВАРНОГО ГРАФА ЗАМЕН ===")
    
    replacements = products[products['replaced_by'].notna()].copy()
    
    if len(replacements) > 0:
        replacement_chains = {}
        for product_id in replacements['product_id']:
            chain_length = 0
            current_id = product_id
            
            while current_id in replacements['product_id'].values:
                current_id = replacements[replacements['product_id'] == current_id]['replaced_by'].iloc[0]
                chain_length += 1
                if chain_length > 10:
                    break
            
            replacement_chains[product_id] = chain_length
        
        chain_lengths = list(replacement_chains.values())
        print(f"Максимальная длина цепочки замен: {max(chain_lengths) if chain_lengths else 0}")
        print(f"Средняя длина цепочки: {np.mean(chain_lengths):.2f}")
        
        products_with_dates = products[products['date_of_create'].notna()].copy()
        products_with_dates['date_of_create'] = pd.to_datetime(products_with_dates['date_of_create'])
        
        replacement_pairs = products_with_dates.merge(
            products_with_dates[['product_id', 'date_of_create']],
            left_on='replaced_by',
            right_on='product_id',
            suffixes=('_old', '_new')
        )
        
        if len(replacement_pairs) > 0:
            replacement_pairs['days_between'] = (replacement_pairs['date_of_create_new'] - replacement_pairs['date_of_create_old']).dt.days
            print(f"Среднее время между заменами: {replacement_pairs['days_between'].mean():.1f} дней")

analyze_replacement_graph(products)


df_merged = products.merge(
    products,
    left_on='replaced_by',
    right_on='product_id',
    suffixes=('_old', '_new')
)

comparison_columns = ['type', 'model', 'brand', 'country'] + \
                    [f'category_name_{i}' for i in range(1, 5)]

detailed_mismatches = []

for col in comparison_columns:
    col_old = f'{col}_old'
    col_new = f'{col}_new'
    
    mismatches = df_merged[
        df_merged[col_old].astype(str).str.lower() != df_merged[col_new].astype(str).str.lower()
    ]
    
    for _, row in mismatches.iterrows():
        detailed_mismatches.append({
            'column': col,
            'product_id_old': row['product_id_old'],
            'product_id_new': row['product_id_new'],
            'old_value': row[col_old],
            'new_value': row[col_new]
        })

detailed_mismatch_df = pd.DataFrame(detailed_mismatches)

print("=== ИЗМЕНЕНИЯ ДЛЯ КАЖДОГО ВИДА ЗАМЕН ===")

if not detailed_mismatch_df.empty:
    for column in detailed_mismatch_df['column'].unique():
        column_data = detailed_mismatch_df[detailed_mismatch_df['column'] == column]
        
        print(f"\n┌── {column.upper()} ── ({len(column_data)} изменений)")
        print(f"├{'─' * 78}")
        
        for i, (_, row) in enumerate(column_data.head(10).iterrows()):
            print(f"│ {row['product_id_old']} → {row['product_id_new']}: '{row['old_value']}' → '{row['new_value']}'")
        
        if len(column_data) > 10:
            print(f"│ ... и еще {len(column_data) - 10} изменений")
        
        print(f"└{'─' * 78}")

    print(f"\nОБЩАЯ СТАТИСТИКА:")
    print("-" * 50)
    summary = detailed_mismatch_df.groupby('column').size()
    for col, count in summary.items():
        print(f"  {col}: {count} изменений")
    
    print(f"\nВсего найдено: {len(detailed_mismatch_df)} изменений")


def check_sequence_percent(df):
    total_groups = df['search_number'].nunique()
    
    problem_groups = 0
    for _, group in df.groupby('search_number'):
        positions = sorted(group['product_position'])
        if positions != list(range(positions[0], positions[0] + len(positions))):
            problem_groups += 1
    
    percent = (problem_groups / total_groups) * 100
    print(f'Групп со скачками позиций товаров в выдаче (на тесте): {percent:.1f}%')

result_df = check_sequence_percent(test)


train_device = train['device'].value_counts(normalize=True)
test_device = test['device'].value_counts(normalize=True)

fig, ax = plt.subplots(figsize=(10, 6))

devices = train_device.index
x = range(len(devices))
width = 0.35

ax.bar([i - width/2 for i in x], train_device.values, width, label='Train', alpha=0.8)
ax.bar([i + width/2 for i in x], [test_device.get(dev, 0) for dev in devices], width, label='Test', alpha=0.8)

ax.set_xlabel('Устройства')
ax.set_ylabel('Доля (%)')
ax.set_title('Распределение по устройствам: Train vs Test')
ax.set_xticks(x)
ax.set_xticklabels(devices, rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


def analyze_feature_interactions(train):
    print("\n=== АНАЛИЗ ВЛИЯНИЯ ПРИЗНАКОВ ===")

    filters_effect = train.groupby('filters_applied')['relevance'].apply(lambda x: (x >= 2).mean())
    print("Влияние фильтров на конверсию:")
    print(f"  Без фильтров: {filters_effect[0]:.3f}")
    print(f"  С фильтрами: {filters_effect[1]:.3f}")

    device_effect = train.groupby('device')['relevance'].apply(lambda x: (x >= 1).mean())
    print("\nCTR по устройствам:")
    for device, ctr in device_effect.items():
        print(f"  {device}: {ctr:.3f}")

    print("\nВзаимодействия в зависимости от цены:")
    train['price_byn'] = train['price'] / 100
    price_bins = pd.qcut(train['price_byn'], q=20, duplicates='drop')
    price_clicks = train.groupby(price_bins)['relevance'].apply(lambda x: (x >= 1).mean())
    price_carts = train.groupby(price_bins)['relevance'].apply(lambda x: (x >= 2).mean())
    price_conversion = train.groupby(price_bins)['relevance'].apply(lambda x: (x == 3).mean())

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    price_clicks.plot(kind='bar', ax=axes[0], title='Клики (relevance ≥ 1)')
    price_carts.plot(kind='bar', ax=axes[1], title='Добавления в корзину (relevance ≥ 2)')
    price_conversion.plot(kind='bar', ax=axes[2], title='Конверсия (relevance = 3)')
    
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    train = train.drop(columns=['price_byn'])

analyze_feature_interactions(train)


def analyze_query_frequency_and_conversion(train):
    print("\n=== АНАЛИЗ ЧАСТОТНОСТИ ЗАПРОСОВ И КОНВЕРСИИ ===")
    
    query_freq = train.groupby('query')['search_number'].nunique()
    
    print("Статистики частотности запросов:")
    print(f"  Самый частый запрос: '{query_freq.idxmax()}' - {query_freq.max()} раз")
    print(f"  Медианная частота: {query_freq.median()}")
    print(f"  Запросы с частотой 1: {sum(query_freq == 1):,} ({sum(query_freq == 1)/len(query_freq)*100:.1f}%)")

    freq_bins = [1, 2, 5, 10, 50, 100, 1000, float('inf')]
    freq_labels = ['1', '2-5', '6-10', '11-50', '51-100', '101-1000', '1000+']
    
    train['query_freq'] = train['query'].map(query_freq)
    train['freq_group'] = pd.cut(train['query_freq'], bins=freq_bins, labels=freq_labels)
    
    freq_clicks = train.groupby('freq_group')['relevance'].apply(lambda x: (x >= 1).mean())
    freq_carts = train.groupby('freq_group')['relevance'].apply(lambda x: (x >= 2).mean())
    freq_conversion = train.groupby('freq_group')['relevance'].apply(lambda x: (x == 3).mean())
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    freq_clicks.plot(kind='bar', ax=axes[0], title='Клики по частотности запросов (relevance ≥ 1)')
    freq_carts.plot(kind='bar', ax=axes[1], title='Добавления в корзину по частотности запросов (relevance ≥ 2)')
    freq_conversion.plot(kind='bar', ax=axes[2], title='Конверсия по частотности запросов (relevance = 3)')

    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylabel('Доля')
    
    plt.tight_layout()
    plt.show()

    train = train.drop(columns=['query_freq', 'freq_group'])
    
    
analyze_query_frequency_and_conversion(train)


corr_matrix = train[numeric_cols + ['relevance']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Корреляционная матрица')
plt.show()


def mutual_info_matrix(df, target_col):
    X = df.drop(columns=[target_col, 'date'])
    y = df[target_col]
    X_encoded = X.copy()

    object_cols = X.select_dtypes(include=['object']).columns
    for col in object_cols:
        X_encoded[col] = X[col].astype('category')

    categorical_features = X_encoded.select_dtypes(include=['category']).columns
    numeric_features = X_encoded.select_dtypes(include=[np.number]).columns

    for col in categorical_features:
        X_encoded[col] = X_encoded[col].cat.codes

    discrete_mask = [col in categorical_features for col in X_encoded.columns]

    mi_scores = mutual_info_classif(
        X_encoded, 
        y, 
        discrete_features=discrete_mask,
        random_state=config.SEED
    )
    
    mi_df = pd.DataFrame({'feature': X_encoded.columns, 'mutual_info': mi_scores})
    mi_df = mi_df.sort_values('mutual_info', ascending=False)
    
    return mi_df

print("\n=== MUTUAL INFO С ЦЕЛЕВОЙ ПЕРЕМЕННОЙ ===")
#mi_results = mutual_info_matrix(train, 'relevance')
#mi_results


class ProductFeaturesProcessor:
    """Добавление характеристик товаров к сессиям"""
    
    def __init__(self):
        self.mappings = None
        self.products = None
    
    def prepare_products(self, products):
        """Подготовка данных о товарах"""
        self.products = products.drop_duplicates(subset=['product_id'], keep='first')
        self.products['date_of_create'] = pd.to_datetime(self.products['date_of_create'])
        self.mappings = self._build_product_mappings()
        return self.products
    
    def _build_product_mappings(self):
        """Построение маппингов товаров - находит самый поздний товар в цепочке для каждого product_id"""
        product_dict = self.products.set_index('product_id').to_dict('index')
        
        replacement_mapping = {}
        
        for product_id in self.products['product_id']:
            current_id = product_id
            visited = set()

            while (current_id in product_dict and 
                   pd.notna(product_dict[current_id].get('replaced_by')) and
                   product_dict[current_id]['replaced_by'] in product_dict and
                   product_dict[current_id]['replaced_by'] not in visited and
                   product_dict[current_id]['replaced_by'] > 0):
                
                visited.add(current_id)
                current_id = product_dict[current_id]['replaced_by']

            replacement_mapping[product_id] = current_id
        
        return {
            'product_dict': product_dict,
            'replacement_mapping': replacement_mapping
        }
    
        
    def _add_product_features(self, sessions):
        sessions = sessions.copy()
        product_dict = self.mappings['product_dict']
        replacement_mapping = self.mappings['replacement_mapping']
        
        earliest_date_mapping = {}
        groups = {}
        
        for product_id, root_id in replacement_mapping.items():
            if root_id not in groups:
                groups[root_id] = []
            groups[root_id].append(product_id)
        
        for root_id, product_ids in groups.items():
            earliest_date = product_dict[root_id]['date_of_create']
            for pid in product_ids:
                current_date = product_dict[pid]['date_of_create']
                if current_date < earliest_date:
                    earliest_date = current_date
            for pid in product_ids:
                earliest_date_mapping[pid] = earliest_date

        sessions['actual_product_id'] = sessions['product_id'].map(replacement_mapping).fillna(sessions['product_id'])
        
        products_df = pd.DataFrame.from_dict(product_dict, orient='index').reset_index()
        products_df = products_df.rename(columns={'index': 'product_id'})
        
        replacements_info = products_df[['product_id', 'replaced_by', 'date_of_create']].copy()
        replacements_info = replacements_info[replacements_info['replaced_by'].notna() & (replacements_info['replaced_by'] > 0)]
        
        sessions_with_replacements = sessions.merge(replacements_info, on='product_id', how='left')
        
        replacement_products = products_df[['product_id', 'date_of_create']].rename(
            columns={'product_id': 'replacement_id', 'date_of_create': 'date_of_create_replacement'}
        )
        
        sessions_with_replacements = sessions_with_replacements.merge(
            replacement_products, left_on='replaced_by', right_on='replacement_id', how='left'
        )
        
        condition_replacement = (
            sessions_with_replacements['replaced_by'].notna() & 
            sessions_with_replacements['date_of_create_replacement'].notna() &
            (sessions_with_replacements['date_of_create_replacement'] <= sessions_with_replacements['date'])
        )
        
        sessions_with_replacements['product_id_for_features'] = np.where(
            condition_replacement,
            sessions_with_replacements['replaced_by'],
            sessions_with_replacements['product_id']
        )
        
        feature_columns = ['type', 'model', 'brand', 'country', 'category_name_1', 'category_name_2', 'category_name_3', 'category_name_4']
        
        all_product_features = []
        for product_id, product_data in product_dict.items():
            features = {'product_id': product_id}
            for col in feature_columns:
                features[col] = product_data.get(col, 'unknown')
            all_product_features.append(features)
        
        features_df_all = pd.DataFrame(all_product_features)
        
        sessions_with_features = sessions_with_replacements.merge(
            features_df_all, 
            left_on='product_id_for_features', 
            right_on='product_id', 
            how='left',
            suffixes=('', '_features')
        )
        
        sessions_with_features['earliest_date_of_create'] = sessions_with_features['actual_product_id'].map(earliest_date_mapping)
        
        missing_mask = sessions_with_features['type'].isna()
        
        if missing_mask.any():
            sessions_missing = sessions_with_features[missing_mask].copy()
            sessions_missing = sessions_missing.drop(feature_columns, axis=1, errors='ignore')
            sessions_missing = sessions_missing.merge(
                features_df_all, 
                left_on='actual_product_id',
                right_on='product_id', 
                how='left', 
                suffixes=('', '_backup')
            )
            sessions_with_features = pd.concat([sessions_with_features[~missing_mask], sessions_missing], ignore_index=True)
        
        for col in feature_columns:
            if col in sessions_with_features.columns:
                sessions_with_features[col] = sessions_with_features[col].fillna('unknown')
        
        columns_to_drop = [
            'product_id', 'product_id_features', 'replaced_by', 
            'date_of_create_replacement', 'replacement_id', 'product_id_for_features',
            'date_of_create'
        ]
        columns_to_drop = [col for col in columns_to_drop if col in sessions_with_features.columns]
        sessions_with_features = sessions_with_features.drop(columns=columns_to_drop)
        
        sessions_with_features = sessions_with_features.rename(
            columns={'earliest_date_of_create': 'date_of_create'}
        )
                
        return sessions_with_features
    
    def process(self, dfs):
        print("Добавление характеристик товаров...")

        for i, df in enumerate(dfs):
            dfs[i] = self._add_product_features(df)
        print("Обработка завершена!")
        return dfs


class DataSampler:
    """Сэмплирование и разделение данных"""
    
    def __init__(self):
        self.min_group_size = config.SAMPLING_CONFIG['min_group_size']
        self.validation_group_size = config.SAMPLING_CONFIG['validation_group_size']
        self.validation_months = config.SAMPLING_CONFIG['validation_months']
    
    def date_to_datetime(self, dfs):
        for i, df in enumerate(dfs):
            dfs[i]['date'] = pd.to_datetime(df['date'])
        return dfs

    def group_sampling(self, df, group_size):
        """Сэмплирование групп поисковых запросов"""
        df_sorted = df.sort_values(['search_number', 'relevance'], ascending=[True, False])
        result = df_sorted.groupby('search_number').head(group_size)
        
        print("Обработка больших групп.")
        print(f"Итоговый размер: {len(result):,} записей")
        
        final_sizes = result.groupby('search_number').size()
        deep_pos_count = len(result[result['product_position'] >= group_size])
        
        print(f"Групп: {len(final_sizes):,}")
        print(f"Позиции ≥{group_size}: {deep_pos_count:,} ({deep_pos_count/len(result)*100:.1f}%)")
        print(f"Максимальный размер группы: {final_sizes.max()}")
        
        return result.reset_index(drop=True)
    
    def filter_short_groups(self, df, size=None):
        """Фильтрация коротких выдач без создания валидационной выборки"""
        size = self.min_group_size if size is None else size
        df_filtered = df.groupby('search_number').filter(lambda x: len(x) > size)
        print(f"После фильтрации коротких выдач: {len(df_filtered):,} записей")
        return df_filtered
    
    def create_validation_split(self, train_df, months):
        """Создание валидационной выборки из тренировочной"""
        latest_date = train_df['date'].max()
        validation_df = train_df[train_df['date'] >= latest_date - pd.DateOffset(months=months)].copy()
        validation_df = data_sampler.group_sampling(validation_df, self.validation_group_size)
        validation_df = data_sampler.filter_short_groups(validation_df, self.validation_group_size)    
        train_df = train_df[train_df['date'] < latest_date - pd.DateOffset(months=months)]
        
        overall_length = len(train_df) + len(validation_df)
        print(f"Размер обучающей выборки: {len(train_df):,} записей, {len(train_df) / overall_length * 100:.1f}%")
        print(f"Размер валидационной выборки: {len(validation_df):,} записей, {len(validation_df) / overall_length * 100:.1f}%")
        
        return train_df, validation_df



class TextPreprocessor:
    """Предобработка текстовых данных"""
    
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()
        self._normalize_cache = {}
        self._transliterate_cache = {}

    
    @lru_cache(maxsize=10000)
    def _lemmatize_word(self, word):
        """Лемматизация одного слова с кэшированием"""
        if WORD_PATTERNS['word'].match(word):
            return word
        parsed = self.morph.parse(word)[0]
        return parsed.normal_form
    
    def lemmatize_text(self, text):
        """Лемматизация текста"""
        if not isinstance(text, str) or not text.strip():
            return text
        
        words = text.split()
        lemmas = []
        
        for word in words:
            if len(word) > config.TEXT_PROCESSING['lemmatize_min_length'] and WORD_PATTERNS['word'].match(word):
                lemmatized_word = self._lemmatize_word(word)
                lemmatized_word = lemmatized_word.replace('ё', 'е')
                lemmas.append(lemmatized_word)
            else:
                lemmas.append(word)
        
        return ' '.join(lemmas)
    
    @lru_cache(maxsize=10000)
    def normalize_text(self, text):
        """Нормализация текста"""
        if not isinstance(text, str):
            return ""    
            
        if not text.strip():
            return ""    
            
        text = str(text).lower().strip()    
        
        text = text.replace('ё', 'е').replace('і', 'и')
        text = WORD_PATTERNS['comma_numbers'].sub(r'\1.\2', text)
        text = WORD_PATTERNS['percent'].sub(r'\1 процент', text)
        text = WORD_PATTERNS['size'].sub(r'\1 x \2', text)
        text = WORD_PATTERNS['plus'].sub(r'\1 plus', text)
        text = WORD_PATTERNS['letters_digits'].sub(r'\1 \2', text)
        text = WORD_PATTERNS['digits_letters'].sub(r'\1 \2', text)
        text = WORD_PATTERNS['non_word'].sub(' ', text)
        text = WORD_PATTERNS['multi_space'].sub(' ', text)   
        
        return text.strip()
    
    @lru_cache(maxsize=5000)
    def _transliterate_word(self, word):
        """Транслитерация слова в кириллицу"""
        if not word:
            return word
                
        if word in SPECIAL_CASES:
            return SPECIAL_CASES[word]

        result = []
        i = 0
        while i < len(word):
            applied = False
            for pattern, replacement in TRANSLITERATION_RULES:
                pattern_len = len(pattern)
                if i + pattern_len <= len(word) and word[i:i+pattern_len] == pattern:
                    result.append(replacement)
                    i += pattern_len
                    applied = True
                    break
            
            if not applied:
                result.append(word[i])
                i += 1
        
        return ''.join(result)
    
    def process_column(self, df, column_name, lemmatize=False):
        """Предобработка текстовой колонки"""
        print(f"Предобработка колонки {column_name}...")
        
        unique_values = df[column_name].dropna().unique()
        mapping_dict = {}
        
        for value in unique_values:
            if not isinstance(value, str) or not value.strip():
                mapping_dict[value] = value
                continue

            processed_words = []
            for word in value.split():
                if word.isdigit():
                    processed_words.append(word)
                    continue

                has_cyrillic = any('а' <= char <= 'я' or char == 'ё' for char in word.lower())
                has_latin = any('a' <= char <= 'z' for char in word.lower())

                if lemmatize and has_cyrillic and len(word) > config.TEXT_PROCESSING['lemmatize_min_length']:
                    processed_word = self._lemmatize_word(word)
                else:
                    processed_word = word

                if has_latin:
                    processed_word = self._transliterate_word(processed_word)
                    
                processed_words.append(processed_word)
            
            mapping_dict[value] = ' '.join(processed_words)

        df[column_name] = df[column_name].map(mapping_dict)
        return df
    
    def normalize_products(self, df):
        """Нормализация таблицы товаров"""
        print("Нормализация таблицы товаров...")
        
        text_columns = ['type', 'model', 'brand', 'country']
        text_columns += [f'category_name_{i}' for i in range(1, 5)]
        
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(self.normalize_text)

        brands = set(df['brand'].dropna().unique()) if 'brand' in df.columns else set()
        models = set(df['model'].dropna().unique()) if 'model' in df.columns else set()
        types = set(df['type'].dropna().unique()) if 'type' in df.columns else set()
        countries = set(df['country'].dropna().unique()) if 'country' in df.columns else set()

        brands_with_latin = [brand for brand in brands if WORD_PATTERNS['latin'].search(str(brand))]
        models_with_latin = [word for model in models for word in str(model).split(' ') if WORD_PATTERNS['latin'].search(word)]
        
        brands_transl_dict = {self._transliterate_word(brand): brand for brand in brands_with_latin}
        models_transl_dict = {self._transliterate_word(model): model for model in models_with_latin}

        categories = set()
        for i in range(1, 5):
            col = f'category_name_{i}'
            if col in df.columns:
                categories.update(df[col].dropna().unique())
        
        print(f"Созданы справочники: brands({len(brands):,}), models({len(models):,}), categories({len(categories):,})")
        return df, brands, models, categories, types, countries, brands_transl_dict, models_transl_dict


class SpellChecker:
    """Коррекция орфографии"""
    
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.spell_checkers = {}
        self.dicts = {}
        
    def train(self, terms_list, to_split=True):
        """Обучение спеллчекера"""
        sym_spell = SymSpell(config.SPELL_CHECKER_CONFIG['max_edit_distance'],
                             config.SPELL_CHECKER_CONFIG['prefix_length'])
        
        all_terms = set()
        for term_list in terms_list:
            for entry in term_list:
                if to_split:
                    all_terms.update(entry.split(' '))
                else:
                    all_terms.add(entry)
        
        for term in all_terms:
            sym_spell.create_dictionary_entry(term, 1)
        
        print(f"Спеллчекер обучен на {len(all_terms):,} терминах")
        return sym_spell
    
    def setup_spell_checkers(self, brands, models, categories, types, countries, brands_transl_dict, models_transl_dict):
        """Настройка трех спеллчекеров"""
        self.spell_checkers = {
            'brand': {
                'original': self.train([brands]),
                'translated': self.train([brands_transl_dict.keys()])
            },
            'model': {
                'original': self.train([brands, models]),
                'translated': self.train([models_transl_dict.keys()])
            },
            'general': {
                'original': self.train([brands, models, categories, types, countries]),
            }
        }
        
        self.dicts = {
            'brand': brands_transl_dict,
            'model': models_transl_dict
        }
        
        print("Все спеллчекеры обучены!")
    
    def _change_keyboard_layout(self, word):
        """Смена раскладки клавиатуры"""
        cyrillic_total = sum(1 for char in word if char in CYRILLIC_CHARS)
        latin_total = sum(1 for char in word if char in LATIN_CHARS)
        
        if cyrillic_total > latin_total:
            mapping = KEYBOARD_LAYOUTS['ru_to_en']
        elif latin_total > cyrillic_total:
            mapping = KEYBOARD_LAYOUTS['en_to_ru']
        else:
            return word
        
        return ''.join(mapping.get(char, char) for char in word)
    
    def _get_suggested_word(self, word, spell_checker):
        """Получение предложений от спеллчекера"""
        try:
            suggestions = spell_checker.lookup(word, Verbosity.CLOSEST, len(word) // 3)
            if suggestions:
                return suggestions[0].term, suggestions[0].distance
            else:
                return None, None           
        except Exception as e:
            return None, None
    
    def _get_correction(self, word, option):
        """Получение исправления для слова"""
        changed_word = self._change_keyboard_layout(word)
        
        suggestions = [
            list(self._get_suggested_word(word, self.spell_checkers[option]['original'])),
            list(self._get_suggested_word(changed_word, self.spell_checkers[option]['original']))
        ]
        
        if option != 'general':
            for source_word in [word, changed_word]:
                suggested = list(self._get_suggested_word(source_word, self.spell_checkers[option]['translated']))
                if suggested[0] is not None:
                    suggested[0] = self.dicts[option].get(suggested[0])
                suggestions.append(suggested)
        
        valid_suggestions = [s for s in suggestions if s[0] is not None and s[1] is not None]
        
        if not valid_suggestions:
            return None
        
        return min(valid_suggestions, key=lambda x: x[1])[0]

    def correct_query(self, query, option='general'):
        """Коррекция запроса"""
        if not query or not isinstance(query, str):
            return query
            
        words = query.split()
        if not words:
            return query
            
        corrected_words = []
        for word in words:
            if word.isdigit():
                corrected_words.append(word)
                continue

            corrected_word = self._get_correction(word, option)
            corrected_words.append(corrected_word if corrected_word is not None else word)
        
        return ' '.join(corrected_words)


class QueryProcessor:
    """Обработка поисковых запросов"""
    
    def __init__(self, preprocessor, spell_checker):
        self.preprocessor = preprocessor
        self.spell_checker = spell_checker
        self.sorted_brands = None
        
    def _find_brand(self, query_text, min_match_ratio):
        """Поиск бренда в запросе"""
        for entity in self.sorted_brands:
            if f" {entity} " in f" {query_text} ":
                return entity
        
        best_match = None
        best_score = 0
        longest_word_length = 0

        query_words = query_text.split()
        for entity in self.sorted_brands:
            entity_words = entity.split()
            
            if len(entity_words) > 1:
                matched_words = sum(1 for ew in entity_words if ew in query_words)
                
                if matched_words > 0:
                    match_ratio = matched_words / len(entity_words)
                    current_longest = max(len(word) for word in entity_words)
                    
                    if (match_ratio >= min_match_ratio and 
                        (current_longest > longest_word_length or 
                         (current_longest == longest_word_length and match_ratio > best_score))):
                        
                        best_score = match_ratio
                        longest_word_length = current_longest
                        best_match = entity
        
        return best_match
    
    def preprocess_queries(self, df, brands):
        """Предобработка поисковых запросов"""
        self.sorted_brands = sorted(brands, key=len, reverse=True)
        
        unique_queries = df['query'].dropna().unique()
        print(f"Найдено {len(unique_queries):,} уникальных запросов из {len(df):,} всего")

        query_mapping = {}
        
        for query in unique_queries:
            if not query or pd.isna(query):
                continue
                
            normalized = self.preprocessor.normalize_text(query)
            corrected_brand = self.spell_checker.correct_query(normalized, 'brand')
            query_mapping[query] = {
                'brand': self._find_brand(corrected_brand, 0.5),
                'model': self.spell_checker.correct_query(normalized, 'model'), 
                'general': self.spell_checker.correct_query(normalized)
            }
        
        df['query_model'] = df['query'].map(lambda x: query_mapping.get(x, {}).get('model', ''))
        df['query_general'] = df['query'].map(lambda x: query_mapping.get(x, {}).get('general', ''))
        df['found_brand'] = df['query'].map(lambda x: query_mapping.get(x, {}).get('brand', ''))
        
        print("Предобработка завершена!")
        return df



class BM25Processor:
    """BM25 для поиска релевантности"""
    
    def __init__(self):
        self.k1 = config.BM25_CONFIG['k1']
        self.b = config.BM25_CONFIG['b']
        self.vocab_size = config.BM25_CONFIG['vocab_size']
        self.batch_size = config.BM25_CONFIG['batch_size']
        self.query_col = config.BM25_CONFIG['query_column']
        self.doc_col = config.BM25_CONFIG['document_column']
        
        self.vectorizer = None
        self.idf_diag = None
        self.avg_doc_length = None

    
    def fit(self, train_df):
        print("Обучение BM25...")
        
        all_train_texts = pd.concat([train_df[self.query_col], train_df[self.doc_col]]).astype(str)
        
        self.vectorizer = CountVectorizer(
            max_features=self.vocab_size,
            analyzer='word',
            token_pattern=r'(?u)\b\w+\b',
            lowercase=False
        )
        self.vectorizer.fit(all_train_texts)
        print(f"Размер словаря: {len(self.vectorizer.vocabulary_):,}")
        
        n_docs_train = len(train_df)
        df_counts = np.zeros(len(self.vectorizer.vocabulary_))
        doc_lengths_list = []

        for i in range(0, n_docs_train, self.batch_size):
            batch_docs = train_df[self.doc_col].iloc[i:i+self.batch_size].astype(str)
            batch_vectors = self.vectorizer.transform(batch_docs)
            df_counts += np.array((batch_vectors > 0).sum(axis=0)).flatten()
            doc_lengths_list.extend(np.array(batch_vectors.sum(axis=1)).flatten())

        idf = np.log((n_docs_train - df_counts + 0.5) / (df_counts + 0.5))
        self.avg_doc_length = np.mean(doc_lengths_list)
        self.idf_diag = sp.diags(idf)
        
        print(f"Средняя длина документа: {self.avg_doc_length:.2f}")

    
    def transform(self, df):
        unique_pairs = df[[self.query_col, self.doc_col]].drop_duplicates()
        total_pairs = len(unique_pairs)
        print(f"Уникальных пар: {total_pairs:,}")
        
        all_scores = []
        
        for i in range(0, total_pairs, self.batch_size):
            batch_pairs = unique_pairs.iloc[i:i+self.batch_size]

            query_vectors = self.vectorizer.transform(batch_pairs[self.query_col].astype(str))
            doc_vectors = self.vectorizer.transform(batch_pairs[self.doc_col].astype(str))

            doc_vectors_bm25 = doc_vectors.astype(np.float64)
            doc_lengths = np.array(doc_vectors.sum(axis=1)).flatten()
            tf = doc_vectors_bm25.data

            doc_indices = np.repeat(np.arange(len(batch_pairs)), 
                                  np.diff(doc_vectors_bm25.indptr))
            doc_lengths_repeated = doc_lengths[doc_indices]

            bm25_scores = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_lengths_repeated / self.avg_doc_length))
            doc_vectors_bm25.data = bm25_scores

            doc_vectors_bm25 = doc_vectors_bm25 * self.idf_diag
            scores = np.array(query_vectors.multiply(doc_vectors_bm25).sum(axis=1)).flatten()
            all_scores.extend(scores)
            
            if i % (self.batch_size) == 0 and i > 0:
                print(f"Обработано {min(i + self.batch_size, total_pairs):,}/{total_pairs:,} пар")
        
        unique_pairs = unique_pairs.copy()
        unique_pairs['bm25_query_model_score'] = all_scores

        result_df = df.merge(unique_pairs, on=[self.query_col, self.doc_col], how='left')
        return result_df

    
    def fit_transform(self, train_df, dfs):
        self.fit(train_df)
        train_df = self.transform(train_df)

        for i, df in enumerate(dfs):
            dfs[i] = self.transform(df)

        print("BM25 обработка завершена")
        return train_df, dfs


class CosineSimilarityCalculator:
    """Расчет косинусной схожести"""
    
    def __init__(self):
        self.model = SentenceTransformer(config.EMBEDDING_CONFIG['model_name'], config.EMBEDDING_CONFIG['device'])
        self.chunk_size = config.EMBEDDING_CONFIG['chunk_size']
        self.embedding_cache = {}
        self.normalize_embeddings = config.EMBEDDING_CONFIG['normalize_embeddings']
        self.convert_to_tensor = config.EMBEDDING_CONFIG['convert_to_tensor']
        self.show_progress_bar = config.EMBEDDING_CONFIG['show_progress_bar']
        self.text_configs = config.EMBEDDING_CONFIG['text_configs']

    
    def _prepare_texts(self, df):
        print("Формируем набор текстов для обработки...")
        
        text_configs = self.text_configs
        texts = {}
        
        for config in text_configs:
            if config['source'] == 'single_column':
                texts[config['name']] = df[config['column']].fillna('').astype(str)
            elif config['source'] == 'combined_columns':
                combined = ''
                for col in config['columns']:
                    combined += df[col].fillna('').astype(str) + ' '
                texts[config['name']] = combined.str.strip()
        
        texts1 = texts['query_text']
        texts2 = texts['category_text']
        
        all_unique_texts = set(texts1) | set(texts2)
        print(f"Всего уникальных текстов для обработки: {len(all_unique_texts):,}")
        
        return list(all_unique_texts), texts1, texts2

    
    def _compute_embeddings(self, texts):
        print("Вычисляем эмбеддинги...")
        
        embeddings = self.model.encode(
            texts, 
            convert_to_tensor= self.convert_to_tensor, 
            normalize_embeddings=self.normalize_embeddings, 
            show_progress_bar=self.show_progress_bar
        )
        
        self.embedding_cache = dict(zip(texts, embeddings))
        print("Эмбеддинги успешно вычислены")

    
    def calculate(self, df):
        """Расчет косинусной близости"""
        all_texts, texts1, texts2 = self._prepare_texts(df)
        self._compute_embeddings(all_texts)
        
        cosine_similarities = []
        
        for i in range(0, len(df), self.chunk_size):
            end_idx = min(i + self.chunk_size, len(df))
            chunk_texts1 = texts1.iloc[i:end_idx]
            chunk_texts2 = texts2.iloc[i:end_idx]

            emb1 = np.array([self.embedding_cache[text] for text in chunk_texts1])
            emb2 = np.array([self.embedding_cache[text] for text in chunk_texts2])

            chunk_cosine = np.einsum('ij,ij->i', emb1, emb2)
            cosine_similarities.extend(chunk_cosine)

            if i % (self.chunk_size * 5) == 0 and i > 0:
                print(f"Обработано {min(i + self.chunk_size, len(df)):,}/{len(df):,} записей")
        
        del self.embedding_cache
        import gc
        gc.collect()
        
        return df.assign(cosine_similarity=cosine_similarities)


#==== ТЕКСТОВЫЕ ФИЧИ ====
class QueryFeatures:
    def process(self, df):
        df['query_length'] = df['query'].str.len().fillna(0).astype(np.int16)
        df['query_word_count'] = df['query'].str.split().str.len().fillna(0).astype(np.int8)
        df['query_avg_word_length'] = (df['query_length'] / (df['query_word_count'] + 1)).astype(np.float32)
        df['query_complexity'] = (df['query_word_count'] * df['query_avg_word_length']).astype(np.float32)
        
        query_freq = df['query'].value_counts().reset_index()
        query_freq.columns = ['query', 'query_global_frequency']
        df = df.merge(query_freq, on='query', how='left')
        
        df['query_global_frequency_log'] = np.log1p(df['query_global_frequency'].fillna(1)).astype(np.float32)
        
        return df
        
class SimilarityRankFeatures:
    def process(self, df):
                
        df['brand'] = df['brand'].astype(str)
        df['found_brand'] = df['found_brand'].astype(str)
        df['brand_match'] = (df['found_brand'] == df['brand']).astype(int)

        df = df.sort_values([
            'search_number', 'brand_match', 'bm25_query_model_score', 'cosine_similarity'
        ], ascending=[True, False, False, False])
        
        df['similarity_rank'] = df.groupby('search_number').cumcount() + 1
        
        return df


#=== ПРОДУКТОВЫЕ ФИЧИ ===
class TemporalProductMetrics:     
    def process(self, df, is_train, train_metrics=None):
        def calculate_cumulative_metrics(df, initial_metrics=None):
            df = df.sort_values(['actual_product_id', 'date'])
            df['session_counter'] = df.groupby('actual_product_id').cumcount()
            
            if initial_metrics:
                df['initial_views'] = df['actual_product_id'].map(
                    lambda x: initial_metrics.get(x, {}).get('cumulative_views', 0)
                )
                df['initial_clicks'] = df['actual_product_id'].map(
                    lambda x: initial_metrics.get(x, {}).get('cumulative_clicks', 0)
                )
                df['initial_carts'] = df['actual_product_id'].map(
                    lambda x: initial_metrics.get(x, {}).get('cumulative_carts', 0)
                )
                df['initial_purchases'] = df['actual_product_id'].map(
                    lambda x: initial_metrics.get(x, {}).get('cumulative_purchases', 0)
                )
                df['initial_relevance'] = df['actual_product_id'].map(
                    lambda x: initial_metrics.get(x, {}).get('cumulative_relevance_sum', 0)
                )
            else:
                df['initial_views'] = 0
                df['initial_clicks'] = 0
                df['initial_carts'] = 0
                df['initial_purchases'] = 0
                df['initial_relevance'] = 0
            
            df['cumulative_views'] = df.groupby('actual_product_id')['session_counter'].cumsum() + 1 + df['initial_views']
            
            df['is_click'] = (df['relevance'] >= 1).astype(np.int8)
            df['is_cart'] = (df['relevance'] >= 2).astype(np.int8)
            df['is_purchase'] = (df['relevance'] == 3).astype(np.int8)
            
            df['cumulative_clicks'] = df.groupby('actual_product_id')['is_click'].cumsum() + df['initial_clicks']
            df['cumulative_carts'] = df.groupby('actual_product_id')['is_cart'].cumsum() + df['initial_carts']
            df['cumulative_purchases'] = df.groupby('actual_product_id')['is_purchase'].cumsum() + df['initial_purchases']
            df['cumulative_relevance_sum'] = df.groupby('actual_product_id')['relevance'].cumsum() + df['initial_relevance']
            
            df.drop(['session_counter', 'initial_views', 'initial_clicks', 'initial_carts', 
                           'initial_purchases', 'initial_relevance', 'is_click', 'is_cart', 'is_purchase'], 
                          axis=1, inplace=True)
            
            return df

        def calculate_final_metrics(df):
            required_columns = ['cumulative_views', 'cumulative_clicks', 'cumulative_carts', 
                               'cumulative_purchases', 'cumulative_relevance_sum']
            
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 0
            
            mask = df['cumulative_views'] > 0
            df.loc[mask, 'historical_CTR'] = df.loc[mask, 'cumulative_clicks'] / df.loc[mask, 'cumulative_views']
            df.loc[~mask, 'historical_CTR'] = 0.0
            
            df.loc[mask, 'historical_CVR'] = df.loc[mask, 'cumulative_purchases'] / df.loc[mask, 'cumulative_views']
            df.loc[~mask, 'historical_CVR'] = 0.0
            
            df.loc[mask, 'historical_Cart_Rate'] = df.loc[mask, 'cumulative_carts'] / df.loc[mask, 'cumulative_views']
            df.loc[~mask, 'historical_Cart_Rate'] = 0.0

            mask_clicks = df['cumulative_clicks'] > 0
            df.loc[mask_clicks, 'historical_Conversion_Rate'] = df.loc[mask_clicks, 'cumulative_purchases'] / df.loc[mask_clicks, 'cumulative_clicks']
            df.loc[~mask_clicks, 'historical_Conversion_Rate'] = 0.0
            
            mask_carts = df['cumulative_carts'] > 0
            df.loc[mask_carts, 'historical_Purchase_to_Cart_Rate'] = df.loc[mask_carts, 'cumulative_purchases'] / df.loc[mask_carts, 'cumulative_carts']
            df.loc[~mask_carts, 'historical_Purchase_to_Cart_Rate'] = 0.0

            df['historical_data_confidence'] = (np.log1p(df['cumulative_views']) / 10.0).astype(np.float32)
            
            return df

        if is_train:
            df = calculate_cumulative_metrics(df)
            df = calculate_final_metrics(df)

            self.final_product_metrics = df[['actual_product_id', 'cumulative_views', 'cumulative_clicks', 
                                                 'cumulative_carts', 'cumulative_purchases', 'cumulative_relevance_sum']].copy()
        else:
            latest_product_metrics = train_metrics.groupby('actual_product_id').agg({
                'cumulative_views': 'last',
                'cumulative_clicks': 'last', 
                'cumulative_carts': 'last',
                'cumulative_purchases': 'last',
                'cumulative_relevance_sum': 'last'
            }).fillna(0).reset_index()
            
            df = df.merge(
                latest_product_metrics, 
                on='actual_product_id', 
                how='left'
            )

            numeric_cols_to_fill = ['cumulative_views', 'cumulative_clicks', 'cumulative_carts', 
                                   'cumulative_purchases', 'cumulative_relevance_sum']
            for col in numeric_cols_to_fill:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
            
            if 'cumulative_views_x' in df.columns and 'cumulative_views_y' in df.columns:
                df['cumulative_views'] = df['cumulative_views_y']
                df = df.drop(['cumulative_views_x', 'cumulative_views_y'], axis=1)
            elif 'cumulative_views_x' in df.columns:
                df['cumulative_views'] = df['cumulative_views_x']
                df = df.drop('cumulative_views_x', axis=1)
            elif 'cumulative_views_y' in df.columns:
                df['cumulative_views'] = df['cumulative_views_y']
                df = df.drop('cumulative_views_y', axis=1)
            
            df = calculate_final_metrics(df)

        return df


#=== СЕССИОННЫЕ / ВРЕМЕННЫЕ ФИЧИ ===
class SessionContextFeatures:
    def process(self, df):
        df['product_age_days'] = (df['date'] - df['date_of_create']).dt.days
 
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofweek / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofweek / 7)

        df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)

        df['day_of_month_sin'] = np.sin(2 * np.pi * df['date'].dt.day / 31)
        df['day_of_month_cos'] = np.cos(2 * np.pi * df['date'].dt.day / 31)

        df['product_create_month_sin'] = np.sin(2 * np.pi * df['date_of_create'].dt.month / 12)
        df['product_create_month_cos'] = np.cos(2 * np.pi * df['date_of_create'].dt.month / 12)

        df['position_normalized'] = (df['product_position'] / df['response_products']).astype(np.float32)          

        df['position_in_group'] = df.groupby('search_number')['product_position'].rank()
        df['group_size'] = df.groupby('search_number').transform('size')

        df['browse_time_log'] = np.log1p(df['browse_time']).astype(np.float32)
        df['response_time_ms_log'] = np.log1p(df['response_time'] / 1000).astype(np.float32)
        
        df['selected_category_present'] = (df['selected_category_id'] > 0).astype(np.int8)
        
        return df


#=== ЦЕНОВЫЕ ФИЧИ ===
class PriceFeatures:
    def process(self, df):
        df['price_byn'] = (df['price'] / 100).astype(np.float32)
        df['discount_ratio'] = np.where(
            df['price'] > 0,
            df['price_discount'] / df['price'],
            0.0
        )

        category1_avg_price = df.groupby('category_name_1')['price_byn'].transform('mean')
        df['price_position_category1'] = (df['price_byn'] / category1_avg_price).astype(np.float32)
        
        category4_avg_price = df.groupby('category_name_4')['price_byn'].transform('mean')
        df['price_position_category4'] = (df['price_byn'] / category4_avg_price).astype(np.float32)
        
        type_avg_price = df.groupby('type')['price_byn'].transform('mean')
        df['price_position_type'] = (df['price_byn'] / type_avg_price).astype(np.float32)
        
        df['log_price'] = np.log1p(df['price_byn']).astype(np.float32)
        
        df['rank_in_group_by_price'] = df.groupby('search_number')['price_byn'].rank(
            method='dense', ascending=True
        ).astype(np.int16)

        df['rank_in_group_by_discount'] = df.groupby('search_number')['discount_ratio'].rank(
            method='dense', ascending=False
        ).fillna(0).astype(np.int16)

        return df


class CompetitivePriceFeatures:
    def process(self, df, is_train, train_category_stats=None):
        df = df.sort_values(['category_name_1', 'date']).reset_index(drop=True)
        
        def calculate_category_metrics(df, initial_stats=None):
            if initial_stats is not None:
                df = df.merge(initial_stats, on='category_name_1', how='left', suffixes=('', '_train'))
                df['initial_mean'] = df.get('mean', 0)
                df['initial_std'] = df.get('std', 0)
                df['initial_count'] = df.get('count', 0)
            else:
                df['initial_mean'] = 0
                df['initial_std'] = 0
                df['initial_count'] = 0
            
            df['cumulative_price_sum'] = df.groupby('category_name_1')['price_byn'].cumsum() - df['price_byn'] + df['initial_mean'] * df['initial_count']
            df['cumulative_price_count'] = df.groupby('category_name_1').cumcount() + df['initial_count']
            
            df['category_mean'] = np.where(
                df['cumulative_price_count'] > 0,
                df['cumulative_price_sum'] / df['cumulative_price_count'],
                df['initial_mean']
            ).astype(np.float32)
            
            df['price_diff_sq'] = (df['price_byn'] - df['category_mean'])**2
            df['cumulative_price_diff_sq'] = df.groupby('category_name_1')['price_diff_sq'].cumsum() - df['price_diff_sq'] + df['initial_std']**2 * df['initial_count']

            drop_cols = ['cumulative_price_sum', 'cumulative_price_count', 'price_diff_sq', 
                        'cumulative_price_diff_sq', 'initial_mean', 'initial_std', 'initial_count']
            if 'mean' in df.columns:
                drop_cols.extend(['mean', 'std', 'count'])
            
            df.drop(drop_cols, axis=1, inplace=True, errors='ignore')
            
            return df

        if is_train:
            df = calculate_category_metrics(df)
        else:
            if train_category_stats is not None:
                latest_category_stats = train_category_stats.groupby('category_name_1').agg({
                    'price_byn': ['mean', 'std', 'count']
                }).fillna(0)
                latest_category_stats.columns = ['mean', 'std', 'count']
                latest_category_stats = latest_category_stats.reset_index()
                
                df = calculate_category_metrics(df, latest_category_stats)
            else:
                df = calculate_category_metrics(df)
        
        return df



#=== ПОЛЬЗОВАТЕЛЬСКИЕ ФИЧИ ===
class UserBehaviorFeatures:  
    def process(self, df, is_train, train_user_metrics=None):
        df = df.sort_values(['user_number', 'date']).reset_index(drop=True)
        
        def calculate_user_metrics(df, initial_metrics=None):
            df['session_counter'] = df.groupby('user_number').cumcount()
            df['user_activity'] = (df['session_counter'] - 1).clip(lower=0)
            df['user_activity_log'] = np.log1p(df['user_activity']).astype(np.float32)
            
            if initial_metrics:
                df['initial_complexity_sum'] = df['user_number'].map(
                    lambda x: initial_metrics.get(x, {}).get('cumulative_complexity_sum', 0)
                )
                df['initial_sessions_count'] = df['user_number'].map(
                    lambda x: initial_metrics.get(x, {}).get('cumulative_sessions_count', 0)
                )
                df['initial_filters_sum'] = df['user_number'].map(
                    lambda x: initial_metrics.get(x, {}).get('cumulative_filters_sum', 0)
                )
            else:
                df['initial_complexity_sum'] = 0
                df['initial_sessions_count'] = 0
                df['initial_filters_sum'] = 0
                df['initial_device'] = 'new_user'
            
            df['cumulative_complexity_sum'] = df.groupby('user_number')['query_complexity'].cumsum() - df['query_complexity'] + df['initial_complexity_sum']
            df['cumulative_sessions_count'] = df.groupby('user_number').cumcount() + df['initial_sessions_count']
            df['cumulative_filters_sum'] = df.groupby('user_number')['filters_applied'].cumsum() - df['filters_applied'] + df['initial_filters_sum']
            
            df['avg_query_complexity'] = np.where(
                df['cumulative_sessions_count'] > 0,
                df['cumulative_complexity_sum'] / df['cumulative_sessions_count'],
                0.0
            ).astype(np.float32)
            
            df['filters_usage_rate'] = np.where(
                df['cumulative_sessions_count'] > 0,
                df['cumulative_filters_sum'] / df['cumulative_sessions_count'],
                0.0
            ).astype(np.float32)
            
            
            df.drop([
                'session_counter', 'initial_complexity_sum', 'initial_sessions_count', 
                'initial_filters_sum', 'initial_device', 'query_complexity'
            ], axis=1, inplace=True, errors='ignore')
            
            return df

        if is_train:
            df = calculate_user_metrics(df)
        else:
            latest_user_metrics = train_user_metrics.groupby('user_number').agg({
                'cumulative_complexity_sum': 'last',
                'cumulative_sessions_count': 'last',
                'cumulative_filters_sum': 'last',
            }).reset_index()
            
            numeric_cols = ['cumulative_complexity_sum', 'cumulative_sessions_count', 'cumulative_filters_sum']
            latest_user_metrics[numeric_cols] = latest_user_metrics[numeric_cols].fillna(0)
            
            initial_metrics_dict = latest_user_metrics.set_index('user_number').to_dict('index')
            
            df = calculate_user_metrics(df, initial_metrics_dict)
        
        return df


#=== ВЗАИМОДЕЙСТВИЯ ФИЧЕЙ ===
class FeatureInteractions:
    def process(self, df):

        df['product_age_days_x_historical_data_confidence'] = df['product_age_days'] * df['historical_data_confidence']
        df['position_x_historical_ctr'] = df['position_normalized'] * df['historical_CTR']
        df['position_x_historical_cvr'] = df['position_normalized'] * df['historical_Conversion_Rate']
        df['position_x_historical_Cart_Rate'] = df['position_normalized'] * df['historical_Cart_Rate']
        
        df['bm25_x_position'] = df['bm25_query_model_score'] * df['position_normalized']
        df['bm25_x_historical_cvr'] = df['bm25_query_model_score'] * df['historical_CVR']

        return df


#=== ОПТИМИЗАЦИЯ ПАМЯТИ ===
class MemoryOptimizer:
    def process(self, df):      
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        
        return df


class FeatureEngine:
    def __init__(self):
        self.similarity_rank = SimilarityRankFeatures()
        self.temporal_metrics = TemporalProductMetrics()
        self.session_context = SessionContextFeatures()
        self.price = PriceFeatures()
        self.query_features = QueryFeatures()
        self.feature_interactions = FeatureInteractions()
        self.competitive_price = CompetitivePriceFeatures()
        self.user_behavior = UserBehaviorFeatures()
        self.memory_optimizer = MemoryOptimizer()

    def add_all_features(self, sessions, is_train, train_data=None):
        df = sessions.copy()

        df = self.temporal_metrics.process(df, is_train, train_data)
        df = self.query_features.process(df)     
        df = self.user_behavior.process(df, is_train, train_data)
        df = self.price.process(df)
        df = self.competitive_price.process(df, is_train, train_data)
        df = self.session_context.process(df)
        df = self.similarity_rank.process(df)
        df = self.memory_optimizer.process(df)
        df = self.feature_interactions.process(df)
        
        return df


class FeatureNormalizer:
    def __init__(self):
        self.scalers = {}
        self.encoder = None
    
    def fit(self, df):
        self.scalers['historical'] = QuantileTransformer(output_distribution='normal', random_state=config.SEED)
        self.scalers['historical'].fit(df[['historical_CTR', 'historical_CVR', 'historical_Cart_Rate']])

        self.scalers['polynomial'] = QuantileTransformer(output_distribution='normal', random_state=config.SEED)
        self.scalers['polynomial'].fit(df[['bm25_x_historical_cvr']])

        self.scalers['ratio'] = QuantileTransformer(output_distribution='normal', random_state=config.SEED)
        self.scalers['ratio'].fit(df[['discount_ratio']])

        self.scalers['filters'] = RobustScaler()
        self.scalers['filters'].fit(df[['filters_applied']])
    
    def transform(self, df):
        df[['historical_CTR', 'historical_CVR', 'historical_Cart_Rate']] = \
            self.scalers['historical'].transform(df[['historical_CTR', 'historical_CVR', 'historical_Cart_Rate']])
        
        df[['bm25_x_historical_cvr']] = self.scalers['polynomial'].transform(df[['bm25_x_historical_cvr']])
        
        df[['discount_ratio']] = self.scalers['ratio'].transform(df[['discount_ratio']])

        df[['filters_applied']] = self.scalers['filters'].transform(df[['filters_applied']])
        
        df['device_encoded'] = df['device'].astype('category').cat.codes

        df['type_encoded'] = df['type'].astype('category').cat.codes
        df['category_encoded'] = df['category_name_4'].astype('category').cat.codes
        
        return df


print_complete_config_summary(config)


train = train_df.copy()
test = test_df.copy()
products = products_df.copy()

data_sampler = DataSampler()

train = data_sampler.group_sampling(train, config.SAMPLING_CONFIG['max_group_size'])
train = data_sampler.filter_short_groups(train)

train, test = data_sampler.date_to_datetime([train, test])

if config.IS_VALIDATION:
    train, validation = data_sampler.create_validation_split(train, 3)


preprocessor = TextPreprocessor()
spell_checker = SpellChecker(preprocessor)
query_processor = QueryProcessor(preprocessor, spell_checker)

products, brands, models, categories, types, countries, brands_transl_dict, models_transl_dict = preprocessor.normalize_products(products)
products = preprocessor.process_column(products, 'model')

spell_checker.setup_spell_checkers(
    brands, models, categories, types, countries, 
    brands_transl_dict, models_transl_dict
)

product_processor = ProductFeaturesProcessor()
products_prepared = product_processor.prepare_products(products)

if config.IS_VALIDATION:
    train, validation, test = product_processor.process([train, validation, test])
else:
    train, test = product_processor.process([train, test])

print("Предобработка пользовательских запросов train")
train = query_processor.preprocess_queries(train, brands)
train = preprocessor.process_column(train, 'query_model')

print("Предобработка пользовательских запросов test")
test = query_processor.preprocess_queries(test, brands)
test = preprocessor.process_column(test, 'query_model')

if config.IS_VALIDATION:
    print("Предобработка пользовательских запросов validation 1")
    validation = query_processor.preprocess_queries(validation, brands)
    validation = preprocessor.process_column(validation, 'query_model')
    
print("Предобработка завершена!")


bm25_processor = BM25Processor()

if config.IS_VALIDATION:
    train, dfs = bm25_processor.fit_transform(train_df=train, dfs=[validation, test])
    validation, test = dfs
else:
    train, dfs = bm25_processor.fit_transform(train_df=train, dfs=[test])
    test = dfs[0]


cosine_calculator = CosineSimilarityCalculator()
train = cosine_calculator.calculate(train)
test = cosine_calculator.calculate(test)

if config.IS_VALIDATION:
    validation = cosine_calculator.calculate(validation)


feature_engine = FeatureEngine()

print("Создание признаков для train")
train_features = feature_engine.add_all_features(train, is_train=True)

print("Создание признаков для test")
test_features = feature_engine.add_all_features(test, is_train=False, train_data=train_features)

if config.IS_VALIDATION:
    print("Создание признаков для validation")
    validation_features = feature_engine.add_all_features(validation, is_train=False, train_data=train_features)


train_features.info()


normalizer = FeatureNormalizer()
normalizer.fit(train_features)

train_features = normalizer.transform(train_features)
test_features = normalizer.transform(test_features)

if config.IS_VALIDATION:
    validation_features = normalizer.transform(validation_features)
print("Нормализация завершена")


train_sorted = train_features.sort_values(['search_number'])
test_sorted = test_features.sort_values(['search_number'])

X_train = train_sorted
X_test = test_sorted.copy()
y_train = train_sorted['relevance']

train_groups = train_sorted.groupby('search_number').size().tolist()

if config.IS_VALIDATION:
    validation_sorted = validation_features.sort_values(['search_number'])
    X_validation = validation_sorted.copy()
    y_validation = validation_sorted['relevance']
    validation_groups = validation_sorted.groupby('search_number').size().tolist()

existing_columns = [col for col in COLUMNS_TO_DROP if col in X_train.columns]
X_train.drop(existing_columns, axis=1, inplace=True)

existing_columns = [col for col in COLUMNS_TO_DROP if col in X_test.columns]
X_test.drop(existing_columns, axis=1, inplace=True)

if config.IS_VALIDATION:
    existing_columns = [col for col in COLUMNS_TO_DROP if col in X_validation.columns]
    X_validation.drop(existing_columns, axis=1, inplace=True)


X_train.info()


params = config.LGBM_PARAMS

lgb_ranker = lgb.LGBMRanker(     
    objective=params['objective'],
    metric=params['metric'],
    boosting_type=params['boosting_type'],                           
    lambdarank_norm=params['lambdarank_norm'],

    lambdarank_truncation_level=params['lambdarank_truncation_level'],
    ndcg_eval_at=params['ndcg_eval_at'],

    num_leaves=params['num_leaves'],
    min_data_in_leaf=params['min_data_in_leaf'],
    min_child_samples=params['min_child_samples'],
    reg_alpha=params['reg_alpha'],
    reg_lambda=params['reg_lambda'],
    subsample=params['subsample'],
    colsample_bytree=params['colsample_bytree'],
    subsample_freq=params['subsample_freq'],
    
    learning_rate=params['learning_rate'],
    n_estimators=params['n_estimators'],
    n_jobs=params['n_jobs'],

    max_depth=params['max_depth'],
    max_bin=params['max_bin'],
    verbosity=params['verbosity'],
    force_row_wise=params['force_row_wise'],

    random_state=config.SEED
)

lgb_ranker.fit(
    X_train, y_train,
    group=train_groups,
)

test_predictions = lgb_ranker.predict(X_test)
submission = pd.DataFrame({
    'session_product_id': test_sorted['session_product_id'],
    'relevance': test_predictions
})
submission = submission.drop_duplicates(subset=['session_product_id'], keep='first')
submission.to_csv(config.OUTPUT_CONFIG['submission_file'], index=False)
print("Сабмит готов")


feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': lgb_ranker.feature_importances_
}).sort_values('importance', ascending=False)

feature_importance.to_csv(config.OUTPUT_CONFIG['feature_importance_file'], index=False)
print("Feature importance сохранен в lgb_feature_importance.csv")


def plot_feature_importance(feature_importance, top_n=20, save_path=None):
    """
    Визуализация важности признаков
    """
    fig, axes = plt.subplots(2, figsize=(16, 12))

    top_features = feature_importance.head(top_n)

    axes[0].barh(range(len(top_features)), top_features['importance'])
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features['feature'])
    axes[0].set_title(f'Топ {top_n} самых важных признаков')
    axes[0].set_xlabel('Важность')
    axes[0].set_ylabel('Признаки')

    feature_importance['cumulative'] = feature_importance['importance'].cumsum() / feature_importance['importance'].sum()
    axes[1].plot(range(1, len(feature_importance) + 1), feature_importance['cumulative'])
    axes[1].set_xlabel('Количество признаков')
    axes[1].set_ylabel('Кумулятивная важность')
    axes[1].set_title('Кумулятивная важность признаков')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

plot_feature_importance(feature_importance, top_n=20)

