import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

# 1. Загрузка данных (полностью)
print("Загрузка данных...")
train_data = pd.read_csv('/kaggle/input/microsoft-malware-prediction/train.csv')
test_data = pd.read_csv('/kaggle/input/microsoft-malware-prediction/test.csv')

print(f"Train shape: {train_data.shape}")
print(f"Test shape: {test_data.shape}")



pd.set_option('display.max_columns', None)  # показывать все столбцы
pd.set_option('display.width', 1000)   


train_data.columns


# Вывести первые 10 строк для этих столбцов
train_data[train_data.columns[:83]].head()



import pandas as pd

def get_categorical_features(df, target_col=None, id_cols=None):
    """
    Возвращает список имён столбцов с типом 'object' (категориальные),
    исключая целевую переменную и столбцы‑идентификаторы.
    
    Параметры:
    - df: pandas DataFrame
    - target_col: имя целевой переменной (строка, опционально)
    - id_cols: список имён столбцов‑идентификаторов (список строк, опционально)
    
    Возвращает:
    - список имён категориальных столбцов (без target и id_cols)
    """
    # Находим все столбцы с типом object
    cat_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Исключаем целевую переменную, если указана
    if target_col is not None:
        if target_col in cat_features:
            cat_features.remove(target_col)

    # Исключаем идентификаторы, если указаны
    if id_cols is not None:
        for col in id_cols:
            if col in cat_features:
                cat_features.remove(col)

    return cat_features


# Пример использования:
# Предположим, ваш DataFrame называется `train_data`
# Целевая переменная — 'IsMalware' (пример)
# Идентификаторы — ['MachineIdentifier', 'OtherId'] (пример)

cat_features = get_categorical_features(
    train_data,
    target_col='IsMalware',           # замените на имя вашей целевой переменной
    id_cols=['MachineIdentifier']      # замените/дополните список id-столбцов
)

print("Категориальные признаки (без целевой и id):")
print(cat_features)



import pandas as pd
import numpy as np

def analyze_dataframe_separate(df, max_top_values=5, iqr_factor=1.5):
    """
    Раздельный анализ числовых и категориальных столбцов DataFrame.
    
    Параметры:
    - df: pandas DataFrame
    - max_top_values: сколько топ-значений показывать для категориальных
    - iqr_factor: множитель для определения выбросов (по умолчанию 1.5)
    """
    print(f"=== АНАЛИЗ ВСЕГО ДАТАСЕТА ===\n")
    print(f"Размер данных: {df.shape[0]} строк, {df.shape[1]} столбцов\n")
    
    
    # Разделяем столбцы по типам
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    
    print(f"Числовые столбцы ({len(num_cols)}): {num_cols}")
    print(f"Категориальные столбцы ({len(cat_cols)}): {cat_cols}\n")
    
    
    # 1. Анализ числовых столбцов
    print("=" * 50)
    print("АНАЛИЗ ЧИСЛОВЫХ СТОЛБЦОВ")
    print("=" * 50)
    
    for col in num_cols:
        print(f"\n--- {col} (числовой) ---")
        print(f"Тип данных: {df[col].dtype}")
        
        # Пропуски
        na_count = df[col].isna().sum()
        na_percent = (na_count / len(df)) * 100
        print(f"Пропуски: {na_count} ({na_percent:.2f}%)")
        
        
        # Основная статистика
        stats = df[col].describe()
        print("Статистика (describe):")
        print(f"count: {stats['count']:.0f}")
        print(f"mean:  {stats['mean']:.6f}")
        print(f"std:   {stats['std']:.6f}")
        print(f"min:   {stats['min']:.6f}")
        print(f"25%:   {stats['25%']:.6f}")
        print(f"50%:   {stats['50%']:.6f}")
        print(f"75%:   {stats['75%']:.6f}")
        print(f"max:   {stats['max']:.6f}")
        
        # Выбросы (по методу IQR)
        Q1 = stats['25%']
        Q3 = stats['75%']
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"Выбросы (IQR, factor={iqr_factor}): {len(outliers)} значений")
        if len(outliers) > 0:
            print(f"Примеры выбросов: {outliers[col].head(5).tolist()}")
        
        print("-!" * 30)
    
    # 2. Анализ категориальных столбцов
    print("=" * 50)
    print("АНАЛИЗ КАТЕГОРИАЛЬНЫХ СТОЛБЦОВ")
    print("=" * 50)
    
    for col in cat_cols:
        print(f"\n--- {col} (категориальный) ---")
        print(f"Тип данных: {df[col].dtype}")
        
        # Пропуски
        na_count = df[col].isna().sum()
        na_percent = (na_count / len(df)) * 100
        print(f"Пропуски: {na_count} ({na_percent:.2f}%)")
        
        # Уникальность
        nunique = df[col].nunique()
        print(f"Уникальных значений: {nunique}")
        
        # Топ-значения
        print(f"Топ-{max_top_values} значений по частоте:")
        value_counts = df[col].value_counts().head(max_top_values)
        for val, count in value_counts.items():
            val_str = str(val) if len(str(val)) < 50 else str(val)[:47] + "..."
            print(f"{val_str}: {count}")
        
        # Редкие категории (доля < 1%)
        total = len(df[col].dropna())
        rare_cats = df[col].value_counts()[df[col].value_counts() / total < 0.01]
        print(f"Редкие категории (доля <1%): {len(rare_cats)} типов")
        if len(rare_cats) > 0:
            print(f"Примеры редких: {list(rare_cats.head(5).index)}")
        
        # Все уникальные значения (если их мало)
        if nunique <= 10:
            print("Все уникальные значения:")
            all_vals = df[col].unique()
            for val in all_vals:
                val_str = str(val) if len(str(val)) < 50 else str(val)[:47] + "..."
                print(val_str)
        
        print("-!" * 30)


analyze_dataframe_separate(train_data, max_top_values=5, iqr_factor=1.5)



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


# Предположим, что train_data уже загружен
print("Исходный размер train_data:", train_data.shape)

# Отбираем 1 млн строк случайным образом
sample_size = 1_000_000
train_sample = train_data.sample(n=sample_size, random_state=42).reset_index(drop=True)

print("Размер выборки:", train_sample.shape)


# Сохраняем Id и целевую переменную
ids = train_sample['MachineIdentifier'].copy()
target = train_sample['HasDetections'].copy()


# Удаляем Id и целевую из признаков (если они есть в датасете)
features_df = train_sample.drop(['MachineIdentifier', 'HasDetections'], axis=1, errors='ignore')


# Список всех числовых признаков из вашего вывода
features = [
    'IsBeta', 'RtpStateBitfield', 'IsSxsPassiveMode', 'DefaultBrowsersIdentifier',
    'AVProductStatesIdentifier', 'AVProductsInstalled', 'AVProductsEnabled', 'HasTpm',
    'CountryIdentifier', 'CityIdentifier', 'OrganizationIdentifier', 'GeoNameIdentifier',
    'LocaleEnglishNameIdentifier', 'OsBuild', 'OsSuite', 'IsProtected',
    'AutoSampleOptIn', 'SMode', 'IeVerIdentifier', 'Firewall', 'UacLuaenable',
    'Census_OEMNameIdentifier', 'Census_OEMModelIdentifier', 'Census_ProcessorCoreCount',
    'Census_ProcessorManufacturerIdentifier', 'Census_ProcessorModelIdentifier',
    'Census_PrimaryDiskTotalCapacity', 'Census_SystemVolumeTotalCapacity',
    'Census_HasOpticalDiskDrive', 'Census_TotalPhysicalRAM',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches',
    'Census_InternalPrimaryDisplayResolutionHorizontal',
    'Census_InternalPrimaryDisplayResolutionVertical',
    'Census_InternalBatteryNumberOfCharges', 'Census_OSBuildNumber',
    'Census_OSBuildRevision', 'Census_OSInstallLanguageIdentifier',
    'Census_OSUILocaleIdentifier', 'Census_IsPortableOperatingSystem',
    'Census_IsFlightingInternal', 'Census_IsFlightsDisabled', 'Census_ThresholdOptIn',
    'Census_FirmwareManufacturerIdentifier', 'Census_FirmwareVersionIdentifier',
    'Census_IsSecureBootEnabled', 'Census_IsWIMBootEnabled', 'Census_IsVirtualDevice',
    'Census_IsTouchEnabled', 'Census_IsPenCapable',
    'Census_IsAlwaysOnAlwaysConnectedCapable', 'Wdft_IsGamer', 'Wdft_RegionIdentifier'
]

# Проверяем, что все признаки есть в датасете
features = [f for f in features if f in features_df.columns]
print("Количество используемых признаков:", len(features))


# Определяем категориальные признаки
categorical_features = [
    'CountryIdentifier', 'CityIdentifier', 'OrganizationIdentifier',
    'GeoNameIdentifier', 'LocaleEnglishNameIdentifier',
    'OsSuite', 'SMode', 'Firewall', 'UacLuaenable',
    'Census_ProcessorManufacturerIdentifier', 'Census_FirmwareManufacturerIdentifier'
]

# Фильтруем, оставляя только те, что есть в features
categorical_features = [col for col in categorical_features if col in features]
print("Категориальные признаки:", categorical_features)



X = features_df[features]
y = target

# ВСТАВЛЯЕМ БЛОК ЗДЕСЬ — после создания X и y, перед train_test_split
for col in categorical_features:
    X[col] = X[col].astype(str)

# Разбиваем на train/valid (например, 80%/20%)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_pool = Pool(X_train, label=y_train, cat_features=categorical_features)
valid_pool = Pool(X_valid, label=y_valid, cat_features=categorical_features)



# Улучшенная модель
model = CatBoostClassifier(
    iterations=2500,           # больше итераций
    learning_rate=0.03,       # медленнее обучение
    depth=8,                  # глубже деревья
    l2_leaf_reg=9,           # чуть больше регуляризации
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=100
)

model.fit(train_pool, eval_set=valid_pool, plot=False)


# Предсказания
y_pred_proba = model.predict_proba(X_valid)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

# Метрики
metrics = {
    'Accuracy': accuracy_score(y_valid, y_pred),
    'Precision': precision_score(y_valid, y_pred),
    'Recall': recall_score(y_valid, y_pred),
    'F1': f1_score(y_valid, y_pred),
    'AUC-ROC': roc_auc_score(y_valid, y_pred_proba)
}

for k, v in metrics.items():
    print(f"{k}: {v:.4f}")


# --- ПРЕДСКАЗАНИЕ НА ТЕСТОВЫХ ДАННЫХ ПО ЧАНКАМ ---
print("Подготовка тестовых данных и предсказание по чанкам...")

chunk_size = 100000  # Размер чанка: 100 000 строк за раз
total_rows = len(test_data)
test_pred_proba = []  # Список для хранения предсказаний

# Проходим по тестовым данным чанками
for start_idx in range(0, total_rows, chunk_size):
    end_idx = min(start_idx + chunk_size, total_rows)
    print(f"Обрабатываем строки {start_idx}–{end_idx} из {total_rows}...")

    # Берём чанк данных
    chunk = test_data.iloc[start_idx:end_idx].copy()
    X_chunk = chunk[features].copy()

    # Приводим категориальные признаки к строковому типу для чанка
    for col in categorical_features:
        if col in X_chunk.columns:
            # Заменяем пропуски на строку 'nan'
            X_chunk[col] = X_chunk[col].fillna('nan')
            # Приводим к строке
            X_chunk[col] = X_chunk[col].astype(str)

    # Делаем предсказание для текущего чанка
    pred_chunk = model.predict_proba(X_chunk)[:, 1]
    test_pred_proba.extend(pred_chunk)

    # Освобождаем память от временных объектов
    del chunk, X_chunk, pred_chunk
    import gc
    gc.collect()

# Преобразуем список предсказаний в массив NumPy
test_pred_proba = np.array(test_pred_proba)
print(f"\nВсего предсказано {len(test_pred_proba)} строк")

# Создаём DataFrame для сабмишена
submission = pd.DataFrame({
    'MachineIdentifier': test_data['MachineIdentifier'],
    'HasDetections': test_pred_proba
})

# Проверки перед сохранением
assert len(submission) == 7853253, f"Неверное число строк: {len(submission)} (ожидается 7 853 253)"
assert list(submission.columns) == ['MachineIdentifier', 'HasDetections'], "Неверные названия столбцов!"
assert submission['MachineIdentifier'].is_unique, "Дубликаты в MachineIdentifier!"
assert (submission['HasDetections'] >= 0).all() and (submission['HasDetections'] <= 1).all(), \
    "Вероятности выходят за диапазон [0, 1]!"
assert not submission.isna().any().any(), "Есть пропуски в сабмишене!"

# Сохраняем в CSV
submission.to_csv(
    'submission.csv',
    index=False,
    header=True
)

print("\nСабмишен успешно сохранён как 'submission.csv'")
print(f"Количество строк: {len(submission)}")
print(f"Столбцы: {list(submission.columns)}")
print("Пример первых 3 строк:")
print(submission.head(3))






import pandas as pd
import numpy as np
import gc

# Предварительно задаём типы данных для экономии памяти
dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'category',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float16',
        'AVProductStatesIdentifier':                            'category',
        'AVProductsInstalled':                                  'category',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int8',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'float16',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'category',
        'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float16',
        'SmartScreen':                                          'category',
        'Firewall':                                             'float16',
        'UacLuaenable':                                         'category',
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float16',
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'category',
        'Census_ProcessorManufacturerIdentifier':               'category',
        'Census_ProcessorModelIdentifier':                      'float16',
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float32',
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float32',
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float32',
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'int16',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int16',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        'Census_IsFlightingInternal':                           'float16',
        'Census_IsFlightsDisabled':                             'float16',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float16',
        'Census_FirmwareVersionIdentifier':                     'float32',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'float16',
        'Census_IsVirtualDevice':                               'float16',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
        'Wdft_IsGamer':                                         'float16',
        'Wdft_RegionIdentifier':                                'float16',
        'HasDetections':                                        'int8'
        }
# Быстрая загрузка с оптимизацией
train_data = pd.read_csv(
    '/kaggle/input/microsoft-malware-prediction/train.csv',
    dtype=dtypes,
    usecols=lambda col: col not in ['MachineIdentifier']  # сразу исключаем ID
)
test_data = pd.read_csv('../input/test.csv', dtype=dtypes)



# Сэмплирование 50 % данных
sample_frac = 0.5
train_sample = train_data.sample(frac=sample_frac, random_state=42)



# Сохраняем размеры для последующего разделения
n_train = len(train_sample)
n_test = len(test_data)

# Объединяем датасеты
all_data = pd.concat([train_sample, test_data], ignore_index=True)

# Удаляем исходные датасеты, освобождаем память
del train_sample, test_data
gc.collect()



# Автоматически находим категориальные колонки
cat_cols = all_data.select_dtypes(include=['object', 'category']).columns.tolist()

# Кодируем через LabelEncoder (или используем pd.Categorical)
from sklearn.preprocessing import LabelEncoder

for col in cat_cols:
    le = LabelEncoder()
    # Заполняем пропуски перед кодированием
    all_data[col] = all_data[col].fillna('MISSING')
    all_data[col] = le.fit_transform(all_data[col].astype(str))



train_processed = all_data[:n_train]
test_processed = all_data[n_train:]

# Освобождаем память
del all_data
gc.collect()



import lightgbm as lgb

# Целевая переменная
y_train = train_processed['HasDetections']
X_train = train_processed.drop('HasDetections', axis=1)
X_test = test_processed

# Создаём LightGBM датасет
train_lgb = lgb.Dataset(X_train, label=y_train)



params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'nthread': 4,
    'learning_rate': 0.05,
    'max_depth': 5,
    'num_leaves': 40,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1
}

# Обучение
model = lgb.train(
    params,
    train_lgb,
    num_boost_round=1000,
    verbose_eval=100
)



# Предсказания
preds = model.predict(X_test)

# Сабмишен
submission = pd.DataFrame({
    'MachineIdentifier': test_data['MachineIdentifier'],  # восстанавливаем ID
    'HasDetections': preds
})
submission.to_csv('submission_lgbm.csv', index=False)


