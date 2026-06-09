import pandas as pd
import numpy as np
import torch
from fastai.tabular.all import *
import random
import os
import fastai
from sklearn.model_selection import StratifiedKFold

# Periksa versi fastai
print(f"fastai version: {fastai.__version__}")
if float(fastai.__version__.split('.')[0]) < 2:
    raise ImportError("fastai version < 2.0 detected. Please update with: pip install -U fastai")

# Fungsi untuk seeding
def seed_everything(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed, True)

# Set seed
seed_everything(42)

# Fungsi SMAPE
def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error using PyTorch."""
    y_true = torch.tensor(y_true) if not isinstance(y_true, torch.Tensor) else y_true
    y_pred = torch.tensor(y_pred) if not isinstance(y_pred, torch.Tensor) else y_pred
    denom = (torch.abs(y_true) + torch.abs(y_pred)) / 2
    diff = torch.abs(y_true - y_pred)
    return torch.mean(diff / denom) * 100

def smape_metric(preds, targs):
    """SMAPE metric for fastai, handling log-transformed target."""
    y_true = torch.expm1(targs)
    y_pred = torch.expm1(preds)
    return smape(y_true, y_pred)

# Fungsi untuk menambahkan fitur siklik
def add_cyclic_features(df, col, max_val):
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df

# Fungsi feature engineering
def extract_plate_features(df):
    df = df.copy()
    df['region'] = df['plate'].str.extract(r"(\d{2,3})$")
    df['letters'] = df['plate'].str.findall(r"[A-Z]").str.join('')
    df['numbers'] = df['plate'].str.extract(r"(\d{3})")
    return df

def extract_datetime_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['days_since_start'] = np.log1p((df['date'] - df['date'].min()).dt.days)
    df['month_year_interaction'] = df['month'].astype(str) + '_' + df['year'].astype(str)
    return df

def create_advanced_features(df):
    df = df.copy()
    df['is_palindrome_letters'] = df['letters'].apply(lambda x: x == x[::-1])
    df['is_palindrome_numbers'] = df['numbers'].apply(lambda x: x == x[::-1])
    df['unique_letters_count'] = df['letters'].apply(lambda x: len(set(x)))
    df['unique_numbers_count'] = df['numbers'].apply(lambda x: len(set(x)))
    digits = df['numbers'].apply(lambda s: list(map(int, s)))
    df[['d1', 'd2', 'd3']] = pd.DataFrame(digits.tolist(), index=df.index)
    df['sum_numbers'] = np.log1p(np.clip(df['d1'] + df['d2'] + df['d3'], 0, None))
    df['product_numbers'] = np.log1p(np.clip(df['d1'] * df['d2'] * df['d3'], 0, None))
    df['is_BOP'] = (df['letters'] == 'BOP')
    df['is_XAM'] = (df['letters'] == 'XAM')
    df['numbers_int'] = np.log1p(df['numbers'].astype(int))
    df['is_special_combination'] = df['letters'].isin(['BOP', 'XAM', 'AAA', 'MMM', 'BBB', 'CCC']) & (
        df['numbers'].isin(['000', '111', '222', '333', '444', '555', '666', '777', '888', '999', '123', '456', '789'])
    )
    df['has_repeated_numbers'] = df['numbers'].apply(lambda x: len(set(x)) == 1)
    df['number_pattern_score'] = df['numbers'].apply(
        lambda x: 1.0 if len(set(x)) == 1 else (0.75 if x == x[::-1] else (0.5 if x[0] == x[2] else (0.25 if x in ['123', '456', '789'] else 0.0)))
    )
    df['is_popular_letter_pattern'] = df['letters'].isin(['AAA', 'MMM', 'BBB', 'CCC', 'XXX', 'KKK', 'SSS'])
    df['is_rare_combination'] = df['letters'].map(df['letters'].value_counts()) < 5  # Kombinasi huruf langka
    return df

def apply_government_tags(df):
    try:
        from supplemental_russian import GOVERNMENT_CODES
    except ImportError:
        GOVERNMENT_CODES = {}
    def get_gov_tags(row):
        for (letters, num_range, region), (_, is_forbidden, has_advantage, level) in GOVERNMENT_CODES.items():
            if row['letters'] == letters and str(row['region']) == region:
                number = int(row['numbers'])
                if num_range[0] <= number <= num_range[1]:
                    return pd.Series([is_forbidden, has_advantage, level])
        return pd.Series([0, 0, 0])
    
    df = df.copy()
    df[['is_forbidden', 'has_advantage', 'significance_level']] = df.apply(get_gov_tags, axis=1)
    return df

def add_frequency_features(train_df, test_df, columns):
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    for col in columns:
        freq = train_df[col].value_counts().to_dict()
        train_df[f'{col}_freq'] = np.log1p(np.clip(train_df[col].map(freq), 0, None))
        test_df[f'{col}_freq'] = np.log1p(np.clip(test_df[col].map(freq).fillna(train_df[f'{col}_freq'].mean()), 0, None))
    
    return train_df, test_df

def add_price_stats_features(train_df, test_df, columns):
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    for col in columns:
        stats = train_df.groupby(col)['price'].agg(['std', 'median']).rename(columns={'std': f'{col}_price_std', 'median': f'{col}_price_median'})
        stats[f'{col}_price_std'] = stats[f'{col}_price_std'].fillna(stats[f'{col}_price_std'].mean())
        stats[f'{col}_price_std'] = np.clip(stats[f'{col}_price_std'], 0, stats[f'{col}_price_std'].quantile(0.95))
        train_df[f'{col}_price_std'] = np.log1p(train_df[col].map(stats[f'{col}_price_std']))
        train_df[f'{col}_price_median'] = train_df[col].map(stats[f'{col}_price_median'])
        test_df = test_df.merge(
            stats[[f'{col}_price_std', f'{col}_price_median']].reset_index(),
            on=col, how='left'
        )
        test_df[f'{col}_price_std'] = np.log1p(test_df[f'{col}_price_std'].fillna(stats[f'{col}_price_std'].mean()))
        test_df[f'{col}_price_median'] = test_df[f'{col}_price_median'].fillna(stats[f'{col}_price_median'].mean())
    
    return train_df, test_df

def add_high_value_region_feature(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    region_means = train_df.groupby('region')['price'].mean()
    high_value_regions = region_means[region_means > region_means.quantile(0.75)].index
    train_df['is_high_value_region'] = train_df['region'].isin(high_value_regions).astype(int)
    test_df['is_high_value_region'] = test_df['region'].isin(high_value_regions).astype(int)
    
    return train_df, test_df

def add_target_encoding(train_df, test_df, columns, smoothing=100):
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    for col in columns:
        global_mean = train_df['price'].mean()
        agg = train_df.groupby(col)['price'].agg(['mean', 'count'])
        agg[f'{col}_target_enc'] = (agg['mean'] * agg['count'] + global_mean * smoothing) / (agg['count'] + smoothing)
        
        train_df[f'{col}_target_enc'] = train_df[col].map(agg[f'{col}_target_enc'])
        test_df = test_df.merge(
            agg[[f'{col}_target_enc']].reset_index(),
            on=col, how='left'
        )
        test_df[f'{col}_target_enc'] = test_df[f'{col}_target_enc'].fillna(global_mean)
    
    return train_df, test_df

def handle_outliers(df):
    df = df.copy()
    lower_limit = df['price'].quantile(0.01)  # Lebih ketat
    upper_limit = df['price'].quantile(0.99)
    df['price'] = df['price'].clip(lower=lower_limit, upper=upper_limit)
    return df

def engineer_features(df):
    df = extract_plate_features(df)
    df = extract_datetime_features(df)
    df = create_advanced_features(df)
    df = apply_government_tags(df)
    return df

# Load data
train_df = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/train.csv")
test_df = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/test.csv")

# Handle outliers
train_df = handle_outliers(train_df)

# Feature engineering
train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

# Tambahkan fitur siklik
train_df = add_cyclic_features(train_df, 'month', 12)
train_df = add_cyclic_features(train_df, 'day', 31)
train_df = add_cyclic_features(train_df, 'hour', 24)
test_df = add_cyclic_features(test_df, 'month', 12)
test_df = add_cyclic_features(test_df, 'day', 31)
test_df = add_cyclic_features(test_df, 'hour', 24)

# Tambahkan fitur frekuensi, statistik harga, high-value region, dan target encoding
train_df, test_df = add_frequency_features(train_df, test_df, ['letters', 'numbers', 'region'])
train_df, test_df = add_price_stats_features(train_df, test_df, ['letters', 'numbers', 'region'])
train_df, test_df = add_high_value_region_feature(train_df, test_df)
train_df, test_df = add_target_encoding(train_df, test_df, ['letters', 'numbers', 'region'])

# Log-transform target
train_df['price'] = np.log1p(train_df['price'])

# Definisikan kolom kategorikal dan numerik
cat_cols = ['region', 'letters', 'numbers', 'is_palindrome_letters', 'is_palindrome_numbers', 
            'is_BOP', 'is_XAM', 'is_forbidden', 'has_advantage', 'is_special_combination',
            'month_year_interaction', 'has_repeated_numbers', 'is_high_value_region', 
            'is_popular_letter_pattern', 'is_rare_combination']
cont_cols = ['year', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos',
             'unique_letters_count', 'unique_numbers_count', 'd1', 'd2', 'd3', 
             'sum_numbers', 'product_numbers', 'numbers_int', 'significance_level',
             'days_since_start', 'letters_target_enc', 'numbers_target_enc', 'region_target_enc',
             'letters_freq', 'numbers_freq', 'region_freq', 'letters_price_std', 
             'numbers_price_std', 'region_price_std', 'letters_price_median', 
             'numbers_price_median', 'region_price_median', 'number_pattern_score']
dep_var = 'price'

# Stratified split berdasarkan region
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = next(skf.split(train_df, train_df['region']))
train_idx, valid_idx = splits[0], splits[1]

# Pra-pemrosesan dengan fastai
procs = [Categorify, FillMissing, Normalize]
dls = TabularPandas(
    train_df,
    procs=procs,
    cat_names=cat_cols,
    cont_names=cont_cols,
    y_names=dep_var,
    splits=(list(train_idx), list(valid_idx)),
    y_block=RegressionBlock(),
    inplace=False
).dataloaders(bs=32)

# Konfigurasi model
config = tabular_config(embed_p=0.1, ps=[0.3, 0.3, 0.3])  # Regularisasi lebih kuat
learn = tabular_learner(
    dls,
    metrics=[smape_metric],
    layers=[200, 100, 50],  # Arsitektur lebih sederhana
    config=config
)

# Cari learning rate
lr = learn.lr_find().valley

# Callback untuk reduce learning rate dan simpan model terbaik
cbs = [
    ReduceLROnPlateau(monitor='smape_metric', min_delta=0.01, patience=4, factor=0.2),
    SaveModelCallback(monitor='smape_metric', min_delta=0.01, fname='best_model', comp=np.less)
]

# Latih model
learn.fit_one_cycle(20, lr_max=lr/10, wd=0.1, cbs=cbs)  # Regularisasi lebih kuat

# Load model terbaik
learn.load('best_model', weights_only=True)

# Prediksi pada test set
test_dl = dls.test_dl(test_df)
preds, _ = learn.get_preds(dl=test_dl)
test_df['price'] = np.expm1(preds.numpy()).astype(int)
submission = test_df[['id', 'price']]
submission.to_csv('submission_fastai_smape_improved.csv', index=False)
print("Submission file created: submission_fastai_smape_improved.csv")

# Evaluasi SMAPE pada validation set
valid_preds, valid_targs = learn.get_preds()
smape_score = smape(np.expm1(valid_targs), np.expm1(valid_preds))
print(f"Validation SMAPE: {smape_score:.2f}%")

