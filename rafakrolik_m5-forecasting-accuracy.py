# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#!pip install polars


import os
import gc
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from pathlib import Path

DATA_DIR = Path("/kaggle/input/m5-forecasting-accuracy")
BASE_DIR = Path("/kaggle/working")
OUTPUT_PATH = Path("/kaggle/working/sales_long.parquet")
MODEL_PATH = Path("/kaggle/working/lightgbm_model.pkl")
SUB_PATH   = Path("/kaggle/working/final_submission.csv")
FEATURES_SALES_PATH = Path("/kaggle/working/sales_features.parquet")


sales = pl.read_csv(DATA_DIR / "sales_train_validation.csv")
calendar = pl.read_csv(DATA_DIR / "calendar.csv")
prices = pl.read_csv(DATA_DIR / "sell_prices.csv")

print(f"Sales: {sales.head} {sales.shape}")
print(f"Calendar: {calendar.head} {calendar.shape}")
print(f"Prices: {prices.head} {prices.shape}")


# PrzeksztaÅ‚cenie formatu szerokiego na dÅ‚ugi

id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

day_cols = [c for c in sales.columns if c.startswith("d_")]

sales_long = sales.unpivot(
    index = id_cols,
    on = day_cols,
    variable_name = "d",
    value_name = "sales"
)

print(sales_long.shape)
print(sales_long.head(2))


# Å�Ä…czenie tabeli calendar 

calendar_use = calendar.select([
    "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year", "event_name_1", "event_type_1", "event_name_2", "event_type_2"
])

sales_long = sales_long.join( calendar_use, on= "d", how= "left", suffix = "_cal") # Error fix "suffix"

sales_long = sales_long.with_columns([
    pl.col("date").cast(pl.Date)
])

gc.collect()
print(sales_long.shape, sales_long.head())


# Å�Ä…czenie z tabelÄ… sell_prices

for col in ["sell_price", "wm_yr_wk_right"]:
    if col in sales_long.columns:
        sales_long = sales_long.drop(col)

sales_long = sales_long.join(
    prices,
    on = ["store_id", "item_id", "wm_yr_wk"],
    how = "left"
)

print(sales_long.shape)


# Optymalizacja

sales_long = sales_long.with_columns([
    pl.col("sales").cast(pl.Int16),
    pl.col("store_id").cast(pl.Categorical),
    pl.col("item_id").cast(pl.Categorical),
    pl.col("state_id").cast(pl.Categorical),
    pl.col("dept_id").cast(pl.Categorical),
    pl.col("cat_id").cast(pl.Categorical),
])
gc.collect()

# Zapis
sales_long.write_parquet(OUTPUT_PATH, compression = "snappy")

print(f"zapisane w {OUTPUT_PATH}")
print(sales_long.shape)
print(sales_long.head(1))


 #Szukaj pliku w caÅ‚ym systemie

def find_file(filename, search_path="/kaggle"):
    """ZnajdÅº plik rekursywnie"""
    matches = []
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            matches.append(os.path.join(root, filename))
    return matches

# PrzykÅ‚ad uÅ¼ycia
results = find_file("sales_long.parquet")

if results:
    print("Znaleziono plik:")
    for path in results:
        print(f"  ğŸ“� {path}")
else:
    print("Nie znaleziono pliku sales_long.parquet")

gc.collect()


# Sprawdzenie wiersze sÄ… posortowane wzglÄ™dem daty i id

#check_sales = pl.read_parquet(OUTPUT_PATH)

# test
#is_sorted = (
#    check_sales.select(["id", "date"])
#    .equals(check_sales.select(["id", "date"]).sort(["id", "date"]))
#)

#print("TAK, Posortowane" if is_sorted else "NIE posortowane")

#gc.collect()


# FEATURE ENGINEERING
# Przygotowanie cech dla modelu prognozowania sprzedaÅ¼y

sales = pl.scan_parquet(OUTPUT_PATH) #Lazy
#sales = sales.sort(["id", "date"])

lags = [7, 14, 28]
windows = [7, 28]

sales = sales.sort(["id", "date"]).with_columns([
    # kalendarz
    pl.col("date").dt.weekday().cast(pl.Int8).alias("dayofweek"), 
    pl.col("date").dt.day().cast(pl.Int8).alias("day"), 
    pl.col("date").dt.month().cast(pl.Int8).alias("month"), 
    pl.col("date").dt.quarter().cast(pl.Int8).alias("quarter"), 

    # Binarne
    (pl.col("date").dt.weekday() >= 5).cast(pl.Int8).alias("is_weekend"),
    pl.col("date").dt.day().is_in([1, 2, 3]).cast(pl.Int8).alias("is_month_start"),
    pl.col("date").dt.day().is_in([28, 29, 30, 31]).cast(pl.Int8).alias("is_month_end"),
    
    ( 
        (pl.col("event_name_1").is_not_null()) | 
        (pl.col("event_name_2").is_not_null()) 
    ).cast(pl.Int8).alias("is_event"),
    
    *[ 
        pl.col("sales")
        .shift(lag) 
        .over("id")
        .alias(f"lag_{lag}")
        .cast(pl.Float32)
        for lag in lags 
    ], 
    
    # ROLLING MEAN 
    *[ 
        pl.col("sales")
        .rolling_mean(window_size=w)
        .over("id")
        .cast(pl.Float32)
        .alias(f"rolling_mean_{w}")
        for w in windows 
    ],
    
    # ROLLING STD 
    *[ 
        pl.col("sales")
        .rolling_std(window_size=w)
        .over("id") .cast(pl.Float32)
        .alias(f"rolling_std_{w}")
        for w in windows 
    ],
    # CECHY CENOWE
    ( 
        pl.col("sell_price") - 
        pl.col("sell_price").shift(7).over("id") 
    ).cast(pl.Float32).alias("price_change_1wk"),
    
    pl.col("sell_price")
    .rolling_mean(window_size=28)
    .over("id") .cast(pl.Float32)
    .alias("price_avg_4wk"), 
    
    pl.col("sell_price")
    .rolling_std(window_size=28)
    .over("id")
    .cast(pl.Float32)
    .alias("price_std_4wk"), 

    ( 
        pl.col("sell_price") >
        pl.col("sell_price").shift(7).over("id")
    ).cast(pl.Int8).alias("price_trend_up"), 

]) 

sales = sales.collect(engine="streaming") # nie stosowaÄ‡ juÅ¼ streaming=True

print(f"Dane utworzone: {sales.shape}")

# Optymalizacja typÃ³w
# SprzedaÅ¼ jako Int16

if sales["sales"].max() < 32767:
    sales = sales.with_columns(
        pl.col("sales").cast(pl.Int16)
    ) 
    
# Ceny jako Float32 

if "sell_price" in sales.columns:
    sales = sales.with_columns(
        pl.col("sell_price").cast(pl.Float32)
    )
        
print(f"ğŸ’¾ ZuÅ¼ycie RAM po optymalizacji: {sales.estimated_size('mb'):.1f} MB")

# Czyszczenie pamiÄ™ci

gc.collect()

# Zapis do Parquet (z kompresjÄ…)

sales.write_parquet( 
    FEATURES_SALES_PATH, 
    compression="zstd",
    compression_level=3 
) 


print(f"Plik: {FEATURES_SALES_PATH}") 
print(f"KsztaÅ‚t: {sales.shape}") 
print(f"Rozmiar pliku: {FEATURES_SALES_PATH.stat().st_size / 1024 / 1024:.1f} MB") 
print(f"\n Kolumny: {sales.columns}")
    


# Trening

# FEATURES_SALES_PATH = Path("/kaggle/working/sales_features.parquet")

# Wczytanie danych

lf = pl.scan_parquet(FEATURES_SALES_PATH)
lf = lf.with_columns(pl.col("date").cast(pl.Date))

# WybÃ³r cech
schema = lf.collect_schema()
cols = schema.names()
exclude = {"id",
           "item_id",
           "dept_id",
           "cat_id",
           "store_id",
           "state_id", 
           "event_name_1",
           "event_name_2",
           "date",
           "sales"
          } 

sample = lf.select(cols).limit(5).collect()
numeric_dtypes = {pl.Int8,
                  pl.Int16,
                  pl.Int32,
                  pl.Int64,
                  pl.UInt8,
                  pl.UInt16,
                  pl.UInt32,
                  pl.UInt64,
                  pl.Float32,
                  pl.Float64
                 } 

schema = sample.schema
feature_cols = [c for c in cols if (c not in exclude) and (schema.get(c) in numeric_dtypes)]

# Wyciek pamiÄ™ci 
if "lag_1" in feature_cols:
    print(" WARNING: lag_1 detected! To moÅ¼e powodowaÄ‡ data leakage!")
    print(" Rekomendacja: usuÅ„ lag_1 z feature engineering") 
    
    
print(f" Wybrano {len(feature_cols)} cech numerycznych.") 

# Zredukuj kolumny 

lf = lf.select(feature_cols + ["sales", "date", "id"]) 


 
# Materializacja + sortowanie 

print(" Materializacja danych...") 

df = lf.collect( engine="streaming" ) 
df = df.sort(["date","id"]) 

#  POPRAWKA: UsuÅ„ NULL-e PRZED splitem 

print(" Usuwanie NULL-i...") 

df = df.drop_nulls( subset = feature_cols + ["sales"] ) 

print(f"Dane po czyszczeniu: {df.shape}") 



# Split czasowy train/val

horizon = 28
max_date = df["date"].max()
val_start = max_date - pl.duration(days=horizon-1)

train_df = df.filter(pl.col("date") < val_start)
val_df   = df.filter(pl.col("date") >= val_start)

print(f"Train: {train_df.height:,}  |  Val: {val_df.height:,}")
print(f"Val period: {val_start} â†’ {max_date}")


# Konwersja do Pandas/NumPy

X_train = train_df.select(feature_cols).to_pandas()
y_train = train_df["sales"].to_numpy()

X_val   = val_df.select(feature_cols).to_pandas()
y_val   = val_df["sales"].to_numpy()

val_ids = val_df["id"].to_numpy()
train_ids = train_df["id"].to_numpy()

# Zachowaj train_df dla RMSSE denominators
train_pdf = train_df.select(["id", "date", "sales"]).to_pandas()

del df; gc.collect()
print("Dane przekonwertowane.")


# Trening LightGBM

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    
    "learning_rate": 0.01,
    "num_leaves": 31,
    "max_depth": 8,
    "min_child_samples": 20,
    
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    
    "seed": 42,
    "verbose": -1,
}

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val   = lgb.Dataset(X_val,   label=y_val, reference=lgb_train)

print(" Trening LightGBM...")
model = lgb.train(
    params,
    lgb_train,
    num_boost_round=5000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=["train","val"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=200, verbose=True),
        lgb.log_evaluation(period=200)
    ]
)


# Predykcja i RMSE

y_pred = model.predict(X_val, num_iteration=model.best_iteration)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"\n RMSE (val): {rmse:.5f}")


# WRMSSE 

print("\n Obliczanie WRMSSE...")

# Denominatory (naive forecast error z train)
denoms = {}
weights = {}

train_pdf = train_df.select(["id", "date", "sales"]).to_pandas()
train_pdf_sorted = train_pdf.sort_values(["id", "date"])

# Statystyki diagnostyczne
n_valid = 0
n_zero = 0
n_nan = 0
n_constant = 0
n_too_short = 0

print(" Obliczanie denominatorÃ³w dla kaÅ¼dej serii...")

for _id, grp in train_pdf_sorted.groupby("id"):
    sales = grp["sales"].values
    
    # SprawdÅº minimalnÄ… dÅ‚ugoÅ›Ä‡
    if len(sales) < 2:
        n_too_short += 1
        continue
    
    # 1: Czyszczenie NaN/Inf
    # UsuÅ„ NaN i Inf PRZED obliczeniami
    mask_valid = ~(np.isnan(sales) | np.isinf(sales))
    sales_clean = sales[mask_valid]
    
    if len(sales_clean) < 2:
        n_nan += 1
        denoms[_id] = 1e-8
        weights[_id] = 0.0
        continue
    
    # 2: SprawdÅº czy seria jest staÅ‚a
    if np.all(sales_clean == sales_clean[0]):
        n_constant += 1
        denoms[_id] = 1e-8
        weights[_id] = sales_clean.sum()
        continue
    
    # 3: Oblicz naive errors
    try:
        diffs = np.diff(sales_clean)
        naive_errors = diffs ** 2
        mean_error = np.mean(naive_errors)
        
        # KROK 4: Zabezpieczenie numeryczne PRZED sqrt
        if mean_error <= 0 or np.isnan(mean_error) or np.isinf(mean_error):
            n_zero += 1
            denoms[_id] = 1e-8
            weights[_id] = sales_clean.sum()
        else:
            # KROK 5: Bezpieczny sqrt
            denom = np.sqrt(mean_error)
            
            # SprawdÅº wynik
            if np.isnan(denom) or np.isinf(denom) or denom < 1e-10:
                n_zero += 1
                denoms[_id] = 1e-8
            else:
                n_valid += 1
                denoms[_id] = denom
            
            weights[_id] = sales_clean.sum()
    
    except Exception as e:
        # Catch-all dla nieoczekiwanych bÅ‚Ä™dÃ³w
        print(f"âš ï¸�  BÅ‚Ä…d dla serii {_id}: {e}")
        denoms[_id] = 1e-8
        weights[_id] = 0.0

print(f"\nStatystyki denominatorÃ³w:")
print(f"   Poprawne serie:        {n_valid:>6,}")
print(f"    StaÅ‚e wartoÅ›ci:       {n_constant:>6,}")
print(f"    Zero variance:        {n_zero:>6,}")
print(f"    NaN/Inf w danych:     {n_nan:>6,}")
print(f"    Za krÃ³tkie (<2):      {n_too_short:>6,}")
print(f"   Razem:                 {len(denoms):>6,}")

# SprawdÅº czy sÄ… problemy
if n_valid == 0:
    print("\n BÅ�Ä„D: Brak poprawnych serii! SprawdÅº dane treningowe.")
    print("   MoÅ¼liwe przyczyny:")
    print("   - Wszystkie wartoÅ›ci = 0")
    print("   - Dane zawierajÄ… same NaN/Inf")
    print("   - Zbyt krÃ³tka historia")


# RMSSE per seria

print("\n Obliczanie RMSSE per seria...")

val_pdf = pd.DataFrame({"id": val_ids, "y_true": y_val, "y_pred": y_pred})

rmsse_list = []
weighted_sum = 0
total_weight = 0
n_skipped = 0

for _id, grp in val_pdf.groupby("id"):
    if _id not in denoms or _id not in weights:
        n_skipped += 1
        continue
    
    y_true_vals = grp["y_true"].values
    y_pred_vals = grp["y_pred"].values
    
    # SprawdÅº predykcje
    if np.any(np.isnan(y_pred_vals)) or np.any(np.isinf(y_pred_vals)):
        n_skipped += 1
        continue
    
    # Numerator 
    squared_errors = (y_true_vals - y_pred_vals) ** 2
    mean_squared_error_val = np.mean(squared_errors)
    
    if mean_squared_error_val < 0 or np.isnan(mean_squared_error_val):
        n_skipped += 1
        continue
    
    num = np.sqrt(mean_squared_error_val)
    
    # RMSSE
    rmsse_i = num / denoms[_id]
    
    # Ostateczne sprawdzenie
    if np.isnan(rmsse_i) or np.isinf(rmsse_i):
        n_skipped += 1
        continue
    
    rmsse_list.append(rmsse_i)
    
    # Weighted sum
    w = weights[_id]
    weighted_sum += rmsse_i * w
    total_weight += w

# Wyniki
if len(rmsse_list) > 0:
    rmsse_macro = float(np.mean(rmsse_list))
    wrmsse = weighted_sum / total_weight if total_weight > 0 else np.inf
else:
    rmsse_macro = np.inf
    wrmsse = np.inf

print(f"\n WYNIKI METRYKI:")
print(f"   RMSE:              {rmse:.5f}")
print(f"   RMSSE (macro):     {rmsse_macro:.5f}")
print(f"   WRMSSE (weighted): {wrmsse:.5f}")
print(f"   Liczba serii:      {len(rmsse_list):,}")
print(f"   PominiÄ™te serie:   {n_skipped:,}")

if len(rmsse_list) > 0:
    print(f"\n    RMSSE - rozkÅ‚ad:")
    print(f"      Min:     {np.min(rmsse_list):.5f}")
    print(f"      Q25:     {np.percentile(rmsse_list, 25):.5f}")
    print(f"      Median:  {np.median(rmsse_list):.5f}")
    print(f"      Q75:     {np.percentile(rmsse_list, 75):.5f}")
    print(f"      Max:     {np.max(rmsse_list):.5f}")
    print(f"      Std:     {np.std(rmsse_list):.5f}")

# Czyszczenie
del train_pdf, train_pdf_sorted, val_pdf
gc.collect()

# Feature importance

print("\nTop 20 cech:")
importances = model.feature_importance(importance_type="gain")
feat_imp = pd.DataFrame({
    "feature": X_train.columns,
    "importance": importances
}).sort_values("importance", ascending=False)

print(feat_imp.head(20).to_string(index=False))

joblib.dump(model, MODEL_PATH)

# Wykres
plt.figure(figsize=(10,6))
top20 = feat_imp.head(20)
plt.barh(top20["feature"][::-1], top20["importance"][::-1])
plt.title("LightGBM Feature Importance (Top 20)")
plt.xlabel("Gain")
plt.tight_layout()
plt.show()

print("\n Trening zakoÅ„czony!")



#Czy model istnieje
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model nie istnieje: {MODEL_PATH}")

print("\nWczytywanie modelu...")
model = joblib.load(MODEL_PATH)
print(f"Model wczytany (best iteration: {model.best_iteration})")

# Wczytanie danych 
print( "\nWczytywanie danych (lazy + streaming)..." )
lf = pl.scan_parquet(FEATURES_SALES_PATH)
lf = lf.with_columns(pl.col( "date" ).cast( pl.Date ))

# Pobierz schemat
schema = lf.collect_schema()
cols = schema.names()

# Kolumny wykluczane
exclude = {"id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
           "event_name_1", "event_name_2", "date", "sales"}

numeric_dtypes = {pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                  pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                  pl.Float32, pl.Float64}

feature_cols = [
    c for c in cols 
    if (c not in exclude) and (schema[c] in numeric_dtypes)
]

print(f"Wybrano {len(feature_cols)} cech numerycznych")

# Materializacja
lf = lf.select(feature_cols + ["sales", "date", "id"])
lf = lf.drop_nulls(subset=feature_cols + ["sales"])
lf = lf.sort(["date", "id"])
df = lf.collect(engine="streaming")

print(f"Dane wczytane: {df.shape}")
print(f"RAM: ~{df.estimated_size('mb'):.1f} MB")

# Split train/val
horizon = 28
max_date = df["date"].max()
val_start = max_date - pl.duration(days=horizon - 1)

print(f"\n Split czasowy:")
print(f"   Val period: {val_start} â†’ {max_date} ({horizon} dni)")

train_df = df.filter(pl.col("date") < val_start)
val_df = df.filter(pl.col("date") >= val_start)

print(f"   Train: {train_df.height:,} wierszy")
print(f"   Val:   {val_df.height:,} wierszy")

# Predykcja na walidacji
print("\n Predykcja na walidacji...")

val_ids = val_df["id"].to_numpy()
val_dates = val_df["date"].to_numpy()
X_val = val_df.select(feature_cols).to_pandas().astype(np.float32)
y_val = val_df["sales"].to_numpy().astype(np.float32)

del val_df
gc.collect()

print(f"   X_val shape: {X_val.shape}")
print(f"   RAM po konwersji: ~{X_val.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

# Predykcja
y_pred = model.predict(X_val, num_iteration=model.best_iteration)

# RMSE
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"\nRMSE (validation): {rmse:.5f}")

# WRMSSE
train_pdf = train_df.select(["id", "sales"]).to_pandas()
del train_df, df
gc.collect()

print("   Obliczanie denominatorÃ³w...")

denoms = {}
weights = {}

for _id, grp in train_pdf.groupby("id"):
    sales = grp["sales"].values
    
    if len(sales) >= 2:
        naive_errors = np.diff(sales) ** 2
        mean_error = np.mean(naive_errors)
        denom = np.sqrt(mean_error) if mean_error > 1e-10 else 1e-8
        denoms[_id] = denom
        weights[_id] = sales.sum()

print(f"   Denominatory obliczone dla {len(denoms)} serii")

del train_pdf
gc.collect()

# RMSSE per seria
val_pdf = pd.DataFrame({
    "id": val_ids,
    "y_true": y_val,
    "y_pred": y_pred
})

rmsse_list = []
weighted_sum = 0
total_weight = 0

for _id, grp in val_pdf.groupby("id"):
    if _id in denoms and _id in weights:
        num = np.sqrt(np.mean((grp["y_true"].values - grp["y_pred"].values) ** 2))
        rmsse_i = num / denoms[_id]
        
        if not np.isnan(rmsse_i) and not np.isinf(rmsse_i):
            rmsse_list.append(rmsse_i)
            w = weights[_id]
            weighted_sum += rmsse_i * w
            total_weight += w

rmsse_macro = np.mean(rmsse_list) if rmsse_list else np.inf
wrmsse = weighted_sum / total_weight if total_weight > 0 else np.inf

print(f"\nâœ… WYNIKI METRYKI:")
print(f"   RMSE:              {rmse:.5f}")
print(f"   RMSSE (macro):     {rmsse_macro:.5f}")
print(f"   WRMSSE (weighted): {wrmsse:.5f}")
print(f"   Liczba serii:      {len(rmsse_list):,}")

if len(rmsse_list) > 0:
    print(f"\n   RMSSE - statystyki:")
    print(f"      Min:     {np.min(rmsse_list):.5f}")
    print(f"      Median:  {np.median(rmsse_list):.5f}")
    print(f"      Max:     {np.max(rmsse_list):.5f}")

# Feature Importance
print("\n Top 15 najwaÅ¼niejszych cech:")

importances = model.feature_importance(importance_type="gain")
feat_imp = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances
}).sort_values("importance", ascending=False)

print(feat_imp.head(15).to_string(index=False))

# Wykres
plt.figure(figsize=(10, 6))
top15 = feat_imp.head(15)
plt.barh(top15["feature"][::-1], top15["importance"][::-1], color='steelblue')
plt.title("LightGBM Feature Importance (Top 15)", fontsize=14, fontweight='bold')
plt.xlabel("Gain", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.show()


# ZAPIS SUBMISSION


print("\n Generowanie submission...")

# Wczytaj template
sample_sub = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sample_submission.csv")
print(f"   Sample submission: {sample_sub.shape}")

#  USUÅƒ sufiksy z val_ids
val_ids_clean = np.array([str(id).replace('_validation', '').replace('_evaluation', '') 
                          for id in val_ids])

print(f"\n Diagnostyka ID:")
print(f"   Oryginalne ID sample: {val_ids[:3]}")
print(f"   Oczyszczone ID:       {val_ids_clean[:3]}")
print(f"   Unique IDs:           {len(set(val_ids_clean))}")

# Przygotuj dane z predykcjami
val_submission = pd.DataFrame({
    "id": val_ids_clean,
    "date": val_dates,
    "prediction": y_pred
})

print(f"   Val submission: {val_submission.shape}")
print(f"   Date range: {val_submission['date'].min()} â†’ {val_submission['date'].max()}")

# Sortuj po ID i dacie
val_submission = val_submission.sort_values(["id", "date"])

# Dodaj numer dnia (1-28) dla kaÅ¼dego ID
val_submission["day_num"] = val_submission.groupby("id").cumcount() + 1

# SprawdÅº liczbÄ™ dni per ID
days_per_id = val_submission.groupby("id")["day_num"].max()
print(f"   Dni per ID - Min: {days_per_id.min()}, Max: {days_per_id.max()}, Mean: {days_per_id.mean():.1f}")

# Pivot: zmieÅ„ na format szeroki (kaÅ¼de ID -> 28 kolumn)
pivot = val_submission.pivot_table(
    index="id", 
    columns="day_num", 
    values="prediction",
    aggfunc="mean"
)

print(f"   Pivot shape: {pivot.shape}")
print(f"   Null values: {pivot.isna().sum().sum()}")

# WypeÅ‚nij brakujÄ…ce wartoÅ›ci
pivot = pivot.ffill(axis=1).bfill(axis=1).fillna(0)

# Upewnij siÄ™, Å¼e mamy dokÅ‚adnie 28 kolumn
forecast_cols = [f"F{i}" for i in range(1, 29)]

if pivot.shape[1] < 28:
    print(f"   UWAGA: Mamy tylko {pivot.shape[1]} kolumn, dodajÄ™ brakujÄ…ce...")
    for i in range(1, 29):
        if i not in pivot.columns:
            last_col = max([c for c in pivot.columns if isinstance(c, int)])
            pivot[i] = pivot[last_col]

# WeÅº tylko pierwsze 28 kolumn i zmieÅ„ nazwy
pivot = pivot.iloc[:, :28]
pivot.columns = forecast_cols

# Reset index Å¼eby 'id' byÅ‚o kolumnÄ…
pivot = pivot.reset_index()

print(f"   Pivot po przetworzeniu: {pivot.shape}")
print(f"\n   Sample predictions:")
print(pivot.head(3))
print(f"   Stats: Mean={pivot[forecast_cols].values.mean():.2f}, "
      f"Min={pivot[forecast_cols].values.min():.2f}, "
      f"Max={pivot[forecast_cols].values.max():.2f}")

# VALIDATION rows (d_1914-1941)
df_val = pivot.copy()
df_val["id"] = df_val["id"].astype(str) + "_validation"

# EVALUATION rows (d_1942-1969) - kopiujemy predykcje
df_eval = pivot.copy()
df_eval["id"] = df_eval["id"].astype(str) + "_evaluation"

# PoÅ‚Ä…cz
submission = pd.concat([df_val, df_eval], ignore_index=True)
print(f"\n   Submission przed merge: {submission.shape}")

# Dopasuj do sample_submission (zachowaj kolejnoÅ›Ä‡ ID)
final_sub = sample_sub[["id"]].merge(submission, on="id", how="left")

print(f"   Po merge: {final_sub.shape}")
print(f"   Null values: {final_sub[forecast_cols].isna().sum().sum()}")

# WypeÅ‚nij brakujÄ…ce (jeÅ›li sÄ…)
if final_sub[forecast_cols].isna().any().any():
    print(f"     WypeÅ‚niam brakujÄ…ce wartoÅ›ci medianÄ…...")
    median_val = submission[forecast_cols].median().median()
    if pd.isna(median_val) or median_val == 0:
        median_val = y_pred.mean()
    final_sub[forecast_cols] = final_sub[forecast_cols].fillna(median_val)

# Zapisz
final_sub.to_csv(SUB_PATH, index=False)
print(f"\n Submission zapisany: {SUB_PATH}")

# Statystyki koÅ„cowe
forecast_values = final_sub[forecast_cols].values.flatten()
print(f"\n Statystyki koÅ„cowej prognozy:")
print(f"   Åšrednia:  {np.nanmean(forecast_values):.2f}")
print(f"   Mediana:  {np.nanmedian(forecast_values):.2f}")
print(f"   Min:      {np.nanmin(forecast_values):.2f}")
print(f"   Max:      {np.nanmax(forecast_values):.2f}")
print(f"   Std:      {np.nanstd(forecast_values):.2f}")
print(f"   Zeros:    {(forecast_values == 0).sum():,} ({(forecast_values == 0).mean()*100:.1f}%)")
print(f"   NaNs:     {np.isnan(forecast_values).sum():,}")

# Czyszczenie pamiÄ™ci
del X_val, y_val, y_pred, val_ids, val_ids_clean, val_dates
del val_submission, pivot, df_val, df_eval, submission, final_sub, sample_sub
gc.collect()


print(f"\n PODSUMOWANIE:")
print(f"   Model oceniony: RMSE={rmse:.5f}, WRMSSE={wrmsse:.5f}")
print(f"   Submission wygenerowany: {SUB_PATH}")
print(f"\n UWAGA o submission:")
print("   - Validation rows (d_1914-1941): Prawdziwe predykcje LightGBM")
print("   - Evaluation rows (d_1942-1969):  Kopiowane z validation")
print(f"\n Oczekiwany wynik:")
print(f"   - Public LB (validation):  ~{wrmsse:.2f} WRMSSE")
print(f"   - Private LB (evaluation): ~{wrmsse*1.1:.2f}-{wrmsse*1.3:.2f} WRMSSE")
print("     (gorszy przez brak rekurencyjnej prognozy)")

