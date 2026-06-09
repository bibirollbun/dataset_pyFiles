import pandas as pd

train_path = "/kaggle/input/air-toxicity-and-chronic-respiratory-diseases-us/train.csv"
df = pd.read_csv(train_path)


print("Shape (rows, cols):", df.shape)
df.head()


df.info()


df["Incidence"].describe()


df["Incidence"].hist(bins=30)


#Are all categories represented?
df["Age"].value_counts().sort_index()


#The relation between age and Incidence
df.groupby("Age")["Incidence"].mean()


df.groupby("Age")["Incidence"].mean().plot(kind="bar")


df["State.Name"].nunique()


state_incidence = df.groupby("State.Name")["Incidence"].mean().sort_values(ascending=False)
state_incidence


state_incidence.head(25).plot(kind="bar", figsize=(8,4))


df.groupby("Year")["Incidence"].mean()


df.groupby("Year")["Incidence"].mean().plot(figsize=(8,4))


df.duplicated().sum()


df.isnull().sum()


states_25 = sorted(df["State.Name"].unique())
years = sorted(df["Year"].unique())

print("Number of states:", len(states_25))
print("Year range:", min(years), "to", max(years))


import bq_helper
from bq_helper import BigQueryHelper

# create a helper object for our bigquery dataset
historical_air_quality = bq_helper.BigQueryHelper(active_project="bigquery-public-data", \
                                                  dataset_name='epa_historical_air_quality')


bq_assistant = BigQueryHelper('bigquery-public-data', 'epa_historical_air_quality')


historical_air_quality.list_tables()


historical_air_quality.head('co_daily_summary')


# 1. Ortak DeÄŸiÅŸkenler (Eyalet Listesi ve Tarih)
target_states = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 
    'Connecticut', 'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 
    'Idaho', 'Indiana', 'Iowa', 'Michigan', 'Mississippi', 'Montana', 
    'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 
    'Tennessee', 'Texas', 'Washington'
]

# SQL iÃ§in liste formatÄ±
states_tuple = "'" + "', '".join(target_states) + "'"

# 2. Genel Sorgu Fonksiyonu
def get_epa_pollutant_data(table_name, value_col_name):
    """
    Verilen tablo ismine gÃ¶re BigQuery'den optimize edilmiÅŸ veriyi Ã§eker.
    """
    QUERY = f"""
        SELECT 
            state_name as STATE, 
            date_local as DATE,
            -- Birimi bilgi olarak alalÄ±m (TekrarÄ± Ã¶nlemek iÃ§in ANY_VALUE)
            ANY_VALUE(units_of_measure) as UNIT, 
            -- OrtalamayÄ± alÄ±p istediÄŸimiz kolon ismini verelim
            AVG(arithmetic_mean) as {value_col_name}
        FROM 
            `bigquery-public-data.epa_historical_air_quality.{table_name}`
        WHERE 
            state_name IN ({states_tuple})
            AND date_local BETWEEN '1990-01-01' AND '2019-12-31'
        GROUP BY
            state_name, date_local
        ORDER BY
            date_local
    """
    
    print(f"Veri Ã§ekiliyor: {table_name} ...")
    return bq_assistant.query_to_pandas(QUERY)

# 3. Her Kirletici Ä°Ã§in Fonksiyonu Ã‡aÄŸÄ±rma

# NO2 (Azot Dioksit)
df_no2 = get_epa_pollutant_data('no2_daily_summary', 'NO2')

# PM10 (PartikÃ¼l Madde 10)
df_pm10 = get_epa_pollutant_data('pm10_daily_summary', 'PM10')

# PM2.5 (PartikÃ¼l Madde 2.5 - FRM yani Resmi Metod tablosunu kullanÄ±yoruz)
df_pm25 = get_epa_pollutant_data('pm25_frm_daily_summary', 'PM25')

# O3 (Ozon)
df_o3 = get_epa_pollutant_data('o3_daily_summary', 'O3')

# SO2 (KÃ¼kÃ¼rt Dioksit)
df_so2 = get_epa_pollutant_data('so2_daily_summary', 'SO2')

# co (KÃ¼kÃ¼rt Dioksit)
df_co = get_epa_pollutant_data('co_daily_summary', 'CO')

print("\n--- Ä°Å�LEM TAMAMLANDI ---")
print(f"df_no2 satÄ±r sayÄ±sÄ± : {len(df_no2)}")
print(f"df_pm10 satÄ±r sayÄ±sÄ±: {len(df_pm10)}")
print(f"df_pm25 satÄ±r sayÄ±sÄ±: {len(df_pm25)}")
print(f"df_o3 satÄ±r sayÄ±sÄ±  : {len(df_o3)}")
print(f"df_so2 satÄ±r sayÄ±sÄ± : {len(df_so2)}")
print(f"df_co satÄ±r sayÄ±sÄ± : {len(df_co)}")


# Sonucu kontrol et
print(df_no2.head())
print(df_o3.head())
print(df_pm25.head())
print(df_pm10.head())
print(df_so2.head())
print(df_co.head())


def get_annual_average(df, pollutant_col):
    """
    GÃ¼nlÃ¼k veriyi alÄ±r, yÄ±llÄ±k ortalamaya Ã§evirir.
    """
    # 1. Veri boÅŸ mu kontrol et (Hata almamak iÃ§in)
    if df is None or df.empty:
        return None

    # 2. Tarih formatÄ±nÄ± garantiye al ve YIL sÃ¼tunu oluÅŸtur
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['YEAR'] = df['DATE'].dt.year
    
    # 3. Eyalet ve YÄ±la gÃ¶re grupla, ilgili kirleticinin ortalamasÄ±nÄ± al
    df_annual = df.groupby(['STATE', 'YEAR'])[pollutant_col].mean().reset_index()
    
    # 4. Okunabilirlik iÃ§in yuvarla (2 basamak)
    df_annual[pollutant_col] = df_annual[pollutant_col].round(2)
    
    return df_annual

# --- FONKSÄ°YONU TÃœM DATAFRAME'LERE UYGULAMA ---

# df_CO, df_no2 vb. deÄŸiÅŸkenlerin daha Ã¶nce tanÄ±mlÄ± olduÄŸunu varsayÄ±yoruz
df_co_annual   = get_annual_average(df_co, 'CO')
df_no2_annual  = get_annual_average(df_no2, 'NO2')
df_pm10_annual = get_annual_average(df_pm10, 'PM10')
df_pm25_annual = get_annual_average(df_pm25, 'PM25')
df_o3_annual   = get_annual_average(df_o3, 'O3')
df_so2_annual  = get_annual_average(df_so2, 'SO2')

# Ã–rnek Kontrol
print("--- PM2.5 YÄ±llÄ±k Ortalama (Ä°lk 5 SatÄ±r) ---")
if df_pm25_annual is not None:
    print(df_pm25_annual.head())

print("\n--- Ä°Å�LEM TAMAMLANDI ---")
print(f"df_co satÄ±r sayÄ±sÄ± : {len(df_co_annual)}")
print(f"df_no2 satÄ±r sayÄ±sÄ±: {len(df_no2_annual)}")
print(f"df_pm10 satÄ±r sayÄ±sÄ±: {len(df_pm10_annual)}")
print(f"df_pm25 satÄ±r sayÄ±sÄ±  : {len(df_pm25_annual)}")
print(f"df_o3 satÄ±r sayÄ±sÄ± : {len(df_o3_annual)}")
print(f"df_so2 satÄ±r sayÄ±sÄ± : {len(df_so2_annual)}")



# Kontrol edilecek DataFrame listesi ve isimleri
pollutant_list = [
    ('CO (Karbonmonoksit)', df_co_annual),
    ('NO2 (Azot Dioksit)', df_no2_annual),
    ('PM10 (PartikÃ¼l Madde 10)', df_pm10_annual),
    ('PM2.5 (PartikÃ¼l Madde 2.5)', df_pm25_annual),
    ('O3 (Ozon)', df_o3_annual),
    ('SO2 (KÃ¼kÃ¼rt Dioksit)', df_so2_annual)
]

summary_data = []

for name, df in pollutant_list:
    if df is not None and not df.empty:
        min_year = df['YEAR'].min()
        max_year = df['YEAR'].max()
        unique_years = df['YEAR'].nunique() # KaÃ§ farklÄ± yÄ±l var?
        
        summary_data.append({
            'Kirletici TÃ¼rÃ¼': name,
            'BaÅŸlangÄ±Ã§ YÄ±lÄ±': int(min_year),
            'BitiÅŸ YÄ±lÄ±': int(max_year),
            'Toplam YÄ±l SayÄ±sÄ±': unique_years,
            'Eksik YÄ±l Var mÄ±?': 'Evet' if (max_year - min_year + 1) != unique_years else 'HayÄ±r'
        })
    else:
        summary_data.append({
            'Kirletici TÃ¼rÃ¼': name,
            'BaÅŸlangÄ±Ã§ YÄ±lÄ±': '-',
            'BitiÅŸ YÄ±lÄ±': '-',
            'Toplam YÄ±l SayÄ±sÄ±': 0,
            'Eksik YÄ±l Var mÄ±?': '-'
        })

# Tabloyu oluÅŸtur ve gÃ¶ster
df_summary_ranges = pd.DataFrame(summary_data)

print("--- VERÄ° SETÄ° YIL ARALIKLARI Ã–ZETÄ° ---")
print(df_summary_ranges.to_string(index=False))


target_set = set(target_states)

# 2. Analiz Edilecek YÄ±l AralÄ±ÄŸÄ± (Referans Zaman)
all_years = list(range(1990, 2020)) # 1990'dan 2019'a kadar

# 3. Dataframe SÃ¶zlÃ¼ÄŸÃ¼ (DÃ¶ngÃ¼ye sokmak iÃ§in)
pollutants = {
    'CO (Karbonmonoksit)': df_co_annual,
    'NO2 (Azot Dioksit)': df_no2_annual,
    'PM10 (PartikÃ¼l Madde 10)': df_pm10_annual,
    'PM2.5 (PartikÃ¼l Madde 2.5)': df_pm25_annual,
    'O3 (Ozon)': df_o3_annual,
    'SO2 (KÃ¼kÃ¼rt Dioksit)': df_so2_annual
}

# --- ANA ANALÄ°Z DÃ–NGÃœSÃœ ---
for name, df in pollutants.items():
    print(f"\n{'='*20} {name} ANALÄ°ZÄ° {'='*20}")
    
    if df is None or df.empty:
        print("â�Œ Veri Yok veya Ã‡ekilemedi.\n")
        continue

    # --- A. HÄ°Ã‡ OLMAYAN EYALETLERÄ° BULMA ---
    present_states = set(df['STATE'].unique())
    missing_states = target_set - present_states
    
    if missing_states:
        print(f"âš ï¸� HÄ°Ã‡ VERÄ°SÄ° OLMAYAN EYALETLER ({len(missing_states)} Adet):")
        print(f"   {', '.join(sorted(missing_states))}")
    else:
        print("âœ… Hedeflenen 25 eyaletin hepsinde istasyon var.")

    # --- B. EKSÄ°K YILLARI BULMA (GAP ANALYSIS) ---
    # Her eyalet iÃ§in mevcut yÄ±llarÄ± 1 ve 0 olarak tabloya dÃ¶kÃ¼yoruz
    pivot = pd.crosstab(df['STATE'], df['YEAR'])
    
    # Tabloyu 1990-2019 aralÄ±ÄŸÄ±na zorluyoruz (Ã–lÃ§Ã¼lmeyen yÄ±llarÄ± 0 yapar)
    pivot = pivot.reindex(columns=all_years, fill_value=0)
    
    # 0 olan yÄ±llarÄ± say (Eksik YÄ±l SayÄ±sÄ±)
    missing_counts = (pivot == 0).sum(axis=1)
    
    # Sadece eksiÄŸi olanlarÄ± filtrele
    gaps = missing_counts[missing_counts > 0]
    
    if not gaps.empty:
        print(f"\nğŸ“‰ YIL EKSÄ°Ä�Ä° OLAN EYALETLER (Toplam 30 yÄ±l Ã¼zerinden):")
        # En Ã§ok eksiÄŸi olan ilk 5 eyaleti gÃ¶ster
        print(gaps.sort_values(ascending=False).head(5).to_string())
        
        # Kritik UyarÄ±: EÄŸer ortalama eksik yÄ±l sayÄ±sÄ± Ã§ok yÃ¼ksekse
        avg_missing = gaps.mean()
        if avg_missing > 10:
            print(f"\n   â„¹ï¸� NOT: Bu kirletici iÃ§in ortalama {int(avg_missing)} yÄ±l eksik.")
            print("   (Muhtemelen Ã¶lÃ§Ã¼mler 1990'dan Ã§ok sonra baÅŸladÄ±, Ã¶rn: PM2.5)")
    else:
        print("\nâœ¨ Harika! Mevcut eyaletlerin hepsinde 30 yÄ±lÄ±n verisi tam.")

print("\n" + "="*60)


from functools import reduce

# 1. Eldeki DataFrame'leri ve Ä°simlerini Listeye AlalÄ±m
pollutant_list = [
    ('CO', df_co_annual),
    ('NO2', df_no2_annual),
    ('PM10', df_pm10_annual),
    ('PM2.5', df_pm25_annual),
    ('O3', df_o3_annual),
    ('SO2', df_so2_annual)
]

# BoÅŸ olmayanlarÄ± filtrele
valid_dfs = [(name, df) for name, df in pollutant_list if df is not None and not df.empty]

if not valid_dfs:
    print("HiÃ§ geÃ§erli veri Ã§erÃ§evesi yok!")
else:
    # 2. ORTAK ANAHTARLARI BULMA (Intersection)
    # Ä°lk DataFrame'in (State, Year) ikililerini alarak baÅŸlÄ±yoruz
    common_keys = set(zip(valid_dfs[0][1]['STATE'], valid_dfs[0][1]['YEAR']))

    # DiÄŸerleriyle kesiÅŸim kÃ¼mesini alÄ±yoruz
    for name, df in valid_dfs[1:]:
        current_keys = set(zip(df['STATE'], df['YEAR']))
        common_keys = common_keys.intersection(current_keys)

    print(f"âœ… TÃœM TABLOLARDA ORTAK OLAN SATIR SAYISI: {len(common_keys)}")
    print("-" * 60)

    # 3. SÄ°LÄ°NEN VERÄ° RAPORU (Hangi Tablodan Ne Gitti?)
    print("--- SÄ°LÄ°NEN VERÄ° Ã–ZETÄ° (Ortak OlmadÄ±ÄŸÄ± Ä°Ã§in Ã‡Ä±karÄ±lanlar) ---")
    
    for name, df in valid_dfs:
        # Bu tablodaki mevcut anahtarlar
        current_keys = set(zip(df['STATE'], df['YEAR']))
        # Ortak kÃ¼mede olmayanlar (Silinecekler)
        dropped_keys = current_keys - common_keys
        
        if dropped_keys:
            print(f"\nğŸ“Œ {name} Tablosundan Silinenler ({len(dropped_keys)} satÄ±r):")
            
            # Silinenleri DataFrame yapÄ±p Eyalet bazÄ±nda Ã¶zetleyelim
            dropped_df = pd.DataFrame(list(dropped_keys), columns=['STATE', 'YEAR'])
            
            # Her eyalet iÃ§in hangi yÄ±llarÄ±n silindiÄŸini grupla
            summary = dropped_df.groupby('STATE')['YEAR'].apply(list)
            
            # Ä°lk 5 eyaleti Ã¶rnek olarak gÃ¶ster (ekranÄ± doldurmamak iÃ§in)
            for state, years in summary.head(5).items():
                years.sort()
                # YÄ±llarÄ± "1990-1998" gibi aralÄ±k olarak gÃ¶stermek yerine min-max yazalÄ±m
                min_y, max_y = years[0], years[-1]
                if min_y == max_y:
                    print(f"   - {state}: {min_y}")
                else:
                    print(f"   - {state}: {min_y}-{max_y} ({len(years)} yÄ±l)")
            
            if len(summary) > 5:
                print(f"   ... ve {len(summary)-5} eyalet daha.")
        else:
            print(f"\nâœ¨ {name}: HiÃ§ veri kaybÄ± yok (Tam uyumlu).")

    # 4. BÄ°RLEÅ�TÄ°RME Ä°Å�LEMÄ° (INNER JOIN)
    # Sadece verileri (df'leri) bir listede topla
    dfs_to_merge = [item[1] for item in valid_dfs]
    
    # Inner Join ile birleÅŸtir
    df_final_common = reduce(lambda left, right: pd.merge(left, right, on=['STATE', 'YEAR'], how='inner'), dfs_to_merge)
    
    # DÃ¼zenleme: Eyalet ve YÄ±la gÃ¶re sÄ±rala
    df_final_common = df_final_common.sort_values(by=['STATE', 'YEAR']).reset_index(drop=True)

    print("\n" + "="*60)
    print("SONUÃ‡: BÄ°RLEÅ�TÄ°RÄ°LMÄ°Å� ORTAK VERÄ° SETÄ°")
    print("="*60)
    print(df_final_common.head(10))
    print(f"\nToplam SatÄ±r SayÄ±sÄ±: {len(df_final_common)}")
    print(f"Toplam SÃ¼tunlar: {df_final_common.columns.tolist()}")


if 'df_final_common' in locals() and not df_final_common.empty:
    # 1. Temel Ä°statistikler
    min_year = int(df_final_common['YEAR'].min())
    max_year = int(df_final_common['YEAR'].max())
    total_years = df_final_common['YEAR'].nunique()
    
    # 2. Teorik olarak olmasÄ± gereken yÄ±l sayÄ±sÄ± (AralÄ±k farkÄ±)
    expected_duration = max_year - min_year + 1
    
    print("-" * 45)
    print("ğŸ“Š SONUÃ‡ TABLOSU (df_final_common) TARÄ°H ARALIÄ�I")
    print("-" * 45)
    print(f"ğŸ“… BaÅŸlangÄ±Ã§ YÄ±lÄ± : {min_year}")
    print(f"ğŸ“… BitiÅŸ YÄ±lÄ±     : {max_year}")
    print(f"â�±ï¸� Kapsanan SÃ¼re  : {expected_duration} YÄ±l")
    print(f"âœ… Dolu YÄ±l SayÄ±sÄ±: {total_years}")
    
    # 3. Kesinti KontrolÃ¼ (Gap Check)
    if total_years == expected_duration:
        print("   SeÃ§ilen aralÄ±ktaki tÃ¼m yÄ±llar mevcut.")
    else:
        print(f"\nArada {expected_duration - total_years} yÄ±l eksik!")
        # Eksik yÄ±llarÄ± bulup gÃ¶sterelim
        all_years = set(range(min_year, max_year + 1))
        existing_years = set(df_final_common['YEAR'].unique())
        missing = sorted(list(all_years - existing_years))
        print(f"   Eksik YÄ±llar: {missing}")

else:
    print("â�Œ Tablo boÅŸ veya oluÅŸturulmamÄ±ÅŸ.")


df_final_common.head(10)


df_final_common.shape


train_path = "/kaggle/input/air-toxicity-and-chronic-respiratory-diseases-us/train.csv"
df = pd.read_csv(train_path)
df.head(8)


df = df[~df['Year'].between(1990, 1997)]


df['Year'].min(), df['Year'].max()


df = df.rename(columns={"State.Name": "STATE", "Year": "YEAR"})

df["YEAR"] = df["YEAR"].astype(int)
df_final_common["YEAR"] = df_final_common["YEAR"].astype(int)

df["STATE"] = df["STATE"].astype(str).str.strip()
df_final_common["STATE"] = df_final_common["STATE"].astype(str).str.strip()

df_merged = df.merge(
    df_final_common,
    on=["STATE", "YEAR"],
    how="left",
    indicator=True   
)

df_merged.head(10)


missing_cases = df_merged[df_merged["_merge"] == "left_only"]
missing_cases.count()



df_merged.shape


df_merged = df_merged[df_merged["_merge"] != "left_only"]
df_merged.head(10)


df_merged.shape


dup_all = df_merged[df_merged.duplicated(keep=False)]
print("DuplÄ±cated rows:", dup_all.shape[0])
df_merged.info()


df_merged.head(8)


cols_to_drop = [c for c in ["ID", "id", "STATE","_merge"] if c in df_merged.columns]

df_merged = df_merged.drop(columns=cols_to_drop)

df_merged.head()



import re
import numpy as np

def age_to_midpoint(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    m = re.match(r"(\d+)\s*-\s*(\d+)", x)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (a + b) / 2
    return np.nan

if "Age" in df_merged.columns:
    df_merged["Age_mid"] = df_merged["Age"].apply(age_to_midpoint)
    df_merged = df_merged.drop(columns=["Age"])

df_merged[["Age_mid"]].head()


df_merged.head(5)


print("Missing values per column:")
print(df_merged.isna().sum().sort_values(ascending=False).head(20))

print("\nTotal duplicated rows:", df_merged.duplicated().sum())



target = "Incidence"  

x = df_merged.drop(columns=[target])
y = df_merged[target]

print("X shape:", x.shape)
print("y shape:", y.shape)


import matplotlib.pyplot as plt
import numpy as np
tmp_num = df_merged.select_dtypes(include=[np.number]).copy()

corr_matrix = tmp_num.corr()

plt.figure(figsize=(12, 10))
plt.imshow(corr_matrix, aspect="auto")

plt.title("Correlation Matrix (Heatmap)")
plt.colorbar()

ticks = np.arange(len(corr_matrix.columns))
plt.xticks(ticks, corr_matrix.columns, rotation=90)
plt.yticks(ticks, corr_matrix.columns)

plt.tight_layout()
plt.show()



from sklearn.model_selection import train_test_split
X_all = x

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=42
)

X_train.shape, X_test.shape


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

mae_lr, rmse_lr, r2_lr


from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

mae_xgb  = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb   = r2_score(y_test, y_pred_xgb)

mae_xgb, rmse_xgb, r2_xgb



feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": xgb.feature_importances_
}).sort_values(by="importance", ascending=False)

feature_importance



import shap

#  SHAP explainer
explainer = shap.Explainer(xgb, X_train)

shap_values = explainer(X_test)



shap.summary_plot(shap_values, X_test, plot_type="bar")


shap.dependence_plot("PM25", shap_values.values, X_test)


shap.dependence_plot("CO", shap_values.values, X_test)
shap.dependence_plot("O3", shap_values.values, X_test)

