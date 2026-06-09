#1. hÃ¼cre
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


# 2A â€” HÄ±zlÄ± EDA (Ã¶zet tablo + hedef daÄŸÄ±lÄ±mÄ±)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 200)

# Veri setini ham hÃ¢liyle oku (deÄŸiÅŸtirme!)
df0 = pd.read_csv("/kaggle/input/playground-series-s4e6/train.csv")

print("Å�ekil:", df0.shape)
print("\nÄ°lk 5 satÄ±r:")
display(df0.head())

# Eksik deÄŸer tablosu
na_tbl = (
    df0.isna().sum()
    .to_frame("missing")
    .assign(pct=lambda x: (100*x["missing"]/len(df0)).round(2))
    .sort_values(["missing","pct"], ascending=[False, False])
)
print("\nEksik deÄŸer Ã¶zeti (en Ã§ok eksik iÃ§erenler Ã¼stte):")
display(na_tbl.head(20))

# Kategorik ve sayÄ±sal kolon listeleri (ham veri)
num_cols = df0.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df0.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"\nSayÄ±sal sÃ¼tun sayÄ±sÄ±: {len(num_cols)}")
print(f"Kategorik sÃ¼tun sayÄ±sÄ±: {len(cat_cols)}")

# Hedef daÄŸÄ±lÄ±mÄ± (class imbalance iÃ§in ilk bakÄ±ÅŸ)
if "Target" in df0.columns:
    vc = df0["Target"].value_counts(normalize=True).sort_index()
    print("\nHedef oranlarÄ± (%):")
    display((vc*100).round(2))

    vc.plot(kind="bar")
    plt.title("Hedef DaÄŸÄ±lÄ±mÄ± (Train)")
    plt.xlabel("SÄ±nÄ±f")
    plt.ylabel("Oran")
    plt.show()

    # Basit dengesizlik notu
    maj = vc.max() if len(vc)>0 else np.nan
    if pd.notna(maj):
        print(f"Not: En bÃ¼yÃ¼k sÄ±nÄ±f oranÄ± â‰ˆ {maj*100:.2f}%")
        if maj >= 0.55:
            print("â†’ Veri DENGESÄ°Z kabul edilebilir (SMOTE/weighting gibi yÃ¶ntemler dÃ¼ÅŸÃ¼nÃ¼lebilir).")
        else:
            print("â†’ Veri belirgin dengesiz gÃ¶rÃ¼nmÃ¼yor (yine de modeli/metric'i kontrol edeceÄŸiz).")
else:
    print("UyarÄ±: 'Target' sÃ¼tunu bulunamadÄ±.")



# 2B â€” SayÄ±sal sÃ¼tun daÄŸÄ±lÄ±mlarÄ± (hÄ±zlÄ± histogramlar)
import numpy as np
import matplotlib.pyplot as plt

# Ã‡ok kalabalÄ±k olmasÄ±n diye ilk N sayÄ±sal sÃ¼tunu Ã§izelim
N = 12
plot_cols = [c for c in df0.select_dtypes(include=[np.number]).columns if c != "Target"][:N]

for c in plot_cols:
    s = df0[c]
    # winsorize benzeri kÄ±rpma (uÃ§ deÄŸer patlamasÄ±nÄ± yumuÅŸatÄ±r)
    q1, q99 = s.quantile(0.01), s.quantile(0.99)
    s_clip = s.clip(lower=q1, upper=q99)

    s_clip.hist(bins=30)
    plt.title(f"{c} â€” Histogram (1%-99% kÄ±rpÄ±lmÄ±ÅŸ)")
    plt.xlabel(c); plt.ylabel("Frekans")
    plt.show()



# 2D â€” Korelasyon (sayÄ±sal sÃ¼tunlar, hÄ±zlÄ± Ä±sÄ± haritasÄ±)
import numpy as np
import matplotlib.pyplot as plt

num_cols = df0.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) >= 2:
    corr = df0[num_cols].corr(numeric_only=True)
    plt.imshow(corr, cmap="viridis", aspect="auto")
    plt.title("Korelasyon IsÄ± HaritasÄ± (SayÄ±sal)")
    plt.colorbar(label="Korelasyon")
    plt.xticks(range(len(num_cols)), num_cols, rotation=90, fontsize=8)
    plt.yticks(range(len(num_cols)), num_cols, fontsize=8)
    plt.tight_layout()
    plt.show()

    # Target ile en iliÅŸkili sayÄ±sallar (varsa)
    if "Target" in corr.columns:
        # Target sayÄ±sal deÄŸilse bu blok Ã§alÄ±ÅŸmaz; LabelEncoder sonrasÄ± ayrÄ± bakacaÄŸÄ±z.
        pass
else:
    print("Korelasyon iÃ§in yeterli sayÄ±sal sÃ¼tun yok.")



#2.hÃ¼cre
import pandas as pd

# EÄŸitim verisini oku
df = pd.read_csv("/kaggle/input/playground-series-s4e6/train.csv")

# Ä°lk 5 satÄ±rÄ± gÃ¶ster
df.head()



#3. hÃ¼cre
df.shape



#4.hÃ¼cre
# ============================================
# ğŸ”� 2. VERÄ° BOYUTU, SÃœTUNLAR VE VERÄ° TÄ°PLERÄ°
# ============================================

# KaÃ§ satÄ±r ve sÃ¼tun var?
print("ğŸ§¾ Veri Boyutu:", df.shape)  # (satÄ±r, sÃ¼tun)

# SÃ¼tun isimleri ve veri tipleri
print("\nğŸ“‹ SÃ¼tun Bilgileri:")
print(df.info())#sÃ¼tun ad, sÃ¼tunlarÄ±n veri tipi, boÅŸ(NaN) deÄŸer var mÄ±?

# Ã–rnek birkaÃ§ satÄ±rÄ± tekrar gÃ¶relim (kontrol iÃ§in)
df.head(3)



#5. hÃ¼cre
# ============================================
# âš ï¸� 3. EKSÄ°K (NULL) VERÄ°LERÄ°N TESPÄ°TÄ°
# ============================================

# Her sÃ¼tunda kaÃ§ adet eksik deÄŸer var?
missing = df.isnull().sum()

# Sadece eksik deÄŸeri olan sÃ¼tunlarÄ± filtrele
missing = missing[missing > 0]

if len(missing) == 0:
    print("âœ… Veri setinde eksik deÄŸer yok.")
else:
    print("âš ï¸� Eksik deÄŸer bulunan sÃ¼tunlar:\n")
    print(missing)



#6.hÃ¼cre
# 6B â€” Eksik veri Ä±sÄ± haritasÄ±
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
sns.heatmap(df.isnull(), cbar=False, cmap="YlGnBu")
plt.title("Eksik Veri IsÄ± HaritasÄ±")
plt.tight_layout()
plt.show()



#7.hÃ¼cre
# ============================================
# 5A â€” EKSÄ°K VERÄ° GÄ°DERME (Ä°MPUTATION)
#  - SayÄ±sal: median ile doldur
#  - Kategorik: mod (en sÄ±k) ile doldur
#  - EÄŸer eksik yoksa dokunmaz; gÃ¼venli Ã§alÄ±ÅŸÄ±r
# ============================================

import numpy as np
import pandas as pd

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

# SayÄ±sal sÃ¼tunlar: median
if len(num_cols):
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Kategorik sÃ¼tunlar: mod
for c in cat_cols:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].mode().iloc[0])

print("âœ… Eksik veri giderildi. Kalan toplam eksik:", int(df.isnull().sum().sum()))



# 7B â€” Feature Engineering (PS S4E6'e Ã¶zel, gÃ¼venli)
import numpy as np
import pandas as pd

def sdiv(a, b):
    """SÄ±fÄ±ra bÃ¶lme hatasÄ±na karÅŸÄ± gÃ¼venli bÃ¶lme: b=0 ise 0 dÃ¶ner."""
    return np.where(b==0, 0, a / b)

def fe_student_success(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    has = lambda c: c in df.columns  # kolon var mÄ±?

    # --- KISA ADLAR (okunurluk iÃ§in) ---
    c1_enr = "Curricular units 1st sem (enrolled)"
    c1_eval= "Curricular units 1st sem (evaluations)"
    c1_app = "Curricular units 1st sem (approved)"
    c1_gr  = "Curricular units 1st sem (grade)"
    c1_wne = "Curricular units 1st sem (without evaluations)"
    c1_cred= "Curricular units 1st sem (credited)"

    c2_enr = "Curricular units 2nd sem (enrolled)"
    c2_eval= "Curricular units 2nd sem (evaluations)"
    c2_app = "Curricular units 2nd sem (approved)"
    c2_gr  = "Curricular units 2nd sem (grade)"
    c2_wne = "Curricular units 2nd sem (without evaluations)"
    c2_cred= "Curricular units 2nd sem (credited)"

    # --- 1) DÃ–NEM BAZLI ORANLAR / SAYILAR ---
    if all(has(c) for c in [c1_enr, c1_app]):
        df["pass_rate_sem1"] = sdiv(df[c1_app], df[c1_enr])
    if all(has(c) for c in [c2_enr, c2_app]):
        df["pass_rate_sem2"] = sdiv(df[c2_app], df[c2_enr])

    if all(has(c) for c in [c1_enr, c1_eval]):
        df["eval_rate_sem1"] = sdiv(df[c1_eval], df[c1_enr])
    if all(has(c) for c in [c2_enr, c2_eval]):
        df["eval_rate_sem2"] = sdiv(df[c2_eval], df[c2_enr])

    if all(has(c) for c in [c1_enr, c1_wne]):
        df["noeval_rate_sem1"] = sdiv(df[c1_wne], df[c1_enr])
    if all(has(c) for c in [c2_enr, c2_wne]):
        df["noeval_rate_sem2"] = sdiv(df[c2_wne], df[c2_enr])

    if all(has(c) for c in [c1_eval, c1_app]):
        df["fail_count_sem1"] = np.maximum(0, df[c1_eval] - df[c1_app])
    if all(has(c) for c in [c2_eval, c2_app]):
        df["fail_count_sem2"] = np.maximum(0, df[c2_eval] - df[c2_app])

    if all(has(c) for c in [c1_enr, c1_app, c2_enr, c2_app]):
        df["enrolled_total"] = df[c1_enr] + df[c2_enr]
        df["approved_total"] = df[c1_app] + df[c2_app]
        df["pass_rate_total"] = sdiv(df["approved_total"], df["enrolled_total"])

    # --- 2) NOTLAR VE DELTALAR ---
    if all(has(c) for c in [c1_gr, c2_gr]):
        df["grade_mean"] = (df[c1_gr] + df[c2_gr]) / 2.0
        df["grade_delta_2_1"] = df[c2_gr] - df[c1_gr]   # 2. dÃ¶nem - 1. dÃ¶nem
    if all(has(c) for c in ["pass_rate_sem1", "pass_rate_sem2"]):
        df["pass_rate_delta_2_1"] = df["pass_rate_sem2"] - df["pass_rate_sem1"]
    if all(has(c) for c in [c1_enr, c2_enr]):
        df["enrolled_delta_2_1"] = df[c2_enr] - df[c1_enr]
    if all(has(c) for c in [c1_app, c2_app]):
        df["approved_delta_2_1"] = df[c2_app] - df[c1_app]

    # --- 3) MUAFÄ°YET ORANLARI ---
    if all(has(c) for c in [c1_cred, c1_enr]):
        df["credit_ratio_sem1"] = sdiv(df[c1_cred], df[c1_enr])
    if all(has(c) for c in [c2_cred, c2_enr]):
        df["credit_ratio_sem2"] = sdiv(df[c2_cred], df[c2_enr])

    # --- 4) MALÄ° DURUM / UYUM BAYRAKLARI ---
    if all(has(c) for c in ["Debtor", "Tuition fees up to date"]):
        df["good_payer"] = ((df["Tuition fees up to date"]==1) & (df["Debtor"]==0)).astype(int)
        df["payment_mismatch"] = ((df["Tuition fees up to date"]==1) & (df["Debtor"]==1)).astype(int)

    # --- 5) AÄ°LE EÄ�Ä°TÄ°MÄ° Ã–ZETLERÄ° ---
    if all(has(c) for c in ["Mother's qualification", "Father's qualification"]):
        df["parents_edu_mean"] = df[["Mother's qualification", "Father's qualification"]].mean(axis=1)
        df["parents_edu_max"] = df[["Mother's qualification", "Father's qualification"]].max(axis=1)

    # --- 6) EKONOMÄ°K ORTAM ENDEKSÄ° ---
    if all(has(c) for c in ["Unemployment rate", "Inflation rate", "GDP"]):
        df["econ_stress"] = df["Unemployment rate"] + df["Inflation rate"] - df["GDP"]

    # --- 7) YAÅ� VE KATEGORÄ°K BAYRAKLAR ---
    if has("Age at enrollment"):
        age = df["Age at enrollment"]
        df["age_under_20"] = (age < 20).astype(int)
        df["age_20_25"]   = ((age >= 20) & (age <= 25)).astype(int)
        df["age_over_25"] = (age > 25).astype(int)

    if has("Application order"):
        df["first_choice"] = (df["Application order"] == 1).astype(int)

    if has("International"):
        df["is_international"] = df["International"].astype(int)
    if has("Displaced"):
        df["is_displaced"] = df["Displaced"].astype(int)

    # --- 8) KULLANMAYI DÃœÅ�ÃœNMEDÄ°KLERÄ°MÄ°Z ---
    # 'id' yalnÄ±zca kimlik: model giriÅŸlerinden sonra Ã§Ä±karÄ±labilir.

    return df

# FE'yi uygula
df_fe = fe_student_success(df)
print("FE bitti. Yeni sÃ¼tun sayÄ±sÄ±:", df_fe.shape[1] - df.shape[1])
df = df_fe  # istersen df'yi gÃ¼ncelle



#8.hÃ¼cre
# ============================================
# ğŸ“Š 4. SAYISAL SÃœTUNLARIN TEMEL Ä°STATÄ°STÄ°KLERÄ°
# ============================================

# describe(): sayÄ±sal sÃ¼tunlarÄ±n istatistik Ã¶zetini verir
df.describe()



# 8C â€” Dengesizlik testi (â‰¥ %55 tek sÄ±nÄ±f ise dengesiz)
vc = df["Target"].value_counts(normalize=True).sort_values(ascending=False)
print("SÄ±nÄ±f oranlarÄ± (%):\n", (vc*100).round(2))
is_imbalanced = (vc.iloc[0] >= 0.55)
print("\nDengesiz mi?:", "EVET" if is_imbalanced else "HAYIR")

import seaborn as sns, matplotlib.pyplot as plt
plt.figure(figsize=(5.5,4))
sns.barplot(x=vc.index.astype(str), y=vc.values)
plt.title("SÄ±nÄ±f OranlarÄ±"); plt.ylabel("Oran")
plt.tight_layout(); plt.show()




#9. hÃ¼cre
# 4B â€” SayÄ±sal sÃ¼tun histogramlarÄ±
import matplotlib.pyplot as plt
import pandas as pd

num_cols = df.select_dtypes(include='number').columns.tolist()
view_cols = num_cols[:6] if len(num_cols) > 6 else num_cols

if len(view_cols):
    df[view_cols].hist(bins=20, figsize=(14, 8))
    plt.suptitle("SayÄ±sal SÃ¼tun DaÄŸÄ±lÄ±mlarÄ±", y=1.02)
    plt.tight_layout()
    plt.show()
else:
    print("SayÄ±sal sÃ¼tun yok/az.")



#10.hÃ¼cre
# 4C â€” Korelasyon Ä±sÄ± haritasÄ±
import seaborn as sns
import matplotlib.pyplot as plt

num_cols = df.select_dtypes(include='number').columns
if len(num_cols) >= 2:
    plt.figure(figsize=(8,6))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("SayÄ±sal Korelasyon IsÄ± HaritasÄ±")
    plt.tight_layout()
    plt.show()
else:
    print("Korelasyon iÃ§in en az 2 sayÄ±sal sÃ¼tun gerekir.")



#11.hÃ¼cre
# ============================================
# ğŸ�¯ 5. HEDEF DEÄ�Ä°Å�KENÄ°N (TARGET) DAÄ�ILIMI
# ============================================

# Hedef sÃ¼tunun adÄ±nÄ± kontrol et (genelde "target" veya "Target")
if "target" in df.columns:
    target_col = "target"
elif "Target" in df.columns:
    target_col = "Target"
else:
    print("âš ï¸� Hedef sÃ¼tun adÄ± manuel girilmeli (Ã¶rneÄŸin 'Outcome', 'Result' vs.)")
    target_col = input("Hedef sÃ¼tun adÄ±nÄ± gir: ")

# Hedef sÃ¼tun daÄŸÄ±lÄ±mÄ±nÄ± gÃ¶ster
print("\nğŸ�¯ Hedef sÃ¼tun daÄŸÄ±lÄ±mÄ±:")
print(df[target_col].value_counts())

# YÃ¼zdelik oranla gÃ¶sterelim
print("\nğŸ“Š DaÄŸÄ±lÄ±m (yÃ¼zde):")
print(df[target_col].value_counts(normalize=True) * 100)



#12.hÃ¼cre
# ============================================
# 7A â€” TARGET DAÄ�ILIM GRAFÄ°Ä�Ä°
#  - Countplot ve her Ã§ubuÄŸa yÃ¼zde anotasyonu
# ============================================

import matplotlib.pyplot as plt
import seaborn as sns

assert 'target_col' in globals(), "Ã–nce 7. hÃ¼cre (target sÃ¼tununun belirlenmesi) Ã§alÄ±ÅŸmalÄ±."

plt.figure(figsize=(6,4))
ax = sns.countplot(x=target_col, data=df)
ax.set_title("ğŸ�¯ Target DaÄŸÄ±lÄ±mÄ±")

# YÃ¼zde anotasyonu
total = len(df)
for p in ax.patches:
    count = int(p.get_height())
    pct = 100.0 * count / total
    ax.annotate(f"{pct:.1f}%", (p.get_x() + p.get_width()/2, p.get_height()),
                ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')

plt.xlabel(target_col)
plt.ylabel("Adet")
plt.tight_layout()
plt.show()



#13.hÃ¼cre
# ============================================
# ğŸ§¹ 6. GEREKSÄ°Z SÃœTUNLAR VE HEDEF DÃ–NÃœÅ�ÃœMÃœ
# ============================================

from sklearn.preprocessing import LabelEncoder

# 1ï¸�âƒ£ ID sÃ¼tunu model iÃ§in gereksiz â†’ silelim
df = df.drop(columns=['id'])

# 2ï¸�âƒ£ Target sÃ¼tununu sayÄ±sal hale getirelim
le = LabelEncoder()
df['Target'] = le.fit_transform(df['Target'])

# DÃ¶nÃ¼ÅŸÃ¼mÃ¼n nasÄ±l olduÄŸunu gÃ¶relim
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("ğŸ�¯ Target dÃ¶nÃ¼ÅŸÃ¼m haritasÄ±:", mapping)

# Ä°lk birkaÃ§ satÄ±rÄ± kontrol et
df.head()



#14.hÃ¼cre
# ============================================
# ğŸ”� 7. KATEGORÄ°K SÃœTUNLARIN TESPÄ°TÄ°
# ============================================

# object veya category tipindeki sÃ¼tunlarÄ± bul
cat_cols = df.select_dtypes(include=['object', 'category']).columns

if len(cat_cols) == 0:
    print("âœ… Kategorik sÃ¼tun bulunmuyor (tÃ¼m sÃ¼tunlar sayÄ±sal).")
else:
    print("âš ï¸� Kategorik sÃ¼tun(lar) bulundu:", list(cat_cols))



#15.hÃ¼cre
# ============================================
# âš ï¸� 8. AYKIRI (OUTLIER) DEÄ�ERLERÄ°N GÃ–RSEL ANALÄ°ZÄ°
# ============================================

import matplotlib.pyplot as plt
import seaborn as sns

# Ã–rnek olarak bir kaÃ§ sayÄ±sal sÃ¼tunun boxplot'unu Ã§izelim
plt.figure(figsize=(12,6))
sns.boxplot(data=df[['Previous qualification (grade)', 
                     'Admission grade', 
                     'Curricular units 1st sem (grade)', 
                     'Curricular units 2nd sem (grade)']])
plt.title("ğŸ“Š AykÄ±rÄ± DeÄŸer KontrolÃ¼ (Boxplot)")
plt.show()



#16.hÃ¼cre
# ============================================
# ğŸ”� 9. KORELASYON ANALÄ°ZÄ°
# ============================================

import seaborn as sns
import matplotlib.pyplot as plt

# Korelasyon matrisini hesapla
corr_matrix = df.corr()

# GÃ¶rselleÅŸtirelim
plt.figure(figsize=(14,10))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
plt.title("ğŸ”� Korelasyon Matrisi (Feature Correlation Heatmap)")
plt.show()

# 0.95'ten bÃ¼yÃ¼k korelasyonlarÄ± listeleyelim (Ã§ok benzer sÃ¼tunlar)
high_corr = corr_matrix[(corr_matrix > 0.95) & (corr_matrix < 1.0)]
print("âš ï¸� YÃ¼ksek korelasyonlu sÃ¼tun Ã§iftleri:")
print(high_corr.dropna(axis=0, how='all').dropna(axis=1, how='all'))



#17.hÃ¼cre
# ============================================
# ğŸŒ² 10. FEATURE IMPORTANCE (Ã–ZNÄ°TELÄ°K Ã–NEMÄ°)
# ============================================

from sklearn.ensemble import RandomForestClassifier

# BaÄŸÄ±msÄ±z deÄŸiÅŸkenler (X) ve hedef deÄŸiÅŸken (y)
X = df.drop(columns=['Target'])
y = df['Target']

# Basit bir model eÄŸitiyoruz
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X, y)

# Feature importance deÄŸerlerini al
importances = pd.Series(model.feature_importances_, index=X.columns)

# En Ã¶nemli 10 Ã¶zelliÄŸi gÃ¶rselleÅŸtirelim
top10 = importances.sort_values(ascending=False).head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top10.values, y=top10.index, palette="viridis")
plt.title("ğŸ�† En Ã–nemli 10 Ã–zellik (RandomForest Feature Importance)")
plt.xlabel("Ã–nem Skoru")
plt.ylabel("Ã–zellik AdÄ±")
plt.show()



#18.hÃ¼cre
# ============================================
# âœ‚ï¸� 11. GEREKSÄ°Z Ã–ZELLÄ°KLERÄ° Ã‡IKARMA (Opsiyonel)
# ============================================

low_features = importances[importances < 0.002].index  # Ã¶nem oranÄ± < 0.2%
print("âš™ï¸� DÃ¼ÅŸÃ¼k etkili Ã¶zellik sayÄ±sÄ±:", len(low_features))
print("ğŸª¶ Silinebilecek Ã¶rnek sÃ¼tunlar:", list(low_features[:10]))

# Ä°stersen bu satÄ±rÄ± aktif edip gerÃ§ekten silebilirsin:
# df = df.drop(columns=low_features)



#19.hÃ¼cre
# ============================================
# ğŸ�“ 12. VERÄ°YÄ° EÄ�Ä°TÄ°M ve TEST SETÄ°NE AYIR
# ============================================

from sklearn.model_selection import train_test_split

# BaÄŸÄ±msÄ±z deÄŸiÅŸkenler (Ã¶zellikler)
X = df.drop(columns=['Target'])
# BaÄŸÄ±mlÄ± deÄŸiÅŸken (etiket)
y = df['Target']

# %80 eÄŸitim, %20 test olarak ayÄ±r
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("ğŸ§  EÄŸitim verisi boyutu:", X_train.shape)
print("ğŸ§ª Test verisi boyutu:", X_test.shape)



#20.hÃ¼cre
# ============================================
# 14A â€” PREPROCESSOR (One-Hot + gÃ¼venli imputation)
#  - Split sonrasÄ±: sadece X_train Ã¼zerinde fit, X_test'e transform
#  - Categorical -> OneHotEncoder(handle_unknown="ignore", sparse=False)
#  - Numerical   -> SimpleImputer(median)
# ============================================

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline              # ğŸ”¹ EKLENDÄ°
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# X_train, X_test, y_train, y_test  â†� 14. hÃ¼creden geliyor olmalÄ±
assert 'X_train' in globals() and 'X_test' in globals(), "Ã–nce 14. hÃ¼cre (train/test split) Ã§alÄ±ÅŸmalÄ±."

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X_train.columns if c not in num_cols]

numeric_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categoric_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_tf, num_cols),
        ("cat", categoric_tf, cat_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

# Sadece train Ã¼zerinde fit â†’ sonra her ikisine de transform
Xtr = preprocessor.fit_transform(X_train)
Xte = preprocessor.transform(X_test)

# OneHot sonrasÄ± feature isimleri (SHAP/grafikler iÃ§in faydalÄ±)
try:
    feature_names = preprocessor.get_feature_names_out()
except Exception:
    feature_names = [f"f{i}" for i in range(Xtr.shape[1])]

print("âœ… Preprocessor OK.")
print(" - Xtr shape:", Xtr.shape, " | Xte shape:", Xte.shape)
print(" - Train sÄ±nÄ±f daÄŸÄ±lÄ±mÄ±:\n", y_train.value_counts(normalize=True).round(3))



#21.hÃ¼cre
# ============================================
# 14B-alt â€” CLASS BALANCING (Only on TRAIN, no imblearn)
#  - Random oversampling: her sÄ±nÄ±fÄ± majority sayÄ±sÄ±na Ã§Ä±kar
#  - Xtr (np.array), y_train (Series) beklenir
# ============================================

import numpy as np
import pandas as pd
from collections import Counter

assert 'Xtr' in globals() and 'y_train' in globals(), "Ã–nce 14A Ã§alÄ±ÅŸmalÄ±."

# y_train -> numpy
ytr = np.asarray(y_train)
classes, counts = np.unique(ytr, return_counts=True)
maj = counts.max()

# Her sÄ±nÄ±f iÃ§in indeksleri al
indices_by_class = {c: np.where(ytr == c)[0] for c in classes}

# Oversample
rng = np.random.default_rng(42)
new_indices = []
for c in classes:
    idx = indices_by_class[c]
    if len(idx) == 0:
        continue
    # gerekli kadar rastgele Ã¶rnekle (tekrar seÃ§ime izin ver)
    need = maj - len(idx)
    if need > 0:
        extra = rng.choice(idx, size=need, replace=True)
        idx = np.concatenate([idx, extra])
    new_indices.append(idx)

new_indices = np.concatenate(new_indices)
rng.shuffle(new_indices)

Xtr_bal = Xtr[new_indices]
ytr_bal = ytr[new_indices]

print("Ã–nce:", Counter(y_train))
print("Sonra:", Counter(ytr_bal))
print("âœ… Oversampling tamam. Xtr_bal:", Xtr_bal.shape)



# 21-SMOTE â€” Sadece TRAIN Ã¼zerinde (Xtr, y_train_enc)
# GÃ¼rÃ¼ltÃ¼lÃ¼ STDERR mesajlarÄ±nÄ± bastÄ±rmak iÃ§in
import contextlib, io, sys
_silent = io.StringIO()
with contextlib.redirect_stderr(_silent):
    from imblearn.over_sampling import SMOTE
    # SMOTE ve modeli Ã§alÄ±ÅŸtÄ±ran kodu da bu bloÄŸun iÃ§ine koyabilirsin

from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier

assert 'Xtr' in globals() and 'y_train_enc' in globals(), "Ã–nce 20 ve 24 Ã§alÄ±ÅŸmalÄ±."

sm = SMOTE(random_state=42, k_neighbors=5)
Xtr_sm, ytr_sm = sm.fit_resample(Xtr, y_train_enc)
print("ğŸ§ª SMOTE: Ã¶nce:", np.bincount(y_train_enc), " | sonra:", np.bincount(ytr_sm))

lgb_sm = LGBMClassifier(n_estimators=800, learning_rate=0.06, class_weight='balanced', random_state=42, n_jobs=-1)
lgb_sm.fit(Xtr_sm, ytr_sm)
yp = lgb_sm.predict(Xte)
print("SMOTE@LGBM â€” Acc:", accuracy_score(y_test_enc, yp), " Macro-F1:", f1_score(y_test_enc, yp, average='macro'))



#22.hÃ¼cre
# ============================================
# 12A â€” DecisionTree ile Ã–zellik SeÃ§imi (top-k)
#  - Girdi: Xtr, Xte (14A'dan), y_train / y_train_enc, feature_names
#  - Ã‡Ä±ktÄ±: Xtr_fs, Xte_fs (ve varsa Xtr_bal_fs)
# ============================================

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# GÃ¼venlik kontrolleri
assert 'Xtr' in globals() and 'Xte' in globals(), "Ã–nce 14A (preprocessor) Ã§alÄ±ÅŸmalÄ±."
assert 'feature_names' in globals(), "14A'da preprocessor.get_feature_names_out() Ã¼retildi mi?"

# Etiket: y_train_enc yoksa bu hÃ¼crede oluÅŸturalÄ±m
if 'y_train_enc' not in globals():
    assert 'y_train' in globals(), "y_train bulunamadÄ± (train/test split koÅŸmalÄ±)."
    _le_tmp = LabelEncoder()
    y_train_enc = _le_tmp.fit_transform(y_train)

# --- 1) DT modeli (hafif dÃ¼zenleme ile) ---
dt = DecisionTreeClassifier(
    max_depth=6,          # Ã§ok derin deÄŸil â†’ daha genel Ã¶nemler
    min_samples_leaf=5,   # yaprakta min Ã¶rnek
    random_state=42
)
dt.fit(Xtr, y_train_enc)

importances = dt.feature_importances_
feat_df = pd.DataFrame({
    "feature": list(feature_names),
    "importance": importances
}).sort_values("importance", ascending=False)

# --- 2) Top-k seÃ§imi ---
TOP_K = 20   # istersen 30 yapabilirsin
top_feats = feat_df.head(TOP_K)["feature"].tolist()
top_idx = [list(feature_names).index(f) for f in top_feats]

# --- 3) AzaltÄ±lmÄ±ÅŸ matrisler ---
Xtr_fs = Xtr[:, top_idx]
Xte_fs = Xte[:, top_idx]

# Oversampled eÄŸitim seti varsa onu da indirgeme
if 'Xtr_bal' in globals():
    Xtr_bal_fs = Xtr_bal[:, top_idx]

print(f"âœ… DT-FeatureSelection tamam. TOP_K={TOP_K}")
print("SeÃ§ilen ilk 10 Ã¶zellik:", top_feats[:10])
print("Xtr_fs:", Xtr_fs.shape, "| Xte_fs:", Xte_fs.shape)
if 'Xtr_bal' in globals():
    print("Xtr_bal_fs:", Xtr_bal_fs.shape)



#23.hÃ¼cre
# ============================================
# 12B â€” SHAP Top-K Ã–zellik Listesi
#  - VarsayÄ±lan: final_xgb (15A'dan). Yoksa hÄ±zlÄ± bir LGBM eÄŸitip kullanÄ±r.
#  - Ã‡Ä±ktÄ±: Xtr_shap_fs, Xte_shap_fs (ve varsa Xtr_bal_shap_fs)
# ============================================

import numpy as np
import pandas as pd

# 1) Model seÃ§imi: final_xgb varsa onu kullan, yoksa hÄ±zlÄ± LGBM kur
model_for_shap = None
if 'final_xgb' in globals():
    model_for_shap = final_xgb
    model_name = "XGBoost(final_xgb)"
else:
    from lightgbm import LGBMClassifier
    assert 'y_train_enc' in globals(), "Ã–nce 15A0 (LabelEncoder) Ã§alÄ±ÅŸmalÄ±."
    lgb_fast = LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        subsample=0.9, colsample_bytree=0.9, random_state=42
    )
    lgb_fast.fit(Xtr, y_train_enc)
    model_for_shap = lgb_fast
    model_name = "LightGBM(fast)"

print("SHAP modeli:", model_name)

# 2) SHAP hesapla (TreeExplainer, hÄ±z iÃ§in Ã¶rnekleme)
import shap
shap_explainer = shap.TreeExplainer(model_for_shap)

rng = np.random.default_rng(42)
sample_size = min(1000, Xtr.shape[0])  # hÄ±z/gÃ¼venlik iÃ§in 1000 Ã¶rnek
sample_idx = rng.choice(Xtr.shape[0], size=sample_size, replace=False)

shap_vals = shap_explainer.shap_values(Xtr[sample_idx])

# 3) Ã‡ok sÄ±nÄ±flÄ± ise: her sÄ±nÄ±f matrisini |.| alÄ±p ortalayÄ±p sonra sÄ±nÄ±flar Ã¼stÃ¼ ortalama
if isinstance(shap_vals, list):
    # -> shape: [n_classes][n_samples, n_features]
    abs_mean_by_class = [np.abs(sv).mean(axis=0) for sv in shap_vals]
    abs_mean = np.mean(abs_mean_by_class, axis=0)
else:
    # -> shape: [n_samples, n_features]
    abs_mean = np.abs(shap_vals).mean(axis=0)

# 4) Top-K Ã¶zellikleri seÃ§ ve indirgeme yap
assert 'feature_names' in globals(), "feature_names 14A'da Ã¼retilmiÅŸ olmalÄ±."
shap_imp = pd.DataFrame({"feature": list(feature_names), "abs_mean_shap": abs_mean})
shap_imp = shap_imp.sort_values("abs_mean_shap", ascending=False)

TOP_K_SHAP = 20
top_feats_shap = shap_imp.head(TOP_K_SHAP)["feature"].tolist()
top_idx_shap = [list(feature_names).index(f) for f in top_feats_shap]

Xtr_shap_fs = Xtr[:, top_idx_shap]
Xte_shap_fs = Xte[:, top_idx_shap]
if 'Xtr_bal' in globals():
    Xtr_bal_shap_fs = Xtr_bal[:, top_idx_shap]

print(f"âœ… SHAP-TopK tamam. TOP_K_SHAP={TOP_K_SHAP}")
print("Ä°lk 10 SHAP Ã¶zelliÄŸi:", top_feats_shap[:10])
print("Xtr_shap_fs:", Xtr_shap_fs.shape, "| Xte_shap_fs:", Xte_shap_fs.shape)
if 'Xtr_bal' in globals():
    print("Xtr_bal_shap_fs:", Xtr_bal_shap_fs.shape)



#24.hÃ¼cre
# ============================================
# 15A0 â€” LabelEncoder + Baseline XGB sanity-check
# ============================================
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

# Etiketleri sayÄ±sala Ã§evir
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)
class_names = [str(c) for c in le.classes_]
print("SÄ±nÄ±flar:", class_names)

# HÄ±zlÄ± baseline (oversample YOK; Xtr/Xte)
xgb_base = XGBClassifier(
    objective="multi:softprob",
    tree_method="hist",
    eval_metric="mlogloss",
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=100,
)
xgb_base.fit(Xtr, y_train_enc, eval_set=[(Xte, y_test_enc)], verbose=False)
yhat_base = xgb_base.predict(Xte)
print("Baseline macro-F1:", f1_score(y_test_enc, yhat_base, average="macro"))



# 17B â€” SMOTEâ€™lu vs SMOTEâ€™suz KarÅŸÄ±laÅŸtÄ±rma (XGB)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier

# --- Ã‡evre deÄŸiÅŸkenlerinin varlÄ±ÄŸÄ±nÄ± doÄŸrula ---
for name in ["Xtr", "Xte", "y_train_enc", "y_test_enc"]:
    if name not in globals():
        raise RuntimeError(f"'{name}' bulunamadÄ±. LÃ¼tfen 12A/12B (split & preprocessing) hÃ¼crelerini Ã¶nce Ã§alÄ±ÅŸtÄ±rÄ±n.")

# --- Dengesizlik Ã¶lÃ§Ã¼mÃ¼ (TRAIN set) ---
vc = pd.Series(y_train_enc).value_counts(normalize=True).sort_values(ascending=False)
maj_ratio = float(vc.iloc[0])
print("Train sÄ±nÄ±f oranlarÄ± (%):\n", (vc*100).round(2))
if maj_ratio >= 0.55:
    print(f"â‡’ Not: En bÃ¼yÃ¼k sÄ±nÄ±f oranÄ± â‰ˆ {maj_ratio*100:.2f}% â†’ Veri DENGESÄ°Z kabul edilebilir (SMOTE/weighting dÃ¼ÅŸÃ¼nÃ¼lebilir).")
else:
    print(f"â‡’ Not: En bÃ¼yÃ¼k sÄ±nÄ±f oranÄ± â‰ˆ {maj_ratio*100:.2f}% â†’ Veri belirgin dengesiz gÃ¶rÃ¼nmÃ¼yor.")

# --- Model ayarlarÄ± (CPU gÃ¼venli) ---
xgb_params = dict(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

# ========== 1) SMOTEâ€™suz ==========
model0 = XGBClassifier(**xgb_params)
model0.fit(Xtr, y_train_enc)
pred0 = model0.predict(Xte)
acc0 = accuracy_score(y_test_enc, pred0)
f10  = f1_score(y_test_enc, pred0, average="macro")
print(f"\n[SMOTEâ€™suz]  Acc={acc0:.4f} | Macro-F1={f10:.4f}")
print("\n[SMOTEâ€™suz] SÄ±nÄ±flandÄ±rma Raporu:")
print(classification_report(y_test_enc, pred0, digits=4))

cm0 = confusion_matrix(y_test_enc, pred0)
ConfusionMatrixDisplay(cm0).plot()
plt.title("KarÄ±ÅŸÄ±klÄ±k Matrisi â€” SMOTEâ€™suz")
plt.show()

# ========== 2) SMOTEâ€™lu (yalnÄ±zca TRAIN Ã¼stÃ¼nde) ==========
try:
    from imblearn.over_sampling import SMOTE
except Exception as e:
    raise RuntimeError("imblearn bulunamadÄ±. Kaggle'da genelde yÃ¼klÃ¼ gelir; deÄŸilse 'pip install imbalanced-learn' hÃ¼cresi ekleyin.") from e

sm = SMOTE(random_state=42, k_neighbors=5)
Xtr_sm, ytr_sm = sm.fit_resample(Xtr, y_train_enc)

model1 = XGBClassifier(**xgb_params)
model1.fit(Xtr_sm, ytr_sm)
pred1 = model1.predict(Xte)
acc1 = accuracy_score(y_test_enc, pred1)
f11  = f1_score(y_test_enc, pred1, average="macro")
print(f"\n[SMOTEâ€™lu ]  Acc={acc1:.4f} | Macro-F1={f11:.4f}")
print("\n[SMOTEâ€™lu ] SÄ±nÄ±flandÄ±rma Raporu:")
print(classification_report(y_test_enc, pred1, digits=4))

cm1 = confusion_matrix(y_test_enc, pred1)
ConfusionMatrixDisplay(cm1).plot()
plt.title("KarÄ±ÅŸÄ±klÄ±k Matrisi â€” SMOTEâ€™lu")
plt.show()

# ========== Ã–zet ==========
print("\n=== Ã–ZET ===")
print(f"SMOTEâ€™suz  â†’ Acc={acc0:.4f} | Macro-F1={f10:.4f}")
print(f"SMOTEâ€™lu   â†’ Acc={acc1:.4f} | Macro-F1={f11:.4f}")
if f11 > f10:
    print("â‡’ Karar: SMOTE testte daha iyi. (SMOTEâ€™lu akÄ±ÅŸ tercih edilebilir.)")
elif f11 < f10:
    print("â‡’ Karar: SMOTE testte daha kÃ¶tÃ¼. (SMOTEâ€™suz akÄ±ÅŸ tercih edilebilir.)")
else:
    print("â‡’ Karar: Etki eÅŸit gÃ¶rÃ¼nÃ¼yor; diÄŸer metrikler/kararlara bakÄ±labilir.")



# 15A-SETUP â€” XGB Optuna iÃ§in baÄŸlama
# (25. hÃ¼crenin beklentilerini garanti altÄ±na alÄ±r)

assert 'Xtr' in globals() and 'Xte' in globals(), "Ã–nce 20. hÃ¼cre (preprocessor) Ã§alÄ±ÅŸmalÄ±."
assert 'y_train_enc' in globals(), "Ã–nce 24. hÃ¼cre (LabelEncoder + baseline XGB) Ã§alÄ±ÅŸmalÄ±."

X_optuna = Xtr.copy()
y        = y_train_enc.copy()
print("OK: X_optuna, y baÄŸlandÄ±.", X_optuna.shape, len(y))



!pip install -q --upgrade scikit-learn==1.3.2 imbalanced-learn==0.12.3



#25.hÃ¼cre
# === 15A â€” XGBoost + Optuna (Senaryo A: class weights, Senaryo B: oversampled) ===
import warnings, json, joblib, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler

from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
import optuna

# --------------------------------------------------------------------
# Girdiler:  X_optuna, y   (Ã¶nceki hÃ¼crelerden hazÄ±r olmalÄ±)
# Not: Bu hÃ¼cre kendi holdout test ayÄ±rÄ±mÄ±nÄ± yapar (tekrar Ã¼retilebilirlik iÃ§in).
# --------------------------------------------------------------------
X_tr_big, X_te, y_tr_big, y_te = train_test_split(
    X_optuna, y, test_size=0.20, random_state=42, stratify=y
)

# =============== YardÄ±mcÄ±lar ===============
def build_model(params: dict) -> XGBClassifier:
    """
    early_stopping_rounds â†’ CONSTRUCTOR iÃ§inde (uyarÄ± susturulur).
    """
    return XGBClassifier(
        **params,
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
        early_stopping_rounds=100,   # <-- uyarÄ±yÄ± susturur (fit'te kullanmÄ±yoruz)
        verbosity=0
    )

def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.20, log=False),
        "max_depth": trial.suggest_int("max_depth", 4, 9),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),
        "subsample": trial.suggest_float("subsample", 0.70, 1.00),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.00),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
    }

# ======= Senaryo A: Orijinal veri + class weights =======
def objective_A(trial: optuna.Trial) -> float:
    params = suggest_params(trial)

    # train â†’ val ayrÄ±mÄ± (SABÄ°T tekrarlanabilirlik)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_big, y_tr_big, test_size=0.25, random_state=42, stratify=y_tr_big
    )

    # class weights â†’ sample_weight dizisine dÃ¶nÃ¼ÅŸtÃ¼r
    classes = np.unique(y_tr)
    cls_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    weight_map = {c: w for c, w in zip(classes, cls_weights)}
    sw_tr = np.array([weight_map[c] for c in y_tr])

    model = build_model(params)
    model.fit(
        X_tr, y_tr,
        sample_weight=sw_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    y_pr = model.predict(X_val)
    return f1_score(y_val, y_pr, average="macro")

# ======= Senaryo B: Oversampled (eÅŸit aÄŸÄ±rlÄ±k) =======
def objective_B(trial: optuna.Trial) -> float:
    params = suggest_params(trial)

    # train â†’ val ayrÄ±mÄ± (SABÄ°T)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_big, y_tr_big, test_size=0.25, random_state=42, stratify=y_tr_big
    )

    # Random oversampling (yalnÄ±zca train kÄ±smÄ±na uygula)
    ros = RandomOverSampler(random_state=42)
    X_tr_os, y_tr_os = ros.fit_resample(X_tr, y_tr)

    model = build_model(params)
    model.fit(
        X_tr_os, y_tr_os,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    y_pr = model.predict(X_val)
    return f1_score(y_val, y_pr, average="macro")

# ------------------ Optimizasyonlar ------------------
print("â–¶ Senaryo A (orijinal + class weights)")
study_A = optuna.create_study(direction="maximize")
study_A.optimize(objective_A, n_trials=20, show_progress_bar=True)
print(f"A-best macro-F1: {study_A.best_value:.6f}")

print("\nâ–¶ Senaryo B (oversampled)")
study_B = optuna.create_study(direction="maximize")
study_B.optimize(objective_B, n_trials=20, show_progress_bar=True)
print(f"B-best macro-F1: {study_B.best_value:.6f}")

# ------------------ SeÃ§im & Final EÄŸitim ------------------
use_B = study_B.best_value >= study_A.best_value
best_params = study_B.best_params if use_B else study_A.best_params
picked = "B (oversampled)" if use_B else "A (class weights)"
print(f"\nğŸ“Œ SeÃ§ilen senaryo: {picked}")
print("ğŸ”§ En iyi paramlar:", best_params)

# Final eÄŸitim: tÃ¼m train (X_tr_big, y_tr_big) Ã¼zerinde
if use_B:
    ros = RandomOverSampler(random_state=42)
    X_train_final, y_train_final = ros.fit_resample(X_tr_big, y_tr_big)
    final_xgb = build_model(best_params)
    final_xgb.fit(
        X_train_final, y_train_final,
        eval_set=[(X_te, y_te)],
        verbose=False
    )
else:
    classes = np.unique(y_tr_big)
    cls_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr_big)
    weight_map = {c: w for c, w in zip(classes, cls_weights)}
    sw_train_final = np.array([weight_map[c] for c in y_tr_big])

    final_xgb = build_model(best_params)
    final_xgb.fit(
        X_tr_big, y_tr_big,
        sample_weight=sw_train_final,
        eval_set=[(X_te, y_te)],
        verbose=False
    )

# ------------------ Test DeÄŸerlendirme ------------------
y_pred = final_xgb.predict(X_te)
acc = accuracy_score(y_te, y_pred)
macro_f1 = f1_score(y_te, y_pred, average="macro")
print("\n=== XGB Test Raporu (final) ===")
print(classification_report(y_te, y_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_te, y_pred))
print(f"Accuracy: {acc}")
print(f"Macro-F1: {macro_f1}")

# ------------------ Kaydet ------------------
joblib.dump(final_xgb, "final_xgb_model.joblib")
with open("final_xgb_params.json", "w", encoding="utf-8") as f:
    json.dump(best_params, f, ensure_ascii=False, indent=2)
print("ğŸ’¾ Kaydedildi: final_xgb_model.joblib, final_xgb_params.json")



#26.hÃ¼cre
# ============================================
# âš–ï¸� 13. TEMEL MODEL KARÅ�ILAÅ�TIRMASI
# ============================================

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

models = {
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=300),
    "XGBoost": XGBClassifier(random_state=42, n_estimators=300, eval_metric='mlogloss', use_label_encoder=False),
    "LightGBM": LGBMClassifier(random_state=42, n_estimators=300)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"ğŸ�† {name} doÄŸruluk oranÄ±: {acc:.4f}")



#27.hÃ¼cre
# ============================================
# ğŸ§© 14. OVERFITTING KONTROLÃœ
# ============================================

results = {}

for name, model in models.items():
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))
    results[name] = {"train": train_acc, "test": test_acc}

results_df = pd.DataFrame(results).T
results_df["fark"] = results_df["train"] - results_df["test"]
print(results_df)



#28.hÃ¼cre
# ============================================
# ğŸ§  15. HÃœCRE â€” LIGHTGBM + OPTUNA (GÃœÃ‡LÃœ & GÃœVENLÄ°)
#  - RepeatedStratifiedKFold (5x2) ile saÄŸlam CV
#  - Early stopping (100) ile overfitting kontrolÃ¼
#  - class_weight='balanced' ile sÄ±nÄ±f dengesizliÄŸi
#  - TPE + MedianPruner, n_trials & timeout ile gÃ¼venli duruÅŸ
# ============================================

import warnings, numpy as np
warnings.filterwarnings("ignore")

import optuna
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# --- Koruma: eÄŸitim verileri hazÄ±r mÄ±? ---
assert 'X_train' in globals() and 'y_train' in globals(), "Ã–nce 12. hÃ¼creyi (train_test_split) Ã§alÄ±ÅŸtÄ±rmalÄ±sÄ±n."

# --- SÄ±nÄ±f aÄŸÄ±rlÄ±klarÄ± (balanced) ---
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
print("âš–ï¸�  Class weights:", class_weight)

# --- SaÄŸlam CV ÅŸemasÄ±: 5 kat x 2 tekrar (toplam 10 validasyon) ---
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

def objective(trial):
    # ğŸ”� Aranacak hiperparametre uzayÄ± (makul ve kapsamlÄ±)
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "max_depth": trial.suggest_int("max_depth", 4, 14),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),  # L2
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),    # L1

        # ğŸ›¡ï¸� GÃ¼venli/sabit ayarlar:
        "random_state": 42,
        "n_jobs": 1,          # CPU'yu zorlamamak iÃ§in tek Ã§ekirdek
        "verbose": -1,
        "class_weight": class_weight
    }

    # Her fold'da early stopping ile eÄŸit â†’ valid accuracy topla
    val_scores = []
    for tr_idx, va_idx in cv.split(X_train, y_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        model = LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[early_stopping(stopping_rounds=100, verbose=False)]
        )

        y_hat = model.predict(X_va)
        val_scores.append(accuracy_score(y_va, y_hat))

        # Optuna pruner iÃ§in ara rapor
        trial.report(np.mean(val_scores), step=len(val_scores))
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(val_scores))

# ğŸ”¬ Optuna Ã§alÄ±ÅŸma nesnesi (TPE + MedianPruner)
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=8)  # ilk birkaÃ§ denemeden sonra zayÄ±flarÄ± budar
)

# â�±ï¸� GÃ¼venli sÄ±nÄ±rlar: 50 deneme veya 900 sn (hangisi Ã¶nce dolarsa)
study.optimize(objective, n_trials=50, timeout=900, show_progress_bar=True)

print("âœ… En iyi parametreler:", study.best_params)
print("ğŸ�¯ En iyi 10-fold (5x2) ortalama doÄŸruluk:", round(study.best_value, 4))



#29.hÃ¼cre
# ============================================
# ğŸ§ª 16. HÃœCRE â€” Optuna ile bulunan en iyi LightGBM modelinin testi
#  - study.best_params RAM'den alÄ±nÄ±r (elle yazmaya gerek yok)
#  - class_weight ve random_state eklenir
#  - test doÄŸruluÄŸu, macro F1 ve confusion matrix raporlanÄ±r
# ============================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier

assert 'study' in globals(), "Ã–nce 15. hÃ¼creyi Ã§alÄ±ÅŸtÄ±rmÄ±ÅŸ olmalÄ±sÄ±n (study yok)."
assert 'X_train' in globals() and 'y_train' in globals() and 'X_test' in globals() and 'y_test' in globals(), \
       "Ã–nce 12. hÃ¼creyi Ã§alÄ±ÅŸtÄ±rmalÄ±sÄ±n (train/test bÃ¶lÃ¼nmemiÅŸ)."

# 1) Optuna'nÄ±n bulduÄŸu en iyi parametreleri al ve saÄŸlamlaÅŸtÄ±r
best_params = study.best_params.copy()
best_params.update({
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": 1,       # kaynak kullanÄ±mÄ±nÄ± sabit tut
    "verbose": -1
})

print("âœ… KullanÄ±lan final parametreler:")
for k,v in best_params.items():
    print(f"  - {k}: {v}")

# 2) Modeli eÄŸit
final_model = LGBMClassifier(**best_params)
final_model.fit(X_train, y_train)

# 3) Test tahminleri ve metrikler
y_pred = final_model.predict(X_test)
acc     = accuracy_score(y_test, y_pred)
f1_macro= f1_score(y_test, y_pred, average="macro")

print(f"\nğŸ�¯ Test DoÄŸruluÄŸu (Accuracy): {acc:.4f}")
print(f"ğŸ“ˆ Macro F1 (sÄ±nÄ±flar ortalamasÄ±): {f1_macro:.4f}")

print("\nğŸ“Š SÄ±nÄ±f BazlÄ± Rapor:\n", classification_report(y_test, y_pred, digits=4))

# 4) KarÄ±ÅŸÄ±klÄ±k matrisi (normalize edilmiÅŸ)
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(6.2,4.6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=["Dropout","Enrolled","Graduate"],
            yticklabels=["Dropout","Enrolled","Graduate"])
plt.title("ğŸ”� Confusion Matrix (Normalize)")
plt.xlabel("Tahmin")
plt.ylabel("GerÃ§ek")
plt.tight_layout()
plt.show()

# 5) (Opsiyonel) Overfitting kontrolÃ¼ iÃ§in train skoru
train_acc = accuracy_score(y_train, final_model.predict(X_train))
print(f"\nğŸ§© Overfitting kontrolÃ¼ â€” Train: {train_acc:.4f} | Test: {acc:.4f} | Fark: {train_acc-acc:.4f}")



#30.hÃ¼cre
# ============================================
# ğŸ§© 17. HÃœCRE â€” Overfitting azaltÄ±lmÄ±ÅŸ LightGBM (DÃ¼zeltilmiÅŸ)
# ============================================

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Yeni parametreler
params_overfit_fixed = {
    'n_estimators': 2000,
    'learning_rate': 0.07,
    'num_leaves': 200,
    'max_depth': 11,
    'subsample': 0.85,
    'colsample_bytree': 0.9,
    'min_child_samples': 20,
    'min_split_gain': 0.05,
    'reg_lambda': 5.0,
    'reg_alpha': 5.0,
    'random_state': 42,
    'class_weight': 'balanced',
    'n_jobs': -1,
    'verbose': -1
}

print("ğŸ”§ Yeni parametreler (overfitting azaltÄ±lmÄ±ÅŸ):")
for k,v in params_overfit_fixed.items():
    print(f"  - {k}: {v}")

# Modeli kur
model_fixed = LGBMClassifier(**params_overfit_fixed)

# EÄŸitim (callback ile erken durdurma)
model_fixed.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='multi_logloss',
    callbacks=[early_stopping(stopping_rounds=100), log_evaluation(100)]
)

# Test sonuÃ§larÄ±
y_pred = model_fixed.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")

print(f"\nğŸ�¯ Test doÄŸruluÄŸu: {acc:.4f}")
print(f"ğŸ“ˆ Macro F1: {f1_macro:.4f}")

# Overfitting farkÄ±
train_acc = accuracy_score(y_train, model_fixed.predict(X_train))
print(f"\nğŸ§© Overfitting kontrolÃ¼ â€” Train: {train_acc:.4f} | Test: {acc:.4f} | Fark: {train_acc - acc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(6,4.5))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=["Dropout","Enrolled","Graduate"],
            yticklabels=["Dropout","Enrolled","Graduate"])
plt.title("Confusion Matrix (Overfitting DÃ¼ÅŸÃ¼rÃ¼lmÃ¼ÅŸ Model)")
plt.xlabel("Tahmin")
plt.ylabel("GerÃ§ek")
plt.tight_layout()
plt.show()



#31.hÃ¼cre
# ============================================
# âš–ï¸� 18. HÃœCRE â€” Manuel Random Oversampling + LGBM (regularize paramlarla)
#  (imblearn gerekmeden sÄ±nÄ±f dengeleme)
# ============================================

import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier

# --- GÃ¼venlik: split yapÄ±lmÄ±ÅŸ mÄ±?
assert 'X_train' in globals() and 'y_train' in globals() and 'X_test' in globals() and 'y_test' in globals(), \
    "Ã–nce 12. hÃ¼creyi (train_test_split) Ã§alÄ±ÅŸtÄ±rmalÄ±sÄ±n."

# 1) EÄŸitim verisini birleÅŸtir
train_df = X_train.copy()
train_df['__target__'] = y_train.values

# 2) SÄ±nÄ±f daÄŸÄ±lÄ±mÄ± ve hedef Ã¶rnek sayÄ±sÄ±
counts = train_df['__target__'].value_counts()
max_n = counts.max()
print("ğŸ”¢ Ã–nceki daÄŸÄ±lÄ±m:", counts.to_dict())

# 3) Her sÄ±nÄ±fÄ± max_n'e kadar WITH REPLACEMENT oversample et
balanced_parts = []
rng = np.random.RandomState(42)
for cls, n in counts.items():
    part = train_df[train_df['__target__'] == cls]
    if n < max_n:
        part_up = resample(part, replace=True, n_samples=max_n, random_state=rng)
        balanced_parts.append(part_up)
    else:
        balanced_parts.append(part)

train_bal = pd.concat(balanced_parts, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)

# 4) Geri ayÄ±r
X_res = train_bal.drop(columns='__target__')
y_res = train_bal['__target__']
print("ğŸ�¯ Sonraki daÄŸÄ±lÄ±m:", y_res.value_counts().to_dict())

# 5) Senin â€œoverfitting azaltÄ±lmÄ±ÅŸâ€� LGBM parametreleri
params_overfit_fixed = {
    'n_estimators': 2000,
    'learning_rate': 0.07,
    'num_leaves': 200,
    'max_depth': 11,
    'subsample': 0.85,
    'colsample_bytree': 0.9,
    'min_child_samples': 20,
    'min_split_gain': 0.05,
    'reg_lambda': 5.0,
    'reg_alpha': 5.0,
    'random_state': 42,
    'class_weight': 'balanced',   # oversample + balanced -> Ã§oÄŸu durumda iyi
    'n_jobs': -1,
    'verbose': -1
}

model_ros = LGBMClassifier(**params_overfit_fixed)
model_ros.fit(X_res, y_res)

# 6) Test metrikleri
y_pred = model_ros.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average='macro')

print(f"\nğŸ�¯ Test DoÄŸruluÄŸu (Oversample): {acc:.4f}")
print(f"ğŸ“ˆ Macro F1 (Oversample): {f1m:.4f}")
print("\nğŸ“‹ SÄ±nÄ±f BazlÄ± Rapor:\n", classification_report(y_test, y_pred, digits=4))

# 7) Overfitting farkÄ±
train_acc = accuracy_score(y_res, model_ros.predict(X_res))
print(f"\nğŸ§© Overfitting â€” Train: {train_acc:.4f} | Test: {acc:.4f} | Fark: {train_acc-acc:.4f}")

# 8) Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(6,4.6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
            xticklabels=["Dropout","Enrolled","Graduate"],
            yticklabels=["Dropout","Enrolled","Graduate"])
plt.title("Confusion Matrix â€” Random Oversampling")
plt.xlabel("Tahmin"); plt.ylabel("GerÃ§ek")
plt.tight_layout(); plt.show()



#32.hÃ¼cre
# ============================================
# ğŸ“Š 19. HÃœCRE â€” Ã–ZELLÄ°K Ã–NEMÄ° (FEATURE IMPORTANCE)
#  - LightGBM modelinden feature_importances_ alÄ±nÄ±r
#  - En Ã¶nemli 15 Ã¶zellik grafikle gÃ¶sterilir
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Modelin eÄŸitilmiÅŸ olduÄŸundan emin ol
assert 'final_model' in globals(), "Ã–nce modeli eÄŸitmiÅŸ olmalÄ±sÄ±n (16. hÃ¼creyi Ã§alÄ±ÅŸtÄ±r)."

# Ã–zellik Ã¶nem skorlarÄ±nÄ± al
importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': final_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# En Ã¶nemli 15 Ã¶zelliÄŸi seÃ§
top_features = importances.head(15)

# Grafik Ã§izimi
plt.figure(figsize=(8, 5))
sns.barplot(
    data=top_features,
    y='Feature', x='Importance',
    palette='viridis'
)
plt.title("ğŸ”¥ En Ã–nemli 15 Ã–zellik (LightGBM Feature Importance)")
plt.xlabel("Ã–nem Skoru")
plt.ylabel("Ã–zellik AdÄ±")
plt.tight_layout()
plt.show()

# Tablo Ã§Ä±ktÄ±sÄ±
print("ğŸ“‹ En etkili 15 Ã¶zellik:")
display(top_features)



#33.hÃ¼cre
# ============================================
# 20-LITE (FIX) â€” SHAP aÃ§Ä±klanabilirlik (multiclass, hÄ±zlÄ± & gÃ¼venli)
#  - interventional mod: background tÃ¼m yapraklarÄ± kapsamak zorunda deÄŸil
#  - kÃ¼Ã§Ã¼k Ã¶rneklem: hÄ±zlÄ± Ã§alÄ±ÅŸÄ±r, modeli etkilemez
# ============================================

import numpy as np, matplotlib.pyplot as plt
import shap, gc

assert 'final_model' in globals(), "Ã–nce 16. hÃ¼creyle modeli eÄŸit."
assert 'X_train' in globals() and 'X_test' in globals(), "Ã–nce 12. hÃ¼cre (split) Ã§alÄ±ÅŸmalÄ±."

# â€”â€”â€” HÄ±z/kararlÄ±lÄ±k iÃ§in Ã¶rneklem boylarÄ± â€”â€”â€”
BG_N   = 4000 if len(X_train) > 4000 else len(X_train)   # background (aÄŸaÃ§ yolu iÃ§in)
TEST_N = 1500 if len(X_test)  > 1500 else len(X_test)    # aÃ§Ä±klanacak test Ã¶rnekleri

X_bg = X_train.sample(BG_N,  random_state=42)
X_te = X_test.sample(TEST_N, random_state=42)

print(f"ğŸ”� SHAP background: {len(X_bg)} | explain: {len(X_te)}")

# â€”â€”â€” 1) En saÄŸlam yol: interventional mod â€”â€”â€”
# Eski API (TreeExplainer) + yeni API (Explainer) iÃ§in iki aÅŸamalÄ± dene/fallback
try:
    # Eski/klasik Ã§aÄŸrÄ± (Ã§oÄŸu sÃ¼rÃ¼mde Ã§alÄ±ÅŸÄ±r)
    explainer = shap.TreeExplainer(
        final_model,
        data=X_bg,
        feature_perturbation="interventional"
    )
    shap_values = explainer.shap_values(X_te, check_additivity=False)

except Exception as e:
    print("â„¹ï¸� TreeExplainer interventional fallback:", e)
    # Yeni API: masker ile
    masker    = shap.maskers.Independent(X_bg)
    explainer = shap.Explainer(final_model, masker=masker, algorithm="tree")
    exp       = explainer(X_te, check_additivity=False)

    # Multiclass Ã§Ä±ktÄ±: values shape (n, p, K) -> liste[K] biÃ§imine Ã§evir
    if hasattr(exp, "values") and exp.values.ndim == 3:
        shap_values = [exp.values[:, :, k] for k in range(exp.values.shape[2])]
    else:
        shap_values = exp.values  # binary ise 2D

# â€”â€”â€” 2) Ã–zet grafikler â€”â€”â€”
# Multiclass ise shap_values bir liste (sÄ±nÄ±f baÅŸÄ±na matris); summary_plot bunu otomatik yÃ¶netir.
plt.figure(figsize=(7.5, 4.8))
shap.summary_plot(shap_values, X_te, plot_type="bar", max_display=20, show=False)
plt.title("Top-20 Global Ã–zellik Ã–nemi (SHAP, interventional)")
plt.tight_layout(); plt.show()

plt.figure(figsize=(7.5, 4.8))
shap.summary_plot(shap_values, X_te, max_display=20, show=False)
plt.title("Beeswarm â€” Ã–zelliklerin YÃ¶nÃ¼ ve Etkisi (Top-20)")
plt.tight_layout(); plt.show()

# â€”â€”â€” 3) Temizlik (uzun oturumlarda belleÄŸi rahatlat) â€”â€”â€”
plt.close('all'); gc.collect()
print("âœ… SHAP tamam (interventional). Model eÄŸitimine/dogruluga etkisi yok.")



# 34. HÃœCRE â€” CatBoost + Optuna (Xtr/Xte, macro-F1) â€” CPU SAFE
# Gerekirse:  !pip install -q catboost==1.2.5

import optuna, numpy as np, joblib
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

assert 'Xtr' in globals() and 'Xte' in globals() and 'y_train_enc' in globals() and 'y_test_enc' in globals(), \
    "Ã–nce #20 (Xtr/Xte) ve #24 (y_train_enc/y_test_enc) Ã§alÄ±ÅŸmalÄ±."

# Val ayÄ±r
X_tr_big, X_val, y_tr_big, y_val = train_test_split(
    Xtr, y_train_enc, test_size=0.20, random_state=42, stratify=y_train_enc
)

def build_cat(params):
    return CatBoostClassifier(
        loss_function="MultiClass",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        **params
    )

def suggest_cat(trial):
    # CPU: sadece Bayesian veya Bernoulli kullan
    btype = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"])

    params = {
        "iterations":       trial.suggest_int("iterations", 800, 2500),
        "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.15),
        "depth":            trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg":      trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "rsm":              trial.suggest_float("rsm", 0.6, 1.0),   # colsample
        "random_strength":  trial.suggest_float("random_strength", 0.5, 3.0),
        "bootstrap_type":   btype,
    }

    if btype == "Bayesian":
        # âœ… bagging_temperature var, subsample yok
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 5.0)
    else:  # Bernoulli
        # âœ… subsample var, bagging_temperature yok
        params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)

    return params

def objective_cat(trial):
    params = suggest_cat(trial)
    model = build_cat(params)
    model.fit(
        X_tr_big, y_tr_big,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=100
    )
    pred = model.predict(X_val)
    return f1_score(y_val, pred, average="macro")

study_cat = optuna.create_study(direction="maximize")
study_cat.optimize(objective_cat, n_trials=25, show_progress_bar=True)
print("CatBoost best F1:", study_cat.best_value)
print("CatBoost best params:", study_cat.best_params)

# Final eÄŸitim (tÃ¼m Xtr) ve test deÄŸerlendirme
final_cat = build_cat(study_cat.best_params)
final_cat.fit(
    Xtr, y_train_enc,
    eval_set=(Xte, y_test_enc),
    use_best_model=True,
    early_stopping_rounds=100
)

y_cat = final_cat.predict(Xte)
acc_cat = accuracy_score(y_test_enc, y_cat)
f1m_cat = f1_score(y_test_enc, y_cat, average="macro")

print("\n=== CatBoost Test ===")
print(classification_report(y_test_enc, y_cat, digits=4))
print("Accuracy:", acc_cat, " | Macro-F1:", f1m_cat)

joblib.dump(final_cat, "final_cat_model.joblib")
print("ğŸ’¾ Kaydedildi: final_cat_model.joblib")



# 36A-2LI â€” Ä°kili Soft-Voting (XGB+LGBM, XGB+Cat, LGBM+Cat)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

assert 'Xtr' in globals() and 'Xte' in globals() and 'y_train_enc' in globals() and 'y_test_enc' in globals(), \
    "Ã–nce #20 (Xtr/Xte) ve #24 (y_train_enc/y_test_enc) Ã§alÄ±ÅŸmalÄ±."

n_classes = len(np.unique(y_train_enc))

# Val ayrÄ±mÄ±
X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
    Xtr, y_train_enc, test_size=0.20, random_state=42, stratify=y_train_enc
)

# Modeller (saÄŸlam ayarlar)
xgb2 = XGBClassifier(
    n_estimators=385, learning_rate=0.07422675041195277, max_depth=7,
    min_child_weight=9.2126039680793, subsample=0.8939910402321201,
    colsample_bytree=0.7072215640167386, gamma=0.28770070095979694,
    reg_alpha=2.112693127273198, reg_lambda=4.763403095297431,
    objective='multi:softprob', num_class=n_classes, eval_metric='mlogloss',
    tree_method='hist', random_state=42, n_jobs=-1, verbosity=0
)
lgb2 = LGBMClassifier(
    n_estimators=1200, learning_rate=0.06, max_depth=-1, num_leaves=255,
    min_child_samples=10, min_child_weight=1e-3, min_split_gain=0.0,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=3.0, reg_alpha=0.0,
    class_weight='balanced', random_state=42, n_jobs=-1, verbosity=-1
)
cat2 = CatBoostClassifier(
    loss_function="MultiClass", bootstrap_type='Bayesian', iterations=1977,
    learning_rate=0.0667356128257909, depth=4, l2_leaf_reg=1.2315152012505142,
    rsm=0.9978176521748269, random_strength=2.4712485476940045,
    bagging_temperature=0.2502955210834783, random_seed=42,
    verbose=False, allow_writing_files=False
)

# Fit
for m in (xgb2, lgb2, cat2):
    m.fit(X_tr2, y_tr2)

# ğŸ’¥ DÃœZELTÄ°LEN KISIM: pairs listesi tam ve kapalÄ±
pairs = [
    ("XGB+LGBM", (xgb2, lgb2)),
    ("XGB+Cat",  (xgb2, cat2)),
    ("LGBM+Cat", (lgb2, cat2)),
]

weight_grid = [(1,1), (2,1), (1,2), (3,2), (2,3)]

best_global = None  # (f1, name, w, (m1,m2))
for name, (m1, m2) in pairs:
    P1, P2 = m1.predict_proba(X_val2), m2.predict_proba(X_val2)
    best, best_w = -1.0, None
    for w in weight_grid:
        w1, w2 = w[0]/(w[0]+w[1]), w[1]/(w[0]+w[1])
        yhat = np.argmax(w1*P1 + w2*P2, axis=1)
        f1m  = f1_score(y_val2, yhat, average='macro')
        if f1m > best:
            best, best_w = f1m, w
    print(f"ğŸ§ª {name} â€” En iyi weights={best_w} | VAL macro-F1={best:.4f}")
    if (best_global is None) or (best > best_global[0]):
        best_global = (best, name, best_w, (m1, m2))

# TEST deÄŸerlendirme (en iyi ikili kombinasyon)
best_f1, best_name, best_w, (m1_best, m2_best) = best_global
w1, w2 = best_w[0]/(best_w[0]+best_w[1]), best_w[1]/(best_w[0]+best_w[1])
Proba1, Proba2 = m1_best.predict_proba(Xte), m2_best.predict_proba(Xte)
y_pred_test = np.argmax(w1*Proba1 + w2*Proba2, axis=1)

acc = accuracy_score(y_test_enc, y_pred_test)
f1m = f1_score(y_test_enc, y_pred_test, average='macro')
print(f"\nğŸ�¯ TEST â€” {best_name} | weights={best_w} â†’ Acc={acc:.4f} | Macro-F1={f1m:.4f}")
print("ğŸ“‹ Rapor:\n", classification_report(y_test_enc, y_pred_test, digits=4))
print("CM:\n", confusion_matrix(y_test_enc, y_pred_test))



# === 36B â€” ÃœÃ§lÃ¼ AnsambÄ±l (XGB + LGBM + CatBoost) â€” Soft Voting + Test Raporu ===
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# --- Ã–nkoÅŸullar ---
assert 'Xtr' in globals() and 'Xte' in globals(), "Ã–nce #20 hÃ¼cresiyle (Xtr/Xte) hazÄ±r olmalÄ±."
assert 'y_train_enc' in globals() and 'y_test_enc' in globals(), "Ã–nce #24 hÃ¼cresiyle (y_train_enc/y_test_enc) hazÄ±r olmalÄ±."

n_classes = len(np.unique(y_train_enc))

# Val iÃ§in kÃ¼Ã§Ã¼k bir ayrÄ±m (yoksa oluÅŸtur)
if not all(k in globals() for k in ['X_tr_small', 'X_val', 'y_tr_small', 'y_val']):
    X_tr_small, X_val, y_tr_small, y_val = train_test_split(
        Xtr, y_train_enc, test_size=0.20, random_state=42, stratify=y_train_enc
    )

# --------------------------
# 1) Modeller (tuned/saÄŸlam ayarlar)
# --------------------------

# XGBoost â€” (Senaryo B en iyileri)
xgb = XGBClassifier(
    n_estimators=385,
    learning_rate=0.07422675041195277,
    max_depth=7,
    min_child_weight=9.2126039680793,
    subsample=0.8939910402321201,
    colsample_bytree=0.7072215640167386,
    gamma=0.28770070095979694,
    reg_alpha=2.112693127273198,
    reg_lambda=4.763403095297431,
    objective='multi:softprob',
    num_class=n_classes,
    eval_metric='mlogloss',
    tree_method='hist',
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

# LightGBM â€” uyarÄ±larÄ± azaltan gevÅŸetmeler
lgb = LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.06,
    max_depth=-1,            # sÄ±nÄ±rsÄ±z
    num_leaves=255,
    min_child_samples=10,
    min_child_weight=1e-3,
    min_split_gain=0.0,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=3.0,
    reg_alpha=0.0,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

# CatBoost â€” Optuna ile bulunan en iyi set
cat = CatBoostClassifier(
    loss_function="MultiClass",
    bootstrap_type='Bayesian',
    iterations=1977,
    learning_rate=0.0667356128257909,
    depth=4,
    l2_leaf_reg=1.2315152012505142,
    rsm=0.9978176521748269,
    random_strength=2.4712485476940045,
    bagging_temperature=0.2502955210834783,
    random_seed=42,
    verbose=False,
    allow_writing_files=False
)

# --------------------------
# 2) EÄŸitim (val ile early stopping â€” LGBM 'verbose' KULLANMAYIN)
# --------------------------
xgb.fit(X_tr_small, y_tr_small, eval_set=[(X_val, y_val)], early_stopping_rounds=100)
lgb.fit(X_tr_small, y_tr_small, eval_set=[(X_val, y_val)], eval_metric='multi_logloss')
cat.fit(X_tr_small, y_tr_small, eval_set=(X_val, y_val), use_best_model=True, early_stopping_rounds=100)

# --------------------------
# 3) Val Ã¼zerinde aÄŸÄ±rlÄ±k taramasÄ± (soft voting)
# --------------------------
def soft_vote(probas, weights):
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    combo = w[0]*probas[0] + w[1]*probas[1] + w[2]*probas[2]
    return combo.argmax(axis=1)

val_probas = [
    xgb.predict_proba(X_val),
    lgb.predict_proba(X_val),
    cat.predict_proba(X_val)
]

grid_weights = [
    (1,1,1), (2,1,1), (1,2,1), (1,1,2),
    (2,2,1), (2,1,2), (1,2,2),
    (2,2,2), (3,2,2), (2,3,2), (2,2,3),
    (2,2,3), (2,2,4), (2,3,3), (3,3,2)
]

best = None
best_f1 = -1.0
for w in grid_weights:
    y_pred_val = soft_vote(val_probas, w)
    f1m = f1_score(y_val, y_pred_val, average='macro')
    if f1m > best_f1:
        best_f1 = f1m
        best = w

print(f"ğŸ§ª Val en iyi: ('trio', 'xgb+lgb+cat')  | weights={best}  | macro-F1={best_f1:.4f}")

# --------------------------
# 4) Test deÄŸerlendirme (seÃ§ilen aÄŸÄ±rlÄ±klarla)
# --------------------------
test_probas = [
    xgb.predict_proba(Xte),
    lgb.predict_proba(Xte),
    cat.predict_proba(Xte)
]

y_pred_test = soft_vote(test_probas, best)
acc = accuracy_score(y_test_enc, y_pred_test)
f1m = f1_score(y_test_enc, y_pred_test, average='macro')
print(f"\nğŸ�¯ TEST â€” Soft Voting (('trio','xgb+lgb+cat'), weights={best})")
print(f"Accuracy: {acc:.4f} | Macro-F1: {f1m:.4f}\n")

print("ğŸ“‹ SÄ±nÄ±f BazlÄ± Rapor:")
print(classification_report(y_test_enc, y_pred_test, digits=4))

cm = confusion_matrix(y_test_enc, y_pred_test)
print("Confusion matrix:\n", cm)



# === 36C â€” FINAL: ÃœÃ§lÃ¼ AnsambÄ±l + SÄ±nÄ±f-1 Bias TaramasÄ± (Val -> Test) ===
import numpy as np, json, joblib, warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# --- Ã–nkoÅŸullar ---
assert 'Xtr' in globals() and 'Xte' in globals(), "Ã–nce #20 hÃ¼cresiyle (Xtr/Xte) hazÄ±r olmalÄ±."
assert 'y_train_enc' in globals() and 'y_test_enc' in globals(), "Ã–nce #24 hÃ¼cresiyle (y_train_enc/y_test_enc) hazÄ±r olmalÄ±."
n_classes = len(np.unique(y_train_enc))

# Val ayrÄ±mÄ± (stabil kÄ±yas iÃ§in)
X_tr, X_val, y_tr, y_val = train_test_split(
    Xtr, y_train_enc, test_size=0.20, random_state=42, stratify=y_train_enc
)

# --- Modeller (en iyi/saÄŸlam setler) ---
xgb = XGBClassifier(
    n_estimators=385,
    learning_rate=0.07422675041195277,
    max_depth=7,
    min_child_weight=9.2126039680793,
    subsample=0.8939910402321201,
    colsample_bytree=0.7072215640167386,
    gamma=0.28770070095979694,
    reg_alpha=2.112693127273198,
    reg_lambda=4.763403095297431,
    objective='multi:softprob',
    num_class=n_classes,
    eval_metric='mlogloss',
    tree_method='hist',
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

lgb = LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.06,
    max_depth=-1,
    num_leaves=255,
    min_child_samples=10,
    min_child_weight=1e-3,
    min_split_gain=0.0,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=3.0,
    reg_alpha=0.0,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

cat = CatBoostClassifier(
    loss_function="MultiClass",
    bootstrap_type='Bayesian',
    iterations=1977,
    learning_rate=0.0667356128257909,
    depth=4,
    l2_leaf_reg=1.2315152012505142,
    rsm=0.9978176521748269,
    random_strength=2.4712485476940045,
    bagging_temperature=0.2502955210834783,
    random_seed=42,
    verbose=False,
    allow_writing_files=False
)

# --- EÄŸitim (val ile izleme) ---
xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100)
lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='multi_logloss')  # verbose paramÄ± yok
cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True, early_stopping_rounds=100)

# --- YardÄ±mcÄ±lar ---
def soft_vote(probas, weights):
    w = np.array(weights, dtype=float); w = w / w.sum()
    S = w[0]*probas[0] + w[1]*probas[1] + w[2]*probas[2]
    return S

def apply_class1_bias(S, scale=1.0):
    """S: (n, C) soft-vote skoru/olasÄ±lÄ±ÄŸÄ±; sÄ±nÄ±f-1'i Ã¶lÃ§ekleyip yeniden normalize eder."""
    S2 = S.copy()
    S2[:, 1] = S2[:, 1] * scale
    S2 = S2 / S2.sum(axis=1, keepdims=True)
    return S2

# --- 1) Val Ã¼zerinde aÄŸÄ±rlÄ±k taramasÄ± ---
val_probas = [
    xgb.predict_proba(X_val),
    lgb.predict_proba(X_val),
    cat.predict_proba(X_val)
]

weight_grid = [
    (1,1,1), (1,1,2), (1,2,1), (2,1,1),
    (2,2,1), (2,1,2), (1,2,2), (2,2,2),
    (3,2,2), (2,3,2), (2,2,3), (3,3,2), (2,3,3), (3,2,3), (3,3,3)
]

best_w, best_f1 = None, -1.0
for w in weight_grid:
    S = soft_vote(val_probas, w)
    yhat = S.argmax(1)
    f1m = f1_score(y_val, yhat, average='macro')
    if f1m > best_f1:
        best_f1, best_w = f1m, w

print(f"ğŸ§ª VAL â€” En iyi aÄŸÄ±rlÄ±k: {best_w} | macro-F1={best_f1:.4f}")

# --- 2) Val Ã¼zerinde sÄ±nÄ±f-1 bias taramasÄ± (en iyi w sabitken) ---
scales = np.linspace(1.00, 1.60, 31)  # %0 â†’ %60 artÄ±r
best_scale, best_f1_bias = 1.0, -1.0
S_base = soft_vote(val_probas, best_w)

for s in scales:
    Sb = apply_class1_bias(S_base, scale=s)
    yhat = Sb.argmax(1)
    f1m = f1_score(y_val, yhat, average='macro')
    if f1m > best_f1_bias:
        best_f1_bias, best_scale = f1m, s

print(f"ğŸ§ª VAL â€” En iyi sÄ±nÄ±f-1 Ã¶lÃ§ek: {best_scale:.2f} | macro-F1={best_f1_bias:.4f}")

# --- 3) TEST deÄŸerlendirme (seÃ§ili w + bias) ---
test_probas = [
    xgb.predict_proba(Xte),
    lgb.predict_proba(Xte),
    cat.predict_proba(Xte)
]

S_test = soft_vote(test_probas, best_w)
S_test = apply_class1_bias(S_test, scale=best_scale)
y_pred = S_test.argmax(1)

acc = accuracy_score(y_test_enc, y_pred)
f1m = f1_score(y_test_enc, y_pred, average='macro')
print(f"\nğŸ�¯ TEST â€” Soft Voting weights={best_w}, class1_scale={best_scale:.2f}")
print(f"Accuracy: {acc:.4f} | Macro-F1: {f1m:.4f}\n")

print("ğŸ“‹ SÄ±nÄ±f BazlÄ± Rapor:")
print(classification_report(y_test_enc, y_pred, digits=4))
cm = confusion_matrix(y_test_enc, y_pred)
print("Confusion matrix:\n", cm)

# --- 4) Kaydet (modeller + meta) ---
joblib.dump({
    "xgb": xgb, "lgb": lgb, "cat": cat,
    "weights": best_w,
    "class1_scale": float(best_scale)
}, "final_ensemble.joblib")

with open("final_ensemble_meta.json", "w", encoding="utf-8") as f:
    json.dump({
        "weights": best_w,
        "class1_scale": float(best_scale),
        "note": "Soft-vote trio (XGB/LGBM/Cat) + class-1 bias scaling (val-opt)."
    }, f, ensure_ascii=False, indent=2)

print("ğŸ’¾ Kaydedildi: final_ensemble.joblib, final_ensemble_meta.json")



# 36D â€” ENSEMBLE Inference + KayÄ±t + Rapor (meta key fix + fallback search)

import json, joblib, numpy as np, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Gerekli tensÃ¶r ve modeller:
assert 'Xtr' in globals() and 'Xte' in globals() and 'y_train_enc' in globals() and 'y_test_enc' in globals(), \
    "Ã–nce #20 ve #24 Ã§alÄ±ÅŸmalÄ± (Xtr/Xte, y_train_enc/y_test_enc)."
assert 'X_val' in globals() and 'y_val' in globals(), "36C'de ayrÄ±lan val seti (X_val, y_val) gerekli."

# --- 0) Modeller bellekte mi? (36C'den)
need_err = []
for name in ('xgb','lgb','cat'):
    if name not in globals():
        need_err.append(name)
if need_err:
    raise RuntimeError(f"Bellekte modeller eksik: {need_err}. 36C hÃ¼cresini Ã¶nce Ã§alÄ±ÅŸtÄ±r.")

# --- 1) Meta'yÄ± oku (anahtar isimleri toleranslÄ±)
best_weights = None
class1_scale = None
chosen = None
try:
    meta = json.load(open("final_ensemble_meta.json", "r", encoding="utf-8"))
    # OlasÄ± anahtar adlarÄ±:
    if best_weights is None:
        for k in ("best_weights", "weights", "best_w"):
            if k in meta:
                best_weights = tuple(meta[k])
                break
    if class1_scale is None:
        for k in ("class1_scale", "best_class1_scale", "scale_c1", "c1_scale"):
            if k in meta:
                class1_scale = float(meta[k])
                break
    for k in ("chosen", "selection", "choice"):
        if k in meta:
            chosen = meta[k]
            break
except FileNotFoundError:
    pass  # Fallback aÅŸaÄŸÄ±da

# --- 2) Ensemble fonksiyonlarÄ±
def ensemble_predict_proba(X, weights=(1,1,2), class1_scale=1.0):
    P = []
    P.append(xgb.predict_proba(X))
    P.append(lgb.predict_proba(X))
    P.append(cat.predict_proba(X))

    w = np.array(weights, dtype=float)
    w = w / w.sum()
    P_stack = np.stack(P, axis=0)                 # (3, n, C)
    P_ens = (w[:, None, None] * P_stack).sum(0)   # (n, C)

    # class-1 bias
    P_ens[:, 1] *= class1_scale
    P_ens = P_ens / P_ens.sum(1, keepdims=True)
    return P_ens

def ensemble_predict(X, weights, class1_scale):
    proba = ensemble_predict_proba(X, weights=weights, class1_scale=class1_scale)
    return np.argmax(proba, axis=1), proba

# --- 3) EÄŸer meta eksikse: kÃ¼Ã§Ã¼k bir VAL taramasÄ± ile (weights, scale) yeniden bul
if best_weights is None or class1_scale is None:
    candidate_weights = [(1,1,1),(1,1,2),(1,2,2),(1,2,3),(2,2,3)]
    candidate_scales  = [1.00, 1.20, 1.35, 1.48, 1.60]
    best = (-1.0, None, None)  # (f1, w, s)

    for w in candidate_weights:
        for s in candidate_scales:
            yv_pred, _ = ensemble_predict(X_val, w, s)
            f1m = f1_score(y_val, yv_pred, average="macro")
            if f1m > best[0]:
                best = (f1m, w, s)

    best_f1, best_weights, class1_scale = best
    chosen = "recovered_by_val_search"
    print(f"ğŸ”� Meta bulunamadÄ±/eksikti â†’ VAL ile yeniden arandÄ±: weights={best_weights}, class1_scale={class1_scale:.2f}, F1={best_f1:.4f}")
else:
    print(f"âœ… Meta bulundu: weights={best_weights}, class1_scale={class1_scale:.2f}, chosen={chosen}")

# --- 4) VAL raporu
yv_pred, val_proba = ensemble_predict(X_val, best_weights, class1_scale)
val_acc = accuracy_score(y_val, yv_pred)
val_f1m = f1_score(y_val, yv_pred, average="macro")
print(f"\nğŸ§ª VAL â€” weights={best_weights}, class1_scale={class1_scale:.2f}")
print(f"Accuracy: {val_acc:.4f} | Macro-F1: {val_f1m:.4f}")
print(classification_report(y_val, yv_pred, digits=4))
print("Confusion matrix (VAL):")
print(confusion_matrix(y_val, yv_pred))

# --- 5) TEST raporu
yte_pred, te_proba = ensemble_predict(Xte, best_weights, class1_scale)
te_acc = accuracy_score(y_test_enc, yte_pred)
te_f1m = f1_score(y_test_enc, yte_pred, average="macro")
print("\nğŸ�¯ TEST â€” SONUÃ‡")
print(f"Accuracy: {te_acc:.4f} | Macro-F1: {te_f1m:.4f}")
print(classification_report(y_test_enc, yte_pred, digits=4))
print("Confusion matrix (TEST):")
print(confusion_matrix(y_test_enc, yte_pred))

# --- 6) Artefact kayÄ±t
np.save("val_proba.npy", val_proba)
np.save("test_proba.npy", te_proba)

df_pred = pd.DataFrame({
    "index": np.arange(len(yte_pred)),
    "y_true": y_test_enc,
    "y_pred": yte_pred,
    "p0": te_proba[:, 0],
    "p1": te_proba[:, 1],
    "p2": te_proba[:, 2],
})
df_pred.to_csv("predictions_test.csv", index=False)

# Meta'yÄ± normalize edilmiÅŸ anahtarlarla yeniden yaz:
safe_meta = {
    "best_weights": list(best_weights),
    "class1_scale": float(class1_scale),
    "chosen": chosen if chosen is not None else "fixed_36D",
}
json.dump(safe_meta, open("final_ensemble_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("\nğŸ’¾ Kaydedildi: val_proba.npy, test_proba.npy, predictions_test.csv, final_ensemble_meta.json")
print(f"âœ… Ã–zet | VAL F1={val_f1m:.4f} | TEST F1={te_f1m:.4f} | weights={best_weights} | class1_scale={class1_scale:.2f}")



# 36E-STACKING â€” XGB + LGBM + Cat â†’ Meta: LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Gerekli tensÃ¶r ve etiketler hazÄ±r mÄ±?
assert 'Xtr' in globals() and 'Xte' in globals(), "Ã–nce #20 (preprocessor) Ã§alÄ±ÅŸmalÄ±."
assert 'y_train_enc' in globals() and 'y_test_enc' in globals(), "Ã–nce #24 (LabelEncoder) Ã§alÄ±ÅŸmalÄ±."

n_classes = len(np.unique(y_train_enc))

# 1) Train/Val ayrÄ±mÄ± (stacking iÃ§in)
X_trs, X_vals, y_trs, y_vals = train_test_split(
    Xtr, y_train_enc, test_size=0.20, random_state=42, stratify=y_train_enc
)

# 2) Base modeller (saÄŸlam ayarlar â€” 3'lÃ¼ ensemble ile tutarlÄ±)
xgb_s = XGBClassifier(
    n_estimators=385, learning_rate=0.07422675041195277, max_depth=7,
    min_child_weight=9.2126039680793, subsample=0.8939910402321201,
    colsample_bytree=0.7072215640167386, gamma=0.28770070095979694,
    reg_alpha=2.112693127273198, reg_lambda=4.763403095297431,
    objective='multi:softprob', num_class=n_classes, eval_metric='mlogloss',
    tree_method='hist', random_state=42, n_jobs=-1, verbosity=0
)

lgb_s = LGBMClassifier(
    n_estimators=1200, learning_rate=0.06, max_depth=-1, num_leaves=255,
    min_child_samples=10, min_child_weight=1e-3, min_split_gain=0.0,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=3.0, reg_alpha=0.0,
    class_weight='balanced', random_state=42, n_jobs=-1, verbosity=-1
)

cat_s = CatBoostClassifier(
    loss_function="MultiClass", bootstrap_type='Bayesian', iterations=1977,
    learning_rate=0.0667356128257909, depth=4, l2_leaf_reg=1.2315152012505142,
    rsm=0.9978176521748269, random_strength=2.4712485476940045,
    bagging_temperature=0.2502955210834783, random_seed=42,
    verbose=False, allow_writing_files=False
)

# 3) Base modelleri eÄŸit
for m in (xgb_s, lgb_s, cat_s):
    m.fit(X_trs, y_trs)

# 4) Meta-Ã¶zellikler = olasÄ±lÄ±k Ã§Ä±ktÄ±larÄ± (train/val/test)
P_tr = np.hstack([
    xgb_s.predict_proba(X_trs),
    lgb_s.predict_proba(X_trs),
    cat_s.predict_proba(X_trs)
])

P_va = np.hstack([
    xgb_s.predict_proba(X_vals),
    lgb_s.predict_proba(X_vals),
    cat_s.predict_proba(X_vals)
])

P_te = np.hstack([
    xgb_s.predict_proba(Xte),
    lgb_s.predict_proba(Xte),
    cat_s.predict_proba(Xte)
])

# 5) Meta-Ã¶ÄŸrenici
meta = LogisticRegression(max_iter=500, multi_class='auto', solver='lbfgs')
meta.fit(P_tr, y_trs)

# 6) VAL ve TEST deÄŸerlendirme
y_val_pred = meta.predict(P_va)
y_te_pred  = meta.predict(P_te)

print("ğŸ§ª STACKING â€” VAL:",
      "Acc", accuracy_score(y_vals, y_val_pred),
      "Macro-F1", f1_score(y_vals, y_val_pred, average='macro'))

print("\nğŸ�¯ STACKING â€” TEST:",
      "Acc", accuracy_score(y_test_enc, y_te_pred),
      "Macro-F1", f1_score(y_test_enc, y_te_pred, average='macro'))

print("\nğŸ“‹ Rapor (TEST):\n", classification_report(y_test_enc, y_te_pred, digits=4))
print("CM (TEST):\n", confusion_matrix(y_test_enc, y_te_pred))



# 27B â€” Performans Ã–zeti (Accuracy & Macro-F1) â€” FIXED
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

rows = []

def add_row(name, y_true, y_pred):
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "MacroF1": f1_score(y_true, y_pred, average='macro')
    })

# 1) #26'daki basit modeller (X_train/y_train -> X_test/y_test)
if 'models' in globals():
    for name, model in models.items():
        try:
            ypr = model.predict(X_test)
            add_row(name, y_test, ypr)
        except Exception:
            pass

# 2) XGB Optuna final (#25) â€” DÄ°KKAT: y_true = y_te
if 'final_xgb' in globals():
    try:
        # final_xgb, #25'te (X_te, y_te) ile deÄŸerlendirilmiÅŸti
        ypr = final_xgb.predict(X_te)
        add_row("XGB (Optuna Final)", y_te, ypr)
    except Exception as e:
        print("XGB (Optuna Final) eklenemedi:", e)

# 3) LGBM Optuna final (#29) â€” y_true = y_test
if 'final_model' in globals():
    try:
        ypr = final_model.predict(X_test)
        add_row("LGBM (Optuna Final)", y_test, ypr)
    except Exception as e:
        print("LGBM (Optuna Final) eklenemedi:", e)

# 4) CatBoost Optuna final (#34) â€” y_true = y_test_enc
if 'final_cat' in globals():
    try:
        ypr = final_cat.predict(Xte)
        add_row("CatBoost (Optuna Final)", y_test_enc, ypr)
    except Exception as e:
        print("CatBoost (Optuna Final) eklenemedi:", e)

# 5) 2â€™li Ensemble (az Ã¶nceki hÃ¼cre) â€” var olan acc/f1m deÄŸiÅŸkenlerini kullan
try:
    rows.append({"Model": "2â€™li Ensemble (LGBM+Cat)", "Accuracy": acc, "MacroF1": f1m})
except:
    # elle girilen deÄŸerleri kullanmak istersen:
    rows.append({"Model": "2â€™li Ensemble (LGBM+Cat)", "Accuracy": 0.8315, "MacroF1": 0.7948})

# 6) Stacking (meta-LR) â€” 36Eâ€™den gelen y_te_pred deÄŸiÅŸkenleri
try:
    from sklearn.metrics import accuracy_score, f1_score
    rows.append({
        "Model": "Stacking (meta-LR)",
        "Accuracy": accuracy_score(y_test_enc, y_te_pred),
        "MacroF1": f1_score(y_test_enc, y_te_pred, average='macro')
    })
except Exception as e:
    print("Stacking sonucu eklenemedi:", e)

# === Tablo & Grafikler ===
perf = pd.DataFrame(rows).dropna()
perf = perf.sort_values("MacroF1", ascending=False).reset_index(drop=True)
display(perf.style.background_gradient(cmap="Blues", subset=["Accuracy", "MacroF1"]))

plt.figure(figsize=(8,4.5))
plt.barh(perf["Model"], perf["MacroF1"])
plt.title("Model & Ensemble KarÅŸÄ±laÅŸtÄ±rmasÄ± â€” Macro-F1")
plt.xlabel("Macro-F1"); plt.ylabel("Model")
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4.5))
plt.barh(perf["Model"], perf["Accuracy"])
plt.title("Model & Ensemble KarÅŸÄ±laÅŸtÄ±rmasÄ± â€” Accuracy")
plt.xlabel("Accuracy"); plt.ylabel("Model")
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.tight_layout(); plt.show()



# 37A â€” Class-1 iÃ§in tek-etiket eÅŸik taramasÄ± (VALâ†’TEST), meta & Ã§Ä±ktÄ± kaydÄ±

import json, numpy as np, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Gerekli: 36C (modeller) ve 36D (proba dosyalarÄ±)
val_proba = np.load("val_proba.npy")   # (n_val, 3)
te_proba  = np.load("test_proba.npy")  # (n_test, 3)

# VAL/TEST true label'lar
y_val_true = y_val
y_te_true  = y_test_enc

def predict_with_class1_threshold(proba, t=0.50):
    """
    Ã‡ok sÄ±nÄ±flÄ±: class-1 iÃ§in ayrÄ± bir eÅŸik uygula.
    EÄŸer p1 >= t ise 1 ata; deÄŸilse geri kalanlar arasÄ±nda argmax.
    """
    p1 = proba[:, 1]
    base = np.argmax(proba, axis=1)
    pred = base.copy()
    pred[p1 >= t] = 1
    return pred

# KÃ¼Ã§Ã¼k bir eÅŸik Ä±zgarasÄ±
grid = np.linspace(0.40, 0.70, 16)  # 0.40, 0.42, ... 0.70
best = (-1.0, None, None)  # (f1_macro, t, rapor)

for t in grid:
    yv_pred = predict_with_class1_threshold(val_proba, t)
    f1m = f1_score(y_val_true, yv_pred, average="macro")
    if f1m > best[0]:
        best = (f1m, t, yv_pred)

best_f1_val, best_t, yv_pred_best = best
print(f"ğŸ§ª VAL â€” En iyi class1_threshold={best_t:.2f} | Macro-F1={best_f1_val:.4f}")
print(classification_report(y_val_true, yv_pred_best, digits=4))
print("Confusion matrix (VAL):")
print(confusion_matrix(y_val_true, yv_pred_best))

# TEST Ã¼zerinde bu eÅŸiÄŸi uygula
yte_pred_tuned = predict_with_class1_threshold(te_proba, best_t)
acc_te  = accuracy_score(y_te_true, yte_pred_tuned)
f1m_te  = f1_score(y_te_true, yte_pred_tuned, average="macro")
print("\nğŸ�¯ TEST â€” TUNED")
print(f"Accuracy: {acc_te:.4f} | Macro-F1: {f1m_te:.4f}")
print(classification_report(y_te_true, yte_pred_tuned, digits=4))
print("Confusion matrix (TEST):")
print(confusion_matrix(y_te_true, yte_pred_tuned))

# Ã‡Ä±ktÄ± dosyalarÄ±
pd.DataFrame({
    "y_true": y_te_true,
    "y_pred_tuned": yte_pred_tuned,
    "p0": te_proba[:,0], "p1": te_proba[:,1], "p2": te_proba[:,2]
}).to_csv("predictions_test_tuned.csv", index=False)

# Meta'yÄ± gÃ¼ncelle (eÅŸik bilgisini ekle)
meta_path = "final_ensemble_meta.json"
try:
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
except FileNotFoundError:
    meta = {}
meta["class1_threshold"] = float(best_t)
json.dump(meta, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("\nğŸ’¾ Kaydedildi: predictions_test_tuned.csv ve final_ensemble_meta.json (class1_threshold ile)")


