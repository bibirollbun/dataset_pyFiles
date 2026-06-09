import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train_df = pd.read_csv("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv")
train_df


hastalik_listesi = train_df['class_name'].value_counts()
print("Hastalık Dağılımı:",hastalik_listesi)


# ID ve İsim eşleşmesi
category= train_df.groupby('class_id')['class_name'].unique().reset_index()

print("Class id ve Class name eşleşmesi:")
print(category)


# İlk 5 satırın yazdırılması
print("VinBigData Chest X-ray Veri Seti:")
train_df.head()


# 1. Satır ve sütunların boyutlarının kontrolü
print(f"Satır: {len(train_df)}, Sütun: {len(train_df.columns)}")

# 2. Boş değer kontrolü
print(train_df.isna().sum())
train_df.info()


temiz_df = train_df.dropna(subset=['x_min', 'x_max', 'y_min', 'y_max']).copy()

# Alan hesaplama
temiz_df['alan'] = (temiz_df['x_max'] - temiz_df['x_min']) * (temiz_df['y_max'] - temiz_df['y_min'])

maks_bulgu = temiz_df[temiz_df['alan'] == temiz_df['alan'].max()]
min_bulgu = temiz_df[temiz_df['alan'] == temiz_df['alan'].min()]

print("--- En Geniş Alanlı Vaka ---")
print(maks_bulgu)
print("\n--- En Küçük Alanlı Vaka ---")
print(min_bulgu)


df_temiz = train_df.dropna(subset=['x_min', 'x_max', 'y_min', 'y_max']).copy()

df_temiz['alan'] = (df_temiz['x_max'] - df_temiz['x_min']) * (df_temiz['y_max'] - df_temiz['y_min'])

en_buyukler = df_temiz[df_temiz['alan'] == df_temiz['alan'].max()]

print(f"En Büyük Alan: {df_temiz['alan'].max()}")
print(f"Kaç tane var: {len(en_buyukler)}")
print(f"Hangi Hastalıklar: {en_buyukler['class_name'].unique()}")
en_buyukler


# temiz_vakalar = No Finding 
temiz_vakalar = train_df[train_df['class_name'] == 'No finding']

rad_counts = temiz_vakalar['rad_id'].value_counts()

print("Radyologlara göre sağlıklı(No Finding) rapor dağılımı:")
print(rad_counts)

en_aktif_radyolog = rad_counts.idxmax()
vaka_adedi = rad_counts.max()

print(f"\nEn çok sağlıklı vaka raporlayan kişi: {en_aktif_radyolog}")
print(f"Toplam vaka sayısı: {vaka_adedi}")


# R10 = Cardiomegaly
r10_kardiyo = train_df[(train_df['class_name'] == 'Cardiomegaly') & (train_df['rad_id'] == 'R10')]

toplam = len(r10_kardiyo)
print(f"R10 ve Cardiomegaly kesişiminde {toplam} vaka var.")
r10_kardiyo.head()


# Görüntünün tam sol üst köşesine (0,0) dayanan vakaların ayıklanması
kose_vakalar = train_df[(train_df['x_min'] == 0) & (train_df['y_min'] == 0)]

print(f"Köşe koordinatlı (0,0) toplam vaka: {len(kose_vakalar)}")
print("\n Bu vakaların sınıflara göre dağılımı:")
print(kose_vakalar['class_name'].value_counts())

kose_vakalar.head()


# No finding olanları = sağlıklı 
train_df['saglikli'] = (train_df['class_name'] == 'No finding')

rad_skorlari = train_df.groupby('rad_id')['saglikli'].mean().sort_values(ascending=False)

print("Radyologların sağlıklı (No Finding) vaka saptama oranları:")
print(rad_skorlari)

en_iyi = rad_skorlari.idxmax()
en_yuksek = rad_skorlari.max() * 100

print(f"\nEn yüksek sağlıklı vaka oranı: {en_iyi} (oran: %{en_yuksek:.2f})")


df_alan = train_df.dropna(subset=['x_min', 'x_max', 'y_min', 'y_max']).copy()
df_alan['alan'] = (df_alan['x_max'] - df_alan['x_min']) * (df_alan['y_max'] - df_alan['y_min'])

ortalama_boyutlar = df_alan.groupby('class_name')['alan'].mean().sort_values(ascending=False)

print("Hastalık Türlerine Göre Ortalama Alan Dağılımı:")
print(ortalama_boyutlar)

en_yaygin = ortalama_boyutlar.idxmax()
print(f"\n Ortalama en geniş alanı kaplayan hastalık: {en_yaygin}")


# No Finding olanları 1, diğerleri 0
train_df['durum'] = (train_df['class_name'] == 'No finding').astype(int)

plt.figure()
sns.countplot(data=train_df, x='rad_id', hue='durum', palette='magma')

plt.title('Radyologlar: Sağlıklı (No Finding) (1) vs Hasta (0) Rapor Sayıları')
plt.show()


df_box = train_df.dropna(subset=['x_min', 'x_max', 'y_min', 'y_max']).copy()
df_box['alan'] = (df_box['x_max'] - df_box['x_min']) * (df_box['y_max'] - df_box['y_min'])

top5 = df_box['class_name'].value_counts().head().index
plot_data = df_box[df_box['class_name'].isin(top5)]

plt.figure()
sns.boxplot(data=plot_data, x='alan', y='class_name', palette='pastel')

plt.title('Hastalık Türlerine Göre Boyut Değişkenliği')
plt.show()


df_n = train_df.dropna(subset=['x_min', 'x_max', 'y_min', 'y_max']).copy()
df_n['alan'] = (df_n['x_max'] - df_n['x_min']) * (df_n['y_max'] - df_n['y_min'])

# Sadece sayısal sütunların birbirleriyle olan bağına (korelasyonuna) bakılması
korelasyon = df_n.corr(numeric_only=True)

plt.figure()
sns.heatmap(korelasyon, annot=True, cmap='RdYlGn')

plt.title('Değişkenler Arasındaki Bağlar (Korelasyon)')
plt.show()

