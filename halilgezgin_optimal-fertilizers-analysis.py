import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import scipy.stats as ss


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv", index_col="id")
sample = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


train.head()


train.info()


train.isnull().sum()


train.describe()


train["Soil Type"].unique()


train["Crop Type"].unique()


train["Fertilizer Name"].unique()


train.shape


test.shape


test.head()


categorical_cols = train.select_dtypes(include=['object', 'category']).columns

for col in categorical_cols:
    counts = train[col].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{col} Variable', fontsize=14)

    axes[0].bar(counts.index, counts.values, color=plt.cm.Pastel1.colors)
    axes[0].set_title('Bar Plot')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xlabel(col)
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=plt.cm.Pastel1.colors)
    axes[1].set_title('Pie Chart')

    plt.tight_layout()
    plt.show()


numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    data = train[col].dropna()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    fig.suptitle(f'{col} Variable', fontsize=14)

    # Histogram
    axes[0].hist(data, bins=10, color='lightblue', edgecolor='black')
    axes[0].set_title('Histogram')
    axes[0].set_xlabel(col)
    axes[0].set_ylabel('Frequency')

    # Density Plot
    from scipy.stats import gaussian_kde
    density = gaussian_kde(data)
    x_vals = np.linspace(min(data), max(data), 100)
    axes[0].plot(x_vals, density(x_vals), color='green')
    axes[0].fill_between(x_vals, density(x_vals), color='green', alpha=0.3)
    axes[0].set_title('Density Plot')
    axes[0].set_xlabel(col)

    # Box Plot
    axes[1].boxplot(data, vert=False)
    axes[1].set_title('Box Plot')
    axes[1].set_xlabel(col)

    plt.tight_layout()
    plt.show()


for cat in categorical_cols:
    for num in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.violinplot(x=cat, y=num, data=train, inner="quart")
        plt.title(f'{cat} vs {num}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


sns.heatmap(train[numeric_cols].corr(), annot=True, vmin=-1, vmax=1)
plt.show()


from scipy.stats import kruskal


# Scale (sayısal) ve nominal değişkenler
scale_vars = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
nominal_vars = ['Soil Type', 'Crop Type', 'Fertilizer Name']

# Sonuçları tutmak için boş DataFrame
eta_results = pd.DataFrame(index=nominal_vars, columns=scale_vars)

# Her nominal x scale kombinasyonu için eta squared hesapla
for nom in nominal_vars:
    for scale in scale_vars:
        try:
            # Grupları oluştur
            gruplar = [train[scale][train[nom] == kategori].dropna() for kategori in train[nom].dropna().unique()]
            
            # Eğer 2'den az grup varsa atla
            if len(gruplar) < 2:
                eta = None
            else:
                # Kruskal-Wallis testi
                H, p = kruskal(*gruplar)
                k = len(gruplar)
                n = sum([len(g) for g in gruplar])

                # Eta squared hesabı
                eta = (H - k + 1) / (n - k)

            # Crosstab benzeri tabloya yaz
            eta_results.loc[nom, scale] = round(eta, 4) if eta is not None else None

        except Exception as e:
            eta_results.loc[nom, scale] = None


sns.heatmap(eta_results.astype(float), annot=True, vmin=0, vmax=1)
plt.title("Categorical vs Numeric Variable Strength")
plt.show()


from scipy.stats import chi2_contingency


# Cramér's V hesaplama fonksiyonu
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2, p, dof, ex = chi2_contingency(confusion_matrix, correction=False)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))  # düzeltme
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# Boş DataFrame (üçgen matris olacak)
cramers_results = pd.DataFrame(index=nominal_vars, columns=nominal_vars)

# Nominal değişken çiftleri için Cramér's V hesapla
for var1 in nominal_vars:
    for var2 in nominal_vars:
        if var1 == var2:
            cramers_results.loc[var1, var2] = 1.0  # Kendisiyle olan ilişki = 1
        else:
            try:
                value = cramers_v(train[var1], train[var2])
                cramers_results.loc[var1, var2] = round(value, 4)
            except:
                cramers_results.loc[var1, var2] = None


sns.heatmap(cramers_results.astype(float), annot=True, vmin=0, vmax=1)
plt.title("Categorical vs Numeric Variable Strength")
plt.show()




