from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv', index_col='ID')
test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv', index_col='ID')
data_dictionary = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')


missing_percentages = train.isnull().mean() * 100
missing_percentages_sorted = missing_percentages.sort_values(ascending=False)
print(missing_percentages_sorted)


def check_df(dataframe):
    print("First 5 rows:")
    print(dataframe.head(), '\n')
    print("Shape of the dataset:")
    print(dataframe.shape, '\n')
    print("Data types of the columns:")
    print(dataframe.dtypes, '\n')
    print("Missing values in each column:")
    print(dataframe.isnull().sum(), '\n')
    print("Summary statistics for numerical features:")
    print(dataframe.describe(), '\n')
    print("Summary statistics for categorical features:")
    print(dataframe.describe(include=['object', 'category']), '\n')
check_df(train)


def plot_numerical_features(dataframe):
    numerical_features = dataframe.select_dtypes(include=['float64', 'int64']).columns

    for feature in numerical_features:
        plt.figure(figsize=(8, 4))

        # KDE
        plt.subplot(1, 2, 1)
        sns.kdeplot(dataframe[feature], shade=True)
        plt.title(f'{feature} - KDE')

        # Histogram
        plt.subplot(1, 2, 2)
        sns.histplot(dataframe[feature], kde=True)
        plt.title(f'{feature} - Histogram')

        plt.tight_layout()
        plt.show()

plot_numerical_features(train)


def plot_categorical_features(dataframe):
    categorical_features = dataframe.select_dtypes(include=['object', 'category']).columns

    for feature in categorical_features:
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.countplot(x=dataframe[feature])
        plt.title(f'{feature} - Frequency Distribution')

        # Bar Plot
        plt.subplot(1, 2, 2)
        sns.barplot(x=dataframe[feature].value_counts().index, y=dataframe[feature].value_counts())
        plt.title(f'{feature} - Bar Plot')

        plt.tight_layout()
        plt.show()

plot_categorical_features(train)


from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

def plot_kaplan_meier_for_categories(data, log_rank_columns, time_column, event_column):
    kmf = KaplanMeierFitter()  # Tek bir kmf kullanıyoruz

    plt.figure(figsize=(12, 8))

    for column in log_rank_columns:
        groups = data[column].dropna().unique()

        for group in groups:
            group_data = data[data[column] == group]
            kmf.fit(group_data[time_column], group_data[event_column], label=f"{column}: {group}")
            kmf.plot(ci_show=False)

            if kmf.event_table.shape[0] > 0:
                median_survival = kmf.median_survival_time_
                print(f"{column}: {group} | Medyan Sağkalım Süresi: {median_survival:.1f} gün")

        plt.title(f"{column} Değişkenine Göre Kaplan-Meier Eğrisi", fontsize=14)
        plt.xlabel(f"Zaman ({time_column})", fontsize=12)
        plt.ylabel("Sağkalım Olasılığı", fontsize=12)
        plt.legend(title=column, fontsize=10)
        plt.grid()
        plt.show()

    return kmf  # Fonksiyon sonunda kmf objesini döndür



log_rank_columns = ["graft_type", "hla_match_drb1_low", "prod_type", "in_vivo_tcd"]

# Kaplan-Meier analizini uygula ve modelleri kaydet
kmf_before = plot_kaplan_meier_for_categories(train, log_rank_columns, 'efs_time', 'efs')


from lifelines.statistics import logrank_test

#important_groups = ["graft_type", "vent_hist", "rituximab"]

important_groups = ['graft_type', 'vent_hist', 'rituximab', 'hla_match_drb1_low',
       'prod_type', 'mrd_hct', 'in_vivo_tcd', 'melphalan_dose']

def perform_logrank_test(data, group_column):
    mask = data[group_column].notna()
    data_clean = data[mask]
    groups = data_clean[group_column].unique()
    if len(groups) == 2:
        group1 = data_clean[data_clean[group_column] == groups[0]]
        group2 = data_clean[data_clean[group_column] == groups[1]]
        results = logrank_test(group1['efs_time'], group2['efs_time'], group1['efs'], group2['efs'])
        print(f"{group_column} | Log-Rank Test: Statistic = {results.test_statistic:.4f}, p-value = {results.p_value:.4f}")
    else:
        print(f"{group_column} has more than two groups - consider multivariate analysis.")

for group in important_groups:
    print(f"\nLog-Rank Test for {group}")
    perform_logrank_test(train, group)


def create_features(df):
    print("Feature Engineering başlatılıyor...")
    print("Mevcut kolonlar:", df.columns.tolist())

    # Tarih özellikleri (eğer year_hct varsa)
    if 'year_hct' in df.columns:
        df['year_normalized'] = df['year_hct'] - df['year_hct'].min()

    # Yaş grupları oluştur
    if 'age' in df.columns:
        df['age_group'] = pd.qcut(df['age'], q=5, labels=['very_young', 'young', 'middle', 'old', 'very_old'])

        # Yaş ile ilgili ek özellikler
        df['age_squared'] = df['age'] ** 2

    # BMI ile ilgili özellikler
    if 'bmi' in df.columns:
        df['bmi_category'] = pd.cut(df['bmi'],
                                  bins=[0, 18.5, 25, 30, 100],
                                  labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

        if 'age' in df.columns:
            df['bmi_age_interaction'] = df['bmi'] * df['age']

    # Disease duration özellikleri
    if 'disease_duration' in df.columns:
        df['disease_severity'] = np.where(df['disease_duration'] > df['disease_duration'].median(),
                                        'Severe', 'Mild')

    # HLA match özellikleri
    hla_columns = [col for col in df.columns if 'hla' in col.lower()]
    if hla_columns:
        df['total_hla_matches'] = df[hla_columns].sum(axis=1)

    # Graft özellikleri
    if 'graft_type' in df.columns:
        df['is_bone_marrow'] = df['graft_type'].str.contains('bone', case=False, na=False).astype(int)

    print("Feature Engineering tamamlandı.")
    return df

# Feature Engineering uygula
print("Train veri seti özellikleri oluşturuluyor...")
train = create_features(train)

print("\nTest veri seti özellikleri oluşturuluyor...")
test = create_features(test)

# Yeni oluşturulan özellikleri kontrol et
print("\nTrain veri setindeki yeni özellikler:")
print(train.columns.tolist())


missing_percentages = train.isnull().mean() * 100
missing_percentages_sorted = missing_percentages.sort_values(ascending=False)
print(missing_percentages_sorted)


from scipy.stats import chi2

def test_mcar(df, significance_level=0.05):
    """
    Little's MCAR test implementation

    Parameters:
    df : pandas DataFrame
    significance_level : float, default 0.05

    Returns:
    dict : Test sonuçları içeren sözlük
    """
    # Sadece sayısal sütunları seç
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        raise ValueError("Veri setinde sayısal sütun bulunamadı!")

    print(f"\nAnaliz edilen sayısal sütunlar: {', '.join(numeric_df.columns)}")

    # Eksik değer pattern matrisi oluştur
    missing_pattern = numeric_df.isna().astype(int)

    # Her değişken için eksik değer sayısı
    missing_counts = missing_pattern.sum()

    # Her satır için eksik değer sayısı
    missing_per_row = missing_pattern.sum(axis=1)

    # Test istatistiği hesaplama
    n = len(numeric_df)
    k = len(numeric_df.columns)

    # Kovaryans matrisi hesaplama
    means = numeric_df.mean()
    observed_values = numeric_df - means
    observed_values[numeric_df.isna()] = 0

    cov_matrix = observed_values.T @ observed_values / (n - 1)

    # Chi-square test istatistiği hesaplama
    d2 = 0
    for i in range(n):
        row = observed_values.iloc[i]
        valid_cols = ~numeric_df.iloc[i].isna()
        if valid_cols.sum() > 0:
            sub_cov = cov_matrix.loc[valid_cols, valid_cols]
            if sub_cov.shape[0] > 0:
                try:
                    d2 += row[valid_cols] @ np.linalg.inv(sub_cov) @ row[valid_cols]
                except:
                    continue

    # Serbestlik derecesi hesaplama
    df_chi = k * (k + 1) / 2

    # p-değeri hesaplama
    p_value = 1 - chi2.cdf(d2, df_chi)

    return {
        'test_statistic': d2,
        'degrees_of_freedom': df_chi,
        'p_value': p_value,
        'is_mcar': p_value > significance_level,
        'missing_counts': missing_counts,
        'missing_percentages': (missing_counts / n) * 100
    }

def analyze_missing_data(df):
    """
    Veri setindeki eksik değerleri analiz eder
    """
    # Veri tipi bilgisini göster
    print("\nVeri Seti Bilgisi:")
    print("-" * 40)
    print(df.dtypes)

    try:
        results = test_mcar(df)

        print("\nEksik Değer Analizi Sonuçları:")
        print("-" * 40)
        print(f"Test İstatistiği: {results['test_statistic']:.2f}")
        print(f"Serbestlik Derecesi: {results['degrees_of_freedom']:.0f}")
        print(f"P-değeri: {results['p_value']:.4f}")
        print(f"MCAR mi?: {'Evet' if results['is_mcar'] else 'Hayır'}")
        print("\nDeğişkenlerdeki Eksik Değer Yüzdeleri:")
        for col, pct in results['missing_percentages'].items():
            print(f"{col}: %{pct:.1f}")

    except Exception as e:
        print(f"\nHata oluştu: {str(e)}")
        print("\nLütfen veri setinizi kontrol edin ve sayısal olmayan sütunları ayıklayın.")


analyze_missing_data(train)


from scipy import stats

def analyze_mnar(df, target_col):
    """
    MNAR analizi için çeşitli testler ve görselleştirmeler yapar

    Parameters:
    df : pandas DataFrame
    target_col : str, analiz edilecek hedef değişken

    Returns:
    dict : Analiz sonuçlarını içeren sözlük
    """
    results = {}

    # Eksik ve mevcut değerleri ayır
    missing_mask = df[target_col].isna()
    complete_data = df[~missing_mask]
    missing_data = df[missing_mask]

    # Diğer değişkenlerle ilişki analizi
    correlations = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col != target_col:
            # Eksik/mevcut durum ile diğer değişken arasındaki ilişki
            try:
                t_stat, p_value = stats.ttest_ind(
                    df[~df[target_col].isna()][col].dropna(),
                    df[df[target_col].isna()][col].dropna()
                )
                correlations[col] = {
                    't_statistic': t_stat,
                    'p_value': p_value
                }
            except:
                continue

    # Anlamlı ilişkileri filtrele (p < 0.05)
    significant_correlations = {
        k: v for k, v in correlations.items()
        if v['p_value'] < 0.05
    }

    results['correlations'] = correlations
    results['significant_correlations'] = significant_correlations

    # Pattern analizi
    patterns = {}
    for col in df.columns:
        if col != target_col:
            cross_tab = pd.crosstab(
                df[col].isna(),
                df[target_col].isna(),
                normalize='columns'
            )
            patterns[col] = cross_tab

    results['missing_patterns'] = patterns

    return results

def visualize_mnar_patterns(df, target_col):
    """
    MNAR paternlerini görselleştirir
    """
    plt.figure(figsize=(20, 10))

    # Eksik değer oranları
    missing_proportions = df.isnull().mean().sort_values(ascending=True)

    # Eksik değer korelasyon matrisi
    missing_matrix = df.isnull().corr()

    # Alt grafikler
    plt.subplot(1, 2, 1)
    missing_proportions.plot(kind='barh')
    plt.title('Eksik Değer Oranları')
    plt.xlabel('Eksik Değer Oranı')

    plt.subplot(1, 2, 2)
    sns.heatmap(missing_matrix, cmap='coolwarm', center=0)
    plt.title('Eksik Değer Korelasyonları')

    plt.tight_layout()
    plt.show()

def run_mnar_analysis(df, target_col):
    """
    Tam MNAR analizi çalıştırır ve sonuçları raporlar
    """
    print(f"\nMNAR Analizi: {target_col}")
    print("-" * 50)

    results = analyze_mnar(df, target_col)

    print("\nAnlamlı İlişkiler (p < 0.05):")
    for col, stats in results['significant_correlations'].items():
        print(f"\n{col}:")
        print(f"t-istatistiği: {stats['t_statistic']:.4f}")
        print(f"p-değeri: {stats['p_value']:.4f}")

    # Görselleştirme
    visualize_mnar_patterns(df, target_col)

    return results


all_results = {}

for col in train.columns:
    print(f"Analiz ediliyor: {col}")
    if train[col].isnull().sum() > 0:  # Sadece eksik değeri olan kolonları analiz et
        results = run_mnar_analysis(train, col)
        all_results[col] = results
        print("\n" + "="*50 + "\n")
    else:
        print(f"{col} sütununda eksik değer bulunamadı.\n")


def run_mnar_analysis(df, target_col):
    """
    Tam MNAR analizi çalıştırır, sonuçları raporlar ve MNAR olan değişkenleri listeye ekler
    """
    print(f"\nMNAR Analizi: {target_col}")
    print("-" * 50)

    results = analyze_mnar(df, target_col)

    print("\nAnlamlı İlişkiler (p < 0.05):")
    mnar_variables = []
    for col, stats in results['significant_correlations'].items():
        print(f"\n{col}:")
        print(f"t-istatistiği: {stats['t_statistic']:.4f}")
        print(f"p-değeri: {stats['p_value']:.4f}")
        mnar_variables.append(col)

    # Görselleştirme
    visualize_mnar_patterns(df, target_col)

    return results, mnar_variables


# Tüm eksik değerler için analiz ve MNAR listesi
# Tüm eksik değerler için analiz ve MNAR listesi
all_results = {}
mnar_set = set()  # MNAR değişkenleri benzersiz olarak toplanacak

for col in train.columns:
    print(f"Analiz ediliyor: {col}")
    if train[col].isnull().sum() > 0:  # Sadece eksik değeri olan kolonları analiz et
        results, mnar_variables = run_mnar_analysis(train, col)
        all_results[col] = results
        mnar_set.update(mnar_variables)  # MNAR değişkenlerini sete ekle
        print("\n" + "="*50 + "\n")
    else:
        print(f"{col} sütununda eksik değer bulunamadı.\n")

# MNAR değişkenlerini liste olarak döndür
mnar_list = list(mnar_set)

print("MNAR olduğu belirlenen değişkenler (tekrar etmeyen):")
print(mnar_list)



def analyze_mar_relationships_with_decision(df):
    """
    Her eksik değer içeren değişkenin diğer değişkenlerle ilişkisini analiz eder
    ve net bir MAR kararı döner.

    Parameters:
    df : pandas DataFrame

    Returns:
    DataFrame : İlişki analizi sonuçları ve MAR kararı
    """
    # Eksik değer içeren sütunları bul
    cols_with_missing = df.columns[df.isnull().any()].tolist()

    # Sayısal sütunları bul
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Sonuçları saklamak için liste
    results = []

    for col_missing in cols_with_missing:
        # Eksik değer maskesi
        missing_mask = df[col_missing].isnull()

        mar_flag = False  # Bu değişken için MAR tespiti

        for col_test in numeric_cols:
            if col_test != col_missing:
                try:
                    # Eksik vs mevcut değerler için t-testi
                    group1 = df[~missing_mask][col_test].dropna()
                    group2 = df[missing_mask][col_test].dropna()

                    if len(group1) > 0 and len(group2) > 0:
                        t_stat, p_value = stats.ttest_ind(group1, group2)

                        # Etki büyüklüğü (Cohen's d)
                        pooled_std = np.sqrt(((group1.var() * (len(group1) - 1)) +
                                            (group2.var() * (len(group2) - 1))) /
                                           (len(group1) + len(group2) - 2))
                        cohens_d = abs(group1.mean() - group2.mean()) / pooled_std

                        # Anlamlılık kontrolü
                        if p_value < 0.05 and cohens_d > 0.2:
                            mar_flag = True  # Anlamlı bir ilişki var

                        results.append({
                            'Missing_Variable': col_missing,
                            'Test_Variable': col_test,
                            'T_Statistic': t_stat,
                            'P_Value': p_value,
                            'Cohens_d': cohens_d,
                            'Mean_Present': group1.mean(),
                            'Mean_Missing': group2.mean(),
                            'N_Present': len(group1),
                            'N_Missing': len(group2)
                        })
                except:
                    continue

        # MAR kararı
        if mar_flag:
            mar_decision = "Muhtemelen MAR"
        else:
            mar_decision = "MAR değil"

        results.append({
            'Missing_Variable': col_missing,
            'Test_Variable': 'Overall',
            'T_Statistic': None,
            'P_Value': None,
            'Cohens_d': None,
            'Mean_Present': None,
            'Mean_Missing': None,
            'N_Present': None,
            'N_Missing': None,
            'MAR_Decision': mar_decision
        })

    # Sonuçları DataFrame'e dönüştür
    results_df = pd.DataFrame(results)

    # P-değerine göre sırala
    if not results_df.empty:
        results_df = results_df.sort_values(by=['P_Value', 'Missing_Variable'], na_position='last')

    return results_df



mar_results_with_decision = analyze_mar_relationships_with_decision(train)
print(mar_results_with_decision[['Missing_Variable', 'MAR_Decision']])


import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression  # RandomForestRegressor yerine LinearRegression kullanıldı.
from sklearn.preprocessing import LabelEncoder

def advanced_mice_imputation(df):
    print("MICE ile veri doldurma başlatılıyor ( 'hla' dışındaki değişkenler)...")

    # Kopya dataframe oluştur
    df_imputed = df.copy()

    # "hla" ile başlayan sütunları filtrele ve diğer sütunları al
    hla_columns = [col for col in df.columns if col.startswith('hla')]
    other_columns = [col for col in df.columns if col not in hla_columns]

    # Kategorik ve numerik sütunları ayır (sadece "hla" dışındaki sütunlar)
    categorical_columns = df_imputed[other_columns].select_dtypes(include=['object']).columns
    numerical_columns = df_imputed[other_columns].select_dtypes(include=['float64', 'int64']).columns

    print(f"\nKategorik 'hla' dışı değişken sayısı: {len(categorical_columns)}")
    print(f"Numerik 'hla' dışı değişken sayısı: {len(numerical_columns)}")

    # Kategorik değişkenler için Label Encoding (sadece "hla" dışındaki sütunlar)
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_imputed[col] = le.fit_transform(df_imputed[col].astype(str))
        label_encoders[col] = le

    # MICE için değişkenleri hazırla (sadece "hla" dışındaki sütunlar)
    columns_for_mice = list(numerical_columns) + list(categorical_columns)

    print("\nMICE için hazırlanan 'hla' dışı değişkenleri:")
    print(columns_for_mice)

    # MICE Imputer oluştur
    mice_imputer = IterativeImputer(
        estimator=LinearRegression(),  # RandomForestRegressor yerine LinearRegression kullanıldı.
        max_iter=5,  # İterasyon sayısı azaltıldı.
        random_state=42,
        n_nearest_features=5,  # Paralel işleme kullanıldı.
        verbose=0  # Gereksiz çıktıları kaldırdık.
    )

    # MICE uygula (sadece "hla" dışındaki sütunlar)
    print("\nMICE uygulanıyor (sadece 'hla' dışındaki değişkenler)...")
    df_imputed[columns_for_mice] = mice_imputer.fit_transform(df_imputed[columns_for_mice])

    # Kategorik değişkenleri geri dönüştür (sadece "hla" dışındaki sütunlar)
    for col in categorical_columns:
        if col in label_encoders:
            # Sayısal değerleri en yakın tam sayıya yuvarla
            df_imputed[col] = np.round(df_imputed[col])
            # Değer aralığını kontrol et
            min_val = 0
            max_val = len(label_encoders[col].classes_) - 1
            df_imputed[col] = df_imputed[col].clip(min_val, max_val)
            # Inverse transform uygula
            df_imputed[col] = label_encoders[col].inverse_transform(df_imputed[col].astype(int))

    # Doldurma sonrası kontrol (sadece "hla" dışındaki sütunlar)
    print("\nDoldurma sonrası eksik değer kontrolü (sadece 'hla' dışındaki değişkenler):")
    remaining_nulls = df_imputed[other_columns].isnull().sum()
    print(remaining_nulls[remaining_nulls > 0])

    return df_imputed

# Train ve test verileri için MICE uygula
print("Train veri seti için MICE uygulanıyor (sadece 'hla' dışındaki değişkenler)...")
train_imputed = advanced_mice_imputation(train)

print("\nTest veri seti için MICE uygulanıyor (sadece 'hla' dışındaki değişkenler)...")
test_imputed = advanced_mice_imputation(test)

# Doldurulmuş verileri kaydet
train = train_imputed
test = test_imputed

print("\nDoldurma işlemi tamamlandı (sadece 'hla' dışındaki değişkenler)!")


def fill_missing_with_mice(df, columns_to_impute):
    """MICE ile eksik değerleri doldurur."""

    imputer = IterativeImputer(random_state=42)  # Sabit sonuçlar için random_state ayarlayın
    df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])
    return df

# Doldurulacak sütunların listesini oluştur
columns_to_impute = [
    'hla_match_c_high','hla_high_res_8','hla_high_res_10']

# Eksik değerleri MICE ile doldur
test = fill_missing_with_mice(test, columns_to_impute)

# Eksik değerlerin dolduğunu kontrol et
print(test.isnull().sum())


test.isnull().sum()
train.isnull().sum()


from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Sayısal değişkenleri seç ve eksik değer oranı %25'ten az olanları filtrele
numerical_columns = train.select_dtypes(include=['number']).columns.tolist()
missing_percentage = train[numerical_columns].isnull().mean() * 100
columns_with_less_than_25_missing = missing_percentage[missing_percentage < 25].index.tolist()

# 2. MNAR analizi yapılan değişkenlerle ilişki kurabileceğimiz anlamlı değişkenleri almak
mnar_analysis_results = {}  # MNAR analiz sonuçları
mnar_columns = []  # MNAR olan değişkenler

for col in columns_with_less_than_25_missing:
    if col in mnar_list:  # Eğer MNAR değişkeniyse
        mnar_columns.append(col)
        # `all_results` içinde col varsa, sonuçları sakla
        if col in all_results:
            mnar_analysis_results[col] = all_results[col]  # İstatistiksel sonuçları sakla
        else:
            print(f"{col} için MNAR analizi yapılmamış veya eksik sonuçlar var.")

# 3. Eksik değerleri doldurmak için HistGradientBoostingRegressor kullanmak
for target_col in mnar_columns:
    print(f"\nEksik değerler dolduruluyor: {target_col}")

    # Hedef değişkenin eksik olmayan verileriyle eğitim verisi oluştur
    train_data = train.dropna(subset=[target_col])

    # Sadece sayısal değişkenleri kullan
    predictors = [col for col in train_data.columns if col != target_col and train_data[col].dtype in ['float64', 'int64']]

    # Sayısal olmayan değerleri sayıya dönüştür
    for col in predictors:
        if train_data[col].dtype == 'object':  # Eğer nesne (string) türünde bir sütun varsa
            # Bu sütunları dönüştürme (örneğin kategorik verileri etiketle)
            train_data[col] = pd.to_numeric(train_data[col], errors='coerce')  # Sayıya dönüştür, dönüştürülemeyenleri NaN yap

    X = train_data[predictors]
    y = train_data[target_col]

    # Modeli oluştur ve eğit
    model = HistGradientBoostingRegressor()
    model.fit(X, y)

    # Eksik değerlerin olduğu verileri tahmin et
    missing_data = train[train[target_col].isnull()]

    if not missing_data.empty:  # Eksik veriler varsa
        X_missing = missing_data[predictors]

        # Eksik verileri doldur
        predicted_values = model.predict(X_missing)
        train.loc[train[target_col].isnull(), target_col] = predicted_values

        print(f"{target_col} için eksik değerler başarıyla dolduruldu.")
    else:
        print(f"{target_col} için eksik değer bulunmamaktadır.")

# Sonuçları kontrol et
print("\nEksik değerler doldurulmuş veri seti:")
print(train.isnull().sum())



missing_percentages = train.isnull().mean() * 100
missing_percentages_sorted = missing_percentages.sort_values(ascending=False)
print(missing_percentages_sorted)


def identify_column_types(dataframe, cat_th=10, car_th=20):
    categorical = list(dataframe.select_dtypes(include=['object', 'category', 'bool']).columns)
    numeric_to_categorical = [col for col in dataframe.select_dtypes(include=['int64', 'float64']).columns
                              if dataframe[col].nunique() < cat_th]
    categorical_to_cardinal = [col for col in categorical
                               if dataframe[col].nunique() > car_th]
    categorical = list(set(categorical) - set(categorical_to_cardinal)) + numeric_to_categorical
    numerical = list(set(dataframe.select_dtypes(include=['int64', 'float64']).columns) - set(numeric_to_categorical))
    cat_cols = categorical
    num_cols = numerical
    cat_but_car = categorical_to_cardinal
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"Categorical Columns: {len(cat_cols)}")
    print(f"Numerical Columns: {len(num_cols)}")
    print(f"Cardinal Columns: {len(cat_but_car)}")

    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = identify_column_types(train)


cat_cols = [col for col in cat_cols if col != "efs"]
cat_cols
num_cols = [col for col in num_cols if col not in ["efs_time", "ID"]]
num_cols


def find_and_cap_outliers(df, columns, lower_quantile=0.05, upper_quantile=0.95):
    outlier_counts = {}

    for col in columns:
        lower_bound = df[col].quantile(lower_quantile)
        upper_bound = df[col].quantile(upper_quantile)

        # Aykırı değerleri belirle
        lower_outliers = (df[col] < lower_bound).sum()
        upper_outliers = (df[col] > upper_bound).sum()
        total_outliers = lower_outliers + upper_outliers

        # Sonuçları sakla
        outlier_counts[col] = total_outliers

        # Aykırı değerleri baskıla
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    return df, outlier_counts

# Aykırı değerleri bul ve baskıla
columns = num_cols
train, outlier_counts = find_and_cap_outliers(train, columns)

# Aykırı değer sayılarını göster
print("Aykırı değer sayıları:")
for col, count in outlier_counts.items():
    print(f"{col}: {count} adet aykırı değer")

# Baskılanmış veriyi özetle
print("\nBaskılanmış verinin özeti:")
print(train.describe())


import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def clean_column_names(names):
    """Sütun adlarındaki sorunlu karakterleri temizler."""
    try:
        cleaned_names = []
        for name in names:
            name = str(name)  # Eğer name string değilse, stringe çevir
            name = re.sub(r'\+', 'plus', name)
            name = re.sub(r'\-', 'neg', name)
            name = re.sub(r'[<>/(){}\[\]]', '_', name)
            name = re.sub(r'[^a-zA-Z0-9_]', '', name)
            cleaned_names.append(name)
        return np.array(cleaned_names)
    except Exception as e:
        print(f"Sütun adlarını temizlerken hata: {e}")
        # Basit bir fallback: boşlukları altçizgi ile değiştir ve alfanumerik olmayanları kaldır
        return np.array([re.sub(r'[^a-zA-Z0-9_]', '', str(name).replace(' ', '_')) for name in names])

def preprocess_data(train, test, cat_cols, num_cols, target_col):
    """
    Comprehensive data preprocessing for Kaggle competition:
    1. Clean column names
    2. Split training data into train and validation sets
    3. One-hot encode categorical columns
    4. Scale numerical columns
    5. Prepare data for modeling
    """
    try:
        # Veri setlerini kopyala (değişikliklerin orijinali etkilememesi için)
        train = train.copy()
        test = test.copy()
        
        print(f"Train shape: {train.shape}, Test shape: {test.shape}")
        
        # Clean column names for all DataFrames
        try:
            train.columns = clean_column_names(train.columns)
            test.columns = clean_column_names(test.columns)
            # Update cat_cols and num_cols with cleaned names if necessary
            cat_cols = clean_column_names(cat_cols)
            num_cols = clean_column_names(num_cols)
            print("Sütun adları temizlendi")
        except Exception as e:
            print(f"Sütun adlarını temizlerken hata: {e}")
            # Devam et, orijinal sütun adlarıyla çalış
        
        # Eksik değerleri kontrol et ve doldur
        try:
            print(f"Train eksik değerler: {train.isnull().sum().sum()}")
            print(f"Test eksik değerler: {test.isnull().sum().sum()}")
            
            # Eksik sayısal değerleri doldur
            for col in num_cols:
                if col in train.columns and train[col].isnull().any():
                    median_val = train[col].median()
                    train[col] = train[col].fillna(median_val)
                if col in test.columns and test[col].isnull().any():
                    median_val = train[col].median() if col in train.columns else test[col].median()
                    test[col] = test[col].fillna(median_val)
            
            # Eksik kategorik değerleri doldur
            for col in cat_cols:
                if col in train.columns and train[col].isnull().any():
                    mode_val = train[col].mode()[0]
                    train[col] = train[col].fillna(mode_val)
                if col in test.columns and test[col].isnull().any():
                    mode_val = train[col].mode()[0] if col in train.columns else test[col].mode()[0]
                    test[col] = test[col].fillna(mode_val)
            
            print("Eksik değerler dolduruldu")
        except Exception as e:
            print(f"Eksik değerleri doldururken hata: {e}")
            # Basit eksik değer doldurma yöntemi
            train = train.fillna(0)
            test = test.fillna(0)
        
        # Separate features and target
        try:
            if target_col not in train.columns:
                raise ValueError(f"Target column '{target_col}' not found in train data")
            
            y = train[target_col]
            X = train.drop(target_col, axis=1)
            print(f"Target column '{target_col}' extracted")
        except Exception as e:
            print(f"Hedef değişkeni ayırırken hata: {e}")
            raise  # Bu kritik bir hata, devam edemeyiz
        
        # Split training data
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            print(f"Train-validation split: {X_train.shape}, {X_val.shape}")
        except Exception as e:
            print(f"Eğitim-doğrulama ayrımında hata: {e}")
            # Basit bir fallback, tüm veriyi hem eğitim hem de doğrulama seti olarak kullan
            X_train = X_val = X
            y_train = y_val = y
        
        # Kategorik kolonları kontrol et
        try:
            # Kategorik kolonların varlığını ve türünü kontrol et
            valid_cat_cols = [col for col in cat_cols if col in X_train.columns]
            if not valid_cat_cols:
                print("Uyarı: Geçerli kategorik kolon bulunamadı!")
            else:
                print(f"Geçerli kategorik kolonlar: {len(valid_cat_cols)}")
                
                # One-hot encoding
                encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
                
                # Fit and transform categorical columns
                X_train_cat = encoder.fit_transform(X_train[valid_cat_cols])
                X_val_cat = encoder.transform(X_val[valid_cat_cols])
                X_test_cat = encoder.transform(test[valid_cat_cols])
                
                # Get encoded column names
                encoded_train_cols = encoder.get_feature_names_out(valid_cat_cols)
                
                # Create DataFrames with encoded categorical columns
                encoded_train_df = pd.DataFrame(X_train_cat, columns=encoded_train_cols, index=X_train.index)
                encoded_val_df = pd.DataFrame(X_val_cat, columns=encoded_train_cols, index=X_val.index)
                encoded_test_df = pd.DataFrame(X_test_cat, columns=encoded_train_cols, index=test.index)
                
                # Remove original categorical columns and add encoded ones
                X_train = pd.concat([X_train.drop(valid_cat_cols, axis=1), encoded_train_df], axis=1)
                X_val = pd.concat([X_val.drop(valid_cat_cols, axis=1), encoded_val_df], axis=1)
                X_test = pd.concat([test.drop(valid_cat_cols, axis=1), encoded_test_df], axis=1)
                
                print("Kategorik değişkenler one-hot encoding ile dönüştürüldü")
        except Exception as e:
            print(f"Kategorik değişkenleri dönüştürürken hata: {e}")
            # Encoder başarısız olursa, kategorik kolonları kullanma
            encoder = None
            # Dummy değişken oluştur (daha basit bir yaklaşım)
            try:
                for col in valid_cat_cols:
                    X_train[col] = X_train[col].astype('category').cat.codes
                    X_val[col] = X_val[col].astype('category').cat.codes
                    X_test[col] = X_test[col].astype('category').cat.codes
                print("Kategorik değişkenler sayısal kodlara dönüştürüldü")
            except:
                print("Kategorik değişkenleri sayısal kodlara dönüştürme başarısız")
        
        # Numeric columns
        try:
            # Sayısal kolonların varlığını kontrol et
            valid_num_cols = [col for col in num_cols if col in X_train.columns]
            if not valid_num_cols:
                print("Uyarı: Geçerli sayısal kolon bulunamadı!")
            else:
                print(f"Geçerli sayısal kolonlar: {len(valid_num_cols)}")
                
                # Scale numerical columns
                scaler = StandardScaler()
                X_train[valid_num_cols] = scaler.fit_transform(X_train[valid_num_cols])
                X_val[valid_num_cols] = scaler.transform(X_val[valid_num_cols])
                X_test[valid_num_cols] = scaler.transform(X_test[valid_num_cols])
                
                print("Sayısal değişkenler ölçeklendirildi")
        except Exception as e:
            print(f"Sayısal değişkenleri ölçeklendirirken hata: {e}")
            scaler = None
            # Basit bir normalizasyon yapmayı dene
            try:
                for col in valid_num_cols:
                    mean = X_train[col].mean()
                    std = X_train[col].std()
                    if std > 0:
                        X_train[col] = (X_train[col] - mean) / std
                        X_val[col] = (X_val[col] - mean) / std
                        X_test[col] = (X_test[col] - mean) / std
                print("Sayısal değişkenler basit normalizasyon ile ölçeklendirildi")
            except:
                print("Sayısal değişkenleri basit normalizasyon ile ölçeklendirme başarısız")
        
        # Remove target time column if it exists
        try:
            time_col = f'{target_col}_time'
            if time_col in X_train.columns:
                X_train = X_train.drop(time_col, axis=1)
                X_val = X_val.drop(time_col, axis=1)
                print(f"'{time_col}' kolonu kaldırıldı")
        except Exception as e:
            print(f"Zaman kolonu kaldırılırken hata: {e}")
        
        # Final check - Make sure column sets match
        try:
            if not set(X_train.columns) == set(X_val.columns) == set(X_test.columns):
                # Find missing columns
                all_cols = set(X_train.columns) | set(X_val.columns) | set(X_test.columns)
                for df_name, df in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
                    missing = all_cols - set(df.columns)
                    if missing:
                        print(f"{df_name}'de eksik kolonlar: {missing}")
                        for col in missing:
                            df[col] = 0  # Eksik kolonları 0 ile doldur
                print("Tüm veri setlerindeki kolonlar eşleştirildi")
        except Exception as e:
            print(f"Kolon eşleştirmede hata: {e}")
        
        print(f"Ön işleme tamamlandı. X_train şekli: {X_train.shape}, X_test şekli: {X_test.shape}")
        
        return X_train, X_val, y_train, y_val, X_test, encoder, scaler
        
    except Exception as e:
        print(f"Kritik hata - veri ön işleme başarısız: {e}")
        # Kritik hata durumunda basit bir veri seti döndür
        # Bu en kötü durum senaryosu, en azından bir şeyler döndürmeyi dener
        try:
            # Minimum işlenmiş verilerle devam et
            X = train.drop(target_col, axis=1) if target_col in train.columns else train
            y = train[target_col] if target_col in train.columns else pd.Series(0, index=train.index)
            
            X_train = X_val = X
            y_train = y_val = y
            X_test = test
            
            print("Acil durum: Minimum işlenmiş veri döndürülüyor")
            return X_train, X_val, y_train, y_val, X_test, None, None
        except:
            raise ValueError("Veri ön işleme tamamen başarısız oldu, yarışmaya devam edilemiyor")


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
import re

def clean_for_xgboost(X):
    """
    Clean feature names to be compatible with XGBoost requirements.
    XGBoost doesn't allow [, ], or < characters in feature names.
    
    Parameters:
    - X: DataFrame with features
    
    Returns:
    - X_cleaned: DataFrame with cleaned column names
    """
    X_cleaned = X.copy()
    
    # Clean column names specifically for XGBoost requirements
    columns = X_cleaned.columns.tolist()
    cleaned_columns = []
    
    for col in columns:
        # Replace [, ], <, > and other problematic characters with underscore
        col = re.sub(r'[\[\]<>]', '_', col)
        cleaned_columns.append(col)
    
    X_cleaned.columns = cleaned_columns
    return X_cleaned

def build_ensemble_model(X_train, X_val, y_train, y_val, X_test):
    """
    Build an ensemble model for classification using multiple base models.
    
    Parameters:
    - X_train, y_train: Training data
    - X_val, y_val: Validation data
    - X_test: Test data for predictions
    
    Returns:
    - ensemble_model: The trained ensemble model
    - base_models: Dictionary of individual base models
    - test_predictions: Predictions on the test set
    """
    try:
        print("Ensemble model oluşturma başladı...")
        
        # Ensure the feature names are compatible with all models, especially XGBoost
        try:
            X_train_clean = clean_for_xgboost(X_train)
            X_val_clean = clean_for_xgboost(X_val)
            X_test_clean = clean_for_xgboost(X_test)
            print("Veri temizleme başarılı")
        except Exception as e:
            print(f"Veri temizleme hatası: {e}")
            # Basit bir çözüm dene
            X_train_clean = X_train.copy()
            X_val_clean = X_val.copy()
            X_test_clean = X_test.copy()
            # Sorunlu sütun adlarını düzelt
            for df in [X_train_clean, X_val_clean, X_test_clean]:
                df.columns = [re.sub(r'[\[\]<>]', '_', col) for col in df.columns]
            print("Alternatif veri temizleme uygulandı")
        
        # Define base models with different algorithms
        base_models = {}
        model_performances = {}
        
        # Train individual models with try-except blokları
        model_configs = [
            ('random_forest', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gradient_boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('xgboost', xgb.XGBClassifier(n_estimators=100, random_state=42)),
            ('lightgbm', lgb.LGBMClassifier(n_estimators=100, random_state=42))
        ]
        
        print("Bireysel modeller eğitiliyor...")
        
        for name, model in model_configs:
            try:
                print(f"\n{name} eğitiliyor...")
                model.fit(X_train_clean, y_train)
                
                # Make predictions on validation set
                val_preds = model.predict(X_val_clean)
                
                # Calculate accuracy
                accuracy = accuracy_score(y_val, val_preds)
                model_performances[name] = accuracy
                base_models[name] = model
                
                print(f"{name} Doğrulama Doğruluğu: {accuracy:.4f}")
            except Exception as e:
                print(f"{name} eğitimi başarısız: {e}")
                print(f"{name} modeli atlanıyor")
        
        # Check if we have any models
        if not base_models:
            raise ValueError("Hiçbir model başarıyla eğitilemedi!")
        
        # Print model performance comparison
        print("\nModel Performans Özeti:")
        for name, accuracy in sorted(model_performances.items(), key=lambda x: x[1], reverse=True):
            print(f"{name}: {accuracy:.4f}")
        
        # Find the best performing model
        best_model_name = max(model_performances, key=model_performances.get)
        best_model = base_models[best_model_name]
        best_accuracy = model_performances[best_model_name]
        
        print(f"\nEn iyi model: {best_model_name}, doğruluk: {best_accuracy:.4f}")
        
        # Create ensemble only if we have multiple models
        if len(base_models) > 1:
            try:
                # Create a voting classifier ensemble
                voting_models = [(name, model) for name, model in base_models.items()]
                
                # Try soft voting first
                try:
                    print("\nSoft voting ensemble oluşturuluyor...")
                    ensemble_model = VotingClassifier(
                        estimators=voting_models,
                        voting='soft'  # Use probability-based voting
                    )
                    ensemble_model.fit(X_train_clean, y_train)
                except Exception as e:
                    print(f"Soft voting başarısız: {e}")
                    print("Hard voting'e geçiliyor...")
                    
                    ensemble_model = VotingClassifier(
                        estimators=voting_models,
                        voting='hard'  # Use majority voting
                    )
                    ensemble_model.fit(X_train_clean, y_train)
                
                # Make predictions with the ensemble
                try:
                    ensemble_val_preds = ensemble_model.predict(X_val_clean)
                    ensemble_accuracy = accuracy_score(y_val, ensemble_val_preds)
                    
                    print(f"Ensemble Model Doğrulama Doğruluğu: {ensemble_accuracy:.4f}")
                    
                    # Compare with the best individual model
                    if ensemble_accuracy > best_accuracy:
                        print(f"Ensemble, en iyi bireysel modelden ({best_model_name}) {ensemble_accuracy - best_accuracy:.4f} daha iyi performans gösteriyor")
                        print("Son tahminler için ensemble model kullanılıyor")
                        final_model = ensemble_model
                    else:
                        print(f"En iyi bireysel model ({best_model_name}), ensemble'dan {best_accuracy - ensemble_accuracy:.4f} daha iyi performans gösteriyor")
                        print("Son tahminler için en iyi bireysel model kullanılıyor")
                        final_model = best_model
                except Exception as e:
                    print(f"Ensemble değerlendirme hatası: {e}")
                    print("En iyi bireysel model kullanılıyor")
                    final_model = best_model
            except Exception as e:
                print(f"Ensemble oluşturma hatası: {e}")
                print("En iyi bireysel model kullanılıyor")
                final_model = best_model
        else:
            print("Sadece bir model mevcut. Son model olarak kullanılıyor.")
            final_model = best_model
        
        # Generate detailed classification report for the chosen model
        try:
            print("\nSeçilen Model için Sınıflandırma Raporu:")
            final_val_preds = final_model.predict(X_val_clean)
            print(classification_report(y_val, final_val_preds))
        except Exception as e:
            print(f"Sınıflandırma raporu oluşturma hatası: {e}")
        
        # Make predictions on the test set for submission
        try:
            test_predictions = final_model.predict(X_test_clean)
            print(f"Test tahmini yapıldı. Test veri şekli: {X_test_clean.shape}")
        except Exception as e:
            print(f"Test tahmini hatası: {e}")
            # Olası hafıza veya başka sorunlar için daha küçük parçalar halinde tahmin yap
            try:
                print("Batch tahmin deneniyor...")
                chunk_size = 1000
                chunks = [X_test_clean.iloc[i:i+chunk_size] for i in range(0, len(X_test_clean), chunk_size)]
                predictions_list = []
                
                for i, chunk in enumerate(chunks):
                    chunk_pred = final_model.predict(chunk)
                    predictions_list.append(chunk_pred)
                    print(f"Chunk {i+1}/{len(chunks)} tahmin edildi")
                
                test_predictions = np.concatenate(predictions_list)
                print("Batch tahmin başarılı")
            except Exception as e2:
                print(f"Batch tahmin de başarısız: {e2}")
                # Son çare: Rastgele tahminler
                print("UYARI: Tahmin yapılamadı, rastgele değerler kullanılıyor!")
                unique_classes = y_train.unique()
                test_predictions = np.random.choice(unique_classes, size=len(X_test_clean))
        
        print("Ensemble model oluşturma tamamlandı")
        return final_model, base_models, test_predictions
        
    except Exception as main_error:
        print(f"Kritik hata - ensemble model oluşturma başarısız: {main_error}")
        
        # Acil durum: En basit model ile devam et
        try:
            print("Acil durum: Basit bir model ile devam ediliyor...")
            
            # En basit model - Lojistik Regresyon
            simple_model = LogisticRegression(max_iter=2000, solver='liblinear')
            simple_model.fit(X_train, y_train)
            base_models = {'logistic_regression': simple_model}
            test_predictions = simple_model.predict(X_test)
            
            print("Acil durum modeli başarıyla oluşturuldu")
            return simple_model, base_models, test_predictions
            
        except Exception as emergency_error:
            print(f"Acil durum modeli de başarısız: {emergency_error}")
            
            # En son çare: Rastgele tahminler
            print("SON ÇARE: Rastgele tahminler yapılıyor!")
            unique_classes = y_train.unique()
            test_predictions = np.random.choice(unique_classes, size=len(X_test))
            dummy_model = LogisticRegression()  # Eğitilmemiş bir model
            base_models = {'dummy': dummy_model}
            
            return dummy_model, base_models, test_predictions
def stacking_ensemble(X_train, X_val, y_train, y_val, X_test):
    """
    Alternative approach using stacking ensemble instead of voting.
    This can sometimes perform better than simple voting.
    
    Parameters:
    - X_train, y_train: Training data
    - X_val, y_val: Validation data
    - X_test: Test data for predictions
    
    Returns:
    - stacked_model: The trained stacking model
    - test_predictions: Predictions on the test set
    """
    from sklearn.ensemble import StackingClassifier
    
    # Clean data for XGBoost
    X_train_clean = clean_for_xgboost(X_train)
    X_val_clean = clean_for_xgboost(X_val)
    X_test_clean = clean_for_xgboost(X_test)
    
    # Base estimators
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]
    
    # Try to add XGBoost if it works
    try:
        estimators.append(('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42)))
    except:
        print("Skipping XGBoost in stacking ensemble")
    
    # Final estimator
    final_estimator = LogisticRegression(max_iter=1000)
    
    # Create and train the stacking ensemble
    stacked_model = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        stack_method='auto'
    )
    
    print("Training stacking ensemble...")
    stacked_model.fit(X_train_clean, y_train)
    
    # Evaluate on validation set
    val_preds = stacked_model.predict(X_val_clean)
    val_accuracy = accuracy_score(y_val, val_preds)
    print(f"Stacking Ensemble Validation Accuracy: {val_accuracy:.4f}")
    
    # Generate prediction for test set
    test_predictions = stacked_model.predict(X_test_clean)
    
    return stacked_model, test_predictions

def optimize_best_model(X_train, X_val, y_train, y_val, best_model_name):
    """
    Optimize the best-performing model using GridSearchCV.
    
    Parameters:
    - X_train, y_train: Training data
    - X_val, y_val: Validation data
    - best_model_name: Name of the best-performing model
    
    Returns:
    - optimized_model: The optimized model after hyperparameter tuning
    """
    # Clean data for XGBoost
    X_train_clean = clean_for_xgboost(X_train)
    X_val_clean = clean_for_xgboost(X_val)
    
    print(f"\nOptimizing the best model: {best_model_name}")
    
    if best_model_name == 'random_forest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    
    elif best_model_name == 'gradient_boosting':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    
    elif best_model_name == 'xgboost':
        model = xgb.XGBClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    
    elif best_model_name == 'lightgbm':
        model = lgb.LGBMClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    
    # Run GridSearchCV (with fewer parameters and cross-validations for speed)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_clean, y_train)
    
    # Get the best model
    optimized_model = grid_search.best_estimator_
    
    # Print results
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate on validation set
    val_preds = optimized_model.predict(X_val_clean)
    val_accuracy = accuracy_score(y_val, val_preds)
    print(f"Optimized model validation accuracy: {val_accuracy:.4f}")
    
    return optimized_model

def create_submission(test_predictions, test_ids, submission_file='submission.csv'):
    """
    Create a submission file for Kaggle
    
    Parameters:
    - test_predictions: Predictions on the test set
    - test_ids: ID column from the test set
    - submission_file: Output file name
    """
    submission = pd.DataFrame({
        'ID': test_ids,
        'efs': test_predictions
    })
    submission.to_csv(submission_file, index=False)
    print(f"\nSubmission file created: {submission_file}")
    return submission



# Main workflow
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score

try:
    # Test verisi kontrolü
    if 'test' not in globals() or test is None:
        print("Uyarı: Test verisi yüklenemedi!")
        # Varsayılan bir test verisi oluşturabilirsiniz
    
    target_col = 'efs'
    
    # Get test IDs for submission
    try:
        test_ids = test['id'].copy() if 'id' in test.columns else test.index
        print(f"Test veri sayısı: {len(test_ids)}")
    except Exception as e:
        print(f"Test ID'leri alınırken hata: {e}")
        test_ids = pd.Series(range(len(test))) if 'test' in globals() and test is not None else pd.Series([])
    
    # Preprocess data
    try:
        X_train, X_val, y_train, y_val, X_test, encoder, scaler = preprocess_data(train, test, cat_cols, num_cols, target_col)
        print("Veri ön işleme başarılı!")
    except Exception as e:
        print(f"Veri ön işleme hatası: {e}")
        raise
    
    # Build ensemble model
    try:
        final_model, base_models, test_predictions = build_ensemble_model(
            X_train, X_val, y_train, y_val, X_test
        )
        print("Model oluşturma başarılı!")
    except Exception as e:
        print(f"Model oluşturma hatası: {e}")
        raise
    
    # Find best model name for optimization
    try:
        model_performances = {}
        for name, model in base_models.items():
            try:
                clean_X_val = clean_for_xgboost(X_val)
                predictions = model.predict(clean_X_val)
                model_performances[name] = accuracy_score(y_val, predictions)
            except Exception as model_e:
                print(f"{name} modeli değerlendirme hatası: {model_e}")
                model_performances[name] = 0
        
        if model_performances:
            best_model_name = max(model_performances, key=model_performances.get)
            print(f"En iyi model: {best_model_name}, Skor: {model_performances[best_model_name]}")
        else:
            print("Model performansları hesaplanamadı!")
            best_model_name = None
    except Exception as e:
        print(f"Model değerlendirme hatası: {e}")
        best_model_name = None
    
    # Create submission file
    try:
        submission = create_submission(test_predictions, test_ids)
        print("Submission dosyası oluşturuldu!")
    except Exception as e:
        print(f"Submission oluşturma hatası: {e}")
        # Fallback submission oluştur
        if 'test_ids' in locals() and 'test_predictions' in locals():
            submission = pd.DataFrame({'id': test_ids, target_col: test_predictions})
        else:
            print("Fallback submission oluşturulamadı!")
            raise
    
    # Submission dosyasını kaydet
    try:
        submission.to_csv('submission.csv', index=False)
        print("Submission dosyası başarıyla kaydedildi!")
        print(submission.head())
    except Exception as e:
        print(f"Submission dosyası kaydedilirken hata: {e}")
        
except Exception as main_error:
    print(f"Ana iş akışında kritik hata: {main_error}")
    # Kritik hata durumunda basit bir submission oluşturmayı deneyin
    try:
        if 'test' in globals() and test is not None:
            dummy_submission = pd.DataFrame({'id': test['id'] if 'id' in test.columns else test.index})
            dummy_submission[target_col] = 0  # Varsayılan tahmin
            dummy_submission.to_csv('emergency_submission.csv', index=False)
            print("Acil durum submission dosyası oluşturuldu!")
    except:
        print("Acil durum submission dosyası oluşturulamadı!")

submission = submission.rename(columns={'efs': 'prediction'})

