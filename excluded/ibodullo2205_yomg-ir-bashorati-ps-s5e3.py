# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
from IPython.display import display, HTML
from io import BytesIO
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


url = "https://img.freepik.com/free-photo/beautiful-city-view_23-2151002674.jpg"

html_code = f'''
<img src="{url}" style="width:100%; height:300px; object-fit: cover; border-radius: 20px;">
'''
display(HTML(html_code))


# Importing Libraries

import warnings
warnings.filterwarnings("ignore")

import optuna
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, median_absolute_error
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import catboost as cb
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit


# Reading .csv data file
train_data = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
original_data = pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv')


train_data.sample(5)


test_data.sample(5)


original_data.sample(5)


original_data.columns


# Bo'shliqlarni o'chirib tashlash
original_data.columns = original_data.columns.str.strip()


# 'rainfall' ustunidagi xatoni to'g'irlaymiz
original_data['rainfall'] = original_data['rainfall'].map({'yes': 1, 'no': 0})


# Trening jadvali oâ€˜lchamlarini olish
num_train_rows, num_train_columns = train_data.shape

# Test jadvali oâ€˜lchamlarini olish
num_test_rows, num_test_columns = test_data.shape

# Asl jadval oâ€˜lchamlarini olish
num_original_rows, num_original_columns = original_data.shape

# Natijalarni chop etish
print("ğŸ“Š **Trening ma'lumotlari:**")
print(f"â�¡ï¸� Qatorlar soni: {num_train_rows}")
print(f"â�¡ï¸� Ustunlar soni: {num_train_columns}\n")

print("ğŸ“Š **Test ma'lumotlari:**")
print(f"â�¡ï¸� Qatorlar soni: {num_test_rows}")
print(f"â�¡ï¸� Ustunlar soni: {num_test_columns}\n")

print("ğŸ“Š **Asl ma'lumotlar:**")
print(f"â�¡ï¸� Qatorlar soni: {num_original_rows}")
print(f"â�¡ï¸� Ustunlar soni: {num_original_columns}")



# ğŸŸ¡ Trening dataset uchun yoâ€˜qolgan qiymatlar
missing_values_train = pd.DataFrame({
    'Feature': train_data.columns,
    '[TRAIN] No. of Missing Values': train_data.isnull().sum().values,
    '[TRAIN] % of Missing Values': (train_data.isnull().sum().values / len(train_data) * 100)
})

# ğŸ”µ Test dataset uchun yoâ€˜qolgan qiymatlar
missing_values_test = pd.DataFrame({
    'Feature': test_data.columns,
    '[TEST] No. of Missing Values': test_data.isnull().sum().values,
    '[TEST] % of Missing Values': (test_data.isnull().sum().values / len(test_data) * 100)
})

# ğŸ”´ Asl dataset uchun yoâ€˜qolgan qiymatlar
missing_values_original = pd.DataFrame({
    'Feature': original_data.columns,
    '[ORIGINAL] No. of Missing Values': original_data.isnull().sum().values,
    '[ORIGINAL] % of Missing Values': (original_data.isnull().sum().values / len(original_data) * 100)
})

# ğŸŸ¢ Har bir ustundagi unikal qiymatlar soni (faqat training dataset)
unique_values = pd.DataFrame({
    'Feature': train_data.columns,
    'No. of Unique Values[FROM TRAIN]': train_data.nunique().values
})

# ğŸŸ  Har bir ustunning maâ€™lumot turi
feature_types = pd.DataFrame({
    'Feature': train_data.columns,
    'DataType': train_data.dtypes
})

# ğŸ“Š Hammasini birlashtirish
merged_df = pd.merge(missing_values_train, missing_values_test, on='Feature', how='left')
merged_df = pd.merge(merged_df, missing_values_original, on='Feature', how='left')
merged_df = pd.merge(merged_df, unique_values, on='Feature', how='left')
merged_df = pd.merge(merged_df, feature_types, on='Feature', how='left')

# ğŸŒˆ Vizualizatsiya (jadvalni rang bilan bezash)
merged_df.style.background_gradient(cmap='viridis')



# ğŸŸ¡ Trening datasetdagi dublikat qatorlar soni
train_duplicates = train_data.duplicated().sum()

# ğŸ”µ Test datasetdagi dublikat qatorlar soni
test_duplicates = test_data.duplicated().sum()

# ğŸ”´ Asl datasetdagi dublikat qatorlar soni
original_duplicates = original_data.duplicated().sum()

# ğŸ“Š Natijalarni chop etish
print(f"Trening ma'lumotlar to'plamidagi takroriy qatorlar soni: {train_duplicates}")
print(f"Test ma'lumotlar to'plamidagi takroriy qatorlar soni: {test_duplicates}")
print(f"Asl ma'lumotlar to'plamidagi takroriy qatorlar soni: {original_duplicates}")



# Raqamli ustunlarning tavsifini ko'rib chiqish
print("Trening datasetdagi barcha raqamli ustunlarning tavsifi:")
train_data.describe().T.style.background_gradient(cmap='viridis')



# Test datasetdagi raqamli ustunlarning tavsifini ko'rib chiqish
print("Test datasetdagi barcha raqamli ustunlarning tavsifi:")
test_data.describe().T.style.background_gradient(cmap='viridis')


# Original datasetdagi raqamli ustunlarning tavsifini ko'rib chiqish
print("Original datasetdagi barcha raqamli ustunlarning tavsifi:")
original_data.describe().T.style.background_gradient(cmap='viridis')


original_data['day'] = range(1, len(original_data) + 1)
original_data['day'].describe()


numerical_variables = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']
target_variable = 'rainfall' 
categorical_variables = ['winddirection']


# Maxsus ranglar palitrasi
custom_palette = ['#3498db', '#e74c3c', '#2ecc71']

# 'Dataset' ustunini qo'shish
train_data['Dataset'] = 'Train'
test_data['Dataset'] = 'Test'
original_data['Dataset'] = 'Original'

# Funksiya: Har bir o'zgaruvchi uchun boxplot va histogram yaratish
def create_variable_plots(variable):
    sns.set_style('whitegrid')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    sns.boxplot(data=pd.concat([train_data, test_data, original_data.dropna()]), 
                x=variable, y="Dataset", palette=custom_palette, ax=axes[0])
    axes[0].set_xlabel(variable)
    axes[0].set_title(f"Box Plot for {variable}")

    # Histogram (Alohida)
    sns.histplot(data=train_data, x=variable, color=custom_palette[0], kde=True, bins=30, label="Train", ax=axes[1])
    sns.histplot(data=test_data, x=variable, color=custom_palette[1], kde=True, bins=30, label="Test", ax=axes[1])
    sns.histplot(data=original_data.dropna(), x=variable, color=custom_palette[2], kde=True, bins=30, label="Original", ax=axes[1])
    
    axes[1].set_xlabel(variable)
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Histogram for {variable} [TRAIN, TEST & ORIGINAL]")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# Har bir raqamli o'zgaruvchi bo'yicha tahlil qilish
for variable in numerical_variables:
    create_variable_plots(variable)

# 'Dataset' ustunini olib tashlash
for df in [train_data, test_data, original_data]:
    df.drop('Dataset', axis=1, inplace=True)


custom_palette = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green

def create_categorical_barplot(variable):
    sns.set_style('whitegrid')

    train_data_copy = train_data.copy()
    test_data_copy = test_data.copy()
    original_data_copy = original_data.dropna().copy()

    train_data_copy['Dataset'] = 'Train'
    test_data_copy['Dataset'] = 'Test'
    original_data_copy['Dataset'] = 'Original'

    combined_data = pd.concat([train_data_copy, test_data_copy, original_data_copy])

    train_counts = train_data[variable].value_counts().sort_values(ascending=True).index.tolist()

    plt.figure(figsize=(14, 7))
    sns.countplot(
        data=combined_data, 
        x=variable,  
        hue="Dataset", 
        palette=custom_palette, 
        dodge=True,  
        width=0.85,  
        order=train_counts  
    )

    plt.ylabel("Count")
    plt.xlabel(variable)
    plt.title(f"Grouped Count Plot for {variable} [TRAIN, TEST & ORIGINAL]")
    plt.legend(title="Dataset")

    plt.xticks(rotation=45, ha="right")

    plt.show()

for variable in categorical_variables:
    create_categorical_barplot(variable)


# Define custom color palette for Train, Test, and Original datasets
custom_palette = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green

# Function to create Wind Rose plot in a subplot
def create_wind_rose(ax, data, dataset_name, color):
    # Convert wind direction to radians
    wind_direction_radians = np.radians(data['winddirection'].dropna())

    # Create histogram bins (every 10Â°)
    bins = np.linspace(0, 2*np.pi, 37)  # 36 bins (every 10Â°)
    counts, bin_edges = np.histogram(wind_direction_radians, bins=bins)

    # Plot on the polar axis with improved style
    bars = ax.bar(bin_edges[:-1], counts, width=np.radians(10), color=color, edgecolor='black', alpha=0.8)

    # Formatting for professional appearance
    ax.set_theta_zero_location("N")  # North is at 0Â°
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_xticks(np.radians(np.arange(0, 360, 45)))  # Tick labels every 45Â°
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=10, fontweight='bold')

    # Add grid and labels for better readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_yticklabels([])  # Remove radial labels to avoid clutter
    ax.set_title(f"Wind Direction ({dataset_name})", fontsize=12, fontweight='bold', pad=10)

# Create a single row with three wind rose plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': 'polar'})

# Generate wind rose plots for Train, Test, and Original datasets
create_wind_rose(axes[0], train_data, "Train Data", custom_palette[0])  # Blue
create_wind_rose(axes[1], test_data, "Test Data", custom_palette[1])    # Red
create_wind_rose(axes[2], original_data.dropna(), "Original Data", custom_palette[2])  # Green

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


pie_chart_palette = ['#33638d', '#28ae80', '#d3eb0c', '#ff9a0b', '#7e03a8', '#35b779', '#fde725', '#440154', '#90d743', '#482173', '#22a884', '#f8961e']

countplot_color = '#5C67A3'

def create_target_plots(variable):
    sns.set_style('whitegrid')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    plt.subplot(1, 2, 1)
    train_data[variable].value_counts().plot.pie(
        autopct='%1.1f%%', colors=pie_chart_palette, wedgeprops=dict(width=0.3), startangle=140
    )
    plt.title(f"Pie Chart for {variable}")

    plt.subplot(1, 2, 2)
    sns.countplot(
        data=pd.concat([train_data, original_data.dropna()]), 
        x=variable, 
        color=countplot_color,  
        alpha=0.8  
    )
    plt.xlabel(variable)
    plt.ylabel("Count")
    plt.title(f"Bar Graph for {variable} [TRAIN & ORIGINAL Combined]")

    plt.tight_layout()
    
    plt.show()

create_target_plots(target_variable)


# Oâ€˜zgaruvchilar roâ€˜yxatini yaratish
variables = [col for col in train_data.columns if col in numerical_variables] + ['day']

# Oâ€˜zgaruvchilarni mavjud roâ€˜yxatga qoâ€˜shish
test_variables = variables
train_variables = variables + ['rainfall']

# Train va test ma'lumotlari uchun korrelyatsiya matritsalarini hisoblash
corr_train = train_data[train_variables].corr()
corr_test = test_data[test_variables].corr()

# Yuqori uchburchak (upper triangle) uchun niqoblar yaratish
mask_train = np.triu(np.ones_like(corr_train, dtype=bool))
mask_test = np.triu(np.ones_like(corr_test, dtype=bool))

# Matn hajmi va burilish burchagini oâ€˜rnatish
annot_kws = {"size": 8, "rotation": 45}

# Train ma'lumotlari uchun issiqlik xaritasini (heatmap) yaratish
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
ax_train = sns.heatmap(corr_train, mask=mask_train, cmap='viridis', annot=True,
                      square=True, linewidths=.5, xticklabels=1, yticklabels=1, annot_kws=annot_kws)
plt.title('Korrelyatsiya issiqlik xaritasi - Train maâ€™lumotlari')

# Test ma'lumotlari uchun issiqlik xaritasini yaratish
plt.subplot(1, 2, 2)
ax_test = sns.heatmap(corr_test, mask=mask_test, cmap='viridis', annot=True,
                     square=True, linewidths=.5, xticklabels=1, yticklabels=1, annot_kws=annot_kws)
plt.title('Korrelyatsiya issiqlik xaritasi - Test maâ€™lumotlari')

# Chizmalarni yaxshiroq joylashtirish
plt.tight_layout()

# Chizmalarni chiqarish
plt.show()



# Train va Test ma'lumotlari uchun ranglarni belgilash
train_color = '#3498db'  # Ko'k
test_color = '#e74c3c'   # Qizil

# Grafik yaratish
plt.figure(figsize=(12, 5))

# Train ma'lumotlarini chizish
plt.plot(train_data['id'], train_data['day'], linestyle='-', color=train_color, label='Train ma\'lumotlari', alpha=0.7)

# Test ma'lumotlarini chizish
plt.plot(test_data['id'], test_data['day'], linestyle='-', color=test_color, label='Test ma\'lumotlari', alpha=0.7)

# Formatlash
plt.xlabel('ID')
plt.ylabel('Kun')
plt.title('Trend grafigi: Kun va ID o\'rtasidagi bog\'liqlik')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Grafikni chiqarish
plt.show()



# Kutilgan takrorlanuvchi naqshni yaratish (1-365 oralig'ini 6 yil davomida takrorlash)
expected_pattern = np.tile(np.arange(1, 366), 6)  # 1-365 ni aniq 6 marta takrorlaydi

# Noto'g'ri yorliqlarni tekshirish
train_data['expected_day'] = expected_pattern[:len(train_data)]  # Kutilgan naqshni tayinlash
train_data['day_mismatch'] = train_data['day'] != train_data['expected_day']  # Nomuvofiqliklarni belgilash



flag_color = '#8B0000'   # Qoramtir qizil (nomuvofiq kunlar uchun)

# Kutilgan takrorlanuvchi naqshni yaratish (1-365 oralig'ini 6 yil davomida takrorlash)
expected_pattern = np.tile(np.arange(1, 366), 6)  # 1-365 ni aniq 6 marta takrorlaydi

# Kutilgan naqshni tayinlash va nomuvofiqliklarni belgilash
train_data['expected_day'] = expected_pattern[:len(train_data)]
train_data['day_mismatch'] = train_data['day'] != train_data['expected_day']  # Boolean bayroq

# Grafikni yaratish
plt.figure(figsize=(12, 5))

# Train Data (Tren ma'lumotlarini chizish)
plt.plot(train_data['id'], train_data['day'], linestyle='-', color=train_color, label='Train Data', alpha=0.7)

# Test Data (Test ma'lumotlarini chizish)
plt.plot(test_data['id'], test_data['day'], linestyle='-', color=test_color, label='Test Data', alpha=0.7)

# Nomuvofiq kunlarni qizil markerlar bilan belgilash
plt.scatter(
    train_data.loc[train_data['day_mismatch'], 'id'],  # X-o'qi: nomuvofiq kunlarning ID-lari
    train_data.loc[train_data['day_mismatch'], 'day'], # Y-o'qi: mos keluvchi notoâ€˜gâ€˜ri kunlar
    color=flag_color, marker='X', s=80, label='Mismatched Days', alpha=0.9
)

# Formatlash
plt.xlabel('ID')
plt.ylabel('Day')
plt.title('Trend Plot: Day vs ID (Nomuvofiqliklarni belgilash)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Grafikni ko'rsatish
plt.show()



train_data['day'] = train_data['expected_day']

# Oxirgi kun qiymatini train datadan olish
last_train_day = train_data['day'].iloc[-1]

# Test ma'lumotlar toâ€˜plami uchun ketma-ket kun raqamlarini yaratish
test_data['day'] = np.arange(last_train_day + 1, last_train_day + 1 + len(test_data))

train_data.drop(columns=['expected_day', 'day_mismatch'], errors='ignore', inplace=True)  # Drop 'expected_day' if it exists


train_data.columns


# Ranglarni belgilash
train_color = '#3498db'  # Moviy
test_color = '#e74c3c'   # Qizil
rainfall_colors = {0: '#f1c40f', 1: '#2980b9'}  # Toâ€˜q sariq (yomgâ€˜ir yoâ€˜q), Moviy (yomgâ€˜ir bor)

# Grafikka tushirish uchun raqamli ustunlar
numerical_columns = test_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col in ['id', 'day', 'rainfall']:
    if col in numerical_columns:
        numerical_columns.remove(col)

# Har bir raqamli oâ€˜zgaruvchi uchun chizish tsikli
for column in numerical_columns:
    # Maxsus tartib bilan grafik yaratish
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    # ---- Trend grafigi (ID va Oâ€˜zgaruvchi) ----
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(train_data['id'], train_data[column], linestyle='-', color=train_color, label='Trening maâ€™lumotlari', alpha=0.7)
    ax0.plot(test_data['id'], test_data[column], linestyle='-', color=test_color, label='Test maâ€™lumotlari', alpha=0.7)

    ax0.set_xlabel('ID', fontsize=14)
    ax0.set_ylabel(column, fontsize=14)
    ax0.set_title(f'Trend grafigi: {column} va ID', fontsize=16, fontweight='bold')  # âœ… Tuzatish qoâ€˜llandi
    ax0.legend(fontsize=12)
    ax0.grid(True, linestyle='--', alpha=0.5)

    # ---- Chizilgan nuqtali grafik (Kun va Oâ€˜zgaruvchi) ----
    ax1 = fig.add_subplot(gs[1, 0])
    scatter = ax1.scatter(
        train_data['day'], train_data[column],
        c=train_data['rainfall'].map(rainfall_colors), alpha=0.7
    )
    ax1.set_xlabel('Kun', fontsize=14)
    ax1.set_ylabel(column, fontsize=14)
    ax1.set_title(f'Chizilgan nuqtali grafik: {column} va Kun (Yomgâ€˜ir boâ€˜yicha)', fontsize=16, fontweight='bold')  # âœ… Tuzatish qoâ€˜llandi

    # Yomgâ€˜ir uchun maxsus afsona (legend)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Yomgâ€˜ir yoâ€˜q',
               markersize=10, markerfacecolor=rainfall_colors[0]),
        Line2D([0], [0], marker='o', color='w', label='Yomgâ€˜ir bor',
               markersize=10, markerfacecolor=rainfall_colors[1])
    ]
    ax1.legend(handles=legend_elements, title="Yomgâ€˜ir", fontsize=12, title_fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ---- KDE grafigi (Yomgâ€˜ir boâ€˜yicha taqsimot) ----
    ax2 = fig.add_subplot(gs[1, 1])
    sns.kdeplot(data=train_data, x=column, hue='rainfall', palette=rainfall_colors, ax=ax2, fill=True, common_norm=False, alpha=0.6)

    ax2.set_xlabel(column, fontsize=14)
    ax2.set_ylabel('Zichlik', fontsize=14)
    ax2.set_title(f'{column} ning Yomgâ€˜ir boâ€˜yicha taqsimoti (KDE)', fontsize=16, fontweight='bold')  
    ax2.legend(title='Yomgâ€˜ir', fontsize=12, title_fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Maketni moslashtirish
    plt.tight_layout(pad=3.0)
    plt.show()

    # ---- Har bir oâ€˜zgaruvchidan keyin aniq ajratish ----
    plt.figure(figsize=(16, 0.3))  # Boâ€˜sh joy qoâ€˜shish
    plt.axhline(y=0, color='gray', linewidth=5, linestyle='-') 
    plt.axis('off')
    plt.show()



test_data['winddirection'].fillna(test_data['winddirection'].median(), inplace=True)


# Shamol yoâ€˜nalishini sektorlarga ajratish funksiyasini aniqlash
def wind_sector(direction):
    if pd.isna(direction):
        return np.nan  # Yoâ€˜qolgan qiymatlarni keyinroq toâ€˜gâ€˜rilash uchun saqlab qolamiz
    direction = float(direction)
    if direction >= 315 or direction < 45:
        return 'Shimol'  # North
    elif direction >= 45 and direction < 135:
        return 'Sharq'   # East
    elif direction >= 135 and direction < 225:
        return 'Janub'   # South
    else:
        return 'Gâ€˜arb'   # West

# Xususiyatlar muhandisligi funksiyasini qoâ€˜llash
def perform_feature_engineering(df):
    """
    Ma'lumotlar toâ€˜plamiga xususiyatlar muhandisligi (Feature Engineering) qoâ€˜llanadi va 
    ob-havo bashorat qilish uchun yangi xususiyatlar yaratiladi.
    """

    # 1ï¸�âƒ£ **Fasl xususiyatlari** (kunni sinusoidal koâ€˜rinishda ifodalash)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)

    # 2ï¸�âƒ£ **Kecha-kunduz bogâ€˜liq xususiyatlar** (kechagi ob-havo ma'lumotlarini olish)
    df['cloud_lag1'] = df['cloud'].shift(1).fillna(0)
    df['sunshine_lag1'] = df['sunshine'].shift(1).fillna(0)
    df['humidity_lag1'] = df['humidity'].shift(1).fillna(0)

    # 3ï¸�âƒ£ **3 kunlik siljish oâ€˜rtacha qiymatlari** (trendni kuzatish uchun)
    df['cloud_roll3_mean'] = df['cloud'].rolling(window=3, min_periods=1).mean().fillna(method='bfill')
    df['sunshine_roll3_mean'] = df['sunshine'].rolling(window=3, min_periods=1).mean().fillna(method='bfill')
    df['humidity_roll3_mean'] = df['humidity'].rolling(window=3, min_periods=1).mean().fillna(method='bfill')

    # 4ï¸�âƒ£ **Interaksiya xususiyatlari** (oâ€˜zaro bogâ€˜liq xususiyatlarni yaratish)
    df['cloud_humidity'] = (df['cloud'] * df['humidity']).fillna(0)  
    df['sunshine_cloud_ratio'] = (df['sunshine'] / (df['cloud'] + 1e-5)).fillna(0)  

    # 5ï¸�âƒ£ **Meteorologik xususiyatlar**
    df['temp_range'] = (df['maxtemp'] - df['mintemp']).fillna(df['maxtemp'].median())  # Maksimal-minimal harorat farqi
    df['pressure_diff'] = df['pressure'].diff().fillna(0)  # Bosim oâ€˜zgarishi

    # 6ï¸�âƒ£ **Vaqtga bogâ€˜liq interaksiya xususiyatlari**
    df['cloud_day_sin'] = (df['cloud'] * df['day_sin']).fillna(0)
    df['sunshine_day_cos'] = (df['sunshine'] * df['day_cos']).fillna(0)
    df['humidity_roll3_day_sin'] = (df['humidity_roll3_mean'] * df['day_sin']).fillna(0)

    # 7ï¸�âƒ£ **Shamol yoâ€˜nalishini kategoriyalarga ajratish**
    df['wind_sector'] = df['winddirection'].apply(wind_sector).fillna('Nomaâ€™lum')  

    return df
# Xususiyatlar muhandisligini ma'lumotlarga qoâ€˜llash
# Test ma'lumotlarida ID'larni ajratib olish
id_test = test_data['id']

# Train va test ma'lumotlarini birlashtirib, keyin xususiyatlar muhandisligini qoâ€˜llash
full_data = pd.concat([train_data, test_data], axis=0).sort_values('id')
full_data = perform_feature_engineering(full_data)

# Yana train va test boâ€˜lish
train_data = full_data[full_data['rainfall'].notna()]
test_data = full_data[full_data['rainfall'].isna()]

# Raqamli yangi xususiyatlar
newly_created_vars = [
    'day_sin',              # Kunning sinus qiymati (mavsumiylik uchun)
    'day_cos',              # Kunning kosinus qiymati (mavsumiylik uchun)
    'cloud_lag1',           # Oldingi kunning bulut qoplami
    'sunshine_lag1',        # Oldingi kunning quyoshli soatlari
    'humidity_lag1',        # Oldingi kunning namlik darajasi
    'cloud_roll3_mean',     # 3 kunlik oâ€˜rtacha bulut qoplami
    'sunshine_roll3_mean',  # 3 kunlik oâ€˜rtacha quyoshli soatlar
    'humidity_roll3_mean',  # 3 kunlik oâ€˜rtacha namlik darajasi
    'cloud_humidity',       # Bulut va namlik oâ€˜zaro taâ€™siri
    'sunshine_cloud_ratio', # Quyosh nuri va bulut qoplami nisbati
    'temp_range',           # Kunlik harorat diapazoni
    'pressure_diff',        # Oldingi kundan bosim farqi
    'cloud_day_sin',        # Bulut qoplami va mavsumiy sinus komponenti
    'sunshine_day_cos',     # Quyosh nuri va mavsumiy kosinus komponenti
    'humidity_roll3_day_sin', # 3 kunlik namlik tendensiyasi va mavsumiy sinus
]

# Kategorik yangi xususiyatlar
categorical_new_feats = [
    'wind_sector'           # Shamol yoâ€˜nalishi: Shimol, Sharq, Janub, Gâ€˜arb
]



# Yangi yaratilgan xususiyatlarning yomgâ€˜ir yogâ€˜ishi bilan bogâ€˜liqligini hisoblash
corr_train = train_data[newly_created_vars + ['rainfall']].corr()[['rainfall']]

# Heatmap vizualizatsiyasi (rang panelisiz)
plt.figure(figsize=(10, 2))
ax = sns.heatmap(
    corr_train.T,  # Transponatsiya qilib, xususiyatlarni x-oâ€˜qiga joylash
    annot=True, 
    cmap='viridis', 
    linewidths=0.5, 
    cbar=False, 
    fmt=".2f"
)

# X oâ€˜qidagi belgilarni 90Â° burish
plt.xticks(rotation=90, ha="right")
plt.yticks(rotation=0)
plt.title('Korrelyatsiya Xarita - Yangi Xususiyatlar va Yomgâ€˜ir')
plt.show()



columns_to_drop = [
    'day_cos',
    'pressure_diff',
    'cloud_day_sin',
    'sunshine_day_cos',
    'humidity_roll3_day_sin'
]

train_data.drop(columns=columns_to_drop, inplace=True)
test_data.drop(columns=columns_to_drop+['rainfall'], inplace=True)



# Identify numerical variables
columns_to_check = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Remove unwanted variables
columns_to_check = [col for col in columns_to_check if col not in ['rainfall', 'id']]

# Function to remove outliers using IQR and visualize
def remove_outliers_iqr_with_plot(data, column):
    Q1 = data[column].quantile(0.05)
    Q3 = data[column].quantile(0.95)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the data
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    # Calculate the number of rows deleted
    rows_deleted = len(data) - len(filtered_data)
    
    # Create a 1x2 plot for before & after visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Original Data Boxplot
    sns.boxplot(x=data[column], color='lightblue', ax=axes[0], 
                flierprops={'marker': 'o', 'markersize': 5, 'markerfacecolor': 'red'})
    axes[0].set_title(f'Before Outlier Removal: {column}')
    
    # Highlight Q1, Q3, and Bounds in the first plot
    axes[0].axvline(Q1, color='green', linestyle='--', label='Q1 (15th Percentile)')
    axes[0].axvline(Q3, color='blue', linestyle='--', label='Q3 (85th Percentile)')
    axes[0].axvline(lower_bound, color='red', linestyle='-', label='Lower Bound')
    axes[0].axvline(upper_bound, color='red', linestyle='-', label='Upper Bound')
    axes[0].legend()

    # Boxplot after outlier removal
    sns.boxplot(x=filtered_data[column], color='lightgreen', ax=axes[1], 
                flierprops={'marker': 'o', 'markersize': 5, 'markerfacecolor': 'red'})
    axes[1].set_title(f'After Outlier Removal: {column}')

    plt.suptitle(f'Outlier Detection & Removal for {column}')
    plt.tight_layout()
    plt.show()
    
    return filtered_data, rows_deleted

# Apply function to each numerical column and visualize
rows_deleted_total = 0

for column in columns_to_check:
    train_data, rows_deleted = remove_outliers_iqr_with_plot(train_data, column)
    rows_deleted_total += rows_deleted
    print(f"Rows deleted for {column}: {rows_deleted}")

print(f"Total rows deleted: {rows_deleted_total}")


y = train_data['rainfall']


# Identify numerical variables
numerical_variables = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Remove unwanted variables
numerical_variables = [col for col in numerical_variables if col not in ['rainfall', 'id']]

# [FOR TRAIN]
# Identify features with skewness greater than 0.75
skewed_features = train_data[numerical_variables].skew()[train_data[numerical_variables].skew() > 0.75].index.values

# Print the list of variables to be transformed
print("Features to be transformed (skewness > 0.75):")
display(skewed_features)

# Plot skewed features before transformation
for feature in skewed_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(train_data[feature], bins=50, kde=True, color='blue')
    plt.title(f'Distribution of {feature} before log transformation')
    plt.show()

# Apply log1p transformation to skewed features
train_data[skewed_features] = np.log1p(train_data[skewed_features])

# Plot skewed features after transformation
for feature in skewed_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(train_data[feature], bins=50, kde=True, color='green')
    plt.title(f'Distribution of {feature} after log transformation')
    plt.show()



# Identify features with skewness greater than 0.75
skewed_features = test_data[numerical_variables].skew()[test_data[numerical_variables].skew() > 0.75].index.values

# Print the list of variables to be transformed
print("Features to be transformed (skewness > 0.75):")
display(skewed_features)

# Plot skewed features before transformation
for feature in skewed_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(test_data[feature], bins=50, kde=True, color='blue')
    plt.title(f'Distribution of {feature} before log transformation')
    plt.show()

# Apply log1p transformation to skewed features
test_data[skewed_features] = np.log1p(test_data[skewed_features])

# Plot skewed features after transformation
for feature in skewed_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(test_data[feature], bins=50, kde=True, color='green')
    plt.title(f'Distribution of {feature} after log transformation')
    plt.show()


# Selecting specific columns for encoding
columns_to_encode = [
    'wind_sector'
]

train_data_to_encode = train_data[columns_to_encode]
test_data_to_encode = test_data[columns_to_encode]

# Dropping selected columns for scaling
train_data_to_scale = train_data.drop(columns_to_encode, axis=1)
test_data_to_scale = test_data.drop(columns_to_encode, axis=1)

train_data_encoded = pd.get_dummies(train_data_to_encode, columns=columns_to_encode, drop_first=True)
test_data_encoded = pd.get_dummies(test_data_to_encode, columns=columns_to_encode, drop_first=True)


train_data_encoded.sample(3)


from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler
minmax_scaler = MinMaxScaler()

# Fit the scaler on the training data
minmax_scaler.fit(train_data_to_scale.drop(['rainfall'], axis=1))

# Scale the training data
scaled_data_train = minmax_scaler.transform(train_data_to_scale.drop(['rainfall'], axis=1))
scaled_train_df = pd.DataFrame(scaled_data_train, columns=train_data_to_scale.drop(['rainfall'], axis=1).columns)

# Scale the test data using the parameters from the training data
scaled_data_test = minmax_scaler.transform(test_data_to_scale)
scaled_test_df = pd.DataFrame(scaled_data_test, columns=test_data_to_scale.columns)


scaled_train_df.sample(3)


# Concatenate train datasets
train_data_combined = pd.concat([train_data_encoded.reset_index(drop=True), scaled_train_df.reset_index(drop=True)], axis=1)

# Concatenate test datasets
test_data_combined = pd.concat([test_data_encoded.reset_index(drop=True), scaled_test_df.reset_index(drop=True)], axis=1)


# Separate features and target clearly
X = train_data_combined

# Chronological Train-Validation Split (80-20%)
split_index = int(len(X) * 0.8)
X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]


# Feature Scaling (RobustScaler recommended for Logistic Regression)
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Logistic Regression Model with balanced class weights
from sklearn.linear_model import LogisticRegression
logreg_model = LogisticRegression(
    max_iter=1000, 
    random_state=42, 
    penalty='l2',
    class_weight='balanced',
    solver='liblinear'
)

# Train model on scaled features
logreg_model.fit(X_train, y_train)


# Validation Predictions
y_val_pred_proba = logreg_model.predict_proba(X_val_scaled)[:, 1]

# Optimize threshold (Optional: Using ROC Curve or precision-recall)
threshold = 0.5
y_val_pred = (y_val_pred_proba >= threshold).astype(int)


# Evaluate the Model
from sklearn.metrics import roc_auc_score, classification_report, roc_curve

auc_score = roc_auc_score(y_val, y_val_pred_proba)
print(f'Validation ROC-AUC Score: {auc_score:.4f}')
print(classification_report(y_val, y_val_pred))


fpr, tpr, thresholds = roc_curve(y_val, y_val_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="#3498db", label=f"ROC Curve (AUC = {auc_score:.4f})")
plt.plot([0, 1], [0, 1], color="red", linestyle="--")  # Random guess line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression Model")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


test_predictions = logreg_model.predict_proba(test_data_combined)[:, 1]


# Create submission file
submission_df = pd.DataFrame({
    'id': id_test,
    'rainfall': test_predictions  # Predicted probabilities for rainfall
})

# Save to CSV
submission_df.to_csv("submission.csv", index=False)

# Display first 5 rows
submission_df.head(5)




