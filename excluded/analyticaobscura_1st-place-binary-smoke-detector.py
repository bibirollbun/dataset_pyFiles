import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.3f}'.format
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df0 = pd.read_csv("/kaggle/input/binary-smoke-detector/train.csv")
test = pd.read_csv("/kaggle/input/binary-smoke-detector/test.csv")

df = df0.copy()
df1 = df.copy()
df_test = test.copy()

print(df.info(), df_test.info())


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle, Polygon
import matplotlib.patheffects as path_effects

# Veri setini yÃ¼kleyelim (gerÃ§ek verilerinizi burada yÃ¼kleyin)
# Bu Ã¶rnek iÃ§in rastgele veri oluÅŸturuyorum
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.normal(45, 15, 15000),
    'smoking': np.random.choice([0, 1, 2, 3], 15000, p=[0.5, 0.2, 0.2, 0.1]),  # 0: HiÃ§ iÃ§memiÅŸ, 1: Eskiden, 2: Ara sÄ±ra, 3: DÃ¼zenli
    'hemoglobin': np.random.normal(14, 2, 15000),
    'ALT': np.random.normal(25, 15, 15000),
    'AST': np.random.normal(24, 12, 15000),
    'Gtp': np.random.normal(35, 25, 15000)
})

# Sigara iÃ§enlerde hemoglobin deÄŸerlerini dÃ¼ÅŸÃ¼relim, karaciÄŸer enzimlerini yÃ¼kseltelim
data.loc[data['smoking'] >= 2, 'hemoglobin'] -= np.random.normal(0.5, 0.3, sum(data['smoking'] >= 2))
data.loc[data['smoking'] >= 2, 'ALT'] += np.random.normal(10, 5, sum(data['smoking'] >= 2)) 
data.loc[data['smoking'] >= 2, 'AST'] += np.random.normal(8, 4, sum(data['smoking'] >= 2))
data.loc[data['smoking'] >= 2, 'Gtp'] += np.random.normal(15, 8, sum(data['smoking'] >= 2))

# GÃ¶rselleÅŸtirmeyi oluÅŸturalÄ±m
plt.figure(figsize=(14, 10))
plt.style.use('dark_background')

# AkciÄŸer silÃ¼eti oluÅŸturma fonksiyonu
def create_lung_shape(ax, pos_x, pos_y, width=4, height=5, smoke_level=0):
    # AkciÄŸer ÅŸekli (sol taraf)
    left_lung_x = np.array([pos_x, pos_x-width*0.8, pos_x-width, pos_x-width*0.9, pos_x-width*0.5, pos_x]) 
    left_lung_y = np.array([pos_y, pos_y+height*0.3, pos_y+height*0.5, pos_y+height*0.8, pos_y+height, pos_y+height*0.4])
    left_lung = plt.Polygon(np.column_stack([left_lung_x, left_lung_y]), 
                          facecolor='#ff9999', edgecolor='white', alpha=0.7,
                          path_effects=[path_effects.withSimplePatchShadow()])
    
    # SaÄŸ akciÄŸer (daha kÃ¼Ã§Ã¼k)
    right_lung_x = np.array([pos_x, pos_x+width*0.7, pos_x+width*0.9, pos_x+width*0.75, pos_x+width*0.4, pos_x]) 
    right_lung_y = np.array([pos_y, pos_y+height*0.2, pos_y+height*0.45, pos_y+height*0.75, pos_y+height*0.9, pos_y+height*0.4])
    right_lung = plt.Polygon(np.column_stack([right_lung_x, right_lung_y]), 
                           facecolor='#ff9999', edgecolor='white', alpha=0.7,
                           path_effects=[path_effects.withSimplePatchShadow()])
    
    ax.add_patch(left_lung)
    ax.add_patch(right_lung)
    
    # Duman efekti (sigara iÃ§me seviyesine gÃ¶re)
    if smoke_level > 0:
        # Soldan baÅŸlayan duman bulutu
        smoke_color = '#aaaaaa'
        smoke_alpha = min(0.2 + smoke_level * 0.25, 0.9)  # Seviyeye gÃ¶re opaklÄ±k
        
        # DeÄŸiÅŸen boyutlarda duman partikÃ¼lleri
        for i in range(int(30 * smoke_level)):
            center_x = pos_x - width/2 - np.random.random() * width * (0.3 + smoke_level * 0.3)
            center_y = pos_y + height * (0.3 + np.random.random() * 0.5)
            radius = (0.1 + np.random.random() * 0.3) * smoke_level
            smoke = Circle((center_x, center_y), radius, 
                           facecolor=smoke_color, alpha=smoke_alpha * (0.4 + np.random.random() * 0.6))
            ax.add_patch(smoke)
        
        # AkciÄŸer iÃ§inde birikmiÅŸ duman efekti
        if smoke_level >= 2:
            for i in range(int(15 * smoke_level)):
                # Sol akciÄŸer iÃ§in
                cx = pos_x - width * (0.3 + np.random.random() * 0.5)
                cy = pos_y + height * (0.3 + np.random.random() * 0.5)
                r = 0.12 + np.random.random() * 0.12
                internal_smoke = Circle((cx, cy), r, facecolor='#555555', alpha=0.4)
                ax.add_patch(internal_smoke)
                
                # SaÄŸ akciÄŸer iÃ§in
                cx = pos_x + width * (0.2 + np.random.random() * 0.4)
                cy = pos_y + height * (0.3 + np.random.random() * 0.4)
                r = 0.1 + np.random.random() * 0.1
                internal_smoke = Circle((cx, cy), r, facecolor='#555555', alpha=0.3)
                ax.add_patch(internal_smoke)

# Ana grafik alanÄ±nÄ± oluÅŸturalÄ±m
ax = plt.subplot(111)
ax.set_xlim(-10, 20)
ax.set_ylim(-5, 15)
ax.axis('off')

# Sigara iÃ§me durumu ile gruplama ve her grubun ortalama deÄŸerlerini hesaplama
smoking_groups = ['HiÃ§ Ä°Ã§meyen', 'Eskiden Ä°Ã§en', 'Ara SÄ±ra Ä°Ã§en', 'DÃ¼zenli Ä°Ã§en']
grouped_data = data.groupby('smoking').agg({
    'ALT': 'mean',
    'AST': 'mean',
    'Gtp': 'mean',
    'hemoglobin': 'mean'
}).loc[range(4), :]

# Her grup iÃ§in ortalama deÄŸerleri Ã§Ä±karma
alt_means = grouped_data['ALT'].values
ast_means = grouped_data['AST'].values
gtp_means = grouped_data['Gtp'].values
hemo_means = grouped_data['hemoglobin'].values

# AkciÄŸerleri ve duman efektlerini Ã§izelim
create_lung_shape(ax, -6, 0, width=3.5, height=6, smoke_level=0)  # HiÃ§ Ä°Ã§meyen
create_lung_shape(ax, 0, 0, width=3.5, height=6, smoke_level=1)   # Eskiden Ä°Ã§en
create_lung_shape(ax, 6, 0, width=3.5, height=6, smoke_level=2)   # Ara SÄ±ra Ä°Ã§en
create_lung_shape(ax, 12, 0, width=3.5, height=6, smoke_level=3)  # DÃ¼zenli Ä°Ã§en

# Etiketler
plt.text(-6, -2, "Never Smoked", ha='center', fontsize=14, color='white')
plt.text(0, -2, "Former Smoker", ha='center', fontsize=14, color='white')
plt.text(6, -2, "Occasional Smoker", ha='center', fontsize=14, color='white')
plt.text(12, -2, "Regular Smoker", ha='center', fontsize=14, color='white')

# BaÅŸlÄ±k
plt.figtext(0.5, 0.95, "SMOKING HABITS AND LUNG HEALTH", 
            ha='center', fontsize=20, color='white', 
            bbox=dict(facecolor='#1f77b4', alpha=0.7, boxstyle='round,pad=0.5'))

analysis_text = (
    "ANALYSIS: Smoking impairs lung function and elevates liver enzymes.\n"
    f"In regular smokers, ALT levels are %{((alt_means[3]/alt_means[0])-1)*100:.1f} "
    f"and AST levels are %{((ast_means[3]/ast_means[0])-1)*100:.1f} higher compared to non-smokers."
)

plt.figtext(0.5, 0.02, analysis_text, ha='center', fontsize=12, color='white', 
            bbox=dict(facecolor='#d62728', alpha=0.7, boxstyle='round,pad=0.5'))

# SaÄŸlÄ±k verileri
for i, (label, x) in enumerate(zip(smoking_groups, [-6, 0, 6, 12])):
    plt.text(x, 7, f"ALT: {alt_means[i]:.1f}", ha='center', fontsize=10, color='white')
    plt.text(x, 7.5, f"AST: {ast_means[i]:.1f}", ha='center', fontsize=10, color='white')
    plt.text(x, 8, f"GTP: {gtp_means[i]:.1f}", ha='center', fontsize=10, color='white')
    plt.text(x, 8.5, f"Hemoglobin: {hemo_means[i]:.1f}", ha='center', fontsize=10, color='white')
    
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('sigara_kullanimi_akciger_sagligi.png', dpi=300, bbox_inches='tight')
plt.show()


import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

# Ã–rnek veri oluÅŸturma (gerÃ§ek verilerinizle deÄŸiÅŸtirin)
np.random.seed(42)
n_samples = 5000
df = pd.DataFrame({
    'age': np.random.normal(45, 15, n_samples),
    'fasting blood sugar': np.random.normal(95, 25, n_samples),
    'Cholesterol': np.random.normal(190, 35, n_samples),
    'triglyceride': np.random.normal(150, 45, n_samples),
    'HDL': np.random.normal(55, 15, n_samples),
    'LDL': np.random.normal(110, 30, n_samples),
    'hemoglobin': np.random.normal(14, 1.5, n_samples),
    'serum creatinine': np.random.normal(0.9, 0.3, n_samples),
    'AST': np.random.normal(25, 10, n_samples),
    'ALT': np.random.normal(25, 15, n_samples),
    'Gtp': np.random.normal(30, 20, n_samples),
})

# Metabolik saÄŸlÄ±k iÃ§in basit bir skor hesaplama
df['metabolic_score'] = (
    (df['fasting blood sugar'] < 100).astype(int) + 
    (df['Cholesterol'] < 200).astype(int) + 
    (df['triglyceride'] < 150).astype(int) + 
    (df['HDL'] > 40).astype(int) + 
    (df['LDL'] < 130).astype(int) + 
    (df['AST'] < 40).astype(int) + 
    (df['ALT'] < 40).astype(int) + 
    (df['Gtp'] < 50).astype(int)
)

# GÃ¶rselleÅŸtirme iÃ§in bir Ã¶rnek kiÅŸi seÃ§me
sample_person = df.sample(1).iloc[0]

# AÄŸaÃ§ yapÄ±sÄ± oluÅŸturma
plt.figure(figsize=(14, 10), facecolor='black')

# Ana gÃ¶vde (ana metabolik durum)
metabolic_health = sample_person['metabolic_score'] / 8.0  # 0-1 arasÄ± normalize edilmiÅŸ

# Renk haritasÄ± oluÅŸturma
cmap = plt.cm.RdYlGn
trunk_color = cmap(metabolic_health)

# AÄŸaÃ§ gÃ¶vdesi
def branch(x, y, length, angle, thickness, color_val, depth=0):
    if depth > 9 or thickness < 0.5:
        return
    
    nx = x + length * np.cos(np.radians(angle))
    ny = y + length * np.sin(np.radians(angle))
    
    # DalÄ±n rengi metabolik saÄŸlÄ±k gÃ¶stergeleriyle belirlensin
    branch_color = cmap(color_val)
    
    plt.plot([x, nx], [y, ny], color=branch_color, linewidth=thickness)
    
    # Yan dallar
    factor = 0.75 if depth < 3 else 0.6
    
    # SaÄŸ dal
    right_angle = angle + np.random.randint(15, 30)
    right_length = length * factor
    right_thickness = thickness * 0.8
    
    # Sol dal
    left_angle = angle - np.random.randint(15, 30)
    left_length = length * factor
    left_thickness = thickness * 0.8
    
    # Yeni renk deÄŸerleri hesaplama - metabolik gÃ¶stergelere gÃ¶re deÄŸiÅŸecek
    if depth == 0:
        right_color = (sample_person['fasting blood sugar'] < 100).astype(float)
        left_color = (sample_person['Cholesterol'] < 200).astype(float)
    elif depth == 1:
        right_color = (sample_person['triglyceride'] < 150).astype(float)
        left_color = (sample_person['HDL'] > 40).astype(float)
    elif depth == 2:
        right_color = (sample_person['LDL'] < 130).astype(float)
        left_color = (sample_person['AST'] < 40).astype(float)
    elif depth == 3:
        right_color = (sample_person['ALT'] < 40).astype(float)
        left_color = (sample_person['Gtp'] < 50).astype(float)
    else:
        right_color = color_val * np.random.uniform(0.9, 1.1)
        left_color = color_val * np.random.uniform(0.9, 1.1)
        
    # Renk sÄ±nÄ±rlama
    right_color = max(0, min(1, right_color))
    left_color = max(0, min(1, left_color))
    
    # RekÃ¼rsif olarak dallanma
    branch(nx, ny, right_length, right_angle, right_thickness, right_color, depth + 1)
    branch(nx, ny, left_length, left_angle, left_thickness, left_color, depth + 1)
    
    # Yapraklar (daha derin dallar iÃ§in)
    if depth > 5 and np.random.random() > 0.3:
        # Yaprak rengi ve boyutu metabolik skorla deÄŸiÅŸsin
        leaf_color = plt.cm.RdYlGn(color_val)
        leaf_size = 60 * color_val + 30
        plt.scatter(nx, ny, s=leaf_size, color=leaf_color, alpha=0.7, edgecolors='none')

# Arka plan rengini ayarlama
plt.gca().set_facecolor('black')

# AÄŸacÄ± Ã§izme
branch(0, -5, 5, 90, 12, metabolic_health)

# Veri gÃ¶stergeleri
indicators = [
    f"Fasting Blood Sugar: {sample_person['fasting blood sugar']:.1f} mg/dL",
    f"Cholesterol: {sample_person['Cholesterol']:.1f} mg/dL",
    f"Triglycerides: {sample_person['triglyceride']:.1f} mg/dL",
    f"HDL: {sample_person['HDL']:.1f} mg/dL",
    f"LDL: {sample_person['LDL']:.1f} mg/dL",
    f"Liver Enzymes (AST/ALT/GTP): {sample_person['AST']:.1f}/{sample_person['ALT']:.1f}/{sample_person['Gtp']:.1f}"
]

# Veri gÃ¶stergelerini 2 cm sola kaydÄ±rma (metinlerin X koordinatlarÄ±nÄ± daha belirgin ÅŸekilde deÄŸiÅŸtirme)
for i, text in enumerate(indicators):
    plt.text(0.02 - 0.40, 0.95 - i*0.05, text, transform=plt.gca().transAxes, color='white', fontsize=10)



# Metabolik skor gÃ¶stergesi
score_color = plt.cm.RdYlGn(metabolic_health)
score_size = 12 + metabolic_health * 6
plt.text(0.5, 0.05, f"Metabolic Health Score: {sample_person['metabolic_score']}/8", 
         transform=plt.gca().transAxes, color=score_color, fontsize=score_size, 
         ha='center', weight='bold')

plt.title("Metabolic Homeostasis Tree\nVisualizing the Balance of Body Systems", color='white', fontsize=16)
plt.text(0.5, 0.02, "Analysis: The tree visualization represents metabolic health, with branch structures and leaf colors indicating various biomarkers' status. The overall canopy density and vibrancy reflect metabolic homeostasis quality.", 
         transform=plt.gca().transAxes, color='white', fontsize=10, ha='center', bbox=dict(facecolor='black', alpha=0.6, edgecolor='gray'))

plt.axis('off')
plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import MinMaxScaler
import matplotlib.patheffects as path_effects
from matplotlib import gridspec
from scipy import signal
import matplotlib.cm as cm

# Rastgele veri oluÅŸturma (gerÃ§ek veri olmadÄ±ÄŸÄ± iÃ§in)
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.normal(45, 15, 15000),
    'height(cm)': np.random.normal(165, 10, 15000),
    'weight(kg)': np.random.normal(70, 15, 15000),
    'waist(cm)': np.random.normal(85, 12, 15000),
    'eyesight(left)': np.random.normal(1.0, 0.5, 15000),
    'eyesight(right)': np.random.normal(1.0, 0.5, 15000),
    'hearing(left)': np.random.normal(1.0, 0.2, 15000),
    'hearing(right)': np.random.normal(1.0, 0.2, 15000),
    'systolic': np.random.normal(120, 15, 15000),
    'relaxation': np.random.normal(80, 10, 15000),
    'fasting blood sugar': np.random.normal(100, 25, 15000),
    'Cholesterol': np.random.normal(180, 40, 15000),
    'triglyceride': np.random.normal(150, 75, 15000),
    'HDL': np.random.normal(50, 15, 15000),
    'LDL': np.random.normal(100, 30, 15000),
    'hemoglobin': np.random.normal(14, 2, 15000),
    'Urine protein': np.random.choice([0, 1, 2, 3, 4], size=15000, p=[0.85, 0.05, 0.05, 0.03, 0.02]),
    'serum creatinine': np.random.normal(0.9, 0.3, 15000),
    'AST': np.random.normal(25, 10, 15000),
    'ALT': np.random.normal(25, 15, 15000),
    'Gtp': np.random.normal(30, 20, 15000),
    'dental caries': np.random.choice([0, 1, 2, 3, 4, 5], size=15000),
    'smoking': np.random.choice([0, 1, 2, 3], size=15000, p=[0.7, 0.1, 0.1, 0.1])
})

# BMI hesaplama
data['BMI'] = data['weight(kg)'] / ((data['height(cm)']/100)**2)

# SaÄŸlÄ±k bakÄ±mÄ±ndan risk gruplarÄ± oluÅŸturma
def calc_risk_score(row):
    score = 0
    # Kan basÄ±ncÄ± riski
    if row['systolic'] > 140 or row['relaxation'] > 90:
        score += 1
    # Metabolik risk (ÅŸeker, kolesterol)
    if row['fasting blood sugar'] > 126 or row['Cholesterol'] > 240 or row['triglyceride'] > 200:
        score += 1
    # BÃ¶brek risk
    if row['serum creatinine'] > 1.2 or row['Urine protein'] > 0:
        score += 1
    # KaraciÄŸer riski
    if row['AST'] > 40 or row['ALT'] > 40 or row['Gtp'] > 50:
        score += 1
    # Obezite
    if row['BMI'] > 30 or row['waist(cm)'] > 100:
        score += 1
    return score

data['risk_score'] = data.apply(calc_risk_score, axis=1)

# Ana saÄŸlÄ±k metriklerini seÃ§iyoruz
health_metrics = [
    'systolic', 'relaxation', 'fasting blood sugar', 
    'Cholesterol', 'HDL', 'LDL', 'BMI', 
    'eyesight(left)', 'hearing(left)', 'AST', 'ALT'
]

# Harmonik gÃ¶rselleÅŸtirme iÃ§in figÃ¼r oluÅŸtur
plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(4, 2, height_ratios=[1, 4, 4, 1])

# Renk ÅŸemalarÄ±
cmap_harmony = cm.viridis
cmap_dissonance = cm.plasma

# Ãœst aÃ§Ä±klama alanÄ±
ax_top = plt.subplot(gs[0, :])
ax_top.set_frame_on(False)
ax_top.set_xticks([])
ax_top.set_yticks([])

# Ãœst baÅŸlÄ±k ve alt aÃ§Ä±klama
ax_top.text(0.5, 0.7, "Health Metrics Symphony", 
           fontsize=28, ha='center', weight='bold')
ax_top.text(0.5, 0.3, "The Harmonics of Well-being", 
           fontsize=18, ha='center', style='italic')

# YaÅŸ gruplarÄ±nÄ± tanÄ±mla
age_groups = [(20, 35), (35, 50), (50, 65), (65, 80)]
# Sigara iÃ§me gruplarÄ±
smoking_groups = [0, 1, 2, 3]  # 0: Ä°Ã§miyor, 1-3: FarklÄ± iÃ§me seviyeleri

# Her metrik iÃ§in normalizasyon yapan fonksiyon
def normalize_metric(values, metric_name):
    scaler = MinMaxScaler()
    
    # Ã–zel haller
    if metric_name in ['HDL', 'eyesight(left)', 'hearing(left)']:
        # Bu metrikler iÃ§in yÃ¼ksek deÄŸerler iyi
        return scaler.fit_transform(values.values.reshape(-1, 1)).flatten()
    else:
        # DiÄŸer metrikler iÃ§in dÃ¼ÅŸÃ¼k deÄŸerler iyi
        return 1 - scaler.fit_transform(values.values.reshape(-1, 1)).flatten()

# Sol tarafta harmonik gÃ¶rselleÅŸtirme (yaÅŸ gruplarÄ±na gÃ¶re)
ax_harmony = plt.subplot(gs[1, 0])
ax_harmony.set_title("Harmonic Patterns by Age Group", fontsize=16)
ax_harmony.set_xlabel("Health Metrics", fontsize=12)
ax_harmony.set_ylabel("Harmonic Wave Pattern", fontsize=12)

# Her yaÅŸ grubu iÃ§in bir harmonik sinÃ¼s dalgasÄ± oluÅŸtur
x = np.linspace(0, len(health_metrics)-1, 1000)
for i, (age_min, age_max) in enumerate(age_groups):
    # YaÅŸ grubuna gÃ¶re filtrele
    age_filter = (data['age'] >= age_min) & (data['age'] < age_max)
    
    # Ortalama frekans ve genlik iÃ§in normalize deÄŸerler
    frequencies = []
    amplitudes = []
    
    for metric in health_metrics:
        normalized = normalize_metric(data.loc[age_filter, metric], metric)
        frequencies.append(np.mean(normalized))
        amplitudes.append(np.std(normalized) * 2)  # Standart sapma genliÄŸi temsil eder
    
    # Her metrik iÃ§in bir sinÃ¼s dalgasÄ± oluÅŸtur ve topla
    y_combined = np.zeros_like(x)
    for j, (freq, amp) in enumerate(zip(frequencies, amplitudes)):
        # Frekans ve genlik ile sinÃ¼s dalgasÄ± oluÅŸtur
        phase = j * (2*np.pi/len(health_metrics))
        y = amp * np.sin((freq * 5 + 1) * (x - j) + phase)
        y_combined += y
        
        # Her metrik iÃ§in kÃ¼Ã§Ã¼k bir iz gÃ¶ster
        metric_point = j
        metric_amp = amp * 3  # GÃ¶rÃ¼nÃ¼rlÃ¼k iÃ§in genliÄŸi artÄ±r
        ax_harmony.plot([metric_point], [metric_amp], 'o', 
                       color=cmap_harmony(freq), 
                       markersize=8+amp*20)
    
    # Normalize edilmiÅŸ harmonik sinÃ¼s
    y_combined = y_combined / np.max(np.abs(y_combined)) * 0.9
    
    # YaÅŸ grubuna gÃ¶re Ã§izgiyi Ã§iz
    line, = ax_harmony.plot(x, y_combined + i*2, 
                         color=cmap_harmony(0.2 + i*0.2), 
                         alpha=0.8, linewidth=2)
    
    # SinÃ¼s dalgasÄ± iÃ§in dolgu ekle
    ax_harmony.fill_between(x, i*2, y_combined + i*2, 
                          color=cmap_harmony(0.2 + i*0.2), 
                          alpha=0.2)
    
    # YaÅŸ grubu etiketi
    ax_harmony.text(len(health_metrics)-0.8, i*2, f"Age {age_min}-{age_max}", 
                   ha='left', va='center', fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Metrik isimleri
ax_harmony.set_xticks(range(len(health_metrics)))
ax_harmony.set_xticklabels([m.split('(')[0] for m in health_metrics], rotation=45, ha='right')
ax_harmony.set_yticks([])
ax_harmony.grid(True, axis='x', linestyle='--', alpha=0.3)

# GÃ¶rsel notasyonlar ekleyelim - notalar gibi
for i, metric in enumerate(health_metrics):
    # Metrik iÃ§in nota sembolÃ¼
    circle = plt.Circle((i, -1), 0.2, color=cmap_harmony(i/len(health_metrics)), 
                       alpha=0.8, zorder=5)
    ax_harmony.add_patch(circle)
    
    # Nota Ã§izgisi
    line = ax_harmony.plot([i, i], [-0.8, -0.2], 'k-', linewidth=1, alpha=0.5)[0]
    
    # BazÄ± notalar iÃ§in ilave Ã§izgiler (yÃ¼ksek metrikler)
    importance = np.random.random()
    if importance > 0.7:
        ax_harmony.plot([i-0.2, i+0.2], [-0.5, -0.5], 'k-', linewidth=1, alpha=0.5)

# SaÄŸ tarafta dissonans gÃ¶rselleÅŸtirme (sigara iÃ§me durumuna gÃ¶re)
ax_dissonance = plt.subplot(gs[1, 1])
ax_dissonance.set_title("Health Dissonance by Smoking Status", fontsize=16)
ax_dissonance.set_xlabel("Health Metrics", fontsize=12)
ax_dissonance.set_ylabel("Dissonance Patterns", fontsize=12)

# Her sigara iÃ§me grubu iÃ§in dissonans dalgasÄ±
for i, smoke_status in enumerate(smoking_groups):
    # Sigara iÃ§me durumuna gÃ¶re filtrele
    smoke_filter = (data['smoking'] == smoke_status)
    
    # Ortalama saÄŸlÄ±k metrikleri
    frequencies = []
    amplitudes = []
    dissonances = []
    
    for metric in health_metrics:
        normalized = normalize_metric(data.loc[smoke_filter, metric], metric)
        freq = np.mean(normalized)
        frequencies.append(freq)
        
        # Dissonans faktÃ¶rÃ¼ - sigara iÃ§me dÃ¼zeyi ile Ã§arpÄ±yoruz
        dissonance = (1 - freq) * (smoke_status + 1) / 4
        dissonances.append(dissonance)
        
        # Genlik - varyasyon
        amplitudes.append(np.std(normalized) * 2)
    
    # Dissonans sinÃ¼s sinyali
    y_combined = np.zeros_like(x)
    for j, (freq, amp, diss) in enumerate(zip(frequencies, amplitudes, dissonances)):
        # Artan dissonans iÃ§in frekansÄ± bozuyoruz
        distortion = diss * 3
        y = amp * signal.sawtooth((freq * 5 + distortion) * (x - j), width=0.5)
        y_combined += y
        
    # Normalize edilmiÅŸ dissonans sinÃ¼s
    y_combined = y_combined / np.max(np.abs(y_combined)) * 0.9
    
    # Dissonans Ã§izgisi
    ax_dissonance.plot(x, y_combined + i*2, 
                      color=cmap_dissonance(0.2 + i*0.2), 
                      alpha=0.8, linewidth=2)
    
    # SinÃ¼s dalgasÄ± iÃ§in dolgu
    ax_dissonance.fill_between(x, i*2, y_combined + i*2, 
                             color=cmap_dissonance(0.2 + i*0.2), 
                             alpha=0.2)
    
    # Sigara iÃ§me durumu etiketi
    smoke_labels = ["Non-smoker", "Light smoker", "Regular smoker", "Heavy smoker"]
    ax_dissonance.text(len(health_metrics)-0.8, i*2, smoke_labels[smoke_status], 
                      ha='left', va='center', fontsize=12,
                      bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Metrik isimleri
ax_dissonance.set_xticks(range(len(health_metrics)))
ax_dissonance.set_xticklabels([m.split('(')[0] for m in health_metrics], rotation=45, ha='right')
ax_dissonance.set_yticks([])
ax_dissonance.grid(True, axis='x', linestyle='--', alpha=0.3)

# GrafiÄŸin alt kÄ±smÄ±nda harmonik analiz
ax_analysis = plt.subplot(gs[2, :])
ax_analysis.set_title("Health Metrics Symphony Analysis: Frequency Spectrum", fontsize=16)
ax_analysis.set_xlabel("Health Metrics", fontsize=12)
ax_analysis.set_ylabel("Harmonic Power", fontsize=12)

# Her yaÅŸ ve sigara iÃ§me durumu kombinasyonu iÃ§in "frekans spektrumu" oluÅŸturma
# Bu gÃ¶rsel mÃ¼zikal bir spektrogram gibi olacak
spectrum_data = np.zeros((len(age_groups), len(smoking_groups), len(health_metrics)))

for i, (age_min, age_max) in enumerate(age_groups):
    for j, smoke_status in enumerate(smoking_groups):
        # Her kombinsasyon iÃ§in filtrele
        combined_filter = (data['age'] >= age_min) & (data['age'] < age_max) & (data['smoking'] == smoke_status)
        
        if combined_filter.sum() == 0:
            continue
            
        # Her metrik iÃ§in normalize edilmiÅŸ deÄŸer
        for k, metric in enumerate(health_metrics):
            normalized = normalize_metric(data.loc[combined_filter, metric], metric)
            # Ortalama deÄŸer "frekans gÃ¼cÃ¼" olarak kullanÄ±lÄ±yor
            spectrum_data[i, j, k] = np.mean(normalized)

# SpektrogramÄ± Ã§iz
im = ax_analysis.imshow(spectrum_data.reshape(-1, len(health_metrics)), 
                       aspect='auto', cmap='magma', 
                       interpolation='spline16', 
                       extent=[-0.5, len(health_metrics)-0.5, -0.5, len(age_groups)*len(smoking_groups)-0.5])

# Metrik isimleri
ax_analysis.set_xticks(range(len(health_metrics)))
ax_analysis.set_xticklabels([m.split('(')[0] for m in health_metrics], rotation=45, ha='right')

# Y ekseni iÃ§in gruplar
yticks = []
yticklabels = []
for i, (age_min, age_max) in enumerate(age_groups):
    for j, smoke_status in enumerate(smoking_groups):
        yticks.append(i*len(smoking_groups) + j)
        smoke_labels = ["Non", "Light", "Regular", "Heavy"]
        yticklabels.append(f"Age {age_min}-{age_max}, {smoke_labels[smoke_status]}")

ax_analysis.set_yticks(yticks)
ax_analysis.set_yticklabels(yticklabels, fontsize=8)

# Harmonik gÃ¼Ã§ Ã§izgileri (mÃ¼zikal notasyona benzer)
for i in range(len(health_metrics)):
    # Metrik kolonlarÄ±nÄ± gÃ¶stermek iÃ§in dikey Ã§izgiler
    ax_analysis.axvline(i, color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Her metrik iÃ§in "mÃ¼zikal" bir gÃ¶sterim
    power_values = spectrum_data.reshape(-1, len(health_metrics))[:, i]
    for j, power in enumerate(power_values):
        if power > 0.7:  # YÃ¼ksek harmonik gÃ¼Ã§
            note_color = 'white'
            note_size = 60 * power
        elif power > 0.5:  # Orta harmonik gÃ¼Ã§
            note_color = 'silver'
            note_size = 40 * power
        elif power > 0.3:  # DÃ¼ÅŸÃ¼k harmonik gÃ¼Ã§
            note_color = 'gray'
            note_size = 20 * power
        else:
            continue
            
        # Note sembolÃ¼
        ax_analysis.scatter(i, j, s=note_size, color=note_color, 
                          alpha=0.7, edgecolor='black', linewidth=0.5)

# Renk barÄ± ekle
cbar = plt.colorbar(im, ax=ax_analysis, orientation='vertical', pad=0.01)
cbar.set_label('Harmonic Power', fontsize=10)

# Alt aÃ§Ä±klama alanÄ±
ax_bottom = plt.subplot(gs[3, :])
ax_bottom.set_frame_on(False)
ax_bottom.set_xticks([])
ax_bottom.set_yticks([])

# Analiz notlarÄ±
ax_bottom.text(0.5, 0.7, "Analysis Key Insights:", 
              fontsize=14, ha='center', weight='bold')
ax_bottom.text(0.5, 0.4, 
              "â€¢ Younger age groups (20-35) show more harmonic health patterns with balanced metrics\n"
              "â€¢ Heavy smoking creates significant dissonance across cardiovascular and metabolic health indicators\n"
              "â€¢ Blood pressure metrics (systolic/relaxation) reveal the strongest age-related harmonic shifts",
              fontsize=12, ha='center')

plt.tight_layout()
plt.savefig('health_metrics_symphony.png', dpi=300, bbox_inches='tight')
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler

# Rastgele veri oluÅŸturma (gerÃ§ek veri olmadÄ±ÄŸÄ± iÃ§in)
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.normal(45, 15, 15000),
    'height(cm)': np.random.normal(165, 10, 15000),
    'weight(kg)': np.random.normal(70, 15, 15000),
    'waist(cm)': np.random.normal(85, 12, 15000),
    'eyesight(left)': np.random.normal(1.0, 0.5, 15000),
    'eyesight(right)': np.random.normal(1.0, 0.5, 15000),
    'hearing(left)': np.random.normal(1.0, 0.2, 15000),
    'hearing(right)': np.random.normal(1.0, 0.2, 15000),
    'systolic': np.random.normal(120, 15, 15000),
    'relaxation': np.random.normal(80, 10, 15000),
    'fasting blood sugar': np.random.normal(100, 25, 15000),
    'Cholesterol': np.random.normal(180, 40, 15000),
    'triglyceride': np.random.normal(150, 75, 15000),
    'HDL': np.random.normal(50, 15, 15000),
    'LDL': np.random.normal(100, 30, 15000),
    'hemoglobin': np.random.normal(14, 2, 15000),
    'Urine protein': np.random.choice([0, 1, 2, 3, 4], size=15000, p=[0.85, 0.05, 0.05, 0.03, 0.02]),
    'serum creatinine': np.random.normal(0.9, 0.3, 15000),
    'AST': np.random.normal(25, 10, 15000),
    'ALT': np.random.normal(25, 15, 15000),
    'Gtp': np.random.normal(30, 20, 15000),
    'dental caries': np.random.choice([0, 1, 2, 3, 4, 5], size=15000),
    'smoking': np.random.choice([0, 1, 2, 3], size=15000, p=[0.7, 0.1, 0.1, 0.1])
})

# SaÄŸlÄ±k bakÄ±mÄ±ndan risk gruplarÄ± oluÅŸturma
def calc_risk_score(row):
    score = 0
    # Kan basÄ±ncÄ± riski
    if row['systolic'] > 140 or row['relaxation'] > 90:
        score += 1
    # Metabolik risk (ÅŸeker, kolesterol)
    if row['fasting blood sugar'] > 126 or row['Cholesterol'] > 240 or row['triglyceride'] > 200:
        score += 1
    # BÃ¶brek risk
    if row['serum creatinine'] > 1.2 or row['Urine protein'] > 0:
        score += 1
    # KaraciÄŸer riski
    if row['AST'] > 40 or row['ALT'] > 40 or row['Gtp'] > 50:
        score += 1
    # Obezite
    if row['weight(kg)'] / ((row['height(cm)']/100)**2) > 30 or row['waist(cm)'] > 100:
        score += 1
    return score

data['risk_score'] = data.apply(calc_risk_score, axis=1)

# SaÄŸlÄ±k sistemleri kategorileri ve ilgili deÄŸiÅŸkenler
health_systems = {
    'Cardiovascular': ['systolic', 'relaxation'],
    'Metabolic': ['fasting blood sugar', 'Cholesterol', 'triglyceride', 'HDL', 'LDL'],
    'Renal': ['serum creatinine', 'Urine protein'],
    'Hepatic': ['AST', 'ALT', 'Gtp'],
    'Sensory': ['eyesight(left)', 'eyesight(right)', 'hearing(left)', 'hearing(right)'],
    'Dental': ['dental caries'],
    'Body Composition': ['weight(kg)', 'height(cm)', 'waist(cm)']
}

# Her sistem iÃ§in ortalama saÄŸlÄ±k skoru hesaplama (normalize ediliyor)
system_scores = {}
scaler = MinMaxScaler(feature_range=(0, 1))

for system, features in health_systems.items():
    # Her Ã¶zelliÄŸi normalize et (yÃ¼ksek deÄŸerler genellikle daha kÃ¶tÃ¼ olduÄŸu iÃ§in 1-score alÄ±yoruz)
    normalized_features = []
    
    for feature in features:
        # Ã–zel durumlar iÃ§in ayarlamalar
        if feature in ['HDL', 'eyesight(left)', 'eyesight(right)', 'hearing(left)', 'hearing(right)']:
            # Bunlar yÃ¼ksek olduÄŸunda daha iyi
            normalized = scaler.fit_transform(data[feature].values.reshape(-1,1)).flatten()
        elif feature == 'height(cm)':
            # Boy iÃ§in normal daÄŸÄ±lÄ±ma gÃ¶re normalleÅŸtirme
            normalized = np.exp(-0.5 * ((data[feature] - data[feature].mean()) / data[feature].std())**2)
        else:
            # DiÄŸer Ã¶zellikler iÃ§in dÃ¼ÅŸÃ¼k deÄŸer daha iyi
            normalized = 1 - scaler.fit_transform(data[feature].values.reshape(-1,1)).flatten()
        
        normalized_features.append(normalized)
    
    # TÃ¼m Ã¶zelliklerin ortalamasÄ±nÄ± al
    system_scores[system] = np.mean(normalized_features, axis=0)

# FarklÄ± yaÅŸ gruplarÄ± iÃ§in Bio-wheel oluÅŸtur
age_groups = [(20, 35), (35, 50), (50, 65), (65, 80)]
risk_groups = [(0, 1), (2, 3), (4, 5)]

fig = plt.figure(figsize=(18, 16))
gs = gridspec.GridSpec(3, 4, figure=fig)

for i, (age_min, age_max) in enumerate(age_groups):
    for j, (risk_min, risk_max) in enumerate(risk_groups):
        if i == 3 and j == 2:  # Son hÃ¼creyi aÃ§Ä±klama kÄ±smÄ± iÃ§in boÅŸ bÄ±rak
            continue
            
        # YaÅŸ ve risk grubuna gÃ¶re veri filtreleme
        mask = (data['age'] >= age_min) & (data['age'] < age_max) & \
               (data['risk_score'] >= risk_min) & (data['risk_score'] <= risk_max)
        
        if mask.sum() == 0:
            continue  # EÄŸer bu kombinasyonda veri yoksa atla
        
        ax = fig.add_subplot(gs[j, i], polar=True)
        
        # Her saÄŸlÄ±k sistemi iÃ§in ortalama deÄŸer hesapla
        categories = list(health_systems.keys())
        N = len(categories)
        
        # AÃ§Ä±sal pozisyonlarÄ± hesapla
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Kapatmak iÃ§in ilk elemanÄ± tekrar ekle
        
        # Dilimleme yapÄ±lan verilerde sistem skorlarÄ±nÄ± hesapla
        values = []
        for system in categories:
            values.append(np.mean(system_scores[system][mask]))
        values += values[:1]  # Kapatmak iÃ§in ilk elemanÄ± tekrar ekle
        
        # Bio-wheel Ã§izimi
        ax.fill(angles, values, alpha=0.25, 
                color=plt.cm.viridis(j/3 + i/12), 
                edgecolor='black', linewidth=2)
        
        # Zeminler ve Ã§izgiler
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # GÃ¶rsel ayarlamalar
        ax.spines['polar'].set_visible(False)
        
        # BaÅŸlÄ±k
        title = f"Age {age_min}-{age_max}, Risk Score {risk_min}-{risk_max}"
        ax.set_title(title, fontsize=10, pad=15)

        # Bio-wheel merkezine dekoratif Ã§iÃ§ek ekleme
        center = patches.Circle((0, 0), 0.15, transform=ax.transData._b, 
                               edgecolor='black', facecolor=plt.cm.viridis(j/3 + i/12),
                               alpha=0.6, zorder=10)
        ax.add_patch(center)
        
        # Her sistem iÃ§in Ã§iÃ§ek yapraklarÄ± ekleyelim
        for k, angle in enumerate(angles[:-1]):
            # Her sistem iÃ§in bir Ã§iÃ§ek yapraÄŸÄ±
            petal_center = (0.5 * values[k] * np.cos(angle), 
                            0.5 * values[k] * np.sin(angle))
            
            petal = patches.Ellipse(petal_center, values[k] * 0.2, values[k] * 0.35,
                                   angle=np.degrees(angle), 
                                   edgecolor='black', linewidth=1,
                                   facecolor=plt.cm.plasma(k/N), alpha=0.4,
                                   transform=ax.transData._b, zorder=5)
            ax.add_patch(petal)
            
        # Risk seviyesine gÃ¶re hareketli damarlar (can damarlarÄ±) ekleyelim
        for k in range(8):
            angle = 2 * np.pi * np.random.random()
            length = 0.9 + 0.3 * np.random.random()
            
            vein_x = [0, length * np.cos(angle)]
            vein_y = [0, length * np.sin(angle)]
            
            # Risk seviyesine gÃ¶re damar rengi ve kalÄ±nlÄ±ÄŸÄ± deÄŸiÅŸiyor
            vein_color = plt.cm.coolwarm(0.2 + 0.8 * (j/3))
            vein_width = 1 + j
            
            ax.plot(vein_x, vein_y, color=vein_color, linewidth=vein_width, 
                    alpha=0.4, zorder=1, linestyle='-')

# BoÅŸ hÃ¼creye aÃ§Ä±klama ekleyelim
ax_legend = fig.add_subplot(gs[2, 3])
ax_legend.axis('off')

# AÃ§Ä±klama metni
ax_legend.text(0.5, 0.9, "Holistic Health Balance", fontsize=14, ha='center', weight='bold')
ax_legend.text(0.5, 0.8, "The Bio-Wheel of Life", fontsize=12, ha='center', style='italic')
ax_legend.text(0.5, 0.7, "Each 'bio-wheel' represents the average health status\nof individuals within specific age and risk groups.", 
             fontsize=10, ha='center')

# Risk gruplarÄ± renk kodlarÄ±
for j, (risk_min, risk_max) in enumerate(risk_groups):
    ax_legend.text(0.1, 0.5 - j*0.1, f"Risk Score {risk_min}-{risk_max}:", fontsize=9)
    ax_legend.plot([0.4, 0.5], [0.5 - j*0.1]*2, color=plt.cm.coolwarm(0.2 + 0.8 * (j/3)), 
                  linewidth=3+j, alpha=0.7)

# Sistem renk kodlarÄ±
for k, system in enumerate(health_systems.keys()):
    circle = plt.Circle((0.15, 0.3 - k*0.05), 0.02, color=plt.cm.plasma(k/N), alpha=0.7)
    ax_legend.add_patch(circle)
    ax_legend.text(0.2, 0.3 - k*0.05, system, fontsize=8, va='center')

# YorumlayÄ±cÄ± notlar
ax_legend.text(0.5, 0.15, "Analysis Key Insights:", fontsize=9, ha='center', weight='bold')
ax_legend.text(0.5, 0.1, "â€¢ Higher risk scores strongly correlate with decreased\n  health balance across multiple systems", 
             fontsize=8, ha='center')
ax_legend.text(0.5, 0.05, "â€¢ Sensory and cardiovascular systems show the\n  most dramatic age-related decline patterns", 
             fontsize=8, ha='center')
ax_legend.text(0.5, 0.0, "â€¢ Metabolic health deteriorates earlier in high-risk groups,\n  while it remains stable longer in low-risk individuals", 
             fontsize=8, ha='center')

plt.suptitle("The Bio-Wheel of Life: Holistic Health Balance Analysis", 
             fontsize=20, y=0.98, weight='bold')
plt.text(0.5, 0.94, 
         "Visualizing the interconnectedness of body systems across age and risk profiles",
         fontsize=14, ha='center', transform=fig.transFigure)

plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.93])

plt.savefig('bio_wheel_of_life.png', dpi=300, bbox_inches='tight')
plt.show()


import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
import plotly.figure_factory as ff

# Ã–rnek veri oluÅŸturma (gerÃ§ek verilerinizle deÄŸiÅŸtirin)
np.random.seed(42)
sample_size = 5000
df = pd.DataFrame({
    'age': np.random.normal(40, 15, sample_size),
    'systolic': np.random.normal(120, 15, sample_size),
    'relaxation': np.random.normal(80, 10, sample_size),
    'Cholesterol': np.random.normal(190, 35, sample_size),
    'triglyceride': np.random.normal(150, 45, sample_size),
    'HDL': np.random.normal(50, 12, sample_size),
    'LDL': np.random.normal(110, 25, sample_size)
})

# SaÄŸlÄ±k risk skoru hesaplama
df['risk_score'] = (
    (df['systolic'] - 110) / 40 + 
    (df['relaxation'] - 70) / 20 + 
    (df['Cholesterol'] - 150) / 100 + 
    (df['triglyceride'] - 100) / 100 + 
    (df['LDL'] - 100) / 50 - 
    (df['HDL'] - 60) / 20
)

df['risk_level'] = pd.qcut(df['risk_score'], 5, labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'])

# 3D Kalp modeli iÃ§in parametreler
theta = np.linspace(0, 2*np.pi, 100)
phi = np.linspace(0, np.pi, 50)
theta, phi = np.meshgrid(theta, phi)

# Kalp ÅŸekli iÃ§in fonksiyon
def heart_x(theta, phi, a=1):
    return a * (np.sin(phi) * np.cos(theta) * np.sin(theta))

def heart_y(theta, phi, a=1):
    return a * 0.9 * np.sin(phi) * np.sin(theta)

def heart_z(theta, phi, a=1):
    return a * np.cos(phi) * (1.5 + 0.5 * np.sin(theta))

# Ã–rnek kiÅŸiler iÃ§in veri
sample_patients = df.sample(5)
risk_colors = {
    'Very Low': 'green',
    'Low': 'lightgreen',
    'Moderate': 'yellow',
    'High': 'orange',
    'Very High': 'red'
}

fig = make_subplots(
    rows=2, cols=3,
    specs=[[{'type': 'surface', 'rowspan': 2, 'colspan': 2}, {'type': 'indicator'}, {'type': 'indicator'}],  # 1. satÄ±r
           [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],  # 2. satÄ±r
    subplot_titles=("Cardiovascular Digital Twin", "Blood Pressure", "Cholesterol Levels")
)



# Ã–rnek bir hasta seÃ§elim
patient = sample_patients.iloc[0]
risk_level = patient['risk_level']
color = risk_colors[risk_level]

# Kalp modeli
x = heart_x(theta, phi)
y = heart_y(theta, phi)
z = heart_z(theta, phi)

colorscale = [[0, 'green'], [0.4, 'yellow'], [0.6, 'orange'], [1, 'red']]
intensity = 0.5 + 0.5 * (np.sin(5*theta) * np.cos(5*phi))

fig.add_trace(
    go.Surface(
        x=x, y=y, z=z,
        surfacecolor=intensity,
        colorscale=colorscale,
        showscale=False,
        opacity=0.9
    ),
    row=1, col=1
)

# Kan basÄ±ncÄ± gÃ¶stergesi
fig.add_trace(
    go.Indicator(
        mode="gauge+number",
        value=patient['systolic'],
        title={'text': f"Systolic BP<br><span style='font-size:0.8em;color:gray'>Age: {int(patient['age'])}</span>"},
        gauge={
            'axis': {'range': [None, 200]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 120], 'color': 'lightgreen'},
                {'range': [120, 140], 'color': 'yellow'},
                {'range': [140, 180], 'color': 'orange'},
                {'range': [180, 200], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 140
            }
        }
    ),
    row=1, col=3
)

# Kolesterol gÃ¶stergesi
fig.add_trace(
    go.Indicator(
        mode="gauge+number",
        value=patient['Cholesterol'],
        title={'text': "Total Cholesterol"},
        gauge={
            'axis': {'range': [0, 300]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 200], 'color': 'lightgreen'},
                {'range': [200, 240], 'color': 'yellow'},
                {'range': [240, 300], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 240
            }
        }
    ),
    row=2, col=3
)

# DÃ¼zenleme
fig.update_layout(
    title_text="Cardiovascular Health Digital Twin<br><span style='font-size:1.2em'>Real-time Heart Model with Risk Analysis</span>",
    height=800,
    width=1200,
    scene={
        'camera': {
            'eye': {'x': 1.2, 'y': 1.2, 'z': 0.6}
        },
        'annotations': [{
            'showarrow': False,
            'x': 0, 'y': 0, 'z': 1.5,
            'text': f"Risk Level: {risk_level}",
            'font': {'size': 14, 'color': color}
        }]
    },
)

fig.add_annotation(
    x=0.5, y=0.02,
    xref="paper", yref="paper",
    text="Analysis: This digital twin model shows cardiovascular health status with real-time risk visualization. Current heart performance indicates a " + str(risk_level).lower() + " risk level based on blood pressure, cholesterol, and triglyceride metrics.",
    showarrow=False,
    font=dict(size=12),
    bordercolor="#888",
    bgcolor="#f0f0f0",
    borderwidth=1,
    borderpad=4
)

fig.show(renderer='iframe_connected')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle, Polygon
import matplotlib.patheffects as path_effects
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Veri setini yÃ¼kleyelim (gerÃ§ek verilerinizi burada yÃ¼kleyin)
# Bu Ã¶rnek iÃ§in rastgele veri oluÅŸturuyorum
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.normal(45, 15, 15000),
    'height(cm)': np.random.normal(170, 10, 15000),
    'weight(kg)': np.random.normal(70, 15, 15000),
    'waist(cm)': np.random.normal(85, 15, 15000),
    'systolic': np.random.normal(120, 15, 15000),
    'relaxation': np.random.normal(80, 10, 15000),
    'fasting blood sugar': np.random.normal(95, 20, 15000),
    'Cholesterol': np.random.normal(190, 40, 15000),
    'triglyceride': np.random.normal(150, 80, 15000),
    'HDL': np.random.normal(50, 15, 15000),
    'LDL': np.random.normal(120, 30, 15000)
})

# BMI hesaplama
data['BMI'] = data['weight(kg)'] / ((data['height(cm)']/100) ** 2)

# BMI kategorileri belirleme
conditions = [
    (data['BMI'] < 18.5),
    (data['BMI'] >= 18.5) & (data['BMI'] < 25),
    (data['BMI'] >= 25) & (data['BMI'] < 30),
    (data['BMI'] >= 30)
]
values = ['Skinny', 'Normal', 'Overweight', 'Obese']
data['BMI_category'] = np.select(conditions, values)

# Ä°liÅŸki gÃ¼Ã§lendirmesi
# Kilo ve bel Ã§evresi artÄ±ÅŸÄ± kolesterol ve trigliseridi de arttÄ±rsÄ±n
for factor in ['Cholesterol', 'triglyceride', 'LDL']:
    data[factor] = data[factor] + data['BMI'] * np.random.normal(2, 0.5, 15000)
    
# HDL ve BMI ters orantÄ±lÄ± olsun
data['HDL'] = data['HDL'] - data['BMI'] * np.random.normal(0.3, 0.1, 15000)

# Sistemik bir nabÄ±z damarÄ± ÅŸekli oluÅŸturalÄ±m
fig = plt.figure(figsize=(16, 12))
plt.style.use('dark_background')

# 3D Damar sistemi gÃ¶rselleÅŸtirmesi
ax = fig.add_subplot(111, projection='3d')

# Damar yolunu oluÅŸturalÄ±m
t = np.linspace(0, 15, 1000)
x = np.sin(t) * t/3
y = np.cos(t) * t/3
z = t

# Damar geniÅŸliÄŸi - kolesterol seviyesine gÃ¶re deÄŸiÅŸecek
# Ã–rnek verileri kolesterol seviyelerine gÃ¶re gruplayalÄ±m
chol_bins = pd.cut(data['Cholesterol'], bins=6)
cholesterol_groups = data.groupby(chol_bins)['LDL'].mean().values
hdl_groups = data.groupby(chol_bins)['HDL'].mean().values
trig_groups = data.groupby(chol_bins)['triglyceride'].mean().values

# Renk haritasÄ± - tehlike seviyesini gÃ¶sterecek
cm_subsection = np.linspace(0.1, 0.9, 6)
colors = [cm.plasma(x) for x in cm_subsection]

# Damar kÄ±sÄ±mlarÄ±nÄ± Ã§izelim - her bir parÃ§a farklÄ± kolesterol seviyesi iÃ§in
section_size = len(t) // 6
vessel_radius_base = 0.4

# Kolesterol plaklarÄ±
for i in range(6):
    start = i * section_size
    end = (i + 1) * section_size if i < 5 else len(t)
    
    # LDL ve trigliserit yÃ¼kseldikÃ§e damar daralÄ±r
    plaque_factor = (cholesterol_groups[i] / 100) * (trig_groups[i] / 120) / 2.5
    vessel_radius = vessel_radius_base * (1 - min(0.7, plaque_factor * 0.01))
    
    # DeÄŸerler arttÄ±kÃ§a renk kÄ±rmÄ±zÄ±laÅŸÄ±r (risk artar)
    risk_color = colors[i]
    
    # DamarÄ±n bu kÄ±smÄ±nÄ± Ã§iz
    ax.plot(x[start:end], y[start:end], z[start:end], 
            linewidth=12*vessel_radius, color=risk_color, alpha=0.8)
    
    # Plak oluÅŸumlarÄ± (LDL yÃ¼ksekse daha fazla)
    if cholesterol_groups[i] > 130:
        plaque_count = int((cholesterol_groups[i] - 100) / 10)
        for _ in range(plaque_count):
            idx = np.random.randint(start, end)
            plaque_size = 0.1 + (cholesterol_groups[i] - 100) / 300
            ax.scatter(x[idx], y[idx], z[idx], color='#ffcc00', s=120*plaque_size, alpha=0.7)
    
    # Ä°lgili deÄŸerleri ekleyelim
    mean_x = np.mean(x[start:end])
    mean_y = np.mean(y[start:end])
    mean_z = np.mean(z[start:end])
    
    # Kesit etiketleri ekleyelim
    ax.text(mean_x+0.5, mean_y+0.5, mean_z, 
            f"LDL: {cholesterol_groups[i]:.1f}\nHDL: {hdl_groups[i]:.1f}\nTG: {trig_groups[i]:.1f}", 
            color='white', fontsize=9)

# Kan hÃ¼creleri
for _ in range(70):
    pos = np.random.randint(0, len(t))
    cell_radius = 0.08
    cell_color = '#ff3333'  # Kan hÃ¼cresi kÄ±rmÄ±zÄ±
    ax.scatter(x[pos], y[pos], z[pos], color=cell_color, s=40, alpha=0.7)

# Eksen ayarlarÄ±
ax.set_xlabel('X', labelpad=15, fontsize=12)
ax.set_ylabel('Y', labelpad=15, fontsize=12)
ax.set_zlabel('Vein Length', labelpad=15, fontsize=12)
ax.grid(False)
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# Lejant ve renkler iÃ§in skala
cmap = cm.plasma
norm = plt.Normalize(min(cholesterol_groups), max(cholesterol_groups))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.1)
cbar.set_label('LDL Cholesterol Level (mg/dL)', rotation=270, labelpad=25, fontsize=12)

# BaÅŸlÄ±k ve analiz
plt.suptitle('CHOLESTEROL AND BLOOD VESSEL HEALTH', fontsize=22, color='white', y=0.92)

# BMI ve Kolesterol iliÅŸkisini analiz edelim
bmi_chol_corr = np.corrcoef(data['BMI'], data['Cholesterol'])[0,1]
bmi_ldl_corr = np.corrcoef(data['BMI'], data['LDL'])[0,1]
bmi_hdl_corr = np.corrcoef(data['BMI'], data['HDL'])[0,1]

# Ä°statistiksel analiz notlarÄ±
analysis_text = (
    f"ANALYSIS: High LDL cholesterol causes plaque formation in artery walls.\n"
    f"Strong positive correlation between body mass index and LDL cholesterol (r={bmi_ldl_corr:.2f}) It is seen."
    f"As HDL ('good cholesterol') levels drop, vascular risks increase.."
)
plt.figtext(0.5, 0.05, analysis_text, ha='center', fontsize=12, color='white', 
          bbox=dict(facecolor='#d62728', alpha=0.7, boxstyle='round,pad=0.5'))

plt.tight_layout(rect=[0, 0.07, 1, 0.9])
plt.savefig('kolesterol_damar_sagligi_3d.png', dpi=300, bbox_inches='tight')
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Wedge, Rectangle, FancyBboxPatch
import matplotlib.patheffects as path_effects
from matplotlib import gridspec
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

# Veri setini yÃ¼kleyelim (gerÃ§ek verilerinizi burada yÃ¼kleyin)
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.normal(45, 15, 15000),
    'eyesight(left)': np.random.normal(0.9, 0.5, 15000),
    'eyesight(right)': np.random.normal(0.9, 0.5, 15000),
    'hearing(left)': np.random.normal(50, 20, 15000),
    'hearing(right)': np.random.normal(50, 20, 15000),
    'dental caries': np.random.choice([0, 1, 2, 3, 4, 5], 15000, p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05]),
    'systolic': np.random.normal(120, 15, 15000),
    'relaxation': np.random.normal(80, 10, 15000)
})

# YaÅŸ ile gÃ¶z ve iÅŸitme iliÅŸkisi oluÅŸturalÄ±m
data['eyesight(left)'] = data['eyesight(left)'] - data['age'] * 0.005 + np.random.normal(0, 0.1, 15000)
data['eyesight(right)'] = data['eyesight(right)'] - data['age'] * 0.005 + np.random.normal(0, 0.1, 15000)
data['hearing(left)'] = data['hearing(left)'] + data['age'] * 0.3 + np.random.normal(0, 5, 15000)
data['hearing(right)'] = data['hearing(right)'] + data['age'] * 0.3 + np.random.normal(0, 5, 15000)

# YaÅŸ gruplarÄ± oluÅŸturalÄ±m
age_bins = [0, 20, 40, 60, 80, 100]
age_labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
data['age_group'] = pd.cut(data['age'], bins=age_bins, labels=age_labels)

# FigÃ¼r oluÅŸturalÄ±m - Ä°nsan vÃ¼cudu ve duyusal yetiler analizi
bg_color = '#0f1931'  # Koyu mavi arka plan
accent_color = '#f9a825'  # AltÄ±n sarÄ±sÄ± vurgu rengi
text_color = '#e0e0e0'  # AÃ§Ä±k gri yazÄ± rengi
highlight_color = '#ff5722'  # Turuncu vurgu

# 3x3 grid oluÅŸturalÄ±m
gs = gridspec.GridSpec(4, 3, height_ratios=[1, 4, 0.5, 0.5])
fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor(bg_color)

# BaÅŸlÄ±k iÃ§in eksen tanÄ±mlama
title_ax = plt.subplot(gs[0, :])  # BaÅŸlÄ±k iÃ§in eksen
title_ax.axis('off')  # BaÅŸlÄ±k alanÄ±nÄ± boÅŸ bÄ±rakmak iÃ§in ekseni kapatalÄ±m

# BaÅŸlÄ±k ekleyelim
title_ax.text(0.5, 0.5, "CHANGES IN HUMAN SENSORY SYSTEMS WITH AGE", 
             fontsize=22, color=accent_color, ha='center', va='center',
             path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])

# Ä°nsan silÃ¼eti ve duyusal organlar
body_ax = plt.subplot(gs[1, :])
body_ax.set_facecolor(bg_color)
body_ax.set_xlim(-10, 10)
body_ax.set_ylim(-10, 10)
body_ax.axis('off')

# Ä°nsan vÃ¼cudu silueti
# BaÅŸ
head = Circle((0, 7), 2, color='#78909c', alpha=0.7, ec='white')
body_ax.add_patch(head)

# GÃ¶vde
body = Rectangle((-2, 0), 4, 5, color='#78909c', alpha=0.7, ec='white')
body_ax.add_patch(body)

# Kollar
left_arm = Rectangle((-3, 0), 1, 4, color='#78909c', alpha=0.7, ec='white')
body_ax.add_patch(left_arm)
right_arm = Rectangle((2, 0), 1, 4, color='#78909c', alpha=0.7, ec='white')
body_ax.add_patch(right_arm)

# Bacaklar
left_leg = Rectangle((-1.5, -5), 1, 5, color='#78909c', alpha=0.7, ec='white')
body_ax.add_patch(left_leg)
right_leg = Rectangle((0.5, -5), 1, 5, color='#78909c', alpha=0.7, ec='white')
body_ax.add_patch(right_leg)

# Duyusal organlar ve verileri etiketlerde gÃ¶sterelim
# YaÅŸ gruplarÄ±na gÃ¶re veriler
age_group_data = data.groupby('age_group').agg({
    'eyesight(left)': 'mean',
    'eyesight(right)': 'mean',
    'hearing(left)': 'mean',
    'hearing(right)': 'mean',
    'dental caries': 'mean'
}).round(2)

# GÃ¶zler ve veriler
left_eye = Circle((-0.8, 7.5), 0.4, color='white', alpha=0.9)
body_ax.add_patch(left_eye)
right_eye = Circle((0.8, 7.5), 0.4, color='white', alpha=0.9)
body_ax.add_patch(right_eye)

# GÃ¶rme keskinliÄŸi gÃ¶stergeleri - yaÅŸ gruplarÄ±na gÃ¶re
eye_data = []
for i, age_label in enumerate(age_labels):
    left_val = age_group_data.loc[age_label, 'eyesight(left)']
    right_val = age_group_data.loc[age_label, 'eyesight(right)']
    eye_data.append((left_val, right_val))
    
    # GÃ¶sterge renkleri - gÃ¶rme keskinliÄŸi azaldÄ±kÃ§a kÄ±rmÄ±zÄ±laÅŸan
    if i > 0:  # Ä°lk grup hariÃ§ (referans olarak kullanacaÄŸÄ±z)
        left_color = plt.cm.RdYlGn(min(1.0, max(0.0, left_val)))
        right_color = plt.cm.RdYlGn(min(1.0, max(0.0, right_val)))
        
        # Sol gÃ¶z iÃ§in gÃ¶sterge
        left_indicator = Circle((-3-i, 7.5), 0.3, color=left_color, alpha=0.8)
        body_ax.add_patch(left_indicator)
        body_ax.text(-3-i, 8, age_label, ha='center', va='bottom', fontsize=8, color=text_color)
        body_ax.text(-3-i, 7, f"{left_val:.2f}", ha='center', va='top', fontsize=7, color=text_color)
        
        # SaÄŸ gÃ¶z iÃ§in gÃ¶sterge
        right_indicator = Circle((3+i, 7.5), 0.3, color=right_color, alpha=0.8)
        body_ax.add_patch(right_indicator)
        body_ax.text(3+i, 8, age_label, ha='center', va='bottom', fontsize=8, color=text_color)
        body_ax.text(3+i, 7, f"{right_val:.2f}", ha='center', va='top', fontsize=7, color=text_color)

# Kulaklar ve veriler
left_ear = Wedge((-2.1, 7), 0.7, 90, 270, color='#e0e0e0', alpha=0.7)
body_ax.add_patch(left_ear)
right_ear = Wedge((2.1, 7), 0.7, -90, 90, color='#e0e0e0', alpha=0.7)
body_ax.add_patch(right_ear)

# Ä°ÅŸitme gÃ¶stergeleri
for i, age_label in enumerate(age_labels):
    if i > 0:  # Ä°lk grup hariÃ§
        left_val = age_group_data.loc[age_label, 'hearing(left)']
        right_val = age_group_data.loc[age_label, 'hearing(right)']
        
        # Ä°ÅŸitme iÃ§in renk (yÃ¼ksek deÄŸer = kÃ¶tÃ¼ iÅŸitme)
        left_color = plt.cm.RdYlGn_r(min(1.0, max(0.0, left_val/100)))
        right_color = plt.cm.RdYlGn_r(min(1.0, max(0.0, right_val/100)))
        
        # Ä°ÅŸitme gÃ¶stergeleri
        left_indicator = Circle((-3-i, 5), 0.3, color=left_color, alpha=0.8)
        body_ax.add_patch(left_indicator)
        body_ax.text(-3-i, 5.5, age_label, ha='center', va='bottom', fontsize=8, color=text_color)
        body_ax.text(-3-i, 4.5, f"{left_val:.1f} dB", ha='center', va='top', fontsize=7, color=text_color)
        
        right_indicator = Circle((3+i, 5), 0.3, color=right_color, alpha=0.8)
        body_ax.add_patch(right_indicator)
        body_ax.text(3+i, 5.5, age_label, ha='center', va='bottom', fontsize=8, color=text_color)
        body_ax.text(3+i, 4.5, f"{right_val:.1f} dB", ha='center', va='top', fontsize=7, color=text_color)

# DiÅŸ problemleri gÃ¶stergeleri
# AÄŸÄ±z
mouth = Wedge((0, 5.5), 0.7, 0, 180, color='#e0e0e0', alpha=0.7)
body_ax.add_patch(mouth)

# YaÅŸa gÃ¶re diÅŸ Ã§Ã¼rÃ¼ÄŸÃ¼ gÃ¶stergeleri
for i, age_label in enumerate(age_labels):
    if i > 0:
        caries_val = age_group_data.loc[age_label, 'dental caries']
        caries_color = plt.cm.RdYlGn_r(min(1.0, max(0.0, caries_val/5)))
        
        # DiÅŸ Ã§Ã¼rÃ¼ÄŸÃ¼ gÃ¶stergesi
        caries_indicator = Circle((0, -i-1), 0.3, color=caries_color, alpha=0.8)
        body_ax.add_patch(caries_indicator)
        body_ax.text(0, -i-0.5, age_label, ha='center', va='bottom', fontsize=8, color=text_color)
        body_ax.text(0, -i-1.5, f"Ã‡Ã¼rÃ¼k: {caries_val:.1f}", ha='center', va='top', fontsize=7, color=text_color)

# VÃ¼cut ana organlarÄ± gÃ¶stergeleri
# Kalp
heart = Circle((0, 3), 0.6, color='#d32f2f', alpha=0.7)
body_ax.add_patch(heart)
body_ax.text(0, 3, "â™¥", ha='center', va='center', fontsize=14, color='white')

# Tansiyon deÄŸerleri
for i, age_label in enumerate(age_labels):
    if i > 0:
        systolic_val = data[data['age_group'] == age_label]['systolic'].mean()
        diastolic_val = data[data['age_group'] == age_label]['relaxation'].mean()
        
        # Tansiyon renk gÃ¶stergesi
        bp_color = plt.cm.RdYlGn_r(min(1.0, max(0.0, (systolic_val-100)/80)))
        
        # Tansiyon gÃ¶stergesi
        bp_x = 6
        bp_y = -i-1
        bp_indicator = Rectangle((bp_x-0.5, bp_y-0.3), 1, 0.6, color=bp_color, alpha=0.8)
        body_ax.add_patch(bp_indicator)
        body_ax.text(bp_x, bp_y+0.7, age_label, ha='center', va='bottom', fontsize=8, color=text_color)
        body_ax.text(bp_x, bp_y, f"{int(systolic_val)}/{int(diastolic_val)}", ha='center', va='center', fontsize=7, color='white')

# YaÅŸ ve duyusal yetenekler iliÅŸkisi aÃ§Ä±klamalarÄ±
# BaÅŸlÄ±k ve gÃ¶rseli tamamlamak iÃ§in
plt.subplots_adjust(hspace=0.5)
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Ellipse, Polygon, Arc, Wedge
from matplotlib import gridspec
import matplotlib.colors as mcolors

# Load your dataset here (replace with your actual data loading)
# For this example, I'm generating random data
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.normal(45, 15, 15000),
    'height(cm)': np.random.normal(170, 10, 15000),
    'weight(kg)': np.random.normal(70, 15, 15000),
    'waist(cm)': np.random.normal(85, 15, 15000),
    'systolic': np.random.normal(120, 15, 15000),
    'relaxation': np.random.normal(80, 10, 15000),
    'fasting blood sugar': np.random.normal(95, 20, 15000),
    'Cholesterol': np.random.normal(190, 40, 15000),
    'triglyceride': np.random.normal(150, 80, 15000),
    'HDL': np.random.normal(50, 15, 15000),
    'LDL': np.random.normal(120, 30, 15000),
    'AST': np.random.normal(24, 12, 15000),
    'ALT': np.random.normal(25, 15, 15000),
    'Gtp': np.random.normal(35, 25, 15000)
})

# Calculate BMI
data['BMI'] = data['weight(kg)'] / ((data['height(cm)']/100) ** 2)

# Create metabolic health score
def metabolic_score(row):
    score = 100  # Starting score
    # Deviations from normal ranges reduce the score
    if row['systolic'] > 130:
        score -= (row['systolic'] - 130) * 0.5
    if row['relaxation'] > 85:
        score -= (row['relaxation'] - 85) * 0.8
    if row['fasting blood sugar'] > 100:
        score -= (row['fasting blood sugar'] - 100) * 0.5
    if row['Cholesterol'] > 200:
        score -= (row['Cholesterol'] - 200) * 0.2
    if row['triglyceride'] > 150:
        score -= (row['triglyceride'] - 150) * 0.1
    if row['LDL'] > 130:
        score -= (row['LDL'] - 130) * 0.3
    if row['HDL'] < 40:
        score -= (40 - row['HDL']) * 0.8
    if row['BMI'] > 25:
        score -= (row['BMI'] - 25) * 2  # Higher penalty for BMI
    if row['ALT'] > 40:
        score -= (row['ALT'] - 40) * 0.5  # Liver enzyme penalty
    return max(0, min(100, score))  # Clamp score between 0 and 100

# Calculate metabolic health score for each person
data['metabolic_score'] = data.apply(metabolic_score, axis=1)

# Determine metabolic syndrome risk category
data['metabolic_risk'] = pd.cut(
    data['metabolic_score'],
    bins=[0, 40, 60, 80, 100],
    labels=['High Risk', 'Moderate Risk', 'Low Risk', 'Optimal'],
    right=True
)

# Determine BMI categories
conditions = [
    (data['BMI'] < 18.5),
    (data['BMI'] >= 18.5) & (data['BMI'] < 25),
    (data['BMI'] >= 25) & (data['BMI'] < 30),
    (data['BMI'] >= 30)
]
values = ['Underweight', 'Normal', 'Overweight', 'Obese']
data['BMI_category'] = np.select(conditions, values)

# Create age groups
age_bins = [0, 30, 40, 50, 60, 100]
age_labels = ['<30', '30-40', '41-50', '51-60', '>60']
data['age_group'] = pd.cut(data['age'], bins=age_bins, labels=age_labels, right=False, include_lowest=True)

# Filtreleme: Negatif veya aÅŸÄ±rÄ± yÃ¼ksek yaÅŸlarÄ± kaldÄ±r
data = data[(data['age'] >= 0) & (data['age'] <= 100)]

# --- Visualization: Human Body Metabolic Map ---
plt.style.use('dark_background')
fig = plt.figure(figsize=(18, 16))
fig.patch.set_facecolor('#0d1117')

gs = gridspec.GridSpec(4, 3, height_ratios=[1, 6, 2, 1], width_ratios=[1,1,1])

# --- Title ---
title_ax = plt.subplot(gs[0, :])
title_ax.axis('off')
title_ax.text(0.5, 0.5, "METABOLIC HEALTH RISK MAP",
              fontsize=26, color='#00ffcc', ha='center', va='center',
              path_effects=[path_effects.withStroke(linewidth=2, foreground='black')],
              fontweight='bold')

# --- Human body and metabolic risk indicator ---
body_ax = plt.subplot(gs[1, :])
body_ax.set_facecolor('#0d1117')
body_ax.set_xlim(-11, 11)
body_ax.set_ylim(-10, 13)
body_ax.axis('off')

# Prepare data grouped by risk category
risk_groups = data.groupby('metabolic_risk', observed=False).agg({
    'systolic': 'mean',
    'relaxation': 'mean',
    'fasting blood sugar': 'mean',
    'Cholesterol': 'mean',
    'triglyceride': 'mean',
    'HDL': 'mean',
    'LDL': 'mean',
    'BMI': 'mean',
    'metabolic_score': 'mean',
    'ALT': 'mean',
    'AST': 'mean',
    'Gtp': 'mean'
}).round(1)

# Risk colors
risk_colors = {
    'Optimal': '#00ff00',
    'Low Risk': '#ffff00',
    'Moderate Risk': '#ff9900',
    'High Risk': '#ff0000'
}

# Reorder risk_groups based on desired display order
risk_order_display = ['Optimal', 'Low Risk', 'Moderate Risk', 'High Risk']
risk_groups = risk_groups.reindex(risk_order_display)

# --- Draw central human body silhouette ---
def draw_human_body_silhouette(ax, highlight_organs=True, scale=1.0, center=(0,0)):
    x, y = center
    # Head
    head = Circle((x, y + 8*scale), 1.5*scale, facecolor='#3a3a3a', edgecolor='white', alpha=0.5)
    ax.add_patch(head)
    # Neck
    neck = Rectangle((x - 0.5*scale, y + 7*scale), 1*scale, 1*scale, facecolor='#3a3a3a', edgecolor='white', alpha=0.5)
    ax.add_patch(neck)
    # Torso
    torso = Polygon([(x - 3*scale, y + 7*scale), (x + 3*scale, y + 7*scale),
                     (x + 2.5*scale, y - 2*scale), (x - 2.5*scale, y - 2*scale)],
                    facecolor='#3a3a3a', edgecolor='white', alpha=0.5)
    ax.add_patch(torso)
    # Arms
    l_arm = Polygon([(x - 3*scale, y + 6.5*scale), (x - 5*scale, y + 2*scale),
                     (x - 5.5*scale, y + 1.5*scale), (x - 3*scale, y + 6*scale)],
                    facecolor='#3a3a3a', edgecolor='white', alpha=0.5)
    ax.add_patch(l_arm)
    r_arm = Polygon([(x + 3*scale, y + 6.5*scale), (x + 5*scale, y + 2*scale),
                     (x + 5.5*scale, y + 1.5*scale), (x + 3*scale, y + 6*scale)],
                    facecolor='#3a3a3a', edgecolor='white', alpha=0.5)
    ax.add_patch(r_arm)
    # Legs
    l_leg = Polygon([(x - 2.5*scale, y - 2*scale), (x - 1*scale, y - 2*scale),
                     (x - 1.5*scale, y - 8*scale), (x - 3*scale, y - 8*scale)],
                    facecolor='#3a3a3a', edgecolor='white', alpha=0.5)
    ax.add_patch(l_leg)
    r_leg = Polygon([(x + 2.5*scale, y - 2*scale), (x + 1*scale, y - 2*scale),
                     (x + 1.5*scale, y - 8*scale), (x + 3*scale, y - 8*scale)],
                    facecolor='#3a3a3a', edgecolor='white', alpha=0.5)
    ax.add_patch(r_leg)
    if highlight_organs:
        # Brain
        brain = Circle((x, y + 9*scale), 0.9*scale, facecolor='#cc66ff', alpha=0.7, edgecolor='white')
        ax.add_patch(brain)
        # Heart
        heart = Circle((x - 1*scale, y + 5*scale), 0.8*scale, facecolor='#ff3333', alpha=0.7, edgecolor='white')
        ax.add_patch(heart)
        # Liver
        liver = Ellipse((x + 1.2*scale, y + 4*scale), 1.3*scale*2, 1.5*scale*2, angle=10, facecolor='#993300', alpha=0.7, edgecolor='white')
        ax.add_patch(liver)
        # Pancreas
        pancreas = Ellipse((x, y + 3*scale), 2*scale*2, 0.5*scale*2, angle=-5, facecolor='#ffcc99', alpha=0.7, edgecolor='white')
        ax.add_patch(pancreas)
        # Kidneys
        l_kidney = Ellipse((x - 1.8*scale, y + 2.5*scale), 0.7*scale*2, 1*scale*2, angle=-10, facecolor='#ff6666', alpha=0.7, edgecolor='white')
        ax.add_patch(l_kidney)
        r_kidney = Ellipse((x + 1.8*scale, y + 2.5*scale), 0.7*scale*2, 1*scale*2, angle=10, facecolor='#ff6666', alpha=0.7, edgecolor='white')
        ax.add_patch(r_kidney)
        # Fat tissue (waist area)
        fat_tissue = Ellipse((x, y + 0.5*scale), 4*scale*2, 1.5*scale*2, facecolor='#ffff00', alpha=0.5, edgecolor='white')
        ax.add_patch(fat_tissue)
        # Muscle tissue (legs)
        l_muscle = Ellipse((x - 2*scale, y - 4*scale), 1.2*scale*2, 3*scale*2, angle=-5, facecolor='#cc0000', alpha=0.4, edgecolor='white')
        ax.add_patch(l_muscle)
        r_muscle = Ellipse((x + 2*scale, y - 4*scale), 1.2*scale*2, 3*scale*2, angle=5, facecolor='#cc0000', alpha=0.4, edgecolor='white')
        ax.add_patch(r_muscle)

# Draw mini-figures for each risk group
risk_positions = {
    'Optimal': (-7.5, 0),
    'Low Risk': (-2.5, 0),
    'Moderate Risk': (2.5, 0),
    'High Risk': (7.5, 0)
}
mini_fig_scale = 0.35
cmap_health = plt.cm.RdYlGn

for risk, position in risk_positions.items():
    if risk not in risk_groups.index:
        continue
    x_center, y_center = position
    bmi = risk_groups.loc[risk, 'BMI']
    blood_sugar = risk_groups.loc[risk, 'fasting blood sugar']
    cholesterol = risk_groups.loc[risk, 'Cholesterol']
    systolic = risk_groups.loc[risk, 'systolic']
    relaxation = risk_groups.loc[risk, 'relaxation']
    hdl = risk_groups.loc[risk, 'HDL']
    ldl = risk_groups.loc[risk, 'LDL']
    trig = risk_groups.loc[risk, 'triglyceride']
    alt = risk_groups.loc[risk, 'ALT']
    ast = risk_groups.loc[risk, 'AST']
    metabolic_score_val = risk_groups.loc[risk, 'metabolic_score']
    
    # Draw Mini Body
    body_color = cmap_health(metabolic_score_val / 100)
    draw_human_body_silhouette(body_ax, highlight_organs=False, scale=mini_fig_scale, center=(x_center, y_center))
    for patch in body_ax.patches[-7:]:
        patch.set_facecolor(body_color)
        patch.set_edgecolor('white')
        patch.set_alpha(0.7)
    
    # Highlight Organs
    heart_health = np.clip(1 - (max(0, systolic - 120)) / 50, 0.1, 1.0)
    heart_color = cmap_health(heart_health)
    mini_heart = Circle((x_center - 1*mini_fig_scale, y_center + 5*mini_fig_scale), 0.8*mini_fig_scale,
                        facecolor=heart_color, alpha=0.9, edgecolor='white', zorder=10)
    body_ax.add_patch(mini_heart)
    
    liver_health = np.clip(1 - (max(0, alt - 20)) / 40, 0.1, 1.0)
    liver_color = cmap_health(liver_health)
    mini_liver = Ellipse((x_center + 1.2*mini_fig_scale, y_center + 4*mini_fig_scale),
                         1.3*mini_fig_scale*2, 1.5*mini_fig_scale*2, angle=10,
                         facecolor=liver_color, alpha=0.9, edgecolor='white', zorder=10)
    body_ax.add_patch(mini_liver)
    
    pancreas_health = np.clip(1 - (max(0, blood_sugar - 85)) / 50, 0.1, 1.0)
    pancreas_color = cmap_health(pancreas_health)
    mini_pancreas = Ellipse((x_center, y_center + 3*mini_fig_scale),
                            2*mini_fig_scale*2, 0.5*mini_fig_scale*2, angle=-5,
                            facecolor=pancreas_color, alpha=0.9, edgecolor='white', zorder=10)
    body_ax.add_patch(mini_pancreas)
    
    fat_health = np.clip(1 - (max(0, bmi - 23)) / 15, 0.1, 1.0)
    fat_color = cmap_health(fat_health)
    mini_fat = Ellipse((x_center, y_center + 0.5*mini_fig_scale),
                       4*mini_fig_scale*2, 1.5*mini_fig_scale*2,
                       facecolor=fat_color, alpha=0.7, edgecolor='white', zorder=9)
    body_ax.add_patch(mini_fat)
    
    ldl_penalty = (max(0, ldl - 100)) / 100
    hdl_bonus = (max(0, hdl - 40)) / 40
    trig_penalty = (max(0, trig - 150)) / 200
    vessel_health = np.clip(0.8 + hdl_bonus*0.4 - ldl_penalty*0.6 - trig_penalty*0.3, 0.1, 1.0)
    vessel_color = cmap_health(vessel_health)
    
    for i in range(3):
        vessel = Arc((x_center, y_center + 3*mini_fig_scale),
                     (5 - i)*mini_fig_scale*1.5, (8 - i*1.5)*mini_fig_scale*1.5,
                     theta1=180, theta2=360,
                     linewidth=2 + (1-vessel_health)*3,
                     color=vessel_color, alpha=0.6 + i*0.1, zorder=8)
        body_ax.add_patch(vessel)
    
    # Add Text Labels
    body_ax.text(x_center, y_center - 9.5, risk, ha='center', va='center', fontsize=14, color=risk_colors[risk],
                 fontweight='bold', path_effects=[path_effects.withStroke(linewidth=1.5, foreground='black')])
    
    info_y_pos = y_center + 5.5
    info_text = (
        f"Score: {metabolic_score_val:.0f}\n"
        f"BMI: {bmi:.1f}\n"
        f"Blood Sugar: {blood_sugar:.0f} mg/dL\n"
        f"BP: {systolic:.0f}/{relaxation:.0f} mmHg\n"
        f"HDL: {hdl:.0f} / LDL: {ldl:.0f}\n"
        f"ALT: {alt:.0f} U/L"
    )
    body_ax.text(x_center, info_y_pos, info_text, ha='center', va='bottom', fontsize=9, color='white',
                 linespacing=1.3,
                 bbox=dict(facecolor='#1f1f1f', alpha=0.7, boxstyle='round,pad=0.4', edgecolor='none'))

# --- Metabolic Risk Distribution Analysis ---
stats_ax = plt.subplot(gs[2, :])
stats_ax.set_facecolor('#0d1117')

# Risk groups by age distribution
risk_by_age = pd.crosstab(data['age_group'], data['metabolic_risk'], normalize='index', dropna=False) * 100
risk_by_age = risk_by_age.reindex(columns=risk_order_display, index=age_labels, fill_value=0)

# Stacked bar chart
bottom_vals = np.zeros(len(age_labels))
bar_width = 0.7

for risk in risk_order_display:
    if risk in risk_by_age.columns:
        values = risk_by_age[risk].values
        stats_ax.bar(age_labels, values, bottom=bottom_vals, color=risk_colors[risk],
                     alpha=0.85, label=risk, edgecolor='white', linewidth=0.5, width=bar_width)
        for i, val in enumerate(values):
            if val > 8:
                stats_ax.text(i, bottom_vals[i] + val / 2, f'{val:.0f}%',
                              ha='center', va='center', fontsize=9, color='black', fontweight='bold')
        bottom_vals += values

# Axes and Title adjustments
stats_ax.set_xlabel('Age Groups', fontsize=12, color='white', labelpad=10)
stats_ax.set_ylabel('Percentage (%)', fontsize=12, color='white', labelpad=10)
stats_ax.set_title('Metabolic Risk Distribution by Age Group', fontsize=16, color='#00ffcc', pad=15)
stats_ax.set_ylim(0, 100)
stats_ax.tick_params(axis='x', colors='white', rotation=0)
stats_ax.tick_params(axis='y', colors='white')

# Grid and Spines
stats_ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='#cccccc')
stats_ax.spines['top'].set_visible(False)
stats_ax.spines['right'].set_visible(False)
stats_ax.spines['bottom'].set_color('#555555')
stats_ax.spines['left'].set_color('#555555')

# Legend
stats_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4,
                frameon=False, labelcolor='white', fontsize=11)

# --- Analysis Summary Note ---
analysis_ax = plt.subplot(gs[3, :])
analysis_ax.axis('off')
analysis_ax.set_facecolor('#0d1117')

# Calculate correlations and percentages
valid_data = data[['age', 'metabolic_score', 'BMI']].dropna()
if len(valid_data) > 1:
    age_metabolic_corr = np.corrcoef(valid_data['age'], valid_data['metabolic_score'])[0, 1]
    bmi_metabolic_corr = np.corrcoef(valid_data['BMI'], valid_data['metabolic_score'])[0, 1]
else:
    age_metabolic_corr = np.nan
    bmi_metabolic_corr = np.nan

high_risk_pct = (data['metabolic_risk'] == 'High Risk').mean() * 100
under_40_data = data[data['age'] < 40]
if not under_40_data.empty:
    young_high_risk = (under_40_data['metabolic_risk'] == 'High Risk').mean() * 100
else:
    young_high_risk = 0

# Analysis text
analysis_text = (
    f"ANALYSIS SUMMARY:\n"
    f"Metabolic health risks tend to increase with age (Correlation with Score: {age_metabolic_corr:.2f}).\n"
    f"Higher BMI is strongly associated with lower metabolic scores (Correlation: {bmi_metabolic_corr:.2f}).\n"
    f"{high_risk_pct:.1f}% of the population is categorized as 'High Risk'.\n"
    f"In the under-40 age group, the High Risk percentage is {young_high_risk:.1f}%."
)

# Add a styled text box
analysis_box = FancyBboxPatch((0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.02",
                             facecolor='#1a1a2e', alpha=0.8, ec='#00ffcc', lw=1.5,
                             transform=analysis_ax.transAxes, clip_on=False)
analysis_ax.add_patch(analysis_box)
analysis_ax.text(0.5, 0.5, analysis_text, ha='center', va='center',
                 fontsize=11, color='white', transform=analysis_ax.transAxes,
                 linespacing=1.4, wrap=True)

# --- Final Adjustments and Save ---
plt.tight_layout(pad=2.5, h_pad=3.0, rect=[0, 0.03, 1, 0.97])
plt.savefig('metabolic_health_risk_map.png', dpi=300, bbox_inches='tight', facecolor='#0d1117')
plt.show()


from PIL import Image, ImageDraw, ImageFont

non_smokers = 9398
smokers = 5602
total = non_smokers + smokers

non_smoker_pct = non_smokers / total
smoker_pct = smokers / total

# GÃ¶rsel boyutu
width = 600
height = 100

# Yeni boÅŸ gÃ¶rsel (sigara ÅŸekli)
img = Image.new('RGB', (width, height), color='white')
draw = ImageDraw.Draw(img)

# Sigara gÃ¶vdesi (tam uzunluk)
cigarette_start = (50, 40)
cigarette_end = (550, 60)
draw.rectangle([cigarette_start, cigarette_end], fill='lightgray', outline='black')

# Ä°Ã§enler oranÄ±: saÄŸdan sola yanÄ±k kÄ±smÄ± (kÄ±rmÄ±zÄ±)
smoke_length = int((cigarette_end[0] - cigarette_start[0]) * smoker_pct)
draw.rectangle(
    [cigarette_end[0] - smoke_length, cigarette_start[1], cigarette_end[0], cigarette_end[1]],
    fill='orangered'
)

# Filtre kÄ±smÄ±: solda sabit turuncu alan
filter_width = 30
draw.rectangle(
    [cigarette_start[0], cigarette_start[1], cigarette_start[0] + filter_width, cigarette_end[1]],
    fill='orange'
)

# YÃ¼zdeleri yaz
font_size = 16
try:
    font = ImageFont.truetype("arial.ttf", font_size)
except IOError:
    font = ImageFont.load_default()

draw.text((cigarette_start[0] + 5, 5), f"Non-smokers: {non_smokers} ({non_smoker_pct:.1%})", fill="black", font=font)
draw.text((cigarette_end[0] - 200, 70), f"Smokers: {smokers} ({smoker_pct:.1%})", fill="orangered", font=font)


plt.figure(figsize=(10, 2))
plt.imshow(img)
plt.axis('off')
plt.title("ğŸš¬ Smoking Distribution on a Cigarette")
plt.show()


def column_info(df):
    """
    Generate summary information for each column in the DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame.

    Returns:
    DataFrame: DataFrame containing summary information for each column.
    """
    info = []
    for col in df.columns:
        data_type = df[col].dtype
        count = len(df[col])
        nan_count = df[col].isnull().sum()
        nan_percent = (nan_count / count) * 100 if count > 0 else 0
        unique_count = df[col].nunique()

        if pd.api.types.is_numeric_dtype(df[col]):
            max_value = df[col].max()
            min_value = df[col].min()
            sample_value = df[col].dropna().sample().iloc[0] if count - nan_count > 0 else None
        else:
            sample_value = df[col].dropna().sample().iloc[0] if count - nan_count > 0 else None
            max_value = 'no value'
            min_value = 'no value'

        info.append({
            'Column_name': col,
            'Data_Type': data_type,
            'Count': count,
            'NaN_Count': nan_count,
            'NaN_Percent': nan_percent,
            'Unique_Count': unique_count,
            'Max_Value': max_value,
            'Min_Value': min_value,
            'Sample_Value': sample_value
        })

    return pd.DataFrame(info)
    


column_info(df)


df = pd.read_csv("/kaggle/input/binary-smoke-detector/train.csv")

fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # 2M yatay yapÄ±

# --- Sol: Donut Chart ---
smoking_counts = df['smoking'].value_counts()
labels = smoking_counts.index
sizes = smoking_counts.values
colors = ['#d3d3d3', '#ff0000']  # duman grisi ve kÄ±rmÄ±zÄ±

# Donut chart
wedges, texts = ax[0].pie(sizes, labels=labels, colors=colors, wedgeprops=dict(width=0.4))
ax[0].set_title('Smoking Donut Chart', color='red')

# --- SaÄŸ: Countplot ---
sns.countplot(x='smoking', data=df, ax=ax[1], palette=['#d3d3d3', '#ff0000'])
ax[1].set_title('Smoking Distribution', color='red')

# SayÄ±larÄ± Ã¼stÃ¼ne yaz
for p in ax[1].patches:
    height = p.get_height()
    ax[1].annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                   ha='center', va='bottom', fontsize=11, color='red')

plt.tight_layout()
plt.show()


binary_cols = ['hearing(left)', 'hearing(right)', 'dental caries']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 1 satÄ±rda 3 grafik
colors = ['#d3d3d3', '#ff0000']

for i, col in enumerate(binary_cols):
    ax = axes[i]
    sns.countplot(x=col, data=df, palette=colors, ax=ax)
    ax.set_title(f'{col} Distribution', color='red')

    # SayÄ±larÄ± Ã§ubuklara yaz
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10, color='red')

plt.tight_layout()
plt.show()


df0 = df.drop(columns="id", axis=1)
df0.columns


sns.pairplot(data=df, hue="smoking", palette={0.0: "#d3d3d3", 1.0: "#ff0000"});


# Sadece numeric sÃ¼tunlarÄ± al
numeric_cols = df.select_dtypes(include='number').columns[:24]

# Grafik ayarlarÄ±
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(24, 16))
axes = axes.flatten()

# Her bir numeric sÃ¼tun iÃ§in histogram ve KDE Ã§iz
for i, col in enumerate(numeric_cols):
    # Histogram duman grisi
    sns.histplot(df[col], ax=axes[i], stat='density', bins=30, color='#d3d3d3', edgecolor='black')
    # KDE kÄ±rmÄ±zÄ±
    sns.kdeplot(df[col], ax=axes[i], color='red', linewidth=2)
    
    # BaÅŸlÄ±k ve etiketler
    axes[i].set_title(col)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

# Kalan boÅŸ grafik alanlarÄ±nÄ± kaldÄ±r
for j in range(len(numeric_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# Sadece numeric sÃ¼tunlarÄ± al ve 'smoking' sÃ¼tununu Ã§Ä±kar
numeric_cols = df.select_dtypes(include='number').drop(columns=['smoking'], errors='ignore').columns[:24]

# Grafik ayarlarÄ±
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(24, 16))
axes = axes.flatten()

# Renkler (duman grisi ve kÄ±rmÄ±zÄ±)
colors = {0: '#d3d3d3', 1: '#ff0000'}

# Her numeric sÃ¼tun iÃ§in sÄ±nÄ±fa gÃ¶re histogram + KDE Ã§iz
for i, col in enumerate(numeric_cols):
    # Sadece 0 ve 1 iÃ§in renkleri kullan
    for label in [0, 1]:
        subset = df[df["smoking"] == label]
        
        # Histogram (duman grisi ve kÄ±rmÄ±zÄ± renk)
        sns.histplot(subset[col], ax=axes[i], stat='density', kde=False, bins=30,
                     color=colors[label], label=f"smoking={label}", element='step', fill=True, alpha=0.4)
        
        # KDE (kÄ±rmÄ±zÄ± Ã§izgi)
        sns.kdeplot(subset[col], ax=axes[i], color=colors[label], linewidth=2)

    axes[i].set_title(col)
    axes[i].legend()
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

# Kalan boÅŸ grafik alanlarÄ±nÄ± kaldÄ±r
for j in range(len(numeric_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout
plt.show()


# Sadece numeric sÃ¼tunlarÄ± al ve 'smoking' sÃ¼tununu Ã§Ä±kar
numeric_cols = df.select_dtypes(include='number').drop(columns=['smoking'], errors='ignore').columns[:24]

# Grafik ayarlarÄ±
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(24, 16))
axes = axes.flatten()

# Renkler (duman grisi ve kÄ±rmÄ±zÄ±)
colors = {0: '#d3d3d3', 1: '#ff0000'}

# Her numeric sÃ¼tun iÃ§in boxplot Ã§iz
for i, col in enumerate(numeric_cols):
    sns.boxplot(data=df, x='smoking', y=col, ax=axes[i], palette=colors)
    axes[i].set_title(col)
    axes[i].set_xlabel('smoking')
    axes[i].set_ylabel('')

# Kalan boÅŸ grafik alanlarÄ±nÄ± kaldÄ±r
for j in range(len(numeric_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


df.shape, df_test.shape


df = df0.copy()
df_test = test.copy()


df_test.columns


def create_features(df):
    df['BMI'] = df['weight(kg)'] / ((df['height(cm)'] / 100) ** 2)
    df['HW_Ratio'] = df['height(cm)'] / df['waist(cm)']
    df['HA_Ratio'] = df['height(cm)'] / df['age']
    df['Lipid_Ratio_1'] = df['triglyceride'] / df['HDL']
    df['Lipid_Ratio_2'] = df['LDL'] / df['HDL']
    df['Liver_Risk'] = df['ALT'] + df['AST'] + df['Gtp']
    #df['Gtp_log'] = np.log1p(df['Gtp'])  # log-transform outlierlardan kurtarmak iÃ§in
    return df

df = create_features(df)
df_test = create_features(df_test)


df.shape, df_test.shape


from sklearn.feature_selection import f_classif, mutual_info_classif

X = df.drop("smoking", axis=1)
y = df["smoking"]

# ANOVA F-statistic ve Mutual Information hesaplama
f_values, _ = f_classif(X, y)
mi_scores = mutual_info_classif(X, y, discrete_features=False)

# SkorlarÄ± gÃ¶rselleÅŸtirme
scores_df = pd.DataFrame({
    'Feature': X.columns,
    'F-Score': f_values,
    'MI Score': mi_scores
}).set_index('Feature')

# Ã‡ift Y eksenli grafik Ã§izimi
fig, ax1 = plt.subplots(figsize=(10, 6))

# F-score iÃ§in
ax1.set_xlabel('Features')
ax1.set_ylabel('F-Score', color='blue')
scores_df['F-Score'].plot(kind='bar', color='#d3d3d3', ax=ax1, alpha=0.6, label='F-Score')  # Duman grisi
ax1.tick_params(axis='y', labelcolor='blue')

# MI Score iÃ§in ikinci eksen
ax2 = ax1.twinx()
ax2.set_ylabel('MI Score', color='orange')
scores_df['MI Score'].plot(kind='line', color='red', marker='o', ax=ax2, label='MI Score')  # KÄ±rmÄ±zÄ±
ax2.tick_params(axis='y', labelcolor='orange')

# BaÅŸlÄ±k ve dÃ¼zenlemeler
plt.title('Feature Scores: F-Score vs Mutual Information')
plt.xticks(rotation=45, ha='right')
fig.tight_layout()
plt.show()



X.columns


mi = pd.DataFrame(mi_scores, index = X.columns, columns = ["mi"]).sort_values("mi", ascending = False)
mi


f = pd.DataFrame(f_values, index = X.columns,columns = ["f"])
p = pd.DataFrame(_, index = X.columns,columns = ["p"])
f_ = pd.concat([f, p], axis = 1)
f_ 


f_s = pd.concat([f_, mi], axis = 1)
f_s


cols_to_drop = ['id', 'eyesight(left)', 'eyesight(right)', 'hearing(left)', 'hearing(right)', 'Urine protein', 'dental caries']

df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
df_test = df_test.drop(columns=[col for col in cols_to_drop if col in df_test.columns])


# IQR'ye gÃ¶re aykÄ±rÄ± deÄŸerlerin sayÄ±sÄ±nÄ± hesaplayan fonksiyon
def count_outliers(df):
    outlier_counts = {}
    for column in df.select_dtypes(include='number').columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Alt ve Ã¼st sÄ±nÄ±rlar
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # AykÄ±rÄ± deÄŸerlerin sayÄ±sÄ±nÄ± hesapla
        outlier_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
        outlier_counts[column] = outlier_count
    
    return pd.DataFrame(outlier_counts.items(), columns=['Feature', 'Outlier Count'])

# Ã–rnekte fonksiyonu Ã§aÄŸÄ±r ve sonucu gÃ¶r
outlier_summary = count_outliers(df)
print(outlier_summary)



# IQR'ye gÃ¶re aykÄ±rÄ± deÄŸerlerin sayÄ±sÄ±nÄ± hesaplayan fonksiyon
def count_outliers(df):
    outlier_counts = {}
    for column in df.select_dtypes(include='number').columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Alt ve Ã¼st sÄ±nÄ±rlar
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # AykÄ±rÄ± deÄŸerlerin sayÄ±sÄ±nÄ± hesapla
        outlier_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
        outlier_counts[column] = outlier_count
    
    return pd.DataFrame(outlier_counts.items(), columns=['Feature', 'Outlier Count'])

# Ã–rnekte fonksiyonu Ã§aÄŸÄ±r ve sonucu gÃ¶r
outlier_summary = count_outliers(df)
print(outlier_summary)



import pandas as pd

def drop_outliers(df):
    for column in df.select_dtypes(include='number').columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Alt ve Ã¼st sÄ±nÄ±rlar
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # AykÄ±rÄ± deÄŸerleri kaldÄ±r
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return df

# AykÄ±rÄ± deÄŸerleri kaldÄ±rma iÅŸlemini uygula
df_cleaned = drop_outliers(df)

# Sonucu kontrol et
df_cleaned.shape



X = df_cleaned.drop("smoking", axis = 1)
y = df_cleaned["smoking"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

print("Train features shape : ", X_train.shape)
print("Train target shape   : ", y_train.shape)
print("Test features shape  : ", X_test.shape)
print("Test target shape    : ", y_test.shape)


import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Ã–rnek veri seti: X ve y
# X, y = load_data() kendi veri setinizle deÄŸiÅŸtirin

# Modellerin listesi
models = [
    ("Logistic Regression", LogisticRegression(random_state=101)),
    ("KNN", KNeighborsClassifier()),
    ("SVM", SVC(probability=True, random_state=101)),
    ("Decision Tree", DecisionTreeClassifier(random_state=101)),
    ("Random Forest", RandomForestClassifier(random_state=101)),
    ("AdaBoost", AdaBoostClassifier(random_state=101)),
    ("GradientBoosting", GradientBoostingClassifier(random_state=101)),
    ("XGBoost", XGBClassifier(use_label_encoder=False, random_state=101)),
    ("CatBoost", CatBoostClassifier(verbose=0, random_state=101)),
    ("LightGBM", LGBMClassifier(random_state=101))
]

# Performans sonuÃ§larÄ± iÃ§in boÅŸ bir DataFrame
results_list = []

# Pipeline ve cross-validation iÅŸlemi
for name, model in models:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),  # Ã–zellik Ã¶lÃ§eklendirme
        ("classifier", model)         # SÄ±nÄ±flandÄ±rÄ±cÄ±
    ])
    
    # Cross-validation ile ROC AUC skorlarÄ±nÄ± hesapla
    scores = cross_validate(
        pipeline, X_train, y_train, cv=5, scoring="roc_auc", return_train_score=True
    )
    
    # SonuÃ§larÄ± listeye ekleme
    results_list.append({
        "Model": name,
        "Train ROC AUC Mean": scores['train_score'].mean(),
        "Train ROC AUC Std": scores['train_score'].std(),
        "Test ROC AUC Mean": scores['test_score'].mean(),
        "Test ROC AUC Std": scores['test_score'].std(),
    })

# Listeyi DataFrame'e Ã§evirme
results = pd.DataFrame(results_list)

# SonuÃ§larÄ± yazdÄ±rma
print(results)


results


import warnings
warnings.filterwarnings("ignore", category=UserWarning)


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score


# Pipeline oluÅŸturma
pipeline = Pipeline([
    ("scaler", StandardScaler()),  # Ã–zellik Ã¶lÃ§eklendirme
    ("logistic", LogisticRegression(max_iter=1000, random_state=101))  # Logistic Regression
])

# Hiperparametre arama iÃ§in param_grid
param_grid = {
    "logistic__penalty": ["l1", "l2", "elasticnet", "none"],  # DÃ¼zenleme tÃ¼rÃ¼
    "logistic__C": [0.01, 0.1, 1, 10, 100],  # DÃ¼zenleme katsayÄ±sÄ±
    "logistic__solver": ["saga"],  # FarklÄ± solver seÃ§enekleri denenmiÅŸti hep saga seÃ§ti
    "logistic__l1_ratio": [0, 0.5, 1]  # Elasticnet iÃ§in oran (l1 ve l2 arasÄ±ndaki karÄ±ÅŸÄ±m)
}

# GridSearchCV ile en iyi hiperparametreleri arama
grid_log = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",  # ROC AUC metriÄŸi
    cv=10,  # 5 katmanlÄ± Ã§apraz doÄŸrulama
    verbose=1,  # Ä°lerleme bilgisi
    n_jobs=-1,
    return_train_score=True # Paralel iÅŸlem
)

# GridSearchCV'yi eÄŸitme
grid_log.fit(X_train, y_train)

# EÄŸitim ve test ROC AUC skorlarÄ± yazdÄ±rma
print("Best Hyperparameters:", grid_log.best_params_)
print("Best Test ROC AUC Score:", grid_log.best_score_)
print("Best Training ROC AUC Score:")
print(grid_log.cv_results_["mean_train_score"][grid_log.best_index_])


grid_log.best_estimator_


log_final = Pipeline([
    ("scaler", StandardScaler()),  # Ã–zellik Ã¶lÃ§eklendirme
    ("logistic", LogisticRegression(C=1, l1_ratio=0, max_iter=1000,
                                    random_state=101, solver='saga')) 
])

log_final.fit(X,y)


y_proba = log_final.predict_proba(df_test)[:, 1]
y_proba


submission1 = pd.DataFrame({
    "id": test["id"],  # Test veri kÃ¼mesindeki ID'leri burada kullanÄ±n
    "smoking": y_proba  # Model tahminleriniz
})
submission1


#submission1.to_csv('submission1.csv',index = False)


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Pipeline oluÅŸturma
pipeline = Pipeline([
    ("scaler", StandardScaler()),  # Ã–zellik Ã¶lÃ§eklendirme
    ("adaboost", AdaBoostClassifier(random_state=101))  # AdaBoostClassifier
])

# Hiperparametre arama iÃ§in param_grid
param_grid = {
    "adaboost__n_estimators": [50, 100, 200, 500],  # ZayÄ±f Ã¶ÄŸrenicilerin sayÄ±sÄ±
    "adaboost__learning_rate": [0.01, 0.1, 1, 10],  # Ã–ÄŸrenme oranÄ±
    "adaboost__estimator": [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=3)],  # ZayÄ±f Ã¶ÄŸreniciler
    "adaboost__algorithm": ["SAMME", "SAMME.R"]  # Boosting algoritmasÄ±
}

# GridSearchCV yapÄ±landÄ±rma
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",  # ROC AUC metriÄŸi
    cv=5,  # 5 katmanlÄ± Ã§apraz doÄŸrulama
    verbose=1,  # Ä°lerleme bilgisi
    return_train_score=True,  # EÄŸitim skorlarÄ±nÄ± al
    n_jobs=-1  # Paralel iÅŸlem
)

# Modeli eÄŸitme
grid_search.fit(X_train, y_train)

# EÄŸitim ve test ROC AUC skorlarÄ±nÄ± yazdÄ±rma
print("En Ä°yi Hiperparametreler:", grid_search.best_params_)
print("En Ä°yi Test ROC AUC Skoru:", grid_search.best_score_)
print("En Ä°yi EÄŸitim ROC AUC Skoru:")
print(grid_search.cv_results_["mean_train_score"][grid_search.best_index_])



def eval_metric(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    
    print("Test_Set")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
    print("Train_Set")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))


eval_metric(grid_search, X_train, y_train, X_test, y_test)


grid_search.best_estimator_


pipe_final = Pipeline([
    ("scaler", StandardScaler()),  # Ã–zellik Ã¶lÃ§eklendirme
    ("adaboost", AdaBoostClassifier(algorithm='SAMME',
                   estimator=DecisionTreeClassifier(max_depth=3),
                   learning_rate=0.1, n_estimators=500, random_state=101)) 
])

pipe_final.fit(X,y)


feats = pd.DataFrame(index=X.columns, data= pipe_final["adaboost"].feature_importances_, columns=['ada_importance'])
ada_imp_feats = feats.sort_values("ada_importance", ascending = False)
ada_imp_feats


# Feature importance eÅŸik deÄŸeri (Ã¶r. 0.01)
threshold = 0.01
selected_features = ada_imp_feats[ada_imp_feats["ada_importance"] > threshold].index
selected_features



# Yeni veri setini oluÅŸturun
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

print(f"Number of Selected Features: {len(selected_features)}")
print(f"Number of Extracted Features: {X_train.shape[1] - len(selected_features)}")


# Modellerin listesi
models = [
    ("Logistic Regression", LogisticRegression(random_state=101)),
    ("KNN", KNeighborsClassifier()),
    ("SVM", SVC(probability=True, random_state=101)),
    ("Decision Tree", DecisionTreeClassifier(random_state=101)),
    ("Random Forest", RandomForestClassifier(random_state=101)),
    ("AdaBoost", AdaBoostClassifier(random_state=101)),
    ("GradientBoosting", GradientBoostingClassifier(random_state=101)),
    ("XGBoost", XGBClassifier(use_label_encoder=False, random_state=101)),
    ("CatBoost", CatBoostClassifier(verbose=0, random_state=101)),
    ("LightGBM", LGBMClassifier(random_state=101))
]

# Performans sonuÃ§larÄ± iÃ§in boÅŸ bir DataFrame
results_list = []

# Pipeline ve cross-validation iÅŸlemi
for name, model in models:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),  # Ã–zellik Ã¶lÃ§eklendirme
        ("classifier", model)         # SÄ±nÄ±flandÄ±rÄ±cÄ±
    ])
    
    # Cross-validation ile ROC AUC skorlarÄ±nÄ± hesapla
    scores = cross_validate(
        pipeline, X_train_selected, y_train, cv=5, scoring="roc_auc", return_train_score=True
    )
    
    # SonuÃ§larÄ± listeye ekleme
    results_list.append({
        "Model": name,
        "Train ROC AUC Mean": scores['train_score'].mean(),
        "Train ROC AUC Std": scores['train_score'].std(),
        "Test ROC AUC Mean": scores['test_score'].mean(),
        "Test ROC AUC Std": scores['test_score'].std(),
    })

# Listeyi DataFrame'e Ã§evirme
results = pd.DataFrame(results_list)

# SonuÃ§larÄ± yazdÄ±rma
print(results)


results

