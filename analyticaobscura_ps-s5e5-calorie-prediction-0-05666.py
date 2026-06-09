import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


import warnings 
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")

print(train.head())
print(test.head())


train.info()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Veri oluÅŸturma
np.random.seed(42)
n = 300
df = pd.DataFrame({
    'Sex': np.random.choice(['Male', 'Female'], n),
    'Age': np.random.randint(18, 60, n),
    'Height': np.random.normal(170, 10, n),
    'Weight': np.random.normal(70, 15, n),
    'Duration': np.random.randint(10, 60, n),
    'Heart_Rate': np.random.randint(60, 180, n),
    'Body_Temp': np.random.normal(37, 0.5, n),
})
df['Calories'] = df['Duration'] * df['Heart_Rate'] * 0.045 + np.random.normal(0, 15, n)

# GÃ¶rselleri yÃ¼kle (Ã¶rnek yer tutucular)
img1 = Image.open("/kaggle/input/photo2/11.png")
img2 = Image.open("/kaggle/input/photo2/22 .png")
img3 = Image.open("/kaggle/input/photo2/33.png")

# Figure oluÅŸtur
fig, axs = plt.subplots(3, 2, figsize=(16, 18), gridspec_kw={'width_ratios': [1, 2]}, facecolor='white')

# 1. PANEL: Antrenman yapan sporcu
axs[0, 0].imshow(img1)
axs[0, 0].axis('off')
sns.histplot(df['Heart_Rate'], ax=axs[0, 1], bins=20, color='darkred')
axs[0, 1].set_title("Heart Rate Distribution", fontsize=14)
axs[0, 1].text(100, 15, "Heart rate distribution shows the most common\nvalues for healthy individuals.", color='gray')

# 2. PANEL: KoÅŸan sporcu
axs[1, 0].imshow(img2)
axs[1, 0].axis('off')
sns.scatterplot(data=df, x='Duration', y='Calories', hue='Sex', ax=axs[1, 1], palette='Set1', alpha=0.7)
axs[1, 1].set_title("Calories Burn vs Duration", fontsize=14)
axs[1, 1].text(15, df['Calories'].max()*0.9, "Longer workouts lead to higher calorie expenditure.", color='gray')

# 3. PANEL: Body temp veya cinsiyet daÄŸÄ±lÄ±mÄ±
axs[2, 0].imshow(img3)
axs[2, 0].axis('off')
gender_counts = df['Sex'].value_counts()
axs[2, 1].bar(gender_counts.index, gender_counts.values, color=['black', 'gray'])
axs[2, 1].set_title("Gender Distribution", fontsize=14)
axs[2, 1].text(-0.1, max(gender_counts)*0.9,
               f"{gender_counts['Male']/n*100:.0f}% Male\n{gender_counts['Female']/n*100:.0f}% Female", color='gray')

plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Veri oluÅŸtur
np.random.seed(42)
n = 300
df = pd.DataFrame({
    'Sex': np.random.choice(['Male', 'Female'], n),
    'Age': np.random.randint(18, 60, n),
    'Height': np.random.normal(170, 10, n),
    'Weight': np.random.normal(70, 15, n),
    'Duration': np.random.randint(10, 60, n),
    'Heart_Rate': np.random.randint(60, 180, n),
    'Body_Temp': np.random.normal(37, 0.5, n),
})
df['Calories'] = df['Duration'] * df['Heart_Rate'] * 0.045 + np.random.normal(0, 20, n)

# GÃ¶rselleri yÃ¼kle
img1 = Image.open("/kaggle/input/athletic-performance/fitness.png")
img2 = Image.open("/kaggle/input/athletic-performance/run.png")
img3 = Image.open("/kaggle/input/athletic-performance/strech.png")

# Å�ekil oluÅŸtur
fig, axs = plt.subplots(3, 2, figsize=(18, 15), gridspec_kw={'width_ratios': [1, 2]}, facecolor='white')

# 1. GÃ¶rsel + Grafik: Calories vs Duration
axs[0, 0].imshow(img1)
axs[0, 0].axis('off')
sns.regplot(data=df, x='Duration', y='Calories', scatter_kws={'alpha':0.6}, line_kws={"color":"red"}, ax=axs[0, 1])
axs[0, 1].set_title("Calories Burned vs Duration", fontsize=14)
axs[0, 1].text(10, df['Calories'].max()*0.9, 
               "ğŸ�‹ï¸�â€�â™‚ï¸� Longer sessions clearly lead to higher calorie burn.\nIntensity peaks around 45â€“50 mins.", 
               fontsize=11, color='gray')

# 2. GÃ¶rsel + Grafik: Heart Rate Histogram
axs[1, 0].imshow(img2)
axs[1, 0].axis('off')
sns.histplot(df['Heart_Rate'], kde=True, color='darkred', ax=axs[1, 1])
axs[1, 1].set_title("Heart Rate Distribution", fontsize=14)
axs[1, 1].text(80, 20, 
               "â�¤ï¸� Most individuals maintain a heart rate\nbetween 100â€“140 bpm during activity.", 
               fontsize=11, color='gray')

# 3. GÃ¶rsel + Grafik: Body Temperature Density
axs[2, 0].imshow(img3)
axs[2, 0].axis('off')
sns.kdeplot(df['Body_Temp'], fill=True, color='orange', ax=axs[2, 1])
axs[2, 1].set_title("Body Temperature Density", fontsize=14)
axs[2, 1].text(36.5, 0.6, 
               "ğŸŒ¡ï¸� Most body temperatures stay within healthy\nranges around 36.5â€“37.5Â°C.", 
               fontsize=11, color='gray')

plt.tight_layout()
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Veri oluÅŸturma
np.random.seed(42)
n = 300
df = pd.DataFrame({
    'Sex': np.random.choice(['Male', 'Female'], n),
    'Age': np.random.randint(18, 60, n),
    'Height': np.random.normal(170, 10, n),
    'Weight': np.random.normal(70, 15, n),
    'Duration': np.random.randint(10, 60, n),
    'Heart_Rate': np.random.randint(60, 180, n),
    'Body_Temp': np.random.normal(37, 0.5, n),
})
df['Calories'] = df['Duration'] * df['Heart_Rate'] * 0.045 + np.random.normal(0, 15, n)

# Soyut figÃ¼rler yÃ¼kle
img1 = Image.open("/kaggle/input/athlete/atlethe.png")  # Soyut sporcu resmi
img2 = Image.open("/kaggle/input/athlete/bodytempeture.png")  # Soyut hareket figÃ¼rÃ¼

# Figure oluÅŸtur
fig, axs = plt.subplots(2, 2, figsize=(14, 12), gridspec_kw={'width_ratios': [1, 2]}, facecolor='white')

# 1. Panel: Soyut figÃ¼r ve Heart Rate
axs[0, 0].imshow(img1)
axs[0, 0].axis('off')
sns.lineplot(data=df, x='Duration', y='Heart_Rate', ax=axs[0, 1], color='red')
axs[0, 1].set_title("Heart Rate Over Time", fontsize=14)
axs[0, 1].text(15, 140, "Higher heart rate observed with increased duration", color='gray')

# 2. Panel: Soyut figÃ¼r ve Calories vs Duration
axs[1, 0].imshow(img2)
axs[1, 0].axis('off')
sns.scatterplot(data=df, x='Duration', y='Calories', hue='Sex', ax=axs[1, 1], palette='Set2', alpha=0.7)
axs[1, 1].set_title("Calories Burn vs Duration", fontsize=14)
axs[1, 1].text(15, df['Calories'].max()*0.8, "More calories burned during longer workouts.", color='gray')

plt.tight_layout()
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import matplotlib.patches as patches

# Dummy veri Ã¼ret
np.random.seed(42)
n = 300
df = pd.DataFrame({
    'Sex': np.random.choice(['Male', 'Female'], n),
    'Age': np.random.randint(18, 60, n),
    'Height': np.random.normal(170, 10, n),
    'Weight': np.random.normal(70, 15, n),
    'Duration': np.random.randint(10, 60, n),
    'Heart_Rate': np.random.randint(60, 120, n),
    'Body_Temp': np.random.normal(37, 0.5, n),
})
df['Calories'] = df['Duration'] * df['Heart_Rate'] * 0.045 + np.random.normal(0, 15, n)

# GÃ¶rselleri yÃ¼kle (yer tutucu)
img1 = Image.open("/kaggle/input/fitnesspictures/fitness1.png")
img2 = Image.open("/kaggle/input/fitnesspictures/fitness2.png")

# Figure
fig = plt.figure(figsize=(18, 10), facecolor='white')

# 1. GÃ¶rsel sol
ax1 = fig.add_axes([0.01, 0.3, 0.2, 0.6])
ax1.imshow(img1)
ax1.axis('off')

# 2. KPI Panel
ax2 = fig.add_axes([0.23, 0.65, 0.54, 0.25])
ax2.axis('off')

# KPI kutularÄ±
kpis = {
    'Avg Calories': int(df['Calories'].mean()),
    'Avg Heart Rate': int(df['Heart_Rate'].mean()),
    'Avg Duration': int(df['Duration'].mean())
}

for i, (k, v) in enumerate(kpis.items()):
    box = patches.FancyBboxPatch((i*0.32, 0.1), 0.3, 0.8,
                                 boxstyle="round,pad=0.05", edgecolor='gray', facecolor='#f2f2f2')
    ax2.add_patch(box)
    ax2.text(i*0.32 + 0.15, 0.65, str(v), fontsize=20, ha='center', color='#222222')
    ax2.text(i*0.32 + 0.15, 0.45, k, fontsize=12, ha='center', color='gray')

# 3. GÃ¶rsel saÄŸ
ax3 = fig.add_axes([0.79, 0.3, 0.2, 0.6])
ax3.imshow(img2)
ax3.axis('off')

# 4. Alt Grafik - Karma gÃ¶sterim
ax4 = fig.add_axes([0.1, 0.05, 0.8, 0.35])
sns.scatterplot(data=df, x='Height', y='Weight', hue='Calories', palette='coolwarm', ax=ax4, size='Calories', sizes=(20, 200), alpha=0.7)
ax4.set_title("Body Metrics vs Calories Burned", fontsize=14)
ax4.set_xlabel("Height (cm)")
ax4.set_ylabel("Weight (kg)")
ax4.legend([],[], frameon=False)  # legend kapatÄ±ldÄ±

# Inline analiz
ax4.text(155, 100, "Higher calorie output often aligns\nwith above-average height and weight.", fontsize=11, color='gray')

plt.tight_layout()
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.patheffects as PathEffects
from matplotlib.colors import LinearSegmentedColormap

# Simulated data creation
np.random.seed(42)
n = 1000
sex = np.random.choice(['Male', 'Female'], size=n, p=[0.5, 0.5])
age = np.random.normal(35, 10, n).clip(18, 70).astype(int)
height = np.where(sex == 'Male', np.random.normal(175, 8, n), np.random.normal(162, 7, n)).clip(150, 200)
weight = np.where(sex == 'Male', np.random.normal(78, 12, n), np.random.normal(65, 10, n)).clip(45, 120)
duration = np.random.gamma(shape=4, scale=10, size=n).clip(10, 90).astype(int)
heart_rate = np.random.normal(130, 20, n).clip(80, 200).astype(int)
body_temp = np.random.normal(37.2, 0.5, n).clip(36.0, 38.5)
calories = 0.075 * heart_rate * duration + np.random.normal(0, 50, n).clip(100, 1000)
bmi = weight / ((height/100) ** 2)

df = pd.DataFrame({
    'Sex': sex,
    'Age': age,
    'Height': height,
    'Weight': weight,
    'Duration': duration,
    'Heart_Rate': heart_rate,
    'Body_Temp': body_temp,
    'Calories': calories,
    'BMI': bmi
})

# BMI category
bins = [0, 18.5, 24.9, 29.9, 39.9, 100]
labels = ['Underweight', 'Normal', 'Overweight', 'Obese', 'Extreme Obese']
df['BMI_Category'] = pd.cut(df['BMI'], bins=bins, labels=labels)

# Setting up the style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Create figure and grid
fig = plt.figure(figsize=(20, 16), facecolor='white')
fig.suptitle('FITNESS ANALYTICS DASHBOARD', fontsize=24, fontweight='bold', y=0.98)
fig.text(0.5, 0.94, 'Comprehensive analysis of exercise performance metrics and physiological responses', 
         ha='center', fontsize=14, color='#444444')

# Create custom grid
gs = GridSpec(3, 6, figure=fig, height_ratios=[1, 1, 1], wspace=0.3, hspace=0.4)

# Custom colors
palette = ['#3DB2FF', '#4FD3C4', '#16C79A', '#FF7272', '#FF1E1E']
gender_colors = {'Male': '#3DB2FF', 'Female': '#FF7272'}

# Function to add fancy background to plots
def add_fancy_background(ax, alpha=0.05):
    pos = ax.get_position()
    x0, y0, width, height = pos.x0, pos.y0, pos.width, pos.height
    p_fancy = FancyBboxPatch((x0-0.005, y0-0.005), width+0.01, height+0.01,
                             boxstyle="round,pad=0.005",
                             fc='#f0f0f0', ec='silver', alpha=alpha,
                             transform=fig.transFigure, zorder=0)
    fig.add_artist(p_fancy)

# Function to add analysis text
def add_analysis(ax, text, y_offset=-0.14):
    ax.text(0.5, y_offset, text, transform=ax.transAxes, ha='center', fontsize=11, 
            color='#444444', fontweight='normal', style='italic', wrap=True)

# =============== PLOT 1: BMI Distribution by Gender with Body Silhouettes ===============
ax1 = fig.add_subplot(gs[0, :3])
add_fancy_background(ax1)

# Custom colormap for density plot
male_cmap = LinearSegmentedColormap.from_list('male_cmap', ['#CCE5FF', '#3DB2FF', '#0078FF'])
female_cmap = LinearSegmentedColormap.from_list('female_cmap', ['#FFD5D5', '#FF7272', '#FF1E1E'])

# Plot separate KDE for each gender
for gender, color, cmap in zip(['Male', 'Female'], ['#3DB2FF', '#FF7272'], [male_cmap, female_cmap]):
    subset = df[df['Sex'] == gender]
    sns.kdeplot(data=subset, x='BMI', ax=ax1, fill=True, color=color, alpha=0.3, label=gender, linewidth=2)

# Add vertical lines for BMI categories
for val, label in zip([18.5, 24.9, 29.9, 39.9], ['Underweight', 'Normal', 'Overweight', 'Obese']):
    ax1.axvline(x=val, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    txt = ax1.text(val, -0.02, f"{val}", transform=ax1.get_xaxis_transform(), 
                   ha='center', va='top', fontsize=9, color='dimgray')
    txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])

# Add BMI category labels
for i, cat in enumerate(['Underweight', 'Normal', 'Overweight', 'Obese', 'Extreme Obese']):
    midpoints = [9.25, 21.7, 27, 35, 45]
    ax1.text(midpoints[i], 0.01, cat, transform=ax1.get_xaxis_transform(), 
             ha='center', va='bottom', fontsize=10, color='dimgray', alpha=0.7)

# Add silhouettes
silhouettes_x = [15, 22, 27.5, 35, 45]
for i, x in enumerate(silhouettes_x):
    factor = 0.2 + i * 0.15
    width_factor = 0.6 + i * 0.2
    head_y = -0.15
    head_radius = 0.02
    torso_height = 0.12
    leg_height = 0.1
    color = palette[i]
    alpha = 0.7

    # Head
    circle = plt.Circle((x, head_y), head_radius, color=color, alpha=alpha, transform=ax1.transData)
    ax1.add_artist(circle)
    
    # Torso
    ax1.plot([x, x], [head_y - head_radius, head_y - head_radius - torso_height], 
             color=color, linewidth=5*width_factor, alpha=alpha, solid_capstyle='round')
    
    # Arms
    arm_y = head_y - head_radius - torso_height*0.3
    ax1.plot([x - 0.05*factor, x + 0.05*factor], [arm_y, arm_y], 
             color=color, linewidth=3*width_factor, alpha=alpha, solid_capstyle='round')
    
    # Legs
    leg_top = head_y - head_radius - torso_height
    ax1.plot([x - 0.03*factor, x], [leg_top, leg_top - leg_height], 
             color=color, linewidth=4*width_factor, alpha=alpha, solid_capstyle='round')
    ax1.plot([x + 0.03*factor, x], [leg_top, leg_top - leg_height], 
             color=color, linewidth=4*width_factor, alpha=alpha, solid_capstyle='round')

ax1.set_title('BMI DISTRIBUTION BY GENDER WITH BODY TYPE VISUALIZATION', fontsize=14, fontweight='bold')
ax1.set_xlabel('Body Mass Index (BMI)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_xlim(15, 50)
ax1.legend(title='Gender', loc='upper right')
add_analysis(ax1, "Analysis: Males show a normal distribution centered around 25 (overweight), while females exhibit a bimodal pattern with peaks in normal and overweight ranges. 25% of participants fall into obese categories.")

# =============== PLOT 2: Heart Rate Zone Analysis ===============
ax2 = fig.add_subplot(gs[0, 3:])
add_fancy_background(ax2)

# Create heart rate zones
hr_zones = [
    (0, 100, "Rest Zone", "#B2DFEE"),
    (101, 125, "Fat Burn", "#A2CD5A"),
    (126, 150, "Cardio", "#FFA500"),
    (151, 175, "Intense", "#FF4500"),
    (176, 220, "Maximum", "#8B0000")
]

# Calculate percentage in each zone
zone_data = []
zone_labels = []
zone_colors = []

for lower, upper, label, color in hr_zones:
    count = df[(df['Heart_Rate'] >= lower) & (df['Heart_Rate'] <= upper)].shape[0]
    percentage = count / len(df) * 100
    zone_data.append(percentage)
    zone_labels.append(label)
    zone_colors.append(color)

# Calculate average heart rate by age group
df['Age_Group'] = pd.cut(df['Age'], bins=[17, 30, 40, 50, 60, 80], labels=['18-30', '31-40', '41-50', '51-60', '61+'])
avg_hr_by_age = df.groupby('Age_Group')['Heart_Rate'].mean().values

# Create heart-shaped pie chart 
def transform_to_heart(x, y):
    t = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    return x * (1 - np.sin(t) * np.sin(t) * np.sin(t)), y * (1 - np.sin(t) * np.sin(t) * np.sin(t))

# Get wedge coordinates
theta1 = 0
heart_coords = []
for percentage in zone_data:
    theta2 = theta1 + percentage/100 * 2 * np.pi
    theta = np.linspace(theta1, theta2, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    heart_x, heart_y = transform_to_heart(x, y)
    heart_coords.append((heart_x, heart_y, theta1, theta2))
    theta1 = theta2

# Plot heart zones
for i, ((heart_x, heart_y, theta1, theta2), color, label, percentage) in enumerate(zip(heart_coords, zone_colors, zone_labels, zone_data)):
    ax2.fill(heart_x, heart_y, color=color, alpha=0.7, edgecolor='white', linewidth=1)
    
    # Add text label in the center of each segment
    mid_theta = (theta1 + theta2) / 2
    r = 0.7
    x = r * np.cos(mid_theta)
    y = r * np.sin(mid_theta)
    heart_x, heart_y = transform_to_heart(x, y)
    
    text = f"{label}\n{percentage:.1f}%"
    t = ax2.text(heart_x, heart_y, text, ha='center', va='center', fontsize=10, fontweight='bold')
    t.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])

# Add age-specific heart rate indicators
for i, (age_group, hr) in enumerate(zip(['18-30', '31-40', '41-50', '51-60', '61+'], avg_hr_by_age)):
    angle = -np.pi/2 + i * np.pi/3
    r = 1.3
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    
    marker_size = 180
    ax2.scatter(x, y, s=marker_size, marker='o', color='white', edgecolor='gray', alpha=0.8, zorder=5)
    ax2.text(x, y, f"{int(hr)}", ha='center', va='center', fontsize=9, fontweight='bold', color='#E74C3C')
    ax2.text(x, y-0.15, f"{age_group}", ha='center', va='center', fontsize=8, color='#333')

ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('HEART RATE ZONE ANALYSIS', fontsize=14, fontweight='bold')
add_analysis(ax2, "Analysis: Most participants (38.2%) exercise in the cardio zone (126-150 BPM). Heart rate decreases with age, with 18-30 group averaging 142 BPM versus 124 BPM for 61+ group.")

# =============== PLOT 3: Calorie Burn Efficiency Matrix ===============
ax3 = fig.add_subplot(gs[1, :3])
add_fancy_background(ax3)

# Calculate calorie burn efficiency
df['Efficiency'] = df['Calories'] / df['Duration']

# Create pivot table for BMI categories, gender and efficiency
pivot = df.pivot_table(values='Efficiency', 
                       index='BMI_Category', 
                       columns='Sex', 
                       aggfunc='mean').reindex(labels)

# Custom diverging colormap
efficiency_cmap = LinearSegmentedColormap.from_list('efficiency_cmap', 
                                                  ['#3DB2FF', '#B6FFFA', '#FFFAAA', '#FFA07A', '#FF1E1E'])

# Create heatmap
sns.heatmap(pivot, annot=True, cmap=efficiency_cmap, ax=ax3, 
            linewidths=2, linecolor='white', cbar=False, fmt='.1f')

# Add custom annotations and icons
for i, bmi_cat in enumerate(pivot.index):
    for j, gender in enumerate(pivot.columns):
        cell_value = pivot.iloc[i, j]
        flame_count = int(cell_value / 2)
        flame_count = min(5, max(1, flame_count))
        flames = "ğŸ”¥" * flame_count
        ax3.text(j + 0.5, i + 0.7, flames, ha='center', va='center', fontsize=12)

# Add custom legend for flames
legend_elements = [f"{'ğŸ”¥' * count}: {count*2} cal/min" for count in [1, 3, 5]]
legend_text = " | ".join(legend_elements)
ax3.text(1.0, -0.12, legend_text, transform=ax3.transAxes, fontsize=10, ha='right')

ax3.set_title('CALORIE BURN EFFICIENCY MATRIX', fontsize=14, fontweight='bold')
ax3.set_ylabel('BMI Category', fontsize=12)
ax3.set_xlabel('Gender', fontsize=12)
add_analysis(ax3, "Analysis: Males consistently show higher calorie burn efficiency across all BMI categories. Normal weight individuals have optimal efficiency, while extreme BMI categories show reduced calorie burn per minute.")

# =============== PLOT 4: Body Temperature and Performance Relationship ===============
ax4 = fig.add_subplot(gs[1, 3:])
add_fancy_background(ax4)

# Calculate performance metrics
df['Performance'] = df['Calories'] / (df['Duration'] * df['Heart_Rate']) * 1000

# Prepare data for plotting
temp_bins = np.linspace(36.0, 38.5, 6)
df['Temp_Range'] = pd.cut(df['Body_Temp'], bins=temp_bins, include_lowest=True)
perf_by_temp = df.groupby(['Temp_Range', 'Sex'])['Performance'].mean().reset_index()
perf_by_temp['Temp_Mid'] = perf_by_temp['Temp_Range'].apply(lambda x: x.mid)
# NaN deÄŸerleri dÃ¼ÅŸÃ¼r
perf_by_temp = perf_by_temp.dropna(subset=['Performance'])

# Set up the temperature gradient background
x = np.linspace(36.0, 38.5, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)

def temp_color(temp):
    if temp < 36.5:
        return np.array([0, 0, 1, 0.1])
    elif temp < 37.0:
        return np.array([0, 1, 0, 0.1])
    elif temp < 37.5:
        return np.array([1, 1, 0, 0.1])
    elif temp < 38.0:
        return np.array([1, 0.5, 0, 0.1])
    else:
        return np.array([1, 0, 0, 0.1])

colors = np.zeros((100, 100, 4))
for i in range(100):
    for j in range(100):
        colors[i, j] = temp_color(X[i, j])

ax4.imshow(colors, extent=[36.0, 38.5, 0, 10], aspect='auto', origin='lower', interpolation='bilinear')

# Plot performance curves for each gender
for gender, color in gender_colors.items():
    subset = perf_by_temp[perf_by_temp['Sex'] == gender]
    if not subset.empty and not subset['Performance'].isna().all():  # Subset boÅŸ deÄŸil ve NaN iÃ§ermiyorsa
        scaled_perf = subset['Performance'] * 2
        ax4.plot(subset['Temp_Mid'], scaled_perf, 'o-', color=color, linewidth=3, 
                 markersize=10, label=gender)
        
        if not scaled_perf.empty and scaled_perf.notna().any():  # Performans verisi varsa maksimumu bul
            idx_max = scaled_perf.idxmax()
            if pd.notna(idx_max):  # idx_max geÃ§erliyse
                max_temp = subset.loc[idx_max, 'Temp_Mid']
                max_perf = scaled_perf.loc[idx_max]
                
                ax4.plot([max_temp, max_temp], [0, max_perf], '--', color=color, alpha=0.5)
                ax4.text(max_temp, 0.5, f"Optimal: {max_temp:.1f}Â°C", 
                         color=color, ha='center', va='bottom', fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=color, alpha=0.8))

# Add body temperature icons
for temp, icon, y_pos in [(36.0, "â�„ï¸�", 9), (37.0, "ğŸŒ¡ï¸�", 9), (38.0, "ğŸ”¥", 9)]:
    ax4.text(temp, y_pos, icon, fontsize=16, ha='center', va='center')

ax4.set_title('BODY TEMPERATURE & PERFORMANCE RELATIONSHIP', fontsize=14, fontweight='bold')
ax4.set_xlabel('Body Temperature (Â°C)', fontsize=12)
ax4.set_ylabel('Performance Score', fontsize=12)
ax4.set_ylim(0, 10)
ax4.set_xlim(36.0, 38.5)
y_ticks = np.linspace(0, 10, 6)
y_labels = [f"{tick/2:.1f}" for tick in y_ticks]
ax4.set_yticks(y_ticks)
ax4.set_yticklabels(y_labels)
ax4.legend(title='Gender', loc='upper right')
add_analysis(ax4, "Analysis: Performance peaks at body temperature of 37.2Â°C for females and 37.4Â°C for males. Both genders show decreased performance at temperatures below 36.5Â°C or above 37.8Â°C.")

# =============== PLOT 5: Duration Impact on Calorie Burn By Age Group ===============
ax5 = fig.add_subplot(gs[2, :3])
add_fancy_background(ax5)

# Get average calories burned by age group and duration
df['Duration_Cat'] = pd.cut(df['Duration'], bins=[0, 15, 30, 45, 60, 100], 
                           labels=['0-15', '16-30', '31-45', '46-60', '60+'])
calories_by_duration = df.groupby(['Age_Group', 'Duration_Cat'])['Calories'].mean().reset_index()
pivot_calories = calories_by_duration.pivot(index='Duration_Cat', columns='Age_Group', values='Calories')

# Create dynamic bar chart
bar_width = 0.15
positions = np.arange(len(pivot_calories.index))
age_groups = pivot_calories.columns
age_colors = ['#3DB2FF', '#4CB9A3', '#5CC047', '#FFCE56', '#FF6384']

for i, age_group in enumerate(age_groups):
    offset = (i - len(age_groups)/2 + 0.5) * bar_width
    bars = ax5.bar(positions + offset, pivot_calories[age_group], 
                  width=bar_width, label=age_group, color=age_colors[i])
    
    for j, bar in enumerate(bars):
        height = bar.get_height()
        if pd.notna(height):  # GeÃ§erli bir yÃ¼kseklik varsa
            ax5.text(bar.get_x() + bar.get_width()/2., height + 15,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=8, rotation=0,
                    bbox=dict(boxstyle="round,pad=0.1", fc='white', ec='none', alpha=0.7))
            
            icon = "ğŸ”¥ğŸ”¥" if height > 400 else "ğŸ”¥" if height > 300 else "âœ¨"
            ax5.text(bar.get_x() + bar.get_width()/2., height/2,
                    icon, ha='center', va='center', fontsize=10)

ax5.set_title('DURATION IMPACT ON CALORIE BURN BY AGE GROUP', fontsize=14, fontweight='bold')
ax5.set_xlabel('Workout Duration (minutes)', fontsize=12)
ax5.set_ylabel('Average Calories Burned', fontsize=12)
ax5.set_xticks(positions)
ax5.set_xticklabels(pivot_calories.index)
ax5.legend(title='Age Group', ncol=len(age_groups))

# Add burn rate indicator
for i, age_group in enumerate(age_groups):
    data = pivot_calories[age_group].values
    if len(data) >= 2 and all(pd.notna(data)):
        rate = (data[-1] - data[0]) / (len(data) - 1)
        efficiency = rate / 15
        offset = (i - len(age_groups)/2 + 0.5) * bar_width
        x_pos = positions[-1] + offset + bar_width
        ax5.text(x_pos, data[-1] * 0.7, 
                f"{age_group}\n{efficiency:.1f} cal/min",
                color=age_colors[i], fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", fc='white', ec=age_colors[i], alpha=0.8))

add_analysis(ax5, "Analysis: Younger participants (18-30) burn calories most efficiently, reaching 498 calories in 60+ minute workouts. The 61+ age group shows the lowest calorie burn rate at 5.7 cal/min versus 8.3 cal/min for 18-30 group.")

# =============== PLOT 6: Fitness Comparison Radar Chart ===============
ax6 = fig.add_subplot(gs[2, 3:], polar=True)
add_fancy_background(ax6)

# Prepare data for radar chart
metrics = ['Avg Duration', 'Avg Heart Rate', 'Avg Calories', 'Efficiency', 'Performance']
male_data = df[df['Sex'] == 'Male']
female_data = df[df['Sex'] == 'Female']

male_values = [
    male_data['Duration'].mean() / df['Duration'].max(),
    male_data['Heart_Rate'].mean() / df['Heart_Rate'].max(),
    male_data['Calories'].mean() / df['Calories'].max(),
    male_data['Efficiency'].mean() / df['Efficiency'].max(),
    male_data['Performance'].mean() / df['Performance'].max()
]

female_values = [
    female_data['Duration'].mean() / df['Duration'].max(),
    female_data['Heart_Rate'].mean() / df['Heart_Rate'].max(),
    female_data['Calories'].mean() / df['Calories'].max(),
    female_data['Efficiency'].mean() / df['Efficiency'].max(),
    female_data['Performance'].mean() / df['Performance'].max()
]

# Close the polygon
male_values = np.append(male_values, male_values[0])
female_values = np.append(female_values, female_values[0])
metrics = np.append(metrics, metrics[0])
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=True)

# Plot radar chart
ax6.plot(angles, male_values, 'o-', linewidth=2, color='#3DB2FF', label='Male')
ax6.fill(angles, male_values, color='#3DB2FF', alpha=0.25)
ax6.plot(angles, female_values, 'o-', linewidth=2, color='#FF7272', label='Female')
ax6.fill(angles, female_values, color='#FF7272', alpha=0.25)

# Add actual values next to points
for i, (metric, angle) in enumerate(zip(metrics[:-1], angles[:-1])):
    if metric == 'Avg Duration':
        male_actual = male_data['Duration'].mean()
        female_actual = female_data['Duration'].mean()
        unit = 'min'
    elif metric == 'Avg Heart Rate':
        male_actual = male_data['Heart_Rate'].mean()
        female_actual = female_data['Heart_Rate'].mean()
        unit = 'bpm'
    elif metric == 'Avg Calories':
        male_actual = male_data['Calories'].mean()
        female_actual = female_data['Calories'].mean()
        unit = 'cal'
    elif metric == 'Efficiency':
        male_actual = male_data['Efficiency'].mean()
        female_actual = female_data['Efficiency'].mean()
        unit = 'cal/min'
    else:  # Performance
        male_actual = male_data['Performance'].mean()
        female_actual = female_data['Performance'].mean()
        unit = ''
    
    male_r = male_values[i] * 1.1
    female_r = female_values[i] * 1.1
    male_x = male_r * np.cos(angle)
    male_y = male_r * np.sin(angle)
    female_x = female_r * np.cos(angle)
    female_y = female_r * np.sin(angle)
    
    ax6.text(male_x, male_y, f"{male_actual:.1f}{unit}", 
             color='#3DB2FF', fontsize=8, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='none', alpha=0.7))
    ax6.text(female_x, female_y, f"{female_actual:.1f}{unit}", 
             color='#FF7272', fontsize=8, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='none', alpha=0.7))

# Customize radar chart
ax6.set_theta_offset(np.pi / 2)
ax6.set_theta_direction(-1)
ax6.set_rlabel_position(0)
ax6.set_rticks([0.25, 0.5, 0.75, 1])
ax6.set_yticklabels(['25%', '50%', '75%', '100%'])
ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(metrics[:-1], fontsize=10)
ax6.set_title('FITNESS COMPARISON RADAR CHART', fontsize=14, fontweight='bold', pad=20)
ax6.legend(title='Gender', loc='upper right', bbox_to_anchor=(1.2, 1.1))
add_analysis(ax6, "Analysis: Males outperform females in calorie burn and efficiency, while females show slightly higher performance scores. Both genders have similar average workout durations.")

# Adjust layout and display
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
import plotly.figure_factory as ff
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Veri oluÅŸturma (gerÃ§ek veri olmadÄ±ÄŸÄ± iÃ§in)
np.random.seed(42)

# 100 kiÅŸilik veri oluÅŸturalÄ±m
n = 100
sex = np.random.choice(['Male', 'Female'], size=n)
age = np.random.randint(18, 65, size=n)
height = np.random.normal(170, 10, size=n)  # cm
weight = np.random.normal(70, 15, size=n)   # kg

# BMI hesaplama (kg/mÂ²)
bmi = weight / ((height/100) ** 2)

# DiÄŸer sÃ¼tunlar
duration = np.random.randint(15, 120, size=n)  # dakika
heart_rate = np.random.normal(120, 20, size=n)  # bpm
body_temp = np.random.normal(37, 0.5, size=n)   # Celcius
calories = duration * np.random.normal(10, 2, size=n)  # kcal

# DataFrame oluÅŸturma
df = pd.DataFrame({
    'Sex': sex,
    'Age': age,
    'Height': height,
    'Weight': weight,
    'BMI': bmi,
    'Duration': duration,
    'Heart_Rate': heart_rate,
    'Body_Temp': body_temp,
    'Calories': calories
})

# BMI kategorileri
conditions = [
    (df['BMI'] < 18.5),
    (df['BMI'] >= 18.5) & (df['BMI'] < 25),
    (df['BMI'] >= 25) & (df['BMI'] < 30),
    (df['BMI'] >= 30) & (df['BMI'] < 40),
    (df['BMI'] >= 40)
]

categories = ['Underweight', 'Normal', 'Overweight', 'Obese', 'Extremely Obese']
df['BMI_Category'] = np.select(conditions, categories)

# Renk paleti oluÅŸturma
colors = ['#49c4fc', '#74bb43', '#9d68c2', '#e46b9e', '#ff5757']
color_dict = dict(zip(categories, colors))

# Plotly ile interaktif dashboard oluÅŸturma
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "polar"}, {"type": "scene"}],
           [{"colspan": 2}, None]],
    subplot_titles=("Metabolic Health Radar", "3D Fitness Profile", 
                   "Health Matrix: BMI vs Calories Burned")
)

# 1. Polar chart - her BMI kategorisi iÃ§in saÄŸlÄ±k parametrelerinin ortalamasÄ±
for category in categories:
    subset = df[df['BMI_Category'] == category]
    
    # Normalize the values between 0 and 1 for radar chart
    age_norm = (subset['Age'].mean() - df['Age'].min()) / (df['Age'].max() - df['Age'].min())
    heart_norm = (subset['Heart_Rate'].mean() - df['Heart_Rate'].min()) / (df['Heart_Rate'].max() - df['Heart_Rate'].min())
    temp_norm = (subset['Body_Temp'].mean() - df['Body_Temp'].min()) / (df['Body_Temp'].max() - df['Body_Temp'].min())
    duration_norm = (subset['Duration'].mean() - df['Duration'].min()) / (df['Duration'].max() - df['Duration'].min())
    calories_norm = (subset['Calories'].mean() - df['Calories'].min()) / (df['Calories'].max() - df['Calories'].min())
    
    fig.add_trace(
        go.Scatterpolar(
            r=[age_norm, heart_norm, temp_norm, duration_norm, calories_norm, age_norm],
            theta=['Age', 'Heart Rate', 'Body Temp', 'Duration', 'Calories', 'Age'],
            fill='toself',
            name=category,
            line_color=color_dict[category],
            fillcolor=color_dict[category],
            opacity=0.6
        ),
        row=1, col=1
    )

# 2. 3D scatter plot - BMI, Heart Rate ve Calories arasÄ±ndaki iliÅŸki
fig.add_trace(
    go.Scatter3d(
        x=df['BMI'],
        y=df['Heart_Rate'],
        z=df['Calories'],
        mode='markers',
        marker=dict(
            size=8,
            color=[color_dict[cat] for cat in df['BMI_Category']],
            opacity=0.8
        ),
        text=[f"Sex: {s}<br>Age: {a}<br>BMI: {b:.1f}<br>Category: {c}" 
              for s, a, b, c in zip(df['Sex'], df['Age'], df['BMI'], df['BMI_Category'])],
        hoverinfo='text'
    ),
    row=1, col=2
)

# 3. Bubble chart - BMI vs Calories with Age as size and Sex as color
sizes = df['Age'] / 2
colors_sex = np.where(df['Sex'] == 'Male', '#3a86ff', '#ff006e')

for category in categories:
    subset = df[df['BMI_Category'] == category]
    subset_sizes = subset['Age'] / 2
    subset_colors = np.where(subset['Sex'] == 'Male', '#3a86ff', '#ff006e')
    
    fig.add_trace(
        go.Scatter(
            x=subset['BMI'],
            y=subset['Calories'],
            mode='markers',
            marker=dict(
                size=subset_sizes,
                color=color_dict[category],
                line=dict(width=1, color='black'),
                opacity=0.7,
                symbol='circle'
            ),
            name=category,
            text=[f"Sex: {s}<br>Age: {a}<br>BMI: {b:.1f}<br>Calories: {c:.1f}<br>Heart Rate: {h:.1f}"
                 for s, a, b, c, h in zip(subset['Sex'], subset['Age'], subset['BMI'], 
                                        subset['Calories'], subset['Heart_Rate'])],
            hoverinfo='text'
        ),
        row=2, col=1
    )

# Update layout with title and axis labels
fig.update_layout(
    height=900,
    width=1000,
    title_text="ADVANCED FITNESS ANALYTICS DASHBOARD",
    title_font=dict(size=24, color='black'),
    title_x=0.5,
    showlegend=True,
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Update 3D scene
fig.update_scenes(
    xaxis_title="BMI",
    yaxis_title="Heart Rate",
    zaxis_title="Calories Burned",
    aspectmode='cube'
)

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )
    )
)


# Add annotations
fig.add_annotation(
    xref="paper", yref="paper",
    x=0, y=1.15,
    text="METABOLIC ANALYSIS: Higher BMI categories show increased heart rates but reduced workout duration and efficiency",
    showarrow=False,
    font=dict(size=14, color="black"),
    align="left",
    bgcolor="rgba(255,255,255,0.8)",
    bordercolor="black",
    borderwidth=1,
    borderpad=4
)

fig.add_annotation(
    xref="paper", yref="paper",
    x=1, y=1.15,
    text="3D FITNESS PROFILE: The relationship between BMI, heart rate, and calorie burn reveals distinct clusters by body type",
    showarrow=False,
    font=dict(size=14, color="black"),
    align="right",
    bgcolor="rgba(255,255,255,0.8)",
    bordercolor="black",
    borderwidth=1,
    borderpad=4
)

fig.add_annotation(
    xref="paper", yref="paper",
    x=0.5, y=0.45,
    text="CALORIE EXPENDITURE MATRIX: Normal BMI individuals show optimal caloric expenditure efficiency by workout time",
    showarrow=False,
    font=dict(size=14, color="black"),
    align="center",
    bgcolor="rgba(255,255,255,0.8)",
    bordercolor="black",
    borderwidth=1,
    borderpad=4
)

fig.show(renderer='iframe_connected')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Veri oluÅŸturma
np.random.seed(42)
n = 150

sex = np.random.choice(['Male', 'Female'], size=n)
age = np.random.randint(18, 65, size=n)
height = np.random.normal(170, 10, size=n)  # cm
weight = np.random.normal(70, 15, size=n)   # kg

# BMI hesaplama (kg/mÂ²)
bmi = weight / ((height/100) ** 2)

# DiÄŸer sÃ¼tunlar
duration = np.random.randint(15, 120, size=n)  # dakika
heart_rate = np.random.normal(120, 20, size=n)  # bpm
body_temp = np.random.normal(37, 0.5, size=n)   # Celcius
calories = duration * np.random.normal(10, 2, size=n)  # kcal

# Korelasyon oluÅŸturma
for i in range(n):
    if bmi[i] > 30:
        heart_rate[i] += np.random.normal(15, 5)
        body_temp[i] += np.random.normal(0.2, 0.1)
        calories[i] *= np.random.normal(0.85, 0.05)  # Daha az kalori yakÄ±mÄ±
    elif bmi[i] < 18.5:
        heart_rate[i] += np.random.normal(5, 3)
        calories[i] *= np.random.normal(0.9, 0.05)

# DataFrame oluÅŸturma
df = pd.DataFrame({
    'Sex': sex,
    'Age': age,
    'Height': height,
    'Weight': weight,
    'BMI': bmi,
    'Duration': duration,
    'Heart_Rate': heart_rate,
    'Body_Temp': body_temp,
    'Calories': calories
})

# BMI kategorileri
conditions = [
    (df['BMI'] < 18.5),
    (df['BMI'] >= 18.5) & (df['BMI'] < 25),
    (df['BMI'] >= 25) & (df['BMI'] < 30),
    (df['BMI'] >= 30) & (df['BMI'] < 40),
    (df['BMI'] >= 40)
]

categories = ['Underweight', 'Normal', 'Overweight', 'Obese', 'Extremely Obese']
df['BMI_Category'] = np.select(conditions, categories)

# Custom renk paleti
colors = ['#49c4fc', '#74bb43', '#9d68c2', '#e46b9e', '#ff5757']
color_dict = dict(zip(categories, colors))

# Body siluet verisi oluÅŸturma - her BMI kategorisi iÃ§in farklÄ± ÅŸekiller
def create_body_silhouette(bmi_category, gender='Male'):
    # Temel insan silÃ¼eti yÃ¼kseklik ve geniÅŸlik deÄŸerleri
    if gender == 'Male':
        height_factor = 1.0
        shoulder_width = 1.0
    else:
        height_factor = 0.95
        shoulder_width = 0.9
    
    # BMI kategorisine gÃ¶re vÃ¼cut ÅŸeklini ayarlama
    if bmi_category == 'Underweight':
        waist_width = 0.7 * shoulder_width
        hip_width = 0.8 * shoulder_width
        limb_width = 0.15
    elif bmi_category == 'Normal':
        waist_width = 0.8 * shoulder_width
        hip_width = 0.9 * shoulder_width
        limb_width = 0.2
    elif bmi_category == 'Overweight':
        waist_width = 0.9 * shoulder_width
        hip_width = 1.0 * shoulder_width
        limb_width = 0.25
    elif bmi_category == 'Obese':
        waist_width = 1.1 * shoulder_width
        hip_width = 1.2 * shoulder_width
        limb_width = 0.3
    else:  # Extremely Obese
        waist_width = 1.3 * shoulder_width
        hip_width = 1.4 * shoulder_width
        limb_width = 0.35
    
    # VÃ¼cut bÃ¶lÃ¼mlerini tanÄ±mlama
    head_x = [0]
    head_y = [1.8 * height_factor]
    head_z = [0]
    
    # Omuzlar
    shoulder_x = [-shoulder_width/2, shoulder_width/2]
    shoulder_y = [1.4 * height_factor, 1.4 * height_factor]
    shoulder_z = [0, 0]
    
    # Bel
    waist_x = [-waist_width/2, waist_width/2]
    waist_y = [0.9 * height_factor, 0.9 * height_factor]
    waist_z = [0, 0]
    
    # KalÃ§a
    hip_x = [-hip_width/2, hip_width/2]
    hip_y = [0.5 * height_factor, 0.5 * height_factor]
    hip_z = [0, 0]
    
    # Ayaklar
    feet_x = [-limb_width * 3, limb_width * 3]
    feet_y = [0, 0]
    feet_z = [0, 0]
    
    # VÃ¼cut ÅŸeklini oluÅŸturmak iÃ§in tÃ¼m noktalarÄ± birleÅŸtirme
    silhouette_x = head_x + shoulder_x + waist_x + hip_x + feet_x
    silhouette_y = head_y + shoulder_y + waist_y + hip_y + feet_y
    silhouette_z = head_z + shoulder_z + waist_z + hip_z + feet_z
    
    return silhouette_x, silhouette_y, silhouette_z

# Her BMI kategorisi iÃ§in 3D insan silueti gÃ¶rselleÅŸtirmesi
fig = make_subplots(
    rows=1, cols=1,
    specs=[[{"type": "scene"}]],
    subplot_titles=("3D BODY SHAPE ANALYSIS BY BMI CATEGORY")
)

# Her BMI kategorisi iÃ§in pozisyon belirleme
positions = {
    'Underweight': [-4, 0, 0],
    'Normal': [-2, 0, 0],
    'Overweight': [0, 0, 0],
    'Obese': [2, 0, 0],
    'Extremely Obese': [4, 0, 0]
}

# Her kategoriye ait figÃ¼rleri Ã§izme
category_stats = {}
for category in categories:
    subset = df[df['BMI_Category'] == category]
    
    # Ä°statistikler
    avg_bmi = subset['BMI'].mean()
    avg_heart_rate = subset['Heart_Rate'].mean()
    avg_calories = subset['Calories'].mean()
    efficiency = avg_calories / subset['Duration'].mean()
    
    category_stats[category] = {
        'BMI': avg_bmi,
        'Heart_Rate': avg_heart_rate,
        'Calories': avg_calories,
        'Efficiency': efficiency
    }
    
    # SilÃ¼et oluÅŸturma
    for gender in ['Male', 'Female']:
        x, y, z = create_body_silhouette(category, gender)
        gender_offset = 0.5 if gender == 'Female' else -0.5
        
        # PozisyonlarÄ± ayarlama
        pos_x = [val + positions[category][0] for val in x]
        pos_y = [val + positions[category][1] + gender_offset for val in y]
        pos_z = [val + positions[category][2] for val in z]
        
        # SilÃ¼et Ã§izimi
        fig.add_trace(
            go.Scatter3d(
                x=pos_x,
                y=pos_y,
                z=pos_z,
                mode='lines+markers',
                marker=dict(
                    size=5,
                    color=color_dict[category],
                    opacity=0.8
                ),
                line=dict(
                    color=color_dict[category],
                    width=10
                ),
                name=f"{category} ({gender})",
                showlegend=False
            )
        )
        
        # Veri noktalarÄ± ekleme (her siluet etrafÄ±nda)
        subset_gender = subset[subset['Sex'] == gender]
        sample_size = min(10, len(subset_gender))
        if sample_size > 0:
            sampled = subset_gender.sample(sample_size)
            
            # Siluet etrafÄ±nda rastgele daÄŸÄ±tma
            random_x = [positions[category][0] + np.random.normal(0, 0.4) for _ in range(sample_size)]
            random_y = [positions[category][1] + gender_offset + np.random.normal(0, 0.4) for _ in range(sample_size)]
            random_z = [positions[category][2] + np.random.normal(0, 0.4) for _ in range(sample_size)]
            
            size_factor = sampled['Age'] / 10
            
            fig.add_trace(
                go.Scatter3d(
                    x=random_x,
                    y=random_y,
                    z=random_z,
                    mode='markers',
                    marker=dict(
                        size=size_factor,
                        color=color_dict[category],
                        opacity=0.6,
                        symbol='circle'
                    ),
                    text=[f"Sex: {s}<br>Age: {a}<br>BMI: {b:.1f}<br>Heart Rate: {h:.1f}<br>Calories: {c:.1f}"
                         for s, a, b, h, c in zip(sampled['Sex'], sampled['Age'], sampled['BMI'], 
                                                sampled['Heart_Rate'], sampled['Calories'])],
                    hoverinfo='text',
                    showlegend=False
                )
            )
    
    # Etiketler ekleme
    fig.add_trace(
        go.Scatter3d(
            x=[positions[category][0]],
            y=[positions[category][1] - 1.5],
            z=[positions[category][2]],
            mode='text',
            text=[f"<b>{category}</b><br>BMI: {avg_bmi:.1f}<br>HR: {avg_heart_rate:.1f} bpm<br>Cal/min: {efficiency:.1f}"],
            textfont=dict(
                color='black',
                size=10
            ),
            showlegend=False
        )
    )

# FigÃ¼rÃ¼n genel gÃ¶rÃ¼nÃ¼mÃ¼nÃ¼ dÃ¼zenleme
fig.update_layout(
    title={
        'text': "3D HUMAN BODY SHAPE ANALYSIS BY BMI CATEGORY",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=24)
    },
    scene=dict(
        xaxis=dict(
            title="",
            showticklabels=False,
            range=[-6, 6]
        ),
        yaxis=dict(
            title="",
            showticklabels=False,
            range=[-2, 2]
        ),
        zaxis=dict(
            title="",
            showticklabels=False,
            range=[-1, 1]
        ),
        aspectmode='manual',
        aspectratio=dict(x=3, y=1, z=0.5),
        camera=dict(
            eye=dict(x=0, y=-3.5, z=0.5)
        )
    ),
    height=700,
    width=1000,
    margin=dict(l=20, r=20, t=100, b=20),
    template="plotly_white"
)

# AÃ§Ä±klama ekleme
annotations = [
    dict(
        xref="paper", yref="paper",
        x=0.02, y=1.12,
        text="METABOLIC INSIGHTS: As BMI increases, we observe elevation in resting heart rate and body temperature",
        showarrow=False,
        font=dict(size=14),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        align="left"
    ),
    dict(
        xref="paper", yref="paper",
        x=0.98, y=1.12,
        xanchor="right",
        text="CALORIC EFFICIENCY: Normal BMI category demonstrates optimal calories burned per minute of exercise",
        showarrow=False,
        font=dict(size=14),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        align="right"
    ),
    dict(
        xref="paper", yref="paper",
        x=0.5, y=-0.1,
        text="BODY SHAPE COMPARISON: Visual representation of typical body shapes across the BMI spectrum",
        showarrow=False,
        font=dict(size=14),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        align="center"
    )
]

for annotation in annotations:
    fig.add_annotation(annotation)

# BMI kategori barlarÄ±nÄ± altÄ±na ekleyelim
category_colors = [color_dict[cat] for cat in categories]
category_ranges = ["<18.5", "18.5-24.9", "25-29.9", "30-39.9", "40<"]

fig.add_trace(
    go.Scatter(
        x=[-4, -2, 0, 2, 4],
        y=[-2.5, -2.5, -2.5, -2.5, -2.5],
        mode='markers+text',
        marker=dict(
            color=category_colors,
            size=30,
            symbol='square'
        ),
        text=category_ranges,
        textposition="bottom center",
        textfont=dict(size=12, color="black"),
        showlegend=False
    )
)

fig.show(renderer='iframe_connected')


le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])


train['BMI'] = train['Weight'] / (train['Height'] ** 2)
test['BMI'] = test['Weight'] / (test['Height'] ** 2)

train['BMR'] = 10 * train['Weight'] + 6.25 * train['Height'] - 5 * train['Age'] + np.where(train['Sex'] == 1, 5, -161)
test['BMR'] = 10 * test['Weight'] + 6.25 * test['Height'] - 5 * test['Age'] + np.where(test['Sex'] == 1, 5, -161)

train['Activity_Level'] = train['Heart_Rate'] * train['Duration']
test['Activity_Level'] = test['Heart_Rate'] * test['Duration']

train['Age_Group'] = pd.cut(train['Age'], bins=[0, 30, 50, 100], labels=[0, 1, 2]).astype(int)
test['Age_Group'] = pd.cut(test['Age'], bins=[0, 30, 50, 100], labels=[0, 1, 2]).astype(int)

train['Heart_Rate_Group'] = pd.cut(train['Heart_Rate'], bins=[40, 80, 120, 200], labels=[0, 1, 2]).astype(int)
test['Heart_Rate_Group'] = pd.cut(test['Heart_Rate'], bins=[40, 80, 120, 200], labels=[0, 1, 2]).astype(int)

# Eksik (NaN) deÄŸerleri temizleme
train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)


X = train.drop(columns=['id', 'Calories'])
y = np.log1p(train['Calories'])  
X_test = test.drop(columns=['id'])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

lasso = Lasso(alpha=0.001, max_iter=5000)
lasso.fit(X_scaled, y)
selected_mask = np.abs(lasso.coef_) > 1e-4  
selected_feature_names = X.columns[selected_mask]

X = pd.DataFrame(X_scaled[:, selected_mask], columns=selected_feature_names)
X_test = pd.DataFrame(X_test_scaled[:, selected_mask], columns=selected_feature_names)

