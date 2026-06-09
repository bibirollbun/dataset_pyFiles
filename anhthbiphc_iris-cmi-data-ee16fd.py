import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import os
import polars as pl
import kaggle_evaluation.cmi_inference_server
from sklearn.metrics import accuracy_score, f1_score
import joblib
from scipy.spatial.transform import Rotation as R


# Táº£i táº­p dá»¯ liá»‡u
#Dá»¯ liá»‡u cáº£m biáº¿n cho táº­p huáº¥n luyá»‡n (Train Sensor Data): 
train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")

#ThÃ´ng tin nhÃ¢n kháº©u há»�c cá»§a Ä‘á»‘i tÆ°á»£ng trong táº­p huáº¥n luyá»‡n (giá»›i tÃ­nh, tuá»•i, ...): 
train_dem_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")

#Dá»¯ liá»‡u cáº£m biáº¿n cho táº­p kiá»ƒm tra:
test_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")

# ThÃ´ng tin nhÃ¢n kháº©u há»�c cá»§a Ä‘á»‘i tÆ°á»£ng trong táº­p kiá»ƒm tra:
test_dem_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")


train_df.sample(5)


train_dem_df.sample(5)


train_df.info()


train_dem_df.info()


train_df.describe().T


print("No. of rows : ", train_df.shape[0])
print("No. of columns : ", train_df.shape[1])


print("No. of rows : ", test_df.shape[0])
print("No. of columns : ", test_df.shape[1])


print("No. of rows : ", train_dem_df.shape[0])
print("No. of columns : ", train_dem_df.shape[1])


print("No. of rows : ", test_dem_df.shape[0])
print("No. of columns : ", test_dem_df.shape[1])


train_df.describe()


train_dem_df.describe()


test_df.describe()


test_dem_df.describe()


datasets = {
    "Train Data": train_df,
    "Train Demographics": train_dem_df,
    "Test Data": test_df,
    "Test Demographics": test_dem_df,
}

for name, df in datasets.items():
    print(f"Missing values in {name}:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(missing)
    else:
        print("  âœ… No missing values.")
    print()



#Kiá»ƒm tra trÃ¹ng láº·p (Duplicate Rows): .duplicated(): Tráº£ vá»� Series boolean, dÃ²ng nÃ o trÃ¹ng láº·p sáº½ lÃ  True.
#                                     .sum(): Ä�áº¿m tá»•ng sá»‘ dÃ²ng trÃ¹ng láº·p.
#Kiá»ƒm tra xem dá»¯ liá»‡u cÃ³ báº£n ghi nÃ o bá»‹ trÃ¹ng khÃ´ng â€” quan trá»�ng cho viá»‡c lÃ m sáº¡ch dá»¯ liá»‡u.

# Ä�áº¿m cÃ¡c hÃ ng trÃ¹ng láº·p trong train_df
train_duplicates = train_df.duplicated().sum()

# Ä�áº¿m cÃ¡c hÃ ng trÃ¹ng láº·p trong  test_df
test_duplicates = test_df.duplicated().sum()

# Ä�áº¿m cÃ¡c hÃ ng trÃ¹ng láº·p trong train_dem_df (optional)
train_dem_duplicates = train_dem_df.duplicated().sum()
# Ä�áº¿m cÃ¡c hÃ ng trÃ¹ng láº·p trong test_dem_df (optional)
test_dem_duplicates = test_dem_df.duplicated().sum()

# In sá»‘ lÆ°á»£ng dÃ²ng trÃ¹ng:
print(f"Number of duplicate rows in train_df: {train_duplicates}")
print(f"Number of duplicate rows in test_df: {test_duplicates}")
print(f"Number of duplicate rows in train_dem_df: {train_dem_duplicates}")
print(f"Number of duplicate rows in test_dem_df: {test_dem_duplicates}")


train_df.nunique().to_frame("# of unique values")


train_dem_df.nunique().to_frame("# of unique values")


gesture_counts = train_df['gesture'].value_counts()

# TÃ­nh toÃ¡n tá»· lá»‡ pháº§n trÄƒm
percentages = (gesture_counts / gesture_counts.sum()) * 100

# Táº¡o nhÃ£n bao gá»“m cáº£ sá»‘ lÆ°á»£ng vÃ  pháº§n trÄƒm
labels = [f'{label} ({count}, {perc:.1f}%)' for label, count, perc in zip(gesture_counts.index, gesture_counts.values, percentages)]

# Táº¡o biá»ƒu Ä‘á»“ trÃ²n
plt.figure(figsize=(8, 8)) # Set figure size for better readability
plt.pie(gesture_counts, labels=labels, startangle=160, counterclock=False, wedgeprops={'edgecolor': 'black'})
plt.title('Distribution of Gestures')
plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# Biá»ƒu Ä‘á»“ trÃ²n
fig, ax = plt.subplots(figsize=(8, 8))

# TÃ­nh pháº§n trÄƒm cho cÃ¡c nhÃ£n (tÃ¹y chá»�n, cÅ©ng cÃ³ thá»ƒ sá»­ dá»¥ng autopct)
orientation_counts = train_df["orientation"].value_counts().head()
total_count = orientation_counts.sum()
labels = [f'{label} ({count})' for label, count in orientation_counts.items()]

wedges, texts, autotexts = ax.pie(
    orientation_counts,
    autopct='%1.1f%%',
    startangle=90,
    colors=plt.cm.Paired.colors,
    pctdistance=0.85, # Distance of the percentage labels from the center
    wedgeprops=dict(width=0.3, edgecolor='w'), # Create the donut hole
    labels=labels
)

# Váº½ má»™t vÃ²ng trÃ²n á»Ÿ trung tÃ¢m Ä‘á»ƒ biáº¿n nÃ³ thÃ nh biá»ƒu Ä‘á»“ donut
centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)

ax.set_title("Subject's orientation (Donut Chart)")
ax.axis('equal')
plt.show()


# Biá»ƒu Ä‘á»“ cá»™t
fig, ax = plt.subplots(figsize=(8, 6))
orientation_counts.sort_values(ascending=False).plot( # Often good to sort descending for vertical bars
    kind="bar", color='purple', ax=ax, title="Target (Vertical Bar)"
)
ax.set_xlabel("Target")
ax.set_ylabel("Count")
plt.xticks(rotation=20, ha='right') # Rotate x-axis labels for better readability if they overlap
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()


# Biá»ƒu Ä‘á»“ trÃ²n
fig, ax = plt.subplots(figsize=(8, 8))

# TÃ­nh pháº§n trÄƒm cho cÃ¡c nhÃ£n (khÃ´ng báº¯t buá»™c, cÅ©ng cÃ³ thá»ƒ dÃ¹ng autopct)
sequence_type_counts = train_df["sequence_type"].value_counts().head()
total_count = sequence_type_counts.sum()
labels = [f'{label} ({count})' for label, count in sequence_type_counts.items()]

wedges, texts, autotexts = ax.pie(
    sequence_type_counts,
    autopct='%1.1f%%',
    startangle=90,
    colors=plt.cm.Paired.colors,
    pctdistance=0.85, # Distance of the percentage labels from the center
    wedgeprops=dict(width=0.3, edgecolor='w'), # Create the donut hole
    labels=labels
)

# Váº½ hÃ¬nh trÃ²n á»Ÿ giá»¯a Ä‘á»ƒ táº¡o biá»ƒu Ä‘á»“ vÃ nh khuyÃªn
centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)

ax.set_title("Target vs Non-target (Donut Chart)")
ax.axis('equal')
plt.show()


# Biá»ƒu Ä‘á»“ cá»™t dá»�c
fig, ax = plt.subplots(figsize=(8, 6))
sequence_type_counts.sort_values(ascending=False).plot( # Often good to sort descending for vertical bars
    kind="bar", color='lightgreen', ax=ax, title="Target vs Non Target (Vertical Bar)"
)
ax.set_xlabel("Target")
ax.set_ylabel("Count")
plt.xticks(rotation=20, ha='right') # Rotate x-axis labels for better readability if they overlap
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()


# Biá»ƒu Ä‘á»“ trÃ²n
fig, ax = plt.subplots(figsize=(8, 8))

# Váº½ hÃ¬nh trÃ²n á»Ÿ giá»¯a Ä‘á»ƒ táº¡o biá»ƒu Ä‘á»“ vÃ nh khuyÃªn
behavior_counts = train_df["behavior"].value_counts().head()
total_count = behavior_counts.sum()
labels = [f'{label} ({count})' for label, count in behavior_counts.items()]

wedges, texts, autotexts = ax.pie(
    behavior_counts,
    autopct='%1.1f%%',
    startangle=90,
    colors=plt.cm.Paired.colors,
    pctdistance=0.85, # Distance of the percentage labels from the center
    wedgeprops=dict(width=0.3, edgecolor='w'), # Create the donut hole
    labels=labels
)

centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)

ax.set_title("Subject's Behavior During the Sequence (Donut Chart)")
ax.axis('equal')
plt.show()


# Biá»ƒu Ä‘á»“ cá»™t dá»�c
fig, ax = plt.subplots(figsize=(8, 6))
behavior_counts.sort_values(ascending=False).plot( # Often good to sort descending for vertical bars
    kind="bar", color='skyblue', ax=ax, title="Subject's Behavior During the Sequence (Vertical Bar)"
)
ax.set_xlabel("Behavior")
ax.set_ylabel("Count")
plt.xticks(rotation=20, ha='right') # Rotate x-axis labels for better readability if they overlap
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()


# Ma tráº­n tÆ°Æ¡ng quan
plt.figure(figsize = (8, 4))
sns.heatmap(data = train_dem_df.corr(numeric_only = True), linewidth = 0.5, annot = True, linecolor = "black")
plt.title('Train Data Demographic Correlation Heatmap')
plt.show()


# Láº¥y sá»‘ lÆ°á»£ng tá»«ng giÃ¡ trá»‹ trong cá»™t giá»›i tÃ­nh
vc = train_dem_df['sex'].value_counts()

# TrÃ­ch xuáº¥t sá»‘ lÆ°á»£ng (giáº£ sá»­ cá»™t 'count' chÆ°a tá»“n táº¡i)
counts = vc.values

# Táº¡o nhÃ£n (giáº£ sá»­ cá»™t 'count' chÆ°a tá»“n táº¡i)
labels = ['Male', 'Female']  # Assuming 'Male' maps to '1' and 'Female' to '0'

# Váº½ biá»ƒu Ä‘á»“ hÃ¬nh trÃ²n
plt.pie(counts, labels=labels, autopct="%1.1f%%")  # Add percentages to pie slices
plt.title('Gender')
plt.show()


#train' lÃ  má»™t DataFrame (khung dá»¯ liá»‡u)
_, axs = plt.subplots(2, 1, sharex=True)

for sex in range(2):
  ax = axs.ravel()[sex]

  # Lá»�c theo giá»›i tÃ­nh báº±ng cÃ¡ch sá»­ dá»¥ng chá»‰ má»¥c boolean
  sex_filter = train_dem_df['sex'] == sex
  vc = train_dem_df[sex_filter]['age'].value_counts()  # Access column after filtering

  # Váº½ biá»ƒu Ä‘á»“ cá»™t
  ax.bar(vc.index, vc.values, color=['lightblue', 'coral'][sex], label=['Male', 'Female'][sex])
  ax.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax.set_ylabel('count')
  ax.legend()

plt.suptitle('Age distribution with respect to Gender')
axs.ravel()[1].set_xlabel('years')
plt.show()


# Khá»Ÿi táº¡o biá»ƒu Ä‘á»“ 3D
fig = plt.figure(figsize=(10, 10)) # Adjust figure size for better visibility
ax = fig.add_subplot(111, projection='3d')

# Láº¥y cÃ¡c giÃ¡ trá»‹ duy nháº¥t tá»« cá»™t 'gesture' Ä‘á»ƒ tÃ´ mÃ u
unique_gestures = train_df['gesture'].unique()

# Sá»­ dá»¥ng báº£n Ä‘á»“ mÃ u Ä‘á»ƒ cÃ³ Ä‘Æ°á»£c mÃ u sáº¯c riÃªng biá»‡t cho tá»«ng loáº¡i cá»­ chá»‰
# matplotlib.colormaps lÃ  cÃ¡ch Ä‘Æ°á»£c khuyáº¿n nghá»‹ Ä‘á»ƒ truy cáº­p báº£n Ä‘á»“ mÃ u hiá»‡n nay
colors = plt.colormaps['tab10'].resampled(len(unique_gestures)) # 'tab10' is good for categorical data

# Láº·p láº¡i cÃ¡c cá»­ chá»‰ Ä‘á»™c Ä‘Ã¡o vÃ  váº½ chÃºng báº±ng cÃ¡c mÃ u khÃ¡c nhau
for i, gesture_type in enumerate(unique_gestures):
    subset = train_df[train_df['gesture'] == gesture_type]
    ax.scatter(subset['acc_x'], subset['acc_y'], subset['acc_z'],
               color=colors(i),    # Chá»‰ Ä‘á»‹nh má»™t mÃ u duy nháº¥t tá»« báº£n Ä‘á»“ mÃ u
               label=gesture_type, # NhÃ£n cho huyá»�n thoáº¡i
               alpha=0.7,          # Ä�á»™ má»� Ä‘á»¥c (phÃ¹ há»£p vá»›i Ä‘á»™ má»� Ä‘á»¥c Plotly cá»§a báº¡n)
               s=2,
               rasterized=True)                # KÃ­ch thÆ°á»›c cá»§a cÃ¡c Ä‘iá»ƒm (tÆ°Æ¡ng tá»± nhÆ° size_max=1 cá»§a Plotly, Ä‘Æ°á»£c Ä‘iá»�u chá»‰nh Ä‘á»ƒ dá»… nhÃ¬n)

# Ä�áº·t nhÃ£n vÃ  tiÃªu Ä‘á»�
ax.set_xlabel('acc_x', fontsize=14)
ax.set_ylabel('acc_y', fontsize=14)
ax.set_zlabel('acc_z', fontsize=14)
ax.set_title('Linear acceleration along X, Y, Z axes', fontsize=16)

# ThÃªm chÃº thÃ­ch Ä‘á»ƒ phÃ¢n biá»‡t cÃ¡c loáº¡i 'cá»­ chá»‰'
ax.legend(title='Gesture Type', fontsize=10, title_fontsize=12, loc='upper right')

# Ä�áº·t kÃ­ch thÆ°á»›c phÃ´ng chá»¯ nhÃ£n Ä‘Ã¡nh dáº¥u
ax.tick_params(axis='both', which='major', labelsize=12)

# --- MÃ´ phá»�ng máº«u 'plotly_dark' cá»§a Plotly ---
# Ä�áº·t mÃ u ná»�n cá»§a Ã´
ax.set_facecolor('black')

# LÃ m cho ná»�n cá»§a khung trong suá»‘t
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Ä�áº·t mÃ u cáº¡nh khung
ax.xaxis.pane.edgecolor = 'grey'
ax.yaxis.pane.edgecolor = 'grey'
ax.zaxis.pane.edgecolor = 'grey'

# Ä�áº·t mÃ u máº·t khung thá»±c táº¿ thÃ nh trong suá»‘t
ax.xaxis.pane.set_facecolor((0.0, 0.0, 0.0, 0.0))
ax.yaxis.pane.set_facecolor((0.0, 0.0, 0.0, 0.0))
ax.zaxis.pane.set_facecolor((0.0, 0.0, 0.0, 0.0))

# Ä�áº·t mÃ u Ä‘Æ°á»�ng lÆ°á»›i
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()


# Khá»Ÿi táº¡o biá»ƒu Ä‘á»“ 3D
fig = plt.figure(figsize=(10, 10)) # Ä�iá»�u chá»‰nh kÃ­ch thÆ°á»›c hÃ¬nh Ä‘á»ƒ dá»… nhÃ¬n hÆ¡n
ax = fig.add_subplot(111, projection='3d')

# Láº¥y cÃ¡c giÃ¡ trá»‹ duy nháº¥t tá»« 'sequence_type' Ä‘á»ƒ tÃ´ mÃ u
unique_sequence_types = train_df['sequence_type'].unique()

# --- Xá»¬ LÃ� Cáº¢NH BÃ�O Vá»€ VIá»†C KHÃ”NG Sá»¬ Dá»¤NG Ná»®A 1: get_cmap ---
# Sá»­ dá»¥ng matplotlib.colormaps[name] thay vÃ¬ plt.cm.get_cmap(name)
colors = plt.colormaps['viridis'].resampled(len(unique_sequence_types))

# Láº·p láº¡i qua cÃ¡c loáº¡i trÃ¬nh tá»± duy nháº¥t vÃ  biá»ƒu diá»…n chÃºng báº±ng cÃ¡c mÃ u khÃ¡c nhau
for i, seq_type in enumerate(unique_sequence_types):
    subset = train_df[train_df['sequence_type'] == seq_type]
    ax.scatter(subset['acc_x'], subset['acc_y'], subset['acc_z'],
               color=colors(i), # GÃ¡n má»™t mÃ u duy nháº¥t
               label=seq_type,  # NhÃ£n cho chÃº giáº£i
               alpha=0.7,       # Ä�á»™ má»� Ä‘á»¥c
               s=10)            # KÃ­ch thÆ°á»›c cá»§a cÃ¡c Ä‘iá»ƒm (tÆ°Æ¡ng tá»± nhÆ° size_max cá»§a Plotly)

# Ä�áº·t nhÃ£n vÃ  tiÃªu Ä‘á»�
ax.set_xlabel('acc_x', fontsize=14)
ax.set_ylabel('acc_y', fontsize=14)
ax.set_zlabel('acc_z', fontsize=14)
ax.set_title('Linear acceleration along X, Y, Z axes', fontsize=16)

# ThÃªm chÃº thÃ­ch Ä‘á»ƒ phÃ¢n biá»‡t 'sequence_type'
ax.legend(title='Sequence Type', fontsize=12, title_fontsize=14)

# Ä�áº·t kÃ­ch thÆ°á»›c phÃ´ng chá»¯ nhÃ£n Ä‘Ã¡nh dáº¥u
ax.tick_params(axis='both', which='major', labelsize=12)

# Ä�áº·t mÃ u ná»�n tá»‘i vÃ  Ä‘iá»�u chá»‰nh mÃ u lÆ°á»›i/khung
ax.set_facecolor('black')

# --- Xá»¬ LÃ� Cáº¢NH BÃ�O Sá»¬A Ä�á»”I 2, 3, 4: w_xaxis, w_yaxis, w_zaxis ---
# Sá»­ dá»¥ng .xaxis, .yaxis, .zaxis trá»±c tiáº¿p cho cÃ¡c ngÄƒn
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.xaxis.pane.edgecolor = 'grey'
ax.yaxis.pane.edgecolor = 'grey'
ax.zaxis.pane.edgecolor = 'grey'

# Ä�áº·t mÃ u khung thÃ nh trong suá»‘t Ä‘á»ƒ cÃ³ hiá»‡u á»©ng tá»‘i thá»±c sá»±
# Giá»� Ä‘Ã¢y, báº¡n cÃ³ thá»ƒ truy cáº­p trá»±c tiáº¿p thÃ´ng qua .xaxis.pane.set_facecolor
ax.xaxis.pane.set_facecolor((0.0, 0.0, 0.0, 0.0))
ax.yaxis.pane.set_facecolor((0.0, 0.0, 0.0, 0.0))
ax.zaxis.pane.set_facecolor((0.0, 0.0, 0.0, 0.0))

ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()


fig = plt.figure(figsize=(14, 8)) # Adjust figure size as needed
ax = fig.add_subplot(111, projection='3d')

# Váº½ Ä‘Æ°á»�ng 3D
# Vá»� mÃ u sáº¯c, plot3D cá»§a matplotlib khÃ´ng há»— trá»£ trá»±c tiáº¿p Ä‘á»‘i sá»‘ 'color' liÃªn tá»¥c
# nhÆ° px.line_3d. ThÃ´ng thÆ°á»�ng, báº¡n sáº½ sá»­ dá»¥ng má»™t báº£n Ä‘á»“ mÃ u vÃ  Ã¡nh xáº¡ 'rot_w' vÃ o Ä‘Ã³.
# VÃ­ dá»¥ nÃ y sá»­ dá»¥ng má»™t mÃ u Ä‘Æ°á»�ng Ä‘Æ¡n giáº£n. Ä�á»‘i vá»›i mÃ u liÃªn tá»¥c, báº¡n cáº§n
# phÃ¢n Ä‘oáº¡n dá»¯ liá»‡u hoáº·c sá»­ dá»¥ng scatter3D vá»›i c=train['rot_w'].
# ax.plot(train['rot_x'], train['rot_y'], train['rot_z'], color='skyblue', linewidth=0.5)

# Náº¿u báº¡n muá»‘n tÃ´ mÃ u theo 'rot_w' vá»›i gradient, thÆ°á»�ng tá»‘t hÆ¡n nÃªn sá»­ dá»¥ng scatter3D
# náº¿u khÃ´ng nháº¥t thiáº¿t pháº£i dÃ¹ng Ä‘Æ°á»�ng liÃªn tá»¥c, hoáº·c phÃ¢n Ä‘oáº¡n Ä‘Æ°á»�ng.
# Ä�á»‘i vá»›i biá»ƒu Ä‘á»“ Ä‘Æ°á»�ng tháº³ng thá»±c sá»± vá»›i mÃ u liÃªn tá»¥c, viá»‡c nÃ y phá»©c táº¡p hÆ¡n trong matplotlib.
# VÃ­ dá»¥ vá»� scatter3D vá»›i mÃ u:
sc = ax.scatter(train_df['rot_x'], train_df['rot_y'], train_df['rot_z'], c=train_df['rot_w'], cmap='viridis', s=1)
fig.colorbar(sc, ax=ax, label='rot_w')


ax.set_xlabel('rot_x')
ax.set_ylabel('rot_y')
ax.set_zlabel('rot_z')
ax.set_title('3D Path of Gyroscope Rotation Over Time')

# Ä�iá»�u chá»‰nh kÃ­ch thÆ°á»›c phÃ´ng chá»¯ náº¿u cáº§n (tÆ°Æ¡ng tá»± nhÆ° update_layout trong Plotly)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.xaxis.label.set_size(14)
ax.yaxis.label.set_size(14)
ax.zaxis.label.set_size(14)
ax.title.set_size(16)

plt.show()


# LÃ m tan cháº£y DataFrame Ä‘á»ƒ váº½ Ä‘á»“ thá»‹ dá»… dÃ ng hÆ¡n
df_melted = train_df.melt(id_vars='sequence_type', 
                    value_vars=['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5'],
                    var_name='sensor', 
                    value_name='temperature')

# Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='sequence_type', y='temperature', hue='sensor', data=df_melted)
plt.title('Thermopile Sensor Readings by Sequence Type')
plt.xlabel('Sequence Type')
plt.ylabel('Temperature (Â°C)')
plt.legend(title='Sensor')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# NhÃ³m theo sequence_type vÃ  láº¥y giÃ¡ trá»‹ trung bÃ¬nh
import numpy as np
df_grouped = train_df.groupby('sequence_type')[['thm_1','thm_2','thm_3','thm_4','thm_5']].mean()

categories = ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']
N = len(categories)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Close the loop

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

for idx, row in df_grouped.iterrows():
    values = row.values.tolist()
    values += values[:1]  # Close the loop
    ax.plot(angles, values, marker='o', label=idx)
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
plt.legend(title="Sequence Type")
plt.title("Radar Chart of Average Thermopile Readings by Sequence Type")
plt.show()


# LÃ m tan cháº£y DataFrame Ä‘á»ƒ váº½ Ä‘á»“ thá»‹ dá»… dÃ ng hÆ¡n
df_melted = train_df.melt(id_vars='gesture', 
                    value_vars=['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5'],
                    var_name='sensor', 
                    value_name='temperature')

# Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='gesture', y='temperature', hue='sensor', data=df_melted)
plt.title('Thermopile Sensor Readings by Gesture')
plt.xlabel('Gesture')
plt.ylabel('Temperature (Â°C)')
plt.legend(title='Sensor')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# LÃ m tan cháº£y DataFrame Ä‘á»ƒ váº½ Ä‘á»“ thá»‹ dá»… dÃ ng hÆ¡n
df_melted = train_df.melt(id_vars='behavior', 
                    value_vars=['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5'],
                    var_name='sensor', 
                    value_name='temperature')

# Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='behavior', y='temperature', hue='sensor', data=df_melted)
plt.title('Thermopile Sensor Readings by Behavior')
plt.xlabel('Behavior')
plt.ylabel('Temperature (Â°C)')
plt.legend(title='Sensor')
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()


def null_percent(df):
    per=((df.isnull().sum()/len(df))*100).round(2)
    return per[per>0]

print("Nan Values in Train data")
print(null_percent(train_df))


#Tá»•ng há»£p dá»¯ liá»‡u vÃ o dictionary: Táº¡o dictionary datasets chá»©a 4 bá»™ dá»¯ liá»‡u chÃ­nh, giÃºp dá»… quáº£n lÃ½ vÃ  xá»­ lÃ½ láº·p Ä‘i láº·p láº¡i.
datasets = {
    "Train Data": train_df,
    "Train Demographics": train_dem_df,
    "Test Data": test_df,
    "Test Demographics": test_dem_df,
}

# Print shapes
#Giáº£i thÃ­ch: .items(): Duyá»‡t qua tá»«ng cáº·p tÃªn vÃ  DataFrame.
#            .shape: Tráº£ vá»� tuple (sá»‘ dÃ²ng, sá»‘ cá»™t).
#            print(...): In ra tÃªn dataset, sá»‘ dÃ²ng vÃ  sá»‘ cá»™t, giÃºp báº¡n hÃ¬nh dung dá»¯ liá»‡u cÃ³ bao nhiÃªu báº£n ghi vÃ  thuá»™c tÃ­nh.


for name, df in datasets.items():
    num_rows, num_cols = df.shape
    print(f"{name}:")
    print(f"  Number of Rows: {num_rows}")
    print(f"  Number of Columns: {num_cols}\n")


# --- 2.4 Há»£p nháº¥t dá»¯ liá»‡u nhÃ¢n kháº©u há»�c vÃ  cáº£m biáº¿n ---
merged_train = pd.merge(train_df, train_dem_df, on="subject", how="left")
merged_test = pd.merge(test_df, test_dem_df, on="subject", how="left")

print("Merged Train Shape:", merged_train.shape)
print("Merged Test Shape:", merged_test.shape)
display(merged_train.head(2))



# --- Thá»‘ng kÃª mÃ´ táº£ toÃ n bá»™ merged_train ---
merged_train.describe().T



# --- 2.5 PhÃ¢n phá»‘i hÃ nh vi ---
sns.countplot(data=merged_train, x='behavior', order=merged_train['behavior'].value_counts().index)
plt.title("PhÃ¢n phá»‘i cÃ¡c hÃ nh vi trong táº­p huáº¥n luyá»‡n")
plt.xticks(rotation=45)
plt.show()



# --- 2.12 Behavior theo phase & sequence_type ---
sns.countplot(data=merged_train, x='behavior', hue='phase')
plt.title("Behavior theo Phase")
plt.xticks(rotation=45)
plt.show()

sns.countplot(data=merged_train, x='behavior', hue='sequence_type')
plt.title("Behavior theo Sequence Type")
plt.xticks(rotation=45)
plt.show()



# --- 2.10 PhÃ¡t hiá»‡n Outlier trong Ä‘á»™ tuá»•i ---
plt.figure(figsize=(6, 3))
sns.boxplot(x=merged_train['age'])
plt.title("Boxplot - Ä�á»™ tuá»•i")
plt.show()

# Lá»�c náº¿u cáº§n
merged_train = merged_train[merged_train['age'] <= 100]



# --- 2.6 Biá»ƒu Ä‘á»“ histogram cÃ¡c Ä‘áº·c trÆ°ng cáº£m biáº¿n acc_ ---
acc_cols = [col for col in merged_train.columns if col.startswith('acc_')]
merged_train[acc_cols].hist(figsize=(16, 12), bins=30)
plt.suptitle("PhÃ¢n phá»‘i cÃ¡c Ä‘áº·c trÆ°ng gia tá»‘c (acc_)", fontsize=16)
plt.tight_layout()
plt.show()



# --- 2.7 Ma tráº­n tÆ°Æ¡ng quan acc_ ---
plt.figure(figsize=(12, 10))
sns.heatmap(merged_train[acc_cols].corr(), cmap='coolwarm', center=0)
plt.title("TÆ°Æ¡ng quan giá»¯a cÃ¡c Ä‘áº·c trÆ°ng gia tá»‘c (acc_)")
plt.show()



# --- 2.8.1 PhÃ¢n phá»‘i hÃ nh vi theo giá»›i tÃ­nh ---
sns.countplot(data=merged_train, x='behavior', hue='sex', order=merged_train['behavior'].value_counts().index)
plt.title("PhÃ¢n phá»‘i hÃ nh vi theo giá»›i tÃ­nh")
plt.xticks(rotation=45)
plt.show()

# --- 2.8.2 PhÃ¢n phá»‘i tuá»•i theo hÃ nh vi ---
sns.boxplot(data=merged_train, x='behavior', y='age')
plt.title("Tuá»•i trung bÃ¬nh theo hÃ nh vi")
plt.xticks(rotation=45)
plt.show()



#Hiá»ƒn thá»‹ dá»¯ liá»‡u máº«u: .head(2): Xem trÆ°á»›c 2 dÃ²ng Ä‘áº§u cá»§a má»—i báº£ng.
#                      display(...): DÃ¹ng trong notebook Ä‘á»ƒ hiá»ƒn thá»‹ báº£ng dá»¯ liá»‡u Ä‘áº¹p, rÃµ rÃ ng.

display(train_df.head(2))
display(train_dem_df.head(2))


display(test_df.head(2))
display(test_dem_df.head(2))


#Chuáº©n bá»‹ hÃ m thá»‘ng kÃª mÃ´ táº£ cÃ³ lá»�c cá»™t: 
# excluded_prefixes: Bá»™ tiá»�n tá»‘ Ä‘á»ƒ loáº¡i cÃ¡c cá»™t cáº£m biáº¿n nhÆ°: 'acc_': Gia tá»‘c káº¿, 'rot_': Cáº£m biáº¿n xoay, 'thm_': Nhiá»‡t Ä‘á»™/Ä�á»™ áº©m. 'tof_': Time-of-Flight (cáº£m biáº¿n khoáº£ng cÃ¡ch).

# Define excluded prefixes
excluded_prefixes = ('acc_', 'rot_', 'thm_', 'tof_')

#HÃ m filtered_describe: Lá»�c cÃ¡c cá»™t khÃ´ng thuá»™c cáº£m biáº¿n vÃ  cÃ³ kiá»ƒu sá»‘.
#                       DÃ¹ng .describe().T Ä‘á»ƒ thá»‘ng kÃª mÃ´ táº£ (mean, std, min, max,...) vÃ  chuyá»ƒn vá»‹ báº£ng cho dá»… nhÃ¬n.
#                       display: Hiá»ƒn thá»‹ báº£ng thá»‘ng kÃª.

def filtered_describe(df, name):
    # Exclude sensor columns
    filtered_cols = [col for col in df.columns 
                     if not col.startswith(excluded_prefixes) and pd.api.types.is_numeric_dtype(df[col])]
    
    # Describe and style
    print(f'\nâ�¡ï¸� Description of numerical columns in {name}')
    return df[filtered_cols].describe().T.style.background_gradient(cmap='viridis')

# Only analyze train sets!
display(filtered_describe(train_df, "train_df"))
display(filtered_describe(train_dem_df, "train_dem_df"))


#Kiá»ƒm tra thiáº¿u giÃ¡ trá»‹ & thá»‘ng kÃª cáº£m biáº¿n: 
#Danh sÃ¡ch cÃ¡c cá»™t khÃ´ng thuá»™c cáº£m biáº¿n (cÃ³ thá»ƒ lÃ  ID, thÃ´ng tin khÃ¡c). 

excluded_prefixes = ('acc_', 'rot_', 'thm_', 'tof_')
sensor_cols = [col for col in train_df.columns if not col.startswith(excluded_prefixes)]

# Sensor Data Summary for TRAIN
#isnull().sum(): Ä�áº¿m sá»‘ giÃ¡ trá»‹ bá»‹ thiáº¿u.
missing_sensor_train = pd.DataFrame({
    'Feature': sensor_cols,
    '[TRAIN] Missing Count': train_df[sensor_cols].isnull().sum().values,
    '[TRAIN] Missing %': (train_df[sensor_cols].isnull().sum().values / len(train_df)) * 100
})

#nunique(): Ä�áº¿m sá»‘ lÆ°á»£ng giÃ¡ trá»‹ duy nháº¥t.
unique_sensor_train = pd.DataFrame({
    'Feature': sensor_cols,
    'Unique Values [TRAIN]': train_df[sensor_cols].nunique().values
})

#dtypes: Láº¥y kiá»ƒu dá»¯ liá»‡u cá»§a tá»«ng cá»™t.
dtypes_sensor = pd.DataFrame({
    'Feature': sensor_cols,
    'Data Type': train_df[sensor_cols].dtypes.values
})

# Merge all summaries (NO test set)
#merge: Gá»™p cÃ¡c báº£ng thá»‘ng kÃª thÃ nh báº£ng duy nháº¥t theo Feature.
sensor_summary = missing_sensor_train \
    .merge(unique_sensor_train, on='Feature', how='left') \
    .merge(dtypes_sensor, on='Feature', how='left')

# Display styled DataFrame (mask NaNs just for styling)
#fillna(0): Ä�iá»�n giÃ¡ trá»‹ thiáº¿u báº±ng 0 (cho Ä‘áº¹p máº¯t khi hiá»ƒn thá»‹).
#.style.background_gradient: TÃ´ mÃ u ná»�n theo giÃ¡ trá»‹ giÃºp dá»… nhÃ¬n.
styled_df = sensor_summary.fillna(0)
styled_df.style.background_gradient(cmap='viridis')


#Thá»‘ng kÃª tÆ°Æ¡ng tá»± cho nhÃ¢n kháº©u há»�c: TÆ°Æ¡ng tá»± nhÆ° pháº§n thá»‘ng kÃª cáº£m biáº¿n nhÆ°ng Ã¡p dá»¥ng cho dá»¯ liá»‡u nhÃ¢n kháº©u há»�c.

# Cá»™t nhÃ¢n kháº©u há»�c (khÃ´ng loáº¡i trá»«)
dem_cols = train_dem_df.columns

# GiÃ¡ trá»‹ bá»‹ thiáº¿u trong dá»¯ liá»‡u nhÃ¢n kháº©u há»�c cá»§a train 
missing_demo_train = pd.DataFrame({
    'Feature': dem_cols,
    '[TRAIN DEMO] Missing Count': train_dem_df[dem_cols].isnull().sum().values,
    '[TRAIN DEMO] Missing %': (train_dem_df[dem_cols].isnull().sum().values / len(train_dem_df)) * 100
})

# GiÃ¡ trá»‹ duy nháº¥t Ä‘Æ°á»£c tÃ­nh trong dá»¯ liá»‡u nhÃ¢n kháº©u há»�c cá»§a train 
unique_demo_train = pd.DataFrame({
    'Feature': dem_cols,
    'Unique Values [TRAIN DEMO]': train_dem_df[dem_cols].nunique().values
})

# Data types
dtypes_demo = pd.DataFrame({
    'Feature': dem_cols,
    'Data Type': train_dem_df[dem_cols].dtypes.values
})

# TÃ³m táº¯t há»£p nháº¥t (chá»‰ dÃ nh cho Ä‘Ã o táº¡o)
demo_summary = (
    missing_demo_train
    .merge(unique_demo_train, on='Feature', how='left')
    .merge(dtypes_demo, on='Feature', how='left')
)

# Hiá»ƒn thá»‹ tÃ³m táº¯t theo phong cÃ¡ch
demo_summary.style.background_gradient(cmap='viridis')


import pandas as pd

# NhÃ³m cá»™t cáº£m biáº¿n (chá»‰ sá»­ dá»¥ng train_df)
# PhÃ¢n nhÃ³m cÃ¡c biáº¿n cáº£m biáº¿n:
acc_cols = [col for col in train_df.columns if col.startswith('acc_')]
rot_cols = [col for col in train_df.columns if col.startswith('rot_')]
thm_cols = [col for col in train_df.columns if col.startswith('thm_')]
tof_cols = [col for col in train_df.columns if col.startswith('tof_')]

# HÃ m trá»£ giÃºp cho sá»‘ liá»‡u thá»‘ng kÃª tÃ³m táº¯t (chá»‰ dÃ nh cho Ä‘Ã o táº¡o)
#HÃ m thá»‘ng kÃª chi tiáº¿t tá»«ng nhÃ³m cáº£m biáº¿n (Tá»· lá»‡ thiáº¿u giÃ¡ trá»‹; GiÃ¡ trá»‹ nhá»� nháº¥t, lá»›n nháº¥t; Trung bÃ¬nh, Ä‘á»™ lá»‡ch chuáº©n.)
def sensor_summary(df, cols, name, dataset_name):
    summary = pd.DataFrame({
        'Feature': cols,
        f'{dataset_name} Missing %': df[cols].isnull().mean().values * 100,
        f'{dataset_name} Min': df[cols].min().values,
        f'{dataset_name} Max': df[cols].max().values,
        f'{dataset_name} Mean': df[cols].mean().values,
        f'{dataset_name} Std': df[cols].std().values
    })
    summary.insert(0, 'Sensor', name)
    return summary

# TÃ­nh toÃ¡n tÃ³m táº¯t quÃ¡ trÃ¬nh Ä‘Ã o táº¡o cho tá»«ng loáº¡i cáº£m biáº¿n
#Thá»‘ng kÃª tá»•ng há»£p cho táº­p train: Duyá»‡t tá»«ng nhÃ³m cáº£m biáº¿n vÃ  gom káº¿t quáº£ thá»‘ng kÃª láº¡i.
# pd.concat: GhÃ©p báº£ng láº¡i thÃ nh báº£ng thá»‘ng kÃª tá»•ng há»£p.
def train_sensor_summary(train_df, sensor_cols_dict):
    all_train = []
    for sensor_name, cols in sensor_cols_dict.items():
        all_train.append(sensor_summary(train_df, cols, sensor_name, 'Train'))
    train_summary = pd.concat(all_train, ignore_index=True)
    return train_summary

# NhÃ³m cáº£m biáº¿n Ä‘á»ƒ xá»­ lÃ½
sensor_cols_dict = {
    'acc': acc_cols,
    'rot': rot_cols,
    'thm': thm_cols,
    'tof': tof_cols
}

# Chá»‰ tÃ³m táº¯t vá»� chuyáº¿n tÃ u cháº¡y
sensor_train_summary = train_sensor_summary(train_df, sensor_cols_dict)

# Trung bÃ¬nh theo nhÃ³m cáº£m biáº¿n: 
#                             .groupby('Sensor'): Gom nhÃ³m theo loáº¡i cáº£m biáº¿n.
#                             .mean(numeric_only=True): Trung bÃ¬nh cÃ¡c chá»‰ sá»‘ sá»‘ há»�c.
#                             TÃ´ mÃ u báº£ng káº¿t quáº£ giÃºp dá»… phÃ¢n tÃ­ch.

# Hiá»ƒn thá»‹ má»©c trung bÃ¬nh cáº¥p cáº£m biáº¿n cho tÃ u há»�a
summary_by_group = sensor_train_summary.groupby('Sensor').mean(numeric_only=True)

# Hiá»ƒn thá»‹ báº£ng theo kiá»ƒu
summary_by_group.style.background_gradient(cmap='viridis')


#áº¨n cáº£nh bÃ¡o khÃ´ng cáº§n thiáº¿t:  áº¨n cáº£nh bÃ¡o tÆ°Æ¡ng lai cá»§a thÆ° viá»‡n, áº¨n cáº£nh bÃ¡o vá»� kÃ½ tá»± bá»‹ thiáº¿u font khi váº½ biá»ƒu Ä‘á»“.

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="Glyph.*missing from current font")

custom_palette = ['#3498db', '#e74c3c', '#2ecc71']
sensor_prefixes = ('acc_', 'rot_', 'thm_', 'tof_')
categorical_columns = ['phase', 'behavior', 'orientation', 'sequence_type',
                       'adult_child', 'sex', 'handedness']
target_columns = ['gesture']
categorical_tracker = categorical_columns.copy()

def get_numerical_variables(df, excluded_prefixes, excluded_columns):
    return [col for col in df.columns 
            if pd.api.types.is_numeric_dtype(df[col])
            and not col.startswith(excluded_prefixes)
            and col not in excluded_columns]

# ONLY TRAIN SETS!
train_main = train_df.copy()
train_demo = train_dem_df.copy()

train_main['Dataset'] = 'Train'
train_demo['Dataset'] = 'Train'

main_data = train_main
demo_data = train_demo

main_numeric_vars = get_numerical_variables(main_data, sensor_prefixes, categorical_columns)
demo_numeric_vars = get_numerical_variables(demo_data, (), categorical_columns)

def create_variable_plots(variable, dataset_label, data):
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Box plot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=data, x=variable, y="Dataset", palette=custom_palette)
    plt.xlabel(variable)
    plt.title(f" Box Plot for {variable} â€” {dataset_label}")

    # Histogram
    plt.subplot(1, 2, 2)
    for label, color in zip(data['Dataset'].unique(), custom_palette):
        subset = data[data['Dataset'] == label]
        sns.histplot(data=subset, x=variable, kde=True, bins=30, label=label, color=color)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title(f" Histogram for {variable} â€” {dataset_label}")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Chá»‰ váº½ cÃ¡c tÃ­nh nÄƒng bá»‘i cáº£nh cá»§a bá»™ Ä‘Ã o táº¡o
for var in main_numeric_vars:
    create_variable_plots(var, "Context Features", main_data)

# Chá»‰ váº½ cÃ¡c Ä‘áº·c Ä‘iá»ƒm nhÃ¢n kháº©u há»�c cá»§a bá»™ huáº¥n luyá»‡n
for var in demo_numeric_vars:
    create_variable_plots(var, "Demographic Features", demo_data)

# --- Cleanup
train_main.drop('Dataset', axis=1, inplace=True)
train_demo.drop('Dataset', axis=1, inplace=True)


import numpy as np
import pandas as pd

# 1) Sao chÃ©p Ä‘Ã o táº¡o vÃ  kiá»ƒm tra Ä‘á»ƒ chÃºng ta khÃ´ng sá»­a Ä‘á»•i DataFrame gá»‘c
train_temp = train_df.copy()
test_temp  = test_df.copy()

# 2) GIA Tá»�C Káº¾: tÃ­nh toÃ¡n Ä‘á»™ lá»›n táº¡i má»—i dáº¥u thá»�i gian
train_temp['acc_mag'] = np.sqrt(
    train_temp['acc_x']**2 + train_temp['acc_y']**2 + train_temp['acc_z']**2
)
test_temp['acc_mag'] = np.sqrt(
    test_temp['acc_x']**2 + test_temp['acc_y']**2 + test_temp['acc_z']**2
)

# 3) ROTATION: tÃ­nh toÃ¡n â€œgÃ³c quayâ€� tá»« thÃ nh pháº§n quaternion w
# (LÆ°u Ã½: rot_w náº±m trong [-1,1], do Ä‘Ã³ arccos há»£p lá»‡. ChÃºng tÃ´i bá»� qua NaN náº¿u cÃ³.)
train_temp['rot_angle'] = 2 * np.arccos(train_temp['rot_w'].clip(-1,1))
test_temp['rot_angle']  = 2 * np.arccos(test_temp['rot_w'].clip(-1,1))

# 4) NhÃ³m theo sequence_id vÃ  tá»•ng há»£p cÃ¡c tÃ³m táº¯t gia tá»‘c káº¿
acc_agg_funcs = {
    'acc_mag': ['mean', 'std', 'max']
}
train_acc_summary = train_temp.groupby('sequence_id').agg(acc_agg_funcs)
test_acc_summary  = test_temp.groupby('sequence_id').agg(acc_agg_funcs)

# LÃ m pháº³ng cá»™t MultiIndex
train_acc_summary.columns = ['acc_mag_' + stat for stat in ['mean', 'std', 'max']]
test_acc_summary.columns  = ['acc_mag_' + stat for stat in ['mean', 'std', 'max']]

# 5) NhÃ³m theo sequence_id vÃ  tá»•ng há»£p tÃ³m táº¯t vÃ²ng quay
rot_agg_funcs = {
    'rot_angle': ['mean', 'std', 'max']
}
train_rot_summary = train_temp.groupby('sequence_id').agg(rot_agg_funcs)
test_rot_summary  = test_temp.groupby('sequence_id').agg(rot_agg_funcs)

train_rot_summary.columns = ['rot_angle_' + stat for stat in ['mean', 'std', 'max']]
test_rot_summary.columns  = ['rot_angle_' + stat for stat in ['mean', 'std', 'max']]

# 6) NHIá»†T Ä�á»˜: nÄƒm cáº£m biáº¿n thm_1 â€¦ thm_5
thm_cols = [f"thm_{i}" for i in range(1, 6)]

# XÃ¡c Ä‘á»‹nh hÃ m tá»•ng há»£p: trung bÃ¬nh + Ä‘á»™ lá»‡ch chuáº©n
thm_agg_funcs = {col: ['mean', 'std'] for col in thm_cols}

train_thm_summary = train_temp.groupby('sequence_id').agg(thm_agg_funcs)
test_thm_summary  = test_temp.groupby('sequence_id').agg(thm_agg_funcs)

# LÃ m pháº³ng cÃ¡c cá»™t MultiIndex
flattened_thm_cols = []
for sensor in thm_cols:
    for stat in ['mean','std']:
        flattened_thm_cols.append(f"{sensor}_{stat}")

train_thm_summary.columns = flattened_thm_cols
test_thm_summary.columns  = flattened_thm_cols

# 7) THá»œI GIAN BAY: má»—i cáº£m biáº¿n i cÃ³ 64 cá»™t pixel: tof_i_v0 â€¦ tof_i_v63
# ChÃºng ta sáº½ táº¡o má»™t â€œtof_i_mean_at_tsâ€� cho má»—i dáº¥u thá»�i gian, sau Ä‘Ã³ tá»•ng há»£p theo tá»«ng chuá»—i.

def compute_tof_sequence_summary(df):
    # Khá»Ÿi táº¡o má»™t dict Ä‘á»ƒ giá»¯ DataFrames theo tá»«ng chuá»—i
    seq_summaries = {}

    for i in range(1, 6):
        # XÃ¢y dá»±ng danh sÃ¡ch cÃ¡c cá»™t cho cáº£m biáº¿n i
        tof_cols = [f"tof_{i}_v{pix}" for pix in range(64)]
        # Thay tháº¿ -1 báº±ng NaN Ä‘á»ƒ chÃºng khÃ´ng lÃ m lá»‡ch giÃ¡ trá»‹ trung bÃ¬nh; chuyá»ƒn Ä‘á»•i sang float
        ts_grid = df[tof_cols].replace(-1, np.nan).astype(float)
        # TÃ­nh toÃ¡n â€œtrung bÃ¬nh trÃªn táº¥t cáº£ 64 pixelâ€� cho má»—i dáº¥u thá»�i gian
        df[f"tof_{i}_mean_at_ts"] = ts_grid.mean(axis=1)
    
   # BÃ¢y giá»�, nhÃ³m theo id chuá»—i vÃ  tÃ­nh giÃ¡ trá»‹ trung bÃ¬nh vÃ  Ä‘á»™ lá»‡ch chuáº©n cá»§a cÃ¡c giÃ¡ trá»‹ trung bÃ¬nh Ä‘Ã³
    agg_dict = {f"tof_{i}_mean_at_ts": ['mean','std'] for i in range(1, 6)}
    summary = df.groupby('sequence_id').agg(agg_dict)
    # LÃ m pháº³ng cÃ¡c cá»™t MultiIndex
    flat_cols = [f"tof_{i}_{stat}" for i in range(1, 6) for stat in ['mean','std']]
    summary.columns = flat_cols
    return summary

train_tof_summary = compute_tof_sequence_summary(train_temp)
test_tof_summary  = compute_tof_sequence_summary(test_temp)

# 8) Há»£p nháº¥t cÃ¡c tÃ³m táº¯t accel, rotation, thm, tof (trÃªn sequence_id)
train_sensor_summary = (
    train_acc_summary
    .join(train_rot_summary, how='outer')
    .join(train_thm_summary, how='outer')
    .join(train_tof_summary, how='outer')
)

test_sensor_summary = (
    test_acc_summary
    .join(test_rot_summary, how='outer')
    .join(test_thm_summary, how='outer')
    .join(test_tof_summary, how='outer')
)

# 9) ThÃªm cá»™t â€œDatasetâ€� Ä‘á»ƒ chÃºng ta cÃ³ thá»ƒ thá»±c hiá»‡n box+hist song song
train_sensor_summary['Dataset'] = 'Train'
test_sensor_summary['Dataset']  = 'Test'

# 10) GhÃ©p ná»‘i thÃ nh má»™t DataFrame Ä‘á»ƒ váº½ Ä‘á»“ thá»‹
combined_sensor_summary = pd.concat(
    [train_sensor_summary, test_sensor_summary],
    axis=0
).reset_index(drop=True)


import seaborn as sns
import matplotlib.pyplot as plt

# TÃ¡i sá»­ dá»¥ng báº£ng mÃ u tÃ¹y chá»‰nh cá»§a báº¡n
custom_palette = ['#3498db', '#e74c3c', '#2ecc71']

# Chá»‰ lá»�c cÃ¡c máº«u "Ä�Ã o táº¡o"
train_sensor_summary = combined_sensor_summary[combined_sensor_summary['Dataset'] == 'Train']

# XÃ¡c Ä‘á»‹nh hÃ m váº½ Ä‘á»“ thá»‹ (bÃ¢y giá»� chá»‰ sá»­ dá»¥ng train_sensor_summary)
def create_sensor_summary_plots(variable, dataset_label, data):
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Box Plot (left)
    plt.subplot(1, 2, 1)
    sns.boxplot(data=data, x=variable, y="Dataset", palette=[custom_palette[0]])
    plt.xlabel(variable)
    plt.title(f"Box Plot for {variable} â€” {dataset_label}")

    # Histogram (right)
    plt.subplot(1, 2, 2)
    sns.histplot(data=data, x=variable, kde=True, bins=30, color=custom_palette[0], label="Train")
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title(f"Histogram for {variable} â€” {dataset_label}")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Liá»‡t kÃª táº¥t cáº£ cÃ¡c biáº¿n tÃ³m táº¯t cáº£m biáº¿n (loáº¡i trá»« 'Bá»™ dá»¯ liá»‡u')
sensor_summary_vars = [col for col in train_sensor_summary.columns if col != 'Dataset']

# Plot each
for var in sensor_summary_vars:
    create_sensor_summary_plots(var, "Per-Sequence Sensor Summaries", train_sensor_summary)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import textwrap

# XÃ¡c Ä‘á»‹nh báº£ng mÃ u
pie_chart_palette = ['#33638d', '#28ae80', '#d3eb0c', '#ff9a0b', '#7e03a8', '#35b779',
                     '#fde725', '#440154', '#90d743', '#482173', '#22a884', '#f8961e']
custom_palette = ['#3498db']  # Only Train

# ğŸ”¹CÃ¡c biáº¿n phÃ¢n loáº¡i Ä‘á»ƒ phÃ¢n tÃ­ch (khÃ´ng bao gá»“m 'cá»­ chá»‰')
categorical_variables = [col for col in categorical_tracker if col != 'gesture']

#--- Chuáº©n bá»‹ táº­p dá»¯ liá»‡u chÃ­nh (chá»‰ Ä‘Ã o táº¡o)
train_main = train_df.copy()
train_main['dataset'] = 'train'
main_combined = train_main.copy()

# --- Chuáº©n bá»‹ bá»™ dá»¯ liá»‡u nhÃ¢n kháº©u há»�c (chá»‰ Ä‘Ã o táº¡o)
train_demo = train_dem_df.copy()
train_demo['dataset'] = 'train'
demo_combined = train_demo.copy()

#--- Chá»©c nÄƒng váº½ Ä‘á»“ thá»‹ thá»‘ng nháº¥t
def create_categorical_plots(variable, data, source_name):
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Biá»ƒu Ä‘á»“ trÃ²n (bÃªn trÃ¡i)
    plt.subplot(1, 2, 1)
    value_counts = data[variable].value_counts()

    # Thu gá»�n cÃ¡c danh má»¥c nhá»�
    threshold = 0.05 * value_counts.sum()
    filtered_values = value_counts.copy()
    filtered_values[value_counts < threshold] = 0
    filtered_values = filtered_values[filtered_values > 0]
    other_count = value_counts.sum() - filtered_values.sum()
    if other_count > 0:
        filtered_values['Other'] = other_count

    wedges, texts, autotexts = plt.pie(
        filtered_values,
        autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
        colors=pie_chart_palette[:len(filtered_values)],
        startangle=140,
        wedgeprops=dict(width=0.3),
        explode=[0.05 if p > 5 else 0 for p in filtered_values],
        textprops={'fontsize': 10}
    )

    plt.title("\n".join(textwrap.wrap(f"Pie Chart for {variable} â€” {source_name}", width=50)))
    plt.legend(filtered_values.index, loc="upper left", bbox_to_anchor=(1, 1))

    # Biá»ƒu Ä‘á»“ Ä‘áº¿m ngang (bÃªn pháº£i)
    plt.subplot(1, 2, 2)
    sns.countplot(
        data=data,
        y=variable,
        hue='dataset',  # váº«n cÃ²n, nhÆ°ng chá»‰ cÃ³ "tÃ u há»�a"
        palette=custom_palette,
        alpha=0.85
    )
    plt.ylabel(variable)
    plt.xlabel("Count")
    plt.title("\n".join(textwrap.wrap(f"Countplot for {variable} â€” {source_name}", width=50)))
    plt.tight_layout()
    plt.show()

# --- Váº½ biá»ƒu Ä‘á»“ táº¥t cáº£ cÃ¡c biáº¿n phÃ¢n loáº¡i tá»« táº­p dá»¯ liá»‡u chÃ­nh (chá»‰ huáº¥n luyá»‡n)
for var in categorical_variables:
    if var in main_combined.columns:
        create_categorical_plots(var, main_combined, "Context Features")

# --- Váº½ biá»ƒu Ä‘á»“ táº¥t cáº£ cÃ¡c biáº¿n phÃ¢n loáº¡i tá»« táº­p dá»¯ liá»‡u nhÃ¢n kháº©u há»�c (chá»‰ Ä‘Ã o táº¡o)
for var in categorical_variables:
    if var in demo_combined.columns:
        create_categorical_plots(var, demo_combined, "Demographic Features")

# Cleanup
train_main.drop('dataset', axis=1, inplace=True)
train_demo.drop('dataset', axis=1, inplace=True)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import textwrap

# XÃ¡c Ä‘á»‹nh báº£ng mÃ u
pie_chart_palette = ['#33638d', '#28ae80', '#d3eb0c', '#ff9a0b', '#7e03a8', '#35b779',
                     '#fde725', '#440154', '#90d743', '#482173', '#22a884', '#f8961e']

# XÃ¡c Ä‘á»‹nh cá»™t má»¥c tiÃªu
target_columns = ['gesture']
# Chuáº©n bá»‹ dá»¯ liá»‡u chÃ­nh (CHá»ˆ táº­p huáº¥n luyá»‡n)
train_main = train_df.copy()
train_main['dataset'] = 'train'  # giá»¯ láº¡i náº¿u báº¡n muá»‘n thÃªm láº¡i logic sau

# HÃ m váº½ Ä‘á»“ thá»‹ thá»‘ng nháº¥t cho cÃ¡c biáº¿n má»¥c tiÃªu (chá»‰ Ä‘Ã o táº¡o)
def create_target_plots(variable, data, source_name):
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Pie Chart (left)
    plt.subplot(1, 2, 1)
    value_counts = data[variable].value_counts()

    # Thu gá»�n cÃ¡c danh má»¥c nhá»� thÃ nh 'KhÃ¡c'
    threshold = 0.05 * value_counts.sum()
    filtered_values = value_counts.copy()
    filtered_values[value_counts < threshold] = 0
    filtered_values = filtered_values[filtered_values > 0]
    other_count = value_counts.sum() - filtered_values.sum()
    if other_count > 0:
        filtered_values['Other'] = other_count

    wedges, texts, autotexts = plt.pie(
        filtered_values,
        autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
        colors=pie_chart_palette[:len(filtered_values)],
        startangle=140,
        wedgeprops=dict(width=0.3),
        explode=[0.05 if p > 5 else 0 for p in filtered_values],
        textprops={'fontsize': 10}
    )

    plt.title("\n".join(textwrap.wrap(f"Pie Chart for {variable} â€” {source_name}", width=50)))
    plt.legend(filtered_values.index, loc="upper left", bbox_to_anchor=(1, 1))

    # Biá»ƒu Ä‘á»“ Ä‘áº¿m ngang (bÃªn pháº£i)
    plt.subplot(1, 2, 2)
    sns.countplot(
        data=data,
        y=variable,
        color='#3498db',  # just use a single color (train)
        alpha=0.85
    )
    plt.ylabel(variable)
    plt.xlabel("Count")
    plt.title("\n".join(textwrap.wrap(f"Countplot for {variable} â€” {source_name}", width=50)))
    plt.tight_layout()
    plt.show()

# Váº½ cá»™t má»¥c tiÃªu (chá»‰ Ä‘Ã o táº¡o)
for var in target_columns:
    if var in train_main.columns:
        create_target_plots(var, train_main, "Target Variable")

# Cleanup
train_main.drop('dataset', axis=1, inplace=True)


#fix code dÃ²ng dÆ°á»›i
subject_id = train_dem_df['subject'].sample(1).iloc[0]


subj_info = train_dem_df[train_dem_df['subject'] == subject_id].iloc[0]

print(f"â�© Details of the random subject being considered")
print(f"Subject ID:              {subject_id}")
print(f"  â€¢ Age:                 {subj_info['age']} years")
print(f"  â€¢ Adult/Child:         {'Adult' if subj_info['adult_child']==1 else 'Child'}")
print(f"  â€¢ Sex:                 {'Male' if subj_info['sex']==1 else 'Female'}")
print(f"  â€¢ Handedness:          {'Right-handed' if subj_info['handedness']==1 else 'Left-handed'}")
print(f"  â€¢ Height:              {subj_info['height_cm']} cm")
print(f"  â€¢ Shoulderâ†’Wrist:      {subj_info['shoulder_to_wrist_cm']} cm")
print(f"  â€¢ Elbowâ†’Wrist:         {subj_info['elbow_to_wrist_cm']} cm")


# â”€â”€â”€ Ã” 1: Biá»ƒu Ä‘á»“ chuá»—i thá»�i gian IMU â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Ã” nÃ y tÃ­nh toÃ¡n Ä‘á»™ lá»›n gia tá»‘c (âˆš(acc_xÂ²+acc_yÂ²+acc_zÂ²)) vÃ  gÃ³c quay (Î¸ = 2Â·arccos(rot_w))
# cho má»—i dáº¥u thá»�i gian trong chuá»—i cá»§a Ä‘á»‘i tÆ°á»£ng Ä‘Ã£ chá»�n. Sau Ä‘Ã³, nÃ³ váº½ hai Ä‘Æ°á»�ng tháº³ng cho má»—i chuá»—i cá»­ chá»‰:
# â€¢ acc_mag (Ä‘á»™ lá»›n gia tá»‘c)
# â€¢ rot_angle (gÃ³c Ä‘á»‹nh hÆ°á»›ng)
# Ná»�n Ä‘Æ°á»£c tÃ´ bÃ³ng theo â€˜phaâ€™: â€œChuyá»ƒn tiáº¿pâ€� so vá»›i â€œCá»­ chá»‰â€� vá»›i cÃ¡c mÃ u khÃ¡c nhau.
# Má»—i tiÃªu Ä‘á»� biá»ƒu Ä‘á»“ phá»¥ hiá»ƒn thá»‹ tÃªn cá»­ chá»‰ vÃ  sequence_type cá»§a biá»ƒu Ä‘á»“ Ä‘Ã³. Má»—i hÃ ng cÃ³ hai biá»ƒu Ä‘á»“.

import numpy as np
import matplotlib.pyplot as plt

# Chá»�n má»™t chá»§ Ä‘á»� (vÃ­ dá»¥: Ä‘áº§u tiÃªn trong danh sÃ¡ch)
subject_id = train_df['subject'].unique()[0]
subj_df = train_df[train_df['subject'] == subject_id].copy()

# TÃ­nh toÃ¡n cÃ¡c tÃ­nh nÄƒng IMU Ä‘Æ°á»£c suy ra
subj_df['acc_mag'] = np.sqrt(
    subj_df['acc_x']**2 + subj_df['acc_y']**2 + subj_df['acc_z']**2
)
subj_df['rot_w_clipped'] = subj_df['rot_w'].clip(-1, 1)
subj_df['rot_angle'] = 2 * np.arccos(subj_df['rot_w_clipped'])

# Chá»�n má»™t sequence_id cho má»—i nhÃ£n cá»­ chá»‰ duy nháº¥t
# NhÃ³m theo cá»™t 'cá»­ chá»‰' Ä‘Æ°á»£c mÃ£ hÃ³a theo sá»‘ nguyÃªn, sau Ä‘Ã³ láº¥y sequence_id Ä‘áº§u tiÃªn trong má»—i nhÃ³m
gesture_to_seq = subj_df.groupby('gesture')['sequence_id'].first().to_dict()
# Chuyá»ƒn Ä‘á»•i thÃ nh danh sÃ¡ch sequence_ids
seq_ids = list(gesture_to_seq.values())

n = len(seq_ids)
ncols = 2
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharex=False, sharey=False)
axes = axes.flatten()

for i, seq in enumerate(seq_ids):
    ax = axes[i]
    seq_df = subj_df[subj_df['sequence_id'] == seq].sort_values('sequence_counter')
    times = seq_df['sequence_counter']

    # Váº½ hai Ä‘Æ°á»�ng IMU Ä‘Ã£ suy ra
    ax.plot(times, seq_df['acc_mag'],
            label='Acceleration Magnitude', color='tab:blue', linewidth=1.5)
    ax.plot(times, seq_df['rot_angle'],
            label='Rotation Angle (rad)', color='tab:orange', linewidth=1.5)

    # TÃ´ bÃ³ng theo giai Ä‘oáº¡n: â€œChuyá»ƒn tiáº¿pâ€� so vá»›i â€œCá»­ chá»‰â€�
    used_labels = set()
    for phase_label, color in [('Transition', 'lightgray'),
                               ('Gesture', 'lightcoral')]:
        mask = seq_df['phase'] == phase_label
        if mask.any():
            # TÃ¬m cÃ¡c khoáº£ng liÃªn tiáº¿p trong Ä‘Ã³ pha == nhÃ£n pha
            idxs = seq_df.index[mask]
            breaks = np.where(np.diff(idxs) != 1)[0]
            spans = []
            start_idx = idxs[0]
            for b in breaks:
                end_idx = idxs[b]
                spans.append((start_idx, end_idx))
                start_idx = idxs[b + 1]
            spans.append((start_idx, idxs[-1]))
            # Váº½ tá»«ng khoáº£ng; chá»‰ ghi nhÃ£n má»™t láº§n cho má»—i pha trong sÆ¡ Ä‘á»“ phá»¥ nÃ y
            for (start_i, end_i) in spans:
                t0 = seq_df.loc[start_i, 'sequence_counter']
                t1 = seq_df.loc[end_i, 'sequence_counter']
                label_arg = phase_label if phase_label not in used_labels else None
                ax.axvspan(t0, t1, color=color, alpha=0.3, label=label_arg)
                used_labels.add(phase_label)

    # TiÃªu Ä‘á»� vÃ  nhÃ£n cá»§a cá»‘t truyá»‡n phá»¥
    gesture_name = seq_df['gesture'].iloc[0]
    seq_type = seq_df['sequence_type'].iloc[0]
    ax.set_title(f"Subject {subject_id} â€“ Seq {seq} â€“ {gesture_name} ({seq_type})",
                 fontsize=10)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Value", fontsize=9)
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True)

# Táº¯t báº¥t ká»³ Ã´ phá»¥ trá»‘ng nÃ o
for j in range(i + 1, nrows * ncols):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Sá»­ dá»¥ng cÃ¹ng subject_id vÃ  seq_ids (má»™t cho má»—i cá»­ chá»‰ duy nháº¥t) tá»« Ã” 1
therm_df = subj_df.copy()

# Thu tháº­p cÃ¡c cá»™t nhiá»‡t Ä‘iá»‡n vÃ  tÃ­nh toÃ¡n giÃ¡ trá»‹ trung bÃ¬nh theo dáº¥u thá»�i gian cá»§a chÃºng
thm_cols = [f'thm_{i}' for i in range(1, 6)]
therm_df['thm_mean'] = therm_df[thm_cols].mean(axis=1)

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharex=False, sharey=False)
axes = axes.flatten()

for i, seq in enumerate(seq_ids):
    ax = axes[i]
    seq_thm = therm_df[therm_df['sequence_id'] == seq].sort_values('sequence_counter')
    times = seq_thm['sequence_counter']
    # XÃ¡c Ä‘á»‹nh pháº¡m vi y Ä‘á»ƒ tÃ´ bÃ³ng (khÃ´ng nháº¥t thiáº¿t pháº£i cÃ³ vÃ¬ axvspan tráº£i dÃ i toÃ n bá»™ trá»¥c y)
    # Váº½ tá»«ng kÃªnh nhiá»‡t Ä‘iá»‡n
    for col in thm_cols:
        ax.plot(times, seq_thm[col], label=col, linewidth=1)
    # Váº½ giÃ¡ trá»‹ trung bÃ¬nh dÆ°á»›i dáº¡ng Ä‘Æ°á»�ng mÃ u Ä‘en Ä‘áº­m
    ax.plot(times, seq_thm['thm_mean'], label='Average Temperature', color='black', linewidth=2.5)
    # TÃ´ bÃ³ng theo pha báº±ng axvspan
    for phase_label, color in [('Transition', 'lightgray'), ('Gesture', 'lightcoral')]:
        mask = seq_thm['phase'] == phase_label
        if mask.any():
            idxs = seq_thm.index[mask]
            breaks = np.where(np.diff(idxs) != 1)[0]
            spans = []
            start_idx = idxs[0]
            for b in breaks:
                end_idx = idxs[b]
                spans.append((start_idx, end_idx))
                start_idx = idxs[b + 1]
            spans.append((start_idx, idxs[-1]))
            for (start_i, end_i) in spans:
                t0 = seq_thm.loc[start_i, 'sequence_counter']
                t1 = seq_thm.loc[end_i, 'sequence_counter']
                ax.axvspan(t0, t1, color=color, alpha=0.3, label=phase_label)
    # NhÃ£n vÃ  chÃº thÃ­ch cho cá»‘t truyá»‡n phá»¥ nÃ y
    gesture_name = seq_thm['gesture'].iloc[0]
    seq_type = seq_thm['sequence_type'].iloc[0]
    ax.set_title(f"Subject {subject_id} â€“ Seq {seq} â€“ {gesture_name} ({seq_type})", fontsize=10)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Temperature (Â°C)", fontsize=9)
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True)

# Táº¯t báº¥t ká»³ pháº§n phá»¥ nÃ o khÃ´ng sá»­ dá»¥ng
for j in range(i + 1, nrows * ncols):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Sá»­ dá»¥ng cÃ¹ng subject_id vÃ  seq_ids (má»™t cho má»—i cá»­ chá»‰ duy nháº¥t) tá»« Ã” 1
tof_df = subj_df.copy()

# XÃ¡c Ä‘á»‹nh 64 cá»™t cho má»—i cáº£m biáº¿n ToF vÃ  tÃ­nh giÃ¡ trá»‹ trung bÃ¬nh theo dáº¥u thá»�i gian cá»§a chÃºng
mean_cols = []
for i_sensor in range(1, 6):
    pixel_cols = [f'tof_{i_sensor}_v{pix}' for pix in range(64)]
    tof_df[f'tof_{i_sensor}_mean'] = tof_df[pixel_cols].replace(-1, np.nan).mean(axis=1)
    mean_cols.append(f'tof_{i_sensor}_mean')

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharex=False, sharey=False)
axes = axes.flatten()

for i, seq in enumerate(seq_ids):
    ax = axes[i]
    seq_tof = tof_df[tof_df['sequence_id'] == seq].sort_values('sequence_counter')
    times = seq_tof['sequence_counter']

    # Váº½ giÃ¡ trá»‹ trung bÃ¬nh cá»§a tá»«ng cáº£m biáº¿n ToF
    for col in mean_cols:
        ax.plot(times, seq_tof[col], label=col, linewidth=1)

    # TÃ´ bÃ³ng theo pha báº±ng axvspan
    for phase_label, color in [('Transition', 'lightgray'), ('Gesture', 'lightcoral')]:
        mask = seq_tof['phase'] == phase_label
        if mask.any():
            idxs = seq_tof.index[mask]
            breaks = np.where(np.diff(idxs) != 1)[0]
            spans = []
            start_idx = idxs[0]
            for b in breaks:
                end_idx = idxs[b]
                spans.append((start_idx, end_idx))
                start_idx = idxs[b + 1]
            spans.append((start_idx, idxs[-1]))
            for (start_i, end_i) in spans:
                t0 = seq_tof.loc[start_i, 'sequence_counter']
                t1 = seq_tof.loc[end_i, 'sequence_counter']
                ax.axvspan(t0, t1, color=color, alpha=0.3, label=phase_label)

    # NhÃ£n vÃ  chÃº thÃ­ch cho cá»‘t truyá»‡n phá»¥ nÃ y
    gesture_name = seq_tof['gesture'].iloc[0]
    seq_type     = seq_tof['sequence_type'].iloc[0]
    ax.set_title(f"Subject {subject_id} â€“ Seq {seq} â€“ {gesture_name} ({seq_type})", fontsize=10)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Mean ToF Distance", fontsize=9)
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True)

# Táº¯t báº¥t ká»³ pháº§n phá»¥ nÃ o khÃ´ng sá»­ dá»¥ng
for j in range(i + 1, nrows * ncols):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# (1) SÃ¡p nháº­p train_demographics vÃ o train_df náº¿u chÆ°a thá»±c hiá»‡n
train_df = train_df.merge(
    train_dem_df,
    on="subject",
    how="left"
)


import warnings, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# â”€â”€â”€ 1. Chá»�n biáº¿n nhÃ³m â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
group_col = "handedness"
group_map = {0: "Left-handed", 1: "Right-handed"}

# â”€â”€â”€ 2. XÃ¢y dá»±ng báº£ng tÃ³m táº¯t cáº¥p trÃ¬nh tá»± â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
seq_summaries = []

for seq_id, seq in train_df.groupby("sequence_id"):
    
    g_label = seq[group_col].iloc[0]
    if pd.isna(g_label):                   # bá»� qua náº¿u thiáº¿u thÃ´ng tin nhÃ¢n kháº©u há»�c
        continue
    
    # ----- Máº·t náº¡ pha -----
    gest_mask = seq["phase"] == "Gesture"
    
    # ----- 1) Gia tá»‘c giáº­t -----
    # giáº­t = |Î”acc_mag / Î”t| ; sá»­ dá»¥ng max trÃªn toÃ n bá»™ chuá»—i
    acc_mag = np.sqrt(seq["acc_x"]**2 + seq["acc_y"]**2 + seq["acc_z"]**2).values
    jerk = np.abs(np.diff(acc_mag))                         # Î”t = 1 frame
    max_jerk = jerk.max() if len(jerk) else np.nan
    
    # ----- 2) Pháº¡m vi gÃ³c quay (Chá»‰ pha cá»­ chá»‰) -----
    rot_angle = 2*np.arccos(seq["rot_w"].clip(-1,1))
    rot_range = rot_angle[gest_mask].max() - rot_angle[gest_mask].min() if gest_mask.any() else np.nan
    
    # ----- 3) Ä�á»‰nh nhiá»‡t Ä‘iá»‡n Î”T (Ä�á»‰nh cá»­ chá»‰ trá»« Ä‘i Ä‘Æ°á»�ng cÆ¡ sá»Ÿ chuyá»ƒn tiáº¿p) -----
    thm_cols = [f"thm_{i}" for i in range(1,6)]
    thm_mean = seq[thm_cols].ffill().bfill().mean(axis=1)
    baseline = thm_mean[seq["phase"]=="Transition"].median()
    peak_dt  = thm_mean[gest_mask].max() - baseline if gest_mask.any() else np.nan
    
    # ----- 4) Khoáº£ng cÃ¡ch trung bÃ¬nh gáº§n nháº¥t cá»§a ToF (tá»‘i thiá»ƒu trÃªn táº¥t cáº£ 5 cáº£m biáº¿n) -----
    tof_means = []
    for i in range(1,6):
        col = f"tof_{i}_mean"
        if col not in seq:
            # tÃ­nh toÃ¡n tá»©c thá»�i (thay tháº¿ -1 báº±ng NaN)
            pix = seq[[f"tof_{i}_v{p}" for p in range(64)]].replace(-1, np.nan)
            seq[col] = pix.mean(axis=1)
        tof_means.append(seq[col])
    tof_overall = pd.concat(tof_means, axis=1).mean(axis=1)
    min_tof = tof_overall.min()
    
    seq_summaries.append({
        "sequence_id":       seq_id,
        group_col:           g_label,
        "max_jerk":          max_jerk,
        "rot_range":         rot_range,
        "thm_peak_delta":    peak_dt,
        "tof_min_distance":  min_tof
    })

summary_df = pd.DataFrame(seq_summaries).dropna()

# Ã�nh xáº¡ nhÃ£n nhÃ³m thÃ nh chuá»—i cÃ³ thá»ƒ Ä‘á»�c Ä‘Æ°á»£c
summary_df["Group"] = summary_df[group_col].map(group_map)

# â”€â”€â”€ 3. PhÃ¢n phá»‘i Ä‘á»“ thá»‹ â”€ ...
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.35, wspace=0.25)

metrics = [
    ("max_jerk",         "Max Jerk |Î”acc| (m/sÂ²)"),
    ("rot_range",        "Gesture Rot-Angle Range (rad)"),
    ("thm_peak_delta",   "Thermopile Î”T Peak (Â°C)"),
    ("tof_min_distance", "Min ToF Mean Dist. (units)")
]

for ax, (metric, nice_name) in zip(axes.flatten(), metrics):
    sns.violinplot(
        data=summary_df,
        x="Group", y=metric, palette="Set2",
        inner=None, ax=ax
    )
    sns.boxplot(
        data=summary_df,
        x="Group", y=metric,
        width=0.2, showcaps=True, boxprops={'facecolor':'white'},
        showfliers=False, whiskerprops={'linewidth':2},
        ax=ax
    )
    ax.set_title(nice_name, fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel(nice_name, fontsize=10)

fig.suptitle(f"{group_map[0]} vs. {group_map[1]} â€“ Sequence-Level Sensor Features (*Gesture Phase Only)", fontsize=16)
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# â”€â”€â”€ TÃ­nh toÃ¡n thá»�i lÆ°á»£ng pha theo tá»«ng chuá»—i â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Giáº£ sá»­ train_df Ä‘Ã£ há»£p nháº¥t dá»¯ liá»‡u nhÃ¢n kháº©u há»�c vÃ  cá»™t 'pha'

# Ä�áº¿m sá»‘ khung hÃ¬nh trÃªn má»—i pha, trÃªn má»—i chuá»—i
dur_df = (
    train_df
    .groupby(['sequence_id', 'handedness'])['phase']
    .value_counts()
    .unstack(fill_value=0)
    .reset_index()
)

# Chá»‰ giá»¯ láº¡i Chuyá»ƒn tiáº¿p & Cá»­ chá»‰ (trong trÆ°á»�ng há»£p Táº¡m dá»«ng tá»“n táº¡i)
dur_df = dur_df[['sequence_id', 'handedness', 'Transition', 'Gesture']]

# Chuyá»ƒn sang dáº¡ng dÃ i cho biá»ƒu Ä‘á»“ há»™p
dur_long = dur_df.melt(
    id_vars=['sequence_id', 'handedness'],
    value_vars=['Transition', 'Gesture'],
    var_name='phase',
    value_name='duration_frames'
)

# Báº£n Ä‘á»“ thuáº­n tay trÃ¡i thÃ nh nhÃ£n
dur_long['Group'] = dur_long['handedness'].map({0: 'Left-handed', 1: 'Right-handed'})

# â”€â”€â”€ Váº½ cÃ¡c há»™p biá»ƒu Ä‘á»“ ngang cáº¡nh nhau â”€ ...
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
plt.subplots_adjust(wspace=0.2)

for ax, group_label in zip(axes, ['Left-handed', 'Right-handed']):
    subset = dur_long[dur_long['Group'] == group_label]
    sns.boxplot(
        data=subset,
        x='duration_frames',
        y='phase',
        orient='h',
        palette=['lightgray', 'lightcoral'],
        ax=ax
    )
    ax.set_title(f"{group_label}", fontsize=12)
    ax.set_xlabel("Duration (frames)", fontsize=10)
    ax.set_ylabel("Phase", fontsize=10)
    ax.grid(True)

fig.suptitle("Phase Durations by Handedness: Transition vs. Gesture", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()


# â”€â”€â”€ Ã” A: So sÃ¡nh Ä‘á»™ thuáº­n tay cá»§a IMU, Rot, Thermopile vÃ  ToF â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# (2) Chá»�n má»™t cá»­ chá»‰ Ä‘á»ƒ so sÃ¡nh (thay Ä‘á»•i náº¿u cáº§n)
gesture_to_plot = "Write name on leg"

# (3) Lá»�c chá»‰ cÃ¡c hÃ ng cÃ³ cá»­ chá»‰ Ä‘Ã³ vÃ  phÃ¢n chia theo tay thuáº­n
df_gesture = train_df[train_df["gesture"] == gesture_to_plot].copy()

# NhÃ³m thuáº­n tay trÃ¡i (thuáº­n tay == 0) vÃ  NhÃ³m thuáº­n tay pháº£i (thuáº­n tay == 1)
left_group  = df_gesture[df_gesture["handedness"] == 0]
right_group = df_gesture[df_gesture["handedness"] == 1]

# (4) Tá»« má»—i nhÃ³m, chá»�n ngáº«u nhiÃªn má»™t sequence_id (hoáº·c Ä‘áº§u tiÃªn) Ä‘á»ƒ váº½ Ä‘á»“ thá»‹
# Náº¿u khÃ´ng cÃ³ vÃ­ dá»¥ nÃ o dÃ nh cho ngÆ°á»�i thuáº­n tay trÃ¡i, báº¡n cÃ³ thá»ƒ cáº§n má»™t cá»­ chá»‰ khÃ¡c.
if left_group["sequence_id"].nunique() == 0:
    raise ValueError("No leftâ€�handed example of that gesture found. Choose a different gesture.")

left_seq  = left_group["sequence_id"].unique()[0]
right_seq = right_group["sequence_id"].unique()[0]

# (5) TrÃ­ch xuáº¥t hai chuá»—i, sáº¯p xáº¿p theo sequence_counter
df_left  = train_df[(train_df["sequence_id"] == left_seq)].sort_values("sequence_counter")
df_right = train_df[(train_df["sequence_id"] == right_seq)].sort_values("sequence_counter")

# (6)TÃ­nh toÃ¡n cÃ¡c Ä‘áº·c Ä‘iá»ƒm cÃ³ nguá»“n gá»‘c tá»« IMU: acc_mag vÃ  rot_angle
for df in (df_left, df_right):
    df["acc_mag"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
    df["rot_w_clipped"] = df["rot_w"].clip(-1, 1)
    df["rot_angle"] = 2 * np.arccos(df["rot_w_clipped"])

# (7) Váº½ trá»£ lÃ½ Ä‘á»ƒ tÃ´ bÃ³ng cÃ¡c pha
def shade_phases(ax, seq_df):
    used = set()
    for phase_label, color in [("Transition", "lightgray"), ("Gesture", "lightcoral")]:
        mask = seq_df["phase"] == phase_label
        if not mask.any():
            continue
        idxs = seq_df.index[mask]
        diffs = np.where(np.diff(idxs) != 1)[0]
        spans = []
        start = idxs[0]
        for b in diffs:
            end = idxs[b]
            spans.append((start, end))
            start = idxs[b + 1]
        spans.append((start, idxs[-1]))
        for (s, e) in spans:
            t0 = seq_df.loc[s, "sequence_counter"]
            t1 = seq_df.loc[e, "sequence_counter"]
            label = phase_label if phase_label not in used else None
            ax.axvspan(t0, t1, color=color, alpha=0.3, label=label)
            used.add(phase_label)

# (8) BÃ¢y giá»� hÃ£y táº¡o **bá»‘n** hÃ¬nh riÃªng biá»‡t (má»—i hÃ¬nh cÃ³ 2 biá»ƒu Ä‘á»“ phá»¥ cáº¡nh nhau):

# --------------------------------------------------------------------------------
# A.1: IMU: Ä�á»™ lá»›n gia tá»‘c
# --------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_left, df_right],
    [f"Leftâ€�handed (Subj: {df_left['subject'].iloc[0]})",
     f"Rightâ€�handed (Subj: {df_right['subject'].iloc[0]})"]
):
    ax.plot(df_seq["sequence_counter"], df_seq["acc_mag"],
            color="tab:blue", lw=1.5, label="|acc|(m/sÂ²)")
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Acceleration Magnitude", fontsize=9)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)
plt.suptitle("Handedness: Acceleration Magnitude â€“ Left vs. Right", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --------------------------------------------------------------------------------
# A.2: IMU: GÃ³c quay
# --------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_left, df_right],
    [f"Leftâ€�handed (Subj: {df_left['subject'].iloc[0]})",
     f"Rightâ€�handed (Subj: {df_right['subject'].iloc[0]})"]
):
    ax.plot(df_seq["sequence_counter"], df_seq["rot_angle"],
            color="tab:orange", lw=1.5, label="rot_angle (rad)")
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Rotation Angle (rad)", fontsize=9)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)
plt.suptitle("Handedness: Rotation Angle â€“ Left vs. Right", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --------------------------------------------------------------------------------
# A.3: Nhiá»‡t Ä‘iá»‡n: 5 kÃªnh + Trung bÃ¬nh
# --------------------------------------------------------------------------------
# (TÃ­nh giÃ¡ trá»‹ trung bÃ¬nh cá»§a nhiá»‡t Ä‘iá»‡n náº¿u chÆ°a thá»±c hiá»‡n)
for df in (df_left, df_right):
    thm_cols = [f"thm_{i}" for i in range(1,6)]
    df["thm_mean"] = df[thm_cols].mean(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_left, df_right],
    [f"Leftâ€�handed (Subj: {df_left['subject'].iloc[0]})",
     f"Rightâ€�handed (Subj: {df_right['subject'].iloc[0]})"]
):
    times = df_seq["sequence_counter"]
    # Váº½ nÄƒm kÃªnh thÃ´
    for col in [f"thm_{i}" for i in range(1,6)]:
        ax.plot(times, df_seq[col], lw=1, label=col, alpha=0.7)
    #Váº½ biá»ƒu Ä‘á»“ trung bÃ¬nh dÆ°á»›i dáº¡ng in Ä‘áº­m
    ax.plot(times, df_seq["thm_mean"], color="black", lw=2.5, label="thm_mean")
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Thermopile (Â°C)", fontsize=9)
    ax.legend(loc="upper right", fontsize="x-small", ncol=2)
    ax.grid(True)
plt.suptitle("Handedness: Thermopile â€“ Left vs. Right", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --------------------------------------------------------------------------------
# A.4: ToF: NÄƒm kÃªnh khoáº£ng cÃ¡ch trung bÃ¬nh
# --------------------------------------------------------------------------------
# (TÃ­nh toÃ¡n ToF trung bÃ¬nh trÃªn má»—i cáº£m biáº¿n)
for df in (df_left, df_right):
    for i_sensor in range(1,6):
        pixel_cols = [f"tof_{i_sensor}_v{pix}" for pix in range(64)]
        df[f"tof_{i_sensor}_mean"] = df[pixel_cols].replace(-1, np.nan).mean(axis=1)

mean_cols = [f"tof_{i}_mean" for i in range(1,6)]
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_left, df_right],
    [f"Leftâ€�handed (Subj: {df_left['subject'].iloc[0]})",
     f"Rightâ€�handed (Subj: {df_right['subject'].iloc[0]})"]
):
    times = df_seq["sequence_counter"]
    for col in mean_cols:
        ax.plot(times, df_seq[col], lw=1, label=col)
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("ToF Mean Distance", fontsize=9)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)
plt.suptitle("Handedness: ToF Mean Distance â€“ Left vs. Right", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# â”€â”€â”€ 1. XÃ¢y dá»±ng tÃ³m táº¯t cáº¥p Ä‘á»™ trÃ¬nh tá»± cho NgÆ°á»�i lá»›n so vá»›i Tráº» em â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
seq_summaries = []
for seq_id, seq in train_df.groupby("sequence_id"):
    grp = seq["adult_child"].iloc[0]  # 0=child, 1=adult
    # Bá»� qua náº¿u khÃ´ng cÃ³ nhÃ£n pha
    if seq["phase"].isna().all():
        continue
    # 1) Max Jerk (IMU)
    acc_mag = np.sqrt(seq["acc_x"]**2 + seq["acc_y"]**2 + seq["acc_z"]**2).values
    jerk    = np.abs(np.diff(acc_mag))
    max_jerk = jerk.max() if len(jerk) else np.nan
    # 2) Pháº¡m vi xoay trong khi cá»­ chá»‰
    rot_angle = 2 * np.arccos(seq["rot_w"].clip(-1,1))
    mask_g = seq["phase"]=="Gesture"
    rot_range = rot_angle[mask_g].max() - rot_angle[mask_g].min() if mask_g.any() else np.nan
    # 3) Ä�á»‰nh nhiá»‡t Ä‘iá»‡n Î”T
    thm = seq[[f"thm_{i}" for i in range(1,6)]].ffill().bfill().mean(axis=1)
    base = thm[seq["phase"]=="Transition"].median()
    peak_dt = thm[mask_g].max() - base if mask_g.any() else np.nan
    # 4) Khoáº£ng cÃ¡ch tá»‘i thiá»ƒu ToF
    tof_means = []
    for i in range(1,6):
        col = f"tof_{i}_mean"
        if col not in seq:
            pix = seq[[f"tof_{i}_v{p}" for p in range(64)]].replace(-1,np.nan)
            seq[col] = pix.mean(axis=1)
        tof_means.append(seq[col])
    tof_min = pd.concat(tof_means,axis=1).mean(axis=1).min()
    seq_summaries.append({
        "sequence_id": seq_id,
        "adult_child": grp,
        "max_jerk": max_jerk,
        "rot_range": rot_range,
        "thm_peak_delta": peak_dt,
        "tof_min_distance": tof_min
    })

summary_df = pd.DataFrame(seq_summaries).dropna()
summary_df["Group"] = summary_df["adult_child"].map({0:"Child", 1:"Adult"})

# â”€â”€â”€ 2. Váº½ Ä‘á»“ thá»‹ phÃ¢n phá»‘i cáº¡nh nhau â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14,10), sharey=False)
plt.subplots_adjust(hspace=0.4, wspace=0.3)

metrics = [
    ("max_jerk",         "Max Jerk |Î”acc| (m/sÂ²)"),
    ("rot_range",        "Gesture Rot-Range (rad)"),
    ("thm_peak_delta",   "Thermopile Î”T Peak (Â°C)"),
    ("tof_min_distance", "Min ToF Mean Dist. (units)")
]

for ax, (col, label) in zip(axes.flatten(), metrics):
    sns.violinplot(
        x="Group", y=col, data=summary_df,
        palette="Set2", inner=None, ax=ax
    )
    sns.boxplot(
        x="Group", y=col, data=summary_df,
        width=0.2, showcaps=True, boxprops={'facecolor':'white'},
        showfliers=False, whiskerprops={'linewidth':2}, ax=ax
    )
    ax.set_title(label, fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel(label, fontsize=10)

fig.suptitle("Adult vs. Child â€“ Sequence-Level Sensor Feature Distributions (*Gesture Phase Only)", fontsize=16)
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# â”€â”€â”€ TÃ­nh toÃ¡n thá»�i lÆ°á»£ng pha theo trÃ¬nh tá»± cho NgÆ°á»�i lá»›n so vá»›i Tráº» em â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
dur_df = (
    train_df
    .groupby(['sequence_id', 'adult_child'])['phase']
    .value_counts()
    .unstack(fill_value=0)
    .reset_index()
)

# Chá»‰ giá»¯ láº¡i cÃ¡c cá»™t Chuyá»ƒn tiáº¿p & Cá»­ chá»‰
dur_df = dur_df[['sequence_id', 'adult_child', 'Transition', 'Gesture']]

# Tan cháº£y thÃ nh dáº¡ng dÃ i
dur_long = dur_df.melt(
    id_vars=['sequence_id', 'adult_child'],
    value_vars=['Transition', 'Gesture'],
    var_name='phase',
    value_name='duration_frames'
)

# Ã�nh xáº¡ adult_child vÃ o nhÃ£n
dur_long['Group'] = dur_long['adult_child'].map({0: 'Child', 1: 'Adult'})

# â”€â”€â”€ Váº½ cÃ¡c biá»ƒu Ä‘á»“ há»™p ngang cáº¡nh nhau â”€ ...
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
plt.subplots_adjust(wspace=0.3)

for ax, group_label in zip(axes, ['Child', 'Adult']):
    subset = dur_long[dur_long['Group'] == group_label]
    sns.boxplot(
        data=subset,
        x='duration_frames',
        y='phase',
        orient='h',
        palette=['lightgray', 'lightcoral'],
        ax=ax
    )
    ax.set_title(f"{group_label}", fontsize=12)
    ax.set_xlabel("Duration (frames)", fontsize=10)
    ax.set_ylabel("Phase", fontsize=10)
    ax.grid(True)

plt.suptitle("Phase Durations by Adult vs. Child", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()


# â”€â”€â”€ Táº¿ bÃ o B: So sÃ¡nh giá»¯a ngÆ°á»�i lá»›n vÃ  tráº» em Ä‘á»‘i vá»›i IMU, Rot, Thermopile vÃ  ToF â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import numpy as np
import matplotlib.pyplot as plt

# (2) Chá»�n má»™t cá»­ chá»‰ Ä‘á»ƒ so sÃ¡nh (thay Ä‘á»•i náº¿u cáº§n)
gesture_to_plot = "Neck - pinch skin"

# (3)Lá»�c chá»‰ cÃ¡c hÃ ng cÃ³ cá»­ chá»‰ Ä‘Ã³, chia theo adult_child (0=child, 1=adult)
df_gesture = train_df[train_df["gesture"] == gesture_to_plot].copy()
child_group = df_gesture[df_gesture["adult_child"] == 0]
adult_group = df_gesture[df_gesture["adult_child"] == 1]

if (child_group["sequence_id"].nunique() == 0) or (adult_group["sequence_id"].nunique() == 0):
    raise ValueError("Insufficient examples in one of the groups. Pick another gesture.")

child_seq = child_group["sequence_id"].unique()[0]
adult_seq = adult_group["sequence_id"].unique()[0]

df_child = train_df[train_df["sequence_id"] == child_seq].sort_values("sequence_counter")
df_adult = train_df[train_df["sequence_id"] == adult_seq].sort_values("sequence_counter")

# TÃ­nh toÃ¡n cÃ¡c tÃ­nh nÄƒng IMU Ä‘Æ°á»£c láº¥y tá»« má»—i
for df in (df_child, df_adult):
    df["acc_mag"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
    df["rot_w_clipped"] = df["rot_w"].clip(-1, 1)
    df["rot_angle"] = 2 * np.arccos(df["rot_w_clipped"])

# TrÃ¬nh trá»£ giÃºp váº½ Ä‘á»“ thá»‹ Ä‘á»ƒ tÃ´ bÃ³ng cÃ¡c pha (chá»©c nÄƒng sá»­ dá»¥ng láº¡i tá»« Ã” A)
def shade_phases(ax, seq_df):
    used = set()
    for phase_label, color in [("Transition", "lightgray"), ("Gesture", "lightcoral")]:
        mask = seq_df["phase"] == phase_label
        if not mask.any():
            continue
        idxs = seq_df.index[mask]
        diffs = np.where(np.diff(idxs) != 1)[0]
        spans = []
        start = idxs[0]
        for b in diffs:
            end = idxs[b]
            spans.append((start, end))
            start = idxs[b + 1]
        spans.append((start, idxs[-1]))
        for (s, e) in spans:
            t0 = seq_df.loc[s, "sequence_counter"]
            t1 = seq_df.loc[e, "sequence_counter"]
            label = phase_label if phase_label not in used else None
            ax.axvspan(t0, t1, color=color, alpha=0.3, label=label)
            used.add(phase_label)

# --------------------------------------------------------------------------------
# B.1: IMU â€“ Ä�á»™ lá»›n gia tá»‘c
# --------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_child, df_adult],
    [f"Child (Subj: {df_child['subject'].iloc[0]}, Age: {df_child['age'].iloc[0]})",
     f"Adult (Subj: {df_adult['subject'].iloc[0]}, Age: {df_adult['age'].iloc[0]})"]
):
    ax.plot(df_seq["sequence_counter"], df_seq["acc_mag"],
            color="tab:blue", lw=1.5, label="|acc|(m/sÂ²)")
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Acc Magnitude", fontsize=9)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)
plt.suptitle("Adult vs. Child: Acceleration Magnitude", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --------------------------------------------------------------------------------
# B.2: IMU â€“ GÃ³c quay
# --------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_child, df_adult],
    [f"Child (Subj: {df_child['subject'].iloc[0]}, Age: {df_child['age'].iloc[0]})",
     f"Adult (Subj: {df_adult['subject'].iloc[0]}, Age: {df_adult['age'].iloc[0]})"]
):
    ax.plot(df_seq["sequence_counter"], df_seq["rot_angle"],
            color="tab:orange", lw=1.5, label="rot_angle (rad)")
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Rot Angle (rad)", fontsize=9)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)
plt.suptitle("Adult vs. Child: Rotation Angle", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --------------------------------------------------------------------------------
# B.3: Nhiá»‡t Ä‘iá»‡n â€“ 5 kÃªnh + Trung bÃ¬nh
# --------------------------------------------------------------------------------
for df in (df_child, df_adult):
    thm_cols = [f"thm_{i}" for i in range(1,6)]
    df["thm_mean"] = df[thm_cols].mean(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_child, df_adult],
    [f"Child (Subj: {df_child['subject'].iloc[0]}, Age: {df_child['age'].iloc[0]})",
     f"Adult (Subj: {df_adult['subject'].iloc[0]}, Age: {df_adult['age'].iloc[0]})"]
):
    times = df_seq["sequence_counter"]
    for col in [f"thm_{i}" for i in range(1,6)]:
        ax.plot(times, df_seq[col], lw=1, label=col, alpha=0.7)
    ax.plot(times, df_seq["thm_mean"], color="black", lw=2.5, label="thm_mean")
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Thermopile (Â°C)", fontsize=9)
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    ax.grid(True)
plt.suptitle("Adult vs. Child: Thermopile", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --------------------------------------------------------------------------------
# B.4: ToF â€“ NÄƒm kÃªnh khoáº£ng cÃ¡ch trung bÃ¬nh
# --------------------------------------------------------------------------------
for df in (df_child, df_adult):
    for i_sensor in range(1,6):
        pixel_cols = [f"tof_{i_sensor}_v{pix}" for pix in range(64)]
        df[f"tof_{i_sensor}_mean"] = df[pixel_cols].replace(-1, np.nan).mean(axis=1)

mean_cols = [f"tof_{i}_mean" for i in range(1,6)]
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_child, df_adult],
    [f"Child (Subj: {df_child['subject'].iloc[0]}, Age: {df_child['age'].iloc[0]})",
     f"Adult (Subj: {df_adult['subject'].iloc[0]}, Age: {df_adult['age'].iloc[0]})"]
):
    times = df_seq["sequence_counter"]
    for col in mean_cols:
        ax.plot(times, df_seq[col], lw=1, label=col)
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("ToF Mean Distance", fontsize=9)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)
plt.suptitle("Adult vs. Child: ToF Mean Distance", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# â”€â”€â”€ 1. Chá»�n biáº¿n nhÃ³m cho Ã´ nÃ y â”€ ...
group_col = "sex"
group_map = {0: "Female", 1: "Male"}

# â”€â”€â”€ 2. XÃ¢y dá»±ng tÃ³m táº¯t cáº¥p trÃ¬nh tá»± (náº¿u chÆ°a táº¡o) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Sá»­ dá»¥ng láº¡i logic tá»« Ã´ tá»•ng quan trÆ°á»›c Ä‘Ã³, nhÆ°ng nhÃ³m vÃ o 'giá»›i tÃ­nh' thay vÃ¬ 'thuáº­n tay'.

seq_stats = []
for seq_id, seq in train_df.groupby("sequence_id"):
    g = seq[group_col].iloc[0]
    if pd.isna(g):
        continue

    # Phase masks
    gest = seq["phase"] == "Gesture"

    # 1) IMU jerk
    acc = np.sqrt(seq["acc_x"]**2 + seq["acc_y"]**2 + seq["acc_z"]**2).values
    max_jerk = np.abs(np.diff(acc)).max() if len(acc)>1 else np.nan

    # 2) Pháº¡m vi xoay trong cá»­ chá»‰
    rot = 2*np.arccos(seq["rot_w"].clip(-1,1))
    rot_range = rot[gest].max() - rot[gest].min() if gest.any() else np.nan

    # 3) Delta Ä‘á»‰nh nhiá»‡t Ä‘iá»‡n
    thms = seq[[f"thm_{i}" for i in range(1,6)]].ffill().bfill().mean(axis=1)
    baseline = thms[seq["phase"]=="Transition"].median() if (seq["phase"]=="Transition").any() else np.nan
    thm_peak = thms[gest].max() - baseline if gest.any() else np.nan

    # 4) Khoáº£ng cÃ¡ch trung bÃ¬nh tá»‘i thiá»ƒu ToF
    tcols = []
    for i in range(1,6):
        col = f"tof_{i}_mean"
        if col not in seq:
            pix = seq[[f"tof_{i}_v{p}" for p in range(64)]].replace(-1, np.nan)
            seq[col] = pix.mean(axis=1)
        tcols.append(seq[col])
    tof_all = pd.concat(tcols, axis=1).mean(axis=1)
    tof_min = tof_all.min()

    seq_stats.append({
        "sequence_id": seq_id,
        group_col: g,
        "max_jerk": max_jerk,
        "rot_range": rot_range,
        "thm_peak_delta": thm_peak,
        "tof_min_distance": tof_min
    })

summary = pd.DataFrame(seq_stats).dropna()
summary["Group"] = summary[group_col].map(group_map)

#â”€â”€â”€ 3. Váº½ biá»ƒu Ä‘á»“ phÃ¢n phá»‘i Ä‘Æ°á»£c Ä‘iá»�u chá»‰nh â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.35, wspace=0.25)

metrics = [
    ("max_jerk",         "Max Jerk |Î”acc| (m/sÂ²)"),
    ("rot_range",        "Rotation Range (rad)"),
    ("thm_peak_delta",   "Thermopile Î”T Peak (Â°C)"),
    ("tof_min_distance", "Min ToF Mean Dist.")
]

for ax, (col, lbl) in zip(axes.flatten(), metrics):
    sns.violinplot(
        data=summary,
        x="Group", y=col,
        palette="Set2",
        inner=None,
        ax=ax
    )
    sns.boxplot(
        data=summary,
        x="Group", y=col,
        width=0.2,
        showcaps=True,
        boxprops={'facecolor':'white'},
        showfliers=False,
        whiskerprops={'linewidth':2},
        ax=ax
    )
    ax.set_title(lbl, fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel(lbl, fontsize=10)

fig.suptitle("Sex Comparison: Sequenceâ€�Level Sensor Summaries (*Gesture Phase Only)", fontsize=16)
plt.show()


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# â”€â”€â”€ TÃ­nh toÃ¡n thá»�i lÆ°á»£ng pha theo tá»«ng chuá»—i Ä‘á»ƒ so sÃ¡nh Giá»›i tÃ­nh â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Giáº£ sá»­ train_df Ä‘Ã£ Ä‘Æ°á»£c há»£p nháº¥t vá»›i train_dem_df

# NhÃ³m theo sequence_id vÃ  giá»›i tÃ­nh, Ä‘áº¿m sá»‘ khung hÃ¬nh trong má»—i pha
dur_df_sex = (
    train_df
    .groupby(['sequence_id', 'sex'])['phase']
    .value_counts()
    .unstack(fill_value=0)
    .reset_index()
)

# Chá»‰ giá»¯ láº¡i Chuyá»ƒn tiáº¿p & Cá»­ chá»‰
dur_df_sex = dur_df_sex[['sequence_id', 'sex', 'Transition', 'Gesture']]

# Tan cháº£y thÃ nh dáº¡ng dÃ i
dur_long_sex = dur_df_sex.melt(
    id_vars=['sequence_id', 'sex'],
    value_vars=['Transition', 'Gesture'],
    var_name='phase',
    value_name='duration_frames'
)

# Ã�nh xáº¡ giá»›i tÃ­nh vÃ o nhÃ£n
dur_long_sex['Group'] = dur_long_sex['sex'].map({0: 'Female', 1: 'Male'})

# â”€â”€â”€ Váº½ cÃ¡c há»™p biá»ƒu Ä‘á»“ ngang cáº¡nh nhau â”€ ...
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
plt.subplots_adjust(wspace=0.2)

for ax, group_label in zip(axes, ['Female', 'Male']):
    subset = dur_long_sex[dur_long_sex['Group'] == group_label]
    sns.boxplot(
        data=subset,
        x='duration_frames',
        y='phase',
        orient='h',
        palette={'Transition':'lightgray', 'Gesture':'lightcoral'},
        ax=ax
    )
    ax.set_title(f"{group_label}", fontsize=12)
    ax.set_xlabel("Duration (frames)", fontsize=10)
    ax.set_ylabel("Phase", fontsize=10)
    ax.grid(True)

fig.suptitle("Sex Comparison: Transition vs. Gesture Duration", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()


# â”€â”€â”€ Táº¿ bÃ o C: So sÃ¡nh giá»›i tÃ­nh cá»§a IMU, Rot, Thermopile vÃ  ToF â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# (2) Chá»�n má»™t cá»­ chá»‰ Ä‘á»ƒ so sÃ¡nh (thay Ä‘á»•i náº¿u cáº§n)
gesture_to_plot = "Forehead - scratch"

# (3) Lá»�c chá»‰ cÃ¡c hÃ ng cÃ³ cá»­ chá»‰ Ä‘Ã³, chia theo giá»›i tÃ­nh (0=ná»¯, 1=nam)
df_gesture = train_df[train_df["gesture"] == gesture_to_plot].copy()
female_group = df_gesture[df_gesture["sex"] == 0]
male_group   = df_gesture[df_gesture["sex"] == 1]

if (female_group["sequence_id"].nunique() == 0) or (male_group["sequence_id"].nunique() == 0):
    raise ValueError("Not enough examples for each sex. Try a different gesture.")

female_seq = female_group["sequence_id"].unique()[0]
male_seq   = male_group["sequence_id"].unique()[0]

df_fem = train_df[train_df["sequence_id"] == female_seq].sort_values("sequence_counter")
df_male= train_df[train_df["sequence_id"] == male_seq].sort_values("sequence_counter")

# TÃ­nh toÃ¡n cÃ¡c tÃ­nh nÄƒng IMU Ä‘Æ°á»£c suy ra
for df in (df_fem, df_male):
    df["acc_mag"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
    df["rot_w_clipped"] = df["rot_w"].clip(-1, 1)
    df["rot_angle"] = 2 * np.arccos(df["rot_w_clipped"])

# TÃ¡i sá»­ dá»¥ng chá»©c nÄƒng Ä‘á»• bÃ³ng
def shade_phases(ax, seq_df):
    used = set()
    for phase_label, color in [("Transition", "lightgray"), ("Gesture", "lightcoral")]:
        mask = seq_df["phase"] == phase_label
        if not mask.any():
            continue
        idxs = seq_df.index[mask]
        diffs = np.where(np.diff(idxs) != 1)[0]
        spans = []
        start = idxs[0]
        for b in diffs:
            end = idxs[b]
            spans.append((start, end))
            start = idxs[b + 1]
        spans.append((start, idxs[-1]))
        for (s, e) in spans:
            t0 = seq_df.loc[s, "sequence_counter"]
            t1 = seq_df.loc[e, "sequence_counter"]
            label = phase_label if phase_label not in used else None
            ax.axvspan(t0, t1, color=color, alpha=0.3, label=label)
            used.add(phase_label)

# --------------------------------------------------------------------------------
# C.1: IMU â€“ Ä�á»™ lá»›n gia tá»‘c
# --------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_fem, df_male],
    [f"Female (Subj: {df_fem['subject'].iloc[0]}, Sex=0)",
     f"Male   (Subj: {df_male['subject'].iloc[0]}, Sex=1)"]
):
    ax.plot(df_seq["sequence_counter"], df_seq["acc_mag"],
            color="tab:blue", lw=1.5, label="|acc|(m/sÂ²)")
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Acceleration Magnitude", fontsize=9)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)
plt.suptitle("Sex: Acceleration Magnitude â€“ Female vs. Male", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --------------------------------------------------------------------------------
# C.2: IMU â€“ GÃ³c quay
# --------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_fem, df_male],
    [f"Female (Subj: {df_fem['subject'].iloc[0]}, Sex=0)",
     f"Male   (Subj: {df_male['subject'].iloc[0]}, Sex=1)"]
):
    ax.plot(df_seq["sequence_counter"], df_seq["rot_angle"],
            color="tab:orange", lw=1.5, label="rot_angle (rad)")
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Rotation Angle (rad)", fontsize=9)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)
plt.suptitle("Sex: Rotation Angle â€“ Female vs. Male", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --------------------------------------------------------------------------------
# C.3: Nhiá»‡t Ä‘iá»‡n â€“ 5 kÃªnh + Trung bÃ¬nh
# --------------------------------------------------------------------------------
for df in (df_fem, df_male):
    thm_cols = [f"thm_{i}" for i in range(1,6)]
    df["thm_mean"] = df[thm_cols].mean(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_fem, df_male],
    [f"Female (Subj: {df_fem['subject'].iloc[0]}, Sex=0)",
     f"Male   (Subj: {df_male['subject'].iloc[0]}, Sex=1)"]
):
    times = df_seq["sequence_counter"]
    for col in [f"thm_{i}" for i in range(1,6)]:
        ax.plot(times, df_seq[col], lw=1, label=col, alpha=0.7)
    ax.plot(times, df_seq["thm_mean"], color="black", lw=2.5, label="thm_mean")
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Thermopile (Â°C)", fontsize=9)
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    ax.grid(True)
plt.suptitle("Sex: Thermopile â€“ Female vs. Male", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --------------------------------------------------------------------------------
# C.4: ToF â€“ NÄƒm kÃªnh khoáº£ng cÃ¡ch trung bÃ¬nh
# --------------------------------------------------------------------------------
for df in (df_fem, df_male):
    for i_sensor in range(1,6):
        pixel_cols = [f"tof_{i_sensor}_v{pix}" for pix in range(64)]
        df[f"tof_{i_sensor}_mean"] = df[pixel_cols].replace(-1, np.nan).mean(axis=1)

mean_cols = [f"tof_{i}_mean" for i in range(1,6)]
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_fem, df_male],
    [f"Female (Subj: {df_fem['subject'].iloc[0]}, Sex=0)",
     f"Male   (Subj: {df_male['subject'].iloc[0]}, Sex=1)"]
):
    times = df_seq["sequence_counter"]
    for col in mean_cols:
        ax.plot(times, df_seq[col], lw=1, label=col)
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("ToF Mean Distance", fontsize=9)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)
plt.suptitle("Sex: ToF Mean Distance â€“ Female vs. Male", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# â”€â”€â”€ TÃ³m táº¯t cáº¥p Ä‘á»™ chuá»—i cho cáº³ng tay ngáº¯n so vá»›i cáº³ng tay dÃ i â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ChÃºng tÃ´i xÃ¡c Ä‘á»‹nh 4 sá»‘ liá»‡u Ä‘Æ°á»£c Ä‘iá»�u chá»‰nh nhÆ° trÆ°á»›c, nhÆ°ng nhÃ³m theo
# 'ngáº¯n' = cm tá»« vai Ä‘áº¿n cá»• tay â‰¤ trung vá»‹
# 'dÃ i' = cm tá»« vai Ä‘áº¿n cá»• tay > trung vá»‹

# 1) Chá»�n cá»­ chá»‰ (giá»‘ng nhÆ° váº½ Ã´)
gesture_to_plot = "Eyelash - pull hair"
df_g = train_df[train_df["gesture"] == gesture_to_plot].copy()

# 2) XÃ¡c Ä‘á»‹nh chiá»�u dÃ i cáº³ng tay trung bÃ¬nh
median_len = df_g["shoulder_to_wrist_cm"].median()

# 3) TÃ³m táº¯t xÃ¢y dá»±ng
records = []
for seq_id, seq in train_df[train_df["gesture"] == gesture_to_plot].groupby("sequence_id"):
    L = seq["shoulder_to_wrist_cm"].iloc[0]
    grp = "Short" if L <= median_len else "Long"
    # IMU jerk
    acc_mag = np.sqrt(seq["acc_x"]**2 + seq["acc_y"]**2 + seq["acc_z"]**2)
    jerks = np.abs(np.diff(acc_mag))
    max_jerk = jerks.max() if len(jerks)>0 else np.nan
    # Pháº¡m vi quay trong Cá»­ chá»‰
    rot = 2*np.arccos(seq["rot_w"].clip(-1,1))
    rot_rng = rot[seq["phase"]=="Gesture"].max() - rot[seq["phase"]=="Gesture"].min() \
              if (seq["phase"]=="Gesture").any() else np.nan
    # Delta Ä‘á»‰nh nhiá»‡t Ä‘iá»‡n
    thm_cols = [f"thm_{i}" for i in range(1,6)]
    thm_mean = seq[thm_cols].mean(axis=1)
    base = thm_mean[seq["phase"]=="Transition"].median()
    peak_dt = thm_mean[seq["phase"]=="Gesture"].max() - base \
              if (seq["phase"]=="Gesture").any() else np.nan
    # ToF min trung bÃ¬nh
    tof_means = []
    for i in range(1,6):
        pix = seq[[f"tof_{i}_v{p}" for p in range(64)]].replace(-1,np.nan)
        tof_means.append(pix.mean(axis=1))
    tof_all = pd.concat(tof_means,axis=1).mean(axis=1)
    min_tof = tof_all.min()
    records.append({
        "Group":         grp,
        "max_jerk":      max_jerk,
        "rot_range":     rot_rng,
        "thm_peak_dt":   peak_dt,
        "tof_min_dist":  min_tof
    })

summary = pd.DataFrame(records).dropna()

# â”€â”€â”€ Váº½ violin+box Ä‘Æ°á»£c thiáº¿t káº¿ riÃªng cho tá»«ng sá»‘ liá»‡u â”€ ...
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14,10))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

metrics = [
    ("max_jerk",     "Max Jerk |Î”acc| (m/sÂ²)"),
    ("rot_range",    "Rotation Range in Gesture (rad)"),
    ("thm_peak_dt",  "Thermopile Peak Î”T (Â°C)"),
    ("tof_min_dist", "Minimum ToF Mean Dist.")
]

for ax, (col, label) in zip(axes.flatten(), metrics):
    sns.violinplot(data=summary, x="Group", y=col, palette="Set2", ax=ax, inner=None)
    sns.boxplot(data=summary, x="Group", y=col,
                width=0.2, showcaps=True, boxprops={'facecolor':'white'},
                showfliers=False, whiskerprops={'linewidth':2}, ax=ax)
    ax.set_title(label, fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel(label, fontsize=10)

fig.suptitle(f"Forearm Length Comparison (â‰¤{median_len:.1f}cm vs >{median_len:.1f}cm) (*Gesture Phase Only)", fontsize=16)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# â”€â”€â”€ TÃ­nh toÃ¡n thá»�i lÆ°á»£ng pha cho má»—i chuá»—i cho táº¥t cáº£ cÃ¡c chuá»—i â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Giáº£ sá»­ train_df Ä‘Ã£ cÃ³ cá»™t shoulder_to_wrist_cm Ä‘Æ°á»£c há»£p nháº¥t vÃ  cá»™t 'pha'

# Ä�Ã¡nh dáº¥u má»—i chuá»—i lÃ  CÃ¡nh tay ngáº¯n so vá»›i CÃ¡nh tay dÃ i dá»±a trÃªn chiá»�u dÃ i trung bÃ¬nh cá»§a vai_to_wrist_cm trÃªn táº¥t cáº£ cÃ¡c chuá»—i
med = train_df.groupby('sequence_id')['shoulder_to_wrist_cm'].first().median()
seq_meta = (
    train_df
    .groupby('sequence_id')
    .agg({'shoulder_to_wrist_cm':'first'})
    .reset_index()
)
seq_meta['ArmGroup'] = np.where(seq_meta['shoulder_to_wrist_cm'] <= med,
                                'Short Arm', 'Long Arm')

# Ä�áº¿m sá»‘ khung hÃ¬nh trÃªn má»—i pha trÃªn má»—i chuá»—i
dur = (
    train_df
    .groupby(['sequence_id','phase'])['sequence_counter']
    .count()
    .unstack(fill_value=0)
    .reset_index()
)

# Há»£p nháº¥t trong nhÃ£n nhÃ³m cÃ¡nh tay
dur = dur.merge(seq_meta[['sequence_id','ArmGroup']], on='sequence_id')

# Tan cháº£y thÃ nh dáº¡ng dÃ i
dur_long = dur.melt(id_vars=['sequence_id','ArmGroup'],
                    value_vars=['Transition','Gesture'],
                    var_name='Phase', value_name='Duration')

# â”€â”€â”€ Váº½ biá»ƒu Ä‘á»“ há»™p theo chiá»�u ngang: CÃ¡nh tay ngáº¯n so vá»›i cÃ¡nh tay dÃ i â”€ ...
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14,5), sharey=True)
for ax, grp in zip(axes, ['Short Arm','Long Arm']):
    sub = dur_long[dur_long['ArmGroup']==grp]
    sns.boxplot(x='Duration', y='Phase', data=sub,
                palette=['lightgray','lightcoral'], ax=ax)
    ax.set_title(f"{grp} (â‰¤ {med:.1f} cm)", fontsize=12)
    ax.set_xlabel("Duration (frames)", fontsize=10)
    ax.set_ylabel("Phase", fontsize=10)
    ax.grid(True)

plt.suptitle("Phase Durations by Arm Length Group", fontsize=14, y=1.05)
plt.tight_layout()
plt.show()


# â”€â”€â”€ Ã” D: So sÃ¡nh chiá»�u dÃ i tá»« vai Ä‘áº¿n cá»• tay (ngáº¯n so vá»›i dÃ i) Ä‘á»‘i vá»›i IMU, Rot, Thermopile vÃ  ToF â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# (2) Chá»�n má»™t cá»­ chá»‰ Ä‘á»ƒ so sÃ¡nh (thay Ä‘á»•i náº¿u cáº§n)
gesture_to_plot = "Eyelash - pull hair"

# (3) Lá»�c theo cá»­ chá»‰ Ä‘Ã³ vÃ  chia theo chiá»�u dÃ i tá»« vai Ä‘áº¿n cá»• tay báº±ng cÃ¡ch sá»­ dá»¥ng trung vá»‹
df_gesture = train_df[train_df["gesture"] == gesture_to_plot].copy()
median_shoulder = df_gesture["shoulder_to_wrist_cm"].median()

short_group = df_gesture[df_gesture["shoulder_to_wrist_cm"] <= median_shoulder]
long_group  = df_gesture[df_gesture["shoulder_to_wrist_cm"]  > median_shoulder]

if (short_group["sequence_id"].nunique() == 0) or (long_group["sequence_id"].nunique() == 0):
    raise ValueError("Not enough examples in one of the lengthâ€�groups. Try another gesture or adjust threshold.")

short_seq = short_group["sequence_id"].unique()[0]
long_seq  = long_group["sequence_id"].unique()[0]

df_short = train_df[train_df["sequence_id"] == short_seq].sort_values("sequence_counter")
df_long  = train_df[train_df["sequence_id"] == long_seq].sort_values("sequence_counter")

# TÃ­nh toÃ¡n cÃ¡c tÃ­nh nÄƒng cÃ³ nguá»“n gá»‘c tá»« IMU
for df in (df_short, df_long):
    df["acc_mag"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
    df["rot_w_clipped"] = df["rot_w"].clip(-1, 1)
    df["rot_angle"] = 2 * np.arccos(df["rot_w_clipped"])

# TÃ¡i sá»­ dá»¥ng chá»©c nÄƒng Ä‘á»• bÃ³ng
def shade_phases(ax, seq_df):
    used = set()
    for phase_label, color in [("Transition", "lightgray"), ("Gesture", "lightcoral")]:
        mask = seq_df["phase"] == phase_label
        if not mask.any():
            continue
        idxs = seq_df.index[mask]
        diffs = np.where(np.diff(idxs) != 1)[0]
        spans = []
        start = idxs[0]
        for b in diffs:
            end = idxs[b]
            spans.append((start, end))
            start = idxs[b + 1]
        spans.append((start, idxs[-1]))
        for (s, e) in spans:
            t0 = seq_df.loc[s, "sequence_counter"]
            t1 = seq_df.loc[e, "sequence_counter"]
            label = phase_label if phase_label not in used else None
            ax.axvspan(t0, t1, color=color, alpha=0.3, label=label)
            used.add(phase_label)

# --------------------------------------------------------------------------------
# D.1: IMU â€“ Ä�á»™ lá»›n gia tá»‘c
# --------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_short, df_long],
    [f"Short Arm (Shoulderâ€�Wrist â‰¤ {median_shoulder:.1f} cm)",
     f"Long Arm (Shoulderâ€�Wrist > {median_shoulder:.1f} cm)"]
):
    ax.plot(df_seq["sequence_counter"], df_seq["acc_mag"],
            color="tab:blue", lw=1.5, label="|acc|(m/sÂ²)")
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Acceleration Magnitude", fontsize=9)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)
plt.suptitle("Arm Length: Acceleration Magnitude â€“ Short vs. Long", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --------------------------------------------------------------------------------
# D.2: IMU â€“ GÃ³c quay
# --------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_short, df_long],
    [f"Short Arm (Shoulderâ€�Wrist â‰¤ {median_shoulder:.1f} cm)",
     f"Long Arm (Shoulderâ€�Wrist > {median_shoulder:.1f} cm)"]
):
    ax.plot(df_seq["sequence_counter"], df_seq["rot_angle"],
            color="tab:orange", lw=1.5, label="rot_angle (rad)")
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Rotation Angle (rad)", fontsize=9)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)
plt.suptitle("Arm Length: Rotation Angle â€“ Short vs. Long", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --------------------------------------------------------------------------------
# D.3: Nhiá»‡t Ä‘iá»‡n â€“ 5 kÃªnh + Trung bÃ¬nh
# --------------------------------------------------------------------------------
for df in (df_short, df_long):
    thm_cols = [f"thm_{i}" for i in range(1,6)]
    df["thm_mean"] = df[thm_cols].mean(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_short, df_long],
    [f"Short Arm (Shoulderâ€�Wrist â‰¤ {median_shoulder:.1f} cm)",
     f"Long Arm (Shoulderâ€�Wrist > {median_shoulder:.1f} cm)"]
):
    times = df_seq["sequence_counter"]
    for col in [f"thm_{i}" for i in range(1,6)]:
        ax.plot(times, df_seq[col], lw=1, label=col, alpha=0.7)
    ax.plot(times, df_seq["thm_mean"], color="black", lw=2.5, label="thm_mean")
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("Thermopile (Â°C)", fontsize=9)
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    ax.grid(True)
plt.suptitle("Arm Length: Thermopile â€“ Short vs. Long", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --------------------------------------------------------------------------------
# D.4: ToF â€“ NÄƒm kÃªnh khoáº£ng cÃ¡ch trung bÃ¬nh
# --------------------------------------------------------------------------------
for df in (df_short, df_long):
    for i_sensor in range(1,6):
        pixel_cols = [f"tof_{i_sensor}_v{pix}" for pix in range(64)]
        df[f"tof_{i_sensor}_mean"] = df[pixel_cols].replace(-1, np.nan).mean(axis=1)

mean_cols = [f"tof_{i}_mean" for i in range(1,6)]
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, df_seq, title in zip(
    axes,
    [df_short, df_long],
    [f"Short Arm (Shoulderâ€�Wrist â‰¤ {median_shoulder:.1f} cm)",
     f"Long Arm (Shoulderâ€�Wrist > {median_shoulder:.1f} cm)"]
):
    times = df_seq["sequence_counter"]
    for col in mean_cols:
        ax.plot(times, df_seq[col], lw=1, label=col)
    shade_phases(ax, df_seq)
    ax.set_title(f"{title}\nGesture: {gesture_to_plot}", fontsize=11)
    ax.set_xlabel("Sequence Counter", fontsize=9)
    ax.set_ylabel("ToF Mean Distance", fontsize=9)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)
plt.suptitle("Arm Length: ToF Mean Distance â€“ Short vs. Long", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Káº¿t há»£p Ä‘Ã o táº¡o vÃ  kiá»ƒm tra cho cáº£ hai táº­p dá»¯ liá»‡u
combined_main = pd.concat([train_df, test_df], axis=0)
combined_demo = pd.concat([train_dem_df, test_dem_df], axis=0)

# Ä�áº·t má»¥c tiÃªu
target_variable = 'gesture'

# XÃ¡c Ä‘á»‹nh tiá»�n tá»‘ cáº£m biáº¿n vÃ  danh sÃ¡ch loáº¡i trá»«
sensor_prefixes = ('acc_', 'rot_', 'thm_', 'tof_')
excluded_columns = [target_variable]

# XÃ¡c Ä‘á»‹nh cÃ¡c biáº¿n sá»‘ cÃ³ thá»ƒ sá»­ dá»¥ng (khÃ´ng pháº£i cáº£m biáº¿n, chá»‰ sá»‘)
def get_numerical_columns(df, exclude_prefixes, excluded_cols):
    return [col for col in df.columns 
            if pd.api.types.is_numeric_dtype(df[col]) 
            and not col.startswith(exclude_prefixes)
            and col not in excluded_cols]

main_vars = get_numerical_columns(combined_main, sensor_prefixes, excluded_columns)
demo_vars = get_numerical_columns(combined_demo, (), excluded_columns)

# Káº¿t há»£p cáº£ hai Ä‘á»ƒ cÃ³ má»‘i tÆ°Æ¡ng quan toÃ n cáº§u
combined_data = pd.concat([
    combined_main[main_vars].reset_index(drop=True),
    combined_demo[demo_vars].reset_index(drop=True)
], axis=1)

# TÃ­nh toÃ¡n tÆ°Æ¡ng quan vÃ  che dáº¥u tam giÃ¡c trÃªn
corr_all = combined_data.corr()
mask_all = np.triu(np.ones_like(corr_all, dtype=bool))

# Plot
plt.figure(figsize=(18, 10))
ax = sns.heatmap(
    corr_all, mask=mask_all, cmap='viridis', annot=True, 
    square=False, linewidths=.5, annot_kws={"size": 12}
)
plt.title('Correlation Heatmap â€” Combined (Demographic + Main) Data', fontsize=16)

# Rotate x-axis tick labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)

plt.tight_layout()
plt.show()


from sklearn.preprocessing import LabelEncoder

# Sao chÃ©p vÃ  mÃ£ hÃ³a má»¥c tiÃªu
train_corr_df = train_df.copy()
train_corr_df[target_variable] = LabelEncoder().fit_transform(train_corr_df[target_variable])

# TrÃ­ch xuáº¥t cÃ¡c Ä‘áº·c Ä‘iá»ƒm sá»‘ há»£p lá»‡ chá»‰ tá»« tÃ u há»�a
train_main_vars = get_numerical_columns(train_corr_df, sensor_prefixes, excluded_columns)

# Bao gá»“m cáº£ cÃ¡c Ä‘áº·c Ä‘iá»ƒm nhÃ¢n kháº©u há»�c
train_demo_vars = get_numerical_columns(train_dem_df, (), excluded_columns)

# Káº¿t há»£p cáº£ hai
train_all_corr = pd.concat([train_corr_df[train_main_vars], train_dem_df[train_demo_vars],
                            train_corr_df[[target_variable]]], axis=1)

# Chá»‰ tÆ°Æ¡ng quan vá»›i má»¥c tiÃªu
corr_target_only = train_all_corr.corr()[[target_variable]].T

# Plot
plt.figure(figsize=(12, 3))
sns.heatmap(corr_target_only, cmap='viridis', annot=True, linewidths=0.5, cbar=False, annot_kws={"size": 10})
plt.title("Correlation with Target (gesture) â€” Train Data", fontsize=13)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()




