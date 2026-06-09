# --- Installing Libraries ---
!pip install ydata-profiling
!pip install Pillow


# ----- Handling data -----
import pandas as pd
import numpy as np


# ----- Graphics -----
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# ----- EDA Univariate -----
from ydata_profiling import ProfileReport


# ----- Remove the warnings -----
import warnings


# Remove the warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ----- Read the dataset -----
df_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col="id")
df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col="id")


df_train.head(2).style.background_gradient(cmap='Blues').hide(axis="index")


df_test.head(2).style.background_gradient(cmap='Blues').hide(axis="index")


# ----- Dataset Report with ProfileReport for the train set -----

ProfileReport(df_train, title='Train Dataset', 
              minimal = False, 
              progress_bar = False, 
              samples = None, 
              interactions = None,
              correlations = None,
              explorative = True,
              notebook = {'iframe':{'height': '600px'}},
              missing_diagrams = {'heatmap': False, 'dendrogram': True}).to_notebook_iframe()


# ----- Dataset Report with ProfileReport for the test set -----

ProfileReport(df_test, title='Train Dataset', 
              minimal = False, 
              progress_bar = False, 
              samples = None, 
              interactions = None,
              correlations = None,
              explorative = True,
              notebook = {'iframe':{'height': '600px'}},
              missing_diagrams = {'heatmap': False, 'dendrogram': True}).to_notebook_iframe()


# ----- Handling missing values -----

winddirection_median = df_train['winddirection'].median()

df_test['winddirection'].fillna(winddirection_median, inplace=True)


# ----- Checking missing values -----

missing_values = df_test['windspeed'].isnull().sum()
print(f"Number of missing values in 'windspeed' : {missing_values}")


# ----- List of accumulated days for each month -----

days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

month_names = ["January", "February", "March", "April", "May", "June",  
               "July", "August", "September", "October", "November", "December"]



# ----- Function to create the “month” variable -----

def add_month_column(df):
    df['month'] = df['day'].apply(
        lambda x: month_names[next(i for i, days in enumerate(days_per_month, start=0) 
                                   if x <= sum(days_per_month[:i+1]))]
    )
    return df


# ----- Apply the transformation to a DataFrame -----
df_train = add_month_column(df_train)


# ----- Read the dataset -----
df_train.head(5).style.background_gradient(cmap='Blues').hide(axis="index")


def correlation_matrix_3(df, color_0, color_1, color_2, subtitle_1, subtitle_2, subtitle_3):
    
    # Correlation matrix from -1 to 1 (with 3 colors)
    
    # If we want a correlation matrix from 0 to 1, we need 2 colors
    
    # Import the library: import matplotlib.colors as mcolors
    
    # Highlight text properties
    highlight_textprops = [{"fontsize":12, "color":f'#{color_0}', "fontname": "Cover sans", "fontweight": "bold"},
                           {"fontsize":12, "color":f'#33363F', "fontname": "Cover sans"},
                           {"fontsize":12, "color":f'#{color_0}', "fontname": "Cover sans", "fontweight": "bold"},]


    # Axis labels color
    variable_name_textprops = [{"fontsize":8, "color":f'#33363F', "fontname": "Cover sans", "fontweight": "bold"}]
    
    # Correlation matrix
    correlation_matrix = df.corr(numeric_only=True)

    # Figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=400)
    

    # Remove the upper half of the matrix
    mask = np.triu(np.ones_like(df.corr(numeric_only=True), dtype=bool))

    
    color1 = mcolors.to_rgba(f'#{color_0}')  # Negative value -1
    color_intermediate = mcolors.to_rgba(f'#{color_1}')  # Intermediate color (Value 0)
    color2 = mcolors.to_rgba(f'#{color_2}')  # Positive value 1

    
    # Create a custom color palette
    n, m = 256, 1
    cmap_custom = mcolors.LinearSegmentedColormap.from_list('custom', [color1, color_intermediate, color2], N=n, gamma=m)

    # Correlation matrix heatmap
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap=cmap_custom, fmt=".2f", linewidths=0.2, cbar=False,
               annot_kws={"size": 10})
    
    
    # Horizontal and vertical labels
    xy_label = dict(size=6)
    yticks, ylabels = plt.yticks()
    xticks, xlabels = plt.xticks()
    ax.set_xticklabels(xlabels, rotation=0, **xy_label, **variable_name_textprops[0])
    ax.set_yticklabels(ylabels, **xy_label, **variable_name_textprops[0])
    
    
    # Add a title to the heatmap axis
    ax.set_title('Correlation of Numerical Variables', fontsize=20, fontweight='bold', 
             fontname='Lisboa Sans OSF', color = "#33363F")
    
    
    # Title
    # Ajouter du texte à l'axe avec ax.text()
    ax.text(0.40, 0.845, f"{subtitle_1} {subtitle_2} {subtitle_3}", 
        va='bottom', ha='center', fontsize=12, 
        bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.3'), 
        color='black')  # Vous pouvez ajuster les propriétés comme la couleur, la taille de police, etc.
    



%matplotlib inline
correlation_matrix_3(df_train,"243B6E", "FFFCF9", "EA7F1B", "", "", "")


# Calculate absolute values for each month and type of rainfall (rain vs no rain)
rainfall_data = df_train.groupby(['month', 'rainfall']).size().unstack(fill_value=0)

# Calculate the percentage for each month
rainfall_percentage = rainfall_data.div(rainfall_data.sum(axis=1), axis=0) * 100

# Graph size and quality
fig = plt.figure(figsize=(16,12), dpi=400)
gs = fig.add_gridspec(4, 3)
ax = fig.add_subplot(gs[:3, :])

# Plot stacked bars
bars = rainfall_percentage.plot(kind='bar', ax=ax, edgecolor='black', width=0.85, linewidth=0.5, alpha=0.70, zorder=1, color="#F3F5FB")

# ------------- Add values on the bars -------------

for i, bar in enumerate(bars.patches):
    # Find bar position
    bar_width = bar.get_width()
    bar_height = bar.get_height()
    x_pos = bar.get_x()
    y_pos = bar.get_y()

    # Add text inside the bar
    if i > 1:  # Check if it's not the first or second bar
        ax.annotate(f"{np.round((bar_height), 1)}%", (x_pos + bar_width / 2, y_pos + bar_height / 2),
                    ha='center', va='center', color='#202020', fontsize=10)
    else:
        ax.annotate(f"{np.round((bar_height), 1)}%", (x_pos + bar_width / 2, bar_height),
                    ha='center', va='bottom', color='#202020', fontsize=10)

        
# ------------- Bar colors -------------   

for i, bar in enumerate(bars.patches):
    if 12 <= i <= 23:  # Rain
        bar.set_facecolor('#0070C0')
        bar.set_alpha(0.75)
    else:
        bar.set_facecolor('#EA7F1B')
        bar.set_alpha(0.85)
        
# --- Remove borders around the graph ---
plt.box(False)

# ---- Remove ticks on x and y axes ----
plt.tick_params(axis='x', which='both', bottom=False, top=False)
plt.tick_params(axis='y', which='both', left=False, right=False)

# ---- Change tick size and color ----
plt.yticks(fontsize=10, color='#202020')
plt.xticks(fontsize=12, color='#202020')

# ------------- Add grid -------------
plt.grid(axis='x', alpha=0)
plt.grid(axis='y', which='major', alpha=0.75, linestyle='dotted', zorder=0)

# --- Rotate x-axis labels ---
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45) # Write category names

# ------ Space out x and y axis titles ------
ax.xaxis.labelpad = 20
ax.yaxis.labelpad = 26

# ------------------------------------------------ Annotation ------------------------------------------------
ax.axhline(y=np.mean(df_train["rainfall"]).round(3)*100, color='#626262', linestyle='--', alpha=0.5)
ax.annotate('Proportion: \n75.3%', xy=(5.3, 76), xytext=(5.5, 80.3),
                 arrowprops=dict(arrowstyle='->', color="#626262"),
                 fontsize=11, color='#626262', fontname="Cover sans")

# ------------ Axis title colors ----------------

# Color and size
highlight_textprops1 = [{"fontsize":18, "color":'#262626', "fontname": "Lisboa Sans OSF", "fontweight": "heavy"}]

# Define x and y axis labels
ax.set_xlabel(f"Month", **highlight_textprops1[-1])
ax.set_ylabel(f"Proportion (%)", **highlight_textprops1[-1])

# Add a legend
ax.legend(["No Rain", "With Rain"], loc="upper right")

# -------------------------- Title --------------------------

# Title color
highlight_textprops = [{"fontsize":26, "color":'#262626', "fontname": "Lisboa Sans OSF", "fontweight": "heavy"},
                           {"fontsize":22, "color":'#202020', "fontname": "Cover sans"},
                           {"fontsize":22, "color":f"#FF4141", "fontname": "Cover sans", "fontweight": "heavy"}]
# Define the title
# Ajouter un titre général pour l'ensemble des graphiques
fig.suptitle("Proportion of Rainy Days per Month of the Year", fontsize=26, 
             fontweight='bold', 
             fontname="Lisboa Sans OSF",
             y=1.01)

fig.text(0.5, 0.96,  # Ajustez la position verticale si nécessaire
         "December, May, and June are the months with the least rain.", 
         ha='center', 
         fontsize=18, 
         color='#202020',  # Vous pouvez ajuster la couleur pour différencier du titre
         fontname="Lisboa Sans OSF")

# -------------------------- Legend --------------------------

# Text properties
font_properties = {'family': 'Cover sans', 'weight': 'normal', 'size': 12}

# Legend
legend_elements = [
    Rectangle((0, 0), 1, 1, color='#EA7F1B', linewidth=0.5, ec="k", label='No Rain', alpha=0.85),
    Rectangle((0, 0), 1, 1, color='#0070C0', linewidth=0.5, ec="k", label='With Rain', alpha=0.75),
    Line2D([0], [0], color='#626262', linestyle='--', linewidth=1, label='Average', alpha=0.9)
]

plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.16), 
fancybox=False, shadow=False, ncol=3, frameon=False, prop=font_properties)

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plot
plt.show()



# ----Taille et qualité du graphique----
fig = plt.figure(figsize=(12, 8), dpi = 400)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])  # Même hauteur pour tous les graphiques
ax1 = fig.add_subplot(gs[0, 0])


highlight_textprops = {"fontsize":10, "color":'#262626', "fontname": "Lisboa Sans OSF", "fontweight": "heavy"}

# ------------------------------------ Graphique 1 (Température minimale) ------------------------------------
temp_rain = df_train[df_train['rainfall'] > 0]['mintemp']
temp_no_rain = df_train[df_train['rainfall'] == 0]['mintemp']
sns.kdeplot(temp_rain, fill = True, linestyle='-', linewidth=1.2, label="With Rain", color='#0070C0', alpha=0.50, ax=ax1)
sns.kdeplot(temp_no_rain, fill = True, linestyle='-', linewidth=1.2, label="No Rain", color='#EA7F1B', alpha=0.60, ax=ax1)
ax1.set_title("Density of Minimum Temperature", **highlight_textprops, pad=20)

# --------Enlever le cadre en haut et à droite----
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ----Ajouter des grilles----
plt.grid(axis='x', which='major', alpha=0.75, linestyle='dotted', zorder=1)
plt.grid(axis='y', alpha=0, zorder=2)

# ----Ajouter un cadre en bas et à gauche----
ax1.spines['bottom'].set_linewidth(1.2)
ax1.spines['bottom'].set_color('#CAC9CD')

ax1.spines['left'].set_linewidth(1.2)
ax1.spines['left'].set_color('#CAC9CD')

# ----Changer la couleur des barres des graduations sur l'axe x----
ax1.tick_params(axis='x', colors='#CAC9CD', width=1.2)
ax1.tick_params(axis='y', colors='#CAC9CD', width=1.2)

# ----Changer la taille et la couleur des ticks----
plt.xticks(fontsize=6, color='#202020', ha='center', rotation=0)
plt.yticks(fontsize=6, color='#202020')
plt.tick_params(left='on', bottom='on')

# ------Espacer les titres ticks------
ax1.tick_params(axis='y', pad=6)
ax1.tick_params(axis='x', pad=6)

# ------Espacer les titres des axes x et y------
ax1.xaxis.labelpad = 12
ax1.yaxis.labelpad = 16

# Couleur et taille
highlight_textprops1 = [{"fontsize":8, "color":'#262626', "fontname": "Lisboa Sans OSF", "fontweight": "heavy"}]

# Définir le nom de l'axe des x et y
ax1.set_xlabel(f"Temperature (°C)", **highlight_textprops1[-1])
ax1.set_ylabel(f"Density", **highlight_textprops1[-1])

# Annotation
ax1.axvline(x=df_train["mintemp"][df_train.rainfall==0].median(), color='#AF5C11', linestyle='--', alpha = 0.7)
ax1.annotate(f'Median on dry \ndays : \n{df_train["mintemp"][df_train.rainfall==0].median()}°', 
                 xy=(24.65, 0.07), xytext=(30, 0.069),
                 arrowprops=dict(arrowstyle='->', color = "#AF5C11"),
                 fontsize=8, color='#AF5C11', fontname = "Cover sans")

ax1.axvline(x=df_train["mintemp"][df_train.rainfall==1].median(), color='#4B6387', linestyle='--', alpha = 0.7)
ax1.annotate(f'Median on rainy \ndays : \n{df_train["mintemp"][df_train.rainfall==1].median()}°', 
                 xy=(23.5, 0.09), xytext=(13, 0.081),
                 arrowprops=dict(arrowstyle='->', color = "#4B6387"),
                 fontsize=8, color='#4B6387', fontname = "Cover sans")
# ------------------------------------------------------------------------------------------------------------






# ------------------------------------ Graphique 2 (Température maximale) ------------------------------------
ax2 = fig.add_subplot(gs[0, 1])  # En bas à gauche
temp_rain = df_train[df_train['rainfall'] > 0]['maxtemp']
temp_no_rain = df_train[df_train['rainfall'] == 0]['maxtemp']
sns.kdeplot(temp_rain, fill = True, linestyle='-', linewidth=1.2, label="Avec pluie", color='#0070C0', alpha=0.50, ax=ax2)
sns.kdeplot(temp_no_rain, fill = True, linestyle='-', linewidth=1.2, label="Sans pluie", color='#EA7F1B', alpha=0.60, ax=ax2)
ax2.set_title("Density of Maximum Temperature", **highlight_textprops, pad=20)

# --------Enlever le cadre en haut et à droite----
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# ----Ajouter des grilles----
plt.grid(axis='x', which='major', alpha=0.75, linestyle='dotted', zorder=1)
plt.grid(axis='y', alpha=0, zorder=2)

# ----Ajouter un cadre en bas et à gauche----
ax2.spines['bottom'].set_linewidth(1.2)
ax2.spines['bottom'].set_color('#CAC9CD')

ax2.spines['left'].set_linewidth(1.2)
ax2.spines['left'].set_color('#CAC9CD')

# ----Changer la couleur des barres des graduations sur l'axe x----
ax2.tick_params(axis='x', colors='#CAC9CD', width=1.2)
ax2.tick_params(axis='y', colors='#CAC9CD', width=1.2)

# ----Changer la taille et la couleur des ticks----
plt.xticks(fontsize=6, color='#202020', ha='center', rotation=0)
plt.yticks(fontsize=6, color='#202020')
plt.tick_params(left='on', bottom='on')

# ------Espacer les titres ticks------
ax2.tick_params(axis='y', pad=6)
ax2.tick_params(axis='x', pad=6)

# ------Espacer les titres des axes x et y------
ax2.xaxis.labelpad = 12
ax2.yaxis.labelpad = 16

# Couleur et taille
highlight_textprops1 = [{"fontsize":8, "color":'#262626', "fontname": "Lisboa Sans OSF", "fontweight": "heavy"}]

# Définir le nom de l'axe des x et y
ax2.set_xlabel(f"Temperature (°C)", **highlight_textprops1[-1])
ax2.set_ylabel(f"", **highlight_textprops1[-1])

# Annotation
ax2.axvline(x=df_train["maxtemp"][df_train.rainfall==0].median(), color='#AF5C11', linestyle='--', alpha = 0.7)
ax2.annotate(f'Median on dry \ndays : \n{df_train["maxtemp"][df_train.rainfall==0].median()}°', 
                 xy=(29, 0.08), xytext=(35, 0.069),
                 arrowprops=dict(arrowstyle='->', color = "#AF5C11"),
                 fontsize=8, color='#AF5C11', fontname = "Cover sans")

ax2.axvline(x=df_train["maxtemp"][df_train.rainfall==1].median(), color='#4B6387', linestyle='--', alpha = 0.7)
ax2.annotate(f'Median on rainy \ndays : \n{df_train["maxtemp"][df_train.rainfall==1].median()}°', 
                 xy=(27.4, 0.07), xytext=(16, 0.069),
                 arrowprops=dict(arrowstyle='->', color = "#4B6387"),
                 fontsize=8, color='#4B6387', fontname = "Cover sans")
# ------------------------------------------------------------------------------------------------------------







# ------------------------------------ Graphique 3 (Température moyenne) ------------------------------------
ax3 = fig.add_subplot(gs[1, :])  # En bas à droite
temp_rain = df_train[df_train['rainfall'] > 0]['temparature']
temp_no_rain = df_train[df_train['rainfall'] == 0]['temparature']
sns.kdeplot(temp_rain, fill = True, linestyle='-', linewidth=1.2, label="With Rain", color='#0070C0', alpha=0.50, ax=ax3)
sns.kdeplot(temp_no_rain, fill = True, linestyle='-', linewidth=1.2, label="No Rain", color='#EA7F1B', alpha=0.60, ax=ax3)
ax3.set_title("Density of Average Temperature", **highlight_textprops, pad=20)

# --------Enlever le cadre en haut et à droite----
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# ----Ajouter des grilles----
plt.grid(axis='x', which='major', alpha=0.75, linestyle='dotted', zorder=1)
plt.grid(axis='y', alpha=0, zorder=2)

# ----Ajouter un cadre en bas et à gauche----
ax3.spines['bottom'].set_linewidth(1.2)
ax3.spines['bottom'].set_color('#CAC9CD')

ax3.spines['left'].set_linewidth(1.2)
ax3.spines['left'].set_color('#CAC9CD')

# ----Changer la couleur des barres des graduations sur l'axe x----
ax3.tick_params(axis='x', colors='#CAC9CD', width=1.2)
ax3.tick_params(axis='y', colors='#CAC9CD', width=1.2)

# ----Changer la taille et la couleur des ticks----
plt.xticks(fontsize=6, color='#202020', ha='center', rotation=0)
plt.yticks(fontsize=6, color='#202020')
plt.tick_params(left='on', bottom='on')

# ------Espacer les titres ticks------
ax3.tick_params(axis='y', pad=6)
ax3.tick_params(axis='x', pad=6)

# ------Espacer les titres des axes x et y------
ax3.xaxis.labelpad = 12
ax3.yaxis.labelpad = 16

# Couleur et taille
highlight_textprops1 = [{"fontsize":8, "color":'#262626', "fontname": "Lisboa Sans OSF", "fontweight": "heavy"}]

# Définir le nom de l'axe des x et y
ax3.set_xlabel(f"Temperature (°C)", **highlight_textprops1[-1])
ax3.set_ylabel(f"Density", **highlight_textprops1[-1])

# Annotation
ax3.axvline(x=df_train["temparature"][df_train.rainfall==0].median(), color='#AF5C11', linestyle='--', alpha = 0.7)
ax3.annotate(f'Median on dry \ndays : \n{df_train["temparature"][df_train.rainfall==0].median()}°', 
                 xy=(26.4, 0.09), xytext=(31, 0.08),
                 arrowprops=dict(arrowstyle='->', color = "#AF5C11"),
                 fontsize=8, color='#AF5C11', fontname = "Cover sans")

ax3.axvline(x=df_train["temparature"][df_train.rainfall==1].median(), color='#4B6387', linestyle='--', alpha = 0.7)
ax3.annotate(f'Median on rainy \ndays : \n{df_train["temparature"][df_train.rainfall==1].median()}°', 
                 xy=(25.2, 0.08), xytext=(20, 0.069),
                 arrowprops=dict(arrowstyle='->', color = "#4B6387"),
                 fontsize=8, color='#4B6387', fontname = "Cover sans")
# ------------------------------------------------------------------------------------------------------------



# --------------------------Légende--------------------------

# Texte
font_properties = {'family': 'Cover sans', 'weight': 'normal', 'size': 10}
# Légende
legend_elements = [
    Rectangle((0, 0), 1, 1, color='#0070C0', linewidth=0.5, ec="k", label='With Rain', alpha=0.50),
    Rectangle((0, 0), 1, 1, color='#EA7F1B', linewidth=0.5, ec="k", label='No Rain', alpha=0.60)
]


plt.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.34), 
fancybox=False, shadow=False, ncol=3, frameon=False, prop=font_properties)



# Ajouter un titre général pour l'ensemble des graphiques
fig.suptitle("Analysis of Rainfall and Temperatures: Minimum, Maximum, Average", fontsize=22, 
             fontweight='bold', 
             fontname="Lisboa Sans OSF",
             y=1.01)


plt.subplots_adjust(hspace=0.45) 
plt.subplots_adjust(wspace=0.25)

# Ajustement du layout
#plt.tight_layout()
plt.show()


# Filter data based on rainfall
rain_data = df_train[df_train['rainfall'] > 0]
no_rain_data = df_train[df_train['rainfall'] == 0]

# Crée la figure et la disposition de la grille
fig = plt.figure(figsize=(12, 8), dpi = 400)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])  # 2 lignes, 2 colonnes

highlight_textprops = {"fontsize":10, "color":'#262626', "fontname": "Lisboa Sans OSF", "fontweight": "heavy"}



# --------------------------- Graphique 1 (Wind rose avec pluie) ---------------------------
ax1 = fig.add_subplot(gs[0, 0], projection='polar')
ax1.set_theta_direction(-1)
ax1.set_theta_offset(np.pi / 2.0)
ax1.bar(
    np.deg2rad(rain_data['winddirection']),
    rain_data['windspeed'],
    width=np.pi/8,
    bottom=0.0,
    color='#0070C0', 
    alpha=0.50,
)
ax1.set_title('Wind Speed and Direction with Rain', **highlight_textprops, pad=20)

ax1.tick_params(axis='x', colors='#262626', labelsize=8)  # Axe angulaire (direction du vent)
ax1.tick_params(axis='y', colors='#262626', labelsize=6)  # Axe radial (vitesse du vent)
# ------------------------------------------------------------------------------------------------------------




# --------------------------- Graphique 2 (Wind rose sans pluie) ---------------------------
ax2 = fig.add_subplot(gs[0, 1], projection='polar')
ax2.set_theta_direction(-1)
ax2.set_theta_offset(np.pi / 2.0)
ax2.bar(
    np.deg2rad(no_rain_data['winddirection']),
    no_rain_data['windspeed'],
    width=np.pi/8,
    bottom=0.0,
    color='#EA7F1B', 
    alpha=0.60
)
ax2.set_title('Wind Speed and Direction without Rain', **highlight_textprops, pad=20)

ax2.tick_params(axis='x', colors='#262626', labelsize=8)  # Axe angulaire (direction du vent)
ax2.tick_params(axis='y', colors='#262626', labelsize=6)  # Axe radial (vitesse du vent)
# ------------------------------------------------------------------------------------------------------------




# --------------------------- Graphique 3 (Température moyenne) ---------------------------
ax3 = fig.add_subplot(gs[1, :])  # En bas à droite
wind_rain = df_train[df_train['rainfall'] > 0]['windspeed']
wind_no_rain = df_train[df_train['rainfall'] == 0]['windspeed']
sns.kdeplot(wind_rain, fill = True, linestyle='-', linewidth=1.2, label="With Rain", color='#0070C0', alpha=0.50, ax=ax3)
sns.kdeplot(wind_no_rain, fill = True, linestyle='-', linewidth=1.2, label="No Rain", color='#EA7F1B', alpha=0.60, ax=ax3)
ax3.set_title("Wind Speed Density in km/h", **highlight_textprops, pad=20)

# --------Enlever le cadre en haut et à droite----
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# ----Ajouter des grilles----
plt.grid(axis='x', which='major', alpha=0.75, linestyle='dotted', zorder=1)
plt.grid(axis='y', alpha=0, zorder=2)

# ----Ajouter un cadre en bas et à gauche----
ax3.spines['bottom'].set_linewidth(1.2)
ax3.spines['bottom'].set_color('#CAC9CD')

ax3.spines['left'].set_linewidth(1.2)
ax3.spines['left'].set_color('#CAC9CD')

# ----Changer la couleur des barres des graduations sur l'axe x----
ax3.tick_params(axis='x', colors='#CAC9CD', width=1.2)
ax3.tick_params(axis='y', colors='#CAC9CD', width=1.2)

# ----Changer la taille et la couleur des ticks----
plt.xticks(fontsize=6, color='#202020', ha='center', rotation=0)
plt.yticks(fontsize=6, color='#202020')
plt.tick_params(left='on', bottom='on')

# ------Espacer les titres ticks------
ax3.tick_params(axis='y', pad=6)
ax3.tick_params(axis='x', pad=6)

# ------Espacer les titres des axes x et y------
ax3.xaxis.labelpad = 12
ax3.yaxis.labelpad = 16

# Couleur et taille
highlight_textprops1 = [{"fontsize":8, "color":'#262626', "fontname": "Lisboa Sans OSF", "fontweight": "heavy"}]

# Définir le nom de l'axe des x et y
ax3.set_xlabel(f"Wind (km/h)", **highlight_textprops1[-1])
ax3.set_ylabel(f"Density", **highlight_textprops1[-1])

# Annotation
ax3.axvline(x=df_train["windspeed"][df_train.rainfall==0].median(), color='#AF5C11', linestyle='--', alpha = 0.8)
ax3.annotate(f'Median on dry \ndays : \n{df_train["windspeed"][df_train.rainfall==0].median()} km/h', 
                 xy=(16.6, 0.045), xytext=(4, 0.042),
                 arrowprops=dict(arrowstyle='->', color = "#AF5C11"),
                 fontsize=8, color='#AF5C11', fontname = "Cover sans")

ax3.axvline(x=df_train["windspeed"][df_train.rainfall==1].median(), color='#4B6387', linestyle='--', alpha = 0.8)
ax3.annotate(f'Median on rainy \ndays : \n{df_train["windspeed"][df_train.rainfall==1].median()} km/h', 
                 xy=(21.8, 0.044), xytext=(26, 0.04),
                 arrowprops=dict(arrowstyle='->', color = "#4B6387"),
                 fontsize=8, color='#4B6387', fontname = "Cover sans")
# ------------------------------------------------------------------------------------------------------------




# Ajouter un titre général pour l'ensemble des graphiques
fig.suptitle("Analysis of Rainfall and Wind: Direction and Speed", fontsize=22, 
             fontweight='bold', 
             fontname="Lisboa Sans OSF",
             y=1.04)


# --------------------------Légende--------------------------

# Texte
font_properties = {'family': 'Cover sans', 'weight': 'normal', 'size': 10}
# Légende
legend_elements = [
    Rectangle((0, 0), 1, 1, color='#0070C0', linewidth=0.5, ec="k", label='With Rain', alpha=0.50),
    Rectangle((0, 0), 1, 1, color='#EA7F1B', linewidth=0.5, ec="k", label='No Rain', alpha=0.60)
]


plt.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.34), 
fancybox=False, shadow=False, ncol=3, frameon=False, prop=font_properties)



plt.subplots_adjust(hspace=0.35) 



plt.show()


# Crée la figure et la disposition de la grille
fig = plt.figure(figsize=(12, 8), dpi = 400)     # 8 ou 10
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])  # Même hauteur pour tous les graphiques

# --------- Graphique 1 (Nuages - Cloud) ---------
ax1 = fig.add_subplot(gs[0, 0])  # Premier graphique
cloud_rain = df_train[df_train['rainfall'] > 0]['cloud']
cloud_no_rain = df_train[df_train['rainfall'] == 0]['cloud']
sns.kdeplot(cloud_rain, fill = True, linestyle='-', linewidth=1.2, label="With Rain", color='#0070C0', alpha=0.50, ax=ax1)
sns.kdeplot(cloud_no_rain, fill = True, linestyle='-', linewidth=1.2, label="No Rain", color='#EA7F1B', alpha=0.60, ax=ax1)
ax1.set_title("Cloud Density", **highlight_textprops, pad=20)


# --------Enlever le cadre en haut et à droite----
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ----Ajouter des grilles----
plt.grid(axis='x', which='major', alpha=0.75, linestyle='dotted', zorder=1)
plt.grid(axis='y', alpha=0, zorder=2)

# ----Ajouter un cadre en bas et à gauche----
ax1.spines['bottom'].set_linewidth(1.2)
ax1.spines['bottom'].set_color('#CAC9CD')

ax1.spines['left'].set_linewidth(1.2)
ax1.spines['left'].set_color('#CAC9CD')

# ----Changer la couleur des barres des graduations sur l'axe x----
ax1.tick_params(axis='x', colors='#CAC9CD', width=1.2)
ax1.tick_params(axis='y', colors='#CAC9CD', width=1.2)

# ----Changer la taille et la couleur des ticks----
plt.xticks(fontsize=6, color='#202020', ha='center', rotation=0)
plt.yticks(fontsize=6, color='#202020')
plt.tick_params(left='on', bottom='on')

# ------Espacer les titres ticks------
ax1.tick_params(axis='y', pad=6)
ax1.tick_params(axis='x', pad=6)

# ------Espacer les titres des axes x et y------
ax1.xaxis.labelpad = 12
ax1.yaxis.labelpad = 16

# Couleur et taille
highlight_textprops1 = [{"fontsize":8, "color":'#262626', "fontname": "Lisboa Sans OSF", "fontweight": "heavy"}]

# Définir le nom de l'axe des x et y
ax1.set_xlabel(f"Cloud (%)", **highlight_textprops1[-1])
ax1.set_ylabel(f"Density", **highlight_textprops1[-1])

# Annotation
ax1.axvline(x=df_train["cloud"][df_train.rainfall==0].median(), color='#AF5C11', linestyle='--', alpha = 0.8)
ax1.annotate(f'Median on dry \ndays : \n{df_train["cloud"][df_train.rainfall==0].median()}%', 
                 xy=(52.00, 0.05), xytext=(16, 0.049),
                 arrowprops=dict(arrowstyle='->', color = "#AF5C11"),
                 fontsize=8, color='#AF5C11', fontname = "Cover sans")

ax1.axvline(x=df_train["cloud"][df_train.rainfall==1].median(), color='#4B6387', linestyle='--', alpha = 0.8)
ax1.annotate(f'Median on rainy \ndays : \n{df_train["cloud"][df_train.rainfall==1].median()}%', 
                 xy=(86.0, 0.05), xytext=(100, 0.049),
                 arrowprops=dict(arrowstyle='->', color = "#4B6387"),
                 fontsize=8, color='#4B6387', fontname = "Cover sans")
# ------------------------------------------------------------------------------------------------------------







# --------- Graphique 2 (Ensoleillement - Sunshine) ---------
ax2 = fig.add_subplot(gs[0, 1])  # Deuxième graphique
sunshine_rain = df_train[df_train['rainfall'] > 0]['sunshine']
sunshine_no_rain = df_train[df_train['rainfall'] == 0]['sunshine']
sns.kdeplot(sunshine_rain, fill = True, linestyle='-', linewidth=1.2, label="With Rain", color='#0070C0', alpha=0.50, ax=ax2)
sns.kdeplot(sunshine_no_rain, fill = True, linestyle='-', linewidth=1.2, label="No Rain", color='#EA7F1B', alpha=0.60, ax=ax2)
ax2.set_title("Sunshine Density", **highlight_textprops, pad=20)

# --------Enlever le cadre en haut et à droite----
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# ----Ajouter des grilles----
plt.grid(axis='x', which='major', alpha=0.75, linestyle='dotted', zorder=1)
plt.grid(axis='y', alpha=0, zorder=2)

# ----Ajouter un cadre en bas et à gauche----
ax2.spines['bottom'].set_linewidth(1.2)
ax2.spines['bottom'].set_color('#CAC9CD')

ax2.spines['left'].set_linewidth(1.2)
ax2.spines['left'].set_color('#CAC9CD')

# ----Changer la couleur des barres des graduations sur l'axe x----
ax2.tick_params(axis='x', colors='#CAC9CD', width=1.2)
ax2.tick_params(axis='y', colors='#CAC9CD', width=1.2)

# ----Changer la taille et la couleur des ticks----
plt.xticks(fontsize=6, color='#202020', ha='center', rotation=0)
plt.yticks(fontsize=6, color='#202020')
plt.tick_params(left='on', bottom='on')

# ------Espacer les titres ticks------
ax2.tick_params(axis='y', pad=6)
ax2.tick_params(axis='x', pad=6)

# ------Espacer les titres des axes x et y------
ax2.xaxis.labelpad = 12
ax2.yaxis.labelpad = 16

# Couleur et taille
highlight_textprops1 = [{"fontsize":8, "color":'#262626', "fontname": "Lisboa Sans OSF", "fontweight": "heavy"}]

# Définir le nom de l'axe des x et y
ax2.set_xlabel(f"Sunshine (hours)", **highlight_textprops1[-1])
ax2.set_ylabel(f"", **highlight_textprops1[-1])

# Annotation
ax2.axvline(x=df_train["sunshine"][df_train.rainfall==0].median(), color='#AF5C11', linestyle='--', alpha = 0.8)
ax2.annotate(f'Median on dry \ndays : \n{df_train["sunshine"][df_train.rainfall==0].median()}h', 
                 xy=(8.5, 0.20), xytext=(11, 0.20),
                 arrowprops=dict(arrowstyle='->', color = "#AF5C11"),
                 fontsize=8, color='#AF5C11', fontname = "Cover sans")

ax2.axvline(x=df_train["sunshine"][df_train.rainfall==1].median(), color='#4B6387', linestyle='--', alpha = 0.8)
ax2.annotate(f'Median on rainy \ndays : \n{df_train["sunshine"][df_train.rainfall==1].median()}h', 
                 xy=(1.6, 0.20), xytext=(3, 0.20),
                 arrowprops=dict(arrowstyle='->', color = "#4B6387"),
                 fontsize=8, color='#4B6387', fontname = "Cover sans")
# ------------------------------------------------------------------------------------------------------------




# --------- Graphique 3 (Pression - Pressure) ---------
ax3 = fig.add_subplot(gs[1, :])  # Troisième graphique
pressure_rain = df_train[df_train['rainfall'] > 0]['pressure']
pressure_no_rain = df_train[df_train['rainfall'] == 0]['pressure']
sns.kdeplot(pressure_rain, fill = True, linestyle='-', linewidth=1.2, label="With Rain", color='#0070C0', alpha=0.50, ax=ax3)
sns.kdeplot(pressure_no_rain, fill = True, linestyle='-', linewidth=1.2, label="No Rain", color='#EA7F1B', alpha=0.60, ax=ax3)
ax3.set_title("Pressure Density", **highlight_textprops, pad=20)

# --------Enlever le cadre en haut et à droite----
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# ----Ajouter des grilles----
plt.grid(axis='x', which='major', alpha=0.75, linestyle='dotted', zorder=1)
plt.grid(axis='y', alpha=0, zorder=2)

# ----Ajouter un cadre en bas et à gauche----
ax3.spines['bottom'].set_linewidth(1.2)
ax3.spines['bottom'].set_color('#CAC9CD')

ax3.spines['left'].set_linewidth(1.2)
ax3.spines['left'].set_color('#CAC9CD')

# ----Changer la couleur des barres des graduations sur l'axe x----
ax3.tick_params(axis='x', colors='#CAC9CD', width=1.2)
ax3.tick_params(axis='y', colors='#CAC9CD', width=1.2)

# ----Changer la taille et la couleur des ticks----
plt.xticks(fontsize=6, color='#202020', ha='center', rotation=0)
plt.yticks(fontsize=6, color='#202020')
plt.tick_params(left='on', bottom='on')

# ------Espacer les titres ticks------
ax3.tick_params(axis='y', pad=6)
ax3.tick_params(axis='x', pad=6)

# ------Espacer les titres des axes x et y------
ax3.xaxis.labelpad = 12
ax3.yaxis.labelpad = 16

# Couleur et taille
highlight_textprops1 = [{"fontsize":8, "color":'#262626', "fontname": "Lisboa Sans OSF", "fontweight": "heavy"}]

# Définir le nom de l'axe des x et y
ax3.set_xlabel(f"Pressure (hPa)", **highlight_textprops1[-1])
ax3.set_ylabel(f"Density", **highlight_textprops1[-1])

# Annotation
ax3.axvline(x=df_train["pressure"][df_train.rainfall==0].median(), color='#AF5C11', linestyle='--', alpha = 0.8)
ax3.annotate(f'Median on dry \ndays : \n{np.round(df_train["pressure"][df_train.rainfall==0].median(), 2)}', 
                 xy=(1013.2, 0.06), xytext=(1022, 0.06),
                 arrowprops=dict(arrowstyle='->', color = "#AF5C11"),
                 fontsize=8, color='#AF5C11', fontname = "Cover sans")

ax3.axvline(x=df_train["pressure"][df_train.rainfall==1].median(), color='#4B6387', linestyle='--', alpha = 0.8)
ax3.annotate(f'Median on rainy \ndays : \n{np.round(df_train["pressure"][df_train.rainfall==1].median(), 2)}', 
                 xy=(1013, 0.06), xytext=(1002, 0.06),
                 arrowprops=dict(arrowstyle='->', color = "#4B6387"),
                 fontsize=8, color='#4B6387', fontname = "Cover sans")
# ------------------------------------------------------------------------------------------------------------



# --------------------------Légende--------------------------

# Texte
font_properties = {'family': 'Cover sans', 'weight': 'normal', 'size': 10}
# Légende
legend_elements = [
    Rectangle((0, 0), 1, 1, color='#0070C0', linewidth=0.5, ec="k", label='With Rain', alpha=0.50),
    Rectangle((0, 0), 1, 1, color='#EA7F1B', linewidth=0.5, ec="k", label='No Rain', alpha=0.60)
]


plt.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.34), 
fancybox=False, shadow=False, ncol=3, frameon=False, prop=font_properties)



# Ajouter un titre général pour l'ensemble des graphiques
fig.suptitle("Sky Analysis: Clouds, Sunshine, and Atmospheric Pressure", fontsize=22, 
             fontweight='bold', 
             fontname="Lisboa Sans OSF",
             y=1.01)


plt.subplots_adjust(hspace=0.45) 
plt.subplots_adjust(wspace=0.25)

# Ajustement du layout
#plt.tight_layout()
plt.show()


# Crée la figure et la disposition de la grille
fig = plt.figure(figsize=(12, 4), dpi = 400)  # Ajusté la taille de la figure
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])  # 1 ligne, 2 colonnes


# ----------------------------------- Graphique 1 (Point de rosée - Dewpoint) -----------------------------------
ax1 = fig.add_subplot(gs[0])  # Premier graphique (première colonne)
dewpoint_rain = df_train[df_train['rainfall'] > 0]['dewpoint']
dewpoint_no_rain = df_train[df_train['rainfall'] == 0]['dewpoint']
sns.kdeplot(dewpoint_rain, fill = True, linestyle='-', linewidth=1.2, label="With Rain", color='#0070C0', alpha=0.50, ax=ax1)
sns.kdeplot(dewpoint_no_rain, fill = True, linestyle='-', linewidth=1.2, label="No Rain", color='#EA7F1B', alpha=0.60, ax=ax1)
ax1.set_title("Dew Point density", **highlight_textprops, pad=20)


# --------Enlever le cadre en haut et à droite----
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ----Ajouter des grilles----
plt.grid(axis='x', which='major', alpha=0.75, linestyle='dotted', zorder=1)
plt.grid(axis='y', alpha=0, zorder=2)

# ----Ajouter un cadre en bas et à gauche----
ax1.spines['bottom'].set_linewidth(1.2)
ax1.spines['bottom'].set_color('#CAC9CD')

ax1.spines['left'].set_linewidth(1.2)
ax1.spines['left'].set_color('#CAC9CD')

# ----Changer la couleur des barres des graduations sur l'axe x----
ax1.tick_params(axis='x', colors='#CAC9CD', width=1.2)
ax1.tick_params(axis='y', colors='#CAC9CD', width=1.2)

# ----Changer la taille et la couleur des ticks----
plt.xticks(fontsize=6, color='#202020', ha='center', rotation=0)
plt.yticks(fontsize=6, color='#202020')
plt.tick_params(left='on', bottom='on')

# ------Espacer les titres ticks------
ax1.tick_params(axis='y', pad=6)
ax1.tick_params(axis='x', pad=6)

# ------Espacer les titres des axes x et y------
ax1.xaxis.labelpad = 12
ax1.yaxis.labelpad = 16

# Couleur et taille
highlight_textprops1 = [{"fontsize":8, "color":'#262626', "fontname": "Lisboa Sans OSF", "fontweight": "heavy"}]

# Définir le nom de l'axe des x et y
ax1.set_xlabel(f"Dew Point (°C)", **highlight_textprops1[-1])
ax1.set_ylabel(f"Density", **highlight_textprops1[-1])

# Annotation
ax1.axvline(x=df_train["dewpoint"][df_train.rainfall==0].median(), color='#AF5C11', linestyle='--', alpha = 0.8)
ax1.annotate(f'Median on dry \ndays : \n{df_train["dewpoint"][df_train.rainfall==0].median()}°', 
                 xy=(21.2, 0.065), xytext=(8, 0.07),
                 arrowprops=dict(arrowstyle='->', color = "#AF5C11"),
                 fontsize=8, color='#AF5C11', fontname = "Cover sans")

ax1.axvline(x=df_train["dewpoint"][df_train.rainfall==1].median(), color='#4B6387', linestyle='--', alpha = 0.8)
ax1.annotate(f'Median on rainy \ndays : \n{df_train["dewpoint"][df_train.rainfall==1].median()}°', 
                 xy=(22.3, 0.10), xytext=(10, 0.105),
                 arrowprops=dict(arrowstyle='->', color = "#4B6387"),
                 fontsize=8, color='#4B6387', fontname = "Cover sans")
# ------------------------------------------------------------------------------------------------------------




# ----------------------------------- Graphique 2 (Humidité - Humidity) -----------------------------------
ax2 = fig.add_subplot(gs[1])  # Deuxième graphique (deuxième colonne)
humidity_rain = df_train[df_train['rainfall'] > 0]['humidity']
humidity_no_rain = df_train[df_train['rainfall'] == 0]['humidity']
sns.kdeplot(humidity_rain, fill = True, linestyle='-', linewidth=1.2, label="With Rain", color='#0070C0', alpha=0.50, ax=ax2)
sns.kdeplot(humidity_no_rain, fill = True, linestyle='-', linewidth=1.2, label="No Rain", color='#EA7F1B', alpha=0.60, ax=ax2)
ax2.set_title("Humidity density", **highlight_textprops, pad=20)

# --------Enlever le cadre en haut et à droite----
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# ----Ajouter des grilles----
plt.grid(axis='x', which='major', alpha=0.75, linestyle='dotted', zorder=1)
plt.grid(axis='y', alpha=0, zorder=2)

# ----Ajouter un cadre en bas et à gauche----
ax2.spines['bottom'].set_linewidth(1.2)
ax2.spines['bottom'].set_color('#CAC9CD')

ax2.spines['left'].set_linewidth(1.2)
ax2.spines['left'].set_color('#CAC9CD')

# ----Changer la couleur des barres des graduations sur l'axe x----
ax2.tick_params(axis='x', colors='#CAC9CD', width=1.2)
ax2.tick_params(axis='y', colors='#CAC9CD', width=1.2)

# ----Changer la taille et la couleur des ticks----
plt.xticks(fontsize=6, color='#202020', ha='center', rotation=0)
plt.yticks(fontsize=6, color='#202020')
plt.tick_params(left='on', bottom='on')

# ------Espacer les titres ticks------
ax2.tick_params(axis='y', pad=6)
ax2.tick_params(axis='x', pad=6)

# ------Espacer les titres des axes x et y------
ax2.xaxis.labelpad = 12
ax2.yaxis.labelpad = 16

# Couleur et taille
highlight_textprops1 = [{"fontsize":8, "color":'#262626', "fontname": "Lisboa Sans OSF", "fontweight": "heavy"}]

# Définir le nom de l'axe des x et y
ax2.set_xlabel(f"Humidity (%)", **highlight_textprops1[-1])
ax2.set_ylabel(f"", **highlight_textprops1[-1])

# Annotation
ax2.axvline(x=df_train["humidity"][df_train.rainfall==0].median(), color='#AF5C11', linestyle='--', alpha = 0.8)
ax2.annotate(f'Median on dry \ndays : \n{df_train["humidity"][df_train.rainfall==0].median()}%', 
                 xy=(75.2, 0.06), xytext=(50, 0.06),
                 arrowprops=dict(arrowstyle='->', color = "#AF5C11"),
                 fontsize=8, color='#AF5C11', fontname = "Cover sans")

ax2.axvline(x=df_train["humidity"][df_train.rainfall==1].median(), color='#4B6387', linestyle='--', alpha = 0.8)
ax2.annotate(f'Median on rainy \ndays : \n{df_train["humidity"][df_train.rainfall==1].median()}%', 
                 xy=(84.8, 0.06), xytext=(95, 0.06),
                 arrowprops=dict(arrowstyle='->', color = "#4B6387"),
                 fontsize=8, color='#4B6387', fontname = "Cover sans")
# ------------------------------------------------------------------------------------------------------------


# --------------------------Légende--------------------------

# Texte
font_properties = {'family': 'Cover sans', 'weight': 'normal', 'size': 10}
# Légende
legend_elements = [
    Rectangle((0, 0), 1, 1, color='#0070C0', linewidth=0.5, ec="k", label='With Rain', alpha=0.50),
    Rectangle((0, 0), 1, 1, color='#EA7F1B', linewidth=0.5, ec="k", label='No Rain', alpha=0.60)
]


plt.legend(handles=legend_elements, loc='center', bbox_to_anchor=(-0.1, -0.34), 
fancybox=False, shadow=False, ncol=3, frameon=False, prop=font_properties)



# Ajouter un titre général pour l'ensemble des graphiques
fig.suptitle("Humidity Analysis: Dew Point and Humidity", fontsize=22, 
             fontweight='bold', 
             fontname="Lisboa Sans OSF",
             y=1.12)


plt.subplots_adjust(hspace=0.45) 
plt.subplots_adjust(wspace=0.15)

plt.show()



# --- Preprocessing ---
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


df_train.head()


# ----- Create a transform for a new variable -----

def add_month_column(df):
    df['month'] = df['day'].apply(
        lambda x: month_names[next(i for i, days in enumerate(days_per_month, start=0) 
                                   if x <= sum(days_per_month[:i+1]))]
    )
    return df

month_transformer = FunctionTransformer(add_month_column, validate=False)


# ----- Create a transform to delete a variable -----

def remove_day_column(df):
    if 'day' in df.columns:
        df = df.drop(columns=['day'])
    return df

remove_day_transformer = FunctionTransformer(remove_day_column, validate=False)


# ----- Create an encoding transformer -----

def apply_get_dummies(X):
    return pd.get_dummies(X, drop_first=True)

encoding = FunctionTransformer(apply_get_dummies, validate=False)


# ----- Create a transformer to standardize quantitative variables -----

class SelectiveStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        # Sélectionner automatiquement les colonnes numériques
        self.continuous_columns = X.select_dtypes(include=['float64', 'int64']).columns
        self.scaler.fit(X[self.continuous_columns])
        return self
    
    def transform(self, X):
        # Appliquer le scaler seulement sur les colonnes continues
        X_scaled = X.copy()
        X_scaled[self.continuous_columns] = self.scaler.transform(X[self.continuous_columns])
        return X_scaled


# ----- Preprocessing Pipeline -----

preprocessing_pipeline = Pipeline([
    
    ('month_feature', month_transformer),   # Add 'Month'
    ('remove_day', remove_day_transformer), # Remove day
    ('encoder', encoding),                  # Encoder
    ('scaler', SelectiveStandardScaler())   # Standardisation
    
])


# ----- Testing our pipeline -----

data = {
    'day': np.random.randint(1, 101, size=10),
    'pressure': np.random.uniform(990, 1080, size=10),
    'temparature': np.random.uniform(0, 40, size=10)
}

data = pd.DataFrame(data)

processed_data = preprocessing_pipeline.fit_transform(data)

print("Created data :")
print(data)
print("\n \n")

print("\nData after transformation :")
print(processed_data)


# ----- Separating vector and matrix -----
X = df_train.drop(columns = 'rainfall')
y = df_train['rainfall']


# ----- Split into train and test sets -----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# ----- Preprocessing -----
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# ----- Hyperparameters -----
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# ----- Models -----
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ----- Pipeline with Logistic Regression -----

logistic_pipeline = make_pipeline(
    preprocessing_pipeline,
    LogisticRegression(max_iter = 1_000_000)
)


# ----- Hyperparameters -----

param_logistic = { 
    
    # Parameter regulation
    'logisticregression__C': [0.0001, 0.01, 0.03, 0.05, 0.1, 0.2],
    
    # B0 regularization (bias)
    'logisticregression__intercept_scaling': [0.0001, 0.01, 0.05, 0.1, 0.5, 0.9, 0.99, 1],
    
    # Solver
    'logisticregression__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}


grid_search_logistic = GridSearchCV(logistic_pipeline , param_logistic, cv=5, scoring='accuracy', n_jobs=-1)


grid_search_logistic.fit(X_train, y_train)


pd.DataFrame(grid_search_logistic.cv_results_).sort_values("rank_test_score").head(5)


grid_search_logistic.best_params_


# ----- Pipeline with RandomForest Classifier -----

randomforest_pipeline = make_pipeline(
    preprocessing_pipeline,
    RandomForestClassifier(random_state = 450)
)


# ----- Hyperparameters -----

param_rf = {
    'randomforestclassifier__n_estimators': [100, 200, 300],
    'randomforestclassifier__max_depth': [2, 5, 7, 15],
    'randomforestclassifier__min_samples_split': [2, 7, 10],
    'randomforestclassifier__min_samples_leaf': [2, 7, 10],
}


grid_search_rf = GridSearchCV(randomforest_pipeline , param_rf, cv = 5 , scoring='accuracy', n_jobs=-1)


grid_search_rf.fit(X_train, y_train)


pd.DataFrame(grid_search_rf.cv_results_).sort_values("rank_test_score").head()


grid_search_rf.best_params_


# ----- Pipeline with the final model -----

# Best hyperparameters
final_param_rf = {
    'n_estimators': 300,
    'max_depth': 5,
    'min_samples_split': 2,
    'min_samples_leaf': 2,
}

# Final pipeline
final_randomforest_pipeline = make_pipeline(
    preprocessing_pipeline,
    RandomForestClassifier(random_state = 450, **final_param_rf)
)


# --- Train the final model on the entire train set without cross validation ---
final_randomforest_pipeline.fit(X_train, y_train)


# --- Prediction ---
y_pred = final_randomforest_pipeline.predict(X_test)


# ----- Metrics -----
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.model_selection import learning_curve


# ----------- Metrics -----------

# Confusion Matrix
accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred) * 100
recall = recall_score(y_test, y_pred) * 100

# Roc Curve
fpr, tpr, thresholds = roc_curve(y_test, final_randomforest_pipeline.predict_proba(X_test)[:, 1])
# AUC
auc_value = roc_auc_score(y_test, y_pred)*100

# Tradeoff precision/recall
precisions, recalls, thresholds = precision_recall_curve(y_test, final_randomforest_pipeline.predict_proba(X_test)[:, 1])

# Learning Curve
train_sizes, train_scores, val_scores = learning_curve(final_randomforest_pipeline, 
                                                    X_train, 
                                                    y_train, 
                                                    train_sizes= np.linspace(0.1, 1.0, 5), 
                                                    cv = 5)
train_sizes_custom = np.array(train_sizes)


# Création de la figure et des sous-graphiques (2 lignes, 2 colonnes)
fig, axs = plt.subplots(2, 2, figsize=(14, 12), dpi = 400)

highlight_textprops = {"fontsize":14, "color":'#262626', "fontname": "Lisboa Sans OSF", "fontweight": "heavy"}


# ------------------------------ Premier graphique: Matrice de confusion ------------------------------

# Définition des couleurs : noir pour tout, bleu pour la diagonale
custom_colors = ['#DEEBF7', '#08306B']
cmap_custom = mcolors.ListedColormap(custom_colors)



cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=axs[0, 0], colorbar=False, cmap=cmap_custom)  # Placer la matrice de confusion dans le sous-graphe en haut à gauche


# Création d'une matrice de couleurs : 0 = noir, 1 = bleu
color_mask = np.zeros_like(cm)  # Tout en bleu (0)
np.fill_diagonal(color_mask, 1)  # Met la diagonale en bleu (1)


# Affichage de la matrice avec les bonnes couleurs
im = axs[0, 0].imshow(color_mask, cmap=cmap_custom)



axs[0, 0].set_title("Confusion Matrix", **highlight_textprops, pad=20)
axs[0, 0].set_aspect('auto', adjustable='box')

# Modifier les ticks
tick_params = dict(length=3, width=1, color="#C8C7CB") # Les ticks
for spine in axs[0, 0].spines.values(): spine.set_color("#C8C7CB") # Contour     
axs[0, 0].tick_params(axis='both', bottom='on', left='on', **tick_params) #Appliquer les ticks  

# Couleur pour les titres des axes

# Couleur et taille
highlight_textprops1 = [{"fontsize":12, "color":'#262626', "fontname": "Lisboa Sans OSF", "fontweight": "heavy"}]

# Définir le nom de l'axe des x et y
axs[0, 0].set_xlabel(f"\nPredicted Class", **highlight_textprops1[-1])
axs[0, 0].set_ylabel(f"\nTrue Class", labelpad=8, **highlight_textprops1[-1])


# ---Personnaliser les nombres---
for text in axs[0, 0].texts:
    text.set_fontsize(10)
    text.set_fontweight('normal')
    text.set_fontname("Cover sans")

axs[0, 0].texts[0].set_color("white")
axs[0, 0].texts[1].set_color("#262626")
axs[0, 0].texts[2].set_color("#262626")
axs[0, 0].texts[3].set_color("white")

# Couleur pour les ticks des axes

# Texte
ticks_labels = [{'family': 'Cover sans', "color" : "#262626", 'weight': 'normal', 'size': 10}]

axs[0, 0].xaxis.set_ticklabels(['False', 'True'], rotation=0, **ticks_labels[-1])
axs[0, 0].yaxis.set_ticklabels(['False', 'True'], **ticks_labels[-1])

# Espacer les titres ticks
axs[0, 0].tick_params(axis='y', pad=6)
axs[0, 0].tick_params(axis='x', pad=6)

# Ajout de la légende sous la matrice de confusion
font_properties = {'family': 'Cover sans', 'weight': 'normal', 'size': 10}
axs[0, 0].text(0.5, -0.24, 
               f"Accuracy = {accuracy:.1f}%  |  Precision = {precision:.1f}%  |  Recall = {recall:.1f}%", 
               ha='center', va='center', transform=axs[0, 0].transAxes, 
               fontsize=10, color='#262626', fontname="Cover sans")
# ------------------------------ Premier graphique: Matrice de confusion ------------------------------











# ------------------------------ Deuxième graphique: ROC Curve ------------------------------
axs[0, 1].plot(fpr, tpr, linewidth=2, label="ROC curve", color = "#4B6387")
axs[0, 1].fill_between(fpr, tpr, color="#4B6387", alpha=0.3, label=f"AUC ({auc_value:.3f})")
axs[0, 1].plot([0, 1], [0, 1], 'k:', label="Random")
axs[0, 1].set_title("Courbe ROC", **highlight_textprops, pad=20)

# Enlever le cadre en haut et à droite
axs[0, 1].spines['top'].set_visible(False)
axs[0, 1].spines['right'].set_visible(False)
    
# Ajouter un cadre en bas et à gauche
axs[0, 1].spines['bottom'].set_linewidth(1.3)
axs[0, 1].spines['bottom'].set_color('#CAC9CD')

axs[0, 1].spines['left'].set_linewidth(1.3)
axs[0, 1].spines['left'].set_color('#CAC9CD')

# Ajouter des grilles
axs[0, 1].grid(axis='x', which='major', alpha=0.6, linestyle='dotted', zorder=1)
axs[0, 1].grid(axis='y', alpha=0, zorder=2)

# Changer la couleur des barres des graduations sur l'axe x
axs[0, 1].tick_params(axis='x', colors='#CAC9CD', width=1.3)
axs[0, 1].tick_params(axis='y', colors='#CAC9CD', width=1.3)

# Changer la couleur des valeurs des barres des graduations sur l'axe x
for tick in axs[0, 1].get_xticklabels():
    tick.set_color('#202020') 

# Couleur pour les axes
highlight_textprops1 = [{"fontsize":12, "color":'#262626', "fontname": "Lisboa Sans OSF", "fontweight": "heavy"}]

# Changer la couleur des valeurs des barres des graduations sur l'axe y
for tick in axs[0, 1].get_yticklabels():
    tick.set_color('#202020')

# Définir le nom de l'axe des x
axs[0, 1].set_xlabel(f"False positive rate",**highlight_textprops1[-1])
    
# Espacer le nom (à modifier)
axs[0, 1].xaxis.labelpad = 20

    
# Définir le nom de l'axe des x
axs[0, 1].set_ylabel(f"Rate of true positives",**highlight_textprops1[-1])
    
# Espacer le nom (à modifier)
axs[0, 1].yaxis.labelpad = 20

# Définir les limites des axes
axs[0, 1].set_xlim(0.0, 1.02)  # Limite de l'axe des X
axs[0, 1].set_ylim(0.0, 1.02)  # Limite de l'axe des Y

# Légende personnalisée
font_properties = {'family': 'Cover sans', 'weight': 'normal', 'size': 10}
axs[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.19), 
                 fancybox=False, shadow=False, ncol=3, frameon=False, prop=font_properties)
# ---------------------------------------------------------------------------------------------




# ------------------------------ Troisième graphique: Precision vs Recall Tradeoff ------------------------------ 

axs[1, 0].plot(thresholds, precisions[:-1], "--", label="Precision", linewidth=2, color = "#4B6387", zorder=3)
axs[1, 0].plot(thresholds, recalls[:-1], "-", label="Recall", linewidth=2, color = "#C55B6F", zorder=3)
axs[1, 0].vlines(0.5, 0, 1.0, linestyle="dotted", label="50%", color = "#CAC9CD", linewidth = 1.7)    

# Title
axs[1, 0].set_title("Precision/Recall Tradeoff", **highlight_textprops, pad=20)

# Enlever le cadre en haut et à droite
axs[1, 0].spines['top'].set_visible(False)
axs[1, 0].spines['right'].set_visible(False)
    
# Ajouter un cadre en bas et à gauche
axs[1, 0].spines['bottom'].set_linewidth(1.3)
axs[1, 0].spines['bottom'].set_color('#CAC9CD')

axs[1, 0].spines['left'].set_linewidth(1.3)
axs[1, 0].spines['left'].set_color('#CAC9CD')

# Ajouter des grilles
axs[1, 0].grid(axis='x', which='major', alpha=0.5, linestyle='dotted', zorder=1)
axs[1, 0].grid(axis='y', alpha=0, zorder=2)
           
# Changer la couleur des barres des graduations sur l'axe x
axs[1, 0].tick_params(axis='x', colors='#CAC9CD', width=1.3)
axs[1, 0].tick_params(axis='y', colors='#CAC9CD', width=1.3)
    
# Changer la couleur des valeurs des barres des graduations sur l'axe x
for tick in axs[1, 0].get_xticklabels():
    tick.set_color('#202020') 
    
# Changer la couleur des valeurs des barres des graduations sur l'axe y
for tick in axs[1, 0].get_yticklabels():
    tick.set_color('#202020')
        
# Définir le nom de l'axe des x
axs[1, 0].set_xlabel(f"Threshold",**highlight_textprops1[-1])
    
# Espacer le nom (à modifier)
axs[1, 0].xaxis.labelpad = 20
  
# Définir le nom de l'axe des x
axs[1, 0].set_ylabel(f"Metric",**highlight_textprops1[-1])
    
# Espacer le nom (à modifier)
axs[1, 0].yaxis.labelpad = 20

# Limite
axs[1, 0].set_ylim(0.3, 1.05)


# Légende
font_properties = {'family': 'Cover sans', 'weight': 'normal', 'size': 10}
axs[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), 
fancybox=False, shadow=False, ncol=3, frameon=False, prop=font_properties)
# ---------------------------------------------------------------------------------------------









# ------------------------------ Quatrième graphique: Learning Curve (en bas à droite) ------------------------------
axs[1, 1].plot(train_sizes, train_scores.mean(axis = 1), label = "Train", color = "#4B6387", zorder=3)
axs[1, 1].plot(train_sizes, val_scores.mean(axis = 1), label = "Validation", color = "#C55B6F", zorder=3)

# Ajout des points spécifiques
axs[1, 1].scatter(train_sizes, train_scores.mean(axis = 1), color="#4B6387", marker="o", zorder=3)
axs[1, 1].scatter(train_sizes, val_scores.mean(axis = 1), color="#C55B6F", marker="o", zorder=3)

# Title
axs[1, 1].set_title("Learning Curve", **highlight_textprops, pad=20)

# Enlever le cadre en haut et à droite
axs[1, 1].spines['top'].set_visible(False)
axs[1, 1].spines['right'].set_visible(False)
    
# Ajouter un cadre en bas et à gauche
axs[1, 1].spines['bottom'].set_linewidth(1.3)
axs[1, 1].spines['bottom'].set_color('#CAC9CD')

axs[1, 1].spines['left'].set_linewidth(1.3)
axs[1, 1].spines['left'].set_color('#CAC9CD')
    
# Ajouter des grilles
axs[1, 1].grid(axis='x', which='major', alpha=0.5, linestyle='dotted', zorder=1)
axs[1, 1].grid(axis='y', alpha=0, zorder=2)
       
# Changer la couleur des barres des graduations sur l'axe x
axs[1, 1].tick_params(axis='x', colors='#CAC9CD', width=1.3)
axs[1, 1].tick_params(axis='y', colors='#CAC9CD', width=1.3)
      
# Changer la couleur des valeurs des barres des graduations sur l'axe x
for tick in axs[1, 1].get_xticklabels():
    tick.set_color('#202020') 

# Changer la couleur des valeurs des barres des graduations sur l'axe y
for tick in axs[1, 1].get_yticklabels():
    tick.set_color('#202020') 
    
# Définir le nom de l'axe des x
axs[1, 1].set_xlabel(f"Training sample size",
                       **highlight_textprops1[-1])
    
# Espacer le nom (à modifier)
axs[1, 1].xaxis.labelpad = 20

    
# Définir le nom de l'axe des x
axs[1, 1].set_ylabel(f"Accuracy",
                       **highlight_textprops1[-1])
    
# Espacer le nom (à modifier)
axs[1, 1].yaxis.labelpad = 20

# Légende
font_properties = {'family': 'Cover sans', 'weight': 'normal', 'size': 10}
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), 
fancybox=False, shadow=False, ncol=2, frameon=False, prop=font_properties)





# Ajouter un titre général pour l'ensemble des graphiques
fig.suptitle("Random Forest evaluation performance report", fontsize=22, fontweight='bold', fontname="Lisboa Sans OSF")


plt.subplots_adjust(hspace=0.50) 
plt.subplots_adjust(wspace=0.30)

# Ajustement de l'espacement pour éviter que les graphiques ne se chevauchent
#plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajuste le rect pour laisser de l'espace pour le titre général
plt.show()



# Train the final model on the entire dataset
final_randomforest_pipeline.fit(X, y)


y_deployment = final_randomforest_pipeline.predict_proba(df_test)[:, 1]


# Submission
submission = pd.DataFrame({'id': df_test.index , 'rainfall': y_deployment})
submission.to_csv("submission.csv", index = False)

