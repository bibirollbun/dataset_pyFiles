"""ì‹œìŠ¤í…œì—� ì„¤ì¹˜ë�œ í�°íŠ¸ ëª©ë¡�ì�„ ë³´ê³  ì‹¶ì�„ ë•Œ?"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ì‹œìŠ¤í…œì—� ì„¤ì¹˜ë�œ í�°íŠ¸ ëª©ë¡� ê°€ì ¸ì˜¤ê¸°
# ë¦¬ìŠ¤íŠ¸ì�˜ ê°� FontEntry ê°�ì²´ì—�ì„œ í�°íŠ¸ ì�´ë¦„(f.name)ê³¼ í�°íŠ¸ íŒŒì�¼ ê²½ë¡œ(f.fname)ë¥¼ íŠœí”Œë¡œ ì¶”ì¶œ
# setìœ¼ë¡œ ì¤‘ë³µ ì œê±°
font_info = sorted(set((f.name, f.fname) for f in fm.fontManager.ttflist))

# ì�´ì „ í�°íŠ¸ ì�´ë¦„ ì €ì�¥ ë³€ìˆ˜
prev_name = None

print("ì„¤ì¹˜ë�œ í�°íŠ¸ ëª©ë¡�")

for name, path in font_info:
    # í�°íŠ¸ ì�´ë¦„ì�´ ë°”ë€” ë•Œë§Œ ì¤„ ë°”ê¿ˆ
    if name != prev_name:
        print(f"\nğŸ“‚{name}")  # ìƒˆë¡œìš´ í�°íŠ¸ ì�´ë¦„ ì¶œë ¥
        prev_name = name  # ì�´ì „ í�°íŠ¸ ì�´ë¦„ ì—…ë�°ì�´íŠ¸
    
    print(f" - {path}")  # í•´ë‹¹ í�°íŠ¸ì�˜ íŒŒì�¼ ê²½ë¡œ ì¶œë ¥



!apt-get update -qq && apt-get install -qq -y fonts-nanum

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

# ë‚˜ëˆ”ê³ ë”• í�°íŠ¸ ê²½ë¡œ
regular_font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
bold_font_path = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"

# í�°íŠ¸ ë“±ë¡� (Matplotlib ë‚´ë¶€ í�°íŠ¸ ìº�ì‹œì—� ì¶”ê°€)
fm.fontManager.addfont(regular_font_path)
fm.fontManager.addfont(bold_font_path)

# ê¸°ë³¸ í�°íŠ¸ ì„¤ì •
mpl.rcParams['font.family'] = 'NanumGothic'  # ê¸°ë³¸ í�°íŠ¸ ì§€ì •
mpl.rcParams['axes.unicode_minus'] = False

# ğŸ”¥ ì��ë�™ Bold ì �ìš©ì�„ ìœ„í•´ font weight ë§¤í•‘
mpl.rcParams['font.weight'] = 'regular'
mpl.rcParams['axes.titleweight'] = 'bold'  # ì œëª©ì�€ ê¸°ë³¸ì �ìœ¼ë¡œ bold
mpl.rcParams['axes.labelweight'] = 'bold'  # ì¶• ë�¼ë²¨ë�„ bold


########################################################################
# í…ŒìŠ¤íŠ¸
#######################################################################

plt.figure(figsize=(10, 2))
plt.text(0.5, 0.6, "ê¸°ë³¸ì²´ (Regular)", fontsize=12, weight='regular', ha='center', va='center')
plt.text(0.5, 0.4, "êµµì�€ì²´ (Bold)", fontsize=12, weight='bold', ha='center', va='center')
plt.title("ì œëª© (Bold ì �ìš© í™•ì�¸)", fontsize=18, pad=10)
plt.show()


################################################################################################
######################### Pandas DataFrame Korean Font Update ##################################

import pandas as pd
from IPython.core.display import display, HTML

pd.set_option('display.max_columns', None)  # ëª¨ë“  ì—´ í‘œì‹œ
pd.set_option('display.unicode.east_asian_width', True)  # í•œê¸€ ì •ë ¬ ê¹¨ì§� ë°©ì§€

display(HTML("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic&display=swap');
        table { font-family: 'Nanum Gothic', sans-serif !important; }
    </style>
"""))

# í…ŒìŠ¤íŠ¸
df = pd.DataFrame({
    "ì�´ë¦„": ["ê¹€ì² ìˆ˜", "ì�´ì˜�í�¬", "ë°•ì§€ë¯¼"],
    "ë‚˜ì�´": [25, 30, 22],
    "ë�„ì‹œ": ["ì„œìš¸", "ë¶€ì‚°", "ëŒ€êµ¬"]
})

display(df)

################################################################################################
######################## Plotly Korean Font Update #############################################

import plotly.graph_objects as go
import plotly.io as pio

# Plotly ê¸°ë³¸ í�°íŠ¸ ì„¤ì • (ì „ì—­ ì„¤ì •)
# weight = bold or normal

pio.templates["plotly"].layout.font.family = "NanumGothic"


#################### ë�°ì�´í„°ì…‹ í�´ë�” ë°� íŒŒì�¼ì—� ëŒ€í•œ ìš”ì•½ ì •ë³´ ##############################

import os

root_dir = "/kaggle/input"

for dirname, dirnames, filenames in os.walk(root_dir):
    if filenames:  # íŒŒì�¼ì�´ ì�ˆëŠ” í�´ë�”ë§Œ ì¶œë ¥
        
        print(f"\nğŸ“‚ {dirname}")  # í˜„ì�¬ ë””ë ‰í† ë¦¬ ì¶œë ¥
        
        file_count = len(filenames)  # í˜„ì�¬ í�´ë�” ë‚´ íŒŒì�¼ ê°œìˆ˜
        display_count = min(file_count, 5)  # ìµœëŒ€ 5ê°œê¹Œì§€ë§Œ í‘œì‹œ            

        # íŒŒì�¼ ëª©ë¡�ì—�ì„œ ìµœëŒ€ 5ê°œê¹Œì§€ë§Œ í‘œì‹œí•˜ê³ , 5ê°œ ë¯¸ë§Œì�¸ ê²½ìš° ê·¸ê²ƒë“¤ë§Œ í‘œì‹œ
        # enumerateì—�ì„œ iëŠ” ì�¸ë�±ìŠ¤ ë²ˆí˜¸ì�´ê³ , í•¨ìˆ˜ ì•ˆì—� ë“¤ì–´ê°€ëŠ” ê²ƒì�€ ë³€ìˆ˜ëª…(ì¹¼ëŸ¼)
        for i, filename in enumerate(filenames[:display_count]):
            if i == display_count - 1 and file_count <= 5:  
                print(f"  â””â”€â”€ {filename}\n\n")  # ë§ˆì§€ë§‰ íŒŒì�¼ (5ê°œ ì�´í•˜ì�¸ ê²½ìš°)
            else:
                print(f"  â”œâ”€â”€ {filename}")  # ì�¼ë°˜ íŒŒì�¼ ì¶œë ¥
        
        if file_count > 5:
            print(f"  â””â”€â”€ ... (ì´� {file_count}ê°œ íŒŒì�¼)")  # 5ê°œ ì´ˆê³¼ ì‹œ ìš”ì•½ ì¶œë ¥
        


# ê¸°ë³¸,ì‹œê°�í™”

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.express as px

# ì»¬ëŸ¬ íŒ”ë ˆíŠ¸
cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801'

# ì§€ë�„ ì‹œê°�í™” ë�¼ì�´ë¸ŒëŸ¬ë¦¬ folium
"""
ì§€ë�„ì�˜ ìƒ�ì„± ë°� í‘œì‹œ (folium.Map)
ë§ˆì»¤ ì¶”ê°€ (folium.Marker, folium.CircleMarker)
í�´ë¦¬ê³¤, ë�¼ì�¸ ì¶”ê°€ (folium.Polygon, folium.PolyLine)
í�ˆíŠ¸ë§µ ì‹œê°�í™” (folium.plugins.HeatMap)
GeoJSON ë�°ì�´í„° í™œìš© ê°€ëŠ¥ (folium.GeoJson)
"""
import folium 
from folium import plugins

# rasterio: ìœ„ì„± ì�´ë¯¸ì§€ ë°� ì§€ë¦¬ê³µê°„ ë�˜ìŠ¤í„° ë�°ì�´í„°ë¥¼ ì²˜ë¦¬
# ê³µê°„ ë�°ì�´í„°ë¥¼ ì�½ê³ , ì“°ê³ , ë¶„ì„�í•˜ëŠ” ë�° ì‚¬ìš©
!pip install rasterio
import rasterio as rio

# ë‚ ì§œ ë�°ì�´í„° ì²˜ë¦¬
import datetime as dt
from datetime import datetime


power_plant = pd.read_csv('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')


display(power_plant.shape)

pp_columns = (power_plant.columns.to_list())
pp_columns = ", ".join(pp_columns)
display(pp_columns)

display(power_plant.info())

display(power_plant.describe(include="all"))


display(power_plant)


display(power_plant['system:index'])

unique_counts = power_plant.nunique().reset_index()
unique_counts.columns = ['Column', 'Unique Values']

display(unique_counts)



power_plant.set_index('system:index',inplace=True)
power_plant


for col in power_plant.columns:
    display(f"Unique value counts for: {col}")
    display(power_plant[col].value_counts())
    print("\n")


power_plant['name'].value_counts()


plt.figure(figsize=(10,6))  
ax = sns.countplot(x="primary_fuel", data=power_plant)  

# ê°� ë§‰ëŒ€ ë†’ì�´(ì¹´ìš´íŠ¸ ê°’) ê°€ì ¸ì˜¤ê¸°
heights = [p.get_height() for p in ax.patches]

# ìµœì†Ÿê°’ & ìµœëŒ“ê°’ ê³„ì‚°
ymin, ymax = min(heights), max(heights)

# ìµœì†Ÿê°’ ë�¼ì�¸ (ë¹¨ê°„ ì �ì„ )
ax.axhline(y=ymin, color='red', linestyle='dashed', linewidth=1, label=f'Min: {ymin}')

# ìµœëŒ“ê°’ ë�¼ì�¸ (íŒŒë�€ ì �ì„ )
ax.axhline(y=ymax, color='blue', linestyle='dashed', linewidth=1, label=f'Max: {ymax}')

# ë§‰ëŒ€ ìœ„ì—� ê°’ í‘œì‹œ (ë�” ê°„ë‹¨í•œ ë°©ë²•)
for p in ax.patches:
    plt.text(p.get_x() + p.get_width() / 2,  # x ìœ„ì¹˜
             p.get_height()+0.2,  # y ìœ„ì¹˜ (ë§‰ëŒ€ ë†’ì�´)
             int(p.get_height()),  # í‘œì‹œí•  ê°’
             ha='center', va='bottom', fontsize=12)  # ê°€ìš´ë�° ì •ë ¬, í�°íŠ¸ í�¬ê¸° ì„¤ì •

plt.title("Count of Primary Fuel Types", pad=10, weight='medium', fontsize=15)
plt.xlabel("Primary Fuel", labelpad=10)
plt.ylabel("Count", labelpad=10)
plt.xticks(rotation=0)  # xì¶• ë ˆì�´ë¸” íšŒì „ (í•„ìš”í•˜ë©´ ì¡°ì •)

ax.set_ylim(0, ax.get_ylim()[1] * 1.2)

plt.show()



# ë‚´ë¦¼ì°¨ìˆœ ì •ë ¬

plt.figure(figsize=(10,6))

# primary_fuelë³„ ê°œìˆ˜ë¥¼ ê³„ì‚°í•˜ê³  ë‚´ë¦¼ì°¨ìˆœ ì •ë ¬
fuel_counts = power_plant["primary_fuel"].value_counts().sort_values(ascending=False)

# ë§‰ëŒ€ ê·¸ë�˜í”„ ìƒ�ì„±
ax = sns.countplot(x="primary_fuel", data=power_plant, order=fuel_counts.index)

# ìµœì†Ÿê°’ & ìµœëŒ“ê°’ ê³„ì‚°
ymin, ymax = fuel_counts.min(), fuel_counts.max()

# ìµœì†Ÿê°’ & ìµœëŒ“ê°’ ë�¼ì�¸ ì¶”ê°€
ax.axhline(ymin, color='red', linestyle='dashed', linewidth=1, label=f'Min: {ymin}')
ax.axhline(ymax, color='blue', linestyle='dashed', linewidth=1, label=f'Max: {ymax}')

# ë§‰ëŒ€ ìœ„ì—� ê°’ í‘œì‹œ
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2, p.get_height() + 0.2, int(p.get_height()), 
            ha='center', va='bottom', fontsize=12)

plt.title("Count of Primary Fuel Types", pad=10, weight='medium', fontsize=15)
plt.xlabel("Primary Fuel", labelpad=10)
plt.ylabel("Count", labelpad=10)
plt.xticks(rotation=0)
ax.set_ylim(0, ymax * 1.2)

plt.show()


# ìƒ‰ê¹” ê¸°ì¤€ìœ¼ë¡œ ë¶„ë¥˜

import matplotlib.cm as cm

# Stock Holderì�˜ ê¸°ì—¬ë�„ ê³„ì‚° (ì˜¤ë¦„ì°¨ìˆœ ì •ë ¬)
stock_holder_counts = power_plant['owner'].value_counts(ascending=True)

plt.figure(figsize=(10,6))

# ì»¬ëŸ¬ë§µ ì �ìš© (ì˜ˆ: 'viridis', 'Blues', 'coolwarm' ë“± ì„ íƒ� ê°€ëŠ¥)
colors = cm.Blues(stock_holder_counts.rank(pct=True))  # ìˆœìœ„ ê¸°ë°˜ìœ¼ë¡œ ìƒ‰ìƒ� ì„¤ì •

ax = stock_holder_counts.plot(kind='barh', color=colors)

# ë§‰ëŒ€ ìœ„ì—� ìˆ«ì�� í‘œê¸°
for p in ax.patches:
    ax.text(p.get_width() + 0.5, p.get_y() + p.get_height()/2, int(p.get_width()), 
            ha='left', va='center', fontsize=10)

plt.title('Contribution of Stock Holders', pad=10, weight='medium', fontsize=15)
plt.xlabel("Count")

# ê·¸ë�˜í”„ ë„ˆë¹„ ì¡°ì •
ax.set_xlim(0, ax.get_xlim()[1] * 1.05)

plt.show()



"""
íŠ¸ë¦¬ë§µ ì‹œê°�í™” ë¶„ì„�:

1. ë¹„ìœ¨(%)ì�„ ì‹œê°�ì �ìœ¼ë¡œ ë¹„êµ�í•˜ê³  ì‹¶ì�„ ë•Œ
ë°œì „ëŸ‰(estimated_generation_gwh)ê³¼ ìš©ëŸ‰(capacity_mw) ì¤‘ ì–´ëŠ� í•­ëª©ì�´ ë�” í�° ê¸°ì—¬ë�„ë¥¼ ê°€ì§€ëŠ”ì§€ ì‰½ê²Œ í™•ì�¸ ê°€ëŠ¥.

2. ê³„ì¸µì � ë�°ì�´í„°ë¥¼ í‘œí˜„í•˜ê³  ì‹¶ì�„ ë•Œ
ì˜ˆ: íšŒì‚¬ ë‚´ ë¶€ì„œë³„ ì§�ì›� ìˆ˜, ì œí’ˆêµ°ë³„ ë§¤ì¶œ, êµ­ê°€ë³„ GDP ë“±

3. ë‹¤ë¥¸ ê·¸ë�˜í”„(ë§‰ëŒ€ê·¸ë�˜í”„, íŒŒì�´ì°¨íŠ¸)ë³´ë‹¤ ë�” ì§�ê´€ì �ì�¸ ì‹œê°�í™”ë¥¼ ì›�í•  ë•Œ
íŠ¸ë¦¬ë§µì�€ ë§‰ëŒ€ê·¸ë�˜í”„ë³´ë‹¤ ë©´ì �ì�„ í†µí•´ í�¬ê¸°ë¥¼ ë¹„êµ�í•  ìˆ˜ ì�ˆì–´ ì‹œê°�ì �ìœ¼ë¡œ ë�” ëª…í™•í•¨.


power_plant ë�°ì�´í„°ì—�ì„œ ì—°ë�„ë³„(commissioning_year) ë°œì „ëŸ‰(estimated_generation_gwh)ê³¼ ìš©ëŸ‰(capacity_mw)ì�˜ ì´�í•©ì�„ ê³„ì‚°í•¨
groupby('commissioning_year')ì�„ ì‚¬ìš©í•˜ì—¬ ì—°ë�„ë³„ ê·¸ë£¹í™”í•˜ê³  sum()ì�„ ì‚¬ìš©í•˜ì—¬ ê°™ì�€ ì—°ë�„ì—� ì†�í•œ ë°œì „ì†Œë“¤ì�˜ ê°’ì�„ í•©ì‚°
"""

# ê·¸ë£¹í™” ë°� ë�°ì�´í„° ë³€í™˜
temp = power_plant.groupby('commissioning_year')[['estimated_generation_gwh', 'capacity_mw']].sum().reset_index()
temp = temp[temp['commissioning_year'] == max(temp['commissioning_year'])].reset_index(drop=True)


# ë�°ì�´í„° ë³€í™˜ (melt)
# commissioning_year                variable   value ë¡œ ì •ë¦¬ë�¨
"""
melt()ë�€? íŠ¸ë¦¬ë§µì�„ ë§Œë“¤ ë•Œ ë�°ì�´í„° ê°€ê³µì‹œ ì‚¬ìš©í•¨
Wide Format â†’ Long Formatìœ¼ë¡œ ë³€í™˜í•˜ëŠ” í•¨ìˆ˜
ì¦‰, ì—´(column)ì�„ í–‰(row)ìœ¼ë¡œ ë°”ê¿”ì£¼ëŠ” ì—­í• 

path=["variable"]: íŠ¸ë¦¬ë§µì—�ì„œ ê·¸ë£¹ì�„ êµ¬ë¶„í•  ê¸°ì¤€ (ì—¬ê¸°ì„œëŠ” estimated_generation_gwh vs capacity_mw).
values="value": íŠ¸ë¦¬ë§µ ë¸”ë¡� í�¬ê¸°ë¥¼ ê²°ì •í•  ê°’
"""
tm = temp.melt(id_vars="commissioning_year", value_vars=["estimated_generation_gwh", "capacity_mw"])

# ì»¬ëŸ¬ ë³€ìˆ˜ ì •ì�˜ (ê¸°ë³¸ê°’ ì„¤ì •)
act, rec = "blue", "red"

# íŠ¸ë¦¬ë§µ ìƒ�ì„±
fig = px.treemap(tm, path=["commissioning_year", "variable"], values="value", height=225, width=1200,
                 color_discrete_sequence=[act, rec])

fig.data[0].textinfo = 'label+text+value'
fig.show()



# ê·¸ë£¹í™” ë°� ë�°ì�´í„° ë³€í™˜
temp = power_plant.groupby('commissioning_year')[['estimated_generation_gwh', 'capacity_mw']].sum().reset_index()

# commissioning_yearê°€ 0ì�¸ ë�°ì�´í„° ì œê±°
temp = temp[temp['commissioning_year'] > 0]

# 20ì„¸ê¸°(1900~1999) ë�°ì�´í„° í•„í„°ë§�
temp_20th = temp[temp['commissioning_year'] < 2000]

# 21ì„¸ê¸°(2000~í˜„ì�¬) ë�°ì�´í„° í•„í„°ë§�
temp_21st = temp[temp['commissioning_year'] >= 2000]

# ë�°ì�´í„° ë³€í™˜ (melt)
tm_20th = temp_20th.melt(id_vars="commissioning_year", value_vars=["estimated_generation_gwh", "capacity_mw"])
tm_21st = temp_21st.melt(id_vars="commissioning_year", value_vars=["estimated_generation_gwh", "capacity_mw"])

# ì»¬ëŸ¬ ë³€ìˆ˜ ì •ì�˜ (ë¶€ë“œëŸ¬ìš´ ìƒ‰ìƒ�)
pastel_colors = ["#A1D6E2", "#FFDDC1"]  

# 2000ë…„ëŒ€ ì�´ì „
fig_20th = px.treemap(tm_20th, path=["commissioning_year", "variable"], values="value", height=600, width=1100,
                       title="1900~1999 ë°œì „ëŸ‰ ë°� ìš©ëŸ‰", color_discrete_sequence=pastel_colors)

# 2000ë…„ëŒ€ í›„í›„
fig_21st = px.treemap(tm_21st, path=["commissioning_year", "variable"], values="value", height=600, width=1100,
                       title="2000ë…„ ì�´í›„ ë°œì „ëŸ‰ ë°� ìš©ëŸ‰", color_discrete_sequence=pastel_colors)

# íŠ¸ë¦¬ë§µ ì¶œë ¥ (ê°�ê°� ë‹¤ë¥¸ í™”ë©´ì—�ì„œ ì¶œë ¥)
fig_20th.show()
fig_21st.show()



# ê·¸ë£¹í™” ë°� ë�°ì�´í„° ë³€í™˜
temp = power_plant.groupby('commissioning_year')[['estimated_generation_gwh', 'capacity_mw']].sum().reset_index()

# commissioning_yearê°€ 0ì�¸ ë�°ì�´í„° ì œê±°
temp = temp[temp['commissioning_year']>0]

display(temp)

# ìƒ�ê´€í–‰ë ¬ í�ˆíŠ¸ë§µ ì‹œê°�í™”
correlation_matrix = temp.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Power Plant Data')
plt.show()

# ì‚°ì �ë�„ ì‹œê°�í™” (ì—°ë�„ vs ë°œì „ëŸ‰, ì—°ë�„ vs ì„¤ë¹„ ìš©ëŸ‰, ì„¤ë¹„ ìš©ëŸ‰ vs ë°œì „ëŸ‰)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ì—°ë�„ vs ë°œì „ëŸ‰
sns.scatterplot(data=temp, x='commissioning_year', y='estimated_generation_gwh', ax=axes[0])
axes[0].set_title('Commissioning Year vs. Estimated Generation (GWh)', pad=15)

# ì—°ë�„ vs ì„¤ë¹„ ìš©ëŸ‰
sns.scatterplot(data=temp, x='commissioning_year', y='capacity_mw', ax=axes[1])
axes[1].set_title('Commissioning Year vs. Capacity (MW)', pad=15)

# ì„¤ë¹„ ìš©ëŸ‰ vs ë°œì „ëŸ‰
sns.scatterplot(data=temp, x='capacity_mw', y='estimated_generation_gwh', ax=axes[2])
axes[2].set_title('Capacity (MW) vs. Estimated Generation (GWh)', pad=15)

plt.show()



#Estimated generation growth from commissioning year
temp=power_plant.groupby('commissioning_year')[['estimated_generation_gwh','capacity_mw']].sum().reset_index()
temp = temp[temp['commissioning_year']!=0]
temp=temp.melt(id_vars="commissioning_year",value_vars=["estimated_generation_gwh","capacity_mw"],var_name='Year',value_name='Count')

display(temp.head())

fig=px.area(temp,x='commissioning_year',y='Count',color='Year',height=600,title='Production over time',color_discrete_sequence=[rec,dth])
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()


full_grouped = power_plant.groupby(['source', 'primary_fuel'])[['capacity_mw', 'estimated_generation_gwh']].sum().reset_index()
display(full_grouped)


temp_1 = full_grouped.sort_values(by='estimated_generation_gwh',ascending=False)
display(temp_1)

# ìˆœì„œ ì •ë ¬ í›„ ì�¸ë�±ìŠ¤ê°€ ê¼¬ì�„. ê·¸ë�˜ì„œ ì�¸ë�±ìŠ¤ë¥¼ ë¦¬ì…‹í•˜ê³ , ê¸°ì¡´ê°’ì�„ ì œê±°í•¨
# drop ì˜µì…˜ì�„ ë„£ì§€ ì•Šì�„ ê²½ìš° ê¸°ì¡´ ì�¸ë�±ìŠ¤ëŠ” ê·¸ëŒ€ë¡œ ìœ ì§€ë�˜ë©´ì„œ (df í•­ëª©ìœ¼ë¡œ ì¶”ê°€) ìƒˆë¡œìš´ ì •ë ¬ ì�¸ë�±ìŠ¤ê°€ ì¶”ê°€ë�¨
temp_1 = temp_1.reset_index(drop=True)
display(temp_1)

#ìˆ˜ì¹˜í˜• ì��ë£Œì—�ë§Œ ì �ìš©ë�¨
temp_1.style.background_gradient(cmap='Blues')


# ì—°ë£Œë³„ë¡œ ì†ŒìŠ¤ì�˜ ë°œì „ ìš©ëŸ‰ì�„ í™•ì�¸
fig = px.treemap(full_grouped, 
                 path=['primary_fuel', 'source'],  # ê³„ì¸µ êµ¬ì¡°: ì—°ë£Œ -> ì†ŒìŠ¤
                 values='capacity_mw',  # íŠ¸ë¦¬ë§µì�˜ í�¬ê¸°: ë°œì „ ìš©ëŸ‰
                 color='capacity_mw',  # ìƒ‰ìƒ�: ë°œì „ ìš©ëŸ‰ í�¬ê¸°
                 hover_data=['capacity_mw', 'estimated_generation_gwh'],  # í˜¸ë²„ ì‹œ ì¶”ê°€ ì •ë³´
                 color_continuous_scale='Viridis',  # ìƒ‰ìƒ� ìŠ¤ì¼€ì�¼ ì„¤ì •
                 title='Capacity (MW) by Fuel Type and Source')

fig.show()

# ë°œì „ëŸ‰ì—� ëŒ€í•œ íŠ¸ë¦¬ë§µ ìƒ�ì„±
fig = px.treemap(full_grouped, 
                 path=['primary_fuel', 'source'],  # ê³„ì¸µ êµ¬ì¡°: ì—°ë£Œ -> ì†ŒìŠ¤
                 values='estimated_generation_gwh',  # íŠ¸ë¦¬ë§µì�˜ í�¬ê¸°: ì˜ˆìƒ� ë°œì „ëŸ‰
                 color='estimated_generation_gwh',  # ìƒ‰ìƒ�: ë°œì „ëŸ‰ í�¬ê¸°
                 hover_data=['capacity_mw', 'estimated_generation_gwh'],  # í˜¸ë²„ ì‹œ ì¶”ê°€ ì •ë³´
                 color_continuous_scale='Plasma',  # ìƒ‰ìƒ� ìŠ¤ì¼€ì�¼ ì„¤ì •
                 title='Estimated Generation (GWh) by Fuel Type and Source')

fig.show()



# 1. ë�°ì�´í„° melt: 'measure' ì—´ì—� ë‘� ê°€ì§€ ì¸¡ì •ì¹˜ë¥¼ êµ¬ë¶„í•˜ê³ , 'value' ì—´ì—� í•´ë‹¹ ê°’ì�„ ì €ì�¥
melted = full_grouped.melt(id_vars=['source', 'primary_fuel'], 
                           value_vars=['capacity_mw', 'estimated_generation_gwh'],
                           var_name='measure', value_name='value')

# 2. 3ë‹¨ê³„ íŠ¸ë¦¬ë§µ ìƒ�ì„±: ì²« ë²ˆì§¸ ê³„ì¸µì�€ source, ë‘� ë²ˆì§¸ëŠ” primary_fuel, ì„¸ ë²ˆì§¸ëŠ” measure
fig = px.treemap(melted,
                 path=['source', 'primary_fuel', 'measure'],
                 values='value',
                 color='value',
                 hover_data=['value'],
                 color_continuous_scale='Viridis',
                 title='3-Level Treemap: Source â†’ Fuel â†’ Measure (Capacity/Generation)')

fig.show()


# ìƒ�í‚¤ ë‹¤ì�´ì–´ê·¸ë�¨ìœ¼ë¡œ ë¶„ì„�í•´ ê¸°ê¸°

import plotly.graph_objects as go

full_grouped = power_plant.groupby(['primary_fuel','source'])[['capacity_mw', 'estimated_generation_gwh']].sum().reset_index()
full_grouped = full_grouped.sort_values(by='primary_fuel',ascending=False).reset_index(drop=True)

display(full_grouped)

# 1. ë�°ì�´í„° ë³€í™˜: meltë¥¼ ì‚¬ìš©í•˜ì—¬ 'measure' ì—´ ìƒ�ì„±
melted = full_grouped.melt(id_vars=['primary_fuel','source'], 
                           value_vars=['capacity_mw', 'estimated_generation_gwh'],
                           var_name='measure', value_name='value')
display(melted)

# 2. ê°� ê³„ì¸µë³„ ë…¸ë“œ ëª©ë¡� ìƒ�ì„±
fuel_nodes = list(melted['primary_fuel'].unique())         # ì—°ë£Œ ìœ í˜• ë…¸ë“œ
source_nodes = list(melted['source'].unique())               # ì†ŒìŠ¤(ë°œì „ì†Œ) ë…¸ë“œ
measure_nodes = list(melted['measure'].unique())             # ì¸¡ì •ì¹˜ ë…¸ë“œ (capacity_mw, estimated_generation_gwh)

# ëª¨ë“  ë…¸ë“œ ê²°í•©
node_labels= fuel_nodes + source_nodes + measure_nodes
display(node_labels)

# 3. ë§�í�¬ ìƒ�ì„±: ê°� í–‰ë§ˆë‹¤ ì—°ë£Œ â†’ ì†ŒìŠ¤, ì†ŒìŠ¤ â†’ ì¸¡ì •ì¹˜ ë‘� ë§�í�¬ ìƒ�ì„±
source_indices = []
target_indices = []
values = []

   

#iterrows(): DataFrameì�˜ ê°� í–‰ì�„ (ì�¸ë�±ìŠ¤, í–‰ ë�°ì�´í„°) íŠœí”Œ í˜•íƒœë¡œ ë°˜í™˜
for _, row in melted.iterrows():
    
    # ì—°ë£Œì—�ì„œ ì†ŒìŠ¤ë¡œì�˜ ë§�í�¬ 
    fuel_index = fuel_nodes.index(row['primary_fuel'])  # ì—°ë£Œ ê·¸ë£¹ ë‚´ ì�¸ë�±ìŠ¤
    src_index = source_nodes.index(row['source']) + len(fuel_nodes)  # ì†ŒìŠ¤ ê·¸ë£¹ì�˜ ì�¸ë�±ìŠ¤ (ì—°ë£Œ ë…¸ë“œ ê°œìˆ˜ offset)
    value = row['value']
    
    source_indices.append(fuel_index)
    target_indices.append(src_index)
    values.append(value)
    
    # ì†ŒìŠ¤ì—�ì„œ ì¸¡ì •ì¹˜ë¡œì�˜ ë§�í�¬
    measure_index = measure_nodes.index(row['measure']) + len(fuel_nodes) + len(source_nodes)  # ì¸¡ì •ì¹˜ ê·¸ë£¹ ì�¸ë�±ìŠ¤ (ì—°ë£Œ+ì†ŒìŠ¤ offset)
    source_indices.append(src_index)
    target_indices.append(measure_index)
    values.append(value)

# 4. Sankey ë‹¤ì�´ì–´ê·¸ë�¨ ìƒ�ì„±
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=node_labels,
        color="lightblue"
    ),
    link=dict(
        source=source_indices,
        target=target_indices,
        value=values
    )
)])

fig.update_layout(title_text="Sankey Diagram: Fuel Type â†’ Source â†’ Measure", font_size=10)
fig.show()



#total electricity generation in gigwatthour for one year

total_gen = power_plant['estimated_generation_gwh'].sum()
print('Total Generation :'+'{:.3f}'.format(total_gen)+'GW')


# percentage of total generation in gigawatthour

generation = (power_plant.groupby(['primary_fuel'])['estimated_generation_gwh'].sum()).to_frame()
generation = generation.sort_values('estimated_generation_gwh',ascending=False)
generation['percentage_of_total'] = (generation['estimated_generation_gwh']/total_gen)*100
generation


# ë°” ì°¨íŠ¸ ìƒ�ì„±
fig, ax = plt.subplots(figsize=(10, 6))

# cmapì�„ ì‚¬ìš©í•˜ì—¬ ë†’ì�€ ê°’ì�€ ë¶‰ì�€ìƒ‰, ë‚®ì�€ ê°’ì�€ íŒŒë�€ìƒ‰ìœ¼ë¡œ ìƒ‰ìƒ� ì �ìš©
cmap = plt.cm.coolwarm  # ì›�í•˜ëŠ” ìƒ‰ìƒ�ë§µ ì„¤ì •
norm = plt.Normalize(vmin=generation['percentage_of_total'].min(), vmax=generation['percentage_of_total'].max())  # ê°’ ë²”ìœ„ ì •ê·œí™”
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # ìƒ‰ìƒ�ë§µì�„ ê°’ì—� ë§¤í•‘

# ìƒ‰ìƒ� ì �ìš©í•˜ì—¬ ë°” ê·¸ë¦¬ê¸°
bars = ax.bar(generation.index, generation['percentage_of_total'], color=[sm.to_rgba(value) for value in generation['percentage_of_total']])

# ê°� ë§‰ëŒ€ ë†’ì�´(ì¹´ìš´íŠ¸ ê°’) ê°€ì ¸ì˜¤ê¸°
heights = [p.get_height() for p in ax.patches]

# ìµœì†Ÿê°’ & ìµœëŒ“ê°’ ê³„ì‚°
ymin, ymax = min(heights), max(heights)

# ìµœì†Ÿê°’ ë�¼ì�¸ (ë¹¨ê°„ ì �ì„ )
min_line = ax.axhline(y=ymin, color='red', linestyle='dashed', linewidth=1, label=f'Min: {ymin:.2f}%')

# ìµœëŒ“ê°’ ë�¼ì�¸ (íŒŒë�€ ì �ì„ )
max_line = ax.axhline(y=ymax, color='blue', linestyle='dashed', linewidth=1, label=f'Max: {ymax:.2f}%')

# ë§‰ëŒ€ ìœ„ì—� ê°’ í‘œì‹œ (ì†Œìˆ˜ì � 2ì��ë¦¬ê¹Œì§€)
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2,  # x ìœ„ì¹˜
            p.get_height() + 1,  # y ìœ„ì¹˜ (ë§‰ëŒ€ ë†’ì�´)
            f"{p.get_height():.2f}%",  # í‘œì‹œí•  ê°’
            ha='center', va='bottom', fontsize=10)  # ê°€ìš´ë�° ì •ë ¬, í�°íŠ¸ í�¬ê¸° ì„¤ì •

# ì œëª©ê³¼ ë ˆì�´ë¸” ì„¤ì •
plt.title('Percentage of Total Electricity Generation by Fuel Type', fontsize=16, pad=10)
plt.xlabel('Fuel Type', fontsize=14)
plt.ylabel('Percentage of Total Generation (%)', fontsize=14)

# y-axis ì—¬ë°± ëŠ˜ë¦¬ê¸°
plt.ylim(0, ymax + 10)  # yì¶• ë²”ìœ„ë¥¼ 0ë¶€í„° ymax + 10ê¹Œì§€ ì„¤ì •í•˜ì—¬ ì—¬ë°±ì�„ ë„“í�˜

# ê·¸ë�˜í”„ ì¶œë ¥
plt.xticks(ha='center')

# ë²”ë¡€ ì„¤ì •
plt.legend(loc='upper right', framealpha=1, facecolor='whitesmoke')

# ë ˆì�´ì•„ì›ƒ ìµœì �í™”
plt.tight_layout()

plt.show()


generation = (power_plant.groupby(['source'])['estimated_generation_gwh'].sum()).to_frame()
display(generation) #to_frame()ì�€ ì‹œë¦¬ì¦ˆë¥¼ ë�°ì�´í„° í”„ë ˆì�„ í˜•ì‹�ìœ¼ë¡œ ë°”ê¾¼ ê²ƒì�„

generation = generation.sort_values('estimated_generation_gwh',ascending=False)
display(generation)

generation['percentage_of_total'] = (generation['estimated_generation_gwh']/total_gen)*100
display(generation)


# ë°” ì°¨íŠ¸ ìƒ�ì„±
fig, ax = plt.subplots(figsize=(10, 6))

# cmapì�„ ì‚¬ìš©í•˜ì—¬ ë†’ì�€ ê°’ì�€ ë¶‰ì�€ìƒ‰, ë‚®ì�€ ê°’ì�€ íŒŒë�€ìƒ‰ìœ¼ë¡œ ìƒ‰ìƒ� ì �ìš©
cmap = plt.cm.coolwarm  # ì›�í•˜ëŠ” ìƒ‰ìƒ�ë§µ ì„¤ì •
norm = plt.Normalize(vmin=generation['percentage_of_total'].min(), vmax=generation['percentage_of_total'].max())  # ê°’ ë²”ìœ„ ì •ê·œí™”
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # ìƒ‰ìƒ�ë§µì�„ ê°’ì—� ë§¤í•‘

# ìƒ‰ìƒ� ì �ìš©í•˜ì—¬ ìˆ˜í�‰ ë°” ê·¸ë¦¬ê¸°
bars = ax.barh(generation.index, generation['percentage_of_total'], color=[sm.to_rgba(value) for value in generation['percentage_of_total']])

# ê°� ë§‰ëŒ€ ê¸¸ì�´(ê°’) ê°€ì ¸ì˜¤ê¸°
widths = [p.get_width() for p in bars]

# ìµœì†Ÿê°’ & ìµœëŒ“ê°’ ê³„ì‚° (xì¶• ê°’ ê¸°ì¤€)
xmin, xmax = min(widths), max(widths)

# ìµœì†Ÿê°’ ë�¼ì�¸ (ë¹¨ê°„ ì �ì„ )
ax.axvline(x=xmin, color='red', linestyle='dashed', linewidth=1, label=f'Min: {xmin:.2f}%')

# ìµœëŒ“ê°’ ë�¼ì�¸ (íŒŒë�€ ì �ì„ )
ax.axvline(x=xmax, color='blue', linestyle='dashed', linewidth=1, label=f'Max: {xmax:.2f}%')

# ë§‰ëŒ€ ìœ„ì—� ê°’ í‘œì‹œ (ì†Œìˆ˜ì � 2ì��ë¦¬ê¹Œì§€)
for p in bars:
    ax.text(p.get_width() + 1,  # x ìœ„ì¹˜ (ìˆ˜í�‰ ë°©í–¥ìœ¼ë¡œ)
            p.get_y() + p.get_height() / 2,  # y ìœ„ì¹˜ (ìˆ˜ì§� ë°©í–¥ìœ¼ë¡œ)
            f"{p.get_width():.2f}%",  # í‘œì‹œí•  ê°’
            va='center', fontsize=10)  # ê°€ìš´ë�° ì •ë ¬, í�°íŠ¸ í�¬ê¸° ì„¤ì •

# ì œëª©ê³¼ ë ˆì�´ë¸” ì„¤ì •
plt.title('Percentage of Total Electricity Generation by Sources', fontsize=16, pad=10)
plt.xlabel('Percentage of Total Generation (%)', fontsize=14)
plt.ylabel('Sources', fontsize=14)

# x-axis ì—¬ë°± ëŠ˜ë¦¬ê¸°
plt.xlim(0, xmax + 10)  # xì¶• ë²”ìœ„ë¥¼ 0ë¶€í„° xmax + 10ê¹Œì§€ ì„¤ì •í•˜ì—¬ ì—¬ë°±ì�„ ë„“í�˜

# ê·¸ë�˜í”„ ì¶œë ¥
plt.yticks(ha='right')  # yì¶• ë ˆì�´ë¸”ì�„ ë³´ê¸° ì¢‹ê²Œ ì •ë ¬

# ë²”ë¡€ ì„¤ì •
plt.legend(loc='upper right', framealpha=1, facecolor='whitesmoke')

# ë ˆì�´ì•„ì›ƒ ìµœì �í™”
plt.tight_layout()

plt.show()


#total production capcity 

total_cap = power_plant['capacity_mw'].sum()
print('Total Capcity :'+'{:.3f}'.format(total_cap)+'MW')

capcity = (power_plant.groupby(['primary_fuel'])['capacity_mw'].sum()).to_frame()
capcity = capcity.sort_values('capacity_mw',ascending=False)
capcity['percentage_of_total'] = (capcity['capacity_mw']/total_cap)*100

display(capcity)


fig = plt.gcf()
fig.set_size_inches(10, 6)
colors = ['dodgerblue', 'plum', '#F0A30A','#8c564b','orange','green','yellow'] 

bars = capcity['percentage_of_total'].plot(kind='bar',color=colors)

# ë°” ìœ„ì—� ê°’ í‘œì‹œ
for p in bars.patches:
    bars.text(p.get_x() + p.get_width() / 2,  # x ìœ„ì¹˜
              p.get_height() + 1,  # y ìœ„ì¹˜ (ë§‰ëŒ€ ë†’ì�´)
              f"{p.get_height():.2f}%",  # ê°’ í‘œì‹œ (ì†Œìˆ˜ì � 2ì��ë¦¬ë¡œ)
              ha='center', va='bottom', fontsize=10)  # ê°€ìš´ë�° ì •ë ¬

ymin, ymax = plt.ylim()
plt.ylim(0, 100)
plt.xticks(rotation=0)
plt.title("Capacity Percentage by Fuels", pad=10, va="center")
plt.show()


# ì „ì²´ ë°œì „ ìš©ëŸ‰ (capacity_mwì�˜ í•©)
total_cap = power_plant['capacity_mw'].sum()
print('Total Capacity :' + '{:.3f}'.format(total_cap) + ' MW')

# ê°� ì—°ë£Œë³„ ìš©ëŸ‰ê³¼ í�¼ì„¼íŠ¸ ê³„ì‚°
capcity = (power_plant.groupby(['primary_fuel'])['capacity_mw'].sum()).to_frame()
capcity = capcity.sort_values('capacity_mw', ascending=False)
capcity['percentage_of_total'] = (capcity['capacity_mw'] / total_cap) * 100

# ë�°ì�´í„° ì¶œë ¥
display(capcity)

#######################################################################################

# ê·¸ë�˜í”„ ìƒ�ì„±

fig, ax1 = plt.subplots(figsize=(10, 6))

# ë§‰ëŒ€ ê·¸ë�˜í”„ (capacity_mw) - ì™¼ìª½ yì¶•
colors = ['dodgerblue', 'plum', '#F0A30A', '#8c564b', 'orange']
bars = ax1.bar(capcity.index, capcity['capacity_mw'], color=colors)

# ë°” ìœ„ì—� ê°’ í‘œì‹œ (capacity_mw)
for p in bars:
    ax1.text(p.get_x() + p.get_width() / 2, p.get_height() + 100,  # x ìœ„ì¹˜, y ìœ„ì¹˜
             f"{p.get_height():,.0f} MW",  # í‘œì‹œí•  ê°’
             ha='center', va='bottom', fontsize=10)  # ê°€ìš´ë�° ì •ë ¬

# ì™¼ìª½ yì¶• ì„¤ì •
ax1.set_xlabel('Fuel Type', fontsize=14)
ax1.set_ylabel('Capacity (MW)', fontsize=14)
ax1.set_xticklabels(capcity.index, rotation=0)
ax1.set_ylim(0, capcity['capacity_mw'].max() + 1000)  # yì¶• ì—¬ë°± ì¶”ê°€


# ì˜¤ë¥¸ìª½ yì¶• (percentage_of_total) - ì„  ê·¸ë�˜í”„
ax2 = ax1.twinx()  # ax1ì�˜ ì˜¤ë¥¸ìª½ì—� yì¶• ì¶”ê°€
ax2.plot(capcity.index, capcity['percentage_of_total'], color='green', marker='o', label='Percentage of Total', linewidth=2, alpha=0.5)

# ì˜¤ë¥¸ìª½ yì¶• ì„¤ì •
ax2.set_ylabel('Percentage of Total Generation (%)', fontsize=14)
ax2.set_ylim(0, 100)  # í�¼ì„¼íŠ¸ëŠ” 0ë¶€í„° 100ê¹Œì§€

# ì œëª©
plt.title("Capacity and Percentage by Fuel Type", pad=10)

# ë²”ë¡€ (ì„  ê·¸ë�˜í”„ì—�ë§Œ ë²”ë¡€ ì¶”ê°€)
ax2.legend(loc='upper right')

# ë ˆì�´ì•„ì›ƒ ìµœì �í™”
plt.tight_layout()
plt.show()



capcity = (power_plant.groupby(['source'])['capacity_mw'].sum()).to_frame()
capcity = capcity.sort_values('capacity_mw',ascending=False)
capcity['percentage_of_total'] = (capcity['capacity_mw']/total_cap)*100
capcity


#######################################################################################
from textwrap import wrap

# ê·¸ë�˜í”„ ìƒ�ì„±

fig, ax1 = plt.subplots(figsize=(20, 5))

# ë§‰ëŒ€ ê·¸ë�˜í”„ (capacity_mw) - ì™¼ìª½ yì¶•
colors = ['dodgerblue', 'plum', '#F0A30A', '#8c564b', 'orange']
bars = ax1.bar(capcity.index, capcity['capacity_mw'], color=colors)

# ë°” ìœ„ì—� ê°’ í‘œì‹œ (capacity_mw)
for p in bars:
    ax1.text(p.get_x() + p.get_width() / 2, p.get_height() + 100,  # x ìœ„ì¹˜, y ìœ„ì¹˜
             f"{p.get_height():,.0f} MW",  # í‘œì‹œí•  ê°’
             ha='center', va='bottom', fontsize=10)  # ê°€ìš´ë�° ì •ë ¬

# ì™¼ìª½ yì¶• ì„¤ì •
ax1.set_xlabel('source', fontsize=14)
ax1.set_ylabel('Capacity (MW)', fontsize=14)
ax1.set_ylim(0, capcity['capacity_mw'].max() + 1000)  # yì¶• ì—¬ë°± ì¶”ê°€

# ë�¼ë²¨ ê¸¸ì�´ê°€ ë„ˆë¬´ ê¸¸ë•Œ
ax1.set_xticklabels(['\n'.join(wrap(label, width=10)) for label in capcity.index], rotation=0)

# ì˜¤ë¥¸ìª½ yì¶• (percentage_of_total) - ì„  ê·¸ë�˜í”„
ax2 = ax1.twinx()  # ax1ì�˜ ì˜¤ë¥¸ìª½ì—� yì¶• ì¶”ê°€
ax2.plot(capcity.index, capcity['percentage_of_total'], color='green', marker='o', label='Percentage of Total', linewidth=2, alpha=0.5)

# ì˜¤ë¥¸ìª½ yì¶• ì„¤ì •
ax2.set_ylabel('Percentage of Total Generation (%)', fontsize=14)
ax2.set_ylim(0, 100)  # í�¼ì„¼íŠ¸ëŠ” 0ë¶€í„° 100ê¹Œì§€

# ì œëª©
plt.title("Capacity and Percentage by source", pad=10)

# ë²”ë¡€ (ì„  ê·¸ë�˜í”„ì—�ë§Œ ë²”ë¡€ ì¶”ê°€)
ax2.legend(loc='upper right')

# ë ˆì�´ì•„ì›ƒ ìµœì �í™”
plt.tight_layout()
plt.show()



"""
1. plot_points_on_map
ì�´ í•¨ìˆ˜ëŠ” ë�°ì�´í„°í”„ë ˆì�„ì�˜ íŠ¹ì • ë²”ìœ„(í–‰)ì—�ì„œ ìœ„ë�„ ë°� ê²½ë�„ ê°’ì�„ ì‚¬ìš©í•´ folium ì§€ë�„ ìœ„ì—� ë§ˆì»¤ë¥¼ í‘œì‹œí•˜ëŠ” í•¨ìˆ˜ì�…ë‹ˆë‹¤.
ë§¤ê°œë³€ìˆ˜:
dataframe: ë§ˆì»¤ë¥¼ ì¶”ê°€í•  ë�°ì�´í„°í”„ë ˆì�„
begin_index, end_index: ë�°ì�´í„°í”„ë ˆì�„ì—�ì„œ ì„ íƒ�í•  ë²”ìœ„ì�˜ ì�¸ë�±ìŠ¤ (ìŠ¬ë�¼ì�´ì‹±ìš©)
latitude_column, longitude_column: ìœ„ë�„ ë°� ê²½ë�„ ê°’ì�„ í�¬í•¨í•˜ëŠ” ì—´ ì�´ë¦„
latitude_value, longitude_value: ì§€ë�„ì�˜ ì¤‘ì‹¬ ì¢Œí‘œë¡œ ì‚¬ìš©í•  ìœ„ë�„ ë°� ê²½ë�„ ê°’
zoom: ì§€ë�„ì�˜ ì¤Œ ë ˆë²¨
ê¸°ëŠ¥: ì£¼ì–´ì§„ ì�¸ë�±ìŠ¤ ë²”ìœ„ ë‚´ì—�ì„œ ê°� í–‰ì—� ëŒ€í•´ ë§ˆì»¤ë¥¼ ì¶”ê°€í•˜ê³ , ë§ˆì»¤ì—�ëŠ” primary_fuel ì—´ì�˜ ê°’ì�„ íŒ�ì—…ìœ¼ë¡œ í‘œì‹œí•©ë‹ˆë‹¤.
ê²°ê³¼: ì§€ë�„ ê°�ì²´ë¥¼ ë°˜í™˜í•©ë‹ˆë‹¤.
"""

def plot_points_on_map(dataframe,begin_index,end_index,latitude_column,latitude_value,longitude_column,longitude_value,zoom):
    df = dataframe[begin_index:end_index]
    location = [latitude_value,longitude_value]
    plot = folium.Map(location=location,zoom_start=zoom)
    for i in range(0,len(df)):
        popup = folium.Popup(str(df.primary_fuel[i:i+1]))
        folium.Marker([df[latitude_column].iloc[i],df[longitude_column].iloc[i]],popup=popup).add_to(plot)
    return(plot)




"""
2. overlay_image_on_puerto_rico
ì�´ í•¨ìˆ˜ëŠ” Puerto Rico ì§€ì—­ì�˜ íŠ¹ì • ì�´ë¯¸ì§€ íŒŒì�¼ì�„ ì§€ë�„ì—� ì˜¤ë²„ë ˆì�´í•˜ëŠ” í•¨ìˆ˜ì�…ë‹ˆë‹¤.
rasterio ë�¼ì�´ë¸ŒëŸ¬ë¦¬ë¥¼ ì‚¬ìš©í•´ ì�´ë¯¸ì§€ì�˜ íŠ¹ì • ë°´ë“œë¥¼ ì�½ê³  ì�´ë¥¼ folium ì§€ë�„ì—� ì˜¤ë²„ë ˆì�´í•©ë‹ˆë‹¤.

ë§¤ê°œë³€ìˆ˜:
file_name: ì˜¤ë²„ë ˆì�´í•  ì�´ë¯¸ì§€ íŒŒì�¼ ê²½ë¡œ
band_layer: ì‚¬ìš©í•  ë°´ë“œì�˜ ë²ˆí˜¸ (ì�´ë¯¸ì§€ íŒŒì�¼ ë‚´ íŠ¹ì • ë ˆì�´ì–´)
ê¸°ëŠ¥: ì§€ì •ë�œ ì�´ë¯¸ì§€ì�˜ ë°´ë“œë¥¼ ì�½ê³ , ê·¸ ë°´ë“œë¥¼ íŠ¹ì • ìœ„ë�„/ê²½ë�„ ê²½ê³„ ë²”ìœ„(í‘¸ì—�ë¥´í† ë¦¬ì½”)ì—� ë§�ì¶° ì˜¤ë²„ë ˆì�´í•©ë‹ˆë‹¤.
foliumì�˜ ImageOverlayë¥¼ ì‚¬ìš©í•˜ì—¬ ì§€ë�„ ìœ„ì—� ì�´ë¯¸ì§€ë¥¼ ì‹œê°�í™”í•©ë‹ˆë‹¤.
ê²°ê³¼: ì§€ë�„ ê°�ì²´ë¥¼ ë°˜í™˜í•©ë‹ˆë‹¤.
"""

def overlay_image_on_puerto_rico(file_name,band_layer):
    band = rio.open(file_name).read(band_layer)
    m = folium.Map([lat, lon], zoom_start=8)
    folium.raster_layers.ImageOverlay(
        image=band,
        bounds = [[18.6,-67.3,],[17.9,-65.2]],
        colormap=lambda x: (1, 0, 0, x),
    ).add_to(m)
    return m



"""
3. plot_scaled
ì�´ í•¨ìˆ˜ëŠ” ì£¼ì–´ì§„ íŒŒì�¼ ë�°ì�´í„°ë¥¼ 5-95% ë²”ìœ„ë¡œ ì •ê·œí™”í•˜ì—¬ ì�´ë¯¸ì§€ë¥¼ í‘œì‹œí•©ë‹ˆë‹¤.

ë§¤ê°œë³€ìˆ˜:
file_name: ì�´ë¯¸ì§€ë¥¼ í‘œì‹œí•  ë�°ì�´í„° (ì�´ë¯¸ì§€ ë°°ì—´)
ê¸°ëŠ¥: ì�…ë ¥ ë�°ì�´í„°ë¥¼ 5%ì—�ì„œ 95% ì‚¬ì�´ì�˜ ê°’ìœ¼ë¡œ ì •ê·œí™”í•˜ê³  imshowë¡œ ê·¸ë ˆì�´ìŠ¤ì¼€ì�¼ë¡œ ì�´ë¯¸ì§€ë¥¼ ì¶œë ¥í•©ë‹ˆë‹¤.
ê²°ê³¼: ì •ê·œí™”ë�œ ì�´ë¯¸ì§€ë¥¼ í™”ë©´ì—� í‘œì‹œí•©ë‹ˆë‹¤.
"""

def plot_scaled(file_name):
    vmin, vmax = np.nanpercentile(file_name, (5,95))  # 5-95% stretch
    img_plt = plt.imshow(file_name, cmap='gray', vmin=vmin, vmax=vmax)
    plt.show()



"""
4. split_column_into_new_columns
ì�´ í•¨ìˆ˜ëŠ” ë�°ì�´í„°í”„ë ˆì�„ì�˜ í•œ ì—´ì—�ì„œ ì�¼ë¶€ ë¬¸ì��ì—´ì�„ ì¶”ì¶œí•˜ì—¬ ìƒˆë¡œìš´ ì—´ì—� ì €ì�¥í•˜ëŠ” í•¨ìˆ˜ì�…ë‹ˆë‹¤.

ë§¤ê°œë³€ìˆ˜:
dataframe: ë�°ì�´í„°ë¥¼ í�¬í•¨í•œ ë�°ì�´í„°í”„ë ˆì�„
column_to_split: ë�°ì�´í„°ë¥¼ ë‚˜ëˆŒ ì›�ë³¸ ì—´ ì�´ë¦„
new_column_one: ìƒˆë¡œìš´ ì—´ì�˜ ì�´ë¦„
begin_column_one, end_column_one: ì¶”ì¶œí•  ë¬¸ì��ì—´ì�˜ ì‹œì�‘ê³¼ ë�� ì�¸ë�±ìŠ¤
ê¸°ëŠ¥: ì£¼ì–´ì§„ ì—´ì—�ì„œ ë¬¸ì��ì—´ì�„ ì¶”ì¶œí•˜ì—¬ new_column_oneì—� ì €ì�¥í•©ë‹ˆë‹¤. ê°� í–‰ì—� ëŒ€í•´ ì§€ì •ë�œ ë²”ìœ„ì�˜ ë¬¸ì��ì—´ì�„ ì��ë¥´ê³  ìƒˆë¡œìš´ ì—´ì—� í• ë‹¹í•©ë‹ˆë‹¤.
ê²°ê³¼: ìƒˆë¡œìš´ ì—´ì�´ ì¶”ê°€ë�œ ë�°ì�´í„°í”„ë ˆì�„ì�„ ë°˜í™˜í•©ë‹ˆë‹¤.
"""

def split_column_into_new_columns(dataframe,column_to_split,new_column_one,begin_column_one,end_column_one):
    for i in range(0, len(dataframe)):
        dataframe.loc[i, new_column_one] = dataframe.loc[i, column_to_split][begin_column_one:end_column_one]
    return dataframe



pd.set_option('display.max_colwidth', None)

power_plant = pd.read_csv('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')
geo = power_plant[['.geo']]

geo



# ìœ„ ë�°ì�´í„°ì—�ì„œëŠ” ìœ„ë�„/ê²½ë�„ê°€ ì•„ë‹Œ ê²½ë�„/ìœ„ë�„ ìˆœì„œë¡œ í‘œì‹œë�˜ì–´ ì�ˆì�Œ.
power_plant = split_column_into_new_columns(power_plant,'.geo','longitude',31,48) #ê²½ë�„
power_plant = split_column_into_new_columns(power_plant,'.geo','latitude',50,66) #ìœ„ë�„

power_plant['latitude'] = power_plant['latitude'].astype(float)

a = np.array(power_plant['latitude'].values.tolist()) # 18 instead of 8

power_plant['latitude'] = np.where(a < 10, a+10, a).tolist() 

lat=18.200178; lon=-66.664513

plot_points_on_map(power_plant,0,425,'latitude',18,'longitude',-66.4, 9.3)

