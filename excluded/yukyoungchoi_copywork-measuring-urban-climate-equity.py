import os
import gc
import datetime as dt

import warnings
warnings.filterwarnings('ignore')

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from matplotlib.patches import Circle, Rectangle, Arc

%matplotlib inline
%config InlineBackend.figure_format = 'retina'


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


!apt-get update -qq && apt-get install -y fonts-nanum


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


# import corporate response data

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
                print(f"  â””â”€â”€ {filename}\n")  # ë§ˆì§€ë§‰ íŒŒì�¼ (5ê°œ ì�´í•˜ì�¸ ê²½ìš°)
            else:
                print(f"  â”œâ”€â”€ {filename}")  # ì�¼ë°˜ íŒŒì�¼ ì¶œë ¥
        
        if file_count > 5:
            print(f"  â””â”€â”€ ... (ì´� {file_count}ê°œ íŒŒì�¼)")  # 5ê°œ ì´ˆê³¼ ì‹œ ìš”ì•½ ì¶œë ¥



# ì‹œê°�ì �ìœ¼ë¡œ í‘œí˜„í•˜ê³  ì‹¶ë‹¤ë©´ ì�´ë ‡ê²Œ í•  ìˆ˜ ì�ˆë‹¤

import os
from graphviz import Digraph
from IPython.display import Image

# ì‹œì�‘ ë””ë ‰í† ë¦¬
root_dir = "/kaggle/input"
dot = Digraph(comment="Directory Tree", format='png')
dot.attr(rankdir='LR')  # ê°€ë¡œ ë°©í–¥ íŠ¸ë¦¬ë¡œ ì„¤ì • (ì„ íƒ� ì‚¬í•­)

# ê²½ë¡œë¥¼ ë…¸ë“œë¡œ ì¶”ê°€í•˜ëŠ” í•¨ìˆ˜
def add_nodes(dot, parent, path):
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        node_id = full_path.replace("/", "_")  # ê³ ìœ  ID
        dot.node(node_id, entry)
        dot.edge(parent, node_id)

        if os.path.isdir(full_path):
            add_nodes(dot, node_id, full_path)

# ë£¨íŠ¸ ë…¸ë“œ ì¶”ê°€
root_node = root_dir.replace("/", "_")
dot.node(root_node, os.path.basename(root_dir))
add_nodes(dot, root_node, root_dir)

# íŠ¸ë¦¬ ì‹œê°�í™” ì €ì�¥ ë°� í‘œì‹œ
output_path = dot.render('directory_tree', view=True)

# ì�´ë¯¸ì§€ í‘œì‹œ
Image(filename=output_path)




cc_df = pd.read_csv('../input/cdp-unlocking-climate-solutions/Corporations/Corporations Responses/Climate Change/2019_Full_Climate_Change_Dataset.csv')
ws_df = pd.read_csv('../input/cdp-unlocking-climate-solutions/Corporations/Corporations Responses/Water Security/2019_Full_Water_Security_Dataset.csv')


# Import libraries

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the data from the 2020 Questionnaire

fc_df = pd.read_csv("../input/cdp-unlocking-climate-solutions/Cities/Cities Responses/2020_Full_Cities_Dataset.csv")

print(fc_df.shape)
fc_df.head()


# Read in the general data about the cities that responded to the 2020 questionnaire

cities_df = pd.read_csv("../input/cdp-unlocking-climate-solutions/Cities/Cities Disclosing/2020_Cities_Disclosing_to_CDP.csv")
print(cities_df.shape)
cities_df.head()


# In this data, "Kansas City" includes Kansas City, MO, but not Kansas City, KS.

cities_df.loc[cities_df["Organization"] == "Kansas City"]


# Read in the cleaned geospatial data provided by Kaggle (shabou) and merge.
# Note: The original geospatial data in "City Location" has many
# missing values and errors.

# Read in the cleaned geospatial dataset from Kaggle
city_coords = pd.read_csv("../input/cdp-challenge-cities-geolocation-data/CDP-Cities-goegraphical-coordinates.csv")

# Change the key column name to facilitate merge
city_coords.rename(columns={"Account.Number":"Account Number"}, inplace=True)

# Merge the geospatial coordinates into the cities dataframe
cities_df = pd.merge(cities_df, city_coords[["Account Number","lat", "long"]])

# Convert to a GeoDataFrame
cities_gdf = gpd.GeoDataFrame(cities_df, geometry=
                             gpd.points_from_xy(cities_df['long'], cities_df['lat']))

# Set the Coordinate Reference System
cities_gdf.crs = "epsg:4326"

# Drop the old city location column
cities_gdf.drop(columns="City Location", inplace=True)

# Import a world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Remove Antarctica from the map
world = world[(world.pop_est>0) & (world.name!="Antarctica")]

# Plot the city locations on top of the world map
fig, ax = plt.subplots(figsize=(20,20))
ax.set_title("Geographical Distribution of Cities in the 2020 CDP Dataset")
ax.set_aspect('equal')
world.plot(ax=ax, color='lightgrey', edgecolor='white')
cities_gdf.plot(ax=ax, marker='o', color='red', markersize=15);


# Visualize the distribution of cities by region

cities_by_region = cities_df["CDP Region"].value_counts(ascending=True)

plt.barh(y = cities_by_region.keys(), width = cities_by_region.values)
plt.title('Cities by Region');


# Create a new dataframe to hold the hazards reported by each city (in 2020)

# Extract the data
hazards_df = fc_df.loc[(fc_df["Year Reported to CDP"] == 2020) &
                       (fc_df["Question Number"] == "2.1") &
                       (fc_df["Column Number"] == 1) &
                       (~fc_df["Response Answer"].isna()),
                       ["Account Number", "Organization", "Country", "CDP Region", "Year Reported to CDP", "Row Number", "Response Answer"]
                      ].sort_values(by=["Account Number", "Row Number"])

# Rename the columns
hazards_df.rename(columns={"Year Reported to CDP": "Year", "Response Answer": "Hazard"}, inplace=True)

# Add a column to indicate whether the hazards increased risk to already vulnerable populations.

# Extract the data from the survey
data = fc_df.loc[(fc_df["Year Reported to CDP"] == 2020) &
                 (fc_df["Question Number"] == "2.1") &
                 (fc_df["Column Number"] == 5) &
                 (fc_df["Response Answer"] == "Increased risk to already vulnerable populations"),
                 ["Account Number", "Row Number", "Response Answer"]
                ].groupby(["Account Number", "Row Number"])["Response Answer"].count()

# Merge the data into the df
hazards_df = pd.merge(hazards_df, data, how="left", on=["Account Number", "Row Number"])

# Give the column a more descriptive name
hazards_df.rename(columns={"Response Answer": "Risk to VPs"}, inplace=True)

# Fill in missing values with 0 and convert the entire column to integers
hazards_df["Risk to VPs"] = hazards_df["Risk to VPs"].fillna(value=0)
hazards_df["Risk to VPs"] = hazards_df["Risk to VPs"].astype(int)

# Add columns for the vulnearble populations impacted by the hazards.

# Extract the data from the survey
data = fc_df.loc[(fc_df["Year Reported to CDP"] == 2020) &
                 (fc_df["Question Number"] == "2.1") &
                 (fc_df["Column Number"] == 7),
                 ["Account Number", "Row Number", "Response Answer"]
                ]

# Simplify all "Other..." responses to "Other"
data.loc[(~data["Response Answer"].isna()) &
               (data["Response Answer"].str.startswith("Other")),
               "Response Answer"] = "Other"

# Dummy the new column
data = pd.get_dummies(data, columns=["Response Answer"], prefix="", prefix_sep="")

# Aggregate the data for each distinct hazard
data = data.groupby(["Account Number", "Row Number"]).sum()

# Add a column for the total number of VPs affected
data["Total VPs Affected"] = data.sum(axis=1)

# Merge the data into the df
hazards_df = pd.merge(hazards_df, data, how="left", on=["Account Number", "Row Number"])

# Clean up the column names
hazards_df.rename(columns={"Row Number": "Hazard Number", "Hazard": "Hazard Type"}, inplace=True)

print(hazards_df.shape)
hazards_df.head()


# Which hazards do cities most commonly face?

most_common_hazards = hazards_df.groupby("Hazard Type")["Hazard Number"].count().sort_values(ascending=True)

plt.figure(figsize=(10,10))
plt.barh(y = most_common_hazards.keys(), width = most_common_hazards.values)
plt.title('Number of Hazards Reported (2020)');


# What percentage of hazards do cities report are increasing the risk to vulnerable populations?

total = hazards_df["Risk to VPs"].count()
risk = hazards_df["Risk to VPs"].sum()
perc = round((risk/total)*100, 1)

plt.pie(x = [risk, total-risk], labels=["risk", "no risk"]);


# Which hazards most frequently affect already vulnerable populations?

freq_impact_on_vps = hazards_df.groupby("Hazard Type")["Risk to VPs"].mean().sort_values(ascending=True)

plt.figure(figsize=(10,10))
plt.barh(y = freq_impact_on_vps.keys(), width = freq_impact_on_vps.values)
plt.title('Frequency of Impact on Vulnerable Populations (2020)');


# Which hazards have the greatest total impact on vulnerable populations?

most_impact_on_vps = hazards_df.groupby("Hazard Type")["Total VPs Affected"].sum().sort_values(ascending=True)

plt.figure(figsize=(10,10))
plt.barh(y = most_impact_on_vps.keys(), width = most_impact_on_vps.values)
plt.title('Number of Vulnerable Populations Affected by Hazard Type (2020)');


# Which hazards have the greatest average impact on vulnerable populations?

most_impact_on_vps = hazards_df.groupby("Hazard Type")["Total VPs Affected"].mean().sort_values(ascending=True)

plt.figure(figsize=(10,10))
plt.barh(y = most_impact_on_vps.keys(), width = most_impact_on_vps.values)
plt.title('Average Number of Vulnerable Populations Affected per Incident (2020)');


# Which vulnerable populations are most affected by climate hazards overall?

most_affected_vps = hazards_df.iloc[:, 8:19].sum().sort_values(ascending=True)

plt.figure(figsize=(10,5))
plt.barh(y = most_affected_vps.keys(), width = most_affected_vps.values)
plt.title('Number of Vulnerable Populations Affected by Hazards (2020)');


# Which vulnerable populations does each type of hazard affect the most?
# Example 1: Forest fires

forest_fires = hazards_df.loc[hazards_df["Hazard Type"] == "Wild fire > Forest fire"].iloc[:, 8:19].sum().sort_values(ascending=True)

plt.figure(figsize=(10,5))
plt.barh(y = forest_fires.keys(), width = forest_fires.values)
plt.title('Number of Vulnerable Populations Affected by Forest Fires (2020)');


# Which vulnerable populations does each type of hazard affect the most?
# Example 2: Cyclones (hurricanes and typhoons)

cyclones = hazards_df.loc[hazards_df["Hazard Type"] == "Storm and wind > Cyclone (Hurricane / Typhoon)"].iloc[:, 8:19].sum().sort_values(ascending=True)

plt.figure(figsize=(10,5))
plt.barh(y = cyclones.keys(), width = cyclones.values)
plt.title('Number of Vulnerable Populations Affected by Cyclones (2020)');


# Create a new dataframe to hold the actions reported by each city (in 2020)

# Extract the data
actions_df = fc_df.loc[(fc_df["Year Reported to CDP"] == 2020) &
                       (fc_df["Question Number"] == "3.0") &
                       (fc_df["Column Number"] == 2) &
                       (~fc_df["Response Answer"].isna())&
                       (fc_df["Response Answer"] != "No action currently taken"),
                       ["Account Number", "Organization", "Country", "CDP Region", "Year Reported to CDP", "Row Number", "Response Answer"]
                      ].sort_values(by=["Account Number", "Row Number"])

# Simplify all "Other..." responses to "Other"
actions_df.loc[(~actions_df["Response Answer"].isna()) &
               (actions_df["Response Answer"].str.startswith("Other")),
               "Response Answer"] = "Other"

# Rename the columns
actions_df.rename(columns={"Year Reported to CDP": "Year", "Response Answer": "Action"}, inplace=True)

# Add a column for the hazards targeted by the actions.

# Extract the data from the survey
data = fc_df.loc[(fc_df["Year Reported to CDP"] == 2020) &
                 (fc_df["Question Number"] == "3.0") &
                 (fc_df["Column Number"] == 1),
                 ["Account Number", "Row Number", "Response Answer"]
                ]

# Merge the data into the df
actions_df = pd.merge(actions_df, data, how="left", on=["Account Number", "Row Number"])

# Give the column a more descriptive name
actions_df.rename(columns={"Response Answer": "Hazard"}, inplace=True)

# Add a column to indicate whether the action benefits poverty reduction.

# Extract the data from the survey
data = fc_df.loc[(fc_df["Year Reported to CDP"] == 2020) &
                 (fc_df["Question Number"] == "3.0") &
                 (fc_df["Column Number"] == 6) &
                 (fc_df["Response Answer"] == "Poverty reduction / eradication"),
                 ["Account Number", "Row Number", "Response Answer"]
                ].groupby(["Account Number", "Row Number"])["Response Answer"].count()

# Merge the data into the df
actions_df = pd.merge(actions_df, data, how="left", on=["Account Number", "Row Number"])

# Give the column a more descriptive name
actions_df.rename(columns={"Response Answer": "Poverty Reduction"}, inplace=True)

# Clean up the values in the new column
actions_df["Poverty Reduction"] = actions_df["Poverty Reduction"].fillna(value=0)
actions_df["Poverty Reduction"] = actions_df["Poverty Reduction"].astype(int)

# Add a column to indicate whether the action benefits social inclusion.

# Extract the data from the survey
data = fc_df.loc[(fc_df["Year Reported to CDP"] == 2020) &
                 (fc_df["Question Number"] == "3.0") &
                 (fc_df["Column Number"] == 6) &
                 (fc_df["Response Answer"] == "Social inclusion, social justice"),
                 ["Account Number", "Row Number", "Response Answer"]
                ].groupby(["Account Number", "Row Number"])["Response Answer"].count()

# Merge the data into the df
actions_df = pd.merge(actions_df, data, how="left", on=["Account Number", "Row Number"])

# Give the column a more descriptive name
actions_df.rename(columns={"Response Answer": "Social Inclusion"}, inplace=True)

actions_df["Social Inclusion"] = actions_df["Social Inclusion"].fillna(value=0)
actions_df["Social Inclusion"] = actions_df["Social Inclusion"].astype(int)

# Clean up the column names

actions_df.rename(columns={"Row Number": "Action Number", "Action": "Action Type", "Hazard": "Hazard Type"}, inplace=True)

print(actions_df.shape)
actions_df.head()


# What are the most common types of actions that cities take to mitigate climate hazards?

most_common_actions = actions_df.groupby("Action Type")["Action Number"].count().sort_values(ascending=True)

plt.figure(figsize=(10,15))
plt.barh(y = most_common_actions.keys(), width = most_common_actions.values)
plt.title('Number of Actions Reported per Action Type (2020)');


# Which types of hazards had the most actions in response to them?

most_targetted_hazards = actions_df.groupby("Hazard Type")["Action Number"].count().sort_values(ascending=True)

plt.figure(figsize=(10,15))
plt.barh(y = most_targetted_hazards.keys(), width = most_targetted_hazards.values)
plt.title('Number of Actions Reported per Hazard Type (2020)');


# What percentage of hazard responses include actions targeting the most vulnerable?


vp_actions = actions_df.loc[actions_df["Action Type"] == "Projects and policies targeted at those most vulnerable",
               "Action Type"
              ].count()
hazard_responses = actions_df.groupby("Account Number")["Hazard Type"].count().sum()
percentage = round((vp_actions/hazard_responses)*100)

plt.pie(x = [vp_actions, hazard_responses-vp_actions], labels=["vp actions", "no vp actions"]);


# Which types of hazards had the most projects targeting vulnerable populations?

vp_hazards = actions_df.loc[actions_df["Action Type"] == "Projects and policies targeted at those most vulnerable"
              ].groupby(["Hazard Type"])["Action Type"].count().sort_values(ascending=True)

plt.figure(figsize=(10,10))
plt.barh(y = vp_hazards.keys(), width = vp_hazards.values)
plt.title('Number of Actions Targeting Vulnerable Populations by Hazard Type (2020)');


# Which actions have most benefited povery reduction?

pov_red = actions_df.groupby("Action Type")["Poverty Reduction"].sum().sort_values(ascending=True)

plt.figure(figsize=(10,15))
plt.barh(y = pov_red.keys(), width = pov_red.values)
plt.title('Number of Actions Benefitting Poverty Reduction (2020)');


# Which actions most frequently benefit poverty reduction?

freq_pov_red = actions_df.groupby("Action Type")["Poverty Reduction"].mean().sort_values(ascending=True)

plt.figure(figsize=(10,15))
plt.barh(y = freq_pov_red.keys(), width = freq_pov_red.values)
plt.title('Frequency of Benefit to Poverty Reduction (2020)');


# Which actions have benefits for social inclusion?

soc_inc = actions_df.groupby("Action Type")["Social Inclusion"].sum().sort_values(ascending=True)

plt.figure(figsize=(10,15))
plt.barh(y = soc_inc.keys(), width = soc_inc.values)
plt.title('Number of Actions Benefitting Social Inclusion (2020)');


# Which actions most frequently benefit social inclusion?

freq_soc_inc = actions_df.groupby("Action Type")["Social Inclusion"].mean().sort_values(ascending=True)

plt.figure(figsize=(10,15))
plt.barh(y = freq_soc_inc.keys(), width = freq_soc_inc.values)
plt.title('Frequency of Benefit to Poverty Reduction (2020)');


# Create a new dataframe to hold the KPI values for each distinct city (in the 2020 survey)

# Make the new dataframe with just one row per city
equity_df = pd.DataFrame(fc_df.drop_duplicates(subset=["Account Number"], keep='last', ignore_index=True))

# Drop the columns for survey answers
equity_df.drop(equity_df.columns.difference(["Account Number", "Organization", "Country", "CDP Region"]), 
               1, inplace=True)

# Sort the dataframe from account number in ascending order
equity_df.sort_values(by=["Account Number"], ignore_index=True, inplace=True)

# Add a column to indicate whether the city has undertaken *at least one*
# climate change risk assessment that identifies vulnerable populations.

# Extract the data from the survey
data = fc_df.loc[(fc_df["Year Reported to CDP"] == 2020) &
                 (fc_df["Question Number"] == "2.0b") &
                 (fc_df["Column Number"] == 7) &
                 (fc_df["Response Answer"] == "Yes"),
                 ["Account Number", "Response Answer"]
                ].groupby(["Account Number"])["Response Answer"].count()

# Merge the data into the df
equity_df = pd.merge(equity_df, data, how="left", on=["Account Number"])

# Give the column a more descriptive name
equity_df.rename(columns={"Response Answer": "Evaluation of Risk to VPs"}, inplace=True)

# Clean up the column to 0s and 1s as integers
equity_df["Evaluation of Risk to VPs"] = equity_df["Evaluation of Risk to VPs"].fillna(value=0)
equity_df["Evaluation of Risk to VPs"] = equity_df["Evaluation of Risk to VPs"].astype(int)
equity_df.loc[equity_df["Evaluation of Risk to VPs"] >= 1, "Evaluation of Risk to VPs"] = 1

# Add a column for the total number of hazards reported by each city (in 2020)

# Extract the data
data = hazards_df.groupby("Account Number")["Hazard Number"].count()

# Merge the data
equity_df = pd.merge(equity_df, data, how="left", on=["Account Number"])

# Give the column a more descriptive name
equity_df.rename(columns={"Hazard Number":"Total Hazards"}, inplace=True)

# Clean up the missing values and datatype
equity_df["Total Hazards"] = equity_df["Total Hazards"].fillna(value=0)
equity_df["Total Hazards"] = equity_df["Total Hazards"].astype(int)

# Add a column for the number of hazards identified as increasing the risk to vulnerable populations

# Extract the data
data = hazards_df.groupby("Account Number")["Risk to VPs"].sum()

# Merge the data
equity_df = pd.merge(equity_df, data, how="left", on=["Account Number"])

# Give the column a more descriptive name
equity_df.rename(columns={"Risk to VPs":"Hazards Affecting VPs"}, inplace=True)

# Clean up the missing values and datatype
equity_df["Hazards Affecting VPs"] = equity_df["Hazards Affecting VPs"].fillna(value=0)
equity_df["Hazards Affecting VPs"] = equity_df["Hazards Affecting VPs"].astype(int)

# Add a column for the total number of vulnerable populations identified as affected by the hazards

# Extract the data
data = hazards_df.groupby("Account Number")["Total VPs Affected"].sum()

# Merge the data
equity_df = pd.merge(equity_df, data, how="left", on=["Account Number"])

# Clean up the missing values and datatype
equity_df["Total VPs Affected"] = equity_df["Total VPs Affected"].fillna(value=0)
equity_df["Total VPs Affected"] = equity_df["Total VPs Affected"].astype(int)

# Add a column for the average number of vulnerable populations identified per hazard

equity_df["Affected VPs per Hazard"] = round(equity_df["Total VPs Affected"] / equity_df["Total Hazards"], 1)

# Clean up rows where no hazards were reported
equity_df["Affected VPs per Hazard"] = equity_df["Affected VPs per Hazard"].fillna(value=0)

# Add a column for the NORMALIZED average number of vulnerable populations identified per hazard

equity_df["Affected VPs per Hazard (Normalized)"] = (equity_df["Affected VPs per Hazard"] - equity_df["Affected VPs per Hazard"].min()) / (equity_df["Affected VPs per Hazard"].max() - equity_df["Affected VPs per Hazard"].min())

# Add a column for overall awareness of the impact on vulnerable populations
equity_df["Awareness Score"] = (0.5 * equity_df["Evaluation of Risk to VPs"]) + (0.5 * equity_df["Affected VPs per Hazard (Normalized)"])

# Add a column for the total number of actions targeting vulnerable populations

# Extract the data
data = actions_df.loc[actions_df["Action Type"] == "Projects and policies targeted at those most vulnerable",
                      ["Account Number", "Action Type"]
                     ].groupby("Account Number")["Action Type"].count()

# Merge the data
equity_df = pd.merge(equity_df, data, how="left", on=["Account Number"])

# Clean up the missing values and datatype
equity_df["Action Type"] = equity_df["Action Type"].fillna(value=0)
equity_df["Action Type"] = equity_df["Action Type"].astype(int)

# Give the column a more descriptive name
equity_df.rename(columns={"Action Type": "Actions Targeting VPs"}, inplace=True)

# Add a column for the average number of actions targeting vulnerable populations per hazard
equity_df["VP Actions per Hazard"] = round(equity_df["Actions Targeting VPs"] / equity_df["Total Hazards"], 1)

# Clean up rows where no hazards were reported
equity_df["VP Actions per Hazard"] = equity_df["VP Actions per Hazard"].fillna(value=0)

# Identify the rows with "inf" value and drop them from the dataframe
index_to_drop = equity_df.loc[equity_df["VP Actions per Hazard"] == equity_df["VP Actions per Hazard"].max()].index
equity_df.drop(axis=0, index=index_to_drop, inplace=True)

# Add a column for the NORMALIZED average number of actions targeting vulnerable populations per hazard
equity_df["VP Actions per Hazard (Normalized)"] = (equity_df["VP Actions per Hazard"] - equity_df["VP Actions per Hazard"].min()) / (equity_df["VP Actions per Hazard"].max() - equity_df["VP Actions per Hazard"].min())

# Add a column for the total number of actions with benefits for poverty reduction

# Extract the data
data = actions_df.groupby("Account Number")["Poverty Reduction"].sum()

# Merge the data
equity_df = pd.merge(equity_df, data, how="left", on=["Account Number"])

# Clean up the missing values and datatype
equity_df["Poverty Reduction"] = equity_df["Poverty Reduction"].fillna(value=0)
equity_df["Poverty Reduction"] = equity_df["Poverty Reduction"].astype(int)

# Give the column a more descriptive name
equity_df.rename(columns={"Poverty Reduction": "Actions Benefitting Poverty Reduction"}, inplace=True)

# Add a column for the average number of actions benefitting poverty reduction per hazard

equity_df["PR Actions per Hazard"] = round(equity_df["Actions Benefitting Poverty Reduction"] / equity_df["Total Hazards"], 1)

# Clean up rows where no hazards were reported
equity_df["PR Actions per Hazard"] = equity_df["PR Actions per Hazard"].fillna(value=0)

# Add a column for the NORMALIZED average
equity_df["PR Actions per Hazard (Normalized)"] = (equity_df["PR Actions per Hazard"] - equity_df["PR Actions per Hazard"].min()) / (equity_df["PR Actions per Hazard"].max() - equity_df["PR Actions per Hazard"].min())

# Add a column for the total number of actions with benefits for social inclusion

# Extract the data
data = actions_df.groupby("Account Number")["Social Inclusion"].sum()

# Merge the data
equity_df = pd.merge(equity_df, data, how="left", on=["Account Number"])

# Clean up the missing values and datatype
equity_df["Social Inclusion"] = equity_df["Social Inclusion"].fillna(value=0)
equity_df["Social Inclusion"] = equity_df["Social Inclusion"].astype(int)

# Give the column a more descriptive name
equity_df.rename(columns={"Social Inclusion": "Actions Benefitting Social Inclusion"}, inplace=True)

# Add a column for the average number of actions benefitting social inclusion per hazard
equity_df["SI Actions per Hazard"] = round(equity_df["Actions Benefitting Social Inclusion"] / equity_df["Total Hazards"], 1)

# Clean up rows where no hazards were reported
equity_df["SI Actions per Hazard"] = equity_df["SI Actions per Hazard"].fillna(value=0)

# Add a column for the NORMALIZED average
equity_df["SI Actions per Hazard (Normalized)"] = (equity_df["SI Actions per Hazard"] - equity_df["SI Actions per Hazard"].min()) / (equity_df["SI Actions per Hazard"].max() - equity_df["SI Actions per Hazard"].min())

# Add a column for overall action to mitigate of the impact on VPs
equity_df["Action Score"] = (0.5 * equity_df["VP Actions per Hazard (Normalized)"]) + (0.25 * equity_df["PR Actions per Hazard (Normalized)"]) + (0.25 * equity_df["SI Actions per Hazard (Normalized)"])

# Add a column for overall social equity in climate response
equity_df["Overall Climate Equity Score"] = (0.5 * equity_df["Awareness Score"]) + (0.5 * equity_df["Action Score"])

# Add a column for the ranking of the overall score
ranks = equity_df["Overall Climate Equity Score"].rank(ascending=False, method="min").astype(int)
equity_df["Overall Climate Equity Rank"] = ranks

# Create a reduced version of the dataframe focusing on the KPIs and scores
kpi_df = equity_df[["Account Number",
                    "Organization",
                    "Country",
                    "CDP Region",
                    "Total Hazards",
                    "Evaluation of Risk to VPs",
                    "Affected VPs per Hazard (Normalized)",
                    "Awareness Score",
                    "VP Actions per Hazard (Normalized)",
                    "PR Actions per Hazard (Normalized)",
                    "SI Actions per Hazard (Normalized)",
                    "Action Score",
                    "Overall Climate Equity Score",
                    "Overall Climate Equity Rank"
                   ]
                  ]

print(kpi_df.shape)
kpi_df.head()


# Which cities have the highest overall climate equity scores?

highest_overall_scores = kpi_df.sort_values(by="Overall Climate Equity Score", ascending=False).head(20)
highest_overall_scores = highest_overall_scores.sort_values(by="Overall Climate Equity Score", ascending=True)

plt.figure(figsize=(10,8))
plt.barh(y = highest_overall_scores["Organization"], width = highest_overall_scores["Overall Climate Equity Score"])
plt.title('Cities with the Top 20 Overall Climate Equity Scores (2020)');


# Which cities have the highest overall climate equity scores?

kpi_df.sort_values(by="Overall Climate Equity Score", ascending=False).head(20)


# Which cities have the lowest overall climate equity scores?

kpi_df.sort_values(by="Overall Climate Equity Score", ascending=True).head()


# Which cities have the highest awareness scores?

highest_awareness_scores = kpi_df.sort_values(by="Awareness Score", ascending=False).head(20)
highest_awareness_scores = highest_awareness_scores.sort_values(by="Awareness Score", ascending=True)

plt.figure(figsize=(10,8))
plt.barh(y = highest_awareness_scores["Organization"], width = highest_awareness_scores["Awareness Score"])
plt.title('Cities with the Top 20 Awareness Scores (2020)');


# Which cities have the highest awareness scores?

equity_df.iloc[:, 0:11].sort_values(by="Awareness Score", ascending=False).head(10)


# What percentage of cities have climate impact assessments that identify vulnerable populations?

vp_assessments = equity_df.loc[equity_df["Evaluation of Risk to VPs"] == 1, "Account Number"].count()
print(f"{round((vp_assessments/565)*100,1)} percent of cities have climate impact assessments that identify vulnerable populations.")

plt.pie(x = [vp_assessments, 565-vp_assessments], labels=["VP Assessment", "No VP Assessment"]);


# Which cities have the highest action scores?

highest_action_scores = kpi_df.sort_values(by="Action Score", ascending=False).head(20)
highest_action_scores = highest_action_scores.sort_values(by="Action Score", ascending=True)

plt.figure(figsize=(10,8))
plt.barh(y = highest_action_scores["Organization"], width = highest_action_scores["Action Score"])
plt.title('Cities with the Top 20 Action Scores (2020)');


# Which cities have the highest action scores?

equity_df.iloc[:,np.r_[0:4,5,11:21]].sort_values(by="Action Score", ascending=False).head(10)


# Visualize the geographic distribution of climate equity scores

# Merge the geospatial coordinates into the cities dataframe
kpi_df = pd.merge(kpi_df, city_coords[["Account Number","lat", "long"]])

# Convert to a GeoDataFrame
kpi_gdf = gpd.GeoDataFrame(kpi_df, geometry=
                             gpd.points_from_xy(kpi_df['long'], kpi_df['lat']))

# Set the Coordinate Reference System
kpi_gdf.crs = "epsg:4326"

# Import a world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Remove Antarctica from the map
world = world[(world.pop_est>0) & (world.name!="Antarctica")]

# Plot the city climate equity score on the world map
fig, ax = plt.subplots(figsize=(20,30))
ax.set_aspect('equal')
world.plot(ax=ax, color='lightgrey', edgecolor='white')
kpi_gdf.plot(ax=ax,
             column='Overall Climate Equity Score',
             cmap='RdYlGn',
             scheme='user_defined',
             classification_kwds={'bins':[.2, .4, .6, .8]},
             marker='o', 
             markersize=25,
             legend=True,
             legend_kwds={"title":"Climate Equity Score"}
            )

plt.title(label="Geographic Distribution of Climate Equity Scores",
          fontdict={"fontsize":20}
         );


# Create a new dataframe for the CDP's 2020 A-List.

cities = ['Ajuntament de Barcelona', 'Auckland Council',
       'Ayuntamiento de Hermosillo', 'Ayuntamiento de Murcia',
       'Ayuntamiento de Vitoria-Gasteiz', 'BCP Council',
       'Bristol City Council', 'BÃ¦rum Kommune', 'Canberra',
       'City of Adelaide', 'City of Athens', 'City of Baltimore',
       'City of Berkeley', 'City of Berlin', 'City of Boston',
       'City of Boulder', 'City of Buenos Aires', 'City of Calgary',
       'City of Cape Town', 'City of Cleveland', 'City of Columbus',
       'City of Copenhagen', 'City of Denver', 'City of Espoo',
       'City of Eugene', 'City of Flagstaff', 'City of Hayward',
       'City of Helsinki', 'City of Lahti', 'City of Los Angeles',
       'City of Louisville, KY', 'City of Lund', 'City of Melbourne',
       'City of Miami', 'City of Paris', 'City of Park City, UT',
       'City of Philadelphia', 'City of Porto', 'City of San Antonio',
       'City of San Francisco', 'City of San JosÃ©', 'City of Stockholm',
       'City of Sydney', 'City of Toronto', 'City of Turku',
       'City of Vancouver', 'City of West Palm Beach', 'City of Windsor',
       'City Ã–rebro', 'Comune di Firenze', 'Comune di Torino',
       'Cuyahoga County', 'District of Columbia',
       'District of Saanich, BC', 'Egedal Municipality',
       'Gladsaxe Kommune', 'Gobierno Municipal de LeÃ³n de los Aldamas',
       'Government of Hong Kong Special Administrative Region',
       'Greater London Authority', 'Halifax Regional Municipality',
       'HelsingÃ¸r Kommune / Elsinore Municipality',
       'Hoeje-Taastrup Kommune', 'HÃ¸rsholm Kommune',
       'Iskandar Regional Development Authority', 'MalmÃ¶ Stad',
       'Mexico City', 'Moscow Government', 'Municipalidad de PeÃ±alolÃ©n',
       'Municipalidad de San JosÃ©', 'Municipality of Recife',
       'MunicÃ­pio de Braga', 'MunicÃ­pio de Ã�gueda',
       'New Taipei City Government', 'Newcastle City Council',
       'Pingtung County Government', 'Prefeitura do Rio de Janeiro',
       'San Luis Obispo', 'Seoul Metropolitan Government',
       'Stadt Heidelberg', 'Stadt ZÃ¼rich', 'Taichung City Government',
       'Tainan City Government', 'Taoyuan City Government',
       'The Local Government of Quezon City', 'Town of Breckenridge, CO',
       'Town of Vail, CO', 'Village of Park Forest, IL', 'VÃ¤stervik']

regions = ['Europe', 'Southeast Asia and Oceania', 'Latin America', 'Europe',
       'Europe', 'Europe', 'Europe', 'Europe',
       'Southeast Asia and Oceania', 'Southeast Asia and Oceania',
       'Europe', 'North America', 'North America', 'Europe',
       'North America', 'North America', 'Latin America', 'North America',
       'Africa', 'North America', 'North America', 'Europe',
       'North America', 'Europe', 'North America', 'North America',
       'North America', 'Europe', 'Europe', 'North America',
       'North America', 'Europe', 'Southeast Asia and Oceania',
       'North America', 'Europe', 'North America', 'North America',
       'Europe', 'North America', 'North America', 'North America',
       'Europe', 'Southeast Asia and Oceania', 'North America', 'Europe',
       'North America', 'North America', 'North America', 'Europe',
       'Europe', 'Europe', 'North America', 'North America',
       'North America', 'Europe', 'Europe', 'Latin America', 'East Asia',
       'Europe', 'North America', 'Europe', 'Europe', 'Europe',
       'Southeast Asia and Oceania', 'Europe', 'Latin America', 'Europe',
       'Latin America', 'Latin America', 'Latin America', 'Europe',
       'Europe', 'East Asia', 'Europe', 'East Asia', 'Latin America',
       'North America', 'East Asia', 'Europe', 'Europe', 'East Asia',
       'East Asia', 'East Asia', 'Southeast Asia and Oceania',
       'North America', 'North America', 'North America', 'Europe']

a_list = {'Organization': cities, 'CDP Region': regions}
a_list = pd.DataFrame(a_list)

# Add a column to indicate A-List status
a_list["CDP A-List"] = 1

# Merge the a-list data into the kpi dataframe
kpi_df = pd.merge(kpi_df, a_list, how="left", on=["Organization", "CDP Region"])

# Clean up the missing values and datatype
kpi_df["CDP A-List"] = kpi_df["CDP A-List"].fillna(value=0)
kpi_df["CDP A-List"] = kpi_df["CDP A-List"].astype(int)

# Which cities on the A-List are also ranked in the top 88?

overlap = kpi_df.loc[(kpi_df["CDP A-List"] == 1) &
                     (kpi_df["Overall Climate Equity Rank"] <= 88)
                    ].sort_values("Overall Climate Equity Rank", ascending=True)

print("How many cities?", overlap.shape[0])
print("Which cities?",  overlap["Organization"].values.tolist())
overlap.head(10)


# Which cities ranked in the top 88 are NOT on the A-List?

ranked_not_alist = kpi_df.loc[(kpi_df["Overall Climate Equity Rank"] <= 88) &
                              (kpi_df["CDP A-List"] == 0)
                             ].sort_values("Overall Climate Equity Rank", ascending=True)

print("How many cities?", ranked_not_alist.shape[0])
print("Which cities?",  ranked_not_alist["Organization"].values.tolist())
ranked_not_alist.head(10)


# Which cities on the A-List were not ranked in the top 88?

alist_not_ranked = kpi_df.loc[(kpi_df["Overall Climate Equity Rank"] > 88) &
                              (kpi_df["CDP A-List"] == 1)
                             ].sort_values("Overall Climate Equity Rank", ascending=False)

print("How many cities?", alist_not_ranked.shape[0])
print("Which cities?",  alist_not_ranked["Organization"].values.tolist())
alist_not_ranked.head(10)

