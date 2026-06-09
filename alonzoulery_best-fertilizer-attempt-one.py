# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3
conn = sqlite3.connect(":memory:")
from collections import Counter
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#IMPORTS
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import label_ranking_average_precision_score
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
train.to_sql('train', conn, if_exists = 'replace', index = False)
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
data_groups = ['train', 'test']

for name, df in zip(data_groups, [train, test]):
    print(f"{name.upper()} - Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    display(df.head(3))


mini = train['Temparature'].min()
maxi = train['Temparature'].max()
mini_i = train['Humidity'].min()
maxi_i = train['Humidity'].max()
mini_ii = train['Moisture'].min()
maxi_ii = train['Moisture'].max()
print(mini, maxi)
print(mini_i, maxi_i)
print(mini_ii, maxi_ii)


query = """
SELECT *,
CASE 
    WHEN Temparature >= 35 AND Humidity <= 59 AND Moisture <= 34 THEN 'Hot & Dry'
    WHEN Temparature BETWEEN 30 AND 34 AND Humidity BETWEEN 60 AND 66 AND Moisture BETWEEN 35 AND 49 THEN 'Warm & Moist'
    WHEN Temparature BETWEEN 25 AND 29 AND Humidity BETWEEN 67 AND 72 AND Moisture BETWEEN 50 AND 65 THEN 'Cool & Wet'
    WHEN Humidity >= 70 AND Moisture >= 60 THEN 'Rainy-like'
    WHEN Moisture < 30 THEN 'Soil Drought'
    ELSE 'Uncertain'
    END AS Weather
    FROM train"""

train_I = pd.read_sql_query(query, conn)
train_I['Soil Type'] = train_I['Soil Type'].replace('Clayey', 'Clay')
train_I.rename(columns = {'Soil Type': 'Soil_Type'}, inplace = True)
train_I.rename(columns = {'Crop Type': 'Crop_Type'}, inplace = True)
train_I.to_sql('train_I', conn, if_exists = 'replace', index = False)




test.to_sql('test', conn, if_exists = 'replace', index = False)

query = """
SELECT *,
CASE
    WHEN Temparature >= 35 AND Humidity <= 59 AND Moisture <= 34 THEN 'Hot & Dry'
    WHEN Temparature BETWEEN 30 AND 34 AND Humidity BETWEEN 60 AND 66 AND Moisture BETWEEN 35 AND 49 THEN 'Warm & Moist'
    WHEN Temparature BETWEEN 25 AND 29 AND Humidity BETWEEN 67 AND 72 AND Moisture BETWEEN 50 AND 65 THEN 'Cool & Wet'
    WHEN Humidity >= 70 AND Moisture >= 60 THEN 'Rainy-like'
    WHEN Moisture < 30 THEN 'Soil Drought'
    ELSE 'Uncertain'
    END AS Weather
    FROM test"""
test_i = pd.read_sql_query(query, conn)
test_i.rename(columns = {'Soil Type': 'Soil_Type'}, inplace = True)
test_i.rename(columns = {'Crop Type': 'Crop_Type'}, inplace = True)
test_i['Soil_Type'] = test_i['Soil_Type'].replace('Clayey', 'Clay')
test_i['Fertilizer Name'] = ''
test_i.to_sql('test_i', conn, if_exists = 'replace', index = False)
test_i.drop(columns = ['Temparature', 'Humidity', 'Moisture'], inplace = True)
test_i


fig, axes = plt.subplots(1, 3, figsize = (18, 6))
sns.boxplot(x = 'Soil_Type', y = 'Nitrogen', data = train_I, ax = axes[0])
axes[0].set_title('Nitrogen Levels by Soil_Type')
sns.boxplot(x = 'Soil_Type', y = 'Phosphorous', data = train_I, ax = axes[1])
axes[1].set_title('Phosphorous Levels by Soil_Type')
sns.boxplot(x = 'Soil_Type', y = 'Potassium', data = train_I, ax = axes[2])
axes[2].set_title('Potassium Levels by Soil_Type')
plt.tight_layout()
plt.show()


for nutrient in ['Nitrogen', 'Phosphorous', 'Potassium']:
    groups = [group[nutrient].values for name, group in train.groupby('Soil Type')]
    stat, p = kruskal(*groups)
    print(f"{nutrient} - Kruskal-Wallis H-Statistic: {stat:.3f}, p-value: {p:.4f}")


train_I = train_I.copy()
train_I = train_I.rename(columns = {'Soil Type': 'Soil_Type'})


query = """
SELECT Soil_Type, AVG(Nitrogen + Phosphorous + Potassium) AS avg_parts_total,
    AVG(Nitrogen * 1.0 / (Nitrogen + Phosphorous + Potassium)) AS avg_n_pct,
    AVG(Phosphorous * 1.0 / (Nitrogen + Phosphorous + Potassium)) AS avg_p_pct,
    AVG(Potassium * 1.0 / (Nitrogen + Phosphorous + Potassium)) AS avg_k_pct,
    AVG(Nitrogen) AS avg_nitrogen,
    AVG(Phosphorous) AS avg_phosphorous,
    AVG(Potassium) AS avg_potassium
FROM train_I
GROUP BY Soil_Type
"""

ST_PCT_AVG = pd.read_sql_query(query, conn)
ST_PCT_AVG.to_sql('Soil_Type', conn, if_exists = 'replace', index = True)
ST_PCT_AVG

query = """
SELECT id, Soil_Type, Nitrogen, Phosphorous, Potassium
FROM train_I
"""
Chem_Struct = pd.read_sql_query(query, conn)
Chem_Struct.to_sql('Chem_Struct', conn, if_exists = 'replace', index = True)
Chem_Struct


Nitro_counts = 0
Phospho_counts = 0
Potas_counts = 0
for i, j, k in zip(train_I['Nitrogen'], train_I['Phosphorous'], train_I['Potassium']):
    if i > j and i > k:
        Nitro_counts += 1
    elif j > i and j > k:
        Phospho_counts += 1
    elif k > i and k > j:
        Potas_counts += 1

print(Nitro_counts, Phospho_counts, Potas_counts) #Gives totals for each compound.
Chem_Count = [Nitro_counts, Phospho_counts, Potas_counts]

compounds = ['Nitrogen', 'Phosphorous', 'Potassium']
counts = Chem_Count
colors = ['#8B4513', '#556B2F', '#DEB887']

plt.figure(figsize = (8, 5))
plt.bar(compounds, counts, color = colors, edgecolor = 'black')
plt.title('Compound Counts in Sample', fontsize = 14, weight = 'bold')
plt.ylabel('Count')
plt.grid(axis = 'y', linestyle = '--', alpha = 0.6)

for i, val in enumerate(counts):
    plt.text(i, val + max(counts)*0.02, str(val), ha = 'center', va = 'bottom')

plt.tight_layout()
plt.show()


F_names = train['Fertilizer Name']
#F_names.unique()
fertilizer_array = np.array(['28-28', '17-17-17', '10-26-26', 'DAP', '20-20', '14-35-14', 'Urea'])

counts_series = F_names.value_counts()
labels = counts_series.index.tolist()
counts = counts_series.values.tolist()

colors = ['#8B4513', '#556B2F', '#DEB887', '#A0522D', '#6B8E23', '#CD853F', '#D2B48C']
while len(colors) < len(labels):
    colors += colors

plt.figure(figsize = (10, 6))
plt.bar(labels, counts, color = colors[:len(labels)], edgecolor = 'black')
plt.title('Record Count per Fertilizer', fontsize = 16, weight = 'bold')
plt.xlabel('Fertilizer Name')
plt.ylabel('Number of Records')
plt.grid(axis = 'y', linestyle = '--', alpha = 0.5)
for i, val in enumerate(counts):
    plt.text(i, val + max(counts) * 0.02, str(val), ha = 'center', va = 'bottom')

plt.tight_layout()
plt.show()



check = Chem_Struct['Nitrogen'].unique()
check_i = Chem_Struct['Phosphorous'].unique()
check_ii = Chem_Struct['Potassium'].unique()

print(f"Nitrogen(Min): {check.min()}, Nitrogen(Max): {check.max()} \n, Phosphorous(Min): {check_i.min()}, Phosphorous(Max): {check_i.max()} \n, Potassium(Min): {check_ii.min()}, Potassium(Max): {check_ii.max()}")


#Fertilizer Name counts and ranking
i = train.columns
j = test.columns

for col in i:
    if col not in j:
        print(col)

holder_i = train['Fertilizer Name'].unique()
print(holder_i)

plt.figure(figsize = (12, 6))
sns.countplot(data = train, y = 'Fertilizer Name', order = train['Fertilizer Name'].value_counts().index, palette = 'viridis')
plt.title('Distribution of Fertilizer Names')
plt.xlabel('Count')
plt.ylabel('Fertilizer Name')
plt.tight_layout()
plt.show()


nutrient_summary = train.groupby('Fertilizer Name').agg(
    record_count = ('Fertilizer Name', 'count'),
    avg_n = ('Nitrogen', 'mean'),
    avg_pho = ('Phosphorous', 'mean'),
    avg_pot = ('Potassium', 'mean')
).reset_index()
print(nutrient_summary.sort_values(by = 'record_count', ascending = False))


nutrient_summary.set_index('Fertilizer Name')[['avg_n', 'avg_pho', 'avg_pot']].plot(
    kind = 'bar', stacked = True, figsize = (12,6), colormap = 'YlGn'
)
plt.title('Average Nutrient Profile by Fertilizer Name')
plt.ylabel('Average Nutrient Amount')
plt.xlabel('Fertilizer Name')
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()


train_II = train_I.copy()
train_II['Fertilizer Name'] = train.loc[train_II.index, 'Fertilizer Name']
train_II.drop(columns = ['Temparature', 'Humidity', 'Moisture'], inplace = True)
train_II.to_sql('train_II', conn, if_exists = 'replace', index = False)
train_II


fert_by_soil = train_II.groupby(['Soil_Type', 'Fertilizer Name']).size().unstack().fillna(0)
fert_by_soil_pct = fert_by_soil.div(fert_by_soil.sum(axis = 1), axis = 0)
fert_by_soil_pct.plot(kind = 'bar', stacked = True, figsize = (12, 6), colormap = 'terrain')
plt.title('Fertilizer Preference by Soil Type')
plt.ylabel('Proportion of Fertilizer Use')
plt.xlabel('Soil Type')
plt.legend(title = 'Fertilizer Name', bbox_to_anchor = (1.05, 1), loc = 'upper left')
plt.tight_layout()
plt.show()


#train_II['Crop_Type'].value_counts()
le_crop = LabelEncoder()
train_II['Crop_Type_Label'] = le_crop.fit_transform(train_II['Crop_Type'])
test_i['Crop_Type_Label'] = le_crop.transform(test_i['Crop_Type'])

crop_soil_ct = pd.crosstab(train_II['Crop_Type'], train_II['Soil_Type'])
crop_soil_rank = crop_soil_ct.rank(axis = 1, method = 'min', ascending = False)

def get_crop_soil_rank(row):
    try:
        return crop_soil_rank.loc[row['Crop_Type'], row['Soil_Type']]
    except KeyError:
        return 0

train_II['Crop_Soil_Rank'] = train_II.apply(get_crop_soil_rank, axis = 1)
test_i['Crop_Soil_Rank'] = test_i.apply(get_crop_soil_rank, axis = 1)


pd.crosstab(train_II['Crop_Type'], train_II['Soil_Type'])


train_II['Total_NPK'] = train_II[['Nitrogen', 'Phosphorous', 'Potassium']].sum(axis = 1)
train_II['N_ratio'] = train_II['Nitrogen'] / train_II['Total_NPK']
train_II['P_ratio'] = train_II['Phosphorous'] / train_II['Total_NPK']
train_II['K_ratio'] = train_II['Potassium'] / train_II['Total_NPK']

test_i['Total_NPK'] = test_i[['Nitrogen', 'Phosphorous', 'Potassium']].sum(axis = 1)
test_i['N_ratio'] = test_i['Nitrogen'] / test_i['Total_NPK']
test_i['P_ratio'] = test_i['Phosphorous'] / test_i['Total_NPK']
test_i['K_ratio'] = test_i['Potassium'] / test_i['Total_NPK']


features = ['Soil_Type', 'Weather', 'Crop_Type', 'Crop_Soil_Rank', 'Total_NPK']
target = 'Fertilizer Name'

df = train_II.copy()
df = df[features + [target]].dropna()

le = LabelEncoder()
df['target'] = le.fit_transform(df[target])

X = df[features]
y = df['target']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)

cat_cols = ['Soil_Type', 'Weather', 'Crop_Type']
for col in cat_cols:
    X_train[col] = X_train[col].astype('category')
    X_val[col] = X_val[col].astype('category')

model = LGBMClassifier(
    objective = 'multiclass',
    num_class = len(le.classes_),
    n_estimators = 200,
    random_state = 42
)

model.fit(
    X_train,
    y_train,
    eval_set = [(X_val, y_val)],
    categorical_feature = cat_cols,
    
)
#
test_data = test_i.copy()
test_data = test_data[features]
for col in cat_cols:
    test_data[col] = test_data[col].astype('category')

probs = model.predict_proba(test_data)

for idx in range(5):
    top_classes = np.argsort(probs[idx])[::-1][:3]
    top_labels = le.inverse_transform(top_classes)
    print(f"Test Row{idx}:")
    for i, label in enumerate(top_labels):
        print(f"Rank {i + 1}: {label} (prob: {probs[idx][top_classes[i]]:.4f})")
    print("-" * 10)

top_3 = np.argsort(probs, axis = 1)[:, -3:][:, ::-1]
top_3_labels = le.inverse_transform(top_3.flatten()).reshape(top_3.shape)
predictions = [' '.join(row) for row in top_3_labels]
submission = pd.DataFrame({
    'id': test_i['id'],
    'Fertilizer Name': predictions
})

submission.to_csv('submission.csv', index = False)
print("Submission file created as 'submission.csv'")




