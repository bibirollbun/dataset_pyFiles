!pip install pytrends


import os
import re
import json
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pytrends.request import TrendReq

MK_PATH = kagglehub.dataset_download("kaggle/meta-kaggle")
MKC_PATH = kagglehub.dataset_download("kaggle/meta-kaggle-code")

print("Path to Meta-Kaggle dataset files:", MK_PATH)  
print("Path to Meta-Kaggle-Code dataset files:", MKC_PATH)


# Constants:
side_lines_c = '#F39237'
ga_area_c = '#F1E100' #Theory’s Golden Age area color
custom_colors = [
    '#1297a5',
    '#872657']
filename_N_T = "/kaggle/input/start/NLP_vs_TFM.csv"

#==================================================
# Downloading Stage:
pytrends = TrendReq()

pytrends.build_payload(['Natural language processing'], timeframe='2014-01-01 2025-01-01')
df_nlp = pytrends.interest_over_time()

pytrends.build_payload(['Transformer AI'], timeframe='2014-01-01 2025-01-01')
df_transformer = pytrends.interest_over_time()

#==================================================
# Merging & filtering stage
df_N_T = pd.concat([df_nlp['Natural language processing'], df_transformer['Transformer AI']], axis=1)

df_N_T.dropna(inplace=True)

df_N_T.index = pd.to_datetime(df_N_T.index)
df_N_T['year'] = df_N_T.index.year

last_year = df_N_T['year'].max()
df_N_T = df_N_T[df_N_T['year'] != last_year]

if os.path.exists(filename_N_T):
    df_N_T = pd.read_csv(filename_N_T, index_col=0, parse_dates=True)
    df_N_T.to_csv('NLP_vs_TFM.csv')

else:
    df_N_T.to_csv('NLP_vs_TFM.csv')

#==================================================
# Visualization Stage: Highlighting the Theory’s Golden Age:

## 1. Figure Setup

plt.figure(figsize=(10, 4))

years = df_N_T['year'].unique()

## 2. Plotting Data Lines
for i, col in enumerate(['Natural language processing', 'Transformer AI']):
    label = 'NLP' if i == 0 else 'TFM'
    df_gr = df_N_T.groupby('year')[col].sum()
    plt.plot(df_gr, label=label, marker='o', color=custom_colors[i])

## 3. Highlighting Key Periods (Golden Age + Milestones)
plt.title('Google Trends: NLP vs Transformer')
plt.xlabel('Years')
plt.ylabel('Search Interest (0–100)')

plt.gca().set_facecolor('#fafab9')

plt.axvline(x=2019, color=side_lines_c, linestyle='--', linewidth=2, label='2019')
plt.axvline(x=2021, color=side_lines_c, linestyle='--', linewidth=2, label='2021')
plt.axvspan(2019, 2021, color=ga_area_c, alpha=0.9, label='Golden Age')

## 4. Final Styling and Saving Figure
plt.xticks(years)
plt.grid(True, linewidth=1.5, color='black', axis='x', alpha=0.6)
plt.legend()


plt.tight_layout()
plt.savefig('Figure 1. Google Trends: NLP vs Transformer.png')
plt.show()


# Constants:
side_lines_c = '#F39237'
ga_area_c = '#F1E100' #Theory’s Golden Age area color
custom_colors_N_TL = [
    '#1297a5',
    '#872657']

kewords_N_TL = ['NLP','Transfer Learning']

filename_N_TL = "/kaggle/input/start/NLP_vs_TL.csv"

#==================================================
# Downloading Stage:
pytrends = TrendReq()

pytrends.build_payload([kewords_N_TL[0]], timeframe='2014-01-01 2025-01-01')
df_tf = pytrends.interest_over_time()

pytrends.build_payload([kewords_N_TL[1]], timeframe='2014-01-01 2025-01-01')
df_pyt = pytrends.interest_over_time()

#==================================================
# Merging & filtering stage
df_N_TL = pd.concat([df_tf[kewords_N_TL[0]], df_pyt[kewords_N_TL[1]]], axis=1)

df_N_TL.dropna(inplace=True)

df_N_TL.index = pd.to_datetime(df_N_TL.index)
df_N_TL['year'] = df_N_TL.index.year

last_year = df_N_TL['year'].max()
df_N_TL = df_N_TL[df_N_TL['year'] != last_year]


if os.path.exists(filename_N_TL):
    df_N_TL = pd.read_csv(filename_N_TL, index_col=0, parse_dates=True)
    df_N_TL.to_csv('NLP_vs_TL.csv')
else:
    df_N_TL.to_csv('NLP_vs_TL.csv')

#==================================================
# Visualization Stage: Highlighting the Theory’s Golden Age:

## 1. Figure Setup

plt.figure(figsize=(12, 4.3))

years = df_N_TL['year'].unique()

## 2. Plotting Data Lines
for i , col in enumerate(kewords_N_TL):
    df_gr = df_N_TL.groupby('year')[col].sum()
    plt.plot(df_gr , label=col , marker='o', color=custom_colors_N_TL[i])
    plt.legend()

## 3. Highlighting Key Periods (Golden Age + Milestones)
plt.title('Google Trends: NLP vs Transfer Learning')
plt.xlabel('Years')
plt.ylabel('Search Interest (0–100)')

plt.gca().set_facecolor('#fafab9')

plt.axvline(x=2019, color=side_lines_c, linestyle='--', linewidth=2 , label='2019')
plt.axvline(x=2021, color=side_lines_c, linestyle='--', linewidth=2 , label='2021')
plt.axvspan(2019, 2021, color=ga_area_c, alpha=0.9, label='Golden Age')

## 4. Final Styling and Saving Figure
plt.xticks(years)
plt.grid(True ,linewidth=1.5,color='black', axis='x' , alpha=0.6)
plt.legend() 

plt.tight_layout()
plt.savefig('Figure 2. Google Trends: NLP vs Transfer Learning.png')
plt.show()



# Constants:
cscl_papers = []
side_lines_c = '#F39237'
ga_area_c = '#F1E100' #The Golden Age of Theory area color

#==================================================
# Downloading Stage:
with open("/kaggle/input/arxiv/arxiv-metadata-oai-snapshot.json", "r", encoding="utf-8") as f:
    for line in f:
        try:
            paper = json.loads(line)
            if "cs.CL" in paper.get("categories", ""):
                cscl_papers.append(paper)
        except json.JSONDecodeError:
            continue  

df_cl = pd.DataFrame(cscl_papers)

#==================================================
# Merging & filtering stage
def apply_Date(versions):
    date = versions[0]['created']
    return pd.to_datetime(date).year

df_cl['year'] = df_cl['versions'].apply(apply_Date)
df_cl = df_cl.sort_values(by='year')
df_cl = df_cl[df_cl['year']>= 2014]

#==================================================
# Visualization Stage: Highlighting the Theory’s Golden Age:

## 1. Figure Setup
plt.figure(figsize=(12, 4.3))

years = df_cl['year'].unique()

## 2. Plotting Data Lines
df_gr = df_cl.groupby('year').size()
plt.plot(df_gr , label='NLP-Papers' , marker='o',linewidth=2.2)

## 3. Highlighting Key Periods (Golden Age + Milestones)
plt.title('ArXiv: The Flourishing of NLP Research')
plt.xlabel('Years')
plt.ylabel('PapersCount')

plt.gca().set_facecolor('#fafab9')

plt.axvline(x=2019, color=side_lines_c, linestyle='--', linewidth=2 , label='2019')
plt.axvline(x=2021, color=side_lines_c, linestyle='--', linewidth=2 , label='2021')
plt.axhline(y=df_gr[2022], color='darkred', linestyle='-.', linewidth=1.2, label=f'{df_gr[2022]} |22')
plt.axvspan(2019, 2021, color=ga_area_c, alpha=0.9, label='Golden Age')

## 4. Final Styling and Saving Figure
plt.xticks(years)
plt.grid(True ,linewidth=1.5, color='black', axis='x' , alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig('Figure 3. ArXiv: The Flourishing of NLP Research.png')
plt.show()




# Constants
keywords = ['nlp','transformers','lstm','computer vision']

custom_colors = {
    'transformers': '#D32F2F',  
    'nlp': 'indigo',       
    'lstm': '#1976D2',          
    'computer vision':'#E87119'}

growth_summaries = []

#==================================================
# Reading Stage
kernels = pd.read_csv(f'{MK_PATH}/Kernels.csv')
kernel_tags = pd.read_csv(f'{MK_PATH}/KernelTags.csv')
tags = pd.read_csv(f'{MK_PATH}/Tags.csv')

#==================================================
# Merging Stage
kt_temp = kernel_tags.merge(
    kernels[['Id', 'CreationDate', 'TotalViews']],
    left_on='KernelId',
    right_on='Id')

kt = kt_temp.merge(
    tags[['Id', 'Name']],
    left_on='TagId',
    right_on='Id',
    suffixes=('', '_Tag'))

#==================================================
# filtering stage
kt['year'] = pd.to_datetime(kt['CreationDate']).dt.year
last_year = kt['year'].max() 
kt = kt[kt['year'] != last_year] # delete 2025

tag_year_counts = (
    kt.groupby(['Name', 'year',])
    .size()
    .reset_index(name='count')
    .sort_values(['Name', 'year']))

#==================================================
# Visualization Stage: Highlighting the Golden Age

## 1. Figure Setup
plt.figure(figsize=(11, 5.5))


years = sorted(tag_year_counts['year'].dropna().unique())
bar_width = 0.15
positions = np.arange(len(years))

## 2. Plotting Data Lines
for i, tag in enumerate(keywords):
    mask = tag_year_counts['Name'].str.contains(tag, case=False, na=False)
    
    df_sub = tag_year_counts[mask].copy()
    df_sub = df_sub.set_index('year').reindex(years).fillna(0)

    counts = df_sub['count'].values
    x_pos = positions + (i - len(keywords)/2) * bar_width + bar_width/2

    plt.bar(x_pos, counts, width=bar_width, label=tag.upper(), color=custom_colors[tag])

    idx_2019 = years.index(2019)
    idx_2021 = years.index(2021)
    count_2019 = counts[idx_2019]
    count_2021 = counts[idx_2021]
    if count_2019 > 0:
        growth_rate = ((count_2021 - count_2019) / count_2019) * 100
        growth_summaries.append((tag, growth_rate))

## 3. Highlighting Key Periods (Golden Age + Milestones)
plt.title('The Flourishing of notebooks')
plt.ylabel("Count")
plt.xlabel("Year")

plt.gca().set_facecolor('#fafab9')

plt.axvline(x=years.index(2019)-0.32, color='lime', linestyle='--', linewidth=2, label='2019')
plt.axvline(x=years.index(2021)+0.32, color='lime', linestyle='--', linewidth=2, label='2022')


## 4. Final Styling and Saving Figure   
plt.xticks(positions, years)
plt.xlim(left=years.index(2015))
plt.grid(True, axis='y', linestyle='--', alpha=0.55)
plt.legend(title='Tags', prop={'size':10}, loc='upper left') 

plt.tight_layout()
plt.savefig('Figure 4. The Flourishing of notebooks.png')
plt.show()




# Constants
keywords2 = ['nlp', 'computer vision']

custom_colors = {
    'nlp': '#1297a5',
    'computer vision': '#872657'
}

side_lines_c = '#F39237'
ga_area_c = '#F1E100' #The Golden Age of Theory area color

#==================================================
# filtering stage
total_views_supdata = kt[['Name', 'TotalViews', 'year']]

#==================================================
# Visualization Stage: Highlighting the Golden Age

## 1. Figure Setup
plt.figure(figsize=(10, 5))

## 2. Plotting Data Lines
for keyword in keywords2:
    mask = total_views_supdata['Name'].str.contains(keyword, case=False, na=False)
    
    subset = total_views_supdata[mask].copy()
    grouped = subset.groupby('year')['TotalViews'].sum()

    color = custom_colors[keyword]
    plt.plot(grouped.index, grouped.values, label=keyword.upper(), color=color , marker='o')

## 3. Highlighting Key Periods (Golden Age + Milestones)
plt.title("Notebook Viewership Flourishing in The Golden Age of Theory")
plt.xlabel("Year")
plt.ylabel("TotalViews")

plt.gca().set_facecolor('#fafab9')

plt.axvline(x=2019, color=side_lines_c, linestyle='--', linewidth=2 , label='2019')
plt.axvline(x=2021, color=side_lines_c, linestyle='--', linewidth=2 , label='2021')
plt.axvspan(2019, 2021, color=ga_area_c, alpha=0.9, label='Golden Age')

## 4. Final Styling and Saving Figure
plt.grid(True)
plt.legend() 

plt.tight_layout()
plt.savefig('Figure 5. Notebook Viewership Flourishing in The Golden Age of Theory.png')
plt.show()





# Constants
figsize = (6.3,2.7) # Suggested figure size: (9, 4)
side_lines_c = '#F39237'
ga_area_c = '#F1E100' #The Golden Age of Theory area color

keywords_dataset2 = ['tensorflow','pytorch']

custom_colors = {
    'tensorflow': '#1297a5',
    'pytorch': '#872657',
}

#==================================================
# Visualization Stage: Highlighting the Golden Age

## 1. Figure Setup
plt.figure(figsize=(9, 4))


## 2. Plotting Data Lines
for keyword in keywords_dataset2:
    
    mask = kt['Name'].str.contains(keyword, case=False, na=False)

    filtred_data = kt[mask].copy()
    grouped_to = filtred_data.groupby('year').size()

    plt.plot(grouped_to ,color=custom_colors[keyword] , label=keyword.upper() , marker='o')

## 3. Highlighting Key Periods (Golden Age + Milestones)
plt.title("TensorFlow vs. PyTorch Popularity During The Golden Age of Theory")
plt.xlabel('Years')
plt.ylabel('Framework Frequency')

plt.gca().set_facecolor('#fafab9')

plt.axvline(x=2019, color=side_lines_c, linestyle='--', linewidth=2 , label='2019')
plt.axvline(x=2021, color=side_lines_c, linestyle='--', linewidth=2 , label='2021')
plt.axvspan(2019, 2021, color=ga_area_c, alpha=0.9, label='Golden Age')

## 4. Final Styling and Saving Figure
plt.grid(True)
plt.legend() 

plt.tight_layout()
plt.savefig('Figure 6. TensorFlow vs. PyTorch Popularity During The Golden Age of Theory.png')
plt.show()




# Constants
nlp_subfields = {
    'text generation': [
        'text generation', 'language generation', 'sequence generation', 'nlg',
        'text-to-text generation', 'story generation', 'sentence generation',
        'paraphrase generation', 'generative language model'
    ],
    'text classification': [
        'text classification', 'document classification', 'sentiment analysis',
        'opinion mining', 'emotion classification', 'topic classification',
        'news categorization', 'stance detection'
    ],
    'question answering': [
        'question answering', 'qa system', 'reading comprehension', 'qa',
        'open-domain qa', 'factoid questions', 'extractive qa', 'retrieval-based qa',
        'multi-hop qa'
    ]
}

custom_colors = {
    'text generation': '#D32F2F',
    'text classification': '#1976D2',
    'question answering': '#E87119'
}


def classify_subfield(title, abstract, subfields):
    combined_text = f"{title} {abstract}".lower()
    for subfield, keywords in subfields.items():
        for kw in keywords:
            if kw in combined_text:
                return subfield
    return 'other'

#==================================================
# filtering stage
df_cl['subfield'] = df_cl.apply(
    lambda row: classify_subfield(row['title'], row['abstract'], nlp_subfields),axis=1)

df_pivot = df_cl.pivot_table(
    index='year',
    columns='subfield',
    aggfunc='size',
    fill_value=0
).drop('other',axis=1)

#==================================================
# Visualization Stage: Highlighting the Golden Age

## 1. Figure Setup
plt.figure(figsize=(10, 4))
    
years = sorted(df_pivot.index)
bar_width = 0.18
positions = np.arange(len(years))

## 2. Plotting Data Lines
for i, task in enumerate(df_pivot.columns):
    counts = df_pivot[task].reindex(years).fillna(0).values
    x_pos = positions + (i - len(df_pivot.columns) / 2) * bar_width + bar_width / 2

    plt.bar(
        x_pos,
        counts,
        width=bar_width,
        label=task.title(),
        color=custom_colors.get(task, 'gray')
    )


## 3. Highlighting Key Periods (Golden Age + Milestones)
plt.title("NLP Subfields Growth During The Golden Age of Theory ArXiv", fontweight='bold')
plt.xlabel("Year")
plt.ylabel("Number of Papers")

plt.gca().set_facecolor('#fafab9')

plt.axvline(x=years.index(2019)-0.32, color='lime', linestyle='--', linewidth=2, label='2019')
plt.axvline(x=years.index(2021)+0.32, color='lime', linestyle='--', linewidth=2, label='2021')

## 4. Final Styling and Saving Figure
plt.xticks(positions, years)
plt.grid(True, axis='y', linestyle='--', alpha=0.55)
plt.legend(title='NLP Subfield',loc='upper left',title_fontsize=9.8,prop={'size':10})
plt.tight_layout()
plt.savefig("Figure 7. NLP Subfields Growth During The Golden Age of Theory ArXiv.png")
plt.show()



# Constants
nlp_fields = {
    'text generation':      '#d62728', 
    'text classification':  '#2ca02c', 
    'question answering':   '#9467bd',}
    
side_lines_c = '#F39237'
ga_area_c = '#F1E100'  # The Golden Age of Theory area color

#==================================================
# filtering stage
total_views_supdata = kt[['Name', 'TotalViews', 'year']]

#==================================================
# Visualization Stage: Highlighting the Golden Age

## 1. Figure Setup
plt.figure(figsize=(10, 4))

## 2. Plotting Data Lines
for field, color in nlp_fields.items():
    mask = total_views_supdata['Name'].str.contains(field, case=False, na=False)
    subset = total_views_supdata[mask].copy()
    
    # Group by year
    grouped = subset.groupby('year').size()
    
    # Plot
    plt.plot(grouped.index, grouped.values, label=field.title(), color=color, marker='o')


## 3. Highlighting Key Periods (Golden Age + Milestones)
plt.title("Flourishing of NLP Subfields in The Golden Age of Theory")
plt.xlabel("Year")
plt.ylabel("Sub-fields flourish")

plt.gca().set_facecolor('#fafab9')

plt.axvline(x=2019, color=side_lines_c, linestyle='--', linewidth=2, label='2019')
plt.axvline(x=2021, color=side_lines_c, linestyle='--', linewidth=2, label='2021')
plt.axvspan(2019, 2021, color=ga_area_c, alpha=0.9, label='Golden Age')

## 4. Final Styling and Saving Figure
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('Figure 8. Flourishing of NLP Subfields in The Golden Age of Theory (on kaggle).png')
plt.show()



# Constants
keyword_dataset = 'nlp'
side_lines_c = '#F39237'
ga_area_c = '#F1E100' #The Golden Age of Theory area color

#==================================================
# Reading Stage
dataset = pd.read_csv(f'{MK_PATH}/Datasets.csv')

dataset_tasks = pd.read_csv(f'{MK_PATH}/DatasetTasks.csv') 

dataset_tags = pd.read_csv(f'{MK_PATH}/DatasetTags.csv')

tags = pd.read_csv(f'{MK_PATH}/Tags.csv')

#==================================================
# Merging Stage
dataset_w_tags = pd.merge(dataset_tags, dataset, left_on='DatasetId', right_on='Id')

dataset_w_tags = pd.merge(dataset_w_tags, tags, left_on='TagId', right_on='Id')

dataset_w_tags = pd.merge(dataset_w_tags, dataset_tasks[['DatasetId', 'Title']], on='DatasetId')

dataset_w_tags_task = dataset_w_tags[["CreationDate","TotalDownloads", "TotalViews","Name"]]

#==================================================
# filtering stage
mask = (
    dataset_w_tags_task ['Name'].str.contains(keyword_dataset, case=False, na=False))

nlp_datasets = dataset_w_tags_task [mask].copy()

nlp_datasets['CreationDate'] = pd.to_datetime(nlp_datasets['CreationDate'])
nlp_datasets['year'] = nlp_datasets['CreationDate'].dt.year
group_dataset = nlp_datasets.groupby('year').size()
GR = (group_dataset[2021] - group_dataset[2019]) / group_dataset[2019] * 100 # Growth rate

#==================================================
# Visualization Stage: Highlighting the Golden Age

## 1. Figure Setup

plt.figure(figsize=(9, 4))

## 2. Plotting Data Lines
plt.plot(group_dataset, marker='o' , label='NlP Dataset Growth')

## 3. Highlighting Key Periods (Golden Age + Milestones)
plt.title('Dataset Flourishing During the The Golden Age of Theory' , x=0.48 , y=1.12)
plt.xlabel('Date')
plt.ylabel('Number of dataset')

plt.gca().set_facecolor('#fafab9')

plt.axvline(x=2019, color=side_lines_c, linestyle='--', linewidth=2 , label='2019')
plt.axvline(x=2021, color=side_lines_c, linestyle='--', linewidth=2 , label='2021')
plt.axvspan(2019, 2021, color=ga_area_c, alpha=0.9, label='Golden Age')

x = plt.xlim()[-1]
x_pos = x-1.17
y = group_dataset.max()
y_pos = y*1.078
fontsize = 8.5

plt.text(
    x_pos,
    y_pos,
    f"⬆ GR (2019 → 21) NLP: +{GR:.0f}%",
    fontsize=8.5,
    color='#1565c0',
    weight='bold',
    bbox=dict(
        boxstyle='round,pad=0.3',
        facecolor='#e3f2fd',               
        edgecolor='#90caf9',               
        alpha=0.85))

## 4. Final Styling and Saving Figure
plt.xlim(right=2021+0.2)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('Figure 9. Dataset Flourishing During the The Golden Age of Theory.png')
plt.show()



# Constants
side_lines_c = '#F39237'
ga_area_c = '#F1E100' #The Golden Age of Theory area color

keywords_dataset2 = ['nlp','computer vision']

custom_colors = {
    'nlp': '#1297a5',
    'computer vision': '#872657'
}

#==================================================
# Merging & filtering stage

dataset_cv_nlp  = pd.merge(dataset_tags, dataset, left_on='DatasetId', right_on='Id')

dataset_cv_nlp  = pd.merge(dataset_cv_nlp[['DatasetId','TagId','CreationDate','TotalDownloads']] , tags, left_on='TagId', right_on='Id')

dataset_cv_nlp  = pd.merge(dataset_cv_nlp[['DatasetId','TagId','CreationDate','TotalDownloads' ,'Name']] , dataset_tasks[['DatasetId']], on='DatasetId')

dataset_cv_nlp['CreationDate'] = pd.to_datetime(dataset_cv_nlp['CreationDate'])
dataset_cv_nlp['year'] = dataset_cv_nlp['CreationDate'].dt.year


#==================================================
# Visualization Stage: Highlighting the Golden Age

## 1. Figure Setup
plt.figure(figsize=(9, 4))
    
## 2. Plotting Data Lines
for keyword in keywords_dataset2:
    
    mask = dataset_cv_nlp['Name'].str.contains(keyword, case=False, na=False)

    filtred_data = dataset_cv_nlp[mask].copy()
    grouped_to = filtred_data.groupby('year')['TotalDownloads'].sum()

    plt.plot(grouped_to ,color=custom_colors[keyword] , label=keyword.upper() , marker='o')


## 3. Highlighting Key Periods (Golden Age + Milestones)
plt.title("NLP vs. CV Dataset Downloads During The Golden Age of Theory",fontsize=10) #  Suggested default
plt.xlabel('Years')
plt.ylabel('TotalDownloads')

plt.gca().set_facecolor('#fafab9')

plt.axvline(x=2019, color=side_lines_c, linestyle='--', linewidth=2 , label='2019')
plt.axvline(x=2021, color=side_lines_c, linestyle='--', linewidth=2 , label='2021')
plt.axvspan(2019, 2021, color=ga_area_c, alpha=0.9, label='Golden Age')


## 4. Final Styling and Saving Figure
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('Figure 10. NLP vs. CV Dataset Downloads During The Golden Age of Theory.png')
plt.show()



# Constants
side_lines_c = '#F39237'
ga_area_c = '#F1E100' #The Golden Age of Theory area color

keywords = ['nlp','computer vision']

custom_colors = {
    'nlp': '#1297a5',
    'computer vision': '#872657'
}

#==================================================
# Reading Stage
kernels = pd.read_csv(f'{MK_PATH}/Kernels.csv')
kernel_tags = pd.read_csv(f'{MK_PATH}/KernelTags.csv')
tags = pd.read_csv(f'{MK_PATH}/Tags.csv')

forum_topics = pd.read_csv(f'{MK_PATH}/ForumTopics.csv')
forum_topics = forum_topics.dropna(subset=['KernelId'])
forum_topics = forum_topics.replace([np.inf, -np.inf], np.nan)

#==================================================
# Merging & filtering stage
k = pd.merge(
    kernels[['Id','CreationDate',]],
    kernel_tags[['KernelId','TagId']],
    left_on = 'Id',
    right_on= 'KernelId')

k = pd.merge(
    k,
    tags[['Id','Name']],
    left_on = 'TagId',
    right_on = 'Id')

k = k[['KernelId','Name','CreationDate']]
k['CreationDate'] = pd.to_datetime(k['CreationDate'])
k['year'] = k['CreationDate'].dt.year
k = k.drop('CreationDate' , axis=1)


note_fourm = pd.merge(
    k,
    forum_topics[['KernelId', 'CreationDate', 'TotalViews', 'Score']],
    on = 'KernelId',
    suffixes=('', '_form'))

last_year = note_fourm ['year'].max() 
note_fourm  = note_fourm [note_fourm ['year'] != last_year] # delete 2025

#==================================================
# Visualization Stage: Highlighting the Golden Age

## 1. Figure Setup
plt.figure(figsize=(9.5, 4.5))

## 2. Plotting Data Lines
for keyword in keywords:
    
    mask = note_fourm['Name'].str.contains(keyword, case=False, na=False)

    filtred_data = note_fourm[mask].copy()
    grouped_to = filtred_data.groupby('year')['Score'].sum()

    plt.plot(grouped_to ,color=custom_colors[keyword] , label=keyword.upper() , marker='o')

## 3. Highlighting Key Periods (Golden Age + Milestones)
plt.title("NLP vs. CV Formula Topics During The Golden Age of Theory", fontsize=11) # Suggested title (default)
plt.xlabel('Years')
plt.ylabel('Score')

plt.gca().set_facecolor('#fafab9')

plt.axvline(x=2019, color=side_lines_c, linestyle='--', linewidth=2 , label='2019')
plt.axvline(x=2021, color=side_lines_c, linestyle='--', linewidth=2 , label='2021')
plt.axvspan(2019, 2021, color=ga_area_c, alpha=0.9, label='Golden Age')

## 4. Final Styling and Saving Figure
plt.grid(True)
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('Figure 11. NLP vs. CV Formula Topics During The Golden Age of Theory.png')
plt.show()



