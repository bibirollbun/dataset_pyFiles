import zipfile
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ctypes
import gc
from tqdm import tqdm
import pickle
from scipy import stats
import collections
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ä�á»‹nh nghÄ©a Ä‘Æ°á»�ng dáº«n
input_dir = "/kaggle/input/santander-product-recommendation/"
output_dir = "/kaggle/working/"

# Danh sÃ¡ch file cáº§n giáº£i nÃ©n
zip_files = [
    "train_ver2.csv.zip",
    "test_ver2.csv.zip",
    "sample_submission.csv.zip"
]

# Giáº£i nÃ©n tá»«ng file
for zip_file in zip_files:
    zip_path = os.path.join(input_dir, zip_file)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)


df = pd.read_csv(os.path.join(output_dir, "train_ver2.csv"), low_memory=False)


df_test = pd.read_csv(os.path.join(output_dir, "test_ver2.csv"), low_memory=False)



print(df.shape)


#This function reduces memory usage of numeric features by changing their data types to the least possible numeric data type.

#https://towardsdatascience.com/how-to-work-with-million-row-datasets-like-a-pro-76fb5c381cdd
def reduce_memory_usage(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df

df = reduce_memory_usage(df)
df_test = reduce_memory_usage(df_test)
#Converting date features to datetime format

df['fecha_dato'] = pd.to_datetime(df.fecha_dato)
df['fecha_alta'] = pd.to_datetime(df.fecha_alta)

df_test['fecha_dato'] = pd.to_datetime(df_test.fecha_dato)
df_test['fecha_alta'] = pd.to_datetime(df_test.fecha_alta)


# Count of null values for each feature
df.isnull().sum()


df.fillna(value={'ind_nomina_ult1':0,'ind_nom_pens_ult1':0},inplace=True)
days_column = (df['fecha_dato'] - df['fecha_alta']).dt.days

#generate new column 'days' from 
df.insert(loc=6, column='days', value=days_column)

#Drop the 'fetch_alta' column
df.drop(columns=['fecha_alta'],inplace = True)

#replace null values with 0 & rest with 1
df['ult_fec_cli_1t'] = df['ult_fec_cli_1t'].apply(lambda x: 1 if pd.notnull(x) else 0)

# remove column tipodom because it has the same value for all customers
df.drop(columns=['tipodom'],inplace = True)

#cod_prov is a duplicate column to nomprov
df.drop(columns=['cod_prov'],inplace = True)

#converting the ' NA' values to nan
df['age'] = pd.to_numeric(df['age'], errors='coerce')

#converting the ' NA' values to nan
df['antiguedad'] = pd.to_numeric(df['antiguedad'], errors='coerce')


df_group = df.groupby('ncodpers').agg({'ind_empleado': lambda x: x.iloc[-1],
                            'pais_residencia': lambda x: x.iloc[-1],
                            'sexo': lambda x: x.iloc[-1],
                            'age': lambda x: x.iloc[-1],
                            'antiguedad': lambda x: x.iloc[-1],
                            'indrel': lambda x: x.iloc[-1],
                            'indrel_1mes': lambda x: x.iloc[-1],
                            'tiprel_1mes': lambda x: x.iloc[-1],
                            'indresi': lambda x: x.iloc[-1],
                            'indext': lambda x: x.iloc[-1],
                            'conyuemp': lambda x: x.iloc[-1],
                            'canal_entrada': lambda x: x.iloc[-1],
                            'indfall': lambda x: x.iloc[-1],
                            'days': lambda x: x.iloc[-1],
                            'nomprov': lambda x: x.iloc[-1],
                            'ind_actividad_cliente': lambda x: x.iloc[-1],
                            'renta': lambda x: x.iloc[-1],
                            'segmento': lambda x: x.iloc[-1],
                            })


df_group


df_group = pd.DataFrame(df_group).reset_index()


# Average yearly income per segment per province
prov_segmento_mean = df_group[(~df_group['renta'].isnull()) & (~df_group['segmento'].isnull()) & (~df_group['nomprov'].isnull())].groupby(['nomprov','segmento']).renta.mean()

# Overall average yearly income per segment of customer 
segmento_renta_mean = df_group[(~df_group['renta'].isnull()) & (~df_group['segmento'].isnull()) & (~df_group['nomprov'].isnull())].groupby('segmento').renta.mean()

# Dataframe indeces of customers who have province and segment name but "null" yearly income
null_indexes = df_group[(~df_group['nomprov'].isnull()) & (~df_group['segmento'].isnull()) & (df_group['renta'].isnull())].index

#Filling those null indices  with avg. yearly income of every province per segment.
for i in tqdm(null_indexes):
    try:
        if prov_segmento_mean[df_group.iloc[i,14]][df_group.iloc[i,17]]:
            df_group.iloc[i,16] = prov_segmento_mean[df_group.iloc[i,14]][df_group.iloc[i,17]]
    except:
        df_group.iloc[i,16] = segmento_renta_mean['01 - TOP']


#plot after filling the yearly income column
plt.figure(figsize=(20,6))
sns.barplot(x="nomprov", y="renta",data = df_group[(~df_group['renta'].isnull()) & (~df_group['segmento'].isnull()) & (~df_group['nomprov'].isnull())],hue = "segmento")
plt.xticks(rotation ='vertical')
plt.title('Province, Segment vs Avg. gross income')
plt.show()


sns.displot(df_group['ind_empleado'])
plt.title('Employee Index Distribution')
plt.show()


plt.figure(figsize=(25,10))
sns.histplot(df_group['pais_residencia'])
plt.title('Country Residence Distribution')
plt.xticks(rotation ='vertical')
plt.show()


sns.displot(df_group['sexo'])
plt.title('Sex Distribution')
plt.show()


plt.figure(figsize=(10,5))
sns.histplot(df_group['age'])
plt.title('Age Distribution')
plt.show()


plt.figure(figsize=(16, 8))
with sns.plotting_context("notebook", font_scale=1.5):
    sns.set_style("whitegrid")
    sns.boxplot(x=df["age"].dropna(), color="tomato")
    plt.title("Outliers in Age Distribution")
    plt.xlabel("Age")
    plt.show()


plt.figure(figsize=(10,5))
sns.distplot(df_group['antiguedad'])
plt.title('Customer Seniority Distribution')
plt.ticklabel_format(useOffset=False,style = 'plain',axis = 'x')
plt.show()


plt.figure(figsize=(16, 8))
with sns.plotting_context("notebook", font_scale=1.5):
    sns.set_style("whitegrid")
    sns.boxplot(x=df["antiguedad"].dropna(), color="tomato")
    plt.title("Outliers in Antiguedad Distribution")
    plt.xlabel("Antiguedad")
    plt.show()


plt.figure(figsize=(10,5))
sns.histplot(df_group['indresi'])
plt.title('Customer residence Distribution')
plt.show()


plt.figure(figsize=(10,5))
sns.histplot(df_group['indext'])
plt.title('Foreigner index Distribution')
plt.show()


plt.figure(figsize=(10,5))
sns.histplot(df_group['conyuemp'])
plt.title('Conyuemp Distribution')
plt.show()


plt.figure(figsize=(25,5))
sns.histplot(df_group['canal_entrada'])
plt.title('Channel Distribution')
plt.xticks(rotation ='vertical')
plt.show()


plt.figure(figsize=(10,5))
sns.histplot(df_group['indfall'])
plt.title('Deceased index Distribution')
plt.show()


plt.figure(figsize=(10,5))
sns.histplot(df_group['nomprov'])
plt.title('Province Distribution')
plt.xticks(rotation ='vertical')
plt.show()


plt.figure(figsize=(10,5))
sns.distplot(df_group['renta'])
plt.title('Gross Income Distribution')
plt.ticklabel_format(useOffset=False,style = 'plain',axis = 'x')
plt.show()


plt.figure(figsize=(10,5))
sns.histplot(df_group['segmento'])
plt.title('Segment Distribution')
plt.show()


plt.figure(figsize=(10,5))
sns.histplot(df_group['days'])
plt.title('Days Distribution')
plt.show()


#Function to sum all values of the product columns

#https://stackoverflow.com/questions/14529838/apply-multiple-functions-to-multiple-groupby-columns/53096340
def f(x):
    d = {}
    d['ahor'] = x['ind_ahor_fin_ult1'].sum()
    d['aval'] = x['ind_aval_fin_ult1'].sum()
    d['cco'] = x['ind_cco_fin_ult1'].sum()
    d['cder'] = x['ind_cder_fin_ult1'].sum()
    d['cno'] = x['ind_cno_fin_ult1'].sum()
    d['ctju'] = x['ind_ctju_fin_ult1'].sum()
    d['ctma'] = x['ind_ctma_fin_ult1'].sum()
    d['ctop'] = x['ind_ctop_fin_ult1'].sum()
    d['ctpp'] = x['ind_ctpp_fin_ult1'].sum()
    d['deco'] = x['ind_deco_fin_ult1'].sum()
    d['deme'] = x['ind_deme_fin_ult1'].sum()
    d['dela'] = x['ind_dela_fin_ult1'].sum()
    d['ecue'] = x['ind_ecue_fin_ult1'].sum()
    d['fond'] = x['ind_fond_fin_ult1'].sum()
    d['hip'] = x['ind_hip_fin_ult1'].sum()
    d['plan'] = x['ind_plan_fin_ult1'].sum()
    d['pres'] = x['ind_pres_fin_ult1'].sum()
    d['reca'] = x['ind_reca_fin_ult1'].sum()
    d['tjcr'] = x['ind_tjcr_fin_ult1'].sum()
    d['valo'] = x['ind_valo_fin_ult1'].sum()
    d['viv'] = x['ind_viv_fin_ult1'].sum()
    d['nomina'] = x['ind_nomina_ult1'].sum()
    d['nom_pens'] = x['ind_nom_pens_ult1'].sum()
    d['recibo'] = x['ind_recibo_ult1'].sum()
    return pd.Series(d, index=['ahor', 'aval', 'cco', 'cder', 'cno','ctju','ctma','ctop','ctpp','deco','deme','dela','ecue','fond','hip','plan','pres','reca','tjcr','valo','viv','nomina','nom_pens','recibo'])


products = df.groupby('fecha_dato').apply(f).columns
dates = df.fecha_dato.unique()


# count of products purchased every month
df.groupby('fecha_dato').apply(f)


#plot of distribution of products purchased every month
a=0
b=0
fig, ax = plt.subplots(6, 3,figsize=(30,40))
for i in tqdm(dates):
    if(b>2):
        b=0
        a+=1
    sns.barplot(x = products, y  = df.groupby('fecha_dato').apply(f).loc[i], ax = ax[a,b])
    plt.setp(ax[a,b].xaxis.get_majorticklabels(), rotation=90)
    b+=1
plt.show()


num_new_prods = pd.DataFrame(columns=['Month', 1, 2, 3, 4, 5, 6, 7])
a = b = 0
fig, ax = plt.subplots(6, 3, figsize=(30, 40))

for i in tqdm(range(len(dates) - 1)):
    if b > 2:
        b = 0
        a += 1
    
    prev_month = df[df['fecha_dato'] == dates[i]].set_index(['ncodpers']).iloc[:, 21:]
    prev_month_customers = df[df['fecha_dato'] == dates[i]]['ncodpers'].to_numpy()
    next_month = df[(df['fecha_dato'] == dates[i+1]) & (df['ncodpers'].isin(prev_month_customers))].set_index(['ncodpers']).iloc[:, 21:]
    
    new_prod = (prev_month - next_month)
    q = (new_prod == -1).sum(axis=1)
    dicts = collections.Counter(q[q > 0])
    
    # ThÃªm dá»¯ liá»‡u má»›i vÃ o DataFrame
    d = {'Month': dates[i+1]}
    d.update(dicts)
    num_new_prods = pd.concat([num_new_prods, pd.DataFrame([d])], ignore_index=True)
    
    # Váº½ biá»ƒu Ä‘á»“
    sns.barplot(x=list(dicts.keys()), y=list(dicts.values()), ax=ax[a, b])
    ax[a, b].set_xlabel('new_products', fontweight='bold')
    ax[a, b].set_ylabel('count', fontweight='bold')
    ax[a, b].set_title(dates[i+1], fontweight='bold')
    b += 1

fig.tight_layout(pad=3.0)
plt.show()


# dataframe of the above plot
num_new_prods


import warnings

new_prods = ['Month', 'ahor', 'aval', 'cco', 'cder', 'cno', 'ctju', 'ctma', 'ctop', 'ctpp', 'deco', 'deme', 'dela', 'ecue', 'fond', 'hip', 'plan', 'pres', 'reca', 'tjcr', 'valo', 'viv', 'nomina', 'nom_pens', 'recibo']
new_prods = pd.DataFrame(columns=new_prods)

warnings.filterwarnings("ignore", category=FutureWarning)

a = b = 0
fig, ax = plt.subplots(6, 3, figsize=(30, 40))
new_prods_dict = {}

for i in tqdm(range(len(dates) - 1)):
    if b > 2:
        b = 0
        a += 1

    prev_month = df[df['fecha_dato'] == dates[i]].set_index(['ncodpers']).iloc[:, 21:]
    prev_month_customers = df[df['fecha_dato'] == dates[i]]['ncodpers'].to_numpy()
    next_month = df[(df['fecha_dato'] == dates[i+1]) & (df['ncodpers'].isin(prev_month_customers))].set_index(['ncodpers']).iloc[:, 21:]
    
    new_prod = (prev_month - next_month)
    new_products = np.zeros((24,))
    
    for j in range(24):
        new_products[j] = len(new_prod.iloc[:, j][new_prod.iloc[:, j] == -1])

    new_prods_dict['Month'] = dates[i+1]

    for k in range(len(products)):
        new_prods_dict[products[k]] = new_products[k]

    new_prods = pd.concat([new_prods, pd.DataFrame([new_prods_dict])], ignore_index=True)
    
    sns.barplot(x=products, y=new_products, ax=ax[a, b])
    plt.setp(ax[a, b].xaxis.get_majorticklabels(), rotation=90)
    ax[a, b].set_xlabel('products', fontweight='bold')
    ax[a, b].set_ylabel('count', fontweight='bold')
    ax[a, b].set_title(dates[i+1], fontweight='bold')
    b += 1

fig.tight_layout(pad=3.0)
plt.show()

warnings.filterwarnings("default", category=FutureWarning)  # Báº­t láº¡i cáº£nh bÃ¡o sau khi hoÃ n thÃ nh



# dataframe of the above plot
new_prods


# Plot to find average yearly income of customers by their province and the segment they belong to
plt.figure(figsize=(20,6))
sns.barplot(x="nomprov", y="renta",data = df_group[(~df_group['renta'].isnull()) & (~df_group['segmento'].isnull()) & (~df_group['nomprov'].isnull())],hue = "segmento")
plt.xticks(rotation ='vertical')
plt.title('Province, Segment vs Avg. gross income')
plt.show()

