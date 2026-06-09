# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

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


import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


path_to_subm = "/kaggle/input/10-july-2025-ps-s5e7/"
path_to_work = "/kaggle/working/"

st_names_gr3 = [f'3.{i}' for i in range(1,11)]
st_names_gr4 = [f'4.{i}' for i in range(1,11)]
st_names_gr5 = [f'5.{i}' for i in range(1,11)]

st_names     = st_names_gr3 + st_names_gr4 + st_names_gr5

st_names_grs = [st_names_gr3, st_names_gr4, st_names_gr5]


def load_submissions(path, file_names):
    dfs = [pd.read_csv(f'{path}{file_names[i]}'+".csv") for i in range(0,len(file_names))]
    for i in range(0,len(dfs)):
        dfs[i].rename(columns={'Personality': f'{file_names[i]}'}, inplace=True)
    dfsm = pd.merge(dfs[0], dfs[1], on="id")
    for i in range(2,len(dfs)):
        dfsm = pd.merge(dfsm,dfs[i],on='id')
    return dfsm


def majority_vote(row,  file_names=st_names):
    preds = [row[col] for col in file_names]
    count = Counter(preds)
    top = count.most_common()
    return top[0][0]




def pswd_wts_vote(row, gCols, wts):
    intro,extro = 0,0
    if len(gCols) != len(wts): return None
    for i in range(len(wts)):
        for col in gCols[i]:
            if row[col] == "Introvert":
                intro += wts[i]
            else: extro += wts[i]
    if intro >= extro: return "Introvert"
    return "Extrovert"



def major(dfsm,st_names,st_names_grs,wts):
    dfsm['Persona_maj'] = dfsm.apply(lambda x: majority_vote(x,st_names), axis=1)
    dfsm['Persona_wts'] = dfsm.apply(lambda x: pswd_wts_vote(x,st_names_grs,wts), axis=1)
    df_maj = dfsm[['id', 'Persona_maj']]
    df_wts = dfsm[['id', 'Persona_wts']]
    df_maj.to_csv(path_to_work + f"submission_maj_{str(wts)}.csv", index=False)
    df_wts.to_csv(path_to_work + f"submission_wts_{str(wts)}.csv", index=False)
    df_compare = dfsm.query('Persona_maj != Persona_wts')
    display(df_compare[['id','Persona_maj','Persona_wts']])
    return df_wts,df_maj


def load(path,st_names):
    dfs = [pd.read_csv(path + name_subm +'.csv') for name_subm in st_names]
    for i in range(len(dfs)):
        dfs[i] = dfs[i].rename(columns={'Personality': f'{st_names[i]}'})
    dfsm = pd.merge(dfs[0], dfs[1], on="id")
    for i in range(2,len(dfs)):
        dfsm = pd.merge(dfsm,dfs[i],on='id')
    return dfsm


def make_list_vs(st_names):
    list = []
    for i in range(0,len(st_names)-1):
        for j in range(i+1,len(st_names)):
            list.append(st_names[i] + "_vs_" + st_names[j])
    return list


def equ_tt(x,target_1,target_2, pie):
    if pie == -1:
        if x[target_1]==x[target_2]:
            return 0
    else:
        if x[target_1].split()[pie] == x[target_2].split()[pie]:
            return 0
    return 1


def get_mvs(dfs, list_vs,pie):
    for vs in list_vs:
        t = vs.split('_vs_')
        dfs[vs] = dfs.apply(lambda x: equ_tt(x,t[0],t[1],pie), axis=1)
    return dfs


def query_Q_vs(name, st_names, list_vs, dfs):
    Q = []
    for st in st_names:
        vs_between = name + "_vs_" + st
        if vs_between not in list_vs:
            Q.append(0)
        else: Q.append(dfs[vs_between].sum())
    return Q



def matrix_vs(st_names,dfs,pie=-1):
    list_vs = make_list_vs(st_names)
    mvs = get_mvs(dfs, list_vs, pie)
    m1 = pd.DataFrame({'subm':st_names})
    m2 = pd.DataFrame({ name :query_Q_vs(name, st_names, list_vs, mvs) for name in st_names})
    matrix_df = pd.concat([m1,m2],axis=1)
    return matrix_df




dfs = load(path_to_subm, st_names_gr3)
matrix = matrix_vs(st_names_gr3, dfs, -1)
display(matrix)

dfs = load(path_to_subm, st_names_gr4)
matrix = matrix_vs(st_names_gr4, dfs, -1)
display(matrix)

dfs = load(path_to_subm, st_names_gr5)
matrix = matrix_vs(st_names_gr5, dfs, -1)
display(matrix)




dfsm = load_submissions(path_to_subm, st_names)

df1,_  = major(dfsm.copy(),st_names,st_names_grs,[10,5,3]) # LB=0.976_518, v.14 <
df2,_  = major(dfsm.copy(),st_names,st_names_grs,[10,3,1]) # LB=0.976_518, v.15
df3,_  = major(dfsm.copy(),st_names,st_names_grs,[10,2,1]) # LB=0.976_518, v.16

_,df4  = major(dfsm.copy(),st_names,st_names_grs,[ 7,7,7]) # LB=0.975_708, v.17

df5,_  = major(dfsm.copy(),st_names,st_names_grs,[10,5,4]) # LB=0.97?_---, v.18


df     = df5
df.to_csv("submission.csv", index=False)
df




