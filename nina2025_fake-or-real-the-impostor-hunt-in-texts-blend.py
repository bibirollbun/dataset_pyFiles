import pandas as pd

from collections import Counter

path_to_work = '/kaggle/working/'


def load_submissions(path, st_names):
    dfs = [pd.read_csv(path + '/submission ' + name +'.csv') for name in st_names]
    for i in range(0,len(dfs)):
        dfs[i].rename(columns={'real_text_id': f'{st_names[i]}'}, inplace=True)
    dfsm = pd.merge(dfs[0], dfs[1], on="id")
    for i in range(2,len(dfs)):
        dfsm = pd.merge(dfsm,dfs[i],on='id')
    return dfsm


def majority_vote(row, st_names):
    preds = [row[col] for col in st_names]
    count = Counter(preds)
    top = count.most_common()
    return top[0][0]


def pswd_wts_vote(row, gCols, wts):
    A,B = 0,0
    if len(gCols) != len(wts): return None
    for i in range(len(wts)):
        if row[gCols[i]] == 1:
            A += wts[i]
        else: B += wts[i]
    if A >= B: return 1
    return 2


def major(dfsm,st_names,st_names_grs,wts):
    dfsm['Fake_Real__maj'] = dfsm.apply(lambda x: majority_vote(x,st_names), axis=1)
    dfsm['Fake_Real__wts'] = dfsm.apply(lambda x: pswd_wts_vote(x,st_names_grs,wts), axis=1)
    df_maj = dfsm[['id', 'Fake_Real__maj']]
    df_wts = dfsm[['id', 'Fake_Real__wts']]
    df_maj.to_csv(path_to_work + f"submission_maj_{str(wts)}.csv", index=False)
    df_wts.to_csv(path_to_work + f"submission_wts_{str(wts)}.csv", index=False)
    df_compare = dfsm.query('Fake_Real__maj != Fake_Real__wts')
    display(df_compare[['id','Fake_Real__maj','Fake_Real__wts']])
    return df_wts,df_maj


def load(path,st_names):
    dfs = [pd.read_csv(path + '/submission ' + name +'.csv') for name in st_names]
    for i in range(len(dfs)):
        dfs[i] = dfs[i].rename(columns={'real_text_id': f'{st_names[i]}'})
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


path_to_subm = '/kaggle/input/21-juli-2025-fake-or-real'

st_names = ['0.82157','0.84232','0.84233','0.87759','0.87966','0.87967','0.90456']

dfs = load(path_to_subm, st_names)


matrix = matrix_vs(st_names, dfs, -1)

display(matrix)


dfsm = load_submissions(path_to_subm, st_names)

st_names_grs,wts = st_names,[2,3,3,4,5,5,8]

df1,_  = major(dfsm.copy(),st_names,st_names_grs,wts)


dfsm = load_submissions(path_to_subm, st_names)

st_names_grs,wts = st_names,[2,3,3,4,5,5,21]

df2,_  = major(dfsm.copy(),st_names,st_names_grs,wts)


dfsm = load_submissions(path_to_subm, st_names)

st_names_grs,wts = st_names,[1,1,1,1,1,1,1]

_,df3  = major(dfsm.copy(),st_names,st_names_grs,wts)


df = df1  # LB = 0.90041  wts = [2,3,3,4,5,5,8]    version.2
df = df2  # LB = 0.90456  wts = [2,3,3,4,5,5,21]   version.3
df = df3  # LB = 0.89834  majority vote            version.4
df = df2  # Top.LB

df = df.rename(columns={'Fake_Real__wts':'real_text_id', 'Fake_Real__maj':'real_text_id'})
df.to_csv("submission_0.90456.csv", index=False)
df


A = pd.read_csv ('/kaggle/input/21-juli-2025-fake-or-real/submission 0.87966.csv')
B = pd.read_csv ('/kaggle/input/21-juli-2025-fake-or-real/submission 0.87967.csv')

#A,B = A.iloc[0:534],B.iloc[534:1068]
#df = pd.concat([A,B], axis=0)
#df.to_csv('submission__0.87344.csv',index=False)                # LB = 0.87344
#df

A,B = B.iloc[0:534],A.iloc[534:1068]
df = pd.concat([A,B], axis=0)
df.to_csv('submission_0.88589.csv',index=False)                 # LB = 0.88589


path_to_subm = '/kaggle/input/21-juli-2025-fake-or-real'

st_names = ['0.82157','0.84232','0.84233','0.87759','0.87966','0.87967','0.88589','0.90456']

dfs = load(path_to_subm, st_names)

matrix = matrix_vs(st_names, dfs, -1)

display(matrix)


dfsm = load_submissions(path_to_subm, st_names)


st_names_grs,wts = st_names,[2,3,3, 4, 4,4,4,10]  # lb = 0.89834

st_names_grs,wts = st_names,[2,3,3,10, 4,4,4,20]  # lb = 0.90041

st_names_grs,wts = st_names,[2,3,3, 4, 4,4,4,20]  # lb = 0.90456

df,_  = major(dfsm.copy(),st_names,st_names_grs,wts)


df = df.rename(columns={'Fake_Real__wts':'real_text_id', 'Fake_Real__maj':'real_text_id'})

df.to_csv("submission.csv", index=False)

df


# Fake or Real: The Impostor Hunt in Texts | blend - Version 8
# Complete · 15m ago · Notebook Fake or Real: The Impostor Hunt in Texts | blend | Version 8
# 0.88589

# Fake or Real: The Impostor Hunt in Texts | blend - Version 7
# Complete · 18m ago · Notebook Fake or Real: The Impostor Hunt in Texts | blend | Version 7
# 0.87344

# Fake or Real | blend - Version 4
# Complete · 17h ago · Notebook Fake or Real | blend | Version 4
# 0.89834

# Fake or Real: The Impostor Hunt in Texts | blend - Version 3
# Complete · 17h ago · Notebook Fake or Real | blend | Version 3
# 0.90456

# Fake or Real | blend - Version 2
# Complete · 18h ago · Notebook Fake or Real | blend | Version 2
# 0.90041

