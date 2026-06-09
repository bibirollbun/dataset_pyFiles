import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

_ = ["" for i in range(250_000)]

path = '/kaggle/input/27-june-2025-fertilizer/submission__LB__0_'

st_names = ['38_357','38_265','38_261','38_333','GEN_30.2']
obezyany = ['any3_0','any3_1','any3_2','any3_3','g30.2']

def Visual_any3in4_GEN(dfs,ns,gen):
    ns            = ns[:4]
    nslb          = ns
    nsg1          = ns+['GEN_30.2']
    dfs['any3_0'] = dfs.apply(lambda x: f_any3in4(x,0,nslb), axis=1)
    dfs['any3_1'] = dfs.apply(lambda x: f_any3in4(x,1,nslb), axis=1)
    dfs['any3_2'] = dfs.apply(lambda x: f_any3in4(x,2,nslb), axis=1)
    dfs['any3_3'] = dfs.apply(lambda x: f_any3in4(x,3,nslb), axis=1)
    dfs['g30.2']  = dfs.apply(lambda x: f_not_equ(x,7,nsg1), axis=1)
    
    plt.figure(figsize=(10, 3))
    sns.heatmap(dfs.isnull(), cbar=False, cmap="Blues")
    plt.title(f"Heatmap of different lines in Top.4 solutions  and  {gen}")
    plt.show()


def read(ns):
    return [pd.read_csv(path + name_subm +'.csv') for name_subm in ns]


def rename(dfs,ns=st_names):
    for i in range(len(dfs)):
        dfs[i] = dfs[i].rename(columns={'Fertilizer Name': f'{ns[i]}'})


def addany(dfs,obezyany):
    for i in range(len(dfs)):
        dfs[i][f'{obezyany[i]}'] = _


def merge(dfs):
    dfsm = pd.merge(dfs[0], dfs[1], on="id")
    for i in range(2,len(dfs)):
        dfsm = pd.merge(dfsm,dfs[i],on='id')
    return dfsm


def load():
    dfs = read(st_names)
    rename(dfs,st_names)
    addany(dfs,obezyany)
    return merge(dfs)


def f_any3in4(x,ix,ns):
    i = [n for n in range(len(ns))]
    i.remove(ix)
    if x[ns[i[0]]]==x[ns[i[1]]]==x[ns[i[2]]] and x[ns[i[0]]]!=x[ns[ix]]:
        return None
    return ""


def f_not_equ(x,l,ns):
    for i in range(len(ns)):
        for j in range(len(ns)):
            if i!=j and x[ns[i]]==x[ns[j]]:
                return ""
    return None


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
    m2 = pd.DataFrame({ name :query_Q_vs(name,st_names,list_vs,mvs) for name in st_names})
    matrix_df = pd.concat([m1,m2],axis=1)
    return matrix_df


dfs = load()

Visual_any3in4_GEN(dfs, st_names, gen='GEN_30.2')



# for ex. if target == Urea 28-28 DAP
print('target')
matrix = matrix_vs(st_names, dfs, -1)
display(matrix)

# Urea
print('target[0]')
matrix_pie_0 = matrix_vs(st_names, dfs, 0)
display(matrix_pie_0)

# 28-28
print('target[1]')
matrix_pie_1 = matrix_vs(st_names, dfs, 1)
display(matrix_pie_1)

# DAP
print('target[2]')
matrix_pie_2 = matrix_vs(st_names, dfs, 2)
display(matrix_pie_2)


A = pd.read_csv (path + '38_357.csv')   
B = pd.read_csv (path + 'GEN_27.7b.csv')

A,B = A.iloc[0:111_111],B.iloc[111_111:250_001]
df = pd.concat([A,B], axis=0)

df.to_csv('submission.csv',index=False)    

df



# df1 = pd.read_csv(path + 'GEN_27.7a.csv')  # Lb=0.38_262
# df2 = pd.read_csv(path + 'GEN_27.7b.csv')  # Lb=0.38_282
# df3 = pd.read_csv(path + 'GEN_27.7c.csv')  # Lb=0.38.259
# df4 = pd.read_csv(path + 'GEN_28.71.csv')  # Lb=0.38_273
# df5 = pd.read_csv(path + 'GEN_28.74.csv')  # Lb=0.38_273

# A = pd.read_csv (path + '38_285.csv')             # Lb=0.38_285
# B = pd.read_csv (path + 'GEN_27.7b.csv')          # Lb=0.38_282
# D = pd.read_csv (path + 'GEN_29.1.csv')           # Lb=0.38_308

# A,B = A.iloc[0:127_000],B.iloc[127_000:250_001]  # Lb=0.38_325
# A,B = A.iloc[0:120_000],B.iloc[120_000:250_001]  # Lb=0.38_332
# A,B = A.iloc[0:113_000],B.iloc[113_000:250_001]  # Lb=0.38_348
# df = pd.concat([A,B], axis=0)

