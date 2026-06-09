import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

_ = ["" for i in range(250_000)]

path = '/kaggle/input/27-june-2025-fertilizer/submission__LB__0_'


# st_names = ['38_285', '38_215', '38_213', '38_192','37_971']
# obezyany = ['any3_0', 'any3_1', 'any3_2', 'any3_3','any3_4']
# st_names += ['GEN_27.1a', 'GEN_27.1b', 'GEN_27.1c']
# obezyany += [   'g27.1a',    'g27.1b',    'g27.1c']

# def Visual_any3in4_GEN(dfs, ns, gen):
#     ns = ns[:5]
#     nslb = ns
#     nsg1 = ns + ['GEN_27.1a']
#     nsg2 = ns + ['GEN_27.1b']
#     nsg3 = ns + ['GEN_27.1c']
#     dfs['any3_0'] = dfs.apply(lambda x: f_any3in4(x, 0, nslb), axis=1)
#     dfs['any3_1'] = dfs.apply(lambda x: f_any3in4(x, 1, nslb), axis=1)
#     dfs['any3_2'] = dfs.apply(lambda x: f_any3in4(x, 2, nslb), axis=1)
#     dfs['any3_3'] = dfs.apply(lambda x: f_any3in4(x, 3, nslb), axis=1)
#     dfs['any3_4'] = dfs.apply(lambda x: f_any3in4(x, 4, nslb), axis=1)
#     dfs['g27.1a'] = dfs.apply(lambda x: f_not_equ(x, 6, nsg1), axis=1)
#     dfs['g27.1b'] = dfs.apply(lambda x: f_not_equ(x, 7, nsg2), axis=1)
#     dfs['g27.1c'] = dfs.apply(lambda x: f_not_equ(x, 8, nsg3), axis=1)
#     plt.figure(figsize=(10, 3))
#     sns.heatmap(dfs.isnull(), cbar=False, cmap="Blues")
#     plt.title(f"Heatmap of different lines in Top.4 solutions  and  {gen}")
#     plt.show()


# st_names  = ['38_285','38_216','38_213','38_000','37_758']
# obezyany  = ['any3_0','any3_1','any3_2','any3_3','any3_4']
# st_names += ['GEN_27.2a','GEN_27.2b','GEN_27.2c']
# obezyany += [   'g27.2a',   'g27.2b',   'g27.2c']

# def Visual_any3in4_GEN(dfs, ns, gen):
#     ns = ns[:5]
#     nslb = ns
#     nsg1 = ns + ['GEN_27.2a']
#     nsg2 = ns + ['GEN_27.2b']
#     nsg3 = ns + ['GEN_27.2c']
#     dfs['any3_0'] = dfs.apply(lambda x: f_any3in4(x, 0, nslb), axis=1)
#     dfs['any3_1'] = dfs.apply(lambda x: f_any3in4(x, 1, nslb), axis=1)
#     dfs['any3_2'] = dfs.apply(lambda x: f_any3in4(x, 2, nslb), axis=1)
#     dfs['any3_3'] = dfs.apply(lambda x: f_any3in4(x, 3, nslb), axis=1)
#     dfs['any3_4'] = dfs.apply(lambda x: f_any3in4(x, 4, nslb), axis=1)
#     dfs['g27.2a'] = dfs.apply(lambda x: f_not_equ(x, 5, nsg1), axis=1)
#     dfs['g27.2b'] = dfs.apply(lambda x: f_not_equ(x, 6, nsg2), axis=1)
#     dfs['g27.2c'] = dfs.apply(lambda x: f_not_equ(x, 7, nsg3), axis=1)
#     plt.figure(figsize=(10, 3))
#     sns.heatmap(dfs.isnull(), cbar=False, cmap="Blues")
#     plt.title(f"Heatmap of different lines in Top.4 solutions  and  {gen}")
#     plt.show()


# st_names  = ['38_285','38_213','38_000','37_971','37_758']
# obezyany  = ['any3_0','any3_1','any3_2','any3_3','any3_4']
# st_names += ['GEN_27.3a','GEN_27.3b','GEN_27.3c']
# obezyany += [   'g27.3a',   'g27.3b',   'g27.3c']

# def Visual_any3in4_GEN(dfs, ns, gen):
#     ns = ns[:5]
#     nslb = ns
#     nsg1 = ns + ['GEN_27.3a']
#     nsg2 = ns + ['GEN_27.3b']
#     nsg3 = ns + ['GEN_27.3c']
#     dfs['any3_0'] = dfs.apply(lambda x: f_any3in4(x, 0, nslb), axis=1)
#     dfs['any3_1'] = dfs.apply(lambda x: f_any3in4(x, 1, nslb), axis=1)
#     dfs['any3_2'] = dfs.apply(lambda x: f_any3in4(x, 2, nslb), axis=1)
#     dfs['any3_3'] = dfs.apply(lambda x: f_any3in4(x, 3, nslb), axis=1)
#     dfs['any3_4'] = dfs.apply(lambda x: f_any3in4(x, 4, nslb), axis=1)
#     dfs['g27.3a'] = dfs.apply(lambda x: f_not_equ(x, 5, nsg1), axis=1)
#     dfs['g27.3b'] = dfs.apply(lambda x: f_not_equ(x, 6, nsg2), axis=1)
#     dfs['g27.3c'] = dfs.apply(lambda x: f_not_equ(x, 7, nsg3), axis=1)
#     plt.figure(figsize=(10, 3))
#     sns.heatmap(dfs.isnull(), cbar=False, cmap="Blues")
#     plt.title(f"Heatmap of different lines in Top.4 solutions  and  {gen}")
#     plt.show()


st_names  = ['38_285','38_213','38_192','38_000','37_971']
obezyany  = ['any3_0','any3_1','any3_2','any3_3','any3_4']
st_names += ['GEN_27.4a','GEN_27.4b','GEN_27.4c']
obezyany += [   'g27.4a',   'g27.4b',   'g27.4c']
def Visual_any3in4_GEN(dfs, ns, gen):
    ns = ns[:5]
    nslb = ns
    nsg1 = ns + ['GEN_27.4a']
    nsg2 = ns + ['GEN_27.4b']
    nsg3 = ns + ['GEN_27.4c']
    dfs['any3_0'] = dfs.apply(lambda x: f_any3in4(x, 0, nslb), axis=1)
    dfs['any3_1'] = dfs.apply(lambda x: f_any3in4(x, 1, nslb), axis=1)
    dfs['any3_2'] = dfs.apply(lambda x: f_any3in4(x, 2, nslb), axis=1)
    dfs['any3_3'] = dfs.apply(lambda x: f_any3in4(x, 3, nslb), axis=1)
    dfs['any3_4'] = dfs.apply(lambda x: f_any3in4(x, 4, nslb), axis=1)
    dfs['g27.4a'] = dfs.apply(lambda x: f_not_equ(x, 5, nsg1), axis=1)
    dfs['g27.4b'] = dfs.apply(lambda x: f_not_equ(x, 6, nsg2), axis=1)
    dfs['g27.4c'] = dfs.apply(lambda x: f_not_equ(x, 7, nsg3), axis=1)
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

Visual_any3in4_GEN(dfs, st_names, gen='GEN_27.4(a,b,c)')


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


df1 = pd.read_csv(path + 'GEN_27.4a.csv')  # Lb = 0.38_254
df2 = pd.read_csv(path + 'GEN_27.4b.csv')  # Lb = 0.38_254
df3 = pd.read_csv(path + 'GEN_27.4c.csv')  # Lb = ?

df4 = pd.read_csv(path + 'GEN_27.7f.csv')  # Lb = 0.28_278
df5 = pd.read_csv(path + 'GEN_27.7e.csv')  # Lb = ?

df = df5

df.to_csv('submission.csv', index=False)

display(df)

