import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

_ = ["" for i in range(250_000)]


path = '/kaggle/input/19-june-2025-fertilizer/submission__LB__0_'

st_names = ['37_474','37_631','37_780','37_856','GEN_1','GEN_2','GEN_3','GEN_4','GEN_5']
obezyany = ['any3_0','any3_1','any3_2','any3_3','gen_1','gen_2','gen_3','gen_4','gen_5']

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

def Visual_any3in4_GEN(dfs,ns,gen):
    ns             = ns[:4]
    nslb           = ns
    nsg2,nsg4      = ns+['GEN_2'],ns+['GEN_4']
    nsg1,nsg3,nsg5 = ns+['GEN_1'],ns+['GEN_3'],ns+['GEN_5']
    dfs['any3_0']  = dfs.apply(lambda x: f_any3in4(x,0,nslb), axis=1)
    dfs['any3_1']  = dfs.apply(lambda x: f_any3in4(x,1,nslb), axis=1)
    dfs['any3_2']  = dfs.apply(lambda x: f_any3in4(x,2,nslb), axis=1)
    dfs['any3_3']  = dfs.apply(lambda x: f_any3in4(x,3,nslb), axis=1)
    dfs['gen_1']   = dfs.apply(lambda x: f_not_equ(x,4,nsg1), axis=1)
    dfs['gen_2']   = dfs.apply(lambda x: f_not_equ(x,5,nsg2), axis=1)
    dfs['gen_3']   = dfs.apply(lambda x: f_not_equ(x,6,nsg3), axis=1)
    dfs['gen_4']   = dfs.apply(lambda x: f_not_equ(x,7,nsg4), axis=1)
    dfs['gen_5']   = dfs.apply(lambda x: f_not_equ(x,8,nsg5), axis=1)

    plt.figure(figsize=(10, 3))
    sns.heatmap(dfs.isnull(), cbar=False, cmap="Blues")
    plt.title(f"Heatmap of different lines in Top.4 solutions  and  {gen}")
    plt.show()


dfs = load()

Visual_any3in4_GEN(dfs, st_names, gen='GEN_1-5')


# df = pd.read_csv(path + 'GEN_1.csv') # Lb = 0.37_986
# df = pd.read_csv(path + 'GEN_2.csv') # Lb = 0.37_867
# df = pd.read_csv(path + 'GEN_3.csv') # Lb = 0.37_999
# df = pd.read_csv(path + 'GEN_4.csv') # Lb = 0.37_988

df = pd.read_csv(path + 'GEN_5.csv')  # Lb = 0.38_009

df.to_csv('submission_GEN_5.csv', index=False)

display(df)





path = '/kaggle/input/20-june-2025-fertilizer/submission__LB__0_'

#df = pd.read_csv(path + 'GEN_8.csv')   # Lb = 0.37_672
#df = pd.read_csv(path + 'GEN_F1.csv')  # Lb = 0.37_996
#df = pd.read_csv(path + 'GEN_F2.csv')  # Lb = 0.38_005

df = pd.read_csv(path + 'GEN_7.csv')   # Lb = 0.38_005        ( GEN_3 + GEN_5 + GEN_F2) 

df.to_csv('submission.csv', index=False)

display(df)

