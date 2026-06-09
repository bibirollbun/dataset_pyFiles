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

df.to_csv('submission_GEN_7.csv', index=False)

display(df)


path = '/kaggle/input/21-june-2025-fertilizer/submission__LB__0_'

st_names = ['37_631','37_758','37_780','37_856','GEN_11','GEN_13','GEN_15']
obezyany = ['any3_0','any3_1','any3_2','any3_3','gen_11','gen_13','gen_15']

def Visual_any3in4_GEN(dfs,ns,gen):
    ns             = ns[:4]
    nslb           = ns
    nsg1,nsg3,nsg5 = ns+['GEN_11'],ns+['GEN_13'],ns+['GEN_15']
    dfs['any3_0']  = dfs.apply(lambda x: f_any3in4(x,0,nslb), axis=1)
    dfs['any3_1']  = dfs.apply(lambda x: f_any3in4(x,1,nslb), axis=1)
    dfs['any3_2']  = dfs.apply(lambda x: f_any3in4(x,2,nslb), axis=1)
    dfs['any3_3']  = dfs.apply(lambda x: f_any3in4(x,3,nslb), axis=1)
    dfs['gen_11']  = dfs.apply(lambda x: f_not_equ(x,4,nsg1), axis=1)
    dfs['gen_13']  = dfs.apply(lambda x: f_not_equ(x,5,nsg3), axis=1)
    dfs['gen_15']  = dfs.apply(lambda x: f_not_equ(x,8,nsg5), axis=1)

    plt.figure(figsize=(10, 3))
    sns.heatmap(dfs.isnull(), cbar=False, cmap="Blues")
    plt.title(f"Heatmap of different lines in Top.4 solutions  and  {gen}")
    plt.show()


def load(obezyany=obezyany,st_names=st_names):
    dfs = read(st_names)
    rename(dfs,st_names)
    addany(dfs,obezyany)
    return merge(dfs)


dfs = load()

Visual_any3in4_GEN(dfs, st_names, gen='GEN_11,13,15')


df = pd.read_csv(path + 'GEN_11.csv') # Lb = 0.38_089
#df = pd.read_csv(path + 'GEN_13.csv') # Lb = 0.38_084
#df = pd.read_csv(path + 'GEN_15.csv') # Lb = 0.38_088

display(df)


path = '/kaggle/input/22-june-2025-fertilizer/submission__LB__0_'

st_names = ['37_631','37_758','37_780','37_856','GEN_11','GEN_12','GEN_13','GEN_14','GEN_15']
obezyany = ['any3_0','any3_1','any3_2','any3_3','gen_11','gen_12','gen_13','gen_14','gen_15']

def Visual_any3in4_GEN(dfs,ns,gen):
    ns             = ns[:4]
    nslb           = ns
    nsg2,nsg4      = ns+['GEN_12'],ns+['GEN_14']
    nsg1,nsg3,nsg5 = ns+['GEN_11'],ns+['GEN_13'],ns+['GEN_15']
    dfs['any3_0']  = dfs.apply(lambda x: f_any3in4(x,0,nslb), axis=1)
    dfs['any3_1']  = dfs.apply(lambda x: f_any3in4(x,1,nslb), axis=1)
    dfs['any3_2']  = dfs.apply(lambda x: f_any3in4(x,2,nslb), axis=1)
    dfs['any3_3']  = dfs.apply(lambda x: f_any3in4(x,3,nslb), axis=1)
    dfs['gen_11']  = dfs.apply(lambda x: f_not_equ(x,4,nsg1), axis=1)
    dfs['gen_12']  = dfs.apply(lambda x: f_not_equ(x,5,nsg2), axis=1)
    dfs['gen_13']  = dfs.apply(lambda x: f_not_equ(x,6,nsg3), axis=1)
    dfs['gen_14']  = dfs.apply(lambda x: f_not_equ(x,7,nsg3), axis=1)
    dfs['gen_15']  = dfs.apply(lambda x: f_not_equ(x,8,nsg5), axis=1)

    plt.figure(figsize=(10, 3))
    sns.heatmap(dfs.isnull(), cbar=False, cmap="Blues")
    plt.title(f"Heatmap of different lines in Top.4 solutions  and  {gen}")
    plt.show()


def load(obezyany=obezyany,st_names=st_names):
    dfs = read(st_names)
    rename(dfs,st_names)
    addany(dfs,obezyany)
    return merge(dfs)


dfs = load()

Visual_any3in4_GEN(dfs, st_names, gen='GEN_11,12,13,14,15')


df = pd.read_csv(path + 'GEN_12.csv')   # Lb=0.38_114     

display(df)


path = '/kaggle/input/23-june-2025-fertilizer/submission__LB__0_'

st_names = ['37_867','37_758','37_780','37_856','GEN_17','GEN_18.1','GEN_18.2','GEN_18.3','GEN_18.4']
obezyany = ['any3_0','any3_1','any3_2','any3_3','gen_17','gen_18.1','gen_18.2','gen_18.3','gen_18.4']

def Visual_any3in4_GEN(dfs,ns,gen):
    ns             = ns[:4]
    nslb           = ns
    nsg7           = ns+['GEN_17']
    nsg1           = ns+['GEN_18.1']
    nsg2           = ns+['GEN_18.2']
    nsg3           = ns+['GEN_18.3']
    nsg4           = ns+['GEN_18.4']
    dfs['any3_0']  = dfs.apply(lambda x: f_any3in4(x,0,nslb), axis=1)
    dfs['any3_1']  = dfs.apply(lambda x: f_any3in4(x,1,nslb), axis=1)
    dfs['any3_2']  = dfs.apply(lambda x: f_any3in4(x,2,nslb), axis=1)
    dfs['any3_3']  = dfs.apply(lambda x: f_any3in4(x,3,nslb), axis=1)
    dfs['gen_17']  = dfs.apply(lambda x: f_not_equ(x,4,nsg7), axis=1)
    dfs['gen_18.1']= dfs.apply(lambda x: f_not_equ(x,5,nsg1), axis=1)
    dfs['gen_18.2']= dfs.apply(lambda x: f_not_equ(x,6,nsg2), axis=1)
    dfs['gen_18.3']= dfs.apply(lambda x: f_not_equ(x,7,nsg3), axis=1)
    dfs['gen_18.4']= dfs.apply(lambda x: f_not_equ(x,8,nsg4), axis=1)

    plt.figure(figsize=(10, 3))
    sns.heatmap(dfs.isnull(), cbar=False, cmap="Blues")
    plt.title(f"Heatmap of different lines in Top.4 solutions  and  {gen}")
    plt.show()


def load(obezyany=obezyany,st_names=st_names):
    dfs = read(st_names)
    rename(dfs,st_names)
    addany(dfs,obezyany)
    return merge(dfs)


dfs = load()

Visual_any3in4_GEN(dfs, st_names, gen='GEN_17,18.1-18.5')


#df  = pd.read_csv(path + 'GEN_11.csv')             
#df.to_csv(    'submission_GEN_11.csv',   index=False) # Lb = 0.38_089

#df  = pd.read_csv(path + 'GEN_12.csv')             
#df.to_csv(    'submission_GEN_12.csv',   index=False) # Lb = 0.38_114  <

#df  = pd.read_csv(path + 'GEN_13.csv')            
#df.to_csv(    'submission_GEN_13.csv',   index=False) # Lb = 0.38_084

#df  = pd.read_csv(path + 'GEN_14.csv')            
#df.to_csv(    'submission_GEN_14.csv',   index=False) # Lb = 0.38_098

#df  = pd.read_csv(path + 'GEN_15.csv')            
#df.to_csv(    'submission_GEN_15.csv',   index=False) # Lb = 0.38_088

#df  = pd.read_csv(path + 'GEN_17.csv')            
#df.to_csv(    'submission_GEN_17.csv',   index=False) # Lb = 0.38_145  <

#df1 = pd.read_csv(path + 'GEN_18.1.csv')
#df1.to_csv(   'submission_GEN_18.1.csv', index=False) # Lb = 0.38_160  <

#df2 = pd.read_csv(path + 'GEN_18.2.csv')
#df2.to_csv(   'submission_GEN_18.2.csv', index=False) # Lb = 0.38_121  v.7

#df3 = pd.read_csv(path + 'GEN_18.3.csv')
#df3.to_csv(   'submission_GEN_18.3.csv', index=False) # Lb = 0.38_125  v.8

#df4 = pd.read_csv(path + 'GEN_18.4.csv')
#df4.to_csv(   'submission_GEN_18.4.csv', index=False) # Lb = 0.38_131  v.9



path = '/kaggle/input/24-june-2025-fertilizer/submission__LB__0_'

st_names = ['38_213','38_192','38_000','37_971','37_758','GEN_20.1','GEN_20.2','GEN_20.3']
obezyany = ['any3_0','any3_1','any3_2','any3_3','any3_4',   'g20.1',   'g20.2',   'g20.3']

def Visual_any3in4_GEN(dfs,ns,gen):
    nslb          = ns[:5]
    nsg1          = ns+['GEN_20.1']
    nsg2          = ns+['GEN_20.2']
    nsg3          = ns+['GEN_20.3']
    dfs['any3_0'] = dfs.apply(lambda x: f_any3in4(x,0,nslb), axis=1)
    dfs['any3_1'] = dfs.apply(lambda x: f_any3in4(x,1,nslb), axis=1)
    dfs['any3_2'] = dfs.apply(lambda x: f_any3in4(x,2,nslb), axis=1)
    dfs['any3_3'] = dfs.apply(lambda x: f_any3in4(x,3,nslb), axis=1)
    dfs['any3_4'] = dfs.apply(lambda x: f_any3in4(x,4,nslb), axis=1)
    dfs['g20.1']  = dfs.apply(lambda x: f_not_equ(x,5,nsg1), axis=1)
    dfs['g20.2']  = dfs.apply(lambda x: f_not_equ(x,6,nsg2), axis=1)
    dfs['g20.3']  = dfs.apply(lambda x: f_not_equ(x,7,nsg3), axis=1)

    plt.figure(figsize=(10, 3))
    sns.heatmap(dfs.isnull(), cbar=False, cmap="Blues")
    plt.title(f"Heatmap of different lines in Top.4 solutions  and  {gen}")
    plt.show()


def load(obezyany=obezyany,st_names=st_names):
    dfs = read(st_names)
    rename(dfs,st_names)
    addany(dfs,obezyany)
    return merge(dfs)


dfs = load()

Visual_any3in4_GEN(dfs, st_names, gen='GEN_20.1-3')


df1 = pd.read_csv(path + 'GEN_20.1.csv')  # Lb = 0.38_198
df2 = pd.read_csv(path + 'GEN_20.2.csv')  # Lb = ?
df3 = pd.read_csv(path + 'GEN_20.3.csv')  # Lb = ?

df = df2

df.to_csv('submission.csv', index=False)

display(df)

