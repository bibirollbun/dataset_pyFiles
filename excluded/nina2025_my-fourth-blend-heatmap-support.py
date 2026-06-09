import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


path_ds = '/kaggle/input/june-13-2025-fertilizer-1-17-2'

solut_names = ['','Ozan M','Pirhosseinlou','M_Naumov']

def load(path=path_ds):
    
    path += '/submission__LB_'

    df1  = pd.read_csv(path + '0_36918__v01__Ozan M'             + '.csv')
    df1g = pd.read_csv(path + '0_36918__v01__Ozan M__GEN'        + '.csv')
    df2  = pd.read_csv(path + '0_36937__v03__Pirhosseinlou'      + '.csv')
    df2g = pd.read_csv(path + '0_36937__v03__Pirhosseinlou__GEN' + '.csv')
    df3  = pd.read_csv(path + '0_36964__v03__M_Naumov'           + '.csv')
    df3g = pd.read_csv(path + '0_36964__v03__M_Naumov__GEN'      + '.csv')

    df1  = df1 .rename(columns={'Fertilizer Name':'Ozan M'              })
    df1g = df1g.rename(columns={'Fertilizer Name':'Ozan M'       +'_GEN'})
    df2  = df2 .rename(columns={'Fertilizer Name':'Pirhosseinlou'       })
    df2g = df2g.rename(columns={'Fertilizer Name':'Pirhosseinlou'+'_GEN'})
    df3  = df3 .rename(columns={'Fertilizer Name':'M_Naumov'            })
    df3g = df3g.rename(columns={'Fertilizer Name':'M_Naumov'     +'_GEN'})
    
    anyTwo = ["" for i in range(250_000)]
    notEqu = ["" for i in range(250_000)]
    
    df1 ['anyTwo1'],df2 ['anyTwo2'],df3 ['anyTwo3'] = anyTwo,anyTwo,anyTwo
    df1g['Add'],    df2g['Add'],    df3g['Add']     = notEqu,notEqu,notEqu

    dfs1 = pd.merge(df1,  df2,  on="id")
    dfs1 = pd.merge(dfs1, df3,  on="id")
    dfs1 = pd.merge(dfs1, df1g, on="id")
    
    dfs2 = pd.merge(df1,  df2,  on="id")
    dfs2 = pd.merge(dfs2, df3,  on="id")
    dfs2 = pd.merge(dfs2, df2g, on="id")
        
    dfs3 = pd.merge(df1,  df2,  on="id")
    dfs3 = pd.merge(dfs3, df3,  on="id")
    dfs3 = pd.merge(dfs3, df3g, on="id")
        
    return dfs1,dfs2,dfs3


def f_anyTwo1(x,ns=solut_names):
    if (x[ns[1]]==x[ns[2]] or x[ns[1]]==x[ns[3]]) and x[ns[2]]!=x[ns[3]]: 
        return None
    return ""
def f_anyTwo2(x,ns=solut_names):
    if (x[ns[2]]==x[ns[1]] or x[ns[2]]==x[ns[3]]) and x[ns[1]]!=x[ns[3]]: 
        return None
    return ""
def f_anyTwo3(x,ns=solut_names):
    if (x[ns[3]]==x[ns[1]] or x[ns[3]]==x[ns[2]]) and x[ns[1]]!=x[ns[2]]: 
        return None
    return ""

def f_not_equ(x,ns=solut_names):
    if x[ns[1]]!=x[ns[2]] and x[ns[1]]!=x[ns[3]] and x[ns[2]]!=x[ns[3]]: 
        return None
    return ''


dfs1,dfs2,dfs3 = load()

display(dfs1[5_845:5_847])
display(dfs2[5_845:5_847])
display(dfs3[5_845:5_847])


def Visual_AnyTwo_and_Add(dfs, gen):
    dfs['anyTwo1'] = dfs.apply(lambda x: f_anyTwo1(x), axis=1)
    dfs['anyTwo2'] = dfs.apply(lambda x: f_anyTwo2(x), axis=1)
    dfs['anyTwo3'] = dfs.apply(lambda x: f_anyTwo3(x), axis=1)
    dfs['Add'    ] = dfs.apply(lambda x: f_not_equ(x), axis=1)
    
    plt.figure(figsize=(10, 2.5))
    sns.heatmap(dfs.isnull(), cbar=False, cmap="plasma")
    plt.title(f"Heatmap of  only any Two equal  prediction  and  {gen}  Add")
    plt.show()


Visual_AnyTwo_and_Add(dfs1, gen='Ozan M')
Visual_AnyTwo_and_Add(dfs2, gen='Pirhosseinlou')
Visual_AnyTwo_and_Add(dfs3, gen='M_Naumov')


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


path_ds = '/kaggle/input/13-2025-1-17-4/submission__LB_'

solut_names = ['','Ozan M','Pirhosseinlou','M_Naumov', 'Mahog']

def load(path=path_ds):

    df1  = pd.read_csv(path + '0_36918__v01__Ozan M'             + '.csv')
    df2  = pd.read_csv(path + '0_36937__v03__Pirhosseinlou'      + '.csv')
    df3  = pd.read_csv(path + '0_36964__v03__M_Naumov'           + '.csv')
    df4  = pd.read_csv(path + '0_37228__v01__Mahog'              + '.csv')
   
    df1  = df1 .rename(columns={'Fertilizer Name':'Ozan M'              })
    df2  = df2 .rename(columns={'Fertilizer Name':'Pirhosseinlou'       })
    df3  = df3 .rename(columns={'Fertilizer Name':'M_Naumov'            })
    df4  = df4 .rename(columns={'Fertilizer Name':'Mahog'               })
    
    any_ = ["" for i in range(250_000)]
    
    df1['any3o'], df2['any3p'], df3['any3n'], df4['any4'] = any_, any_, any_, any_

    dfs = pd.merge(df1, df2,  on="id")
    dfs = pd.merge(dfs, df3,  on="id")
    dfs = pd.merge(dfs, df4,  on="id")
    
    return dfs


dfs = load()

display(dfs[5_847:5_854])


def f_any3(x,ns=solut_names):
    if x[ns[1]]==x[ns[2]] and x[ns[2]]==x[ns[3]]: 
        return None
    return ""


def f_any4(x,ns=solut_names):
    if x[ns[1]]==x[ns[2]] and x[ns[2]]==x[ns[3]] and x[ns[3]]==x[ns[4]]: 
        return None
    return ""


def Visual_3x3eq_1x4eq(dfs, gen):
    dfs['any3o'] = dfs.apply(lambda x: f_any3(x), axis=1)
    dfs['any3p'] = dfs.apply(lambda x: f_any3(x), axis=1)
    dfs['any3n'] = dfs.apply(lambda x: f_any3(x), axis=1)
    dfs['any4' ] = dfs.apply(lambda x: f_any4(x), axis=1)
    
    plt.figure(figsize=(10, 4))
    sns.heatmap(dfs.isnull(), cbar=False, cmap='plasma') # cmap="'Blues'")
    plt.title(f"Heatmap {gen}  GEN")
    plt.show()


Visual_3x3eq_1x4eq(dfs, gen='Mahog')


# path = path_ds + '/submission__LB_'

# df1 = pd.read_csv(path + '0_37228__v01__Mahog'      +'__GEN.csv')
# df2 = pd.read_csv(path + '0_36996__v10__Lion_Li_Li' +'__GEN.csv')
# df3 = pd.read_csv(path + '0_36964__v03__M_Naumov'   +'__GEN.csv')    LB = 0.36_977
# df  = df1                            

# path = '/kaggle/input/13-june-2025-fertilizer-1-17/'
# A = pd.read_csv (path+'submission__LB_0_37228__v01__Mahog.csv')
# B = pd.read_csv (path+'submission__LB_0_36964__v03__M_Naumov.csv')   LB = 0.37_277 
# A,B = A.iloc[0:249_980],B.iloc[249_980:250_001]
# df = pd.concat([A,B], axis=0)

# path = '/kaggle/input/june-13-2025-fertilizer-1-17-2/'
# df1 = pd.read_csv(path + '0_36918__v01__Ozan M'        +'__GEN.csv')
# df2 = pd.read_csv(path + '0_36937__v03__Pirhosseinlou' +'__GEN.csv')
# df3 = pd.read_csv(path + '0_36964__v03__M_Naumov'      +'__GEN.csv') LB = 0.36_983
# df  = df3

# df.to_csv('submission.csv', index=False)



df = pd.read_csv(path_ds + '0_37______v01__Mahog__GEN.csv')

df.to_csv('submission.csv', index=False)

df

