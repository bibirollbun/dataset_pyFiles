import ast
import numpy as np
import pandas as pd


def v_blend(path_to_ds, file_short_names, dk):

    def read(dk,i):
        tnm = dk["subm"][i]["name"]
        FiN = dk["path"] + tnm + ".csv"
        return pd.read_csv(FiN).rename(columns={'target':tnm, dk["target"]:tnm})
        
    def merge(dfs_subm):
        df_subms = pd.merge(dfs_subm[0],  dfs_subm[1], on=[dk['id']])
        for i in range(2, len(dk["subm"])): 
            df_subms = pd.merge(df_subms, dfs_subm[i], on=[dk['id']])
        return df_subms
        
    def da(dk,sorting_direction):
        
        df_subms = merge([read(dk,i) for i in range(len(dk["subm"]))])
        cols = [col for col in df_subms.columns if col != dk['id']]
        short_name_cols = [c for c in cols]
        
        def alls(x, sd=sorting_direction,cs=cols):
            reverse = True if sd=='desc' else False
            tes = {c: x[c] for c in cs}.items()
            subms_sorted = [t[0] for t in sorted(tes,key=lambda k:k[1],reverse=reverse)]
            return subms_sorted
            
        def summa(x,cs,wts,ic_alls): 
            return sum([x[cs[j]] * (wts[0][j] + wts[1][ic_alls[j]]) for j in range(len(cs))])
            
        wts = [
            [[e['weight'] for e in dk["subm"]], [w for w in dk["subwts" ]]],
            [[e['weight'] for e in dk["subm2"]],[w for w in dk["subwts2"]]],
            [[e['weight'] for e in dk["subm3"]],[w for w in dk["subwts3"]]],
        ]
        def correct(x, cs=cols, wts=wts):
            i = [x['alls'].index(c) for c in short_name_cols]
            if len(short_name_cols) == 3:
                if   0.000 < x['mx-m'] <= 0.070: return summa(x,cs,wts[0],i)
                if   0.070 < x['mx-m'] <= 0.140: return summa(x,cs,wts[1],i)
                else:                            return summa(x,cs,wts[2],i)
            if len(short_name_cols) == 2:
                if   0.000 < x['mx-m'] <= 0.001: return summa(x,cs,wts[0],i)
                if   0.001 < x['mx-m'] <= 0.003: return summa(x,cs,wts[1],i)
                else:                            return summa(x,cs,wts[2],i)
                
        def amxm(x, cs=cols):
            list_values = x[cs].to_list()
            mxm = abs(max(list_values)-min(list_values))
            return mxm
            
        df_subms['mx-m']       = df_subms.apply(lambda x: amxm   (x), axis=1)
        df_subms['alls']       = df_subms.apply(lambda x: alls   (x), axis=1)
        df_subms[dk["target"]] = df_subms.apply(lambda x: correct(x), axis=1)
        schema_rename = { old_nc:new_shnc for old_nc, new_shnc in zip(cols, short_name_cols) }
        df_subms = df_subms.rename(columns=schema_rename)
        df_subms = df_subms.rename(columns={dk["target"]:"ensemble"})
        df_subms.insert(loc=1, column=' _ ', value=['   '] * len(df_subms))
        df_subms[' _ '] = df_subms[' _ '].astype(str)
        pd.set_option('display.max_rows',100)
        pd.set_option('display.float_format', '{:.4f}'.format)
        vcols = [dk['id']] + [' _ '] + short_name_cols + [' _ '] + ['mx-m'] + [' _ '] + ['alls'] + [' _ '] + ['ensemble']
        df_subms = df_subms[vcols]
        display(df_subms.head(5))
        pd.set_option('display.float_format', '{:.5f}'.format)
        df_subms = df_subms.rename(columns={"ensemble":dk["target"]})
        df_subms.to_csv(f'tida_{sorting_direction}.csv', index=False)
        return df_subms[[dk['id'],dk['target']]]
   
    def ensemble_da(dk): 
        dfD = da(dk,'desc')
        dfA = da(dk,'asc')
        dfA[dk['target']] = dk['desc'] * dfD[dk['target']] + \
                            dk['asc']  * dfA[dk['target']]
        return dfA
    
    return  ensemble_da(dk)


def data_in_col(i,matrix):
    data = [row[i] for row in matrix]
    return data

def quantity(i,js):
    return {"c" : i, "q" : sum(1 for subm in cols[i] if subm == subms[js])}

def dossier(js):
    return {
        'name' : subms[js],
        'q_in' : [quantity(i,js) for i in range(len(subms))]
    }

def info_mx_m(df):
    matrix = [ast.literal_eval(row.alls) for row in df.itertuples()]
    subms = sorted(matrix[0])
    df_subms = pd.DataFrame({f'col_{i}': data_in_col(i,matrix) for i in range(len(subms))})
    fig,(ax1,ax2,ax3) = plt.subplots(ncols=5,figsize=(12,3))
    axs = [ax1,ax2,ax3]
    for i in range(len(subms)):
        axs[i] = sns.countplot(x=df_subms[f"col_{i}"],ax=axs[i])
    plt.tight_layout()
    dossiers = [dossier(js) for js in range(len(subms))]
    for one_dossier in dossiers: 
        print(one_dossier['name'])
        for q in one_dossier['q_in']:
            print("\t",q)



%%time

path ='/kaggle/input/30-august-2025-ps-s5e8/' + 'submission '

fins = ['0.97772','0.97770','0.97768']

params_A = {
        'path'   : path,
        'id'     : 'id',
        'target' : "y",
        'desc'   : 0.70,
        'asc'    : 0.30,
        'subwts' : [+0.03, -0.02, -0.01],
        'subm'   : [
             { 'name':fins[0],'weight':0.85, },
             { 'name':fins[1],'weight':0.14, },
             { 'name':fins[2],'weight':0.01, },
        ],
        'subwts2': [+0.04, -0.01, -0.03],
        'subm2'  : [
             { 'name':fins[0],'weight':0.79, },
             { 'name':fins[1],'weight':0.19, },            # LB = 0.97772
             { 'name':fins[2],'weight':0.02, },
        ],
        'subwts3': [+0.03, -0.01, -0.02],
        'subm3'  : [
             { 'name':fins[0],'weight':0.85, },
             { 'name':fins[1],'weight':0.14, },
             { 'name':fins[2],'weight':0.01, },
        ]
    }

df = v_blend ( path, fins, params_A )

df.to_csv('submission_A.csv', index=False)

display(df)


params_B = {
        'path'   : path,
        'id'     : 'id',
        'target' : "y",
        'desc'   : 0.70,
        'asc'    : 0.30,
        'subwts' : [-0.03, +0.07, -0.04],
        'subm'   : [
             { 'name':fins[0],'weight':0.85, },
             { 'name':fins[1],'weight':0.13, },
             { 'name':fins[2],'weight':0.02, },
        ],
        'subwts2': [-0.04, +0.07, -0.03],
        'subm2'  : [
             { 'name':fins[0],'weight':0.79, },
             { 'name':fins[1],'weight':0.19, },            # LB = ?
             { 'name':fins[2],'weight':0.03, },
        ],
        'subwts3': [+0.03, +0.07, -0.04],
        'subm3'  : [
             { 'name':fins[0],'weight':0.85, },
             { 'name':fins[1],'weight':0.13, },
             { 'name':fins[2],'weight':0.02, },
        ]
    }


df = v_blend ( path, fins, params_B )

df.to_csv('submission_B.csv', index=False)

display(df)

print('\n','~ ~ ~ ~ ~ ~ ~','\n')


path ='/kaggle/working/' + 'submission_'

fins = ['A','B']

params_v18 = {
        'path'   : path,
        'id'     : 'id',
        'target' : "y",
        'desc'   : 0.70,
        'asc'    : 0.30,
        'subwts' : [+0.13, -0.13],
        'subm'   : [
             { 'name':fins[0],'weight':0.55, },
             { 'name':fins[1],'weight':0.45, },
        ],
        'subwts2': [+0.21, -0.21],
        'subm2'  : [
             { 'name':fins[0],'weight':0.50, },
             { 'name':fins[1],'weight':0.50, },            # LB = ?
        ],
        'subwts3': [+0.13, -0.13],
        'subm3'  : [
             { 'name':fins[0],'weight':0.45, },
             { 'name':fins[1],'weight':0.55, },
        ]
    }


df = v_blend ( path, fins, params_v18 )

df.to_csv('submission.csv', index=False)

display(df)

