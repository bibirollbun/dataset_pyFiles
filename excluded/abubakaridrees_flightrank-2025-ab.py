# %load_ext cudf.pandas

# RAPIDS cuDF 25.02 cuML 25.02 -> add to UTILITY SCRIPT 


import pandas as pd


def iBlend(path_to_ds, file_short_names, sls):

    def tida(sls):
        
        def read_subm(sls,i):
            tnm = sls["subm"][i]["name"]
            FiN = sls["path"] + tnm + ".csv"
            df = pd.read_csv(FiN).rename(columns={'target':tnm, sls["target"]:tnm})
            del df["ranker_id"]
            return df
        
        dfs_subm = [read_subm(sls,i) for i in range(len(sls["subm"]))]
        
        df_subms = pd.merge(dfs_subm[0],  dfs_subm[1], on=['Id'])  
        for i in range(2, len(sls["subm"])): 
            df_subms = pd.merge(df_subms, dfs_subm[i], on=['Id'])
            
        cols = [col for col in df_subms.columns if col != "Id"]
        short_name_cols = [c.replace(sls["prefix"], '') for c in cols]
        corrects = [wt for wt in sls["subwts"]]
        weights = [subm['weight'] for subm in sls["subm"]]
        
        def alls(x, cs=cols):
            tes = {c: x[c] for c in cs}.items()
            subms_sorted = [
              t[0].replace(sls["prefix"], '')
              for t in sorted(tes,key=lambda k:k[1],reverse=True if sls["sort"]=='desc' else False)]
            return subms_sorted
        
        def correct(x, cs=cols, w=weights, cw=corrects):
            ic = [x['alls'].index(c) for c in short_name_cols]
            cS = [x[cols[j]] * (w[j] + cw[ic[j]]) for j in range(len(cols))]
            return sum(cS)
        
        df_subms['alls']        = df_subms.apply(lambda x: alls   (x), axis=1)
        df_subms[sls["target"]] = df_subms.apply(lambda x: correct(x), axis=1)
        
        schema_rename = { old_nc:new_shnc for old_nc, new_shnc in zip(cols, short_name_cols) }
        
        df_subms = df_subms.rename(columns=schema_rename)
        df_subms = df_subms.rename(columns={sls["target"]:"ensemble"})
        
        df_subms.insert(loc=1, column=' _ ', value=['   '] * sls["q_rows"])
        
        df_subms[' _ '] = df_subms[' _ '].astype(str)
        pd.set_option('display.max_rows',100)
        pd.set_option('display.float_format', '{:.2f}'.format)
        vcols = ['Id'] + [' _ '] + short_name_cols + [' _ '] + ['alls'] + [' _ '] + ['ensemble']
        df_subms = df_subms[vcols]
        display(df_subms.head(7))
        pd.set_option('display.float_format', '{:.0f}'.format)
        df_subms = df_subms.rename(columns={"ensemble":sls["target"]})
        return df_subms
        

    sample_subm = pd.read_csv(path_to_ds + file_short_names[1] + ".csv")

    
    def ensemble_tida(sls,submission=sample_subm):   
        sls['sort'] = 'desc'
        dfs = tida(sls)
        dfD = dfs[['Id', sls['target']]]
        dfD.to_csv(f'tida_desc.csv', index=False)
        sls['sort'] = 'asc'
        dfs = tida(sls)
        dfA = dfs[['Id', sls['target']]]
        dfA.to_csv(f'tida_asc.csv',  index=False)
        target,d,a = sls['target'],sls['desc'],sls['asc']
        submission[target] = round((dfD[target] * d + a * dfA[target]),0)
        submission[target] = submission[sls['target']].round().astype(int)
        return submission

    
    submission = ensemble_tida(sls)

    
    return submission


# Archiv

# path_to_ds ='/kaggle/input/20-juli-2025-flightrank/submission '

# file_short_names = ['0.42163','0.43916','0.47635','0.48388']

# params_v5 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "dynamic",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.50,
#       'asc'   : 0.50,
#       'subwts': [+0.11, +0.04, -0.04, -0.11],                          # LB = 0.48379,  v.5
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.05, },
#         { 'name':file_short_names[1],'weight':0.08, },
#         { 'name':file_short_names[2],'weight':0.13, },
#         { 'name':file_short_names[3],'weight':0.74, },
#       ]
#     }

# params_v6 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "dynamic",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.30,
#       'asc'   : 0.70,
#       'subwts': [+0.21, -0.07, -0.07, -0.07],                          # LB = 0.48443,  v.6
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.10, },
#         { 'name':file_short_names[1],'weight':0.10, },
#         { 'name':file_short_names[2],'weight':0.10, },
#         { 'name':file_short_names[3],'weight':0.70, },
#       ]
#     }

# params_v7 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "dynamic",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.30,
#       'asc'   : 0.70,
#       'subwts': [+0.21, -0.03, -0.07, -0.11],                          # LB = 0.48462,  v.7
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.10, },
#         { 'name':file_short_names[1],'weight':0.10, },
#         { 'name':file_short_names[2],'weight':0.10, },
#         { 'name':file_short_names[3],'weight':0.70, },
#       ]
#     }

# params_v8 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "dynamic",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.30,
#       'asc'   : 0.70,
#       'subwts': [+0.11, -0.01, -0.03, -0.07],                          # LB = 0.48397
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.10, },
#         { 'name':file_short_names[1],'weight':0.10, },
#         { 'name':file_short_names[2],'weight':0.10, },
#         { 'name':file_short_names[3],'weight':0.70, },
#       ]
#     }

# params_v9 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "dynamic",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.30,
#       'asc'   : 0.70,
#       'subwts': [+0.21, -0.02, -0.07, -0.12],                          # LB = 0.48517,  v.10
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.05, },
#         { 'name':file_short_names[1],'weight':0.08, },
#         { 'name':file_short_names[2],'weight':0.13, },
#         { 'name':file_short_names[3],'weight':0.74, },
#       ]
#     }


file_short_names = ['0.47635','0.48388','0.48397','0.48425']

path_to_ds ='/kaggle/input/20-juli-2025-flightrank/submission '


params_v12 = {
      'path'  : path_to_ds,                                 
      'sort'  : "dynamic",
      'target': "selected",
      'q_rows': 6_897_776,
      'prefix': "subm_",
      'desc'  : 0.30,
      'asc'   : 0.70,
      'subwts': [+0.33, -0.11, -0.11, -0.11],                          # LB = 0.48507,  v.12
      'subm'  : [
        { 'name':file_short_names[0],'weight':0.25, },
        { 'name':file_short_names[1],'weight':0.25, },
        { 'name':file_short_names[2],'weight':0.25, },
        { 'name':file_short_names[3],'weight':0.25, },
      ]
    }

params_v13 = {
      'path'  : path_to_ds,                                 
      'sort'  : "dynamic",
      'target': "selected",
      'q_rows': 6_897_776,
      'prefix': "subm_",
      'desc'  : 0.30,
      'asc'   : 0.70,
      'subwts': [+0.33, -0.04, -0.11, -0.18],                          # LB = 0.48452,  v.13
      'subm'  : [
        { 'name':file_short_names[0],'weight':0.25, },
        { 'name':file_short_names[1],'weight':0.25, },
        { 'name':file_short_names[2],'weight':0.25, },
        { 'name':file_short_names[3],'weight':0.25, },
      ]
    }

params_v14 = {
      'path'  : path_to_ds,                                 
      'sort'  : "dynamic",
      'target': "selected",
      'q_rows': 6_897_776,
      'prefix': "subm_",
      'desc'  : 0.20,
      'asc'   : 0.80,
      'subwts': [+0.21, -0.02, -0.07, -0.12],                          # LB = 0.48526,  v.14
      'subm'  : [
        { 'name':file_short_names[0],'weight':0.30, },
        { 'name':file_short_names[1],'weight':0.30, },
        { 'name':file_short_names[2],'weight':0.30, },
        { 'name':file_short_names[3],'weight':0.10, },
      ]
    }

params_v11 = {
      'path'  : path_to_ds,                                 
      'sort'  : "dynamic",
      'target': "selected",
      'q_rows': 6_897_776,
      'prefix': "subm_",
      'desc'  : 0.30,
      'asc'   : 0.70,
      'subwts': [+0.21, -0.02, -0.07, -0.12],                          # LB = 0.48590,  v.11
      'subm'  : [
        { 'name':file_short_names[0],'weight':0.30, },
        { 'name':file_short_names[1],'weight':0.30, },
        { 'name':file_short_names[2],'weight':0.30, },
        { 'name':file_short_names[3],'weight':0.10, },
      ]
    }

params_v15 = {
      'path'  : path_to_ds,                                 
      'sort'  : "dynamic",
      'target': "selected",
      'q_rows': 6_897_776,
      'prefix': "subm_",
      'desc'  : 0.40,
      'asc'   : 0.60,
      'subwts': [+0.21, -0.02, -0.07, -0.12],                          # LB = 0.48608,  v.15
      'subm'  : [
        { 'name':file_short_names[0],'weight':0.30, },
        { 'name':file_short_names[1],'weight':0.30, },
        { 'name':file_short_names[2],'weight':0.30, },
        { 'name':file_short_names[3],'weight':0.10, },
      ]
    }

params_v17 = {
      'path'  : path_to_ds,                                 
      'sort'  : "dynamic",
      'target': "selected",
      'q_rows': 6_897_776,
      'prefix': "subm_",
      'desc'  : 0.45,
      'asc'   : 0.55,
      'subwts': [+0.21, -0.02, -0.07, -0.12],                          # LB = 0.48544,  v.17
      'subm'  : [
        { 'name':file_short_names[0],'weight':0.30, },
        { 'name':file_short_names[1],'weight':0.30, },
        { 'name':file_short_names[2],'weight':0.30, },
        { 'name':file_short_names[3],'weight':0.10, },
      ]
    }

params_v18 = {
      'path'  : path_to_ds,                                 
      'sort'  : "dynamic",
      'target': "selected",
      'q_rows': 6_897_776,
      'prefix': "subm_",
      'desc'  : 0.45,
      'asc'   : 0.55,
      'subwts': [+0.21, -0.02, -0.07, -0.12],                          # LB = 0.48572,  v.18
      'subm'  : [
        { 'name':file_short_names[0],'weight':0.27, },
        { 'name':file_short_names[1],'weight':0.30, },
        { 'name':file_short_names[2],'weight':0.30, },
        { 'name':file_short_names[3],'weight':0.13, },
      ]
    }

params_v19 = {
      'path'  : path_to_ds,                                 
      'sort'  : "dynamic",
      'target': "selected",
      'q_rows': 6_897_776,
      'prefix': "subm_",
      'desc'  : 0.40,
      'asc'   : 0.60,
      'subwts': [+0.21, -0.02, -0.07, -0.12],                          # LB = 0.48572,      v.19
      'subm'  : [
        { 'name':file_short_names[0],'weight':0.27, },
        { 'name':file_short_names[1],'weight':0.33, },
        { 'name':file_short_names[2],'weight':0.33, },
        { 'name':file_short_names[3],'weight':0.07, },
      ]
    }

params_v20 = {
      'path'  : path_to_ds,                                 
      'sort'  : "dynamic",
      'target': "selected",
      'q_rows': 6_897_776,
      'prefix': "subm_",
      'desc'  : 0.40,
      'asc'   : 0.60,
      'subwts': [+0.17, -0.02, -0.05, -0.10],                          # LB = 0.48581,      v.20
      'subm'  : [
        { 'name':file_short_names[0],'weight':0.27, },
        { 'name':file_short_names[1],'weight':0.33, },
        { 'name':file_short_names[2],'weight':0.33, },
        { 'name':file_short_names[3],'weight':0.07, },
      ]
    }

params_v21 = {
      'path'  : path_to_ds,                                 
      'sort'  : "dynamic",
      'target': "selected",
      'q_rows': 6_897_776,
      'prefix': "subm_",
      'desc'  : 0.40,
      'asc'   : 0.60,
      'subwts': [+0.20, -0.02, -0.07, -0.11],                          # LB = ?,            v.21
      'subm'  : [
        { 'name':file_short_names[0],'weight':0.29, },
        { 'name':file_short_names[1],'weight':0.30, },
        { 'name':file_short_names[2],'weight':0.30, },
        { 'name':file_short_names[3],'weight':0.11, },
      ]
    }


%%time

params = params_v11  #  LB=0.48590
params = params_v12  #  LB=0.48507
params = params_v13  #  LB=0.48452
params = params_v14  #  LB=0.48590
params = params_v15  #  LB=0.48608  < 
params = params_v17  #  LB=0.48544
params = params_v18  #  LB=0.48572
params = params_v19  #  LB=0.48572
params = params_v20  #  LB=0.48581

params = params_v21  #


df = iBlend ( path_to_ds, file_short_names, params )

df.to_csv('submission_tida.csv', index=False)


df1 = pd.read_csv("submission_tida.csv")

df2 = pd.read_csv("/kaggle/input/20-juli-2025-flightrank/submission 0.43916.csv")
df3 = pd.read_csv("/kaggle/input/20-juli-2025-flightrank/submission 0.42163.csv")
df4 = pd.read_csv("/kaggle/input/20-juli-2025-flightrank/submission 0.41226.csv")


dfs = [
    df1,df2,df3,df4
]

def rank2score(sr, eps=1e-6):
    n = sr.max()
    return 1.0 - (sr - 1) / (n + eps)

score_frames = []
for i, df in enumerate(dfs):
    tmp = df[['Id', 'ranker_id', 'selected']].copy()
    tmp['score'] = tmp.groupby('ranker_id')['selected'].transform(rank2score)
    score_frames.append(tmp[['Id', 'ranker_id', 'score']].rename(columns={'score': f'score_{i}'}))

merged = score_frames[0]
for i in range(1, 4):
    merged = merged.merge(score_frames[i], on=['Id', 'ranker_id'], how='left')

weights = [0.997, 0.001, 0.001, 0.001]
score_cols = [f'score_{i}' for i in range(4)]
w = pd.Series(weights, index=score_cols)
merged['score_mean'] = (merged[score_cols] * w).sum(axis=1) / w.sum()

def score2rank(s):
    return s.rank(method='first', ascending=False).astype(int)

merged['selected'] = merged.groupby('ranker_id')['score_mean'].transform(score2rank)


out = merged[['Id', 'ranker_id', 'selected']]

out.to_csv("submission.csv", index=False, float_format='%.0f')

out

