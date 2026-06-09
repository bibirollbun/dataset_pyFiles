['0.48425','0.49507', '0.49343', '0.49472', '0.49508', '0.49857']


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
        pd.set_option('display.float_format', '{:.3f}'.format)
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


# Archive

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


# Archive

# file_short_names = ['0.47635','0.48388','0.48397','0.48425']

# path_to_ds ='/kaggle/input/20-juli-2025-flightrank/submission '


# params_v12 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "dynamic",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.30,
#       'asc'   : 0.70,
#       'subwts': [+0.33, -0.11, -0.11, -0.11],                          # LB = 0.48507,  v.12
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.25, },
#         { 'name':file_short_names[1],'weight':0.25, },
#         { 'name':file_short_names[2],'weight':0.25, },
#         { 'name':file_short_names[3],'weight':0.25, },
#       ]
#     }

# params_v13 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "dynamic",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.30,
#       'asc'   : 0.70,
#       'subwts': [+0.33, -0.04, -0.11, -0.18],                          # LB = 0.48452,  v.13
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.25, },
#         { 'name':file_short_names[1],'weight':0.25, },
#         { 'name':file_short_names[2],'weight':0.25, },
#         { 'name':file_short_names[3],'weight':0.25, },
#       ]
#     }

# params_v14 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "dynamic",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.20,
#       'asc'   : 0.80,
#       'subwts': [+0.21, -0.02, -0.07, -0.12],                          # LB = 0.48526,  v.14
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.30, },
#         { 'name':file_short_names[1],'weight':0.30, },
#         { 'name':file_short_names[2],'weight':0.30, },
#         { 'name':file_short_names[3],'weight':0.10, },
#       ]
#     }

# params_v11 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "dynamic",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.30,
#       'asc'   : 0.70,
#       'subwts': [+0.21, -0.02, -0.07, -0.12],                          # LB = 0.48590,  v.11
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.30, },
#         { 'name':file_short_names[1],'weight':0.30, },
#         { 'name':file_short_names[2],'weight':0.30, },
#         { 'name':file_short_names[3],'weight':0.10, },
#       ]
#     }

# params_v15 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "dynamic",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.40,
#       'asc'   : 0.60,
#       'subwts': [+0.21, -0.02, -0.07, -0.12],                          # LB = 0.48608,  v.15
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.30, },
#         { 'name':file_short_names[1],'weight':0.30, },
#         { 'name':file_short_names[2],'weight':0.30, },
#         { 'name':file_short_names[3],'weight':0.10, },
#       ]
#     }

# params_v17 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "dynamic",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.45,
#       'asc'   : 0.55,
#       'subwts': [+0.21, -0.02, -0.07, -0.12],                          # LB = 0.48544,  v.17
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.30, },
#         { 'name':file_short_names[1],'weight':0.30, },
#         { 'name':file_short_names[2],'weight':0.30, },
#         { 'name':file_short_names[3],'weight':0.10, },
#       ]
#     }

# params_v18 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "dynamic",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.45,
#       'asc'   : 0.55,
#       'subwts': [+0.21, -0.02, -0.07, -0.12],                          # LB = 0.48572,  v.18
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.27, },
#         { 'name':file_short_names[1],'weight':0.30, },
#         { 'name':file_short_names[2],'weight':0.30, },
#         { 'name':file_short_names[3],'weight':0.13, },
#       ]
#     }

# params_v19 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "dynamic",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.40,
#       'asc'   : 0.60,
#       'subwts': [+0.21, -0.02, -0.07, -0.12],                          # LB = 0.48572,      v.19
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.27, },
#         { 'name':file_short_names[1],'weight':0.33, },
#         { 'name':file_short_names[2],'weight':0.33, },
#         { 'name':file_short_names[3],'weight':0.07, },
#       ]
#     }

# params_v20 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "dynamic",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.40,
#       'asc'   : 0.60,
#       'subwts': [+0.17, -0.02, -0.05, -0.10],                          # LB = 0.48581,      v.20
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.27, },
#         { 'name':file_short_names[1],'weight':0.33, },
#         { 'name':file_short_names[2],'weight':0.33, },
#         { 'name':file_short_names[3],'weight':0.07, },
#       ]
#     }

# params_v21 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "asc\desc",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.40,
#       'asc'   : 0.60,
#       'subwts': [+0.20, -0.02, -0.07, -0.11],                          # LB = 0.48608,  v.21 < v.15
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.29, },
#         { 'name':file_short_names[1],'weight':0.30, },
#         { 'name':file_short_names[2],'weight':0.30, },
#         { 'name':file_short_names[3],'weight':0.11, },
#       ]
#     }


# Archive

# path_to_ds ='/kaggle/input/20-juli-2025-flightrank/submission '

# file_short_names = ['0.47635','0.48388','0.48397','0.48425','0.49343']

# params_v23 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "asc\desc",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.30,
#       'asc'   : 0.70,
#       'subwts': [+0.17, +0.04, -0.03, -0.07, -0.11],
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.12, },
#         { 'name':file_short_names[1],'weight':0.13, },
#         { 'name':file_short_names[2],'weight':0.13, },
#         { 'name':file_short_names[3],'weight':0.04, },
#         { 'name':file_short_names[4],'weight':0.58, },
#       ]
#     }

# params_v24 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "asc\desc",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.30,
#       'asc'   : 0.70,
#       'subwts': [+0.17, +0.04, -0.03, -0.07, -0.11],
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.10, },
#         { 'name':file_short_names[1],'weight':0.10, },
#         { 'name':file_short_names[2],'weight':0.10, },
#         { 'name':file_short_names[3],'weight':0.03, },
#         { 'name':file_short_names[4],'weight':0.67, },
#       ]
#     }

# params_v25 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "asc\desc",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.30,
#       'asc'   : 0.70,
#       'subwts': [+0.17, +0.04, -0.03, -0.07, -0.11],
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.03, },
#         { 'name':file_short_names[1],'weight':0.03, },
#         { 'name':file_short_names[2],'weight':0.03, },
#         { 'name':file_short_names[3],'weight':0.11, },
#         { 'name':file_short_names[4],'weight':0.80, },
#       ]
#     }



# Archive

# path_to_ds ='/kaggle/input/20-juli-2025-flightrank/submission '

# file_short_names = ['0.48507','0.48425','0.49343']

# params_v26 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "asc\desc",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.40,
#       'asc'   : 0.60,
#       'subwts': [+0.10, -0.03, -0.07],
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.10, },
#         { 'name':file_short_names[1],'weight':0.05, },  # Lb= 0.49315
#         { 'name':file_short_names[2],'weight':0.85, },
#       ]
#     }

# params_v27 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "asc\desc",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.30,
#       'asc'   : 0.70,
#       'subwts': [+0.10, -0.03, -0.07],
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.14, },
#         { 'name':file_short_names[1],'weight':0.07, },  # Lb= 0.49260
#         { 'name':file_short_names[2],'weight':0.79, },
#       ]
#     }

# params_v30 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "asc\desc",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.40,
#       'asc'   : 0.60,
#       'subwts': [+0.08, -0.02, -0.06],
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.30, },
#         { 'name':file_short_names[1],'weight':0.20, },  # Lb= 0.49563
#         { 'name':file_short_names[2],'weight':0.50, },
#       ]
#     }

# params_v28 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "asc\desc",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.40,
#       'asc'   : 0.60,
#       'subwts': [+0.10, -0.03, -0.07],
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.30, },
#         { 'name':file_short_names[1],'weight':0.20, },  # Lb= 0.49560
#         { 'name':file_short_names[2],'weight':0.50, },
#       ]
#     }

# params_v29 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "asc\desc",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.40,
#       'asc'   : 0.60,
#       'subwts': [+0.12, -0.04, -0.08],
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.30, },
#         { 'name':file_short_names[1],'weight':0.20, },  # Lb= 0.49591       Top.v29
#         { 'name':file_short_names[2],'weight':0.50, },
#       ]
#     }

# params_v31 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "asc\desc",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.40,
#       'asc'   : 0.60,
#       'subwts': [+0.13, -0.045, -0.085],
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.31, },
#         { 'name':file_short_names[1],'weight':0.21, },  # Lb= 0.49481
#         { 'name':file_short_names[2],'weight':0.48, },
#       ]
#     }

# params_v32 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "asc\desc",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.40,
#       'asc'   : 0.60,
#       'subwts': [+0.18, -0.5, -0.12],
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.30, },
#         { 'name':file_short_names[1],'weight':0.20, },  # Lb= 0.49187
#         { 'name':file_short_names[2],'weight':0.50, },
#       ]
#     }


# params_v35 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "asc\desc",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.38,
#       'asc'   : 0.62,
#       'subwts': [+0.12, -0.04, -0.08],
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.30, },
#         { 'name':file_short_names[1],'weight':0.20, },
#         { 'name':file_short_names[2],'weight':0.50, },
#       ]
#     } # LB=0.49554 # Top.v29.Lb=0.49591 asc/desc 0.60 x 0.40

# params_v36 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "asc\desc",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.42,
#       'asc'   : 0.58,
#       'subwts': [+0.12, -0.04, -0.08],
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.30, },
#         { 'name':file_short_names[1],'weight':0.20, },  #  Top.v29.Lb=0.49591 asc/desc 0.60 x 0.40
#         { 'name':file_short_names[2],'weight':0.50, },
#       ]
#     } # LB=0.49591 # Top.v29.Lb=0.49591 asc/desc 0.60 x 0.40


# params_v37 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "asc\desc",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.43,
#       'asc'   : 0.57,                                   # LB=0.49591
#       'subwts': [+0.12, -0.04, -0.08],
#       'subm'  : [
#         { 'name':file_short_names[0],'weight':0.30, },
#         { 'name':file_short_names[1],'weight':0.20, },  #  Top.v29.Lb=0.49591 asc/desc 0.60 x 0.40
#         { 'name':file_short_names[2],'weight':0.50, },  #  Top.v36.Lb=0.49591 asc/desc 0.58 x 0.42
#       ]
#     } 

# params_v39 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "asc\desc",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.41,
#       'asc'   : 0.59,                                   # LB=0.49591
#       'subwts': [+0.11, -0.036, -0.074],
#       'subm'  : [
#        {'name':file_short_names[0],'weight':0.30,}, # Top.v29.Lb=0.49591 asc/desc 0.60 x 0.40
#        {'name':file_short_names[1],'weight':0.20,}, # Top.v36.Lb=0.49591 asc/desc 0.58 x 0.42
#        {'name':file_short_names[2],'weight':0.50,}, # Top.v37.Lb=0.49591 asc/desc 0.57 x 0.43
#       ]
#     }

# params_v40 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "asc\desc",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.45,
#       'asc'   : 0.55,                                   # LB=0.49472
#       'subwts': [+0.40, -0.15, -0.25],
#       'subm'  : [
#        {'name':file_short_names[0],'weight':0.30,},
#        {'name':file_short_names[1],'weight':0.20,},
#        {'name':file_short_names[2],'weight':0.50,},
#       ]
#     }

# params_v41 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "asc\desc",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.45,
#       'asc'   : 0.55,                                   # LB=0.49.591
#       'subwts': [+0.11, -0.04, -0.07],
#       'subm'  : [
#        {'name':file_short_names[0],'weight':0.23,},
#        {'name':file_short_names[1],'weight':0.22,},
#        {'name':file_short_names[2],'weight':0.55,},
#       ]
#     }

# params_v42 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "asc\desc",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.45,
#       'asc'   : 0.55,                                   # LB=0.49279
#       'subwts': [+0.11, -0.04, -0.07],
#       'subm'  : [
#        {'name':file_short_names[0],'weight':0.28,},
#        {'name':file_short_names[1],'weight':0.27,},
#        {'name':file_short_names[2],'weight':0.45,},
#       ]
#     }

# params_v43 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "asc\desc",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.45,
#       'asc'   : 0.55,                                   # LB=0.49426
#       'subwts': [+0.11, -0.04, -0.07],
#       'subm'  : [
#        {'name':file_short_names[0],'weight':0.20,},
#        {'name':file_short_names[1],'weight':0.30,},
#        {'name':file_short_names[2],'weight':0.50,},
#       ]
#     }

# params_v44 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "asc\desc",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.44,
#       'asc'   : 0.54,                                   # LB=0.49499
#       'subwts': [+0.11, -0.04, -0.07],
#       'subm'  : [
#        {'name':file_short_names[0],'weight':0.30,},
#        {'name':file_short_names[1],'weight':0.20,},
#        {'name':file_short_names[2],'weight':0.50,},
#       ]
#     }

# path_to_ds ='/kaggle/input/20-juli-2025-flightrank/submission '

# file_short_names = ['0.48425','0.49068','0.49343', '0.49472']

# params_v46 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "asc\desc",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.30,                      # Lb=0.49407
#       'asc'   : 0.70,                                
#       'subwts': [+0.12, -0.01, -0.04, 0.07], # <- ERROR!!!
#       'subm'  : [
#        {'name':file_short_names[0],'weight':0.15,},
#        {'name':file_short_names[1],'weight':0.22,},
#        {'name':file_short_names[2],'weight':0.28,},
#        {'name':file_short_names[2],'weight':0.35,},
#       ]
#     }

# params_v47 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "asc\desc",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.40,                      # Lb=0.49426
#       'asc'   : 0.60,                                
#       'subwts': [+0.12, -0.01, -0.04, 0.07], # <- ERROR!!!
#       'subm'  : [
#        {'name':file_short_names[0],'weight':0.15,},
#        {'name':file_short_names[1],'weight':0.22,},
#        {'name':file_short_names[2],'weight':0.28,},
#        {'name':file_short_names[2],'weight':0.35,},
#       ]
#     }

# params_v48 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "asc\desc",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.50,                      # Lb=0.49398
#       'asc'   : 0.50,                                
#       'subwts': [+0.12, -0.01, -0.04, 0.07], # <- ERROR!!!
#       'subm'  : [
#        {'name':file_short_names[0],'weight':0.15,},
#        {'name':file_short_names[1],'weight':0.22,},
#        {'name':file_short_names[2],'weight':0.28,},
#        {'name':file_short_names[2],'weight':0.35,},
#       ]
#     }

# params_v49 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "asc\desc",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.60,                      # Lb=0.49435
#       'asc'   : 0.40,                                
#       'subwts': [+0.12, -0.01, -0.04, -0.07],
#       'subm'  : [
#        {'name':file_short_names[0],'weight':0.15,},
#        {'name':file_short_names[1],'weight':0.22,},
#        {'name':file_short_names[2],'weight':0.28,},
#        {'name':file_short_names[2],'weight':0.35,},
#       ]
#     }

# params_v50 = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "asc\desc",
#       'target': "selected",
#       'q_rows': 6_897_776,
#       'prefix': "subm_",
#       'desc'  : 0.70,                      # Lb=0.49527
#       'asc'   : 0.30,                                
#       'subwts': [+0.12, -0.01, -0.04, -0.07],
#       'subm'  : [
#        {'name':file_short_names[0],'weight':0.15,},
#        {'name':file_short_names[1],'weight':0.22,},
#        {'name':file_short_names[2],'weight':0.28,},
#        {'name':file_short_names[2],'weight':0.35,},
#       ]
#     }

# params = params_v11  #  LB=0.48590
# params = params_v12  #  LB=0.48507
# params = params_v13  #  LB=0.48452
# params = params_v14  #  LB=0.48590
# params = params_v15  #  LB=0.48608  < 
# params = params_v17  #  LB=0.48544
# params = params_v18  #  LB=0.48572
# params = params_v19  #  LB=0.48572
# params = params_v20  #  LB=0.48581
# params = params_v21  #  LB=0.48608  <

# params = params_v23  #  LB=0.49423  <
# params = params_v24  #  LB=0.49233
# params = params_v25  #  LB=0.49279

# params = params_v26  #  LB=0.49315
# params = params_v27  #  LB=0.49260
# params = params_v28  #  LB=0.49560 
# params = params_v29  #  LB=0.49591  < Top
# params = params_v30  #  LB=0.49563
# params = params_v31  #  LB=0.49481
# params = params_v32  #  LB=0.49187

# params = params_v35  #  LB=0.49554
# params = params_v36  #  LB=0.49591
# params = params_v37  #  LB=0.49591
# params = params_v39  #  LB=0.49591
# params = params_v40  #  LB=0.49472
# params = params_v41  #  LB=0.49591
# params = params_v42  #  LB=0.49279
# params = params_v43  #  LB=0.49426
# params = params_v44  #  LB=0.49499

# params = params_v46  #  LB=?        asc/desc weights = [ 30 x 70 ] ! err.subwts
# params = params_v47  #  LB=?        asc/desc weights = [ 40 x 60 ] ! err.subwts
# params = params_v48  #  LB=?        asc/desc weights = [ 50 x 50 ] ! err.subwts
# params = params_v49  #  LB=0.49435  asc/desc weights = [ 60 x 40 ]
# params = params_v50  #  LB=0.49527  asc/desc weights = [ 70 x 30 ]


path_to_ds ='/kaggle/input/20-juli-2025-flightrank/submission '

file_short_names = ['0.48425','0.48507', '0.49343', '0.49472', '0.49508', '0.49857']

params_v53 = {
      'path'  : path_to_ds,                                 
      'sort'  : "asc\desc",
      'target': "selected",            # LB=0.49885
      'q_rows': 6_897_776, 
      'prefix': "subm_",
      'desc'  : 0.70,
      'asc'   : 0.30,                                       
      'subwts': [+0.15, +0.08, -0.01, -0.04, -0.07, -0.11],
      'subm'  : [
       {'name':file_short_names[0],'weight':0.05,},
       {'name':file_short_names[1],'weight':0.05,},
       {'name':file_short_names[2],'weight':0.11,},
       {'name':file_short_names[3],'weight':0.12,},
       {'name':file_short_names[4],'weight':0.13,},
       {'name':file_short_names[5],'weight':0.54,},
      ]
    }


params_v54 = {
      'path'  : path_to_ds,                                 
      'sort'  : "asc\desc",
      'target': "selected",            # LB=0.49885   !   double
      'q_rows': 6_897_776, 
      'prefix': "subm_",
      'desc'  : 0.70,
      'asc'   : 0.30,                                       
      'subwts': [+0.15, +0.08, -0.01, -0.04, -0.07, -0.11],
      'subm'  : [
       {'name':file_short_names[0],'weight':0.05,},
       {'name':file_short_names[1],'weight':0.05,},
       {'name':file_short_names[2],'weight':0.11,},
       {'name':file_short_names[3],'weight':0.12,},
       {'name':file_short_names[4],'weight':0.13,},
       {'name':file_short_names[5],'weight':0.54,},
      ]
    }
                    #'0.48425',
file_short_names = [          '0.48507', '0.49343', '0.49472', '0.49508', '0.49857']

params_v55 = {
      'path'  : path_to_ds,                                 
      'sort'  : "asc\desc",
      'target': "selected",            # LB=0.50004
      'q_rows': 6_897_776, 
      'prefix': "subm_",
      'desc'  : 0.70,
      'asc'   : 0.30,                                       
      'subwts': [+0.14, +0.07, -0.03, -0.07, -0.11],
      'subm'  : [
       {'name':file_short_names[0],'weight':0.05,},
       {'name':file_short_names[1],'weight':0.11,},
       {'name':file_short_names[2],'weight':0.12,},
       {'name':file_short_names[3],'weight':0.13,},
       {'name':file_short_names[4],'weight':0.59,},
      ]
    }
                    #'0.48425','0.48507', 
file_short_names = [                    '0.49343', '0.49472', '0.49508', '0.49857']

params_v56 = {
      'path'  : path_to_ds,                                 
      'sort'  : "asc\desc",
      'target': "selected",            # LB=0.49490
      'q_rows': 6_897_776, 
      'prefix': "subm_",
      'desc'  : 0.70,
      'asc'   : 0.30,                                       
      'subwts': [+0.11, +0.02, -0.05, -0.08],
      'subm'  : [
       {'name':file_short_names[0],'weight':0.11,},
       {'name':file_short_names[1],'weight':0.12,},
       {'name':file_short_names[2],'weight':0.13,},
       {'name':file_short_names[2],'weight':0.64,},
      ]
    }

                    #'0.48425',           '0.49343', '0.49472',
file_short_names = [            '0.48507',                    '0.49508', '0.49857']

params_v57 = {
      'path'  : path_to_ds,                                 
      'sort'  : "asc\desc",
      'target': "selected",            # LB=?
      'q_rows': 6_897_776, 
      'prefix': "subm_",
      'desc'  : 0.70,
      'asc'   : 0.30,                                       
      'subwts': [+0.11, -0.04, -0.07],
      'subm'  : [
       {'name':file_short_names[1],'weight':0.07,},
       {'name':file_short_names[2],'weight':0.14,},
       {'name':file_short_names[2],'weight':0.79,},
      ]
    }


params = params_v57


%%time

# params = params_v53  Lb=0.49_885
# params = params_v55  Lb=0.50_004
# params = params_v56  Lb=0.49_490


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

