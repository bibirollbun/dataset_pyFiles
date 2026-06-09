import pandas as pd


def iBlend(path_to_ds, file_short_names, sls):

    def tida(sls):
        
        def read_subm(sls,i):
            tnm = sls["subm"][i]["name"]
            FiN = sls["path"] + tnm + ".csv"
            return pd.read_csv(FiN).rename(columns={'target':tnm, sls["target"]:tnm})
        
        dfs_subm = [read_subm(sls,i) for i in range(len(sls["subm"]))]
        df_subms = pd.merge(dfs_subm[0],  dfs_subm[1], on=['ID'])
        
        for i in range(2, len(sls["subm"])): 
            df_subms = pd.merge(df_subms, dfs_subm[i], on=['ID'])
            
        cols = [col for col in df_subms.columns if col != "ID"]
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
        pd.set_option('display.float_format', '{:.7f}'.format)
        vcols = ['ID'] + [' _ '] + short_name_cols + [' _ '] + ['alls'] + [' _ '] + ['ensemble']
        df_subms = df_subms[vcols]
        display(df_subms.head(7))
        pd.set_option('display.float_format', '{:.11f}'.format)
        df_subms = df_subms.rename(columns={"ensemble":sls["target"]})
        
        return df_subms
        

    sample_subm = pd.read_csv(path_to_ds + file_short_names[1] + ".csv")

    
    def ensemble_tida(sls,submission=sample_subm):   
        sls['sort'] = 'desc'
        dfs = tida(sls)
        dfD = dfs[['ID', sls['target']]]
        dfD.to_csv(f'tida_desc.csv', index=False)
        sls['sort'] = 'asc'
        dfs = tida(sls)
        dfA = dfs[['ID', sls['target']]]
        dfA.to_csv(f'tida_asc.csv',  index=False)
        target,d,a = sls['target'],sls['desc'],sls['asc']
        submission[target] = dfD[target] * d + a * dfA[target]
        return submission

    submission = ensemble_tida(sls)
    
    return submission


path_to_ds ='/kaggle/input/13-juli-2025-drw/submission '

file_short_names = ['0.73799','0.72837','0.70871']

params = {
      'path'  : path_to_ds,                                 
      'sort'  : "dynamic",
      'target': "prediction",
      'q_rows': 538_150,
      'prefix': "subm_",
      'desc'  : 0.30,
      'asc'   : 0.70,
      'subwts': [+0.40, -0.10, -0.30],                          # LB = 0.80384,  v.8
      'subm'  : [
        { 'name':file_short_names[0],'weight':0.37, },
        { 'name':file_short_names[1],'weight':0.34, },
        { 'name':file_short_names[2],'weight':0.29, },
      ]
    }

path_to_ds ='/kaggle/input/13-juli-2025-drw/submission '

file_short_names = ['0.73799','0.72837','0.70871']

params = {
      'path'  : path_to_ds,                                 
      'sort'  : "dynamic",
      'target': "prediction",
      'q_rows': 538_150,
      'prefix': "subm_",
      'desc'  : 0.20,
      'asc'   : 0.80,
      'subwts': [+0.50, -0.15, -0.35],                          # LB = 0.78384,  v.9
      'subm'  : [
        { 'name':file_short_names[0],'weight':0.38, },
        { 'name':file_short_names[1],'weight':0.34, },
        { 'name':file_short_names[2],'weight':0.28, },
      ]
    }


path_to_ds ='/kaggle/input/13-juli-2025-drw/submission '

file_short_names = ['0.73799','0.72837','0.70871']

params = {
      'path'  : path_to_ds,                                 
      'sort'  : "dynamic",
      'target': "prediction",
      'q_rows': 538_150,
      'prefix': "subm_",
      'desc'  : 0.33,
      'asc'   : 0.67,
      'subwts': [+0.40, -0.10, -0.30],                          # LB = 0.80925,  v.10
      'subm'  : [
        { 'name':file_short_names[0],'weight':0.34, },
        { 'name':file_short_names[1],'weight':0.33, },
        { 'name':file_short_names[2],'weight':0.33, },
      ]
    }


path_to_ds ='/kaggle/input/13-juli-2025-drw/submission '

file_short_names = ['0.73799','0.72837','0.70871','0.81760']

params = {
      'path'  : path_to_ds,                                 
      'sort'  : "dynamic",
      'target': "prediction",
      'q_rows': 538_150,
      'prefix': "subm_",
      'desc'  : 0.33,
      'asc'   : 0.67,
      'subwts': [+1.00, -0.20, -0.30, -0.50],                   # LB = 0.82968,  v.11
      'subm'  : [
        { 'name':file_short_names[0],'weight':0.34, },
        { 'name':file_short_names[1],'weight':0.33, },
        { 'name':file_short_names[2],'weight':0.33, },
        { 'name':file_short_names[3],'weight':1.00, },
      ]
    }


path_to_ds ='/kaggle/input/15-juli-2025-drw/submission '

file_short_names = ['0.83975','0.86767','0.89178']

params = {
      'path'  : path_to_ds,                                 
      'sort'  : "dynamic",
      'target': "prediction",
      'q_rows': 538_150,
      'prefix': "subm_",
      'desc'  : 0.30,
      'asc'   : 0.70,
      'subwts': [+0.40, -0.10, -0.30],                          # LB = 0.93980
      'subm'  : [
        { 'name':file_short_names[0],'weight':0.20, },
        { 'name':file_short_names[1],'weight':0.30, },
        { 'name':file_short_names[2],'weight':0.50, },
      ]
    }


path_to_ds ='/kaggle/input/15-juli-2025-drw/submission '

file_short_names = ['0.83975','0.86767','0.88377','0.89178','0.90038']

params = {
      'path'  : path_to_ds,                                 
      'sort'  : "dynamic",
      'target': "prediction",
      'q_rows': 538_150,
      'prefix': "subm_",
      'desc'  : 0.30,
      'asc'   : 0.70,
      'subwts': [+0.20, +0.10, -0.05,-0.10,-0.15],              # LB = ?
      'subm'  : [
         { 'name':file_short_names[0],'weight':0.20, },
         { 'name':file_short_names[1],'weight':0.20, },
         { 'name':file_short_names[2],'weight':0.21, },
         { 'name':file_short_names[3],'weight':0.22, },
         { 'name':file_short_names[4],'weight':0.23, },
      ]
    }


df = iBlend ( path_to_ds, file_short_names, params )

df.to_csv('submission.csv', index=False)

display(df)

