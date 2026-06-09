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
        pd.set_option('display.float_format', '{:.4f}'.format)
        vcols = ['ID'] + [' _ '] + short_name_cols + [' _ '] + ['alls'] + [' _ '] + ['ensemble']
        df_subms = df_subms[vcols]
        display(df_subms.head(7))
        pd.set_option('display.float_format', '{:.7f}'.format)
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



path_to_ds ='/kaggle/input/24-juli-2025-drw/submission '

file_short_names = ['0.95167','0.95166','0.95165','0.95164','0.95163']
params = {
      'path'   : path_to_ds,                                 
      'sort'   : "asc/desc",
      'target' : "prediction",
      'q_rows' : 538_150,
      'prefix' : "subm_",
      'desc'   : 0.700003,
      'asc'    : 0.300007,
    
      'subwts' : [+0.000004, +0.000002, -0.000001, -0.000002, -0.000003], # 0.95167
      'subm'   : [
         { 'name':file_short_names[0],'weight':0.99995, },
         { 'name':file_short_names[1],'weight':0.00002, },
         { 'name':file_short_names[2],'weight':0.00001, },
         { 'name':file_short_names[3],'weight':0.00001, },
         { 'name':file_short_names[4],'weight':0.00001, },
      ],
    }


file_short_names = ['0.95167','0.95166','0.95165','0.95164','0.95163']
params = {
      'path'   : path_to_ds,                                 
      'sort'   : "asc/desc",
      'target' : "prediction",
      'q_rows' : 538_150,
      'prefix' : "subm_",
      'desc'   : 0.20,
      'asc'    : 0.50,
    
      'subwts' : [+0.01, +0.02, +0.03, +0.04, +0.05],                    # 0.92359
      'subm'   : [
         { 'name':file_short_names[0],'weight':0.20, },
         { 'name':file_short_names[1],'weight':0.20, },
         { 'name':file_short_names[2],'weight':0.20, },
         { 'name':file_short_names[3],'weight':0.20, },
         { 'name':file_short_names[4],'weight':0.20, },
      ],
    }


file_short_names = ['0.95167','0.95166','0.95165','0.95164','0.95163']
params = {
      'path'   : path_to_ds,                                 
      'sort'   : "asc/desc",
      'target' : "prediction",
      'q_rows' : 538_150,
      'prefix' : "subm_",
      'desc'   : 0.50,
      'asc'    : 0.50,
    
      'subwts' : [+0.01, +0.02, +0.03, +0.04, +0.05],                    # 0.92358
      'subm'   : [
         { 'name':file_short_names[0],'weight':0.20, },
         { 'name':file_short_names[1],'weight':0.20, },
         { 'name':file_short_names[2],'weight':0.20, },
         { 'name':file_short_names[3],'weight':0.20, },
         { 'name':file_short_names[4],'weight':0.20, },
      ],
    }

file_short_names = ['0.95167','0.95166','0.95165','0.95164','0.95163']
params = {
      'path'   : path_to_ds,                                 
      'sort'   : "asc/desc",
      'target' : "prediction",
      'q_rows' : 538_150,
      'prefix' : "subm_",
      'desc'   : 0.50,
      'asc'    : 0.50,
    
      'subwts' : [-0.01, -0.02, -0.03, -0.04, -0.05],                    # 0.92358
      'subm'   : [
         { 'name':file_short_names[0],'weight':0.20, },
         { 'name':file_short_names[1],'weight':0.20, },
         { 'name':file_short_names[2],'weight':0.20, },
         { 'name':file_short_names[3],'weight':0.20, },
         { 'name':file_short_names[4],'weight':0.20, },
      ],
    }

file_short_names = ['0.90006','0.95109','0.95129','0.95164','0.95167']
params = {
      'path'   : path_to_ds,                                 
      'sort'   : "asc/desc",
      'target' : "prediction",
      'q_rows' : 538_150,
      'prefix' : "subm_",
      'desc'   : 0.30,
      'asc'    : 0.70,
    
      'subwts' : [+0.03, +0.01, 0.0, -0.01, -0.03], 
      'subm'   : [
         { 'name':file_short_names[0],'weight':0.01, },
         { 'name':file_short_names[1],'weight':0.02, },
         { 'name':file_short_names[2],'weight':0.03, },
         { 'name':file_short_names[3],'weight':0.04, },
         { 'name':file_short_names[4],'weight':0.90, },
      ],
    }

file_short_names = ['0.90006','0.95129','0.95164','0.95167'] #'0.95109',
params = {
      'path'   : path_to_ds,                                 
      'sort'   : "asc/desc",
      'target' : "prediction",
      'q_rows' : 538_150,
      'prefix' : "subm_",
      'desc'   : 0.29,
      'asc'    : 0.71,
    
      'subwts' : [+0.03, -0.005, 0.010, -0.015], #, -0.03], 
      'subm'   : [
         { 'name':file_short_names[0],'weight':0.01 },
         { 'name':file_short_names[1],'weight':0.02 },
         { 'name':file_short_names[2],'weight':0.03 },
         { 'name':file_short_names[3],'weight':0.94 },
         #{ 'name':file_short_names[4],'weight':0.880, },
      ],
    }


df = iBlend ( path_to_ds, file_short_names, params )

df.to_csv('submission.csv', index=False)

display(df)

