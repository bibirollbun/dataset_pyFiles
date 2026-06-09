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
        
        weights1,corrects1  = [subm['weight'] for subm in sls["subm"]], [wt for wt in sls["subwts"] ]
        weights2,corrects2  = [subm['weight'] for subm in sls["subm2"]],[wt for wt in sls["subwts2"]]
        weights3,corrects3  = [subm['weight'] for subm in sls["subm3"]],[wt for wt in sls["subwts3"]]
        weights4,corrects4  = [subm['weight'] for subm in sls["subm4"]],[wt for wt in sls["subwts4"]]
        weights5,corrects5  = [subm['weight'] for subm in sls["subm5"]],[wt for wt in sls["subwts5"]]
        
        def alls(x, cs=cols):
            tes = {c: x[c] for c in cs}.items()
            subms_sorted = [
              t[0].replace(sls["prefix"], '')
              for t in sorted(tes,key=lambda k:k[1],reverse=True if sls["sort"]=='desc' else False)]
            return subms_sorted
        
        def correct(x, cs=cols, 
                    w1=weights1, cw1=corrects1, 
                    w2=weights2, cw2=corrects2,
                    w3=weights3, cw3=corrects3, 
                    w4=weights4, cw4=corrects4,
                    w5=weights5, cw5=corrects5
                   ):
            ic = [x['alls'].index(c) for c in short_name_cols]

            mxm = x['abs(mx-m)']

            if   0.00 < mxm <= 0.26:
                cS = [x[cols[j]] * (w1[j] + cw1[ic[j]]) for j in range(len(cols))]
            elif 0.26 < mxm <= 0.50:
                cS = [x[cols[j]] * (w2[j] + cw2[ic[j]]) for j in range(len(cols))]
            elif 0.50 < mxm <= 0.74:
                cS = [x[cols[j]] * (w3[j] + cw3[ic[j]]) for j in range(len(cols))]
            elif 0.74 < mxm <= 1.00:
                cS = [x[cols[j]] * (w4[j] + cw4[ic[j]]) for j in range(len(cols))]
            else:
                cS = [x[cols[j]] * (w5[j] + cw5[ic[j]]) for j in range(len(cols))]
            return sum(cS)

        def amxm(x, cs=cols):
            list_values = x[cs].to_list()
            mxm = abs(max(list_values)-min(list_values))
            return mxm

        df_subms['abs(mx-m)']   = df_subms.apply(lambda x: amxm   (x), axis=1)
        
        df_subms['alls']        = df_subms.apply(lambda x: alls   (x), axis=1)
        df_subms[sls["target"]] = df_subms.apply(lambda x: correct(x), axis=1)
        
        schema_rename = { old_nc:new_shnc for old_nc, new_shnc in zip(cols, short_name_cols) }
        
        df_subms = df_subms.rename(columns=schema_rename)
        df_subms = df_subms.rename(columns={sls["target"]:"ensemble"})
        
        df_subms.insert(loc=1, column=' _ ', value=['   '] * sls["q_rows"])
        
        df_subms[' _ '] = df_subms[' _ '].astype(str)
        pd.set_option('display.max_rows',100)
        pd.set_option('display.float_format', '{:.3f}'.format)
        vcols = ['ID'] + [' _ '] + short_name_cols + [' _ '] + ['abs(mx-m)'] + [' _ '] + ['alls'] + [' _ '] + ['ensemble']
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


# Archive

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
      'subwts': [+1.00, -0.40, -0.60],                          # LB = 0.93598
      'subm'  : [
        { 'name':file_short_names[0],'weight':0.40, },
        { 'name':file_short_names[1],'weight':0.60, },
        { 'name':file_short_names[2],'weight':1.00, },
      ]
    }


path_to_ds ='/kaggle/input/15-juli-2025-drw/submission '

file_short_names = ['0.88377','0.89178','0.90038']

params = {
      'path'  : path_to_ds,                                 
      'sort'  : "dynamic",
      'target': "prediction",
      'q_rows': 538_150,
      'prefix': "subm_",
      'desc'  : 0.45,
      'asc'   : 0.55,
      'subwts': [+0.55, -0.20, -0.35],                          # LB = 0.89864
      'subm'  : [
        { 'name':file_short_names[0],'weight':0.27, },
        { 'name':file_short_names[1],'weight':0.33, },
        { 'name':file_short_names[2],'weight':0.40, },
      ]
    }


path_to_ds ='/kaggle/input/15-juli-2025-drw/submission '

file_short_names = ['0.88377','0.89178','0.90038']

params = {
      'path'  : path_to_ds,                                 
      'sort'  : "dynamic",
      'target': "prediction",
      'q_rows': 538_150,
      'prefix': "subm_",
      'desc'  : 0.20,
      'asc'   : 0.80,
      'subwts': [+0.50, -0.20, -0.30],                          # LB = 0.90174
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
      'subwts': [+0.40, +0.05, -0.05,-0.15,-0.25],              # LB = 0.93794
      'subm'  : [
         { 'name':file_short_names[0],'weight':0.20, },
         { 'name':file_short_names[1],'weight':0.30, },
         { 'name':file_short_names[2],'weight':0.01, },
         { 'name':file_short_names[3],'weight':0.47, },
         { 'name':file_short_names[4],'weight':0.02, },
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
      'subwts': [+0.20, +0.10, -0.05,-0.10,-0.15],              # LB = 0.94828
      'subm'  : [
         { 'name':file_short_names[0],'weight':0.20, },
         { 'name':file_short_names[1],'weight':0.20, },
         { 'name':file_short_names[2],'weight':0.20, },
         { 'name':file_short_names[3],'weight':0.20, },
         { 'name':file_short_names[4],'weight':0.20, },
      ]
    }


params = {
      'path'  : path_to_ds,                                 
      'sort'  : "dynamic",
      'target': "prediction",
      'q_rows': 538_150,
      'prefix': "subm_",
      'desc'  : 0.30,
      'asc'   : 0.70,
      'subwts': [+0.20, +0.10, -0.05,-0.10,-0.15],              # LB = 0.94857 (only Horizont)
      'subm'  : [
         { 'name':file_short_names[0],'weight':0.20, },
         { 'name':file_short_names[1],'weight':0.20, },
         { 'name':file_short_names[2],'weight':0.21, },
         { 'name':file_short_names[3],'weight':0.22, },
         { 'name':file_short_names[4],'weight':0.23, },
      ],
      'subwts2': [+0.18, +0.09, -0.04,-0.09,-0.14],             # LB = 0.94859
      'subm2'  : [
         { 'name':file_short_names[0],'weight':0.19, },
         { 'name':file_short_names[1],'weight':0.20, },
         { 'name':file_short_names[2],'weight':0.21, },
         { 'name':file_short_names[3],'weight':0.22, },
         { 'name':file_short_names[4],'weight':0.23, },
      ]
    }


path_to_ds ='/kaggle/input/15-juli-2025-drw/submission '

file_short_names = ['0.83975','0.86767','0.88377','0.89178','0.90038']


# option.1 ------- ------- ------- ------- ------- ------- ------- LB = 0.94915

# params = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "dynamic",
#       'target': "prediction",
#       'q_rows': 538_150,
#       'prefix': "subm_",
#       'desc'  : 0.30,
#       'asc'   : 0.70,
#       'subwts': [+0.19, +0.10, -0.05,-0.10,-0.14],             
#       'subm'  : [
#          { 'name':file_short_names[0],'weight':0.19, },
#          { 'name':file_short_names[1],'weight':0.20, },
#          { 'name':file_short_names[2],'weight':0.21, },
#          { 'name':file_short_names[3],'weight':0.22, },
#          { 'name':file_short_names[4],'weight':0.23, },
#       ],
#       'subwts2': [+0.17, +0.09, -0.04,-0.09,-0.13],             
#       'subm2'  : [
#          { 'name':file_short_names[0],'weight':0.19, },
#          { 'name':file_short_names[1],'weight':0.20, },
#          { 'name':file_short_names[2],'weight':0.21, },
#          { 'name':file_short_names[3],'weight':0.22, },
#          { 'name':file_short_names[4],'weight':0.23, },
#       ],
#       'subwts3': [+0.15, +0.08, -0.03,-0.08,-0.12],             
#       'subm3'  : [
#          { 'name':file_short_names[0],'weight':0.19, },
#          { 'name':file_short_names[1],'weight':0.20, },
#          { 'name':file_short_names[2],'weight':0.21, },
#          { 'name':file_short_names[3],'weight':0.22, },
#          { 'name':file_short_names[4],'weight':0.23, },
#       ],
#       'subwts4': [+0.14, +0.07, -0.02,-0.07,-0.11],             
#       'subm4'  : [
#          { 'name':file_short_names[0],'weight':0.19, },
#          { 'name':file_short_names[1],'weight':0.20, },
#          { 'name':file_short_names[2],'weight':0.21, },
#          { 'name':file_short_names[3],'weight':0.22, },
#          { 'name':file_short_names[4],'weight':0.23, },
#       ],
#       'subwts5': [+0.11, +0.06, -0.01,-0.06,-0.10],             
#       'subm5'  : [
#          { 'name':file_short_names[0],'weight':0.19, },
#          { 'name':file_short_names[1],'weight':0.20, },
#          { 'name':file_short_names[2],'weight':0.21, },
#          { 'name':file_short_names[3],'weight':0.22, },
#          { 'name':file_short_names[4],'weight':0.23, },
#       ],
#     }

# ID	prediction
# 0	1	-0.1542508
# 1	2	 0.2589274
# 2	3	-1.4203646
# 3	4	-0.2054324
# 4	5	 0.1035272
# ...	...	...
# 538145	538146	-0.3137427
# 538146	538147	 0.1481161
# 538147	538148	-0.9100966
# 538148	538149	 0.7786920
# 538149	538150	-0.4061519
# 538150 rows × 2 columns


# option.2 ------- ------- ------- ------- ------- ------- ------- LB = 0.94889


# params = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "dynamic",
#       'target': "prediction",
#       'q_rows': 538_150,
#       'prefix': "subm_",
#       'desc'  : 0.30,
#       'asc'   : 0.70,
#       'subwts': [+0.19, +0.10, -0.05,-0.10,-0.14],             
#       'subm'  : [
#          { 'name':file_short_names[0],'weight':0.21, },
#          { 'name':file_short_names[1],'weight':0.22, },
#          { 'name':file_short_names[2],'weight':0.23, },
#          { 'name':file_short_names[3],'weight':0.24, },
#          { 'name':file_short_names[4],'weight':0.25, },
#       ],
#       'subwts2': [+0.17, +0.09, -0.04,-0.09,-0.13],             
#       'subm2'  : [
#          { 'name':file_short_names[0],'weight':0.22, },
#          { 'name':file_short_names[1],'weight':0.23, },
#          { 'name':file_short_names[2],'weight':0.24, },
#          { 'name':file_short_names[3],'weight':0.25, },
#          { 'name':file_short_names[4],'weight':0.26, },
#       ],
#       'subwts3': [+0.15, +0.08, -0.03,-0.08,-0.12],             
#       'subm3'  : [
#          { 'name':file_short_names[0],'weight':0.23, },
#          { 'name':file_short_names[1],'weight':0.24, },
#          { 'name':file_short_names[2],'weight':0.25, },
#          { 'name':file_short_names[3],'weight':0.26, },
#          { 'name':file_short_names[4],'weight':0.27, },
#       ],
#       'subwts4': [+0.14, +0.07, -0.02,-0.07,-0.11],             
#       'subm4'  : [
#          { 'name':file_short_names[0],'weight':0.24, },
#          { 'name':file_short_names[1],'weight':0.25, },
#          { 'name':file_short_names[2],'weight':0.26, },
#          { 'name':file_short_names[3],'weight':0.27, },
#          { 'name':file_short_names[4],'weight':0.28, },
#       ],
#       'subwts5': [+0.11, +0.06, -0.01,-0.06,-0.10],             
#       'subm5'  : [
#          { 'name':file_short_names[0],'weight':0.25, },
#          { 'name':file_short_names[1],'weight':0.26, },
#          { 'name':file_short_names[2],'weight':0.27, },
#          { 'name':file_short_names[3],'weight':0.28, },
#          { 'name':file_short_names[4],'weight':0.29, },
#       ],
#     }

# 	ID	prediction
# 0	1	-0.1673348
# 1	2	 0.2849654
# 2	3	-1.8152673
# 3	4	-0.2242138
# 4	5	 0.1223858
# ...	...	...
# 538145	538146	-0.3409893
# 538146	538147	 0.1720437
# 538147	538148	-1.1151354
# 538148	538149	 1.0286348
# 538149	538150	-0.4889277
# 538150 rows × 2 columns


# # option.3 ------- ------- ------- ------- ------- ------- ------- LB = 0.94717

# params = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "dynamic",
#       'target': "prediction",
#       'q_rows': 538_150,
#       'prefix': "subm_",
#       'desc'  : 0.30,
#       'asc'   : 0.70,
#       'subwts': [+0.19, +0.10, -0.05,-0.10,-0.14],             
#       'subm'  : [
#          { 'name':file_short_names[0],'weight':0.19, },
#          { 'name':file_short_names[1],'weight':0.20, },
#          { 'name':file_short_names[2],'weight':0.21, },
#          { 'name':file_short_names[3],'weight':0.22, },
#          { 'name':file_short_names[4],'weight':0.23, },
#       ],
#       'subwts2': [+0.17, +0.09, -0.04,-0.09,-0.13],             
#       'subm2'  : [
#          { 'name':file_short_names[0],'weight':0.18, },
#          { 'name':file_short_names[1],'weight':0.20, },
#          { 'name':file_short_names[2],'weight':0.21, },
#          { 'name':file_short_names[3],'weight':0.22, },
#          { 'name':file_short_names[4],'weight':0.24, },
#       ],
#       'subwts3': [+0.15, +0.08, -0.03,-0.08,-0.12],             
#       'subm3'  : [
#          { 'name':file_short_names[0],'weight':0.17, },
#          { 'name':file_short_names[1],'weight':0.20, },
#          { 'name':file_short_names[2],'weight':0.21, },
#          { 'name':file_short_names[3],'weight':0.22, },
#          { 'name':file_short_names[4],'weight':0.25, },
#       ],
#       'subwts4': [+0.14, +0.07, -0.02,-0.07,-0.11],             
#       'subm4'  : [
#          { 'name':file_short_names[0],'weight':0.16, },
#          { 'name':file_short_names[1],'weight':0.20, },
#          { 'name':file_short_names[2],'weight':0.21, },
#          { 'name':file_short_names[3],'weight':0.22, },
#          { 'name':file_short_names[4],'weight':0.26, },
#       ],
#       'subwts5': [+0.11, +0.06, -0.01,-0.06,-0.10],             
#       'subm5'  : [
#          { 'name':file_short_names[0],'weight':0.15, },
#          { 'name':file_short_names[1],'weight':0.20, },
#          { 'name':file_short_names[2],'weight':0.21, },
#          { 'name':file_short_names[3],'weight':0.22, },
#          { 'name':file_short_names[4],'weight':0.27, },
#       ],
#     }


# ID	prediction
# 0	1	-0.1542508
# 1	2	 0.2589274
# 2	3	-1.3990823
# 3	4	-0.2054324
# 4	5	 0.1073946
# ...	...	...
# 538145	538146	-0.3137427
# 538146	538147	 0.1482668
# 538147	538148	-0.8857567
# 538148	538149	 0.7211133
# 538149	538150	-0.4089252


# option.4 ------- ------- ------- ------- ------- ------- ------- LB = 0.94900

# params = {
#       'path'  : path_to_ds,                                 
#       'sort'  : "dynamic",
#       'target': "prediction",
#       'q_rows': 538_150,
#       'prefix': "subm_",
#       'desc'  : 0.26,
#       'asc'   : 0.74,
#       'subwts': [+0.19, +0.10, -0.05,-0.10,-0.14],             
#       'subm'  : [
#          { 'name':file_short_names[0],'weight':0.19, },
#          { 'name':file_short_names[1],'weight':0.20, },
#          { 'name':file_short_names[2],'weight':0.21, },
#          { 'name':file_short_names[3],'weight':0.22, },
#          { 'name':file_short_names[4],'weight':0.23, },
#       ],
#       'subwts2': [+0.17, +0.09, -0.04,-0.09,-0.13],             
#       'subm2'  : [
#          { 'name':file_short_names[0],'weight':0.19, },
#          { 'name':file_short_names[1],'weight':0.20, },
#          { 'name':file_short_names[2],'weight':0.21, },
#          { 'name':file_short_names[3],'weight':0.22, },
#          { 'name':file_short_names[4],'weight':0.225,},
#       ],
#       'subwts3': [+0.15, +0.08, -0.03,-0.08,-0.12],             
#       'subm3'  : [
#          { 'name':file_short_names[0],'weight':0.19, },
#          { 'name':file_short_names[1],'weight':0.20, },
#          { 'name':file_short_names[2],'weight':0.21, },
#          { 'name':file_short_names[3],'weight':0.22, },
#          { 'name':file_short_names[4],'weight':0.220,},
#       ],
#       'subwts4': [+0.14, +0.07, -0.02,-0.07,-0.11],             
#       'subm4'  : [
#          { 'name':file_short_names[0],'weight':0.19, },
#          { 'name':file_short_names[1],'weight':0.20, },
#          { 'name':file_short_names[2],'weight':0.21, },
#          { 'name':file_short_names[3],'weight':0.22, },
#          { 'name':file_short_names[4],'weight':0.215,},
#       ],
#       'subwts5': [+0.11, +0.06, -0.01,-0.06,-0.10],             
#       'subm5'  : [
#          { 'name':file_short_names[0],'weight':0.19, },
#          { 'name':file_short_names[1],'weight':0.20, },
#          { 'name':file_short_names[2],'weight':0.21, },
#          { 'name':file_short_names[3],'weight':0.22, },
#          { 'name':file_short_names[4],'weight':0.210,},
#       ],
#     }


# ID	prediction
# 0	1	-0.1575648
# 1	2	 0.2557592
# 2	3	-1.4114128
# 3	4	-0.2071729
# 4	5	 0.0966335
# ...	...	...
# 538145	538146	-0.3189531
# 538146	538147	 0.1430990
# 538147	538148	-0.9122206
# 538148	538149	 0.7518473
# 538149	538150	-0.4125638
# 538150 rows × 2 columns


# option.5 ------- ------- ------- ------- ------- ------- ------- LB = ?

params = {
      'path'  : path_to_ds,                                 
      'sort'  : "dynamic",
      'target': "prediction",
      'q_rows': 538_150,
      'prefix': "subm_",
      'desc'  : 0.35,
      'asc'   : 0.65,
      'subwts': [+0.19, +0.10, -0.05,-0.10,-0.14],             
      'subm'  : [
         { 'name':file_short_names[0],'weight':0.19, },
         { 'name':file_short_names[1],'weight':0.20, },
         { 'name':file_short_names[2],'weight':0.21, },
         { 'name':file_short_names[3],'weight':0.22, },
         { 'name':file_short_names[4],'weight':0.23, },
      ],
      'subwts2': [+0.14, +0.10, +0.05,-0.10,-0.19],             
      'subm2'  : [
         { 'name':file_short_names[0],'weight':0.19, },
         { 'name':file_short_names[1],'weight':0.20, },
         { 'name':file_short_names[2],'weight':0.21, },
         { 'name':file_short_names[3],'weight':0.22, },
         { 'name':file_short_names[4],'weight':0.23,},
      ],
      'subwts3': [+0.25, +0.25, 0, -0.25,-0.25],             
      'subm3'  : [
         { 'name':file_short_names[0],'weight':0.19, },
         { 'name':file_short_names[1],'weight':0.20, },
         { 'name':file_short_names[2],'weight':0.21, },
         { 'name':file_short_names[3],'weight':0.22, },
         { 'name':file_short_names[4],'weight':0.23,},
      ],
      'subwts4': [-0.05, -0.10, +0.30,-0.10,-0.05],             
      'subm4'  : [
         { 'name':file_short_names[0],'weight':0.19, },
         { 'name':file_short_names[1],'weight':0.20, },
         { 'name':file_short_names[2],'weight':0.21, },
         { 'name':file_short_names[3],'weight':0.22, },
         { 'name':file_short_names[4],'weight':0.23,},
      ],
      'subwts5': [+0.05, +0.10, -0.30, +0.10, +0.05],             
      'subm5'  : [
         { 'name':file_short_names[0],'weight':0.19, },
         { 'name':file_short_names[1],'weight':0.20, },
         { 'name':file_short_names[2],'weight':0.21, },
         { 'name':file_short_names[3],'weight':0.22, },
         { 'name':file_short_names[4],'weight':0.23,},
      ],
}

# 	ID	prediction
# 0	1	-0.1501083
# 1	2	 0.2628877
# 2	3	-1.4322685
# 3	4	-0.2032566
# 4	5	 0.1207345
# ...	...	...
# 538145	538146	-0.3072296
# 538146	538147	 0.1369802
# 538147	538148	-0.7683847
# 538148	538149	 0.9015924
# 538149	538150	-0.3144665
# 538150 rows × 2 columns



# option.1 LB=0.94915                  # option.2 LB=0.94889        # option.3 LB=0.94717

# ID	prediction                |    ID	 prediction         |    ID	    prediction

# 0	1	-0.1542508                |    0 1	 -0.1673348         |    0	1	-0.1542508
# 1	2	 0.2589274                |    1 2	  0.2849654         |    1	2	 0.2589274
# 2	3	-1.4203646                |    2 3	 -1.8152673         |    2	3	-1.3990823
# 3	4	-0.2054324                |    3 4	 -0.2242138         |    3	4	-0.2054324
# 4	5	 0.1035272                |    4 5	  0.1223858         |    4	5	 0.1073946
# ...	...	...
# 538145	538146	-0.3137427    |    538146	 -0.3409893     |    538145	538146	-0.3137427
# 538146	538147	 0.1481161    |    538147	  0.1720437     |    538146	538147	 0.1482668
# 538147	538148	-0.9100966    |    538148	 -1.1151354     |    538147	538148	-0.8857567
# 538148	538149	 0.7786920    |    538149	  1.0286348     |    538148	538149	 0.7211133
# 538149	538150	-0.4061519    |    538150	 -0.4889277     |    538149	538150	-0.4089252
# 538150 rows × 2 columns


# path_to_ds ='/kaggle/input/15-juli-2025-drw/submission '

# file_short_names = ['0.83975','0.86767','0.89178','0.90038']


# coming soon..


# path_to_ds ='/kaggle/input/15-juli-2025-drw/submission '

# file_short_names = ['0.83975','0.86767','0.88377','0.89178']


# coming soon..


df = iBlend ( path_to_ds, file_short_names, params )

df.to_csv('submission.csv', index=False)

display(df)

