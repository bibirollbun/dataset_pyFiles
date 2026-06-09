import ast
import pandas as pd

from bokeh.plotting import figure, gridplot
from bokeh.io import output_file, show, output_notebook
output_notebook()


# hv-blend + visualizations using Bokeh

def dossier(js,subms,cols):
    def quant(i,js,subms,cols):
        return {"c" : i, "q" : sum([1 for subm in cols[i] if subm == subms[js]])}
    return {
        'name' : subms[js],
        'q_in' : [quant(i,js,subms,cols) for i in range(len(subms))]
    }

def bokeh_show(colors):
    alls = pd.read_csv(f'tida_alls.csv')
    matrix = [ast.literal_eval(str(row.alls)) for row in alls.itertuples()]
    subms = sorted(matrix[0])
    cols = [[data[i] for data in matrix] for i in range(len(subms))]
    df_subms = pd.DataFrame({f'col_{i}': [x[i] for x in matrix] for i in range(len(subms))})
    dossiers = [dossier(js,subms,cols) for js in range(len(subms))]
    figures,i = [],0
    subm_names = [one_dossier['name'] for one_dossier in dossiers]
    for one_dossier in dossiers: 
        i_col = 'alls.column ' + str(one_dossier['q_in'][i]['c'])
        qs = [one['q'] for one in one_dossier['q_in']]
        if len(colors) == 5: subm_names = [ name.replace("subm_","") for name in subm_names]
        width = 259 if len(colors) == 3 else 174-13
        f = figure(x_range=subm_names,width=width, height=174, title=i_col)
        f.vbar(x=subm_names, width=0.5, top=qs, color=colors)
        figures.append(f)
        i+=1
    grid = gridplot([figures])
    output_file('tida_alls.html')
    show(grid)


def bokeh_show2(colors, blend_params):
    alls = pd.read_csv(f'tida_alls.csv')
    matrix = [ast.literal_eval(str(row.alls)) for row in alls.itertuples()]
    subms = sorted(matrix[0])
    cols = [[data[i] for data in matrix] for i in range(len(subms))]
    df_subms = pd.DataFrame({f'col_{i}': [x[i] for x in matrix] for i in range(len(subms))})
    dossiers = [dossier(js,subms,cols) for js in range(len(subms))]
    figures,i = [],0
    subm_names = [one_dossier['name'] for one_dossier in dossiers]
    for one_dossier in dossiers: 
        i_col = 'alls.column ' + str(one_dossier['q_in'][i]['c'])
        qs = [one['q'] for one in one_dossier['q_in']]
        if len(colors) == 5: subm_names = [ name.replace("subm_","") for name in subm_names]
        width = 259 if len(colors) == 3 else 174-13
        f = figure(x_range=subm_names,width=width, height=174, title=i_col)
        f.vbar(x=subm_names, width=0.5, top=qs, color=colors)
        figures.append(f)
        i+=1
    grid = gridplot([figures])
    output_file('tida_alls.html')
    show(grid)
    sub_wts = blend_params['subwts']
    main_wts = [subm['weight'] for subm in blend_params['subm']]
    acc_mass = []
    for j in range(len(dossiers)):
        one_dossier = dossiers[j]
        qs = [one['q'] for one in one_dossier['q_in']]
        mass = sum([qs[h] * (main_wts[j] + sub_wts[h]) for h in range(len(sub_wts))])
        acc_mass.append(round(mass,1))
    y_names = [name + " - " + str(mass) for name,mass in zip(subm_names,acc_mass)]
    f = figure(y_range=y_names, width=450, height=150)
    f.hbar(y=y_names, height=0.5, right=acc_mass, left=0, color=colors)
    output_file('tida_alls2.html')
    show(f)
    

def h_blend(path_to_ds, file_short_names, dk):
    def da(dk,sorting_direction):
        def read_subm(dk,i):
            tnm = dk["subm"][i]["name"]
            FiN = dk["path"] + tnm + ".csv"
            return pd.read_csv(FiN).rename(columns={'target':tnm, dk["target"]:tnm})
        dfs_subm = [read_subm(dk,i) for i in range(len(dk["subm"]))]
        df_subms = pd.merge(dfs_subm[0],  dfs_subm[1], on=[dk['id']])
        for i in range(2, len(dk["subm"])): 
            df_subms = pd.merge(df_subms, dfs_subm[i], on=[dk['id']])
        cols = [col for col in df_subms.columns if col != dk['id']]
        short_name_cols = [c for c in cols]
        corrects = [wt for wt in dk["subwts"]]
        weights = [subm['weight'] for subm in dk["subm"]]
        def alls(x, sd=sorting_direction,cs=cols):
            reverse = True if sd=='desc' else False
            tes = {c: x[c] for c in cs}.items()
            subms_sorted = [t[0] for t in sorted(tes,key=lambda _:_[1],reverse=reverse)]
            return subms_sorted
        def correct(x, cs=cols, w=weights, cw=corrects):
            ic = [x['alls'].index(c) for c in short_name_cols]
            cS = [x[cols[j]] * (w[j] + cw[ic[j]]) for j in range(len(cols))]
            return sum(cS)
        df_subms['alls']       = df_subms.apply(lambda x: alls   (x), axis=1)
        df_subms[dk["target"]] = df_subms.apply(lambda x: correct(x), axis=1)
        schema_rename = { old_nc:new_shnc for old_nc, new_shnc in zip(cols, short_name_cols) }
        df_subms = df_subms.rename(columns=schema_rename)
        df_subms = df_subms.rename(columns={dk["target"]:"ensemble"})
        df_subms.insert(loc=1, column=' _ ', value=['   '] * len(df_subms))
        df_subms[' _ '] = df_subms[' _ '].astype(str)
        pd.set_option('display.max_rows',8)
        pd.set_option('display.float_format', '{:.4f}'.format)
        vcols = [dk['id']]+[' _ '] + short_name_cols + [' _ ']+['alls']+[' _ ']+['ensemble']
        df_subms = df_subms[vcols]
        display(df_subms.head(5))
        pd.set_option('display.float_format', '{:.5f}'.format)
        df_subms = df_subms.rename(columns={"ensemble":dk["target"]})
        df_subms.to_csv(f'tida_alls.csv', index=False)
        return df_subms
        
    sample_subm = pd.read_csv(path_to_ds + file_short_names[1] + ".csv")

    def ensemble_da(dk,submission=sample_subm): 
        _id,target,d,a = dk['id'],dk['target'],dk['desc'],dk['asc']
        dfs = da(dk,'desc')
        dfD = dfs[[_id, target]]
        dfD.to_csv(f'tida_desc.csv', index=False)
        dfs = da(dk,'asc')
        dfA = dfs[[_id, target]]
        dfA.to_csv(f'tida_asc.csv',  index=False)
        submission[target] = dfD[target] * d + a * dfA[target]
        return submission
    
    return ensemble_da(dk)


# It's the same thing - development is happening in real time.
# these functions will be merged together

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
        ]
        
        def correct(x, cs=cols, wts=wts):
            i = [x['alls'].index(c) for c in short_name_cols]
            if   0.00 < x['mx-m'] <= 0.10: return summa(x,cs,wts[0],i)
            else:                          return summa(x,cs,wts[1],i)
                
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



%%time

path07 = '/kaggle/input/7-august-2025-ps-s5e8/submission '
path13 = '/kaggle/input/13-august-2025-ps-s5e8/'
path14 = '/kaggle/input/14-august-2025-ps-s5e8/'
path15 = '/kaggle/input/15-august-2025-ps-s5e8/'
path16 = '/kaggle/input/16-august-2025-ps-s5e8/'

subm_A = pd.read_csv(path13 + 'subm A.csv' )
subm_F = pd.read_csv(path14 + 'subm F.csv' )  # 'Group_1_1.csv'
#--------------------------------------------
subm_B = pd.read_csv(path13 + 'subm B.csv' )
subm_Q = pd.read_csv(path16 + 'subm Q.csv' )  # 'Group_1_2.csv'
#--------------------------------------------
subm_C = pd.read_csv(path13 + 'subm C.csv' )
subm_R = pd.read_csv(path07 + '0.97689.csv')  # 'Group_1_3.csv'
#================================================================
subm_D = pd.read_csv(path13 + 'subm C.csv' )
subm_G = pd.read_csv(path14 + 'subm G.csv' )  # 'Group_2_1.csv'
#--------------------------------------------
subm_E = pd.read_csv(path13 + 'subm E.csv' )
subm_H = pd.read_csv(path14 + 'subm H.csv' )  # 'Group_2_2.csv'
#--------------------------------------------
subm_I = pd.read_csv(path15 + 'subm I.csv' )
subm_T = pd.read_csv(path07 + '0.97677.csv')  # 'Group_2_3.csv'
#================================================================
subm_J = pd.read_csv(path16 + 'subm J.csv' )
subm_O = pd.read_csv(path16 + 'subm O.csv' )  # 'Group_3_1.csv'
#--------------------------------------------
subm_K = pd.read_csv(path16 + 'subm K.csv' )
subm_P = pd.read_csv(path16 + 'subm P.csv' )  # 'Group_3_2.csv'
#--------------------------------------------
subm_L = pd.read_csv(path16 + 'subm L.csv' )
subm_Q = pd.read_csv(path16 + 'subm Q.csv' )  # 'Group_3_3.csv'
#--------------------------------------------
subm_M = pd.read_csv(path16 + 'subm M.csv' )
subm_S = pd.read_csv(path16 + 'subm S.csv' )  # 'Group_3_4.csv'
#--------------------------------------------
subm_N = pd.read_csv(path16 + 'subm N.csv' )
subm_U = pd.read_csv(path07 + '0.97681.csv')  # 'Group_3_5.csv'

def straight_blend(df1,df2, file_name, wts=[0.50,0.50]):
    df1['y'] = df1['y'] *wts[0] + wts[1]* df2['y']
    df1.to_csv(file_name, index=False)
    #return df1

straight_blend(subm_A, subm_F, 'Group_1_1.csv')
straight_blend(subm_B, subm_Q, 'Group_1_2.csv')
straight_blend(subm_C, subm_R, 'Group_1_3.csv')
#================================================
straight_blend(subm_D, subm_G, 'Group_2_1.csv')
straight_blend(subm_E, subm_H, 'Group_2_2.csv')
straight_blend(subm_E, subm_H, 'Group_2_3.csv')
#================================================
straight_blend(subm_J, subm_O, 'Group_3_1.csv')
straight_blend(subm_K, subm_P, 'Group_3_2.csv')
straight_blend(subm_L, subm_Q, 'Group_3_3.csv')
straight_blend(subm_M, subm_S, 'Group_3_4.csv')
straight_blend(subm_N, subm_U, 'Group_3_5.csv')


# Part_3__Group_1

path ='/kaggle/working/'

file_short_names = ['Group_1_1','Group_1_2','Group_1_3']
params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "y",                  # Lb = ?
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [+0.21, -0.07, -0.14],                 
      'subm'   : [
         { 'name':file_short_names[0],'weight':0.30, },
         { 'name':file_short_names[1],'weight':0.40, },
         { 'name':file_short_names[2],'weight':0.30, },
      ]
    }

df = h_blend ( path, file_short_names, params )
bokeh_show2(["blue","mediumblue",'midnightblue'],params)
df.to_csv('subm_g1.csv', index=False)
display(df)


# Part_3__Group_2

path ='/kaggle/working/'

file_short_names = ['Group_2_1','Group_2_2','Group_2_3']
params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "y",                  # Lb = ?
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [+0.21, -0.07, -0.14],                 
      'subm'   : [
         { 'name':file_short_names[0],'weight':0.20, },
         { 'name':file_short_names[1],'weight':0.60, },
         { 'name':file_short_names[2],'weight':0.20, },
      ]
    }

df = h_blend ( path, file_short_names, params )
bokeh_show2(["limegreen","forestgreen","darkgreen"],params)
df.to_csv('subm_g2.csv', index=False)
display(df)


# Part_3__Group_3

path ='/kaggle/working/'

file_short_names = ['Group_3_1','Group_3_2','Group_3_3','Group_3_4','Group_3_5']
params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "y",#                                Lb = ?
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [+0.21, +0.10, -0.04, -0.10, -0.17],                 
      'subm'   : [
         { 'name':file_short_names[0],'weight':0.20, },
         { 'name':file_short_names[1],'weight':0.20, },
         { 'name':file_short_names[2],'weight':0.20, },
         { 'name':file_short_names[3],'weight':0.20, },
         { 'name':file_short_names[4],'weight':0.20, },
      ]
    }

df = h_blend ( path, file_short_names, params )
bokeh_show2(["lightcoral","red","crimson","brown","sienna"], params)
df.to_csv('subm_g3.csv', index=False)
display(df)


path ='/kaggle/working/'

# file_short_names = ['subm_g1','subm_g2','subm_g3']
# params = {
#       'path'   : path,
#       'id'     : 'id',                 
#       'target' : "y",#                                 v.48  Lb = 0.97737
#       'desc'   : 0.70,
#       'asc'    : 0.30,
#       'subwts' : [+0.21, -0.07, -0.14],                 
#       'subm'   : [
#          { 'name':file_short_names[0],'weight':0.20, },
#          { 'name':file_short_names[1],'weight':0.65, },
#          { 'name':file_short_names[2],'weight':0.15, },
#       ]
#     }

# file_short_names = ['subm_g1','subm_g2','subm_g3']
# params = {
#       'path'   : path,
#       'id'     : 'id',                 
#       'target' : "y",#                                 v.49  Lb = 0.97739
#       'desc'   : 0.70,
#       'asc'    : 0.30,
#       'subwts' : [+0.21, -0.07, -0.14],                 
#       'subm'   : [
#          { 'name':file_short_names[0],'weight':0.10, },
#          { 'name':file_short_names[1],'weight':0.74, },
#          { 'name':file_short_names[2],'weight':0.16, },
#       ]
#     }

file_short_names = ['subm_g1','subm_g2','subm_g3']
params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "y",#                                 v.50  Lb = ?
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [+0.21, -0.07, -0.14],                 
      'subm'   : [
         { 'name':file_short_names[0],'weight':0.05, },
         { 'name':file_short_names[1],'weight':0.77, },
         { 'name':file_short_names[2],'weight':0.18, },
      ]
    }


df = h_blend ( path, file_short_names, params )
bokeh_show2(["darkblue","darkgreen","crimson"],params)
df.to_csv('submission.csv', index=False)
display(df)


# ===================== Part.1,2 ==========================

path ='/kaggle/input/13-august-2025-ps-s5e8/'

file_short_names = ['subm C','subm D','subm E']
params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "y",                  # Lb = 0.97729
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [+0.21, -0.07, -0.14],                 
      'subm'   : [
         { 'name':file_short_names[0],'weight':0.20, },
         { 'name':file_short_names[1],'weight':0.30, },
         { 'name':file_short_names[2],'weight':0.50, },
      ]
    }

df = h_blend ( path, file_short_names, params )
bokeh_show2(["blue","mediumblue",'midnightblue'],params)
df.to_csv('subm_y1.csv', index=False)
display(df)


# ===================== Part.1,2 ==========================

path ='/kaggle/input/14-august-2025-ps-s5e8/'

file_short_names = ['subm F','subm G','subm H']
params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "y",                  # Lb = no data
      'desc'   : 0.30,
      'asc'    : 0.70,
      'subwts' : [+0.10, -0.02, -0.08],                 
      'subm'   : [
         { 'name':file_short_names[0],'weight':0.15, },
         { 'name':file_short_names[1],'weight':0.25, },
         { 'name':file_short_names[2],'weight':0.60, },
      ]
    }

df = h_blend ( path, file_short_names, params )
bokeh_show2(["limegreen","forestgreen","darkgreen"],params)
df.to_csv('subm_y2.csv', index=False)
display(df)


# ===================== Part.1 ==========================

path ='/kaggle/input/15-august-2025-ps-s5e8/'

file_short_names = ['subm I','subm J','subm K']
params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "y",                  # Lb = no data
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [+0.10, -0.02, -0.08],                 
      'subm'   : [
         { 'name':file_short_names[0],'weight':0.30, },
         { 'name':file_short_names[1],'weight':0.40, },
         { 'name':file_short_names[2],'weight':0.30, },
      ]
    }

df = h_blend ( path, file_short_names, params )
bokeh_show2(["crimson","red","lightcoral"],params)
df.to_csv('subm_y3.csv', index=False)
display(df)


# ===================== Part.2 ==========================

path ='/kaggle/input/16-august-2025-ps-s5e8/'

file_short_names = ['subm J','subm M','subm P']
params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "y",                  # Lb = no data
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [+0.21, -0.07, -0.14],                 
      'subm'   : [
         { 'name':file_short_names[0],'weight':0.33, },
         { 'name':file_short_names[1],'weight':0.34, },
         { 'name':file_short_names[2],'weight':0.33, },
      ]
    }

df = h_blend ( path, file_short_names, params )
bokeh_show2 (["red","crimson","lightcoral"], params )
df.to_csv('subm_y3.csv', index=False)
display(df)


# ===================== Part.1 ==========================

path = '/kaggle/input/16-august-2025-ps-s5e8/'

file_short_names = ['subm L','subm M','subm N']
params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "y",                  # Lb = no data
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [+0.11, -0.03, -0.08],                 
      'subm'   : [
         { 'name':file_short_names[0],'weight':0.33, },
         { 'name':file_short_names[1],'weight':0.34, },
         { 'name':file_short_names[2],'weight':0.33, },
      ]
    }

df = h_blend ( path, file_short_names, params )
bokeh_show(["maroon","chocolate","sienna"])
df.to_csv('subm_y4.csv', index=False)
display(df)


# ===================== Part.2 ==========================

path = '/kaggle/input/16-august-2025-ps-s5e8/'

file_short_names = ['subm K','subm I','subm L','subm O']
params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "y",                  # Lb = no data
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [+0.42, -0.04, -0.14, -0.24],                 
      'subm'   : [
         { 'name':file_short_names[0],'weight':0.25, },
         { 'name':file_short_names[1],'weight':0.25, },
         { 'name':file_short_names[2],'weight':0.25, },
         { 'name':file_short_names[3],'weight':0.25, },
      ]
    }

df = h_blend ( path, file_short_names, params )
bokeh_show2(["maroon","sienna","chocolate","sandybrown"], params) 
df.to_csv('subm_y4.csv', index=False)
display(df)


# ===================== Part.1 ==========================

path = '/kaggle/input/16-august-2025-ps-s5e8/'

file_short_names = ['subm O','subm P','subm S']
params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "y",                  # Lb = no data
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [+0.11, -0.03, -0.08],                 
      'subm'   : [
         { 'name':file_short_names[0],'weight':0.33, },
         { 'name':file_short_names[1],'weight':0.34, },
         { 'name':file_short_names[2],'weight':0.33, },
      ]
    }

df = h_blend ( path, file_short_names, params )

bokeh_show(["darkmagenta","magenta","mediumorchid"])

df.to_csv('subm_y5.csv', index=False)
display(df)


# Archive  Part.1 

# path ='/kaggle/working/'

# file_short_names = ['subm_y1','subm_y2','subm_y3']
# params = {
#       'path'   : path,
#       'id'     : 'id',                 
#       'target' : "y",#                                                 Lb = 0.97736
#       'desc'   : 0.70,
#       'asc'    : 0.30,
#       'subwts' : [+0.14, -0.03, -0.11],                 
#       'subm'   : [
#          { 'name':file_short_names[0],'weight':0.50, },
#          { 'name':file_short_names[1],'weight':0.25, },
#          { 'name':file_short_names[2],'weight':0.25, },
#       ]
#     }

# file_short_names = ['subm_y1','subm_y2','subm_y3']
# params = {
#       'path'   : path,
#       'id'     : 'id',                 
#       'target' : "y",#                                                 Lb = 0.97738
#       'desc'   : 0.70,
#       'asc'    : 0.30,
#       'subwts' : [+0.14, -0.03, -0.11],                 
#       'subm'   : [
#          { 'name':file_short_names[0],'weight':0.25, },
#          { 'name':file_short_names[1],'weight':0.50, },
#          { 'name':file_short_names[2],'weight':0.25, },
#       ]
#     }

# file_short_names = ['subm_y1','subm_y2','subm_y3']
# params = {
#       'path'   : path,
#       'id'     : 'id',                 
#       'target' : "y",#                                                 Lb = 0.97734
#       'desc'   : 0.70,
#       'asc'    : 0.30,
#       'subwts' : [+0.14, -0.03, -0.11],                 
#       'subm'   : [
#          { 'name':file_short_names[0],'weight':0.25, },
#          { 'name':file_short_names[1],'weight':0.25, },
#          { 'name':file_short_names[2],'weight':0.50, },
#       ]
#     }

# file_short_names = ['subm_y1','subm_y2','subm_y3']
# params = {
#       'path'   : path,
#       'id'     : 'id',                 
#       'target' : "y",#                                                 Lb = 0.97738
#       'desc'   : 0.70,
#       'asc'    : 0.30,
#       'subwts' : [+0.14, -0.03, -0.11],                 
#       'subm'   : [
#          { 'name':file_short_names[0],'weight':0.17, },
#          { 'name':file_short_names[1],'weight':0.70, },
#          { 'name':file_short_names[2],'weight':0.13, },
#       ]
#     }

# file_short_names = ['subm_y1','subm_y2','subm_y3','subm_y4','subm_y5']
# params = {
#       'path'   : path,
#       'id'     : 'id',                 
#       'target' : "y",#                                                 Lb = 0.97732
#       'desc'   : 0.70,
#       'asc'    : 0.30,
#       'subwts' : [+0.07, +0.04, -0.01, -0.03, -0.07],                 
#       'subm'   : [
#          { 'name':file_short_names[0],'weight':0.20, },
#          { 'name':file_short_names[1],'weight':0.20, },
#          { 'name':file_short_names[2],'weight':0.20, },
#          { 'name':file_short_names[3],'weight':0.20, },
#          { 'name':file_short_names[4],'weight':0.20, },
#       ]
#     }

# file_short_names = ['subm_y1','subm_y2','subm_y3','subm_y4','subm_y5']
# params = {
#       'path'   : path,
#       'id'     : 'id',                 
#       'target' : "y",#                                         v39     Lb = 0.97737
#       'desc'   : 0.70,
#       'asc'    : 0.30,
#       'subwts' : [+0.07, +0.04, -0.01, -0.03, -0.07],                 
#       'subm'   : [
#          { 'name':file_short_names[0],'weight':0.41, },
#          { 'name':file_short_names[1],'weight':0.38, },
#          { 'name':file_short_names[2],'weight':0.07, },
#          { 'name':file_short_names[3],'weight':0.07, },
#          { 'name':file_short_names[4],'weight':0.07, },
#       ]
#     }

# file_short_names = ['subm_y1','subm_y2','subm_y3','subm_y4','subm_y5']
# params = {
#       'path'   : path,
#       'id'     : 'id',                 
#       'target' : "y",#                                         v40     Lb = 0.97734
#       'desc'   : 0.70,
#       'asc'    : 0.30,
#       'subwts' : [+0.07, +0.04, -0.01, -0.03, -0.07],                 
#       'subm'   : [
#          { 'name':file_short_names[0],'weight':0.70, },
#          { 'name':file_short_names[1],'weight':0.21, },
#          { 'name':file_short_names[2],'weight':0.05, },
#          { 'name':file_short_names[3],'weight':0.03, },
#          { 'name':file_short_names[4],'weight':0.01, },
#       ]
#     }


# ===================== Part.1 ==========================

path ='/kaggle/working/'

file_short_names = ['subm_y1','subm_y2','subm_y3','subm_y4','subm_y5']
params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "y",#                                         v41     Lb = 0.97738
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [+0.07, +0.04, -0.01, -0.03, -0.07],                 
      'subm'   : [
         { 'name':file_short_names[0],'weight':0.21, },
         { 'name':file_short_names[1],'weight':0.70, },
         { 'name':file_short_names[2],'weight':0.05, },
         { 'name':file_short_names[3],'weight':0.03, },
         { 'name':file_short_names[4],'weight':0.01, },
      ]
    }

df = h_blend ( path, file_short_names, params )
bokeh_show(["darkblue","darkgreen","crimson","sienna","darkmagenta"])
df.to_csv('submission_part_1.csv', index=False)
display(df)


# ===================== Part.2 ==========================

path ='/kaggle/working/'

file_short_names = ['subm_y1','subm_y2','subm_y3','subm_y4']

# params = {
#       'path'   : path,
#       'id'     : 'id',                 
#       'target' : "y",#                                         v43     Lb = 0.97738  
#       'desc'   : 0.70,
#       'asc'    : 0.30,
#       'subwts' : [+0.13, -0.01, -0.04, -0.08],                 
#       'subm'   : [
#          { 'name':file_short_names[0],'weight':0.10, },
#          { 'name':file_short_names[1],'weight':0.70, },
#          { 'name':file_short_names[2],'weight':0.10, },
#          { 'name':file_short_names[3],'weight':0.10, },
#       ]
#     }

# params = {
#       'path'   : path,
#       'id'     : 'id',                 
#       'target' : "y",#                                          v44     Lb = 0.97739
#       'desc'   : 0.65,
#       'asc'    : 0.35,
#       'subwts' : [+0.18, -0.02, -0.05, -0.11],     
#       'subm'   : [
#          { 'name':file_short_names[0],'weight':0.07, },
#          { 'name':file_short_names[1],'weight':0.79, },
#          { 'name':file_short_names[2],'weight':0.07, },
#          { 'name':file_short_names[3],'weight':0.07, },
#       ]
#     }

params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "y",#                                          v45     Lb = ?
      'desc'   : 0.60,
      'asc'    : 0.40,
      'subwts' : [+0.21, -0.02, -0.07, -0.12],     
      'subm'   : [
         { 'name':file_short_names[0],'weight':0.05, },
         { 'name':file_short_names[1],'weight':0.85, },
         { 'name':file_short_names[2],'weight':0.05, },
         { 'name':file_short_names[3],'weight':0.05, },
      ]
    }

df = h_blend ( path, file_short_names, params )
bokeh_show2 (["darkblue","darkgreen","crimson","sienna"],params)
df.to_csv('submission_part_2.csv', index=False)
display(df)

