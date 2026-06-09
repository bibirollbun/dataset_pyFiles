import ast
import pandas as pd

from bokeh.plotting import figure, gridplot
from bokeh.io import output_file, show, output_notebook
output_notebook()


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
    submission_names = [one_dossier['name'] for one_dossier in dossiers]
    for one_dossier in dossiers: 
        i_col = 'alls.column ' + str(one_dossier['q_in'][i]['c'])
        qs = [one['q'] for one in one_dossier['q_in']]
        f1 = figure(x_range=submission_names,width=259, height=174, title=i_col)
        f1.vbar(x=submission_names, width=0.5, top=qs, color=colors)
        figures.append(f1)
        i+=1
    grid = gridplot([figures])
    output_file('tida_alls.html')
    show(grid)


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


path ='/kaggle/input/13-august-2025-ps-s5e8/'

file_short_names = ['subm C','subm D','subm E']
params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "y",                  # Lb = 0.97729
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [+0.10, -0.02, -0.08],                 
      'subm'   : [
         { 'name':file_short_names[0],'weight':0.33, },
         { 'name':file_short_names[1],'weight':0.34, },
         { 'name':file_short_names[2],'weight':0.33, },
      ]
    }

df = h_blend ( path, file_short_names, params )

bokeh_show(["purple","blue","red"])

df.to_csv('subm_y1.csv', index=False)
display(df.head(5))


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
         { 'name':file_short_names[0],'weight':0.20, },
         { 'name':file_short_names[1],'weight':0.30, },
         { 'name':file_short_names[2],'weight':0.50, },
      ]
    }

df = h_blend ( path, file_short_names, params )

bokeh_show(["maroon","orange","green"])

df.to_csv('subm_y2.csv', index=False)
display(df.head(5))


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

bokeh_show(["teal","darkslategray","lime"])

df.to_csv('subm_y3.csv', index=False)
display(df.head(5))


path ='/kaggle/working/'

# ------------------------------------------------------

# file_short_names = ['subm_y1','subm_y2','subm_y3']
# params = {
#       'path'   : path,
#       'id'     : 'id',                 
#       'target' : "y",#                               v.1      Lb = 0.97736
#       'desc'   : 0.70,
#       'asc'    : 0.30,
#       'subwts' : [+0.14, -0.03, -0.11],                 
#       'subm'   : [
#          { 'name':file_short_names[0],'weight':0.50, },
#          { 'name':file_short_names[1],'weight':0.25, },
#          { 'name':file_short_names[2],'weight':0.25, },
#       ]
#     }

# ------------------------------------------------------

# file_short_names = ['subm_y1','subm_y2','subm_y3']
# params = {
#       'path'   : path,
#       'id'     : 'id',                 
#       'target' : "y",#                               v.2      Lb = 0.97738
#       'desc'   : 0.70,
#       'asc'    : 0.30,
#       'subwts' : [+0.14, -0.03, -0.11],                 
#       'subm'   : [
#          { 'name':file_short_names[0],'weight':0.25, },
#          { 'name':file_short_names[1],'weight':0.50, },
#          { 'name':file_short_names[2],'weight':0.25, },
#       ]
#     }

# ------------------------------------------------------

# file_short_names = ['subm_y1','subm_y2','subm_y3']
# params = {
#       'path'   : path,
#       'id'     : 'id',                 
#       'target' : "y",#                               v.3      Lb = 0.97734
#       'desc'   : 0.70,
#       'asc'    : 0.30,
#       'subwts' : [+0.14, -0.03, -0.11],                 
#       'subm'   : [
#          { 'name':file_short_names[0],'weight':0.25, },
#          { 'name':file_short_names[1],'weight':0.25, },
#          { 'name':file_short_names[2],'weight':0.50, },
#       ]
#     }

# ------------------------------------------------------

file_short_names = ['subm_y1','subm_y2','subm_y3']
params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "y",#                               v.4      Lb = ?
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [+0.14, -0.03, -0.11],                 
      'subm'   : [
         { 'name':file_short_names[0],'weight':0.17, },
         { 'name':file_short_names[1],'weight':0.70, },
         { 'name':file_short_names[2],'weight':0.13, },
      ]
    }

# ------------------------------------------------------

df = h_blend ( path, file_short_names, params )

bokeh_show(["dimgray","navy","silver"])

df.to_csv('submission.csv', index=False)
display(df.head(5))

