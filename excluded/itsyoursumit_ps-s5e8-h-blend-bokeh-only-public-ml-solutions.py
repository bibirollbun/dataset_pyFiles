import ast
import numpy as np
import pandas as pd

from bokeh.plotting import figure, gridplot
from bokeh.io import output_file, show, output_notebook
output_notebook()


def bokeh_show(params, colors, show_figures1):
    def dossier(js,subms,cols):
        def quant(i,js,subms,cols):
            return {"c" : i, "q" : sum([1 for subm in cols[i] if subm == subms[js]])}
        return {
            'name' : subms[js],
            'q_in' : [quant(i,js,subms,cols) for i in range(len(subms))]
        }
    alls = pd.read_csv(f'tida_alls.csv')
    matrix = [ast.literal_eval(str(row.alls)) for row in alls.itertuples()]
    subms = sorted(matrix[0])
    cols = [[data[i] for data in matrix] for i in range(len(subms))]
    df_subms = pd.DataFrame({f'col_{i}': [x[i] for x in matrix] for i in range(len(subms))})
    dossiers = [dossier(js,subms,cols) for js in range(len(subms))]
    subm_names = [one_dossier['name'] for one_dossier in dossiers]
    figures1,qss,i = [],[],0
    for one_dossier in dossiers: 
        i_col = 'alls. ' + str(one_dossier['q_in'][i]['c'])
        qs = [one['q'] for one in one_dossier['q_in']]
        x_names = [name.replace("Group","").replace("subm_","") for name in subm_names]
        width = 151  if len(colors) == 5\
            else (121 if len(colors) == 8\
            else (131 if len(colors)  == 9\
            else (141 if len(colors)  == 10 else 185)))
        f = figure(x_range=x_names,width=width, height=174, title=i_col)
        f.vbar(x=x_names, width=0.585, top=qs, color=colors)
        figures1.append(f)
        qss.append(qs)
        i+=1
    grid = gridplot([figures1])
    output_file('tida_alls.html')
    if show_figures1 == True: show(grid)
    sub_wts = params['subwts']
    main_wts = [subm['weight'] for subm in params['subm']]
    mms,acc_mass = [],[]
    for j in range(len(dossiers)):
        one_dossier = dossiers[j]
        qs = [one['q'] for one in one_dossier['q_in']]
        mm = [qs[h] * (main_wts[j] + sub_wts[h]) for h in range(len(sub_wts))]
        mass = sum(mm)
        mms.append(mm)
        acc_mass.append(round(mass))                        #subm_names[::-1]
    y_names = [name + " - " + str(mass) for name,mass in zip(subm_names[::-1],acc_mass)]
    f1 = figure(y_range=y_names, width=313, height=174, title='relations of general masses')
    f1.hbar(y=y_names, height=0.585, right=acc_mass, left=0, color=colors)
    output_file('tida_alls2.html')
    alls = [f'alls.{i}' for i in range(len(dossiers))]
    subm = [f'sub{i}'   for i in range(len(dossiers))] 
    mmsT  = np.asarray(mms).T
    data = {'cols' : alls}
    for i in range(len(dossiers)): data[f'sub{i}'] = mmsT[i,:]
    f2 = figure(y_range=alls, height=174, width=274, title="relations of columns masses")
    f2.hbar_stack(subm, y='cols', height=0.585, color=colors, source=data)
    qssT  = np.asarray(qss).T
    data = {'cols' : alls}
    for i in range(len(dossiers)): data[f'sub{i}'] = qssT[i,:]
    f3 = figure(y_range=alls, height=174, width=215, title="ratios in columns")
    f3.hbar_stack(subm, y='cols', height=0.585, color=colors, source=data)
    grid = gridplot([[f3,f2,f1]])
    show(grid)
    

def h_blend(path, fs_names, params, color, show_figures1=False):
    #colors_alls = ["sienna","green","blue","red","silver",'gold']
    colors_alls  = ['mediumblue',"green",'crimson',"magenta","sienna",
                    'red',"dimgray","black","silver",'gold']
    colors_Red   = ["firebrick",'tomato',"red","crimson","orangered"]
    colors_Green = ["limegreen","forestgreen","darkgreen",'lime',"green"]
    colors_Blue  = ['midnightblue',"royalblue","mediumblue","blue","steelblue"]
    colors_Brown = ["maroon","chocolate","sienna","sandybrown",'brown']
    if color == 'alls':  colors = [colors_alls [i] for i in range(len(fs_names))]
    if color == 'red':   colors = [colors_Red  [i] for i in range(len(fs_names))]
    if color == 'green': colors = [colors_Green[i] for i in range(len(fs_names))]
    if color == 'blue':  colors = [colors_Blue [i] for i in range(len(fs_names))]
    if color == 'brown': colors = [colors_Brown[i] for i in range(len(fs_names))]
    dk = params
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
        # display(df_subms.head(3))
        pd.set_option('display.float_format', '{:.5f}'.format)
        df_subms = df_subms.rename(columns={"ensemble":dk["target"]})
        df_subms.to_csv(f'tida_alls.csv', index=False) # so far it's 
        return df_subms
        
    sample_subm = pd.read_csv(path + fs_names[1] + ".csv")

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

    da = ensemble_da(dk)
    
    bokeh_show(params, colors, show_figures1)
    
    return da


import numpy as np

from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from bokeh.util.hex import hexbin

n = 50000
x = np.random.standard_normal(n)
y = np.random.standard_normal(n)

bins = hexbin(x, y, 0.1)

p = figure(tools="wheel_zoom,reset", match_aspect=True, background_fill_color='#440154')
p.grid.visible = False

p.hex_tile(q="q", r="r", size=0.1, line_color=None, source=bins,
           fill_color=linear_cmap('counts', 'Viridis256', 0, max(bins.counts)))

output_file("hex_tile.html")

output_notebook()

show(p)


# Group 1

path = '/kaggle/input/21-august-2025-ps-s5e8/' + 'submission '

fs_names = ['0.97742','0.97735','0.97731','0.97706','0.97574']

params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "y",
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [ +0.11,+0.04,-0.02,-0.05,-0.08 ],             
      'subm'   : [
         { 'name':fs_names[0],'weight':+0.57 },
         { 'name':fs_names[1],'weight':+0.13 },
         { 'name':fs_names[2],'weight':+0.17 },
         { 'name':fs_names[3],'weight':+0.12 },
         { 'name':fs_names[4],'weight':+0.01 },
      ]
    }

df = h_blend ( path, fs_names, params, color='red', show_figures1=True )

df.to_csv('Group_1.csv', index=False)

# display(df)


# Group 2

path = '/kaggle/input/21-august-2025-ps-s5e8/' + 'submission '

fs_names = ['0.97743','0.97729','0.97710','0.97694','0.97118']

params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "y",
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [ +0.14,+0.07,-0.03,-0.07,-0.11 ],             
      'subm'   : [
         { 'name':fs_names[0],'weight':+0.40 },
         { 'name':fs_names[1],'weight':+0.30 },
         { 'name':fs_names[2],'weight':+0.17 },
         { 'name':fs_names[3],'weight':+0.11 },
         { 'name':fs_names[4],'weight':+0.02 }, 
      ]
    }

df = h_blend ( path, fs_names, params, color='green', show_figures1=True )

df.to_csv('Group_2.csv', index=False)

# display(df)


# Group 3

path = '/kaggle/input/21-august-2025-ps-s5e8/' + 'submission '

fs_names = ['0.97762','0.97756','0.97754','0.97661','0.97540']
 
params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "y",
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [ +0.21,+0.11,-0.04,-0.10,-0.17 ],             
      'subm'   : [
         { 'name':fs_names[0],'weight':+0.74 },
         { 'name':fs_names[1],'weight':+0.11 },
         { 'name':fs_names[2],'weight':+0.10 },
         { 'name':fs_names[3],'weight':+0.03 },
         { 'name':fs_names[4],'weight':+0.02 },
      ]
    }

df = h_blend ( path, fs_names, params, color='blue', show_figures1=True )

df.to_csv('Group_3.csv', index=False)

# display(df)


path = '/kaggle/working/'

fs_names = ['Group_1','Group_2','Group_3']
 
params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "y",
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [ +0.21,-0.07,-0.14 ],             
      'subm'   : [
         { 'name':fs_names[0],'weight':+0.45 },
         { 'name':fs_names[1],'weight':+0.15 },
         { 'name':fs_names[2],'weight':+0.40 },
      ]
    }

params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "y",
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [ +0.21,-0.07,-0.14 ],             
      'subm'   : [
         { 'name':fs_names[0],'weight':+0.18 },
         { 'name':fs_names[1],'weight':+0.12 },
         { 'name':fs_names[2],'weight':+0.70 },
      ]
    }

df = h_blend ( path, fs_names, params, color='alls')

df.to_csv('submission.csv', index=False)

# display(df)

