import os,ast,copy
import numpy as np
import pandas as pd

from bokeh.plotting import figure, gridplot 
from bokeh.io import output_file, show, output_notebook
output_notebook()


def color_scheme(dk,color):
    colors    = ['red','green','blue']
    clr_alls  = ['crimson',"forestgreen",'mediumblue',"darkmagenta",'gold',"silver"]
    clr_alls2 = ['red',"green",'blue',"silver",'gold']
    clr_alls3 = ['darkmagenta',"forestgreen",'mediumblue']
    clr_alls4 = ['crimson',"forestgreen",'mediumblue','darkmagenta']
    clr_Red   = ["firebrick","orangered","crimson",'tomato',"red"]
    clr_Red4  = ["firebrick","orangered","crimson",'tomato']
    clr_Green = ["darkgreen","limegreen","green",'lime',"forestgreen"]
    clr_Green2= ['olivedrab',"darkgreen","forestgreen"]
    clr_Green3= ["darkmagenta",'olivedrab',"darkgreen"]
    clr_Blue  = ['midnightblue',"royalblue","mediumblue","blue","steelblue",'cyan']
    clr_Blue4 = ['midnightblue',"royalblue","mediumblue","steelblue"]
    clr_Brown = ["maroon","sienna","chocolate","sandybrown",'brown']
    clr_Brown3= ["maroon","sienna","sandybrown"]
    clr_Brown4= ["maroon","sienna","chocolate","sandybrown"]
    clr_Two   = ['crimson','mediumblue']
    clr_Two2  = ['crimson','darkgreen']
    clr_tes3  = ['limegreen',"magenta",'red']
    clr_tes3b = ['darkmagenta',"magenta",'red']
    clr_tes6  = ['limegreen'] + clr_Brown
    clr_tes7  = ['limegreen'] + clr_Brown4 + ["magenta"]+["darkmagenta"]
    clr_tes8  = clr_Red4 + clr_Blue4
    clr_tes9  = clr_Red4 + ['darkmagenta'] + clr_Blue4
    clr_tes10 = clr_Brown + clr_Green
    clr_tes11 = clr_Brown + ['red','darkmagenta'] + clr_Green
    l = len(dk['subm'])
    if color == 'Two2':  colors = clr_Two2   [0:l]
    if color == 'Two':   colors = clr_Two    [0:l]
    if color == 'alls':  colors = clr_alls   [0:l]
    if color == 'alls2': colors = clr_alls2  [0:l]
    if color == 'alls3': colors = clr_alls3  [0:l]
    if color == 'alls4': colors = clr_alls4  [0:l]
    if color == 'red':   colors = clr_Red    [0:l]
    if color == 'green': colors = clr_Green  [0:l]
    if color == 'green2':colors = clr_Green2 [0:l]
    if color == 'green3':colors = clr_Green3 [0:l]
    if color == 'blue':  colors = clr_Blue   [0:l]
    if color == 'brown': colors = clr_Brown  [0:l]
    if color == 'brown3':colors = clr_Brown3 [0:l]
    if color == 'tes3':  colors = clr_tes3   [0:l]
    if color == 'tes3b': colors = clr_tes3b  [0:l]
    if color == 'tes6':  colors = clr_tes6   [0:l]
    if color == 'tes7':  colors = clr_tes7   [0:l]
    if color == 'tes8':  colors = clr_tes8   [0:l]
    if color == 'tes9':  colors = clr_tes9   [0:l]
    if color == 'tes10': colors = clr_tes10  [0:l]
    if color == 'tes11': colors = clr_tes11  [0:l]
    return colors


def bokeh_show(
        params,
        df_cross,
        colors, 
        show_figures1, 
        show_figures2, wps_fig2,
        color_cross):
    
    def dossier(js,subms,cols):
        def quant(i,js,subms,cols):
            return {"c" : i, "q" : sum([1 for subm in cols[i] if subm == subms[js]])}
        return {
            'name' : subms[js],
            'q_in' : [quant(i,js,subms,cols) for i in range(len(subms))]
        }
    alls = pd.read_csv(f'tida_desc.csv')
    matrix = [ast.literal_eval(str(row.alls)) for row in alls.itertuples()]
    subms = sorted(matrix[0])
    cols = [[data[i] for data in matrix] for i in range(len(subms))]
    df_subms = pd.DataFrame({f'col_{i}': [x[i] for x in matrix] for i in range(len(subms))})
    dossiers = [dossier(js,subms,cols) for js in range(len(subms))]
    subm_names = [one_dossier['name'] for one_dossier in dossiers]
    figures1,qss,i = [],[],0
    height = 85 if len(colors)==2\
        else 134 if len(colors)==3 else (154 if len(colors)==4 else 174)
    for one_dossier in dossiers: 
        i_col = 'alls. ' + str(one_dossier['q_in'][i]['c'])
        qs = [one['q'] for one in one_dossier['q_in']]
        x_names = [name.replace("Group","").replace("subm_","") for name in subm_names]
        width = 157  if len(colors) == 5\
            else (121 if len(colors) == 8\
            else (131 if len(colors) == 9\
            else (141 if len(colors) == 10\
            else (171 if len(colors) == 11 else 133))))
        f = figure(x_range=x_names,width=width, height=height, title=i_col)
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
    y_names = [name + " - " + str(mass) for name,mass in zip(subm_names,acc_mass)]
    f1 = figure(y_range=y_names, width=313, height=height, title='relations of general masses')
    f1.hbar(y=y_names, height=0.585, right=acc_mass, left=0, color=colors)
    output_file('tida_alls2.html')
    alls = [f'alls.{i}' for i in range(len(dossiers))]
    subm = [f'sub{i}'   for i in range(len(dossiers))] 
    mmsT  = np.asarray(mms).T
    data = {'cols' : alls}
    for i in range(len(dossiers)): data[f'sub{i}'] = mmsT[i,:]
    f2 = figure(y_range=alls, height=height, width=274, title=" ( !? )")
    f2.hbar_stack(subm, y='cols', height=0.585, color=colors, source=data)
    qssT  = np.asarray(qss).T
    data = {'cols' : alls}
    for i in range(len(dossiers)): data[f'sub{i}'] = qssT[i,:]
    f3 = figure(y_range=alls, height=height, width=215, title="ratios in columns")
    f3.hbar_stack(subm, y='cols', height=0.585, color=colors, source=data)
    grid = gridplot([[f3,f2,f1]])
    show(grid)
    if show_figures2 == True:
        def read(params,i):
            FiN = params["path"] + params["subm"][i]["name"] + ".csv"
            target_name_back = {'target':params["target"],'pred':params["target"]}
            return pd.read_csv(FiN).rename(columns=target_name_back)
        dfs = [read(params,i) for i in range(len(params["subm"]))] + [df_cross]
        f   = figure(width=800, height=274)
        f.title.text = 'Click on legend entries to mute the corresponding lines'
        b,e                   = 21000,21555
        if wps_fig2==500: b,e = 21025,21525
        if wps_fig2==250: b,e = 21150,21400
        if wps_fig2==100: b,e = 21227,21327
        if wps_fig2== 50: b,e = 21254,21304
        line_x     = [dfs[i][b:e]['id']            for i in range(len(dfs))]
        line_y     = [dfs[i][b:e]['accident_risk'] for i in range(len(dfs))]
        color      = colors + [color_cross]
        alpha      = [0.8 for i in range(len(dfs)-1)] + [0.95]
        lws        = [1.0 for i in range(len(dfs)-1)] + [1.00]
        legend = subm_names + ['cross']
        for i in range(len(legend)):
            f.line(line_x[i], line_y[i], line_width=lws[i], color=color[i], alpha=alpha[i],
                   muted_color='white',legend_label=legend[i])
        f.legend.location = "top_left"
        f.legend.click_policy="mute"
        show(f)


def h_blend(params,color,cross='silver',figures1=False,figures2=False,wf2=555,details=False):
# ------------------------------------------------------------------------ 
# def h_blend(path_to_ds, 
#             file_short_names, 
#             params, 
#             type_sort,
#             color,
#             show_figures1=False, show_figures2=False, show_details=False,
#             color_cross='lightsteelblue'):
# ------------------------------------------------------------------------   
    color_cross = cross

    dk = copy.deepcopy(params)

    show_details,show_figures1,show_figures2 = details,figures1,figures2

    file_short_names = [subm['name'] for subm in params['subm']]
    type_sort    = params['type_sort'][0]
    dk['asc']    = params['type_sort'][1]
    dk['desc']   = params['type_sort'][2]
    dk['id']     = params['id_target'][0]
    dk['target'] = params['id_target'][1]
# ------------------------------------------------------------------------
    def read(dk,i):
        tnm = dk["subm"][i]["name"]
        FiN = dk["path"] + tnm + ".csv"
        return pd.read_csv(FiN).rename(columns={
            'target':tnm, 'pred':tnm, dk["target"]:tnm})
        
    def merge(dfs_subm):
        df_subms = pd.merge(dfs_subm[0],  dfs_subm[1], on=[dk['id']])
        for i in range(2, len(dk["subm"])): 
            df_subms = pd.merge(df_subms, dfs_subm[i], on=[dk['id']])
        return df_subms
        
    def da(dk,sorting_direction,show_details):
        
        df_subms = merge([read(dk,i) for i in range(len(dk["subm"]))])
        cols = [col for col in df_subms.columns if col != dk['id']]
        short_name_cols = [c for c in cols]
        
        def alls1(x, sd=sorting_direction,cs=cols):
            reverse = True if sd=='desc' else False
            tes = {c: x[c] for c in cs}.items()
            subms_sorted = [t[0] for t in sorted(tes,key=lambda k:k[1],reverse=reverse)]
            return subms_sorted

        import random

        def alls2(x, sd=sorting_direction,cs=cols):
            reverse = True if sd=='desc' else False
            tes = {c: x[c] for c in cs}.items()
            subms_random = [t[0] for t in tes]
            random.shuffle(subms_random)
            return subms_random

        alls = alls1 if type_sort == 'asc/desc' else alls2
            
        def summa(x,cs,wts,ic_alls): 
            return sum([x[cs[j]] * (wts[0][j] + wts[1][ic_alls[j]]) for j in range(len(cs))])
            
        wts = [[[e['weight'] for e in dk["subm"]], [w for w in dk["subwts" ]]]]
          
        def correct(x, cs=cols, wts=wts):
            i = [x['alls'].index(c) for c in short_name_cols]
            return summa(x,cs,wts[0],i)

        if len(wts) == 1:
            correct_sub_weights = [wt for wt in dk["subwts"]]
            weights = [subm['weight'] for subm in dk["subm"]]
            def correct(x, cs=cols, w=weights, cw=correct_sub_weights):
                ic = [x['alls'].index(c) for c in short_name_cols]
                cS = [x[cols[j]] * (w[j] + cw[ic[j]]) for j in range(len(cols))]
                return sum(cS)
                   
        def amxm(x, cs=cols):
            list_values = x[cs].to_list()
            mxm = abs(max(list_values)-min(list_values))
            return mxm

        if len(wts) > 1:
            df_subms['mx-m']   = df_subms.apply(lambda x: amxm   (x), axis=1)
        df_subms['alls']       = df_subms.apply(lambda x: alls   (x), axis=1)
        df_subms[dk["target"]] = df_subms.apply(lambda x: correct(x), axis=1)
        schema_rename = { old_nc:new_shnc for old_nc, new_shnc in zip(cols, short_name_cols) }
        df_subms = df_subms.rename(columns=schema_rename)
        df_subms = df_subms.rename(columns={dk["target"]:"ensemble"})
        df_subms.insert(loc=1, column=' _ ', value=['   '] * len(df_subms))
        df_subms[' _ '] = df_subms[' _ '].astype(str)
        pd.set_option('display.max_rows',100)
        pd.set_option('display.float_format', '{:.4f}'.format)
        vcols = [dk['id']]+[' _ '] + short_name_cols + [' _ ']+['alls']+[' _ ']+['ensemble']
        if len(wts) > 1: vcols.append([' _ '] + ['mx-m'])
        df_subms = df_subms[vcols]
        if show_details: display(df_subms.head(5))
        pd.set_option('display.float_format', '{:.5f}'.format)
        df_subms = df_subms.rename(columns={"ensemble":dk["target"]})
        df_subms.to_csv(f'tida_{sorting_direction}.csv', index=False)
        return df_subms[[dk['id'],dk['target']]]
   
    def ensemble_da(dk,        show_details): 
        dfD    = da(dk,'desc', show_details)
        dfA    = da(dk,'asc',  show_details)
        dfA[dk['target']] = dk['desc']*dfD[dk['target']] + dfA[dk['target']]*dk['asc']
        return dfA

    da = ensemble_da(dk,show_details)
    colors = color_scheme(dk, color)
    bokeh_show(dk, da, colors, show_figures1, show_figures2, wf2, color_cross)
    return  da


def matrix_vs(path,fs_names):
    def load(path,fs_names):
        dfs = [pd.read_csv(path + name_subm +'.csv') for name_subm in fs_names]
        for i in range(len(dfs)):
            dfs[i] = dfs[i].rename(columns={"accident_risk": f'{fs_names[i]}'})
        dfsm = pd.merge(dfs[0], dfs[1], on="id")
        for i in range(2,len(dfs)):
            dfsm = pd.merge(dfsm,dfs[i],on='id')
        return dfsm   
    def make_list_vs(fs_names):
        list = []
        for i in range(0,len(fs_names)-1):
            for j in range(i+1,len(fs_names)):
                list.append(fs_names[i] + "_vs_" + fs_names[j])
        return list
    def get_mvs(dfs, list_vs):
        def get_abs_distance(x,t1,t2):
            return abs(x[t1]-x[t2])
        for vs in list_vs:
            t = vs.split('_vs_')
            dfs[vs] = dfs.apply(lambda x: get_abs_distance(x,t[0],t[1]), axis=1)
        return dfs   
    def distance_vs(name, st_names, list_vs, dfs):
        distances = []
        for st in st_names:
            vs_between = name + "_vs_" + st
            if vs_between not in list_vs:
                distances.append(0)
            else: distances.append(round(dfs[vs_between].sum()))
        return distances
    dfs = load(path,fs_names)
    list_vs = make_list_vs(fs_names)
    mvs = get_mvs(dfs, list_vs)
    m1 = pd.DataFrame({'subm':fs_names})
    m2 = pd.DataFrame({ name :distance_vs(name, fs_names, list_vs, mvs) for name in fs_names})
    matrix = pd.concat([m1,m2],axis=1)
    return matrix


def blend(path, short_name1, short_name2, w=[0.70, 0.30]):
    df1 = pd.read_csv(path + short_name1 + '.csv')
    df2 = pd.read_csv(path + short_name2 + '.csv')
    df1["accident_risk"] = df1["accident_risk"]*w[0] + w[1]*df2["accident_risk"]
    return df1


%%time

path = '/kaggle/input/2-october-2025-ps-s5e10/submission_'

fsn = [
    '0.05547.a','0.05550.a','0.05550.b','0.05552.a','0.05552.b',
    '0.05552.c','0.05552.d','0.05553.e','0.05553.d','0.05555.a'
]

matrix_distances = matrix_vs(path, fsn)
matrix_distances


df_A = blend(path,fsn[0],fsn[9]); df_A.to_csv('A.csv',index=False)
df_B = blend(path,fsn[1],fsn[8]); df_B.to_csv('B.csv',index=False)
df_C = blend(path,fsn[2],fsn[7]); df_C.to_csv('C.csv',index=False)
df_D = blend(path,fsn[3],fsn[4]); df_D.to_csv('D.csv',index=False)
df_E = blend(path,fsn[5],fsn[6]); df_E.to_csv('E.csv',index=False)

params = {
      'path'     : f'/kaggle/working/',            
      'id_target': ['id',"accident_risk"],          
      'type_sort': ['asc/desc',0.30,0.70],
      'subwts'   : [ +0.07,+0.03, 0, -0.03,-0.07 ],
      'subm'     : [
         { 'name': f'A','weight':+0.60 },
         { 'name': f'B','weight':+0.10 },
         { 'name': f'C','weight':+0.10 },
         { 'name': f'D','weight':+0.10 },
         { 'name': f'E','weight':+0.10 },]
}

df_cross = h_blend(params,color='brown',figures1=True,figures2=True,wf2=250,details=True)

df_cross.to_csv('submission_0.05547.b.csv',  index=False)

matrix_distances = matrix_vs('/kaggle/working/',['A','B','C','D','E'])

display(matrix_distances, df_cross)

for f in ['A','B','C','D','E']: os.remove(f'/kaggle/working/{f}.csv')


%%time

df_A = blend(path,fsn[0],fsn[9]); df_A.to_csv('A.csv',index=False)
df_B = blend(path,fsn[1],fsn[8]); df_B.to_csv('B.csv',index=False)
df_C = blend(path,fsn[2],fsn[7]); df_C.to_csv('C.csv',index=False)
df_D = blend(path,fsn[3],fsn[4]); df_D.to_csv('D.csv',index=False)
df_E = blend(path,fsn[5],fsn[6]); df_E.to_csv('E.csv',index=False)

params = {
      'path'     : f'/kaggle/working/',            
      'id_target': ['id',"accident_risk"],          
      'type_sort': ['asc/desc',0.30,0.70],
      'subwts'   : [ +0.07,+0.03, 0, -0.03,-0.07 ],
      'subm'     : [
         { 'name': f'A','weight':+0.60 },
         { 'name': f'B','weight':+0.10 },
         { 'name': f'C','weight':+0.10 },          # LB = 0.055_47
         { 'name': f'D','weight':+0.10 },
         { 'name': f'E','weight':+0.10 },]
}

df_cross1 = h_blend(params,color='red',figures1=True,figures2=True,wf2=250,details=True)

df_cross1.to_csv('g1.csv',index=False)

matrix_distances = matrix_vs('/kaggle/working/',['A','B','C','D','E'])

display(matrix_distances, df_cross1)


%%time

df_A = blend(path,fsn[0],fsn[9],w=[0.85,0.15]); df_A.to_csv('A.csv',index=False)
df_B = blend(path,fsn[1],fsn[8],w=[0.70,0.30]); df_B.to_csv('B.csv',index=False)
df_C = blend(path,fsn[2],fsn[7],w=[0.70,0.30]); df_C.to_csv('C.csv',index=False)
df_D = blend(path,fsn[3],fsn[4],w=[0.65,0.35]); df_D.to_csv('D.csv',index=False)
df_E = blend(path,fsn[5],fsn[6],w=[0.60,0.40]); df_E.to_csv('E.csv',index=False)

params = {
      'path'     : f'/kaggle/working/',            
      'id_target': ['id',"accident_risk"],          
      'type_sort': ['asc/desc',0.30,0.70],
      'subwts'   : [ +0.05,+0.02, 0, -0.02,-0.05 ],
      'subm'     : [
         { 'name': f'A','weight':+0.70 },
         { 'name': f'B','weight':+0.08 },
         { 'name': f'C','weight':+0.08 },                        # LB = ?
         { 'name': f'D','weight':+0.07 },
         { 'name': f'E','weight':+0.07 },]
}

df_cross2 = h_blend(params,color='green',figures1=True,figures2=True,wf2=250,details=True)

df_cross2.to_csv('g2.csv',index=False)

matrix_distances = matrix_vs('/kaggle/working/',['A','B','C','D','E'])

display(matrix_distances, df_cross2)


%%time

df_A = blend(path,fsn[0],fsn[3],w=[0.70,0.30]); df_A.to_csv('A.csv',index=False)
df_B = blend(path,fsn[1],fsn[8],w=[0.74,0.26]); df_B.to_csv('B.csv',index=False)
df_C = blend(path,fsn[2],fsn[9],w=[0.80,0.20]); df_C.to_csv('C.csv',index=False)
df_D = blend(path,fsn[7],fsn[4],w=[0.60,0.40]); df_D.to_csv('D.csv',index=False)
df_E = blend(path,fsn[6],fsn[5],w=[0.50,0.50]); df_E.to_csv('E.csv',index=False)

params = {
      'path'     : f'/kaggle/working/',            
      'id_target': ['id',"accident_risk"],          
      'type_sort': ['asc/desc',0.30,0.70],
      'subwts'   : [ +0.04,+0.01, 0, -0.01,-0.04 ],
      'subm'     : [
         { 'name': f'A','weight':+0.70 },
         { 'name': f'B','weight':+0.08 },
         { 'name': f'C','weight':+0.08 },                        # LB = ?
         { 'name': f'D','weight':+0.07 },
         { 'name': f'E','weight':+0.07 },]
}

df_cross3 = h_blend(params,color='blue',figures1=True,figures2=True,wf2=250,details=True)

df_cross3.to_csv('g3.csv',index=False)

matrix_distances = matrix_vs('/kaggle/working/',['A','B','C','D','E'])

display(matrix_distances, df_cross3)


params = {
      'path'     : f'/kaggle/working/',            
      'id_target': ['id',"accident_risk"],          
      'type_sort': ['asc/desc',0.30,0.70],
      'subwts'   : [ +0.13, -0.05,-0.08 ],
      'subm'     : [
         { 'name': f'g1','weight':+0.32 },
         { 'name': f'g2','weight':+0.33 }, 
         { 'name': f'g3','weight':+0.35 },]                  
}

df_alls = h_blend(params,color='alls',figures1=True,figures2=True,wf2=250,details=True)

df_alls.to_csv('submission_0.05547.c.csv',  index=False)

matrix_distances = matrix_vs('/kaggle/working/',['g1','g2','g3'])

display(matrix_distances, df_alls)

for f in ['A','B','C','D','E']+['g1','g2','g3']: os.remove(f'/kaggle/working/{f}.csv')


params = {
      'path'     : f'/kaggle/input/2-october-2025-ps-s5e10/submission_',            
      'id_target': ['id',"accident_risk"],          
      'type_sort': ['asc/desc',0.30,0.70],
      'subwts'   : [ +0.07,+0.04, -0.04,-0.07 ],
      'subm'     : [
         { 'name': f'0.05547.a','weight':+0.90 },
         { 'name': f'0.05552.a','weight':+0.09 },
         { 'name': f'0.05552.b','weight':+0.06 },
         { 'name': f'0.05553.e','weight':-0.05 },]
}

df_g4 = h_blend(params,color='red',figures1=True,figures2=True,wf2=250,details=True)

df_g4.to_csv('g4.csv',index=False)

matrix_distances =\
    matrix_vs(f'/kaggle/input/2-october-2025-ps-s5e10/submission_',
              ['0.05547.a','0.05552.a','0.05552.b','0.05553.e'])
matrix_distances


params = {
      'path'     : f'/kaggle/input/2-october-2025-ps-s5e10/submission_',            
      'id_target': ['id',"accident_risk"],          
      'type_sort': ['asc/desc',0.30,0.70],
      'subwts'   : [ +0.07,-0.03,-0.04 ],
      'subm'     : [
         { 'name': f'0.05550.a','weight':+0.850 },
         { 'name': f'0.05552.c','weight':+0.065 },
         { 'name': f'0.05553.d','weight':+0.085 },]
}

df_g5 = h_blend(params,color='green',figures1=True,figures2=True,wf2=250,details=True)

df_g5.to_csv('g5.csv',index=False)

matrix_distances =\
    matrix_vs(f'/kaggle/input/2-october-2025-ps-s5e10/submission_',
              ['0.05550.a','0.05552.c','0.05553.d'])
matrix_distances


params = {
      'path'     : f'/kaggle/input/2-october-2025-ps-s5e10/submission_',            
      'id_target': ['id',"accident_risk"],          
      'type_sort': ['asc/desc',0.30,0.70],
      'subwts'   : [ +0.07,-0.03,-0.04 ],
      'subm'     : [
         { 'name': f'0.05550.a','weight':+0.850 },
         { 'name': f'0.05552.c','weight':+0.065 },
         { 'name': f'0.05555.a','weight':+0.085 },]
}

df_g6 = h_blend(params,color='blue',figures1=True,figures2=True,wf2=250,details=True)

df_g6.to_csv('g6.csv',index=False)

matrix_distances =\
    matrix_vs(f'/kaggle/input/2-october-2025-ps-s5e10/submission_',
              ['0.05550.b','0.05552.d','0.05555.a'])
matrix_distances


params = {
      'path'     : f'/kaggle/working/',            
      'id_target': ['id',"accident_risk"],          
      'type_sort': ['asc/desc',0.30,0.70],
      'subwts'   : [ +0.13, -0.05,-0.08 ],
      'subm'     : [
         { 'name': f'g4','weight':+0.850 },
         { 'name': f'g5','weight':+0.080 },                     # LB = ?
         { 'name': f'g6','weight':+0.070 },]                  
}

df_alls2 = h_blend(params,color='alls2',figures1=True,figures2=True,wf2=250,details=True)

df_alls2.to_csv('submission_0.05547.d.csv', index=False)

matrix_distances = matrix_vs('/kaggle/working/',['g4','g5','g6'])

display(matrix_distances, df_alls2)

for f in ['g4','g5','g6']: os.remove(f'/kaggle/working/{f}.csv')


# redirrect
df = pd.read_csv('/kaggle/input/2-october-2025-ps-s5e10/submission_0.05547.a.csv')
df.to_csv('submission_0.05547.a.csv', index=False)

params = {
      'path'     : f'/kaggle/working/submission_',            
      'id_target': ['id',"accident_risk"],          
      'type_sort': ['asc/desc',0.30,0.70],
      'subwts'   : [ +0.04, +0.02, -0.02,-0.04 ],
      'subm'     : [
         { 'name': f'0.05547.a','weight':-0.04 },
         { 'name': f'0.05547.b','weight':+0.10 },
         { 'name': f'0.05547.c','weight':+0.20 },
         { 'name': f'0.05547.d','weight':+0.74 },]       
}

df = h_blend(params,color='alls4',figures1=True,figures2=True,wf2=250,details=True)

df.to_csv('submission_0.05547.e.csv', index=False)

matrix_distances =\
    matrix_vs('/kaggle/working/submission_',
        ['0.05547.a','0.05547.b','0.05547.c','0.05547.d','0.05547.e'])

display(matrix_distances, df)


params = {
      'path'     : f'/kaggle/working/submission_',            
      'id_target': ['id',"accident_risk"],          
      'type_sort': ['asc/desc',0.30,0.70],
      'subwts'   : [ +0.04, +0.02,-0.02,-0.04 ],
      'subm'     : [
         { 'name': f'0.05547.a','weight':-0.11 },
         { 'name': f'0.05547.b','weight':+0.11 },
         { 'name': f'0.05547.c','weight':+0.21 },
         { 'name': f'0.05547.d','weight':+0.79 },]       
}

df = h_blend(params,color='alls4',figures1=True,figures2=True,wf2=250,details=True)

df.to_csv('submission_0.05547.f.csv', index=False)

matrix_distances =\
    matrix_vs('/kaggle/working/submission_',
        ['0.05547.a','0.05547.b','0.05547.c','0.05547.d','0.05547.e','0.05547.f'])

display(matrix_distances)


params = {
      'path'     : f'/kaggle/input/2-october-2025-ps-s5e10/submission_',            
      'id_target': ['id',"accident_risk"],          
      'type_sort': ['asc/desc',0.30,0.70],
      'subwts'   : [ +0.03,+0.02,+0.01, 0, -0.01,-0.02,-0.03 ],
      'subm'     : [
         { 'name': f'0.05547.a','weight':+0.63 },
         { 'name': f'0.05550.c','weight':+0.18 },
         { 'name': f'0.05551.a','weight':+0.11 },
         { 'name': f'0.05552.a','weight':+0.08 },
         { 'name': f'0.05552.b','weight':+0.05 },
         { 'name': f'0.05553.c','weight':-0.02 },
         { 'name': f'0.05553.b','weight':-0.03 },]       
}

df = h_blend(params,color='tes7',figures1=True,figures2=True,wf2=250,details=True)

df.to_csv('submission_0.05547.g.csv',  index=False)

matrix_distances =\
 matrix_vs('/kaggle/input/2-october-2025-ps-s5e10/submission_',
  ['0.05547.a','0.05550.c','0.05551.a','0.05552.a','0.05552.b','0.05553.c','0.05553.b'])

display(matrix_distances)


params = {
      'path'     : f'/kaggle/input/2-october-2025-ps-s5e10/submission_',            
      'id_target': ['id',"accident_risk"],          
      'type_sort': ['asc/desc',0.30,0.70],
      'subwts'   : [ +0.04,+0.02,-0.01,-0.02,-0.03 ],
      'subm'     : [
         { 'name': f'0.05547.a','weight':+0.70 },
         { 'name': f'0.05550.c','weight':+0.14 },
         { 'name': f'0.05551.a','weight':+0.08 },
         { 'name': f'0.05552.a','weight':+0.05 },
         { 'name': f'0.05552.b','weight':+0.03 },]       
}

df = h_blend(params,color='tes7',figures1=True,figures2=True,wf2=250,details=True)

df.to_csv('submission_0.05547.h.csv',  index=False)

matrix_distances =\
 matrix_vs('/kaggle/input/2-october-2025-ps-s5e10/submission_',
  ['0.05547.a','0.05550.c','0.05551.a','0.05552.a','0.05552.b'])

display(matrix_distances)


params = {
      'path'     : f'/kaggle/input/2-october-2025-ps-s5e10/submission_',            
      'id_target': ['id',"accident_risk"],          
      'type_sort': ['asc/desc',0.30,0.70],
      'subwts'   : [ +0.04,+0.02,-0.01,-0.02,-0.03 ],
      'subm'     : [
         { 'name': f'0.05547.a','weight':+0.85 },
         { 'name': f'0.05550.c','weight':+0.09 }, # prev,err +0.10 -> sum.wts=101%
         { 'name': f'0.05551.a','weight':+0.03 },
         { 'name': f'0.05552.a','weight':+0.02 },
         { 'name': f'0.05552.b','weight':+0.01 },]       
}

df = h_blend(params,color='tes7',figures1=True,figures2=True,wf2=250,details=True)

df.to_csv('submission_0.05547.i.csv',  index=False)

matrix_distances =\
 matrix_vs('/kaggle/input/2-october-2025-ps-s5e10/submission_',
  ['0.05547.a','0.05550.c','0.05551.a','0.05552.a','0.05552.b'])

display(matrix_distances)


params = {
      'path'     : f'/kaggle/input/2-october-2025-ps-s5e10/submission_',            
      'id_target': ['id',"accident_risk"],          
      'type_sort': ['asc/desc',0.30,0.70],
      'subwts'   : [ +0.03,-0.01,-0.02 ],
      'subm'     : [
         { 'name': f'0.05547.a','weight':+0.90 },
         { 'name': f'0.05550.c','weight':+0.07 },
         { 'name': f'0.05552.a','weight':+0.03 },]       
}

df = h_blend(params,color='tes7',figures1=True,figures2=True,wf2=250,details=True)

df.to_csv('submission_0.05547.j.csv',  index=False)

matrix_distances =\
 matrix_vs('/kaggle/input/2-october-2025-ps-s5e10/submission_',
  ['0.05547.a','0.05550.c','0.05552.a'])

display(matrix_distances)


params = {
      'path'     : f'/kaggle/input/2-october-2025-ps-s5e10/submission_',            
      'id_target': ['id',"accident_risk"],          
      'type_sort': ['asc/desc',0.30,0.70],
      'subwts'   : [ +0.11,-0.04,-0.07 ],
      'subm'     : [
         { 'name': f'0.05547.a','weight':+0.60 },
         { 'name': f'0.05548.a','weight':+0.35 },
         { 'name': f'0.05550.c','weight':+0.05 },]       
}

df = h_blend(params,color='tes3',figures1=True,figures2=True,wf2=250,details=True)

df.to_csv('submission_0.05546.a.csv',  index=False)

matrix_distances =\
 matrix_vs('/kaggle/input/2-october-2025-ps-s5e10/submission_',
  ['0.05547.a','0.05548.a','0.05550.c'])

display(matrix_distances)


params = {
      'path'     : f'/kaggle/input/2-october-2025-ps-s5e10/submission_',            
      'id_target': ['id',"accident_risk"],          
      'type_sort': ['asc/desc',0.30,0.70],
      'subwts'   : [ +0.11,-0.04,-0.07 ],
      'subm'     : [
         { 'name': f'0.05547.e','weight':+0.65 },
         { 'name': f'0.05548.a','weight':+0.32 },
         { 'name': f'0.05550.c','weight':+0.03 },]       
}

df = h_blend(params,color='tes3b',figures1=True,figures2=True,wf2=250,details=True)

df.to_csv('submission_0.05546.b.csv',  index=False)

matrix_distances =\
 matrix_vs('/kaggle/input/2-october-2025-ps-s5e10/submission_',
  ['0.05547.e','0.05548.a','0.05550.c'])

display(matrix_distances)


%%time

path = '/kaggle/input/5-october-2025-ps-s5e10/submission_'

fsn = [
    '0.05546.a','0.05547.a','0.05547.b','0.05547.c','0.05547.d','0.05547.e',
                '0.05547.f','0.05547.g','0.05547.h','0.05547.i','0.05547.j','0.05546.b',

    '0.05548.a','0.05550.c'
]

matrix_distances = matrix_vs(path, fsn)

scheme_rename = {col : col.replace('0.055','') for col in matrix_distances.columns}

matrix_distances = matrix_distances.rename(columns=scheme_rename)

matrix_distances


%%time

p = '/kaggle/input/5-october-2025-ps-s5e10/submission_'

_46a = pd.read_csv(p + '0.05546.a.csv')

_47a = pd.read_csv(p + '0.05547.a.csv')
_47b = pd.read_csv(p + '0.05547.b.csv')
_47c = pd.read_csv(p + '0.05547.c.csv')
_47d = pd.read_csv(p + '0.05547.d.csv')
_47e = pd.read_csv(p + '0.05547.e.csv')
_47f = pd.read_csv(p + '0.05547.f.csv')
_47g = pd.read_csv(p + '0.05547.g.csv')
_47h = pd.read_csv(p + '0.05547.h.csv')
_47i = pd.read_csv(p + '0.05547.i.csv')
_47j = pd.read_csv(p + '0.05547.j.csv')

_46b = pd.read_csv(p + '0.05546.b.csv')

df = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

t = "accident_risk"

df[t] = _46a[t] *1.01 - _47a[t] * 0.01
df[t] =   df[t] *1.01 - _47b[t] * 0.01
df[t] =   df[t] *1.01 - _47c[t] * 0.01
df[t] =   df[t] *1.01 - _47d[t] * 0.01
df[t] =   df[t] *1.01 - _47e[t] * 0.01
df[t] =   df[t] *1.01 - _47f[t] * 0.01
df[t] =   df[t] *1.01 - _47g[t] * 0.01
df[t] =   df[t] *1.01 - _47h[t] * 0.01
df[t] =   df[t] *1.01 - _47i[t] * 0.01
df[t] =   df[t] *1.01 - _47j[t] * 0.01

df[t] =   df[t] *1.01 - _46b[t] * 0.01

df.to_csv('submission_0.05546.c.csv', index=False)
df


%%time

target = t

dfs_47 = [_47b,_47c,_47d,_47e,_47f,_47g,_47h,_47i,_47j,_46b]

df[target] =   _46a[target] *1.02 - 0.02* _47a[target]

for df_47 in dfs_47:    
    df[target] = df[target] *1.02 - 0.02* df_47[target]

df.to_csv('submission_0.05546.d.csv', index=False)    # 46.a < 46.c < 46.d.Top.1
df


df[target] =   _46a[target] *1.0414 - 0.0414* _47a[target]

for df_47 in dfs_47:    
    df[target] = df[target] *1.0414 - 0.0414* df_47[target]

df.to_csv('submission_0.05546.e.csv', index=False)    # 46.e.LB = ?


df.to_csv('submission.csv',  index=False)

df

