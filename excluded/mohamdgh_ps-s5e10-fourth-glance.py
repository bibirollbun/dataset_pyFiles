import os, ast
import numpy as np
import pandas as pd

from bokeh.plotting import figure, gridplot 
from bokeh.io import output_file, show, output_notebook
output_notebook()


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
            'name' : subms[js], 'color' : colors[js],
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
    height = 101 if len(colors)==2\
        else 134 if len(colors)==3 else (154 if len(colors)==4 else 174)
    for one_dossier in dossiers: 
        i_col = 'alls. ' + str(one_dossier['q_in'][i]['c'])
        qs = [one['q'] for one in one_dossier['q_in']]
        x_names = [name.replace("Group","").replace("subm_","") for name in subm_names]
        width = 157  if len(colors) == 5\
            else (121 if len(colors) == 8\
            else (131 if len(colors) == 9\
            else (141 if len(colors) == 10\
            else (171 if len(colors) == 11 else 111))))
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
    f2 = figure(y_range=alls, height=height, width=274, title=" ( relations of columns masses )")
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
        b,e        = 21000,21021
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


def color_scheme(dk,color):
    colors    = ['red','green','blue']
    clr_silver= ['gray','darkgray','silver','gold']
    clr_silver3=['gold','darkgray','silver']
    clr_silverr=['gold','gray','darkgray','silver']
    clr_alls  = ['crimson',"forestgreen",'mediumblue','gold',"darkmagenta","silver"]
    clr_alls2 = ["silver",'gold']
    clr_alls3 = ['darkmagenta',"forestgreen",'mediumblue']
    clr_alls3b= ['crimson',"darkgreen",'mediumblue']
    clr_alls3c= ['tomato',"limegreen",'royalblue']
    clr_alls4 = ['red',"forestgreen",'mediumblue','crimson']
    clr_alls4m= ['red',"forestgreen",'mediumblue',"darkmagenta"]
    clr_alls5 = ['darkmagenta',"crimson","darkgreen",'mediumblue',"magenta"]
    clr_Red   = ["crimson","orangered","red",'tomato','gold',"firebrick",]
    clr_Red3  = ["crimson","red","tomato"]
    clr_Red4  = ["firebrick","orangered","crimson",'tomato']
    clr_Green = ["green","limegreen","darkgreen","forestgreen",'lime']
    clr_Green2= ['olivedrab',"darkgreen","forestgreen"]
    clr_Green3= ["darkgreen","forestgreen","limegreen",]
    clr_Blue  = ["mediumblue","steelblue","blue","royalblue",'midnightblue','cyan']
    clr_Blue3 = ['midnightblue','mediumblue',"royalblue"]
    clr_Blue4 = ['midnightblue',"royalblue","mediumblue","steelblue"]
    clr_Brown = ["maroon","sienna","sandybrown","chocolate",'brown',]
    clr_Brown3= ["maroon","sienna","sandybrown"]
    clr_Brown4= ["maroon","sienna","chocolate","sandybrown"]
    clr_magent= ['darkmagenta','magenta']
    clr_Two   = ['olivedrab','gold']
    clr_Two2  = ['crimson','darkgreen']
    clr_tes3  = ['limegreen',"magenta",'red']
    clr_tes3b = ['darkmagenta',"magenta",'red']
    clr_tes6  = ['limegreen'] + clr_Brown
    clr_tes7  = ['gold',"silver","darkgray","gray","tomato","darkgreen","mediumblue"]
    clr_tes8  = clr_Red4 + clr_Blue4
    clr_tes9  = clr_Red3 + clr_Green3 + clr_Blue3
    clr_tes10 = clr_Brown + clr_Green
    clr_tes11 = clr_Brown + ['red','darkmagenta'] + clr_Green
    l = len(dk['subm'])
    if color == 'Two2'  : colors = clr_Two2   [0:l]
    if color == 'Two'   : colors = clr_Two    [0:l]
    if color == 'alls'  : colors = clr_alls   [0:l]
    if color == 'alls2' : colors = clr_alls2  [0:l]
    if color == 'alls3' : colors = clr_alls3  [0:l]
    if color == 'alls3b': colors = clr_alls3b [0:l]
    if color == 'alls3c': colors = clr_alls3c [0:l]
    if color == 'alls4' : colors = clr_alls4  [0:l]
    if color == 'alls4m': colors = clr_alls4m [0:l]
    if color == 'alls5' : colors = clr_alls5  [0:l]
    if color == 'red'   : colors = clr_Red    [0:l]
    if color == 'red3'  : colors = clr_Red3   [0:l]
    if color == 'green' : colors = clr_Green  [0:l]
    if color == 'green2': colors = clr_Green2 [0:l]
    if color == 'green3': colors = clr_Green3 [0:l]
    if color == 'blue'  : colors = clr_Blue   [0:l]
    if color == 'blue3' : colors = clr_Blue3  [0:l]
    if color == 'brown' : colors = clr_Brown  [0:l]
    if color == 'brown3': colors = clr_Brown3 [0:l]
    if color == 'tes3'  : colors = clr_tes3   [0:l]
    if color == 'tes3b' : colors = clr_tes3b  [0:l]
    if color == 'tes6'  : colors = clr_tes6   [0:l]
    if color == 'tes7'  : colors = clr_tes7   [0:l]
    if color == 'tes8'  : colors = clr_tes8   [0:l]
    if color == 'tes9'  : colors = clr_tes9   [0:l]
    if color == 'tes10' : colors = clr_tes10  [0:l]
    if color == 'tes11' : colors = clr_tes11  [0:l]
    if color == 'magent': colors = clr_magent [0:l]
    if color == 'silver': colors = clr_silver [0:l]
    if color == 'silverr':colors = clr_silverr[0:l]
    if color == 'silver3':colors = clr_silver3[0:l]
    return colors


def h_blend(params,color,cross='silver',
            figures1=False,figures2=False,wf2=555,
            details=False):

    import copy

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
        pd.set_option('display.float_format', '{:.5f}'.format)
        vcols = [dk['id']]+[' _ '] + short_name_cols + [' _ ']+['alls']+[' _ ']+['ensemble']
        if len(wts) > 1: vcols.append([' _ '] + ['mx-m'])
        df_subms = df_subms[vcols]
        if show_details and sorting_direction=='asc': display(df_subms.head(5))
        df_subms = df_subms.rename(columns={"ensemble":dk["target"]})
        for snc in short_name_cols: df_subms[snc] = df_subms[snc].round(7)
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


def blend(path, short_name1, short_name2, short_name3, w=[0.33, 0.34, 0.33]):
    df1 = pd.read_csv(path + short_name1 + '.csv')
    df2 = pd.read_csv(path + short_name2 + '.csv')
    df3 = pd.read_csv(path + short_name3 + '.csv')
    df1["accident_risk"] =\
        df1["accident_risk"]*w[0] +\
        df2["accident_risk"]*w[1] +\
        df3["accident_risk"]*w[2]
    return df1


def procedure_Cage(FiN_import,n_iter=4,ks1=[1.0054,0.0021],ks2=[1.00037,0.00037]):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import warnings; warnings.filterwarnings('ignore')
    
    sub_sample = pd.read_csv('../input/playground-series-s5e10/sample_submission.csv') 
    sub_import = pd.read_csv(FiN_import) 
    per = sub_import['accident_risk'].values
    # ..................................................................................................
    sns.set()
    plt.figure(figsize=(5, 2))
    plt.hist(per, bins=80)
    plt.gca().set_facecolor('mintcream')
    plt.suptitle('Before | accident_risk', y=0.96, fontsize=12, c='navy')
    # ..................................................................................................
    print('- - - - - - - ',FiN_import)
    min_per  = np.min(per);  print('Min:',  round(min_per, 7))
    max_per  = np.max(per);  print('Max:',  round(max_per, 7))
    mean_per = np.mean(per); print('Mean:', round(mean_per,7))
    print('-------')
    R = -0.0
    guide = mean_per - R
    # ....................................
    per1 = [f for f in per if f < guide]
    per2 = [f for f in per if f > guide]
    print(len(per1),'-',len(per2))
    print('-------')
    N = n_iter
    for _ in range(N):
        for i in range(len(per)):
            per_guide = (per[i] + guide) / 2            
            if per[i] <= guide:
                per[i] = (per[i] *ks1[0]) - (per_guide *ks1[1])
            else:
                per[i] = (per[i] *ks2[0]) - (per_guide *ks2[1])
    # .......................................................................
    sns.set()
    plt.figure(figsize=(5, 2))
    plt.hist(per, bins=80)
    plt.gca().set_facecolor('snow')
    plt.suptitle('After | accident_risk', y=0.96, fontsize=11, c='navy')
    # .......................................................................
    min_per  = np.min(per);  print('Min:',  round(min_per, 7))
    max_per  = np.max(per);  print('Max:',  round(max_per, 7))
    mean_per = np.mean(per); print('Mean:', round(mean_per,7)); 
    # .......................................................................
    print('- - - - - - - ', 'Cage '+FiN_import, '\n')
    # .......................................................................
    
    sub_sample['accident_risk'] = per
    return sub_sample 


def Cage_by_MehranKazeminia(file_name_to_Cage):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import warnings; warnings.filterwarnings('ignore')
    
    sub_import = pd.read_csv(file_name_to_Cage)
    
    per = sub_import['accident_risk'].values
    # ...................................................................................
    sns.set()
    plt.figure(figsize=(5, 2))
    plt.hist(per, bins=80)
    plt.gca().set_facecolor('mintcream')
    plt.suptitle('Before | accident_risk', y=0.96, fontsize=13, c='navy')
    # ...................................................................................
    min_per  = np.min (per) ;print('Min:',  round(min_per, 3))
    max_per  = np.max (per) ;print('Max:',  round(max_per, 3))
    mean_per = np.mean(per) ;print('Mean:', round(mean_per,3))
    # ...................................................................................
    R = 0.0              # Adjusting the R value can increase the accuracy of the guide.
    guide = mean_per - R
    # ....................................
    per1 = [f for f in per if f < guide]
    per2 = [f for f in per if f > guide]
    
    print(len(per1), len(per2))
    # .......................................................................
    for i in range(len(per)):
        
        per_guide = (per[i] + guide) / 2
            
        if per[i] <= guide: per[i] = (per[i]* 1.30) - (per_guide* 0.30)
        if per[i] >  guide: per[i] = (per[i]* 1.00) - (per_guide* 0.00)
    # .......................................................................
    sns.set()
    plt.figure(figsize=(5, 2))
    plt.hist(per, bins=80)
    plt.gca().set_facecolor('snow')
    plt.suptitle('After | accident_risk', y=0.96, fontsize=12, c='navy')
    
    # .......................................................................
    min_per  = np.min (per) ;print('Min:',  round(min_per, 3))
    max_per  = np.max (per) ;print('Max:',  round(max_per, 3))
    mean_per = np.mean(per) ;print('Mean:', round(mean_per,3))
    
    # -------------------------------------------------------------------------
    # After the after
    # -------------------------------------------------------------------------
    
    for i in range(len(per)): 
        if per[i] < (min_per+7): per[i] = per[i] ** 0.994
        if per[i] > (max_per-9): per[i] = per[i] ** 1.006
    # .......................................................................
    sns.set()
    plt.figure(figsize=(5, 2))
    plt.hist(per, bins=80)
    plt.gca().set_facecolor('ghostwhite')
    plt.suptitle('After the after | accident_risk', y=0.96, fontsize=11, c='navy')
    # .......................................................................
    min_per  = np.min (per) ;print('Min:',  round(min_per, 3))
    max_per  = np.max (per) ;print('Max:',  round(max_per, 3))
    mean_per = np.mean(per) ;print('Mean:', round(mean_per,3))
    # -------------------------------------------------------------------------
    df = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
    df['accident_risk'] = per
    return df
    

def display_distances(params):
    files = [subm['name'] for subm in params['subm']]
    distances = matrix_vs ( params['path'], files )            
    display(distances)


params = {
      'path'     : '/kaggle/input/22-october-2025-ps-s5e10/submission_',            
      'id_target': ['id',"accident_risk"],          
      'type_sort': ['asc/desc',0.30,0.70],
      'subwts'   : [ 0.0,-0.01,-0.01,-0.01,+0.03 ],       
      'subm'     : [ 
         { 'name': f'0.05539.a','weight': 0.02 },
         { 'name': f'0.05539.b','weight': 0.02 },
         { 'name': f'0.05539.c','weight': 0.02 },
         { 'name': f'0.05539.d','weight': 0.02 },
         { 'name': f'0.05539.e','weight': 0.92 },]
}
df_cross = h_blend ( params, color='alls5', figures1=True, details=True)

display_distances  ( params )


df_cross.to_csv('submission.csv',index=False)
df_cross

