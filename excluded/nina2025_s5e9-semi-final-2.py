file_names_semi_final_2 = [ 
    '26.38202',   '26.38309',  '26.38441',  '26.4572cv',  
    '26.38482',   '26.38549',  '26.38519',  '26.4562cv',  '26.37987',
]

import shutil

def copy_file_to_Kaggle_working(file_name):
    dst_path = '/kaggle/working/'+file_name+'.csv'
    src_path = '/kaggle/input/25-september-2025-ps-s5e9/submission_'+file_name+'.csv'
    shutil.copy(src_path, dst_path)

for file in file_names_semi_final_2: copy_file_to_Kaggle_working(file)

path = '/kaggle/working/' 


import ast
import numpy as np
import pandas as pd

from bokeh.plotting import figure, gridplot 
from bokeh.io import output_file, show, output_notebook
output_notebook()


def color_scheme(color):
    colors    = ['red','green','blue']
    clr_alls  = ['crimson',"forestgreen",'mediumblue',"silver",'gold']
    clr_alls2 = ['red',"green",'blue',"silver",'gold']
    clr_alls3 = ['darkmagenta',"forestgreen",'mediumblue']
    clr_Red   = ["firebrick","orangered","crimson",'tomato',"red"]
    clr_Red4  = ["firebrick","orangered","crimson",'tomato']
    clr_Green = ["darkgreen","limegreen","green","forestgreen",'lime']
    clr_Green2= ['olivedrab',"darkgreen","forestgreen"]
    clr_Green3= ["darkmagenta",'olivedrab',"darkgreen"]
    clr_Blue  = ['midnightblue',"royalblue","mediumblue","blue","steelblue",'cyan']
    clr_Blue4 = ['midnightblue',"royalblue","mediumblue","steelblue"]
    clr_Brown = ["maroon","sienna","chocolate","sandybrown",'brown']
    clr_Brown3= ["maroon","sienna","sandybrown"]
    clr_Two   = ['crimson','mediumblue']
    clr_Two2  = ['crimson','darkgreen']
    clr_tes9  = clr_Red4 + ['darkmagenta'] + clr_Blue4
    clr_tes10 = ['darkmagenta'] + clr_Red + clr_Blue
    clr_tes11 = ['darkmagenta'] + clr_Brown + clr_Green
    if color == 'Two2':  colors = clr_Two2   [0:len(fs_names)]
    if color == 'Two':   colors = clr_Two    [0:len(fs_names)]
    if color == 'alls':  colors = clr_alls   [0:len(fs_names)]
    if color == 'alls2': colors = clr_alls2  [0:len(fs_names)]
    if color == 'alls3': colors = clr_alls3  [0:len(fs_names)]
    if color == 'red':   colors = clr_Red    [0:len(fs_names)]
    if color == 'green': colors = clr_Green  [0:len(fs_names)]
    if color == 'green2':colors = clr_Green2 [0:len(fs_names)]
    if color == 'green3':colors = clr_Green3 [0:len(fs_names)]
    if color == 'blue':  colors = clr_Blue   [0:len(fs_names)]
    if color == 'brown': colors = clr_Brown  [0:len(fs_names)]
    if color == 'brown3':colors = clr_Brown3 [0:len(fs_names)]
    if color == 'tes9':  colors = clr_tes9   [0:len(fs_names)]
    if color == 'tes10': colors = clr_tes10  [0:len(fs_names)]
    if color == 'tes11': colors = clr_tes11  [0:len(fs_names)]
    return colors


def bokeh_show(
        nr,
        params,
        df_cross,
        colors, 
        show_figures1, 
        show_figures2,
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
    f2 = figure(y_range=alls, height=height, width=274, title="relations of columns masses")
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
        f   = figure(width=800, height=250)
        f.title.text = 'Click on legend entries to mute the corresponding lines'
        b,e = 21000,21221
        line_x     = [dfs[i][b:e]['id']             for i in range(len(dfs))]
        line_y     = [dfs[i][b:e]['BeatsPerMinute'] for i in range(len(dfs))]
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


def v_blend(nr,
            path_to_ds, 
            file_short_names, 
            params, 
            type_sort,
            color,
            show_figures1=False, show_figures2=False, show_details=False,
            color_cross='yellow'):
    
    dk = params

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

    colors = color_scheme(color)

    oo = nr == 0 and type_sort == 'random'

    if oo:
        bokeh_show(nr, dk, da, colors, show_figures1, show_figures2, color_cross)
    
    return  da


# Cage procedure to follow:
# 1. https://www.kaggle.com/code/mehrankazeminia/s5e9-songs-rmse-cage
# 2. https://www.kaggle.com/code/somepatt/a-little-bit-of-iterations-lb-26-38097

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def MKcage3(FiN_import, n_iter=5, ks1=[1.10, 0.10], ks2=[1.05, 0.05]):
    sub_sample = pd.read_csv('../input/playground-series-s5e9/sample_submission.csv') 
    sub_import = pd.read_csv(FiN_import) 
    per = sub_import['BeatsPerMinute'].values
    # ..................................................................................................
    sns.set()
    plt.figure(figsize=(5, 2))
    plt.hist(per, bins=80)
    plt.gca().set_facecolor('mintcream')
    plt.suptitle('Before | BeatsPerMinute', y=0.96, fontsize=12, c='navy')
    # ..................................................................................................
    print('- - - - - - - ',FiN_import)
    min_per  = np.min(per);  print('Min:',  round(min_per, 3))
    max_per  = np.max(per);  print('Max:',  round(max_per, 3))
    mean_per = np.mean(per); print('Mean:', round(mean_per,3))
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
    plt.suptitle('After | BeatsPerMinute', y=0.96, fontsize=11, c='navy')
    # .......................................................................
    min_per = np.min(per);   print('Min:',  round(min_per, 3))
    max_per = np.max(per);   print('Max:',  round(max_per, 3))
    mean_per = np.mean(per); print('Mean:', round(mean_per,3)); 
    # .......................................................................
    print('- - - - - - - ', 'Cage '+FiN_import, '\n')
    # .......................................................................
    sub_sample['BeatsPerMinute'] = per
    return sub_sample 


def MKcage5(file_name_to_Cage):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import warnings; warnings.filterwarnings('ignore')
    
    sub_import = pd.read_csv(file_name_to_Cage)
    per = sub_import['BeatsPerMinute'].values
    # ...................................................................................
    sns.set()
    plt.figure(figsize=(5, 2))
    plt.hist(per, bins=80)
    plt.gca().set_facecolor('mintcream')
    plt.suptitle('Before | BeatsPerMinute', y=0.96, fontsize=13, c='navy')
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
    plt.suptitle('After | BeatsPerMinute', y=0.96, fontsize=12, c='navy')
    # .......................................................................
    min_per  = np.min (per) ;print('Min:',  round(min_per, 3))
    max_per  = np.max (per) ;print('Max:',  round(max_per, 3))
    mean_per = np.mean(per) ;print('Mean:', round(mean_per,3))
    # -----------------------------------------------------------------------
    # After the after
    # -----------------------------------------------------------------------
    for i in range(len(per)): 
        if per[i] < (min_per+7): per[i] = per[i] ** 0.994
        if per[i] > (max_per-9): per[i] = per[i] ** 1.006
    # .......................................................................
    sns.set()
    plt.figure(figsize=(5, 2))
    plt.hist(per, bins=80)
    plt.gca().set_facecolor('ghostwhite')
    plt.suptitle('After the after | BeatsPerMinute', y=0.96, fontsize=11, c='navy')
    # .......................................................................
    min_per  = np.min (per) ;print('Min:',  round(min_per, 3))
    max_per  = np.max (per) ;print('Max:',  round(max_per, 3))
    mean_per = np.mean(per) ;print('Mean:', round(mean_per,3))
    # .......................................................................
    df_sample = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
    df_sample['BeatsPerMinute'] = per
    return df_sample


def MKcage6(file_name_to_cage):
    sub_import = pd.read_csv(file_name_to_cage) 
    per = sub_import['BeatsPerMinute'].values
    # .......................................................................
    sns.set()
    plt.figure(figsize=(7, 4))
    plt.hist(per, bins=80)
    plt.gca().set_facecolor('lightgreen')
    plt.suptitle('Before | BeatsPerMinute', y=0.96, fontsize=16, c='navy')
    # .......................................................................
    min_per  = np.min( per) ;print('Min:',          round(min_per, 3))
    max_per  = np.max (per) ;print('Max:',          round(max_per, 3))
    mean_per = np.mean(per) ;print('Mean:',         round(mean_per,3))
    ptp_per  = np.ptp (per) ;print('Peak to Peak:', round(ptp_per, 3))

    def new_range(per, old_min, old_max, new_min, new_max):
        if (old_max == old_min): 
            return [new_min for p in per]
        new_per = []
        for p in per:
            if (old_min+7) < p < (old_max-9):
                new_per.append(p)
            else:
                percentage = (p - old_min) / (old_max - old_min)
                new_p = new_min + (percentage * (new_max - new_min))
                new_per.append(new_p)   
        return new_per
    # .......................................................................
    per = new_range(per, min_per, max_per, 97.5, 152.0)
    print(len(per))
    sns.set()
    plt.figure(figsize=(7, 4))
    plt.hist(per, bins=80)
    plt.gca().set_facecolor('pink')
    plt.suptitle('After | BeatsPerMinute', y=0.96, fontsize=16, c='navy')
    # .......................................................................
    min_per    = np.min (per)  ;print('Min:',          round(min_per, 3))
    max_per    = np.max (per)  ;print('Max:',          round(max_per, 3))
    mean_per   = np.mean(per)  ;print('Mean:',         round(mean_per,3))
    ptp_per    = np.ptp (per)  ;print('Peak to Peak:', round(ptp_per, 3))
    sub_sample = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
    sub_sample['BeatsPerMinute'] = per
    return sub_sample
    

def r_blend(qnt_iter, list_wts, path, fs_name, params, color):
    rs_lena = qnt_iter * len(list_wts[0]) * len(list_wts[1])
    n = 0
    for i in range(len(list_wts[0])):
        for j in  range(len(list_wts[1])):
            pv0  = list_wts[0][i]
            pv1  = list_wts[1][j]
            p0   =      pv0        / 100
            p1   =            pv1  / 100
            p2   = 1 - (pv0 + pv1) / 100
            for p in range(len(params['subm'])):
                params['subm'][p]['weight'] = [p0,p1,p2][p]
            for q in range(qnt_iter):
                df_R_cross = v_blend(n,path,fs_names,params,type_sort='random',color=color)
                df_R_cross.to_csv(f'submission_R_cross{n}.csv', index=False)
                n += 1    # ;print(df_R_cross.head(3)) ;print('~'*49, " ", n)
    dfs = [pd.read_csv(f'submission_R_cross{n}.csv') for n in range(rs_lena)]
    df_sample = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
    df_sample["BeatsPerMinute"] = sum([df["BeatsPerMinute"] for df in dfs]) / rs_lena
    for n in range(rs_lena): os.remove(f'submission_R_cross{n}.csv')
    return df_sample
    

def straight_blend2(file_name1, file_name2, wts=[0.70, 0.30]):
    df1 = pd.read_csv(file_name1)
    df2 = pd.read_csv(file_name2)
    df2['BeatsPerMinute'] =\
        df1['BeatsPerMinute'] *wts[0] +\
        df2['BeatsPerMinute'] *wts[1]
    return df2
    

def straight_blend3(file_name2, file_name1, file_name3, wts=[0.07, 0.03, 0.90]):
    df2 = pd.read_csv(file_name2)
    df1 = pd.read_csv(file_name1)
    df3 = pd.read_csv(file_name3)
    df3['BeatsPerMinute'] =\
        df2['BeatsPerMinute'] *wts[0] +\
        df1['BeatsPerMinute'] *wts[1] +\
        df3['BeatsPerMinute'] *wts[2]
    return df3


%%time

fs_names = ['26.38202','26.38482']

params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "BeatsPerMinute",                  
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [ +0.04,-0.04 ],       
      'subm'   : [
         { 'name':fs_names[0], 'weight':+0.70 },
         { 'name':fs_names[1], 'weight':+0.30 },]
}

qnt_iter, list_wts = 11, [[70],[30]]

df1_G1 = r_blend (qnt_iter, list_wts, path, fs_names, params, color='brown')

df1_G1 . to_csv  ('blend_G1.csv', index=False)  ;display(df1_G1.head(3))

df2_G1 = MKcage3 ('blend_G1.csv')

df2_G1 . to_csv  ('G1.csv',       index=False)  ;display(df2_G1.head(3))


%%time

fs_names = ['G1','26.38309','26.38549']

params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "BeatsPerMinute",                  
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [ +0.03,-0.01,-0.02 ],       
      'subm'   : [
         { 'name':fs_names[0], 'weight':+0.74 },
         { 'name':fs_names[1], 'weight':+0.19 },
         { 'name':fs_names[2], 'weight':+0.07 },]
}

qnt_iter, list_wts = 11, [[74,77],[18,19]]

df1_G2 = r_blend (qnt_iter, list_wts, path, fs_names, params, color='red')

df1_G2 . to_csv  ('blend_G2.csv',index=False)  ;display(df1_G2.head(3))

df2_G2 = MKcage3 ('blend_G2.csv')

df2_G2 . to_csv  ('G2.csv',      index=False)  ;display(df2_G2.head(3))


%%time

fs_names = ['G2','26.38441','26.38519']

params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "BeatsPerMinute",                  
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [ +0.05,-0.02,-0.03 ],       
      'subm'   : [
         { 'name':fs_names[0], 'weight':+0.80 },
         { 'name':fs_names[1], 'weight':+0.12 },
         { 'name':fs_names[2], 'weight':+0.08 },]
}

qnt_iter, list_wts = 11, [[78,80],[12,14]]

df1_G3 = r_blend (qnt_iter, list_wts, path, fs_names, params, color='green')

df1_G3 . to_csv  ('blend_G3.csv',index=False)  ;display(df1_G3.head(3))

df2_G3 = MKcage3 ('blend_G3.csv')

df2_G3 . to_csv  ('G3.csv',      index=False)  ;display(df2_G3.head(3))


%%time

fs_names = ['G3','26.4572cv','26.4562cv']

params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "BeatsPerMinute",                  
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [ +0.03,-0.02,-0.01 ],       
      'subm'   : [
         { 'name':fs_names[0], 'weight':+0.90 },
         { 'name':fs_names[1], 'weight':+0.07 },
         { 'name':fs_names[2], 'weight':+0.03 },]
}

qnt_iter, list_wts = 11, [[90,92],[4,5]]

df1_G4 = r_blend (qnt_iter, list_wts, path, fs_names, params, color='blue')

df1_G4 . to_csv  ('blend_G4.csv',index=False)  ;display(df1_G4.head(3))

df2_G4 = MKcage3 ('blend_G4.csv')

df2_G4 . to_csv  ('G4.csv',      index=False)  ;display(df2_G4.head(3))



# G4 = {G3,g41,g42} <- G3.{G2,g31,g32} <- G2.{G1,g21,g22} <- G1.{g11,g12}


df_G5 = straight_blend2('/kaggle/working/26.37987.csv', 'G4.csv', wts=[0.95, 0.05])
df_G5 . to_csv('G5.csv', index=False)
display(df_G5)

for file in file_names_semi_final_2: os.remove(file +'.csv')

df_MKC3 = MKcage3('/kaggle/working/G5.csv');  df_MKC3.to_csv('MKC3.csv',index=False)
df_MKC5 = MKcage5('/kaggle/working/G5.csv');  df_MKC5.to_csv('MKC5.csv',index=False)
df_MKC6 = MKcage6('/kaggle/working/G5.csv');  df_MKC6.to_csv('MKC5.csv',index=False)

df = df_MKC3  # LB = 26.38_133      G5 = 26.37987 x G4  <- wts.[0.70, 0.30]
df = df_MKC3  # LB = 26.38_064      G5 = 26.37987 x G4  <- wts.[0.95, 0.05]
df = df_MKC5  # LB = 26.37_999      G5 = 26.37987 x G4  <- wts.[0.95, 0.05]
df = df_MKC6  # LB = ?              G5 = 26.37987 x G4  <- wts.[0.95, 0.05]

df.to_csv('submission.csv', index=False)

df

