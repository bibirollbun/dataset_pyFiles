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


def v_blend(path_to_ds, 
            file_short_names, 
            params, 
            type_sort,
            color,
            show_figures1=False, show_figures2=False, show_details=False,
            color_cross='brown'):
    
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

    bokeh_show(dk, da, colors, show_figures1, show_figures2, color_cross)
    
    return  da


# procedure to follow:
# 1. https://www.kaggle.com/code/mehrankazeminia/s5e9-songs-rmse-cage
# 2. https://www.kaggle.com/code/somepatt/a-little-bit-of-iterations-lb-26-38097


def straight_blend3(file_name2, file_name1, file_name3, wts=[0.07, 0.03, 0.90]):
    df2 = pd.read_csv(file_name2)
    df1 = pd.read_csv(file_name1)
    df3 = pd.read_csv(file_name3)
    df3['BeatsPerMinute'] =\
        df2['BeatsPerMinute'] *0.07 +\
        df1['BeatsPerMinute'] *0.03 +\
        df3['BeatsPerMinute'] *0.90
    return df3
    

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def f_st2cage(FiN_import, n_iter=5, ks1=[1.10, 0.10], ks2=[1.05, 0.05]):
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


def Cage_by_MehranKazeminia(file_name_to_Cage):
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
    plt.suptitle('After the after | BeatsPerMinute', y=0.96, fontsize=11, c='navy')
    # .......................................................................
    min_per  = np.min (per) ;print('Min:',  round(min_per, 3))
    max_per  = np.max (per) ;print('Max:',  round(max_per, 3))
    mean_per = np.mean(per) ;print('Mean:', round(mean_per,3))
    
    # -------------------------------------------------------------------------
    # semi-final
    # -------------------------------------------------------------------------
    
    df_semi_final =\
        pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
    
    df_semi_final['BeatsPerMinute'] = per
    
    return df_semi_final


path = '/kaggle/input/24-september-2025-ps-s5e9/submission_'

fs_names = ['26.38299','26.38494','26.38547']

params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "BeatsPerMinute",                  
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [ +0.14,-0.04,-0.10 ],       
      'subm'   : [
         { 'name':fs_names[0], 'weight':+0.47 },
         { 'name':fs_names[1], 'weight':+0.32 },
         { 'name':fs_names[2], 'weight':+0.21 },]
}
df1_G1 = v_blend(
    path, fs_names, 
    params, type_sort='asc/desc',
    color='red', show_figures1=True, show_figures2=True, show_details=True)

df1_G1.to_csv('blend_Group_1.csv', index=False)
display(df1_G1)

df2_G1 = f_st2cage('blend_Group_1.csv')
df2_G1.to_csv('Cage_blend_Group_1.csv',index=False)
display(df2_G1)


path = '/kaggle/input/24-september-2025-ps-s5e9/submission_'

fs_names = ['26.38482','26.38493','26.38531']

params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "BeatsPerMinute",                  
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [ +0.11,-0.04,-0.07 ],       
      'subm'   : [
         { 'name':fs_names[0], 'weight':+0.37 },
         { 'name':fs_names[1], 'weight':+0.33 },
         { 'name':fs_names[2], 'weight':+0.30 },]
}
df1_G2 = v_blend(
    path, fs_names, 
    params, type_sort='asc/desc',
    color='green', show_figures1=True, show_figures2=True, show_details=True)

df1_G2.to_csv('blend_Group_2.csv', index=False)
display(df1_G2)

df2_G2 = f_st2cage('blend_Group_2.csv')
df2_G2.to_csv('Cage_blend_Group_2.csv',index=False)
display(df2_G2)


path = '/kaggle/input/24-september-2025-ps-s5e9/submission_'

fs_names = ['26.38382','26.38438','26.38518']

params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "BeatsPerMinute",                  
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [ +0.11,-0.04,-0.07 ],       
      'subm'   : [
         { 'name':fs_names[0], 'weight':+0.45 },
         { 'name':fs_names[1], 'weight':+0.32 },
         { 'name':fs_names[2], 'weight':+0.23 },]
}
df1_G3 = v_blend(
    path, fs_names, 
    params, type_sort='asc/desc',
    color='blue', show_figures1=True, show_figures2=True, show_details=True)

df1_G3.to_csv('blend_Group_3.csv', index=False)
display(df1_G3)

df2_G3 = f_st2cage('blend_Group_3.csv')
df2_G3.to_csv('Cage_blend_Group_3.csv',index=False)
display(df2_G3)


path = '/kaggle/working/blend_'

fs_names = ['Group_1', 'Group_2', 'Group_3']

params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "BeatsPerMinute",                  
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [ +0.21,-0.07,-0.14 ],       
      'subm'   : [
         { 'name':fs_names[0], 'weight':+0.33 },
         { 'name':fs_names[1], 'weight':+0.30 },
         { 'name':fs_names[2], 'weight':+0.37 },]
}
df123_blend_of_blend = v_blend(
    path, fs_names, 
    params, type_sort='asc/desc',
    color='alls', show_figures1=True, show_figures2=True, show_details=True)

df123_blend_of_blend.to_csv('blend_of_blend_Groups.(123).csv', index=False)
display(df123_blend_of_blend)

#----------------------------------------------------------------------------

path = '/kaggle/working/Cage_blend_'

fs_names = ['Group_1', 'Group_2', 'Group_3']

params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "BeatsPerMinute",                  
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [ +0.21,-0.07,-0.14 ],       
      'subm'   : [
         { 'name':fs_names[0], 'weight':+0.33 },
         { 'name':fs_names[1], 'weight':+0.30 },
         { 'name':fs_names[2], 'weight':+0.37 },]
}
df123_blend_of_Cage = v_blend(
    path, fs_names, 
    params, type_sort='asc/desc',
    color='alls2', show_figures1=True, show_figures2=True, show_details=True)

df123_blend_of_Cage.to_csv('blend_of_Cage__Groups.(123).csv', index=False)
display(df123_blend_of_Cage)


df_CAGE_blend_of_blend = Cage_by_MehranKazeminia('blend_of_blend_Groups.(123).csv')
df_CAGE_blend_of_blend . to_csv('df_CAGE_blend_of_blend.csv',index=False)
print('\n')
df_CAGE_blend_of_Cage  = Cage_by_MehranKazeminia('blend_of_Cage__Groups.(123).csv')
df_CAGE_blend_of_Cage  . to_csv('df_CAGE_blend_of_Cage_.csv', index=False)


def straight_blend(file_name1, file_name2, wts=[0.70, 0.30]):
    df1 = pd.read_csv(file_name1)
    df2 = pd.read_csv(file_name2)
    df1['BeatsPerMinute'] = df1['BeatsPerMinute'] *0.70 + 0.30* df2['BeatsPerMinute']
    return df1


FiN_26_37992 = '/kaggle/input/24-september-2025-ps-s5e9/submission_26.37992.csv'
    

df_blend_of_blend = straight_blend(FiN_26_37992, 'blend_of_blend_Groups.(123).csv')
df_blend_of_Cage  = straight_blend(FiN_26_37992, 'blend_of_Cage__Groups.(123).csv')

df_CAGE_blend_of_blend = straight_blend(FiN_26_37992, 'df_CAGE_blend_of_blend.csv')
df_CAGE_blend_of_Cage  = straight_blend(FiN_26_37992, 'df_CAGE_blend_of_Cage_.csv')

display(
    df_blend_of_blend      .head(4),
    df_blend_of_Cage       .head(4),
    df_CAGE_blend_of_blend .head(4),
    df_CAGE_blend_of_Cage  .head(4)
)


option = 'option.1' # LB = 26.38019
option = 'option.2' # LB = 26.38017
option = 'option.3' # LB = 26.40476
option = 'option.4' # LB = ?
option = 'option.5' # LB = ?


df_blend_of_blend.to_csv('blend_of_blend.csv', index=False)

if option == 'option.1': 
    df_subm = df_blend_of_blend


df_blend_of_Cage.to_csv('blend_of_Cage.csv', index=False)

if option == 'option.2':
    df_subm = df_blend_of_Cage


df_CAGE_blend_of_blend.to_csv('CAGE_blend_of_blend.csv', index=False)

if option == 'option.3':
    df_subm = df_CAGE_blend_of_blend


df_CAGE_blend_of_Cage.to_csv('CAGE_blend_of_Cage.csv', index=False)

if option == 'option.4':
    df_subm = df_CAGE_blend_of_Cage


df5 = straight_blend3('blend_of_Cage.csv','blend_of_blend.csv', FiN_26_37992)

if option == 'option.5':
    df_subm = df5


df_subm.to_csv("submission.csv", index=False)

df_subm

