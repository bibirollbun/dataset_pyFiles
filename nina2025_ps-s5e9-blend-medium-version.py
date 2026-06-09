import ast
import numpy as np
import pandas as pd

from bokeh.plotting import figure, gridplot
from bokeh.io import output_file, show, output_notebook
output_notebook()


# Funcs: v_blend, bokeh_show

def color_scheme(color):
    colors    = ['red','green','blue']
    clr_alls  = ['crimson',"forestgreen",'mediumblue',"darkmagenta",'gold']
    clr_alls3 = ['darkmagenta',"forestgreen",'mediumblue']
    clr_Red   = ["firebrick","orangered","crimson",'tomato',"red"]
    clr_Red4  = ["firebrick","orangered","crimson",'tomato']
    clr_Green = ["green","darkgreen","limegreen","forestgreen",'lime']
    clr_Green2= ['olivedrab',"darkgreen","forestgreen"]
    clr_Green3= ["darkmagenta",'olivedrab',"darkgreen"]
    clr_Blue  = ['midnightblue',"royalblue","mediumblue","blue","steelblue"]
    clr_Blue4  = ['midnightblue',"royalblue","mediumblue","steelblue"]
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
        show_figures2):
    
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
        width = 151  if len(colors) == 5\
            else (121 if len(colors) == 8\
            else (131 if len(colors) == 9\
            else (141 if len(colors) == 10\
            else (171 if len(colors) == 11 else 185))))
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
        f   = figure(width=800, height=300)
        f.title.text = 'Click on legend entries to mute the corresponding lines'
        b,e = 21000,21221
        line_x     = [dfs[i][b:e]['id']             for i in range(len(dfs))]
        line_y     = [dfs[i][b:e]['BeatsPerMinute'] for i in range(len(dfs))]
        color      = colors + ['black']
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
            color, 
            show_figures1=False, show_figures2=False, show_details=False):
    
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
        
        def alls(x, sd=sorting_direction,cs=cols):
            reverse = True if sd=='desc' else False
            tes = {c: x[c] for c in cs}.items()
            subms_sorted = [t[0] for t in sorted(tes,key=lambda k:k[1],reverse=reverse)]
            return subms_sorted
            
        def summa(x,cs,wts,ic_alls): 
            return sum([x[cs[j]] * (wts[0][j] + wts[1][ic_alls[j]]) for j in range(len(cs))])
            
        wts = [[[e['weight'] for e in dk["subm"]], [w for w in dk["subwts" ]]]]
        
        if "subm2" in dk and "subwts2" in dk:
            wts.append([[e['weight'] for e in dk["subm2"]],[w for w in dk["subwts2"]]])
        if "subm3" in dk and "subwts3" in dk:
            wts.append([[e['weight'] for e in dk["subm3"]],[w for w in dk["subwts3"]]])
          
        def correct(x, cs=cols, wts=wts):
            i = [x['alls'].index(c) for c in short_name_cols]
            if len(wts) == 3:
                if   0.11 < x['mx-m'] <= 0.21: return summa(x,cs,wts[2],i)
                if   0.21 < x['mx-m'] <= 0.33: return summa(x,cs,wts[1],i)
            if len(wts) == 2:
                if   0.21 < x['mx-m'] <= 0.33: return summa(x,cs,wts[1],i)
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
        if show_details: display(df_subms.head(4))
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

    bokeh_show(dk,da,colors,show_figures1,show_figures2)
    
    return  da


# previous events, LB = 26.38_091

# # collecting files and moving them into one folder '/kaggle/working/subms/'

# folder = '/kaggle/working/subms/'

# path04 = '/kaggle/input/4-september-2025-ps-s5e9/'
# path07 = '/kaggle/input/7-september-2025-ps-s5e9/'
# path09 = '/kaggle/input/9-september-2025-ps-s5e9/'
# path12 = '/kaggle/input/12-september-2025-ps-s5e9/'

# import os, shutil

# if not os.path.isdir(folder): os.mkdir(folder)

# shutil.copyfile(path12 + 'submission_26.38124.csv', folder + '26.38_124.csv')
# shutil.copyfile(path09 + 'submission_26.38299.csv', folder + '26.38_299.csv')
# shutil.copyfile(path09 + 'submission_26.38309.csv', folder + '26.38_309.csv')
# shutil.copyfile(path09 + 'submission_26.4572cv.csv',folder + '26.4572cv.csv')
# shutil.copyfile(path12 + 'submission_26.38482.csv', folder + '26.38_482.csv')
# shutil.copyfile(path12 + 'submission_26.38124.csv', folder + '26.38_547.csv')
# shutil.copyfile(path12 + 'GROUPS_G.csv',            folder + '26.38_304.csv')
# shutil.copyfile(path12 + 'GROUPS_H.csv',            folder + '26.38_306.csv')
# shutil.copyfile(path04 + 'submission_26.38519.csv', folder + '26.38_519.csv')
# shutil.copyfile(path12 + 'submission_26.38509.csv', folder + '26.38_509.csv')
# shutil.copyfile(path07 + 'submission_26.38441.csv', folder + '26.38_441.csv')

# print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

# path = folder

# fs_names = ['26.38_124','26.38_299','26.38_304','26.38_306','26.38_309','26.38_441',
#                         '26.38_482','26.38_509','26.38_519','26.38_547','26.4572cv']
# params = {
#       'path'   : path,
#       'id'     : 'id',                 
#       'target' : "BeatsPerMinute",                  
#       'desc'   : 0.70,
#       'asc'    : 0.30,
#       'subwts' : [ +0.0008,+0.0005,+0.0003,+0.0001, 0,0,
#                    -0.0001,-0.0002,-0.0003,-0.0004,-0.0007],                 
#       'subm'   : [
#          { 'name':fs_names[0], 'weight':+0.9954 },
#          { 'name':fs_names[1], 'weight':+0.0007 },
#          { 'name':fs_names[2], 'weight':+0.0007 },
#          { 'name':fs_names[3], 'weight':+0.0007 },
#          { 'name':fs_names[4], 'weight':+0.0007 },
#          { 'name':fs_names[5], 'weight':+0.0004 }, 
#          { 'name':fs_names[6], 'weight':+0.0004 },
#          { 'name':fs_names[7], 'weight':+0.0004 },
#          { 'name':fs_names[8], 'weight':+0.0004 },
#          { 'name':fs_names[9], 'weight':+0.0003 },
#          { 'name':fs_names[10],'weight':+0.0003 },
#       ]
# }

# df_cross = v_blend (path, 
#                     fs_names, 
#                     params, 
#                     color='tes10', 
#                     show_figures1=True, show_figures2=True, show_details=True)

# df_cross.to_csv('submission_1.csv', index=False)

# df_cross

# print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

# # procedure to follow:
# # 1. https://www.kaggle.com/code/mehrankazeminia/s5e9-songs-rmse-cage
# # 2. https://www.kaggle.com/code/somepatt/a-little-bit-of-iterations-lb-26-38097

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np

# def iter10(file_name_submission, ks1=[1.1, 0.1], ks2=[1.0, 0.0]):
#     sub_sample = pd.read_csv('../input/playground-series-s5e9/sample_submission.csv') 
#     sub_import = pd.read_csv('/kaggle/input/12-september-2025-ps-s5e9/submission_26.38135.csv') 
#     per = sub_import['BeatsPerMinute'].values
#     # ..................................................................................................
#     sns.set()
#     plt.figure(figsize=(5, 2))
#     plt.hist(per, bins=80)
#     plt.gca().set_facecolor('lightgreen')
#     plt.suptitle('Before | BeatsPerMinute', y=0.96, fontsize=16, c='navy')
#     # ..................................................................................................
#     print('- - - - - - - ',file_name_submission)
#     min_per  = np.min(per);  print('Min:',  round(min_per, 3))
#     max_per  = np.max(per);  print('Max:',  round(max_per, 3))
#     mean_per = np.mean(per); print('Mean:', round(mean_per,3))
#     print('-------')
#     R = -0.0
#     guide = mean_per - R
#     # ....................................
#     per1 = [f for f in per if f < guide]
#     per2 = [f for f in per if f > guide]
#     len(per1), len(per2)
    
#     N = 5
#     for _ in range(N):
#         for i in range(len(per)):
#             per_guide = (per[i] + guide) / 2            
#             if per[i] <= guide:
#                 per[i] = (per[i] *ks1[0]) - (per_guide *ks1[1])
#             else:
#                 per[i] = (per[i] *ks2[0]) - (per_guide *ks2[1])
#     # .......................................................................
#     sns.set()
#     plt.figure(figsize=(5, 2))
#     plt.hist(per, bins=80)
#     plt.gca().set_facecolor('pink')
#     plt.suptitle('After | BeatsPerMinute', y=0.96, fontsize=16, c='navy')
#     # .......................................................................
#     min_per = np.min(per);   print('Min:',  round(min_per, 3))
#     max_per = np.max(per);   print('Max:',  round(max_per, 3))
#     mean_per = np.mean(per); print('Mean:', round(mean_per,3)); print('- '*17, '\n')
#     # .......................................................................
#     sub_sample['BeatsPerMinute'] = per
#     sub_sample.to_csv(file_name_submission, index=False) 
#     sub_sample 
    

# iter10('26.38_097_1.csv',  ks1=[1.11, 0.08], ks2=[1.0, 0.0])
# iter10('26.38_097_2.csv',  ks1=[1.12, 0.09], ks2=[1.0, 0.0])
# iter10('26.38_097_3.csv',  ks1=[1.13, 0.10], ks2=[1.0, 0.0])
# iter10('26.38_097_4.csv',  ks1=[1.14, 0.11], ks2=[1.0, 0.0])
# iter10('26.38_097_5.csv',  ks1=[1.15, 0.12], ks2=[1.0, 0.0])

# iter10('26.38_097_6.csv',  ks1=[1.11, 0.08], ks2=[0.95, 0.08])
# iter10('26.38_097_7.csv',  ks1=[1.12, 0.09], ks2=[0.97, 0.09])
# iter10('26.38_097_8.csv',  ks1=[1.13, 0.10], ks2=[0.98, 0.10])
# iter10('26.38_097_9.csv',  ks1=[1.14, 0.11], ks2=[0.99, 0.11])
# iter10('26.38_097_10.csv', ks1=[1.15, 0.12], ks2=[1.00, 0.12])

# top = pd.read_csv('/kaggle/input/12-september-2025-ps-s5e9/submission_26.38097.csv')
# top.to_csv('26.38_097.csv',index=False)

# print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

# path = '/kaggle/working/'

# fs_names = ['26.38_097','26.38_097_1','26.38_097_2','26.38_097_3','26.38_097_4',
#             '26.38_097_5','26.38_097_6','26.38_097_7','26.38_097_8','26.38_097_9',
#             '26.38_097_10']

# params = {
#       'path'   : path,
#       'id'     : 'id',                 
#       'target' : "BeatsPerMinute",                  
#       'desc'   : 0.70,
#       'asc'    : 0.30,
#       'subwts' : [ +0.0021,+0.0011,+0.0007,+0.0003, 0,
#                    -0.0001,-0.0003,-0.0005,-0.0007,-0.0011,-0.0015,],       
#       'subm'   : [
#          { 'name':fs_names[0], 'weight':+0.9935 },
#          { 'name':fs_names[1], 'weight':+0.0008 },
#          { 'name':fs_names[2], 'weight':+0.0008 },
#          { 'name':fs_names[3], 'weight':+0.0008 },
#          { 'name':fs_names[4], 'weight':+0.0008 },
#          { 'name':fs_names[5], 'weight':+0.0008 },
#          { 'name':fs_names[6], 'weight':+0.0005 },
#          { 'name':fs_names[7], 'weight':+0.0005 },
#          { 'name':fs_names[8], 'weight':+0.0005 },
#          { 'name':fs_names[9], 'weight':+0.0005 },
#          { 'name':fs_names[10],'weight':+0.0005 },
#       ]
# }

# df_cross = v_blend (path, 
#                     fs_names, 
#                     params, 
#                     color='tes11', 
#                     show_figures1=True, show_figures2=True, show_details=True)

# df_cross.to_csv('submission_2.csv', index=False)

# df_cross

# print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

# # Submission

# subm_1 = pd.read_csv('submission_1.csv')
# subm_2 = pd.read_csv('submission_2.csv')

# submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')

# target = 'BeatsPerMinute'

# submission[target] = subm_1[target] *0.005 + 0.995*subm_2[target]
# submission.to_csv('submission.csv', index=False) 
# display(submission) 


def generator(FiN_subm, FiN_dist, k1,k2):
    sub_import = pd.read_csv(FiN_dist)
    per = sub_import['BeatsPerMinute'].values
    min_per,max_per,mean_per = np.min(per),np.max(per),np.mean(per)
    print(f'Min:{round(min_per,3)}, Max:{round(max_per,3)}, Mean:{round(mean_per,3)}')
    guide = mean_per - 0.0 # Adjusting the 0.0 value can increase the accuracy of the guide.
    # ....................................
    per1 = [f for f in per if f < guide]
    per2 = [f for f in per if f > guide]
    print(f'length_per1:{len(per1)}, length_per1:{len(per2)}')
    for i in range(len(per)):     
        if per[i] < (min_per+7): per[i] = per[i] ** k1 # 0.994
        if per[i] > (max_per-9): per[i] = per[i] ** k2 # 1.006
    # ....................................        
    min_per,max_per,mean_per = np.min(per),np.max(per),np.mean(per)
    print(f'Min:{round(min_per,3)}, Max:{round(max_per,3)}, Mean:{round(mean_per,3)}')
    print(f'.................................... {FiN_subm}\n')
    sub_sample = pd.read_csv('../input/playground-series-s5e9/sample_submission.csv')
    sub_sample['BeatsPerMinute'] = per
    sub_sample.to_csv(FiN_subm+'.csv', index=False)


FiN_dist1 = '/kaggle/input/14-september-2025-ps-s5e9/submission_26.38089.csv'

FiN_dist2 = '/kaggle/input/14-september-2025-ps-s5e9/submission_26.38091.csv'
FiN_dist3 = '/kaggle/input/12-september-2025-ps-s5e9/submission_26.38094.csv'

generator('26.38_078_11', FiN_dist1, 0.9944, 1.0056)
generator('26.38_078_12', FiN_dist1, 0.9943, 1.0057)
generator('26.38_078_13', FiN_dist1, 0.9942, 1.0058)
generator('26.38_078_14', FiN_dist1, 0.9941, 1.0059)
generator('26.38_078_15', FiN_dist1, 0.9940, 1.0060) # original
generator('26.38_078_16', FiN_dist1, 0.9939, 1.0061)
generator('26.38_078_17', FiN_dist1, 0.9938, 1.0062)
generator('26.38_078_18', FiN_dist1, 0.9937, 1.0063)
generator('26.38_078_19', FiN_dist1, 0.9936, 1.0064)

generator('26.38_078_21', FiN_dist2, 0.9936, 1.0056)
generator('26.38_078_22', FiN_dist2, 0.9937, 1.0057)
generator('26.38_078_23', FiN_dist2, 0.9938, 1.0058)
generator('26.38_078_24', FiN_dist2, 0.9939, 1.0059)
generator('26.38_078_25', FiN_dist2, 0.9940, 1.0060) # original
generator('26.38_078_26', FiN_dist2, 0.9941, 1.0061)
generator('26.38_078_27', FiN_dist2, 0.9942, 1.0062)
generator('26.38_078_28', FiN_dist2, 0.9943, 1.0063)
generator('26.38_078_29', FiN_dist2, 0.9944, 1.0064)

generator('26.38_078_31', FiN_dist3, 0.9944, 1.0056)
generator('26.38_078_32', FiN_dist3, 0.9943, 1.0057)
generator('26.38_078_33', FiN_dist3, 0.9942, 1.0058)
generator('26.38_078_34', FiN_dist3, 0.9941, 1.0059)
generator('26.38_078_35', FiN_dist3, 0.9940, 1.0060) # original
generator('26.38_078_36', FiN_dist3, 0.9939, 1.0061)
generator('26.38_078_37', FiN_dist3, 0.9938, 1.0062)
generator('26.38_078_38', FiN_dist3, 0.9937, 1.0063)
generator('26.38_078_39', FiN_dist3, 0.9936, 1.0064)


# v18,v19,v20 = LB = 26.38_077


# path = '/kaggle/working/'
# fs_names = ['26.38_078_11','26.38_078_12','26.38_078_13','26.38_078_14',
#             '26.38_078_15','26.38_078_16','26.38_078_17','26.38_078_18','26.38_078_19']
# params = {
#       'path'   : path,
#       'id'     : 'id',                 
#       'target' : "BeatsPerMinute",                  
#       'desc'   : 0.70,
#       'asc'    : 0.30,
#       'subwts' : [ +0.0014,+0.0007,+0.0004,+0.0002, 0, -0.0001,-0.0005,-0.0008,-0.0013],       
#       'subm'   : [
#          { 'name':fs_names[0], 'weight':+0.015 },
#          { 'name':fs_names[1], 'weight':+0.015 },
#          { 'name':fs_names[2], 'weight':+0.015 },
#          { 'name':fs_names[3], 'weight':+0.015 },
#          { 'name':fs_names[4], 'weight':+0.88 }, # original
#          { 'name':fs_names[5], 'weight':+0.015},
#          { 'name':fs_names[6], 'weight':+0.015 },
#          { 'name':fs_names[7], 'weight':+0.015 },
#          { 'name':fs_names[8], 'weight':+0.015 },
#       ]
# }
# df_cross = v_blend (path, fs_names, params, color='tes9', show_figures2=True)
# df_cross.to_csv('subm_1.csv', index=False)

# path = '/kaggle/working/'
# fs_names = ['26.38_078_21','26.38_078_22','26.38_078_23','26.38_078_24',
#             '26.38_078_25','26.38_078_26','26.38_078_27','26.38_078_28','26.38_078_29']
# params = {
#       'path'   : path,
#       'id'     : 'id',                 
#       'target' : "BeatsPerMinute",                  
#       'desc'   : 0.70,
#       'asc'    : 0.30,
#       'subwts' : [ +0.0014,+0.0007,+0.0004,+0.0002, 0, -0.0001,-0.0005,-0.0008,-0.0013],       
#       'subm'   : [
#          { 'name':fs_names[0], 'weight':+0.015 },
#          { 'name':fs_names[1], 'weight':+0.015 },
#          { 'name':fs_names[2], 'weight':+0.015 },
#          { 'name':fs_names[3], 'weight':+0.015 },
#          { 'name':fs_names[4], 'weight':+0.88 }, # original
#          { 'name':fs_names[5], 'weight':+0.015},
#          { 'name':fs_names[6], 'weight':+0.015 },
#          { 'name':fs_names[7], 'weight':+0.015 },
#          { 'name':fs_names[8], 'weight':+0.015 },
#       ]
# }
# df_cross = v_blend (path, fs_names, params, color='tes9', show_figures2=True)
# df_cross.to_csv('subm_2.csv', index=False)


# path = '/kaggle/working/'
# fs_names = ['26.38_078_31','26.38_078_32','26.38_078_33','26.38_078_34',
#             '26.38_078_35','26.38_078_36','26.38_078_37','26.38_078_38','26.38_078_39']
# params = {
#       'path'   : path,
#       'id'     : 'id',                 
#       'target' : "BeatsPerMinute",                  
#       'desc'   : 0.70,
#       'asc'    : 0.30,
#       'subwts' : [ +0.0014,+0.0007,+0.0004,+0.0002, 0, -0.0001,-0.0005,-0.0008,-0.0013],       
#       'subm'   : [
#          { 'name':fs_names[0], 'weight':+0.015 },
#          { 'name':fs_names[1], 'weight':+0.015 },
#          { 'name':fs_names[2], 'weight':+0.015 },
#          { 'name':fs_names[3], 'weight':+0.015 },
#          { 'name':fs_names[4], 'weight':+0.88 }, # original
#          { 'name':fs_names[5], 'weight':+0.015},
#          { 'name':fs_names[6], 'weight':+0.015 },
#          { 'name':fs_names[7], 'weight':+0.015 },
#          { 'name':fs_names[8], 'weight':+0.015 },
#       ]
# }
# df_cross = v_blend (path, fs_names, params, color='tes9', show_figures2=True)
# df_cross.to_csv('subm_3.csv', index=False)


path = '/kaggle/working/'
fs_names = ['26.38_078_11','26.38_078_12','26.38_078_13','26.38_078_14',
            '26.38_078_15','26.38_078_16','26.38_078_17','26.38_078_18','26.38_078_19']
params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "BeatsPerMinute",                  
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [ +0.0014,+0.0007,+0.0004,+0.0002, 0, -0.0001,-0.0005,-0.0008,-0.0013],       
      'subm'   : [
         { 'name':fs_names[0], 'weight':+0.025 },
         { 'name':fs_names[1], 'weight':+0.025 },
         { 'name':fs_names[2], 'weight':+0.025 },
         { 'name':fs_names[3], 'weight':+0.025 },
         { 'name':fs_names[4], 'weight':+0.80 }, # original
         { 'name':fs_names[5], 'weight':+0.025},
         { 'name':fs_names[6], 'weight':+0.025 },
         { 'name':fs_names[7], 'weight':+0.025 },
         { 'name':fs_names[8], 'weight':+0.025 },
      ]
}
df_cross = v_blend (path, fs_names, params, color='tes9', show_figures2=True)
df_cross.to_csv('subm_1.csv', index=False)

path = '/kaggle/working/'
fs_names = ['26.38_078_21','26.38_078_22','26.38_078_23','26.38_078_24',
            '26.38_078_25','26.38_078_26','26.38_078_27','26.38_078_28','26.38_078_29']
params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "BeatsPerMinute",                  
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [ +0.0014,+0.0007,+0.0004,+0.0002, 0, -0.0001,-0.0005,-0.0008,-0.0013],       
      'subm'   : [
         { 'name':fs_names[0], 'weight':+0.025 },
         { 'name':fs_names[1], 'weight':+0.025 },
         { 'name':fs_names[2], 'weight':+0.025 },
         { 'name':fs_names[3], 'weight':+0.025 },
         { 'name':fs_names[4], 'weight':+0.80 }, # original
         { 'name':fs_names[5], 'weight':+0.025},
         { 'name':fs_names[6], 'weight':+0.025 },
         { 'name':fs_names[7], 'weight':+0.025 },
         { 'name':fs_names[8], 'weight':+0.025 },
      ]
}
df_cross = v_blend (path, fs_names, params, color='tes9', show_figures2=True)
df_cross.to_csv('subm_2.csv', index=False)


path = '/kaggle/working/'
fs_names = ['26.38_078_31','26.38_078_32','26.38_078_33','26.38_078_34',
            '26.38_078_35','26.38_078_36','26.38_078_37','26.38_078_38','26.38_078_39']
params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "BeatsPerMinute",                  
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [ +0.0014,+0.0007,+0.0004,+0.0002, 0, -0.0001,-0.0005,-0.0008,-0.0013],       
      'subm'   : [
         { 'name':fs_names[0], 'weight':+0.025 },
         { 'name':fs_names[1], 'weight':+0.025 },
         { 'name':fs_names[2], 'weight':+0.025 },
         { 'name':fs_names[3], 'weight':+0.025 },
         { 'name':fs_names[4], 'weight':+0.80 }, # original
         { 'name':fs_names[5], 'weight':+0.025},
         { 'name':fs_names[6], 'weight':+0.025 },
         { 'name':fs_names[7], 'weight':+0.025 },
         { 'name':fs_names[8], 'weight':+0.025 },
      ]
}
df_cross = v_blend (path, fs_names, params, color='tes9', show_figures2=True)
df_cross.to_csv('subm_3.csv', index=False)


path = '/kaggle/working/'
fs_names = ['subm_1','subm_2','subm_3']
params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "BeatsPerMinute",                 
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [ +0.08,-0.03,-0.05],       
      'subm'   : [
         { 'name':fs_names[0], 'weight':+0.80 },  # v.18.LB = 26.38_077  
         { 'name':fs_names[1], 'weight':+0.10 },
         { 'name':fs_names[2], 'weight':+0.10 },
      ]
}
df_cross = v_blend (path, fs_names, params, color='alls3', 
                    show_figures1=True, show_details=True)
df_cross.to_csv('submission_1.csv', index=False)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "BeatsPerMinute",                  
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [ +0.08,-0.03,-0.05],       
      'subm'   : [
         { 'name':fs_names[0], 'weight':+0.70 },  # v.19.LB = 26.38_077
         { 'name':fs_names[1], 'weight':+0.15 },
         { 'name':fs_names[2], 'weight':+0.15 },
      ]
}
df_cross = v_blend (path, fs_names, params, color='alls3')
df_cross.to_csv('submission_2.csv', index=False)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "BeatsPerMinute",                  
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [ +0.08,-0.03,-0.05],       
      'subm'   : [
         { 'name':fs_names[0], 'weight':+0.40 },  # v.20.LB = 26.38_077,  v18 < v20 < v19
         { 'name':fs_names[1], 'weight':+0.30 },
         { 'name':fs_names[2], 'weight':+0.30 },
      ]
}
df_cross = v_blend (path, fs_names, params, color='alls3')
df_cross.to_csv('submission_3.csv', index=False)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "BeatsPerMinute",                  
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [ +0.08,-0.03,-0.05],       
      'subm'   : [
         { 'name':fs_names[0], 'weight':+0.74 },  # v.21.LB = ?
         { 'name':fs_names[1], 'weight':+0.13 },
         { 'name':fs_names[2], 'weight':+0.13 },
      ]
}
df_cross = v_blend (path, fs_names, params, color='alls3', show_figures2=True)
df_cross.to_csv('submission.csv', index=False)
df_cross

