import numpy as np
import pandas as pd

import os,ast,shutil,copy

from bokeh.plotting import figure, gridplot 
from bokeh.io import output_file, show, output_notebook
output_notebook()


def bokeh_show(
        params,
        df_cross,
        show_figures1, 
        show_figures2, wps_fig2,
        color_cross):

    colors = [subm['color'] for subm in params['subm']]
    
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
    height = 100 if len(colors)==2\
        else 134 if len(colors)==3 else (154 if len(colors)==4 else 174)
    for one_dossier in dossiers: 
        i_col = 'alls. ' + str(one_dossier['q_in'][i]['c'])
        qs = [one['q'] for one in one_dossier['q_in']]
        x_names = [name.replace("Group","").replace("subm_","") for name in subm_names]
        width = 157  if len(colors) == 5\
            else (140 if len(colors) == 4\
            else (121 if len(colors) == 8\
            else (131 if len(colors) == 9\
            else (141 if len(colors) == 10\
            else (171 if len(colors) == 11 else 130)))))
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
    f1 = figure(y_range=y_names, width=270, height=height, title='relations of general masses')
    f1.hbar(y=y_names, height=0.555, right=acc_mass, left=0, color=colors)
    output_file('tida_alls2.html')
    alls = [f'alls.{i}' for i in range(len(dossiers))]
    subm = [f'sub{i}'   for i in range(len(dossiers))] 
    mmsT  = np.asarray(mms).T
    data = {'cols' : alls}
    for i in range(len(dossiers)): data[f'sub{i}'] = mmsT[i,:]
    f2 = figure(y_range=alls, height=height, width=270, title="relations of columns masses")
    f2.hbar_stack(subm, y='cols', height=0.555, color=colors, source=data)
    qssT  = np.asarray(qss).T
    data = {'cols' : alls}
    for i in range(len(dossiers)): data[f'sub{i}'] = qssT[i,:]
    f3 = figure(y_range=alls, height=height, width=245, title="ratios in columns")
    f3.hbar_stack(subm, y='cols', height=0.555, color=colors, source=data)
    grid = gridplot([[f3,f2,f1]])
    show(grid)
    if show_figures2 == True:
        def read(params,i):
            FiN = params["path"] + params["subm"][i]["name"] + ".csv"
            target_name_back = {'target':params["target"],'pred':params["target"]}
            return pd.read_csv(FiN).rename(columns=target_name_back)
        dfs = [read(params,i) for i in range(len(params["subm"]))] + [df_cross]
        _height = 358 if len(params["subm"]) == 11 else 254
        f   = figure(width=785, height=_height)
        f.title.text = 'Click on legend entries to mute the corresponding lines'
        b,e        = 21000,21121
        line_x     = [dfs[i][b:e]['id']             for i in range(len(dfs))]
        line_y     = [dfs[i][b:e]['diagnosed_diabetes'] for i in range(len(dfs))]
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


def matrix_vs(path,fs_names):
    def load(path,fs_names):
        dfs = [pd.read_csv(path + name_subm +'.csv') for name_subm in fs_names]
        for i in range(len(dfs)):
            dfs[i] = dfs[i].rename(columns={"diagnosed_diabetes": f'{fs_names[i]}'})
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


def display_distances(params):
    files = [subm['name'] for subm in params['subm']]
    distances = matrix_vs ( params['path'], files )            
    display(distances)


def arr_colors(color):
    sg = ['silver','gainsboro']
    if color=='g'     or color=='g': return ['darkmagenta','limegreen']              + sg
    if color=='b'     or color=='b': return ['royalblue','darkmagenta']              + sg
    if color=='2'     or color=='2': return ['limegreen','royalblue','darkmagenta']  + sg
    if color=='M'     or color=='M': return ['darkmagenta','darkorchid','magenta']   + sg
    if color=='red'   or color=='r': return ['red','crimson','firebrick']            + sg
    if color=='Red'   or color=='R': return ['red','tomato','crimson']               + sg
    if color=='Green' or color=='G': return ['forestgreen','limegreen', 'darkgreen'] + sg
    if color=='Blue'  or color=='B': return ['blue','royalblue','mediumblue']        + sg
    if color=='RGB'   or color=='S': return ['mediumblue','darkgreen','crimson']     + sg
    return ['black','dimgray','gray'] + sg


def convert(schema):
    colors = arr_colors(schema[2])
    dicts  = [
        {'name': schema[0][i],'weight':schema[1][i],'color':colors[i]} 
        for i in range(len(schema[0]))
    ]
    return {'subm':dicts}


def h_blend(
        params, _update={},
        cross='silver',
        details=False,
        fig1=False, fig2=False, wf2=555, 
        dtls=False, dist=False, subm=''):

    if isinstance(params, list): params = convert(params)

    if 'path' in _update: params.update(_update)
    
    color_cross, dk  = cross, copy.deepcopy(params)

    if details == True:
        dist = True
        show_details,show_figures1,show_figures2 = True,True,True
    else:
        show_details,show_figures1,show_figures2 = dtls,fig1,fig2
        
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
            
        wts = [[[e['weight'] for e in dk["subm"]], [w for w in dk["subwts"]]]]
          
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
                
        if len(wts) > 1 or "subwts2" in dk:

            wts = [
                [[e['weight'] for e in dk["subm"]], [w for w in dk["subwts" ]]],
                [[e['weight'] for e in dk["subm2"]],[w for w in dk["subwts2"]]],
                [[e['weight'] for e in dk["subm3"]],[w for w in dk["subwts3"]]],
            ]

            def correct(x, cs=cols, wts=wts):
                i = [x['alls'].index(c) for c in short_name_cols]
                if   0.0540 < x['mx-m'] <= 0.0740: return summa(x,cs,wts[2],i)
                if   0.0000 < x['mx-m'] <= 0.0050: return summa(x,cs,wts[1],i)
                else:                              return summa(x,cs,wts[0],i)
                   
        def amxm(x, cs=cols):
            list_values = x[cs].to_list()
            mxm = abs(max(list_values)-min(list_values))
            return mxm

        if len(wts) > 1 or "subwts2" in dk:
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
        if len(wts) > 1: 
            vcols = [dk['id']] + [' _ '] + short_name_cols + [' _ '] + ['mx-m'] + [' _ '] +\
                      ['alls'] + [' _ '] + ['ensemble']
        else:
            vcols = [dk['id']] + [' _ '] + short_name_cols + [' _ '] +\
                      ['alls'] + [' _ '] + ['ensemble']
        df_subms = df_subms[vcols]
        if show_details and sorting_direction=='desc': display(df_subms.head(5))
        pd.set_option('display.float_format', '{:.5f}'.format)
        df_subms = df_subms.rename(columns={"ensemble":dk["target"]})
        if sorting_direction=='desc': 
            df_subms.to_csv(f'tida_{sorting_direction}.csv', index=False)
        return df_subms[[dk['id'],dk['target']]]
   
    def ensemble_da(dk,        show_details): 
        dfD    = da(dk,'desc', show_details)
        dfA    = da(dk,'asc',  show_details)
        dfA[dk['target']] = dk['desc']*dfD[dk['target']] + dfA[dk['target']]*dk['asc']
        return dfA

    da = ensemble_da(dk,show_details)
    bokeh_show(dk, da, show_figures1, show_figures2, wf2, color_cross)
    if dist == True: display_distances(params)
    if subm != '': da.to_csv(subm, index=False)
    return  da


def voting(rem_left=0,rem_right=4):
    print(f'\nVoting:\n')
    df_Top = pd.read_csv('/kaggle/input/05-december-2025-ps-s5e12/0.70370.b.csv')
    target = 'diagnosed_diabetes'
    tida_desc = pd.read_csv('/kaggle/working/tida_desc.csv')
    df_soluts = tida_desc[tida_desc.columns[2:13]]
    display(df_soluts)
    qnt_Majority_votes = df_soluts.shape[1] - rem_right
    df_vote = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
    np_soluts_sorted = np.sort(df_soluts.to_numpy(), axis=1)
    df_vote[target] = np.mean(np_soluts_sorted[:,rem_left:qnt_Majority_votes], axis=1)
    df_vote[target] = df_Top[target] *0.00001 + 0.99999* df_vote[target]
    print(f'\nComparison: df_Top _x  vs  df_vote _y\n')
    df_comparison = pd.merge(df_Top, df_vote, on='id')
    df_comparison["delta_proba's"] = df_Top[target] - df_vote[target]
    display  (df_comparison)
    print(sum(df_comparison["delta_proba's"]),
              df_comparison["delta_proba's"][df_comparison["delta_proba's"] < 0].sum())
    return df_vote


path_To_ds = '/kaggle/input/01-december-2025-ps-s5e12/'
path_wpage = '/kaggle/working/submissions/'
s          = 'submission '

if os.path.isdir(path_wpage): shutil.rmtree(path_wpage) 
    
os.mkdir(path_wpage)

shutil.copy(path_To_ds + '0.70108.csv', path_wpage +'1.1.csv')
shutil.copy(path_To_ds+s+'0.69850.csv', path_wpage +'1.2.csv')
shutil.copy(path_To_ds + '0.70165.csv', path_wpage +'2.1.csv')
shutil.copy(path_To_ds + '0.70175.csv', path_wpage +'2.2.csv')


up21 = {'path'     : path_wpage,            
        'id_target': ['id','diagnosed_diabetes'],          
        'type_sort': ['asc/desc',0.30,0.70 ],
        'subwts'   : [w/150 for w in [+5,-5]],}
up22 = {'path'     : path_wpage,            
        'id_target': ['id','diagnosed_diabetes'],          
        'type_sort': ['asc/desc',0.30,0.70 ],
        'subwts'   : [w/150 for w in [+5,-5]],}
up31 = {'path'     : path_wpage,          
        'id_target': ['id','diagnosed_diabetes'],          
        'type_sort': ['asc/desc',0.30,0.70 ],
        'subwts'   : [w/200 for w in [10, -3,-7]],}
up32 = {'path'     : path_wpage,          
        'id_target': ['id','diagnosed_diabetes'],          
        'type_sort': ['asc/desc',0.30,0.70 ],
        'subwts'   : [w/250 for w in [10, -3,-7]],
}
loc_21_wts, loc_22_wts = [0.50,0.50],[0.50,0.50]

loc_31_wts = [0.31,0.34,0.35] # up31.subwts=[w/200 for w in [+10,-3,-7]]
gro_32_wts = [0.06,0.17,0.77] # up32.subwts=[w/250 for w in [+10,-3,-7]]


%%time

up32_subwts = [[+10,-3,-7],[+7,-2,-5],[+7,-1,-6],[+7,0,-7],[+4,-4, 0],
               [-8,-2,+10],[-5,-2,+7],[-6,-1,+7],[-7,0,+7],[-4,+4, 0], [0,0,0]]

for i in range(10,21):

    if i >=10 and i < 15: up32['subwts'] = np.asarray(up32_subwts[i-11]) / 250 # 0..5
    if i >=15 and i < 20: up32['subwts'] = np.asarray(up32_subwts[i-11]) / 250 # 5..10
    if i == 20:           up32['subwts'] = np.asarray(up32_subwts[10])   / 250 # 10

    if 10<=i<=15:
        loc_31_wts[0]+=0.001; loc_31_wts[1]-=0.001
        gro_32_wts[0]-=0.001; gro_32_wts[1]+=0.001
    else:
        loc_31_wts[0]+=0.001; loc_31_wts[2]-=0.001
        gro_32_wts[1]+=0.001; gro_32_wts[2]-=0.001
    
    h_blend([['1.1','1.2']      ,loc_21_wts,'g'], up21, subm=f'{path_wpage}g1.csv')
    h_blend([[      '2.1','2.2'],loc_22_wts,'b'], up22, subm=f'{path_wpage}g2.csv')
    h_blend([['1.2','2.1','2.2'],loc_31_wts,'2'], up31, subm=f'{path_wpage}g3.csv')
    
    h_blend([['g1','g2','g3']   ,gro_32_wts,'M'], up32, subm=f'{path_wpage}gs{i}.csv')


weights = [1/11 for i in range(11)]
files   = [f'gs{i}' for i in range(10,21)]
colors  = ['darkmagenta','darkorchid','mediumorchid','magenta','indigo','darkviolet','deeppink','hotpink','palevioletred','orchid','violet',]
params = {
      'path'     : path_wpage,            
      'id_target': ['id',"diagnosed_diabetes"],          
      'type_sort': ['asc/desc',0.30,0.70 ],
      'subwts'   : [e/100 for e in [+1,-1,+1,-1,+1, 0, -1,+1,-1,+1,-1]],
}
params.update({'subm':[{'name':f,'weight':w,'color':c} for f,w,c in zip(files,weights,colors)]})
        
df_hb = h_blend(params, fig1=True, fig2=True)


rem_left,rem_right = 4,0


def Vote(rem_left=rem_left,rem_right=rem_right):
    df_Top = pd.read_csv('/kaggle/input/05-december-2025-ps-s5e12/0.70370.b.csv')
    target = 'diagnosed_diabetes'
    tida_desc = pd.read_csv('/kaggle/working/tida_desc.csv')
    df_soluts = tida_desc[tida_desc.columns[2:13]]
    display(df_soluts)
    qnt_Majority_votes = df_soluts.shape[1] - rem_right
    df_vote = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
    np_soluts_sorted = np.sort(df_soluts.to_numpy(), axis=1)
    df_vote[target] = np.mean(np_soluts_sorted[:,rem_left:qnt_Majority_votes], axis=1)
    df_vote[target] = df_Top[target] *0.0001 + 0.9999* df_vote[target]
    print(f'\nComparison: df_Top _x  vs  df_vote _y\n')
    df_comparison = pd.merge(df_Top, df_vote, on='id')
    df_comparison["delta_proba's"] = df_Top[target] - df_vote[target]
    display  (df_comparison)
    print(sum(df_comparison["delta_proba's"]),
              df_comparison["delta_proba's"][df_comparison["delta_proba's"] < 0].sum())
    return df_vote

df_vote_0 = Vote()

df_vote_0.to_csv(f'{path_wpage}vote_0.csv',index=False)


%%time

df_Vote = df_vote_0

weights = [1/11 for i in range(11)]

files = ['gs11','gs12','gs13','gs14','gs15','gs16','gs17','gs18','gs19','gs20','vote_0']

colors = ['darkorchid','mediumorchid','magenta','indigo','darkviolet','deeppink','hotpink','palevioletred','orchid','violet','limegreen']

_colors = 'green,seagreen,darkgreen,forestgreen,lime'.split(',')
         
for i in range(1,5):
    def prep_Vote():
        params = {
              'path'     : path_wpage,            
              'id_target': ['id',"diagnosed_diabetes"],          
              'type_sort': ['asc/desc',0.30,0.70 ],
              'subwts'   : [e/300 for e in [+5,+4,+3,+2,+1, 0, -1,-2,-3,-4,-5]],
        }; params.update(
        {'subm':[{'name':f,'weight':w,'color':c} for f,w,c in zip(files,weights,colors)]})  
        h_blend(params, fig1=True)

    prep_Vote()
    
    files  = files [1:]; files .append(f'vote_{i}')
    colors = colors[1:]; colors.append(_colors[i-1])

    df_Vote = voting(rem_left=3,rem_right=1)
    
    df_Vote.to_csv(f'{path_wpage}vote_{i}.csv',index=False)

    if i == 4: prep_Vote()


# deleting a folder and all its contents, our "voting comrades" were kept there

if os.path.isdir(path_wpage): shutil.rmtree(path_wpage)


df_Vote.to_csv('Vote.2c.csv',index=False)
df_Vote

