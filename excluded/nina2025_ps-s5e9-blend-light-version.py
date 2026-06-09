import ast
import numpy as np
import pandas as pd

from bokeh.plotting import figure, gridplot
from bokeh.models import Legend, LegendItem
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
            else (131 if len(colors)  == 9\
            else (141 if len(colors)  == 10 else 185)))
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


def v_blend(path_to_ds, 
            file_short_names, 
            params, 
            color, 
            show_figures1=False, show_dtls=False): 
    dk = params

    clr_alls  = ["darkmagenta","crimson",'darkgreen',"mediumblue",'gold']
    clr_Red   = ["firebrick","orangered","crimson",'tomato',"red"]
    clr_Green = ['lime',"darkgreen","limegreen","forestgreen","green"]
    clr_Blue  = ['midnightblue',"royalblue","mediumblue","blue","steelblue"]
    clr_Brown = ["maroon","sienna","chocolate","sandybrown",'brown']
    clr_Two   = ['crimson','mediumblue']
    clr_Two2  = ['crimson','darkgreen']
    if color == 'Two2':  colors = clr_Two2  [0:len(fs_names)]
    if color == 'Two':   colors = clr_Two   [0:len(fs_names)]
    if color == 'alls':  colors = clr_alls  [0:len(fs_names)]
    if color == 'red':   colors = clr_Red   [0:len(fs_names)]
    if color == 'green': colors = clr_Green [0:len(fs_names)]
    if color == 'blue':  colors = clr_Blue  [0:len(fs_names)]
    if color == 'brown': colors = clr_Brown [0:len(fs_names)]

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
        
    def da(dk,sorting_direction,show_dtls):
        
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
        pd.set_option('display.float_format', '{:.5f}'.format)
        vcols = [dk['id']]+[' _ '] + short_name_cols + [' _ ']+['alls']+[' _ ']+['ensemble']
        if len(wts) > 1: vcols.append([' _ '] + ['mx-m'])
        df_subms = df_subms[vcols]
        if show_dtls: display(df_subms.head(4))
        pd.set_option('display.float_format', '{:.5f}'.format)
        df_subms = df_subms.rename(columns={"ensemble":dk["target"]})
        df_subms.to_csv(f'tida_{sorting_direction}.csv', index=False)
        return df_subms[[dk['id'],dk['target']]]
   
    def ensemble_da(dk,        show_dtls): 
        dfD    = da(dk,'desc', show_dtls)
        dfA    = da(dk,'asc',  show_dtls)
        dfA[dk['target']] = dk['desc']*dfD[dk['target']] + dfA[dk['target']]*dk['asc']
        return dfA

    da = ensemble_da(dk,show_dtls)

    bokeh_show(dk,colors,show_figures1)
    
    return  da


path = '/kaggle/input/14-september-2025-ps-s5e9/' + 'submission_'

fs_names = ['26.38078','26.38084','26.38089','26.38091']

params = {
      'path'   : path,
      'id'     : 'id',                 
      'target' : "BeatsPerMinute",
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [ -0.0007,+0.0001,+0.0002,+0.0004 ],
      'subm'   : [
         { 'name':fs_names[0],'weight':+0.9959 },
         { 'name':fs_names[1],'weight':+0.0010 },
         { 'name':fs_names[2],'weight':+0.0011 },
         { 'name':fs_names[3],'weight':+0.0020 },
      ]
}

df_cross = v_blend ( path, fs_names, params, color='alls',show_figures1=True, show_dtls=True)

dfs  = [
    pd.read_csv('/kaggle/input/14-september-2025-ps-s5e9/submission_26.38078.csv'),
    pd.read_csv('/kaggle/input/14-september-2025-ps-s5e9/submission_26.38084.csv'),
    pd.read_csv('/kaggle/input/14-september-2025-ps-s5e9/submission_26.38089.csv'),
    pd.read_csv('/kaggle/input/14-september-2025-ps-s5e9/submission_26.38091.csv'),
    df_cross
]

f = figure(width=800, height=300)
f.title.text = 'Click on legend entries to mute the corresponding lines'

b,e  = 1945,1983

_id  = [df[b:e]['id']  for df in dfs]
_BPM = [df[b:e]['BeatsPerMinute'] for df in dfs]
color=["darkmagenta","crimson",'darkgreen',"mediumblue",'gold']
alpha,lws =[0.8, 0.8, 0.8, 0.8, 0.95], [1,1,1,1,2]

legend = fs_names + ['cross']

for i in range(len(legend)):
    f.line(_id[i], _BPM[i], line_width=lws[i], color=color[i], alpha=alpha[i],
           muted_color='white',legend_label=legend[i])

f.legend.location = "top_left"
f.legend.click_policy="mute"

show(f)


df_cross.to_csv('submission.csv', index=False)
df_cross

