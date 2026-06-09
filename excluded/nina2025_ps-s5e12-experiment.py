import os,shutil
import numpy as np
import pandas as pd
import xgboost as xgb, lightgbm as lgbm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import warnings; warnings.filterwarnings('ignore')
from catboost import CatBoostClassifier


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


seed, n_estimators = 41, {'CAT':500, 'XGB':500, 'lightGBM':500}

wpage = f'/kaggle/working/wpage/'

if os.path.isdir(wpage): shutil.rmtree(wpage) 
    
os.mkdir(wpage)

y = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")['diagnosed_diabetes']


def f_XGB(folds=5, working_page='/kaggle/working/'):
    train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
    test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
    subm  = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")
    print("train", train.shape, "test", test.shape)
    
    X = train.drop(["id","diagnosed_diabetes"], axis=1)
    y = train["diagnosed_diabetes"]

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    
    for col in cat_cols:
        X[col] = X[col].astype("category").cat.codes
        test[col] = test[col].astype("category").cat.codes

    models, oof_xgb = [], np.zeros(len(X))
    
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}/{folds}")
        X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]
    
        xgb_model = xgb.XGBClassifier(
            n_estimators     = n_estimators['XGB'],
            max_depth        = 12,
            learning_rate    = 0.03,
            subsample        = 0.83,
            colsample_bytree = 0.19,
            eval_metric      ="auc",
            random_state     = seed,
            device           ='cuda',
        )     
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=200)
    
        oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:,1]
        models.append(xgb_model)

    print("XGB OOF AUC:", roc_auc_score(y, oof_xgb),'\n')

    npy_oof = f'{working_page}oof_xgb_folds_{folds}.npy'
    
    np.save(npy_oof, oof_xgb)

    X_test, preds_N_folds, folds_files = test.drop(columns=["id"]),[],[]
    
    for fold, model in enumerate(models):
        pred          = model.predict_proba(X_test)[:, 1]
        preds_N_folds . append(pred)
        subm_one_fold = pd.DataFrame({'id': test['id'], 'diagnosed_diabetes': pred})
        filename_fold = f'XGB_{folds}_fold_{fold}.csv'
        subm_one_fold . to_csv(working_page+filename_fold, index=False)
        folds_files   . append(filename_fold)

    subm_file_name = "subm_XGB.csv"
    subm ["diagnosed_diabetes"] = np.mean(preds_N_folds, axis=0)
    subm.to_csv(working_page + subm_file_name, index=False)
    return subm_file_name, {'oof':oof_xgb,'npy':npy_oof}, folds_files


subm_xgb_7, xgb_7, fold_files_7 = f_XGB(folds=7, working_page=wpage)
subm_xgb_4, xgb_4, fold_files_4 = f_XGB(folds=4, working_page=wpage)
subm_xgb_5, xgb_5, fold_files_5 = f_XGB(folds=5, working_page=wpage)

score745 = roc_auc_score ( y, (xgb_7['oof'] +xgb_4['oof'] +xgb_5['oof'])/3 )

print("XGB direct oof's AUC:", score745)


files7 = [file.replace('.csv','') for file in fold_files_7]

params = {
      'path'     : wpage,            
      'id_target': ['id',"diagnosed_diabetes"],          
      'type_sort': ['asc/desc',0.30,0.70 ],
      'subwts'   : [w/200 for w in [ +8,+5,+3, -2,-3,-4,-7 ]],
      'subm'     : [
          {'name': files7[0], 'weight':+1/7, 'color':'crimson'   },
          {'name': files7[1], 'weight':+1/7, 'color':'dimgray'   },
          {'name': files7[2], 'weight':+1/7, 'color':'gray'      },
          {'name': files7[3], 'weight':+1/7, 'color':'darkgray'  },
          {'name': files7[4], 'weight':+1/7, 'color':'silver'    },
          {'name': files7[5], 'weight':+1/7, 'color':'gainsboro' },
          {'name': files7[6], 'weight':+1/7, 'color':'whitesmoke'},]
}
df = h_blend(params, details=True, subm=f'{wpage}xgb_7.csv')

df['direct addition'] = pd.read_csv(wpage + subm_xgb_7)['diagnosed_diabetes']

display(df)

# -----------------------------------------------------------------------------

files4 = [file.replace('.csv','') for file in fold_files_4]

params = {
      'path'     : wpage,            
      'id_target': ['id',"diagnosed_diabetes"],          
      'type_sort': ['asc/desc',0.30,0.70 ],
      'subwts'   : [w/200 for w in [ +8,+5,-4,-9 ]],
      'subm'     : [
          {'name': files4[0], 'weight':+0.25, 'color':'crimson' },
          {'name': files4[1], 'weight':+0.25, 'color':'gray'    },
          {'name': files4[2], 'weight':+0.25, 'color':'darkgray'},
          {'name': files4[3], 'weight':+0.25, 'color':'silver'  },]
}
df = h_blend(params, details=True, subm=f'{wpage}xgb_4.csv')

df['direct addition'] = pd.read_csv(wpage + subm_xgb_4)['diagnosed_diabetes']

display(df)

# -----------------------------------------------------------------------------

files5 = [file.replace('.csv','') for file in fold_files_5]

params = {
      'path'     : wpage,            
      'id_target': ['id',"diagnosed_diabetes"],          
      'type_sort': ['asc/desc',0.30,0.70 ],
      'subwts'   : [w/200 for w in [ +8,+5, -2,-4,-7 ]],
      'subm'     : [
          {'name': files5[0], 'weight':+0.20, 'color':'crimson'  },
          {'name': files5[1], 'weight':+0.20, 'color':'gray'     },
          {'name': files5[2], 'weight':+0.20, 'color':'darkgray' },
          {'name': files5[3], 'weight':+0.20, 'color':'silver'   },
          {'name': files5[4], 'weight':+0.20, 'color':'gainsboro'},
      ]
}
df = h_blend(params, details=True, subm=f'{wpage}xgb_5.csv')

df['direct addition'] = pd.read_csv(wpage + subm_xgb_5)['diagnosed_diabetes']

display(df)

# -----------------------------------------------------------------------------

params = {
      'path'     : wpage,            
      'id_target': ['id',"diagnosed_diabetes"],          
      'type_sort': ['asc/desc',0.30,0.70 ],
      'subwts'   : [w/200 for w in [ +3,-1,-2 ]],
      'subm'     : [
          {'name': f'xgb_4', 'weight':+1/3, 'color':'crimson' },
          {'name': f'xgb_5', 'weight':+1/3, 'color':'darkgray'},
          {'name': f'xgb_7', 'weight':+1/3, 'color':'gray'    },]
}
df = h_blend(params, details=True, subm=f'{wpage}xgb.csv')

df


def f_LGBM(folds=5, working_page='/kaggle/working/'):
    train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
    test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
    subm  = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")
    print("train", train.shape, "test", test.shape)

    X = train.drop(columns=['id',"diagnosed_diabetes"])
    y = train["diagnosed_diabetes"]
    
    for c in X.columns:
        if X[c].dtype == object:
            combined = pd.concat([X[c], test[c]], axis=0).astype(str)
            codes, uniques = pd.factorize(combined)
            X[c] = codes[:len(X)]
            test[c] = codes[len(X):]
        
    params = {
        "objective"       :"binary",
        "metric"          :"auc",
        "boosting"        :"gbdt",
        "learning_rate"   : 0.025,
        "num_leaves"      : 85,
        "feature_fraction": 0.74,
        "bagging_fraction": 0.54,
        "bagging_freq"    : 5,
        "seed"            : seed,
        'device'          :'gpu',
        "verbosity"       : -1,
        'metric'          : 'AUC', 
        'objective'       : 'binary', 
        'max_depth'       : 7, 
    }

    models, oof_lgbm = [], np.zeros(len(X))
    
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"Fold {fold}/{folds}")
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    
        train_data = lgbm.Dataset(X_tr,  label = y_tr)
        valid_data = lgbm.Dataset(X_val, label = y_val)
    
        model = lgbm.train(
            params,
            train_data,
            num_boost_round = n_estimators['lightGBM'],
            valid_sets      = [train_data, valid_data],
            valid_names     = ["train", "valid"],
            callbacks       = [lgbm.early_stopping(stopping_rounds=150), lgbm.log_evaluation(period=150) ]
        )
    
        oof_lgbm[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    
        models.append(model)


    print("\nLGBM OOF AUC:", roc_auc_score(y, oof_lgbm),'\n')

    npy_oof = f'{working_page}oof_lgbm_folds_{folds}.npy'
    
    np.save(npy_oof, oof_lgbm)

    X_test, preds_N_folds, folds_files = test.drop(columns=["id"]),[],[]

    for fold, model in enumerate(models):
        pred          = model.predict(X_test)
        preds_N_folds . append(pred)
        subm_one_fold = pd.DataFrame({'id': test['id'], 'diagnosed_diabetes': pred})
        filename_fold = f'LGBM_{folds}_fold_{fold}.csv'
        subm_one_fold . to_csv(working_page+filename_fold, index=False)
        folds_files   . append(filename_fold)

    subm_file_name = "subm_LGBM.csv"
    subm ["diagnosed_diabetes"] = np.mean(preds_N_folds, axis=0)
    subm.to_csv(working_page + subm_file_name, index=False)
    return subm_file_name, {'oof':oof_lgbm,'npy':npy_oof}, folds_files


subm_lgbm_7, lgbm_7, fold_files_7 = f_LGBM(folds=7, working_page=wpage)
subm_lgbm_4, lgbm_4, fold_files_4 = f_LGBM(folds=4, working_page=wpage)
subm_lgbm_5, lgbm_5, fold_files_5 = f_LGBM(folds=5, working_page=wpage)

score745 = roc_auc_score ( y, (lgbm_7['oof'] +lgbm_4['oof'] +lgbm_5['oof'])/3 )

print("LGBM direct oof's AUC:", score745)


files7 = [file.replace('.csv','') for file in fold_files_7]

params = {
      'path'     : wpage,            
      'id_target': ['id',"diagnosed_diabetes"],          
      'type_sort': ['asc/desc',0.30,0.70 ],
      'subwts'   : [w/200 for w in [ +8,+5,+3, -2,-3,-4,-7 ]],
      'subm'     : [
          {'name': files7[0], 'weight':+1/7, 'color':'green' },
          {'name': files7[1], 'weight':+1/7, 'color':'dimgray'   },
          {'name': files7[2], 'weight':+1/7, 'color':'gray'      },
          {'name': files7[3], 'weight':+1/7, 'color':'darkgray'  },
          {'name': files7[4], 'weight':+1/7, 'color':'silver'    },
          {'name': files7[5], 'weight':+1/7, 'color':'gainsboro' },
          {'name': files7[6], 'weight':+1/7, 'color':'whitesmoke'},]
}
df = h_blend(params, details=True, subm=f'{wpage}lgbm_7.csv')

df['direct addition'] = pd.read_csv(wpage + subm_lgbm_7)['diagnosed_diabetes']

display(df)

# -----------------------------------------------------------------------------

files4 = [file.replace('.csv','') for file in fold_files_4]

params = {
      'path'     : wpage,            
      'id_target': ['id',"diagnosed_diabetes"],          
      'type_sort': ['asc/desc',0.30,0.70 ],
      'subwts'   : [w/200 for w in [ +8,+5,-4,-9 ]],
      'subm'     : [
          {'name': files4[0], 'weight':+0.25, 'color':'green'},
          {'name': files4[1], 'weight':+0.25, 'color':'gray'     },
          {'name': files4[2], 'weight':+0.25, 'color':'darkgray' },
          {'name': files4[3], 'weight':+0.25, 'color':'silver'   },]
}
df = h_blend(params, details=True, subm=f'{wpage}lgbm_4.csv')

df['direct addition'] = pd.read_csv(wpage + subm_lgbm_4)['diagnosed_diabetes']

display(df)

# -----------------------------------------------------------------------------

files5 = [file.replace('.csv','') for file in fold_files_5]

params = {
      'path'     : wpage,            
      'id_target': ['id',"diagnosed_diabetes"],          
      'type_sort': ['asc/desc',0.30,0.70 ],
      'subwts'   : [w/200 for w in [ +8,+5, -2,-4,-7 ]],
      'subm'     : [
          {'name': files5[0], 'weight':+0.20, 'color':'green'},
          {'name': files5[1], 'weight':+0.20, 'color':'gray'     },
          {'name': files5[2], 'weight':+0.20, 'color':'darkgray' },
          {'name': files5[3], 'weight':+0.20, 'color':'silver'   },
          {'name': files5[4], 'weight':+0.20, 'color':'gainsboro'},
      ]
}
df = h_blend(params, details=True, subm=f'{wpage}lgbm_5.csv')

df['direct addition'] = pd.read_csv(wpage + subm_lgbm_5)['diagnosed_diabetes']

display(df)

# -----------------------------------------------------------------------------

params = {
      'path'     : wpage,            
      'id_target': ['id',"diagnosed_diabetes"],          
      'type_sort': ['asc/desc',0.30,0.70 ],
      'subwts'   : [w/200 for w in [ +3,-1,-2 ]],
      'subm'     : [
          {'name': f'lgbm_4', 'weight':+1/3, 'color':'green' },
          {'name': f'lgbm_5', 'weight':+1/3, 'color':'darkgray'},
          {'name': f'lgbm_7', 'weight':+1/3, 'color':'gray'    },]
}
df = h_blend(params, details=True, subm=f'{wpage}lgbm.csv')

df


def f_CAT(folds=5, working_page='/kaggle/working/'):
    train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
    test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
    subm  = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")
    
    print("train", train.shape, "test", test.shape)
    
    X = train.drop(columns=['id', 'diagnosed_diabetes'])
    y = train['diagnosed_diabetes']
    
    # --- CatBoost specific: specify categorical features by index
    cols = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']
    cat_features_indices = [X.columns.get_loc(col) for col in cols]
    
    models, oof_cat = [], np.zeros(len(X))
    
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}/{folds}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
        model = CatBoostClassifier(
            iterations    = n_estimators['CAT'],
            learning_rate = 0.1,
            depth         = 7,
            l2_leaf_reg   = 3,
            cat_features  = cat_features_indices,
            eval_metric   ='AUC',
            random_seed   = seed,
            verbose       = 200,
        )
    
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)
    
        oof_cat[val_idx] = model.predict_proba(X_val)[:, 1]
    
        models.append(model)
    
    
    print("\nLGBM OOF AUC:", roc_auc_score(y, oof_cat),'\n')

    npy_oof = f'{working_page}oof_cat_folds_{folds}.npy'
    
    np.save(npy_oof, oof_cat)

    X_test, preds_N_folds, folds_files = test.drop(columns=["id"]),[],[]

    for fold, model in enumerate(models):
        pred          = model.predict_proba(X_test)[:, 1]
        preds_N_folds . append(pred)
        subm_one_fold = pd.DataFrame({'id': test['id'], 'diagnosed_diabetes': pred})
        filename_fold = f'CAT_{folds}_fold_{fold}.csv'
        subm_one_fold . to_csv(working_page+filename_fold, index=False)
        folds_files   . append(filename_fold)

    subm_file_name = "subm_CAT.csv"
    subm ["diagnosed_diabetes"] = np.mean(preds_N_folds, axis=0)
    subm.to_csv(working_page + subm_file_name, index=False)
    return subm_file_name, {'oof':oof_cat,'npy':npy_oof}, folds_files


subm_cat_7, cat_7, fold_files_7 = f_CAT(folds=7, working_page=wpage)
subm_cat_4, cat_4, fold_files_4 = f_CAT(folds=4, working_page=wpage)
subm_cat_5, cat_5, fold_files_5 = f_CAT(folds=5, working_page=wpage)

score745 = roc_auc_score ( y, (cat_7['oof'] +cat_4['oof'] +cat_5['oof'])/3 )

print("CAT direct oof's AUC:", score745)


files7 = [file.replace('.csv','') for file in fold_files_7]

params = {
      'path'     : wpage,            
      'id_target': ['id',"diagnosed_diabetes"],          
      'type_sort': ['asc/desc',0.30,0.70 ],
      'subwts'   : [w/200 for w in [ +8,+5,+3, -2,-3,-4,-7 ]],
      'subm'     : [
          {'name': files7[0], 'weight':+1/7, 'color':'mediumblue' },
          {'name': files7[1], 'weight':+1/7, 'color':'dimgray'   },
          {'name': files7[2], 'weight':+1/7, 'color':'gray'      },
          {'name': files7[3], 'weight':+1/7, 'color':'darkgray'  },
          {'name': files7[4], 'weight':+1/7, 'color':'silver'    },
          {'name': files7[5], 'weight':+1/7, 'color':'gainsboro' },
          {'name': files7[6], 'weight':+1/7, 'color':'whitesmoke'},]
}
df = h_blend(params, details=True, subm=f'{wpage}cat_7.csv')

df['direct addition'] = pd.read_csv(wpage + subm_cat_7)['diagnosed_diabetes']

display(df)

# -----------------------------------------------------------------------------

files4 = [file.replace('.csv','') for file in fold_files_4]

params = {
      'path'     : wpage,            
      'id_target': ['id',"diagnosed_diabetes"],          
      'type_sort': ['asc/desc',0.30,0.70 ],
      'subwts'   : [w/200 for w in [ +8,+5,-4,-9 ]],
      'subm'     : [
          {'name': files4[0], 'weight':+0.25, 'color':'mediumblue'},
          {'name': files4[1], 'weight':+0.25, 'color':'gray'     },
          {'name': files4[2], 'weight':+0.25, 'color':'darkgray' },
          {'name': files4[3], 'weight':+0.25, 'color':'silver'   },]
}
df = h_blend(params, details=True, subm=f'{wpage}cat_4.csv')

df['direct addition'] = pd.read_csv(wpage + subm_cat_4)['diagnosed_diabetes']

display(df)

# -----------------------------------------------------------------------------

files5 = [file.replace('.csv','') for file in fold_files_5]

params = {
      'path'     : wpage,            
      'id_target': ['id',"diagnosed_diabetes"],          
      'type_sort': ['asc/desc',0.30,0.70 ],
      'subwts'   : [w/200 for w in [ +8,+5, -2,-4,-7 ]],
      'subm'     : [
          {'name': files5[0], 'weight':+0.20, 'color':'mediumblue'},
          {'name': files5[1], 'weight':+0.20, 'color':'gray'     },
          {'name': files5[2], 'weight':+0.20, 'color':'darkgray' },
          {'name': files5[3], 'weight':+0.20, 'color':'silver'   },
          {'name': files5[4], 'weight':+0.20, 'color':'gainsboro'},
      ]
}
df = h_blend(params, details=True, subm=f'{wpage}cat_5.csv')

df['direct addition'] = pd.read_csv(wpage + subm_cat_5)['diagnosed_diabetes']

display(df)

# -----------------------------------------------------------------------------

params = {
      'path'     : wpage,            
      'id_target': ['id',"diagnosed_diabetes"],          
      'type_sort': ['asc/desc',0.30,0.70 ],
      'subwts'   : [w/200 for w in [ +3,-1,-2 ]],
      'subm'     : [
          {'name': f'cat_4', 'weight':+1/3, 'color':'mediumblue'},
          {'name': f'cat_5', 'weight':+1/3, 'color':'darkgray' },
          {'name': f'cat_7', 'weight':+1/3, 'color':'gray'     },]
}
df = h_blend(params, details=True, subm=f'{wpage}cat.csv')

df


params = {
      'path'     : wpage,            
      'id_target': ['id',"diagnosed_diabetes"],          
      'type_sort': ['asc/desc',0.30,0.70 ],
      'subwts'   : [w/200 for w in [ +3,-1,-2 ]],
      'subm'     : [
          {'name': f'cat', 'weight':+0.30, 'color':'mediumblue'},
          {'name': f'lgbm','weight':+0.30, 'color':'green'     },
          {'name': f'xgb', 'weight':+0.40, 'color':'crimson'   },]
}
df = h_blend(params, details=True)


df.to_csv('submission.csv',index=False)
df

