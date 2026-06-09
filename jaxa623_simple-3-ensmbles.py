path = "/kaggle/input/4-september-2025-ps-s5e9/" + "submission_"

file_short_names = ['26.38470', '26.38471', '26.38486']


import pandas as pd


def h_blend(path, fs_names, params):
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
        display(df_subms.head(5))
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
    
    return da


params = {
    'path'   : path,
    'id'     : 'id',
    'target' : "BeatsPerMinute",
    'desc'   : 0.70,    # 最终结果 = desc融合*0.7 + asc融合*0.3
    'asc'    : 0.30,
    'subwts' : [+0.10, -0.03, -0.07],   # 排序修正，可随便先给
    'subm'   : [
         { 'name':file_short_names[0], 'weight':0.4 },
         { 'name':file_short_names[1], 'weight':0.4 },
         { 'name':file_short_names[2], 'weight':0.2 },
    ]
}


df = h_blend(path, file_short_names, params)


df.to_csv("submission_myblend.csv", index=False)




