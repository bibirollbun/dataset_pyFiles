import pandas as pd
import numpy as np

def v_blend(path_to_ds, file_short_names, dk):
    def read(dk, i):
        tnm = dk["subm"][i]["name"]
        FiN = dk["path"] + tnm + ".csv"
        df = pd.read_csv(FiN)
        if 'target' in df.columns:
            df = df.rename(columns={'target': tnm})
        elif 'y' in df.columns:
            df = df.rename(columns={'y': tnm})
        return df

    def merge(dfs_subm):
        df_subms = pd.merge(dfs_subm[0], dfs_subm[1], on=[dk['id']])
        for i in range(2, len(dfs_subm)):
            df_subms = pd.merge(df_subms, dfs_subm[i], on=[dk['id']])
        return df_subms

    def da(dk, sorting_direction):
        df_subms = merge([read(dk, i) for i in range(len(dk["subm"]))])
        cols = [col for col in df_subms.columns if col != dk['id']]
        short_name_cols = [c for c in cols]

        def alls(x, sd=sorting_direction, cs=cols):
            reverse = True if sd == 'desc' else False
            tes = {c: x[c] for c in cs}.items()
            subms_sorted = [t[0] for t in sorted(tes, key=lambda k: k[1], reverse=reverse)]
            return subms_sorted

        def summa(x, cs, wts, ic_alls):
            return sum([x[cs[j]] * (wts[0][j] + wts[1][ic_alls[j]]) for j in range(len(cs))])

        wts = [
            [[e['weight'] for e in dk["subm"]], [w for w in dk["subwts"]]],
            [[e['weight'] for e in dk["subm2"]], [w for w in dk["subwts2"]]],
            [[e['weight'] for e in dk["subm3"]], [w for w in dk["subwts3"]]],
            [[e['weight'] for e in dk["subm4"]], [w for w in dk["subwts4"]]],
        ]

        def correct(x, cs=cols, wts=wts):
            i = [x['alls'].index(c) for c in short_name_cols]
            if 0.00 < x['mx-m'] <= 0.10:
                return summa(x, cs, wts[0], i)
            elif 0.10 < x['mx-m'] <= 0.15:
                return summa(x, cs, wts[1], i)
            elif 0.15 < x['mx-m'] <= 0.20:
                return summa(x, cs, wts[2], i)
            else:
                return summa(x, cs, wts[3], i)

        def amxm(x, cs=cols):
            list_values = x[cs].to_list()
            mxm = abs(max(list_values) - min(list_values))
            return mxm

        df_subms['mx-m'] = df_subms.apply(lambda x: amxm(x), axis=1)
        df_subms['alls'] = df_subms.apply(lambda x: alls(x), axis=1)
        df_subms[dk["target"]] = df_subms.apply(lambda x: correct(x), axis=1)
        schema_rename = {old_nc: new_shnc for old_nc, new_shnc in zip(cols, short_name_cols)}
        df_subms = df_subms.rename(columns=schema_rename)
        df_subms = df_subms.rename(columns={dk["target"]: "ensemble"})
        df_subms.insert(loc=1, column=' _ ', value=['   '] * len(df_subms))
        df_subms[' _ '] = df_subms[' _ '].astype(str)
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.float_format', '{:.4f}'.format)
        vcols = [dk['id']] + [' _ '] + short_name_cols + [' _ '] + ['mx-m'] + [' _ '] + ['alls'] + [' _ '] + ['ensemble']
        df_subms = df_subms[vcols]
        display(df_subms.head(5))
        pd.set_option('display.float_format', '{:.5f}'.format)
        df_subms = df_subms.rename(columns={"ensemble": dk["target"]})
        df_subms.to_csv(f'/kaggle/working/tida_{sorting_direction}.csv', index=False)
        return df_subms[[dk['id'], dk['target']]]

    def ensemble_da(dk):
        dfD = da(dk, 'desc')
        dfA = da(dk, 'asc')
        dfA[dk['target']] = dk['desc'] * dfD[dk['target']] + dk['asc'] * dfA[dk['target']]
        return dfA

    return ensemble_da(dk)

# Direct blend function
def direct_blend(subms, file_name, wts):
    result = subms[0].copy()
    result['y'] = sum(wts[i] * subms[i]['y'] for i in range(len(subms)))
    result.to_csv(file_name, index=False)
    return result

# Define path
input_path = '/kaggle/input/aug25-ps-s5e8/7-august-2025-ps-s5e8/'

# Group 1: High-scoring submissions
group_1_fins = [
    'submission(3)',        # 0.97717
    'submission_Group_1',    # 0.97684
    'submission',            # 0.97645
    'submission (1)',        # 0.97495
    'submission_vBlend'      # 0.97558
]
group_1_params = {
    'path': input_path,
    'id': 'id',
    'target': 'y',
    'desc': 0.80,
    'asc': 0.20,
    'subwts': [+0.08, +0.05, +0.02, -0.03, -0.04],
    'subm': [
        {'name': group_1_fins[0], 'weight': 0.40},  # 0.97717
        {'name': group_1_fins[1], 'weight': 0.25},  # 0.97684
        {'name': group_1_fins[2], 'weight': 0.20},  # 0.97645
        {'name': group_1_fins[3], 'weight': 0.10},  # 0.97495
        {'name': group_1_fins[4], 'weight': 0.05},  # 0.97558
    ],
    'subwts2': [+0.07, +0.04, +0.015, -0.025, -0.035],
    'subm2': [
        {'name': group_1_fins[0], 'weight': 0.45},
        {'name': group_1_fins[1], 'weight': 0.23},
        {'name': group_1_fins[2], 'weight': 0.18},
        {'name': group_1_fins[3], 'weight': 0.09},
        {'name': group_1_fins[4], 'weight': 0.05},
    ],
    'subwts3': [+0.06, +0.03, +0.01, -0.02, -0.03],
    'subm3': [
        {'name': group_1_fins[0], 'weight': 0.50},
        {'name': group_1_fins[1], 'weight': 0.21},
        {'name': group_1_fins[2], 'weight': 0.16},
        {'name': group_1_fins[3], 'weight': 0.08},
        {'name': group_1_fins[4], 'weight': 0.05},
    ],
    'subwts4': [+0.05, +0.02, +0.005, -0.015, -0.025],
    'subm4': [
        {'name': group_1_fins[0], 'weight': 0.55},
        {'name': group_1_fins[1], 'weight': 0.19},
        {'name': group_1_fins[2], 'weight': 0.14},
        {'name': group_1_fins[3], 'weight': 0.07},
        {'name': group_1_fins[4], 'weight': 0.05},
    ],
}

# Group 2: AutoGluon and other models
group_2_fins = [
    'submission_AutoGluon_02',      # 0.97093
    'submission_LightGBM_BAG_L3',   # 0.97093
    'submission_AutoGluon',         # 0.97012
    'submission_XGBoost_Tuned_0.9645',  # 0.96697
    'lightbgm'                      # 0.96441 (assumed)
]
group_2_params = {
    'path': input_path,
    'id': 'id',
    'target': 'y',
    'desc': 0.80,
    'asc': 0.20,
    'subwts': [+0.07, +0.05, +0.03, -0.02, -0.03],
    'subm': [
        {'name': group_2_fins[0], 'weight': 0.30},  # 0.97093
        {'name': group_2_fins[1], 'weight': 0.25},  # 0.97093
        {'name': group_2_fins[2], 'weight': 0.20},  # 0.97012
        {'name': group_2_fins[3], 'weight': 0.15},  # 0.96697
        {'name': group_2_fins[4], 'weight': 0.10},  # 0.96441
    ],
    'subwts2': [+0.06, +0.04, +0.02, -0.015, -0.025],
    'subm2': [
        {'name': group_2_fins[0], 'weight': 0.32},
        {'name': group_2_fins[1], 'weight': 0.23},
        {'name': group_2_fins[2], 'weight': 0.18},
        {'name': group_2_fins[3], 'weight': 0.14},
        {'name': group_2_fins[4], 'weight': 0.13},
    ],
    'subwts3': [+0.05, +0.03, +0.015, -0.01, -0.02],
    'subm3': [
        {'name': group_2_fins[0], 'weight': 0.34},
        {'name': group_2_fins[1], 'weight': 0.21},
        {'name': group_2_fins[2], 'weight': 0.16},
        {'name': group_2_fins[3], 'weight': 0.15},
        {'name': group_2_fins[4], 'weight': 0.14},
    ],
    'subwts4': [+0.04, +0.02, +0.01, -0.005, -0.015],
    'subm4': [
        {'name': group_2_fins[0], 'weight': 0.36},
        {'name': group_2_fins[1], 'weight': 0.19},
        {'name': group_2_fins[2], 'weight': 0.14},
        {'name': group_2_fins[3], 'weight': 0.16},
        {'name': group_2_fins[4], 'weight': 0.15},
    ],
}

# Apply v-blend to Group 1
print("Processing Group 1...")
group_1_df = v_blend(input_path, group_1_fins, group_1_params)
group_1_df.to_csv('/kaggle/working/submission_Group_1_v2.csv', index=False)
display(group_1_df.head())

# Apply v-blend to Group 2
print("Processing Group 2...")
group_2_df = v_blend(input_path, group_2_fins, group_2_params)
group_2_df.to_csv('/kaggle/working/submission_Group_2_v2.csv', index=False)
display(group_2_df.head())

# Direct blend of Group 1 and Group 2
print("Creating final submission...")
final_submission = direct_blend([group_1_df, group_2_df], '/kaggle/working/submission_vBlend_v2.csv', wts=[0.80, 0.20])
display(final_submission.head())




