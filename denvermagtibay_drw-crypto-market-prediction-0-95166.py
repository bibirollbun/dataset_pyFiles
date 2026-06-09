import pandas as pd


# Where your individual model outputs live
path_to_ds = '/kaggle/input/21-juli-2025-drw/submission '

# The basenames of each CSV (without ".csv" or folder path)
file_short_names = [
    '0.95109',
    '0.95004',
    '0.95002',
    '0.94857',
    '0.90222'
]


# Ensemble configuration: blend ratios, static weights, and per-tier corrections
params = {
    'path':   path_to_ds,
    'sort':   "dynamic",     # placeholder; will be set to 'desc' or 'asc'
    'target': "prediction",  # name of the final output column
    'q_rows': 538_150,       # number of test rows (for display spacing)

    'prefix': "subm_",       # prefix your code adds when renaming columns

    # Final blend weights for the two passes:
    'desc': 0.32,  # weight for the descending-sort pass
    'asc':  0.68,  # weight for the ascending-sort pass

    # Tier 1: if maxâ€“min spread in [0.00, 0.50]
    'subwts':   [+0.015, +0.002, -0.002, -0.005, -0.010],
    'subm': [
        {'name': file_short_names[0], 'weight': 0.86},
        {'name': file_short_names[1], 'weight': 0.04},
        {'name': file_short_names[2], 'weight': 0.04},
        {'name': file_short_names[3], 'weight': 0.05},
        {'name': file_short_names[4], 'weight': 0.01},
    ],

    # Tier 2: if spread in (0.50, 1.00]
    'subwts2':  [+0.020, +0.002, -0.002, -0.007, -0.013],
    'subm2': [
        {'name': file_short_names[0], 'weight': 0.83},
        {'name': file_short_names[1], 'weight': 0.053},
        {'name': file_short_names[2], 'weight': 0.053},
        {'name': file_short_names[3], 'weight': 0.054},
        {'name': file_short_names[4], 'weight': 0.010},
    ],

    # Tier 3: if spread in (1.00, 1.50]
    'subwts3':  [+0.025, +0.002, -0.002, -0.010, -0.015],
    'subm3': [
        {'name': file_short_names[0], 'weight': 0.82},
        {'name': file_short_names[1], 'weight': 0.057},
        {'name': file_short_names[2], 'weight': 0.057},
        {'name': file_short_names[3], 'weight': 0.057},
        {'name': file_short_names[4], 'weight': 0.010},
    ],

    # Tier 4: if spread in (1.50, 2.00]
    'subwts4':  [+0.030, +0.002, -0.002, -0.010, -0.020],
    'subm4': [
        {'name': file_short_names[0], 'weight': 0.79},
        {'name': file_short_names[1], 'weight': 0.07},
        {'name': file_short_names[2], 'weight': 0.07},
        {'name': file_short_names[3], 'weight': 0.07},
        {'name': file_short_names[4], 'weight': 0.00},
    ],

    # Tier 5: if spread > 2.00
    'subwts5':  [+0.035, +0.002, -0.002, -0.012, -0.023],
    'subm5': [
        {'name': file_short_names[0], 'weight': 0.78},
        {'name': file_short_names[1], 'weight': 0.07},
        {'name': file_short_names[2], 'weight': 0.07},
        {'name': file_short_names[3], 'weight': 0.07},
        {'name': file_short_names[4], 'weight': 0.01},
    ],
}


def iBlend(path_to_ds, file_short_names, sls):
    import pandas as pd

    def tida(sls):
        # 1) Read & rename individual submissions
        def read_subm(sls, i):
            fn = sls["path"] + sls["subm"][i]["name"] + ".csv"
            return pd.read_csv(fn).rename(
                columns={'target': sls["subm"][i]["name"],
                         sls["target"]: sls["subm"][i]["name"]}
            )

        # Merge on ID
        dfs = [read_subm(sls, i) for i in range(len(sls["subm"]))]
        df = dfs[0].merge(dfs[1], on="ID")
        for d in dfs[2:]:
            df = df.merge(d, on="ID")

        # 2) Prepare column lists
        cols = [c for c in df if c != "ID"]
        short_cols = [c.replace(sls["prefix"], "") for c in cols]

        # 3) Extract static weights & corrections for each tier
        weights = [
            [m["weight"] for m in sls[k]]
            for k in ("subm", "subm2", "subm3", "subm4", "subm5")
        ]
        corrections = [
            sls[k] for k in ("subwts", "subwts2", "subwts3", "subwts4", "subwts5")
        ]

        # 4) Helpers: spread, ranking, weighted sum
        def spread(x): return abs(x[cols].max() - x[cols].min())

        def ranking(x):
            items = x[cols].items()
            rev = (sls["sort"] == "desc")
            sorted_names = [n for n, _ in sorted(items, key=lambda p: p[1], reverse=rev)]
            return [n.replace(sls["prefix"], "") for n in sorted_names]

        def apply_weights(x):
            sp = x["spread"]
            tier = min(int(sp // 0.5), 4)  # 0â†’tier1, 1â†’tier2, â€¦, 4â†’tier5
            ws, cs = weights[tier], corrections[tier]
            ranks = x["ranks"]
            return sum(
                x[cols[j]] * (ws[j] + cs[ranks[j]])
                for j in range(len(cols))
            )

        # 5) Compute
        df["spread"] = df.apply(spread, axis=1)
        df["ranks"]  = df.apply(ranking, axis=1).apply(
            lambda lst: [lst.index(c) for c in short_cols]
        )
        df[sls["target"]] = df.apply(apply_weights, axis=1)

        # 6) Return only ID + final prediction
        return df[["ID", sls["target"]]]

    # 7) Twoâ€�pass blend: desc then asc
    sample = pd.read_csv(path_to_ds + file_short_names[1] + ".csv")
    def ensemble_tida(sls):
        sls["sort"] = "desc"
        d1 = tida(sls)
        d1.to_csv("tida_desc.csv", index=False)

        sls["sort"] = "asc"
        d2 = tida(sls)
        d2.to_csv("tida_asc.csv", index=False)

        sample[sls["target"]] = d1[sls["target"]] * sls["desc"] + d2[sls["target"]] * sls["asc"]
        return sample

    return ensemble_tida(sls)


# Run the ensemble and save your final submission.csv
df = iBlend(path_to_ds, file_short_names, params)
df.to_csv('submission1.csv', index=False)
display(df)


import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1) List the directories holding your submission CSVs
dirs = [
    '/kaggle/input/convergence-drw',
    '/kaggle/input/13-juli-2025-drw',
    '/kaggle/input/15-juli-2025-drw',
    '/kaggle/input/21-juli-2025-drw'
]

# 2) Gather all .csv files from those directories
submission_files = []
for d in dirs:
    for fname in os.listdir(d):
        if fname.endswith('.csv'):
            submission_files.append(os.path.join(d, fname))
submission_files = sorted(submission_files)  # sort for reproducibility

# 3) Load the first submission to get 'ID' and a reference scale
first_df   = pd.read_csv(submission_files[0])
df         = pd.DataFrame({'ID': first_df['ID']})
first_name = os.path.splitext(os.path.basename(submission_files[0]))[0]
df[first_name] = first_df['prediction']

# 4) Load each of the other submissions into its own column
for path in submission_files[1:]:
    tmp = pd.read_csv(path)
    col = os.path.splitext(os.path.basename(path))[0]
    df[col] = tmp['prediction'].values

# 5) Build the prediction matrix (rows Ã— n_submissions)
pred_mat = df.drop(columns='ID').values

# 6) Standardize and extract the first principal component
scaler = StandardScaler()
pred_std = scaler.fit_transform(pred_mat)
pca     = PCA(n_components=1, random_state=42)
pc1     = pca.fit_transform(pred_std).ravel()

# 7) Rescale PC1 to the min/max of the first submission
orig = df[first_name]
pc1_scaled = (pc1 - pc1.min()) / (pc1.max() - pc1.min()) \
             * (orig.max() - orig.min()) + orig.min()

# 8) Use PC1 as your final prediction and save
df['prediction'] = pc1_scaled
submission = df[['ID','prediction']]
submission.to_csv('submission.csv', index=False)
display(submission.head())

