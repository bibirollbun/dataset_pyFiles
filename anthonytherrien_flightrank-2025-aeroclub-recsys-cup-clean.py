# ---------------------------
#  Import libraries
# ---------------------------
import pandas as pd


# ---------------------------
#  Helper function: rank to score
# ---------------------------
def rank_to_score(sr, eps=1e-5):
    n = sr.max()
    return 1.0 - (sr - 1) / (n + eps)

# ---------------------------
#  Helper function: score to rank
# ---------------------------
def score_to_rank(s):
    return s.rank(method='first', ascending=False).astype(int)


# ---------------------------
#  iBlend function
# ---------------------------
def iBlend(path_to_ds, file_short_names, sls):

    # Internal helper function: read submissions
    def read_subm(sls, i):
        tnm = sls["subm"][i]["name"]
        filename = f"{sls['path']}{tnm}.csv"
        df = pd.read_csv(filename).rename(columns={'target': tnm, sls["target"]: tnm})
        del df["ranker_id"]
        return df

    # Internal helper function: blending logic
    def tida(sls):
        dfs_subm = [read_subm(sls, i) for i in range(len(sls["subm"]))]

        # Merge all submissions by Id
        df_subms = dfs_subm[0]
        for df in dfs_subm[1:]:
            df_subms = pd.merge(df_subms, df, on='Id')

        cols = [col for col in df_subms.columns if col != "Id"]
        short_name_cols = [c.replace(sls["prefix"], '') for c in cols]
        corrects = sls["subwts"]
        weights = [subm['weight'] for subm in sls["subm"]]

        # Compute sorted order
        def alls(x, cs=cols):
            tes = {c: x[c] for c in cs}.items()
            subms_sorted = [
                t[0].replace(sls["prefix"], '')
                for t in sorted(tes, key=lambda k: k[1], reverse=(sls["sort"] == 'desc'))
            ]
            return subms_sorted

        # Compute weighted ensemble
        def correct(x, cs=cols, w=weights, cw=corrects):
            ic = [x['alls'].index(c) for c in short_name_cols]
            cS = [x[cols[j]] * (w[j] + cw[ic[j]]) for j in range(len(cols))]
            return sum(cS)

        df_subms['alls'] = df_subms.apply(alls, axis=1)
        df_subms[sls["target"]] = df_subms.apply(correct, axis=1)

        schema_rename = {old_nc: new_shnc for old_nc, new_shnc in zip(cols, short_name_cols)}
        df_subms = df_subms.rename(columns=schema_rename).rename(columns={sls["target"]: "ensemble"})

        submission_cols = ['Id', 'ensemble']
        df_submission = df_subms[submission_cols].rename(columns={"ensemble": sls["target"]})
        return df_submission

    # Load a sample submission template
    sample_subm = pd.read_csv(f"{path_to_ds}{file_short_names[1]}.csv")

    # Generate ascending and descending submissions, then blend
    def ensemble_tida(sls, submission=sample_subm):
        sls['sort'] = 'desc'
        dfD = tida(sls)
        sls['sort'] = 'asc'
        dfA = tida(sls)

        target, d, a = sls['target'], sls['desc'], sls['asc']
        submission[target] = ((dfD[target] * d) + (dfA[target] * a)).round().astype(int)
        return submission

    # Final blended submission
    submission = ensemble_tida(sls)
    return submission


# ---------------------------
#  Main function
# ---------------------------
def main():
    # Paths and filenames
    path_to_ds = '/kaggle/input/20-juli-2025-flightrank/submission '
    file_short_names = ['0.48507', '0.48425', '0.49343']

    # Parameters configuration
    params = {
        'path': path_to_ds,
        'sort': "asc\\desc",
        'target': "selected",
        'q_rows': 6_897_776,
        'prefix': "subm_",
        'desc': 0.44,
        'asc': 0.54,
        'subwts': [+0.11, -0.04, -0.07],
        'subm': [
            {'name': file_short_names[0], 'weight': 0.30},
            {'name': file_short_names[1], 'weight': 0.20},
            {'name': file_short_names[2], 'weight': 0.50},
        ]
    }

    # Generate blended submission
    df_submission = iBlend(path_to_ds, file_short_names, params)
    df_submission.to_csv('/kaggle/working/submission_tida.csv', index=False)

    # Load additional submissions for simple ensemble
    df2 = pd.read_csv(f"{path_to_ds}0.43916.csv")
    df3 = pd.read_csv(f"{path_to_ds}0.42163.csv")
    df4 = pd.read_csv(f"{path_to_ds}0.41226.csv")

    # List of dataframes
    dfs = [df_submission, df2, df3, df4]

    # Convert ranks to scores for ensemble
    score_frames = []
    for i, df in enumerate(dfs):
        tmp = df[['Id', 'ranker_id', 'selected']].copy()
        tmp['score'] = tmp.groupby('ranker_id')['selected'].transform(rank_to_score)
        score_frames.append(tmp[['Id', 'ranker_id', 'score']].rename(columns={'score': f'score_{i}'}))

    # Merge scores
    merged = score_frames[0]
    for frame in score_frames[1:]:
        merged = merged.merge(frame, on=['Id', 'ranker_id'], how='left')

    # Ensemble weights
    weights = [0.997, 0.001, 0.001, 0.001]
    score_cols = [f'score_{i}' for i in range(4)]
    w = pd.Series(weights, index=score_cols)

    # Compute weighted score mean
    merged['score_mean'] = (merged[score_cols] * w).sum(axis=1) / w.sum()

    # Convert scores back to ranks
    merged['selected'] = merged.groupby('ranker_id')['score_mean'].transform(score_to_rank)

    # Final submission file
    final_submission = merged[['Id', 'ranker_id', 'selected']]
    final_submission.to_csv("/kaggle/working/submission.csv", index=False, float_format='%.0f')


# ---------------------------
#  Execute main function
# ---------------------------
if __name__ == "__main__":
    main()

