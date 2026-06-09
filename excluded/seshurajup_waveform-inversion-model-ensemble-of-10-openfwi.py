import pandas as pd
from glob import glob
exps = glob(f"/kaggle/input/waveform-inversion-exps/**/*")
datasets = ['CurveFault_A', 'CurveFault_B', 'CurveVel_A', 'CurveVel_B', 'FlatFault_A', 'FlatFault_B', 'FlatVel_A', 'FlatVel_B', 'Style_A', 'Style_B']
short_datasets = {'cfa':'CurveFault_A', 'cfb':'CurveFault_B', 'cva':'CurveVel_A', 'cvb':'CurveVel_B', 'ffa':'FlatFault_A', 'ffb':'FlatFault_B', 'fva':'FlatVel_A', 'fvb':'FlatVel_B', 'sta':'Style_A', 'stb':'Style_B'}
datasets_map = {x.lower(): x for x in datasets}
lb_scores = [
  {"name": "exp_40", "lb": 391.7},
  {"name": "exp_39", "lb": 258.3},
  {"name": "exp_38", "lb": 390.2},
  {"name": "exp_37", "lb": 259.1},
  {"name": "exp_36", "lb": 249.7},
  {"name": "exp_35", "lb": 304.8},
  {"name": "exp_34", "lb": 272.0},
  {"name": "exp_33", "lb": 381.5},
  {"name": "exp_32", "lb": 328.3},
  {"name": "exp_31", "lb": 310.4},
  {"name": "exp_30", "lb": 399.0},
  {"name": "exp_29", "lb": 418.7},
  {"name": "exp_28", "lb": 278.8},
  {"name": "exp_27", "lb": 319.4},
  {"name": "exp_26", "lb": 338.8},
  {"name": "exp_25", "lb": 459.7},
  {"name": "exp_24", "lb": 474.8},
  {"name": "exp_23", "lb": 461.2},
  {"name": "exp_22", "lb": 414.2},
  {"name": "exp_21", "lb": 279.4},
  {"name": "exp_20", "lb": 262.8},
  {"name": "exp_19", "lb": 288.5},
  {"name": "exp_18", "lb": 391.5},
  {"name": "exp_17", "lb": 359.0},
  {"name": "exp_16", "lb": 272.6},
  {"name": "exp_15", "lb": 378.7},
  {"name": "exp_14", "lb": 324.6},
  {"name": "exp_13", "lb": 269.1},
  {"name": "exp_12", "lb": 364.5},
  {"name": "exp_11", "lb": 340.2},
  {"name": "exp_10", "lb": 429.7},
  {"name": "exp_9", "lb": 285.8},
  {"name": "exp_8", "lb": 387.9},
  {"name": "exp_7", "lb": 348.1},
  {"name": "exp_6", "lb": 366.2},
  {"name": "exp_5", "lb": 408.7},
  {"name": "exp_4", "lb": 308.4},
  {"name": "exp_3", "lb": 405.1},
  {"name": "exp_2", "lb": 261.0},
  {"name": "exp_1", "lb": 379.3}
]
lb_scores_hash = {x['name']: x['lb'] for x in lb_scores}
def get_dataset(exp):
    exp_name = exp.split("/")[-1].split("_")[1:4]
    dataset = exp_name[0]
    if "l1" in exp_name[2] or "l2" in exp_name[2]:
        dataset = '_'.join(exp_name[0:2])
    if dataset in datasets_map:
        dataset = datasets_map[dataset]
    elif dataset in short_datasets:
        dataset = short_datasets[dataset]
    return dataset
exp_df = pd.DataFrame([{"exp_id": exp.split("/")[4], "exp_sub": exp, "dataset": get_dataset(exp), "LB Score": lb_scores_hash[exp.split("/")[4]]} for exp in exps]).sort_values(['dataset','LB Score']).reset_index(drop=True)
exp_df


selected_exps = [ 27, 21, 31, 39, 6, 37, 24, 23, 18, 3]
best_exp_df = exp_df[exp_df['exp_id'].isin([f'exp_{eid}' for eid in selected_exps])].reset_index(drop=True)
best_exp_df


sub_cat = pd.read_csv("/kaggle/input/waveform-inversion-test-2-datasets/submission_categories.csv")
sub_cat


dataset_2_test_ids = dict(sub_cat.groupby(['dataset'])['id'].agg(list))


from tqdm import tqdm
exp_dfs = {}
for i, row in tqdm(best_exp_df.iterrows(), total=best_exp_df.shape[0]):
    dataset = row['dataset']
    print("Loading", row['exp_sub'])
    df = pd.read_csv(row['exp_sub'])
    test_ids = dataset_2_test_ids[dataset]
    df['oid'] = df['oid_ypos'].apply(lambda x: x.split("_")[0])
    df = df[df['oid'].isin(test_ids)].drop(columns=['oid'])
    exp_dfs[dataset] = df


%%time
comp_sub = pd.read_csv("/kaggle/input/waveform-inversion/sample_submission.csv")
oid_ypos_2_index = {v:k for k,v in dict(comp_sub['oid_ypos']).items()}
comp_sub


%%time
sub = pd.concat([df for dataset,df in exp_dfs.items()], ignore_index=False)
sub['index'] = sub['oid_ypos'].apply(lambda x: oid_ypos_2_index[x])
sub = sub.set_index("index").sort_index()
sub


sub.shape, comp_sub.shape


%%time
sub.to_csv("submission.csv", index=False)




