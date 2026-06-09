import pandas as pd


def add_target_groups(data_df, source_column='label_group', target_column='target'):
    target_groups = data_df.groupby(source_column).indices
    data_df[target_column]=data_df[source_column].map(target_groups)
    return data_df

def get_targets_shape(train_df):
    all_targets = add_target_groups(train_df).target.to_list()
    all_targets_lens = [len(t) for t in all_targets]
    targets_shape = []
    for size in range(min(all_targets_lens), max(all_targets_lens)+1):
        count = all_targets_lens.count(size) / len(all_targets)
        targets_shape.append((size,count))
    return targets_shape

df = pd.read_csv("/kaggle/input/shopee-product-matching/train.csv")

x = get_targets_shape(df)


data_dist_df = pd.DataFrame(x, columns=['grp_size', 'prob'])


data_dist_df.to_csv('data_dist_df.csv', index=False)




