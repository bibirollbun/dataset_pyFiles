import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import chi2

class Files:
    train = '/kaggle/input/playground-series-s5e4/train.csv'
    test = '/kaggle/input/playground-series-s5e4/test.csv'


data = pd.read_csv(Files.train)
target = 'Listening_Time_minutes'
irrelevant_cols = ['id']

X = data.drop(columns = irrelevant_cols + [target])
y = data[target]


def traverse(node: dict, parents: list, ancestors: list) -> None:
    if 'split_feature' in node: # selects non-leaf nodes
        feature = node['split_feature']
        for parent in parents:
            ancestors.append((parent, feature))
        parents = parents.copy()
        parents.append(feature)

        if 'left_child' in node:
            traverse(node['left_child'], parents, ancestors)
        if 'right_child' in node:
            traverse(node['right_child'], parents, ancestors)
            

def tally_results(ancestors: list, num_features: int, doubled: bool = False) -> pd.DataFrame:
    parent_count = {}
    child_count = {}
    tuple_count = {}
    total = 0
    for p, c in ancestors:
        total += 1
        parent_count[p] = 0 if p not in parent_count else parent_count[p] + 1
        child_count[c] = 0 if c not in child_count else child_count[c] + 1
        tuple_count[(p, c)] = 0 if (p, c) not in tuple_count else tuple_count[(p, c)] + 1

    p_vals = {}
    degrees_of_freedom = (num_features - 1)**2
    for (p, c), observed in tuple_count.items():
        if doubled:
            if (c, p) in p_vals: 
                continue
            expected = (parent_count[p] * child_count[c] + 
                        parent_count[c] * child_count[p]) / total
            try: 
                observed += tuple_count[(c, p)]
            except:
                pass
        else:
            expected = parent_count[p] * child_count[c] / total
        chi2_stat = (observed - expected)**2 / expected
        p_vals[(p, c)] = (1 - chi2.cdf(chi2_stat, df=degrees_of_freedom), 
                          'Attractive' if observed > expected else 'Repulsive')

    df_items = [{'Parent': p, 'Child': c, 'Relationship': freq, 'p_value': val} for (p, c), (val, freq) in p_vals.items()]
    return pd.DataFrame(df_items)
    

def ranked_feature_relationships(X: pd.DataFrame, y: pd.Series, doubled: bool = False) -> pd.DataFrame:
    X, y = X.copy(), y.copy()
    cat_cols = X.select_dtypes('object').columns.to_list()
    X[cat_cols] = X[cat_cols].astype('category')

    features = X.columns.to_list()

    dtrain = lgb.Dataset(X, label=y, categorical_feature=cat_cols)
    params = {
        "objective": "regression",
        "boosting": "gbdt",
        "verbose": -1,
        "max_depth": 5,
    }
    model = lgb.train(params, dtrain, num_boost_round=1000)

    trees = model.dump_model()["tree_info"]
    ancestors = []
    for tree in trees:
        parents = []
        traverse(tree['tree_structure'], parents, ancestors)

    p_values = tally_results(ancestors, len(features), doubled=doubled)
    p_values[['Parent', 'Child']] = p_values[['Parent', 'Child']].map(lambda x: features[x])
    p_values = p_values.sort_values('p_value', ascending=True).reset_index(drop=True)
    if doubled:
        p_values = p_values.rename(columns={'Parent' : 'Feature_1', 'Child' : 'Feature_2'})
    return p_values


p_vals = ranked_feature_relationships(X, y)
p_vals.query('Parent != Child and p_value < 0.05').reset_index(drop=True).head(20)


p_vals.query('Parent == Child and p_value < 0.05').reset_index(drop=True).head(20)


symmetric_p_vals = ranked_feature_relationships(X, y, doubled=True)
symmetric_p_vals.query('Feature_1 != Feature_2 and p_value < 0.05').reset_index(drop=True).head(20)


symmetric_p_vals.query('Feature_1 == Feature_2 and p_value < 0.05').reset_index(drop=True).head(20)

