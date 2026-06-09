import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

SOCIAL_PATH = '/kaggle/input/mercor-cheating-detection/social_graph.csv'
TEST_PRED_PATH = '/kaggle/input/mercor-cheating-detection-lgb-xgb-cat/submission.csv' 
ALPHA = 0.2       
MAX_ITER = 2          

def run_LP():
    print("1. Read submission...")
    df_test = pd.read_csv(TEST_PRED_PATH)
    
    test_user_set = set(df_test['user_hash'])

    print("2. Read Social Graph (only keep Test <-> Test edges)...")
    df_social = pd.read_csv(SOCIAL_PATH)
    
    mask = df_social['user_a'].isin(test_user_set) & df_social['user_b'].isin(test_user_set)
    df_edges = df_social[mask].copy()

    user_to_idx = {u: i for i, u in enumerate(df_test['user_hash'])}

    row = df_edges['user_a'].map(user_to_idx).values
    col = df_edges['user_b'].map(user_to_idx).values
    data = np.ones(len(row))
    
    num_users = len(df_test)
    
    adj_matrix = sp.coo_matrix((data, (row, col)), shape=(num_users, num_users))
    adj_matrix = adj_matrix + adj_matrix.T
    
    adj_norm = normalize(adj_matrix, norm='l1', axis=1)

    print("3. Begining LP...")
    
    y_init = df_test['prediction'].values
    y_current = y_init.copy()
    

    node_degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    has_neighbor_mask = node_degrees > 0

    for i in range(MAX_ITER):
        neighbor_avg = adj_norm.dot(y_current)
        y_current[has_neighbor_mask] = (
            ALPHA * neighbor_avg[has_neighbor_mask] + 
            (1 - ALPHA) * y_init[has_neighbor_mask]
        )
        
    submission = df_test.copy()
    submission['prediction'] = y_current
    
    output_file = 'submission.csv'
    submission.to_csv(output_file, index=False)
    print("4. Output submission!")

if __name__ == "__main__":
    run_LP()




