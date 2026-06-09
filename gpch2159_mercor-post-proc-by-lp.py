!git clone https://github.com/chpoonag/kaggle.git ./github_tmp
!pip install -qr ./github_tmp/requirements.txt
!mv ./github_tmp/src ./src
!rm -r ./github_tmp ./sample_data


import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from scipy.stats import rankdata
from matplotlib import pyplot as plt
from IPython.display import display

SOCIAL_PATH = '/kaggle/input/mercor-cheating-detection/social_graph.csv'
TEST_PRED_PATH = '/kaggle/input/mercor-cheating-detection-lgb-xgb-cat/submission.csv' 
ALPHA = 0.15       
MAX_ITER = 2          

RUN_LP = False
ENSEMBLE_SUBMISSION = True


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
    y_current_ranked = rankdata(y_current)
    y_current_ranked = (y_current_ranked-y_current_ranked.min()) / (y_current_ranked.max()-y_current_ranked.min())
    
    submission['prediction'] = y_current
    submission.to_csv(
        'submission_lp_raw_probs.csv', 
        index=False
    )
    print(f"\n<< y_current >>\n")
    display(submission)
    
    # submission['prediction'] = y_current_ranked
    # submission.to_csv(
    #     'submission_lp_ranked.csv', 
    #     index=False
    # )
    # print(f"\n<< y_current_ranked >>\n")
    # display(submission)
    print("4. Output submission!")

    plt.hist(y_current_ranked, bins=30, alpha=.5, label="y_current_ranked")
    plt.hist(y_current, bins=30, alpha=.5, label="y_current")
    plt.legend()
    plt.savefig("hist_submission_preds.png")
    plt.show()


import kagglehub, os, shutil
from src.utils.kaggle_utils import setup_kaggle
from src.utils.file_utils import get_folder_size


def download_and_copy_kernel_files(
    kernel_path: str, 
    all_nb_dst_path: str):
    nb_files_path = kagglehub.notebook_output_download(
        kernel_path, 
        force_download=False
    )
    nb_files_size = get_folder_size(nb_files_path) / 10**9  # Convert to GiB
    dst_path = os.path.join(all_nb_dst_path, kernel_path)
    shutil.copytree(
        nb_files_path, 
        dst_path,
        dirs_exist_ok=True
    )
    cum_output_size = get_folder_size(all_nb_dst_path) / 10**9  # Convert to GiB
    return {"dst_path": dst_path, "cum_output_size" : cum_output_size}


def quantile_quantize_probs(probs, n_bins=10, normalize=True):
    """True quantile splitting - equal # samples per bin."""
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(probs, q=quantiles[1:-1])  # Exclude 0,1 endpoints
    ret = np.digitize(probs, bins)
    if normalize:
        ret = (ret-ret.min()) / (ret.max()-ret.min() + 1e-12)
    return ret
    
def linear_quantize_probs(probs, n_bins=10, normalize=True):
    """Quantize [0,1] probs to 0-n_bins integers."""
    ret = np.digitize(probs, np.linspace(0, 1, n_bins+1))
    if normalize:
        ret = (ret-ret.min()) / (ret.max()-ret.min() + 1e-12)
    return ret


def ensemble_submission(df_dirs):
    base_df = pd.read_csv(df_dirs[0])
    preds = []
    for p in df_dirs:
        df = pd.read_csv(p)
        # df.plot(kind='hist', bins=30)
        plt.hist(df['prediction'], alpha=0.5, bins=30, label=p)
        preds.append(df['prediction'])
    combined_preds = pd.DataFrame(preds).mean()
    base_df['prediction'] = combined_preds
    plt.hist(base_df['prediction'], alpha=0.5, bins=30, label="combined")
    plt.legend()
    plt.show()
    
    return base_df


def shrink_to_extremes_2thresholds(probs, low_threshold=0.3, high_threshold=0.7, scale_factor=0.5):
    """
    Shrink toward nearest extreme using 2 splitting thresholds.
    
    Args:
        probs: [0,1] probability array
        low_threshold: split <low→0, >high→1, middle→shrink to closest
        high_threshold: 
        scale_factor: shrink distance by this factor (0.8=80% closer)
    """
    result = probs.copy()
    
    # < low_threshold → shrink toward 0
    mask_low = probs < low_threshold
    result[mask_low] = probs[mask_low] * scale_factor
    
    # > high_threshold → shrink toward 1
    mask_high = probs > high_threshold
    result[mask_high] = 1 - (1 - probs[mask_high]) * scale_factor
    
    # Middle: shrink toward closest extreme
    mask_middle = (probs >= low_threshold) & (probs <= high_threshold)
    middle_probs = probs[mask_middle]
    pivot = (low_threshold + high_threshold) / 2
    
    low_side = middle_probs < pivot
    result[mask_middle][low_side] = middle_probs[low_side] * scale_factor
    result[mask_middle][~low_side] = 1 - (1 - middle_probs[~low_side]) * scale_factor
    
    return np.clip(result, 0, 1)


download_and_copy_kernel_files(
    kernel_path = "gpch2159/mercor-train/versions/49",
    all_nb_dst_path = "./notebook-outputs"
)


df_dirs = [
    "/kaggle/working/notebook-outputs/gpch2159/mercor-train/versions/49/submission_xgb_gnn.csv",
    '/kaggle/input/mercor-cheating-detection-lgb-xgb-cat/submission.csv' 
]


sub_df = pd.read_csv(TEST_PRED_PATH)
plt.hist(sub_df.prediction, bins=30, label='before', alpha=.5)
sub_df.prediction = shrink_to_extremes_2thresholds(sub_df.prediction)
# sub_df.prediction.plot(kind="hist", bins=30)
plt.hist(sub_df.prediction, bins=30, label='after', alpha=.5)
sub_df.to_csv("submission_shrink.csv", index=False)
sub_df


sub_df = pd.read_csv(TEST_PRED_PATH)
sub_df.prediction = quantile_quantize_probs(sub_df.prediction, n_bins=10)
sub_df.prediction.plot(kind="hist", bins=30)
sub_df.to_csv("submission_quantile.csv", index=False)
sub_df


sub_df = pd.read_csv(TEST_PRED_PATH)
sub_df.prediction = linear_quantize_probs(sub_df.prediction, n_bins=10)
sub_df.prediction.plot(kind="hist", bins=30)
sub_df.to_csv("submission_linear.csv", index=False)
sub_df


# if __name__ == "__main__":
if ENSEMBLE_SUBMISSION:
    sub_df = ensemble_submission(df_dirs=df_dirs)
    sub_df.to_csv("./submission_ensemble.csv", index=False)
if RUN_LP:
    run_LP()




