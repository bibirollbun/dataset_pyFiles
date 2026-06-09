import numpy as np
import pandas as pd
import pandas.api.types
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

import lightgbm as lgb
from scipy.stats import zscore
from numpy import nanmean, nanstd
np.seterr(invalid='ignore')

import copy
import os


noise_test_candidate = [30]

# prior seeds that helped
magic_noise_seeds = [2, 3, 9, 17, 21, 22, 24, 27]

magic_noise_seeds = magic_noise_seeds + noise_test_candidate

#we are just keeping of track of seeds that didn't help (unused)
bad_seeds = [1, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 23, 25, 26, 28]

#value of 0.0000003 has been determined to provide useful noise level (lower than expected)
#this value should be kept through this experiment
magic_noise_factor = 0.0000003

magic_noise_seeds


df_train = pd.read_csv("/kaggle/input/isic-2024-challenge/train-metadata.csv")
df_test = pd.read_csv("/kaggle/input/isic-2024-challenge/test-metadata.csv")

#verify these are set before committing!!!
max_estimators = 8000    
early_stopping_rounds = 500  #high number assures we train through any "lucky" cv bounces

#set to false if want to see full train results / save models.... (false if want to save on GPU)

SCORING, QUICK_TEST, SAVE_TRAIN = 1, 2, 3

#set to QUICK_TEST when submitting to avoid wasting GPU
mode = QUICK_TEST

#ugly duckling tabular analysis
do_ud = True

#load OOF imagenet preds
load_train_imagepreds_and_cv = True

do_imagenet_uglyduckling = False
import_ud_auged_train = False

#how much gaussian noise to oof image preds
#best LB improvement with 0.005 so far 
oof_imagenet_folds_noise_std = 0.005  # Standard deviation of the Gaussian noise

#assures mode = SCORING on submit
if len(df_test) > 3:
    mode = SCORING

if mode == QUICK_TEST:
    df_train = df_train.head(50000)  #10k too few for successful train
    max_estimators = 500
    do_ud = False  #True for saving out augmented train data
    
if mode == SAVE_TRAIN:
    max_estimators = 500
    do_ud = True  #True for saving out augmented train data

#to keep same code base across imagenet / non-imagenet notebooks
main_imagenet_column_name = "imagenet_predict"
image_net_columns = [main_imagenet_column_name]

df_train


if load_train_imagepreds_and_cv:
    oof_image_net_preds = pd.read_csv("/kaggle/input/isic-2024-imagenet-gen-2-output/oof_predictions.csv")

    # Rename the 'oof_prediction' column to 'imagenet_predict'
    oof_image_net_preds = oof_image_net_preds.rename(columns={'oof_prediction': main_imagenet_column_name})

    # Merge with df_train
    df_train = df_train.merge(oof_image_net_preds[['fold', 'imagenet_predict']], left_index=True, right_index=True, how='left')


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def comp_score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, min_tpr: float=0.80):
    v_gt = abs(np.floor(np.asarray(solution.values))-1)
    v_pred = np.array([1.0 - x for x in submission.values])
    max_fpr = abs(1-min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return partial_auc

def pauc_score_func(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    y_true_df = pd.DataFrame(y_true, columns=['target'])
    y_pred_df = pd.DataFrame(y_pred, columns=['prediction'])
    return comp_score(y_true_df, y_pred_df, "", min_tpr=0.80)

def process_imagenet_column(df, main_imagenet_column_name, target_column_name, random_seed=42, noise_mean=0, noise_std=0.01):
    # Set the random seed for reproducibility 
    np.random.seed(random_seed) 
    
    noise = np.random.normal(loc=noise_mean, scale=noise_std, size=len(df))
    df[main_imagenet_column_name] += noise
    
    # Clip the adjusted values to ensure they remain between 0 and 1
    df[main_imagenet_column_name] = df[main_imagenet_column_name].clip(0, 1)
    
    return df

print("Before adjust pauc:", pauc_score_func(df_train["target"], df_train["imagenet_predict"]))

target_column_name = 'target'
random_seed = 42  # Set a fixed random seed for reproducibility

df_train = process_imagenet_column(df_train, main_imagenet_column_name, target_column_name, random_seed, noise_mean=0, noise_std=oof_imagenet_folds_noise_std)

# Print the final results
print("\nFinal results:")
print(df_train[[main_imagenet_column_name, target_column_name]].head())

print("After adjust pauc:", pauc_score_func(df_train["target"], df_train["imagenet_predict"]))



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import timm
from torchvision import transforms
from PIL import Image
import io
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ISICDataset(Dataset):
    def __init__(self, hdf5_file, isic_ids, targets=None, transform=None):
        self.hdf5_file = h5py.File(hdf5_file, 'r')
        self.isic_ids = isic_ids
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, idx):
        img_bytes = self.hdf5_file[self.isic_ids[idx]][()]
        img = Image.open(io.BytesIO(img_bytes))
        img = np.array(img)
        
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']
        
        target = self.targets[idx] if self.targets is not None else torch.tensor(-1)
        return img, target

    def __del__(self):
        self.hdf5_file.close()

base_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

@torch.no_grad()
def ensemble_predict(models, test_loader, device):
    all_predictions = []
    
    for inputs, _ in tqdm(test_loader, desc="Predicting"):
        inputs = inputs.to(device)
        fold_predictions = torch.stack([model(inputs).softmax(dim=1)[:, 1] for model in models])
        avg_predictions = fold_predictions.mean(dim=0)
        all_predictions.append(avg_predictions.cpu())
    
    return torch.cat(all_predictions).numpy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df_test = pd.read_csv("/kaggle/input/isic-2024-challenge/test-metadata.csv")
TEST_HDF5_FILE_PATH = '/kaggle/input/isic-2024-challenge/test-image.hdf5'
TRAIN_HDF5_FILE_PATH = '/kaggle/input/isic-2024-challenge/train-image.hdf5'

#only for verifiying can work with larger test dataset
#df_test = pd.read_csv("/kaggle/input/isic-2024-challenge/train-metadata.csv")
#TEST_HDF5_FILE_PATH = '/kaggle/input/isic-2024-challenge/train-image.hdf5'

model_configs = [
    ("/kaggle/input/isic-2024-multifold-v2-offsite-train/v2_model_fold_2_epoch_1.pth", 'tf_efficientnetv2_b1'),
    ("/kaggle/input/isic-2024-multifold-v2-offsite-train/v2_model_fold_4_epoch_1.pth", 'tf_efficientnetv2_b1'),
    ("/kaggle/input/imagenet-143lb-from-isic-2024-imagenet-model-a/best_model.pth", 'tf_efficientnetv2_b1'),
    ("/kaggle/input/isic-2024-effnetv2b0-lb-0-151/effnetv2b0_151lb_isic2024.pth", 'tf_efficientnet_b0')
]

models = [timm.create_model(model_type, pretrained=False, num_classes=2).to(device) for _, model_type in model_configs]
for model, (model_path, _) in zip(models, model_configs):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

test_dataset = ISICDataset(
    hdf5_file=TEST_HDF5_FILE_PATH,
    isic_ids=df_test['isic_id'].values,
    transform=base_transform,
)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

predictions = ensemble_predict(models, test_loader, device)
df_test[main_imagenet_column_name] = predictions

print(df_test[main_imagenet_column_name].head())



from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

BATCH_SIZE = 128
NUM_WORKERS = 4

# Set a fixed number of threads for NumPy
os.environ['OMP_NUM_THREADS'] = '1'

def get_infer_model(num_classes=2, model_class = "", model_file = "", is_imagenet = False):
    
    model = timm.create_model(model_class, pretrained=False)
    
    # Replace the classifier if your MODEL_PATH doesn't include it
    # If MODEL_PATH includes the classifier, you can remove these lines
    if not is_imagenet:
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, num_classes)
    
    # Load your trained weights
    if torch.cuda.is_available(): 
        model.load_state_dict(torch.load(model_file))
    else:
        model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        
    model = model.to(device)
    
    return model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Set seed for reproducibility
set_seed()

# Ensure deterministic GPU operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

def extract_features(model, loader, device):
    features = []
    isic_ids = []
    model.eval()
    with torch.no_grad():
        for inputs, batch_isic_ids in tqdm(loader, desc="Extracting features"):
            inputs = inputs.to(device)
            output = model(inputs)
            features.append(output.cpu())
            isic_ids.extend(batch_isic_ids)
    return torch.cat(features).numpy(), isic_ids

def calculate_outlier_scores(features):
    print("Normalizing features...")
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    print("Calculating Isolation Forest scores...")
    iso_forest = IsolationForest(contamination=0.1, random_state=42, n_jobs=1)  # n_jobs=1 for determinism
    outlier_scores_if = iso_forest.fit_predict(features_normalized)
    outlier_scores_if = (outlier_scores_if * -1 + 1) / 2
    
    print("Calculating Local Outlier Factor scores...")
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, n_jobs=1)  # n_jobs=1 for determinism
    outlier_scores_lof = lof.fit_predict(features_normalized)
    outlier_scores_lof = (outlier_scores_lof * -1 + 1) / 2
    
    print("Combining scores...")
    outlier_scores_combined = (outlier_scores_if + outlier_scores_lof) / 2
    
    return outlier_scores_if, outlier_scores_lof, outlier_scores_combined

def add_outlier_scores_to_df(df, isic_ids, outlier_scores_if, outlier_scores_lof, outlier_scores_combined):
    print("Adding outlier scores to dataframe...")
    temp_df = pd.DataFrame({
        'isic_id': isic_ids,
        'outlier_score_if': outlier_scores_if,
        'outlier_score_lof': outlier_scores_lof,
        'outlier_score_combined': outlier_scores_combined
    })
    return df.merge(temp_df, on='isic_id', how='left')

def gpu_correlation(tensor):
    centered = tensor - tensor.mean(dim=0)
    cov = torch.matmul(centered.t(), centered) / (tensor.size(0) - 1)
    std = torch.sqrt(torch.diag(cov))
    cor = cov / torch.ger(std, std)
    return cor

def plot_correlation_heatmap(df, target_column, feature_columns, device):
    print("Plotting correlation heatmap...")
    tensor = torch.tensor(df[feature_columns + [target_column]].values, dtype=torch.float32).to(device)
    corr = gpu_correlation(tensor)
    corr_cpu = corr.cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_cpu, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                xticklabels=feature_columns + [target_column],
                yticklabels=feature_columns + [target_column])
    plt.title('Correlation Heatmap: Outlier Scores vs Target')
    plt.show()
    
    return corr_cpu

if do_imagenet_uglyduckling:
    print(f"Using device: {device}")

    print("Loading the model...")
    model = get_infer_model(num_classes=2, model_class="tf_efficientnetv2_b1", model_file="/kaggle/input/isic-2024-tf-efficientnetv2-b1/best_model.pth", is_imagenet=False)

    # Recreate DataLoaders with deterministic settings
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, 
                              pin_memory=True, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, 
                             pin_memory=True, worker_init_fn=seed_worker, generator=g)

    print("Extracting features for train set...")
    train_features, train_isic_ids = extract_features(model, train_loader, device)
    print("Extracting features for test set...")
    test_features, test_isic_ids = extract_features(model, test_loader, device)

    print("Calculating outlier scores for train set...")
    train_outlier_scores_if, train_outlier_scores_lof, train_outlier_scores_combined = calculate_outlier_scores(train_features)
    print("Calculating outlier scores for test set...")
    test_outlier_scores_if, test_outlier_scores_lof, test_outlier_scores_combined = calculate_outlier_scores(test_features)

    print("Adding outlier scores to dataframes...")
    df_train = add_outlier_scores_to_df(df_train, train_isic_ids, train_outlier_scores_if, train_outlier_scores_lof, train_outlier_scores_combined)
    df_test = add_outlier_scores_to_df(df_test, test_isic_ids, test_outlier_scores_if, test_outlier_scores_lof, test_outlier_scores_combined)

    outlier_score_columns = ['outlier_score_if', 'outlier_score_lof', 'outlier_score_combined']
    corr_matrix = plot_correlation_heatmap(df_train, 'target', outlier_score_columns, device)

    print("Correlations with target:")
    target_correlations = corr_matrix[-1, :-1]
    for col, corr in zip(outlier_score_columns, target_correlations):
        print(f"{col}: {corr:.4f}")
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Delete large variables
    del train_features, test_features
    del train_outlier_scores_if, train_outlier_scores_lof, train_outlier_scores_combined
    del test_outlier_scores_if, test_outlier_scores_lof, test_outlier_scores_combined
    del corr_matrix

    # Force garbage collection
    import gc
    gc.collect()

    image_net_columns.extend(outlier_score_columns) 
    print(image_net_columns)


num_cols = [
    'age_approx', 'clin_size_long_diam_mm', 'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext', 
    'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L', 
    'tbp_lv_Lext', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean', 
    'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB',
    'tbp_lv_deltaLBnorm', 'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM',
    'tbp_lv_nevi_confidence', 'tbp_lv_norm_border', 'tbp_lv_norm_color',
    'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdL',
    'tbp_lv_stdLExt', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle',
    'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z',
]

num_cols = num_cols + image_net_columns

# anatom_site_general (why isn't this in here?)
cat_cols = ["sex", "tbp_tile_type", "tbp_lv_location", "tbp_lv_location_simple"]

def feature_engineering(df):
    df["lesion_size_ratio"] = df["tbp_lv_minorAxisMM"] / df["clin_size_long_diam_mm"]
    df["lesion_shape_index"] = df["tbp_lv_areaMM2"] / (df["tbp_lv_perimeterMM"] ** 2)
    df["hue_contrast"] = (df["tbp_lv_H"] - df["tbp_lv_Hext"]).abs()
    df["luminance_contrast"] = (df["tbp_lv_L"] - df["tbp_lv_Lext"]).abs()
    df["lesion_color_difference"] = np.sqrt(df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2)
    df["border_complexity"] = df["tbp_lv_norm_border"] + df["tbp_lv_symm_2axis"]
    df["color_uniformity"] = df["tbp_lv_color_std_mean"] / df["tbp_lv_radial_color_std_max"]
    df["3d_position_distance"] = np.sqrt(df["tbp_lv_x"] ** 2 + df["tbp_lv_y"] ** 2 + df["tbp_lv_z"] ** 2) 
    df["perimeter_to_area_ratio"] = df["tbp_lv_perimeterMM"] / df["tbp_lv_areaMM2"]
    df["lesion_visibility_score"] = df["tbp_lv_deltaLBnorm"] + df["tbp_lv_norm_color"]
    df["combined_anatomical_site"] = df["anatom_site_general"] + "_" + df["tbp_lv_location"]
    df["symmetry_border_consistency"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_norm_border"]
    df["color_consistency"] = df["tbp_lv_stdL"] / df["tbp_lv_Lext"]
    
    df["size_age_interaction"] = df["clin_size_long_diam_mm"] * df["age_approx"]
    df["hue_color_std_interaction"] = df["tbp_lv_H"] * df["tbp_lv_color_std_mean"]
    df["lesion_severity_index"] = (df["tbp_lv_norm_border"] + df["tbp_lv_norm_color"] + df["tbp_lv_eccentricity"]) / 3
    df["shape_complexity_index"] = df["border_complexity"] + df["lesion_shape_index"]
    df["color_contrast_index"] = df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"] + df["tbp_lv_deltaLBnorm"]
    df["log_lesion_area"] = np.log(df["tbp_lv_areaMM2"] + 1)
    df["normalized_lesion_size"] = df["clin_size_long_diam_mm"] / df["age_approx"]
    df["mean_hue_difference"] = (df["tbp_lv_H"] + df["tbp_lv_Hext"]) / 2
    df["std_dev_contrast"] = np.sqrt((df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2) / 3)
    df["color_shape_composite_index"] = (df["tbp_lv_color_std_mean"] + df["tbp_lv_area_perim_ratio"] + df["tbp_lv_symm_2axis"]) / 3
    df["3d_lesion_orientation"] = np.arctan2(df_train["tbp_lv_y"], df_train["tbp_lv_x"])
    df["overall_color_difference"] = (df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"]) / 3
    df["symmetry_perimeter_interaction"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_perimeterMM"]
    df["comprehensive_lesion_index"] = (df["tbp_lv_area_perim_ratio"] + df["tbp_lv_eccentricity"] + df["tbp_lv_norm_color"] + df["tbp_lv_symm_2axis"]) / 4

    new_num_cols = [
        "lesion_size_ratio", "lesion_shape_index", "hue_contrast",
        "luminance_contrast", "lesion_color_difference", "border_complexity",
        "color_uniformity", "3d_position_distance", "perimeter_to_area_ratio",
        "lesion_visibility_score", "symmetry_border_consistency", "color_consistency",

        "size_age_interaction", "hue_color_std_interaction", "lesion_severity_index", 
        "shape_complexity_index", "color_contrast_index", "log_lesion_area",
        "normalized_lesion_size", "mean_hue_difference", "std_dev_contrast",
        "color_shape_composite_index", "3d_lesion_orientation", "overall_color_difference",
        "symmetry_perimeter_interaction", "comprehensive_lesion_index",
    ]
    new_cat_cols = ["combined_anatomical_site"]
    
    return df, new_num_cols, new_cat_cols

# Generate features for test and train
df_train, new_num_cols, new_cat_cols = feature_engineering(df_train.copy())
df_test, _, _ = feature_engineering(df_test.copy())

num_cols = num_cols + new_num_cols
cat_cols = cat_cols + new_cat_cols


def additional_feature_engineering(df):
    # Asymmetry features
    df['asymmetry_ratio'] = df['tbp_lv_symm_2axis'] / df['tbp_lv_perimeterMM']
    df['asymmetry_area_ratio'] = df['tbp_lv_symm_2axis'] / df['tbp_lv_areaMM2']

    # Color variation features
    df['color_variation_intensity'] = df['tbp_lv_norm_color'] * df['tbp_lv_deltaLBnorm']
    df['color_contrast_ratio'] = df['tbp_lv_deltaLBnorm'] / (df['tbp_lv_L'] + 1e-5)

    # Shape complexity features
    df['shape_irregularity'] = df['tbp_lv_perimeterMM'] / (2 * np.sqrt(np.pi * df['tbp_lv_areaMM2']))
    df['border_density'] = df['tbp_lv_norm_border'] / df['tbp_lv_perimeterMM']

    # Size-related features
    df['size_age_ratio'] = df['clin_size_long_diam_mm'] / (df['age_approx'] + 1e-5)
    df['area_diameter_ratio'] = df['tbp_lv_areaMM2'] / (df['clin_size_long_diam_mm']**2 + 1e-5)

    # Location-based features
    df['location_size_interaction'] = df.apply(lambda row: f"{row['tbp_lv_location_simple']}_{bin_size(row['clin_size_long_diam_mm'])}", axis=1)
    df['location_age_interaction'] = df.apply(lambda row: f"{row['tbp_lv_location_simple']}_{bin_age(row['age_approx'])}", axis=1)

    # 3D position features
    df['3d_position_norm'] = np.sqrt(df['tbp_lv_x']**2 + df['tbp_lv_y']**2 + df['tbp_lv_z']**2)
    df['3d_position_angle_xy'] = np.arctan2(df['tbp_lv_y'], df['tbp_lv_x'])
    df['3d_position_angle_xz'] = np.arctan2(df['tbp_lv_z'], df['tbp_lv_x'])

    # Color space transformations
    df['lab_chroma'] = np.sqrt(df['tbp_lv_A']**2 + df['tbp_lv_B']**2)
    df['lab_hue'] = np.arctan2(df['tbp_lv_B'], df['tbp_lv_A'])

    # Texture-related features
    df['texture_contrast'] = df['tbp_lv_stdL'] / (df['tbp_lv_L'] + 1e-5)
    df['texture_uniformity'] = 1 / (1 + df['tbp_lv_color_std_mean'])

    # Color difference features
    df['color_difference_AB'] = np.sqrt(df['tbp_lv_deltaA']**2 + df['tbp_lv_deltaB']**2)
    df['color_difference_total'] = np.sqrt(df['tbp_lv_deltaA']**2 + df['tbp_lv_deltaB']**2 + df['tbp_lv_deltaL']**2)

    # Anatomical site encoding
    df['anatom_site_encoded'] = df['anatom_site_general'].map(anatom_site_encoding)

    # Sex encoding
    df['sex_encoded'] = df['sex'].map({'male': 0, 'female': 1})

    return df

def bin_size(size):
    if size < 5:
        return 'very_small'
    elif size < 10:
        return 'small'
    elif size < 20:
        return 'medium'
    else:
        return 'large'

def bin_age(age):
    if age < 30:
        return 'young'
    elif age < 60:
        return 'middle_aged'
    else:
        return 'senior'

# Encoding for anatomical sites
anatom_site_encoding = {
    'torso': 0,
    'upper extremity': 1,
    'lower extremity': 2,
    'head/neck': 3,
    'palms/soles': 4,
    'oral/genital': 5
}

# Add these new features to your existing feature engineering pipeline
df_train = additional_feature_engineering(df_train)
df_test = additional_feature_engineering(df_test)

# Update your feature lists
new_num_cols = [
    'asymmetry_ratio', 'asymmetry_area_ratio', 'color_variation_intensity',
    'color_contrast_ratio', 'shape_irregularity', 'border_density',
    'size_age_ratio', 'area_diameter_ratio', '3d_position_norm',
    '3d_position_angle_xy', '3d_position_angle_xz', 'lab_chroma', 'lab_hue',
    'texture_contrast', 'texture_uniformity', 'color_difference_AB',
    'color_difference_total', 'anatom_site_encoded', 'sex_encoded'
]
new_cat_cols = ['location_size_interaction', 'location_age_interaction']

num_cols += new_num_cols
cat_cols += new_cat_cols


def ugly_duckling_processing(df, num_cols):
    ud_columns = num_cols.copy()
    new_num_cols = []
    
    #if false - only do location-based ugly ducklings
    include_patient_wide_ud = False  
    
    counter = 0
    
    def calc_ugly_duckling_scores(group, grouping):
        nonlocal counter
        counter += 1
        if counter % 10 == 0: print(".", end="", flush=True)
        z_scores = group[ud_columns].apply(lambda x: zscore(x, nan_policy='omit'))
        ud_scores = np.abs(z_scores)
        prefix = 'ud_' if grouping == 'patient' else 'ud_loc_'
        ud_scores.columns = [f'{prefix}{col}' for col in ud_columns]
        return ud_scores

    print("Analyzing ducklings", end="", flush=True)
    ud_location_col = 'tbp_lv_location'
    ud_scores_loc = df.groupby(['patient_id', ud_location_col])[ud_columns + ['patient_id', ud_location_col]].apply(
        lambda x: calc_ugly_duckling_scores(x, 'location')
    ).reset_index(level=[0, 1], drop=True)
    
    print("\nConcat ducklings")
    df = pd.concat([df, ud_scores_loc], axis=1)
    
    if include_patient_wide_ud:
        print("Analyzing ducklings (part 2)", end="", flush=True)
        ud_scores_patient = df.groupby('patient_id')[ud_columns + ['patient_id']].apply(
            lambda x: calc_ugly_duckling_scores(x, 'patient')
        ).reset_index(level=0, drop=True)
        df = pd.concat([df, ud_scores_patient], axis=1)
        print()  # New line after progress indicator

    print("Extending ducklings")
    new_num_cols.extend([f'ud_loc_{col}' for col in ud_columns])
    if include_patient_wide_ud:
        new_num_cols.extend([f'ud_{col}' for col in ud_columns])

    print("Enhancing ugly duckling features", end="", flush=True)
    
    # 1. Percentile-based ugly duckling scores
    def calc_percentile_ud_scores(group):
        nonlocal counter
        counter += 1
        if counter % 10 == 0: print(".", end="", flush=True)
        percentiles = group[ud_columns].rank(pct=True)
        return percentiles.add_prefix('ud_percentile_')
    
    counter = 0  # Reset counter for percentile calculation
    ud_percentiles = df.groupby('patient_id')[ud_columns].apply(calc_percentile_ud_scores).reset_index(level=0, drop=True)
    df = pd.concat([df, ud_percentiles], axis=1)
    new_num_cols.extend([f'ud_percentile_{col}' for col in ud_columns])
    print()  # New line after progress indicator

    # 2. Ugly duckling count features
    threshold = 2.0  # You can adjust this threshold
    if include_patient_wide_ud:
        ud_count = (df[[f'ud_{col}' for col in ud_columns]].abs() > threshold).sum(axis=1)
        df['ud_count_patient'] = ud_count
        new_num_cols.append('ud_count_patient')
    
    ud_count_loc = (df[[f'ud_loc_{col}' for col in ud_columns]].abs() > threshold).sum(axis=1)
    df['ud_count_location'] = ud_count_loc
    new_num_cols.append('ud_count_location')

    # 3. Ugly duckling severity features
    if include_patient_wide_ud:
        df['ud_max_severity_patient'] = df[[f'ud_{col}' for col in ud_columns]].abs().max(axis=1)
        new_num_cols.append('ud_max_severity_patient')
    df['ud_max_severity_location'] = df[[f'ud_loc_{col}' for col in ud_columns]].abs().max(axis=1)
    new_num_cols.append('ud_max_severity_location')

    # 4. Ugly duckling consistency features
    if include_patient_wide_ud:
        df['ud_consistency_patient'] = df[[f'ud_{col}' for col in ud_columns]].abs().std(axis=1)
        new_num_cols.append('ud_consistency_patient')
    df['ud_consistency_location'] = df[[f'ud_loc_{col}' for col in ud_columns]].abs().std(axis=1)
    new_num_cols.append('ud_consistency_location')

    return df, new_num_cols

if do_ud:
    df_train, ud_num_cols = ugly_duckling_processing(df_train.copy(), num_cols)
    df_test, _ = ugly_duckling_processing(df_test.copy(), num_cols)

    # Update the list of columns to train on
    num_cols = num_cols + ud_num_cols


train_cols = num_cols + cat_cols


if mode == SAVE_TRAIN:
    df_train.to_csv("isic_2024_train_auged.csv", index=False)


if import_ud_auged_train:
    #everything including catcols
    train_cols = ['age_approx', 'clin_size_long_diam_mm', 'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext', 'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L', 'tbp_lv_Lext', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean', 'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm', 'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM', 'tbp_lv_nevi_confidence', 'tbp_lv_norm_border', 'tbp_lv_norm_color', 'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdL', 'tbp_lv_stdLExt', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z', 'lesion_size_ratio', 'lesion_shape_index', 'hue_contrast', 'luminance_contrast', 'lesion_color_difference', 'border_complexity', 'color_uniformity', '3d_position_distance', 'perimeter_to_area_ratio', 'lesion_visibility_score', 'symmetry_border_consistency', 'color_consistency', 'size_age_interaction', 'hue_color_std_interaction', 'lesion_severity_index', 'shape_complexity_index', 'color_contrast_index', 'log_lesion_area', 'normalized_lesion_size', 'mean_hue_difference', 'std_dev_contrast', 'color_shape_composite_index', '3d_lesion_orientation', 'overall_color_difference', 'symmetry_perimeter_interaction', 'comprehensive_lesion_index', 'asymmetry_ratio', 'asymmetry_area_ratio', 'color_variation_intensity', 'color_contrast_ratio', 'shape_irregularity', 'border_density', 'size_age_ratio', 'area_diameter_ratio', '3d_position_norm', '3d_position_angle_xy', '3d_position_angle_xz', 'lab_chroma', 'lab_hue', 'texture_contrast', 'texture_uniformity', 'color_difference_AB', 'color_difference_total', 'anatom_site_encoded', 'sex_encoded', 'ud_loc_age_approx', 'ud_loc_clin_size_long_diam_mm', 'ud_loc_tbp_lv_A', 'ud_loc_tbp_lv_Aext', 'ud_loc_tbp_lv_B', 'ud_loc_tbp_lv_Bext', 'ud_loc_tbp_lv_C', 'ud_loc_tbp_lv_Cext', 'ud_loc_tbp_lv_H', 'ud_loc_tbp_lv_Hext', 'ud_loc_tbp_lv_L', 'ud_loc_tbp_lv_Lext', 'ud_loc_tbp_lv_areaMM2', 'ud_loc_tbp_lv_area_perim_ratio', 'ud_loc_tbp_lv_color_std_mean', 'ud_loc_tbp_lv_deltaA', 'ud_loc_tbp_lv_deltaB', 'ud_loc_tbp_lv_deltaL', 'ud_loc_tbp_lv_deltaLB', 'ud_loc_tbp_lv_deltaLBnorm', 'ud_loc_tbp_lv_eccentricity', 'ud_loc_tbp_lv_minorAxisMM', 'ud_loc_tbp_lv_nevi_confidence', 'ud_loc_tbp_lv_norm_border', 'ud_loc_tbp_lv_norm_color', 'ud_loc_tbp_lv_perimeterMM', 'ud_loc_tbp_lv_radial_color_std_max', 'ud_loc_tbp_lv_stdL', 'ud_loc_tbp_lv_stdLExt', 'ud_loc_tbp_lv_symm_2axis', 'ud_loc_tbp_lv_symm_2axis_angle', 'ud_loc_tbp_lv_x', 'ud_loc_tbp_lv_y', 'ud_loc_tbp_lv_z', 'ud_loc_lesion_size_ratio', 'ud_loc_lesion_shape_index', 'ud_loc_hue_contrast', 'ud_loc_luminance_contrast', 'ud_loc_lesion_color_difference', 'ud_loc_border_complexity', 'ud_loc_color_uniformity', 'ud_loc_3d_position_distance', 'ud_loc_perimeter_to_area_ratio', 'ud_loc_lesion_visibility_score', 'ud_loc_symmetry_border_consistency', 'ud_loc_color_consistency', 'ud_loc_size_age_interaction', 'ud_loc_hue_color_std_interaction', 'ud_loc_lesion_severity_index', 'ud_loc_shape_complexity_index', 'ud_loc_color_contrast_index', 'ud_loc_log_lesion_area', 'ud_loc_normalized_lesion_size', 'ud_loc_mean_hue_difference', 'ud_loc_std_dev_contrast', 'ud_loc_color_shape_composite_index', 'ud_loc_3d_lesion_orientation', 'ud_loc_overall_color_difference', 'ud_loc_symmetry_perimeter_interaction', 'ud_loc_comprehensive_lesion_index', 'ud_loc_asymmetry_ratio', 'ud_loc_asymmetry_area_ratio', 'ud_loc_color_variation_intensity', 'ud_loc_color_contrast_ratio', 'ud_loc_shape_irregularity', 'ud_loc_border_density', 'ud_loc_size_age_ratio', 'ud_loc_area_diameter_ratio', 'ud_loc_3d_position_norm', 'ud_loc_3d_position_angle_xy', 'ud_loc_3d_position_angle_xz', 'ud_loc_lab_chroma', 'ud_loc_lab_hue', 'ud_loc_texture_contrast', 'ud_loc_texture_uniformity', 'ud_loc_color_difference_AB', 'ud_loc_color_difference_total', 'ud_loc_anatom_site_encoded', 'ud_loc_sex_encoded', 'ud_percentile_age_approx', 'ud_percentile_clin_size_long_diam_mm', 'ud_percentile_tbp_lv_A', 'ud_percentile_tbp_lv_Aext', 'ud_percentile_tbp_lv_B', 'ud_percentile_tbp_lv_Bext', 'ud_percentile_tbp_lv_C', 'ud_percentile_tbp_lv_Cext', 'ud_percentile_tbp_lv_H', 'ud_percentile_tbp_lv_Hext', 'ud_percentile_tbp_lv_L', 'ud_percentile_tbp_lv_Lext', 'ud_percentile_tbp_lv_areaMM2', 'ud_percentile_tbp_lv_area_perim_ratio', 'ud_percentile_tbp_lv_color_std_mean', 'ud_percentile_tbp_lv_deltaA', 'ud_percentile_tbp_lv_deltaB', 'ud_percentile_tbp_lv_deltaL', 'ud_percentile_tbp_lv_deltaLB', 'ud_percentile_tbp_lv_deltaLBnorm', 'ud_percentile_tbp_lv_eccentricity', 'ud_percentile_tbp_lv_minorAxisMM', 'ud_percentile_tbp_lv_nevi_confidence', 'ud_percentile_tbp_lv_norm_border', 'ud_percentile_tbp_lv_norm_color', 'ud_percentile_tbp_lv_perimeterMM', 'ud_percentile_tbp_lv_radial_color_std_max', 'ud_percentile_tbp_lv_stdL', 'ud_percentile_tbp_lv_stdLExt', 'ud_percentile_tbp_lv_symm_2axis', 'ud_percentile_tbp_lv_symm_2axis_angle', 'ud_percentile_tbp_lv_x', 'ud_percentile_tbp_lv_y', 'ud_percentile_tbp_lv_z', 'ud_percentile_lesion_size_ratio', 'ud_percentile_lesion_shape_index', 'ud_percentile_hue_contrast', 'ud_percentile_luminance_contrast', 'ud_percentile_lesion_color_difference', 'ud_percentile_border_complexity', 'ud_percentile_color_uniformity', 'ud_percentile_3d_position_distance', 'ud_percentile_perimeter_to_area_ratio', 'ud_percentile_lesion_visibility_score', 'ud_percentile_symmetry_border_consistency', 'ud_percentile_color_consistency', 'ud_percentile_size_age_interaction', 'ud_percentile_hue_color_std_interaction', 'ud_percentile_lesion_severity_index', 'ud_percentile_shape_complexity_index', 'ud_percentile_color_contrast_index', 'ud_percentile_log_lesion_area', 'ud_percentile_normalized_lesion_size', 'ud_percentile_mean_hue_difference', 'ud_percentile_std_dev_contrast', 'ud_percentile_color_shape_composite_index', 'ud_percentile_3d_lesion_orientation', 'ud_percentile_overall_color_difference', 'ud_percentile_symmetry_perimeter_interaction', 'ud_percentile_comprehensive_lesion_index', 'ud_percentile_asymmetry_ratio', 'ud_percentile_asymmetry_area_ratio', 'ud_percentile_color_variation_intensity', 'ud_percentile_color_contrast_ratio', 'ud_percentile_shape_irregularity', 'ud_percentile_border_density', 'ud_percentile_size_age_ratio', 'ud_percentile_area_diameter_ratio', 'ud_percentile_3d_position_norm', 'ud_percentile_3d_position_angle_xy', 'ud_percentile_3d_position_angle_xz', 'ud_percentile_lab_chroma', 'ud_percentile_lab_hue', 'ud_percentile_texture_contrast', 'ud_percentile_texture_uniformity', 'ud_percentile_color_difference_AB', 'ud_percentile_color_difference_total', 'ud_percentile_anatom_site_encoded', 'ud_percentile_sex_encoded', 'ud_count_location', 'ud_max_severity_location', 'ud_consistency_location', 'sex', 'tbp_tile_type', 'tbp_lv_location', 'tbp_lv_location_simple', 'combined_anatomical_site', 'location_size_interaction', 'location_age_interaction']
    cat_cols = ['sex', 'tbp_tile_type', 'tbp_lv_location', 'tbp_lv_location_simple', 'combined_anatomical_site', 'location_size_interaction', 'location_age_interaction']

    df_train = pd.read_csv("/kaggle/input/isic-2024-tabular-feature-generation/isic_2024_train_auged.csv")

    missing_cols = set(df_train.columns) - set(df_test.columns)
    new_cols = pd.DataFrame({col: df_train[col].iloc[0] for col in missing_cols}, index=df_test.index).astype(df_train[list(missing_cols)].dtypes)
    df_test = pd.concat([df_test, new_cols], axis=1)[df_train.columns]


# Encode categories
category_encoder = OrdinalEncoder(
    categories='auto',
    dtype=int,
    handle_unknown='use_encoded_value',
    unknown_value=-2,
    encoded_missing_value=-1,
)

X_cat = category_encoder.fit_transform(df_train[cat_cols])
for c, cat_col in enumerate(cat_cols):
    df_train[cat_col] = X_cat[:, c]

X_cat = category_encoder.fit_transform(df_test[cat_cols])
for c, cat_col in enumerate(cat_cols):
    df_test[cat_col] = X_cat[:, c]


#stuff to remove
base_removals_1 = [
    "ud_percentile_log_lesion_area", "ud_count_location", "texture_uniformity", "ud_loc_tbp_lv_y",
    "log_lesion_area", "ud_percentile_symmetry_border_consistency", "ud_percentile_3d_position_distance", "ud_loc_color_contrast_ratio",
    "ud_loc_3d_position_angle_xy", "tbp_lv_perimeterMM", "normalized_lesion_size", "ud_loc_area_diameter_ratio",
    "ud_consistency_location", "tbp_lv_deltaA", "tbp_lv_C", "ud_percentile_tbp_lv_norm_color",
    "ud_loc_std_dev_contrast", "ud_loc_texture_uniformity", "ud_loc_tbp_lv_radial_color_std_max", "perimeter_to_area_ratio",
    "lab_hue", "ud_loc_color_consistency", "tbp_lv_B", "ud_percentile_age_approx",
    "ud_loc_3d_position_distance", "ud_percentile_lesion_shape_index", "ud_loc_tbp_lv_area_perim_ratio", "tbp_lv_minorAxisMM",
    "ud_loc_tbp_lv_deltaB", "ud_percentile_color_contrast_index", "ud_loc_overall_color_difference", "shape_complexity_index",
    "ud_loc_border_complexity", "sex_encoded", "mean_hue_difference", "ud_loc_lesion_visibility_score",
    "tbp_lv_A", "tbp_lv_Bext", "ud_loc_tbp_lv_stdL", "ud_loc_asymmetry_ratio",
    "ud_loc_tbp_lv_deltaA"
    ]


# Remove items from train_cols that are in features
train_cols = [col for col in train_cols if col not in base_removals_1]


base_removals_2 = ["ud_loc_tbp_lv_eccentricity", "ud_percentile_size_age_interaction", "ud_percentile_tbp_lv_C", "ud_percentile_lesion_visibility_score", "ud_percentile_tbp_lv_L",
                   "ud_loc_tbp_lv_symm_2axis_angle", "asymmetry_area_ratio", "ud_loc_mean_hue_difference", "tbp_lv_H", "ud_percentile_tbp_lv_Lext", "ud_percentile_tbp_lv_deltaL",
                   "ud_percentile_size_age_ratio", "comprehensive_lesion_index", "tbp_lv_deltaB", "ud_loc_symmetry_border_consistency", "ud_percentile_tbp_lv_perimeterMM", "hue_color_std_interaction"]

# Remove items from train_cols that are in features
train_cols = [col for col in train_cols if col not in base_removals_2]


base_removals_3 = ["tbp_lv_Hext", "ud_percentile_color_difference_AB", "hue_contrast", "ud_loc_color_difference_total"]

# Remove items from train_cols that are in features
train_cols = [col for col in train_cols if col not in base_removals_3]

print("Updated train_cols:", train_cols)


if not load_train_imagepreds_and_cv:
    gkf = GroupKFold(n_splits=5)

    df_train["fold"] = -1
    for idx, (train_idx, val_idx) in enumerate(gkf.split(df_train, df_train["target"], groups=df_train["patient_id"])):
        df_train.loc[val_idx, "fold"] = idx


df_train['robustness_score'] = 100

df_train.loc[df_train['lesion_id'].notna(), 'robustness_score'] = 200

df_train.loc[df_train['iddx_1'] == "Indeterminate", 'robustness_score'] = 50

df_train['robustness_score'] += 0.25 * df_train['tbp_lv_dnn_lesion_confidence'].fillna(0)

def create_sample_weights(df, robustness_column='robustness_score', min_weight=1, max_weight=10):
    min_score = df[robustness_column].min()
    max_score = df[robustness_column].max()
    weights = min_weight + (max_weight - min_weight) * (df[robustness_column] - min_score) / (max_score - min_score)
    return weights


class EarlyStoppingException(Exception):
    pass

class EarlyStoppingByPAUC:
    def __init__(self, stopping_cycles, period=1, eval_set=None):
        self.stopping_cycles = stopping_cycles
        self.period = period
        self.eval_set = eval_set
        self.best_score = -np.inf
        self.best_iteration = 0
        self.best_model = None
        self.counter = 0

    def __call__(self, env):
        if self.eval_set is None or env.iteration % self.period != 0:
            return False

        y_true = self.eval_set[0][1]
        y_pred = env.model.predict(self.eval_set[0][0], num_iteration=env.iteration)
        current_score = pauc_score_func(y_true, y_pred)

        print(f"Iteration {env.iteration}, Current pAUC: {current_score:.5f}")

        if current_score > self.best_score:
            self.best_score = current_score
            self.best_iteration = env.iteration
            self.best_model_state = copy.deepcopy(env.model)
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.stopping_cycles:
            print(f"Early stopping at iteration {env.iteration}")
            print(f"Best iteration: {self.best_iteration}")
            print(f"Best pAUC: {self.best_score:.5f}")
            raise EarlyStoppingException()
            return True

        return False

    
def train_lgbm_with_early_stopping(df_train, train_cols, use_early_stopping=True, early_stopping_rounds=100):
    lgb_params = {
        'objective': 'binary',
        "random_state": 42,
        "n_estimators": max_estimators,
        'learning_rate': 0.001,
        'num_leaves': 35,
        'min_data_in_leaf': 55,
        'bagging_freq': 1,
        'pos_bagging_fraction': 0.84,
        'neg_bagging_fraction': 0.11,
        'feature_fraction': 0.55,
        'lambda_l1': 0.21,
        'lambda_l2': 0.7,
        "verbosity": -1
    }
    scores = []
    models = []
    for fold in range(5):
        df_train_fold = df_train[df_train["fold"] != fold].reset_index(drop=True)
        df_valid_fold = df_train[df_train["fold"] == fold].reset_index(drop=True)
        
        train_weights = create_sample_weights(df_train_fold)
        
        train_dataset = lgb.Dataset(df_train_fold[train_cols], df_train_fold["target"], weight=train_weights)
        
        valid_dataset = lgb.Dataset(df_valid_fold[train_cols], df_valid_fold["target"], reference=train_dataset)
        
        eval_set = [(df_valid_fold[train_cols], df_valid_fold["target"])]
        callbacks = []
        
        if use_early_stopping:
            rounds_between_pauc_check = 50
            early_stopping_callback = EarlyStoppingByPAUC(
                stopping_cycles=early_stopping_rounds // rounds_between_pauc_check,
                period=rounds_between_pauc_check,
                eval_set=eval_set
            )
            callbacks.append(early_stopping_callback)
        
        try:
            model = lgb.train(
                lgb_params,
                train_dataset,
                valid_sets=[valid_dataset],
                callbacks=callbacks,
                num_boost_round=lgb_params['n_estimators']
            )

        except EarlyStoppingException:
            print(f"Training stopped early.")
        else:
            print(f"Completed without early stopping")

        model = early_stopping_callback.best_model_state
        score = early_stopping_callback.best_score
        print(f"Using best model from iteration {early_stopping_callback.best_iteration}")
        print(f"Fold {fold} / Partial AUC Score: {score:.5f}")            
        
        scores.append(score)        
        models.append(model)
    
    print("\nAverage pAUC:", np.mean(scores))
    return models, scores

# Usage
use_early_stopping = True
models, scores = train_lgbm_with_early_stopping(df_train, train_cols, use_early_stopping, early_stopping_rounds)
np.mean(scores)


importances = np.mean([model.feature_importance(importance_type='gain') for model in models], axis=0)
df_imp = pd.DataFrame({
    "feature": models[0].feature_name(),  # Assuming all models have the same feature names
    "importance": importances
}).sort_values("importance", ascending=False).reset_index(drop=True)

# Print top 10 most important features
print(df_imp.head(30))


model_directory="saved_models"

def save_models(models):
    # Create the directory if it doesn't exist
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    
    for i, model in enumerate(models):
        model_path = os.path.join(model_directory, f"model_fold_{i}.txt")
        model.save_model(model_path)
    
    print(f"Models saved in {model_directory}")

def load_models(directory="saved_models"):
    models = []
    for i in range(5):  # Assuming 5-fold cross-validation
        model_path = os.path.join(directory, f"model_fold_{i}.txt")
        if os.path.exists(model_path):
            model = lgb.Booster(model_file=model_path)
            models.append(model)
    
    print(f"Loaded {len(models)} models from {directory}")
    return models

save_models(models)


import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_probabilities(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    else:
        return sigmoid(model.predict(X))

def stacking_ensemble_lgb_new(models, X_train, y_train, X_test, train_cols, meta_model=LogisticRegression()):
    num_models = len(models)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(y_train), num_models))
    test_preds = np.zeros((len(X_test), num_models))

    for i, model in enumerate(models):
        # Generate out-of-fold predictions
        for train_idx, val_idx in kf.split(X_train):
            oof_preds[val_idx, i] = get_probabilities(model, X_train.iloc[val_idx][train_cols])
        
        # Generate test predictions
        test_preds[:, i] = get_probabilities(model, X_test[train_cols])

    # Train meta-model
    meta_model.fit(oof_preds, y_train)
    final_preds = meta_model.predict_proba(test_preds)[:, 1]
    return final_preds


#just average models (values look different than as produces by ensemble test code)
#preds = np.mean([model.predict(df_test[train_cols]) for model in models], 0)

# Stacking Ensemble
preds = stacking_ensemble_lgb_new(models, df_train, df_train["target"], df_test, train_cols)
preds


def generate_seeded_noise(seed, size, noise_factor):
    np.random.seed(seed)
    return np.random.normal(loc=0, scale=noise_factor, size=size)

def apply_stacked_noise(preds, magic_noise_seeds, noise_factor=0.025):
    stacked_noise = np.zeros_like(preds)
    for i, noise_seed in enumerate(magic_noise_seeds):
        layer_noise = generate_seeded_noise(noise_seed, len(preds), noise_factor)
        stacked_noise += layer_noise
    
    # Modify predictions with stacked noise
    modified_preds = np.clip(preds + stacked_noise, 0, 1)
    
    return modified_preds

preds = apply_stacked_noise(preds, magic_noise_seeds, magic_noise_factor)

preds


df_sub = pd.read_csv("/kaggle/input/isic-2024-challenge/sample_submission.csv")
df_sub["target"] = preds
df_sub.to_csv("submission.csv", index=False)
df_sub

