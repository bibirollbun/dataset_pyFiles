
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


print("--- Block 1: Loading and Preparing Data ---")
try:
    csv_path = '/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv'
    df = pd.read_csv(csv_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Could not find data at Kaggle path. Please update the path.")
    exit()

df_findings = df[df['class_name'] != 'No finding'].copy()
image_summary = df_findings.groupby('image_id').agg(all_labels=('class_name', list)).reset_index()
print(f"Processed {len(image_summary)} unique images with one or more findings.")



print("\n--- Block 2: Creating and Encoding Labeling Versions ---")
# (Functions are unchanged)
def version_1_pure_majority(labels):
    if not labels: return "No_Finding_Consensus"
    return Counter(labels).most_common(1)[0][0]

def version_2_no_clear_majority_is_noise(labels):
    if not labels: return "No_Finding_Consensus"
    counts = Counter(labels).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]: return -1
    return counts[0][0]

def version_3_plus_disagreement_is_noise(labels, disagreement_threshold):
    if not labels: return "No_Finding_Consensus"
    counts = Counter(labels)
    majority_label, majority_count = counts.most_common(1)[0]
    if len(labels) - majority_count >= disagreement_threshold: return -1
    return majority_label

df_versions = image_summary[['image_id']].copy()
all_labels_list = image_summary['all_labels'].tolist()
df_versions['V1_Majority'] = [version_1_pure_majority(labels) for labels in all_labels_list]
df_versions['V2_TieIsNoise'] = [version_2_no_clear_majority_is_noise(labels) for labels in all_labels_list]
df_versions['V3_Noise_d1'] = [version_3_plus_disagreement_is_noise(labels, 1) for labels in all_labels_list]
df_versions['V4_Noise_d2'] = [version_3_plus_disagreement_is_noise(labels, 2) for labels in all_labels_list]
df_versions['V5_Noise_d3'] = [version_3_plus_disagreement_is_noise(labels, 3) for labels in all_labels_list]

all_string_labels = set()
version_cols = ['V1_Majority', 'V2_TieIsNoise', 'V3_Noise_d1', 'V4_Noise_d2', 'V5_Noise_d3']
for col in version_cols:
    for label in df_versions[col].unique():
        if isinstance(label, str): all_string_labels.add(label)
label_encoder = LabelEncoder().fit(list(all_string_labels))
for col in version_cols:
    encoded_col_name = f"{col}_encoded"
    temp_col = df_versions[col].copy()
    string_mask = df_versions[col].apply(lambda x: isinstance(x, str))
    if string_mask.any(): temp_col[string_mask] = label_encoder.transform(df_versions.loc[string_mask, col])
    df_versions[encoded_col_name] = temp_col.astype(int)
print("Versions created and encoded.")


print("\n--- Block 2A: Summary Statistics for Each Labeling Version ---")
summary_data = []
total_points = len(df_versions)
for col in version_cols:
    noise_count = (df_versions[col] == -1).sum()
    summary_data.append({
        'Version': col,
        'Noise Points': noise_count,
        'Noise Percentage (%)': f"{(noise_count/total_points)*100:.2f}",
        'Clean Points': total_points - noise_count,
        'Unique Labels (Clean)': df_versions[df_versions[col] != -1][col].nunique()
    })
print(pd.DataFrame(summary_data))


print("\n--- Block 3: Calculating Comparison Matrices (ARI & AMI) ---")
versions_encoded = [f"{v}_encoded" for v in version_cols]
ari_matrix_i, ami_matrix_i = [pd.DataFrame(index=version_cols, columns=version_cols, dtype=float) for _ in range(2)]
ari_matrix_ii, ami_matrix_ii = [pd.DataFrame(index=version_cols, columns=version_cols, dtype=float) for _ in range(2)]
ari_matrix_iii, ami_matrix_iii = [pd.DataFrame(index=version_cols, columns=version_cols, dtype=float) for _ in range(2)]

print("Pre-computing singleton versions for Method (iii)...")
singleton_versions = {}
for v_name_enc in versions_encoded:
    labels = df_versions[v_name_enc].copy()
    noise_mask = labels == -1
    labels.loc[noise_mask] = np.arange(-2, -2 - noise_mask.sum(), -1)
    singleton_versions[v_name_enc] = labels
print("Pre-computation complete.")

print("Running pairwise comparisons (this may take a few minutes for Method iii)...")
total_pairs = len(versions_encoded) ** 2
pair_count = 0

for i in range(len(versions_encoded)):
    for j in range(len(versions_encoded)):
        pair_count += 1
        v_a_name_enc, v_b_name_enc = versions_encoded[i], versions_encoded[j]
        v_a_name_orig, v_b_name_orig = version_cols[i], version_cols[j]
        
        # ===== SOLUTION 1: Progress Indicator =====
        print(f"--> Processing pair {pair_count}/{total_pairs}: {v_a_name_orig} vs {v_b_name_orig}")

        labels_a = df_versions[v_a_name_enc]
        labels_b = df_versions[v_b_name_enc]

        # Method (i)
        ari_matrix_i.loc[v_a_name_orig, v_b_name_orig] = adjusted_rand_score(labels_a, labels_b)
        ami_matrix_i.loc[v_a_name_orig, v_b_name_orig] = adjusted_mutual_info_score(labels_a, labels_b)

        # Method (ii)
        noise_indices = df_versions[(labels_a == -1) | (labels_b == -1)].index
        clean_labels_a = labels_a.drop(noise_indices)
        clean_labels_b = labels_b.drop(noise_indices)
        if len(clean_labels_a) > 1:
            ari_matrix_ii.loc[v_a_name_orig, v_b_name_orig] = adjusted_rand_score(clean_labels_a, clean_labels_b)
            ami_matrix_ii.loc[v_a_name_orig, v_b_name_orig] = adjusted_mutual_info_score(clean_labels_a, clean_labels_b)
        else:
            ari_matrix_ii.loc[v_a_name_orig, v_b_name_orig] = np.nan
            ami_matrix_ii.loc[v_a_name_orig, v_b_name_orig] = np.nan

        # Method (iii)
        s_a = singleton_versions[v_a_name_enc]
        s_b = singleton_versions[v_b_name_enc]
        s_b_rebased = s_b.copy()
        s_a_noise_min_id = s_a[s_a < 0].min()
        if pd.isna(s_a_noise_min_id): s_a_noise_min_id = -1
        s_b_noise_mask = s_b_rebased < 0
        s_b_rebased.loc[s_b_noise_mask] = np.arange(s_a_noise_min_id - 1, s_a_noise_min_id - 1 - s_b_noise_mask.sum(), -1)
        ari_matrix_iii.loc[v_a_name_orig, v_b_name_orig] = adjusted_rand_score(s_a, s_b_rebased)
        ami_matrix_iii.loc[v_a_name_orig, v_b_name_orig] = adjusted_mutual_info_score(s_a, s_b_rebased)

print("Successfully calculated all comparison matrices.")


print("\n--- Saving Processed Data and Visualizing ---")
output_dir = 'project_outputs'
os.makedirs(output_dir, exist_ok=True)
csv_output_path = os.path.join(output_dir, 'vindr_cxr_ground_truth_versions.csv')
df_versions.to_csv(csv_output_path, index=False)
print(f"Processed DataFrame saved to: {csv_output_path}")

def plot_heatmap(matrix, title, filename, cmap='viridis'):
    plt.figure(figsize=(12, 9))
    sns.heatmap(matrix, annot=True, cmap=cmap, fmt=".3f", vmin=-0.1, vmax=1, annot_kws={"size": 12})
    plt.title(title, fontsize=18, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Heatmap saved to: {filename}")

matrices = {'ARI': [ari_matrix_i, ari_matrix_ii, ari_matrix_iii], 'AMI': [ami_matrix_i, ami_matrix_ii, ami_matrix_iii]}
method_titles = ['Method (i) Noise as single cluster', 'Method (ii) Non-Noise Points Only', 'Method (iii) Noise as singletons']
cmaps = ['viridis', 'plasma', 'magma']

for metric_name, matrix_list in matrices.items():
    for i, matrix in enumerate(matrix_list):
        title = f'{metric_name} Comparison | {method_titles[i]}'
        filename = os.path.join(output_dir, f'{metric_name}_{method_titles[i].replace(" ", "_")}.png')
        plot_heatmap(matrix, title, filename, cmap=cmaps[i])

print("\n\nProject Complete.")


import pandas as pd
from collections import Counter
import os
import pydicom
import matplotlib.pyplot as plt

# ---
# This script finds two example images from the VinDR-CXR dataset:
# 1. High Agreement: An image where multiple radiologists all gave the exact same diagnosis.
# 2. High Disagreement: An image where radiologists provided several different diagnoses.
# It then finds the file paths of these images, saves them as PNG files,
# and saves all information to a text file.
# ---

# Load the training data from the standard Kaggle path
try:
    csv_path = '/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv'
    df = pd.read_csv(csv_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Could not find data at Kaggle path. Please ensure the path is correct.")
    exit()

# Remove 'No finding' class
df_findings = df[df['class_name'] != 'No finding'].copy()

# Group by image_id to get a list of all diagnoses
image_summary = df_findings.groupby('image_id').agg(
    all_labels=('class_name', list)
).reset_index()

# Add helper columns
image_summary['num_labels'] = image_summary['all_labels'].apply(len)
image_summary['unique_labels'] = image_summary['all_labels'].apply(lambda x: len(set(x)))


# --- Find a "High Agreement" Image ---
high_agreement_candidates = image_summary[
    (image_summary['num_labels'] > 2) & 
    (image_summary['unique_labels'] == 1)
]
high_agreement_example = high_agreement_candidates.iloc[0] if not high_agreement_candidates.empty else None


# --- Find a "High Disagreement" Image ---
high_disagreement_candidates = image_summary[image_summary['unique_labels'] >= 3]
high_disagreement_example = high_disagreement_candidates.iloc[0] if not high_disagreement_candidates.empty else None


# --- Function to find the file path for a given image_id ---
def find_image_path(image_id, search_dir='/kaggle/input/vinbigdata-chest-xray-abnormalities-detection'):
    """Searches for a DICOM file matching the image_id."""
    for root, dirs, files in os.walk(search_dir):
        if f"{image_id}.dicom" in files:
            return os.path.join(root, f"{image_id}.dicom")
    return "File not found."

# ---  Function to read a DICOM file and save it as a PNG ---
def save_dicom_as_png(dicom_path, output_png_path):
    """Reads a DICOM file and saves it as a PNG image."""
    if not os.path.exists(dicom_path):
        print(f"Cannot save PNG. DICOM file not found at: {dicom_path}")
        return
    
    print(f"Reading DICOM file from: {dicom_path}")
    # Read the DICOM file using pydicom
    dicom_file = pydicom.dcmread(dicom_path)
    
    # Get the pixel data
    pixel_array = dicom_file.pixel_array
    
    # Save the pixel data as a PNG using matplotlib
    plt.imsave(output_png_path, pixel_array, cmap='gray')
    print(f"  --> Successfully saved image to: {output_png_path}")


# --- Find paths and save images ---
output_dir = 'project_outputs'
os.makedirs(output_dir, exist_ok=True)

# Process High Agreement Example
if high_agreement_example is not None:
    high_agreement_path = find_image_path(high_agreement_example['image_id'])
    high_agreement_png_path = os.path.join(output_dir, 'high_agreement_example.png')
    save_dicom_as_png(high_agreement_path, high_agreement_png_path)
else:
    high_agreement_path = "N/A"

# Process High Disagreement Example
if high_disagreement_example is not None:
    high_disagreement_path = find_image_path(high_disagreement_example['image_id'])
    high_disagreement_png_path = os.path.join(output_dir, 'high_disagreement_example.png')
    save_dicom_as_png(high_disagreement_path, high_disagreement_png_path)
else:
    high_disagreement_path = "N/A"


# --- Prepare the output content for the text file ---
report_content = []
report_content.append("="*50)
report_content.append("RESULTS")
report_content.append("="*50 + "\n")

if high_agreement_example is not None:
    report_content.append("--- LEFT COLUMN (HIGH AGREEMENT) ---")
    report_content.append(f"Image ID: {high_agreement_example['image_id']}")
    report_content.append(f"Saved Image File: {os.path.basename(high_agreement_png_path)}")
    report_content.append("\nExpert Labels:")
    for label in high_agreement_example['all_labels']:
        report_content.append(f"  - {label}")
    report_content.append("\nResult:")
    report_content.append(f"  - V1-V5 Label: {high_agreement_example['all_labels'][0]}")
    report_content.append("  - This is a 'clean' data point. All our versions agree.")
else:
    report_content.append("Could not find a high-agreement example.")

report_content.append("\n" + "-"*50 + "\n")

if high_disagreement_example is not None:
    majority_vote = Counter(high_disagreement_example['all_labels']).most_common(1)[0][0]
    report_content.append("--- RIGHT COLUMN (HIGH DISAGREEMENT) ---")
    report_content.append(f"Image ID: {high_disagreement_example['image_id']}")
    report_content.append(f"Saved Image File: {os.path.basename(high_disagreement_png_path)}")
    report_content.append("\nExpert Labels:")
    for label in high_disagreement_example['all_labels']:
        report_content.append(f"  - {label}")
    report_content.append("\nResult:")
    report_content.append(f"  - V1 Label: {majority_vote} (Majority)")
    report_content.append("  - V3 Label: Noise (1+ Disagreement)")
    report_content.append("  - This is a 'noisy' data point where the final label depends on our rules.")
else:
    report_content.append("Could not find a high-disagreement example.")

report_content.append("\n" + "="*50)

# --- Save the text report and print to console ---
final_report_string = "\n".join(report_content)
output_filepath = os.path.join(output_dir, 'examples.txt')

try:
    with open(output_filepath, 'w') as f:
        f.write(final_report_string)
    print(f"\nSUCCESS: Text report saved to '{output_filepath}'")
except Exception as e:
    print(f"\nERROR: Could not save text report. {e}")

print("\n" + final_report_string)


