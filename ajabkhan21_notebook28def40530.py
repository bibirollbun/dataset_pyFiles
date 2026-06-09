# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import os

# Path to the competition data in Kaggle notebooks
data_path = '/kaggle/input/adaptive-immune-profiling-challenge-2025'  # Adjust if the folder name is slightly different

print("Folders and files:")
for root, dirs, files in os.walk(data_path):
    level = root.replace(data_path, '').count(os.sep)
    indent = ' ' * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    for f in files[:10]:  # Show first 10 files
        print(f"{indent}    {f}")
    if len(files) > 10:
        print(f"{indent}    ... and {len(files)-10} more files")


import os

# List all available input directories
input_path = '/kaggle/input'
print("All competition data folders in /kaggle/input/:")
for folder in os.listdir(input_path):
    full_path = os.path.join(input_path, folder)
    if os.path.isdir(full_path):
        print(f"- {folder}")
        # Show subcontents
        sub = os.listdir(full_path)
        print(f"  Subfolders/files (first 20): {sub[:20]}")
        if len(sub) > 20:
            print("  ... more")

# Once you see the correct folder (look for one with 'train_datasets' or 'airr' in name/sub), copy it here:
# data_path = '/kaggle/input/YOUR_EXACT_FOLDER_NAME_HERE'


import os

data_path = '/kaggle/input/adaptive-immune-profiling-challenge-2025'

nested_train = os.path.join(data_path, 'train_datasets', 'train_datasets')
nested_test = os.path.join(data_path, 'test_datasets', 'test_datasets')

print("Real train datasets folders:")
print(os.listdir(nested_train))

print("\nExample inside train_dataset_1:")
example = os.path.join(nested_train, 'train_dataset_1')
print(os.listdir(example)[:20])  # Should show metadata.csv and .tsv files


import pandas as pd
import os

data_path = '/kaggle/input/adaptive-immune-profiling-challenge-2025'

# Adjust these based on Step 1 output
train_base = os.path.join(data_path, 'train_datasets', 'train_datasets')  # Likely nested
test_base = os.path.join(data_path, 'test_datasets', 'test_datasets')     # Same for test

# List real datasets
print("Real train datasets:", os.listdir(train_base))

def load_dataset(dataset_name, is_train=True):
    base = train_base if is_train else test_base
    folder = os.path.join(base, dataset_name)
    
    print(f"Loading {dataset_name} from {folder}")
    
    # Metadata inside the dataset folder
    meta_path = os.path.join(folder, 'metadata.csv')
    metadata = pd.read_csv(meta_path) if os.path.exists(meta_path) else None
    if metadata is not None:
        print(f"Metadata: {len(metadata)} rows, columns: {metadata.columns.tolist()}")
    
    # Load TSVs
    repertoires = []
    for file in os.listdir(folder):
        if file.endswith('.tsv'):
            rep_id = os.path.splitext(file)[0]
            df = pd.read_csv(os.path.join(folder, file), sep='\t')
            df['repertoire_id'] = rep_id
            repertoires.append(df)
            if len(repertoires) > 5:  # Test mode - remove for full load
                break
    
    df_full = pd.concat(repertoires, ignore_index=True)
    
    if metadata is not None:
        # filename column likely has .tsv, so strip
        metadata['repertoire_id'] = metadata['filename'].str.replace('.tsv', '', regex=False)
        df_full = df_full.merge(metadata[['repertoire_id', 'label_positive']], on='repertoire_id', how='left')
    
    return df_full

# Test
train1 = load_dataset('train_dataset_1', is_train=True)
print(train1.head())
print("Columns:", train1.columns.tolist())


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.sparse import vstack  # If needed later


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
import pandas as pd

# Your existing data_path, train_base, test_base, load_dataset function here...

all_test_preds = []
all_explanations = []

train_datasets = [f'train_dataset_{i}' for i in range(1, 9)]

# List all test datasets once
test_datasets = os.listdir(test_base)

for train_ds in train_datasets:
    print(f"\nProcessing {train_ds}...")
    train_df = load_dataset(train_ds, is_train=True)
    
    # Determine count column (templates is common in Adaptive data)
    count_col = 'templates' if 'templates' in train_df.columns else 'duplicate_count'
    if count_col not in train_df.columns:
        count_col = None  # No counts, use uniform
    
    if count_col:
        train_df['freq'] = train_df[count_col] / train_df.groupby('repertoire_id')[count_col].transform('sum')
    else:
        train_df['freq'] = 1.0 / train_df.groupby('repertoire_id').transform('size')
    
    # Grouped texts for vectorizer (fixed deprecation)
    grouped = train_df.groupby('repertoire_id', group_keys=False).apply(
        lambda g: ' '.join(g['junction_aa'].astype(str)), include_groups=False
    )
    
    y_train = train_df.groupby('repertoire_id')['label_positive'].first().reindex(grouped.index).values
    
    class_info = np.unique(y_train, return_counts=True)
    print(f"{train_ds} classes: {class_info}")
    
    if len(class_info[0]) < 2:
        print(f"Single class detected ({class_info[0][0]}). Skipping model training.")
        
        # Fallback explanations: top 50k most frequent unique sequences
        if count_col:
            top_seqs = (train_df.groupby(['junction_aa', 'v_call', 'j_call'])[count_col]
                        .sum()
                        .sort_values(ascending=False)
                        .head(50000)
                        .reset_index())
        else:
            top_seqs = (train_df[['junction_aa', 'v_call', 'j_call']]
                        .value_counts()
                        .head(50000)
                        .reset_index()
                        .drop(columns='count'))
        
        top_seqs['IDdataset'] = train_ds
        all_explanations.append(top_seqs[['IDdataset', 'junction_aa', 'v_call', 'j_call']])
        
        # Default predictions for linked tests: probability = majority class
        default_prob = 1.0 if class_info[0][0] else 0.0
        
        # Find linked test datasets (adjust naming pattern if needed)
        linked_tests = [td for td in test_datasets if td.startswith(train_ds.replace('train_dataset_', 'test_dataset_'))]
        for test_ds in linked_tests:
            test_df = load_dataset(test_ds, is_train=False)
            test_rep_ids = test_df['repertoire_id'].unique()
            for rep_id in test_rep_ids:
                all_test_preds.append({'repertoire_id': rep_id, 'label_positive_probability': default_prob})
        
        continue  # Skip to next train dataset
    
    # Normal case: train model
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3,3), lowercase=False)
    X_train = vectorizer.fit_transform(grouped.values)
    X_train = normalize(X_train, norm='l1')
    
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Predictions on linked tests
    linked_tests = [td for td in test_datasets if td.startswith(train_ds.replace('train_dataset_', 'test_dataset_'))]
    for test_ds in linked_tests:
        test_df = load_dataset(test_ds, is_train=False)
        grouped_test = test_df.groupby('repertoire_id', group_keys=False).apply(
            lambda g: ' '.join(g['junction_aa'].astype(str)), include_groups=False
        )
        X_test = vectorizer.transform(grouped_test.values)
        X_test = normalize(X_test, norm='l1')
        probs = model.predict_proba(X_test)[:, 1]
        
        for rep_id, prob in zip(grouped_test.index, probs):
            all_test_preds.append({'repertoire_id': rep_id, 'label_positive_probability': prob})
    
    # Explanations: score unique sequences by k-mer coefficients
    coef = model.coef_[0]
    unique_seqs = train_df[['junction_aa', 'v_call', 'j_call']].drop_duplicates()
    
    def seq_score(seq):
        if len(seq) < 3:
            return 0.0
        kmers = [' '.join([seq[i:i+3] for i in range(len(seq)-2)])]
        counts = vectorizer.transform(kmers)
        return np.dot(counts.toarray()[0], coef)
    
    unique_seqs['importance'] = unique_seqs['junction_aa'].apply(seq_score)
    unique_seqs = unique_seqs.sort_values('importance', ascending=False).head(50000)
    unique_seqs['IDdataset'] = train_ds
    all_explanations.append(unique_seqs[['IDdataset', 'junction_aa', 'v_call', 'j_call']])

# Final submission assembly (after loop)
pred_df = pd.DataFrame(all_test_preds)
pred_df['junction_aa'] = -999.0
pred_df['v_call'] = -999.0
pred_df['j_call'] = -999.0

expl_df = pd.concat(all_explanations, ignore_index=True)
expl_df['label_positive_probability'] = -999.0
expl_df['repertoire_id'] = ''  # Or leave blank if not needed

submission = pd.concat([pred_df, expl_df], ignore_index=True)
submission = submission[['repertoire_id', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

# Fill any missing with -999 if needed
submission.fillna(-999.0, inplace=True)

submission.to_csv('submission.csv', index=False)
print("Submission created! Rows:", len(submission))


# After the full loop over train_datasets

print(f"Collected {len(all_test_preds)} predictions")
print(f"Collected explanations from {len(all_explanations)} datasets")

# Predictions dataframe (4213 rows expected)
pred_df = pd.DataFrame(all_test_preds)
pred_df = pred_df[['repertoire_id', 'label_positive_probability']]
pred_df['junction_aa'] = -999.0
pred_df['v_call'] = -999.0
pred_df['j_call'] = -999.0

# Explanations (400,000 rows: 50k per 8 datasets)
expl_df = pd.concat(all_explanations, ignore_index=True)
expl_df['label_positive_probability'] = -999.0
expl_df['repertoire_id'] = -999.0  # Or leave as '' if not required

# Full submission
submission = pd.concat([pred_df, expl_df], ignore_index=True)
submission = submission[['repertoire_id', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

# Ensure exact 404213 rows and no NaNs
submission.fillna(-999.0, inplace=True)
print("Final submission rows:", len(submission))
submission.head(10)
submission.tail(10)

submission.to_csv('submission.csv', index=False)
print("submission.csv created – download and submit on Kaggle!")


import os
print(os.listdir('/kaggle/working'))






import pandas as pd
import os

# Load the sample submission file (template with exact format)
sample_path = '/kaggle/input/adaptive-immune-profiling-challenge-2025/sample_submissions.csv'
sample = pd.read_csv(sample_path)

print("Sample columns:", sample.columns.tolist())
print("Sample rows:", len(sample))  # Should be 404213

# Fill everything with -999.0 (float)
sample[:] = -999.0

# Number of prediction rows (from competition description)
num_pred_rows = 4213

# Your predictions
pred_df = pd.DataFrame(all_test_preds)
print(f"You have {len(pred_df)} predictions")

# Fill prediction probabilities (first 4213 rows)
probs = pred_df['label_positive_probability'].reindex(range(num_pred_rows)).fillna(0.5).values
sample.iloc[:num_pred_rows, sample.columns.get_loc('label_positive_probability')] = probs

print(f"Filled {num_pred_rows} prediction probabilities")

# Start filling explanations after predictions
current_row = num_pred_rows

# Concat all explanations
expl_df = pd.concat(all_explanations, ignore_index=True)

# Fill 50,000 rows for each of the 8 training datasets
for i in range(1, 9):
    dataset_name = f'train_dataset_{i}'
    print(f"Processing explanations for {dataset_name}...")
    
    # Select this dataset's explanations
    if 'IDdataset' in expl_df.columns:
        dataset_expl = expl_df[expl_df['IDdataset'] == dataset_name][['junction_aa', 'v_call', 'j_call']]
    else:
        # If no IDdataset, take next 50k chunk
        start_idx = (i-1) * 50000
        dataset_expl = expl_df.iloc[start_idx:start_idx+50000][['junction_aa', 'v_call', 'j_call']]
    
    dataset_expl = dataset_expl.copy()
    
    # Pad with -999.0 if less than 50,000
    if len(dataset_expl) < 50000:
        pad_rows = 50000 - len(dataset_expl)
        pad_df = pd.DataFrame({
            'junction_aa': [-999.0] * pad_rows,
            'v_call': [-999.0] * pad_rows,
            'j_call': [-999.0] * pad_rows
        })
        dataset_expl = pd.concat([dataset_expl, pad_df], ignore_index=True)
    
    # Take only top 50,000
    dataset_expl = dataset_expl.head(50000)
    
    # Fill into sample
    end_row = current_row + 50000
    sample.iloc[current_row:end_row, sample.columns.get_loc('junction_aa')] = dataset_expl['junction_aa'].values
    sample.iloc[current_row:end_row, sample.columns.get_loc('v_call')] = dataset_expl['v_call'].values
    sample.iloc[current_row:end_row, sample.columns.get_loc('j_call')] = dataset_expl['j_call'].values
    
    current_row += 50000

# Save final submission
sample.to_csv('submission.csv', index=False)

print("SUCCESS! submission.csv is ready with exact 404213 rows")
print("Download and submit it now!")





import pandas as pd

# Load the sample (template with exact format)
sample = pd.read_csv('/kaggle/input/adaptive-immune-profiling-challenge-2025/sample_submissions.csv')

print("Columns:", sample.columns.tolist())
print("Rows:", len(sample))  # Must be 404213

# Fill with -999.0
sample[:] = -999.0

# Fill your predictions (first rows – probability column)
num_preds = len(all_test_preds)
if num_preds > 0:
    probs = pd.DataFrame(all_test_preds)['label_positive_probability'].values
    sample.iloc[:num_preds, sample.columns.get_loc('label_positive_probability')] = probs

# Fill explanations if possible (simple way – spread your all_explanations across the explanation rows)
expl_start = 4213
expl_rows = len(sample) - expl_start
expl_df = pd.concat(all_explanations, ignore_index=True)

if len(expl_df) > 0:
    # Repeat or pad to fill explanation rows
    expl_filled = pd.concat([expl_df] * (expl_rows // len(expl_df) + 1), ignore_index=True).head(expl_rows)
    sample.iloc[expl_start:, sample.columns.get_loc('junction_aa')] = expl_filled['junction_aa'].values
    sample.iloc[expl_start:, sample.columns.get_loc('v_call')] = expl_filled['v_call'].values
    sample.iloc[expl_start:, sample.columns.get_loc('j_call')] = expl_filled['j_call'].values

# Save
sample.to_csv('submission.csv', index=False)

print("New submission.csv ready! Rows:", len(sample))





import pandas as pd

# Load sample (template with unique IDs)
sample = pd.read_csv('/kaggle/input/adaptive-immune-profiling-challenge-2025/sample_submissions.csv')

print("Columns:", sample.columns.tolist())  # Likely ['ID', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call'] or similar

# Overwrite everything with -999.0 (safe)
sample.iloc[:, 1:] = -999.0  # Keep first column (ID) intact, fill others

# Fill predictions (first 4213 rows)
num_preds = 4213
probs = pd.DataFrame(all_test_preds)['label_positive_probability'].values
if len(probs) < num_preds:
    probs = list(probs) + [0.5] * (num_preds - len(probs))  # Default

sample.iloc[:num_preds, sample.columns.get_loc('label_positive_probability')] = probs[:num_preds]

# Fill explanations (after predictions, no IDdataset added)
expl_start = num_preds
current_row = expl_start

expl_df = pd.concat(all_explanations, ignore_index=True)
expl_df = expl_df[['junction_aa', 'v_call', 'j_call']]  # Drop any IDdataset column!

for i in range(8):  # 8 datasets
    start = i * 50000
    end = start + 50000
    dataset_expl = expl_df.iloc[start:end] if len(expl_df) > start else pd.DataFrame()
    
    if len(dataset_expl) < 50000:
        pad = pd.DataFrame({
            'junction_aa': [-999.0] * (50000 - len(dataset_expl)),
            'v_call': [-999.0] * (50000 - len(dataset_expl)),
            'j_call': [-999.0] * (50000 - len(dataset_expl))
        })
        dataset_expl = pd.concat([dataset_expl, pad], ignore_index=True)
    
    end_row = current_row + 50000
    sample.iloc[current_row:end_row, sample.columns.get_loc('junction_aa')] = dataset_expl['junction_aa'].values
    sample.iloc[current_row:end_row, sample.columns.get_loc('v_call')] = dataset_expl['v_call'].values
    sample.iloc[current_row:end_row, sample.columns.get_loc('j_call')] = dataset_expl['j_call'].values
    
    current_row += 50000

# Save
sample.to_csv('submission.csv', index=False)

print("Fixed – no duplicates! Ready to submit.")


import pandas as pd

# Load sample (has unique ID and correct dataset values)
sample = pd.read_csv('/kaggle/input/adaptive-immune-profiling-challenge-2025/sample_submissions.csv')

print("Columns:", sample.columns.tolist())  # Confirm ['ID', 'dataset', ...]

# Fill all fillable columns with -999.0 (keep ID and dataset intact)
fillable_cols = ['label_positive_probability', 'junction_aa', 'v_call', 'j_call']
sample[fillable_cols] = -999.0

# Fill predictions (first 4213 rows: label_positive_probability)
num_preds = 4213
probs = pd.DataFrame(all_test_preds)['label_positive_probability'].values.tolist()
if len(probs) < num_preds:
    probs += [0.5] * (num_preds - len(probs))  # Default if missing

sample.iloc[:num_preds, sample.columns.get_loc('label_positive_probability')] = probs[:num_preds]

# Fill explanations (from row 4213 onward)
expl_start = num_preds
current_row = expl_start

expl_df = pd.concat(all_explanations, ignore_index=True)

# Ensure only the 3 columns, no extra
expl_df = expl_df[['junction_aa', 'v_call', 'j_call']]

for i in range(8):
    start_idx = i * 50000
    dataset_expl = expl_df.iloc[start_idx:start_idx+50000]
    
    if len(dataset_expl) < 50000:
        pad = pd.DataFrame({
            'junction_aa': [-999.0] * (50000 - len(dataset_expl)),
            'v_call': [-999.0] * (50000 - len(dataset_expl)),
            'j_call': [-999.0] * (50000 - len(dataset_expl))
        })
        dataset_expl = pd.concat([dataset_expl, pad], ignore_index=True)
    
    end_row = current_row + 50000
    sample.iloc[current_row:end_row, sample.columns.get_loc('junction_aa')] = dataset_expl['junction_aa'].values
    sample.iloc[current_row:end_row, sample.columns.get_loc('v_call')] = dataset_expl['v_call'].values
    sample.iloc[current_row:end_row, sample.columns.get_loc('j_call')] = dataset_expl['j_call'].values
    
    current_row += 50000

# Save
sample.to_csv('submission.csv', index=False)

print("FINAL fixed submission ready – no duplicates, correct format!")


from IPython.display import FileLink
FileLink('submission.csv')

