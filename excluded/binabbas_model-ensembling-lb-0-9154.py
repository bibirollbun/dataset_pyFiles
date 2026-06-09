import pandas as pd
import numpy as np
import os

# Get the current directory
preds_path = r"c:/Users/abdal/Desktop/data/nlp/preds5"
all_files = os.listdir(preds_path)
csv_files = [f for f in all_files if f.endswith('.csv') and f != 'submit.csv']

prob_cols = [f'class_{i}_prob' for i in range(8)]

# Filter only files that have all prob_cols
valid_csv_files = []
for f in csv_files:
    file_path = os.path.join(preds_path, f)
    try:
        df = pd.read_csv(file_path, nrows=1)
        if all(col in df.columns for col in prob_cols):
            valid_csv_files.append(f)
        else:
            print(f"Skipping {f}: missing probability columns.")
    except Exception as e:
        print(f"Skipping {f}: {e}")

if not valid_csv_files:
    print("No valid CSV files with probability columns found to ensemble.")
else:
    # Read the first valid CSV to initialize the ensemble DataFrame
    first_file_path = os.path.join(preds_path, valid_csv_files[0])
    ensemble_df = pd.read_csv(first_file_path)
    
    # Initialize sum of probabilities with the first file
    sum_probs = ensemble_df[prob_cols].values
    
    # Iterate over the rest of the valid CSV files and add their probabilities
    for file_name in valid_csv_files[1:]:
        file_path = os.path.join(preds_path, file_name)
        current_df = pd.read_csv(file_path)
        if len(current_df) == len(ensemble_df) and (current_df['id'] == ensemble_df['id']).all():
            sum_probs += current_df[prob_cols].values
        else:
            print(f"Warning: File {file_name} has different IDs or length. Skipping or implement merging.")

    # Calculate average probabilities
    avg_probs = sum_probs / len(valid_csv_files)
    
    # Get the argmax for each row (ID)
    predicted_labels = np.argmax(avg_probs, axis=1)
    
    # Create submission DataFrame
    submit_df = pd.DataFrame({
        'id': ensemble_df['id'],
        'label': predicted_labels
    })
    
    # Save to submit.csv
    submit_file_path = os.path.join(preds_path, 'submit.csv')
    submit_df.to_csv(submit_file_path, index=False)
    
    print(f"Ensembled predictions saved to {submit_file_path}")
    print(f"Files ensembled: {valid_csv_files}")

    # --- Confident Pseudo Labeling ---
    # Find confident pseudo-labels in test set
    confident_mask = (avg_probs.max(axis=1) >= 0.85)
    confident_ids = ensemble_df.loc[confident_mask, 'id']
    confident_labels = predicted_labels[confident_mask]
    confident_probs = avg_probs[confident_mask]

    # Load test questions
    test_path = os.path.join(preds_path, 'test.csv')
    test_df = pd.read_csv(test_path)

    # Merge confident pseudo-labels with test questions
    confident_test = test_df[test_df['id'].isin(confident_ids)].copy()
    confident_test['label'] = confident_labels

    # Load train data
    train_path = r"c:/Users/abdal/Desktop/data/nlp/data/train.csv"
    train_df = pd.read_csv(train_path)

    # Ensure train_df has columns: Question, label
    # Combine train and confident pseudo-labeled test (drop id column from test)
    pseudo_train = pd.concat([train_df, confident_test[['Question', 'label']]], ignore_index=True)

    # Save the new pseudo-labeled train set (without id column)
    pseudo_train_path = os.path.join(preds_path, 'pseudo_train.csv')
    pseudo_train.to_csv(pseudo_train_path, index=False)
    print(f"Pseudo-labeled train set saved to {pseudo_train_path}. Total samples: {len(pseudo_train)}")




