## imports used by the basic code template provided.

import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import glob
import sys
import argparse
from collections import defaultdict
from typing import Iterator, Tuple, Union, List

## imports that are additionally used by this notebook

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline



## some utility functions such as data loaders, etc.

def load_data_generator(data_dir: str, metadata_filename='metadata.csv') -> Iterator[
    Union[Tuple[str, pd.DataFrame, bool], Tuple[str, pd.DataFrame]]]:
    """
    A generator to load immune repertoire data.

    This function operates in two modes:
    1.  If metadata is found, it yields data based on the metadata file.
    2.  If metadata is NOT found, it uses glob to find and yield all '.tsv'
        files in the directory.

    Args:
        data_dir (str): The path to the directory containing the data.

    Yields:
        An iterator of tuples. The format depends on the mode:
        - With metadata: (repertoire_id, pd.DataFrame, label_positive)
        - Without metadata: (filename, pd.DataFrame)
    """
    metadata_path = os.path.join(data_dir, metadata_filename)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        for row in metadata_df.itertuples(index=False):
            file_path = os.path.join(data_dir, row.filename)
            try:
                repertoire_df = pd.read_csv(file_path, sep='\t')
                yield row.repertoire_id, repertoire_df, row.label_positive
            except FileNotFoundError:
                print(f"Warning: File '{row.filename}' listed in metadata not found. Skipping.")
                continue
    else:
        search_pattern = os.path.join(data_dir, '*.tsv')
        tsv_files = glob.glob(search_pattern)
        for file_path in sorted(tsv_files):
            try:
                filename = os.path.basename(file_path)
                repertoire_df = pd.read_csv(file_path, sep='\t')
                yield filename, repertoire_df
            except Exception as e:
                print(f"Warning: Could not read file '{file_path}'. Error: {e}. Skipping.")
                continue


def load_full_dataset(data_dir: str) -> pd.DataFrame:
    """
    Loads all TSV files from a directory and concatenates them into a single DataFrame.

    This function handles two scenarios:
    1. If metadata.csv exists, it loads data based on the metadata and adds
       'repertoire_id' and 'label_positive' columns.
    2. If metadata.csv does not exist, it loads all .tsv files and adds
       a 'filename' column as an identifier.

    Args:
        data_dir (str): The path to the data directory.

    Returns:
        pd.DataFrame: A single, concatenated DataFrame containing all the data.
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    df_list = []
    data_loader = load_data_generator(data_dir=data_dir)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        total_files = len(metadata_df)
        for rep_id, data_df, label in tqdm(data_loader, total=total_files, desc="Loading files"):
            data_df['ID'] = rep_id
            data_df['label_positive'] = label
            df_list.append(data_df)
    else:
        search_pattern = os.path.join(data_dir, '*.tsv')
        total_files = len(glob.glob(search_pattern))
        for filename, data_df in tqdm(data_loader, total=total_files, desc="Loading files"):
            data_df['ID'] = os.path.basename(filename).replace(".tsv", "")
            df_list.append(data_df)

    if not df_list:
        print("Warning: No data files were loaded.")
        return pd.DataFrame()

    full_dataset_df = pd.concat(df_list, ignore_index=True)
    return full_dataset_df


def load_and_encode_kmers(data_dir: str, k: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loading and k-mer encoding of repertoire data.

    Args:
        data_dir: Path to data directory
        k: K-mer length

    Returns:
        Tuple of (encoded_features_df, metadata_df)
        metadata_df always contains 'ID', and 'label_positive' if available
    """
    from collections import Counter

    metadata_path = os.path.join(data_dir, 'metadata.csv')
    data_loader = load_data_generator(data_dir=data_dir)

    repertoire_features = []
    metadata_records = []

    search_pattern = os.path.join(data_dir, '*.tsv')
    total_files = len(glob.glob(search_pattern))

    for item in tqdm(data_loader, total=total_files, desc=f"Encoding {k}-mers"):
        if os.path.exists(metadata_path):
            rep_id, data_df, label = item
        else:
            filename, data_df = item
            rep_id = os.path.basename(filename).replace(".tsv", "")
            label = None

        kmer_counts = Counter()
        for seq in data_df['junction_aa'].dropna():
            for i in range(len(seq) - k + 1):
                kmer_counts[seq[i:i + k]] += 1

        repertoire_features.append({
            'ID': rep_id,
            **kmer_counts
        })

        metadata_record = {'ID': rep_id}
        if label is not None:
            metadata_record['label_positive'] = label
        metadata_records.append(metadata_record)

        del data_df, kmer_counts

    features_df = pd.DataFrame(repertoire_features).fillna(0).set_index('ID')
    features_df.fillna(0)
    metadata_df = pd.DataFrame(metadata_records)

    return features_df, metadata_df


def save_tsv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep='\t', index=False)


def get_repertoire_ids(data_dir: str) -> list:
    """
    Retrieves repertoire IDs from the metadata file or filenames in the directory.

    Args:
        data_dir (str): The path to the data directory.

    Returns:
        list: A list of repertoire IDs.
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        repertoire_ids = metadata_df['repertoire_id'].tolist()
    else:
        search_pattern = os.path.join(data_dir, '*.tsv')
        tsv_files = glob.glob(search_pattern)
        repertoire_ids = [os.path.basename(f).replace('.tsv', '') for f in sorted(tsv_files)]

    return repertoire_ids


def generate_random_top_sequences_df(n_seq: int = 50000) -> pd.DataFrame:
    """
    Generates a random DataFrame simulating top important sequences.

    Args:
        n_seq (int): Number of sequences to generate.

    Returns:
        pd.DataFrame: A DataFrame with columns 'ID', 'dataset', 'junction_aa', 'v_call', 'j_call'.
    """
    seqs = set()
    while len(seqs) < n_seq:
        seq = ''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), size=15))
        seqs.add(seq)
    data = {
        'junction_aa': list(seqs),
        'v_call': ['TRBV20-1'] * n_seq,
        'j_call': ['TRBJ2-7'] * n_seq,
        'importance_score': np.random.rand(n_seq)
    }
    return pd.DataFrame(data)


def validate_dirs_and_files(train_dir: str, test_dirs: List[str], out_dir: str) -> None:
    assert os.path.isdir(train_dir), f"Train directory `{train_dir}` does not exist."
    train_tsvs = glob.glob(os.path.join(train_dir, "*.tsv"))
    assert train_tsvs, f"No .tsv files found in train directory `{train_dir}`."
    metadata_path = os.path.join(train_dir, "metadata.csv")
    assert os.path.isfile(metadata_path), f"`metadata.csv` not found in train directory `{train_dir}`."

    for test_dir in test_dirs:
        assert os.path.isdir(test_dir), f"Test directory `{test_dir}` does not exist."
        test_tsvs = glob.glob(os.path.join(test_dir, "*.tsv"))
        assert test_tsvs, f"No .tsv files found in test directory `{test_dir}`."

    try:
        os.makedirs(out_dir, exist_ok=True)
        test_file = os.path.join(out_dir, "test_write_permission.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        print(f"Failed to create or write to output directory `{out_dir}`: {e}")
        sys.exit(1)


def concatenate_output_files(out_dir: str) -> None:
    """
    Concatenates all test predictions and important sequences TSV files from the output directory.

    This function finds all files matching the patterns:
    - *_test_predictions.tsv
    - *_important_sequences.tsv

    and concatenates them to match the expected output format of submissions.csv.

    Args:
        out_dir (str): Path to the output directory containing the TSV files.

    Returns:
        pd.DataFrame: Concatenated DataFrame with predictions followed by important sequences.
                     Columns: ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']
    """
    predictions_pattern = os.path.join(out_dir, '*_test_predictions.tsv')
    sequences_pattern = os.path.join(out_dir, '*_important_sequences.tsv')

    predictions_files = sorted(glob.glob(predictions_pattern))
    sequences_files = sorted(glob.glob(sequences_pattern))

    df_list = []

    for pred_file in predictions_files:
        try:
            df = pd.read_csv(pred_file, sep='\t')
            df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not read predictions file '{pred_file}'. Error: {e}. Skipping.")
            continue

    for seq_file in sequences_files:
        try:
            df = pd.read_csv(seq_file, sep='\t')
            df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not read sequences file '{seq_file}'. Error: {e}. Skipping.")
            continue

    if not df_list:
        print("Warning: No output files were found to concatenate.")
        concatenated_df = pd.DataFrame(
            columns=['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call'])
    else:
        concatenated_df = pd.concat(df_list, ignore_index=True)
    submissions_file = os.path.join(out_dir, 'submissions.csv')
    concatenated_df.to_csv(submissions_file, index=False)
    print(f"Concatenated output written to `{submissions_file}`.")


def get_dataset_pairs(train_dir: str, test_dir: str) -> List[Tuple[str, List[str]]]:
    """Returns list of (train_path, [test_paths]) tuples for dataset pairs."""
    test_groups = defaultdict(list)
    for test_name in sorted(os.listdir(test_dir)):
        if test_name.startswith("test_dataset_"):
            base_id = test_name.replace("test_dataset_", "").split("_")[0]
            test_groups[base_id].append(os.path.join(test_dir, test_name))

    pairs = []
    for train_name in sorted(os.listdir(train_dir)):
        if train_name.startswith("train_dataset_"):
            train_id = train_name.replace("train_dataset_", "")
            train_path = os.path.join(train_dir, train_name)
            pairs.append((train_path, test_groups.get(train_id, [])))

    return pairs



## A Classifier class that implements functionality of a baseline prediction + identification of sequences that explain the labels
## This implementation will be used in the provided `ImmuneStatePredictor` class provided through the code template, as shown in the next chunk.


class KmerClassifier:
    """L1-regularized logistic regression for k-mer count data."""

    def __init__(self, c_values=None, cv_folds=5,
                 opt_metric='balanced_accuracy', random_state=123, n_jobs=1):
        if c_values is None:
            c_values = [1, 0.1, 0.05, 0.03]
        self.c_values = c_values
        self.cv_folds = cv_folds
        self.opt_metric = opt_metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_C_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.model_ = None
        self.feature_names_ = None
        self.val_score_ = None

    def _make_pipeline(self, C):
        """Create standardization + L1 logistic regression pipeline."""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                penalty='l1', C=C, solver='liblinear',
                random_state=self.random_state, max_iter=1000
            ))
        ])

    def _get_scorer(self):
        """Get scoring function for optimization."""
        if self.opt_metric == 'balanced_accuracy':
            return 'balanced_accuracy'
        elif self.opt_metric == 'roc_auc':
            return 'roc_auc'
        else:
            raise ValueError(f"Unknown metric: {self.opt_metric}")

    def tune_and_fit(self, X, y, val_size=0.2):
        """Perform CV tuning on train split and fit, with optional validation split."""

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        if val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_size, random_state=self.random_state, stratify=y)
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=self.random_state)
        scorer = self._get_scorer()

        results = []
        for C in self.c_values:
            pipeline = self._make_pipeline(C)
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scorer,
                                     n_jobs=self.n_jobs)
            results.append({
                'C': C,
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            })

        self.cv_results_ = pd.DataFrame(results)
        best_idx = self.cv_results_['mean_score'].idxmax()
        self.best_C_ = self.cv_results_.loc[best_idx, 'C']
        self.best_score_ = self.cv_results_.loc[best_idx, 'mean_score']

        print(f"Best C: {self.best_C_} (CV {self.opt_metric}: {self.best_score_:.4f})")

        # Fit on training split with best hyperparameter
        self.model_ = self._make_pipeline(self.best_C_)
        self.model_.fit(X_train, y_train)

        if X_val is not None:
            if scorer == 'balanced_accuracy':
                self.val_score_ = balanced_accuracy_score(y_val, self.model_.predict(X_val))
            else:  # roc_auc
                self.val_score_ = roc_auc_score(y_val, self.model_.predict_proba(X_val)[:, 1])
            print(f"Validation {self.opt_metric}: {self.val_score_:.4f}")

        return self

    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model_.predict_proba(X)[:, 1]

    def predict(self, X):
        """Predict class labels."""
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model_.predict(X)

    def get_feature_importance(self):
        """
        Get feature importance from L1 coefficients.

        Returns:
            pd.DataFrame with columns ['feature', 'coefficient', 'abs_coefficient']
        """
        if self.model_ is None:
            raise ValueError("Model not fitted.")

        coef = self.model_.named_steps['classifier'].coef_[0]

        if self.feature_names_ is not None:
            feature_names = self.feature_names_
        else:
            feature_names = [f"feature_{i}" for i in range(len(coef))]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coef,
            'abs_coefficient': np.abs(coef)
        })

        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)

        return importance_df

    def score_all_sequences(self, sequences_df, sequence_col='junction_aa'):
        """
        Score all sequences using model coefficients.

        Parameters:
            sequences_df: DataFrame with unique sequences
            sequence_col: Column name containing sequences

        Returns:
            DataFrame with added 'importance_score' column
        """
        if self.model_ is None:
            raise ValueError("Model not fitted.")

        scaler = self.model_.named_steps['scaler']
        coefficients = self.model_.named_steps['classifier'].coef_[0]
        coefficients = coefficients / scaler.scale_

        kmer_to_index = {kmer: idx for idx, kmer in enumerate(self.feature_names_)}
        k = len(self.feature_names_[0])

        scores = []
        total_seqs = len(sequences_df)
        for seq in tqdm(sequences_df[sequence_col], total=total_seqs, desc="Scoring sequences"):
            counts = np.zeros(len(kmer_to_index), dtype=np.uint8)
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i + k]
                if kmer in kmer_to_index:
                    counts[kmer_to_index[kmer]] = 1
            scores.append(np.dot(counts, coefficients))

        result_df = sequences_df.copy()
        result_df['importance_score'] = scores
        return result_df


def prepare_data(X_df, labels_df, id_col='ID', label_col='label_positive'):
    """
    Merge feature matrix with labels, ensuring alignment.

    Parameters:
        X_df: DataFrame with samples as rows (index contains IDs)
        labels_df: DataFrame with ID column and label column
        id_col: Name of ID column in labels_df
        label_col: Name of label column in labels_df

    Returns:
        X: Feature matrix aligned with labels
        y: Binary labels
        common_ids: IDs that were kept
    """
    if id_col in labels_df.columns:
        labels_indexed = labels_df.set_index(id_col)[label_col]
    else:
        # Assume labels_df index is already the ID
        labels_indexed = labels_df[label_col]

    common_ids = X_df.index.intersection(labels_indexed.index)

    if len(common_ids) == 0:
        raise ValueError("No common IDs found between feature matrix and labels")

    X = X_df.loc[common_ids]
    y = labels_indexed.loc[common_ids]

    print(f"Aligned {len(common_ids)} samples with labels")

    return X, y, common_ids


## Main ImmuneStatePredictor class, where this notebook fills in a baseline predictor implementation within the placeholders 
## and replaces any example code lines with actual code that makes sense

class ImmuneStatePredictor:
    """
    A template for predicting immune states from TCR repertoire data.

    Participants should implement the logic for training, prediction, and
    sequence identification within this class.
    """

    def __init__(self, n_jobs: int = 1, device: str = 'cpu', **kwargs):
        """
        Initializes the predictor.

        Args:
            n_jobs (int): Number of CPU cores to use for parallel processing.
            device (str): The device to use for computation (e.g., 'cpu', 'cuda').
            **kwargs: Additional hyperparameters for the model.
        """
        self.train_ids_ = None
        total_cores = os.cpu_count()
        if n_jobs == -1:
            self.n_jobs = total_cores
        else:
            self.n_jobs = min(n_jobs, total_cores)
        self.device = device
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: 'cuda' was requested but is not available. Falling back to 'cpu'.")
            self.device = 'cpu'
        else:
            self.device = device
        # --- your code starts here ---
        # Example: Store hyperparameters, the actual model, identified important sequences, etc.

        # NOTE: we encourage you to use self.n_jobs and self.device if appropriate in
        # your implementation instead of hardcoding these values because your code may later be run in an
        # environment with different hardware resources.

        self.model = None
        self.important_sequences_ = None
        # --- your code ends here ---

    def fit(self, train_dir_path: str):
        """
        Trains the model on the provided training data.

        Args:
            train_dir_path (str): Path to the directory with training TSV files.

        Returns:
            self: The fitted predictor instance.
        """

        # --- your code starts here ---
        # Load the data, prepare suited representations as needed, train your model,
        # and find the top k important sequences that best explain the labels.
        # Example: Load the data. One possibility could be to use the provided utility function as shown below.

        # full_train_dataset_df = load_full_dataset(train_dir_path)

        X_train_df, y_train_df = load_and_encode_kmers(train_dir_path, k=3)  # Example of loading and encoding kmers

        #   Model Training
        #    Example: self.model = SomeClassifier().fit(X_train, y_train)

        X_train, y_train, train_ids = prepare_data(X_train_df, y_train_df,
                                                   id_col='ID', label_col='label_positive')

        self.model = KmerClassifier(
            c_values=[1, 0.2, 0.1, 0.05, 0.03],
            cv_folds=5,
            opt_metric='roc_auc',
            random_state=123,
            n_jobs=self.n_jobs
        )

        self.model.tune_and_fit(X_train, y_train)

        self.train_ids_ = train_ids

        #   Identify important sequences (can be done here or in the dedicated method)
        #    Example:
        self.important_sequences_ = self.identify_associated_sequences(train_dir_path=train_dir_path, top_k=50000)

        # --- your code ends here ---
        print("Training complete.")
        return self

    def predict_proba(self, test_dir_path: str) -> pd.DataFrame:
        """
        Predicts probabilities for examples in the provided path.

        Args:
            test_dir_path (str): Path to the directory with test TSV files.

        Returns:
            pd.DataFrame: A DataFrame with 'ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call' columns.
        """
        print(f"Making predictions for data in {test_dir_path}...")
        if self.model is None:
            raise RuntimeError("The model has not been fitted yet. Please call `fit` first.")

        # --- your code starts here ---

        # Example: Load the data. One possibility could be to use the provided utility function as shown below.

        # full_test_dataset_df = load_full_dataset(test_dir_path)

        X_test_df, _ = load_and_encode_kmers(test_dir_path, k=3)

        if self.model.feature_names_ is not None:
            X_test_df = X_test_df.reindex(columns=self.model.feature_names_, fill_value=0)

        repertoire_ids = X_test_df.index.tolist()

        # Prediction
        #    Example:
        # draw random probabilities for demonstration purposes

        probabilities = self.model.predict_proba(X_test_df)

        # --- your code ends here ---

        predictions_df = pd.DataFrame({
            'ID': repertoire_ids,
            'dataset': [os.path.basename(test_dir_path)] * len(repertoire_ids),
            'label_positive_probability': probabilities
        })

        # to enable compatibility with the expected output format that includes junction_aa, v_call, j_call columns
        predictions_df['junction_aa'] = -999.0
        predictions_df['v_call'] = -999.0
        predictions_df['j_call'] = -999.0

        predictions_df = predictions_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

        print(f"Prediction complete on {len(repertoire_ids)} examples in {test_dir_path}.")
        return predictions_df

    def identify_associated_sequences(self, train_dir_path: str, top_k: int = 50000) -> pd.DataFrame:
        """
        Identifies the top "k" important sequences (rows) from the training data that best explain the labels.

        Args:
            top_k (int): The number of top sequences to return (based on some scoring mechanism).
            train_dir_path (str): Path to the directory with training TSV files.

        Returns:
            pd.DataFrame: A DataFrame with 'ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call' columns.
        """
        dataset_name = os.path.basename(train_dir_path)

        # --- your code starts here ---
        # Return the top k sequences, sorted based on some form of importance score.
        # Example:
        # all_sequences_scored = self._score_all_sequences()

        full_df = load_full_dataset(train_dir_path)
        unique_seqs = full_df[['junction_aa', 'v_call', 'j_call']].drop_duplicates()
        all_sequences_scored = self.model.score_all_sequences(unique_seqs, sequence_col='junction_aa')

        # --- your code ends here ---

        top_sequences_df = all_sequences_scored.nlargest(top_k, 'importance_score')
        top_sequences_df = top_sequences_df[['junction_aa', 'v_call', 'j_call']]
        top_sequences_df['dataset'] = dataset_name
        top_sequences_df['ID'] = range(1, len(top_sequences_df)+1)
        top_sequences_df['ID'] = top_sequences_df['dataset'] + '_seq_top_' + top_sequences_df['ID'].astype(str)
        top_sequences_df['label_positive_probability'] = -999.0 # to enable compatibility with the expected output format
        top_sequences_df = top_sequences_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

        return top_sequences_df



## The `main` workflow that uses your implementation of the ImmuneStatePredictor class to train, identify important sequences and predict test labels


def _train_predictor(predictor: ImmuneStatePredictor, train_dir: str):
    """Trains the predictor on the training data."""
    print(f"Fitting model on examples in ` {train_dir} `...")
    predictor.fit(train_dir)


def _generate_predictions(predictor: ImmuneStatePredictor, test_dirs: List[str]) -> pd.DataFrame:
    """Generates predictions for all test directories and concatenates them."""
    all_preds = []
    for test_dir in test_dirs:
        print(f"Predicting on examples in ` {test_dir} `...")
        preds = predictor.predict_proba(test_dir)
        if preds is not None and not preds.empty:
            all_preds.append(preds)
        else:
            print(f"Warning: No predictions returned for {test_dir}")
    if all_preds:
        return pd.concat(all_preds, ignore_index=True)
    return pd.DataFrame()


def _save_predictions(predictions: pd.DataFrame, out_dir: str, train_dir: str) -> None:
    """Saves predictions to a TSV file."""
    if predictions.empty:
        raise ValueError("No predictions to save - predictions DataFrame is empty")

    preds_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_test_predictions.tsv")
    save_tsv(predictions, preds_path)
    print(f"Predictions written to `{preds_path}`.")


def _save_important_sequences(predictor: ImmuneStatePredictor, out_dir: str, train_dir: str) -> None:
    """Saves important sequences to a TSV file."""
    seqs = predictor.important_sequences_
    if seqs is None or seqs.empty:
        raise ValueError("No important sequences available to save")

    seqs_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_important_sequences.tsv")
    save_tsv(seqs, seqs_path)
    print(f"Important sequences written to `{seqs_path}`.")


def main(train_dir: str, test_dirs: List[str], out_dir: str, n_jobs: int, device: str) -> None:
    validate_dirs_and_files(train_dir, test_dirs, out_dir)
    predictor = ImmuneStatePredictor(n_jobs=n_jobs,
                                     device=device)  # instantiate with any other parameters as defined by you in the class
    _train_predictor(predictor, train_dir)
    predictions = _generate_predictions(predictor, test_dirs)
    _save_predictions(predictions, out_dir, train_dir)
    _save_important_sequences(predictor, out_dir, train_dir)


def run():
    parser = argparse.ArgumentParser(description="Immune State Predictor CLI")
    parser.add_argument("--train_dir", required=True, help="Path to training data directory")
    parser.add_argument("--test_dirs", required=True, nargs="+", help="Path(s) to test data director(ies)")
    parser.add_argument("--out_dir", required=True, help="Path to output directory")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of CPU cores to use. Use -1 for all available cores.")
    parser.add_argument("--device", type=str, default='cpu', choices=['cpu', 'cuda'],
                        help="Device to use for computation ('cpu' or 'cuda').")
    args = parser.parse_args()
    main(args.train_dir, args.test_dirs, args.out_dir, args.n_jobs, args.device)


train_datasets_dir = "/kaggle/input/adaptive-immune-profiling-challenge-2025/train_datasets/train_datasets"
test_datasets_dir = "/kaggle/input/adaptive-immune-profiling-challenge-2025/test_datasets/test_datasets"
results_dir = "/kaggle/working/results"

train_test_dataset_pairs = get_dataset_pairs(train_datasets_dir, test_datasets_dir)

for train_dir, test_dirs in train_test_dataset_pairs:
    main(train_dir=train_dir, test_dirs=test_dirs, out_dir=results_dir, n_jobs=4, device="cpu")

concatenate_output_files(results_dir)

