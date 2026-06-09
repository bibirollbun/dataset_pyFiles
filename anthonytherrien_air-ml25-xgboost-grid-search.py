import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score


# Define function to parse a dataset folder of TSV files
def parse_tsv_files(folder_path, feature_columns=('v_call', 'j_call')):
    folder_name = os.path.basename(folder_path)
    files = os.listdir(folder_path)
    tsv_files = [f for f in files if f.endswith('.tsv')]
    other_files = [f.name for f in os.scandir(folder_path) if not f.name.endswith('.tsv')]

    print(f'Loading {len(tsv_files)} .tsv files from {folder_name} (remaining: {other_files}).')

    metadata = None
    if "metadata.csv" in files:
        metadata = pd.read_csv(os.path.join(folder_path, "metadata.csv"))
        metadata.set_index("filename", inplace=True)

    dataset_rows = []

    for tsv_file in tqdm(tsv_files, desc=f"Loading {folder_name}"):
        file_path = os.path.join(folder_path, tsv_file)
        file_name, _ = os.path.splitext(tsv_file)

        try:
            df = pd.read_csv(file_path, sep="\t")
        except Exception as e:
            print(f"Error loading {tsv_file}: {e}")
            continue

        row = {
            "ID": file_name,
            "dataset": folder_name
        }

        if metadata is not None:
            row["label_positive"] = int(metadata.at[tsv_file, "label_positive"])

        for col in feature_columns:
            counts = df[col].value_counts() / len(df)
            row.update(counts.to_dict())

        dataset_rows.append(row)

    return dataset_rows


# Define function to load all train and test folders
def load_all_datasets(train_folders, test_folders, train_base_path, test_base_path):
    train_rows = []
    test_rows = []

    for folder in tqdm(train_folders, desc="Train folders"):
        path = os.path.join(train_base_path, folder)
        train_rows += parse_tsv_files(path)

    for folder in tqdm(test_folders, desc="Test folders"):
        path = os.path.join(test_base_path, folder)
        test_rows += parse_tsv_files(path)

    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


# Define function to run XGB Grid Search
def run_grid_search_xgb(X_train, y_train, base_params, param_grid, n_splits):
    cv_splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    model = xgb.XGBClassifier(
        **base_params,
        use_label_encoder=False
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_log_loss',
        cv=cv_splitter,
        n_jobs=-1,
        verbose=2,
        refit=True
    )

    print("\nRunning Grid Search...")
    grid_search.fit(X_train, y_train)

    print("Best Params:", grid_search.best_params_)
    print("Best Neg LogLoss:", grid_search.best_score_)

    return grid_search.best_estimator_


# Define function to train the best model on full data
def train_final_model(model, X_train, y_train):
    print("\nTraining final model...")
    model.fit(X_train, y_train)
    return model


# Define function to generate predictions
def generate_predictions(model, X_test, dataset_test):
    print("\nGenerating predictions...")
    probabilities = model.predict_proba(X_test)[:, 1]

    predictions_df = pd.DataFrame({
        "ID": dataset_test["ID"],
        "dataset": dataset_test["dataset"],
        "label_positive_probability": probabilities
    })

    return predictions_df


# Define function to create submission.csv
def prepare_submission(predictions_df, sample_path, output_path):
    sample_df = pd.read_csv(sample_path)
    sample_df = sample_df.drop(columns=['label_positive_probability'])

    merged = pd.merge(
        sample_df,
        predictions_df,
        on=['ID', 'dataset'],
        how='left'
    )

    merged = merged.fillna(0.5)
    merged = merged.drop_duplicates(subset=['ID', 'dataset'], keep='first')

    merged.to_csv(output_path, index=False)
    print(f"Saved submission to: {output_path}")


# Define main workflow
def main():
    PATH_DATASET = "/kaggle/input/adaptive-immune-profiling-challenge-2025"
    PATH_TRAIN_DATASETS = os.path.join(PATH_DATASET, 'train_datasets', 'train_datasets')
    PATH_TEST_DATASETS = os.path.join(PATH_DATASET, 'test_datasets', 'test_datasets')

    train_folders = sorted(os.listdir(PATH_TRAIN_DATASETS))
    test_folders = sorted(os.listdir(PATH_TEST_DATASETS))

    print("Loading Train/Test datasets...")
    dataset_train, dataset_test = load_all_datasets(
        train_folders,
        test_folders,
        PATH_TRAIN_DATASETS,
        PATH_TEST_DATASETS
    )

    X_train = dataset_train.drop(['label_positive', 'ID', 'dataset'], axis=1).fillna(0)
    y_train = dataset_train['label_positive']
    X_test = dataset_test.drop(['ID', 'dataset'], axis=1).fillna(0)

    excluded_features = ['TCRBV6-01', 'TCRBVA']
    X_train = X_train.drop(excluded_features, axis=1, errors='ignore')

    for col in set(X_train.columns) - set(X_test.columns):
        X_test[col] = 0

    X_test = X_test[X_train.columns]

    base_params = {
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        'random_state': 42,
        'importance_type': 'gain'
    }

    param_grid = {
        'colsample_bytree': [0.58, 0.63842335244, 0.70],
        'learning_rate': [0.02, 0.03256233, 0.05],
        'max_depth': [18, 20, 22],
        'reg_alpha': [0.05, 0.087314708, 0.15],
        'reg_lambda': [0.003, 0.0072173962, 0.02],
        'subsample': [0.7, 0.8, 0.9]
    }

    best_model = run_grid_search_xgb(
        X_train,
        y_train,
        base_params,
        param_grid,
        n_splits=8
    )

    final_model = train_final_model(
        best_model,
        X_train,
        y_train
    )

    predictions = generate_predictions(
        final_model,
        X_test,
        dataset_test
    )

    sample_path = os.path.join(PATH_DATASET, "sample_submissions.csv")
    prepare_submission(
        predictions,
        sample_path,
        "submission.csv"
    )


# Run main workflow
if __name__ == "__main__":
    main()

