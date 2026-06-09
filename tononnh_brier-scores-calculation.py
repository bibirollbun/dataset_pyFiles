from sklearn.metrics import brier_score_loss, mean_squared_error
import numpy as np
import pandas as pd
from typing import Callable


def calculate_brier_score(df_predictions: pd.DataFrame, df_ground_truth: pd.DataFrame, loss_function: Callable):
    """
    Calculates the Brier Score from two Pandas DataFrames, aligning them by ID columns.
    Considers only ground truth values of 0 or 1, using the alternative Brier Score formula.
    Assumes specific column names: 'ID' for IDs, 'Pred' for predictions in df_predictions,
    and 'Pred' for ground truth in df_ground_truth. Handles potential column name conflicts
    after merging.

    Args:
        df_predictions (pd.DataFrame): DataFrame with predictions ('ID' and 'Pred' columns).
        df_ground_truth (pd.DataFrame): DataFrame with ground truth ('ID' and 'Pred' columns).

    Returns:
        float: The Brier Score, or None if an error occurs or no valid/matching data is found.
    """
    ID_COL = "ID"
    PRED_COL = "Pred"
    
    try:
        if not isinstance(df_predictions, pd.DataFrame) or not isinstance(df_ground_truth, pd.DataFrame):
            raise ValueError("Inputs must be Pandas DataFrames.")

        for col in [ID_COL, PRED_COL]:
            if col not in df_predictions.columns:
                raise ValueError(f"Column '{col}' not found in prediction DataFrame.")
        for col in [ID_COL, PRED_COL]:
            if col not in df_ground_truth.columns:
                raise ValueError(f"Column '{col}' not found in ground truth DataFrame.")

        merged_df = pd.merge(df_predictions,
                             df_ground_truth,
                             left_on=ID_COL,
                             right_on=ID_COL,
                             how='inner',
                             suffixes=('_pred', '_gt'))

        if merged_df.empty:
            print("Warning: No matching IDs found between DataFrames. Returning None.")
            return None

        predictions = merged_df[f'{PRED_COL}_pred'].to_numpy()
        ground_truth = merged_df[f'{PRED_COL}_gt'].to_numpy()


        if not np.all((ground_truth == 0) | (ground_truth == 1) | (ground_truth == 0.5)):
            raise ValueError("Ground truth values must be 0, 0.5, or 1.")
        if not np.all((predictions >= 0) & (predictions <= 1)):
            raise ValueError("Prediction probabilities must be between 0 and 1.")

        mask = (ground_truth == 0) | (ground_truth == 1)
        masked_ground_truth = ground_truth[mask]
        masked_predictions = predictions[mask]

        if masked_ground_truth.size == 0:
            print("Warning: No valid ground truth values (0 or 1) found after alignment and masking. Returning None.")
            return None

        brier_score = loss_function(masked_ground_truth, masked_predictions)
        return brier_score

    except ValueError as e:
        print(f"Error: {e}")
        return None



sample_submission = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage1.csv')
ground_truth = pd.read_csv('/kaggle/input/mmlm2025-ground-truth/ground_truth.csv')

test_1 = pd.read_csv('/kaggle/input/mmlm2025-linear-regression/submission.csv')
# from https://www.kaggle.com/code/kerta27/mmlm2025-randomforest-optuna
test_2 = pd.read_csv('/kaggle/input/mixture-of-experts-moe-2-gcn-model/submission.csv')
# from https://www.kaggle.com/code/kumarandatascientist/mixture-of-experts-moe-2-gcn-model


mean_squared_error(ground_truth['Pred'], sample_submission['Pred'])


mean_squared_error(ground_truth['Pred'], test_1['Pred'])
# 0.00758


mean_squared_error(ground_truth['Pred'], test_2['Pred'])
# 0.14889


print(calculate_brier_score(sample_submission,ground_truth, brier_score_loss))
print(calculate_brier_score(sample_submission,ground_truth, mean_squared_error))


print(calculate_brier_score(test_1,ground_truth, brier_score_loss))
# 0.00758


print(calculate_brier_score(test_2,ground_truth, brier_score_loss))
# 0.14889




