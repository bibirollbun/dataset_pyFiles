# IMPORTS
import numpy as np
import pandas as pd
import pandas.api.types
import sklearn.metrics
from typing import Union
import ast
import optuna


# METRICS
class ParticipantVisibleError(Exception):
    pass


class HostVisibleError(Exception):
    pass


def treat_as_participant_error(error_message: str, solution: Union[pd.DataFrame, np.ndarray]) -> bool:
    ''' Many metrics can raise more errors than can be handled manually. This function attempts
    to identify errors that can be treated as ParticipantVisibleError without leaking any competition data.

    If the solution is purely numeric, and there are no numbers in the error message,
    then the error message is sufficiently unlikely to leak usable data and can be shown to participants.

    We expect this filter to reject many safe messages. It's intended only to reduce the number of errors we need to manage manually.
    '''
    # This check treats bools as numeric
    if isinstance(solution, pd.DataFrame):
        solution_is_all_numeric = all([pandas.api.types.is_numeric_dtype(x) for x in solution.dtypes.values])
        solution_has_bools = any([pandas.api.types.is_bool_dtype(x) for x in solution.dtypes.values])
    elif isinstance(solution, np.ndarray):
        solution_is_all_numeric = pandas.api.types.is_numeric_dtype(solution)
        solution_has_bools = pandas.api.types.is_bool_dtype(solution)

    if not solution_is_all_numeric:
        return False

    for char in error_message:
        if char.isnumeric():
            return False
    if solution_has_bools:
        if 'true' in error_message.lower() or 'false' in error_message.lower():
            return False
    return True


def safe_call_score(metric_function, solution, submission, **metric_func_kwargs):
    '''
    Call score. If that raises an error and that already been specifically handled, just raise it.
    Otherwise make a conservative attempt to identify potential participant visible errors.
    '''
    try:
        score_result = metric_function(solution, submission, **metric_func_kwargs)
    except Exception as err:
        error_message = str(err)
        if err.__class__.__name__ == 'ParticipantVisibleError':
            raise ParticipantVisibleError(error_message)
        elif err.__class__.__name__ == 'HostVisibleError':
            raise HostVisibleError(error_message)
        else:
            if treat_as_participant_error(error_message, solution):
                raise ParticipantVisibleError(error_message)
            else:
                raise err
    return score_result


def verify_valid_probabilities(df: pd.DataFrame, df_name: str):
    """ Verify that the dataframe contains valid probabilities.

    The dataframe must be limited to the target columns; do not pass in any ID columns.
    """
    if not pandas.api.types.is_numeric_dtype(df.values):
        raise ParticipantVisibleError(f'All target values in {df_name} must be numeric')

    if df.min().min() < 0:
        raise ParticipantVisibleError(f'All target values in {df_name} must be at least zero')

    if df.max().max() > 1:
        raise ParticipantVisibleError(f'All target values in {df_name} must be no greater than one')

    if not np.allclose(df.sum(axis=1), 1):
        raise ParticipantVisibleError(f'Target values in {df_name} do not add to one within all rows')

def score(sol: pd.DataFrame, sub: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    Version of macro-averaged ROC-AUC score that ignores all classes that have no true positive labels.
    '''
    solution = sol.copy()
    submission = sub.copy()
    
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if not pandas.api.types.is_numeric_dtype(submission.values):
        bad_dtypes = {x: submission[x].dtype  for x in submission.columns if not pandas.api.types.is_numeric_dtype(submission[x])}
        raise ParticipantVisibleError(f'Invalid submission data types found: {bad_dtypes}')

    solution_sums = solution.sum(axis=0)
    scored_columns = list(solution_sums[solution_sums > 0].index.values)
    assert len(scored_columns) > 0

    return safe_call_score(sklearn.metrics.roc_auc_score, solution[scored_columns].values, submission[scored_columns].values, average='macro')


submission_effnet = pd.read_csv('/kaggle/input/effnetfinal/submission_effnet.csv')
submission_regnet = pd.read_csv('/kaggle/input/regnetfinal/submission_regnet.csv')

if not submission_effnet['row_id'].equals(submission_regnet['row_id']):
    print("Warning: Row IDs of submissions do not match. Blending may be incorrect.")


meansub_effnet = submission_effnet.copy()
meansub_effnet['row_id'] = meansub_effnet['row_id'].str.split('_').str[0]
meansub_effnet = meansub_effnet.groupby('row_id', as_index=False).max()

meansub_regnet = submission_regnet.copy()
meansub_regnet['row_id'] = meansub_regnet['row_id'].str.split('_').str[0]
meansub_regnet = meansub_regnet.groupby('row_id', as_index=False).max()

if not meansub_effnet['row_id'].equals(meansub_regnet['row_id']):
    print("Warning: Row IDs of submissions do not match. Blending may be incorrect.")


train = pd.read_csv("/kaggle/input/birdclef-2025/train.csv")
train["row_id"] = train["filename"].str.extract(r"/([^/.]+)\.")[0]

targets = meansub_effnet.copy()
for col in targets.columns:
    if col != 'row_id':
        targets[col] = 0
targets = targets.set_index("row_id")

for _, row in train.iterrows():
    rid = row["row_id"]
    primary = row["primary_label"]

    if rid in targets.index and primary in targets.columns:
        targets.at[rid, primary] = 1
    
        sec_list = ast.literal_eval(row["secondary_labels"])
        for s in sec_list:
            targets.at[rid, s] = 1

targets = targets.reset_index()
targets = targets.drop(columns=[""])


meansub_effnet_sorted = meansub_effnet.sort_values(by='row_id').reset_index(drop=True)
meansub_regnet_sorted = meansub_regnet.sort_values(by='row_id').reset_index(drop=True)
targets_sorted = targets.sort_values(by='row_id').reset_index(drop=True)


if not meansub_effnet['row_id'].equals(meansub_regnet['row_id']):
    print("Warning: Row IDs of submissions do not match. Blending may be incorrect.")
if not meansub_effnet['row_id'].equals(targets['row_id']):
    print("Warning: Row IDs of submissions do not match. Blending may be incorrect.")


print(score(targets_sorted, meansub_effnet_sorted, 'row_id'))
print(score(targets_sorted, meansub_regnet_sorted, 'row_id'))


weight_effnet = 1
weight_regnet = 0
result = meansub_effnet_sorted.set_index('row_id').multiply(weight_effnet).add(meansub_regnet_sorted.set_index('row_id').multiply(weight_regnet), fill_value=0).reset_index()
print(score(targets_sorted, result, 'row_id'))


results = []

# Iterate over weights for effnet from 0.0 to 1.0 in steps of 0.1
for w_eff in np.arange(0.0, 1.01, 0.025):
    w_reg = 1.0 - w_eff

    # Compute the weighted sum of predictions
    combined = (
        meansub_effnet_sorted.set_index('row_id').multiply(w_eff)
        .add(meansub_regnet_sorted.set_index('row_id').multiply(w_reg), fill_value=0)
        .reset_index()
    )
    
    # Compute the score
    sc = score(targets_sorted, combined, 'row_id')
    
    # Append to results
    results.append({
        'weight_effnet': round(w_eff, 1),
        'weight_regnet': round(w_reg, 1),
        'score': sc
    })

# Create a DataFrame from results
scores_df = pd.DataFrame(results)

# Display the table of weights vs. score
print(scores_df.to_string(index=False))

# Find best combination
best_row = scores_df.loc[scores_df['score'].idxmax()]
print(f"\nBest score: {best_row['score']} at weight_effnet={best_row['weight_effnet']}, weight_regnet={best_row['weight_regnet']}")


# Define the objective function for Optuna
def objective(trial):
    # Suggest a weight for effnet between 0 and 1
    w_eff = trial.suggest_float("weight_effnet", 0.0, 1.0)
    w_reg = 1.0 - w_eff

    # Combine the two model predictions
    combined = (
        meansub_effnet_sorted.set_index('row_id').multiply(w_eff)
        .add(meansub_regnet_sorted.set_index('row_id').multiply(w_reg), fill_value=0)
        .reset_index()
    )

    # Compute and return the score (to maximize)
    return score(targets_sorted, combined, 'row_id')

# Create a study that maximizes the score
study = optuna.create_study(direction="maximize")

# Run the optimization for a given number of trials (e.g., 50)
study.optimize(objective, n_trials=50)

# Extract the best trial
best_trial = study.best_trial
best_w_eff = best_trial.params["weight_effnet"]
best_w_reg = 1.0 - best_w_eff
best_score = best_trial.value

# Print the best result
print(f"Best score: {best_score:.6f} at weight_effnet={best_w_eff:.3f}, weight_regnet={best_w_reg:.3f}")

# (Optional) If you want a DataFrame of all trials:
trials_df = study.trials_dataframe()
from ace_tools import display_dataframe_to_user
display_dataframe_to_user(name="Optuna Trial Results", dataframe=trials_df)





