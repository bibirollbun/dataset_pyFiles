import pandas as pd
import numpy as np
from tqdm import tqdm


"""F Beta customized for the data format of the MABe challenge."""

import json

from collections import defaultdict

import pandas as pd
import polars as pl


class HostVisibleError(Exception):
    pass


def single_lab_f1(lab_solution: pl.DataFrame, lab_submission: pl.DataFrame, beta: float = 1) -> float:
    label_frames: defaultdict[str, set[int]] = defaultdict(set) # key is video/agent/target/action from solution
    prediction_frames: defaultdict[str, set[int]] = defaultdict(set) # key is video/agent/target/action from submission

    for row in lab_solution.to_dicts():
        label_frames[row['label_key']].update(range(row['start_frame'], row['stop_frame']))

    for video in lab_solution['video_id'].unique():
        active_labels: str = lab_solution.filter(pl.col('video_id') == video)['behaviors_labeled'].first()  # ty: ignore
        active_labels: set[str] = set(json.loads(active_labels)) # set of agent,target,action from solution
        predicted_mouse_pairs: defaultdict[str, set[int]] = defaultdict(set) # key is agent,target from submission

        for row in lab_submission.filter(pl.col('video_id') == video).to_dicts(): # every submission row is converted to a dict
            # Since the labels are sparse, we can't evaluate prediction keys not in the active labels.
            if ','.join([str(row['agent_id']), str(row['target_id']), row['action']]) not in active_labels:
                continue # these submission rows are ignored
           
            new_frames = set(range(row['start_frame'], row['stop_frame']))
            # Ignore truly redundant predictions.
            new_frames = new_frames.difference(prediction_frames[row['prediction_key']])
            prediction_pair = ','.join([str(row['agent_id']), str(row['target_id'])])
            if predicted_mouse_pairs[prediction_pair].intersection(new_frames):
                # A single agent can have multiple targets per frame (ex: evading all other mice) but only one action per target per frame.
                raise HostVisibleError('Multiple predictions for the same frame from one agent/target pair')
            prediction_frames[row['prediction_key']].update(new_frames)
            predicted_mouse_pairs[prediction_pair].update(new_frames)

    tps = defaultdict(int) # key is action
    fns = defaultdict(int) # key is action
    fps = defaultdict(int) # key is action
    for key, pred_frames in prediction_frames.items():
        action = key.split('_')[-1]
        matched_label_frames = label_frames[key]
        tps[action] += len(pred_frames.intersection(matched_label_frames))
        fns[action] += len(matched_label_frames.difference(pred_frames))
        fps[action] += len(pred_frames.difference(matched_label_frames))

    distinct_actions = set()
    for key, frames in label_frames.items():
        action = key.split('_')[-1]
        distinct_actions.add(action)
        if key not in prediction_frames:
            fns[action] += len(frames)

    action_f1s = []
    for action in distinct_actions:
        # print(f"{tps[action]:8} {fns[action]:8} {fps[action]:8}")
        if tps[action] + fns[action] + fps[action] == 0:
            action_f1s.append(0)
        else:
            action_f1s.append((1 + beta**2) * tps[action] / ((1 + beta**2) * tps[action] + beta**2 * fns[action] + fps[action]))
    return sum(action_f1s) / len(action_f1s)


def mouse_fbeta(solution: pd.DataFrame, submission: pd.DataFrame, beta: float = 1) -> float:
    """
    Doctests:
    >>> solution = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 10, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ... ])
    >>> submission = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 10},
    ... ])
    >>> mouse_fbeta(solution, submission)
    1.0

    >>> solution = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 10, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ... ])
    >>> submission = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'mount', 'start_frame': 0, 'stop_frame': 10}, # Wrong action
    ... ])
    >>> mouse_fbeta(solution, submission)
    0.0

    >>> solution = pd.DataFrame([
    ...     {'video_id': 123, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 9, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ...     {'video_id': 123, 'agent_id': 1, 'target_id': 2, 'action': 'mount', 'start_frame': 15, 'stop_frame': 24, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ... ])
    >>> submission = pd.DataFrame([
    ...     {'video_id': 123, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 9},
    ... ])
    >>> "%.12f" % mouse_fbeta(solution, submission)
    '0.500000000000'

    >>> solution = pd.DataFrame([
    ...     {'video_id': 123, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 9, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ...     {'video_id': 123, 'agent_id': 1, 'target_id': 2, 'action': 'mount', 'start_frame': 15, 'stop_frame': 24, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ...     {'video_id': 345, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 9, 'lab_id': 2, 'behaviors_labeled': '["1,2,attack"]'},
    ...     {'video_id': 345, 'agent_id': 1, 'target_id': 2, 'action': 'mount', 'start_frame': 15, 'stop_frame': 24, 'lab_id': 2, 'behaviors_labeled': '["1,2,attack"]'},
    ... ])
    >>> submission = pd.DataFrame([
    ...     {'video_id': 123, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 9},
    ... ])
    >>> "%.12f" % mouse_fbeta(solution, submission)
    '0.250000000000'

    >>> # Overlapping solution events, one prediction matching both.
    >>> solution = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 10, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 10, 'stop_frame': 20, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ... ])
    >>> submission = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 20},
    ... ])
    >>> mouse_fbeta(solution, submission)
    1.0

    >>> solution = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 10, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 30, 'stop_frame': 40, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ... ])
    >>> submission = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 40},
    ... ])
    >>> mouse_fbeta(solution, submission)
    0.6666666666666666
    """
    if len(solution) == 0 or len(submission) == 0:
        raise ValueError('Missing solution or submission data')

    expected_cols = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']

    for col in expected_cols:
        if col not in solution.columns:
            raise ValueError(f'Solution is missing column {col}')
        if col not in submission.columns:
            raise ValueError(f'Submission is missing column {col}')

    solution: pl.DataFrame = pl.DataFrame(solution)
    submission: pl.DataFrame = pl.DataFrame(submission)
    assert (solution['start_frame'] <= solution['stop_frame']).all()
    assert (submission['start_frame'] <= submission['stop_frame']).all()
    solution_videos = set(solution['video_id'].unique())
    # Need to align based on video IDs as we can't rely on the row IDs for handling public/private splits.
    submission = submission.filter(pl.col('video_id').is_in(solution_videos))

    solution = solution.with_columns(
        pl.concat_str(
            [
                pl.col('video_id').cast(pl.Utf8),
                pl.col('agent_id').cast(pl.Utf8),
                pl.col('target_id').cast(pl.Utf8),
                pl.col('action'),
            ],
            separator='_',
        ).alias('label_key'),
    )
    submission = submission.with_columns(
        pl.concat_str(
            [
                pl.col('video_id').cast(pl.Utf8),
                pl.col('agent_id').cast(pl.Utf8),
                pl.col('target_id').cast(pl.Utf8),
                pl.col('action'),
            ],
            separator='_',
        ).alias('prediction_key'),
    )

    lab_scores = []
    for lab in solution['lab_id'].unique():
        lab_solution = solution.filter(pl.col('lab_id') == lab).clone()
        lab_videos = set(lab_solution['video_id'].unique())
        lab_submission = submission.filter(pl.col('video_id').is_in(lab_videos)).clone()
        # print(len(lab_solution), len(lab_videos), len(lab_submission), single_lab_f1(lab_solution, lab_submission, beta=beta))
        lab_scores.append(single_lab_f1(lab_solution, lab_submission, beta=beta))

    return sum(lab_scores) / len(lab_scores)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, beta: float = 1) -> float:
    """
    F1 score for the MABe Challenge
    """
    solution = solution.drop(row_id_column_name, axis='columns', errors='ignore')
    submission = submission.drop(row_id_column_name, axis='columns', errors='ignore')
    return mouse_fbeta(solution, submission, beta=beta)


train = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')
train_subset = train.query("~ lab_id.str.startswith('MABe22_')")



def create_solution_df(dataset):
    """Create the solution dataframe for validating out-of-fold predictions.

    From https://www.kaggle.com/code/ambrosm/mabe-validated-baseline-without-machine-learning/
    
    Parameters:
    dataset: (a subset of) the train dataframe
    
    Return values:
    solution: solution dataframe in the correct format for the score() function
    """
    solution = []
    for train_idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
    
        # Load annotation file
        lab_id = row['lab_id']
        if lab_id.startswith('MABe22'): continue
        video_id = row['video_id']
        path = f"/kaggle/input/MABe-mouse-behavior-detection/train_annotation/{lab_id}/{video_id}.parquet"
        try:
            annot = pd.read_parquet(path.replace('train_tracking', 'train_annotation'))
        except FileNotFoundError:
            print(f"No annotations for {path}")
            continue
    
        # Add all annotations to the solution
        self_annotation = annot.target_id == annot.agent_id
        annot['lab_id'] = lab_id
        annot['video_id'] = video_id
        annot['behaviors_labeled'] = row['behaviors_labeled']
        annot['agent_id'] = annot['agent_id'].apply(lambda s: f"mouse{s}")
        annot['target_id'] = np.where(self_annotation, 'self', annot['target_id'].apply(lambda s: f"mouse{s}"))
        solution.append(annot)
    
    solution = pd.concat(solution)
    return solution

solution = create_solution_df(train_subset)


def predict_without_ml(dataset, traintest):
    """Predict actions without machine learning
    
    Parameters:
    dataset: (a subset of) the train or test dataframe
    traintest: 'train' or 'test'
    
    Return values:
    submission: submission dataframe
    """
    submission = []
    for train_idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
    
        # Load video
        lab_id = row['lab_id']
        if lab_id.startswith('MABe22'): continue
        video_id = row['video_id']
        path = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
    
        # Determine the behaviors of this video
        if type(row.behaviors_labeled) != str:
            continue
        vid_behaviors = json.loads(row['behaviors_labeled'])
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])
    
        # Determine start_frame and stop_frame
        start_frame = vid.video_frame.min()
        stop_frame = vid.video_frame.max() + 1
    
        # Predict all possible actions as often as possible
        for (agent, target), actions in vid_behaviors.groupby(['agent', 'target']):
            batch_length = int(np.ceil((stop_frame - start_frame) / len(actions)))
            for i, (_, action_row) in enumerate(actions.iterrows()):
                batch_start = start_frame + i * batch_length
                batch_stop = min(batch_start + batch_length, stop_frame)
                submission.append((video_id, agent, target, action_row['action'], batch_start, batch_stop))
    
    # Convert to dataframe
    submission = pd.DataFrame(submission, columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])

    return submission

submission = predict_without_ml(train_subset, 'train')


%%time
score(solution, submission, '')



test = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv')
submission = predict_without_ml(test, 'test')
submission.index.name = 'row_id'
submission.to_csv('submission.csv')
!head submission.csv




