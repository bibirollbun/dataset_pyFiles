VALIDATE_OR_SUBMIT = 'submit'
THRESHOLD = 0.27
N_ESTIMATORS = 100
VERBOSE = False


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import itertools
import warnings
import json
import os
import lightgbm

from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.model_selection import cross_val_predict, GroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


class TrainOnSubsetClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, n_samples):
        self.estimator = estimator
        self.n_samples = n_samples

    def fit(self, X, y):
        downsample = len(X) // self.n_samples
        downsample = max(downsample, 1)
        self.estimator.fit(np.array(X, copy=False)[::downsample],
                           np.array(y, copy=False)[::downsample])
        self.classes_ = self.estimator.classes_
        return self

    def predict_proba(self, X):
        if len(self.classes_) == 1:
            return np.full((len(X), 1), 1.0)
        probs = self.estimator.predict_proba(np.array(X))
        return probs
        
    def predict(self, X):
        return self.estimator.predict(np.array(X))


import polars as pl
from collections import defaultdict

class HostVisibleError(Exception):
    pass

def single_lab_f1(lab_solution: pl.DataFrame, lab_submission: pl.DataFrame, beta: float = 1) -> float:
    label_frames: defaultdict[str, set[int]] = defaultdict(set)
    prediction_frames: defaultdict[str, set[int]] = defaultdict(set)

    for row in lab_solution.to_dicts():
        label_frames[row['label_key']].update(range(row['start_frame'], row['stop_frame']))

    for video in lab_solution['video_id'].unique():
        active_labels: str = lab_solution.filter(pl.col('video_id') == video)['behaviors_labeled'].first()
        active_labels: set[str] = set(json.loads(active_labels))
        predicted_mouse_pairs: defaultdict[str, set[int]] = defaultdict(set)

        for row in lab_submission.filter(pl.col('video_id') == video).to_dicts():
            if ','.join([str(row['agent_id']), str(row['target_id']), row['action']]) not in active_labels:
                continue
           
            new_frames = set(range(row['start_frame'], row['stop_frame']))
            new_frames = new_frames.difference(prediction_frames[row['prediction_key']])
            prediction_pair = ','.join([str(row['agent_id']), str(row['target_id'])])
            if predicted_mouse_pairs[prediction_pair].intersection(new_frames):
                raise HostVisibleError('Multiple predictions for the same frame from one agent/target pair')
            prediction_frames[row['prediction_key']].update(new_frames)
            predicted_mouse_pairs[prediction_pair].update(new_frames)

    tps = defaultdict(int)
    fns = defaultdict(int)
    fps = defaultdict(int)
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
        if tps[action] + fns[action] + fps[action] == 0:
            action_f1s.append(0)
        else:
            action_f1s.append((1 + beta**2) * tps[action] / ((1 + beta**2) * tps[action] + beta**2 * fns[action] + fps[action]))
    return sum(action_f1s) / len(action_f1s)

def mouse_fbeta(solution: pd.DataFrame, submission: pd.DataFrame, beta: float = 1) -> float:
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
    submission = submission.filter(pl.col('video_id').is_in(solution_videos))

    solution = solution.with_columns(
        pl.concat_str([pl.col('video_id').cast(pl.Utf8), pl.col('agent_id').cast(pl.Utf8),
                       pl.col('target_id').cast(pl.Utf8), pl.col('action')],
                      separator='_').alias('label_key'))
    submission = submission.with_columns(
        pl.concat_str([pl.col('video_id').cast(pl.Utf8), pl.col('agent_id').cast(pl.Utf8),
                       pl.col('target_id').cast(pl.Utf8), pl.col('action')],
                      separator='_').alias('prediction_key'))

    lab_scores = []
    for lab in solution['lab_id'].unique():
        lab_solution = solution.filter(pl.col('lab_id') == lab).clone()
        lab_videos = set(lab_solution['video_id'].unique())
        lab_submission = submission.filter(pl.col('video_id').is_in(lab_videos)).clone()
        lab_scores.append(single_lab_f1(lab_solution, lab_submission, beta=beta))

    return sum(lab_scores) / len(lab_scores)

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, beta: float = 1) -> float:
    solution = solution.drop(row_id_column_name, axis='columns', errors='ignore')
    submission = submission.drop(row_id_column_name, axis='columns', errors='ignore')
    return mouse_fbeta(solution, submission, beta=beta)


train = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')
train['n_mice'] = 4 - train[['mouse1_strain', 'mouse2_strain', 'mouse3_strain', 'mouse4_strain']].isna().sum(axis=1)
train_without_mabe22 = train.query("~ lab_id.str.startswith('MABe22_')")

test = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv')
body_parts_tracked_list = list(np.unique(train.body_parts_tracked))


def create_solution_df(dataset):
    solution = []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        lab_id = row['lab_id']
        if lab_id.startswith('MABe22'): continue
        video_id = row['video_id']
        path = f"/kaggle/input/MABe-mouse-behavior-detection/train_annotation/{lab_id}/{video_id}.parquet"
        try:
            annot = pd.read_parquet(path)
        except FileNotFoundError:
            if VERBOSE: print(f"No annotations for {path}")
            continue
    
        annot['lab_id'] = lab_id
        annot['video_id'] = video_id
        annot['behaviors_labeled'] = row['behaviors_labeled']
        annot['target_id'] = np.where(annot.target_id != annot.agent_id, annot['target_id'].apply(lambda s: f"mouse{s}"), 'self')
        annot['agent_id'] = annot['agent_id'].apply(lambda s: f"mouse{s}")
        solution.append(annot)
    
    solution = pd.concat(solution)
    return solution

if VALIDATE_OR_SUBMIT == 'validate':
    solution = create_solution_df(train_without_mabe22)


drop_body_parts =  ['headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
                    'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright', 
                    'spine_1', 'spine_2', 'tail_middle_1', 'tail_middle_2', 'tail_midpoint']

def generate_mouse_data(dataset, traintest, traintest_directory=None, generate_single=True, generate_pair=True):
    assert traintest in ['train', 'test']
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
    for _, row in dataset.iterrows():
        
        lab_id = row.lab_id
        if lab_id.startswith('MABe22'): continue
        video_id = row.video_id

        if type(row.behaviors_labeled) != str:
            if VERBOSE: print('No labeled behaviors:', lab_id, video_id, type(row.behaviors_labeled), row.behaviors_labeled)
            continue

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
        if len(np.unique(vid.bodypart)) > 5:
            vid = vid.query("~ bodypart.isin(@drop_body_parts)")
        pvid = vid.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])
        if pvid.isna().any().any():
            if VERBOSE and traintest == 'test': print('video with missing values', video_id, traintest, len(vid), 'frames')
        else:
            if VERBOSE and traintest == 'test': print('video with all values', video_id, traintest, len(vid), 'frames')
        del vid
        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T
        pvid /= row.pix_per_cm_approx

        vid_behaviors = json.loads(row.behaviors_labeled)
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])
        
        if traintest == 'train':
            try:
                annot = pd.read_parquet(path.replace('train_tracking', 'train_annotation'))
            except FileNotFoundError:
                continue

        if generate_single:
            vid_behaviors_subset = vid_behaviors.query("target == 'self'")
            for mouse_id_str in np.unique(vid_behaviors_subset.agent):
                try:
                    mouse_id = int(mouse_id_str[-1])
                    vid_agent_actions = np.unique(vid_behaviors_subset.query("agent == @mouse_id_str").action)
                    single_mouse = pvid.loc[:, mouse_id]
                    assert len(single_mouse) == len(pvid)
                    single_mouse_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': mouse_id_str,
                        'target_id': 'self',
                        'video_frame': single_mouse.index
                    })
                    if traintest == 'train':
                        single_mouse_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=single_mouse.index)
                        annot_subset = annot.query("(agent_id == @mouse_id) & (target_id == @mouse_id)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            single_mouse_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'single', single_mouse, single_mouse_meta, single_mouse_label
                    else:
                        if VERBOSE: print('- test single', video_id, mouse_id)
                        yield 'single', single_mouse, single_mouse_meta, vid_agent_actions
                except KeyError:
                    pass

        if generate_pair:
            vid_behaviors_subset = vid_behaviors.query("target != 'self'")
            if len(vid_behaviors_subset) > 0:
                for agent, target in itertools.permutations(np.unique(pvid.columns.get_level_values('mouse_id')), 2):
                    agent_str = f"mouse{agent}"
                    target_str = f"mouse{target}"
                    vid_agent_actions = np.unique(vid_behaviors_subset.query("(agent == @agent_str) & (target == @target_str)").action)
                    if len(vid_agent_actions) == 0:
                        continue
                    mouse_pair = pd.concat([pvid[agent], pvid[target]], axis=1, keys=['A', 'B'])
                    assert len(mouse_pair) == len(pvid)
                    mouse_pair_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': agent_str,
                        'target_id': target_str,
                        'video_frame': mouse_pair.index
                    })
                    if traintest == 'train':
                        mouse_pair_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=mouse_pair.index)
                        annot_subset = annot.query("(agent_id == @agent) & (target_id == @target)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            mouse_pair_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'pair', mouse_pair, mouse_pair_meta, mouse_pair_label
                    else:
                        if VERBOSE: print('- test pair', video_id, agent, target)
                        yield 'pair', mouse_pair, mouse_pair_meta, vid_agent_actions


def transform_single(single_mouse, body_parts_tracked):
    """Distance features + velocity + acceleration"""
    available_body_parts = single_mouse.columns.get_level_values(0)
    X = pd.DataFrame({
            f"{part1}+{part2}": np.square(single_mouse[part1] - single_mouse[part2]).sum(axis=1, skipna=False)
            for part1, part2 in itertools.combinations(body_parts_tracked, 2) if part1 in available_body_parts and part2 in available_body_parts
        })
    X = X.reindex(columns=[f"{part1}+{part2}" for part1, part2 in itertools.combinations(body_parts_tracked, 2)], copy=False)

    if 'ear_left' in single_mouse.columns and 'ear_right' in single_mouse.columns and 'tail_base' in single_mouse.columns:
        shifted = single_mouse[['ear_left', 'ear_right', 'tail_base']].shift(10)
        velocity_features = pd.DataFrame({
            'velocity_left': np.square(single_mouse['ear_left'] - shifted['ear_left']).sum(axis=1, skipna=False),
            'velocity_right': np.square(single_mouse['ear_right'] - shifted['ear_right']).sum(axis=1, skipna=False),
            'velocity_tail': np.square(single_mouse['tail_base'] - shifted['tail_base']).sum(axis=1, skipna=False),
        })
        accel_features = pd.DataFrame({
            'accel_left': velocity_features['velocity_left'].diff(10),
            'accel_right': velocity_features['velocity_right'].diff(10),
            'accel_tail': velocity_features['velocity_tail'].diff(10),
        })
        X = pd.concat([X, velocity_features, accel_features], axis=1)
    return X

def transform_pair(mouse_pair, body_parts_tracked):
    """Inter-mouse distances"""
    available_body_parts_A = mouse_pair['A'].columns.get_level_values(0)
    available_body_parts_B = mouse_pair['B'].columns.get_level_values(0)
    X = pd.DataFrame({
            f"12+{part1}+{part2}": np.square(mouse_pair['A'][part1] - mouse_pair['B'][part2]).sum(axis=1, skipna=False)
            for part1, part2 in itertools.product(body_parts_tracked, repeat=2) if part1 in available_body_parts_A and part2 in available_body_parts_B
        })
    X = X.reindex(columns=[f"12+{part1}+{part2}" for part1, part2 in itertools.product(body_parts_tracked, repeat=2)], copy=False)
    return X


def predict_multiclass(pred, meta):
    ama = np.argmax(pred, axis=1)
    ama = np.where(pred.max(axis=1) >= THRESHOLD, ama, -1)
    ama = pd.Series(ama, index=meta.video_frame)
    changes_mask = (ama != ama.shift(1)).values
    ama_changes = ama[changes_mask]
    meta_changes = meta[changes_mask]
    mask = ama_changes.values >= 0
    mask[-1] = False
    submission_part = pd.DataFrame({
        'video_id': meta_changes['video_id'][mask].values,
        'agent_id': meta_changes['agent_id'][mask].values,
        'target_id': meta_changes['target_id'][mask].values,
        'action': pred.columns[ama_changes[mask].values],
        'start_frame': ama_changes.index[mask],
        'stop_frame': ama_changes.index[1:][mask[:-1]]
    })
    stop_video_id = meta_changes['video_id'][1:][mask[:-1]].values
    stop_agent_id = meta_changes['agent_id'][1:][mask[:-1]].values
    stop_target_id = meta_changes['target_id'][1:][mask[:-1]].values
    for i in range(len(submission_part)):
        video_id = submission_part.video_id.iloc[i]
        agent_id = submission_part.agent_id.iloc[i]
        target_id = submission_part.target_id.iloc[i]
        if stop_video_id[i] != video_id or stop_agent_id[i] != agent_id or stop_target_id[i] != target_id:
            new_stop_frame = meta.query("(video_id == @video_id)").video_frame.max() + 1
            submission_part.iat[i, submission_part.columns.get_loc('stop_frame')] = new_stop_frame
    assert (submission_part.stop_frame > submission_part.start_frame).all(), 'stop <= start'
    if VERBOSE: print('  actions found:', len(submission_part))
    return submission_part


def cross_validate_classifier(binary_classifier, X, label, meta):
    oof = pd.DataFrame(index=meta.video_frame)
    for action in label.columns:
        action_mask = ~ label[action].isna().values
        X_action = X[action_mask]
        y_action = label[action][action_mask].values.astype(int)
        p = y_action.mean()
        baseline_score = p / (1 + p)
        groups_action = meta.video_id[action_mask]
        if len(np.unique(groups_action)) < 5:
            continue

        if not (y_action == 0).all():
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                oof_action = cross_val_predict(binary_classifier, X_action, y_action, groups=groups_action, cv=GroupKFold(), method='predict_proba')
            oof_action = oof_action[:, 1]
        else:
            oof_action = np.zeros(len(y_action))
        f1 = f1_score(y_action, (oof_action >= THRESHOLD), zero_division=0)
        ch = '>' if f1 > baseline_score else '=' if f1 == baseline_score else '<'
        if VERBOSE: print(f"  F1: {f1:.3f} {ch} ({baseline_score:.3f}) {action}")
        oof_column = np.zeros(len(label))
        oof_column[action_mask] = oof_action
        oof[action] = oof_column

    submission_part = predict_multiclass(oof, meta)
    return submission_part


def submit(body_parts_tracked_str, switch_tr, binary_classifier, X_tr, label, meta):
    model_list = []
    for action in label.columns:
        action_mask = ~ label[action].isna().values
        y_action = label[action][action_mask].values.astype(int)

        if not (y_action == 0).all():
            model = clone(binary_classifier)
            model.fit(X_tr[action_mask], y_action)
            assert len(model.classes_) == 2
            model_list.append((action, model))

    body_parts_tracked = json.loads(body_parts_tracked_str)
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]
    if VALIDATE_OR_SUBMIT == 'submit':
        test_subset = test[test.body_parts_tracked == body_parts_tracked_str]
        generator = generate_mouse_data(test_subset, 'test',
                                        generate_single=(switch_tr == 'single'), 
                                        generate_pair=(switch_tr == 'pair'))
    if VERBOSE: print(f"n_videos: {len(test_subset)}")
    for switch_te, data_te, meta_te, actions_te in generator:
        assert switch_te == switch_tr
        try:
            if switch_te == 'single':
                X_te = transform_single(data_te, body_parts_tracked)
            else:
                X_te = transform_pair(data_te, body_parts_tracked)
            if VERBOSE and len(X_te) == 0: print("ERROR: X_te is empty")
            del data_te
    
            pred = pd.DataFrame(index=meta_te.video_frame)
            for action, model in model_list:
                if action in actions_te:
                    pred[action] = model.predict_proba(X_te)[:, 1]
            del X_te
            if pred.shape[1] != 0:
                submission_part = predict_multiclass(pred, meta_te)
                submission_list.append(submission_part)
            else:
                if VERBOSE: print(f"  ERROR: no useful training data")
        except KeyError:
            if VERBOSE: print(f'  ERROR: KeyError because of missing bodypart ({switch_tr})')
            del data_te


f1_list = []
submission_list = []

for section in range(1, len(body_parts_tracked_list)):
    body_parts_tracked_str = body_parts_tracked_list[section]
    try:
        body_parts_tracked = json.loads(body_parts_tracked_str)
        print(f"{section}. Processing videos with {body_parts_tracked}")
        if len(body_parts_tracked) > 5:
            body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]
    
        train_subset = train[train.body_parts_tracked == body_parts_tracked_str]
        single_mouse_list = []
        single_mouse_label_list = []
        single_mouse_meta_list = []
        mouse_pair_list = []
        mouse_pair_label_list = []
        mouse_pair_meta_list = []
    
        for switch, data, meta, label in generate_mouse_data(train_subset, 'train'):
            if switch == 'single':
                single_mouse_list.append(data)
                single_mouse_meta_list.append(meta)
                single_mouse_label_list.append(label)
            else:
                mouse_pair_list.append(data)
                mouse_pair_meta_list.append(meta)
                mouse_pair_label_list.append(label)
    
        binary_classifier = make_pipeline(
            SimpleImputer(),
            TrainOnSubsetClassifier(
                lightgbm.LGBMClassifier(
                    n_estimators=N_ESTIMATORS,
                    learning_rate=0.03,
                    min_child_samples=40,
                    verbose=-1),
                100000)
        )
    
        if len(single_mouse_list) > 0:
            single_mouse = pd.concat(single_mouse_list)
            single_mouse_label = pd.concat(single_mouse_label_list)
            single_mouse_meta = pd.concat(single_mouse_meta_list)
            del single_mouse_list, single_mouse_label_list, single_mouse_meta_list
            assert len(single_mouse) == len(single_mouse_label)
            assert len(single_mouse) == len(single_mouse_meta)
            
            X_tr = transform_single(single_mouse, body_parts_tracked)
            del single_mouse
            print(f"{X_tr.shape=}")
    
            if VALIDATE_OR_SUBMIT == 'validate':
                submission_part = cross_validate_classifier(binary_classifier, X_tr, single_mouse_label, single_mouse_meta)
                submission_list.append(submission_part)
            else:
                submit(body_parts_tracked_str, 'single', binary_classifier, X_tr, single_mouse_label, single_mouse_meta)
            del X_tr
                
        if len(mouse_pair_list) > 0:
            mouse_pair = pd.concat(mouse_pair_list)
            mouse_pair_label = pd.concat(mouse_pair_label_list)
            mouse_pair_meta = pd.concat(mouse_pair_meta_list)
            del mouse_pair_list, mouse_pair_label_list, mouse_pair_meta_list
            assert len(mouse_pair) == len(mouse_pair_label)
            assert len(mouse_pair) == len(mouse_pair_meta)
        
            X_tr = transform_pair(mouse_pair, body_parts_tracked)
            del mouse_pair
            print(f"{X_tr.shape=}")
    
            if VALIDATE_OR_SUBMIT == 'validate':
                submission_part = cross_validate_classifier(binary_classifier, X_tr, mouse_pair_label, mouse_pair_meta)
                submission_list.append(submission_part)
            else:
                submit(body_parts_tracked_str, 'pair', binary_classifier, X_tr, mouse_pair_label, mouse_pair_meta)
            del X_tr
                
    except Exception as e:
        print(f'***Exception*** {e}')
    print()


def robustify(submission, dataset, traintest, traintest_directory=None):
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"

    old_submission = submission.copy()
    submission = submission[submission.start_frame < submission.stop_frame]
    if len(submission) != len(old_submission):
        print("ERROR: Dropped frames with start >= stop")
    
    old_submission = submission.copy()
    group_list = []
    for _, group in submission.groupby(['video_id', 'agent_id', 'target_id']):
        group = group.sort_values('start_frame')
        mask = np.ones(len(group), dtype=bool)
        last_stop_frame = 0
        for i, (_, row) in enumerate(group.iterrows()):
            if row['start_frame'] < last_stop_frame:
                mask[i] = False
            else:
                last_stop_frame = row['stop_frame']
        group_list.append(group[mask])
    submission = pd.concat(group_list)
    if len(submission) != len(old_submission):
        print("ERROR: Dropped duplicate frames")

    s_list = []
    for idx, row in dataset.iterrows():
        lab_id = row['lab_id']
        if lab_id.startswith('MABe22'):
            continue
        video_id = row['video_id']
        if (submission.video_id == video_id).any():
            continue

        if VERBOSE: print(f"Video {video_id} has no predictions.")
        
        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
    
        vid_behaviors = eval(row['behaviors_labeled'])
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])
    
        start_frame = vid.video_frame.min()
        stop_frame = vid.video_frame.max() + 1
    
        for (agent, target), actions in vid_behaviors.groupby(['agent', 'target']):
            batch_length = int(np.ceil((stop_frame - start_frame) / len(actions)))
            for i, (_, action_row) in enumerate(actions.iterrows()):
                batch_start = start_frame + i * batch_length
                batch_stop = min(batch_start + batch_length, stop_frame)
                s_list.append((video_id, agent, target, action_row['action'], batch_start, batch_stop))

    if len(s_list) > 0:
        submission = pd.concat([
            submission,
            pd.DataFrame(s_list, columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
        ])
        print("ERROR: Filled empty videos")

    submission = submission.reset_index(drop=True)
    return submission


if VALIDATE_OR_SUBMIT == 'validate':
    submission = pd.concat(submission_list)
    submission_robust = robustify(submission, train, 'train')
    print(f"# OOF score with competition metric: {score(solution, submission_robust, ''):.4f}")

    f1_df = pd.DataFrame(f1_list, columns=['body_parts_tracked_str', 'action', 'binary F1 score'])
    print(f"# Average of {len(f1_df)} binary F1 scores {f1_df['binary F1 score'].mean():.4f}")


if VALIDATE_OR_SUBMIT != 'validate':
    if len(submission_list) > 0:
        submission = pd.concat(submission_list)
    else:
        submission = pd.DataFrame(
            dict(
                video_id=438887472,
                agent_id='mouse1',
                target_id='self',
                action='rear',
                start_frame='278',
                stop_frame='500'
            ), index=[44])
    if VALIDATE_OR_SUBMIT == 'submit':
        submission_robust = robustify(submission, test, 'test')
    submission_robust.index.name = 'row_id'
    submission_robust.to_csv('submission.csv')
    print(f"Submission saved with {len(submission_robust)} predictions")
    submission_robust.head()

