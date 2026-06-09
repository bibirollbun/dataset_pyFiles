"""F Beta customized for the data format of the MABe challenge."""

import json

from collections import defaultdict

import pandas as pd
import polars as pl


class HostVisibleError(Exception):
    pass


def single_lab_f1(lab_solution: pl.DataFrame, lab_submission: pl.DataFrame, beta: float = 1) -> float:
    label_frames: defaultdict[str, set[int]] = defaultdict(set)
    prediction_frames: defaultdict[str, set[int]] = defaultdict(set)

    for row in lab_solution.to_dicts():
        label_frames[row['label_key']].update(range(row['start_frame'], row['stop_frame']))

    for video in lab_solution['video_id'].unique():
        active_labels: str = lab_solution.filter(pl.col('video_id') == video)['behaviors_labeled'].first()  # ty: ignore
        active_labels: set[str] = set(json.loads(active_labels))
        predicted_mouse_pairs: defaultdict[str, set[int]] = defaultdict(set)

        for row in lab_submission.filter(pl.col('video_id') == video).to_dicts():
            # Since the labels are sparse, we can't evaluate prediction keys not in the active labels.
            if ','.join([str(row['agent_id']), str(row['target_id']), row['action']]) not in active_labels:
                continue

            new_frames = set(range(row['start_frame'], row['stop_frame']))
            # Ignore truly redundant predictions.
            new_frames = new_frames.difference(prediction_frames[row['prediction_key']])
            prediction_pair = ','.join([str(row['agent_id']), str(row['target_id'])])
            if predicted_mouse_pairs[prediction_pair].intersection(new_frames):
                # A single agent can have multiple targets per frame (ex: evading all other mice) but only one action per target per frame.
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
        lab_scores.append(single_lab_f1(lab_solution, lab_submission, beta=beta))

    return sum(lab_scores) / len(lab_scores)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, beta: float = 1) -> float:
    """
    F1 score for the MABe Challenge
    """
    solution = solution.drop(row_id_column_name, axis='columns', errors='ignore')
    submission = submission.drop(row_id_column_name, axis='columns', errors='ignore')
    return mouse_fbeta(solution, submission, beta=beta)


from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from sklearn.base import clone
from xgboost import XGBClassifier
from tqdm.notebook import tqdm
from catboost import CatBoostClassifier
import joblib
import os
import numpy as np
import itertools
import warnings
import optuna
import joblib
import glob
import gc

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')


class CFG:
    # ===============================
    # æ•°æ�®è·¯å¾„ï¼ˆå›ºå®šï¼‰
    # ===============================
    train_path = "/kaggle/input/MABe-mouse-behavior-detection/train.csv"
    test_path = "/kaggle/input/MABe-mouse-behavior-detection/test.csv"
    train_annotation_path = "/kaggle/input/MABe-mouse-behavior-detection/train_annotation"
    train_tracking_path = "/kaggle/input/MABe-mouse-behavior-detection/train_tracking"
    test_tracking_path = "/kaggle/input/MABe-mouse-behavior-detection/test_tracking"

    # ===============================
    # æ¨¡å�‹ä¿�å­˜è·¯å¾„
    # ===============================
    # æ¨¡å�‹æœ€ç»ˆä¼šä¿�å­˜åœ¨ï¼š
    # /kaggle/working/models/xgb_cat/{section}/{action}/(xgb_i.pkl, cat_i.pkl)
    model_path = "/kaggle/input/mabe-xgb-catb/models"
    model_name = "xgb_cat"

    # ===============================
    # è®­ç»ƒ / æ��äº¤æµ�ç¨‹
    # ===============================
    # å�¯é€‰: "validate" / "submit"
    mode = "submit"

    # ===============================
    # CV è®¾ç½®
    # ===============================
    n_splits = 3
    cv = StratifiedGroupKFold(n_splits)
    random_state = 42
    # -----------------------
    # XGBoost é…�ç½®ï¼ˆå�Œ T4 æŠ˜çº§è½®è®­ï¼‰
    # -----------------------
    xgb_params = dict(
        verbosity=0,  # å…³é—­å†—ä½™è¾“å‡º
        random_state=random_state,
        n_estimators=200,  # è¿­ä»£æ¬¡æ•°ï¼ˆå�¯æ ¹æ�®é€Ÿåº¦è°ƒæ•´ä¸º 200ï¼‰
        learning_rate=0.08,  # å­¦ä¹ ç�‡ï¼ˆé…�å�ˆæ—©å�œä½¿ç”¨ï¼‰
        max_depth=6,  # æ ‘æ·±åº¦ï¼ˆå¹³è¡¡æ‹Ÿå�ˆèƒ½åŠ›ä¸�è¿‡æ‹Ÿå�ˆï¼‰
        min_child_weight=5,  # æœ€å°�å­�èŠ‚ç‚¹æ�ƒé‡�ï¼ˆæŠ‘åˆ¶è¿‡æ‹Ÿå�ˆï¼‰
        subsample=0.8,  # æ ·æœ¬é‡‡æ ·æ¯”ä¾‹ï¼ˆå‡�å°‘è®¡ç®—é‡�ï¼‰
        colsample_bytree=0.8,  # ç‰¹å¾�é‡‡æ ·æ¯”ä¾‹ï¼ˆå‡�å°‘è¿‡æ‹Ÿå�ˆï¼‰
        tree_method="gpu_hist",  # å¼ºåˆ¶ GPU è®­ç»ƒï¼ˆT4 ä¼˜åŒ–ï¼‰
        gpu_id=0,  # å�•å�¡å� ä½�ï¼ˆè®­ç»ƒæ—¶æŒ‰æŠ˜åˆ‡æ�¢ 0/1ï¼‰
        predictor="gpu_predictor",  # GPU é¢„æµ‹ï¼ˆæ��é€Ÿï¼‰
        eval_metric="logloss",  # è¯„ä¼°æŒ‡æ ‡ï¼ˆä¸� CatBoost ä¸€è‡´ï¼‰
        early_stopping_rounds=30,  # æ—©å�œï¼ˆé�¿å…�æ— æ•ˆè¿­ä»£ï¼‰
    )
    
    # -----------------------
    # CatBoost é…�ç½®ï¼ˆä¿®å¤� GPU RSM æŠ¥é”™ + å�Œ T4 å¹¶è¡Œï¼‰
    # -----------------------
    cat_params = dict(
        loss_function="Logloss",  # æ�Ÿå¤±å‡½æ•°ï¼ˆäºŒåˆ†ç±»ï¼‰
        eval_metric="Logloss",  # è¯„ä¼°æŒ‡æ ‡ï¼ˆä¸� XGBoost ä¸€è‡´ï¼‰
        iterations=250,  # è¿­ä»£æ¬¡æ•°ï¼ˆå�¯è°ƒæ•´ä¸º 250 æ��é€Ÿï¼‰
        depth=6,  # æ ‘æ·±åº¦ï¼ˆä¸� XGBoost ä¿�æŒ�ä¸€è‡´ï¼‰
        learning_rate=0.05,  # å­¦ä¹ ç�‡ï¼ˆé…�å�ˆæ—©å�œä½¿ç”¨ï¼‰
        random_seed=random_state,
        verbose=False,  # å…³é—­å†—ä½™è¾“å‡º
        task_type="GPU",  # å¼ºåˆ¶ GPU è®­ç»ƒ
        devices="0,1",  # å�Œ T4 GPUï¼ˆå­—ç¬¦ä¸²æ ¼å¼�ï¼Œå…¼å®¹æ‰€æœ‰ç‰ˆæœ¬ï¼‰
        gpu_ram_part=0.7,  # æ¯�å�¡æ˜¾å­˜é™�åˆ¶ï¼ˆ16GB T4 é¢„ç•™ 30% æ˜¾å­˜ï¼‰
        bootstrap_type="MVS",  # é‡‡æ ·æ–¹å¼�ï¼ˆæ”¯æŒ� subsampleï¼Œæ€§èƒ½æœ€ä¼˜ï¼‰
        subsample=0.8,  # æ ·æœ¬é‡‡æ ·æ¯”ä¾‹ï¼ˆå‡�å°‘è®¡ç®—é‡�ï¼‰
        # colsample_bylevel=0.8,  # å…³é”®åˆ é™¤ï¼šGPU æ¨¡å¼�äºŒåˆ†ç±»ä¸�æ”¯æŒ�è¯¥å�‚æ•°
        early_stopping_rounds=30,  # æ—©å�œï¼ˆé�¿å…�æ— æ•ˆè¿­ä»£ï¼‰
    )


train = pd.read_csv(CFG.train_path)
train['n_mice'] = 4 - train[['mouse1_strain', 'mouse2_strain', 'mouse3_strain', 'mouse4_strain']].isna().sum(axis=1)
train_without_mabe22 = train.query("~lab_id.str.startswith('MABe22_')")

test = pd.read_csv(CFG.test_path)


body_parts_tracked_list = list(np.unique(train.body_parts_tracked))


def create_solution_df(dataset):
    solution = []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
    
        lab_id = row['lab_id']
        if lab_id.startswith('MABe22'): 
            continue
        
        video_id = row['video_id']
        path = f"{CFG.train_annotation_path}/{lab_id}/{video_id}.parquet"
        try:
            annot = pd.read_parquet(path)
        except FileNotFoundError:
            continue
    
        annot['lab_id'] = lab_id
        annot['video_id'] = video_id
        annot['behaviors_labeled'] = row['behaviors_labeled']
        annot['target_id'] = np.where(annot.target_id != annot.agent_id, annot['target_id'].apply(lambda s: f"mouse{s}"), 'self')
        annot['agent_id'] = annot['agent_id'].apply(lambda s: f"mouse{s}")
        solution.append(annot)
    
    solution = pd.concat(solution)
    
    return solution

if CFG.mode == 'validate':
    solution = create_solution_df(train_without_mabe22)


import json
import itertools
import pandas as pd
import numpy as np
import gc
from tqdm import tqdm

drop_body_parts = [
    'headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
    'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright', 
    'spine_1', 'spine_2', 'tail_middle_1', 'tail_middle_2', 'tail_midpoint'
]



def generate_mouse_data(dataset, traintest, traintest_directory=None, generate_single=True, generate_pair=True):
    import json
    import itertools
    import pandas as pd
    import numpy as np
    import gc
    import os
    
    # ---------------------- è®¾ç½®æ•°æ�®é›†è·¯å¾„ ----------------------
    if traintest_directory is None:
        if traintest == 'test':
            traintest_directory = "/kaggle/input/MABe-mouse-behavior-detection/test_tracking"
        else:
            traintest_directory = "/kaggle/input/MABe-mouse-behavior-detection/train_tracking"
        print(f"ä½¿ç”¨æ•°æ�®é›†è·¯å¾„ï¼š{traintest_directory}")
    
    # ---------------------- è¿‡æ»¤æœ‰æ•ˆæ•°æ�® ----------------------
    mask_lab = dataset.lab_id.str.startswith('MABe22', na=True)
    mask_behavior = dataset.behaviors_labeled.apply(
        lambda x: isinstance(x, str) and x.strip() != ''
    )
    valid_dataset = dataset[~mask_lab & mask_behavior].copy()
    
    if len(valid_dataset) == 0:
        print(f"\tâš ï¸�  æ— æœ‰æ•ˆæ•°æ�®å�¯å¤„ç�†ï¼Œè¿”å›�ç©ºè¿­ä»£å™¨")
        return
    
    # ---------------------- è¿­ä»£å¤„ç�†æ¯�ä¸ªè§†é¢‘ ----------------------
    for _, row in valid_dataset.iterrows():
        lab_id = row.lab_id
        video_id = str(int(row.video_id))
        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        
        print(f"\tğŸ”� æ­£åœ¨å¤„ç�†è§†é¢‘ï¼švideo_id={video_id}, lab_id={lab_id}, è·¯å¾„={path}")
        
        if not os.path.exists(path):
            print(f"\tâš ï¸�  è·³è¿‡ä¸�å­˜åœ¨çš„æ–‡ä»¶ï¼š{path}")
            continue
        
        # è¯»å�–è¿½è¸ªæ–‡ä»¶
        try:
            vid = pd.read_parquet(path)
            print(f"\tâœ… æˆ�åŠŸè¯»å�–è¿½è¸ªæ–‡ä»¶ï¼Œæ•°æ�®å½¢çŠ¶ï¼š{vid.shape}ï¼Œåˆ—å��ï¼š{list(vid.columns)}")
            print(f"\tâœ… è¿½è¸ªæ–‡ä»¶ä¸­ mouse_id å”¯ä¸€å€¼ï¼š{vid['mouse_id'].unique()}")
            print(f"\tâœ… è¿½è¸ªæ–‡ä»¶ä¸­ bodypart å”¯ä¸€å€¼ï¼š{list(vid['bodypart'].unique())[:10]}...ï¼ˆå…±{len(vid['bodypart'].unique())}ä¸ªï¼‰")
        except Exception as e:
            print(f"\tâš ï¸�  è¯»å�–æ–‡ä»¶ {path} å¤±è´¥ï¼š{str(e)}")
            continue
        
        # è¿‡æ»¤ä¸�éœ€è¦�çš„èº«ä½“éƒ¨ä½�ï¼ˆä¾�èµ–å…¨å±€ drop_body_partsï¼Œè‹¥æœªå®šä¹‰åˆ™è·³è¿‡è¿‡æ»¤ï¼‰
        if 'drop_body_parts' in globals() and len(np.unique(vid.bodypart)) > 5:
            before_drop = len(vid['bodypart'].unique())
            vid = vid.query("~ bodypart.isin(@drop_body_parts)")
            after_drop = len(vid['bodypart'].unique())
            print(f"\tâœ… èº«ä½“éƒ¨ä½�è¿‡æ»¤ï¼šä»� {before_drop} ä¸ªå‡�å°‘åˆ° {after_drop} ä¸ªï¼Œä¿�ç•™ï¼š{list(vid['bodypart'].unique())[:10]}...")
        
        # é€�è§†è¡¨æ•´ç�†æ•°æ�®
        try:
            pvid = vid.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])
            pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T
            pvid /= row.pix_per_cm_approx
            print(f"\tâœ… é€�è§†è¡¨æ•´ç�†å®Œæˆ�ï¼Œå½¢çŠ¶ï¼š{pvid.shape}ï¼Œåˆ—å��å±‚çº§ï¼š{pvid.columns.names}")
            print(f"\tâœ… é€�è§†è¡¨ä¸­æœ‰æ•ˆå°�é¼ IDï¼š{list(pvid.columns.get_level_values('mouse_id').unique())}")
        except Exception as e:
            print(f"\tâš ï¸�  æ•´ç�†æ–‡ä»¶ {path} æ•°æ�®å¤±è´¥ï¼š{str(e)}")
            del vid
            gc.collect()
            continue
        
        # æ£€æŸ¥é€�è§†è¡¨æ˜¯å�¦ä¸ºç©º
        if pvid.empty or len(pvid.columns) == 0:
            print(f"\tâš ï¸�  é€�è§†è¡¨ä¸ºç©ºæˆ–æ— æœ‰æ•ˆåˆ—ï¼Œè·³è¿‡è¯¥è§†é¢‘")
            del vid, pvid
            gc.collect()
            continue
        
        del vid
        gc.collect()
        
        # ---------------------- è§£æ��è¡Œä¸ºæ ‡ç­¾ï¼ˆä¿®å¤�ï¼šæ”¯æŒ� mouseX,self,action æ ¼å¼�ï¼‰----------------------
        try:
            vid_behaviors = json.loads(row.behaviors_labeled)
            vid_behaviors = sorted(list({b.replace("'", "").strip() for b in vid_behaviors}))
            print(f"\tâœ… è§£æ��è¡Œä¸ºæ ‡ç­¾ï¼šå�Ÿå§‹è¡Œä¸ºåˆ—è¡¨={vid_behaviors[:5]}...ï¼ˆå…±{len(vid_behaviors)}ä¸ªï¼‰")
            
            if traintest == 'test':
                # æµ‹è¯•é›†ï¼šæ ‡ç­¾æ˜¯ "mouseA,mouseB,action" æˆ– "mouseA,self,action" æ ¼å¼�ï¼Œç›´æ�¥æ‹†åˆ†
                vid_behaviors_df = []
                for behavior_str in vid_behaviors:
                    parts = behavior_str.split(',')
                    if len(parts) == 3:
                        agent_str = parts[0].strip()  # æ¯”å¦‚ "mouse1"
                        target_str = parts[1].strip()  # æ¯”å¦‚ "mouse2" æˆ– "self"
                        action = parts[2].strip()      # æ¯”å¦‚ "approach" æˆ– "rear"
                        
                        # éªŒè¯�æ ¼å¼�ï¼šagent å¿…é¡»æ˜¯ "mouseX"ï¼Œtarget å�¯ä»¥æ˜¯ "mouseX" æˆ– "self"
                        if agent_str.startswith('mouse') and (target_str.startswith('mouse') or target_str == 'self'):
                            vid_behaviors_df.append([agent_str, target_str, action])
                            # å�ªæ‰“å�°å‰�10ä¸ªè¡Œä¸ºï¼Œé�¿å…�æ—¥å¿—è¿‡é•¿
                            if len(vid_behaviors_df) <= 10:
                                print(f"\t\tğŸ”� æ‹†åˆ†è¡Œä¸ºï¼šagent={agent_str}, target={target_str}, action={action}")
                        else:
                            print(f"\t\tâš ï¸�  æ— æ•ˆæ ¼å¼�ï¼š{behavior_str}ï¼ˆagentéœ€ä¸ºmouseXï¼Œtargetéœ€ä¸ºmouseXæˆ–selfï¼‰ï¼Œè·³è¿‡")
                    else:
                        print(f"\t\tâš ï¸�  æ— æ•ˆæ ¼å¼�ï¼š{behavior_str}ï¼ˆæ‹†åˆ†å��ä¸�æ˜¯3éƒ¨åˆ†ï¼‰ï¼Œè·³è¿‡")
                
                # è½¬æ�¢ä¸ºDataFrameå¹¶è¿‡æ»¤ç©ºæ•°æ�®
                vid_behaviors = pd.DataFrame(vid_behaviors_df, columns=['agent', 'target', 'action'])
                vid_behaviors = vid_behaviors.dropna()
            else:
                # è®­ç»ƒé›†ï¼šå�Ÿæœ‰é€»è¾‘ï¼ˆä¸‰ç»´åˆ—è¡¨ï¼‰
                vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])
            
            print(f"\tâœ… æœ€ç»ˆè¡Œä¸ºæ•°æ�®æ¡†ï¼š\n{vid_behaviors.head()}")
            if vid_behaviors.empty:
                print(f"\tâš ï¸�  æ— æœ‰æ•ˆè¡Œä¸ºæ•°æ�®ï¼Œè·³è¿‡è¯¥è§†é¢‘")
                del pvid, vid_behaviors
                gc.collect()
                continue
            
            # éªŒè¯�è¡Œä¸ºæ•°æ�®æ¡†ä¸­çš„å°�é¼ IDæ˜¯å�¦å­˜åœ¨äº�è¿½è¸ªæ–‡ä»¶ä¸­
            valid_mouse_ids = [f"mouse{mid}" for mid in pvid.columns.get_level_values('mouse_id').unique()]
            invalid_agents = [a for a in vid_behaviors['agent'].unique() if a not in valid_mouse_ids]
            invalid_targets = [t for t in vid_behaviors['target'].unique() if t not in valid_mouse_ids and t != 'self']
            
            if invalid_agents:
                print(f"\tâš ï¸�  æ— æ•ˆagentï¼š{invalid_agents}ï¼ˆè¿½è¸ªæ–‡ä»¶ä¸­å�ªæœ‰ {valid_mouse_ids}ï¼‰ï¼Œä¼šè¿‡æ»¤æ�‰è¿™äº›è¡Œä¸º")
            if invalid_targets:
                print(f"\tâš ï¸�  æ— æ•ˆtargetï¼š{invalid_targets}ï¼ˆè¿½è¸ªæ–‡ä»¶ä¸­å�ªæœ‰ {valid_mouse_ids}ï¼‰ï¼Œä¼šè¿‡æ»¤æ�‰è¿™äº›è¡Œä¸º")
            
            # è¿‡æ»¤æ�‰æ— æ•ˆå°�é¼ IDçš„è¡Œä¸º
            vid_behaviors = vid_behaviors[
                (vid_behaviors['agent'].isin(valid_mouse_ids)) & 
                ((vid_behaviors['target'].isin(valid_mouse_ids)) | (vid_behaviors['target'] == 'self'))
            ].reset_index(drop=True)
            print(f"\tâœ… è¿‡æ»¤å��æœ‰æ•ˆè¡Œä¸ºæ•°ï¼š{len(vid_behaviors)}")

        except Exception as e:
            print(f"\tâš ï¸�  è§£æ��è¡Œä¸ºæ ‡ç­¾å¤±è´¥ï¼š{str(e)}")
            print(f"\tâš ï¸�  è¡Œä¸ºæ ‡ç­¾å�Ÿå§‹æ•°æ�®ï¼š{str(row.behaviors_labeled)[:100]}...")
            del pvid
            gc.collect()
            continue
        
        # ---------------------- è®­ç»ƒæ¨¡å¼�ï¼šè¯»å�–æ ‡æ³¨æ–‡ä»¶ ----------------------
        annot = None
        if traintest == 'train':
            try:
                annot_path = path.replace('train_tracking', 'train_annotation')
                if os.path.exists(annot_path):
                    annot = pd.read_parquet(annot_path)
                else:
                    print(f"\tâš ï¸�  è®­ç»ƒæ ‡æ³¨æ–‡ä»¶ä¸�å­˜åœ¨ï¼š{annot_path}ï¼Œè·³è¿‡è¯¥è§†é¢‘")
                    del pvid, vid_behaviors
                    gc.collect()
                    continue
            except Exception as e:
                print(f"\tâš ï¸�  è¯»å�–æ ‡æ³¨æ–‡ä»¶å¤±è´¥ï¼š{str(e)}")
                del pvid, vid_behaviors
                gc.collect()
                continue
        
        # ---------------------- ç”Ÿæˆ�å�•é¼ è¡Œä¸ºæ•°æ�® ----------------------
        single_generated = 0
        if generate_single:
            vid_behaviors_subset = vid_behaviors.query("target == 'self'")
            print(f"\tğŸ”� å�•é¼ è¡Œä¸ºå­�é›†å½¢çŠ¶ï¼š{vid_behaviors_subset.shape}")
            if len(vid_behaviors_subset) > 0:
                for mouse_id_str in np.unique(vid_behaviors_subset.agent):
                    try:
                        mouse_id = int(mouse_id_str[-1])
                        vid_agent_actions = np.unique(vid_behaviors_subset.query("agent == @mouse_id_str").action)
                        print(f"\tğŸ”� å�•é¼  {mouse_id_str} å¯¹åº”çš„è¡Œä¸ºï¼š{vid_agent_actions}")
                        
                        # æ£€æŸ¥å°�é¼ æ•°æ�®æ˜¯å�¦å­˜åœ¨
                        if mouse_id not in pvid.columns.get_level_values('mouse_id'):
                            print(f"\tâš ï¸�  å�•é¼  {mouse_id_str} åœ¨é€�è§†è¡¨ä¸­æ— æ•°æ�®ï¼Œè·³è¿‡")
                            continue
                        
                        single_mouse = pvid.loc[:, mouse_id]
                        assert len(single_mouse) == len(pvid), "å�•é¼ æ•°æ�®é•¿åº¦ä¸�åŒ¹é…�"
                        
                        single_mouse_meta = pd.DataFrame({
                            'video_id': video_id,
                            'agent_id': mouse_id_str,
                            'target_id': 'self',
                            'video_frame': single_mouse.index
                        })
                        
                        if traintest == 'train':
                            single_mouse_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=single_mouse.index)
                            annot_subset = annot.query("(agent_id == @mouse_id) & (target_id == @mouse_id)")
                            for _, annot_row in annot_subset.iterrows():
                                mask = (single_mouse_label.index >= annot_row['start_frame']) & (single_mouse_label.index <= annot_row['stop_frame'])
                                single_mouse_label.loc[mask, annot_row.action] = 1.0
                            yield 'single', single_mouse, single_mouse_meta, single_mouse_label
                        else:
                            yield 'single', single_mouse, single_mouse_meta, vid_agent_actions
                        single_generated += 1
                    except KeyError as e:
                        print(f"\tâš ï¸�  å�•é¼ è¡Œä¸ºå¤„ç�†å¤±è´¥ï¼ˆKeyErrorï¼‰ï¼š{str(e)}")
                        continue
                    except Exception as e:
                        print(f"\tâš ï¸�  å�•é¼ è¡Œä¸ºå¤„ç�†å¤±è´¥ï¼š{str(e)}")
                        continue
        print(f"\tâœ… å�•é¼ æ•°æ�®ç”Ÿæˆ�æ€»æ•°ï¼š{single_generated}")
        
        # ---------------------- ç”Ÿæˆ�å�Œé¼ è¡Œä¸ºæ•°æ�® ----------------------
        pair_generated = 0
        if generate_pair:
            vid_behaviors_subset = vid_behaviors.query("target != 'self'")
            print(f"\tğŸ”� å�Œé¼ è¡Œä¸ºå­�é›†å½¢çŠ¶ï¼š{vid_behaviors_subset.shape}")
            if len(vid_behaviors_subset) > 0:
                mouse_ids = np.unique(pvid.columns.get_level_values('mouse_id'))
                print(f"\tğŸ”� å�¯ç”¨å°�é¼ IDï¼š{mouse_ids}")
                for agent, target in itertools.permutations(mouse_ids, 2):
                    agent_str = f"mouse{agent}"
                    target_str = f"mouse{target}"
                    vid_agent_actions = np.unique(
                        vid_behaviors_subset.query("(agent == @agent_str) & (target == @target_str)").action
                    )
                    print(f"\tğŸ”� å�Œé¼  {agent_str}â†’{target_str} å¯¹åº”çš„è¡Œä¸ºï¼š{vid_agent_actions}")
                    if len(vid_agent_actions) == 0:
                        continue
                    
                    try:
                        mouse_pair = pd.concat([pvid[agent], pvid[target]], axis=1, keys=['A', 'B'])
                        assert len(mouse_pair) == len(pvid), "å�Œé¼ æ•°æ�®é•¿åº¦ä¸�åŒ¹é…�"
                        
                        mouse_pair_meta = pd.DataFrame({
                            'video_id': video_id,
                            'agent_id': agent_str,
                            'target_id': target_str,
                            'video_frame': mouse_pair.index
                        })
                        
                        if traintest == 'train':
                            mouse_pair_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=mouse_pair.index)
                            annot_subset = annot.query("(agent_id == @agent) & (target_id == @target)")
                            for _, annot_row in annot_subset.iterrows():
                                mask = (mouse_pair_label.index >= annot_row['start_frame']) & (mouse_pair_label.index <= annot_row['stop_frame'])
                                mouse_pair_label.loc[mask, annot_row.action] = 1.0
                            yield 'pair', mouse_pair, mouse_pair_meta, mouse_pair_label
                        else:
                            yield 'pair', mouse_pair, mouse_pair_meta, vid_agent_actions
                        pair_generated += 1
                    except KeyError as e:
                        print(f"\tâš ï¸�  å�Œé¼ è¡Œä¸ºå¤„ç�†å¤±è´¥ï¼ˆKeyErrorï¼‰ï¼š{str(e)}")
                        continue
                    except Exception as e:
                        print(f"\tâš ï¸�  å�Œé¼ è¡Œä¸ºå¤„ç�†å¤±è´¥ï¼š{str(e)}")
                        continue
        print(f"\tâœ… å�Œé¼ æ•°æ�®ç”Ÿæˆ�æ€»æ•°ï¼š{pair_generated}")
        
        # æ¸…ç�†ä¸´æ—¶æ•°æ�®
        del pvid, vid_behaviors
        if annot is not None:
            del annot
        gc.collect()


def safe_rolling(series, window, func, min_periods=None):
    if min_periods is None:
        min_periods = max(1, window // 4)
    return series.rolling(window, min_periods=min_periods, center=True).apply(func, raw=True)

def _scale(n_frames_at_30fps, fps, ref=30.0):
    return max(1, int(round(n_frames_at_30fps * float(fps) / ref)))

def _scale_signed(n_frames_at_30fps, fps, ref=30.0):
    if n_frames_at_30fps == 0:
        return 0
    s = 1 if n_frames_at_30fps > 0 else -1
    mag = max(1, int(round(abs(n_frames_at_30fps) * float(fps) / ref)))
    return s * mag

def _fps_from_meta(meta_df, fallback_lookup, default_fps=30.0):
    if 'frames_per_second' in meta_df.columns and pd.notnull(meta_df['frames_per_second']).any():
        return float(meta_df['frames_per_second'].iloc[0])
    vid = meta_df['video_id'].iloc[0]
    return float(fallback_lookup.get(vid, default_fps))

def add_curvature_features(X, center_x, center_y, fps):
    vel_x = center_x.diff()
    vel_y = center_y.diff()
    acc_x = vel_x.diff()
    acc_y = vel_y.diff()

    cross_prod = vel_x * acc_y - vel_y * acc_x
    vel_mag = np.sqrt(vel_x**2 + vel_y**2)
    curvature = np.abs(cross_prod) / (vel_mag**3 + 1e-6)

    for w in [25, 50, 75]:
        ws = _scale(w, fps)
        X[f'curv_mean_{w}'] = curvature.rolling(ws, min_periods=max(1, ws // 5)).mean()

    angle = np.arctan2(vel_y, vel_x)
    angle_change = np.abs(angle.diff())
    w = 30
    ws = _scale(w, fps)
    X[f'turn_rate_{w}'] = angle_change.rolling(ws, min_periods=max(1, ws // 5)).sum()

    return X

def add_multiscale_features(X, center_x, center_y, fps):
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)

    scales = [20, 40, 60, 80]
    for scale in scales:
        ws = _scale(scale, fps)
        if len(speed) >= ws:
            X[f'sp_m{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).mean()
            X[f'sp_s{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).std()

    if len(scales) >= 2 and f'sp_m{scales[0]}' in X.columns and f'sp_m{scales[-1]}' in X.columns:
        X['sp_ratio'] = X[f'sp_m{scales[0]}'] / (X[f'sp_m{scales[-1]}'] + 1e-6)

    return X

def add_state_features(X, center_x, center_y, fps):
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)
    w_ma = _scale(15, fps)
    speed_ma = speed.rolling(w_ma, min_periods=max(1, w_ma // 3)).mean()

    try:
        bins = [-np.inf, 0.5 * fps, 2.0 * fps, 5.0 * fps, np.inf]
        speed_states = pd.cut(speed_ma, bins=bins, labels=[0, 1, 2, 3]).astype(float)

        for window in [20, 40, 60, 80]:
            ws = _scale(window, fps)
            if len(speed_states) >= ws:
                for state in [0, 1, 2, 3]:
                    X[f's{state}_{window}'] = (
                        (speed_states == state).astype(float)
                        .rolling(ws, min_periods=max(1, ws // 5)).mean()
                    )
                state_changes = (speed_states != speed_states.shift(1)).astype(float)
                X[f'trans_{window}'] = state_changes.rolling(ws, min_periods=max(1, ws // 5)).sum()
    except Exception:
        pass

    return X

def add_longrange_features(X, center_x, center_y, fps):
    for window in [30, 60, 120]:
        ws = _scale(window, fps)
        if len(center_x) >= ws:
            X[f'x_ml{window}'] = center_x.rolling(ws, min_periods=max(5, ws // 6)).mean()
            X[f'y_ml{window}'] = center_y.rolling(ws, min_periods=max(5, ws // 6)).mean()

    for span in [30, 60, 120]:
        s = _scale(span, fps)
        X[f'x_e{span}'] = center_x.ewm(span=s, min_periods=1).mean()
        X[f'y_e{span}'] = center_y.ewm(span=s, min_periods=1).mean()

    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)  # cm/s
    for window in [30, 60, 120]:
        ws = _scale(window, fps)
        if len(speed) >= ws:
            X[f'sp_pct{window}'] = speed.rolling(ws, min_periods=max(5, ws // 6)).rank(pct=True)

    return X

def add_interaction_features(X, mouse_pair, avail_A, avail_B, fps):
    if 'body_center' not in avail_A or 'body_center' not in avail_B:
        return X

    rel_x = mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x']
    rel_y = mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y']
    rel_dist = np.sqrt(rel_x**2 + rel_y**2)

    A_vx = mouse_pair['A']['body_center']['x'].diff()
    A_vy = mouse_pair['A']['body_center']['y'].diff()
    B_vx = mouse_pair['B']['body_center']['x'].diff()
    B_vy = mouse_pair['B']['body_center']['y'].diff()

    A_lead = (A_vx * rel_x + A_vy * rel_y) / (np.sqrt(A_vx**2 + A_vy**2) * rel_dist + 1e-6)
    B_lead = (B_vx * (-rel_x) + B_vy * (-rel_y)) / (np.sqrt(B_vx**2 + B_vy**2) * rel_dist + 1e-6)

    for window in [30, 60]:
        ws = _scale(window, fps)
        X[f'A_ld{window}'] = A_lead.rolling(ws, min_periods=max(1, ws // 6)).mean()
        X[f'B_ld{window}'] = B_lead.rolling(ws, min_periods=max(1, ws // 6)).mean()

    approach = -rel_dist.diff()
    chase = approach * B_lead
    w = 30
    ws = _scale(w, fps)
    X[f'chase_{w}'] = chase.rolling(ws, min_periods=max(1, ws // 6)).mean()

    for window in [60, 120]:
        ws = _scale(window, fps)
        A_sp = np.sqrt(A_vx**2 + A_vy**2)
        B_sp = np.sqrt(B_vx**2 + B_vy**2)
        X[f'sp_cor{window}'] = A_sp.rolling(ws, min_periods=max(1, ws // 6)).corr(B_sp)

    return X


def transform_single(single_mouse, body_parts_tracked, fps):
    available_body_parts = single_mouse.columns.get_level_values(0)

    X = pd.DataFrame({
        f"{p1}+{p2}": np.square(single_mouse[p1] - single_mouse[p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.combinations(body_parts_tracked, 2)
        if p1 in available_body_parts and p2 in available_body_parts
    })
    X = X.reindex(columns=[f"{p1}+{p2}" for p1, p2 in itertools.combinations(body_parts_tracked, 2)], copy=False)

    if all(p in single_mouse.columns for p in ['ear_left', 'ear_right', 'tail_base']):
        lag = _scale(10, fps)
        shifted = single_mouse[['ear_left', 'ear_right', 'tail_base']].shift(lag)
        speeds = pd.DataFrame({
            'sp_lf': np.square(single_mouse['ear_left'] - shifted['ear_left']).sum(axis=1, skipna=False),
            'sp_rt': np.square(single_mouse['ear_right'] - shifted['ear_right']).sum(axis=1, skipna=False),
            'sp_lf2': np.square(single_mouse['ear_left'] - shifted['tail_base']).sum(axis=1, skipna=False),
            'sp_rt2': np.square(single_mouse['ear_right'] - shifted['tail_base']).sum(axis=1, skipna=False),
        })
        X = pd.concat([X, speeds], axis=1)

    if 'nose+tail_base' in X.columns and 'ear_left+ear_right' in X.columns:
        X['elong'] = X['nose+tail_base'] / (X['ear_left+ear_right'] + 1e-6)

    if all(p in available_body_parts for p in ['nose', 'body_center', 'tail_base']):
        v1 = single_mouse['nose'] - single_mouse['body_center']
        v2 = single_mouse['tail_base'] - single_mouse['body_center']
        X['body_ang'] = (v1['x'] * v2['x'] + v1['y'] * v2['y']) / (
            np.sqrt(v1['x']**2 + v1['y']**2) * np.sqrt(v2['x']**2 + v2['y']**2) + 1e-6)

    if 'body_center' in available_body_parts:
        cx = single_mouse['body_center']['x']
        cy = single_mouse['body_center']['y']

        for w in [5, 15, 30, 60]:
            ws = _scale(w, fps)
            roll = dict(min_periods=1, center=True)
            X[f'cx_m{w}'] = cx.rolling(ws, **roll).mean()
            X[f'cy_m{w}'] = cy.rolling(ws, **roll).mean()
            X[f'cx_s{w}'] = cx.rolling(ws, **roll).std()
            X[f'cy_s{w}'] = cy.rolling(ws, **roll).std()
            X[f'x_rng{w}'] = cx.rolling(ws, **roll).max() - cx.rolling(ws, **roll).min()
            X[f'y_rng{w}'] = cy.rolling(ws, **roll).max() - cy.rolling(ws, **roll).min()
            X[f'disp{w}'] = np.sqrt(cx.diff().rolling(ws, min_periods=1).sum()**2 +
                                     cy.diff().rolling(ws, min_periods=1).sum()**2)
            X[f'act{w}'] = np.sqrt(cx.diff().rolling(ws, min_periods=1).var() +
                                   cy.diff().rolling(ws, min_periods=1).var())

        X = add_curvature_features(X, cx, cy, fps)
        X = add_multiscale_features(X, cx, cy, fps)
        X = add_state_features(X, cx, cy, fps)
        X = add_longrange_features(X, cx, cy, fps)

    if all(p in available_body_parts for p in ['nose', 'tail_base']):
        nt_dist = np.sqrt((single_mouse['nose']['x'] - single_mouse['tail_base']['x'])**2 +
                          (single_mouse['nose']['y'] - single_mouse['tail_base']['y'])**2)
        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            X[f'nt_lg{lag}'] = nt_dist.shift(l)
            X[f'nt_df{lag}'] = nt_dist - nt_dist.shift(l)

    if all(p in available_body_parts for p in ['ear_left', 'ear_right']):
        ear_d = np.sqrt((single_mouse['ear_left']['x'] - single_mouse['ear_right']['x'])**2 +
                        (single_mouse['ear_left']['y'] - single_mouse['ear_right']['y'])**2)
        for off in [-30, -20, -10, 10, 20, 30]:
            o = _scale_signed(off, fps)
            X[f'ear_o{off}'] = ear_d.shift(-o)
        w = _scale(30, fps)
        X['ear_con'] = ear_d.rolling(w, min_periods=1, center=True).std() / \
                       (ear_d.rolling(w, min_periods=1, center=True).mean() + 1e-6)

    return X.astype(np.float32, copy=False)

def transform_pair(mouse_pair, body_parts_tracked, fps):
    avail_A = mouse_pair['A'].columns.get_level_values(0)
    avail_B = mouse_pair['B'].columns.get_level_values(0)

    X = pd.DataFrame({
        f"12+{p1}+{p2}": np.square(mouse_pair['A'][p1] - mouse_pair['B'][p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.product(body_parts_tracked, repeat=2)
        if p1 in avail_A and p2 in avail_B
    })
    X = X.reindex(columns=[f"12+{p1}+{p2}" for p1, p2 in itertools.product(body_parts_tracked, repeat=2)], copy=False)

    if ('A', 'ear_left') in mouse_pair.columns and ('B', 'ear_left') in mouse_pair.columns:
        lag = _scale(10, fps)
        shA = mouse_pair['A']['ear_left'].shift(lag)
        shB = mouse_pair['B']['ear_left'].shift(lag)
        speeds = pd.DataFrame({
            'sp_A': np.square(mouse_pair['A']['ear_left'] - shA).sum(axis=1, skipna=False),
            'sp_AB': np.square(mouse_pair['A']['ear_left'] - shB).sum(axis=1, skipna=False),
            'sp_B': np.square(mouse_pair['B']['ear_left'] - shB).sum(axis=1, skipna=False),
        })
        X = pd.concat([X, speeds], axis=1)

    if 'nose+tail_base' in X.columns and 'ear_left+ear_right' in X.columns:
        X['elong'] = X['nose+tail_base'] / (X['ear_left+ear_right'] + 1e-6)

    if all(p in avail_A for p in ['nose', 'tail_base']) and all(p in avail_B for p in ['nose', 'tail_base']):
        dir_A = mouse_pair['A']['nose'] - mouse_pair['A']['tail_base']
        dir_B = mouse_pair['B']['nose'] - mouse_pair['B']['tail_base']
        X['rel_ori'] = (dir_A['x'] * dir_B['x'] + dir_A['y'] * dir_B['y']) / (
            np.sqrt(dir_A['x']**2 + dir_A['y']**2) * np.sqrt(dir_B['x']**2 + dir_B['y']**2) + 1e-6)

    if all(p in avail_A for p in ['nose']) and all(p in avail_B for p in ['nose']):
        cur = np.square(mouse_pair['A']['nose'] - mouse_pair['B']['nose']).sum(axis=1, skipna=False)
        lag = _scale(10, fps)
        shA_n = mouse_pair['A']['nose'].shift(lag)
        shB_n = mouse_pair['B']['nose'].shift(lag)
        past = np.square(shA_n - shB_n).sum(axis=1, skipna=False)
        X['appr'] = cur - past

    if 'body_center' in avail_A and 'body_center' in avail_B:
        cd = np.sqrt((mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x'])**2 +
                     (mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y'])**2)
        X['v_cls'] = (cd < 5.0).astype(float)
        X['cls']   = ((cd >= 5.0) & (cd < 15.0)).astype(float)
        X['med']   = ((cd >= 15.0) & (cd < 30.0)).astype(float)
        X['far']   = (cd >= 30.0).astype(float)

    if 'body_center' in avail_A and 'body_center' in avail_B:
        cd_full = np.square(mouse_pair['A']['body_center'] - mouse_pair['B']['body_center']).sum(axis=1, skipna=False)

        for w in [5, 15, 30, 60]:
            ws = _scale(w, fps)
            roll = dict(min_periods=1, center=True)
            X[f'd_m{w}']  = cd_full.rolling(ws, **roll).mean()
            X[f'd_s{w}']  = cd_full.rolling(ws, **roll).std()
            X[f'd_mn{w}'] = cd_full.rolling(ws, **roll).min()
            X[f'd_mx{w}'] = cd_full.rolling(ws, **roll).max()

            d_var = cd_full.rolling(ws, **roll).var()
            X[f'int{w}'] = 1 / (1 + d_var)

            Axd = mouse_pair['A']['body_center']['x'].diff()
            Ayd = mouse_pair['A']['body_center']['y'].diff()
            Bxd = mouse_pair['B']['body_center']['x'].diff()
            Byd = mouse_pair['B']['body_center']['y'].diff()
            coord = Axd * Bxd + Ayd * Byd
            X[f'co_m{w}'] = coord.rolling(ws, **roll).mean()
            X[f'co_s{w}'] = coord.rolling(ws, **roll).std()

    if 'nose' in avail_A and 'nose' in avail_B:
        nn = np.sqrt((mouse_pair['A']['nose']['x'] - mouse_pair['B']['nose']['x'])**2 +
                     (mouse_pair['A']['nose']['y'] - mouse_pair['B']['nose']['y'])**2)
        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            X[f'nn_lg{lag}']  = nn.shift(l)
            X[f'nn_ch{lag}']  = nn - nn.shift(l)
            is_cl = (nn < 10.0).astype(float)
            X[f'cl_ps{lag}']  = is_cl.rolling(l, min_periods=1).mean()

    if 'body_center' in avail_A and 'body_center' in avail_B:
        Avx = mouse_pair['A']['body_center']['x'].diff()
        Avy = mouse_pair['A']['body_center']['y'].diff()
        Bvx = mouse_pair['B']['body_center']['x'].diff()
        Bvy = mouse_pair['B']['body_center']['y'].diff()
        val = (Avx * Bvx + Avy * Bvy) / (np.sqrt(Avx**2 + Avy**2) * np.sqrt(Bvx**2 + Bvy**2) + 1e-6)

        for off in [-30, -20, -10, 0, 10, 20, 30]:
            o = _scale_signed(off, fps)
            X[f'va_{off}'] = val.shift(-o)

        w = _scale(30, fps)
        X['int_con'] = cd_full.rolling(w, min_periods=1, center=True).std() / \
                       (cd_full.rolling(w, min_periods=1, center=True).mean() + 1e-6)

        X = add_interaction_features(X, mouse_pair, avail_A, avail_B, fps)

    return X.astype(np.float32, copy=False)


import numpy as np
import pandas as pd
import json

def robustify(submission, dataset, traintest, traintest_directory=None):
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
    
    old_submission = submission.copy()
    
    # --------------------------
    # æ ¸å¿ƒä¿®å¤�ï¼šå°† start_frame/stop_frame è½¬ä¸ºæ•´æ•°
    # --------------------------
    # å¤„ç�†å­—ç¬¦ä¸²è½¬æ•´æ•°ï¼ˆå…¼å®¹ "123" è¿™ç±»å­—ç¬¦ä¸²ï¼Œæ— æ•ˆå€¼è½¬ä¸º NaN å��å¡«å……ä¸º 0ï¼‰
    submission['start_frame'] = pd.to_numeric(submission['start_frame'], errors='coerce').fillna(0).astype(int)
    submission['stop_frame'] = pd.to_numeric(submission['stop_frame'], errors='coerce').fillna(0).astype(int)
    
    # è¿‡æ»¤ start >= stop çš„æ— æ•ˆè¡Œï¼ˆç�°åœ¨æ˜¯æ•´æ•°æ¯”è¾ƒï¼Œæ— ç±»å�‹é”™è¯¯ï¼‰
    submission = submission[submission.start_frame < submission.stop_frame]
    if len(submission) != len(old_submission):
        print("ERROR: Dropped frames with start >= stop")
    
    old_submission = submission.copy()
    group_list = []
    # æŒ‰ video_id + agent_id + target_id åˆ†ç»„ï¼Œé�¿å…�è·¨ä¸»ä½“çš„å¸§å†²çª�
    for _, group in submission.groupby(['video_id', 'agent_id', 'target_id']):
        group = group.sort_values('start_frame')  # æŒ‰å¼€å§‹å¸§æ�’åº�
        mask = np.ones(len(group), dtype=bool)
        last_stop_frame = 0
        for i, (_, row) in enumerate(group.iterrows()):
            # ç�°åœ¨ start_frame æ˜¯æ•´æ•°ï¼Œå�¯æ­£å¸¸æ¯”è¾ƒ
            if row['start_frame'] < last_stop_frame:
                mask[i] = False  # è¿‡æ»¤é‡�å� çš„åŠ¨ä½œï¼ˆå½“å‰�å¼€å§‹å¸§ < ä¸Šä¸€ä¸ªç»“æ�Ÿå¸§ï¼‰
            else:
                last_stop_frame = row['stop_frame']  # æ›´æ–°ä¸Šä¸€ä¸ªåŠ¨ä½œçš„ç»“æ�Ÿå¸§
        group_list.append(group[mask])
        
    submission = pd.concat(group_list)
    
    if len(submission) != len(old_submission):
        print("ERROR: Dropped duplicate frames")
        
    s_list = []
    # å¤„ç�†æ— é¢„æµ‹ç»“æ�œçš„è§†é¢‘ï¼Œå¡«å……é»˜è®¤åŠ¨ä½œå¸§
    for idx, row in dataset.iterrows():
        lab_id = row['lab_id']
        if lab_id.startswith('MABe22'):
            continue
        
        video_id = row['video_id']
        if (submission.video_id == video_id).any():
            continue  # å·²æœ‰é¢„æµ‹ç»“æ�œï¼Œè·³è¿‡
        
        if type(row.behaviors_labeled) != str:
            continue

        print(f"Video {video_id} has no predictions.")
        
        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
    
        vid_behaviors = json.loads(row['behaviors_labeled'])
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])
    
        start_frame = vid.video_frame.min()
        stop_frame = vid.video_frame.max() + 1
    
        # æŒ‰ agent + target åˆ†ç»„ï¼Œå�‡åŒ€åˆ†é…�åŠ¨ä½œå¸§
        for (agent, target), actions in vid_behaviors.groupby(['agent', 'target']):
            batch_length = int(np.ceil((stop_frame - start_frame) / len(actions)))
            for i, (_, action_row) in enumerate(actions.iterrows()):
                batch_start = start_frame + i * batch_length
                batch_stop = min(batch_start + batch_length, stop_frame)
                s_list.append((video_id, agent, target, action_row['action'], batch_start, batch_stop))

    if len(s_list) > 0:
        # æ–°å¢�çš„å¡«å……è¡Œï¼Œstart_frame/stop_frame æœ¬èº«æ˜¯æ•´æ•°ï¼Œæ— éœ€è½¬æ�¢
        submission = pd.concat([
            submission,
            pd.DataFrame(s_list, columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
        ])
        print("ERROR: Filled empty videos")

    submission = submission.reset_index(drop=True)
    
    return submission


def predict_multiclass(pred, meta, thresholds):
    """
    ç”Ÿæˆ�æ¯”èµ›è¦�æ±‚çš„å¸§åŒºé—´é¢„æµ‹ç»“æ�œï¼ˆä¿�ç•™å�Ÿå§‹è¡Œä¸ºå�˜åŒ–æ£€æµ‹é€»è¾‘ï¼‰
    è¾“å‡ºåˆ—ï¼švideo_id, agent_id, target_id, action, start_frame, stop_frame
    """
    import pandas as pd
    import numpy as np
    
    # 1. å�–æ¦‚ç�‡æœ€å¤§çš„è¡Œä¸ºå’Œå¯¹åº”æ¦‚ç�‡
    ama = np.argmax(pred.values, axis=1)
    max_proba = pred.max(axis=1).values

    # 2. è¡Œä¸ºé˜ˆå€¼åŒ¹é…�
    threshold_array = np.array([thresholds.get(col, 0.27) for col in pred.columns])
    action_thresholds = threshold_array[ama]

    # 3. è¿‡æ»¤æ¦‚ç�‡ä½�äº�é˜ˆå€¼çš„è¡Œä¸ºï¼ˆæ ‡è®°ä¸º-1ï¼‰
    ama = np.where(max_proba >= action_thresholds, ama, -1)
    ama = pd.Series(ama, index=meta['video_frame'].values)  # ç´¢å¼•=å�Ÿå§‹å¸§å�·
    
    # 4. æ£€æµ‹è¡Œä¸ºå�˜åŒ–ç‚¹ï¼ˆå’Œä¸Šä¸€å¸§ä¸�å�Œå�³ä¸ºå�˜åŒ–ï¼‰
    changes_mask = (ama != ama.shift(1)).values
    ama_changes = ama[changes_mask]
    meta_changes = meta[changes_mask].reset_index(drop=True)  # é‡�ç½®ç´¢å¼•é�¿å…�å¯¹é½�é”™è¯¯
    
    # 5. è¿‡æ»¤æ— è¡Œä¸ºçš„å�˜åŒ–ç‚¹ï¼Œæœ€å��ä¸€ä¸ªå�˜åŒ–ç‚¹è®¾ä¸ºFalseï¼ˆé�¿å…�ç´¢å¼•è¶Šç•Œï¼‰
    mask = ama_changes.values >= 0
    if len(mask) > 0:
        mask[-1] = False
    
    # 6. ç”Ÿæˆ�æ ¸å¿ƒé¢„æµ‹ç»“æ�œï¼ˆæ¯”èµ›è¦�æ±‚çš„5åˆ—ï¼‰
    submission_part = pd.DataFrame()
    if np.any(mask):
        submission_part = pd.DataFrame({
            'video_id': meta_changes.loc[mask, 'video_id'].values.astype(str),  # ç¡®ä¿�ä¸ºå­—ç¬¦ä¸²
            'agent_id': meta_changes.loc[mask, 'agent_id'].values,
            'target_id': meta_changes.loc[mask, 'target_id'].values,
            'action': pred.columns[ama_changes[mask].values],
            'start_frame': ama_changes[mask].index.values.astype(int),  # å¸§å�·ä¸ºæ•´æ•°
            'stop_frame': ama_changes.index[1:][mask[:-1]].astype(int)
        })
        
        # 7. ä¿®æ­£è·¨è§†é¢‘/agent/targetçš„ç»“æ�Ÿå¸§
        stop_meta = meta_changes.iloc[1:].reset_index(drop=True)
        stop_video_id = stop_meta.loc[mask[:-1], 'video_id'].values.astype(str)
        stop_agent_id = stop_meta.loc[mask[:-1], 'agent_id'].values
        stop_target_id = stop_meta.loc[mask[:-1], 'target_id'].values
        
        for i in range(len(submission_part)):
            if (stop_video_id[i] != submission_part['video_id'].iloc[i] or
                stop_agent_id[i] != submission_part['agent_id'].iloc[i] or
                stop_target_id[i] != submission_part['target_id'].iloc[i]):
                # ä¿®æ­£ä¸ºå½“å‰�è§†é¢‘çš„æœ€å¤§å¸§+1
                max_frame = meta.query("video_id == @submission_part['video_id'].iloc[i]")['video_frame'].max()
                submission_part.iloc[i, submission_part.columns.get_loc('stop_frame')] = int(max_frame) + 1
    
    print(f"\t\tâœ… predict_multiclass ç”Ÿæˆ� {len(submission_part)} è¡ŒåŒºé—´ç»“æ�œ")
    return submission_part


def tune_threshold(oof_action, y_action):
    def objective(trial):
        threshold = trial.suggest_float("threshold", 0, 1, step=0.01)
        return f1_score(y_action, (oof_action >= threshold), zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1000, n_jobs=-1)
    return study.best_params["threshold"]


import pandas as pd
import numpy as np
import gc
import os
import warnings
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import joblib
import torch

def cross_validate_classifier(X, label, meta, body_parts_tracked_str, section):
    """
    æœ€ç»ˆé€‚é…�ç‰ˆï¼šå®Œå…¨åŒ¹é…�ä½ çš„ CFG é…�ç½®ï¼Œæ— ä»»ä½•å�‚æ•°å†²çª�
    æ ¸å¿ƒï¼šfit ä¸­ä»…ä¼  eval_set å’Œ verboseï¼Œåˆ é™¤æ‰€æœ‰æ�„é€ å‡½æ•°å·²æœ‰çš„å�‚æ•°
    """
    # åˆ�å§‹åŒ– OOF ç»“æ�œï¼ˆç”¨ meta å�Ÿå§‹ç´¢å¼•å¯¹é½�ï¼Œé�¿å…�å¸§é”™ä½�ï¼‰
    oof = pd.DataFrame(index=meta.index)
    f1_list = []
    submission_list = []
    thresholds = {}

    # é��å�†æ¯�ä¸ªåŠ¨ä½œï¼ˆå¤šæ ‡ç­¾åˆ†ç±»ï¼‰
    for action in label.columns:
        # -----------------------
        # 1. æ•°æ�®ç­›é€‰ï¼šè¿‡æ»¤ç©ºæ ‡ç­¾å’Œæ— æ•ˆæ•°æ�®
        # -----------------------
        action_mask = ~label[action].isna().values  # é��ç©ºæ ‡ç­¾æ�©ç �
        y_action = label[action][action_mask].values.astype(int)  # åŠ¨ä½œæ ‡ç­¾ï¼ˆ0/1ï¼‰
        X_action = X[action_mask]  # åŠ¨ä½œç‰¹å¾�
        groups_action = meta.video_id[action_mask]  # åˆ†ç»„ï¼ˆæŒ‰è§†é¢‘ IDï¼Œé�¿å…�å�Œè§†é¢‘è·¨æŠ˜ï¼‰

        # è‡³å°‘éœ€è¦� 2 ä¸ªä¸�å�Œè§†é¢‘æ‰�èƒ½å�š CVï¼ˆå�¦åˆ™è®­ç»ƒæ— æ„�ä¹‰ï¼‰
        if len(np.unique(groups_action)) < 2:
            print(f"\tè·³è¿‡åŠ¨ä½œ {action}ï¼šè§†é¢‘æ•°é‡�ä¸�è¶³ 2 ä¸ª")
            continue

        # é��å…¨è´Ÿæ ·æœ¬ï¼ˆå…¨è´Ÿæ ·æœ¬ç›´æ�¥ F1=0ï¼Œæ— éœ€è®­ç»ƒï¼‰
        if not (y_action == 0).all():
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)  # å¿½ç•¥å†—ä½™è­¦å‘Š

                    # ===========================
                    # 2. CV é…�ç½®ï¼š2 æŠ˜æ��é€Ÿï¼ˆå¹³è¡¡é€Ÿåº¦ä¸�ç¨³å®šæ€§ï¼‰
                    # ===========================
                    n_splits = min(2, len(np.unique(groups_action)))  # æœ€å¤š 2 æŠ˜ï¼ˆè¦†ç›– CFG çš„ 3 æŠ˜ï¼Œä¼˜å…ˆæ��é€Ÿï¼‰
                    cv = GroupKFold(n_splits=n_splits)
                    oof_action = np.zeros(len(y_action))  # å­˜å‚¨å½“å‰�åŠ¨ä½œçš„ OOF é¢„æµ‹
                    preds = []  # å­˜å‚¨æ¯�æŠ˜æ¨¡å�‹ï¼ˆXGB+CatBoostï¼‰

                    # ===========================
                    # 3. å¤šæŠ˜è®­ç»ƒï¼šXGB æŠ˜çº§è½®è®­ + CatBoost å�Œ GPU å¹¶è¡Œ
                    # ===========================
                    for fold, (train_idx, valid_idx) in enumerate(cv.split(X_action, y_action, groups_action)):
                        print(f"\tåŠ¨ä½œ {action} - æŠ˜ {fold+1}/{n_splits} è®­ç»ƒä¸­...")

                        # -----------------------
                        # XGBoostï¼šæŠ˜çº§è½®è®­ï¼ˆå�Œ T4 äº¤æ›¿ä½¿ç”¨ï¼‰
                        # -----------------------
                        current_xgb_params = CFG.xgb_params.copy()
                        current_xgb_params["gpu_id"] = fold % 2  # æŠ˜ 0â†’GPU0ï¼ŒæŠ˜1â†’GPU1ï¼ˆè¦†ç›– CFG çš„ gpu_id=0ï¼‰
                        xgb_model = XGBClassifier(**current_xgb_params)
                        
                        # å…³é”®ä¿®æ”¹ï¼šfit ä¸­å�ªä¼  eval_set å’Œ verboseï¼Œåˆ é™¤æ‰€æœ‰å…¶ä»–å�‚æ•°ï¼�
                        # å�Ÿå› ï¼ševal_metricã€�early_stopping_rounds å·²åœ¨ CFG.xgb_params ä¸­å®šä¹‰ï¼ˆæ•°å€¼ 30ï¼‰
                        xgb_model.fit(
                            X_action.iloc[train_idx], y_action[train_idx],
                            eval_set=[(X_action.iloc[valid_idx], y_action[valid_idx])],
                            verbose=False  # ä»…ä¿�ç•™ 2 ä¸ªå¿…è¦�å�‚æ•°ï¼Œå½»åº•æ�œç»�é‡�å¤�
                        )
                        
                        # é¢„æµ‹ï¼ˆæ—§ç‰ˆæœ¬å…¼å®¹ï¼šç”¨ iteration_range æ›¿ä»£ ntree_limitï¼‰
                        if hasattr(xgb_model, "best_iteration") and xgb_model.best_iteration > 0:
                            xgb_pred = xgb_model.predict_proba(
                                X_action.iloc[valid_idx],
                                iteration_range=(0, xgb_model.best_iteration)  # ä»…ç”¨æœ€ä¼˜è¿­ä»£
                            )[:, 1]
                        else:
                            xgb_pred = xgb_model.predict_proba(X_action.iloc[valid_idx])[:, 1]

                        # -----------------------
                        # CatBoostï¼šå�Œ GPU å¹¶è¡Œè®­ç»ƒï¼ˆæ— éœ€æ‰‹åŠ¨åˆ‡æ�¢ï¼Œç›´æ�¥ç”¨ CFG å�‚æ•°ï¼‰
                        # -----------------------
                        cat_model = CatBoostClassifier(**CFG.cat_params)
                        cat_model.fit(
                            X_action.iloc[train_idx], y_action[train_idx],
                            eval_set=(X_action.iloc[valid_idx], y_action[valid_idx]),
                            verbose=False
                        )
                        
                        # é¢„æµ‹ï¼ˆç”¨æ—©å�œå��çš„æœ€ä¼˜è¿­ä»£ï¼‰
                        if hasattr(cat_model, "best_iteration") and cat_model.best_iteration > 0:
                            cat_pred = cat_model.predict_proba(
                                X_action.iloc[valid_idx],
                                ntree_end=cat_model.best_iteration
                            )[:, 1]
                        else:
                            cat_pred = cat_model.predict_proba(X_action.iloc[valid_idx])[:, 1]

                        # -----------------------
                        # æ¨¡å�‹è��å�ˆï¼šæ¦‚ç�‡å¹³å�‡ï¼ˆç®€å�•é«˜æ•ˆï¼‰
                        # -----------------------
                        fold_pred = (xgb_pred + cat_pred) / 2
                        oof_action[valid_idx] = fold_pred  # å›�å¡« OOF ç»“æ�œ

                        # ä¿�å­˜å½“å‰�æŠ˜æ¨¡å�‹ï¼ˆå��ç»­æ��äº¤ç”¨ï¼‰
                        preds.append((xgb_model, cat_model))

                        # æ¸…ç�†å½“å‰�æŠ˜èµ„æº�ï¼ˆé�¿å…�æ˜¾å­˜/å†…å­˜å †ç§¯ï¼‰
                        del xgb_model, cat_model, xgb_pred, cat_pred, fold_pred
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()  # æ¸…ç�† PyTorch æ˜¾å­˜æ®‹ç•™

                    # ===========================
                    # 4. é˜ˆå€¼ä¼˜åŒ–ï¼šå¯»æ‰¾æœ€ä¼˜ F1 é˜ˆå€¼ï¼ˆé�¿å…�é»˜è®¤ 0.5 å��è§�ï¼‰
                    # ===========================
                    threshold = tune_threshold(oof_action, y_action)  # è‡ªå®šä¹‰é˜ˆå€¼ä¼˜åŒ–å‡½æ•°
                    thresholds[action] = threshold

                    # è®¡ç®— F1 åˆ†æ•°ï¼ˆé›¶é™¤ä¿�æŠ¤ï¼‰
                    f1 = f1_score(y_action, (oof_action >= threshold), zero_division=0)
                    f1_list.append((body_parts_tracked_str, action, f1))
                    print(f"\tåŠ¨ä½œ {action} - F1: {f1:.4f} | æœ€ä¼˜é˜ˆå€¼: {threshold:.2f}")

                    # ===========================
                    # 5. æ¨¡å�‹ä¿�å­˜ï¼šæŒ‰ sectionâ†’action åˆ†å±‚å­˜å‚¨ï¼ˆå��ç»­æ��äº¤å¤�ç”¨ï¼‰
                    # ===========================
                    save_dir = f"{CFG.model_path}/{CFG.model_name}/{section}/{action}"
                    os.makedirs(save_dir, exist_ok=True)
                    joblib.dump(preds, f"{save_dir}/xgb_cat_trainer.pkl")  # ä¿�å­˜æ‰€æœ‰æŠ˜æ¨¡å�‹
                    joblib.dump(threshold, f"{save_dir}/best_threshold.pkl")  # ä¿�å­˜æœ€ä¼˜é˜ˆå€¼

            except Exception as e:
                # è®­ç»ƒå¤±è´¥æ—¶ï¼ŒOOF å¡«å…… 0ï¼Œé�¿å…�ä¸­æ–­æ•´ä½“æµ�ç¨‹
                print(f"\tåŠ¨ä½œ {action} è®­ç»ƒå¤±è´¥: {str(e)}")
                oof_action = np.zeros(len(y_action))
        else:
            # å…¨è´Ÿæ ·æœ¬ï¼šF1=0ï¼Œé˜ˆå€¼=0
            oof_action = np.zeros(len(y_action))
            thresholds[action] = 0.0
            print(f"\tåŠ¨ä½œ {action} - F1: 0.0000 | å…¨è´Ÿæ ·æœ¬")

        # -----------------------
        # 6. OOF å›�å¡«ï¼šå°†å½“å‰�åŠ¨ä½œçš„ OOF ç»“æ�œå�ˆå¹¶åˆ°å…¨å±€
        # -----------------------
        oof_column = pd.Series(0.0, index=meta.index)  # å…¨å±€åˆ�å§‹åŒ–
        oof_column[action_mask] = oof_action  # æŒ‰æ�©ç �å›�å¡«å½“å‰�åŠ¨ä½œç»“æ�œ
        oof[action] = oof_column.values

        # æ¸…ç�†å½“å‰�åŠ¨ä½œèµ„æº�
        del oof_action, y_action, X_action, groups_action, action_mask, oof_column
        gc.collect()

    # -----------------------
    # 7. ç”Ÿæˆ�æ��äº¤ç»“æ�œï¼ˆæŒ‰ç«�èµ›æ ¼å¼�è¦�æ±‚ï¼‰
    # -----------------------
    submission_part = predict_multiclass(oof, meta, thresholds)  # è‡ªå®šä¹‰å¤šæ ‡ç­¾é¢„æµ‹å‡½æ•°
    submission_list.append(submission_part)

    return submission_list, f1_list, thresholds


def submit(body_parts_tracked_str, switch_tr, section, thresholds):
    """
    å®Œæ•´é¢„æµ‹æµ�ç¨‹ï¼šæ•°æ�®ç”Ÿæˆ�â†’ç‰¹å¾�è½¬æ�¢â†’æ¨¡å�‹é¢„æµ‹â†’åŒºé—´ç”Ÿæˆ�
    è¾“å‡ºç¬¦å�ˆæ¯”èµ›è¦�æ±‚çš„é¢„æµ‹æ‰¹æ¬¡åˆ—è¡¨
    """
    import glob
    import joblib
    import pandas as pd
    import numpy as np
    import gc
    
    # å…¨å±€å�˜é‡�ä¾�èµ–ï¼ˆç¡®ä¿�æ��å‰�å®šä¹‰ï¼‰
    global drop_body_parts, CFG, test, generate_mouse_data, _fps_from_meta
    global transform_single, transform_pair, predict_multiclass
    
    # 1. è§£æ��å¹¶è¿‡æ»¤èº«ä½“éƒ¨ä½�
    body_parts_tracked = json.loads(body_parts_tracked_str)
    if 'drop_body_parts' in globals() and len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]

    # 2. ç­›é€‰å½“å‰�Sectionçš„æµ‹è¯•æ•°æ�®
    test_subset = test[test['body_parts_tracked'] == body_parts_tracked_str]
    if len(test_subset) == 0:
        print(f"\tâš ï¸� å½“å‰�Sectionæ— æµ‹è¯•æ•°æ�®ï¼Œè·³è¿‡")
        return []

    # 3. åˆ�å§‹åŒ–æ•°æ�®ç”Ÿæˆ�å™¨
    generator = generate_mouse_data(
        test_subset,
        'test',
        generate_single=(switch_tr == 'single'),
        generate_pair=(switch_tr == 'pair')
    )

    # 4. æ�„å»ºFPSæŸ¥æ‰¾å­—å…¸
    fps_lookup = (
        test_subset[['video_id', 'frames_per_second']]
        .drop_duplicates('video_id')
        .set_index('video_id')['frames_per_second']
        .to_dict()
    )

    submission_list = []

    # 5. è¿­ä»£é¢„æµ‹
    for switch_te, data_te, meta_te, actions_te in generator:
        assert switch_te == switch_tr, f"æ•°æ�®ç±»å�‹ä¸�åŒ¹é…�ï¼š{switch_te} != {switch_tr}"
        try:
            # 5.1 è�·å�–FPS
            fps_i = _fps_from_meta(meta_te, fps_lookup, default_fps=30.0)

            # 5.2 ç‰¹å¾�è½¬æ�¢
            if switch_te == 'single':
                X_te = transform_single(data_te, body_parts_tracked, fps_i).astype(np.float32)
            else:
                X_te = transform_pair(data_te, body_parts_tracked, fps_i).astype(np.float32)

            del data_te
            gc.collect()

            # 5.3 åˆ�å§‹åŒ–é¢„æµ‹æ¦‚ç�‡DataFrame
            pred_prob = pd.DataFrame(index=meta_te['video_frame'].values)
            print(f"\t\tâœ… åˆ�å§‹åŒ–é¢„æµ‹ç»“æ�œï¼Œå¸§æ•°é‡�ï¼š{len(pred_prob)}")

            # 5.4 é€�ä¸ªè¡Œä¸ºé¢„æµ‹ï¼ˆXGBoost+CatBoostè��å�ˆï¼‰
            for action in actions_te:
                model_files = glob.glob(
                    f"{CFG.model_path}/{CFG.model_name}/{section}/{action}/xgb_cat_trainer.pkl"
                )
                if len(model_files) != 1:
                    print(f"\t\tâš ï¸� æœªæ‰¾åˆ°{action}çš„æ¨¡å�‹æ–‡ä»¶ï¼Œè·³è¿‡")
                    continue

                models = joblib.load(model_files[0])
                if not isinstance(models, list) or len(models) == 0:
                    print(f"\t\tâš ï¸� {action}æ¨¡å�‹æ ¼å¼�é”™è¯¯ï¼Œè·³è¿‡")
                    continue

                # å¤šæŠ˜è��å�ˆ
                fold_probs = []
                for (xgb_model, cat_model) in models:
                    xgb_prob = xgb_model.predict_proba(X_te)[:, 1]
                    cat_prob = cat_model.predict_proba(X_te)[:, 1]
                    fold_probs.append((xgb_prob + cat_prob) / 2)

                pred_prob[action] = np.mean(fold_probs, axis=0)
                del models, fold_probs
                gc.collect()
                print(f"\t\tâœ… å®Œæˆ�{action}é¢„æµ‹")

            del X_te
            gc.collect()

            # 5.5 ç”Ÿæˆ�åŒºé—´ç»“æ�œ
            if pred_prob.shape[1] > 0:
                submission_part = predict_multiclass(pred_prob, meta_te, thresholds)
                if len(submission_part) > 0:
                    submission_list.append(submission_part)
                    print(f"\t\tâœ… æ–°å¢�é¢„æµ‹æ‰¹æ¬¡ï¼Œè¡Œæ•°ï¼š{len(submission_part)}")

        except Exception as e:
            print(f"\tâš ï¸� å¤„ç�†{switch_te}æ•°æ�®å‡ºé”™ï¼š{type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            if 'data_te' in locals():
                del data_te
            gc.collect()
            continue

    print(f"\tâœ… submitæ‰§è¡Œå®Œæˆ�ï¼Œç”Ÿæˆ�{len(submission_list)}ä¸ªæœ‰æ•ˆæ‰¹æ¬¡")
    return submission_list


import os
import joblib

# å…¨å±€å®šä¹‰é˜ˆå€¼å­—å…¸ï¼ˆç¡®ä¿�ä½œç”¨åŸŸæ­£ç¡®ï¼‰
thresholds = {
    "single": {},
    "pair": {}
}

if CFG.mode != "validate":
    # æ ¹è·¯å¾„ï¼šmodels/xgb_cat/
    thresholds_root = f"{CFG.model_path}/{CFG.model_name}"
    default_threshold = 0.5  # å…œåº•é»˜è®¤å€¼

    # å…³é”®ï¼šè¡Œä¸ºç±»å�‹æ˜ å°„è¡¨ï¼ˆæ ¹æ�® MABe æŒ‘æˆ˜çš„è¡Œä¸ºå®šä¹‰åˆ†ç±»ï¼‰
    # æ‰‹åŠ¨å°†æ‰€æœ‰è¡Œä¸ºåˆ’åˆ†ä¸º singleï¼ˆå�•é¼ è¡Œä¸ºï¼‰å’Œ pairï¼ˆå�Œé¼ äº¤äº’è¡Œä¸ºï¼‰
    action_type_map = {
        # å�•é¼ è¡Œä¸ºï¼ˆsingleï¼‰
        "rear": "single",           # ç«™ç«‹
        "selfgroom": "single",      # è‡ªæˆ‘æ¢³ç�†
        "rest": "single",           # ä¼‘æ�¯
        "dig": "single",            # æŒ–æ�˜
        "climb": "single",          # æ”€çˆ¬
        "freeze": "single",         # å†»ç»“
        "biteobject": "single",     # å’¬ç‰©ä½“
        "exploreobject": "single",  # æ�¢ç´¢ç‰©ä½“
        "run": "single",            # å¥”è·‘

        # å�Œé¼ äº¤äº’è¡Œä¸ºï¼ˆpairï¼‰
        "approach": "pair",         # æ�¥è¿‘
        "attack": "pair",           # æ”»å‡»
        "avoid": "pair",            # èº²é�¿
        "chase": "pair",            # è¿½é€�
        "chaseattack": "pair",      # è¿½é€�æ”»å‡»
        "huddle": "pair",           # ä¾�å��
        "reciprocalsniff": "pair",  # äº’ç›¸å—…é—»
        "sniffgenital": "pair",     # å—…ç”Ÿæ®–å™¨
        "dominance": "pair",        # æ”¯é…�
        "escape": "pair",           # é€ƒè·‘
        "follow": "pair",           # è·Ÿéš�
        "sniff": "pair",            # å—…é—»
        "defend": "pair",           # é˜²å¾¡
        "mount": "pair",            # çˆ¬è·¨
        "sniffface": "pair",         # å—…é�¢éƒ¨
        "sniffbody": "pair",        # å—…èº«ä½“
        "attemptmount": "pair",     # å°�è¯•çˆ¬è·¨
        "shepherd": "pair",         # é©±èµ¶
        "allogroom": "pair",        # äº’ç›¸æ¢³ç�†
        "disengage": "pair",        # è„±ç¦»
        "dominancegroom": "pair",   # æ”¯é…�æ€§æ¢³ç�†
        "dominancemount": "pair",   # æ”¯é…�æ€§çˆ¬è·¨
        "ejaculate": "pair",        # å°„ç²¾
        "genitalgroom": "pair",     # ç”Ÿæ®–å™¨æ¢³ç�†
        "intromit": "pair"          # æ�’å…¥
        # å¦‚æœ‰é�—æ¼�è¡Œä¸ºï¼ŒæŒ‰ç›¸å�Œæ ¼å¼�è¡¥å……
    }

    # 1. é��å�†æ‰€æœ‰ Section æ–‡ä»¶å¤¹ï¼ˆ1-9ï¼‰
    for section_name in os.listdir(thresholds_root):
        section_path = os.path.join(thresholds_root, section_name)
        # å�ªå¤„ç�†æ•°å­—å‘½å��çš„ Section ç›®å½•ï¼ˆå¦‚ 1ã€�2ã€�3...ï¼‰
        if not (os.path.isdir(section_path) and section_name.isdigit()):
            continue
        print(f"\n===== å¤„ç�† Section {section_name} =====")

        # 2. é��å�†å½“å‰� Section ä¸‹çš„æ‰€æœ‰è¡Œä¸ºæ–‡ä»¶å¤¹ï¼ˆå¦‚ attackã€�rear ç­‰ï¼‰
        for action_name in os.listdir(section_path):
            action_path = os.path.join(section_path, action_name)
            # è·³è¿‡ submit æ–‡ä»¶å¤¹å’Œé��ç›®å½•æ–‡ä»¶ï¼ˆå�ªå¤„ç�†è¡Œä¸ºæ–‡ä»¶å¤¹ï¼‰
            if not os.path.isdir(action_path) or action_name == "submit":
                continue

            # 3. æ£€æŸ¥å½“å‰�è¡Œä¸ºæ–‡ä»¶å¤¹ä¸‹æ˜¯å�¦æœ‰ best_threshold.pkl
            threshold_file = os.path.join(action_path, "best_threshold.pkl")
            if os.path.exists(threshold_file):
                print(f"æ‰¾åˆ°è¡Œä¸º {action_name} çš„é˜ˆå€¼æ–‡ä»¶ï¼š{threshold_file}")
                try:
                    # è¯»å�–é˜ˆå€¼ï¼ˆå�‡è®¾ pkl ç›´æ�¥å­˜å‚¨æ•°å€¼ï¼Œå¦‚ 0.62ï¼‰
                    threshold_val = joblib.load(threshold_file)
                    # ç¡®ä¿�è¯»å�–åˆ°çš„æ˜¯æ•°å€¼ï¼ˆé˜²æ­¢æ–‡ä»¶æ ¼å¼�é”™è¯¯ï¼‰
                    if not isinstance(threshold_val, (int, float)):
                        raise ValueError("é˜ˆå€¼æ–‡ä»¶å†…å®¹ä¸�æ˜¯æ•°å€¼")

                    # 4. ç¡®å®šè¡Œä¸ºç±»å�‹ï¼ˆsingle/pairï¼‰
                    if action_name in action_type_map:
                        behavior_type = action_type_map[action_name]
                    else:
                        # æœªçŸ¥è¡Œä¸ºé»˜è®¤å½’ä¸º pairï¼ˆå�¯æ ¹æ�®å®�é™…æƒ…å†µè°ƒæ•´ï¼‰
                        behavior_type = "pair"
                        print(f"âš ï¸� æœªçŸ¥è¡Œä¸º {action_name}ï¼Œé»˜è®¤å½’ä¸º pair ç±»å�‹")

                    # 5. å­˜å…¥é˜ˆå€¼å­—å…¸ï¼ˆè‹¥å�Œä¸€è¡Œä¸ºåœ¨å¤šä¸ª Section å‡ºç�°ï¼Œä¿�ç•™æœ€å��ä¸€ä¸ªå€¼ï¼‰
                    thresholds[behavior_type][action_name] = threshold_val
                    print(f"âœ… åŠ è½½æˆ�åŠŸï¼š{behavior_type}/{action_name} = {threshold_val}")

                except Exception as e:
                    print(f"â�Œ è¯»å�– {threshold_file} å¤±è´¥ï¼š{str(e)}ï¼Œä½¿ç”¨é»˜è®¤å€¼ {default_threshold}")
                    # è¯»å�–å¤±è´¥æ—¶ç”¨é»˜è®¤å€¼å…œåº•
                    behavior_type = action_type_map.get(action_name, "pair")
                    thresholds[behavior_type][action_name] = default_threshold

    # 6. æ ¡éªŒå¹¶è¡¥å……æœªåŠ è½½çš„è¡Œä¸ºï¼ˆä½¿ç”¨é»˜è®¤å€¼ï¼‰
    print("\n" + "="*60)
    print("æœ€ç»ˆé˜ˆå€¼æ±‡æ€»ï¼š")
    print(f"single è¡Œä¸ºï¼ˆ{len(thresholds['single'])} ä¸ªï¼‰ï¼š{thresholds['single'].keys()}")
    print(f"pair è¡Œä¸ºï¼ˆ{len(thresholds['pair'])} ä¸ªï¼‰ï¼š{thresholds['pair'].keys()}")

    # ä¸ºæ‰€æœ‰å·²çŸ¥è¡Œä¸ºè®¾ç½®é»˜è®¤å€¼ï¼ˆç¡®ä¿�æ²¡æœ‰é�—æ¼�ï¼‰
    for action, btype in action_type_map.items():
        if action not in thresholds[btype]:
            thresholds[btype][action] = default_threshold
            print(f"âš ï¸� è¡Œä¸º {btype}/{action} æœªæ‰¾åˆ°é˜ˆå€¼æ–‡ä»¶ï¼Œä½¿ç”¨é»˜è®¤å€¼ {default_threshold}")

else:
    print("å½“å‰�ä¸º validate æ¨¡å¼�ï¼Œä½¿ç”¨ç©ºé˜ˆå€¼å­—å…¸")

# ç¡®è®¤å�˜é‡�å�¯ç”¨
print(f"\né˜ˆå€¼å­—å…¸çŠ¶æ€�ï¼šsingle={len(thresholds['single'])} ä¸ªï¼Œpair={len(thresholds['pair'])} ä¸ª")


# ---------------------- ä¸»å¾ªç�¯ï¼šå®Œæ•´å¤„ç�†æµ�ç¨‹ + æ¯”èµ›æ ¼å¼�æ��äº¤ ----------------------
import pandas as pd
import numpy as np
import gc
import torch

# å…¨å±€å�˜é‡�ï¼ˆç¡®ä¿�æ��å‰�å®šä¹‰ï¼‰
# CFG, drop_body_parts, thresholds, action_type_map, body_parts_tracked_list
# test, generate_mouse_data, _fps_from_meta, transform_single, transform_pair
# predict_multiclass, submit, robustify

# è¯»å�–æµ‹è¯•é›†å…ƒæ•°æ�®
test_meta = pd.read_csv("/kaggle/input/MABe-mouse-behavior-detection/test.csv")
print(f"æˆ�åŠŸè¯»å�–æµ‹è¯•é›†å…ƒæ•°æ�®ï¼Œå…±{len(test_meta)}æ�¡è®°å½•")

# åˆ�å§‹åŒ–ç»“æ�œåˆ—è¡¨
submission_list = []

# å¤„ç�†æ‰€æœ‰Section
total_sections = len(body_parts_tracked_list) - 1
print(f"å¼€å§‹å¤„ç�†{total_sections}ä¸ªSectionçš„è§†é¢‘æ•°æ�®...\n")

for section in range(1, len(body_parts_tracked_list)):
    body_parts_tracked_str = body_parts_tracked_list[section]
    section_str = f"{section}"
    
    try:
        # 1. è§£æ��èº«ä½“éƒ¨ä½�
        body_parts_tracked = json.loads(body_parts_tracked_str)
        print(f"===== {section}/{total_sections} å¤„ç�†ä¸­ =====")
        print(f"è¿½è¸ªçš„èº«ä½“éƒ¨ä½�ï¼š{body_parts_tracked}")
        
        # 2. è¿‡æ»¤èº«ä½“éƒ¨ä½�
        if 'drop_body_parts' in globals() and len(body_parts_tracked) > 5:
            body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]
            print(f"è¿‡æ»¤å��ä¿�ç•™ï¼š{body_parts_tracked}")
        
        # 3. ç­›é€‰æµ‹è¯•æ•°æ�®
        data_subset = test_meta[test_meta['body_parts_tracked'].astype(str) == body_parts_tracked_str].copy()
        print(f"\tâœ… ç­›é€‰æµ‹è¯•æ•°æ�®ï¼š{len(data_subset)}æ�¡")
        
        if len(data_subset) == 0:
            print(f"\tâš ï¸� æ— æœ‰æ•ˆæ•°æ�®ï¼Œè·³è¿‡\n")
            continue
        
        # 4. è°ƒç”¨submitå‡½æ•°é¢„æµ‹ï¼ˆå�•é¼ +å�Œé¼ ï¼‰
        # å�•é¼ é¢„æµ‹
        single_sub = submit(
            body_parts_tracked_str=body_parts_tracked_str,
            switch_tr='single',
            section=section,
            thresholds=thresholds["single"].get(section_str, {})
        )
        if single_sub:
            submission_list.extend(single_sub)
            print(f"\tâœ… å�•é¼ é¢„æµ‹æ–°å¢�{len(single_sub)}ä¸ªæ‰¹æ¬¡")
        
        # å�Œé¼ é¢„æµ‹
        pair_sub = submit(
            body_parts_tracked_str=body_parts_tracked_str,
            switch_tr='pair',
            section=section,
            thresholds=thresholds["pair"].get(section_str, {})
        )
        if pair_sub:
            submission_list.extend(pair_sub)
            print(f"\tâœ… å�Œé¼ é¢„æµ‹æ–°å¢�{len(pair_sub)}ä¸ªæ‰¹æ¬¡")
        
        # æ¸…ç�†å†…å­˜
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"===== {section}/{total_sections} å¤„ç�†å®Œæˆ� =====\n")
    
    except Exception as e:
        print(f"\tâ�Œ å¤„ç�†Section{section}å‡ºé”™ï¼š{type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"===== {section}/{total_sections} å¤„ç�†å¤±è´¥ =====\n")
        continue

# ---------------------- æ¯”èµ›è¦�æ±‚çš„æ��äº¤é€»è¾‘ ----------------------
print("="*50)
print("æ‰€æœ‰Sectionå¤„ç�†å®Œæˆ�ï¼�å¼€å§‹ç”Ÿæˆ�æ��äº¤æ–‡ä»¶...")

if CFG.mode == 'submit':
    # å�ˆå¹¶æ‰€æœ‰é¢„æµ‹ç»“æ�œ
    if len(submission_list) > 0:
        submission = pd.concat(submission_list, ignore_index=True)
        print(f"å�ˆå¹¶å��é¢„æµ‹ç»“æ�œè¡Œæ•°ï¼š{len(submission)}")
    else:
        # æ— é¢„æµ‹ç»“æ�œæ—¶ï¼Œç”Ÿæˆ�é»˜è®¤ç©ºç»“æ�œï¼ˆæ¯”èµ›è¦�æ±‚çš„æ ¼å¼�ï¼‰
        submission = pd.DataFrame(
            dict(
                video_id='438887472',  # å­—ç¬¦ä¸²æ ¼å¼�
                agent_id='mouse1',
                target_id='self',
                action='rear',
                start_frame=278,  # æ•´æ•°æ ¼å¼�
                stop_frame=500
            ),
            index=[0]  # ç´¢å¼•ä¸ºæ•´æ•°
        )
        print(f"æ— æœ‰æ•ˆé¢„æµ‹ç»“æ�œï¼Œç”Ÿæˆ�é»˜è®¤æ ¼å¼�å� ä½�ç¬¦")
    
    # æ¯”èµ›è¦�æ±‚çš„robustifyå��å¤„ç�†
    submission_robust = robustify(submission, test, 'test')
    
    # è®¾ç½®ç´¢å¼•å��ç§°ä¸ºrow_idï¼ˆæ¯”èµ›è¦�æ±‚ï¼‰
    submission_robust.index.name = 'row_id'
    
    # ä¿�å­˜æ��äº¤æ–‡ä»¶
    submission_robust.to_csv('submission.csv')
    print(f"\nâœ… æ��äº¤æ–‡ä»¶ç”Ÿæˆ�æˆ�åŠŸï¼�")
    print(f"æ��äº¤æ–‡ä»¶å½¢çŠ¶ï¼š{submission_robust.shape}")
    print(f"æ��äº¤æ–‡ä»¶åˆ—å��ï¼š{list(submission_robust.columns)}")
    print(f"æ��äº¤æ–‡ä»¶å‰�5è¡Œï¼š\n{submission_robust.head()}")

print("="*50)


if CFG.mode == 'validate':  
    submission = pd.concat(submission_list)
    submission_robust = robustify(submission, train, 'train')
    print(f"Competition metric: {score(solution, submission_robust, ''):.4f}")

    f1_df = pd.DataFrame(f1_list, columns=['body_parts_tracked_str', 'action'    , 'binary F1 score'])
    print(f"Mean F1:            {f1_df['binary F1 score'].mean():.4f}")
  
    joblib.dump(thresholds, f"{CFG.model_name}/thresholds.pkl")
    joblib.dump(f1_df, f"{CFG.model_name}/scores.pkl")


if CFG.mode == 'submit':
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
        
    submission_robust = robustify(submission, test, 'test')
    submission_robust.index.name = 'row_id'
    submission_robust.to_csv('submission.csv')
    submission.head()


submission

