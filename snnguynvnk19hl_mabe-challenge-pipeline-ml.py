import pandas as pd
import numpy as np
import json
import itertools
import warnings
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier

# ============= CONFIGURATION =============
validate_or_submit = 'submit' 
verbose = True

# ============= LOAD DATA =============
print("Loading training and test metadata...")
train = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')
train['n_mice'] = 4 - train[['mouse1_strain', 'mouse2_strain', 'mouse3_strain', 'mouse4_strain']].isna().sum(axis=1)
train_without_mabe22 = train.query("~ lab_id.str.startswith('MABe22_')")

test = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv')
body_parts_tracked_list = list(np.unique(train.body_parts_tracked))

print(f"Found {len(body_parts_tracked_list)} unique body part configurations")

# ============= ENHANCED UTILITY CLASSES =============
class TrainOnSubsetClassifier:
    """Wrapper to train classifier on subset for memory efficiency"""
    def __init__(self, clf, max_samples):
        self.clf = clf
        self.max_samples = max_samples
    
    def fit(self, X, y):
        if len(X) > self.max_samples:
            idx = np.random.choice(len(X), self.max_samples, replace=False)
            X_subset = X[idx] if hasattr(X, 'iloc') else X[idx]
            y_subset = y[idx] if hasattr(y, 'iloc') else y[idx]
        else:
            X_subset, y_subset = X, y
        
        self.clf.fit(X_subset, y_subset)
        return self
    
    def predict_proba(self, X):
        return self.clf.predict_proba(X)
    
    @property
    def classes_(self):
        return self.clf.classes_

# ============= ENHANCED FEATURE ENGINEERING =============
def transform_single_enhanced(single_mouse, body_parts_tracked):
    """Enhanced transform for single mouse with temporal features"""
    
    # Original distance features
    X = pd.DataFrame({
        f"{part1}+{part2}": np.square(single_mouse[part1] - single_mouse[part2]).sum(axis=1, skipna=False)
        for part1, part2 in itertools.combinations(body_parts_tracked, 2)
    })
    
    # Enhanced temporal features
    if 'ear_left' in single_mouse.columns and 'ear_right' in single_mouse.columns and 'tail_base' in single_mouse.columns:
        # Speed features (original)
        shifted = single_mouse[['ear_left', 'ear_right', 'tail_base']].shift(10)
        speed_features = pd.DataFrame({
            'speed_left': np.square(single_mouse['ear_left'] - shifted['ear_left']).sum(axis=1, skipna=False),
            'speed_right': np.square(single_mouse['ear_right'] - shifted['ear_right']).sum(axis=1, skipna=False),
            'speed_left2': np.square(single_mouse['ear_left'] - shifted['tail_base']).sum(axis=1, skipna=False),
            'speed_right2': np.square(single_mouse['ear_right'] - shifted['tail_base']).sum(axis=1, skipna=False),
        })
        
        # Acceleration features (new)
        shifted_20 = single_mouse[['ear_left', 'ear_right']].shift(20)
        if not shifted_20.isna().all().all():
            accel_features = pd.DataFrame({
                'accel_left': speed_features['speed_left'] - np.square(shifted['ear_left'] - shifted_20['ear_left']).sum(axis=1, skipna=False),
                'accel_right': speed_features['speed_right'] - np.square(shifted['ear_right'] - shifted_20['ear_right']).sum(axis=1, skipna=False),
            })
            X = pd.concat([X, speed_features, accel_features], axis=1)
        else:
            X = pd.concat([X, speed_features], axis=1)
    
    return X

def transform_pair_enhanced(mouse_pair, body_parts_tracked):
    """Enhanced transform for mouse pairs with social features"""
    
    # Filter body parts for memory efficiency
    drop_body_parts = ['ear_left', 'ear_right',
                      'headpiece_bottombackleft', 'headpiece_bottombackright', 
                      'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
                      'headpiece_topbackleft', 'headpiece_topbackright', 
                      'headpiece_topfrontleft', 'headpiece_topfrontright', 
                      'tail_midpoint']
    
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]
    
    # Original inter-mouse distance features
    X = pd.DataFrame({
        f"12+{part1}+{part2}": np.square(mouse_pair['A'][part1] - mouse_pair['B'][part2]).sum(axis=1, skipna=False)
        for part1, part2 in itertools.product(body_parts_tracked, repeat=2)
    })

    # Enhanced social interaction features
    if 'nose' in body_parts_tracked and 'tail_base' in body_parts_tracked:
        # Face-to-face distance (important for social behaviors)
        X['face_distance'] = np.square(mouse_pair['A']['nose'] - mouse_pair['B']['nose']).sum(axis=1, skipna=False)
        # Following behavior indicator  
        X['following'] = np.square(mouse_pair['A']['nose'] - mouse_pair['B']['tail_base']).sum(axis=1, skipna=False)

    # Original speed features (keep for compatibility)
    if ('A', 'ear_left') in mouse_pair.columns and ('B', 'ear_left') in mouse_pair.columns:
        shifted_A = mouse_pair['A']['ear_left'].shift(10)
        shifted_B = mouse_pair['B']['ear_left'].shift(10)
        X = pd.concat([
            X,
            pd.DataFrame({
                'speed_left_A': np.square(mouse_pair['A']['ear_left'] - shifted_A).sum(axis=1, skipna=False),
                'speed_left_AB': np.square(mouse_pair['A']['ear_left'] - shifted_B).sum(axis=1, skipna=False),
                'speed_left_B': np.square(mouse_pair['B']['ear_left'] - shifted_B).sum(axis=1, skipna=False),
            })
        ], axis=1)
    
    return X

# ============= DYNAMIC THRESHOLD OPTIMIZATION =============
def find_optimal_thresholds(oof_predictions, labels, default_threshold=0.27):
    """Find optimal threshold for each action"""
    optimal_thresholds = {}
    
    for action in oof_predictions.columns:
        if action in labels.columns:
            mask = ~labels[action].isna()
            if mask.sum() > 100:  # Sufficient data for optimization
                y_true = labels[action][mask].values.astype(int)
                y_pred_proba = oof_predictions[action][mask].values
                
                best_f1 = 0
                best_thresh = default_threshold
                
                # Quick grid search
                for thresh in [0.15, 0.2, 0.25, 0.27, 0.3, 0.35, 0.4, 0.45, 0.5]:
                    f1 = f1_score(y_true, y_pred_proba >= thresh, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = thresh
                        
                optimal_thresholds[action] = best_thresh
            else:
                optimal_thresholds[action] = default_threshold
    
    return optimal_thresholds

# ============= ENHANCED MODEL CREATION =============
def create_simple_ensemble():
    """Create simple but effective ensemble"""
    lgb = LGBMClassifier(
        n_estimators=500,
        max_depth=6, 
        learning_rate=0.05,
        random_state=42,
        verbosity=-1,
        force_row_wise=True
    )
    
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    
    ensemble = VotingClassifier([('lgb', lgb), ('rf', rf)], voting='soft')
    
    return make_pipeline(
        SimpleImputer(),
        StandardScaler(), 
        TrainOnSubsetClassifier(ensemble, 25000)
    )

# ============= ENHANCED MULTICLASS PREDICTION =============
def predict_multiclass_optimized(pred, meta, thresholds=None):
    """Enhanced multiclass prediction with optimized thresholds"""
    if thresholds is None:
        thresholds = {col: 0.27 for col in pred.columns}
    
    # Apply action-specific thresholds
    ama = np.argmax(pred.values, axis=1)
    max_proba = pred.max(axis=1).values
    
    # Use action-specific thresholds
    threshold_array = np.array([thresholds.get(col, 0.27) for col in pred.columns])
    action_thresholds = threshold_array[ama]
    
    ama = np.where(max_proba >= action_thresholds, ama, -1)
    ama = pd.Series(ama, index=meta.video_frame)
    
    # Keep only start and stop frames
    changes_mask = (ama != ama.shift(1)).values
    ama_changes = ama[changes_mask]
    meta_changes = meta[changes_mask]
    
    # mask selects the start frames
    mask = ama_changes.values >= 0  # start of action
    mask[-1] = False
    
    submission_part = pd.DataFrame({
        'video_id': meta_changes['video_id'][mask].values,
        'agent_id': meta_changes['agent_id'][mask].values,
        'target_id': meta_changes['target_id'][mask].values,
        'action': pred.columns[ama_changes[mask].values],
        'start_frame': ama_changes.index[mask],
        'stop_frame': ama_changes.index[1:][mask[:-1]]
    })
    
    # Fix stop_frame for video boundaries
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
    if verbose: 
        print('  actions found:', len(submission_part))
        
    return submission_part

# ============= DATA GENERATION =============
def generate_mouse_data(dataset, traintest, traintest_directory=None, generate_single=True, generate_pair=True):
    """Generate batches of data in coordinate representation"""
    
    assert traintest in ['train', 'test']
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
        
    for _, row in dataset.iterrows():
        # Load the video and pivot it so that one frame = one row
        lab_id = row.lab_id
        if lab_id.startswith('MABe22'): 
            continue
        video_id = row.video_id
        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        
        vid = pd.read_parquet(path)
        pvid = vid.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])
        
        if pvid.isna().any().any():
            if verbose and traintest == 'test': 
                print('video with missing values', video_id, traintest, len(vid), 'frames')
        else:
            if verbose and traintest == 'test': 
                print('video with all values', video_id, traintest, len(vid), 'frames')
        
        del vid
        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T  # mouse_id, body_part, xy
        pvid /= row.pix_per_cm_approx  # convert to cm

        # Determine the behaviors of this video
        vid_behaviors = json.loads(row.behaviors_labeled)
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])
        
        # Load the annotations for training
        if traintest == 'train':
            try:
                annot = pd.read_parquet(path.replace('train_tracking', 'train_annotation'))
            except FileNotFoundError:
                # MABe22 and one more training file lack annotations
                continue

        # Create single_mouse dataframes
        if generate_single:
            vid_behaviors_subset = vid_behaviors.query("target == 'self'")
            for mouse_id_str in np.unique(vid_behaviors_subset.agent):
                try:
                    mouse_id = int(mouse_id_str[-1])
                    vid_agent_actions = np.unique(vid_behaviors_subset.query("agent == @mouse_id_str").action)
                    single_mouse = pvid.loc[:, mouse_id]
                    
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
                        if verbose: 
                            print('- test single', video_id, mouse_id)
                        yield 'single', single_mouse, single_mouse_meta, vid_agent_actions
                except KeyError:
                    pass  # Skip if no data for selected agent mouse

        # Create mouse_pair dataframes
        if generate_pair:
            vid_behaviors_subset = vid_behaviors.query("target != 'self'")
            if len(vid_behaviors_subset) > 0:
                for agent, target in itertools.permutations(np.unique(pvid.columns.get_level_values('mouse_id')), 2):
                    agent_str = f"mouse{agent}"
                    target_str = f"mouse{target}"
                    vid_agent_actions = np.unique(vid_behaviors_subset.query("(agent == @agent_str) & (target == @target_str)").action)
                    
                    mouse_pair = pd.concat([pvid[agent], pvid[target]], axis=1, keys=['A', 'B'])
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
                        if verbose: 
                            print('- test pair', video_id, agent, target)
                        yield 'pair', mouse_pair, mouse_pair_meta, vid_agent_actions

# ============= ENHANCED CROSS-VALIDATION =============
def cross_validate_classifier_enhanced(binary_classifier, X, label, meta):
    """Enhanced cross-validation with optimized thresholds"""
    
    oof = pd.DataFrame(index=meta.video_frame)
    
    for action in label.columns:
        # Filter for samples with defined target
        action_mask = ~label[action].isna().values
        X_action = X[action_mask]
        y_action = label[action][action_mask].values.astype(int)
        p = y_action.mean()
        baseline_score = p / (1 + p)
        groups_action = meta.video_id[action_mask]
        
        if len(np.unique(groups_action)) < 3:  # Need at least 3 groups for 3-fold CV
            continue
            
        if ~(y_action == 0).all():
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                oof_action = cross_val_predict(
                    binary_classifier, X_action, y_action, 
                    groups=groups_action, cv=GroupKFold(n_splits=3),  # Reduced to 3-fold
                    method='predict_proba'
                )
            oof_action = oof_action[:, 1]
        else:
            oof_action = np.zeros(len(y_action))
            
        # Store OOF predictions
        oof_column = np.zeros(len(label))
        oof_column[action_mask] = oof_action
        oof[action] = oof_column

    # Find optimal thresholds
    optimal_thresholds = find_optimal_thresholds(oof, label)
    
    if verbose:
        print("Optimal thresholds:", {k: f"{v:.3f}" for k, v in optimal_thresholds.items()})
    
    # Make multiclass prediction with optimized thresholds
    submission_part = predict_multiclass_optimized(oof, meta, optimal_thresholds)
    submission_list.append(submission_part)

# ============= SUBMISSION GENERATION =============
def submit_enhanced(body_parts_tracked_str, switch_tr, binary_classifier, X_tr, label, meta):
    """Enhanced submission with optimized thresholds"""
    
    # Fit binary classifier for every action
    model_list = []
    for action in label.columns:
        action_mask = ~label[action].isna().values
        y_action = label[action][action_mask].values.astype(int)

        if ~(y_action == 0).all():
            model = clone(binary_classifier)
            model.fit(X_tr[action_mask], y_action)
            assert len(model.classes_) == 2
            model_list.append((action, model))

    # Compute test predictions in batches
    body_parts_tracked = json.loads(body_parts_tracked_str)
    
    if validate_or_submit == 'submit':
        test_subset = test[test.body_parts_tracked == body_parts_tracked_str]
        generator = generate_mouse_data(test_subset, 'test',
                                      generate_single=(switch_tr == 'single'), 
                                      generate_pair=(switch_tr == 'pair'))
    else:
        test_subset = stresstest.query("body_parts_tracked == @body_parts_tracked_str")
        generator = generate_mouse_data(test_subset, 'test',
                                      traintest_directory='stresstest_tracking',
                                      generate_single=(switch_tr == 'single'),
                                      generate_pair=(switch_tr == 'pair'))
    
    if verbose: 
        print(f"n_videos: {len(test_subset)}")
    
    for switch_te, data_te, meta_te, actions_te in generator:
        assert switch_te == switch_tr
        try:
            # Transform from coordinate to distance representation
            if switch_te == 'single':
                X_te = transform_single_enhanced(data_te, body_parts_tracked)
            else:
                X_te = transform_pair_enhanced(data_te, body_parts_tracked)
                
            if verbose and len(X_te) == 0: 
                print("ERROR: X_te is empty")
            del data_te

            # Compute binary predictions
            pred = pd.DataFrame(index=meta_te.video_frame)
            for action, model in model_list:
                if action in actions_te:
                    pred[action] = model.predict_proba(X_te)[:, 1]
            del X_te

            # Compute multiclass predictions with default thresholds
            if pred.shape[1] != 0:
                submission_part = predict_multiclass_optimized(pred, meta_te)
                submission_list.append(submission_part)
            else:
                if verbose: 
                    print(f"  ERROR: no useful training data")
                    
        except KeyError:
            if verbose: 
                print(f'  ERROR: KeyError because of missing bodypart ({switch_tr})')
            if 'data_te' in locals():
                del data_te

# ============= ROBUSTIFICATION =============
def robustify(submission, dataset, traintest, traintest_directory=None):
    """Ensure submission conforms to competition rules"""
    
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"

    # Rule 1: Ensure that start_frame < stop_frame
    old_submission = submission.copy()
    submission = submission[submission.start_frame < submission.stop_frame]
    if len(submission) != len(old_submission):
        print("ERROR: Dropped frames with start >= stop")
    
    # Rule 2: Avoid multiple predictions for same frame from one agent/target pair
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

    # Rule 3: Submit something for every video (simplified fallback)
    s_list = []
    for idx, row in dataset.iterrows():
        lab_id = row['lab_id']
        if lab_id.startswith('MABe22'):
            continue
        video_id = row['video_id']
        if (submission.video_id == video_id).any():
            continue

        if verbose: 
            print(f"Video {video_id} has no predictions.")
        
        # Simple fallback prediction
        s_list.append((video_id, 'mouse1', 'self', 'rear', 100, 200))

    if len(s_list) > 0:
        submission = pd.concat([
            submission,
            pd.DataFrame(s_list, columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
        ])
        print("ERROR: Filled empty videos")

    submission = submission.reset_index(drop=True)
    return submission

# ============= MAIN PROCESSING LOOP =============
print("Starting enhanced processing loop...")

f1_list = []
submission_list = []

for section in range(1, len(body_parts_tracked_list)):  # skip index 0 (MABe22)
    body_parts_tracked_str = body_parts_tracked_list[section]
    
    try:
        body_parts_tracked = json.loads(body_parts_tracked_str)
        print(f"\n{section}. Processing videos with {body_parts_tracked}")
    
        # Read training data for this body parts configuration
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
    
        # Create enhanced binary classifier
        binary_classifier = create_simple_ensemble()
    
        # Process single-mouse actions
        if len(single_mouse_list) > 0:
            # Concatenate all batches
            single_mouse = pd.concat(single_mouse_list)
            single_mouse_label = pd.concat(single_mouse_label_list)
            single_mouse_meta = pd.concat(single_mouse_meta_list)
            del single_mouse_list, single_mouse_label_list, single_mouse_meta_list
            
            # Enhanced feature engineering
            X_tr = transform_single_enhanced(single_mouse, body_parts_tracked)
            del single_mouse
            print(f"Single mouse features: {X_tr.shape}")
    
            if validate_or_submit == 'validate':
                cross_validate_classifier_enhanced(binary_classifier, X_tr, single_mouse_label, single_mouse_meta)
            else:
                submit_enhanced(body_parts_tracked_str, 'single', binary_classifier, X_tr, single_mouse_label, single_mouse_meta)
            del X_tr
                
        # Process mouse-pair actions  
        if len(mouse_pair_list) > 0:
            # Concatenate all batches
            mouse_pair = pd.concat(mouse_pair_list)
            mouse_pair_label = pd.concat(mouse_pair_label_list)
            mouse_pair_meta = pd.concat(mouse_pair_meta_list)
            del mouse_pair_list, mouse_pair_label_list, mouse_pair_meta_list
        
            # Enhanced feature engineering
            X_tr = transform_pair_enhanced(mouse_pair, body_parts_tracked)
            del mouse_pair
            print(f"Mouse pair features: {X_tr.shape}")
    
            if validate_or_submit == 'validate':
                cross_validate_classifier_enhanced(binary_classifier, X_tr, mouse_pair_label, mouse_pair_meta)
            else:
                submit_enhanced(body_parts_tracked_str, 'pair', binary_classifier, X_tr, mouse_pair_label, mouse_pair_meta)
            del X_tr
                
    except Exception as e:
        print(f'***Exception*** {e}')

    print()

# ============= FINALIZATION =============
print("Finalizing submission...")

if validate_or_submit != 'validate':
    if len(submission_list) > 0:
        submission = pd.concat(submission_list)
    else:
        # Fallback submission
        submission = pd.DataFrame({
            'video_id': [438887472],
            'agent_id': ['mouse1'],
            'target_id': ['self'],
            'action': ['rear'],
            'start_frame': [278],
            'stop_frame': [500]
        })
    
    if validate_or_submit == 'submit':
        submission_robust = robustify(submission, test, 'test')
    else:
        submission_robust = robustify(submission, stresstest, 'stresstest', 'stresstest_tracking')
    
    submission_robust.index.name = 'row_id'
    submission_robust.to_csv('submission.csv')
    
    print(f"Final submission shape: {submission_robust.shape}")
    print("Submission saved to submission.csv")
    
    # Display first few rows
    print("\nFirst few rows of submission:")
    print(submission_robust.head())

