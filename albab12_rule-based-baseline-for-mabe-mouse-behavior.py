import pandas as pd
import numpy as np
import json
import itertools
import warnings
import gc
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Masking
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration
class CFG:
    SEQ_LENGTH = 64
    BATCH_SIZE = 256
    EPOCHS = 15
    VERBOSE = 1
    USE_AMP = True
    
if CFG.USE_AMP:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# --- Data Loading ---
print("Loading training and test metadata...")
train = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')
test = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv')
body_parts_tracked_list = list(np.unique(train.body_parts_tracked))
print(f"Found {len(body_parts_tracked_list)} unique body part configurations.")

# --- Feature Engineering ---
def feature_engineering_single_mouse(mouse_data, body_parts):
    features = []
    core_parts = ['nose', 'ear_left', 'ear_right', 'tail_base']
    parts_to_use = [p for p in core_parts if p in body_parts]
    for part in parts_to_use:
        features.append(mouse_data[part]['x'].rename(f'{part}_x'))
        features.append(mouse_data[part]['y'].rename(f'{part}_y'))
        features.append(mouse_data[part]['x'].diff().fillna(0).rename(f'{part}_vx'))
        features.append(mouse_data[part]['y'].diff().fillna(0).rename(f'{part}_vy'))
    df = pd.concat(features, axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    return pd.DataFrame(scaled_features, index=df.index, columns=df.columns)

def feature_engineering_pair_mouse(mouse_pair_data, body_parts):
    features = []
    core_parts = ['nose', 'ear_left', 'ear_right', 'tail_base']
    parts_to_use = [p for p in core_parts if p in body_parts]
    for mouse_id in ['A', 'B']:
        for part in parts_to_use:
            features.append(mouse_pair_data[mouse_id][part]['x'].rename(f'{mouse_id}_{part}_x'))
            features.append(mouse_pair_data[mouse_id][part]['y'].rename(f'{mouse_id}_{part}_y'))
            features.append(mouse_pair_data[mouse_id][part]['x'].diff().fillna(0).rename(f'{mouse_id}_{part}_vx'))
            features.append(mouse_pair_data[mouse_id][part]['y'].diff().fillna(0).rename(f'{mouse_id}_{part}_vy'))
    if 'nose' in parts_to_use and 'tail_base' in parts_to_use:
        dist_n2n = np.linalg.norm(mouse_pair_data['A']['nose'].values - mouse_pair_data['B']['nose'].values, axis=1)
        features.append(pd.Series(dist_n2n, index=mouse_pair_data.index, name='dist_nose2nose'))
        dist_n2t_A = np.linalg.norm(mouse_pair_data['A']['nose'].values - mouse_pair_data['B']['tail_base'].values, axis=1)
        features.append(pd.Series(dist_n2t_A, index=mouse_pair_data.index, name='dist_A_nose_B_tail'))
        dist_n2t_B = np.linalg.norm(mouse_pair_data['B']['nose'].values - mouse_pair_data['A']['tail_base'].values, axis=1)
        features.append(pd.Series(dist_n2t_B, index=mouse_pair_data.index, name='dist_B_nose_A_tail'))
    df = pd.concat(features, axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    return pd.DataFrame(scaled_features, index=df.index, columns=df.columns)

# --- Keras Data Generator ---
class DataGenerator(Sequence):
    def __init__(self, features, labels=None, batch_size=CFG.BATCH_SIZE, seq_length=CFG.SEQ_LENGTH, is_test=False):
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_test = is_test
        self.num_features = features.shape[1]
        if self.is_test:
            self.indices = np.arange(len(self.features))
        else:
            self.indices = np.arange(self.seq_length - 1, len(self.features))

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_pos_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = np.zeros((len(batch_pos_indices), self.seq_length, self.num_features), dtype=np.float32)
        for i, pos_idx in enumerate(batch_pos_indices):
            start_pos = pos_idx - self.seq_length + 1
            end_pos = pos_idx + 1
            X[i,] = self.features.iloc[start_pos:end_pos].values
        if self.is_test:
            return X
        else:
            y = self.labels.iloc[batch_pos_indices].values.astype(np.float32)
            return X, y
    
    def on_epoch_end(self):
        if not self.is_test:
            np.random.shuffle(self.indices)

# --- Model Building ---
def build_model(num_features):
    model = Sequential([
        Masking(mask_value=0., input_shape=(CFG.SEQ_LENGTH, num_features)),
        Bidirectional(LSTM(128, return_sequences=False)),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])
    return model

# --- Data Loading Generator ---
def generate_mouse_data(dataset, traintest, traintest_directory=None, generate_single=True, generate_pair=True):
    # This function remains largely the same
    assert traintest in ['train', 'test']
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
    for _, row in dataset.iterrows():
        lab_id, video_id = row.lab_id, row.video_id
        if lab_id.startswith('MABe22'): continue
        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        try:
            vid = pd.read_parquet(path)
        except Exception: continue
        pvid = vid.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])
        del vid
        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T
        pvid /= row.pix_per_cm_approx
        if pd.isna(row.behaviors_labeled): continue
        vid_behaviors = pd.DataFrame([b.split(',') for b in sorted(list({b.replace("'", "") for b in json.loads(row.behaviors_labeled)}))], columns=['agent', 'target', 'action'])
        annot = None
        if traintest == 'train':
            try:
                annot = pd.read_parquet(path.replace('train_tracking', 'train_annotation'))
            except FileNotFoundError: continue
        if generate_single:
            for mouse_id_str in np.unique(vid_behaviors.query("target == 'self'").agent):
                try:
                    mouse_id = int(mouse_id_str[-1])
                    actions = np.unique(vid_behaviors.query("agent == @mouse_id_str").action)
                    data = pvid.loc[:, mouse_id]
                    meta = pd.DataFrame({'video_id': video_id, 'agent_id': mouse_id_str, 'target_id': 'self', 'video_frame': data.index})
                    if traintest == 'train':
                        labels = pd.DataFrame(0.0, columns=actions, index=data.index)
                        subset = annot.query("(agent_id == @mouse_id) & (target_id == @mouse_id)")
                        for _, r in subset.iterrows():
                            labels.loc[r['start_frame']:r['stop_frame'], r.action] = 1.0
                        yield 'single', data, meta, labels
                    else: yield 'single', data, meta, actions
                except (KeyError, ValueError): pass
        if generate_pair:
            if len(vid_behaviors.query("target != 'self'")) > 0:
                for agent, target in itertools.permutations(np.unique(pvid.columns.get_level_values('mouse_id')), 2):
                    try:
                        agent_str, target_str = f"mouse{agent}", f"mouse{target}"
                        actions = np.unique(vid_behaviors.query("(agent == @agent_str) & (target == @target_str)").action)
                        data = pd.concat([pvid[agent], pvid[target]], axis=1, keys=['A', 'B'])
                        meta = pd.DataFrame({'video_id': video_id, 'agent_id': agent_str, 'target_id': target_str, 'video_frame': data.index})
                        if traintest == 'train':
                            labels = pd.DataFrame(0.0, columns=actions, index=data.index)
                            subset = annot.query("(agent_id == @agent) & (target_id == @target)")
                            for _, r in subset.iterrows():
                                labels.loc[r['start_frame']:r['stop_frame'], r.action] = 1.0
                            yield 'pair', data, meta, labels
                        else: yield 'pair', data, meta, actions
                    except (KeyError, ValueError): pass

# --- Post-Processing and Submission Formatting ---
def predict_multiclass_optimized(pred, meta, thresholds=None):
    # This function remains largely the same
    if pred.empty or pred.isnull().all().all(): return pd.DataFrame()
    if thresholds is None: thresholds = {col: 0.5 for col in pred.columns}
    pred_smooth = pred.rolling(5, min_periods=1, center=True).mean()
    ama = np.argmax(pred_smooth.values, axis=1)
    max_proba = pred_smooth.max(axis=1).values
    threshold_array = np.array([thresholds.get(col, 0.5) for col in pred.columns])
    action_thresholds = threshold_array[ama]
    ama = np.where(max_proba >= action_thresholds, ama, -1)
    ama = pd.Series(ama, index=meta.video_frame)
    changes_mask = (ama != ama.shift(1)).values
    ama_changes, meta_changes = ama[changes_mask], meta[changes_mask]
    mask = ama_changes.values >= 0
    if len(mask) > 0: mask[-1] = False
    if np.sum(mask) == 0: return pd.DataFrame()
    submission_part = pd.DataFrame({
        'video_id': meta_changes['video_id'][mask].values, 'agent_id': meta_changes['agent_id'][mask].values,
        'target_id': meta_changes['target_id'][mask].values, 'action': pred.columns[ama_changes[mask].values],
        'start_frame': ama_changes.index[mask], 'stop_frame': ama_changes.index[1:][mask[:-1]]})
    duration = submission_part.stop_frame - submission_part.start_frame
    return submission_part[(duration >= 2) & (duration <= 10000)]

def robustify(submission, dataset):
    # This function remains largely the same
    if submission is None or submission.empty:
        submission = pd.DataFrame(columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
    submission = submission[submission.start_frame < submission.stop_frame]
    group_list = []
    if not submission.empty:
        for _, group in submission.groupby(['video_id', 'agent_id', 'target_id']):
            group = group.sort_values('start_frame')
            mask = np.ones(len(group), dtype=bool)
            last_stop_frame = -1
            for i, (_, row) in enumerate(group.iterrows()):
                if row['start_frame'] < last_stop_frame: mask[i] = False
                else: last_stop_frame = row['stop_frame']
            group_list.append(group[mask])
    if group_list: submission = pd.concat(group_list)
    s_list = []
    for _, row in dataset.iterrows():
        if row.lab_id.startswith('MABe22'): continue
        if not (submission.video_id == row.video_id).any():
            s_list.append((row.video_id, 'mouse1', 'self', 'rear', 100, 200))
    if len(s_list) > 0:
        submission = pd.concat([submission, pd.DataFrame(s_list, columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])])
    return submission.reset_index(drop=True)

# ============= NEW MAIN PROCESSING LOOP =============
models = {}
print("--- Starting Training Phase ---")

for section, body_parts_tracked_str in enumerate(body_parts_tracked_list):
    if pd.isna(body_parts_tracked_str): continue
    try:
        body_parts_tracked = json.loads(body_parts_tracked_str)
        print(f"\nProcessing config {section}: {len(body_parts_tracked)} body parts")
        train_subset = train[train.body_parts_tracked == body_parts_tracked_str]
        if len(train_subset) == 0: continue

        # Process each behavior type (single/pair)
        for switch, feature_func in [('single', feature_engineering_single_mouse), ('pair', feature_engineering_pair_mouse)]:
            print(f"  Processing {switch} mouse actions...")
            
            # Aggregate all data for this config to find all possible actions
            all_actions_in_config = set()
            gen_args = {'generate_single': switch=='single', 'generate_pair': switch=='pair'}
            data_gen_for_actions = generate_mouse_data(train_subset, 'train', **gen_args)
            for _, _, _, labels in data_gen_for_actions:
                all_actions_in_config.update(labels.columns)

            # Train one model per action
            for action in sorted(list(all_actions_in_config)):
                print(f"    Training model for action: {action}")
                
                features_list, labels_list = [], []
                data_gen_for_training = generate_mouse_data(train_subset, 'train', **gen_args)
                
                for _, data, _, labels in data_gen_for_training:
                    if action in labels.columns:
                        features = feature_func(data, body_parts_tracked)
                        features_list.append(features)
                        labels_list.append(labels[[action]])

                if not features_list: continue

                # Now concat and train
                X_action = pd.concat(features_list).reset_index(drop=True)
                y_action = pd.concat(labels_list).reset_index(drop=True)

                if y_action.sum().iloc[0] < 20:
                    print(f"      Skipping {action} due to insufficient positive samples.")
                    continue
                
                train_gen = DataGenerator(X_action, y_action)
                model = build_model(num_features=X_action.shape[1])
                model.fit(train_gen, epochs=CFG.EPOCHS, verbose=CFG.VERBOSE)
                models[(body_parts_tracked_str, switch, action)] = model
                
    except Exception as e:
        print(f'  ***Exception***: {e}')
        import traceback
        traceback.print_exc()

# --- Prediction Phase ---
print("\n--- Starting Prediction Phase ---")
submission_list = []
for section, body_parts_tracked_str in enumerate(body_parts_tracked_list):
    if pd.isna(body_parts_tracked_str): continue
    try:
        body_parts_tracked = json.loads(body_parts_tracked_str)
        test_subset = test[test.body_parts_tracked == body_parts_tracked_str]
        if len(test_subset) == 0: continue
        
        print(f"\nPredicting for config {section}: {len(body_parts_tracked)} body parts")
        for switch, feature_func in [('single', feature_engineering_single_mouse), ('pair', feature_engineering_pair_mouse)]:
            print(f"  Predicting {switch} mouse actions...")
            gen_args = {'generate_single': switch=='single', 'generate_pair': switch=='pair'}
            generator = generate_mouse_data(test_subset, 'test', **gen_args)
            
            for _, data_te, meta_te, actions_te in generator:
                if len(data_te) < CFG.SEQ_LENGTH: continue
                X_te = feature_func(data_te, body_parts_tracked)
                pred_df = pd.DataFrame(index=X_te.index)

                for action in actions_te:
                    model_key = (body_parts_tracked_str, switch, action)
                    if model_key in models:
                        print(f"    Predicting action: {action}")
                        model = models[model_key]
                        test_gen = DataGenerator(X_te.reset_index(drop=True), is_test=True)
                        predictions = model.predict(test_gen, verbose=CFG.VERBOSE)
                        
                        pred_aligned = np.full(len(X_te), np.nan)
                        num_preds = len(predictions)
                        pred_aligned[test_gen.indices[:num_preds]] = predictions.flatten()
                        pred_df[action] = pd.Series(pred_aligned, index=X_te.index).interpolate(limit_direction='both')

                if not pred_df.empty:
                    submission_part = predict_multiclass_optimized(pred_df, meta_te)
                    submission_list.append(submission_part)
    except Exception as e:
        print(f'  ***Exception***: {e}')
        import traceback
        traceback.print_exc()

# --- Finalization ---
print("\nFinalizing submission...")
if submission_list:
    submission = pd.concat([s for s in submission_list if not s.empty])
else:
    submission = pd.DataFrame()

submission_robust = robustify(submission, test)
submission_robust.index.name = 'row_id'
submission_robust.to_csv('submission.csv')

print(f"\nFinal submission shape: {submission_robust.shape}")
print("Submission saved to submission.csv")
print(submission_robust.head())

