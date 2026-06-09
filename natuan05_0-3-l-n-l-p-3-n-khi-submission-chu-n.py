import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

# --- 1. Cáº¤U HÃŒNH ---
DATA_PATH = '/kaggle/input/MABe-mouse-behavior-detection/'
NUM_VIDEOS_TO_TRAIN = 20 # Sá»‘ lÆ°á»£ng video dÃ¹ng Ä‘á»ƒ train (CÃ ng nhiá»�u cÃ ng tá»‘t)

# Ä�á»�c metadata
try:
    df_train_meta = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
except FileNotFoundError:
    print("âš ï¸� Lá»—i: KhÃ´ng tÃ¬m tháº¥y file train.csv")

# --- 2. HÃ€M Táº O Ä�áº¶C TRÆ¯NG (FEATURE ENGINEERING) ---
def calculate_features_with_memory(df):
    # a. TÃ­nh Váº­t lÃ½ (Distance & Velocity)
    try:
        dx = df['mouse1_body_center_x'] - df['mouse2_body_center_x']
        dy = df['mouse1_body_center_y'] - df['mouse2_body_center_y']
        df['distance'] = np.sqrt(dx**2 + dy**2)
    except KeyError:
        df['distance'] = 0 
        
    vx = df['mouse1_body_center_x'].diff().fillna(0)
    vy = df['mouse1_body_center_y'].diff().fillna(0)
    df['velocity_m1'] = np.sqrt(vx**2 + vy**2)
    
    try:
        vx2 = df['mouse2_body_center_x'].diff().fillna(0)
        vy2 = df['mouse2_body_center_y'].diff().fillna(0)
        df['velocity_m2'] = np.sqrt(vx2**2 + vy2**2)
    except KeyError:
        df['velocity_m2'] = 0
        
    # b. Táº¡o KÃ½ á»©c (Rolling Window - 10 frames)
    w = 10
    df['dist_mean_10'] = df['distance'].rolling(window=w).mean().fillna(0)
    df['dist_std_10'] = df['distance'].rolling(window=w).std().fillna(0)
    df['vel1_mean_10'] = df['velocity_m1'].rolling(window=w).mean().fillna(0)
    df['vel2_mean_10'] = df['velocity_m2'].rolling(window=w).mean().fillna(0)
    
    return df

# --- 3. HÃ€M LOAD & Lá»ŒC Dá»® LIá»†U (CHá»ˆ Láº¤Y Cáº¶P 1-2 Ä�á»‚ TRAIN) ---
def get_train_data(idx):
    row = df_train_meta.iloc[idx]
    lab_id, video_id = row['lab_id'], row['video_id']
    pix_per_cm = row['pix_per_cm_approx'] if row['pix_per_cm_approx'] > 0 else 1.0
    
    # Load Tracking
    t_path = os.path.join(DATA_PATH, 'train_tracking', lab_id, f'{video_id}.parquet')
    a_path = os.path.join(DATA_PATH, 'train_annotation', lab_id, f'{video_id}.parquet')
    
    try:
        df_track = pd.read_parquet(t_path)
    except FileNotFoundError: return None

    # Pivot & Chuáº©n hÃ³a CM
    px = df_track.pivot(index='video_frame', columns=['mouse_id', 'bodypart'], values='x')
    px.columns = [f"mouse{m}_{bp}_x" for m, bp in px.columns]
    py = df_track.pivot(index='video_frame', columns=['mouse_id', 'bodypart'], values='y')
    py.columns = [f"mouse{m}_{bp}_y" for m, bp in py.columns]
    df_wide = pd.concat([px, py], axis=1).sort_index(axis=1)
    df_wide = df_wide / pix_per_cm 

    # Load Annotation & Lá»�c chá»‰ láº¥y tÆ°Æ¡ng tÃ¡c cáº·p 1-2
    try:
        df_annot = pd.read_parquet(a_path)
        df_wide['label'] = 'other'
        # Chá»‰ láº¥y dÃ²ng cÃ³ agent=1, target=2 HOáº¶C agent=2, target=1
        mask = ((df_annot['agent_id'] == 1) & (df_annot['target_id'] == 2)) | \
               ((df_annot['agent_id'] == 2) & (df_annot['target_id'] == 1))
        pair_annot = df_annot[mask]
        
        for _, r in pair_annot.iterrows():
            if r['stop_frame'] <= df_wide.index.max():
                df_wide.loc[r['start_frame']:r['stop_frame'], 'label'] = r['action']
    except:
        return None # Bá»� qua video lá»—i nhÃ£n

    return df_wide.fillna(0)

# --- 4. PIPELINE CHÃ�NH: Gá»˜P DATA -> CÃ‚N Báº°NG -> TRAIN ---
features = ['distance', 'velocity_m1', 'velocity_m2', 
            'dist_mean_10', 'dist_std_10', 'vel1_mean_10', 'vel2_mean_10']

# A. Gá»™p dá»¯ liá»‡u nhiá»�u video
all_data = []
print(f"â�³ Ä�ang xá»­ lÃ½ {NUM_VIDEOS_TO_TRAIN} video...")
for i in tqdm(range(NUM_VIDEOS_TO_TRAIN)):
    df = get_train_data(i)
    if df is not None and len(df) > 0:
        df = calculate_features_with_memory(df)
        all_data.append(df[features + ['label']])

df_big_train = pd.concat(all_data, ignore_index=True)
print(f"âœ… KÃ­ch thÆ°á»›c táº­p Train thÃ´: {df_big_train.shape}")

# B. CÃ¢n báº±ng dá»¯ liá»‡u (Undersampling an toÃ n)
print("âš–ï¸� Ä�ang cÃ¢n báº±ng dá»¯ liá»‡u...")
others = df_big_train[df_big_train['label'] == 'other']
actions = df_big_train[df_big_train['label'] != 'other']

min_sample = min(len(others), len(actions))
df_bal = pd.concat([
    resample(others, replace=False, n_samples=min_sample, random_state=42),
    resample(actions, replace=False, n_samples=min_sample, random_state=42)
])
print(f"âœ… Dá»¯ liá»‡u sau cÃ¢n báº±ng: {df_bal.shape} (Má»—i phe {min_sample} dÃ²ng)")

# C. Train Model
print("ğŸš€ Ä�ang huáº¥n luyá»‡n Random Forest...")
le = LabelEncoder()
y_train = le.fit_transform(df_bal['label'])
X_train = df_bal[features]

model_big = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model_big.fit(X_train, y_train)
print("ğŸ�‰ Ä�Ã£ huáº¥n luyá»‡n xong model_big!")
print(f"CÃ¡c nhÃ£n Ä‘Ã£ há»�c: {le.classes_}")


import pandas as pd
import numpy as np
import os
import gc
import lightgbm as lgb
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import itertools

# --- 1. Cáº¤U HÃŒNH ---
DATA_PATH = '/kaggle/input/MABe-mouse-behavior-detection/'
NUM_VIDEOS_TRAIN = 50 

# --- 2. HÃ€M FEATURE ENGINEERING AN TOÃ€N (FIX Lá»–I KEYERROR) ---
def calculate_features_safe(df):
    # Khá»Ÿi táº¡o máº·c Ä‘á»‹nh báº±ng 0.0 Ä‘á»ƒ trÃ¡nh lá»—i thiáº¿u cá»™t
    df['distance'] = 0.0
    df['velocity_m1'] = 0.0
    df['velocity_m2'] = 0.0
    
    # 1. TÃ­nh Khoáº£ng cÃ¡ch (Chá»‰ khi cÃ³ Ä‘á»§ cáº£ 2 chuá»™t)
    if 'mouse1_body_center_x' in df.columns and 'mouse2_body_center_x' in df.columns:
        dx = df['mouse1_body_center_x'] - df['mouse2_body_center_x']
        dy = df['mouse1_body_center_y'] - df['mouse2_body_center_y']
        df['distance'] = np.sqrt(dx**2 + dy**2)
    
    # 2. TÃ­nh Váº­n tá»‘c M1
    if 'mouse1_body_center_x' in df.columns:
        vx1 = df['mouse1_body_center_x'].diff().fillna(0)
        vy1 = df['mouse1_body_center_y'].diff().fillna(0)
        df['velocity_m1'] = np.sqrt(vx1**2 + vy1**2)
        
    # 3. TÃ­nh Váº­n tá»‘c M2
    if 'mouse2_body_center_x' in df.columns:
        vx2 = df['mouse2_body_center_x'].diff().fillna(0)
        vy2 = df['mouse2_body_center_y'].diff().fillna(0)
        df['velocity_m2'] = np.sqrt(vx2**2 + vy2**2)
        
    # 4. TÃ­nh KÃ½ á»©c (Rolling Window)
    # VÃ¬ cÃ¡c cá»™t trÃªn Ä‘Ã£ Ä‘Æ°á»£c khá»Ÿi táº¡o (dÃ¹ lÃ  0), nÃªn Ä‘oáº¡n nÃ y luÃ´n an toÃ n
    w = 10
    df['dist_mean_10'] = df['distance'].rolling(window=w).mean().fillna(0)
    df['vel1_mean_10'] = df['velocity_m1'].rolling(window=w).mean().fillna(0)
    df['vel2_mean_10'] = df['velocity_m2'].rolling(window=w).mean().fillna(0)
    
    return df

features = ['distance', 'velocity_m1', 'velocity_m2', 'dist_mean_10', 'vel1_mean_10', 'vel2_mean_10']

# --- 3. HÃ€M LOAD DATA "ALL-PAIRS" (Ä�Ãƒ THÃŠM TRY-EXCEPT) ---
def load_train_data_all_pairs(meta_row):
    try:
        video_id = meta_row['video_id']
        lab_id = meta_row['lab_id']
        pix_per_cm = meta_row['pix_per_cm_approx'] if meta_row['pix_per_cm_approx'] > 0 else 1.0
        
        t_path = os.path.join(DATA_PATH, 'train_tracking', lab_id, f'{video_id}.parquet')
        a_path = os.path.join(DATA_PATH, 'train_annotation', lab_id, f'{video_id}.parquet')
        
        if not os.path.exists(t_path) or not os.path.exists(a_path): return None
        
        # Load & Pivot
        df_track = pd.read_parquet(t_path)
        px = df_track.pivot(index='video_frame', columns=['mouse_id', 'bodypart'], values='x')
        px.columns = [f"mouse{m}_{bp}_x" for m, bp in px.columns]
        py = df_track.pivot(index='video_frame', columns=['mouse_id', 'bodypart'], values='y')
        py.columns = [f"mouse{m}_{bp}_y" for m, bp in py.columns]
        df_wide = pd.concat([px, py], axis=1).sort_index(axis=1)
        
        # Interpolate & Normalize
        df_wide = df_wide.interpolate(limit=5).fillna(0)
        df_wide = df_wide / pix_per_cm 
        
        # Load Annotation
        df_annot = pd.read_parquet(a_path)

        
        mouse_ids = sorted(list(set([int(c.split('_')[0].replace('mouse', '')) for c in df_wide.columns if 'mouse' in c])))
        pairs_data = []
        
        for m1, m2 in itertools.combinations(mouse_ids, 2):#lá»�c ra cÃ¡c tá»• há»£p chuá»™t cÃ³ thá»ƒ cÃ³ (1, 2) (2, 3)
            #cols1 láº¥y cÃ¡c cols cÃ³ chá»©a sá»‘ m1
            cols1 = [c for c in df_wide.columns if f'mouse{m1}_' in c]
            cols2 = [c for c in df_wide.columns if f'mouse{m2}_' in c]
            if not cols1 or not cols2: continue

            #copy má»™t báº£ng lá»�c ra cols1, cols2 tá»©c chá»‰ cÃ³ mouse_m1, mouse_m2 tá»« df_wide
            df_pair = df_wide[cols1 + cols2].copy()
            rename_dict = {}
            for c in cols1: rename_dict[c] = c.replace(f'mouse{m1}_', 'mouse1_')
            for c in cols2: rename_dict[c] = c.replace(f'mouse{m2}_', 'mouse2_')
            df_pair.rename(columns=rename_dict, inplace=True)
            #BÆ°á»›c trÃªn Ä‘á»•i tÃªn tá»« mouse_m1 thÃ nh mouse 1, ...
            
            # Check xem sau khi rename cÃ³ Ä‘á»§ cá»™t body_center khÃ´ng, náº¿u khÃ´ng thÃ¬ hÃ m safe sáº½ xá»­ lÃ½
            
            df_pair['label'] = 'other'
            mask = ((df_annot['agent_id'] == m1) & (df_annot['target_id'] == m2)) | \
                   ((df_annot['agent_id'] == m2) & (df_annot['target_id'] == m1))
            
            pair_annot = df_annot[mask]
            for _, r in pair_annot.iterrows():
                if r['stop_frame'] <= df_pair.index.max():
                    df_pair.loc[r['start_frame']:r['stop_frame'], 'label'] = r['action']
                    
            pairs_data.append(df_pair)
            
        if pairs_data:
            return pd.concat(pairs_data, ignore_index=True)
    except Exception as e:
        # print(f"Skipping video {meta_row['video_id']} due to error: {e}")
        return None
        
    return None

# --- 4. CHUáº¨N Bá»Š Dá»® LIá»†U TRAIN ---
print("â�³ Ä�ang táº¡o dá»¯ liá»‡u 'All-Pairs' tá»« 50 video (Safe Mode)...")
try:
    df_meta = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
except:
    df_meta = pd.DataFrame()

all_train_dfs = []
# DÃ¹ng tqdm Ä‘á»ƒ theo dÃµi tiáº¿n Ä‘á»™
for i in tqdm(range(min(NUM_VIDEOS_TRAIN, len(df_meta)))):
    df = load_train_data_all_pairs(df_meta.iloc[i])
    if df is not None:
        # DÃ¹ng hÃ m SAFE thay vÃ¬ hÃ m thÆ°á»�ng
        df = calculate_features_safe(df) 
        all_train_dfs.append(df[features + ['label']])

if len(all_train_dfs) > 0:
    df_train_big = pd.concat(all_train_dfs, ignore_index=True)
    del all_train_dfs
    gc.collect()

    print(f"âœ… Dá»¯ liá»‡u thÃ´: {df_train_big.shape}")

    # --- 5. CÃ‚N Báº°NG Dá»® LIá»†U ---
    print("âš–ï¸� Ä�ang cÃ¢n báº±ng dá»¯ liá»‡u...")
    others = df_train_big[df_train_big['label'] == 'other']
    actions = df_train_big[df_train_big['label'] != 'other']

    n_sample = min(len(others), len(actions))
    # Náº¿u video quÃ¡ Ã­t action, ta láº¥y toÃ n bá»™ action vÃ  má»™t pháº§n other
    if n_sample == 0: n_sample = 1000 # Fallback
    
    df_train_bal = pd.concat([
        resample(others, replace=False, n_samples=n_sample, random_state=42),
        actions # Láº¥y háº¿t action (vÃ¬ thÆ°á»�ng action Ã­t hÆ¡n other)
    ])
    
    # Náº¿u action nhiá»�u hÆ¡n other (hiáº¿m), ta cÅ©ng cÃ³ thá»ƒ resample action. 
    # NhÆ°ng code trÃªn Æ°u tiÃªn giá»¯ láº¡i toÃ n bá»™ hÃ nh vi hiáº¿m.
    
    print(f"âœ… Dá»¯ liá»‡u train cuá»‘i cÃ¹ng: {df_train_bal.shape}")

    # Label Encoding
    le = LabelEncoder()
    y_train = le.fit_transform(df_train_bal['label'])
    X_train = df_train_bal[features]

    # --- 6. HUáº¤N LUYá»†N LIGHTGBM ---
    print("ğŸš€ Ä�ang huáº¥n luyá»‡n LightGBM...")
    params = {
        'objective': 'multiclass',
        'num_class': len(le.classes_),
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'random_state': 42,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'verbosity': -1
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    print("ğŸ�‰ Huáº¥n luyá»‡n xong!")

else:
    print("â�Œ KhÃ´ng load Ä‘Æ°á»£c dá»¯ liá»‡u nÃ o. Kiá»ƒm tra láº¡i Ä‘Æ°á»�ng dáº«n hoáº·c dataset.")

# --- 7. HÃ€M POST-PROCESSING ---
def run_length_encoding_pro(predictions, agent_id, target_id):
    events = []
    if len(predictions) == 0: return events
    
    current_label = predictions[0]
    start_frame = 0
    
    for i in range(1, len(predictions)):
        if predictions[i] != current_label:
            if current_label != 'other':
                events.append({
                    'agent_id': agent_id,
                    'target_id': target_id,
                    'action': current_label,
                    'start_frame': start_frame,
                    'stop_frame': i - 1
                })
            current_label = predictions[i]
            start_frame = i
            
    if current_label != 'other':
        events.append({
            'agent_id': agent_id,
            'target_id': target_id,
            'action': current_label,
            'start_frame': start_frame,
            'stop_frame': len(predictions) - 1
        })
    return events

# --- 8. Táº O SUBMISSION (SAFE MODE) ---
# --- 8. Táº O SUBMISSION (Ä�Ãƒ FIX FORMAT MOUSE ID) ---
if 'model' in locals(): # Chá»‰ cháº¡y náº¿u train thÃ nh cÃ´ng
    print("ğŸ“� Ä�ang táº¡o submission (Correct Format)...")
    try:
        df_test_meta = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    except: df_test_meta = pd.DataFrame()

    submission_rows = []
    row_id_counter = 0

    for idx, row in tqdm(df_test_meta.iterrows(), total=len(df_test_meta)):
        try:
            video_id = row['video_id']
            lab_id = row['lab_id']
            pix_per_cm = row['pix_per_cm_approx'] if row['pix_per_cm_approx'] > 0 else 1.0
            
            t_path = os.path.join(DATA_PATH, 'test_tracking', lab_id, f'{video_id}.parquet')
            if not os.path.exists(t_path): continue
            
            df_track = pd.read_parquet(t_path)
            px = df_track.pivot(index='video_frame', columns=['mouse_id', 'bodypart'], values='x')
            px.columns = [f"mouse{m}_{bp}_x" for m, bp in px.columns]
            py = df_track.pivot(index='video_frame', columns=['mouse_id', 'bodypart'], values='y')
            py.columns = [f"mouse{m}_{bp}_y" for m, bp in py.columns]
            df_wide = pd.concat([px, py], axis=1).sort_index(axis=1)
            
            df_wide = df_wide.interpolate(limit=5).fillna(0)
            df_wide = df_wide / pix_per_cm
            
            # Láº¥y ID chuá»™t dÆ°á»›i dáº¡ng sá»‘ (1, 2...)
            mouse_ids = sorted(list(set([int(c.split('_')[0].replace('mouse', '')) for c in df_wide.columns if 'mouse' in c])))
            
            for m1, m2 in itertools.permutations(mouse_ids, 2):
                cols1 = [c for c in df_wide.columns if f'mouse{m1}_' in c]
                cols2 = [c for c in df_wide.columns if f'mouse{m2}_' in c]
                if not cols1 or not cols2: continue
                
                df_pair = df_wide[cols1 + cols2].copy()
                rename_dict = {}
                for c in cols1: rename_dict[c] = c.replace(f'mouse{m1}_', 'mouse1_')
                for c in cols2: rename_dict[c] = c.replace(f'mouse{m2}_', 'mouse2_')
                df_pair.rename(columns=rename_dict, inplace=True)
                
                # DÃ¹ng hÃ m SAFE features
                df_pair = calculate_features_safe(df_pair)
                
                X_test = pd.DataFrame(0.0, index=df_pair.index, columns=features)
                for c in features:
                    if c in df_pair.columns: X_test[c] = df_pair[c]
                
                # Dá»± Ä‘oÃ¡n
                y_pred_idx = model.predict(X_test)
                y_pred_lbl = le.inverse_transform(y_pred_idx)
                
                # --- Sá»¬A Lá»–I Táº I Ä�Ã‚Y: Ã‰P KIá»‚U Vá»€ "mouse1", "mouse2" ---
                agent_str = f"mouse{m1}"
                target_str = f"mouse{m2}"
                
                events = run_length_encoding_pro(y_pred_lbl, agent_str, target_str)
                
                for event in events:
                    submission_rows.append({
                        'row_id': row_id_counter,
                        'video_id': video_id,
                        'agent_id': event['agent_id'],   # BÃ¢y giá»� lÃ  "mouse1" thay vÃ¬ 1
                        'target_id': event['target_id'], # BÃ¢y giá»� lÃ  "mouse2" thay vÃ¬ 2
                        'action': event['action'],
                        'start_frame': event['start_frame'],
                        'stop_frame': event['stop_frame']
                    })
                    row_id_counter += 1
                    
            del df_wide
            gc.collect()
        except Exception as e:
            print(f"Error processing test video {video_id}: {e}")
            continue

    # LÆ°u file
    sub = pd.DataFrame(submission_rows)
    if len(sub) == 0:
        sub = pd.DataFrame(columns=['row_id', 'video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])

    sub.to_csv('submission.csv', index=False)
    print(f"âœ… Ä�Ã£ táº¡o submission.csv CHUáº¨N FORMAT vá»›i {len(sub)} dÃ²ng.")
    display(sub.head())

