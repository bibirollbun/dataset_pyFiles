!pip install -q tensorflow


import tensorflow as tf
import pandas as pd
import numpy as np
import os
import gc
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CẤU HÌNH (CONFIG)
# ==========================================
class CFG:
    RAW_DIR = "/kaggle/input/cafa-6-protein-function-prediction"
    TRAIN_TERMS = os.path.join(RAW_DIR, "Train/train_terms.tsv")
    
    EMBED_DIR = "/kaggle/input/cafa-6-t5-embeddings" 
    TRAIN_EMBEDS = os.path.join(EMBED_DIR, "train_embeds.npy")
    TRAIN_IDS = os.path.join(EMBED_DIR, "train_ids.npy")
    TEST_EMBEDS = os.path.join(EMBED_DIR, "test_embeds.npy")
    TEST_IDS = os.path.join(EMBED_DIR, "test_ids.npy")

    IA_PATH = "/kaggle/input/cafa-6-protein-function-prediction/IA.tsv" 
    
    INPUT_DIM = 1024
    BATCH_SIZE = 256
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    
    # Giữ nguyên key là BPO/CCO/MFO để dễ quản lý
    NUM_LABELS = {'BPO': 1500, 'CCO': 800, 'MFO': 800}


# ==========================================
# 2. HÀM LOAD & PREPARE DỮ LIỆU
# ==========================================
def load_dataset_fixed():
    print("--- 1. LOADING & CLEANING IDs ---")
    
    # A. Load Embeddings
    X = np.load(CFG.TRAIN_EMBEDS)
    train_ids_raw = np.load(CFG.TRAIN_IDS)
    
    # B. Clean Embeddings IDs (Cắt bỏ 'sp|...|')
    train_ids_clean = []
    for uid in train_ids_raw:
        uid_str = str(uid).strip()
        if '|' in uid_str:
            parts = uid_str.split('|')
            if len(parts) >= 2:
                train_ids_clean.append(parts[1])
            else:
                train_ids_clean.append(uid_str)
        else:
            train_ids_clean.append(uid_str)
            
    # Ép kiểu string
    train_ids_clean = [str(x) for x in train_ids_clean]
    
    # Tạo Dictionary Map
    id_to_idx = {uid: i for i, uid in enumerate(train_ids_clean)}
    
    # C. Load Labels
    terms_df = pd.read_csv(CFG.TRAIN_TERMS, sep='\t')
    terms_df['EntryID'] = terms_df['EntryID'].astype(str).str.strip()
    
    # Check khớp lệnh
    common = set(train_ids_clean).intersection(set(terms_df['EntryID']))
    print(f"   > IDs khớp nhau: {len(common)}")
    
    return X, id_to_idx, terms_df


def get_data_for_aspect(aspect_name, X, id_to_idx, terms_df):
    print(f"\n--- PREPARING {aspect_name} ---")
    
    # [SỬA LỖI QUAN TRỌNG] Mapping từ tên Aspect sang ký hiệu trong file (P, C, F)
    # BPO -> P, CCO -> C, MFO -> F
    aspect_map = {
        'BPO': 'P',
        'CCO': 'C',
        'MFO': 'F'
    }
    target_code = aspect_map[aspect_name]
    print(f"   > Mapping {aspect_name} -> '{target_code}' in dataset")
    
    # 1. Filter aspect
    df_aspect = terms_df[terms_df['aspect'] == target_code].copy()
    
    if len(df_aspect) == 0:
        raise ValueError(f"Không tìm thấy dữ liệu cho code '{target_code}'. Kiểm tra lại file terms!")

    # 2. Top-K Terms
    top_k = CFG.NUM_LABELS[aspect_name]
    top_terms = df_aspect['term'].value_counts().head(top_k).index.tolist()
    df_aspect = df_aspect[df_aspect['term'].isin(top_terms)]
    
    # 3. Pivot Table (Tạo One-Hot)
    print("   > Creating pivot table (this may take a moment)...")
    df_aspect['val'] = 1
    label_matrix = df_aspect.pivot_table(index='EntryID', columns='term', values='val', fill_value=0)
    
    # Ép kiểu index
    label_ids = [str(x) for x in label_matrix.index]
    
    # 4. Tìm giao thoa
    valid_ids = list(set(label_ids).intersection(set(id_to_idx.keys())))
    
    print(f"   > Label Matrix Rows: {len(label_matrix)}")
    print(f"   > Valid IDs (Intersection): {len(valid_ids)}")
    
    if len(valid_ids) == 0:
        raise ValueError(f"Không tìm thấy protein nào cho aspect {aspect_name}!")

    # 5. Tạo dữ liệu train
    # Lấy vector embedding tương ứng
    indices = [id_to_idx[uid] for uid in valid_ids]
    X_sub = X[indices]
    
    # Lấy nhãn tương ứng (cần reindex label_matrix theo valid_ids để đảm bảo thứ tự)
    # Lưu ý: label_matrix.loc[valid_ids] sẽ tự sắp xếp theo thứ tự valid_ids
    Y_sub = label_matrix.loc[valid_ids].values
    
    print(f"   > Final Train Data: X={X_sub.shape}, Y={Y_sub.shape}")
    return X_sub, Y_sub, label_matrix.columns.tolist()


# ==========================================
# 2.1 HELPER FUNCTIONS CHO IA LOSS
# ==========================================
def load_ia_weights(ia_path):
    """Đọc file IA.tsv và trả về dictionary {GO_ID: Score}"""
    print(f"Loading IA weights from: {ia_path}")
    
    # Đọc file TSV (không có header hoặc header tùy file, thường là cot 1: Term, cot 2: IA)
    # Giả định file có 2 cột: TermID và IA_Score
    try:
        df_ia = pd.read_csv(ia_path, sep='\t', header=None, names=['term', 'ia'])
        # Chuyển về dict để tra cứu cho nhanh
        return dict(zip(df_ia['term'], df_ia['ia']))
    except Exception as e:
        print(f"⚠️ Error reading IA file: {e}")
        return {}

def get_weighted_loss(class_weights):
    """
    Tạo hàm loss tùy chỉnh: Weighted Binary Crossentropy.
    class_weights: Numpy array chứa trọng số IA tương ứng với từng label.
    """
    # Chuyển weights thành Tensor hằng số (shape: 1, num_classes)
    weights_tensor = tf.constant(class_weights[None, :], dtype=tf.float32)
    
    def weighted_loss(y_true, y_pred):
        # Tránh lỗi log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Công thức Binary Crossentropy chuẩn
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        
        # Nhân với trọng số IA
        weighted_bce = bce * weights_tensor
        
        # Trả về trung bình loss
        return tf.reduce_mean(weighted_bce)
    
    return weighted_loss


def create_model(num_classes, weights_array=None):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(CFG.INPUT_DIM,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='sigmoid')
    ])
    
    # --- PHẦN THAY ĐỔI ---
    if weights_array is not None:
        # Nếu có weights, dùng Custom Loss
        loss_fn = get_weighted_loss(weights_array)
        #print("   > Using Weighted Binary Crossentropy (IA based)")
    else:
        # Nếu không (hoặc lỗi), dùng mặc định
        loss_fn = 'binary_crossentropy'
        #print("   > Using Standard Binary Crossentropy")
        
    model.compile(optimizer='adam', loss=loss_fn, metrics=['binary_accuracy'])
    return model


# A. Load Global Data
X_global, id_to_idx, terms_df = load_dataset_fixed()


# B. Load Test Data
print("\nLoading Test Data...")
if os.path.exists(CFG.TEST_EMBEDS):
    X_test = np.load(CFG.TEST_EMBEDS)
    test_ids = np.load(CFG.TEST_IDS)
    print(f"Test loaded: {X_test.shape}")
else:
    print("WARNING: Test files not found. Creating dummy test data just to run code.")
    # Dummy để code không crash nếu bạn chưa có file test
    X_test = np.zeros((10, CFG.INPUT_DIM)) 
    test_ids = np.array([f"TEST_{i}" for i in range(10)])


submissions = []

global_ia_weights = load_ia_weights(CFG.IA_PATH)

# C. Loop Training
for aspect in ['BPO', 'CCO', 'MFO']:
    try:
        # Get Data
        X_train, Y_train, target_terms = get_data_for_aspect(aspect, X_global, id_to_idx, terms_df)
        
        # Split
        x_tr, x_val, y_tr, y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

        print(f"Preparing weights for {len(target_terms)} terms...")
        weights_list = []
        for term in target_terms:
            # Nếu term có trong IA file thì lấy, không thì mặc định là 1.0
            # Mẹo: Có thể lấy mặc định là trung bình IA hoặc 1.0
            w = global_ia_weights.get(term, 1.0) 
            weights_list.append(w)
        
        weights_array = np.array(weights_list, dtype=np.float32)
        
        # Train
        print(f"Training {aspect} model...")
        model = create_model(len(target_terms), weights_array=weights_array) 
        
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        model.fit(x_tr, y_tr, validation_data=(x_val, y_val), 
                  epochs=CFG.EPOCHS, batch_size=CFG.BATCH_SIZE, 
                  callbacks=[early_stop], verbose=1)
        
        # Predict
        print(f"Predicting {aspect}...")
        preds = model.predict(X_test, batch_size=CFG.BATCH_SIZE, verbose=1)
        
        # Format Result
        df_pred = pd.DataFrame(preds, columns=target_terms)
        df_pred['EntryID'] = test_ids
        melted = df_pred.melt(id_vars='EntryID', var_name='term', value_name='score')
        melted = melted[melted['score'] > 0.006] # Chỉ lấy điểm > 0.005
        submissions.append(melted)
        
        # Cleanup
        del model, x_tr, y_tr, df_pred, melted, X_train, Y_train
        tf.keras.backend.clear_session()
        gc.collect()
        
    except Exception as e:
        print(f"Error processing {aspect}: {e}")
        import traceback
        traceback.print_exc()
        continue


# # ==========================================
# # D. SAVE SUBMISSION (TSV FORMAT)
# # ==========================================
# print("\nSaving final submission...")

# if len(submissions) > 0:
#     # 1. Gộp tất cả các aspect lại
#     final_df = pd.concat(submissions, axis=0, ignore_index=True)
    
#     # 2. Sắp xếp lại cho đẹp (Protein ID tăng dần, Score giảm dần) - Optional
#     # Giúp file dễ nhìn hơn nếu bạn mở ra check
#     print("Sorting data...")
#     final_df.sort_values(by=['EntryID', 'score'], ascending=[True, False], inplace=True)
    
#     # 3. Lưu file .tsv
#     output_filename = "submission.tsv"
    
#     print(f"Writing to {output_filename}...")
#     # sep='\t': Dùng tab làm dấu phân cách
#     # index=False: Không lưu cột số thứ tự dòng (0,1,2...)
#     # float_format='%.3f': Làm tròn 3 số thập phân để giảm dung lượng file (Optional)
#     final_df.to_csv(output_filename, sep='\t', index=False, float_format='%.3f', header=False) 
    
#     print(f"Done! Saved {len(final_df)} rows to {output_filename}")
#     print(final_df.head())

# else:
#     print("No predictions made.")


# ==========================================
# E. HELPER FUNCTIONS CHO POST-PROCESSING
# ==========================================
from collections import defaultdict

def parse_obo(obo_file):
    """Đọc file .obo để hiểu quan hệ cha-con giữa các GO terms"""
    print(f"Loading OBO file: {obo_file}")
    parents = defaultdict(list)
    children = defaultdict(list)
    term_id = None
    
    with open(obo_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('id: '):
                term_id = line.split('id: ')[1]
            elif line.startswith('is_a: ') and term_id:
                parent_id = line.split('is_a: ')[1].split(' ! ')[0]
                parents[term_id].append(parent_id)
                children[parent_id].append(term_id)
    return parents, children

def get_descendants(term, children_map, cache=None):
    """Tìm tất cả các con cháu của một GO term (để lan truyền tính chất NOT)"""
    if cache is None: cache = {}
    if term in cache: return cache[term]
    
    descendants = set()
    stack = [term]
    while stack:
        current = stack.pop()
        if current in children_map:
            for child in children_map[current]:
                if child not in descendants:
                    descendants.add(child)
                    stack.append(child)
    
    cache[term] = descendants
    return descendants


import os, gc
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# ==========================================
# F. POST-PROCESSING (DIRECT MEMORY PROCESSING)
# ==========================================
print("\n[START] Post-processing pipeline (Direct from RAM)...")

OBO_PATH = "/kaggle/input/cafa-6-protein-function-prediction/Train/go-basic.obo"
GOA_PATH = "/kaggle/input/protein-go-annotations/goa_uniprot_all.csv"

# Kiểm tra xem list submissions có dữ liệu không
if 'submissions' in globals() and len(submissions) > 0 and os.path.exists(OBO_PATH) and os.path.exists(GOA_PATH):
    
    # 1. GỘP BIẾN SUBMISSIONS TỪ RAM
    print("[1/6] Consolidating predictions from memory...")
    # Gộp tất cả các mảnh (aspects) lại thành 1 DataFrame
    sub = pd.concat(submissions, axis=0, ignore_index=True)
    
    # [QUAN TRỌNG] Đổi tên cột khớp với logic xử lý bên dưới
    # Code train sinh ra: ['EntryID', 'term', 'score']
    # Code xử lý cần: ['protein_id', 'go_term', 'score']
    sub.rename(columns={'EntryID': 'protein_id', 'term': 'go_term'}, inplace=True)
    
    # Xóa biến submissions gốc để giải phóng RAM ngay lập tức
    del submissions
    gc.collect()
    
    # Lấy danh sách ID mục tiêu
    target_ids = set(sub['protein_id'])
    print(f"   > Focusing on {len(target_ids)} target proteins.")

    # 2. PARSE OBO
    print("[2/6] Parsing Ontology Tree...")
    parents_map, children_map = parse_obo(OBO_PATH)

    # 3. ĐỌC GOA FILE THEO CHUNK
    print("[3/6] Scanning GOA file in chunks...")
    
    ground_truth_pairs = set() 
    negative_dict = {} 

    chunk_size = 1_000_000 
    reader = pd.read_csv(GOA_PATH, chunksize=chunk_size, usecols=['protein_id', 'go_term', 'qualifier'])

    for chunk in tqdm(reader, desc="Processing GOA Chunks"):
        # Chỉ giữ lại protein nằm trong target_ids
        chunk = chunk[chunk['protein_id'].isin(target_ids)]
        if chunk.empty: continue
        
        is_neg = chunk['qualifier'].str.contains('NOT', na=False)
        
        # Gom Negative
        neg_chunk = chunk[is_neg]
        for pid, term in zip(neg_chunk['protein_id'], neg_chunk['go_term']):
            if pid not in negative_dict: negative_dict[pid] = set()
            negative_dict[pid].add(term)
            
        # Gom Positive (Ground Truth)
        pos_chunk = chunk[~is_neg]
        ground_truth_pairs.update(zip(pos_chunk['protein_id'], pos_chunk['go_term']))
    
    print(f"   > Found {len(ground_truth_pairs)} Ground Truth pairs.")
    print(f"   > Found negative evidence for {len(negative_dict)} proteins.")
    
    del reader, chunk
    gc.collect()

    # 4. NEGATIVE PROPAGATION
    print("[4/6] Propagating Negative Terms...")
    blacklist_pairs = set()
    cache_descendants = {}
    
    for pid, terms in tqdm(negative_dict.items(), desc="Propagating"):
        all_neg_terms = set(terms)
        for t in list(all_neg_terms):
            descendants = get_descendants(t, children_map, cache_descendants)
            all_neg_terms.update(descendants)
            
        for t in all_neg_terms:
            blacklist_pairs.add((pid, t))
            
    print(f"   > Final Blacklist size: {len(blacklist_pairs)} pairs.")
    del negative_dict, cache_descendants, parents_map, children_map
    gc.collect()

    # 5. LỌC SUBMISSION (TRỰC TIẾP TRÊN BIẾN SUB)
    print("[5/6] Refining Submission...")
    print(f"   > Original rows: {len(sub)}")
    
    # Tạo mask lọc (nhanh hơn drop)
    # Logic: Giữ lại nếu (KHÔNG nằm trong blacklist) VÀ (KHÔNG nằm trong Ground Truth)
    # Ground truth sẽ được gộp vào sau với điểm 1.0
    valid_mask = []
    
    # Chuyển columns sang list/numpy để zip nhanh hơn truy cập dataframe
    pids = sub['protein_id'].values
    terms = sub['go_term'].values
    
    for pid, term in zip(pids, terms):
        pair = (pid, term)
        if (pair in blacklist_pairs) or (pair in ground_truth_pairs):
            valid_mask.append(False)
        else:
            valid_mask.append(True)
            
    sub = sub[valid_mask]
    print(f"   > Rows after filtering: {len(sub)}")
    
    del blacklist_pairs, valid_mask, pids, terms
    gc.collect()

    # 6. MERGE VÀ LƯU FILE
    print("[6/6] Merging Ground Truth and Saving...")
    
    if len(ground_truth_pairs) > 0:
        gt_df = pd.DataFrame(list(ground_truth_pairs), columns=['protein_id', 'go_term'])
        gt_df['score'] = 1.0
        final_submission = pd.concat([gt_df, sub], axis=0, ignore_index=True)
    else:
        final_submission = sub

    # Lưu file kết quả cuối cùng
    output_filename = "submission.tsv"
    # Sắp xếp nhẹ để dễ nhìn (có thể bỏ qua nếu RAM quá căng)
    # final_submission.sort_values(by=['protein_id', 'score'], ascending=[True, False], inplace=True)
    
    final_submission.to_csv(output_filename, sep='\t', index=False, header=False, float_format='%.3f')
    
    print(f"[SUCCESS] Saved {len(final_submission)} rows to {output_filename}")
    
    # Dọn dẹp sạch sẽ
    del sub, ground_truth_pairs, final_submission
    gc.collect()

else:
    print("[ERROR] 'submissions' list is empty/missing OR dataset files not found.")

