# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm.auto import tqdm
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


# Set some display options for pandas for better readability
pd.set_option('display.max_columns', 100)
sns.set_style('whitegrid')

# Define the path to your data
DATA_PATH = '/kaggle/input/MABe-mouse-behavior-detection/'

# Load the metadata files
print("Loading train.csv (metadata)...")
df_train_meta = pd.read_csv(DATA_PATH + 'train.csv')

print("Loading test.csv (metadata)...")
df_test_meta = pd.read_csv(DATA_PATH + 'test.csv')

print("\n--- Train Metadata ---")
print(f"Shape: {df_train_meta.shape}")
display(df_train_meta.head())

print("\n--- Test Metadata ---")
print(f"Shape: {df_test_meta.shape}")
display(df_test_meta.head())


# Select the first video from the metadata as our sample
sample_video_meta = df_train_meta.iloc[0]
sample_lab_id = sample_video_meta['lab_id']
sample_video_id = sample_video_meta['video_id']

print(f"Loading sample video...\n  Lab ID: {sample_lab_id}\n  Video ID: {sample_video_id}")


# Construct the file paths using the lab and video IDs
tracking_path = os.path.join(DATA_PATH, 'train_tracking', sample_lab_id, f'{sample_video_id}.parquet')
annotation_path = os.path.join(DATA_PATH, 'train_annotation', sample_lab_id, f'{sample_video_id}.parquet')

# Load the actual data from the parquet files
df_tracking_sample = pd.read_parquet(tracking_path)
df_annot_sample = pd.read_parquet(annotation_path)

print("\n--- Sample Tracking Data ---")
print(f"Shape: {df_tracking_sample.shape}")
print("Info:")
df_tracking_sample.info()
print("\nFirst 5 rows:")
display(df_tracking_sample.head())

print("\n\n--- Sample Annotation Data ---")
print(f"Shape: {df_annot_sample.shape}")
print("Info:")
df_annot_sample.info()
print("\nFirst 5 rows:")
display(df_annot_sample.head())


# 1. See what bodyparts are available
unique_bodyparts = df_tracking_sample['bodypart'].unique()
print(f"Unique bodyparts tracked: {unique_bodyparts}\n")


# 2. Pivot the table to get a "wide" format
# We want one row per video_frame, and columns for each mouse's bodypart's x and y coordinates.

print("Pivoting data from long to wide format...")


# Create a pivot for the 'x' coordinates
pivot_x = df_tracking_sample.pivot(
    index='video_frame', 
    columns=['mouse_id', 'bodypart'], 
    values='x'
)
# Rename columns for clarity, e.g., (1, 'nose') -> 'mouse1_nose_x'
pivot_x.columns = [f"mouse{m}_{bp}_x" for m, bp in pivot_x.columns]


# Create a pivot for the 'y' coordinates
pivot_y = df_tracking_sample.pivot(
    index='video_frame', 
    columns=['mouse_id', 'bodypart'], 
    values='y'
)
# Rename columns for clarity
pivot_y.columns = [f"mouse{m}_{bp}_y" for m, bp in pivot_y.columns]


# 3. Merge the x and y pivots into a single DataFrame
df_wide_sample = pd.concat([pivot_x, pivot_y], axis=1)

# Sort columns alphabetically for consistent order
df_wide_sample = df_wide_sample.sort_index(axis=1)


print("Pivoting complete.\n")
print("--- Reshaped Wide DataFrame ---")
print(f"Shape: {df_wide_sample.shape}")
display(df_wide_sample.head())


# --- BƯỚC 4: GÁN NHÃN HÀNH VI (MERGE LABELS) ---
# Tạo một cột 'label' mới, mặc định là 'other' (không có hành vi gì đặc biệt)
df_wide_sample['label'] = 'other'

# Duyệt qua từng dòng trong bảng ghi chú (annotation) để gán nhãn
# Logic: Nếu hành vi diễn ra từ frame 10 đến frame 20, ta gán nhãn đó cho các dòng 10-20
for _, row in df_annot_sample.iterrows():
    start = row['start_frame']
    stop = row['stop_frame']
    action = row['action']
    
    # Chỉ gán nhãn nếu frame nằm trong phạm vi dữ liệu của chúng ta
    # (Dùng .loc để gán giá trị cho các dòng từ start đến stop)
    if stop <= df_wide_sample.index.max():
        df_wide_sample.loc[start:stop, 'label'] = action

print("Đã gán nhãn xong!")
print("Phân phối các hành vi trong dữ liệu mẫu:")
print(df_wide_sample['label'].value_counts())

# --- BƯỚC 5: XỬ LÝ DỮ LIỆU THIẾU (MISSING VALUES) ---
# Như bạn thấy trong output cũ, có rất nhiều giá trị NaN (do camera bị che khuất)
# Model Machine Learning không hiểu NaN, nên ta phải điền số vào.
# Cách đơn giản nhất (MVP): Điền số 0
df_wide_sample = df_wide_sample.fillna(0)

print("\nĐã xử lý NaN. Dữ liệu sẵn sàng để train!")
display(df_wide_sample.head())


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- 1. CHUẨN BỊ DỮ LIỆU ---
# X: Dữ liệu đầu vào (Bỏ cột label đi)
X = df_wide_sample.drop(columns=['label'])
# y: Nhãn cần dự đoán
y = df_wide_sample['label']

# --- 2. MÃ HÓA NHÃN (ENCODING) ---
# Biến đổi chữ thành số (Ví dụ: 'attack' -> 0, 'investigation' -> 1...)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Các hành vi tìm thấy:", le.classes_)
print(f"Dữ liệu X shape: {X.shape}, y shape: {y_encoded.shape}")

# --- BƯỚC: FEATURE ENGINEERING (TẠO ĐẶC TRƯNG MỚI) ---
# Tính khoảng cách giữa Mouse 1 và Mouse 2 (Dựa trên Body Center)
# Công thức Pythagoras: căn bậc hai của ((x1-x2)^2 + (y1-y2)^2)

dx = df_wide_sample['mouse1_body_center_x'] - df_wide_sample['mouse2_body_center_x']
dy = df_wide_sample['mouse1_body_center_y'] - df_wide_sample['mouse2_body_center_y']

# Tạo cột mới 'distance_m1_m2'
df_wide_sample['distance_m1_m2'] = np.sqrt(dx**2 + dy**2)

print("Đã tạo thêm đặc trưng: distance_m1_m2")

# --- QUAN TRỌNG: CẬP NHẬT LẠI X ---
# Bây giờ X sẽ bao gồm cả cột khoảng cách mới này
X = df_wide_sample.drop(columns=['label'])

# ... (Sau đó bạn chạy lại đoạn code chia tập train/test và huấn luyện như cũ)

# --- 3. CHIA TẬP TRAIN / TEST ---
# Dành 80% để học, 20% để kiểm tra
# stratify=y_encoded giúp đảm bảo tập test vẫn có đủ các loại hành vi (kể cả hành vi hiếm)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- 4. HUẤN LUYỆN MODEL ---
print("\nĐang huấn luyện mô hình Random Forest (vui lòng chờ)...")
# Dùng Random Forest vì nó mạnh và không cần chuẩn hóa dữ liệu phức tạp
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# --- 5. ĐÁNH GIÁ KẾT QUẢ ---
print("Đã học xong! Đang dự đoán trên tập Test...")
y_pred = model.predict(X_test)

# In kết quả
acc = accuracy_score(y_test, y_pred)
print(f"\n>>> ĐỘ CHÍNH XÁC (Accuracy): {acc:.2%}")
print("\nChi tiết từng hành vi:")
print(classification_report(y_test, y_pred, target_names=le.classes_))


import matplotlib.pyplot as plt
import seaborn as sns

# Lấy thông tin độ quan trọng từ model đã train
importances = model.feature_importances_
feature_names = X.columns

# Tạo bảng dữ liệu để vẽ
feature_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Vẽ biểu đồ Top 10
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(15), palette='viridis')
plt.title('TOP 15 ĐẶC TRƯNG QUAN TRỌNG NHẤT')
plt.xlabel('Điểm quan trọng (Càng cao càng tốt)')
plt.ylabel('Tên đặc trưng')
plt.show()

# In ra giá trị cụ thể của 'distance_m1_m2'
dist_rank = feature_imp_df[feature_imp_df['Feature'] == 'distance_m1_m2']
print("Thứ hạng của Distance:")
print(dist_rank)


# ... (Phần hàm get_video_data giữ nguyên như cũ) ...

# --- CHẠY THỬ NGHIỆM CHIẾN THUẬT "TRAIN 1 VIDEO - TEST 1 VIDEO" ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

print("1. Đang load Video TRAIN (Index 0)...")
df_train = get_video_data(0)

print("2. Đang load Video TEST (Index 1)...")
# Thử lấy video index khác nếu video 1 bị lỗi, ví dụ index 2
df_test = get_video_data(1) 

if df_train is not None and df_test is not None:
    # Chuẩn bị dữ liệu
    X_train = df_train.drop(columns=['label'])
    y_train_raw = df_train['label']
    
    X_test = df_test.drop(columns=['label'])
    y_test_raw = df_test['label']
    
    # --- QUAN TRỌNG: BƯỚC ĐỒNG BỘ HÓA CỘT (FIX LỖI VALUE ERROR) ---
    # Ép X_test phải có cấu trúc cột Y HỆT X_train
    # - Cột nào thiếu (ví dụ headpiece): Tự động thêm vào và điền 0
    # - Cột nào thừa (ví dụ neck): Tự động vứt bỏ
    print("-> Đang đồng bộ hóa các cột dữ liệu...")
    feature_columns = X_train.columns
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)
    
    print(f"Kích thước sau đồng bộ -> Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Mã hóa nhãn (Label Encoding)
    le = LabelEncoder()
    # Gộp nhãn cả 2 tập để học hết các tên hành vi
    all_labels = list(y_train_raw.unique()) + list(y_test_raw.unique())
    le.fit(all_labels)
    
    y_train = le.transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    
    # Huấn luyện
    print("\n3. Đang huấn luyện Random Forest...")
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Dự đoán
    print("4. Đang dự đoán trên Video Test...")
    y_pred = model.predict(X_test)
    
    # Kết quả
    acc = accuracy_score(y_test, y_pred)
    print(f"\n>>> ĐỘ CHÍNH XÁC THỰC TẾ (Cross-Video): {acc:.2%}")
    
    # Xem đặc trưng quan trọng
    importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 5 đặc trưng quan trọng nhất:")
    print(importances.head(5))
    
    print("\nThứ hạng của Distance (Khoảng cách):")
    print(importances[importances['Feature'] == 'distance_m1_m2'])


# --- BƯỚC 8: FEATURE ENGINEERING & SELECTION (CAI NGHIỆN TỌA ĐỘ) ---

def calculate_features(df):
    # 1. Tính Khoảng cách (Đã làm)
    try:
        dx = df['mouse1_body_center_x'] - df['mouse2_body_center_x']
        dy = df['mouse1_body_center_y'] - df['mouse2_body_center_y']
        df['distance'] = np.sqrt(dx**2 + dy**2)
    except KeyError:
        df['distance'] = 0
        
    # 2. Tính Vận tốc (Velocity) của Mouse 1
    # Vận tốc = Khoảng cách di chuyển so với frame trước đó
    # Dùng .diff() để trừ dòng hiện tại cho dòng trước
    vx = df['mouse1_body_center_x'].diff().fillna(0)
    vy = df['mouse1_body_center_y'].diff().fillna(0)
    df['velocity_m1'] = np.sqrt(vx**2 + vy**2)
    
    # 3. Tính Vận tốc của Mouse 2 (Nếu có)
    try:
        vx2 = df['mouse2_body_center_x'].diff().fillna(0)
        vy2 = df['mouse2_body_center_y'].diff().fillna(0)
        df['velocity_m2'] = np.sqrt(vx2**2 + vy2**2)
    except KeyError:
        df['velocity_m2'] = 0
        
    return df

print("Đang tạo đặc trưng mới (Vận tốc, Khoảng cách)...")
df_train_eng = calculate_features(df_train.copy())
df_test_eng = calculate_features(df_test.copy())

# --- QUAN TRỌNG NHẤT: CHỌN LỌC ĐẶC TRƯNG ---
# Chúng ta chỉ lấy đúng 3 cột này để train. Vứt hết x, y đi!
selected_features = ['distance', 'velocity_m1', 'velocity_m2']

print(f"Chỉ train trên các đặc trưng: {selected_features}")

X_train_new = df_train_eng[selected_features]
y_train_new = y_train # Dùng lại nhãn đã encode ở bước trước

X_test_new = df_test_eng[selected_features]
y_test_new = y_test   # Dùng lại nhãn đã encode ở bước trước

# --- TRAIN LẠI MODEL ---
print("\nĐang train model trên dữ liệu tinh gọn...")
model_new = RandomForestClassifier(n_estimators=60, random_state=52)
model_new.fit(X_train_new, y_train_new)

# Dự đoán
print("Đang dự đoán...")
y_pred_new = model_new.predict(X_test_new)

# Kết quả
acc_new = accuracy_score(y_test_new, y_pred_new)
print(f"\n>>> ĐỘ CHÍNH XÁC MỚI: {acc_new:.2%}")

# Xem cái gì quan trọng nhất bây giờ
importances = pd.DataFrame({
    'Feature': selected_features,
    'Importance': model_new.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nBảng xếp hạng độ quan trọng:")
print(importances)


# --- BƯỚC 9: FEATURE ENGINEERING VỚI KÝ ỨC (ROLLING WINDOW) ---

def calculate_features_with_memory(df):
    # 1. TÍNH CƠ BẢN (VẬT LÝ)
    # Khoảng cách
    try:
        dx = df['mouse1_body_center_x'] - df['mouse2_body_center_x']
        dy = df['mouse1_body_center_y'] - df['mouse2_body_center_y']
        df['distance'] = np.sqrt(dx**2 + dy**2)
    except KeyError:
        df['distance'] = 0
        
    # Vận tốc (Mouse 1 & 2)
    vx = df['mouse1_body_center_x'].diff().fillna(0)
    vy = df['mouse1_body_center_y'].diff().fillna(0)
    df['velocity_m1'] = np.sqrt(vx**2 + vy**2)
    
    try:
        vx2 = df['mouse2_body_center_x'].diff().fillna(0)
        vy2 = df['mouse2_body_center_y'].diff().fillna(0)
        df['velocity_m2'] = np.sqrt(vx2**2 + vy2**2)
    except KeyError:
        df['velocity_m2'] = 0
        
    # 2. TÍNH KÝ ỨC (ROLLING WINDOW)
    # Window = 10 frames (Tương đương 0.33 giây nếu video 30fps)
    # Ý nghĩa: "Trong 0.3 giây vừa qua, chuyện gì đã xảy ra?"
    w = 10
    
    # Khoảng cách trung bình trong 10 frame qua (Đang gần lại hay xa ra?)
    df['dist_mean_10'] = df['distance'].rolling(window=w).mean().fillna(0)
    # Độ biến động khoảng cách (Chuột có đang giật cục không?)
    df['dist_std_10'] = df['distance'].rolling(window=w).std().fillna(0)
    
    # Vận tốc trung bình (Nó đang chạy bền hay chỉ giật mình 1 cái?)
    df['vel1_mean_10'] = df['velocity_m1'].rolling(window=w).mean().fillna(0)
    df['vel2_mean_10'] = df['velocity_m2'].rolling(window=w).mean().fillna(0)
    
    return df

print("Đang tạo bộ não có ký ức cho model...")
# Áp dụng hàm mới vào dữ liệu cũ
# (Lưu ý: Dùng .copy() để không ảnh hưởng dữ liệu gốc)
df_train_mem = calculate_features_with_memory(df_train.copy())
df_test_mem = calculate_features_with_memory(df_test.copy())

# --- DANH SÁCH ĐẶC TRƯNG MỚI ---
# Chúng ta sẽ train trên 7 đặc trưng này
features_memory = [
    'distance', 'velocity_m1', 'velocity_m2',       # Tức thời
    'dist_mean_10', 'dist_std_10',                  # Ký ức về khoảng cách
    'vel1_mean_10', 'vel2_mean_10'                  # Ký ức về vận tốc
]

print(f"Các đặc trưng sẽ dùng: {features_memory}")

# --- HUẤN LUYỆN LẠI ---
print("\nĐang train model (Random Forest)...")
model_mem = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model_mem.fit(df_train_mem[features_memory], y_train) # Dùng y_train cũ

# --- DỰ ĐOÁN & KẾT QUẢ ---
print("Đang dự đoán...")
y_pred_mem = model_mem.predict(df_test_mem[features_memory])

acc_mem = accuracy_score(y_test, y_pred_mem) # Dùng y_test cũ
print(f"\n>>> ĐỘ CHÍNH XÁC MỚI (Với ký ức): {acc_mem:.2%}")

# --- XEM CÁI GÌ QUAN TRỌNG NHẤT ---
importances = pd.DataFrame({
    'Feature': features_memory,
    'Importance': model_mem.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nBảng xếp hạng độ quan trọng:")
print(importances)


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 1. Tính ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred_mem)

# 2. Vẽ biểu đồ nhiệt (Heatmap) cho dễ nhìn
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, 
            yticklabels=le.classes_)
plt.xlabel('Model Dự đoán (Predicted)')
plt.ylabel('Sự thật (Actual)')
plt.title('Ma trận nhầm lẫn: Model đang lầm đường lạc lối ở đâu?')
plt.show()

# 3. In lại báo cáo chi tiết để đối chiếu
print(classification_report(y_test, y_pred_mem, target_names=le.classes_))


# --- BƯỚC 10: XỬ LÝ MẤT CÂN BẰNG DỮ LIỆU ---

print("Đang huấn luyện lại với chế độ 'Nghiêm khắc' (Balanced Weights)...")

# THAY ĐỔI DUY NHẤT: Thêm class_weight='balanced'
# Nó sẽ tự động tính toán: Hành vi nào ít xuất hiện -> Trọng số cao -> Phạt nặng nếu sai
model_balanced = RandomForestClassifier(
    n_estimators=50, 
    random_state=42, 
    n_jobs=-1,
    class_weight='balanced'  # <--- CHÌA KHÓA LÀ Ở ĐÂY
)

model_balanced.fit(df_train_mem[features_memory], y_train)

# Dự đoán
print("Đang dự đoán...")
y_pred_balanced = model_balanced.predict(df_test_mem[features_memory])

# Kết quả
acc_balanced = accuracy_score(y_test, y_pred_balanced)
print(f"\n>>> ĐỘ CHÍNH XÁC (Balanced): {acc_balanced:.2%}")

# Xem chi tiết từng hành vi (Chú ý cột RECALL của 'attack')
print("\nChi tiết hiệu năng:")
print(classification_report(y_test, y_pred_balanced, target_names=le.classes_))


# --- BƯỚC 11: CẬP NHẬT HÀM LOAD DATA (CHUẨN HÓA PIXEL -> CM) ---

def get_video_data_normalized(idx):
    # 1. Lấy thông tin metadata
    row = df_train_meta.iloc[idx]
    lab_id = row['lab_id']
    video_id = row['video_id']
    
    # --- CHÌA KHÓA: Lấy tỉ lệ quy đổi (Pixel per CM) ---
    pix_per_cm = row['pix_per_cm_approx']
    if pd.isna(pix_per_cm) or pix_per_cm == 0:
        pix_per_cm = 1.0 # Tránh chia cho 0
    
    print(f"Loading Video {video_id} (Lab: {lab_id}) - Scale: {pix_per_cm} pix/cm")
    
    # 2. Tạo đường dẫn
    t_path = os.path.join(DATA_PATH, 'train_tracking', lab_id, f'{video_id}.parquet')
    a_path = os.path.join(DATA_PATH, 'train_annotation', lab_id, f'{video_id}.parquet')
    
    try:
        df_track = pd.read_parquet(t_path)
    except FileNotFoundError:
        return None
        
    # 3. Pivot (Xoay bảng)
    px = df_track.pivot(index='video_frame', columns=['mouse_id', 'bodypart'], values='x')
    px.columns = [f"mouse{m}_{bp}_x" for m, bp in px.columns]
    
    py = df_track.pivot(index='video_frame', columns=['mouse_id', 'bodypart'], values='y')
    py.columns = [f"mouse{m}_{bp}_y" for m, bp in py.columns]
    
    df_wide = pd.concat([px, py], axis=1).sort_index(axis=1)
    
    # 4. --- CHUẨN HÓA NGAY TẠI ĐÂY ---
    # Chia toàn bộ tọa độ cho pix_per_cm để đưa về đơn vị CM
    df_wide = df_wide / pix_per_cm
    
    # 5. Load Nhãn
    try:
        df_annot = pd.read_parquet(a_path)
        df_wide['label'] = 'other'
        for _, row in df_annot.iterrows():
            if row['stop_frame'] <= df_wide.index.max():
                df_wide.loc[row['start_frame']:row['stop_frame'], 'label'] = row['action']
    except:
        df_wide['label'] = 'unknown'

    return df_wide.fillna(0)

# --- CHẠY LẠI QUY TRÌNH (DÙNG HÀM MỚI) ---
print("1. Load dữ liệu đã chuẩn hóa (cm)...")
df_train_norm = get_video_data_normalized(0)
df_test_norm = get_video_data_normalized(1) # Hoặc đổi sang index khác nếu muốn test video khác

if df_train_norm is not None and df_test_norm is not None:
    # Feature Engineering (Dùng lại hàm cũ calculate_features_with_memory)
    # Vì tọa độ đã là cm, nên Distance/Velocity tính ra sẽ là cm và cm/frame -> Rất chuẩn!
    print("2. Tạo đặc trưng (Features)...")
    df_train_final = calculate_features_with_memory(df_train_norm.copy())
    df_test_final = calculate_features_with_memory(df_test_norm.copy())
    
    # Đồng bộ cột (Tránh lỗi thiếu cột)
    feature_cols = [c for c in df_train_final.columns if c in features_memory] # Chỉ lấy các features đã chọn
    # Hoặc dùng lại list features_memory cũ:
    # features_memory = ['distance', 'velocity_m1', 'velocity_m2', 'dist_mean_10', 'dist_std_10', 'vel1_mean_10', 'vel2_mean_10']
    
    X_train = df_train_final[features_memory]
    y_train = le.transform(df_train_final['label']) # Dùng lại LabelEncoder cũ
    
    X_test = df_test_final[features_memory]
    y_test = le.transform(df_test_final['label'])
    
    # Train lại với Balanced Weight
    print("3. Train model (Normalized + Balanced)...")
    model_norm = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, class_weight='balanced')
    model_norm.fit(X_train, y_train)
    
    # Dự đoán
    print("4. Dự đoán...")
    y_pred_norm = model_norm.predict(X_test)
    
    print(f"\n>>> ĐỘ CHÍNH XÁC (Sau khi chuẩn hóa CM): {accuracy_score(y_test, y_pred_norm):.2%}")
    print("\nChi tiết hiệu năng:")
    print(classification_report(y_test, y_pred_norm, target_names=le.classes_))

