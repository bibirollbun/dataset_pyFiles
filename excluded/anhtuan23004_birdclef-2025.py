import pandas as pd # Thao tác dữ liệu 
import numpy as np
import os
import librosa # Xử lý âm thanh

# Trực quan hóa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import warnings
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict

# Xử lý tín hiệu
from scipy import stats
from scipy.signal import periodogram

import gc  # Dọn rác
import json
import time

warnings.filterwarnings('ignore')

# Thiết lập tùy chọn hiển thị
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

# Cho các ô tương tác trong sổ tay
%matplotlib inline


class Config:
    # Tùy chọn chế độ gỡ lỗi
    DEBUG_MODE = False  # Đặt thành False để xử lý toàn bộ tập dữ liệu
    VISUALIZE = True   # Có hiển thị biểu đồ trực quan không?
    SAMPLE_SIZE = 100  # Số lượng tệp sẽ được phân tích khi bật chế độ debug
    
    # Đường dẫn thư mục
    BASE_DIR = '/kaggle/input/birdclef-2025/'  # Thư mục gốc chứa tập dữ liệu
    TRAIN_AUDIO_DIR = os.path.join(BASE_DIR, 'train_audio')  # Thư mục chứa file âm thanh huấn luyện
    TRAIN_SOUNDSCAPES_DIR = os.path.join(BASE_DIR, 'train_soundscapes')  # Thư mục chứa soundscape huấn luyện
    TEST_SOUNDSCAPES_DIR = os.path.join(BASE_DIR, 'test_soundscapes')    # Thư mục chứa soundscape kiểm thử
    OUTPUT_DIR = "/kaggle/working/"  # Thư mục để lưu kết quả đầu ra
    
    # Thông số âm thanh
    SR = 32000  # Tần số lấy mẫu (sampling rate)
    
    def __init__(self):
        # Điều chỉnh đường dẫn khi chạy trong môi trường local
        if not os.path.exists(self.BASE_DIR):
            print("Đang điều chỉnh đường dẫn cho môi trường local...")
            self.BASE_DIR = "./data"
            self.TRAIN_AUDIO_DIR = "./data/train_audio"
            self.TRAIN_SOUNDSCAPES_DIR = "./data/train_soundscapes"
            self.TEST_SOUNDSCAPES_DIR = "./data/test_soundscapes" 
            self.OUTPUT_DIR = "./output"
            os.makedirs(self.OUTPUT_DIR, exist_ok=True)

# Khởi tạo cấu hình
cfg = Config()

# In ra các đường dẫn
print(f"Thư mục gốc: {cfg.BASE_DIR}")
print(f"Thư mục âm thanh huấn luyện: {cfg.TRAIN_AUDIO_DIR}")
print(f"Thư mục lưu kết quả: {cfg.OUTPUT_DIR}")


# Hàm tải file âm thanh và trả về dạng sóng âm (waveform)
def load_audio_file(file_path, sr=Config.SR):
    """
    Tải file âm thanh và trả về sóng âm
    """
    try:
        y, _ = librosa.load(file_path, sr=sr)
        return y
    except Exception as e:
        print(f"Lỗi khi tải {file_path}: {e}")
        return None

# Hàm đánh giá chất lượng âm thanh
def assess_audio_quality(y):
    """
    Đánh giá chất lượng âm thanh dựa trên thống kê tín hiệu
    Trả về điểm chất lượng từ 0 đến 1
    """
    if y is None or len(y) == 0:
        return 0
    
    # Tính các thống kê tín hiệu
    signal_std = np.std(y)
    signal_var = np.var(y)
    signal_rms = np.sqrt(np.mean(np.square(y)))
    signal_pwr = np.mean(np.square(y))
    
    # Tính tổng thống kê
    t_stat = signal_std + signal_var + signal_rms + signal_pwr
    
    # Chuẩn hóa về khoảng từ 0 đến 1 (ước lượng)
    quality_score = min(1.0, t_stat * 100)
    
    return quality_score

# Hàm trích xuất đặc trưng tần số
def extract_frequency_characteristics(y, sr=Config.SR):
    """
    Trích xuất các đặc trưng miền tần số từ âm thanh
    """
    if y is None or len(y) == 0:
        return None
    
    # Tính mật độ phổ công suất (Power Spectral Density)
    freqs, psd = periodogram(y, fs=sr)
    
    # Xác định tần số trội nhất
    dominant_freq = freqs[np.argmax(psd)]
    
    # Giới hạn phân tích trong dải tần nghe được: 20Hz - 20kHz
    idx_range = np.where((freqs >= 20) & (freqs <= 20000))[0]
    
    if len(idx_range) > 0:
        audible_freqs = freqs[idx_range]
        audible_psd = psd[idx_range]
        
        # Tính trọng tâm phổ (trung bình có trọng số của tần số)
        if np.sum(audible_psd) > 0:
            spectral_centroid = np.sum(audible_freqs * audible_psd) / np.sum(audible_psd)
        else:
            spectral_centroid = 0
            
        # Tính độ rộng phổ (độ phân tán của tần số)
        if np.sum(audible_psd) > 0:
            spectral_bandwidth = np.sqrt(np.sum(((audible_freqs - spectral_centroid) ** 2) * audible_psd) / np.sum(audible_psd))
        else:
            spectral_bandwidth = 0
            
        # Tính độ phẳng phổ (tỉ lệ trung bình hình học / trung bình số học)
        # Cho biết âm thanh mang tính nhiễu (1) hay mang tính giai điệu (0)
        audible_psd_nonzero = audible_psd[audible_psd > 0]
        if len(audible_psd_nonzero) > 0:
            spectral_flatness = stats.mstats.gmean(audible_psd_nonzero) / np.mean(audible_psd_nonzero)
        else:
            spectral_flatness = 0
    else:
        spectral_centroid = 0
        spectral_bandwidth = 0
        spectral_flatness = 0
    
    return {
        'dominant_frequency': dominant_freq,       # Tần số trội nhất
        'spectral_centroid': spectral_centroid,    # Trọng tâm phổ
        'spectral_bandwidth': spectral_bandwidth,  # Độ rộng phổ
        'spectral_flatness': spectral_flatness     # Độ phẳng phổ
    }

# Hàm trích xuất loại âm thanh từ tên file
def extract_call_type_from_filename(filename):
    """
    Trích xuất loại âm thanh từ tên file nếu có
    Mẫu tên thường gặp: XC######-{call_type}.ogg
    """
    call_types = [
        'song', 'call', 'alarm', 'flight', 'drum', 'display', 'duet', 
        'juvenile', 'begging', 'contact', 'excited', 'warn', 'wing',
        'dawn', 'growl', 'chatter', 'trill', 'whistle'
    ]
    
    filename_lower = filename.lower()
    
    for call_type in call_types:
        if call_type in filename_lower:
            return call_type
    
    # Nếu không tìm thấy loại âm thanh cụ thể nào
    if 'song' in filename_lower:
        return 'song'
    elif 'call' in filename_lower:
        return 'call'
    else:
        return 'unknown'



# Thử tải metadata (dữ liệu mô tả)
def load_metadata(cfg):
    try:
        metadata_path = os.path.join(cfg.BASE_DIR, 'train.csv')
        if os.path.exists(metadata_path):
            metadata_df = pd.read_csv(metadata_path)
            print(f"Đã tải metadata với {len(metadata_df)} dòng")
            return metadata_df
        else:
            print("Không tìm thấy file metadata, chuyển sang phân tích từ hệ thống tệp")
            return None
    except Exception as e:
        print(f"Lỗi khi tải metadata: {e}")
        return None

metadata_df = load_metadata(cfg)

# Hiển thị một mẫu của metadata nếu có
if metadata_df is not None:
    print("\nCác cột trong metadata:", metadata_df.columns.tolist())
    print("\nMẫu dữ liệu metadata:")
    display(metadata_df.head())
    
    # Đếm số loài duy nhất
    unique_species = metadata_df['primary_label'].nunique()
    print(f"\nSố lượng loài duy nhất: {unique_species}")


# Thử tải dữ liệu phân loại sinh học (taxonomy)
def load_taxonomy(cfg):
    try:
        taxonomy_path = os.path.join(cfg.BASE_DIR, 'taxonomy.csv')
        if os.path.exists(taxonomy_path):
            taxonomy_df = pd.read_csv(taxonomy_path)
            print(f"Đã tải dữ liệu taxonomy với {len(taxonomy_df)} dòng")
            return taxonomy_df
        else:
            print("Không tìm thấy file taxonomy")
            return None
    except Exception as e:
        print(f"Lỗi khi tải taxonomy: {e}")
        return None

taxonomy_df = load_taxonomy(cfg)

# Hiển thị một mẫu của dữ liệu taxonomy nếu có
if taxonomy_df is not None:
    print("\nCác cột trong taxonomy:", taxonomy_df.columns.tolist())
    print("\nMẫu dữ liệu taxonomy:")
    display(taxonomy_df.head())
    
    # Đếm số loài theo lớp (class)
    if 'class' in taxonomy_df.columns:
        class_counts = taxonomy_df['class'].value_counts()
        print("\nSố lượng loài theo lớp:")
        display(class_counts)


def get_species_from_directory(cfg):
    """Lấy danh sách các loài từ cấu trúc thư mục"""
    if os.path.exists(cfg.TRAIN_AUDIO_DIR):
        species_list = [d for d in os.listdir(cfg.TRAIN_AUDIO_DIR) 
                        if os.path.isdir(os.path.join(cfg.TRAIN_AUDIO_DIR, d))]
        print(f"Tìm thấy {len(species_list)} thư mục loài")
        return species_list
    else:
        print(f"Không tìm thấy thư mục audio huấn luyện: {cfg.TRAIN_AUDIO_DIR}")
        return []

# Lấy danh sách loài từ metadata (nếu có) hoặc từ thư mục
if metadata_df is not None:
    species_list = metadata_df['primary_label'].unique().tolist()
    print(f"Tìm thấy {len(species_list)} loài từ metadata")
else:
    species_list = get_species_from_directory(cfg)

# Tạo ánh xạ giữa tên loài và đường dẫn thư mục tương ứng
species_dir_mapping = {}
if os.path.exists(cfg.TRAIN_AUDIO_DIR):
    for species in species_list:
        species_dir = os.path.join(cfg.TRAIN_AUDIO_DIR, species)
        if os.path.isdir(species_dir):
            species_dir_mapping[species] = species_dir
    
    print(f"Đã ánh xạ {len(species_dir_mapping)} loài tới thư mục tương ứng")

# Đếm số lượng mẫu âm thanh cho từng loài
samples_per_species = {}
for species, species_dir in species_dir_mapping.items():
    audio_files = [f for f in os.listdir(species_dir) 
                   if f.endswith(('.mp3', '.ogg', '.wav'))]
    samples_per_species[species] = len(audio_files)

# Chuyển sang DataFrame để dễ phân tích
samples_df = pd.DataFrame({
    'species': list(samples_per_species.keys()),
    'sample_count': list(samples_per_species.values())
}).sort_values('sample_count', ascending=False)

# In thống kê
print("\nThống kê số mẫu âm thanh theo loài:")
print(f"- Số mẫu ít nhất: {samples_df['sample_count'].min()}")
print(f"- Số mẫu nhiều nhất: {samples_df['sample_count'].max()}")
print(f"- Số mẫu trung bình: {samples_df['sample_count'].mean():.1f}")
print(f"- Số mẫu trung vị: {samples_df['sample_count'].median():.1f}")
print(f"- Tổng số mẫu: {samples_df['sample_count'].sum()}")

# Hiển thị các loài có ít mẫu nhất
print("\nCác loài có ít mẫu âm thanh nhất:")
display(samples_df.nsmallest(10, 'sample_count'))


# Vẽ biểu đồ phân bố số mẫu âm thanh theo loài
plt.figure(figsize=(12, 6))
plt.hist(samples_df['sample_count'], bins=50, edgecolor='black')
plt.title('Phân bố số mẫu âm thanh theo loài')
plt.xlabel('Số lượng mẫu âm thanh')
plt.ylabel('Số lượng loài')
plt.grid(True, alpha=0.3)
plt.show()

# Vẽ biểu đồ 30 loài có nhiều mẫu âm thanh nhất
plt.figure(figsize=(14, 10))
top_species = samples_df.head(30)
sns.barplot(x='sample_count', y='species', data=top_species)
plt.title('Top 30 loài có nhiều mẫu âm thanh nhất')
plt.xlabel('Số lượng tệp âm thanh')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


def analyze_audio_length(species_list, species_dir_mapping, cfg):
    """Phân tích độ dài của các tệp âm thanh"""
    results = {
        'audio_lengths': [],          # Tổng hợp độ dài của tất cả tệp âm thanh
        'species_lengths': {}         # Độ dài theo từng loài
    }

    # Nếu đang ở chế độ debug, chọn ngẫu nhiên một số loài để phân tích
    if cfg.DEBUG_MODE:
        np.random.seed(42)
        if len(species_list) > 20:
            species_subset = np.random.choice(species_list, 20, replace=False).tolist()
        else:
            species_subset = species_list
    else:
        species_subset = species_list

    print(f"Đang phân tích độ dài âm thanh cho {len(species_subset)} loài")

    for species_idx, species in enumerate(species_subset):
        species_dir = species_dir_mapping.get(species)
        if not species_dir or not os.path.exists(species_dir):
            continue

        audio_files = [f for f in os.listdir(species_dir)
                       if f.endswith(('.mp3', '.ogg', '.wav'))]

        # Nếu đang debug, chỉ lấy một mẫu nhỏ
        if cfg.DEBUG_MODE and len(audio_files) > cfg.SAMPLE_SIZE:
            audio_files = np.random.choice(audio_files, cfg.SAMPLE_SIZE, replace=False).tolist()

        species_lengths = []

        # Xử lý từng tệp âm thanh
        for file_idx, filename in enumerate(tqdm(audio_files, desc=f"Loài {species_idx+1}/{len(species_subset)}")):
            file_path = os.path.join(species_dir, filename)
            audio = load_audio_file(file_path, sr=cfg.SR)

            if audio is not None:
                audio_length = len(audio) / cfg.SR  # Độ dài tính bằng giây
                results['audio_lengths'].append(audio_length)
                species_lengths.append(audio_length)

        # Lưu kết quả độ dài theo từng loài
        if species_lengths:
            results['species_lengths'][species] = species_lengths

    return results


# Chạy phân tích độ dài âm thanh
audio_length_results = analyze_audio_length(species_list, species_dir_mapping, cfg)

if audio_length_results['audio_lengths']:
    audio_lengths = np.array(audio_length_results['audio_lengths'])

    print("\nThống kê độ dài âm thanh:")
    print(f"- Số lượng tệp được phân tích: {len(audio_lengths)}")
    print(f"- Độ dài trung bình: {np.mean(audio_lengths):.2f} giây")
    print(f"- Độ dài nhỏ nhất: {np.min(audio_lengths):.2f} giây")
    print(f"- Độ dài lớn nhất: {np.max(audio_lengths):.2f} giây")
    print(f"- Độ dài trung vị: {np.median(audio_lengths):.2f} giây")

    # Vẽ biểu đồ phân bố độ dài âm thanh
    plt.figure(figsize=(12, 6))
    plt.hist(audio_lengths, bins=50, edgecolor='black')
    plt.axvline(np.median(audio_lengths), color='red', linestyle='dashed', linewidth=1, label=f'Trung vị: {np.median(audio_lengths):.2f}s')
    plt.axvline(np.mean(audio_lengths), color='green', linestyle='dashed', linewidth=1, label=f'Trung bình: {np.mean(audio_lengths):.2f}s')
    plt.title('Phân bố độ dài các tệp âm thanh')
    plt.xlabel('Độ dài (giây)')
    plt.ylabel('Số lượng')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
else:
    print("Không có tệp âm thanh nào được phân tích về độ dài")


def analyze_call_types(species_list, species_dir_mapping, cfg):
    """Phân tích loại tiếng kêu từ tên tệp"""
    results = {
        'call_types': {},           # Lưu loại tiếng kêu theo từng loài
        'overall_counts': Counter() # Đếm tổng số loại tiếng kêu
    }

    # Chọn một tập nhỏ nếu ở chế độ DEBUG
    if cfg.DEBUG_MODE:
        if len(species_list) > 30:
            species_subset = np.random.choice(species_list, 30, replace=False).tolist()
        else:
            species_subset = species_list
    else:
        species_subset = species_list

    print(f"Đang phân tích loại tiếng kêu cho {len(species_subset)} loài")

    for species_idx, species in enumerate(species_subset):
        species_dir = species_dir_mapping.get(species)
        if not species_dir or not os.path.exists(species_dir):
            continue

        audio_files = [f for f in os.listdir(species_dir)
                       if f.endswith(('.mp3', '.ogg', '.wav'))]

        species_call_types = Counter()

        # Phân tích từng tên tệp
        for filename in audio_files:
            # Trích xuất loại tiếng kêu từ tên tệp
            call_type = extract_call_type_from_filename(filename)
            species_call_types[call_type] += 1
            results['overall_counts'][call_type] += 1

        # Lưu kết quả theo loài
        if species_call_types:
            results['call_types'][species] = dict(species_call_types)

    return results


# Thực thi hàm phân tích
call_type_results = analyze_call_types(species_list, species_dir_mapping, cfg)

if call_type_results['overall_counts']:
    # Tạo DataFrame chứa kết quả
    call_counts = pd.DataFrame({
        'call_type': list(call_type_results['overall_counts'].keys()),
        'count': list(call_type_results['overall_counts'].values())
    }).sort_values('count', ascending=False)

    print("\nPhân bố các loại tiếng kêu:")
    display(call_counts)

    # Vẽ biểu đồ cột
    plt.figure(figsize=(12, 8))
    sns.barplot(x='count', y='call_type', data=call_counts)
    plt.title('Phân bố các loại tiếng kêu của chim')
    plt.xlabel('Số lượng')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Vẽ biểu đồ tròn
    plt.figure(figsize=(10, 10))
    plt.pie(call_counts['count'], labels=call_counts['call_type'],
            autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Tỉ lệ các loại tiếng kêu của chim')
    plt.tight_layout()
    plt.show()
else:
    print("Không có loại tiếng kêu nào được phân tích")


# Chuẩn bị dữ liệu cho biểu đồ sunburst
if call_type_results['call_types']:
    # Tạo danh sách các từ điển (dict) chứa dữ liệu
    sunburst_data = []
    
    for species, call_types in call_type_results['call_types'].items():
        for call_type, count in call_types.items():
            sunburst_data.append({
                'species': species,        # Tên loài chim
                'call_type': call_type,    # Loại tiếng kêu (song, call, alarm,...)
                'count': count             # Số lượng mẫu có loại tiếng kêu đó
            })
    
    # Chuyển dữ liệu thành DataFrame
    sunburst_df = pd.DataFrame(sunburst_data)
    
    # Vẽ biểu đồ Sunburst
    fig = px.sunburst(
        sunburst_df,
        path=['call_type', 'species'],  # Cấp 1 là call_type, cấp 2 là species
        values='count',                 # Dựa trên số lượng file
        title='Biểu đồ Sunburst: Phân bố loại tiếng kêu theo loài chim',
        width=800,
        height=800
    )
    
    fig.update_layout(margin=dict(t=30, b=30, l=0, r=0))
    fig.show()
else:
    print("Không có dữ liệu loại tiếng kêu để trực quan hóa bằng sunburst")


def analyze_quality_ratings(metadata_df):
    """Phân tích đánh giá chất lượng từ dữ liệu metadata"""
    if metadata_df is None or 'rating' not in metadata_df.columns:
        print("Không có đánh giá chất lượng trong metadata")
        return None

    # Tính toán các thống kê mô tả cho cột rating
    quality_stats = metadata_df['rating'].describe()
    print("\nThống kê đánh giá chất lượng:")
    display(quality_stats)

    # Đếm số lượng của từng giá trị đánh giá
    rating_counts = metadata_df['rating'].value_counts().sort_index()
    print("\nSố lượng của từng mức đánh giá:")
    display(rating_counts)

    # Vẽ biểu đồ phân bố các đánh giá
    plt.figure(figsize=(10, 6))
    sns.countplot(x='rating', data=metadata_df, palette='viridis')
    plt.title('Phân bố mức đánh giá chất lượng âm thanh')
    plt.xlabel('Đánh giá (0=không rõ, 1=thấp, 5=cao)')
    plt.ylabel('Số lượng')
    plt.grid(True, alpha=0.3)
    plt.show()

    # Tính trung bình đánh giá theo loài
    species_ratings = metadata_df.groupby('primary_label')['rating'].agg(['mean', 'count']).reset_index()
    species_ratings = species_ratings.sort_values('mean', ascending=False)

    print("\nNhững loài có điểm đánh giá trung bình cao nhất:")
    display(species_ratings.head(10))

    print("\nNhững loài có điểm đánh giá trung bình thấp nhất:")
    display(species_ratings.tail(10))

    return species_ratings

# Thực hiện phân tích đánh giá chất lượng nếu metadata có sẵn
species_ratings = None
if metadata_df is not None:
    species_ratings = analyze_quality_ratings(metadata_df)

# Hàm để tính toán đánh giá chất lượng của riêng chúng ta trên một mẫu tệp âm thanh
def analyze_computed_quality(species_list, species_dir_mapping, cfg):
    """Tính toán đánh giá chất lượng của chính hệ thống trên các tệp âm thanh"""
    results = {
        'quality_scores': [],       # Danh sách điểm chất lượng từng tệp
        'species_scores': {}        # Điểm theo từng loài
    }

    # Chọn một tập con loài để phân tích nếu đang ở chế độ debug
    if cfg.DEBUG_MODE:
        if len(species_list) > 15:  # Phân tích tối đa 15 loài khi debug
            species_subset = np.random.choice(species_list, 15, replace=False).tolist()
        else:
            species_subset = species_list
    else:
        species_subset = species_list

    print(f"Đang phân tích chất lượng đã tính toán cho {len(species_subset)} loài")

    for species_idx, species in enumerate(species_subset):
        # Lấy thư mục chứa âm thanh của loài
        species_dir = species_dir_mapping.get(species)
        if not species_dir or not os.path.exists(species_dir):
            continue

        # Lấy danh sách các tệp âm thanh của loài
        audio_files = [f for f in os.listdir(species_dir)
                       if f.endswith(('.mp3', '.ogg', '.wav'))]

        # Nếu đang ở chế độ debug, chỉ xử lý một mẫu nhỏ
        if cfg.DEBUG_MODE:
            if len(audio_files) > cfg.SAMPLE_SIZE:
                audio_files = np.random.choice(audio_files, cfg.SAMPLE_SIZE, replace=False).tolist()

        species_scores = []

        # Xử lý từng tệp âm thanh
        for filename in tqdm(audio_files, desc=f"Đang tính chất lượng cho {species}"):
            file_path = os.path.join(species_dir, filename)

            # Tải và đánh giá chất lượng âm thanh
            audio = load_audio_file(file_path, sr=cfg.SR)

            if audio is not None:
                # Tính điểm chất lượng
                quality = assess_audio_quality(audio)
                results['quality_scores'].append((species, filename, quality))
                species_scores.append(quality)

        # Lưu kết quả theo loài
        if species_scores:
            results['species_scores'][species] = species_scores

    return results

# Thực hiện phân tích chất lượng đã tính toán
computed_quality_results = analyze_computed_quality(species_list, species_dir_mapping, cfg)

if computed_quality_results['quality_scores']:
    # Trích xuất các điểm đánh giá chất lượng
    quality_scores = [score for _, _, score in computed_quality_results['quality_scores']]

    print("\nThống kê điểm chất lượng đã tính:")
    print(f"- Trung bình: {np.mean(quality_scores):.4f}")
    print(f"- Thấp nhất: {np.min(quality_scores):.4f}")
    print(f"- Cao nhất: {np.max(quality_scores):.4f}")

    # Vẽ biểu đồ histogram của các điểm chất lượng
    plt.figure(figsize=(12, 6))
    plt.hist(quality_scores, bins=50, edgecolor='black')
    plt.title('Phân bố điểm chất lượng âm thanh đã tính')
    plt.xlabel('Điểm chất lượng (0-1)')
    plt.ylabel('Số lượng')
    plt.grid(True, alpha=0.3)
    plt.show()

    # Tính điểm chất lượng trung bình theo loài
    species_quality = {}
    for species, scores in computed_quality_results['species_scores'].items():
        if scores:
            species_quality[species] = np.mean(scores)

    # Chuyển đổi sang DataFrame
    species_quality_df = pd.DataFrame({
        'species': list(species_quality.keys()),
        'avg_quality': list(species_quality.values())
    }).sort_values('avg_quality', ascending=False)

    # Vẽ biểu đồ chất lượng trung bình theo loài
    plt.figure(figsize=(14, 10))
    sns.barplot(x='avg_quality', y='species', data=species_quality_df)
    plt.title('Điểm chất lượng trung bình theo loài (tính toán)')
    plt.xlabel('Điểm chất lượng trung bình')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def analyze_frequency_patterns(species_list, species_dir_mapping, cfg):
    """Phân tích các đặc trưng tần số trong các tệp âm thanh"""
    results = {
        'frequency_data': {}  # Lưu dữ liệu tần số theo loài
    }

    # Chọn một tập con loài để phân tích nếu ở chế độ debug
    if cfg.DEBUG_MODE:
        if len(species_list) > 10:  # Phân tích tối đa 10 loài khi debug
            species_subset = np.random.choice(species_list, 10, replace=False).tolist()
        else:
            species_subset = species_list
    else:
        species_subset = species_list

    print(f"Đang phân tích đặc trưng tần số cho {len(species_subset)} loài")

    for species_idx, species in enumerate(species_subset):
        # Lấy thư mục âm thanh tương ứng với loài
        species_dir = species_dir_mapping.get(species)
        if not species_dir or not os.path.exists(species_dir):
            continue

        # Lấy danh sách tệp âm thanh của loài
        audio_files = [f for f in os.listdir(species_dir)
                       if f.endswith(('.mp3', '.ogg', '.wav'))]

        # Nếu đang ở chế độ debug, chỉ xử lý một mẫu nhỏ
        if cfg.DEBUG_MODE:
            if len(audio_files) > cfg.SAMPLE_SIZE:
                audio_files = np.random.choice(audio_files, cfg.SAMPLE_SIZE // 2, replace=False).tolist()

        species_freq_data = []

        # Xử lý từng tệp âm thanh
        for filename in tqdm(audio_files, desc=f"Phân tích tần số cho {species}"):
            file_path = os.path.join(species_dir, filename)

            # Tải và phân tích tệp âm thanh
            audio = load_audio_file(file_path, sr=cfg.SR)

            if audio is not None:
                # Trích xuất đặc trưng tần số
                freq_data = extract_frequency_characteristics(audio, sr=cfg.SR)
                if freq_data:
                    freq_data['species'] = species
                    freq_data['filename'] = filename
                    species_freq_data.append(freq_data)

        # Lưu kết quả theo loài
        if species_freq_data:
            results['frequency_data'][species] = species_freq_data

    return results

# Chạy phân tích mẫu tần số
freq_pattern_results = analyze_frequency_patterns(species_list, species_dir_mapping, cfg)

if freq_pattern_results['frequency_data']:
    # Trích xuất dữ liệu đặc trưng tần số cho từng loài
    species_freq_data = {}
    for species, freq_list in freq_pattern_results['frequency_data'].items():
        if freq_list:
            species_freq_data[species] = {
                'dominant_frequency': [item['dominant_frequency'] for item in freq_list],
                'spectral_centroid': [item['spectral_centroid'] for item in freq_list],
                'spectral_bandwidth': [item['spectral_bandwidth'] for item in freq_list],
                'spectral_flatness': [item['spectral_flatness'] for item in freq_list]
            }

    # Tính giá trị trung vị cho mỗi đặc trưng theo loài
    median_values = {}
    for species, freq_data in species_freq_data.items():
        median_values[species] = {
            'dominant_frequency': np.median(freq_data['dominant_frequency']),
            'spectral_centroid': np.median(freq_data['spectral_centroid']),
            'spectral_bandwidth': np.median(freq_data['spectral_bandwidth']),
            'spectral_flatness': np.median(freq_data['spectral_flatness'])
        }

    # Chuyển đổi sang DataFrame
    freq_df = pd.DataFrame.from_dict(median_values, orient='index')

    print("\nĐặc trưng tần số theo từng loài:")
    display(freq_df.head())

    # Vẽ biểu đồ cho từng đặc trưng tần số
    for characteristic in ['dominant_frequency', 'spectral_centroid', 'spectral_bandwidth', 'spectral_flatness']:
        plt.figure(figsize=(14, 8))
        sorted_df = freq_df.sort_values(characteristic, ascending=False)
        plt.barh(range(len(sorted_df)), sorted_df[characteristic])
        plt.yticks(range(len(sorted_df)), sorted_df.index)
        plt.title(f'Trung vị {characteristic.replace("_", " ").title()} theo loài')
        plt.xlabel('Giá trị (Hz đối với tần số)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def analyze_time_series(species_list, species_dir_mapping, cfg):
    """Phân tích chuỗi thời gian trong tiếng hót của các loài chim"""
    
    # Chọn một loài để phân tích chi tiết
    if species_list and species_dir_mapping:
        # Cố gắng tìm một loài có mẫu tiếng hót rõ ràng
        target_species = None
        for species in ['norspe1', 'comyel', 'rebnut', 'amerob', 'houspa']:
            if species in species_dir_mapping:
                target_species = species
                break

        # Nếu không tìm được, chọn loài đầu tiên trong danh sách
        if target_species is None and species_list:
            target_species = species_list[0]

        species_dir = species_dir_mapping.get(target_species)
        if not species_dir or not os.path.exists(species_dir):
            print(f"Không tìm thấy thư mục cho loài {target_species}")
            return

        print(f"Đang phân tích chuỗi thời gian cho loài {target_species}")
        
        # Lấy các tệp âm thanh
        audio_files = [f for f in os.listdir(species_dir)
                       if f.endswith(('.mp3', '.ogg', '.wav'))]

        if not audio_files:
            print("Không tìm thấy tệp âm thanh")
            return

        # Chọn một tệp để phân tích
        filename = audio_files[0]
        file_path = os.path.join(species_dir, filename)

        # Tải âm thanh
        audio = load_audio_file(file_path, sr=cfg.SR)

        if audio is None:
            print(f"Không thể tải tệp âm thanh: {file_path}")
            return

        # Vẽ biểu đồ dạng sóng
        plt.figure(figsize=(14, 5))
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(len(audio)) / cfg.SR, audio)
        plt.title(f'Dạng sóng của {target_species} - {filename}')
        plt.xlabel('Thời gian (giây)')
        plt.ylabel('Biên độ')
        plt.grid(True, alpha=0.3)

        # Mật độ phổ công suất
        plt.subplot(2, 1, 2)
        freqs, psd = periodogram(audio, fs=cfg.SR)
        plt.semilogy(freqs, psd)
        plt.title('Mật độ phổ công suất')
        plt.xlabel('Tần số (Hz)')
        plt.ylabel('Công suất/Tần số (dB/Hz)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Trích xuất phong bì năng lượng như một chuỗi thời gian
        def extract_envelope(signal, frame_size=1024, hop_length=512):
            energy = []
            for i in range(0, len(signal) - frame_size, hop_length):
                chunk = signal[i:i + frame_size]
                energy.append(np.sum(chunk ** 2) / frame_size)
            return np.array(energy)

        # Lấy phong bì năng lượng
        envelope = extract_envelope(audio)

        # Vẽ phong bì năng lượng
        plt.figure(figsize=(14, 5))
        plt.plot(np.arange(len(envelope)) * (512 / cfg.SR), envelope)
        plt.title(f'Phong bì năng lượng của {target_species} - {filename}')
        plt.xlabel('Thời gian (giây)')
        plt.ylabel('Năng lượng')
        plt.grid(True, alpha=0.3)
        plt.show()

        # Cố gắng khớp mô hình ARIMA với phong bì
        try:
            # Chỉ lấy mẫu nhỏ để xử lý nhanh hơn
            sample_size = min(len(envelope), 1000)
            sample_envelope = envelope[:sample_size]

            # Khớp mô hình ARIMA
            model = ARIMA(sample_envelope, order=(5, 1, 0))
            model_fit = model.fit()

            # Hiển thị tóm tắt mô hình
            print("\nTóm tắt mô hình ARIMA:")
            print(model_fit.summary())

            # Vẽ mô hình khớp so với dữ liệu thực tế
            plt.figure(figsize=(14, 5))
            plt.plot(sample_envelope, label='Thực tế')
            plt.plot(model_fit.fittedvalues, color='red', label='Dự đoán')
            plt.title(f'Mô hình ARIMA khớp với {target_species}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

            # Dự đoán 100 điểm tiếp theo
            forecast = model_fit.forecast(steps=100)

            # Vẽ biểu đồ dự đoán
            plt.figure(figsize=(14, 5))
            plt.plot(range(len(sample_envelope)), sample_envelope, label='Thực tế')
            plt.plot(range(len(sample_envelope), len(sample_envelope) + 100), forecast, color='red', label='Dự đoán')
            plt.title(f'Dự báo ARIMA cho {target_species}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

        except Exception as e:
            print(f"Lỗi khi khớp mô hình ARIMA: {e}")

# Chạy phân tích chuỗi thời gian
analyze_time_series(species_list, species_dir_mapping, cfg)

