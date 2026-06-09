# ========================================================================
# CELL 1: LIBRARY IMPORTS & ENVIRONMENT SETUP - KAGGLE SAFE
# Robust library imports with memory protection
# ========================================================================

print("NFL BIG DATA BOWL 2026 - BCS FRAMEWORK CHAMPIONSHIP ANALYSIS")
print("="*70)
print("Setting up robust environment for championship submission...")

# =====================================================================
# CORE DATA SCIENCE LIBRARIES
# =====================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import gc
import os
from pathlib import Path
from functools import wraps
import json

# Statistical libraries
try:
    import scipy.stats as stats
    from scipy.stats import pointbiserialr, f_oneway, pearsonr
    SCIPY_AVAILABLE = True
    print("SciPy statistical functions available")
except ImportError:
    SCIPY_AVAILABLE = False
    print("SciPy not available, using basic statistics")

# System libraries for monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
    print("psutil available")
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not available, using basic memory tracking")

# =====================================================================
# ENVIRONMENT DETECTION
# =====================================================================

ENVIRONMENT = 'KAGGLE' if os.path.exists('/kaggle/input') else 'LOCAL'
print(f"Environment: {ENVIRONMENT}")

# Memory limits for Kaggle
MEMORY_LIMIT_GB = 12 if ENVIRONMENT == 'KAGGLE' else 20
SAMPLE_FRAC = 0.12 if ENVIRONMENT == 'KAGGLE' else 0.25
PARQUET_DIR = '/kaggle/working/parquet_cache' if ENVIRONMENT == 'KAGGLE' else './parquet_cache'

print(f"Memory limit: {MEMORY_LIMIT_GB}GB, Sample fraction: {SAMPLE_FRAC*100:.0f}%")

# =====================================================================
# PERFORMANCE MONITORING & TIMING
# =====================================================================

def get_memory_usage():
    """Get current memory usage in MB"""
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except:
            pass
    return 0

def get_memory_gb():
    """Get current memory usage in GB"""
    return get_memory_usage() / 1024

def aggressive_gc():
    """Aggressive garbage collection"""
    gc.collect()
    gc.collect()
    gc.collect()

def check_memory_safe(context=""):
    """Check if memory is safe to continue"""
    current = get_memory_gb()
    if current >= MEMORY_LIMIT_GB:
        print(f"   MEMORY LIMIT: {current:.2f}GB [{context}]")
        return False
    if current >= MEMORY_LIMIT_GB - 2:
        print(f"   Memory warning: {current:.2f}GB")
    return True

def timing_decorator(func):
    """Professional timing decorator with memory tracking"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = get_memory_usage()
        
        print(f"\n Executing {func.__name__}...")
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = get_memory_usage()
        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory
        
        print(f"   Time: {func.__name__}: {execution_time:.2f}s")
        print(f"   Memory: {end_memory:.1f}MB ({memory_delta:+.1f}MB)")
        
        return result
    return wrapper

# =====================================================================
# DATA TYPE OPTIMIZATION
# =====================================================================

def optimize_data_types(df):
    """Enhanced data type optimization for memory efficiency"""
    if df is None or len(df) == 0:
        return df
    
    for col in df.columns:
        col_type = df[col].dtype
        
        try:
            if col_type == 'object':
                numeric_df = pd.to_numeric(df[col], errors='ignore')
                if numeric_df.dtype != 'object':
                    df[col] = numeric_df
                    col_type = df[col].dtype
                else:
                    if df[col].nunique() < len(df) * 0.1:
                        df[col] = df[col].astype('category')
                    continue
            
            if 'int' in str(col_type):
                c_min = df[col].min()
                c_max = df[col].max()
                
                if pd.isna(c_min) or pd.isna(c_max):
                    continue
                
                if c_min >= 0:
                    if c_max <= 255:
                        df[col] = df[col].astype('uint8')
                    elif c_max <= 65535:
                        df[col] = df[col].astype('uint16')
                    elif c_max <= 4294967295:
                        df[col] = df[col].astype('uint32')
                else:
                    if c_min >= -128 and c_max <= 127:
                        df[col] = df[col].astype('int8')
                    elif c_min >= -32768 and c_max <= 32767:
                        df[col] = df[col].astype('int16')
                    elif c_min >= -2147483648 and c_max <= 2147483647:
                        df[col] = df[col].astype('int32')
            
            elif 'float' in str(col_type):
                df[col] = pd.to_numeric(df[col], downcast='float')
                
        except Exception:
            continue
    
    return df

# =====================================================================
# VISUALIZATION SETUP
# =====================================================================

plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.titlesize'] = 14

COLORS = {
    'blue': '#1f77b4', 'red': '#d62728', 'green': '#2ca02c',
    'orange': '#ff7f0e', 'purple': '#9467bd', 'brown': '#8c564b',
    'pink': '#e377c2', 'gray': '#7f7f7f', 'olive': '#bcbd22',
    'cyan': '#17becf', 'gold': '#FFD700', 'elite': '#FFD700'
}

# =====================================================================
# NFL CONSTANTS
# =====================================================================

FIELD_LENGTH = 120
FIELD_WIDTH = 53.3

DEFENSIVE_POSITIONS = {
    'CB': 'Cornerback', 'CORNERBACK': 'CB', 'CORNER': 'CB',
    'S': 'Safety', 'SAFETY': 'S',
    'FS': 'Free Safety', 'FREE SAFETY': 'FS', 'FREE_SAFETY': 'FS',
    'SS': 'Strong Safety', 'STRONG SAFETY': 'SS', 'STRONG_SAFETY': 'SS',
    'LB': 'Linebacker', 'LINEBACKER': 'LB',
    'MLB': 'Middle Linebacker', 'MIDDLE LINEBACKER': 'MLB',
    'OLB': 'Outside Linebacker', 'OUTSIDE LINEBACKER': 'OLB',
    'ILB': 'Inside Linebacker', 'INSIDE LINEBACKER': 'ILB',
    'DE': 'Defensive End', 'DEFENSIVE END': 'DE',
    'DT': 'Defensive Tackle', 'DEFENSIVE TACKLE': 'DT',
    'NT': 'Nose Tackle', 'NOSE TACKLE': 'NT'
}

OFFENSIVE_POSITIONS = {
    'WR': 'Wide Receiver', 'WIDE RECEIVER': 'WR', 'WIDE_RECEIVER': 'WR',
    'TE': 'Tight End', 'TIGHT END': 'TE', 'TIGHT_END': 'TE',
    'RB': 'Running Back', 'RUNNING BACK': 'RB', 'RUNNING_BACK': 'RB',
    'FB': 'Fullback', 'FULLBACK': 'FB',
    'QB': 'Quarterback', 'QUARTERBACK': 'QB'
}

# =====================================================================
# GLOBAL FRAMEWORK CONFIGURATION
# =====================================================================

championship_data = {}
championship_bcs_results = {}

CHAMPIONSHIP_CONFIG = {
    'version': 'BCS v2.0 Championship Edition',
    'target_execution_time': '15-25 minutes',
    'optimization_level': 'Kaggle Memory-Safe',
    'innovation_focus': 'Ball Convergence Score + Motion Analysis'
}

warnings.filterwarnings('ignore')

print(f"\nCHAMPIONSHIP ENVIRONMENT CONFIGURED:")
print(f"   Framework: {CHAMPIONSHIP_CONFIG['version']}")
print(f"   Target: {CHAMPIONSHIP_CONFIG['target_execution_time']}")
print(f"   Focus: {CHAMPIONSHIP_CONFIG['innovation_focus']}")
print(f"   Initial memory: {get_memory_usage():.1f} MB")

print("CELL 1 COMPLETE - ROBUST ENVIRONMENT SETUP!")


# ========================================================================
# CELL 2: MEMORY-SAFE DATA LOADING - ALL 37 FILES
# Load ALL files with parquet cache
# ========================================================================

print("\nMEMORY-SAFE DATA LOADING - ALL FILES")
print("="*55)

os.makedirs(PARQUET_DIR, exist_ok=True)

# INCREASED LIMITS - load ALL files!
MAX_TRACKING_FILES = 50  # Was 18, now 50 to get all
SAMPLE_FRAC_INPUT = 0.20  # 20% for large input files
SAMPLE_FRAC_OUTPUT = 1.0  # 100% for small output files (keep all)

def merge_two_parquets_on_disk(path1, path2, output_path, is_tracking=True):
    """Merge two parquet files on disk to save memory"""
    try:
        df1 = pd.read_parquet(path1)
        df2 = pd.read_parquet(path2)
        
        if is_tracking:
            combined = pd.concat([df1, df2], ignore_index=True)
        else:
            common = set(df1.columns) & set(df2.columns)
            keys = [c for c in ['game_id', 'play_id', 'player_id'] if c in common]
            
            if keys and len(df2) < len(df1) * 0.5:
                combined = df1.merge(df2, on=keys, how='left', suffixes=('', '_r'))
            else:
                combined = pd.concat([df1, df2], ignore_index=True)
        
        combined.to_parquet(output_path, index=False, compression='snappy')
        rows = len(combined)
        
        del df1, df2, combined
        aggressive_gc()
        
        return rows
    except Exception as e:
        print(f"      Merge error: {e}")
        import shutil
        if os.path.exists(path1):
            shutil.copy(path1, output_path)
        return 0

def standardize_columns(df):
    """Standardize column names for compatibility"""
    if df is None or len(df) == 0:
        return df
    
    mapping = {
        'gameId': 'game_id', 'playId': 'play_id', 'nflId': 'player_id',
        'frameId': 'frame_id', 'displayName': 'player_name',
        'officialPosition': 'position', 'club': 'team',
        'pff_role': 'role_category', 's': 'speed', 'a': 'acceleration'
    }
    
    rename = {k: v for k, v in mapping.items() if k in df.columns}
    if rename:
        df = df.rename(columns=rename)
    
    return df

@timing_decorator
def discover_and_load_complete_data():
    """Load ALL files with smart sampling"""
    
    print(f"\n   Memory: {get_memory_gb():.2f}GB")
    
    # Check for final parquet cache
    final_supp = os.path.join(PARQUET_DIR, 'FINAL_supplementary.parquet')
    final_track = os.path.join(PARQUET_DIR, 'FINAL_tracking.parquet')
    
    # FORCE RELOAD: Delete old cache to get all files
    # Remove this block after first successful run with all files
    for old_file in Path(PARQUET_DIR).glob('*.pq'):
        os.remove(old_file)
    for old_file in Path(PARQUET_DIR).glob('*.parquet'):
        os.remove(old_file)
    print(f"   Cleared old cache to reload ALL files")
    
    # Uncomment below after first successful run:
    # if os.path.exists(final_supp) and os.path.exists(final_track):
    #     print(f"\n   FAST PATH: Loading from Parquet cache!")
    #     supplementary = pd.read_parquet(final_supp)
    #     tracking = pd.read_parquet(final_track)
    #     print(f"      supplementary: {len(supplementary):,} rows")
    #     print(f"      tracking: {len(tracking):,} rows")
    #     return {
    #         'supplementary': supplementary,
    #         'combined_tracking': tracking,
    #         'file_info': [],
    #         'loading_stats': [{'source': 'parquet_cache'}]
    #     }, {}, {}
    
    # =========================================================
    # STEP 1: FILE DISCOVERY
    # =========================================================
    
    print(f"\n   STEP 1: FILE DISCOVERY")
    
    DATA_PATH = None
    
    if ENVIRONMENT == 'KAGGLE' and os.path.exists('/kaggle/input'):
        for item in os.listdir('/kaggle/input'):
            candidate = f'/kaggle/input/{item}'
            if os.path.isdir(candidate):
                DATA_PATH = Path(candidate)
                print(f"      Found Kaggle data: {DATA_PATH}")
                break
    
    if DATA_PATH is None:
        for p in ['.', './data', './input', '../input']:
            if os.path.exists(p) and list(Path(p).rglob('*.csv')):
                DATA_PATH = Path(p)
                break
    
    if DATA_PATH is None:
        print(f"      NO DATA FOUND!")
        return {'supplementary': pd.DataFrame(), 'combined_tracking': pd.DataFrame(), 'file_info': []}, {}, {}
    
    # Find all CSV files recursively
    all_csv_files = list(DATA_PATH.rglob('*.csv'))
    print(f"      TOTAL CSV FILES: {len(all_csv_files)}")
    
    # =========================================================
    # STEP 2: FILE CATEGORIZATION (input vs output vs supp)
    # =========================================================
    
    print(f"\n   STEP 2: FILE CATEGORIZATION")
    
    file_categories = {
        'supplementary': [],
        'input_tracking': [],   # Large files ~40-50MB
        'output_tracking': []   # Small files ~1MB
    }
    file_info_list = []
    
    for file_path in all_csv_files:
        filename = file_path.name.lower()
        size_mb = file_path.stat().st_size / (1024 * 1024)
        
        file_info = {
            'filename': file_path.name,
            'path': file_path,
            'size_mb': size_mb,
            'category': 'unknown'
        }
        
        # Categorize by name
        if 'supplementary' in filename or 'play' in filename or 'game' in filename:
            file_categories['supplementary'].append(file_info)
            file_info['category'] = 'supplementary'
        elif 'input' in filename:
            file_categories['input_tracking'].append(file_info)
            file_info['category'] = 'input_tracking'
        elif 'output' in filename:
            file_categories['output_tracking'].append(file_info)
            file_info['category'] = 'output_tracking'
        elif 'tracking' in filename or 'week' in filename:
            if size_mb > 10:
                file_categories['input_tracking'].append(file_info)
                file_info['category'] = 'input_tracking'
            else:
                file_categories['output_tracking'].append(file_info)
                file_info['category'] = 'output_tracking'
        else:
            # Default to supplementary for unknown
            file_categories['supplementary'].append(file_info)
            file_info['category'] = 'supplementary'
        
        file_info_list.append(file_info)
    
    # Print summary
    supp_size = sum(f['size_mb'] for f in file_categories['supplementary'])
    input_size = sum(f['size_mb'] for f in file_categories['input_tracking'])
    output_size = sum(f['size_mb'] for f in file_categories['output_tracking'])
    
    print(f"      Supplementary: {len(file_categories['supplementary'])} files ({supp_size:.1f}MB)")
    print(f"      Input tracking: {len(file_categories['input_tracking'])} files ({input_size:.1f}MB)")
    print(f"      Output tracking: {len(file_categories['output_tracking'])} files ({output_size:.1f}MB)")
    print(f"      TOTAL: {len(all_csv_files)} files ({supp_size + input_size + output_size:.1f}MB)")
    
    # =========================================================
    # STEP 3: CONVERT ALL FILES TO PARQUET
    # =========================================================
    
    print(f"\n   STEP 3: CONVERTING ALL FILES TO PARQUET")
    
    supp_parquets = []
    track_parquets = []
    loading_stats = []
    
    # --- Supplementary files (load 100%) ---
    print(f"\n      --- Supplementary ({len(file_categories['supplementary'])} files, 100%) ---")
    
    for i, f in enumerate(file_categories['supplementary']):
        pq_path = os.path.join(PARQUET_DIR, f"s_{i}.pq")
        
        try:
            print(f"      {f['filename']} ({f['size_mb']:.1f}MB)...", end=' ')
            
            df = pd.read_csv(f['path'])
            df = optimize_data_types(df)
            df = standardize_columns(df)
            df['source_file'] = f['filename']
            
            df.to_parquet(pq_path, index=False, compression='snappy')
            pq_size = os.path.getsize(pq_path)/(1024*1024)
            
            print(f"OK {len(df):,}r -> {pq_size:.1f}MB")
            supp_parquets.append(pq_path)
            
            loading_stats.append({
                'category': 'supplementary',
                'filename': f['filename'],
                'rows': len(df)
            })
            
            del df
            aggressive_gc()
            
        except Exception as e:
            print(f"FAILED: {e}")
    
    # --- Output tracking files (load 100% - small files) ---
    print(f"\n      --- Output tracking ({len(file_categories['output_tracking'])} files, 100%) ---")
    
    for i, f in enumerate(sorted(file_categories['output_tracking'], key=lambda x: x['size_mb'])):
        pq_path = os.path.join(PARQUET_DIR, f"to_{i}.pq")
        
        try:
            print(f"      {f['filename']} ({f['size_mb']:.1f}MB)...", end=' ')
            
            df = pd.read_csv(f['path'])
            df = optimize_data_types(df)
            df = standardize_columns(df)
            df['source_file'] = f['filename']
            df['data_type'] = 'output'
            
            df.to_parquet(pq_path, index=False, compression='snappy')
            pq_size = os.path.getsize(pq_path)/(1024*1024)
            
            print(f"OK {len(df):,}r -> {pq_size:.1f}MB")
            track_parquets.append(pq_path)
            
            loading_stats.append({
                'category': 'output_tracking',
                'filename': f['filename'],
                'rows': len(df)
            })
            
            del df
            aggressive_gc()
            
        except Exception as e:
            print(f"FAILED: {e}")
    
    # --- Input tracking files (sample 20% - large files) ---
    print(f"\n      --- Input tracking ({len(file_categories['input_tracking'])} files, {SAMPLE_FRAC_INPUT*100:.0f}%) ---")
    
    for i, f in enumerate(sorted(file_categories['input_tracking'], key=lambda x: x['size_mb'])):
        pq_path = os.path.join(PARQUET_DIR, f"ti_{i}.pq")
        
        mem = get_memory_gb()
        if mem > MEMORY_LIMIT_GB:
            print(f"      Memory limit reached ({mem:.1f}GB)!")
            break
        
        try:
            print(f"      {f['filename']} ({f['size_mb']:.0f}MB) [M:{mem:.1f}G]...", end=' ')
            
            # Chunked sampling for large files
            chunks = []
            rows_loaded = 0
            
            for chunk in pd.read_csv(f['path'], chunksize=200000):
                sampled = chunk.sample(frac=SAMPLE_FRAC_INPUT, random_state=42)
                sampled = optimize_data_types(sampled)
                chunks.append(sampled)
                rows_loaded += len(sampled)
                
                if get_memory_gb() > MEMORY_LIMIT_GB - 2:
                    print(f"mem limit...", end=' ')
                    break
            
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                del chunks
            else:
                df = pd.DataFrame()
            
            if len(df) == 0:
                print("empty")
                continue
            
            df = standardize_columns(df)
            df['source_file'] = f['filename']
            df['data_type'] = 'input'
            
            df.to_parquet(pq_path, index=False, compression='snappy')
            pq_size = os.path.getsize(pq_path)/(1024*1024)
            
            print(f"OK {len(df):,}r -> {pq_size:.1f}MB")
            track_parquets.append(pq_path)
            
            loading_stats.append({
                'category': 'input_tracking',
                'filename': f['filename'],
                'rows': len(df)
            })
            
            del df
            aggressive_gc()
            
        except Exception as e:
            print(f"FAILED: {e}")
            aggressive_gc()
    
    print(f"\n   Memory after conversions: {get_memory_gb():.2f}GB")
    print(f"   Parquets created: {len(supp_parquets)} supp, {len(track_parquets)} track")
    
    # =========================================================
    # STEP 4: INCREMENTAL MERGE ON DISK
    # =========================================================
    
    print(f"\n   STEP 4: INCREMENTAL MERGE ON DISK")
    
    # Merge supplementary
    print(f"\n      --- Merging supplementary ({len(supp_parquets)} files) ---")
    
    if len(supp_parquets) == 0:
        supp_final_path = os.path.join(PARQUET_DIR, 'supp_merged.pq')
        pd.DataFrame().to_parquet(supp_final_path)
    elif len(supp_parquets) == 1:
        supp_final_path = supp_parquets[0]
    else:
        current = supp_parquets[0]
        for i, next_pq in enumerate(supp_parquets[1:], 1):
            output = os.path.join(PARQUET_DIR, f'supp_m{i}.pq')
            print(f"      Merging {i}/{len(supp_parquets)-1}...", end=' ')
            rows = merge_two_parquets_on_disk(current, next_pq, output, is_tracking=False)
            print(f"OK {rows:,} rows")
            current = output
            aggressive_gc()
        supp_final_path = current
    
    # Merge tracking
    print(f"\n      --- Merging tracking ({len(track_parquets)} files) ---")
    
    if len(track_parquets) == 0:
        track_final_path = os.path.join(PARQUET_DIR, 'track_merged.pq')
        pd.DataFrame().to_parquet(track_final_path)
    elif len(track_parquets) == 1:
        track_final_path = track_parquets[0]
    else:
        current = track_parquets[0]
        for i, next_pq in enumerate(track_parquets[1:], 1):
            output = os.path.join(PARQUET_DIR, f'track_m{i}.pq')
            
            mem = get_memory_gb()
            
            # Print every 5th merge or last one
            if i % 5 == 0 or i == len(track_parquets) - 1:
                print(f"      Merging {i}/{len(track_parquets)-1} [M:{mem:.1f}G]...", end=' ')
                rows = merge_two_parquets_on_disk(current, next_pq, output, is_tracking=True)
                print(f"OK {rows:,} rows")
            else:
                rows = merge_two_parquets_on_disk(current, next_pq, output, is_tracking=True)
            
            if mem > MEMORY_LIMIT_GB:
                print(f"      Memory limit - stopping merge")
                break
            
            current = output
            aggressive_gc()
        track_final_path = current
    
    print(f"\n   Memory after merge: {get_memory_gb():.2f}GB")
    
    # =========================================================
    # STEP 5: SAVE FINAL AND LOAD
    # =========================================================
    
    print(f"\n   STEP 5: SAVING FINAL PARQUETS")
    
    import shutil
    
    if os.path.exists(supp_final_path):
        shutil.copy(supp_final_path, final_supp)
        size = os.path.getsize(final_supp)/(1024*1024)
        print(f"      FINAL_supplementary.parquet ({size:.1f}MB)")
    
    if os.path.exists(track_final_path):
        shutil.copy(track_final_path, final_track)
        size = os.path.getsize(final_track)/(1024*1024)
        print(f"      FINAL_tracking.parquet ({size:.1f}MB)")
    
    # Load final data
    print(f"\n   Loading final data...")
    
    aggressive_gc()
    
    if os.path.exists(final_supp):
        supplementary = pd.read_parquet(final_supp)
        print(f"      supplementary: {len(supplementary):,} rows x {len(supplementary.columns)} cols")
    else:
        supplementary = pd.DataFrame()
    
    aggressive_gc()
    
    if os.path.exists(final_track):
        tracking = pd.read_parquet(final_track)
        print(f"      tracking: {len(tracking):,} rows x {len(tracking.columns)} cols")
    else:
        tracking = pd.DataFrame()
    
    # Build data summary
    data_summary = {}
    
    for name, df in [('supplementary', supplementary), ('combined_tracking', tracking)]:
        if len(df) > 0:
            data_summary[name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'key_columns': [c for c in ['game_id', 'play_id', 'player_id', 'frame_id'] if c in df.columns],
                'sample_columns': list(df.columns)[:15]
            }
    
    return {
        'supplementary': supplementary,
        'combined_tracking': tracking,
        'file_info': file_info_list,
        'loading_stats': loading_stats
    }, data_summary, file_categories

# Execute data loading
championship_data, data_summary, file_structure = discover_and_load_complete_data()

aggressive_gc()

print(f"\nMemory after data loading: {get_memory_usage():.1f} MB")
print("="*55)

# Detailed validation
print(f"\nFINAL DATASET VERIFICATION:")
print(f"="*55)

for dataset_name, df in championship_data.items():
    if isinstance(df, pd.DataFrame) and len(df) > 0:
        mem = df.memory_usage(deep=True).sum()/(1024*1024)
        print(f"\n{dataset_name}:")
        print(f"   Shape: {len(df):,} rows x {len(df.columns)} cols")
        print(f"   Memory: {mem:.1f}MB")
        print(f"   Columns: {list(df.columns)[:10]}...")
        
        # Show data sources
        if 'source_file' in df.columns:
            sources = df['source_file'].nunique()
            print(f"   Source files: {sources}")
        
        # Show data types if tracking
        if 'data_type' in df.columns:
            type_dist = df['data_type'].value_counts()
            print(f"   Data types: {dict(type_dist)}")
            
    elif dataset_name in ['file_info', 'loading_stats']:
        print(f"\n{dataset_name}: {len(df)} items")

# Summary
total_records = sum(len(df) for df in championship_data.values() if isinstance(df, pd.DataFrame))
files_loaded = len(championship_data.get('loading_stats', []))

print(f"\n" + "="*55)
print(f"SUMMARY:")
print(f"   Files loaded: {files_loaded}")
print(f"   Total records: {total_records:,}")
print(f"   Memory used: {get_memory_usage():.1f}MB")

if total_records > 100000:
    print(f"   STATUS: CHAMPIONSHIP DATASET READY!")
else:
    print(f"   STATUS: Dataset loaded (consider increasing sample rate)")

print(f"\nTIP: Comment out cache clearing code for instant reload!")


# ========================================================================
# CELL 3: ENHANCED SUPPLEMENTARY INTEGRATION - KAGGLE SAFE
# Advanced integration with memory protection
# ========================================================================

print("\nENHANCED SUPPLEMENTARY INTEGRATION")
print("="*55)

@timing_decorator
def enhance_comprehensive_supplementary_integration_fixed():
    """Enhanced supplementary integration with memory protection"""
    
    print(f"\n   Memory: {get_memory_gb():.2f}GB")
    
    if not championship_data:
        print("      No championship data available")
        return None
    
    supplementary_df = championship_data.get('supplementary', pd.DataFrame())
    tracking_df = championship_data.get('combined_tracking', pd.DataFrame())
    file_info = championship_data.get('file_info', [])
    
    print(f"      Processing comprehensive dataset:")
    print(f"         Supplementary records: {len(supplementary_df):,}")
    print(f"         Tracking records: {len(tracking_df):,}")
    print(f"         Source files: {len(file_info)}")
    
    if len(supplementary_df) == 0 and len(tracking_df) == 0:
        print("      No data available for enhancement")
        return None
    
    # =====================================================================
    # STEP 1: COMPREHENSIVE VARIABLE DISCOVERY
    # =====================================================================
    
    print(f"\n      STEP 1: VARIABLE DISCOVERY")
    
    all_columns = set()
    
    if len(supplementary_df) > 0:
        all_columns.update(supplementary_df.columns)
        print(f"         Supplementary columns: {list(supplementary_df.columns)[:10]}...")
    
    if len(tracking_df) > 0:
        all_columns.update(tracking_df.columns)
        print(f"         Tracking columns: {list(tracking_df.columns)[:10]}...")
    
    print(f"         Total unique columns: {len(all_columns)}")
    
    # =====================================================================
    # STEP 2: SMART PLAYER ID MAPPING
    # =====================================================================
    
    print(f"\n      STEP 2: PLAYER ID MAPPING")
    
    supplementary_player_cols = []
    tracking_player_cols = []
    
    if len(supplementary_df) > 0:
        for col in supplementary_df.columns:
            col_lower = col.lower()
            if ('player' in col_lower and 'id' in col_lower) or ('nfl' in col_lower and 'id' in col_lower):
                supplementary_player_cols.append(col)
    
    if len(tracking_df) > 0:
        for col in tracking_df.columns:
            col_lower = col.lower()
            if ('player' in col_lower and 'id' in col_lower) or ('nfl' in col_lower and 'id' in col_lower):
                tracking_player_cols.append(col)
    
    print(f"         Player ID columns:")
    print(f"            Supplementary: {supplementary_player_cols}")
    print(f"            Tracking: {tracking_player_cols}")
    
    # =====================================================================
    # STEP 3: INTELLIGENT DATA CONSOLIDATION
    # =====================================================================
    
    print(f"\n      STEP 3: DATA CONSOLIDATION")
    
    if len(supplementary_df) > 0:
        enhanced_supplementary = supplementary_df.copy()
        base_dataset = 'supplementary'
        print(f"         Base dataset: {base_dataset} ({len(enhanced_supplementary):,} records)")
        
        if supplementary_player_cols:
            primary_player_col = supplementary_player_cols[0]
            if primary_player_col != 'player_id':
                enhanced_supplementary['player_id'] = enhanced_supplementary[primary_player_col]
                print(f"            Standardized player ID from: {primary_player_col}")
    
    elif len(tracking_df) > 0:
        print(f"         Creating supplementary from tracking data...")
        
        tracking_player_col = tracking_player_cols[0] if tracking_player_cols else None
        
        if tracking_player_col:
            groupby_cols = ['game_id', 'play_id']
            if tracking_player_col in tracking_df.columns:
                groupby_cols.append(tracking_player_col)
            
            agg_dict = {}
            if 'frame_id' in tracking_df.columns:
                agg_dict['frame_id'] = ['count', 'min', 'max']
            if 'x' in tracking_df.columns:
                agg_dict['x'] = ['mean', 'std', 'min', 'max']
            if 'y' in tracking_df.columns:
                agg_dict['y'] = ['mean', 'std', 'min', 'max']
            
            if agg_dict:
                tracking_summary = tracking_df.groupby(groupby_cols).agg(agg_dict).reset_index()
                
                new_columns = []
                for col in tracking_summary.columns:
                    if isinstance(col, tuple):
                        new_columns.append('_'.join([str(c) for c in col if c != '']))
                    else:
                        new_columns.append(col)
                tracking_summary.columns = new_columns
                
                if tracking_player_col != 'player_id' and tracking_player_col in tracking_summary.columns:
                    tracking_summary['player_id'] = tracking_summary[tracking_player_col]
                
                enhanced_supplementary = tracking_summary
                base_dataset = 'tracking_derived'
                print(f"            Created from tracking: {len(enhanced_supplementary):,} records")
            else:
                enhanced_supplementary = pd.DataFrame()
                base_dataset = 'empty'
        else:
            enhanced_supplementary = pd.DataFrame()
            base_dataset = 'empty'
    else:
        enhanced_supplementary = pd.DataFrame()
        base_dataset = 'empty'
    
    # =====================================================================
    # STEP 4: POSITION MAPPING
    # =====================================================================
    
    print(f"\n      STEP 4: POSITION MAPPING")
    
    position_mapped = False
    position_sources = []
    
    if len(enhanced_supplementary) > 0:
        position_cols = [col for col in enhanced_supplementary.columns if 'position' in col.lower()]
        if position_cols:
            position_sources.append(('supplementary', position_cols[0], enhanced_supplementary))
    
    if len(tracking_df) > 0:
        position_cols = [col for col in tracking_df.columns if 'position' in col.lower()]
        if position_cols:
            position_sources.append(('tracking', position_cols[0], tracking_df))
    
    print(f"         Found {len(position_sources)} position data sources")
    
    if position_sources and len(enhanced_supplementary) > 0:
        source_name, position_col, source_df = position_sources[0]
        print(f"         Using position data from: {source_name}")
        
        if source_name == 'tracking':
            enhanced_id_cols = [col for col in enhanced_supplementary.columns 
                              if any(term in col.lower() for term in ['player', 'nfl', 'id'])]
            tracking_id_cols = [col for col in source_df.columns 
                              if any(term in col.lower() for term in ['player', 'nfl', 'id'])]
            
            matching_pairs = []
            for enh_col in enhanced_id_cols:
                for track_col in tracking_id_cols:
                    if (enh_col.lower().replace('_', '') == track_col.lower().replace('_', '') or
                        ('nfl' in enh_col.lower() and 'nfl' in track_col.lower()) or
                        ('player' in enh_col.lower() and 'player' in track_col.lower())):
                        matching_pairs.append((enh_col, track_col))
            
            if matching_pairs:
                enh_id_col, track_id_col = matching_pairs[0]
                
                try:
                    position_mapping = source_df.groupby(track_id_col)[position_col].first().reset_index()
                    position_mapping.columns = [enh_id_col, 'position_standardized']
                    
                    print(f"            Position mapping: {len(position_mapping):,} unique players")
                    
                    before_merge = len(enhanced_supplementary)
                    enhanced_supplementary = enhanced_supplementary.merge(
                        position_mapping, on=enh_id_col, how='left'
                    )
                    
                    position_mapped = True
                    mapped_positions = enhanced_supplementary['position_standardized'].notna().sum()
                    mapping_rate = mapped_positions / len(enhanced_supplementary)
                    print(f"            Position mapping rate: {mapping_rate*100:.1f}%")
                    
                except Exception as e:
                    print(f"            Position mapping failed: {e}")
        
        elif source_name == 'supplementary':
            enhanced_supplementary['position_standardized'] = enhanced_supplementary[position_col]
            position_mapped = True
            print(f"            Position data available in supplementary")
    
    if position_mapped and 'position_standardized' in enhanced_supplementary.columns:
        print(f"         Standardizing position names...")
        
        position_mapping_dict = {
            **DEFENSIVE_POSITIONS, **OFFENSIVE_POSITIONS,
            'CORNERBACK': 'CB', 'CORNER': 'CB',
            'SAFETY': 'S', 'FREE_SAFETY': 'FS', 'STRONG_SAFETY': 'SS',
            'LINEBACKER': 'LB', 'MIDDLE_LINEBACKER': 'MLB', 'OUTSIDE_LINEBACKER': 'OLB',
            'WIDE_RECEIVER': 'WR', 'RECEIVER': 'WR',
            'TIGHT_END': 'TE', 'DEFENSIVE_END': 'DE', 'DEFENSIVE_TACKLE': 'DT',
            'RUNNING_BACK': 'RB', 'QUARTERBACK': 'QB'
        }
        
        pos_series = enhanced_supplementary['position_standardized'].copy()
        if hasattr(pos_series, 'cat'):
            pos_series = pos_series.astype(str)
        
        enhanced_supplementary['position_standardized'] = pos_series.map(
            position_mapping_dict
        ).fillna(pos_series)
        
        enhanced_supplementary['role_category'] = enhanced_supplementary['position_standardized'].map(
            lambda x: 'DEFENDER' if str(x) in DEFENSIVE_POSITIONS 
                     else 'RECEIVER' if str(x) in OFFENSIVE_POSITIONS 
                     else 'OTHER'
        )
        
        position_dist = enhanced_supplementary['position_standardized'].value_counts()
        print(f"         Position distribution (top 10):")
        for pos, count in position_dist.head(10).items():
            role = 'DEF' if pos in DEFENSIVE_POSITIONS else 'OFF' if pos in OFFENSIVE_POSITIONS else 'OTH'
            print(f"            {pos} ({role}): {count:,}")
    
    if not position_mapped:
        print(f"         No position mapping achieved, using defaults")
        enhanced_supplementary['position_standardized'] = 'UNKNOWN'
        enhanced_supplementary['role_category'] = 'UNKNOWN'
    
    # =====================================================================
    # STEP 5: TACTICAL CONTEXT
    # =====================================================================
    
    print(f"\n      STEP 5: TACTICAL CONTEXT")
    
    def safe_categorical_fillna(series, fill_value='UNKNOWN'):
        if hasattr(series, 'cat') and fill_value not in series.cat.categories:
            series = series.cat.add_categories([fill_value])
        return series.fillna(fill_value)
    
    tactical_vars_found = {}
    
    # Route information
    route_cols = [col for col in enhanced_supplementary.columns 
                  if any(pattern in col.lower() for pattern in ['route', 'pattern'])]
    
    if route_cols:
        route_col = route_cols[0]
        enhanced_supplementary['route_type_clean'] = safe_categorical_fillna(
            enhanced_supplementary[route_col], 'UNKNOWN'
        )
        tactical_vars_found['route'] = route_col
        route_dist = enhanced_supplementary['route_type_clean'].value_counts()
        print(f"         Route types: {len(route_dist)} unique")
        for route, count in route_dist.head(5).items():
            print(f"            {route}: {count:,}")
    else:
        enhanced_supplementary['route_type_clean'] = 'UNKNOWN'
        print(f"         No route data found")
    
    # Coverage information
    coverage_cols = [col for col in enhanced_supplementary.columns 
                     if any(pattern in col.lower() for pattern in ['coverage', 'defense', 'scheme'])]
    
    if coverage_cols:
        coverage_col = coverage_cols[0]
        enhanced_supplementary['coverage_type_clean'] = safe_categorical_fillna(
            enhanced_supplementary[coverage_col], 'UNKNOWN'
        )
        tactical_vars_found['coverage'] = coverage_col
        coverage_dist = enhanced_supplementary['coverage_type_clean'].value_counts()
        print(f"         Coverage types: {len(coverage_dist)} unique")
        for coverage, count in coverage_dist.head(5).items():
            print(f"            {coverage}: {count:,}")
    else:
        enhanced_supplementary['coverage_type_clean'] = 'UNKNOWN'
        print(f"         No coverage data found")
    
    # EPA information
    epa_cols = [col for col in enhanced_supplementary.columns 
                if any(pattern in col.lower() for pattern in ['expected', 'epa', 'points']) 
                and 'point' in col.lower()]
    
    if epa_cols:
        epa_col = epa_cols[0]
        try:
            epa_series = enhanced_supplementary[epa_col]
            if hasattr(epa_series, 'cat'):
                epa_series = pd.to_numeric(epa_series.astype(str), errors='coerce')
            
            enhanced_supplementary['epa_situation'] = pd.cut(
                epa_series,
                bins=[-100, -1, 1, 3, 100],
                labels=['HIGH_PRESSURE', 'PRESSURE', 'NEUTRAL', 'FAVORABLE']
            )
            tactical_vars_found['epa'] = epa_col
            epa_dist = enhanced_supplementary['epa_situation'].value_counts()
            print(f"         EPA situations: {len(epa_dist)} categories")
        except Exception as e:
            enhanced_supplementary['epa_situation'] = 'NEUTRAL'
            print(f"         EPA processing failed: {e}")
    else:
        enhanced_supplementary['epa_situation'] = 'NEUTRAL'
        print(f"         No EPA data found")
    
    # Additional tactical context
    additional_tactical = {}
    for col in enhanced_supplementary.columns:
        col_lower = col.lower()
        if 'formation' in col_lower:
            additional_tactical['formation'] = col
        elif 'play_action' in col_lower:
            additional_tactical['play_action'] = col
        elif 'dropback' in col_lower:
            additional_tactical['dropback'] = col
    
    print(f"         Additional tactical: {list(additional_tactical.keys())}")
    
    # =====================================================================
    # STEP 6: DATA QUALITY ASSESSMENT
    # =====================================================================
    
    print(f"\n      STEP 6: DATA QUALITY ASSESSMENT")
    
    quality_metrics = {
        'total_records': len(enhanced_supplementary),
        'position_completeness': (enhanced_supplementary['position_standardized'] != 'UNKNOWN').mean() if 'position_standardized' in enhanced_supplementary.columns else 0,
        'tactical_completeness': len(tactical_vars_found) / 3,
        'identifier_completeness': 0,
        'unique_games': enhanced_supplementary['game_id'].nunique() if 'game_id' in enhanced_supplementary.columns else 0,
        'unique_plays': 0
    }
    
    id_cols = ['game_id', 'play_id', 'player_id']
    available_ids = [col for col in id_cols if col in enhanced_supplementary.columns]
    quality_metrics['identifier_completeness'] = len(available_ids) / len(id_cols)
    
    if all(col in enhanced_supplementary.columns for col in ['game_id', 'play_id']):
        quality_metrics['unique_plays'] = enhanced_supplementary.groupby(['game_id', 'play_id']).ngroups
    
    print(f"         Data quality metrics:")
    print(f"            Total records: {quality_metrics['total_records']:,}")
    print(f"            Position completeness: {quality_metrics['position_completeness']*100:.1f}%")
    print(f"            Tactical completeness: {quality_metrics['tactical_completeness']*100:.1f}%")
    print(f"            ID completeness: {quality_metrics['identifier_completeness']*100:.1f}%")
    print(f"            Unique games: {quality_metrics['unique_games']:,}")
    print(f"            Unique plays: {quality_metrics['unique_plays']:,}")
    
    overall_quality = np.mean([
        quality_metrics['position_completeness'],
        quality_metrics['tactical_completeness'],
        quality_metrics['identifier_completeness']
    ])
    print(f"            Overall quality: {overall_quality*100:.1f}%")
    
    # =====================================================================
    # STEP 7: INDEX CREATION (memory-efficient - counts only)
    # =====================================================================
    
    print(f"\n      STEP 7: INDEX CREATION")
    
    enhanced_indices = {}
    
    if all(col in enhanced_supplementary.columns for col in ['game_id', 'play_id']):
        play_count = enhanced_supplementary.groupby(['game_id', 'play_id']).ngroups
        enhanced_indices['play_count'] = play_count
        print(f"         Play groups: {play_count:,}")
    
    if 'player_id' in enhanced_supplementary.columns:
        player_count = enhanced_supplementary['player_id'].nunique()
        enhanced_indices['player_count'] = player_count
        print(f"         Player groups: {player_count:,}")
    
    if 'position_standardized' in enhanced_supplementary.columns:
        position_count = enhanced_supplementary['position_standardized'].nunique()
        enhanced_indices['position_count'] = position_count
        print(f"         Position groups: {position_count:,}")
    
    # Update championship data
    championship_data['enhanced_supplementary'] = enhanced_supplementary
    championship_data['tactical_vars_found'] = tactical_vars_found
    championship_data['additional_tactical'] = additional_tactical
    championship_data['enhanced_indices'] = enhanced_indices
    championship_data['data_quality_metrics'] = quality_metrics
    
    enhancement_summary = {
        'total_records_processed': quality_metrics['total_records'],
        'position_mapping_success': position_mapped,
        'tactical_variables_found': len(tactical_vars_found),
        'overall_data_quality': overall_quality,
        'indices_created': len(enhanced_indices),
        'id_mapping_success': quality_metrics['identifier_completeness'] > 0.5
    }
    
    print(f"\n      ENHANCEMENT SUMMARY:")
    print(f"         Records processed: {enhancement_summary['total_records_processed']:,}")
    print(f"         Position mapping: {'OK' if enhancement_summary['position_mapping_success'] else 'Limited'}")
    print(f"         Tactical variables: {enhancement_summary['tactical_variables_found']}")
    print(f"         Overall quality: {enhancement_summary['overall_data_quality']*100:.1f}%")
    
    aggressive_gc()
    
    return enhancement_summary

# Execute
comprehensive_enhancement_results = enhance_comprehensive_supplementary_integration_fixed()

print(f"\nMemory after enhancement: {get_memory_usage():.1f} MB")
print("CELL 3 COMPLETE - SUPPLEMENTARY INTEGRATION DONE!")


# ========================================================================
# CELL 4: TRAJECTORY EXTRACTION - KAGGLE SAFE
# Memory-optimized trajectory processing
# ========================================================================

print("\nTRAJECTORY EXTRACTION - MEMORY OPTIMIZED")
print("="*50)

@timing_decorator
def extract_massive_dataset_trajectories():
    """Extract trajectories with memory optimization"""
    
    print(f"\n   Memory: {get_memory_gb():.2f}GB")
    
    if 'combined_tracking' not in championship_data or len(championship_data['combined_tracking']) == 0:
        print("      No tracking data available")
        return None
    
    tracking_df = championship_data['combined_tracking'].copy()
    total_records = len(tracking_df)
    
    print(f"      Processing tracking dataset:")
    print(f"         Total records: {total_records:,}")
    print(f"         Memory: {tracking_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # =====================================================================
    # STEP 1: DATA PREPARATION
    # =====================================================================
    
    print(f"\n      STEP 1: DATA PREPARATION")
    
    available_cols = tracking_df.columns.tolist()
    print(f"         Available columns ({len(available_cols)}): {available_cols[:10]}...")
    
    # Column mapping
    essential_cols = {}
    id_patterns = {
        'game_id': ['game_id', 'gameid', 'game'],
        'play_id': ['play_id', 'playid', 'play'],
        'player_id': ['nfl_id', 'nflid', 'player_id', 'playerid'],
        'frame_id': ['frame_id', 'frameid', 'frame'],
        'x': ['x', 'pos_x', 'position_x'],
        'y': ['y', 'pos_y', 'position_y']
    }
    
    for standard_name, candidates in id_patterns.items():
        for candidate in candidates:
            if candidate in available_cols:
                essential_cols[standard_name] = candidate
                break
    
    print(f"         Column mapping: {essential_cols}")
    
    missing_essential = [col for col in ['game_id', 'play_id', 'x', 'y'] if col not in essential_cols]
    if missing_essential:
        print(f"         Missing essential columns: {missing_essential}")
        return None
    
    # Create working dataset
    print(f"         Creating working dataset...")
    working_cols = ['game_id', 'play_id', 'x', 'y']
    if 'player_id' in essential_cols:
        working_cols.append('player_id')
    if 'frame_id' in essential_cols:
        working_cols.append('frame_id')
    
    col_mapping = {essential_cols[std_col]: std_col for std_col in working_cols if std_col in essential_cols}
    
    working_tracking = tracking_df[list(col_mapping.keys())].copy()
    working_tracking = working_tracking.rename(columns=col_mapping)
    
    print(f"         Working dataset: {len(working_tracking):,} records")
    
    # Data cleaning
    before_cleaning = len(working_tracking)
    
    coord_mask = (
        working_tracking['x'].between(-20, 140) &
        working_tracking['y'].between(-10, 70) &
        working_tracking['x'].notna() &
        working_tracking['y'].notna()
    )
    working_tracking = working_tracking[coord_mask]
    
    id_mask = working_tracking['game_id'].notna() & working_tracking['play_id'].notna()
    working_tracking = working_tracking[id_mask]
    
    after_cleaning = len(working_tracking)
    print(f"         Cleaned: {before_cleaning:,} -> {after_cleaning:,} ({after_cleaning/before_cleaning*100:.1f}% retained)")
    
    # Sampling if needed
    max_trajectories_target = 150000
    
    if 'player_id' in working_tracking.columns:
        total_potential = len(working_tracking.groupby(['game_id', 'play_id', 'player_id']))
    else:
        total_potential = len(working_tracking.groupby(['game_id', 'play_id']))
    
    print(f"         Potential trajectories: {total_potential:,}")
    
    if total_potential > max_trajectories_target:
        sample_ratio = max_trajectories_target / total_potential
        print(f"         Applying sampling: {sample_ratio:.3f} ratio")
        
        unique_plays = working_tracking[['game_id', 'play_id']].drop_duplicates()
        sampled_plays = unique_plays.sample(n=int(len(unique_plays) * sample_ratio), random_state=42)
        working_tracking = working_tracking.merge(sampled_plays, on=['game_id', 'play_id'])
        print(f"            Sampled: {len(working_tracking):,} records")
    
    # =====================================================================
    # STEP 2: TRAJECTORY EXTRACTION
    # =====================================================================
    
    print(f"\n      STEP 2: TRAJECTORY EXTRACTION")
    
    if 'player_id' in working_tracking.columns and 'frame_id' in working_tracking.columns:
        grouping_cols = ['game_id', 'play_id', 'player_id']
        sort_col = 'frame_id'
        extraction_mode = 'full'
    elif 'player_id' in working_tracking.columns:
        grouping_cols = ['game_id', 'play_id', 'player_id']
        sort_col = None
        extraction_mode = 'player_only'
    else:
        working_tracking = working_tracking.sort_values(['game_id', 'play_id', 'x', 'y'])
        working_tracking['synthetic_player'] = working_tracking.groupby(['game_id', 'play_id']).cumcount()
        grouping_cols = ['game_id', 'play_id', 'synthetic_player']
        sort_col = None
        extraction_mode = 'synthetic'
        print(f"            Using synthetic player grouping")
    
    print(f"         Extraction mode: {extraction_mode}")
    print(f"         Grouping by: {grouping_cols}")
    
    trajectory_groups = working_tracking.groupby(grouping_cols)
    total_groups = len(trajectory_groups)
    
    print(f"         Trajectory groups: {total_groups:,}")
    
    # Batch processing
    trajectories = []
    processed_groups = 0
    batch_size = 5000
    progress_interval = 10000
    
    print(f"         Starting extraction (batch: {batch_size:,})...")
    
    for group_key, group_data in trajectory_groups:
        processed_groups += 1
        
        if processed_groups % progress_interval == 0:
            print(f"            Progress: {processed_groups:,}/{total_groups:,} ({processed_groups/total_groups*100:.1f}%)")
            gc.collect()
        
        if len(group_data) < 3:
            continue
        
        try:
            if sort_col and sort_col in group_data.columns:
                group_data = group_data.sort_values(sort_col)
                frame_sequence = group_data[sort_col].values
            else:
                group_data = group_data.sort_values(['x', 'y'])
                frame_sequence = np.arange(len(group_data))
            
            x_sequence = group_data['x'].values
            y_sequence = group_data['y'].values
            
            if np.std(x_sequence) < 0.5 and np.std(y_sequence) < 0.5:
                continue
            
            # Movement calculations
            dx = np.diff(x_sequence)
            dy = np.diff(y_sequence)
            distances = np.sqrt(dx**2 + dy**2)
            velocities = distances.copy()
            directions = np.arctan2(dy, dx)
            
            if len(directions) > 1:
                direction_diffs = np.diff(directions)
                direction_diffs = np.where(direction_diffs > np.pi, direction_diffs - 2*np.pi, direction_diffs)
                direction_diffs = np.where(direction_diffs < -np.pi, direction_diffs + 2*np.pi, direction_diffs)
            else:
                direction_diffs = np.array([])
            
            accelerations = np.diff(velocities) if len(velocities) > 1 else np.array([])
            
            # Summary stats
            total_distance = np.sum(distances)
            net_displacement = np.sqrt((x_sequence[-1] - x_sequence[0])**2 + 
                                     (y_sequence[-1] - y_sequence[0])**2)
            
            path_efficiency = net_displacement / total_distance if total_distance > 0 else 0
            max_velocity = np.max(velocities) if len(velocities) > 0 else 0
            avg_velocity = np.mean(velocities) if len(velocities) > 0 else 0
            max_acceleration = np.max(np.abs(accelerations)) if len(accelerations) > 0 else 0
            significant_changes = np.sum(np.abs(direction_diffs) > np.pi/6) if len(direction_diffs) > 0 else 0
            
            # Quality filter
            if not (total_distance > 5 and len(frame_sequence) >= 3 and max_velocity < 15 and path_efficiency <= 2.0):
                continue
            
            trajectory = {
                'game_id': int(group_key[0]) if pd.notna(group_key[0]) else 0,
                'play_id': int(group_key[1]) if pd.notna(group_key[1]) else 0,
                'player_id': int(group_key[2]) if len(group_key) > 2 and pd.notna(group_key[2]) else 0,
                'frame_sequence': frame_sequence.tolist(),
                'x_sequence': x_sequence.tolist(),
                'y_sequence': y_sequence.tolist(),
                'distances_per_frame': distances.tolist(),
                'velocities_per_frame': velocities.tolist(),
                'directions_per_frame': directions.tolist(),
                'accelerations_per_frame': accelerations.tolist(),
                'total_frames': len(frame_sequence),
                'total_distance': float(total_distance),
                'net_displacement': float(net_displacement),
                'path_efficiency': float(path_efficiency),
                'max_velocity': float(max_velocity),
                'avg_velocity': float(avg_velocity),
                'max_acceleration': float(max_acceleration),
                'direction_changes_count': int(significant_changes),
                'start_x': float(x_sequence[0]),
                'start_y': float(y_sequence[0]),
                'end_x': float(x_sequence[-1]),
                'end_y': float(y_sequence[-1]),
                'movement_variance': float(np.var(distances)),
                'velocity_consistency': float(1 / (1 + np.var(velocities))) if len(velocities) > 0 else 0,
                'extraction_mode': extraction_mode
            }
            
            trajectories.append(trajectory)
            
        except Exception:
            continue
        
        if processed_groups % batch_size == 0:
            gc.collect()
    
    # =====================================================================
    # STEP 3: QUALITY ASSESSMENT
    # =====================================================================
    
    print(f"\n      STEP 3: QUALITY ASSESSMENT")
    
    if trajectories:
        trajectories_df = pd.DataFrame(trajectories)
        
        print(f"         Extraction results:")
        print(f"            Trajectories: {len(trajectories_df):,}")
        print(f"            Success rate: {len(trajectories_df)/total_groups*100:.1f}%")
        print(f"            Memory: {trajectories_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        print(f"         Movement characteristics:")
        print(f"            Avg frames: {trajectories_df['total_frames'].mean():.1f}")
        print(f"            Avg distance: {trajectories_df['total_distance'].mean():.1f} yards")
        print(f"            Avg efficiency: {trajectories_df['path_efficiency'].mean():.3f}")
        
        # Quality scoring
        trajectories_df['quality_score'] = (
            np.minimum(trajectories_df['total_frames'] / 20, 1.0) * 0.3 +
            np.minimum(trajectories_df['total_distance'] / 30, 1.0) * 0.3 +
            trajectories_df['path_efficiency'].clip(0, 1) * 0.2 +
            np.minimum(trajectories_df['avg_velocity'] / 4, 1.0) * 0.2
        )
        
        high_quality = len(trajectories_df[trajectories_df['quality_score'] >= 0.4])
        medium_quality = len(trajectories_df[trajectories_df['quality_score'].between(0.2, 0.4)])
        low_quality = len(trajectories_df[trajectories_df['quality_score'] < 0.2])
        
        print(f"         Quality distribution:")
        print(f"            High (>=0.4): {high_quality:,}")
        print(f"            Medium (0.2-0.4): {medium_quality:,}")
        print(f"            Low (<0.2): {low_quality:,}")
        
        massive_trajectory_stats = {
            'total_trajectories': len(trajectories_df),
            'original_records_processed': total_records,
            'extraction_success_rate': len(trajectories_df) / total_groups,
            'avg_frames': trajectories_df['total_frames'].mean(),
            'avg_distance': trajectories_df['total_distance'].mean(),
            'avg_efficiency': trajectories_df['path_efficiency'].mean(),
            'quality_distribution': {'high': high_quality, 'medium': medium_quality, 'low': low_quality},
            'extraction_mode': extraction_mode,
            'sampling_applied': total_potential > max_trajectories_target
        }
        
        return {
            'trajectories_df': trajectories_df,
            'massive_trajectory_stats': massive_trajectory_stats,
            'column_mappings': essential_cols
        }
    
    else:
        print(f"         No trajectories extracted")
        return None

# Execute
massive_trajectory_results = extract_massive_dataset_trajectories()

aggressive_gc()

print(f"\nMemory after extraction: {get_memory_usage():.1f} MB")
print("CELL 4 COMPLETE - TRAJECTORY EXTRACTION DONE!")

if massive_trajectory_results:
    stats = massive_trajectory_results['massive_trajectory_stats']
    print(f"\nSUMMARY:")
    print(f"   Original records: {stats['original_records_processed']:,}")
    print(f"   Trajectories: {stats['total_trajectories']:,}")
    print(f"   Success rate: {stats['extraction_success_rate']*100:.1f}%")


# ========================================================================
# CELL 5: MOTION ANALYSIS - KAGGLE SAFE
# Memory-optimized motion analysis with batch processing
# ========================================================================

print("\nMOTION ANALYSIS - MEMORY OPTIMIZED")
print("="*45)

@timing_decorator
def calculate_massive_motion_analysis():
    """Motion analysis with memory optimization"""
    
    print(f"\n   Memory: {get_memory_gb():.2f}GB")
    
    if massive_trajectory_results is None:
        print("      No trajectory data available")
        return None
    
    trajectories_df = massive_trajectory_results['trajectories_df'].copy()
    total_trajectories = len(trajectories_df)
    
    print(f"      Analyzing trajectories:")
    print(f"         Total: {total_trajectories:,}")
    print(f"         Memory: {trajectories_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # =====================================================================
    # STEP 1: BATCH PROCESSING SETUP
    # =====================================================================
    
    print(f"\n      STEP 1: BATCH PROCESSING SETUP")
    
    batch_size = 10000
    total_batches = (total_trajectories + batch_size - 1) // batch_size
    
    print(f"         Processing in {total_batches} batches of {batch_size:,}")
    
    enhanced_trajectories = []
    
    # =====================================================================
    # STEP 2: BATCH PROCESSING
    # =====================================================================
    
    print(f"\n      STEP 2: BATCH PROCESSING")
    
    for batch_num in range(total_batches):
        batch_start = batch_num * batch_size
        batch_end = min((batch_num + 1) * batch_size, total_trajectories)
        
        if batch_num % 2 == 0 or batch_num == total_batches - 1:
            print(f"         Batch {batch_num + 1}/{total_batches}: {batch_start:,}-{batch_end:,}")
        
        batch_trajectories = trajectories_df.iloc[batch_start:batch_end]
        batch_enhanced = []
        
        for idx, trajectory in batch_trajectories.iterrows():
            try:
                x_seq = np.array(trajectory['x_sequence'])
                y_seq = np.array(trajectory['y_sequence'])
                velocities = np.array(trajectory['velocities_per_frame'])
                accelerations = np.array(trajectory['accelerations_per_frame'])
                directions = np.array(trajectory['directions_per_frame'])
                
                if len(x_seq) < 3:
                    continue
                
                # Path deviation analysis
                start_pos = np.array([x_seq[0], y_seq[0]])
                end_pos = np.array([x_seq[-1], y_seq[-1]])
                
                if not np.array_equal(start_pos, end_pos):
                    path_vec = end_pos - start_pos
                    path_length = np.linalg.norm(path_vec)
                    
                    if path_length > 0:
                        points = np.column_stack([x_seq, y_seq])
                        point_vecs = points - start_pos
                        projections = np.dot(point_vecs, path_vec) / path_length
                        projection_points = start_pos + np.outer(projections, path_vec) / path_length
                        deviations = np.linalg.norm(points - projection_points, axis=1)
                        
                        max_path_deviation = np.max(deviations)
                        avg_path_deviation = np.mean(deviations)
                        path_deviation_variance = np.var(deviations)
                    else:
                        max_path_deviation = avg_path_deviation = path_deviation_variance = 0
                else:
                    max_path_deviation = avg_path_deviation = path_deviation_variance = 0
                
                # Acceleration analysis
                if len(accelerations) > 0:
                    accel_magnitude = np.abs(accelerations)
                    accel_75th = np.percentile(accel_magnitude, 75) if len(accel_magnitude) > 4 else 0
                    accel_90th = np.percentile(accel_magnitude, 90) if len(accel_magnitude) > 4 else 0
                    
                    acceleration_bursts = np.sum(accel_magnitude > accel_75th)
                    explosive_bursts = np.sum(accel_magnitude > accel_90th)
                    
                    high_accel_mask = accel_magnitude > accel_75th
                    diff_mask = np.diff(np.concatenate(([False], high_accel_mask, [False])).astype(int))
                    starts = np.where(diff_mask == 1)[0]
                    ends = np.where(diff_mask == -1)[0]
                    
                    max_sustained_acceleration = np.max(ends - starts) if len(starts) > 0 and len(ends) > 0 else 0
                    acceleration_consistency = 1 - (np.std(accel_magnitude) / (np.mean(accel_magnitude) + 0.001))
                else:
                    acceleration_bursts = explosive_bursts = max_sustained_acceleration = 0
                    acceleration_consistency = 0
                
                # Directional analysis
                if len(directions) > 1:
                    direction_changes = np.abs(np.diff(directions))
                    direction_changes = np.minimum(direction_changes, 2*np.pi - direction_changes)
                    
                    minor_cuts = np.sum((direction_changes > np.pi/12) & (direction_changes <= np.pi/6))
                    moderate_cuts = np.sum((direction_changes > np.pi/6) & (direction_changes <= np.pi/3))
                    sharp_cuts = np.sum(direction_changes > np.pi/3)
                    
                    direction_efficiency = 1 - (sharp_cuts / max(len(direction_changes), 1))
                    
                    high_speed_cuts = np.sum((direction_changes > np.pi/6) & (velocities[1:len(direction_changes)+1] > 2.5)) if len(velocities) >= len(direction_changes) else 0
                else:
                    minor_cuts = moderate_cuts = sharp_cuts = high_speed_cuts = 0
                    direction_efficiency = 1.0
                
                # Reaction analysis
                if len(velocities) > 0:
                    velocity_threshold = 0.8
                    significant_movement = velocities > velocity_threshold
                    
                    reaction_delay = np.argmax(significant_movement) if np.any(significant_movement) else len(velocities)
                    peak_velocity_frame = np.argmax(velocities)
                    velocity_cv = np.std(velocities) / (np.mean(velocities) + 0.001)
                    
                    if len(velocities) > 5:
                        sprint_threshold = np.percentile(velocities, 75)
                        sprint_frames = np.sum(velocities > sprint_threshold)
                        sprint_ratio = sprint_frames / len(velocities)
                    else:
                        sprint_frames = sprint_ratio = 0
                else:
                    reaction_delay = peak_velocity_frame = velocity_cv = sprint_frames = sprint_ratio = 0
                
                # Effort score
                max_accel = trajectory['max_acceleration']
                max_velocity = trajectory['max_velocity']
                
                accel_component = min(max_accel / 4.0, 1.0) * 0.6 + min(explosive_bursts / 3.0, 1.0) * 0.4
                complexity_component = min(avg_path_deviation / 8.0, 1.0) * 0.4 + min(sharp_cuts / 8.0, 1.0) * 0.6
                velocity_component = min(max_velocity / 6.0, 1.0) * 0.6 + min(high_speed_cuts / 4.0, 1.0) * 0.4
                sustained_component = min(sprint_ratio * 2.0, 1.0) * 0.6 + min(max_sustained_acceleration / 5.0, 1.0) * 0.4
                efficiency_component = trajectory['path_efficiency'] * 0.5 + direction_efficiency * 0.5
                reaction_component = max(0, 1 - reaction_delay / 10.0) * 0.7 + max(0, 1 - velocity_cv) * 0.3
                
                comprehensive_effort_score = (
                    accel_component * 0.25 +
                    complexity_component * 0.20 +
                    velocity_component * 0.20 +
                    sustained_component * 0.15 +
                    efficiency_component * 0.10 +
                    reaction_component * 0.10
                ) * 100
                
                # Player categorization
                receiver_score = (
                    (trajectory['path_efficiency'] > 0.6) * 0.2 +
                    (trajectory['total_distance'] > 20) * 0.2 +
                    (sharp_cuts >= 2) * 0.2 +
                    (max_velocity > 3.0) * 0.2 +
                    (comprehensive_effort_score > 40) * 0.2
                )
                
                defender_score = (
                    (trajectory['total_distance'] > 15) * 0.2 +
                    (trajectory['direction_changes_count'] > 3) * 0.2 +
                    (acceleration_bursts > 2) * 0.2 +
                    (avg_path_deviation > 3) * 0.2 +
                    (comprehensive_effort_score > 30) * 0.2
                )
                
                enhanced_traj = {
                    'game_id': trajectory['game_id'],
                    'play_id': trajectory['play_id'],
                    'player_id': trajectory['player_id'],
                    'total_frames': trajectory['total_frames'],
                    'total_distance': trajectory['total_distance'],
                    'path_efficiency': trajectory['path_efficiency'],
                    'max_velocity': trajectory['max_velocity'],
                    'avg_velocity': trajectory['avg_velocity'],
                    'max_acceleration': trajectory['max_acceleration'],
                    'direction_changes_count': trajectory['direction_changes_count'],
                    'max_path_deviation': float(max_path_deviation),
                    'avg_path_deviation': float(avg_path_deviation),
                    'acceleration_bursts': int(acceleration_bursts),
                    'explosive_bursts': int(explosive_bursts),
                    'minor_cuts': int(minor_cuts),
                    'moderate_cuts': int(moderate_cuts),
                    'sharp_cuts': int(sharp_cuts),
                    'high_speed_cuts': int(high_speed_cuts),
                    'direction_efficiency': float(direction_efficiency),
                    'reaction_delay_frames': int(reaction_delay),
                    'sprint_ratio': float(sprint_ratio),
                    'comprehensive_effort_score': float(comprehensive_effort_score),
                    'receiver_likelihood': float(receiver_score),
                    'defender_likelihood': float(defender_score),
                    'is_potential_receiver': bool(receiver_score > 0.6),
                    'is_potential_defender': bool(defender_score > 0.6),
                    'start_x': trajectory['start_x'],
                    'start_y': trajectory['start_y'],
                    'end_x': trajectory['end_x'],
                    'end_y': trajectory['end_y'],
                    'quality_score': trajectory['quality_score']
                }
                
                batch_enhanced.append(enhanced_traj)
                
            except Exception:
                continue
        
        enhanced_trajectories.extend(batch_enhanced)
        
        if (batch_num + 1) % 5 == 0:
            gc.collect()
    
    # =====================================================================
    # STEP 3: MOTION PATTERN CLASSIFICATION
    # =====================================================================
    
    print(f"\n      STEP 3: MOTION PATTERN CLASSIFICATION")
    
    if enhanced_trajectories:
        enhanced_df = pd.DataFrame(enhanced_trajectories)
        
        print(f"         Enhanced trajectories: {len(enhanced_df):,}")
        print(f"         Success rate: {len(enhanced_df)/total_trajectories*100:.1f}%")
        
        # Motion pattern classification
        def classify_motion_pattern(df):
            effort = df['comprehensive_effort_score']
            efficiency = df['path_efficiency']
            cuts = df['sharp_cuts']
            acceleration = df['max_acceleration']
            distance = df['total_distance']
            
            conditions = [
                (effort >= 80) & (cuts >= 5) & (acceleration > 3),
                (effort >= 70) & (efficiency >= 0.8),
                (cuts >= 8) & (effort >= 50),
                (effort >= 60) & (distance > 40),
                (efficiency >= 0.8) & (effort < 50),
                (acceleration > 4) & (cuts <= 2),
                (cuts >= 5) & (efficiency < 0.6)
            ]
            
            choices = [
                'ELITE_AGGRESSIVE', 'ELITE_EFFICIENT', 'HIGHLY_EVASIVE',
                'SUSTAINED_EFFORT', 'CONSERVATIVE', 'SPEED_FOCUSED', 'SCRAMBLING'
            ]
            
            return np.select(conditions, choices, default='MODERATE')
        
        enhanced_df['comprehensive_motion_pattern'] = classify_motion_pattern(enhanced_df)
        
        pattern_dist = enhanced_df['comprehensive_motion_pattern'].value_counts()
        print(f"         Motion patterns:")
        for pattern, count in pattern_dist.items():
            print(f"            {pattern}: {count:,} ({count/len(enhanced_df)*100:.1f}%)")
        
        print(f"         Performance metrics:")
        print(f"            Avg effort: {enhanced_df['comprehensive_effort_score'].mean():.1f}")
        print(f"            Elite (>80): {len(enhanced_df[enhanced_df['comprehensive_effort_score'] > 80]):,}")
        print(f"            Potential receivers: {enhanced_df['is_potential_receiver'].sum():,}")
        print(f"            Potential defenders: {enhanced_df['is_potential_defender'].sum():,}")
        
        # Memory optimization
        enhanced_df = enhanced_df.astype({
            'game_id': 'int32',
            'play_id': 'int32',
            'player_id': 'int32',
            'total_frames': 'uint16',
            'acceleration_bursts': 'uint8',
            'sharp_cuts': 'uint8'
        })
        
        massive_motion_stats = {
            'total_enhanced': len(enhanced_df),
            'original_trajectories': total_trajectories,
            'enhancement_success_rate': len(enhanced_df) / total_trajectories,
            'avg_comprehensive_effort': enhanced_df['comprehensive_effort_score'].mean(),
            'motion_patterns': pattern_dist.to_dict(),
            'receiver_identification': enhanced_df['is_potential_receiver'].sum(),
            'defender_identification': enhanced_df['is_potential_defender'].sum(),
            'elite_performers': len(enhanced_df[enhanced_df['comprehensive_effort_score'] > 80])
        }
        
        print(f"         Memory: {enhanced_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return {
            'enhanced_trajectories_df': enhanced_df,
            'massive_motion_stats': massive_motion_stats
        }
    
    else:
        print(f"         No enhanced trajectories created")
        return None

# Execute
massive_motion_results = calculate_massive_motion_analysis()

aggressive_gc()

print(f"\nMemory after analysis: {get_memory_usage():.1f} MB")
print("CELL 5 COMPLETE - MOTION ANALYSIS DONE!")


# ========================================================================
# CELL 6: BALL-IN-AIR ANALYSIS - KAGGLE SAFE
# Robust ball-in-air window creation with error handling
# ========================================================================

print("\nBALL-IN-AIR ANALYSIS")
print("="*45)

@timing_decorator
def create_fixed_ball_in_air_analysis():
    """Create ball-in-air analysis with robust error handling"""
    
    print(f"\n   Memory: {get_memory_gb():.2f}GB")
    
    if massive_motion_results is None:
        print("      Missing massive_motion_results")
        return None
    
    if 'enhanced_supplementary' not in championship_data:
        print("      Missing enhanced_supplementary")
        return None
    
    enhanced_trajectories = massive_motion_results['enhanced_trajectories_df']
    enhanced_supplementary = championship_data['enhanced_supplementary']
    
    print(f"      Processing datasets:")
    print(f"         Enhanced trajectories: {len(enhanced_trajectories):,}")
    print(f"         Supplementary records: {len(enhanced_supplementary):,}")
    
    # =====================================================================
    # STEP 1: ROBUST PLAY GROUPING
    # =====================================================================
    
    print(f"\n      STEP 1: PLAY GROUPING")
    
    required_cols = ['game_id', 'play_id', 'end_x', 'end_y', 'comprehensive_effort_score']
    missing_cols = [col for col in required_cols if col not in enhanced_trajectories.columns]
    
    if missing_cols:
        print(f"         Missing columns: {missing_cols}")
        return None
    
    play_groups = enhanced_trajectories.groupby(['game_id', 'play_id'])
    total_plays = len(play_groups)
    
    print(f"         Total unique plays: {total_plays:,}")
    
    min_trajectories = 3
    valid_plays = []
    
    for (game_id, play_id), group in play_groups:
        if len(group) >= min_trajectories:
            valid_plays.append((game_id, play_id, len(group)))
    
    print(f"         Valid plays (>={min_trajectories} trajectories): {len(valid_plays):,}")
    
    if len(valid_plays) == 0:
        print(f"         No valid plays found")
        return None
    
    # =====================================================================
    # STEP 2: BALL-IN-AIR WINDOW CREATION
    # =====================================================================
    
    print(f"\n      STEP 2: BALL-IN-AIR WINDOW CREATION")
    
    ball_windows = []
    processed_count = 0
    error_count = 0
    success_count = 0
    
    print(f"         Processing {len(valid_plays):,} plays...")
    
    for game_id, play_id, traj_count in valid_plays:
        processed_count += 1
        
        if processed_count % 1000 == 0:
            print(f"            Progress: {processed_count:,}/{len(valid_plays):,} ({processed_count/len(valid_plays)*100:.1f}%)")
        
        try:
            play_trajectories = enhanced_trajectories[
                (enhanced_trajectories['game_id'] == game_id) & 
                (enhanced_trajectories['play_id'] == play_id)
            ]
            
            if len(play_trajectories) < min_trajectories:
                continue
            
            # Frame analysis
            estimated_play_duration = max(10, min(50, len(play_trajectories) * 2))
            frame_release = 5
            frame_arrival = max(frame_release + 5, estimated_play_duration - 5)
            ball_in_air_duration = frame_arrival - frame_release
            
            if ball_in_air_duration < 3:
                ball_in_air_duration = 8
                frame_arrival = frame_release + ball_in_air_duration
            
            # Ball location estimation
            receivers = play_trajectories[
                play_trajectories.get('is_potential_receiver', False) == True
            ]
            
            if len(receivers) > 0:
                effort_weights = receivers['comprehensive_effort_score'].values
                if np.sum(effort_weights) > 0:
                    effort_weights = effort_weights / np.sum(effort_weights)
                    ball_land_x = np.average(receivers['end_x'], weights=effort_weights)
                    ball_land_y = np.average(receivers['end_y'], weights=effort_weights)
                else:
                    ball_land_x = receivers['end_x'].mean()
                    ball_land_y = receivers['end_y'].mean()
                estimation_method = 'receiver_convergence'
            else:
                high_effort = play_trajectories[
                    play_trajectories['comprehensive_effort_score'] > 50
                ]
                
                if len(high_effort) > 0:
                    ball_land_x = high_effort['end_x'].mean()
                    ball_land_y = high_effort['end_y'].mean()
                    estimation_method = 'effort_convergence'
                else:
                    ball_land_x = play_trajectories['end_x'].mean()
                    ball_land_y = play_trajectories['end_y'].mean()
                    estimation_method = 'geometric_centroid'
            
            if pd.isna(ball_land_x) or pd.isna(ball_land_y):
                error_count += 1
                continue
            
            ball_land_x = float(np.clip(ball_land_x, 0, 120))
            ball_land_y = float(np.clip(ball_land_y, 0, 53.3))
            
            # Tactical context
            play_context = enhanced_supplementary[
                (enhanced_supplementary['game_id'] == game_id) & 
                (enhanced_supplementary['play_id'] == play_id)
            ]
            
            tactical_context = {
                'route_type': 'UNKNOWN',
                'coverage_type': 'UNKNOWN',
                'epa_situation': 'NEUTRAL'
            }
            
            if len(play_context) > 0:
                context_row = play_context.iloc[0]
                if 'route_type_clean' in context_row.index:
                    tactical_context['route_type'] = str(context_row['route_type_clean'])
                if 'coverage_type_clean' in context_row.index:
                    tactical_context['coverage_type'] = str(context_row['coverage_type_clean'])
                if 'epa_situation' in context_row.index:
                    tactical_context['epa_situation'] = str(context_row['epa_situation'])
            
            # Performance metrics
            effort_scores = play_trajectories['comprehensive_effort_score'].values
            
            performance_metrics = {
                'total_players': int(len(play_trajectories)),
                'receivers_identified': int(len(receivers)),
                'avg_effort_score': float(np.mean(effort_scores)),
                'max_effort_score': float(np.max(effort_scores)),
                'min_effort_score': float(np.min(effort_scores)),
                'total_distance': float(play_trajectories['total_distance'].sum()),
                'avg_path_efficiency': float(play_trajectories['path_efficiency'].mean()),
                'elite_performers': int(np.sum(effort_scores > 80)),
                'high_performers': int(np.sum(effort_scores > 60))
            }
            
            ball_window = {
                'game_id': int(game_id),
                'play_id': int(play_id),
                'frame_release': int(frame_release),
                'frame_arrival': int(frame_arrival),
                'ball_in_air_duration': int(ball_in_air_duration),
                'total_play_frames': int(estimated_play_duration),
                'ball_land_x': ball_land_x,
                'ball_land_y': ball_land_y,
                'estimation_method': estimation_method,
                'route_type': tactical_context['route_type'],
                'coverage_type': tactical_context['coverage_type'],
                'epa_situation': tactical_context['epa_situation'],
                **performance_metrics,
                'data_quality': 'simplified',
                'processing_success': True
            }
            
            ball_windows.append(ball_window)
            success_count += 1
            
        except Exception as e:
            error_count += 1
            if error_count <= 3:
                print(f"            Error: {game_id}-{play_id}: {e}")
            continue
    
    # =====================================================================
    # STEP 3: RESULTS VALIDATION
    # =====================================================================
    
    print(f"\n      STEP 3: RESULTS VALIDATION")
    
    if ball_windows:
        ball_windows_df = pd.DataFrame(ball_windows)
        
        print(f"         Ball windows created: {len(ball_windows_df):,}")
        print(f"         Success rate: {success_count/len(valid_plays)*100:.1f}%")
        print(f"         Error count: {error_count:,}")
        
        print(f"         Statistics:")
        print(f"            Avg duration: {ball_windows_df['ball_in_air_duration'].mean():.1f} frames")
        print(f"            Avg players/play: {ball_windows_df['total_players'].mean():.1f}")
        print(f"            Avg effort: {ball_windows_df['avg_effort_score'].mean():.1f}")
        
        elite_plays = ball_windows_df['elite_performers'].sum()
        print(f"            Elite performances: {elite_plays:,}")
        
        ball_stats = {
            'total_windows': len(ball_windows_df),
            'success_rate': success_count / len(valid_plays),
            'error_rate': error_count / len(valid_plays),
            'avg_duration': ball_windows_df['ball_in_air_duration'].mean(),
            'avg_players_per_play': ball_windows_df['total_players'].mean(),
            'avg_effort_score': ball_windows_df['avg_effort_score'].mean(),
            'total_elite_performances': elite_plays,
            'unique_games': ball_windows_df['game_id'].nunique()
        }
        
        return {
            'fixed_ball_windows_df': ball_windows_df,
            'fixed_ball_stats': ball_stats
        }
    
    else:
        print(f"         No ball windows created")
        print(f"         Errors: {error_count:,}")
        return None

# Execute
fixed_ball_results = create_fixed_ball_in_air_analysis()

# CRITICAL: Save to championship_data for Cell 7+
if fixed_ball_results:
    championship_data['ball_windows_df'] = fixed_ball_results['fixed_ball_windows_df']
    print(f"\nBall windows saved: {len(championship_data['ball_windows_df']):,} records")

aggressive_gc()

print(f"\nMemory after ball analysis: {get_memory_usage():.1f} MB")

if fixed_ball_results:
    print("CELL 6 COMPLETE - BALL-IN-AIR ANALYSIS DONE!")
    stats = fixed_ball_results['fixed_ball_stats']
    print(f"\nRESULTS:")
    print(f"   Windows: {stats['total_windows']:,}")
    print(f"   Success rate: {stats['success_rate']*100:.1f}%")
    print(f"   Avg effort: {stats['avg_effort_score']:.1f}")
    print(f"   Elite performances: {stats['total_elite_performances']:,}")
else:
    print("CELL 6 FAILED - Check data")


# ========================================================================
# CELL 7: BCS CALCULATION CORE - KAGGLE SAFE
# Ball Convergence Score with 4 components
# ========================================================================

print("\nBCS CALCULATION CORE")
print("="*45)

@timing_decorator
def calculate_working_championship_bcs_core():
    """Calculate championship BCS using reliable approach"""
    
    print(f"\n   Memory: {get_memory_gb():.2f}GB")
    
    if massive_motion_results is None:
        print("      Missing massive_motion_results")
        return None
    
    if 'ball_windows_df' not in championship_data:
        print("      Missing ball_windows_df")
        print("      Check Cell 6 ran correctly!")
        return None
    
    enhanced_trajectories = massive_motion_results['enhanced_trajectories_df']
    ball_windows = championship_data['ball_windows_df']
    
    print(f"      BCS processing:")
    print(f"         Enhanced trajectories: {len(enhanced_trajectories):,}")
    print(f"         Ball windows: {len(ball_windows):,}")
    
    # =====================================================================
    # STEP 1: TRAJECTORY-BALL INTEGRATION
    # =====================================================================
    
    print(f"\n      STEP 1: TRAJECTORY-BALL INTEGRATION")
    
    ball_simple = ball_windows[['game_id', 'play_id', 'ball_land_x', 'ball_land_y']].copy()
    
    trajectory_ball_merged = enhanced_trajectories.merge(
        ball_simple,
        on=['game_id', 'play_id'],
        how='inner'
    )
    
    print(f"         Merged records: {len(trajectory_ball_merged):,}")
    print(f"         Integration rate: {len(trajectory_ball_merged)/len(enhanced_trajectories)*100:.1f}%")
    
    if len(trajectory_ball_merged) == 0:
        print("      No records after merge")
        return None
    
    # =====================================================================
    # STEP 2: BCS CALCULATIONS
    # =====================================================================
    
    print(f"\n      STEP 2: BCS CALCULATIONS")
    
    bcs_records = []
    
    print(f"         Calculating BCS for {len(trajectory_ball_merged):,} trajectories...")
    
    for idx, traj in trajectory_ball_merged.iterrows():
        if idx % 20000 == 0:
            print(f"            Progress: {idx:,}/{len(trajectory_ball_merged):,}")
        
        try:
            ball_x, ball_y = traj['ball_land_x'], traj['ball_land_y']
            end_x, end_y = traj['end_x'], traj['end_y']
            
            if pd.isna(ball_x) or pd.isna(ball_y) or pd.isna(end_x) or pd.isna(end_y):
                continue
            
            # COMPONENT 1: PROXIMITY SCORE (40%)
            final_distance = np.sqrt((end_x - ball_x)**2 + (end_y - ball_y)**2)
            
            if final_distance <= 2:
                proximity_score = 100
            elif final_distance <= 5:
                proximity_score = 85 - (final_distance - 2) * 5
            elif final_distance <= 10:
                proximity_score = 70 - (final_distance - 5) * 4
            elif final_distance <= 20:
                proximity_score = 50 - (final_distance - 10) * 2
            else:
                proximity_score = max(0, 30 - (final_distance - 20) * 1)
            
            # COMPONENT 2: PATH EFFICIENCY (30%)
            path_efficiency = traj['path_efficiency']
            efficiency_score = min(100, path_efficiency * 100)
            
            # COMPONENT 3: EFFORT SCORE (20%)
            effort_score = min(100, traj['comprehensive_effort_score'])
            
            # COMPONENT 4: VELOCITY FACTOR (10%)
            max_velocity = traj['max_velocity']
            velocity_score = min(100, max_velocity * 20)
            
            # COMBINED BCS SCORE
            base_bcs_score = (
                proximity_score * 0.40 +
                efficiency_score * 0.30 +
                effort_score * 0.20 +
                velocity_score * 0.10
            )
            
            # Role determination
            if traj['is_potential_receiver']:
                player_role = 'RECEIVER'
            elif traj['is_potential_defender']:
                player_role = 'DEFENDER'
            else:
                player_role = 'OTHER'
            
            # Performance tier
            if base_bcs_score >= 80:
                performance_tier = 'ELITE'
            elif base_bcs_score >= 65:
                performance_tier = 'EXCELLENT'
            elif base_bcs_score >= 50:
                performance_tier = 'GOOD'
            elif base_bcs_score >= 35:
                performance_tier = 'AVERAGE'
            else:
                performance_tier = 'BELOW_AVERAGE'
            
            bcs_record = {
                'game_id': traj['game_id'],
                'play_id': traj['play_id'],
                'player_id': traj['player_id'],
                'proximity_score': float(proximity_score),
                'efficiency_score': float(efficiency_score),
                'effort_score': float(effort_score),
                'velocity_score': float(velocity_score),
                'base_bcs_score': float(base_bcs_score),
                'epa_weighted_bcs': float(base_bcs_score),
                'final_distance_to_ball': float(final_distance),
                'player_role': player_role,
                'performance_tier': performance_tier,
                'comprehensive_effort_score': traj['comprehensive_effort_score'],
                'path_efficiency': traj['path_efficiency'],
                'total_distance': traj['total_distance'],
                'max_velocity': traj['max_velocity']
            }
            
            bcs_records.append(bcs_record)
            
        except Exception:
            continue
    
    # =====================================================================
    # STEP 3: RESULTS VALIDATION
    # =====================================================================
    
    print(f"\n      STEP 3: BCS RESULTS")
    
    if bcs_records:
        bcs_df = pd.DataFrame(bcs_records)
        
        print(f"         BCS calculations: {len(bcs_df):,}")
        print(f"         Success rate: {len(bcs_df)/len(trajectory_ball_merged)*100:.1f}%")
        print(f"         Average BCS: {bcs_df['base_bcs_score'].mean():.1f}")
        
        print(f"         Component analysis:")
        print(f"            Proximity: {bcs_df['proximity_score'].mean():.1f}")
        print(f"            Efficiency: {bcs_df['efficiency_score'].mean():.1f}")
        print(f"            Effort: {bcs_df['effort_score'].mean():.1f}")
        print(f"            Velocity: {bcs_df['velocity_score'].mean():.1f}")
        
        tier_dist = bcs_df['performance_tier'].value_counts()
        print(f"         Performance tiers:")
        for tier, count in tier_dist.items():
            print(f"            {tier}: {count:,} ({count/len(bcs_df)*100:.1f}%)")
        
        role_dist = bcs_df['player_role'].value_counts()
        print(f"         Role distribution:")
        for role, count in role_dist.items():
            avg_bcs = bcs_df[bcs_df['player_role'] == role]['base_bcs_score'].mean()
            print(f"            {role}: {count:,} (avg BCS: {avg_bcs:.1f})")
        
        working_bcs_stats = {
            'total_bcs_calculations': len(bcs_df),
            'avg_bcs_score': bcs_df['base_bcs_score'].mean(),
            'bcs_std': bcs_df['base_bcs_score'].std(),
            'performance_distribution': tier_dist.to_dict(),
            'role_distribution': role_dist.to_dict(),
            'elite_performers': len(bcs_df[bcs_df['performance_tier'] == 'ELITE'])
        }
        
        return {
            'championship_bcs_df': bcs_df,
            'championship_bcs_stats': working_bcs_stats
        }
    
    else:
        print(f"         No BCS calculations completed")
        return None

# Execute
championship_bcs_results = calculate_working_championship_bcs_core()

# Save to championship_data
if championship_bcs_results:
    championship_data['championship_bcs_df'] = championship_bcs_results['championship_bcs_df']
    championship_data['championship_bcs_stats'] = championship_bcs_results['championship_bcs_stats']

aggressive_gc()

print(f"\nMemory after BCS: {get_memory_usage():.1f} MB")

if championship_bcs_results:
    print("CELL 7 COMPLETE - BCS CALCULATION DONE!")
    stats = championship_bcs_results['championship_bcs_stats']
    print(f"\nRESULTS:")
    print(f"   BCS calculations: {stats['total_bcs_calculations']:,}")
    print(f"   Average BCS: {stats['avg_bcs_score']:.1f}")
    print(f"   Elite performers: {stats['elite_performers']:,}")
else:
    print("CELL 7 FAILED - Check Cell 6")


# ========================================================================
# CELL 8: ADVANCED BCS AGGREGATION - KAGGLE SAFE (COMPLETE)
# Player-level analysis with position weighting - ALL COLUMNS INCLUDED
# ========================================================================

print("\nADVANCED BCS AGGREGATION")
print("="*45)

@timing_decorator
def calculate_advanced_player_bcs_aggregation():
    """Advanced BCS aggregation with position weighting - COMPLETE VERSION"""
    
    print(f"\n   Memory: {get_memory_gb():.2f}GB")
    
    if championship_bcs_results is None:
        print("      Missing BCS results")
        return None
    
    bcs_df = championship_bcs_results['championship_bcs_df'].copy()
    enhanced_traj = massive_motion_results['enhanced_trajectories_df']
    
    print(f"      Processing {len(bcs_df):,} BCS records from {bcs_df['player_id'].nunique():,} players")
    
    # =====================================================================
    # STEP 1: COMPREHENSIVE PLAYER AGGREGATION
    # =====================================================================
    
    print(f"\n      STEP 1: PLAYER AGGREGATION")
    
    player_agg = bcs_df.groupby('player_id').agg({
        # BCS Core metrics
        'base_bcs_score': ['mean', 'std', 'min', 'max', 'median', 'count'],
        # Component scores
        'proximity_score': ['mean', 'std', 'min', 'max'],
        'efficiency_score': ['mean', 'std', 'min', 'max'],
        'effort_score': ['mean', 'std', 'min', 'max'],
        'velocity_score': ['mean', 'std', 'min', 'max'],
        # Performance metrics
        'final_distance_to_ball': ['mean', 'std', 'min'],
        'comprehensive_effort_score': ['mean', 'max', 'std'],
        'path_efficiency': ['mean', 'max', 'std'],
        'total_distance': ['mean', 'sum', 'max', 'std'],
        'max_velocity': ['mean', 'max', 'std'],
        # Context
        'game_id': 'nunique',
        'play_id': 'nunique'
    }).reset_index()
    
    # Flatten columns
    player_agg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                          for col in player_agg.columns]
    
    # Rename key columns
    rename_map = {
        'base_bcs_score_mean': 'avg_bcs',
        'base_bcs_score_std': 'bcs_std',
        'base_bcs_score_min': 'min_bcs',
        'base_bcs_score_max': 'max_bcs',
        'base_bcs_score_median': 'median_bcs',
        'base_bcs_score_count': 'total_plays',
        'game_id_nunique': 'games_played',
        'play_id_nunique': 'unique_plays',
        'proximity_score_mean': 'avg_proximity',
        'proximity_score_std': 'proximity_std',
        'proximity_score_min': 'min_proximity',
        'proximity_score_max': 'max_proximity',
        'efficiency_score_mean': 'avg_efficiency',
        'efficiency_score_std': 'efficiency_std',
        'efficiency_score_min': 'min_efficiency',
        'efficiency_score_max': 'max_efficiency',
        'effort_score_mean': 'avg_effort',
        'effort_score_std': 'effort_std',
        'effort_score_min': 'min_effort',
        'effort_score_max': 'max_effort',
        'velocity_score_mean': 'avg_velocity',
        'velocity_score_std': 'velocity_std',
        'velocity_score_min': 'min_velocity',
        'velocity_score_max': 'max_velocity',
        'final_distance_to_ball_mean': 'avg_distance_to_ball',
        'final_distance_to_ball_std': 'distance_to_ball_std',
        'final_distance_to_ball_min': 'min_distance_to_ball',
        'comprehensive_effort_score_mean': 'avg_comp_effort',
        'comprehensive_effort_score_max': 'max_comp_effort',
        'comprehensive_effort_score_std': 'comp_effort_std',
        'path_efficiency_mean': 'avg_path_efficiency',
        'path_efficiency_max': 'max_path_efficiency',
        'path_efficiency_std': 'path_efficiency_std',
        'total_distance_mean': 'avg_total_distance',
        'total_distance_sum': 'cumulative_distance',
        'total_distance_max': 'max_single_distance',
        'total_distance_std': 'distance_std',
        'max_velocity_mean': 'avg_max_velocity',
        'max_velocity_max': 'peak_velocity',
        'max_velocity_std': 'velocity_variance'
    }
    player_agg = player_agg.rename(columns={k: v for k, v in rename_map.items() if k in player_agg.columns})
    
    print(f"         Base aggregation: {len(player_agg):,} players")
    
    # =====================================================================
    # STEP 2: PLAYER ROLE INFERENCE
    # =====================================================================
    
    print(f"\n      STEP 2: PLAYER ROLE INFERENCE")
    
    # Get dominant role and tier
    player_roles = bcs_df.groupby('player_id').agg({
        'player_role': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'OTHER',
        'performance_tier': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'AVERAGE'
    }).reset_index()
    player_roles.columns = ['player_id', 'dominant_role', 'dominant_tier']
    
    player_agg = player_agg.merge(player_roles, on='player_id', how='left')
    
    # Trajectory features for position inference
    traj_agg = enhanced_traj.groupby('player_id').agg({
        'is_potential_receiver': 'mean',
        'is_potential_defender': 'mean',
        'sharp_cuts': 'mean',
        'acceleration_bursts': 'mean',
        'sprint_ratio': 'mean'
    }).reset_index()
    traj_agg.columns = ['player_id', 'receiver_likelihood', 'defender_likelihood', 
                        'avg_sharp_cuts', 'avg_accel_bursts', 'avg_sprint_ratio']
    
    player_agg = player_agg.merge(traj_agg, on='player_id', how='left')
    
    # Position inference
    def infer_position(row):
        if row['dominant_role'] == 'RECEIVER':
            if row.get('peak_velocity', 0) > 5 and row.get('avg_sharp_cuts', 0) > 2:
                return 'WR_SPEED'
            elif row.get('avg_path_efficiency', 0) > 0.75:
                return 'WR_ROUTE'
            elif row.get('avg_total_distance', 0) < 20:
                return 'RB'
            else:
                return 'TE'
        elif row['dominant_role'] == 'DEFENDER':
            if row.get('avg_comp_effort', 0) > 60:
                return 'CB'
            elif row.get('avg_total_distance', 0) > 20:
                return 'S'
            elif row.get('avg_accel_bursts', 0) > 3:
                return 'LB'
            else:
                return 'DL'
        return 'OTHER'
    
    player_agg['inferred_position'] = player_agg.apply(infer_position, axis=1)
    
    pos_dist = player_agg['inferred_position'].value_counts()
    print(f"         Position distribution:")
    for pos, count in pos_dist.head(6).items():
        avg = player_agg[player_agg['inferred_position'] == pos]['avg_bcs'].mean()
        print(f"            {pos}: {count:,} players ({avg:.1f} avg BCS)")
    
    # =====================================================================
    # STEP 3: CONSISTENCY & RELIABILITY METRICS
    # =====================================================================
    
    print(f"\n      STEP 3: CONSISTENCY METRICS")
    
    # Fill NaN std with 0
    std_cols = [col for col in player_agg.columns if '_std' in col]
    for col in std_cols:
        player_agg[col] = player_agg[col].fillna(0)
    
    # Consistency scores (lower std = more consistent)
    player_agg['bcs_consistency'] = 100 - np.minimum(player_agg['bcs_std'] * 3, 60)
    player_agg['proximity_consistency'] = 100 - np.minimum(player_agg['proximity_std'] * 2, 50)
    player_agg['efficiency_consistency'] = 100 - np.minimum(player_agg['efficiency_std'] * 2, 50)
    player_agg['effort_consistency'] = 100 - np.minimum(player_agg['effort_std'] * 2, 50)
    
    # Overall consistency (weighted average)
    player_agg['overall_consistency'] = (
        player_agg['bcs_consistency'] * 0.4 +
        player_agg['proximity_consistency'] * 0.25 +
        player_agg['efficiency_consistency'] * 0.2 +
        player_agg['effort_consistency'] * 0.15
    )
    
    # Reliability based on sample size
    player_agg['sample_reliability'] = np.minimum(player_agg['total_plays'] / 15 * 100, 100)
    player_agg['game_coverage'] = np.minimum(player_agg['games_played'] / 10 * 100, 100)
    
    # Performance range
    player_agg['bcs_range'] = player_agg['max_bcs'] - player_agg['min_bcs']
    player_agg['bcs_iqr'] = player_agg['bcs_range'] * 0.5  # Approximate
    
    # Upside potential and floor risk
    player_agg['upside_potential'] = player_agg['max_bcs'] - player_agg['avg_bcs']
    player_agg['floor_risk'] = player_agg['avg_bcs'] - player_agg['min_bcs']
    
    print(f"         Avg consistency: {player_agg['overall_consistency'].mean():.1f}")
    print(f"         Avg sample reliability: {player_agg['sample_reliability'].mean():.1f}")
    print(f"         Avg game coverage: {player_agg['game_coverage'].mean():.1f}")
    
    # =====================================================================
    # STEP 4: COMPONENT STRENGTH ANALYSIS
    # =====================================================================
    
    print(f"\n      STEP 4: COMPONENT ANALYSIS")
    
    def get_component_analysis(row):
        comp_scores = {
            'proximity': row.get('avg_proximity', 0),
            'efficiency': row.get('avg_efficiency', 0),
            'effort': row.get('avg_effort', 0),
            'velocity': row.get('avg_velocity', 0)
        }
        sorted_comps = sorted(comp_scores.items(), key=lambda x: x[1], reverse=True)
        strongest = sorted_comps[0][0]
        weakest = sorted_comps[-1][0]
        gap = sorted_comps[0][1] - sorted_comps[-1][1]
        return strongest, weakest, gap
    
    component_analysis = player_agg.apply(get_component_analysis, axis=1, result_type='expand')
    player_agg['strongest_component'] = component_analysis[0]
    player_agg['weakest_component'] = component_analysis[1]
    player_agg['component_gap'] = component_analysis[2]
    
    # Component balance score (lower gap = more balanced)
    player_agg['component_balance'] = 100 - np.minimum(player_agg['component_gap'], 50)
    
    strength_dist = player_agg['strongest_component'].value_counts()
    print(f"         Strongest component distribution:")
    for comp, count in strength_dist.items():
        print(f"            {comp}: {count:,} players")
    
    # =====================================================================
    # STEP 5: POSITION-SPECIFIC WEIGHTING
    # =====================================================================
    
    print(f"\n      STEP 5: POSITION-SPECIFIC WEIGHTING")
    
    POSITION_WEIGHTS = {
        'WR_SPEED': {'proximity': 0.45, 'efficiency': 0.30, 'effort': 0.10, 'velocity': 0.15},
        'WR_ROUTE': {'proximity': 0.40, 'efficiency': 0.40, 'effort': 0.10, 'velocity': 0.10},
        'TE': {'proximity': 0.40, 'efficiency': 0.30, 'effort': 0.20, 'velocity': 0.10},
        'RB': {'proximity': 0.35, 'efficiency': 0.30, 'effort': 0.25, 'velocity': 0.10},
        'CB': {'proximity': 0.30, 'efficiency': 0.25, 'effort': 0.30, 'velocity': 0.15},
        'S': {'proximity': 0.25, 'efficiency': 0.30, 'effort': 0.35, 'velocity': 0.10},
        'LB': {'proximity': 0.35, 'efficiency': 0.25, 'effort': 0.30, 'velocity': 0.10},
        'DL': {'proximity': 0.45, 'efficiency': 0.15, 'effort': 0.30, 'velocity': 0.10},
        'OTHER': {'proximity': 0.35, 'efficiency': 0.30, 'effort': 0.25, 'velocity': 0.10}
    }
    
    def calc_position_weighted_bcs(row):
        pos = row['inferred_position']
        weights = POSITION_WEIGHTS.get(pos, POSITION_WEIGHTS['OTHER'])
        return (
            row['avg_proximity'] * weights['proximity'] +
            row['avg_efficiency'] * weights['efficiency'] +
            row['avg_effort'] * weights['effort'] +
            row['avg_velocity'] * weights['velocity']
        )
    
    player_agg['position_weighted_bcs'] = player_agg.apply(calc_position_weighted_bcs, axis=1)
    player_agg['bcs_weight_adjustment'] = player_agg['position_weighted_bcs'] - player_agg['avg_bcs']
    
    print(f"         Avg standard BCS: {player_agg['avg_bcs'].mean():.1f}")
    print(f"         Avg position-weighted BCS: {player_agg['position_weighted_bcs'].mean():.1f}")
    
    # =====================================================================
    # STEP 6: COMPOSITE SCORING
    # =====================================================================
    
    print(f"\n      STEP 6: COMPOSITE SCORING")
    
    player_agg['composite_score'] = (
        player_agg['position_weighted_bcs'] * 0.35 +
        player_agg['avg_bcs'] * 0.20 +
        player_agg['overall_consistency'] * 0.15 +
        player_agg['sample_reliability'] * 0.10 +
        player_agg['max_bcs'] * 0.10 +
        (100 - player_agg['avg_distance_to_ball'].clip(0, 20) * 5) * 0.10
    )
    
    def assign_tier(row):
        score = row['composite_score']
        plays = row['total_plays']
        consistency = row['overall_consistency']
        
        if score >= 70 and plays >= 8 and consistency >= 60:
            return 'CHAMPIONSHIP_ELITE'
        elif score >= 65 and plays >= 5:
            return 'ELITE'
        elif score >= 58 and plays >= 4:
            return 'EXCELLENT'
        elif score >= 50:
            return 'GOOD'
        elif score >= 42:
            return 'AVERAGE'
        else:
            return 'DEVELOPING'
    
    player_agg['championship_tier'] = player_agg.apply(assign_tier, axis=1)
    
    tier_dist = player_agg['championship_tier'].value_counts()
    print(f"         Championship tiers:")
    for tier, count in tier_dist.items():
        avg_score = player_agg[player_agg['championship_tier'] == tier]['composite_score'].mean()
        print(f"            {tier}: {count:,} ({avg_score:.1f} avg)")
    
    # =====================================================================
    # STEP 7: PERCENTILE RANKINGS
    # =====================================================================
    
    print(f"\n      STEP 7: PERCENTILE RANKINGS")
    
    player_agg['bcs_percentile'] = player_agg['avg_bcs'].rank(pct=True) * 100
    player_agg['composite_percentile'] = player_agg['composite_score'].rank(pct=True) * 100
    player_agg['consistency_percentile'] = player_agg['overall_consistency'].rank(pct=True) * 100
    
    # Component percentiles
    for comp in ['proximity', 'efficiency', 'effort', 'velocity']:
        col = f'avg_{comp}'
        if col in player_agg.columns:
            player_agg[f'{comp}_percentile'] = player_agg[col].rank(pct=True) * 100
    
    print(f"         Percentile rankings computed")
    
    # =====================================================================
    # STEP 8: TOP PERFORMERS & STORAGE
    # =====================================================================
    
    print(f"\n      STEP 8: TOP PERFORMERS")
    
    qualified = player_agg[player_agg['total_plays'] >= 3].copy()
    top_by_composite = qualified.nlargest(25, 'composite_score')
    elite_players = qualified[qualified['championship_tier'].isin(['CHAMPIONSHIP_ELITE', 'ELITE'])]
    
    print(f"         Qualified players: {len(qualified):,}")
    print(f"         Elite performers: {len(elite_players):,}")
    
    print(f"\n         TOP 10 BY COMPOSITE:")
    for rank, (_, p) in enumerate(top_by_composite.head(10).iterrows(), 1):
        print(f"            {rank:2d}. Player {int(p['player_id'])} ({p['inferred_position']}): "
              f"{p['composite_score']:.1f} | {p['avg_bcs']:.1f} BCS | {int(p['total_plays'])} plays")
    
    # Store in championship_data
    championship_data['player_aggregations'] = player_agg
    championship_data['qualified_players'] = qualified
    championship_data['elite_players'] = elite_players
    championship_data['top_performers'] = top_by_composite
    championship_data['position_weights'] = POSITION_WEIGHTS
    
    aggregation_stats = {
        'total_players': len(player_agg),
        'qualified_players': len(qualified),
        'elite_players': len(elite_players),
        'championship_elite': len(player_agg[player_agg['championship_tier'] == 'CHAMPIONSHIP_ELITE']),
        'avg_bcs': float(player_agg['avg_bcs'].mean()),
        'avg_composite': float(player_agg['composite_score'].mean()),
        'avg_consistency': float(player_agg['overall_consistency'].mean()),
        'top_performer_id': int(top_by_composite.iloc[0]['player_id']) if len(top_by_composite) > 0 else None,
        'top_performer_score': float(top_by_composite.iloc[0]['composite_score']) if len(top_by_composite) > 0 else 0
    }
    
    aggressive_gc()
    
    return {
        'player_aggregations_df': player_agg,
        'qualified_players_df': qualified,
        'elite_players_df': elite_players,
        'top_performers_df': top_by_composite,
        'aggregation_stats': aggregation_stats,
        'position_weights': POSITION_WEIGHTS
    }

# Execute
player_bcs_results = calculate_advanced_player_bcs_aggregation()

aggressive_gc()

print(f"\nMemory after aggregation: {get_memory_usage():.1f} MB")

if player_bcs_results:
    print("CELL 8 COMPLETE - ADVANCED AGGREGATION DONE!")
    s = player_bcs_results['aggregation_stats']
    print(f"\nRESULTS:")
    print(f"   Players: {s['total_players']:,} total | {s['qualified_players']:,} qualified | {s['elite_players']:,} elite")
    print(f"   Top performer: Player {s['top_performer_id']} ({s['top_performer_score']:.1f})")
    print(f"   Avg composite: {s['avg_composite']:.1f}")
else:
    print("CELL 8 FAILED - Check Cell 7")


# ========================================================================
# CELL 9: ADVANCED POSITION ANALYSIS - KAGGLE SAFE (COMPLETE)
# Position hierarchy with ALL statistics
# ========================================================================

print("\nADVANCED POSITION ANALYSIS")
print("="*45)

@timing_decorator
def calculate_advanced_position_analysis():
    """Advanced position analysis with hierarchy and benchmarking - COMPLETE"""
    
    print(f"\n   Memory: {get_memory_gb():.2f}GB")
    
    if player_bcs_results is None or championship_bcs_results is None:
        print("      Missing required data")
        return None
    
    bcs_df = championship_bcs_results['championship_bcs_df'].copy()
    player_agg = player_bcs_results['player_aggregations_df'].copy()
    enhanced_traj = massive_motion_results['enhanced_trajectories_df']
    
    print(f"      Analyzing: {len(bcs_df):,} records | {len(player_agg):,} players")
    
    # =====================================================================
    # STEP 1: POSITION INFERENCE
    # =====================================================================
    
    print(f"\n      STEP 1: POSITION INFERENCE")
    
    traj_features = enhanced_traj[['game_id', 'play_id', 'player_id', 
                                    'is_potential_receiver', 'is_potential_defender',
                                    'sharp_cuts', 'acceleration_bursts', 'sprint_ratio',
                                    'max_path_deviation', 'direction_efficiency',
                                    'start_x', 'start_y', 'end_x', 'end_y']].copy()
    
    position_df = bcs_df.merge(traj_features, on=['game_id', 'play_id', 'player_id'], 
                               how='left', suffixes=('', '_traj'))
    
    def infer_detailed_position(row):
        role = row['player_role']
        
        if role == 'RECEIVER':
            if row['max_velocity'] > 4.5 and row.get('sharp_cuts', 0) >= 3:
                return 'WR_DEEP'
            elif row['path_efficiency'] > 0.8 and row.get('direction_efficiency', 0) > 0.7:
                return 'WR_SLOT'
            elif row['max_velocity'] > 4.0:
                return 'WR_OUTSIDE'
            elif row['total_distance'] < 15 and row.get('acceleration_bursts', 0) > 2:
                return 'RB_RECEIVING'
            elif row['total_distance'] > 15:
                return 'TE_RECEIVING'
            else:
                return 'TE_BLOCKING'
        elif role == 'DEFENDER':
            effort = row['comprehensive_effort_score']
            distance = row['total_distance']
            
            if effort > 65 and distance > 28:
                return 'CB_SHADOW'
            elif effort > 55 and distance > 22:
                return 'CB_ZONE'
            elif distance > 25 and row.get('sprint_ratio', 0) > 0.3:
                return 'S_FREE'
            elif distance > 18:
                return 'S_STRONG'
            elif effort > 50 and row.get('acceleration_bursts', 0) > 3:
                return 'LB_COVERAGE'
            elif distance > 12:
                return 'LB_RUN'
            elif row.get('acceleration_bursts', 0) > 2:
                return 'DE_SPEED'
            else:
                return 'DL_INTERIOR'
        return 'OTHER'
    
    position_df['detailed_position'] = position_df.apply(infer_detailed_position, axis=1)
    
    position_groups = {
        'WR': ['WR_DEEP', 'WR_SLOT', 'WR_OUTSIDE'],
        'TE': ['TE_RECEIVING', 'TE_BLOCKING'],
        'RB': ['RB_RECEIVING'],
        'CB': ['CB_SHADOW', 'CB_ZONE'],
        'S': ['S_FREE', 'S_STRONG'],
        'LB': ['LB_COVERAGE', 'LB_RUN'],
        'DL': ['DE_SPEED', 'DL_INTERIOR']
    }
    
    def get_position_group(pos):
        for group, positions in position_groups.items():
            if pos in positions:
                return group
        return 'OTHER'
    
    position_df['position_group'] = position_df['detailed_position'].apply(get_position_group)
    
    detailed_dist = position_df['detailed_position'].value_counts()
    print(f"         Position distribution:")
    for pos, count in detailed_dist.head(10).items():
        avg_bcs = position_df[position_df['detailed_position'] == pos]['base_bcs_score'].mean()
        print(f"            {pos}: {count:,} ({avg_bcs:.1f} avg BCS)")
    
    # =====================================================================
    # STEP 2: COMPREHENSIVE POSITION STATISTICS
    # =====================================================================
    
    print(f"\n      STEP 2: POSITION STATISTICS")
    
    position_stats = {}
    
    for pos in position_df['detailed_position'].unique():
        pos_data = position_df[position_df['detailed_position'] == pos]
        
        if len(pos_data) < 10:
            continue
        
        stats = {
            # Volume metrics
            'count': len(pos_data),
            'player_count': pos_data['player_id'].nunique(),
            'plays_per_player': len(pos_data) / max(pos_data['player_id'].nunique(), 1),
            
            # BCS metrics
            'avg_bcs': pos_data['base_bcs_score'].mean(),
            'std_bcs': pos_data['base_bcs_score'].std(),
            'median_bcs': pos_data['base_bcs_score'].median(),
            'min_bcs': pos_data['base_bcs_score'].min(),
            'max_bcs': pos_data['base_bcs_score'].max(),
            'bcs_range': pos_data['base_bcs_score'].max() - pos_data['base_bcs_score'].min(),
            
            # Percentiles
            'p10_bcs': pos_data['base_bcs_score'].quantile(0.10),
            'p25_bcs': pos_data['base_bcs_score'].quantile(0.25),
            'p50_bcs': pos_data['base_bcs_score'].quantile(0.50),
            'p75_bcs': pos_data['base_bcs_score'].quantile(0.75),
            'p90_bcs': pos_data['base_bcs_score'].quantile(0.90),
            'p95_bcs': pos_data['base_bcs_score'].quantile(0.95),
            
            # Component scores
            'avg_proximity': pos_data['proximity_score'].mean(),
            'avg_efficiency': pos_data['efficiency_score'].mean(),
            'avg_effort': pos_data['effort_score'].mean(),
            'avg_velocity': pos_data['velocity_score'].mean(),
            
            # Performance metrics
            'avg_distance_to_ball': pos_data['final_distance_to_ball'].mean(),
            'min_distance_to_ball': pos_data['final_distance_to_ball'].min(),
            'avg_total_distance': pos_data['total_distance'].mean(),
            'avg_max_velocity': pos_data['max_velocity'].mean(),
            'avg_path_efficiency': pos_data['path_efficiency'].mean(),
            
            # Tier distribution
            'elite_count': len(pos_data[pos_data['performance_tier'] == 'ELITE']),
            'excellent_count': len(pos_data[pos_data['performance_tier'] == 'EXCELLENT']),
            'good_count': len(pos_data[pos_data['performance_tier'] == 'GOOD']),
            'elite_rate': len(pos_data[pos_data['performance_tier'] == 'ELITE']) / len(pos_data) * 100,
            'excellent_rate': len(pos_data[pos_data['performance_tier'].isin(['ELITE', 'EXCELLENT'])]) / len(pos_data) * 100
        }
        
        # Component strength ranking
        components = {
            'proximity': stats['avg_proximity'],
            'efficiency': stats['avg_efficiency'],
            'effort': stats['avg_effort'],
            'velocity': stats['avg_velocity']
        }
        sorted_comps = sorted(components.items(), key=lambda x: x[1], reverse=True)
        stats['strongest_component'] = sorted_comps[0][0]
        stats['weakest_component'] = sorted_comps[-1][0]
        stats['component_gap'] = sorted_comps[0][1] - sorted_comps[-1][1]
        
        position_stats[pos] = stats
    
    print(f"         Statistics for {len(position_stats)} positions")
    
    # =====================================================================
    # STEP 3: POSITION GROUP HIERARCHY
    # =====================================================================
    
    print(f"\n      STEP 3: POSITION HIERARCHY")
    
    group_stats = {}
    
    for group in ['WR', 'TE', 'RB', 'CB', 'S', 'LB', 'DL']:
        group_data = position_df[position_df['position_group'] == group]
        
        if len(group_data) < 20:
            continue
        
        group_stats[group] = {
            'count': len(group_data),
            'player_count': group_data['player_id'].nunique(),
            'avg_bcs': group_data['base_bcs_score'].mean(),
            'std_bcs': group_data['base_bcs_score'].std(),
            'median_bcs': group_data['base_bcs_score'].median(),
            'p90_bcs': group_data['base_bcs_score'].quantile(0.90),
            'elite_rate': len(group_data[group_data['performance_tier'] == 'ELITE']) / len(group_data) * 100,
            'avg_proximity': group_data['proximity_score'].mean(),
            'avg_efficiency': group_data['efficiency_score'].mean(),
            'avg_effort': group_data['effort_score'].mean(),
            'avg_velocity': group_data['velocity_score'].mean(),
            'avg_distance_to_ball': group_data['final_distance_to_ball'].mean(),
            'subtypes': position_df[position_df['position_group'] == group]['detailed_position'].value_counts().to_dict()
        }
    
    receiver_hierarchy = sorted(
        [(g, s) for g, s in group_stats.items() if g in ['WR', 'TE', 'RB']],
        key=lambda x: x[1]['avg_bcs'], reverse=True
    )
    
    defender_hierarchy = sorted(
        [(g, s) for g, s in group_stats.items() if g in ['CB', 'S', 'LB', 'DL']],
        key=lambda x: x[1]['avg_bcs'], reverse=True
    )
    
    print(f"         RECEIVER HIERARCHY:")
    for rank, (pos, stats) in enumerate(receiver_hierarchy, 1):
        print(f"            {rank}. {pos}: {stats['player_count']:,} players | {stats['avg_bcs']:.1f} avg | {stats['elite_rate']:.1f}% elite")
    
    print(f"         DEFENDER HIERARCHY:")
    for rank, (pos, stats) in enumerate(defender_hierarchy, 1):
        print(f"            {rank}. {pos}: {stats['player_count']:,} players | {stats['avg_bcs']:.1f} avg | {stats['elite_rate']:.1f}% elite")
    
    # =====================================================================
    # STEP 4: POSITION BENCHMARKS
    # =====================================================================
    
    print(f"\n      STEP 4: POSITION BENCHMARKS")
    
    benchmarks = {}
    for pos, stats in position_stats.items():
        benchmarks[pos] = {
            'elite_threshold': stats['p90_bcs'],
            'excellent_threshold': stats['p75_bcs'],
            'good_threshold': stats['median_bcs'],
            'below_average_threshold': stats['p25_bcs'],
            'floor': stats['p10_bcs'],
            'ceiling': stats['max_bcs'],
            'typical_range': (stats['p25_bcs'], stats['p75_bcs']),
            'consistency_benchmark': 100 - min(stats['std_bcs'] * 2, 50)
        }
    
    print(f"         Benchmarks for {len(benchmarks)} positions")
    
    # =====================================================================
    # STEP 5: COMPONENT MATRIX
    # =====================================================================
    
    print(f"\n      STEP 5: COMPONENT MATRIX")
    
    component_matrix = pd.DataFrame(index=list(position_stats.keys()))
    
    for pos, stats in position_stats.items():
        component_matrix.loc[pos, 'proximity'] = stats['avg_proximity']
        component_matrix.loc[pos, 'efficiency'] = stats['avg_efficiency']
        component_matrix.loc[pos, 'effort'] = stats['avg_effort']
        component_matrix.loc[pos, 'velocity'] = stats['avg_velocity']
        component_matrix.loc[pos, 'strongest'] = stats['strongest_component']
        component_matrix.loc[pos, 'weakest'] = stats['weakest_component']
        component_matrix.loc[pos, 'count'] = stats['count']
    
    print(f"         Component matrix: {len(component_matrix)} positions x {len(component_matrix.columns)} metrics")
    
    # =====================================================================
    # STEP 6: MATCHUP INTELLIGENCE
    # =====================================================================
    
    print(f"\n      STEP 6: MATCHUP INTELLIGENCE")
    
    matchup_matrix = {}
    
    for rec_group in ['WR', 'TE', 'RB']:
        if rec_group not in group_stats:
            continue
        for def_group in ['CB', 'S', 'LB', 'DL']:
            if def_group not in group_stats:
                continue
            
            matchup_key = f"{rec_group}_vs_{def_group}"
            rec_stats = group_stats[rec_group]
            def_stats = group_stats[def_group]
            
            bcs_diff = rec_stats['avg_bcs'] - def_stats['avg_bcs']
            p90_diff = rec_stats['p90_bcs'] - def_stats['p90_bcs']
            
            if bcs_diff > 5:
                offensive_advantage = 'STRONG'
                win_prob = min(0.80, 0.65 + bcs_diff * 0.02)
            elif bcs_diff > 0:
                offensive_advantage = 'SLIGHT'
                win_prob = 0.50 + bcs_diff * 0.02
            elif bcs_diff > -5:
                offensive_advantage = 'CONTESTED'
                win_prob = 0.50 + bcs_diff * 0.02
            else:
                offensive_advantage = 'DEFENSIVE'
                win_prob = max(0.35, 0.50 + bcs_diff * 0.02)
            
            matchup_matrix[matchup_key] = {
                'receiver_group': rec_group,
                'defender_group': def_group,
                'receiver_avg_bcs': rec_stats['avg_bcs'],
                'defender_avg_bcs': def_stats['avg_bcs'],
                'bcs_differential': bcs_diff,
                'p90_differential': p90_diff,
                'offensive_advantage': offensive_advantage,
                'estimated_win_prob': win_prob,
                'receiver_elite_rate': rec_stats['elite_rate'],
                'defender_elite_rate': def_stats['elite_rate']
            }
    
    sorted_matchups = sorted(matchup_matrix.items(), key=lambda x: x[1]['bcs_differential'], reverse=True)
    
    print(f"         Top offensive matchups:")
    for matchup, data in sorted_matchups[:3]:
        print(f"            {matchup}: {data['bcs_differential']:+.1f} diff | {data['estimated_win_prob']*100:.1f}% win")
    
    print(f"         Top defensive matchups:")
    for matchup, data in sorted_matchups[-3:]:
        print(f"            {matchup}: {data['bcs_differential']:+.1f} diff | {data['estimated_win_prob']*100:.1f}% win")
    
    # =====================================================================
    # STEP 7: STRATEGIC INSIGHTS
    # =====================================================================
    
    print(f"\n      STEP 7: STRATEGIC INSIGHTS")
    
    strategic_insights = {
        'receiver_hierarchy': [g for g, _ in receiver_hierarchy],
        'defender_hierarchy': [g for g, _ in defender_hierarchy],
        'best_matchups_for_offense': [m for m, d in sorted_matchups[:3]],
        'worst_matchups_for_offense': [m for m, d in sorted_matchups[-3:]],
        'highest_ceiling_position': max(position_stats.items(), key=lambda x: x[1]['max_bcs'])[0],
        'most_consistent_position': min(position_stats.items(), key=lambda x: x[1]['std_bcs'])[0],
        'highest_elite_rate': max(position_stats.items(), key=lambda x: x[1]['elite_rate'])[0],
        'best_proximity_position': max(position_stats.items(), key=lambda x: x[1]['avg_proximity'])[0],
        'best_effort_position': max(position_stats.items(), key=lambda x: x[1]['avg_effort'])[0]
    }
    
    print(f"         Key insights:")
    print(f"            Best receiver group: {receiver_hierarchy[0][0] if receiver_hierarchy else 'N/A'}")
    print(f"            Best defender group: {defender_hierarchy[0][0] if defender_hierarchy else 'N/A'}")
    print(f"            Highest ceiling: {strategic_insights['highest_ceiling_position']}")
    print(f"            Most consistent: {strategic_insights['most_consistent_position']}")
    
    # =====================================================================
    # STEP 8: STORAGE
    # =====================================================================
    
    print(f"\n      STEP 8: STORAGE")
    
    championship_data['position_analysis_df'] = position_df
    championship_data['position_stats'] = position_stats
    championship_data['position_group_stats'] = group_stats
    championship_data['position_benchmarks'] = benchmarks
    championship_data['component_matrix'] = component_matrix
    championship_data['matchup_matrix'] = matchup_matrix
    championship_data['position_hierarchy'] = {
        'receiver': dict(receiver_hierarchy),
        'defender': dict(defender_hierarchy)
    }
    championship_data['strategic_insights'] = strategic_insights
    
    position_analysis_stats = {
        'total_records': len(position_df),
        'unique_players': position_df['player_id'].nunique(),
        'detailed_positions': len(position_stats),
        'position_groups': len(group_stats),
        'matchups_analyzed': len(matchup_matrix),
        'receiver_avg_bcs': position_df[position_df['player_role'] == 'RECEIVER']['base_bcs_score'].mean(),
        'defender_avg_bcs': position_df[position_df['player_role'] == 'DEFENDER']['base_bcs_score'].mean()
    }
    
    print(f"         Positions: {position_analysis_stats['detailed_positions']}")
    print(f"         Matchups: {position_analysis_stats['matchups_analyzed']}")
    
    aggressive_gc()
    
    return {
        'position_analysis_df': position_df,
        'position_stats': position_stats,
        'group_stats': group_stats,
        'benchmarks': benchmarks,
        'component_matrix': component_matrix,
        'matchup_matrix': matchup_matrix,
        'receiver_hierarchy': receiver_hierarchy,
        'defender_hierarchy': defender_hierarchy,
        'strategic_insights': strategic_insights,
        'position_analysis_stats': position_analysis_stats
    }

# Execute
position_analysis_results = calculate_advanced_position_analysis()

aggressive_gc()

print(f"\nMemory after position analysis: {get_memory_usage():.1f} MB")

if position_analysis_results:
    print("CELL 9 COMPLETE - POSITION ANALYSIS DONE!")
    s = position_analysis_results['position_analysis_stats']
    rh = position_analysis_results['receiver_hierarchy']
    dh = position_analysis_results['defender_hierarchy']
    print(f"\nRESULTS:")
    print(f"   {s['detailed_positions']} positions | {s['matchups_analyzed']} matchups")
    print(f"   Receivers: {' > '.join([p for p, _ in rh])}")
    print(f"   Defenders: {' > '.join([p for p, _ in dh])}")
else:
    print("CELL 9 FAILED")


# ========================================================================
# CELL 10: BATTLE DETECTION SYSTEM - KAGGLE SAFE (COMPLETE)
# Battle analysis with ALL metrics
# ========================================================================

print("\nBATTLE DETECTION SYSTEM")
print("="*45)

@timing_decorator
def calculate_advanced_battle_system():
    """Battle detection with tactical intelligence - COMPLETE"""
    
    print(f"\n   Memory: {get_memory_gb():.2f}GB")
    
    if position_analysis_results is None or massive_motion_results is None:
        print("      Missing required data")
        return None
    
    position_df = position_analysis_results['position_analysis_df'].copy()
    ball_windows = championship_data.get('ball_windows_df', pd.DataFrame())
    enhanced_traj = massive_motion_results['enhanced_trajectories_df']
    matchup_matrix = position_analysis_results.get('matchup_matrix', {})
    
    print(f"      Battle data:")
    print(f"         Position records: {len(position_df):,}")
    print(f"         Ball windows: {len(ball_windows):,}")
    
    if len(ball_windows) == 0:
        print("      No ball windows available")
        return None
    
    # =====================================================================
    # STEP 1: COORDINATE INTEGRATION
    # =====================================================================
    
    print(f"\n      STEP 1: COORDINATE INTEGRATION")
    
    traj_coords = enhanced_traj[['game_id', 'play_id', 'player_id', 
                                  'start_x', 'start_y', 'end_x', 'end_y']].copy()
    
    battle_df = position_df.merge(traj_coords, on=['game_id', 'play_id', 'player_id'],
                                   how='left', suffixes=('', '_coord'))
    
    if 'end_x_coord' in battle_df.columns:
        battle_df['end_x'] = battle_df['end_x'].fillna(battle_df['end_x_coord'])
        battle_df['end_y'] = battle_df['end_y'].fillna(battle_df['end_y_coord'])
    
    valid_coords = battle_df[['end_x', 'end_y']].notna().all(axis=1).sum()
    print(f"         Battle-ready: {len(battle_df):,} | Valid coords: {valid_coords:,}")
    
    def get_field_zone(x, y):
        if x < 20:
            h_zone = 'OWN_ENDZONE'
        elif x < 40:
            h_zone = 'OWN_TERRITORY'
        elif x < 60:
            h_zone = 'MIDFIELD'
        elif x < 80:
            h_zone = 'OPP_TERRITORY'
        elif x < 100:
            h_zone = 'REDZONE'
        else:
            h_zone = 'OPP_ENDZONE'
        
        if y < 17.7:
            v_zone = 'LEFT'
        elif y < 35.6:
            v_zone = 'MIDDLE'
        else:
            v_zone = 'RIGHT'
        
        return f"{h_zone}_{v_zone}"
    
    # =====================================================================
    # STEP 2: BATTLE DETECTION
    # =====================================================================
    
    print(f"\n      STEP 2: BATTLE DETECTION")
    
    BATTLE_RADIUS_TIGHT = 5.0
    BATTLE_RADIUS_STANDARD = 8.0
    BATTLE_RADIUS_EXTENDED = 12.0
    MIN_BCS_THRESHOLD = 25
    ELITE_BCS_THRESHOLD = 60
    HIGH_INTENSITY_THRESHOLD = 110
    
    play_groups = battle_df.groupby(['game_id', 'play_id'])
    total_plays = len(play_groups)
    
    battles = []
    processed = 0
    
    print(f"         Processing {total_plays:,} plays...")
    
    for (game_id, play_id), play_data in play_groups:
        processed += 1
        if processed % 2000 == 0:
            print(f"            Progress: {processed:,}/{total_plays:,} | Battles: {len(battles):,}")
        
        try:
            ball_info = ball_windows[(ball_windows['game_id'] == game_id) & 
                                      (ball_windows['play_id'] == play_id)]
            if len(ball_info) == 0:
                continue
            
            ball_row = ball_info.iloc[0]
            ball_x, ball_y = ball_row['ball_land_x'], ball_row['ball_land_y']
            
            route_type = ball_row.get('route_type', 'UNKNOWN')
            coverage_type = ball_row.get('coverage_type', 'UNKNOWN')
            epa_situation = ball_row.get('epa_situation', 'NEUTRAL')
            field_zone = get_field_zone(ball_x, ball_y)
            
            receivers = play_data[(play_data['player_role'] == 'RECEIVER') & 
                                   (play_data['base_bcs_score'] >= MIN_BCS_THRESHOLD)]
            defenders = play_data[(play_data['player_role'] == 'DEFENDER') & 
                                   (play_data['base_bcs_score'] >= MIN_BCS_THRESHOLD)]
            
            if len(receivers) == 0 or len(defenders) == 0:
                continue
            
            for _, rec in receivers.iterrows():
                if pd.isna(rec.get('end_x')) or pd.isna(rec.get('end_y')):
                    continue
                
                rx, ry = rec['end_x'], rec['end_y']
                rec_ball_dist = np.sqrt((rx - ball_x)**2 + (ry - ball_y)**2)
                
                if rec_ball_dist > 20:
                    continue
                
                nearby_defenders = []
                
                for _, defender in defenders.iterrows():
                    if pd.isna(defender.get('end_x')) or pd.isna(defender.get('end_y')):
                        continue
                    
                    dx, dy = defender['end_x'], defender['end_y']
                    dist = np.sqrt((rx - dx)**2 + (ry - dy)**2)
                    
                    if dist <= BATTLE_RADIUS_EXTENDED:
                        def_ball_dist = np.sqrt((dx - ball_x)**2 + (dy - ball_y)**2)
                        nearby_defenders.append({
                            'player_id': defender['player_id'],
                            'position': defender.get('detailed_position', 'DEF'),
                            'position_group': defender.get('position_group', 'DEF'),
                            'bcs': defender['base_bcs_score'],
                            'proximity': defender['proximity_score'],
                            'effort': defender['effort_score'],
                            'distance_to_receiver': dist,
                            'distance_to_ball': def_ball_dist,
                            'end_x': dx,
                            'end_y': dy
                        })
                
                if len(nearby_defenders) == 0:
                    continue
                
                nearby_defenders.sort(key=lambda x: x['distance_to_receiver'])
                primary_def = nearby_defenders[0]
                
                primary_dist = primary_def['distance_to_receiver']
                if primary_dist <= BATTLE_RADIUS_TIGHT:
                    battle_radius_type = 'TIGHT'
                    battle_intensity_mult = 1.2
                elif primary_dist <= BATTLE_RADIUS_STANDARD:
                    battle_radius_type = 'STANDARD'
                    battle_intensity_mult = 1.0
                else:
                    battle_radius_type = 'EXTENDED'
                    battle_intensity_mult = 0.8
                
                tight_count = sum(1 for d in nearby_defenders if d['distance_to_receiver'] <= BATTLE_RADIUS_TIGHT)
                standard_count = sum(1 for d in nearby_defenders if d['distance_to_receiver'] <= BATTLE_RADIUS_STANDARD)
                extended_count = len(nearby_defenders)
                
                rec_bcs = rec['base_bcs_score']
                rec_proximity = rec['proximity_score']
                rec_effort = rec['effort_score']
                
                def_bcs = primary_def['bcs']
                def_proximity = primary_def['proximity']
                def_effort = primary_def['effort']
                
                # Multiple defender metrics
                multi_def_avg_bcs = np.mean([d['bcs'] for d in nearby_defenders[:3]])
                multi_def_total_bcs = sum([d['bcs'] for d in nearby_defenders])
                
                bcs_advantage = rec_bcs - def_bcs
                bcs_advantage_multi = rec_bcs - multi_def_avg_bcs
                proximity_advantage = rec_proximity - def_proximity
                effort_advantage = rec_effort - def_effort
                
                battle_intensity = (rec_bcs + def_bcs) * battle_intensity_mult
                multi_intensity = rec_bcs + multi_def_total_bcs
                
                rec_position = rec.get('detailed_position', rec.get('position_group', 'REC'))
                rec_group = rec.get('position_group', 'WR')
                def_group = primary_def['position_group']
                matchup_key = f"{rec_group}_vs_{def_group}"
                
                matchup_intel = matchup_matrix.get(matchup_key, {})
                expected_win_prob = matchup_intel.get('estimated_win_prob', 0.5)
                matchup_advantage = matchup_intel.get('offensive_advantage', 'UNKNOWN')
                
                outcome_score = (
                    bcs_advantage * 0.3 +
                    proximity_advantage * 0.25 +
                    (rec_ball_dist < primary_def['distance_to_ball']) * 10 +
                    effort_advantage * 0.15 -
                    (tight_count - 1) * 3
                )
                
                if outcome_score > 8:
                    outcome = 'DOMINANT_OFFENSIVE'
                elif outcome_score > 3:
                    outcome = 'OFFENSIVE_WIN'
                elif outcome_score > -3:
                    outcome = 'CONTESTED'
                elif outcome_score > -8:
                    outcome = 'DEFENSIVE_WIN'
                else:
                    outcome = 'DOMINANT_DEFENSIVE'
                
                is_elite = rec_bcs >= ELITE_BCS_THRESHOLD or def_bcs >= ELITE_BCS_THRESHOLD
                is_high_intensity = battle_intensity >= HIGH_INTENSITY_THRESHOLD
                is_multi_defender = tight_count >= 2
                
                if is_elite and is_high_intensity:
                    battle_class = 'CHAMPIONSHIP_BATTLE'
                elif is_elite:
                    battle_class = 'ELITE_BATTLE'
                elif is_high_intensity:
                    battle_class = 'HIGH_INTENSITY'
                elif is_multi_defender:
                    battle_class = 'MULTI_DEFENDER'
                else:
                    battle_class = 'STANDARD'
                
                # Separation quality
                if rec_ball_dist < 3:
                    separation_quality = 'EXCELLENT'
                elif rec_ball_dist < 6:
                    separation_quality = 'GOOD'
                elif rec_ball_dist < 10:
                    separation_quality = 'MODERATE'
                else:
                    separation_quality = 'POOR'
                
                battle = {
                    'game_id': int(game_id),
                    'play_id': int(play_id),
                    'receiver_id': int(rec['player_id']),
                    'primary_defender_id': int(primary_def['player_id']),
                    'receiver_position': rec_position,
                    'receiver_group': rec_group,
                    'defender_position': primary_def['position'],
                    'defender_group': def_group,
                    'matchup_key': matchup_key,
                    'receiver_bcs': float(rec_bcs),
                    'defender_bcs': float(def_bcs),
                    'multi_defender_avg_bcs': float(multi_def_avg_bcs),
                    'bcs_advantage': float(bcs_advantage),
                    'bcs_advantage_multi': float(bcs_advantage_multi),
                    'proximity_advantage': float(proximity_advantage),
                    'effort_advantage': float(effort_advantage),
                    'receiver_proximity': float(rec_proximity),
                    'defender_proximity': float(def_proximity),
                    'battle_distance': float(primary_dist),
                    'receiver_ball_distance': float(rec_ball_dist),
                    'defender_ball_distance': float(primary_def['distance_to_ball']),
                    'battle_intensity': float(battle_intensity),
                    'multi_intensity': float(multi_intensity),
                    'outcome_score': float(outcome_score),
                    'battle_radius_type': battle_radius_type,
                    'battle_type': f"1v{tight_count}" if tight_count > 0 else "1v1",
                    'battle_class': battle_class,
                    'outcome': outcome,
                    'separation_quality': separation_quality,
                    'tight_defenders': tight_count,
                    'standard_defenders': standard_count,
                    'extended_defenders': extended_count,
                    'route_type': route_type,
                    'coverage_type': coverage_type,
                    'epa_situation': epa_situation,
                    'field_zone': field_zone,
                    'expected_win_prob': float(expected_win_prob),
                    'matchup_advantage': matchup_advantage,
                    'is_elite_battle': is_elite,
                    'is_high_intensity': is_high_intensity,
                    'is_multi_defender': is_multi_defender,
                    'receiver_x': float(rx),
                    'receiver_y': float(ry),
                    'defender_x': float(primary_def['end_x']),
                    'defender_y': float(primary_def['end_y']),
                    'ball_x': float(ball_x),
                    'ball_y': float(ball_y)
                }
                
                battles.append(battle)
                
        except Exception:
            continue
    
    # =====================================================================
    # STEP 3: BATTLE ANALYTICS
    # =====================================================================
    
    print(f"\n      STEP 3: BATTLE ANALYTICS")
    
    if not battles:
        print("         No battles detected")
        return None
    
    battles_df = pd.DataFrame(battles)
    
    print(f"         Total battles: {len(battles_df):,}")
    
    class_dist = battles_df['battle_class'].value_counts()
    print(f"         Battle classes:")
    for cls, count in class_dist.items():
        print(f"            {cls}: {count:,} ({count/len(battles_df)*100:.1f}%)")
    
    outcome_dist = battles_df['outcome'].value_counts()
    print(f"         Outcomes:")
    for outcome, count in outcome_dist.items():
        print(f"            {outcome}: {count:,} ({count/len(battles_df)*100:.1f}%)")
    
    # =====================================================================
    # STEP 4: MATCHUP ANALYSIS
    # =====================================================================
    
    print(f"\n      STEP 4: MATCHUP ANALYSIS")
    
    matchup_analysis = battles_df.groupby('matchup_key').agg({
        'game_id': 'count',
        'bcs_advantage': 'mean',
        'outcome_score': 'mean',
        'battle_intensity': 'mean',
        'receiver_bcs': 'mean',
        'defender_bcs': 'mean',
        'is_elite_battle': 'sum',
        'outcome': lambda x: (x.isin(['DOMINANT_OFFENSIVE', 'OFFENSIVE_WIN'])).sum() / len(x) * 100
    }).reset_index()
    
    matchup_analysis.columns = ['matchup_key', 'battle_count', 'avg_bcs_advantage', 'avg_outcome_score',
                                'avg_intensity', 'avg_receiver_bcs', 'avg_defender_bcs',
                                'elite_battles', 'offensive_win_rate']
    matchup_analysis = matchup_analysis.sort_values('battle_count', ascending=False)
    
    print(f"         Top matchups:")
    for _, row in matchup_analysis.head(5).iterrows():
        print(f"            {row['matchup_key']}: {int(row['battle_count']):,} | "
              f"{row['avg_bcs_advantage']:+.1f} adv | {row['offensive_win_rate']:.1f}% off win")
    
    # =====================================================================
    # STEP 5: ZONE ANALYSIS
    # =====================================================================
    
    print(f"\n      STEP 5: ZONE ANALYSIS")
    
    zone_analysis = battles_df.groupby('field_zone').agg({
        'game_id': 'count',
        'battle_intensity': 'mean',
        'bcs_advantage': 'mean',
        'is_elite_battle': 'mean',
        'outcome': lambda x: (x.isin(['DOMINANT_OFFENSIVE', 'OFFENSIVE_WIN'])).sum() / len(x) * 100
    }).reset_index()
    
    zone_analysis.columns = ['field_zone', 'battle_count', 'avg_intensity', 'avg_advantage',
                             'elite_rate', 'offensive_win_rate']
    zone_analysis = zone_analysis.sort_values('battle_count', ascending=False)
    
    print(f"         Top zones:")
    for _, row in zone_analysis.head(5).iterrows():
        print(f"            {row['field_zone']}: {int(row['battle_count']):,} | {row['offensive_win_rate']:.1f}% off win")
    
    # Elite battles
    elite_battles = battles_df[battles_df['is_elite_battle']]
    championship_battles = battles_df[battles_df['battle_class'] == 'CHAMPIONSHIP_BATTLE']
    
    print(f"         Elite battles: {len(elite_battles):,}")
    print(f"         Championship battles: {len(championship_battles):,}")
    
    # =====================================================================
    # STEP 6: STATISTICS & STORAGE
    # =====================================================================
    
    print(f"\n      STEP 6: STATISTICS")
    
    offensive_wins = battles_df['outcome'].isin(['DOMINANT_OFFENSIVE', 'OFFENSIVE_WIN']).sum()
    defensive_wins = battles_df['outcome'].isin(['DOMINANT_DEFENSIVE', 'DEFENSIVE_WIN']).sum()
    contested = (battles_df['outcome'] == 'CONTESTED').sum()
    
    battle_stats = {
        'total_battles': len(battles_df),
        'elite_battles': len(elite_battles),
        'championship_battles': len(championship_battles),
        'high_intensity_battles': len(battles_df[battles_df['is_high_intensity']]),
        'multi_defender_battles': len(battles_df[battles_df['is_multi_defender']]),
        'elite_rate': len(elite_battles) / len(battles_df) * 100,
        'offensive_win_rate': offensive_wins / len(battles_df) * 100,
        'defensive_win_rate': defensive_wins / len(battles_df) * 100,
        'contested_rate': contested / len(battles_df) * 100,
        'avg_battle_intensity': battles_df['battle_intensity'].mean(),
        'avg_bcs_advantage': battles_df['bcs_advantage'].mean(),
        'avg_battle_distance': battles_df['battle_distance'].mean(),
        'avg_receiver_ball_distance': battles_df['receiver_ball_distance'].mean(),
        'unique_receivers': battles_df['receiver_id'].nunique(),
        'unique_defenders': battles_df['primary_defender_id'].nunique(),
        'unique_matchups': battles_df['matchup_key'].nunique(),
        'outcome_distribution': outcome_dist.to_dict(),
        'class_distribution': class_dist.to_dict(),
        'battle_type_distribution': battles_df['battle_type'].value_counts().to_dict(),
        'most_active_receiver': int(battles_df['receiver_id'].value_counts().index[0]),
        'most_active_defender': int(battles_df['primary_defender_id'].value_counts().index[0])
    }
    
    championship_data['battles_df'] = battles_df
    championship_data['elite_battles_df'] = elite_battles
    championship_data['championship_battles_df'] = championship_battles
    championship_data['battle_stats'] = battle_stats
    championship_data['matchup_analysis'] = matchup_analysis
    championship_data['zone_analysis'] = zone_analysis
    
    print(f"         Total: {battle_stats['total_battles']:,}")
    print(f"         Offensive win: {battle_stats['offensive_win_rate']:.1f}%")
    print(f"         Avg intensity: {battle_stats['avg_battle_intensity']:.1f}")
    
    aggressive_gc()
    
    return {
        'battles_df': battles_df,
        'elite_battles_df': elite_battles,
        'championship_battles_df': championship_battles,
        'matchup_analysis': matchup_analysis,
        'zone_analysis': zone_analysis,
        'battle_stats': battle_stats
    }

# Execute
battle_detection_results = calculate_advanced_battle_system()

aggressive_gc()

print(f"\nMemory after battles: {get_memory_usage():.1f} MB")

if battle_detection_results:
    print("CELL 10 COMPLETE - BATTLE DETECTION DONE!")
    s = battle_detection_results['battle_stats']
    print(f"\nRESULTS:")
    print(f"   {s['total_battles']:,} battles | {s['elite_battles']:,} elite | {s['championship_battles']:,} championship")
    print(f"   Offensive: {s['offensive_win_rate']:.1f}% | Defensive: {s['defensive_win_rate']:.1f}%")
else:
    print("CELL 10 FAILED")


# ========================================================================
# CELL 11: ELITE PLAYER IDENTIFICATION - KAGGLE SAFE (COMPLETE)
# Multi-factor ranking system - ALL FEATURES
# ========================================================================

print("\nELITE PLAYER IDENTIFICATION")
print("="*45)

@timing_decorator
def identify_elite_players_championship():
    """Elite player identification with multi-factor scoring - COMPLETE"""
    
    print(f"\n   Memory: {get_memory_gb():.2f}GB")
    
    if player_bcs_results is None or battle_detection_results is None:
        print("      Missing required data")
        return None
    
    player_agg = player_bcs_results['player_aggregations_df'].copy()
    battles_df = battle_detection_results['battles_df'].copy()
    
    print(f"      Elite identification:")
    print(f"         Players: {len(player_agg):,}")
    print(f"         Battles: {len(battles_df):,}")
    
    # =====================================================================
    # STEP 1: BATTLE PERFORMANCE INTEGRATION
    # =====================================================================
    
    print(f"\n      STEP 1: BATTLE PERFORMANCE")
    
    # Receiver battle stats
    receiver_battles = battles_df.groupby('receiver_id').agg({
        'game_id': 'count',
        'bcs_advantage': ['mean', 'std'],
        'battle_intensity': 'mean',
        'outcome': lambda x: (x.isin(['DOMINANT_OFFENSIVE', 'OFFENSIVE_WIN'])).sum(),
        'is_elite_battle': 'sum',
        'outcome_score': 'mean',
        'separation_quality': lambda x: (x == 'EXCELLENT').sum() + (x == 'GOOD').sum() * 0.5
    }).reset_index()
    
    receiver_battles.columns = ['player_id', 'battles_as_receiver', 'avg_bcs_advantage', 
                                'bcs_advantage_std', 'avg_battle_intensity', 'offensive_wins',
                                'elite_battles_receiver', 'avg_outcome_score', 'separation_score']
    
    receiver_battles['receiver_win_rate'] = (
        receiver_battles['offensive_wins'] / receiver_battles['battles_as_receiver'] * 100
    )
    
    # Defender battle stats
    defender_battles = battles_df.groupby('primary_defender_id').agg({
        'game_id': 'count',
        'bcs_advantage': ['mean', 'std'],
        'battle_intensity': 'mean',
        'outcome': lambda x: (x.isin(['DOMINANT_DEFENSIVE', 'DEFENSIVE_WIN'])).sum(),
        'is_elite_battle': 'sum',
        'outcome_score': 'mean'
    }).reset_index()
    
    defender_battles.columns = ['player_id', 'battles_as_defender', 'avg_bcs_disadvantage',
                                'bcs_disadvantage_std', 'avg_battle_intensity_def', 'defensive_wins',
                                'elite_battles_defender', 'avg_outcome_score_def']
    
    defender_battles['defender_win_rate'] = (
        defender_battles['defensive_wins'] / defender_battles['battles_as_defender'] * 100
    )
    
    # Merge battle stats
    player_elite = player_agg.merge(receiver_battles, on='player_id', how='left')
    player_elite = player_elite.merge(defender_battles, on='player_id', how='left')
    
    # Fill NaN values
    battle_cols = ['battles_as_receiver', 'battles_as_defender', 'offensive_wins', 
                   'defensive_wins', 'elite_battles_receiver', 'elite_battles_defender',
                   'separation_score']
    for col in battle_cols:
        if col in player_elite.columns:
            player_elite[col] = player_elite[col].fillna(0)
    
    # Combined battle metrics
    player_elite['total_battles'] = (
        player_elite['battles_as_receiver'].fillna(0) + 
        player_elite['battles_as_defender'].fillna(0)
    )
    player_elite['total_elite_battles'] = (
        player_elite['elite_battles_receiver'].fillna(0) + 
        player_elite['elite_battles_defender'].fillna(0)
    )
    
    print(f"         Battle stats integrated")
    print(f"         Players with battles: {(player_elite['total_battles'] > 0).sum():,}")
    
    # =====================================================================
    # STEP 2: ENSURE ALL REQUIRED COLUMNS EXIST
    # =====================================================================
    
    print(f"\n      STEP 2: COLUMN VALIDATION")
    
    # Create game_coverage if not exists
    if 'game_coverage' not in player_elite.columns:
        if 'games_played' in player_elite.columns:
            player_elite['game_coverage'] = np.minimum(player_elite['games_played'] / 10 * 100, 100)
        else:
            player_elite['game_coverage'] = 50
    
    # Create sample_reliability if not exists
    if 'sample_reliability' not in player_elite.columns:
        player_elite['sample_reliability'] = np.minimum(player_elite['total_plays'] / 15 * 100, 100)
    
    # Create upside_potential if not exists
    if 'upside_potential' not in player_elite.columns:
        player_elite['upside_potential'] = player_elite['max_bcs'] - player_elite['avg_bcs']
    
    # Create floor_risk if not exists
    if 'floor_risk' not in player_elite.columns:
        player_elite['floor_risk'] = player_elite['avg_bcs'] - player_elite['min_bcs']
    
    # Create overall_consistency if not exists
    if 'overall_consistency' not in player_elite.columns:
        player_elite['overall_consistency'] = 100 - np.minimum(player_elite['bcs_std'].fillna(0) * 3, 60)
    
    print(f"         All required columns validated")
    
    # =====================================================================
    # STEP 3: MULTI-FACTOR ELITE SCORING
    # =====================================================================
    
    print(f"\n      STEP 3: ELITE SCORING")
    
    # Factor 1: BCS Performance (35%)
    player_elite['bcs_factor'] = (
        player_elite['avg_bcs'] * 0.5 +
        player_elite['position_weighted_bcs'] * 0.3 +
        player_elite['max_bcs'] * 0.2
    )
    player_elite['bcs_factor_normalized'] = (
        (player_elite['bcs_factor'] - player_elite['bcs_factor'].min()) /
        (player_elite['bcs_factor'].max() - player_elite['bcs_factor'].min() + 0.001) * 100
    )
    
    # Factor 2: Consistency (20%)
    player_elite['consistency_factor'] = player_elite['overall_consistency'].fillna(50)
    
    # Factor 3: Battle Performance (20%)
    player_elite['battle_win_rate'] = np.where(
        player_elite['dominant_role'] == 'RECEIVER',
        player_elite['receiver_win_rate'].fillna(50),
        player_elite['defender_win_rate'].fillna(50)
    )
    
    player_elite['battle_factor'] = (
        player_elite['battle_win_rate'] * 0.4 +
        np.minimum(player_elite['total_battles'] / 10, 1) * 100 * 0.3 +
        np.minimum(player_elite['total_elite_battles'] / 5, 1) * 100 * 0.3
    )
    
    # Factor 4: Sample Size & Reliability (15%)
    player_elite['reliability_factor'] = (
        player_elite['sample_reliability'].fillna(50) * 0.6 +
        player_elite['game_coverage'].fillna(50) * 0.4
    )
    
    # Factor 5: Upside Potential (10%)
    player_elite['upside_factor'] = (
        player_elite['upside_potential'].fillna(0) * 0.6 +
        (100 - player_elite['floor_risk'].fillna(50)) * 0.4
    )
    
    # Combined Elite Score
    player_elite['elite_score'] = (
        player_elite['bcs_factor_normalized'] * 0.35 +
        player_elite['consistency_factor'] * 0.20 +
        player_elite['battle_factor'].fillna(50) * 0.20 +
        player_elite['reliability_factor'] * 0.15 +
        player_elite['upside_factor'] * 0.10
    )
    
    print(f"         Elite score: {player_elite['elite_score'].min():.1f} - {player_elite['elite_score'].max():.1f}")
    print(f"         Avg: {player_elite['elite_score'].mean():.1f}")
    
    # =====================================================================
    # STEP 4: ELITE TIER CLASSIFICATION
    # =====================================================================
    
    print(f"\n      STEP 4: TIER CLASSIFICATION")
    
    def classify_elite_tier(row):
        score = row['elite_score']
        plays = row['total_plays']
        consistency = row['consistency_factor']
        battles = row['total_battles']
        
        if score >= 75 and plays >= 10 and consistency >= 65:
            return 'HALL_OF_FAME'
        elif score >= 70 and plays >= 8 and consistency >= 60:
            return 'ALL_PRO'
        elif score >= 65 and plays >= 6:
            return 'PRO_BOWL'
        elif score >= 60 and plays >= 5:
            return 'ELITE'
        elif score >= 55 and plays >= 4:
            return 'ABOVE_AVERAGE'
        elif score >= 50 and plays >= 3:
            return 'STARTER'
        elif score >= 45:
            return 'ROTATIONAL'
        elif score >= 40:
            return 'DEVELOPMENTAL'
        else:
            return 'PRACTICE_SQUAD'
    
    player_elite['elite_tier'] = player_elite.apply(classify_elite_tier, axis=1)
    
    tier_dist = player_elite['elite_tier'].value_counts()
    print(f"         Tier distribution:")
    for tier, count in tier_dist.items():
        avg_score = player_elite[player_elite['elite_tier'] == tier]['elite_score'].mean()
        print(f"            {tier}: {count:,} ({avg_score:.1f} avg)")
    
    # =====================================================================
    # STEP 5: POSITION-SPECIFIC RANKINGS
    # =====================================================================
    
    print(f"\n      STEP 5: POSITION RANKINGS")
    
    position_rankings = {}
    
    for pos in player_elite['inferred_position'].unique():
        pos_players = player_elite[player_elite['inferred_position'] == pos].copy()
        
        if len(pos_players) < 3:
            continue
        
        # Rank within position
        pos_players['position_rank'] = pos_players['elite_score'].rank(ascending=False, method='min')
        pos_players['position_percentile'] = pos_players['elite_score'].rank(pct=True) * 100
        
        # Update main dataframe
        player_elite.loc[player_elite['inferred_position'] == pos, 'position_rank'] = (
            player_elite.loc[player_elite['inferred_position'] == pos, 'elite_score'].rank(ascending=False, method='min')
        )
        player_elite.loc[player_elite['inferred_position'] == pos, 'position_percentile'] = (
            player_elite.loc[player_elite['inferred_position'] == pos, 'elite_score'].rank(pct=True) * 100
        )
        
        # Get top performers
        top_at_position = pos_players.nlargest(10, 'elite_score')
        
        position_rankings[pos] = {
            'total_players': len(pos_players),
            'avg_elite_score': pos_players['elite_score'].mean(),
            'top_player_id': int(top_at_position.iloc[0]['player_id']) if len(top_at_position) > 0 else None,
            'top_player_score': float(top_at_position.iloc[0]['elite_score']) if len(top_at_position) > 0 else 0,
            'elite_count': len(pos_players[pos_players['elite_tier'].isin(['HALL_OF_FAME', 'ALL_PRO', 'PRO_BOWL', 'ELITE'])]),
            'top_10': top_at_position[['player_id', 'elite_score', 'avg_bcs', 'total_plays']].to_dict('records')
        }
    
    # Fill NaN position ranks
    player_elite['position_rank'] = player_elite['position_rank'].fillna(999)
    player_elite['position_percentile'] = player_elite['position_percentile'].fillna(50)
    
    print(f"         Rankings for {len(position_rankings)} positions")
    
    # Show top at each major position
    print(f"         Position leaders:")
    for pos in ['WR_SPEED', 'WR_ROUTE', 'TE', 'CB', 'S', 'LB']:
        if pos in position_rankings:
            data = position_rankings[pos]
            print(f"            {pos}: Player {data['top_player_id']} ({data['top_player_score']:.1f})")
    
    # =====================================================================
    # STEP 6: ROLE-BASED ANALYSIS
    # =====================================================================
    
    print(f"\n      STEP 6: ROLE-BASED ANALYSIS")
    
    role_analysis = {}
    
    for role in ['RECEIVER', 'DEFENDER', 'OTHER']:
        role_players = player_elite[player_elite['dominant_role'] == role]
        
        if len(role_players) < 5:
            continue
        
        role_analysis[role] = {
            'total_players': len(role_players),
            'avg_elite_score': role_players['elite_score'].mean(),
            'avg_bcs': role_players['avg_bcs'].mean(),
            'avg_consistency': role_players['consistency_factor'].mean(),
            'elite_count': len(role_players[role_players['elite_tier'].isin(['HALL_OF_FAME', 'ALL_PRO', 'PRO_BOWL', 'ELITE'])]),
            'hall_of_fame': len(role_players[role_players['elite_tier'] == 'HALL_OF_FAME']),
            'all_pro': len(role_players[role_players['elite_tier'] == 'ALL_PRO']),
            'pro_bowl': len(role_players[role_players['elite_tier'] == 'PRO_BOWL']),
            'top_performers': role_players.nlargest(5, 'elite_score')[
                ['player_id', 'inferred_position', 'elite_score', 'elite_tier']
            ].to_dict('records')
        }
    
    print(f"         Role analysis:")
    for role, data in role_analysis.items():
        print(f"            {role}: {data['total_players']:,} players | "
              f"{data['elite_count']} elite | {data['avg_elite_score']:.1f} avg score")
    
    # =====================================================================
    # STEP 7: PERFORMANCE PROFILES
    # =====================================================================
    
    print(f"\n      STEP 7: PERFORMANCE PROFILES")
    
    def create_performance_profile(row):
        strengths = []
        weaknesses = []
        
        # Check each component
        if row['avg_proximity'] >= 70:
            strengths.append('PROXIMITY')
        elif row['avg_proximity'] < 40:
            weaknesses.append('PROXIMITY')
        
        if row['avg_efficiency'] >= 70:
            strengths.append('EFFICIENCY')
        elif row['avg_efficiency'] < 40:
            weaknesses.append('EFFICIENCY')
        
        if row['avg_effort'] >= 70:
            strengths.append('EFFORT')
        elif row['avg_effort'] < 40:
            weaknesses.append('EFFORT')
        
        if row['avg_velocity'] >= 70:
            strengths.append('VELOCITY')
        elif row['avg_velocity'] < 40:
            weaknesses.append('VELOCITY')
        
        # Battle performance
        if row.get('battle_win_rate', 50) >= 60:
            strengths.append('BATTLES')
        elif row.get('battle_win_rate', 50) < 40:
            weaknesses.append('BATTLES')
        
        # Consistency
        if row['consistency_factor'] >= 70:
            strengths.append('CONSISTENCY')
        elif row['consistency_factor'] < 50:
            weaknesses.append('CONSISTENCY')
        
        # Create profile type
        if len(strengths) >= 4:
            profile = 'COMPLETE'
        elif 'PROXIMITY' in strengths and 'EFFICIENCY' in strengths:
            profile = 'TECHNICIAN'
        elif 'EFFORT' in strengths and 'VELOCITY' in strengths:
            profile = 'ATHLETIC'
        elif 'BATTLES' in strengths:
            profile = 'COMPETITOR'
        elif 'CONSISTENCY' in strengths:
            profile = 'RELIABLE'
        elif len(weaknesses) >= 3:
            profile = 'PROJECT'
        else:
            profile = 'BALANCED'
        
        return profile, ','.join(strengths), ','.join(weaknesses)
    
    profiles = player_elite.apply(create_performance_profile, axis=1, result_type='expand')
    player_elite['performance_profile'] = profiles[0]
    player_elite['strengths'] = profiles[1]
    player_elite['weaknesses'] = profiles[2]
    
    # Create strongest/weakest component if not exists
    if 'strongest_component' not in player_elite.columns:
        def get_strongest_weakest(row):
            comps = {
                'proximity': row.get('avg_proximity', 0),
                'efficiency': row.get('avg_efficiency', 0),
                'effort': row.get('avg_effort', 0),
                'velocity': row.get('avg_velocity', 0)
            }
            sorted_comps = sorted(comps.items(), key=lambda x: x[1], reverse=True)
            return sorted_comps[0][0], sorted_comps[-1][0]
        
        comp_analysis = player_elite.apply(get_strongest_weakest, axis=1, result_type='expand')
        player_elite['strongest_component'] = comp_analysis[0]
        player_elite['weakest_component'] = comp_analysis[1]
    
    profile_dist = player_elite['performance_profile'].value_counts()
    print(f"         Profile distribution:")
    for profile, count in profile_dist.items():
        avg_score = player_elite[player_elite['performance_profile'] == profile]['elite_score'].mean()
        print(f"            {profile}: {count:,} ({avg_score:.1f} avg)")
    
    # =====================================================================
    # STEP 8: COMPARATIVE RANKINGS
    # =====================================================================
    
    print(f"\n      STEP 8: COMPARATIVE RANKINGS")
    
    # Overall rankings
    player_elite['overall_rank'] = player_elite['elite_score'].rank(ascending=False, method='min')
    
    # Percentile rankings
    player_elite['overall_percentile'] = player_elite['elite_score'].rank(pct=True) * 100
    player_elite['bcs_percentile'] = player_elite['avg_bcs'].rank(pct=True) * 100
    player_elite['consistency_percentile'] = player_elite['consistency_factor'].rank(pct=True) * 100
    
    # Category rankings
    player_elite['bcs_rank'] = player_elite['avg_bcs'].rank(ascending=False, method='min')
    player_elite['consistency_rank'] = player_elite['consistency_factor'].rank(ascending=False, method='min')
    player_elite['battle_rank'] = player_elite['battle_factor'].rank(ascending=False, method='min')
    
    # Qualified players only
    qualified = player_elite[player_elite['total_plays'] >= 3].copy()
    
    # Top 25 overall
    top_25_overall = qualified.nlargest(25, 'elite_score')
    
    print(f"         TOP 15 ELITE PLAYERS:")
    for rank, (_, p) in enumerate(top_25_overall.head(15).iterrows(), 1):
        print(f"            {rank:2d}. Player {int(p['player_id'])} ({p['inferred_position']}) - {p['elite_tier']}")
        print(f"                Score: {p['elite_score']:.1f} | BCS: {p['avg_bcs']:.1f} | "
              f"Plays: {int(p['total_plays'])} | Profile: {p['performance_profile']}")
    
    # =====================================================================
    # STEP 9: ELITE PLAYER CARDS
    # =====================================================================
    
    print(f"\n      STEP 9: ELITE PLAYER CARDS")
    
    elite_cards = []
    
    top_elite = qualified[qualified['elite_tier'].isin(['HALL_OF_FAME', 'ALL_PRO', 'PRO_BOWL', 'ELITE'])]
    
    for _, player in top_elite.iterrows():
        card = {
            'player_id': int(player['player_id']),
            'elite_score': float(player['elite_score']),
            'elite_tier': player['elite_tier'],
            'overall_rank': int(player['overall_rank']),
            'position': player['inferred_position'],
            'position_rank': int(player.get('position_rank', 0)),
            'role': player['dominant_role'],
            'performance_profile': player['performance_profile'],
            
            # BCS metrics
            'avg_bcs': float(player['avg_bcs']),
            'max_bcs': float(player['max_bcs']),
            'min_bcs': float(player['min_bcs']),
            'bcs_percentile': float(player['bcs_percentile']),
            
            # Components
            'proximity': float(player['avg_proximity']),
            'efficiency': float(player['avg_efficiency']),
            'effort': float(player['avg_effort']),
            'velocity': float(player['avg_velocity']),
            
            # Battle performance
            'total_battles': int(player['total_battles']),
            'battle_win_rate': float(player.get('battle_win_rate', 50)),
            'elite_battles': int(player['total_elite_battles']),
            
            # Consistency & reliability
            'consistency_score': float(player['consistency_factor']),
            'sample_size': int(player['total_plays']),
            'games_played': int(player.get('games_played', 0)),
            
            # Strengths & weaknesses (as lists)
            'strengths': player['strengths'].split(',') if player['strengths'] else [],
            'weaknesses': player['weaknesses'].split(',') if player['weaknesses'] else [],
            'strongest_component': player['strongest_component'],
            'weakest_component': player['weakest_component'],
            
            # Potential
            'upside_potential': float(player.get('upside_potential', 0)),
            'floor_risk': float(player.get('floor_risk', 0))
        }
        elite_cards.append(card)
    
    print(f"         Elite cards: {len(elite_cards)}")
    
    # =====================================================================
    # STEP 10: STATISTICS & STORAGE
    # =====================================================================
    
    print(f"\n      STEP 10: STORAGE")
    
    elite_stats = {
        'total_players_analyzed': len(player_elite),
        'qualified_players': len(qualified),
        
        # Tier counts
        'hall_of_fame': len(player_elite[player_elite['elite_tier'] == 'HALL_OF_FAME']),
        'all_pro': len(player_elite[player_elite['elite_tier'] == 'ALL_PRO']),
        'pro_bowl': len(player_elite[player_elite['elite_tier'] == 'PRO_BOWL']),
        'elite': len(player_elite[player_elite['elite_tier'] == 'ELITE']),
        'total_elite': len(top_elite),
        
        # Score statistics
        'avg_elite_score': player_elite['elite_score'].mean(),
        'max_elite_score': player_elite['elite_score'].max(),
        'elite_score_std': player_elite['elite_score'].std(),
        
        # Distributions
        'tier_distribution': tier_dist.to_dict(),
        'profile_distribution': profile_dist.to_dict(),
        'role_analysis': role_analysis,
        
        # Top performer
        'top_player_id': int(top_25_overall.iloc[0]['player_id']),
        'top_player_score': float(top_25_overall.iloc[0]['elite_score']),
        'top_player_tier': top_25_overall.iloc[0]['elite_tier'],
        
        # Rankings created
        'positions_ranked': len(position_rankings),
        'elite_cards_created': len(elite_cards)
    }
    
    # Store in championship_data
    championship_data['elite_players_df'] = player_elite
    championship_data['elite_qualified'] = qualified
    championship_data['top_25_elite'] = top_25_overall
    championship_data['position_rankings'] = position_rankings
    championship_data['role_analysis'] = role_analysis
    championship_data['elite_cards'] = elite_cards
    championship_data['elite_stats'] = elite_stats
    
    print(f"         Total elite: {elite_stats['total_elite']:,}")
    print(f"         HOF: {elite_stats['hall_of_fame']} | AP: {elite_stats['all_pro']} | PB: {elite_stats['pro_bowl']}")
    
    aggressive_gc()
    
    return {
        'elite_players_df': player_elite,
        'qualified_df': qualified,
        'top_25_df': top_25_overall,
        'position_rankings': position_rankings,
        'role_analysis': role_analysis,
        'elite_cards': elite_cards,
        'elite_stats': elite_stats
    }

# Execute
elite_identification_results = identify_elite_players_championship()

aggressive_gc()

print(f"\nMemory after elite ID: {get_memory_usage():.1f} MB")

if elite_identification_results:
    print("CELL 11 COMPLETE - ELITE IDENTIFICATION DONE!")
    s = elite_identification_results['elite_stats']
    print(f"\nRESULTS:")
    print(f"   {s['total_elite']:,} elite players")
    print(f"   {s['hall_of_fame']} HOF | {s['all_pro']} All-Pro | {s['pro_bowl']} Pro Bowl")
    print(f"   Top: Player {s['top_player_id']} ({s['top_player_score']:.1f}) - {s['top_player_tier']}")
else:
    print("CELL 11 FAILED")


# ========================================================================
# CELL 12: STATISTICAL VALIDATION - KAGGLE SAFE (COMPLETE)
# Comprehensive statistical validation with ALL metrics
# ========================================================================

print("\nSTATISTICAL VALIDATION")
print("="*45)

@timing_decorator
def perform_statistical_validation():
    """Comprehensive statistical validation of BCS framework - COMPLETE"""
    
    print(f"\n   Memory: {get_memory_gb():.2f}GB")
    
    if championship_bcs_results is None or elite_identification_results is None:
        print("      Missing required data")
        return None
    
    bcs_df = championship_bcs_results['championship_bcs_df'].copy()
    player_elite = elite_identification_results['elite_players_df'].copy()
    battles_df = battle_detection_results['battles_df'].copy()
    
    print(f"      Validation data:")
    print(f"         Play-level: {len(bcs_df):,}")
    print(f"         Player-level: {len(player_elite):,}")
    print(f"         Battles: {len(battles_df):,}")
    
    validation_results = {}
    
    # =====================================================================
    # STEP 1: DISTRIBUTION ANALYSIS
    # =====================================================================
    
    print(f"\n      STEP 1: DISTRIBUTION ANALYSIS")
    
    bcs_distribution = {
        'mean': bcs_df['base_bcs_score'].mean(),
        'std': bcs_df['base_bcs_score'].std(),
        'median': bcs_df['base_bcs_score'].median(),
        'min': bcs_df['base_bcs_score'].min(),
        'max': bcs_df['base_bcs_score'].max(),
        'skewness': bcs_df['base_bcs_score'].skew(),
        'kurtosis': bcs_df['base_bcs_score'].kurtosis(),
        'iqr': bcs_df['base_bcs_score'].quantile(0.75) - bcs_df['base_bcs_score'].quantile(0.25),
        'p10': bcs_df['base_bcs_score'].quantile(0.10),
        'p25': bcs_df['base_bcs_score'].quantile(0.25),
        'p50': bcs_df['base_bcs_score'].quantile(0.50),
        'p75': bcs_df['base_bcs_score'].quantile(0.75),
        'p90': bcs_df['base_bcs_score'].quantile(0.90),
        'p95': bcs_df['base_bcs_score'].quantile(0.95),
        'p99': bcs_df['base_bcs_score'].quantile(0.99)
    }
    
    print(f"         BCS Distribution:")
    print(f"            Mean: {bcs_distribution['mean']:.2f} +/- {bcs_distribution['std']:.2f}")
    print(f"            Median: {bcs_distribution['median']:.2f}")
    print(f"            Range: [{bcs_distribution['min']:.1f}, {bcs_distribution['max']:.1f}]")
    print(f"            IQR: {bcs_distribution['iqr']:.2f}")
    print(f"            Skewness: {bcs_distribution['skewness']:.3f}")
    print(f"            Kurtosis: {bcs_distribution['kurtosis']:.3f}")
    
    if abs(bcs_distribution['skewness']) < 0.5 and abs(bcs_distribution['kurtosis']) < 1:
        normality_assessment = 'APPROXIMATELY_NORMAL'
    elif abs(bcs_distribution['skewness']) < 1:
        normality_assessment = 'SLIGHTLY_SKEWED'
    else:
        normality_assessment = 'NON_NORMAL'
    
    bcs_distribution['normality_assessment'] = normality_assessment
    print(f"            Normality: {normality_assessment}")
    
    validation_results['bcs_distribution'] = bcs_distribution
    
    # =====================================================================
    # STEP 2: COMPONENT CORRELATION ANALYSIS
    # =====================================================================
    
    print(f"\n      STEP 2: COMPONENT CORRELATION")
    
    components = ['proximity_score', 'efficiency_score', 'effort_score', 'velocity_score']
    
    # Full correlation matrix between components
    correlation_matrix = {}
    
    for i, comp1 in enumerate(components):
        correlation_matrix[comp1] = {}
        for comp2 in components:
            if comp1 in bcs_df.columns and comp2 in bcs_df.columns:
                corr = bcs_df[comp1].corr(bcs_df[comp2])
                correlation_matrix[comp1][comp2] = round(corr, 4)
    
    # Component importance (correlation with BCS)
    component_importance = {}
    
    print(f"         Component correlations with BCS:")
    for comp in components:
        if comp in bcs_df.columns:
            corr = bcs_df[comp].corr(bcs_df['base_bcs_score'])
            component_importance[comp] = {
                'correlation': round(corr, 4),
                'r_squared': round(corr**2, 4),
                'variance_explained_pct': round(corr**2 * 100, 2)
            }
            print(f"            {comp}: r = {corr:.3f} (R2 = {corr**2:.3f}, {corr**2*100:.1f}%)")
    
    # Sort by importance
    sorted_importance = sorted(component_importance.items(), 
                               key=lambda x: x[1]['r_squared'], reverse=True)
    
    print(f"         Component ranking by importance:")
    for rank, (comp, data) in enumerate(sorted_importance, 1):
        print(f"            {rank}. {comp}: {data['variance_explained_pct']:.1f}%")
    
    validation_results['correlation_matrix'] = correlation_matrix
    validation_results['component_importance'] = component_importance
    
    # =====================================================================
    # STEP 3: ROLE COMPARISON ANALYSIS
    # =====================================================================
    
    print(f"\n      STEP 3: ROLE COMPARISON")
    
    role_comparison = {}
    
    for role in ['RECEIVER', 'DEFENDER', 'OTHER']:
        role_data = bcs_df[bcs_df['player_role'] == role]
        
        if len(role_data) < 100:
            continue
        
        role_comparison[role] = {
            'count': len(role_data),
            'mean_bcs': role_data['base_bcs_score'].mean(),
            'std_bcs': role_data['base_bcs_score'].std(),
            'median_bcs': role_data['base_bcs_score'].median(),
            'elite_count': len(role_data[role_data['performance_tier'] == 'ELITE']),
            'elite_rate': len(role_data[role_data['performance_tier'] == 'ELITE']) / len(role_data) * 100,
            'avg_proximity': role_data['proximity_score'].mean(),
            'avg_efficiency': role_data['efficiency_score'].mean(),
            'avg_effort': role_data['effort_score'].mean(),
            'avg_velocity': role_data['velocity_score'].mean()
        }
    
    # Statistical comparison between roles
    if 'RECEIVER' in role_comparison and 'DEFENDER' in role_comparison:
        rec_bcs = bcs_df[bcs_df['player_role'] == 'RECEIVER']['base_bcs_score']
        def_bcs = bcs_df[bcs_df['player_role'] == 'DEFENDER']['base_bcs_score']
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((rec_bcs.var() + def_bcs.var()) / 2)
        cohens_d = (rec_bcs.mean() - def_bcs.mean()) / pooled_std if pooled_std > 0 else 0
        
        role_comparison['receiver_vs_defender'] = {
            'mean_difference': rec_bcs.mean() - def_bcs.mean(),
            'cohens_d': round(cohens_d, 3),
            'effect_size': 'LARGE' if abs(cohens_d) > 0.8 else 'MEDIUM' if abs(cohens_d) > 0.5 else 'SMALL'
        }
        
        print(f"         Receiver vs Defender:")
        print(f"            Mean diff: {role_comparison['receiver_vs_defender']['mean_difference']:.2f}")
        print(f"            Effect size (d): {cohens_d:.3f} ({role_comparison['receiver_vs_defender']['effect_size']})")
    
    print(f"         Role stats:")
    for role, data in role_comparison.items():
        if role != 'receiver_vs_defender':
            print(f"            {role}: {data['mean_bcs']:.1f} +/- {data['std_bcs']:.1f} (n={data['count']:,})")
    
    validation_results['role_comparison'] = role_comparison
    
    # =====================================================================
    # STEP 4: TIER VALIDATION
    # =====================================================================
    
    print(f"\n      STEP 4: TIER VALIDATION")
    
    tier_validation = {}
    tiers = ['ELITE', 'EXCELLENT', 'GOOD', 'AVERAGE', 'BELOW_AVERAGE']
    
    for tier in tiers:
        tier_data = bcs_df[bcs_df['performance_tier'] == tier]
        
        if len(tier_data) < 50:
            continue
        
        tier_validation[tier] = {
            'count': len(tier_data),
            'percentage': len(tier_data) / len(bcs_df) * 100,
            'mean_bcs': tier_data['base_bcs_score'].mean(),
            'std_bcs': tier_data['base_bcs_score'].std(),
            'min_bcs': tier_data['base_bcs_score'].min(),
            'max_bcs': tier_data['base_bcs_score'].max(),
            'bcs_range': (tier_data['base_bcs_score'].min(), tier_data['base_bcs_score'].max())
        }
    
    # Verify tier separation
    tier_means = [(tier, data['mean_bcs']) for tier, data in tier_validation.items()]
    tier_means.sort(key=lambda x: x[1], reverse=True)
    
    tier_separation_valid = all(
        tier_means[i][1] > tier_means[i+1][1] 
        for i in range(len(tier_means)-1)
    )
    
    print(f"         Tier validation:")
    for tier, data in tier_validation.items():
        print(f"            {tier}: {data['mean_bcs']:.1f} +/- {data['std_bcs']:.1f} "
              f"[{data['min_bcs']:.1f}-{data['max_bcs']:.1f}] (n={data['count']:,})")
    print(f"         Tier separation valid: {tier_separation_valid}")
    
    tier_validation['separation_valid'] = tier_separation_valid
    validation_results['tier_validation'] = tier_validation
    
    # =====================================================================
    # STEP 5: BATTLE OUTCOME VALIDATION
    # =====================================================================
    
    print(f"\n      STEP 5: BATTLE OUTCOME VALIDATION")
    
    battle_validation = {}
    
    offensive_wins = battles_df[battles_df['outcome'].isin(['DOMINANT_OFFENSIVE', 'OFFENSIVE_WIN'])]
    defensive_wins = battles_df[battles_df['outcome'].isin(['DOMINANT_DEFENSIVE', 'DEFENSIVE_WIN'])]
    contested = battles_df[battles_df['outcome'] == 'CONTESTED']
    
    battle_validation['offensive_wins'] = {
        'count': len(offensive_wins),
        'avg_bcs_advantage': offensive_wins['bcs_advantage'].mean(),
        'avg_outcome_score': offensive_wins['outcome_score'].mean()
    }
    
    battle_validation['defensive_wins'] = {
        'count': len(defensive_wins),
        'avg_bcs_advantage': defensive_wins['bcs_advantage'].mean(),
        'avg_outcome_score': defensive_wins['outcome_score'].mean()
    }
    
    battle_validation['contested'] = {
        'count': len(contested),
        'avg_bcs_advantage': contested['bcs_advantage'].mean() if len(contested) > 0 else 0
    }
    
    # Predictive accuracy
    correct_predictions = len(battles_df[
        ((battles_df['bcs_advantage'] > 5) & battles_df['outcome'].isin(['DOMINANT_OFFENSIVE', 'OFFENSIVE_WIN'])) |
        ((battles_df['bcs_advantage'] < -5) & battles_df['outcome'].isin(['DOMINANT_DEFENSIVE', 'DEFENSIVE_WIN'])) |
        ((battles_df['bcs_advantage'].between(-5, 5)) & (battles_df['outcome'] == 'CONTESTED'))
    ])
    
    prediction_accuracy = correct_predictions / len(battles_df) * 100
    battle_validation['prediction_accuracy'] = round(prediction_accuracy, 2)
    
    # Correlation between BCS advantage and outcome
    outcome_numeric = battles_df['outcome'].map({
        'DOMINANT_OFFENSIVE': 2, 'OFFENSIVE_WIN': 1, 'CONTESTED': 0,
        'DEFENSIVE_WIN': -1, 'DOMINANT_DEFENSIVE': -2
    })
    
    bcs_outcome_corr = battles_df['bcs_advantage'].corr(outcome_numeric)
    battle_validation['bcs_outcome_correlation'] = round(bcs_outcome_corr, 4)
    
    print(f"         Battle outcome validation:")
    print(f"            Off wins avg advantage: {battle_validation['offensive_wins']['avg_bcs_advantage']:+.2f}")
    print(f"            Def wins avg advantage: {battle_validation['defensive_wins']['avg_bcs_advantage']:+.2f}")
    print(f"            BCS-Outcome correlation: r = {bcs_outcome_corr:.3f}")
    print(f"            Prediction accuracy: {prediction_accuracy:.1f}%")
    
    validation_results['battle_validation'] = battle_validation
    
    # =====================================================================
    # STEP 6: RELIABILITY ANALYSIS
    # =====================================================================
    
    print(f"\n      STEP 6: RELIABILITY ANALYSIS")
    
    reliability_analysis = {}
    
    # Split-half reliability (odd vs even plays)
    player_plays = bcs_df.groupby('player_id').apply(
        lambda x: x.reset_index(drop=True)
    ).reset_index(level=0, drop=True)
    
    player_plays['play_order'] = player_plays.groupby('player_id').cumcount()
    
    odd_plays = player_plays[player_plays['play_order'] % 2 == 1]
    even_plays = player_plays[player_plays['play_order'] % 2 == 0]
    
    odd_avg = odd_plays.groupby('player_id')['base_bcs_score'].mean()
    even_avg = even_plays.groupby('player_id')['base_bcs_score'].mean()
    
    # Align on common players
    common_players = odd_avg.index.intersection(even_avg.index)
    if len(common_players) > 10:
        split_half_corr = odd_avg[common_players].corr(even_avg[common_players])
        # Spearman-Brown correction
        reliability_coefficient = (2 * split_half_corr) / (1 + split_half_corr)
    else:
        split_half_corr = 0
        reliability_coefficient = 0
    
    reliability_analysis['split_half_correlation'] = round(split_half_corr, 4)
    reliability_analysis['reliability_coefficient'] = round(reliability_coefficient, 4)
    
    # Assess reliability
    if reliability_coefficient >= 0.9:
        reliability_assessment = 'EXCELLENT'
    elif reliability_coefficient >= 0.8:
        reliability_assessment = 'GOOD'
    elif reliability_coefficient >= 0.7:
        reliability_assessment = 'ACCEPTABLE'
    else:
        reliability_assessment = 'QUESTIONABLE'
    
    reliability_analysis['assessment'] = reliability_assessment
    
    # Intra-player consistency (coefficient of variation)
    player_consistency = bcs_df.groupby('player_id').agg({
        'base_bcs_score': ['mean', 'std', 'count']
    })
    player_consistency.columns = ['mean_bcs', 'std_bcs', 'n_plays']
    player_consistency['cv'] = player_consistency['std_bcs'] / player_consistency['mean_bcs']
    
    qualified_players = player_consistency[player_consistency['n_plays'] >= 5]
    avg_cv = qualified_players['cv'].mean()
    
    reliability_analysis['avg_coefficient_of_variation'] = round(avg_cv, 4)
    reliability_analysis['qualified_players'] = len(qualified_players)
    
    print(f"         Reliability analysis:")
    print(f"            Split-half correlation: r = {split_half_corr:.3f}")
    print(f"            Reliability coefficient: {reliability_coefficient:.3f}")
    print(f"            Assessment: {reliability_assessment}")
    print(f"            Avg CV (qualified): {avg_cv:.3f}")
    print(f"            Qualified players: {len(qualified_players):,}")
    
    validation_results['reliability_analysis'] = reliability_analysis
    
    # =====================================================================
    # STEP 7: CONVERGENT VALIDITY
    # =====================================================================
    
    print(f"\n      STEP 7: CONVERGENT VALIDITY")
    
    convergent_validity = {}
    validity_correlations = {}
    
    related_metrics = [
        ('comprehensive_effort_score', 'Effort'),
        ('path_efficiency', 'Path Efficiency'),
        ('total_distance', 'Distance'),
        ('max_velocity', 'Velocity'),
        ('final_distance_to_ball', 'Ball Proximity (inverse)')
    ]
    
    for metric, name in related_metrics:
        if metric in bcs_df.columns:
            corr = bcs_df['base_bcs_score'].corr(bcs_df[metric])
            validity_correlations[name] = round(corr, 4)
    
    print(f"         Convergent validity correlations:")
    for name, corr in validity_correlations.items():
        direction = '+' if corr > 0 else ''
        strength = 'strong' if abs(corr) > 0.5 else 'moderate' if abs(corr) > 0.3 else 'weak'
        print(f"            {name}: r = {direction}{corr:.3f} ({strength})")
    
    # Expected negative correlation with distance to ball
    if 'Ball Proximity (inverse)' in validity_correlations:
        ball_corr = validity_correlations['Ball Proximity (inverse)']
        validity_assessment = 'VALID' if ball_corr < -0.3 else 'PARTIALLY_VALID'
    else:
        validity_assessment = 'CANNOT_ASSESS'
    
    convergent_validity['correlations'] = validity_correlations
    convergent_validity['assessment'] = validity_assessment
    
    print(f"         Convergent validity: {validity_assessment}")
    
    validation_results['convergent_validity'] = convergent_validity
    
    # =====================================================================
    # STEP 8: DISCRIMINANT VALIDITY
    # =====================================================================
    
    print(f"\n      STEP 8: DISCRIMINANT VALIDITY")
    
    discriminant_validity = {}
    
    if 'performance_tier' in bcs_df.columns:
        elite_bcs = bcs_df[bcs_df['performance_tier'] == 'ELITE']['base_bcs_score']
        non_elite_bcs = bcs_df[bcs_df['performance_tier'] != 'ELITE']['base_bcs_score']
        
        if len(elite_bcs) > 0 and len(non_elite_bcs) > 0:
            mean_diff = elite_bcs.mean() - non_elite_bcs.mean()
            pooled_std = np.sqrt((elite_bcs.var() + non_elite_bcs.var()) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            # Non-overlap percentage (approximate)
            non_overlap = 2 * (1 - 0.5 * (1 + np.tanh(cohens_d / np.sqrt(2)))) * 100
            
            discriminant_validity['elite_nonelite'] = {
                'elite_mean': elite_bcs.mean(),
                'nonelite_mean': non_elite_bcs.mean(),
                'mean_difference': mean_diff,
                'cohens_d': round(cohens_d, 3),
                'effect_size': 'LARGE' if cohens_d > 0.8 else 'MEDIUM' if cohens_d > 0.5 else 'SMALL',
                'separation': 'EXCELLENT' if cohens_d > 1.5 else 'GOOD' if cohens_d > 1.0 else 'MODERATE',
                'non_overlap_pct': round(non_overlap, 1)
            }
            
            print(f"         Elite vs Non-Elite:")
            print(f"            Elite mean: {elite_bcs.mean():.2f}")
            print(f"            Non-elite mean: {non_elite_bcs.mean():.2f}")
            print(f"            Mean diff: {mean_diff:.2f}")
            print(f"            Effect size (d): {cohens_d:.2f}")
            print(f"            Separation: {discriminant_validity['elite_nonelite']['separation']}")
    
    validation_results['discriminant_validity'] = discriminant_validity
    
    # =====================================================================
    # STEP 9: VALIDATION SUMMARY
    # =====================================================================
    
    print(f"\n      STEP 9: VALIDATION SUMMARY")
    
    validation_checks = {
        'distribution_normal': bcs_distribution['normality_assessment'] in ['APPROXIMATELY_NORMAL', 'SLIGHTLY_SKEWED'],
        'tier_separation': tier_validation.get('separation_valid', False),
        'battle_predictive': battle_validation.get('prediction_accuracy', 0) > 50,
        'reliability_acceptable': reliability_analysis.get('reliability_coefficient', 0) >= 0.7,
        'convergent_valid': convergent_validity.get('assessment', '') == 'VALID',
        'discriminant_valid': discriminant_validity.get('elite_nonelite', {}).get('cohens_d', 0) > 0.8
    }
    
    checks_passed = sum(validation_checks.values())
    total_checks = len(validation_checks)
    validation_score = checks_passed / total_checks * 100
    
    if validation_score >= 80:
        overall_assessment = 'STRONG_VALIDITY'
    elif validation_score >= 60:
        overall_assessment = 'ACCEPTABLE_VALIDITY'
    elif validation_score >= 40:
        overall_assessment = 'WEAK_VALIDITY'
    else:
        overall_assessment = 'POOR_VALIDITY'
    
    validation_summary = {
        'checks_passed': checks_passed,
        'total_checks': total_checks,
        'validation_score': round(validation_score, 1),
        'overall_assessment': overall_assessment,
        'individual_checks': validation_checks,
        'component_importance': component_importance
    }
    
    print(f"         Validation checks:")
    for check, passed in validation_checks.items():
        status = 'PASS' if passed else 'FAIL'
        print(f"            [{status}] {check}")
    
    print(f"\n         VALIDATION SCORE: {validation_score:.1f}% ({checks_passed}/{total_checks})")
    print(f"         OVERALL: {overall_assessment}")
    
    validation_results['validation_summary'] = validation_summary
    
    championship_data['validation_results'] = validation_results
    championship_data['validation_summary'] = validation_summary
    
    aggressive_gc()
    
    return validation_results

# Execute
statistical_validation_results = perform_statistical_validation()

aggressive_gc()

print(f"\nMemory after validation: {get_memory_usage():.1f} MB")

if statistical_validation_results:
    print("CELL 12 COMPLETE - STATISTICAL VALIDATION DONE!")
    summary = statistical_validation_results['validation_summary']
    print(f"\nRESULTS:")
    print(f"   Score: {summary['validation_score']:.1f}% ({summary['checks_passed']}/{summary['total_checks']})")
    print(f"   Assessment: {summary['overall_assessment']}")
else:
    print("CELL 12 FAILED")


# ========================================================================
# CELL 13: BUSINESS APPLICATIONS - KAGGLE SAFE (COMPLETE)
# Actionable NFL business insights - ALL FEATURES
# ========================================================================

print("\nBUSINESS APPLICATIONS")
print("="*45)

@timing_decorator
def generate_business_applications():
    """Generate NFL business applications from BCS framework - COMPLETE"""
    
    print(f"\n   Memory: {get_memory_gb():.2f}GB")
    
    if elite_identification_results is None or statistical_validation_results is None:
        print("      Missing required data")
        return None
    
    player_elite = elite_identification_results['elite_players_df'].copy()
    elite_cards = elite_identification_results['elite_cards']
    position_rankings = elite_identification_results['position_rankings']
    battles_df = battle_detection_results['battles_df'].copy()
    matchup_analysis = battle_detection_results['matchup_analysis'].copy()
    zone_analysis = championship_data.get('zone_analysis', pd.DataFrame())
    position_stats = championship_data.get('position_stats', {})
    validation_summary = statistical_validation_results['validation_summary']
    
    print(f"      Business data:")
    print(f"         Elite players: {len(elite_cards)}")
    print(f"         Position rankings: {len(position_rankings)}")
    print(f"         Validation score: {validation_summary['validation_score']:.1f}%")
    
    business_applications = {}
    
    # =====================================================================
    # APPLICATION 1: PLAYER EVALUATION & SCOUTING
    # =====================================================================
    
    print(f"\n      APP 1: PLAYER SCOUTING")
    
    scouting_reports = []
    
    qualified = player_elite[player_elite['total_plays'] >= 5]
    top_prospects = qualified.nlargest(50, 'elite_score')
    
    for _, player in top_prospects.iterrows():
        # Determine player archetype
        strengths = player['strengths'].split(',') if player['strengths'] else []
        weaknesses = player['weaknesses'].split(',') if player['weaknesses'] else []
        
        if player['performance_profile'] == 'COMPLETE':
            archetype = 'FRANCHISE_PLAYER'
            projection = 'STAR'
        elif player['performance_profile'] == 'ATHLETIC':
            archetype = 'PHYSICAL_SPECIMEN'
            projection = 'HIGH_CEILING'
        elif player['performance_profile'] == 'TECHNICIAN':
            archetype = 'POLISHED_ROUTE_RUNNER'
            projection = 'RELIABLE_STARTER'
        elif player['performance_profile'] == 'COMPETITOR':
            archetype = 'GAMER'
            projection = 'CLUTCH_PERFORMER'
        elif player['performance_profile'] == 'RELIABLE':
            archetype = 'STEADY_PERFORMER'
            projection = 'STARTER'
        else:
            archetype = 'DEVELOPMENTAL'
            projection = 'DEPTH'
        
        # Calculate draft value (hypothetical)
        if player['elite_score'] >= 75:
            draft_value = 'ROUND_1'
        elif player['elite_score'] >= 68:
            draft_value = 'ROUND_2'
        elif player['elite_score'] >= 62:
            draft_value = 'ROUND_3'
        elif player['elite_score'] >= 55:
            draft_value = 'ROUND_4_5'
        else:
            draft_value = 'LATE_ROUND_UDFA'
        
        report = {
            'player_id': int(player['player_id']),
            'position': player['inferred_position'],
            'elite_score': float(player['elite_score']),
            'elite_tier': player['elite_tier'],
            'archetype': archetype,
            'projection': projection,
            'draft_value': draft_value,
            
            # Key metrics
            'bcs_grade': float(player['avg_bcs']),
            'ceiling': float(player['max_bcs']),
            'floor': float(player['min_bcs']),
            'consistency': float(player['consistency_factor']),
            
            # Strengths & weaknesses
            'primary_strengths': strengths[:3] if strengths else ['UNKNOWN'],
            'areas_for_development': weaknesses[:2] if weaknesses else ['NONE'],
            'strongest_component': player['strongest_component'],
            'weakest_component': player['weakest_component'],
            
            # Sample & reliability
            'sample_size': int(player['total_plays']),
            'games': int(player.get('games_played', 0)),
            'reliability': 'HIGH' if player['total_plays'] >= 10 else 'MEDIUM' if player['total_plays'] >= 5 else 'LOW',
            
            # Battle performance
            'total_battles': int(player['total_battles']),
            'battle_win_rate': float(player.get('battle_win_rate', 50))
        }
        
        scouting_reports.append(report)
    
    print(f"         Scouting reports: {len(scouting_reports)}")
    
    # Position-specific scouting tiers
    position_tiers = {}
    for pos, data in position_rankings.items():
        if data['total_players'] >= 5:
            top_10 = data.get('top_10', [])
            position_tiers[pos] = {
                'tier_1': top_10[:3] if len(top_10) >= 3 else top_10,
                'tier_2': top_10[3:7] if len(top_10) >= 7 else top_10[3:] if len(top_10) > 3 else [],
                'tier_3': top_10[7:10] if len(top_10) >= 10 else top_10[7:] if len(top_10) > 7 else [],
                'position_depth': data['total_players'],
                'elite_count': data['elite_count']
            }
    
    business_applications['scouting'] = {
        'reports': scouting_reports,
        'position_tiers': position_tiers,
        'total_prospects_evaluated': len(scouting_reports)
    }
    
    print(f"         Position tiers: {len(position_tiers)}")
    
    # =====================================================================
    # APPLICATION 2: GAME PLANNING & MATCHUP STRATEGY
    # =====================================================================
    
    print(f"\n      APP 2: GAME PLANNING")
    
    matchup_strategy = {}
    
    # Identify favorable matchups
    if len(matchup_analysis) > 0:
        favorable_offense = matchup_analysis[matchup_analysis['avg_bcs_advantage'] > 3].nlargest(5, 'avg_bcs_advantage')
        favorable_defense = matchup_analysis[matchup_analysis['avg_bcs_advantage'] < -3].nsmallest(5, 'avg_bcs_advantage')
        
        matchup_strategy['offensive_advantages'] = []
        for _, row in favorable_offense.iterrows():
            matchup_strategy['offensive_advantages'].append({
                'matchup': row['matchup_key'],
                'bcs_advantage': float(row['avg_bcs_advantage']),
                'win_rate': float(row['offensive_win_rate']),
                'sample_size': int(row['battle_count']),
                'recommendation': f"TARGET {row['matchup_key'].split('_vs_')[0]} vs {row['matchup_key'].split('_vs_')[1]}"
            })
        
        matchup_strategy['defensive_advantages'] = []
        for _, row in favorable_defense.iterrows():
            matchup_strategy['defensive_advantages'].append({
                'matchup': row['matchup_key'],
                'bcs_advantage': float(row['avg_bcs_advantage']),
                'win_rate': float(row['offensive_win_rate']),
                'sample_size': int(row['battle_count']),
                'recommendation': f"AVOID {row['matchup_key'].split('_vs_')[0]} vs {row['matchup_key'].split('_vs_')[1]}"
            })
    
    # Field zone strategy - hot zones and danger zones
    if len(zone_analysis) > 0:
        matchup_strategy['hot_zones'] = zone_analysis.nlargest(3, 'offensive_win_rate')[
            ['field_zone', 'battle_count', 'offensive_win_rate']
        ].to_dict('records')
        
        matchup_strategy['danger_zones'] = zone_analysis.nsmallest(3, 'offensive_win_rate')[
            ['field_zone', 'battle_count', 'offensive_win_rate']
        ].to_dict('records')
    else:
        matchup_strategy['hot_zones'] = []
        matchup_strategy['danger_zones'] = []
    
    # Coverage tendency analysis
    coverage_analysis = battles_df.groupby('coverage_type').agg({
        'game_id': 'count',
        'bcs_advantage': 'mean',
        'outcome': lambda x: (x.isin(['DOMINANT_OFFENSIVE', 'OFFENSIVE_WIN'])).sum() / len(x) * 100
    }).reset_index()
    coverage_analysis.columns = ['coverage', 'battles', 'avg_advantage', 'offensive_success']
    
    matchup_strategy['coverage_tendencies'] = coverage_analysis.to_dict('records')
    
    business_applications['game_planning'] = matchup_strategy
    
    print(f"         Offensive advantages: {len(matchup_strategy.get('offensive_advantages', []))}")
    print(f"         Defensive advantages: {len(matchup_strategy.get('defensive_advantages', []))}")
    print(f"         Hot zones: {len(matchup_strategy.get('hot_zones', []))}")
    print(f"         Danger zones: {len(matchup_strategy.get('danger_zones', []))}")
    
    # =====================================================================
    # APPLICATION 3: PERSONNEL DECISIONS
    # =====================================================================
    
    print(f"\n      APP 3: PERSONNEL DECISIONS")
    
    personnel_decisions = {}
    
    # Contract value assessment
    contract_tiers = {
        'PREMIUM': [],      # Top 10% - max contracts
        'ABOVE_MARKET': [], # Top 25% - above average
        'MARKET': [],       # 25-50% - market value
        'BELOW_MARKET': [], # 50-75% - team-friendly
        'MINIMUM': []       # Bottom 25% - minimum deals
    }
    
    for _, player in qualified.iterrows():
        percentile = player['overall_percentile']
        
        tier_data = {
            'player_id': int(player['player_id']),
            'position': player['inferred_position'],
            'elite_score': float(player['elite_score']),
            'percentile': float(percentile),
            'consistency': float(player['consistency_factor']),
            'elite_tier': player['elite_tier']
        }
        
        if percentile >= 90:
            contract_tiers['PREMIUM'].append(tier_data)
        elif percentile >= 75:
            contract_tiers['ABOVE_MARKET'].append(tier_data)
        elif percentile >= 50:
            contract_tiers['MARKET'].append(tier_data)
        elif percentile >= 25:
            contract_tiers['BELOW_MARKET'].append(tier_data)
        else:
            contract_tiers['MINIMUM'].append(tier_data)
    
    personnel_decisions['contract_tiers'] = {
        tier: {'count': len(players), 'players': players[:10]}
        for tier, players in contract_tiers.items()
    }
    
    # Release candidates
    release_candidates = qualified[
        (qualified['elite_score'] < 45) & 
        (qualified['consistency_factor'] < 50) &
        (qualified['total_plays'] >= 5)
    ].nsmallest(10, 'elite_score')
    
    personnel_decisions['release_candidates'] = release_candidates[
        ['player_id', 'inferred_position', 'elite_score', 'consistency_factor', 'elite_tier']
    ].to_dict('records')
    
    # Extension candidates
    extension_candidates = qualified[
        (qualified['elite_score'] >= 65) &
        (qualified['consistency_factor'] >= 60) &
        (qualified['upside_potential'] >= 10)
    ].nlargest(10, 'elite_score')
    
    personnel_decisions['extension_candidates'] = extension_candidates[
        ['player_id', 'inferred_position', 'elite_score', 'upside_potential', 'elite_tier']
    ].to_dict('records')
    
    business_applications['personnel'] = personnel_decisions
    
    print(f"         Contract tiers assigned: {sum(len(v['players']) for v in personnel_decisions['contract_tiers'].values())}")
    print(f"         Release candidates: {len(personnel_decisions['release_candidates'])}")
    print(f"         Extension candidates: {len(personnel_decisions['extension_candidates'])}")
    
    # =====================================================================
    # APPLICATION 4: COACHING & DEVELOPMENT
    # =====================================================================
    
    print(f"\n      APP 4: COACHING & DEVELOPMENT")
    
    coaching_development = {}
    
    # Development priorities by position
    development_priorities = {}
    
    for pos, data in position_rankings.items():
        if data['total_players'] < 5:
            continue
        
        pos_players = qualified[qualified['inferred_position'] == pos]
        
        if len(pos_players) < 3:
            continue
        
        # Find high-upside players needing development
        development_candidates = pos_players[
            (pos_players['upside_potential'] > 15) &
            (pos_players['consistency_factor'] < 60)
        ]
        
        if len(development_candidates) > 0:
            # Identify common weaknesses
            all_weaknesses = []
            for _, p in development_candidates.iterrows():
                if p['weaknesses']:
                    all_weaknesses.extend(p['weaknesses'].split(','))
            
            weakness_counts = pd.Series(all_weaknesses).value_counts() if all_weaknesses else pd.Series()
            
            development_priorities[pos] = {
                'candidates': len(development_candidates),
                'avg_upside': development_candidates['upside_potential'].mean(),
                'primary_focus': weakness_counts.index[0] if len(weakness_counts) > 0 else 'GENERAL',
                'secondary_focus': weakness_counts.index[1] if len(weakness_counts) > 1 else 'GENERAL',
                'top_prospects': development_candidates.nlargest(3, 'upside_potential')[
                    ['player_id', 'elite_score', 'upside_potential', 'weakest_component']
                ].to_dict('records')
            }
    
    coaching_development['position_development'] = development_priorities
    
    # Skill-specific training recommendations
    component_training = {
        'PROXIMITY': {
            'focus': 'Route precision and ball tracking',
            'drills': ['Ball location drills', 'Catch radius work', 'Spatial awareness'],
            'candidates': len(qualified[qualified['weakest_component'] == 'proximity'])
        },
        'EFFICIENCY': {
            'focus': 'Route efficiency and path optimization',
            'drills': ['Route tree work', 'Footwork fundamentals', 'Break point technique'],
            'candidates': len(qualified[qualified['weakest_component'] == 'efficiency'])
        },
        'EFFORT': {
            'focus': 'Sustained effort and motor',
            'drills': ['Conditioning', 'Finish drills', 'Competitive periods'],
            'candidates': len(qualified[qualified['weakest_component'] == 'effort'])
        },
        'VELOCITY': {
            'focus': 'Speed and acceleration',
            'drills': ['Sprint work', 'Explosion training', 'Speed technique'],
            'candidates': len(qualified[qualified['weakest_component'] == 'velocity'])
        }
    }
    
    coaching_development['skill_training'] = component_training
    
    business_applications['coaching'] = coaching_development
    
    print(f"         Development priorities: {len(development_priorities)} positions")
    print(f"         Skill training programs: {len(component_training)}")
    
    # =====================================================================
    # APPLICATION 5: BROADCAST & ANALYTICS INSIGHTS
    # =====================================================================
    
    print(f"\n      APP 5: BROADCAST INSIGHTS")
    
    broadcast_insights = {}
    
    # Pre-game storylines
    storylines = []
    
    # Elite matchups to watch
    elite_battles = championship_data.get('elite_battles_df', pd.DataFrame())
    if len(elite_battles) > 0:
        top_matchups = elite_battles.groupby(['receiver_id', 'primary_defender_id']).agg({
            'battle_intensity': 'mean',
            'game_id': 'count',
            'bcs_advantage': 'mean'
        }).reset_index()
        top_matchups.columns = ['receiver', 'defender', 'avg_intensity', 'battles', 'avg_advantage']
        top_matchups = top_matchups[top_matchups['battles'] >= 2].nlargest(5, 'avg_intensity')
        
        for _, m in top_matchups.iterrows():
            storylines.append({
                'type': 'ELITE_MATCHUP',
                'headline': f"Player {int(m['receiver'])} vs Player {int(m['defender'])}",
                'narrative': f"High-intensity battle averaging {m['avg_intensity']:.1f} intensity over {int(m['battles'])} encounters",
                'key_stat': f"BCS Advantage: {m['avg_advantage']:+.1f}"
            })
    
    # Breakout candidates
    breakout = qualified[
        (qualified['max_bcs'] >= 80) & 
        (qualified['avg_bcs'] < 60) &
        (qualified['upside_potential'] >= 20)
    ].nlargest(5, 'upside_potential')
    
    for _, p in breakout.iterrows():
        storylines.append({
            'type': 'BREAKOUT_WATCH',
            'headline': f"Player {int(p['player_id'])} ({p['inferred_position']})",
            'narrative': f"Flashed elite potential with {p['max_bcs']:.1f} peak BCS",
            'key_stat': f"Upside: +{p['upside_potential']:.1f} points"
        })
    
    # Consistency stories
    most_consistent = qualified.nlargest(3, 'consistency_factor')
    for _, p in most_consistent.iterrows():
        storylines.append({
            'type': 'CONSISTENCY_KING',
            'headline': f"Player {int(p['player_id'])} ({p['inferred_position']})",
            'narrative': f"Most reliable performer with {p['consistency_factor']:.1f} consistency score",
            'key_stat': f"Avg BCS: {p['avg_bcs']:.1f}"
        })
    
    broadcast_insights['storylines'] = storylines
    
    # Real-time graphics data
    graphics_data = {
        'bcs_leaderboard': qualified.nlargest(10, 'avg_bcs')[
            ['player_id', 'inferred_position', 'avg_bcs', 'elite_tier']
        ].to_dict('records'),
        'consistency_leaders': qualified.nlargest(10, 'consistency_factor')[
            ['player_id', 'inferred_position', 'consistency_factor', 'avg_bcs']
        ].to_dict('records'),
        'battle_win_leaders': qualified[qualified['total_battles'] >= 3].nlargest(10, 'battle_win_rate')[
            ['player_id', 'inferred_position', 'battle_win_rate', 'total_battles']
        ].to_dict('records') if 'battle_win_rate' in qualified.columns else [],
        'upside_leaders': qualified.nlargest(10, 'upside_potential')[
            ['player_id', 'inferred_position', 'upside_potential', 'max_bcs', 'avg_bcs']
        ].to_dict('records')
    }
    
    broadcast_insights['graphics_data'] = graphics_data
    
    # Telestrator talking points
    component_importance = validation_summary.get('component_importance', {})
    proximity_pct = component_importance.get('proximity_score', {}).get('variance_explained_pct', 40)
    
    broadcast_insights['talking_points'] = [
        f"BCS framework validated with {validation_summary['validation_score']:.0f}% confidence",
        f"Proximity accounts for {proximity_pct:.0f}% of performance variance",
        f"Elite players ({len(elite_cards)}) consistently outperform in battle situations",
        f"Position hierarchy shows clear differentiation in ball convergence patterns",
        f"Battle prediction accuracy: {statistical_validation_results['battle_validation']['prediction_accuracy']:.0f}%"
    ]
    
    business_applications['broadcast'] = broadcast_insights
    
    print(f"         Storylines: {len(storylines)}")
    print(f"         Graphics packages: {len(graphics_data)}")
    print(f"         Talking points: {len(broadcast_insights['talking_points'])}")
    
    # =====================================================================
    # APPLICATION 6: EXECUTIVE SUMMARY & ROI
    # =====================================================================
    
    print(f"\n      APP 6: EXECUTIVE SUMMARY")
    
    executive_summary = {
        'framework_validity': {
            'validation_score': validation_summary['validation_score'],
            'assessment': validation_summary['overall_assessment'],
            'confidence_level': 'HIGH' if validation_summary['validation_score'] >= 80 else 'MEDIUM' if validation_summary['validation_score'] >= 60 else 'LOW'
        },
        
        'key_findings': [
            f"Analyzed {len(player_elite):,} players across {player_elite['inferred_position'].nunique()} positions",
            f"Identified {len(elite_cards)} elite performers using multi-factor analysis",
            f"Battle prediction accuracy: {statistical_validation_results['battle_validation']['prediction_accuracy']:.1f}%",
            f"Framework reliability: {statistical_validation_results['reliability_analysis']['assessment']}",
            f"BCS-Outcome correlation: r = {statistical_validation_results['battle_validation']['bcs_outcome_correlation']:.3f}"
        ],
        
        'business_value': {
            'scouting_reports': len(scouting_reports),
            'position_tiers': len(position_tiers),
            'matchup_strategies': len(matchup_strategy.get('offensive_advantages', [])) + len(matchup_strategy.get('defensive_advantages', [])),
            'personnel_recommendations': len(personnel_decisions.get('extension_candidates', [])) + len(personnel_decisions.get('release_candidates', [])),
            'development_programs': len(development_priorities),
            'broadcast_content': len(storylines)
        },
        
        'competitive_advantage': {
            'unique_metric': 'Ball Convergence Score (BCS)',
            'innovation': '4-component weighted analysis with position-specific optimization',
            'differentiator': 'Battle detection and outcome prediction system',
            'scalability': 'Framework applicable to any passing play with tracking data'
        },
        
        'recommendations': [
            'Implement BCS in weekly game planning for matchup analysis',
            'Use position-weighted scoring for player evaluation and contracts',
            'Deploy battle detection for real-time broadcast analytics',
            'Integrate development priorities into coaching curriculum',
            'Leverage hot zones and danger zones for play calling strategy'
        ]
    }
    
    business_applications['executive_summary'] = executive_summary
    
    # =====================================================================
    # STORAGE
    # =====================================================================
    
    print(f"\n      STORING APPLICATIONS")
    
    championship_data['business_applications'] = business_applications
    championship_data['scouting_reports'] = scouting_reports
    championship_data['position_tiers'] = position_tiers
    championship_data['matchup_strategy'] = matchup_strategy
    championship_data['personnel_decisions'] = personnel_decisions
    championship_data['coaching_development'] = coaching_development
    championship_data['broadcast_insights'] = broadcast_insights
    championship_data['executive_summary'] = executive_summary
    
    application_stats = {
        'total_applications': 6,
        'scouting_reports': len(scouting_reports),
        'position_tiers': len(position_tiers),
        'matchup_strategies': len(matchup_strategy.get('offensive_advantages', [])),
        'personnel_recommendations': len(personnel_decisions.get('extension_candidates', [])),
        'development_programs': len(development_priorities),
        'broadcast_storylines': len(storylines),
        'executive_recommendations': len(executive_summary['recommendations'])
    }
    
    aggressive_gc()
    
    return {
        'business_applications': business_applications,
        'application_stats': application_stats
    }

# Execute
business_application_results = generate_business_applications()

aggressive_gc()

print(f"\nMemory after business apps: {get_memory_usage():.1f} MB")

if business_application_results:
    print("CELL 13 COMPLETE - BUSINESS APPLICATIONS DONE!")
    s = business_application_results['application_stats']
    print(f"\nRESULTS:")
    print(f"   Scouting reports: {s['scouting_reports']}")
    print(f"   Position tiers: {s['position_tiers']}")
    print(f"   Matchup strategies: {s['matchup_strategies']}")
    print(f"   Personnel recommendations: {s['personnel_recommendations']}")
    print(f"   Development programs: {s['development_programs']}")
    print(f"   Broadcast storylines: {s['broadcast_storylines']}")
else:
    print("CELL 13 FAILED")


# ========================================================================
# CELL 14: VISUALIZATION 1 - Violin Plot: Average BCS by Position
# Distribution of Average BCS by Inferred Position
# ========================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

print("\n" + "="*60)
print("VISUALIZATION 1: Distribution of Average BCS by Position")
print("="*60)

# Get player_agg from results
player_agg = player_bcs_results['player_aggregations_df'].copy()

# Set clean style
plt.style.use('seaborn-v0_8-white')
sns.set_context("notebook", font_scale=1.1)

# Create figure
plt.figure(figsize=(12, 7))

# Order positions by median BCS (descending)
position_order = (player_agg.groupby('inferred_position')['avg_bcs']
                  .median()
                  .sort_values(ascending=False)
                  .index.tolist())

# Create violin plot
ax = sns.violinplot(
    data=player_agg,
    x='inferred_position',
    y='avg_bcs',
    order=position_order,
    palette='viridis',
    inner='quartile',
    linewidth=1
)

# Add overall median line
median_bcs = player_agg['avg_bcs'].median()
plt.axhline(y=median_bcs, color='red', linestyle='--', linewidth=1.5, 
            label=f'Overall Median: {median_bcs:.1f}')

# Formatting
plt.title('Distribution of Average BCS by Inferred Position', 
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Inferred Position', fontsize=12)
plt.ylabel('Average BCS Score', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(loc='upper right')

# Add sample sizes
for i, pos in enumerate(position_order):
    n = len(player_agg[player_agg['inferred_position'] == pos])
    plt.text(i, plt.ylim()[0] + 1, f'n={n}', ha='center', va='bottom', fontsize=8, color='gray')

plt.tight_layout()
plt.savefig('viz_01_bcs_by_position.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\nVisualization 1 complete - BCS distribution by position")
print(f"   Positions analyzed: {len(position_order)}")
print(f"   Overall median BCS: {median_bcs:.2f}")


# ========================================================================
# CELL 15: VISUALIZATION 2 - Scatter: Upside Potential vs Floor Risk
# Risk vs Potential Analysis by Championship Tier
# ========================================================================

print("\n" + "="*60)
print("VISUALIZATION 2: Upside Potential vs Floor Risk by Tier")
print("="*60)

# Create figure
plt.figure(figsize=(11, 8))

# Define tier order and color palette
tier_order = ['CHAMPIONSHIP_ELITE', 'ELITE', 'EXCELLENT', 'GOOD', 'AVERAGE', 'DEVELOPING']
tier_colors = {
    'CHAMPIONSHIP_ELITE': '#FFD700',  # Gold
    'ELITE': '#C0C0C0',               # Silver
    'EXCELLENT': '#CD7F32',           # Bronze
    'GOOD': '#2ecc71',                # Green
    'AVERAGE': '#3498db',             # Blue
    'DEVELOPING': '#95a5a6'           # Gray
}

# Plot each tier
for tier in tier_order:
    tier_data = player_agg[player_agg['championship_tier'] == tier]
    if len(tier_data) > 0:
        plt.scatter(
            tier_data['upside_potential'],
            tier_data['floor_risk'],
            label=f'{tier} (n={len(tier_data)})',
            color=tier_colors.get(tier, 'gray'),
            alpha=0.65,
            s=60,
            edgecolor='white',
            linewidth=0.5
        )

# Add quadrant lines at medians
median_upside = player_agg['upside_potential'].median()
median_floor = player_agg['floor_risk'].median()
plt.axvline(x=median_upside, color='gray', linestyle=':', alpha=0.6)
plt.axhline(y=median_floor, color='gray', linestyle=':', alpha=0.6)

# Add quadrant labels
plt.text(plt.xlim()[1] * 0.85, plt.ylim()[1] * 0.9, 'High Risk\nHigh Upside', 
         ha='center', fontsize=9, color='gray', alpha=0.7)
plt.text(plt.xlim()[0] + 2, plt.ylim()[1] * 0.9, 'High Risk\nLow Upside', 
         ha='center', fontsize=9, color='gray', alpha=0.7)
plt.text(plt.xlim()[1] * 0.85, plt.ylim()[0] + 2, 'Low Risk\nHigh Upside', 
         ha='center', fontsize=9, color='gray', alpha=0.7)
plt.text(plt.xlim()[0] + 2, plt.ylim()[0] + 2, 'Low Risk\nLow Upside', 
         ha='center', fontsize=9, color='gray', alpha=0.7)

# Formatting
plt.title('Upside Potential vs Floor Risk by Championship Tier', 
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Upside Potential (Max BCS - Avg BCS)', fontsize=12)
plt.ylabel('Floor Risk (Avg BCS - Min BCS)', fontsize=12)
plt.legend(title='Championship Tier', loc='upper right', fontsize=9, title_fontsize=10)

plt.tight_layout()
plt.savefig('viz_02_risk_vs_potential.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\nVisualization 2 complete - Risk vs Potential analysis")
print(f"   Median upside potential: {median_upside:.2f}")
print(f"   Median floor risk: {median_floor:.2f}")


# ========================================================================
# CELL 16: VISUALIZATION 3 - KDE: Component Balance by Dominant Role
# Component Balance Distribution Analysis
# ========================================================================

print("\nVISUALIZATION 3: COMPONENT BALANCE BY ROLE")
print("="*50)

print(f"   Memory: {get_memory_gb():.2f}GB")

# Verify data availability
if player_bcs_results is None:
    print("   ERROR: player_bcs_results not available")
else:
    # Get player_agg
    player_agg = player_bcs_results['player_aggregations_df'].copy()
    
    # Verify component_balance exists
    if 'component_balance' not in player_agg.columns:
        print("   WARNING: Creating component_balance column")
        if 'component_gap' in player_agg.columns:
            player_agg['component_balance'] = 100 - np.minimum(player_agg['component_gap'], 50)
        else:
            # Calculate from component scores
            comp_cols = ['avg_proximity', 'avg_efficiency', 'avg_effort', 'avg_velocity']
            if all(col in player_agg.columns for col in comp_cols):
                comp_std = player_agg[comp_cols].std(axis=1)
                player_agg['component_balance'] = 100 - np.minimum(comp_std * 2, 50)
            else:
                player_agg['component_balance'] = 50  # Default
    
    print(f"   Players: {len(player_agg):,}")
    print(f"   Roles: {player_agg['dominant_role'].value_counts().to_dict()}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Define role colors
    role_colors = {
        'RECEIVER': '#2ecc71',
        'DEFENDER': '#e74c3c',
        'OTHER': '#3498db'
    }
    
    role_stats = {}
    
    # Plot KDE for each role
    for role in ['RECEIVER', 'DEFENDER', 'OTHER']:
        role_data = player_agg[player_agg['dominant_role'] == role]['component_balance']
        if len(role_data) > 10:
            sns.kdeplot(
                data=role_data,
                label=f'{role} (n={len(role_data)}, mean={role_data.mean():.1f})',
                color=role_colors.get(role, 'gray'),
                linewidth=2.5,
                fill=True,
                alpha=0.25,
                ax=ax
            )
            role_stats[role] = {'count': len(role_data), 'mean': role_data.mean(), 'std': role_data.std()}
    
    # Add overall mean line
    overall_mean = player_agg['component_balance'].mean()
    ax.axvline(x=overall_mean, color='black', linestyle='--', linewidth=1.5, 
               label=f'Overall Mean: {overall_mean:.1f}')
    
    # Formatting
    ax.set_title('Component Balance Distribution by Dominant Role', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Component Balance Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(title='Dominant Role', loc='upper left', fontsize=10)
    
    # Add interpretation note
    ax.text(0.98, 0.02, 'Higher = More Balanced Components', 
            transform=ax.transAxes, ha='right', va='bottom', 
            fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig('viz_03_component_balance_kde.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Store in championship_data
    championship_data['viz_03_stats'] = {
        'overall_mean': overall_mean,
        'role_stats': role_stats
    }
    
    print(f"\n   Visualization 3 saved: viz_03_component_balance_kde.png")
    print(f"   Overall mean balance: {overall_mean:.2f}")
    for role, stats in role_stats.items():
        print(f"   {role}: mean={stats['mean']:.1f}, std={stats['std']:.1f}")

aggressive_gc()


# ========================================================================
# CELL 17: VISUALIZATION 4 - Bar: Battle Outcome by BCS Advantage
# Win Rate Analysis by BCS Advantage Bracket
# ========================================================================

print("\nVISUALIZATION 4: BATTLE OUTCOMES BY BCS ADVANTAGE")
print("="*50)

print(f"   Memory: {get_memory_gb():.2f}GB")

# Verify data availability
if battle_detection_results is None:
    print("   ERROR: battle_detection_results not available")
else:
    # Get battles data
    battles_df = battle_detection_results['battles_df'].copy()
    
    print(f"   Battles: {len(battles_df):,}")
    
    # Create BCS advantage brackets
    def categorize_advantage(adv):
        if adv < -5:
            return '< -5 (Def Adv)'
        elif adv < 0:
            return '-5 to 0'
        elif adv < 5:
            return '0 to 5'
        elif adv < 10:
            return '5 to 10'
        else:
            return '10+ (Off Adv)'
    
    battles_df['advantage_bracket'] = battles_df['bcs_advantage'].apply(categorize_advantage)
    
    # Calculate win rates per bracket
    bracket_order = ['< -5 (Def Adv)', '-5 to 0', '0 to 5', '5 to 10', '10+ (Off Adv)']
    
    battle_summary = []
    for bracket in bracket_order:
        bracket_data = battles_df[battles_df['advantage_bracket'] == bracket]
        if len(bracket_data) > 0:
            off_wins = bracket_data['outcome'].isin(['DOMINANT_OFFENSIVE', 'OFFENSIVE_WIN']).sum()
            win_rate = off_wins / len(bracket_data) * 100
            battle_summary.append({
                'bracket': bracket,
                'offensive_win_rate': win_rate,
                'total_battles': len(bracket_data),
                'offensive_wins': off_wins
            })
    
    battle_summary_df = pd.DataFrame(battle_summary)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Color gradient based on win rate
    colors = ['#c0392b', '#e74c3c', '#f39c12', '#2ecc71', '#27ae60']
    colors = colors[:len(battle_summary_df)]
    
    # Bar plot
    bars = ax.bar(
        battle_summary_df['bracket'],
        battle_summary_df['offensive_win_rate'],
        color=colors,
        edgecolor='white',
        linewidth=1.5
    )
    
    # Add value labels on bars
    for bar, row in zip(bars, battle_summary_df.itertuples()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%\n(n={row.total_battles})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add 50% reference line
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, label='50% Baseline')
    
    # Formatting
    ax.set_title('Battle Outcome by BCS Advantage Range', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('BCS Advantage Bracket', fontsize=12)
    ax.set_ylabel('Offensive Win Rate (%)', fontsize=12)
    plt.xticks(rotation=15, ha='right')
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right')
    
    # Add interpretation box
    ax.text(0.02, 0.98, 'BCS Advantage Predicts Battle Outcomes', 
            transform=ax.transAxes, ha='left', va='top', 
            fontsize=10, style='italic', color='gray',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('viz_04_win_rate_by_advantage.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Store in championship_data
    championship_data['viz_04_stats'] = {
        'total_battles': len(battles_df),
        'bracket_summary': battle_summary_df.to_dict('records')
    }
    championship_data['battle_summary_df'] = battle_summary_df
    
    print(f"\n   Visualization 4 saved: viz_04_win_rate_by_advantage.png")
    print(f"   Total battles analyzed: {len(battles_df):,}")
    print(f"\n   Win Rate by Bracket:")
    for _, row in battle_summary_df.iterrows():
        print(f"      {row['bracket']}: {row['offensive_win_rate']:.1f}% ({row['total_battles']:,} battles)")

aggressive_gc()


# ========================================================================
# CELL 18: VISUALIZATION 5 - Horizontal Bar: Top 10 Players by Composite
# Elite Player Identification
# ========================================================================

print("\nVISUALIZATION 5: TOP 10 ELITE PLAYERS")
print("="*50)

print(f"   Memory: {get_memory_gb():.2f}GB")

# Verify data availability
if player_bcs_results is None:
    print("   ERROR: player_bcs_results not available")
else:
    # Get player_agg
    player_agg = player_bcs_results['player_aggregations_df'].copy()
    
    # Get top 10 players by composite score
    top_10 = player_agg.nlargest(10, 'composite_score')[
        ['player_id', 'composite_score', 'inferred_position', 'championship_tier', 'avg_bcs']
    ].copy()
    
    print(f"   Top 10 composite scores: {top_10['composite_score'].min():.1f} - {top_10['composite_score'].max():.1f}")
    
    # Create display labels
    top_10['player_label'] = top_10.apply(
        lambda x: f"Player {int(x['player_id'])} ({x['inferred_position']})", axis=1
    )
    
    # Sort for horizontal bar (ascending so highest is on top)
    top_10 = top_10.sort_values('composite_score', ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(11, 8))
    
    # Color by tier
    tier_colors = {
        'CHAMPIONSHIP_ELITE': '#FFD700',
        'ELITE': '#C0C0C0',
        'EXCELLENT': '#CD7F32',
        'GOOD': '#2ecc71',
        'AVERAGE': '#3498db',
        'DEVELOPING': '#95a5a6'
    }
    colors = [tier_colors.get(tier, 'gray') for tier in top_10['championship_tier']]
    
    # Horizontal bar plot
    bars = ax.barh(
        top_10['player_label'],
        top_10['composite_score'],
        color=colors,
        edgecolor='white',
        linewidth=1.5,
        height=0.7
    )
    
    # Add value labels
    for bar, (_, row) in zip(bars, top_10.iterrows()):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                f'{width:.1f} ({row["championship_tier"]})',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Formatting
    ax.set_title('Top 10 Players by Composite Score', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Composite Score', fontsize=12)
    ax.set_ylabel('Player', fontsize=12)
    
    # Extend x-axis for labels
    ax.set_xlim(0, top_10['composite_score'].max() * 1.25)
    
    # Add legend for tiers
    from matplotlib.patches import Patch
    unique_tiers = top_10['championship_tier'].unique()
    legend_elements = [Patch(facecolor=tier_colors.get(tier, 'gray'), label=tier) 
                       for tier in tier_colors.keys() if tier in unique_tiers]
    ax.legend(handles=legend_elements, title='Tier', loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('viz_05_top10_players.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Store in championship_data
    championship_data['viz_05_stats'] = {
        'top_player_id': int(top_10.iloc[-1]['player_id']),
        'top_composite_score': float(top_10.iloc[-1]['composite_score']),
        'top_10_list': top_10.iloc[::-1][['player_id', 'composite_score', 'inferred_position', 'championship_tier']].to_dict('records')
    }
    
    print(f"\n   Visualization 5 saved: viz_05_top10_players.png")
    print(f"\n   TOP 10 PLAYERS BY COMPOSITE SCORE:")
    for rank, (_, row) in enumerate(top_10.iloc[::-1].iterrows(), 1):
        print(f"      {rank:2d}. Player {int(row['player_id'])} ({row['inferred_position']}): "
              f"{row['composite_score']:.1f} - {row['championship_tier']}")

aggressive_gc()

print("\n" + "="*50)
print("ALL 5 VISUALIZATIONS COMPLETE")
print("="*50)
print("\nFiles saved:")
print("   1. viz_01_bcs_by_position.png")
print("   2. viz_02_risk_vs_potential.png")
print("   3. viz_03_component_balance_kde.png")
print("   4. viz_04_win_rate_by_advantage.png")
print("   5. viz_05_top10_players.png")

