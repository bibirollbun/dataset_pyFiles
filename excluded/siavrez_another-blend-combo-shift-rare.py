!pip install -qqq onnxruntime --no-index --find-links=file:/kaggle/input/onnxruntime


import numpy as np
from typing import Union
import os
import gc
import time
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as AT
import concurrent.futures
from tqdm import tqdm
import onnxruntime as ort

# Configuration
TEST_AUDIO_DIR_DEFAULT = '/kaggle/input/birdclef-2025/test_soundscapes/'
TRAIN_AUDIO_DIR_DEBUG = '/kaggle/input/birdclef-2025/train_soundscapes/'
TAXONOMY_FILE = "/kaggle/input/birdclef-2025/taxonomy.csv"
SUBMISSION_FILE = "submission.csv"
SUBMISSION_FILE_PSD = "submission_psd.csv"
SUBMISSION_FILE_PSD2 = "submission_psd2.csv"
SUBMISSION_FILE_PSD3 = "submission_psd3.csv"

# Model paths for both types - changed to .onnx extension
MODEL_PATHS_EF3_PSD = [f'/kaggle/input/new15sec-onnx-v2/ef3_{i}.onnx' for i in range(1,2)]
MODEL_PATHS_SER_PSD = [f'/kaggle/input/new15sec-onnx-v2/ser_{i}.onnx' for i in range(1,2)]
MODEL_PATHS_NF_PSD = [f'/kaggle/input/new15sec-onnx-v2/nfnet_{i}.onnx' for i in range(1,2)]

MODEL_PATHS_FB_PSD2 = [f'/kaggle/input/new122sec-onnx-v2/fbnet_{i}.onnx' for i in range(1,2)]
MODEL_PATHS_NF_PSD2 = [f'/kaggle/input/new20sec-onnx/ef3_{i}.onnx' for i in range(1,2)]

MODEL_PATHS_FB_PSD3 = [f'/kaggle/input/new30secvis-onnx/fbnet_{i}.onnx' for i in range(1,2)]
MODEL_PATHS_NF_PSD3 = [f'/kaggle/input/new30secvis-onnx/nfnet_{i}.onnx' for i in range(1,2)]

PSD_MODEL_PATHS = MODEL_PATHS_EF3_PSD + MODEL_PATHS_SER_PSD + MODEL_PATHS_NF_PSD
PSD2_MODEL_PATHS = MODEL_PATHS_FB_PSD2 + MODEL_PATHS_NF_PSD2
PSD3_MODEL_PATHS = MODEL_PATHS_FB_PSD3 + MODEL_PATHS_NF_PSD3

# Debugging
DEBUG_MODE = False  # Set to True to enable debug mode
DEBUG_START_NUM = 5
DEBUG_NUM_FILES = 8

# Audio Parameters
WAV_SEC = 5
SAMPLE_RATE = 32000
INFER_DURATION_SEC = 5
TRAIN_DURATION_SEC = 10

# Mel Spectrogram Parameters
N_FFT = 2048
WIN_LENGTH = 2048
HOP_LENGTH = 512
F_MIN = 20
F_MAX = 16000
N_MELS = 256
MEL_CENTER = True
MEL_PAD_MODE = "reflect"
MEL_POWER = 2.0
MEL_NORM = 'slaney'
MEL_SCALE = "htk"

# Model Parameters
BASE_MODEL_NAME_1 = 'seresnext26t_32x4d'
BASE_MODEL_NAME_2 = 'tf_efficientnetv2_b3.in21k'
BASE_MODEL_NAME_3 = 'eca_nfnet_l0'
PRETRAINED_MODELS = False
IN_CHANNELS_TIMM = 1
NUM_CLASSES = 206 # Derived from taxonomy.csv, but useful to have as a constant for AttBlockV2

# Inference Parameters
TTA_DELTA = 2
APPLY_POWER_TOP_K = 30
APPLY_POWER_EXPONENT = 2
MAX_WORKERS_THREADPOOL = 5

# End Configuration

def apply_power_to_low_ranked_cols(
    p: np.ndarray,
    top_k: int = APPLY_POWER_TOP_K,
    exponent: Union[int, float] = APPLY_POWER_EXPONENT,
    inplace: bool = True
) -> np.ndarray:
    if not inplace:
        p = p.copy()

    tail_cols = np.argsort(-p.max(axis=0))[top_k:]

    p[:, tail_cols] = p[:, tail_cols] ** exponent
    return p

amount_to_add = {32: 0.05, 58: 0.05, 5: 0.05, 1: 0.05, 7: 0.065, 2: 0.065, 17: 0.065, 44: 0.065, 52: 0.1, 48: 0.1, 50: 0.1, 57: 0.1, 27: 0.1, 33: 0.1, 49: 0.1, 20: 0.1, 26: 0.1, 23: 0.1, 24: 0.1, 11: 0.1, 0: 0.1, 10: 0.1, 39: 0.1}
SHAPE = (12, 206)
amount_array = np.zeros(SHAPE, dtype=float)
for idx, amount in amount_to_add.items():
    if idx < SHAPE[1]:  # Check bounds
        amount_array[:, idx] = amount
def add_amounts_to_top_predictions(predictions, amount_array=amount_array, top_k=5):

    predictions = predictions.astype(float)  # Ensure float type for addition
    
    # Get top k indices for all rows at once
    top_k_indices = np.argsort(predictions, axis=1)[:, -top_k:]
    
    # Create a boolean mask for top k positions
    top_k_mask = np.zeros_like(predictions, dtype=bool)
    row_indices = np.arange(predictions.shape[0])[:, np.newaxis]
    top_k_mask[row_indices, top_k_indices] = True
    # Apply amounts only where mask is True (top k positions)
    mask_and_amount = top_k_mask & (amount_array > 0)
    predictions[mask_and_amount] += amount_array[mask_and_amount]
        
    return predictions
test_audio_dir = TEST_AUDIO_DIR_DEFAULT
file_list = [f for f in sorted(os.listdir(test_audio_dir))]
file_list = [file.split('.')[0] for file in file_list if file.endswith('.ogg')]

debug = DEBUG_MODE
if len(file_list) == 0 or debug: # Simplified debug check
    debug = True # Ensure debug is true if file_list was initially empty
    test_audio_dir = TRAIN_AUDIO_DIR_DEBUG
    file_list = [f for f in sorted(os.listdir(test_audio_dir))]
    file_list = [file.split('.')[0] for file in file_list if file.endswith('.ogg')]
    if DEBUG_NUM_FILES > 0 : # Allow running on all files in debug if DEBUG_NUM_FILES is 0 or less
        file_list = file_list[DEBUG_START_NUM : DEBUG_START_NUM + DEBUG_NUM_FILES]

print('Debug mode:', debug)
print('Number of test soundscapes:', len(file_list))

min_segment = SAMPLE_RATE * WAV_SEC # Retained as it was explicitly defined
taxonomy_df = pd.read_csv(TAXONOMY_FILE)
class_labels = sorted(taxonomy_df['primary_label'].unique().tolist())
# Ensure NUM_CLASSES matches the actual number of labels
if NUM_CLASSES != len(class_labels):
    print(f"Warning: NUM_CLASSES ({NUM_CLASSES}) does not match number of labels from taxonomy ({len(class_labels)}). Using length from taxonomy.")

mel_spectrogram = AT.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    center=MEL_CENTER,
    f_min=F_MIN,
    f_max=F_MAX,
    pad_mode=MEL_PAD_MODE,
    power=MEL_POWER,
    norm=MEL_NORM,
    n_mels=N_MELS,
    mel_scale=MEL_SCALE,
)

def normalize_std(spec, eps=1e-6):
    mean = torch.mean(spec)
    std = torch.std(spec)
    return torch.where(std == 0, spec-mean, (spec - mean) / (std+eps))

def audio_to_mel(filepath=None):
    waveform, sr = torchaudio.load(filepath, backend="soundfile") # sr should be SAMPLE_RATE
    if sr != SAMPLE_RATE:
        # Resample if necessary, though problem context implies data is prepared
        print(f"Warning: Sample rate of {filepath} is {sr}, not {SAMPLE_RATE}. Resampling not implemented here.")
    len_wav = waveform.shape[1]
    waveform = waveform[0,:].reshape(1, len_wav)
    PREDS = []
    # The loop processes 12 segments of 5 seconds each from a 60-second audio file
    # Each segment is 5 * SAMPLE_RATE frames
    for i in range(12): # Assuming 60s audio, 12 * 5s chunks
        waveform2 = waveform[:, i * SAMPLE_RATE * WAV_SEC : (i * SAMPLE_RATE * WAV_SEC) + (SAMPLE_RATE * WAV_SEC)]
        melspec = mel_spectrogram(waveform2)
        melspec = torch.log(melspec+1e-6)
        melspec = normalize_std(melspec)
        melspec = torch.unsqueeze(melspec, dim=0)

        PREDS.append(melspec)
    return torch.vstack(PREDS)

# Load ONNX models
psd_models = []
psd_model_types = []

for path in PSD_MODEL_PATHS:
    model_type = 'PSD'
    print(f"{path}  - {model_type}")
    
    # Load ONNX model
    ort_session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    
    psd_models.append(ort_session)
    psd_model_types.append(model_type)

psd2_models = []
psd2_model_types = []

for path in PSD2_MODEL_PATHS:
    model_type = 'PSD2'
    print(f"{path}  - {model_type}")
    
    # Load ONNX model
    ort_session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    
    psd2_models.append(ort_session)
    psd2_model_types.append(model_type)

psd3_models = []
psd3_model_types = []

for path in PSD3_MODEL_PATHS:
    model_type = 'PSD3'
    print(f"{path}  - {model_type}")
    
    # Load ONNX model
    ort_session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    
    psd3_models.append(ort_session)
    psd3_model_types.append(model_type)

def prediction_combined(afile):
    global pred_psd, pred_psd2, pred_psd3
    path = test_audio_dir + afile + '.ogg'
    
    sig = audio_to_mel(path)
    
    # Process PSD models
    psd_outputs = None
    for model_idx, (ort_session, model_type) in enumerate(zip(psd_models, psd_model_types)):
        
        # Convert PyTorch tensor to numpy for ONNX
        sig_np = sig.detach().cpu().numpy().astype(np.float32)
        
        # Get input name from ONNX model
        input_name = ort_session.get_inputs()[0].name
        
        # Run ONNX inference
        ort_outputs = ort_session.run(None, {input_name: sig_np})
        p = ort_outputs[0]  # Assuming first output is the predictions
        
        # Apply sigmoid if needed (depends on ONNX model export)
        # p = 1 / (1 + np.exp(-p))  # sigmoid function if needed
        
        if p.ndim > 2:
            p = p.squeeze()
        p = add_amounts_to_top_predictions(p)
        p = apply_power_to_low_ranked_cols(p, top_k=APPLY_POWER_TOP_K, exponent=APPLY_POWER_EXPONENT)
        if psd_outputs is None: 
            psd_outputs = p
        else: 
            psd_outputs += p

    psd_outputs /= len(psd_models)
    
    # Process PSD2 models
    psd2_outputs = None
    for model_idx, (ort_session, model_type) in enumerate(zip(psd2_models, psd2_model_types)):
        
        # Convert PyTorch tensor to numpy for ONNX
        sig_np = sig.detach().cpu().numpy().astype(np.float32)
        
        # Get input name from ONNX model
        input_name = ort_session.get_inputs()[0].name
        
        # Run ONNX inference
        ort_outputs = ort_session.run(None, {input_name: sig_np})
        p = ort_outputs[0]  # Assuming first output is the predictions
        
        # Apply sigmoid if needed (depends on ONNX model export)
        # p = 1 / (1 + np.exp(-p))  # sigmoid function if needed
        
        if p.ndim > 2:
            p = p.squeeze()
        p = add_amounts_to_top_predictions(p)       
        p = apply_power_to_low_ranked_cols(p, top_k=APPLY_POWER_TOP_K, exponent=APPLY_POWER_EXPONENT)
        if psd2_outputs is None: 
            psd2_outputs = p
        else: 
            psd2_outputs += p

    psd2_outputs /= len(psd2_models)
    
    # Process PSD3 models
    psd3_outputs = None
    for model_idx, (ort_session, model_type) in enumerate(zip(psd3_models, psd3_model_types)):
        
        # Convert PyTorch tensor to numpy for ONNX
        sig_np = sig.detach().cpu().numpy().astype(np.float32)
        
        # Get input name from ONNX model
        input_name = ort_session.get_inputs()[0].name
        
        # Run ONNX inference
        ort_outputs = ort_session.run(None, {input_name: sig_np})
        p = ort_outputs[0]  # Assuming first output is the predictions
        
        # Apply sigmoid if needed (depends on ONNX model export)
        # p = 1 / (1 + np.exp(-p))  # sigmoid function if needed
        
        if p.ndim > 2:
            p = p.squeeze()
        p = add_amounts_to_top_predictions(p)
        p = apply_power_to_low_ranked_cols(p, top_k=APPLY_POWER_TOP_K, exponent=APPLY_POWER_EXPONENT)
        if psd3_outputs is None: 
            psd3_outputs = p
        else: 
            psd3_outputs += p

    psd3_outputs /= len(psd3_models)
    
    # Store results for all 3 model types
    chunks = [[] for i in range(12)]
    for i in range(len(chunks)):
        chunk_end_time = (i + 1) * 5
        row_id = afile + '_' + str(chunk_end_time)
        
        # Store PSD results
        pred_psd['row_id'].append(row_id)
        bird_no = 0
        for bird in class_labels:
            pred_psd[bird].append(psd_outputs[i,bird_no])
            bird_no += 1
            
        # Store PSD2 results
        pred_psd2['row_id'].append(row_id)
        bird_no = 0
        for bird in class_labels:
            pred_psd2[bird].append(psd2_outputs[i,bird_no])
            bird_no += 1
            
        # Store PSD3 results
        pred_psd3['row_id'].append(row_id)
        bird_no = 0
        for bird in class_labels:
            pred_psd3[bird].append(psd3_outputs[i,bird_no])
            bird_no += 1
            
    gc.collect()

# Initialize prediction dictionaries
pred_psd = {'row_id': []}
for species_code in class_labels:
    pred_psd[species_code] = []

pred_psd2 = {'row_id': []}
for species_code in class_labels:
    pred_psd2[species_code] = []

pred_psd3 = {'row_id': []}
for species_code in class_labels:
    pred_psd3[species_code] = []

# Process all models in single ThreadPoolExecutor
print("Processing PSD, PSD2, and PSD3 models...")
start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_THREADPOOL) as executor:
    _ = list(executor.map(prediction_combined, tqdm(file_list, desc="Processing audio files")))
end_t = time.time()

if debug == True:
    print("Total processing time:", 700*(end_t - start)/60/DEBUG_NUM_FILES if DEBUG_NUM_FILES > 0 else "DEBUG_NUM_FILES is 0, cannot calculate rate")

# Save and process PSD results
results_psd = pd.DataFrame(pred_psd, columns = ['row_id'] + class_labels)
results_psd.to_csv(SUBMISSION_FILE_PSD, index=False)

# Apply temporal smoothing to PSD results
sub_psd = pd.read_csv(SUBMISSION_FILE_PSD)
cols = sub_psd.columns[1:]
groups = sub_psd['row_id'].str.rsplit('_', n=1).str[0]
groups = groups.values
for group in np.unique(groups):
    current_group_mask = (groups == group)
    predictions = sub_psd.loc[current_group_mask, cols].values
    new_predictions = predictions.copy()
    for i in range(1, predictions.shape[0]-1):
        new_predictions[i] = (predictions[i-1] * 0.2) + (predictions[i] * 0.6) + (predictions[i+1] * 0.2)
    new_predictions[0] = (predictions[0] * 0.8) + (predictions[1] * 0.2)
    new_predictions[-1] = (predictions[-1] * 0.8) + (predictions[-2] * 0.2)
    sub_psd.loc[current_group_mask, cols] = new_predictions
sub_psd.to_csv(SUBMISSION_FILE_PSD, index=False)

# Save and process PSD2 results
results_psd2 = pd.DataFrame(pred_psd2, columns = ['row_id'] + class_labels)
results_psd2.to_csv(SUBMISSION_FILE_PSD2, index=False)

# Apply temporal smoothing to PSD2 results
sub_psd2 = pd.read_csv(SUBMISSION_FILE_PSD2)
cols = sub_psd2.columns[1:]
groups = sub_psd2['row_id'].str.rsplit('_', n=1).str[0]
groups = groups.values
for group in np.unique(groups):
    current_group_mask = (groups == group)
    predictions = sub_psd2.loc[current_group_mask, cols].values
    new_predictions = predictions.copy()
    for i in range(1, predictions.shape[0]-1):
        new_predictions[i] = (predictions[i-1] * 0.2) + (predictions[i] * 0.6) + (predictions[i+1] * 0.2)
    new_predictions[0] = (predictions[0] * 0.8) + (predictions[1] * 0.2)
    new_predictions[-1] = (predictions[-1] * 0.8) + (predictions[-2] * 0.2)
    sub_psd2.loc[current_group_mask, cols] = new_predictions
sub_psd2.to_csv(SUBMISSION_FILE_PSD2, index=False)

# Save and process PSD3 results
results_psd3 = pd.DataFrame(pred_psd3, columns = ['row_id'] + class_labels)
results_psd3.to_csv(SUBMISSION_FILE_PSD3, index=False)

# Apply temporal smoothing to PSD3 results
sub_psd3 = pd.read_csv(SUBMISSION_FILE_PSD3)
cols = sub_psd3.columns[1:]
groups = sub_psd3['row_id'].str.rsplit('_', n=1).str[0]
groups = groups.values
for group in np.unique(groups):
    current_group_mask = (groups == group)
    predictions = sub_psd3.loc[current_group_mask, cols].values
    new_predictions = predictions.copy()
    for i in range(1, predictions.shape[0]-1):
        new_predictions[i] = (predictions[i-1] * 0.2) + (predictions[i] * 0.6) + (predictions[i+1] * 0.2)
    new_predictions[0] = (predictions[0] * 0.8) + (predictions[1] * 0.2)
    new_predictions[-1] = (predictions[-1] * 0.8) + (predictions[-2] * 0.2)
    sub_psd3.loc[current_group_mask, cols] = new_predictions
sub_psd3.to_csv(SUBMISSION_FILE_PSD3, index=False)

# Average the three submissions using rank averaging
print("Averaging PSD, PSD2, and PSD3 submissions...")
sub_psd = pd.read_csv(SUBMISSION_FILE_PSD)
sub_psd2 = pd.read_csv(SUBMISSION_FILE_PSD2)
sub_psd3 = pd.read_csv(SUBMISSION_FILE_PSD3)

# Ensure all dataframes have the same row_id order
sub_psd = sub_psd.sort_values('row_id').reset_index(drop=True)
sub_psd2 = sub_psd2.sort_values('row_id').reset_index(drop=True)
sub_psd3 = sub_psd3.sort_values('row_id').reset_index(drop=True)

# Convert to ranks and average (excluding row_id column)
sub_final = sub_psd.copy()
cols = sub_final.columns[1:]  # All columns except row_id
sub_psd[cols] = sub_psd[cols].rank(axis=0, pct=True)
sub_psd2[cols] = sub_psd2[cols].rank(axis=0, pct=True)
sub_psd3[cols] = sub_psd3[cols].rank(axis=0, pct=True)

# Weighted average of the three models
sub_final[cols] = 0.5*sub_psd[cols] + 0.25*sub_psd2[cols] + 0.25*sub_psd3[cols]

sub_final.to_csv(SUBMISSION_FILE, index=False)
print(f"Final averaged submission saved to {SUBMISSION_FILE}")
print("Used weights: PSD=0.5, PSD2=0.25, PSD3=0.25") 


sub_final.head()

