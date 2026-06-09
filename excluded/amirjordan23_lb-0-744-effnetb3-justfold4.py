import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import time
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models, transforms  
import timm

class InferenceConfig:
    sampling_rate = 32000
    num_classes = 206
    n_mels = 64
    fmin = 40
    fmax = 15000
    n_fft = 1024
    hop_length = 320
    segment_duration = 5  # seconds for each segment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_soundscapes = "/kaggle/input/birdclef-2025/train_soundscapes"
    test_soundscapes = "/kaggle/input/birdclef-2025/test_soundscapes"
    pretrained_model_path = "/kaggle/input/tf-efficientnet/pytorch/tf-efficientnet-b3/1/tf_efficientnet_b3_aa-84b4657e.pth"
    batch_size = 16
    max_dev_files = 2
    time_limit = 540
    
def load_audio_file(file_path, config):
    try:
        y, _ = librosa.load(file_path, sr=config.sampling_rate)
        return y
    except Exception as e:
        print(f"Error loading audio file {file_path}: {str(e)}")
        return None

def get_audio_segments(audio, config, num_segments=12):
    segments = []
    segment_length = config.segment_duration * config.sampling_rate
    
    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        
        if end_idx <= len(audio):
            segment = audio[start_idx:end_idx]
        else:
            segment = np.zeros(segment_length)
            segment[:len(audio)-start_idx] = audio[start_idx:]
        
        segment = np.concatenate([segment, segment])
        segments.append(segment)
    
    return segments

def get_model(config):
    model = models.efficientnet_b3(weights=None)
    
    if os.path.exists(config.pretrained_model_path):
        print(f"Loading pre-trained weights from: {config.pretrained_model_path}")
        pretrained_dict = torch.load(config.pretrained_model_path, map_location=config.device)
        
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"Successfully loaded {len(pretrained_dict)} layers from pre-trained model")
    else:
        print(f"Warning: Pre-trained model not found at {config.pretrained_model_path}")
        print("Initializing model with random weights")
    
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, config.num_classes)
    return model

def audio_to_melspec_batch(audio_batch, config):
    melspec_batch = []
    
    for audio in audio_batch:
        melspec = librosa.feature.melspectrogram(
            y=audio,
            sr=config.sampling_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            fmin=config.fmin,
            fmax=config.fmax
        )
        melspec_db = librosa.power_to_db(melspec, ref=np.max)
        melspec_3channel = np.stack([melspec_db, melspec_db, melspec_db])
        melspec_batch.append(melspec_3channel)
    
    melspec_tensor = torch.FloatTensor(np.array(melspec_batch))
    transform = transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
    melspec_tensor = torch.stack([transform(m) for m in melspec_tensor])
    
    return melspec_tensor

def create_submission(model_path, config, class_names):
    start_time = time.time()
    
    test_dir = config.test_soundscapes
    train_dir = config.train_soundscapes
    
    is_submission_mode = any(f.endswith('.ogg') for f in os.listdir(test_dir)) if os.path.exists(test_dir) else False
    
    audio_dir = test_dir if is_submission_mode else train_dir
    
    print(f"Running in {'submission' if is_submission_mode else 'development'} mode")
    print(f"Using audio directory: {audio_dir}")
    print(f"Using device: {config.device}")
    
    model = get_model(config)
    
    print(f"Loading fine-tuned model from: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=config.device)
        
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
            
        print("Successfully loaded fine-tuned model weights")
    except Exception as e:
        print(f"Error loading model weights: {str(e)}")
        raise
        
    model = model.to(config.device)
    model.eval()

    predictions = []

    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.ogg')])
    
    if not is_submission_mode:
        audio_files = audio_files[:config.max_dev_files]
        print(f"Development mode: Using only first {config.max_dev_files} files for testing")

    for audio_file in tqdm(audio_files):
        elapsed_time = time.time() - start_time
        if is_submission_mode and elapsed_time > config.time_limit:
            print(f"WARNING: Approaching time limit ({elapsed_time:.1f}s/{config.time_limit}s). Stopping processing.")
            break
            
        file_path = os.path.join(audio_dir, audio_file)
        base_name = os.path.splitext(audio_file)[0]
        
        try:
            audio = load_audio_file(file_path, config)
            if audio is None:
                continue
                
            segments = get_audio_segments(audio, config)
            
            for batch_start in range(0, len(segments), config.batch_size):
                batch_end = min(batch_start + config.batch_size, len(segments))
                batch_segments = segments[batch_start:batch_end]
                batch_indices = list(range(batch_start, batch_end))
                
                batch_melspec = audio_to_melspec_batch(batch_segments, config)
                batch_melspec = batch_melspec.to(config.device)
                
                with torch.no_grad():
                    batch_outputs = model(batch_melspec)
                    batch_probs = torch.sigmoid(batch_outputs).cpu().numpy()
                
                for i, segment_idx in enumerate(batch_indices):
                    probs = batch_probs[i]
                    
                    formatted_probs = [f"{prob:.10f}" for prob in probs]
                    
                    row_id = f"{base_name}_{(segment_idx + 1) * 5}"
                    pred_dict = {'row_id': row_id}
                    pred_dict.update({class_name: formatted_prob for class_name, formatted_prob in zip(class_names, formatted_probs)})
                    predictions.append(pred_dict)
                    
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
            continue

    submission_df = pd.DataFrame(predictions)
    
    if 'row_id' not in submission_df.columns:
        print("WARNING: No predictions were generated!")
        submission_df = pd.DataFrame(columns=['row_id'] + class_names)
    else:
        submission_df = submission_df[['row_id'] + class_names]
    
    for col in class_names:
        if col in submission_df.columns:
            submission_df[col] = submission_df[col].astype(float)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Generated {len(submission_df)} predictions")
    
    return submission_df


if __name__ == "__main__":
    config = InferenceConfig()
    model_path = "/kaggle/input/efficientnet-b3-fold4/best_fold4.pth"  
    
    train_df = pd.read_csv("/kaggle/input/birdclef-2025/train.csv")
    class_names = sorted(train_df['primary_label'].unique())
    
    submission_df = create_submission(model_path, config, class_names)
    
    submission_df.to_csv("/kaggle/working/submission.csv", index=False)
    print("Submission saved to /kaggle/working/submission.csv")

