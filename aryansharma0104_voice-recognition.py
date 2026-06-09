import os

archive_path = '../input/tensorflow-speech-recognition-challenge/train.7z'
output_dir = "/kaggle/working"
# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print(f"Attempting extraction using system 7z utility...")

!7z x {archive_path} -o{output_dir} -mmt=on -y > /dev/null 2>&1

print("\n--- 7z Command Output Above ---")
print("Check the specified output directory for files.")


import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle
from collections import Counter, defaultdict
import seaborn as sns
from tensorflow.keras.metrics import top_k_categorical_accuracy


# ==================== CONFIGURATION ====================
SAMPLE_RATE = 16000
DURATION = 1.0
SAMPLES_PER_AUDIO = int(SAMPLE_RATE * DURATION)

N_MFCC = 40
HOP_LENGTH = 512

# Training parameters - Optimized for large speaker set
BATCH_SIZE = 128  # Larger batch for 2000+ speakers
EPOCHS = 100
LEARNING_RATE = 0.001
MIN_SAMPLES_PER_SPEAKER = 12  # Increased for better quality


# ==================== CRITICAL FIX: SPEAKER ID EXTRACTION ====================

def extract_speaker_id(filename):
    name = filename.replace('.wav', '')
    # Extract only the speaker hash (first part before _nohash)
    speaker_base = name.split('_')[0]
    return speaker_base


# ==================== DATA LOADING WITH VERIFICATION ====================

def load_audio(filepath, sr=SAMPLE_RATE, duration=DURATION):
    """Load and preprocess audio file"""
    try:
        audio, _ = librosa.load(filepath, sr=sr, duration=duration)
        if len(audio) < SAMPLES_PER_AUDIO:
            audio = np.pad(audio, (0, SAMPLES_PER_AUDIO - len(audio)), mode='constant')
        else:
            audio = audio[:SAMPLES_PER_AUDIO]
        return audio
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def extract_mfcc_features(audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """Extract MFCC features with deltas"""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=HOP_LENGTH)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
    return features.T


def load_speaker_dataset(data_path, min_samples=MIN_SAMPLES_PER_SPEAKER):
    """Load dataset with CORRECT speaker grouping"""
    
    print(f"\n{'='*80}")
    print(" "*25 + "LOADING DATASET")
    print(f"{'='*80}")
    
    speaker_files = {}
    total_files = 0
    
    # Get word folders
    word_folders = [f for f in os.listdir(data_path) 
                   if os.path.isdir(os.path.join(data_path, f))
                   and f != '_background_noise_']
    
    print(f"\nScanning {len(word_folders)} word folders...")
    
    # Scan all files
    for word in word_folders:
        folder_path = os.path.join(data_path, word)
        for filename in os.listdir(folder_path):
            if filename.endswith('.wav'):
                # âœ… Extract BASE speaker ID (groups recordings from same person)
                speaker_id = extract_speaker_id(filename)
                filepath = os.path.join(folder_path, filename)
                
                if speaker_id not in speaker_files:
                    speaker_files[speaker_id] = []
                speaker_files[speaker_id].append(filepath)
                total_files += 1
    
    print(f"âœ“ Found {total_files} audio files")
    print(f"âœ“ Found {len(speaker_files)} UNIQUE speakers (before filtering)")
    
    # Show grouping verification
    print(f"\nâœ… SPEAKER GROUPING VERIFICATION:")
    sample_speaker = list(speaker_files.keys())[0]
    sample_files = speaker_files[sample_speaker][:3]
    print(f"  Example speaker: {sample_speaker}")
    print(f"  Their files:")
    for f in sample_files:
        print(f"    - {os.path.basename(f)}")
    
    # Filter speakers
    valid_speakers = {spk: files for spk, files in speaker_files.items() 
                     if len(files) >= min_samples}
    
    removed = len(speaker_files) - len(valid_speakers)
    
    print(f"\nğŸ“Š FILTERING RESULTS:")
    print(f"  Minimum samples required: {min_samples}")
    print(f"  Speakers kept: {len(valid_speakers)}")
    print(f"  Speakers removed: {removed}")
    print(f"  Total training samples: {sum(len(f) for f in valid_speakers.values())}")
    
    # Statistics
    counts = [len(files) for files in valid_speakers.values()]
    print(f"\nğŸ“ˆ SAMPLES PER SPEAKER:")
    print(f"  Min: {min(counts)}")
    print(f"  Max: {max(counts)}")
    print(f"  Mean: {np.mean(counts):.1f}")
    print(f"  Median: {np.median(counts):.0f}")
    
    # Top speakers
    top = sorted(valid_speakers.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    print(f"\nğŸ�† TOP 5 SPEAKERS:")
    for spk, files in top:
        print(f"  {spk}: {len(files)} samples")
    
    print(f"{'='*80}\n")
    
    return valid_speakers


def extract_features_from_dataset(speaker_files):
    """Extract features from all audio files"""
    
    print(f"\n{'='*80}")
    print(" "*25 + "EXTRACTING FEATURES")
    print(f"{'='*80}\n")
    
    all_features = []
    all_labels = []
    total = sum(len(f) for f in speaker_files.values())
    processed = 0
    errors = 0
    
    for speaker_id, filepaths in speaker_files.items():
        for filepath in filepaths:
            processed += 1
            
            if processed % 1000 == 0:
                print(f"Progress: {processed}/{total} ({processed/total*100:.1f}%) - Errors: {errors}")
            
            audio = load_audio(filepath)
            if audio is None:
                errors += 1
                continue
            
            try:
                features = extract_mfcc_features(audio)
                all_features.append(features)
                all_labels.append(speaker_id)
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  Error: {filepath}: {e}")
    
    print(f"\nâœ“ Successfully processed: {len(all_features)}")
    print(f"âœ— Errors: {errors}")
    
    # Convert to arrays
    X = np.array(all_features)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(all_labels)
    y_categorical = keras.utils.to_categorical(y_encoded)
    
    print(f"\n{'='*80}")
    print(" "*28 + "DATASET SUMMARY")
    print(f"{'='*80}")
    print(f"\nFeatures shape: {X.shape}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Time steps: {X.shape[1]}")
    print(f"  Features: {X.shape[2]}")
    print(f"\nNumber of speakers: {len(label_encoder.classes_)}")
    print(f"Labels shape: {y_categorical.shape}")
    
    # âœ… VERIFICATION: Show sample speaker IDs
    print(f"\nâœ… SAMPLE SPEAKER IDs (first 10):")
    for i, spk in enumerate(label_encoder.classes_[:10]):
        print(f"  {i+1}. {spk}")
    
    print(f"{'='*80}\n")
    
    return X, y_categorical, label_encoder


# ==================== IMPROVED MODEL ====================

def build_speaker_model(input_shape, num_speakers):
    """Build improved model for large-scale speaker recognition"""
    from tensorflow.keras import regularizers
    
    model = models.Sequential()
    
    # Block 1
    model.add(layers.Conv1D(128, 3, padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(128, 3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.3))
    
    # Block 2
    model.add(layers.Conv1D(256, 3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(256, 3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.4))
    
    # Block 3
    model.add(layers.Conv1D(512, 3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dropout(0.5))
    
    # Dense layers
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))
    
    # Output
    model.add(layers.Dense(num_speakers, activation='softmax'))
    
    return model


# ==================== TRAINING WITH TOP-K ACCURACY ====================

def train_speaker_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS):
    """Train model with Top-K accuracy tracking (backward compatible)"""
    from tensorflow.keras.metrics import top_k_categorical_accuracy

    # Define callbacks safely
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1
        ),
        # ğŸ§  Older Keras doesnâ€™t support restore_best_weights
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_speaker_model.h5',
            monitor='val_accuracy',  # Safer for compatibility
            save_best_only=True,
            verbose=1
        )
    ]

    # Compile model with backward compatibility for optimizer & metrics
    try:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                lambda y_true, y_pred: top_k_categorical_accuracy(y_true, y_pred, k=5),
                lambda y_true, y_pred: top_k_categorical_accuracy(y_true, y_pred, k=10)
            ]
        )
    except TypeError:
        model.compile(
            optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    print(f"\n{'='*80}")
    print(" "*30 + "TRAINING STARTED")
    print(f"{'='*80}\n")

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # âœ… Reload best weights manually (for older TF versions)
    try:
        model.load_weights('best_speaker_model.h5')
        print("\n[INFO] Best model weights restored from checkpoint.")
    except Exception as e:
        print(f"[WARN] Could not restore best weights: {e}")

    return history



# ==================== VISUALIZATION ====================

def plot_training_history(history):
    """Plot training metrics"""
    keys = history.history.keys()
    acc_key = 'accuracy' if 'accuracy' in keys else 'acc'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in keys else 'val_acc'
    
    has_top5 = 'top_5_accuracy' in keys
    
    if has_top5:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Top-1 Accuracy
        axes[0, 0].plot(history.history[acc_key], label='Train', linewidth=2)
        axes[0, 0].plot(history.history[val_acc_key], label='Val', linewidth=2)
        axes[0, 0].set_title('Top-1 Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Top-5 Accuracy
        axes[0, 1].plot(history.history['top_5_accuracy'], label='Train', linewidth=2)
        axes[0, 1].plot(history.history['val_top_5_accuracy'], label='Val', linewidth=2)
        axes[0, 1].set_title('Top-5 Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loss
        axes[1, 0].plot(history.history['loss'], label='Train', linewidth=2)
        axes[1, 0].plot(history.history['val_loss'], label='Val', linewidth=2)
        axes[1, 0].set_title('Loss', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Overfitting Gap
        axes[1, 1].plot(
            np.array(history.history[acc_key]) - np.array(history.history[val_acc_key]),
            linewidth=2, color='red'
        )
        axes[1, 1].axhline(0, color='gray', linestyle='--')
        axes[1, 1].set_title('Overfitting Gap', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Train - Val Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(history.history[acc_key], label='Train', linewidth=2)
        axes[0].plot(history.history[val_acc_key], label='Val', linewidth=2)
        axes[0].set_title('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history.history['loss'], label='Train', linewidth=2)
        axes[1].plot(history.history['val_loss'], label='Val', linewidth=2)
        axes[1].set_title('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("âœ“ Saved: training_history.png")


# ==================== EVALUATION & ANALYSIS ====================

def evaluate_model(model, X_test, y_test, label_encoder):
    """Comprehensive evaluation"""
    
    print(f"\n{'='*80}")
    print(" "*30 + "EVALUATION")
    print(f"{'='*80}\n")
    
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Top-K accuracy
    print("ğŸ�¯ ACCURACY METRICS:")
    for k in [1, 3, 5, 10, 20]:
        top_k = np.argsort(y_pred_probs, axis=1)[:, -k:]
        acc = np.mean([y_true[i] in top_k[i] for i in range(len(y_true))])
        print(f"  Top-{k:2d}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Confidence analysis
    max_conf = np.max(y_pred_probs, axis=1)
    correct = (y_pred == y_true)
    
    print(f"\nğŸ”� CONFIDENCE:")
    print(f"  All: {np.mean(max_conf):.3f}")
    print(f"  Correct: {np.mean(max_conf[correct]):.3f}")
    print(f"  Incorrect: {np.mean(max_conf[~correct]):.3f}")
    
    # Per-speaker stats
    speaker_acc = {}
    for i in range(len(y_true)):
        spk = label_encoder.inverse_transform([y_true[i]])[0]
        if spk not in speaker_acc:
            speaker_acc[spk] = {'correct': 0, 'total': 0}
        speaker_acc[spk]['total'] += 1
        if y_pred[i] == y_true[i]:
            speaker_acc[spk]['correct'] += 1
    
    accs = [(s, d['correct']/d['total']) for s, d in speaker_acc.items()]
    
    print(f"\nğŸ�† BEST SPEAKERS:")
    for spk, acc in sorted(accs, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {spk}: {acc:.2%}")
    
    print(f"\nâš ï¸� WORST SPEAKERS:")
    for spk, acc in sorted(accs, key=lambda x: x[1])[:5]:
        print(f"  {spk}: {acc:.2%}")
    
    print(f"{'='*80}\n")
    
    return np.mean(correct)


# ==================== PREDICTION ====================

def predict_speaker(model, audio_file, label_encoder, top_k=5):
    """Predict speaker with proper feature extraction"""
    
    print(f"\n{'='*80}")
    print(f"PREDICTING: {os.path.basename(audio_file)}")
    print(f"{'='*80}\n")
    
    # Expected speaker from filename
    expected = extract_speaker_id(os.path.basename(audio_file))
    print(f"Expected speaker: {expected}")
    
    # Check if in training set
    if expected in label_encoder.classes_:
        print(f"âœ“ Speaker IS in training set")
    else:
        print(f"âœ— Speaker NOT in training set - prediction will be wrong!")
    
    # Load and predict
    audio = load_audio(audio_file)
    if audio is None:
        return None, None
    
    features = extract_mfcc_features(audio)
    features = np.expand_dims(features, axis=0)
    
    probs = model.predict(features, verbose=0)[0]
    
    # Top K
    top_idx = np.argsort(probs)[-top_k:][::-1]
    top_spk = label_encoder.inverse_transform(top_idx)
    top_conf = probs[top_idx]
    
    print(f"\nTop {top_k} Predictions:")
    print("-" * 80)
    
    for i, (spk, conf) in enumerate(zip(top_spk, top_conf), 1):
        bar = "â–ˆ" * int(conf * 40) + "â–‘" * (40 - int(conf * 40))
        marker = "âœ“" if spk == expected else " "
        print(f"{marker} {i}. {spk:25s} {conf:6.2%} â”‚{bar}â”‚")
    
    print("-" * 80)
    print(f"\nâœ“ PREDICTED: {top_spk[0]} (confidence: {top_conf[0]:.2%})")
    
    if top_spk[0] == expected:
        print(f"âœ“ CORRECT PREDICTION!")
    else:
        print(f"âœ— WRONG - Expected: {expected}")
    
    print(f"{'='*80}\n")
    
    return top_spk[0], top_conf[0]


# ==================== SAVE/LOAD ====================

def save_model_and_encoder(model, label_encoder, model_path='speaker_model.h5', 
                           encoder_path='label_encoder.pkl'):
    """Save model and encoder"""
    model.save(model_path)
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"\nâœ“ Saved: {model_path}")
    print(f"âœ“ Saved: {encoder_path}")


def load_model_and_encoder(model_path='speaker_model.h5', 
                           encoder_path='label_encoder.pkl'):
    """Load model and encoder"""
    model = keras.models.load_model(model_path)
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"âœ“ Loaded: {model_path}")
    print(f"âœ“ Loaded: {encoder_path}")
    print(f"âœ“ Recognizes {len(label_encoder.classes_)} speakers")
    return model, label_encoder


    print("\n" + "="*80)
    print(" "*20 + "SPEAKER RECOGNITION SYSTEM")
    print("="*80)
    
    DATA_PATH = '/kaggle/working/train/audio/'
# Load
    print("\n[1/6] Loading dataset...")
    speaker_files = load_speaker_dataset(DATA_PATH, MIN_SAMPLES_PER_SPEAKER)


   # Extract features
    print("\n[2/6] Extracting features...")
    X, y, label_encoder = extract_features_from_dataset(speaker_files)


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Split
print("\n[3/6] Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1)
    )
print(f"Train: {len(X_train)}, Val: {len(X_val)}")


    # Build
    print("\n[4/6] Building model...")
    model = build_speaker_model((X.shape[1], X.shape[2]), y.shape[1])
    model.summary()


    # Train
    print("\n[5/6] Training...")
    history = train_speaker_model(model, X_train, y_train, X_val, y_val, EPOCHS)
    


   # Plot
    plot_training_history(history)


    # Evaluate
    print("\n[6/6] Evaluating...")
    acc = evaluate_model(model, X_val, y_val, label_encoder)


    # Save
save_model_and_encoder(model, label_encoder)
    
    print("\n" + "="*80)
    print(" "*25 + "TRAINING COMPLETE!")
    print(f" "*20 + f"Validation Accuracy: {acc:.2%}")
    print("="*80 + "\n")
    
 # model, label_encoder, speaker_files


import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
from glob import glob


# ==================== AUDIO PLAYBACK ====================

def play_audio(audio_path, sr=16000):
    """
    Play audio file in Jupyter/Kaggle notebook
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate (default: 16000)
    
    Returns:
        Audio widget for playback
    """
    if not os.path.exists(audio_path):
        print(f"â�Œ File not found: {audio_path}")
        return None
    
    try:
        # Load audio
        audio, sample_rate = librosa.load(audio_path, sr=sr)
        
        print(f"ğŸ�µ Playing: {os.path.basename(audio_path)}")
        print(f"   Duration: {len(audio)/sample_rate:.2f}s")
        print(f"   Sample rate: {sample_rate} Hz\n")
        
        # Create audio widget
        return ipd.Audio(audio, rate=sample_rate)
    
    except Exception as e:
        print(f"â�Œ Error loading audio: {e}")
        return None


def visualize_audio(audio_path, sr=16000):
    """
    Visualize audio waveform and spectrogram
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
    """
    if not os.path.exists(audio_path):
        print(f"â�Œ File not found: {audio_path}")
        return
    
    try:
        # Load audio
        audio, sample_rate = librosa.load(audio_path, sr=sr)
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # 1. Waveform
        axes[0].plot(np.linspace(0, len(audio)/sample_rate, len(audio)), audio, linewidth=0.5)
        axes[0].set_title(f'Waveform: {os.path.basename(audio_path)}', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        img = librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='hz', ax=axes[1])
        axes[1].set_title('Spectrogram (Frequency over Time)', fontsize=14, fontweight='bold')
        fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
        
        # 3. Mel Spectrogram (what the model sees)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        img2 = librosa.display.specshow(mel_spec_db, sr=sample_rate, x_axis='time', y_axis='mel', ax=axes[2])
        axes[2].set_title('Mel Spectrogram (Model Input)', fontsize=14, fontweight='bold')
        fig.colorbar(img2, ax=axes[2], format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig('audio_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("âœ“ Visualization saved as 'audio_visualization.png'\n")
        
    except Exception as e:
        print(f"â�Œ Error visualizing audio: {e}")


# ==================== FIND SIMILAR SAMPLES ====================

def find_speaker_samples(speaker_id, data_path='/kaggle/working/train/audio/', 
                         max_samples=5, random_sample=True):
    """
    Find audio samples for a specific speaker
    
    Args:
        speaker_id: Speaker ID to find (e.g., 'ec201020')
        data_path: Path to audio data directory
        max_samples: Maximum number of samples to return
        random_sample: If True, randomly sample files; if False, return first N
    
    Returns:
        List of file paths
    """
    print(f"\nğŸ”� Searching for samples of speaker: {speaker_id}")
    
    # Search all word folders
    all_files = []
    word_folders = [f for f in os.listdir(data_path) 
                   if os.path.isdir(os.path.join(data_path, f))
                   and f != '_background_noise_']
    
    for word in word_folders:
        folder_path = os.path.join(data_path, word)
        pattern = os.path.join(folder_path, f"{speaker_id}_*.wav")
        files = glob(pattern)
        all_files.extend(files)
    
    if not all_files:
        print(f"â�Œ No samples found for speaker: {speaker_id}")
        return []
    
    print(f"âœ“ Found {len(all_files)} samples")
    
    # Sample files
    if random_sample and len(all_files) > max_samples:
        selected = np.random.choice(all_files, max_samples, replace=False).tolist()
    else:
        selected = all_files[:max_samples]
    
    print(f"âœ“ Selected {len(selected)} samples:\n")
    for i, f in enumerate(selected, 1):
        word = os.path.basename(os.path.dirname(f))
        filename = os.path.basename(f)
        print(f"   {i}. [{word}] {filename}")
    
    return selected


# ==================== COMPARE AUDIO SAMPLES ====================

def compare_audio_samples(test_audio, predicted_speaker_id, 
                         data_path='/kaggle/working/train/audio/',
                         num_comparisons=3, sr=16000):
    """
    Compare test audio with samples from predicted speaker
    
    Args:
        test_audio: Path to test audio file
        predicted_speaker_id: Predicted speaker ID
        data_path: Path to training data
        num_comparisons: Number of comparison samples to show
        sr: Sample rate
    """
    print("\n" + "="*80)
    print(" "*20 + "AUDIO COMPARISON")
    print("="*80 + "\n")
    
    # Play test audio
    print("ğŸ�§ TEST AUDIO:")
    print(f"   File: {os.path.basename(test_audio)}")
    test_widget = play_audio(test_audio, sr)
    if test_widget:
        ipd.display(test_widget)
    print()
    
    # Find samples of predicted speaker
    comparison_files = find_speaker_samples(
        predicted_speaker_id, 
        data_path, 
        max_samples=num_comparisons
    )
    
    if not comparison_files:
        print("\nâš ï¸�  No comparison samples available")
        return
    
    # Play comparison samples
    print(f"\nğŸ�§ PREDICTED SPEAKER SAMPLES (Speaker: {predicted_speaker_id}):")
    print("   Listen to these to verify if they sound similar:\n")
    
    for i, audio_file in enumerate(comparison_files, 1):
        word = os.path.basename(os.path.dirname(audio_file))
        filename = os.path.basename(audio_file)
        
        print(f"   {i}. [{word}] {filename}")
        widget = play_audio(audio_file, sr)
        if widget:
            ipd.display(widget)
        print()
    
    print("="*80 + "\n")


# ==================== SIDE-BY-SIDE COMPARISON ====================

def compare_spectrograms(audio1_path, audio2_path, 
                        label1="Test Audio", label2="Predicted Speaker Sample",
                        sr=16000):
    """
    Compare spectrograms of two audio files side-by-side
    
    Args:
        audio1_path: Path to first audio file
        audio2_path: Path to second audio file
        label1: Label for first audio
        label2: Label for second audio
        sr: Sample rate
    """
    print("\n" + "="*80)
    print(" "*20 + "SPECTROGRAM COMPARISON")
    print("="*80 + "\n")
    
    try:
        # Load both audio files
        audio1, sr1 = librosa.load(audio1_path, sr=sr)
        audio2, sr2 = librosa.load(audio2_path, sr=sr)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Audio 1 - Waveform
        axes[0, 0].plot(audio1, linewidth=0.5, color='blue')
        axes[0, 0].set_title(f'{label1} - Waveform', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Audio 1 - Mel Spectrogram
        mel1 = librosa.feature.melspectrogram(y=audio1, sr=sr, n_mels=128)
        mel1_db = librosa.power_to_db(mel1, ref=np.max)
        img1 = librosa.display.specshow(mel1_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[0, 1])
        axes[0, 1].set_title(f'{label1} - Mel Spectrogram', fontsize=12, fontweight='bold')
        fig.colorbar(img1, ax=axes[0, 1], format='%+2.0f dB')
        
        # Audio 2 - Waveform
        axes[1, 0].plot(audio2, linewidth=0.5, color='green')
        axes[1, 0].set_title(f'{label2} - Waveform', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Sample')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Audio 2 - Mel Spectrogram
        mel2 = librosa.feature.melspectrogram(y=audio2, sr=sr, n_mels=128)
        mel2_db = librosa.power_to_db(mel2, ref=np.max)
        img2 = librosa.display.specshow(mel2_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[1, 1])
        axes[1, 1].set_title(f'{label2} - Mel Spectrogram', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Time (s)')
        fig.colorbar(img2, ax=axes[1, 1], format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig('spectrogram_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("âœ“ Comparison saved as 'spectrogram_comparison.png'\n")
        print("ğŸ“Š Visual Comparison Tips:")
        print("   â€¢ Similar patterns in spectrograms suggest same speaker")
        print("   â€¢ Look for similar formant structure (bright horizontal bands)")
        print("   â€¢ Check pitch patterns and intensity distribution")
        print()
        
    except Exception as e:
        print(f"â�Œ Error comparing spectrograms: {e}")


# ==================== ALL-IN-ONE FUNCTION ====================

def predict_and_listen(test_audio_path, model, label_encoder,
                      data_path='/kaggle/working/train/audio/',
                      num_comparisons=3, show_spectrograms=True):
    """
    Complete workflow: Predict, play audio, and compare with samples
    
    Args:
        test_audio_path: Path to test audio file
        model: Trained model
        label_encoder: Label encoder
        data_path: Path to training data
        num_comparisons: Number of comparison samples
        show_spectrograms: Whether to show spectrogram comparison
    """
    # Import prediction function
    from prediction_script import extract_mfcc_features, load_audio as load_audio_features
    
    print("\n" + "ğŸ�µ "*30)
    print("PREDICT AND LISTEN - COMPLETE WORKFLOW")
    print("ğŸ�µ "*30 + "\n")
    
    # 1. Make prediction
    print("="*80)
    print("STEP 1: MAKING PREDICTION")
    print("="*80 + "\n")
    
    audio = load_audio_features(test_audio_path)
    if audio is None:
        return
    
    features = extract_mfcc_features(audio)
    features_batch = np.expand_dims(features, axis=0)
    predictions = model.predict(features_batch, verbose=0)[0]
    
    # Get top 5 predictions
    top_indices = np.argsort(predictions)[-5:][::-1]
    top_speakers = label_encoder.inverse_transform(top_indices)
    top_confidences = predictions[top_indices]
    
    print(f"ğŸ�¯ Top 5 Predictions:")
    for i, (speaker, conf) in enumerate(zip(top_speakers, top_confidences), 1):
        bar = "â–ˆ" * int(conf * 30) + "â–‘" * (30 - int(conf * 30))
        print(f"   {i}. {speaker:15s} {conf:6.2%} â”‚{bar}â”‚")
    
    predicted_speaker = top_speakers[0]
    confidence = top_confidences[0]
    
    print(f"\nâœ“ Predicted Speaker: {predicted_speaker} ({confidence:.2%} confidence)\n")
    
    # 2. Play test audio
    print("="*80)
    print("STEP 2: LISTEN TO TEST AUDIO")
    print("="*80 + "\n")
    
    print("ğŸ�§ YOUR TEST AUDIO:")
    test_widget = play_audio(test_audio_path)
    if test_widget:
        ipd.display(test_widget)
    
    # 3. Visualize test audio
    print("\nğŸ“Š VISUALIZING TEST AUDIO:")
    visualize_audio(test_audio_path)
    
    # 4. Compare with predicted speaker samples
    print("\n" + "="*80)
    print("STEP 3: COMPARE WITH PREDICTED SPEAKER SAMPLES")
    print("="*80 + "\n")
    
    compare_audio_samples(
        test_audio_path, 
        predicted_speaker, 
        data_path, 
        num_comparisons
    )
    
    # 5. Spectrogram comparison (if available)
    if show_spectrograms:
        comparison_files = find_speaker_samples(predicted_speaker, data_path, max_samples=1)
        if comparison_files:
            print("\n" + "="*80)
            print("STEP 4: SPECTROGRAM COMPARISON")
            print("="*80 + "\n")
            
            compare_spectrograms(
                test_audio_path,
                comparison_files[0],
                label1="Your Test Audio",
                label2=f"Speaker {predicted_speaker} Sample"
            )
    
    print("\n" + "="*80)
    print("âœ“ ANALYSIS COMPLETE!")
    print("="*80 + "\n")
    
    print("ğŸ’¡ Next Steps:")
    print("   â€¢ Listen to both the test audio and comparison samples")
    print("   â€¢ Do they sound like the same person?")
    print("   â€¢ Check the spectrograms - similar patterns = likely same speaker")
    print("   â€¢ If prediction seems wrong, the speaker might not be in training set")
    print()
    
    return predicted_speaker, confidence, top_speakers, top_confidences


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" "*20 + "AUDIO PLAYBACK TOOL - USAGE GUIDE")
    print("="*80 + "\n")
    
    print("ğŸ“– AVAILABLE FUNCTIONS:\n")
    
    print("1ï¸�âƒ£  PLAY AUDIO:")
    print("    >>> audio = play_audio('/path/to/audio.wav')")
    print("    >>> ipd.display(audio)")
    print()
    
    print("2ï¸�âƒ£  VISUALIZE AUDIO:")
    print("    >>> visualize_audio('/path/to/audio.wav')")
    print()
    
    print("3ï¸�âƒ£  FIND SPEAKER SAMPLES:")
    print("    >>> samples = find_speaker_samples('ec201020', num_samples=5)")
    print()
    
    print("4ï¸�âƒ£  COMPARE AUDIO:")
    print("    >>> compare_audio_samples(")
    print("            test_audio='/path/to/test.wav',")
    print("            predicted_speaker_id='ec201020',")
    print("            num_comparisons=3")
    print("        )")
    print()
    
    print("5ï¸�âƒ£  COMPARE SPECTROGRAMS:")
    print("    >>> compare_spectrograms(")
    print("            audio1_path='/path/to/test.wav',")
    print("            audio2_path='/path/to/reference.wav'")
    print("        )")
    print()
    
    print("6ï¸�âƒ£  ALL-IN-ONE (RECOMMENDED):")
    print("    >>> predict_and_listen(")
    print("            test_audio_path='/path/to/test.wav',")
    print("            model=model,")
    print("            label_encoder=label_encoder,")
    print("            num_comparisons=3")
    print("        )")
    print()
    
    print("="*80 + "\n")


ls '/kaggle/working/train/audio/dog' | head -n 10


results = predict_speaker(model, '/kaggle/working/train/audio/dog/00f0204f_nohash_0.wav', label_encoder)


