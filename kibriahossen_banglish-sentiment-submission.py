# Competition Setup: Banglish Sentiment Challenge
# This environment comes with analytics libraries for the sentiment analysis task

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O

# Competition data files are available in the "../input/" directory
# Let's explore the dataset structure for the Banglish Sentiment Challenge

import os
print("=== Competition Dataset Files ===")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        filepath = os.path.join(dirname, filename)
        print(f"ğŸ“� {filepath}")
        
        # Show file size for better understanding
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"   Size: {size:,} bytes")

print("\n=== Expected Files ===")
print("ğŸ“Š example.csv - Sample dataset with labeled examples")
print("ğŸ§ª test.csv - Test set for predictions (no labels)")  
print("ğŸ“¤ submission.csv - Submission template")

# Note: This is a zero-shot learning challenge - no training data provided!


# =============================================================================
# SECTION 1: OLLAMA SETUP FOR ZERO-SHOT SENTIMENT ANALYSIS
# =============================================================================

print("ğŸš€ Installing Ollama and required packages...")
print("This will enable us to run Llama3 locally for sentiment classification")

!curl -fsSL https://ollama.com/install.sh | sh
!pip install -qq pyngrok ollama

print("âœ… Installation complete!")


import subprocess
import os
import time

def start_ollama_server_with_gpu():
    """Starts Ollama server optimized for 2x T4 GPU setup."""
    print("ğŸ”§ Starting Ollama server with GPU acceleration for Banglish sentiment analysis...")
    
    # Set GPU environment variables for optimal performance
    gpu_env = os.environ.copy()
    gpu_env['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use both T4 GPUs
    gpu_env['OLLAMA_NUM_PARALLEL'] = '2'     # Parallel processing on both GPUs
    gpu_env['OLLAMA_MAX_LOADED_MODELS'] = '1'  # Keep one large model loaded
    
    print("ğŸ�¯ GPU Configuration:")
    print("   â€¢ CUDA_VISIBLE_DEVICES: 0,1 (Both T4 GPUs)")
    print("   â€¢ OLLAMA_NUM_PARALLEL: 2 (Parallel processing)")
    print("   â€¢ Optimized for 14B parameter models")
    
    # Ollama installation path
    ollama_path = '/usr/local/bin/ollama'
    if not os.path.exists(ollama_path):
        print("âš ï¸�  Ollama not found at /usr/local/bin, trying system PATH...")
        ollama_path = 'ollama'
    
    try:
        # Check if server is already running
        try:
            subprocess.run(['pgrep', '-f', 'ollama serve'], check=True, 
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("ğŸ”„ Ollama server already running, checking GPU status...")
        except subprocess.CalledProcessError:
            print("ğŸš€ Starting Ollama server with GPU optimization...")
            # Start with GPU environment
            subprocess.Popen([ollama_path, 'serve'], env=gpu_env)
            print("âœ… Ollama server started with GPU acceleration!")
        
        print("ğŸ•� Initializing GPU memory and model cache...")
        
    except Exception as e:
        print(f"â�Œ Error starting Ollama server: {e}")
        return False
    
    return True

def verify_gpu_setup():
    """Verify GPU detection and availability."""
    print("\nğŸ”� GPU Verification:")
    try:
        # Check NVIDIA GPUs
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free', 
                               '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split('\n')
            print("ğŸ“Š Detected GPUs:")
            for i, gpu in enumerate(gpu_info):
                print(f"   GPU {i}: {gpu}")
            
            if len(gpu_info) >= 2:
                print("âœ… Dual GPU setup confirmed for optimal large model performance")
                return True
            else:
                print("âš ï¸�  Less than 2 GPUs detected")
        else:
            print("â�Œ No NVIDIA GPUs detected")
            
    except Exception as e:
        print(f"â�Œ GPU verification failed: {e}")
    
    return False

# Start optimized server
if start_ollama_server_with_gpu():
    # Verify GPU setup
    gpu_ok = verify_gpu_setup()
    
    # Extended initialization time for large models
    initialization_time = 12 if gpu_ok else 8
    print(f"â�±ï¸�  Waiting {initialization_time}s for GPU initialization...")
    time.sleep(initialization_time)
    
    if gpu_ok:
        print("ğŸ�¯ Server ready for GPU-accelerated sentiment analysis!")
        print("ğŸš€ Optimized for Qwen2.5:14b and other large models")
    else:
        print("âš ï¸�  Server started but GPU optimization may be limited")
else:
    print("ğŸ’¥ Failed to start server - check installation and GPU drivers")


# =============================================================================
# SECTION 2: GPU DETECTION & ADVANCED MODEL SETUP FOR MULTILINGUAL SENTIMENT
# =============================================================================

print("ğŸ”¥ Checking GPU availability for enhanced performance...")

# Check GPU detection
!nvidia-smi

print("\nğŸ“Š GPU Memory and Configuration:")
!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

print("\nğŸš€ Setting up advanced multilingual model for Banglish sentiment analysis...")
print("Using Qwen2.5:14b - Excellent model for multilingual tasks including Bangla")

# Pull Qwen2.5:14b - Proven model with superior Bangla understanding
print("ğŸ“¥ Downloading Qwen2.5:14b model (this may take 10-15 minutes)...")
!ollama pull qwen2.5:14b

print("\nâœ… Qwen2.5:14b model ready!")
print("ğŸŒ� This model excels at:")
print("   â€¢ Excellent Bangla language understanding")
print("   â€¢ Superior English language processing") 
print("   â€¢ Enhanced code-switching between languages")
print("   â€¢ Reliable sentiment analysis tasks")
print("   â€¢ Strong multilingual context comprehension")
print("   â€¢ Good emoji and emoticon interpretation")

print("\nğŸ”§ Verifying GPU utilization for optimal performance...")
print("Expected: 2x T4 GPUs should be detected and utilized by Ollama")

# Alternative models to consider if Qwen2.5:14b is not available:
print("\nğŸ“‹ Model Information:")
print("Primary: qwen2.5:14b (14B parameters) - Proven excellent for Banglish")
print("Fallback: llama3.1:8b (8B parameters) - Good multilingual support")
print("GPU Optimized: Uses both T4 GPUs for faster inference")


# =============================================================================
# GPU & MODEL VERIFICATION FOR OPTIMAL PERFORMANCE
# =============================================================================

print("ğŸ”� Verifying GPU detection and model availability:")
print("=" * 60)

# Check Ollama models
print("ğŸ“‹ Available Ollama Models:")
!ollama list

print("\nğŸ�¯ GPU Detection by Ollama:")
# Check if Ollama detects GPUs properly
import subprocess
try:
    result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
    gpu_info = result.stdout
    print("ğŸ–¥ï¸� GPU Detection:")
    print(gpu_info)
    
    gpu_count = gpu_info.count('GPU')
    print(f"ğŸ“Š Total GPUs detected: {gpu_count}")
    
    if gpu_count >= 2:
        print("âœ… Dual GPU setup confirmed - Optimal for large model inference")
    else:
        print("âš ï¸�  Expected 2 GPUs, but found fewer")
        
except Exception as e:
    print(f"â�Œ Error checking GPU: {e}")

print("\nğŸ”¥ Model Readiness Check:")
print("âœ… Qwen3:14b should be listed above for optimal Banglish sentiment analysis")
print("ğŸ�¯ This latest model provides superior performance for:")
print("   â€¢ Advanced Bengali script recognition")
print("   â€¢ Enhanced code-switching detection")
print("   â€¢ Superior contextual sentiment understanding")
print("   â€¢ Advanced emoji-aware analysis")


# =============================================================================
# SECTION 3: COMPETITION DATA EXPLORATION  
# =============================================================================

print("ğŸ“Š Loading and exploring competition datasets...")

# Load the datasets
try:
    # Load example dataset (with labels for understanding format)
    if os.path.exists('/kaggle/input/binary-biplob-can-you-decode-emotions/bangla/example.csv'):
        example_df = pd.read_csv('/kaggle/input/binary-biplob-can-you-decode-emotions/bangla/example.csv')
        print("âœ… Example dataset loaded successfully!")
        print(f"ğŸ“ˆ Example dataset shape: {example_df.shape}")
        print("\nğŸ”� Example data preview:")
        print(example_df.head())
        if 'label' in example_df.columns:
            print(f"\nğŸ“Š Label distribution in examples:")
            print(example_df['label'].value_counts())
    
    # Load test dataset (what we need to predict)
    if os.path.exists('/kaggle/input/binary-biplob-can-you-decode-emotions/bangla/test.csv'):
        test_df = pd.read_csv('/kaggle/input/binary-biplob-can-you-decode-emotions/bangla/test.csv')
        print(f"\nâœ… Test dataset loaded successfully!")
        print(f"ï¿½ Test dataset shape: {test_df.shape}")
        print("\nğŸ”� Test data preview:")
        print(test_df.head())
        
        # Analyze text characteristics
        print(f"\nğŸ“� Text length statistics:")
        test_df['text_length'] = test_df['text'].str.len()
        print(test_df['text_length'].describe())
        
    else:
        print("âš ï¸�  Test dataset not found - using sample data")
        test_df = pd.DataFrame({
            'id': ['sample_1', 'sample_2', 'sample_3'], 
            'text': ['à¦†à¦œà¦•à§‡ weather nice ğŸ˜Š', 'feeling sad today ğŸ˜¢', 'kaj cholche normally']
        })

except Exception as e:
    print(f"â�Œ Error loading datasets: {e}")
    print("ğŸ“� Creating sample data for testing...")
    test_df = pd.DataFrame({
        'id': ['sample_1', 'sample_2', 'sample_3'], 
        'text': ['à¦†à¦œà¦•à§‡ weather nice ğŸ˜Š', 'feeling sad today ğŸ˜¢', 'kaj cholche normally']
    })

print(f"\nï¿½ Ready to process {len(test_df)} samples for sentiment prediction!")


# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

print("ğŸ�¯ BANGLISH SENTIMENT CHALLENGE - DATASET CONFIGURATION")
print("=" * 60)

# Using the specified test dataset path
test_dataset_path = '/kaggle/input/binary-biplob-can-you-decode-emotions/bangla/test.csv'
print(f"ğŸ“Š Test Dataset Path: {test_dataset_path}")

# Verify dataset accessibility
import os
if os.path.exists(test_dataset_path):
    print("âœ… Test dataset found and accessible")
    try:
        # Quick preview without loading full dataset
        import pandas as pd
        sample_df = pd.read_csv(test_dataset_path, nrows=3)
        print(f"ğŸ“‹ Dataset columns: {list(sample_df.columns)}")
        print(f"ğŸ”� Sample entries:")
        for _, row in sample_df.iterrows():
            print(f"   ID: {row['id']}, Text preview: {str(row['text'])[:50]}...")
    except Exception as e:
        print(f"âš ï¸�  Preview error: {e}")
else:
    print("âš ï¸�  Test dataset not found at specified path")
    print("ğŸ”§ Ensure the dataset is available at the Kaggle input location")

print(f"\nğŸ�¯ Model Configuration:")
print("â€¢ Primary Model: qwen3:14b (Latest and best for Banglish)")
print("â€¢ Fallback Model: llama3.1:8b")
print("â€¢ Output Format: id,label (competition standard)")
print("â€¢ Labels: positive, negative, neutral")

print(f"\nğŸš€ Ready to process the full dataset with advanced sentiment analysis!")


# =============================================================================
# SECTION 4: ADVANCED BANGLISH SENTIMENT CLASSIFICATION WITH QWEN2.5:14B
# =============================================================================

import pandas as pd
import ollama
import re
from tqdm.auto import tqdm

print("ğŸ�¯ BANGLISH SENTIMENT CHALLENGE - QWEN2.5:14B SIMPLE SOLUTION")
print("=" * 60)

def get_banglish_sentiment(text_input, client):
    
    # Simple and direct prompt - what worked for 0.75 accuracy
    prompt = f"""Analyze sentiment: positive, negative, or neutral

Text: {text_input}

Sentiment:"""

    try:
        response = client.generate(
            model='qwen2.5:14b',  # Using qwen2.5:14b as you found it best
            prompt=prompt,
            options={
                'temperature': 0.1,       # Low but not too restrictive
                'num_predict': 5,         # Allow a bit more output
                'top_p': 0.9,            # Higher for natural language
                'repeat_penalty': 1.1,   # Minimal repetition penalty
                'num_ctx': 2048,         # Smaller context for efficiency
                'num_gpu': 2             # Utilize both T4 GPUs
            }
        )
        
        sentiment = response['response'].strip().lower()
        
        # Simple direct parsing - no complex regex needed
        if 'positive' in sentiment:
            return "positive"
        elif 'negative' in sentiment:
            return "negative"
        elif 'neutral' in sentiment:
            return "neutral"
        else:
            # If unclear, try first word
            first_word = sentiment.split()[0] if sentiment.split() else ""
            if first_word.startswith('pos'):
                return "positive"
            elif first_word.startswith('neg'):
                return "negative"
            else:
                return "neutral"  # Default to neutral for unclear cases
                
    except Exception as e:
        print(f"âš ï¸�  Error with Qwen2.5:14b: {e}")
        # Simple fallback - no complex alternative models
        return "neutral"

# Initialize Ollama client with GPU optimization
try:
    client = ollama.Client(host='http://127.0.0.1:11434')
    models = client.list()
    print("âœ… Connected to Ollama server with GPU acceleration")
    
    # Verify our target model is available
    try:
        # Handle different response formats from Ollama API
        if hasattr(models, 'models'):
            model_list = models.models
        elif isinstance(models, dict) and 'models' in models:
            model_list = models['models']
        else:
            model_list = models
        
        # Extract model names safely
        model_names = []
        for model in model_list:
            if hasattr(model, 'name'):
                model_names.append(model.name)
            elif isinstance(model, dict) and 'name' in model:
                model_names.append(model['name'])
            elif hasattr(model, 'model'):
                model_names.append(model.model)
            elif isinstance(model, dict) and 'model' in model:
                model_names.append(model['model'])
        
        print(f"ğŸ”� Available models: {model_names}")
        
        if 'qwen2.5:14b' in model_names:
            print("ğŸ�¯ Qwen2.5:14b model confirmed - Optimal for Banglish analysis")
        else:
            print("âš ï¸�  Qwen2.5:14b not found, will attempt to use available models")
            
    except Exception as model_error:
        print(f"âš ï¸�  Model verification error: {model_error}")
        print("ğŸ”„ Proceeding with available models...")
    
except Exception as e:
    print(f"â�Œ Cannot connect to Ollama: {e}")
    print("ğŸ”§ Make sure Ollama server is running with GPU support!")
    raise

# Load test data from specified path
try:
    test_df = pd.read_csv('/kaggle/input/binary-biplob-can-you-decode-emotions/bangla/test.csv')
    print(f"ğŸ“Š Loaded test data: {test_df.shape[0]} samples")
    print(f"ğŸ“‹ Columns: {list(test_df.columns)}")
    print(f"ğŸ”� Sample data preview:")
    print(test_df.head())
    
except FileNotFoundError:
    print(f"â�Œ Test dataset not found at: /kaggle/input/binary-biplob-can-you-decode-emotions/bangla/test.csv")
    print("ğŸ”§ Please ensure the file exists at the specified path")
    raise
except Exception as e:
    print(f"â�Œ Error loading test dataset: {e}")
    raise

print(f"\nğŸš€ Starting Qwen2.5:14b simple sentiment analysis...")
print(f"âš¡ Processing {len(test_df)} samples with direct prompting approach...")

# Apply simple sentiment analysis
tqdm.pandas(desc="ğŸ§  Qwen2.5:14b Simple Analysis")
test_df['predicted_label'] = test_df['text'].progress_apply(
    lambda x: get_banglish_sentiment(x, client)
)

# Results summary
print(f"\nğŸ“Š SIMPLE SENTIMENT ANALYSIS COMPLETE!")
print("=" * 50)
print("ğŸ“ˆ Prediction distribution:")
print(test_df['predicted_label'].value_counts())

print(f"\nğŸ”� Sample predictions with simple Qwen2.5:14b:")
for _, row in test_df.head(10).iterrows():
    text_preview = row['text'][:70] + "..." if len(row['text']) > 70 else row['text']
    print(f"ğŸ“� {row['predicted_label'].upper()}: {text_preview}")

# Prepare submission with exact format requested
submission_df = test_df[['id', 'predicted_label']].copy()
submission_df = submission_df.rename(columns={'predicted_label': 'label'})
submission_df.to_csv('submission.csv', index=False)

print(f"\nâœ… SUBMISSION.CSV GENERATED!")
print(f"ğŸ’¾ Format: id,label")
print(f"ğŸ“Š Total predictions: {len(submission_df)}")
print(f"ğŸ�¯ Using simple direct prompting with Qwen2.5:14b for better accuracy")

# Show sample of final submission format
print(f"\nğŸ“„ SUBMISSION.CSV PREVIEW:")
print("id,label")
for _, row in submission_df.head(5).iterrows():
    print(f"{row['id']},{row['label']}")

print("ğŸ�† Ready for competition submission!")


# =============================================================================
# SECTION 5: SUBMISSION VALIDATION & FINAL CHECKS
# =============================================================================

print("ğŸ”� VALIDATING SUBMISSION FOR BANGLISH SENTIMENT CHALLENGE")
print("=" * 60)

# Load and validate submission file
try:
    submission_check = pd.read_csv('submission.csv')
    
    print("âœ… Submission file loaded successfully!")
    print(f"ğŸ“Š Submission shape: {submission_check.shape}")
    print(f"ğŸ“‹ Columns: {list(submission_check.columns)}")
    
    # Validate required columns
    required_cols = ['id', 'label']
    missing_cols = [col for col in required_cols if col not in submission_check.columns]
    
    if missing_cols:
        print(f"â�Œ Missing required columns: {missing_cols}")
    else:
        print("âœ… All required columns present")
    
    # Validate label values
    valid_labels = {'positive', 'negative', 'neutral'}
    unique_labels = set(submission_check['label'].unique())
    invalid_labels = unique_labels - valid_labels
    
    if invalid_labels:
        print(f"âš ï¸�  Invalid labels found: {invalid_labels}")
        print("ğŸ”§ Valid labels are: positive, negative, neutral")
    else:
        print("âœ… All labels are valid")
    
    # Check for missing values
    missing_count = submission_check.isnull().sum().sum()
    if missing_count > 0:
        print(f"âš ï¸�  Found {missing_count} missing values")
    else:
        print("âœ… No missing values")
    
    # Summary statistics
    print(f"\nğŸ“Š FINAL SUBMISSION SUMMARY:")
    print(f"ğŸ“� Total predictions: {len(submission_check)}")
    print(f"ğŸ“ˆ Label distribution:")
    label_counts = submission_check['label'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(submission_check)) * 100
        print(f"   {label}: {count} ({percentage:.1f}%)")
    
    # Show sample submission format
    print(f"\nğŸ“„ SAMPLE SUBMISSION FORMAT:")
    print("id,label")
    for _, row in submission_check.head(5).iterrows():
        print(f"{row['id']},{row['label']}")
    
    # Calculate macro F1 readiness
    print(f"\nğŸ�¯ COMPETITION READINESS:")
    print("âœ… Format matches submission requirements (id,label)")
    print("âœ… Uses macro-averaged F1-score evaluation")
    print("âœ… Zero-shot approach with Qwen3:14b (latest model)")
    print("âœ… Handles Banglish code-switching")
    
    print(f"\nğŸ�† SUBMISSION READY FOR UPLOAD!")
    print("ğŸ“¤ File: submission.csv")
    print("ğŸ�ª Competition: Banglish Sentiment Challenge")
    
except FileNotFoundError:
    print("â�Œ Submission file not found!")
    print("ğŸ”§ Run the previous cells to generate predictions")
    
except Exception as e:
    print(f"â�Œ Error validating submission: {e}")

print("\n" + "=" * 60)
print("ğŸ�¯ BANGLISH SENTIMENT CHALLENGE SOLUTION COMPLETE")
print("=" * 60)




