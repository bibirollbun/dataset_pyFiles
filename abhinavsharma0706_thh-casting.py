import os
import numpy as np
import pandas as pd

# Constants
N_MODELS = 45      # Number of models
N_SAMPLES = 75     # Number of samples
CHANNELS = ['channel_44', 'channel_45', 'channel_46']  # Channels

# Input and Output Directories
INPUT_DIR = '/kaggle/input/trojan-horse-hunt-in-space'
CLEAN_MODEL_PATH = os.path.join(INPUT_DIR, 'clean_model')
POISONED_MODELS_PATH = os.path.join(INPUT_DIR, 'poisoned_models')
SUBMISSION_PATH = '/kaggle/working/submission.csv'  # Updated path for submission

DEBUG = False
if DEBUG:
    INPUT_DIR = './data'
    CLEAN_MODEL_PATH = os.path.join(INPUT_DIR, 'clean_model')
    POISONED_MODELS_PATH = os.path.join(INPUT_DIR, 'poisoned_models')
    SUBMISSION_PATH = '/kaggle/working/submission.csv'  # Updated path for submission

print("âœ… All configurations are set.")
print(f"Models: {N_MODELS}")
print(f"Samples: {N_SAMPLES}")
print(f"Channels: {CHANNELS}")
print(f"All Channels: {N_SAMPLES * len(CHANNELS)}")

def create_zero_trigger_submission():
    print("ğŸ“Š Generating submission data...")
    
    zero_trigger = np.zeros(N_SAMPLES * len(CHANNELS))
    print(f"Triggers: {len(zero_trigger)}")
    
    data = np.tile(zero_trigger, (N_MODELS, 1))
    print(f"Data shape: {data.shape}")
    
    df = pd.DataFrame(data)
    channel_cols = [
        f"{ch}_{i+1}"
        for ch in CHANNELS
        for i in range(N_SAMPLES)
    ]
    
    print(f"Channel columns: {len(channel_cols)}")
    print(f"First 5 columns: {channel_cols[:5]}")
    
    df.columns = channel_cols    
    df.insert(0, "model_id", range(1, N_MODELS + 1))    
    df.index = df.index + 1
    
    print("âœ… Submission data generated successfully.")
    return df

def save_and_validate_submission(df):
    print("ğŸ’¾ Saving submission data...")
    
    # CSV
    df.to_csv(SUBMISSION_PATH, index=False)
    print(f"âœ… Submission saved to: {SUBMISSION_PATH}")
    
    # Validate shape
    print("\nğŸ”� Data Shape:")
    print(f"Shape: {df.shape}")
    print(f"Expected shape: ({N_MODELS}, {N_SAMPLES * len(CHANNELS) + 1})")  # Including model_id
    
    # Display first few rows
    print(f"\nğŸ“‹ First 3 rows:")
    print(df.head(3))
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    print(f"\nMissing values: {missing_values}")
    
    # File size
    file_size = os.path.getsize(SUBMISSION_PATH)
    print(f"File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    return True

def main():
    print("ğŸš€ Generating submission...")
    print("=" * 60)
    
    try:
        df = create_zero_trigger_submission()
        save_and_validate_submission(df)
        
        print("\n" + "=" * 60)
        print("ğŸ�‰ Submission data ready.")
        
        return df
        
    except Exception as e:
        print(f"â�Œ Error: {str(e)}")
        raise

if __name__ == "__main__":
    submission_df = main()
    
    print("\nğŸ“Š Submission Data:")
    print(f"â€¢ Number of rows: {len(submission_df)}")
    print(f"â€¢ Number of columns: {len(submission_df.columns) - 1}")  # Excluding model_id
    print(f"â€¢ Value range: [{submission_df.iloc[:, 1:].min().min():.3f}, {submission_df.iloc[:, 1:].max().max():.3f}]")
    print(f"â€¢ Final path: {SUBMISSION_PATH}")
    
    print("\nğŸ�¯ Kaggle Result Ready!")


