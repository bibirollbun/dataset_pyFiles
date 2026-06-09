# !pip install --upgrade protobuf==3.20.3

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import warnings
warnings.filterwarnings('ignore')


# Display system information
print(f"TensorFlow Version: {tf.__version__}")
gpu_devices = tf.config.list_physical_devices('GPU')
print(f"GPU Devices: {gpu_devices}")


# Load test data and submission template
test_df = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')
submission_template = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/sample_submission.csv')

print(f"Test Data Shape: {test_df.shape}")
print(f"Submission Template Shape: {submission_template.shape}")


# Prepare test text with separators
test_df['text_combined'] = (test_df['prompt'] + ' [SEP] ' + 
                             test_df['response_a'] + ' [SEP] ' + 
                             test_df['response_b'])

SEQUENCE_LENGTH = 512

print("\n" + "-" * 70)
print("LOADING TOKENIZER AND PREPROCESSING TEST DATA")
print("-" * 70)



# Load pre-trained tokenizer
tokenizer_path = '/kaggle/input/aditya-model/tokenizer_gru.pkl'
with open(tokenizer_path, 'rb') as file:
    text_tokenizer = pickle.load(file)



# Convert text to sequences and pad
test_sequences = text_tokenizer.texts_to_sequences(test_df['text_combined'])
X_test_padded = pad_sequences(test_sequences, maxlen=SEQUENCE_LENGTH, 
                              padding='post', truncating='post')

print(f"Processed test shape: {X_test_padded.shape}")


# Custom layer for handling mask equality
class NotEqual(keras.layers.Layer):
    def __init__(self, cast_to_float=True, **kwargs):
        super().__init__(**kwargs)
        self.cast_to_float = cast_to_float
    
    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[0]
        
        result = tf.math.not_equal(inputs, 0)
        if self.cast_to_float:
            return tf.cast(result, tf.float32)
        return result
    
    def get_config(self):
        config = super().get_config()
        config.update({"cast_to_float": self.cast_to_float})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)



# Model loading utility with config patching
def load_model_with_fix(model_filepath):
    import json
    import h5py
    import tempfile
    import shutil
    import os
    
    # Read model configuration
    with h5py.File(model_filepath, 'r') as hf:
        config_json = json.loads(hf.attrs['model_config'])
        
        # Fix NotEqual layer configuration
        for layer_config in config_json['config']['layers']:
            if layer_config['class_name'] == 'NotEqual':
                for node in layer_config.get('inbound_nodes', []):
                    if len(node['args']) > 1 and isinstance(node['args'][1], int):
                        node['args'] = [node['args'][0]]
    
    # Create temporary file with fixed config
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
        temp_filepath = temp_file.name
    
    shutil.copy2(model_filepath, temp_filepath)
    
    # Update model config in temporary file
    with h5py.File(temp_filepath, 'r+') as hf:
        del hf.attrs['model_config']
        hf.attrs['model_config'] = json.dumps(config_json)
    
    # Load model with custom objects
    loaded_model = keras.models.load_model(
        temp_filepath,
        custom_objects={'NotEqual': NotEqual},
        compile=False
    )
    
    # Clean up temporary file
    os.remove(temp_filepath)
    
    return loaded_model


# Load and predict with all fold models
NUM_FOLDS = 5
all_predictions = []

print("\n" + "-" * 70)
print("LOADING MODELS AND GENERATING PREDICTIONS")
print("-" * 70)

for fold_id in range(NUM_FOLDS):
    model_filepath = f'/kaggle/input/aditya-model/gru_model_fold_{fold_id}.h5'
    print(f"Processing Fold {fold_id}: {model_filepath}")
    
    try:
        fold_model = load_model_with_fix(model_filepath)
        fold_preds = fold_model.predict(X_test_padded, batch_size=32, verbose=0)
        all_predictions.append(fold_preds)
        print(f"  Fold {fold_id} output shape: {fold_preds.shape}")
    except Exception as error:
        print(f"  Error loading Fold {fold_id}: {repr(error)}")
        import traceback
        traceback.print_exc()


# Ensemble predictions
if all_predictions:
    ensemble_preds = tf.reduce_mean(all_predictions, axis=0).numpy()
    print(f"\nEnsembled predictions shape: {ensemble_preds.shape}")
else:
    print("\nWarning: No models were successfully loaded!")
    ensemble_preds = None

print("\n" + "-" * 70)
print("COMPUTING ENSEMBLE PREDICTIONS")
print("-" * 70)

final_predictions = np.mean(all_predictions, axis=0)
print(f"Final prediction array shape: {final_predictions.shape}")

print("\n" + "-" * 70)
print("PREPARING SUBMISSION FILE")
print("-" * 70)


# Create submission dataframe
submission_df = submission_template.copy()
submission_df['winner_model_a'] = final_predictions[:, 0]
submission_df['winner_model_b'] = final_predictions[:, 1]
submission_df['winner_tie'] = final_predictions[:, 2]


# Save submission
submission_path = '/kaggle/working/submission.csv'
submission_df.to_csv(submission_path, index=False)

print(f"Submission saved to: {submission_path}")
print(f"Submission shape: {submission_df.shape}")

print("\n" + "-" * 70)
print("SUBMISSION PREVIEW")
print("-" * 70)
print(submission_df.head(20))

print("\n" + "-" * 70)
print("PREDICTION DISTRIBUTION STATISTICS")
print("-" * 70)

for idx, label in enumerate(['Model A', 'Model B', 'Tie']):
    pred_col = final_predictions[:, idx]
    print(f"{label:8s} - Mean: {pred_col.mean():.4f} | "
          f"Std: {pred_col.std():.4f} | "
          f"Min: {pred_col.min():.4f} | "
          f"Max: {pred_col.max():.4f}")

print("\n" + "-" * 70)
print("PROBABILITY VALIDATION")
print("-" * 70)

row_sums = final_predictions.sum(axis=1)
print(f"Probability sum - Min: {float(row_sums.min()):.6f}")
print(f"Probability sum - Max: {float(row_sums.max()):.6f}")
print(f"Probability sum - Mean: {float(row_sums.mean()):.6f}")

print("\n" + "-" * 70)
print("INFERENCE PIPELINE COMPLETED")
print("-" * 70)
print(f"Final submission: {submission_path}")

