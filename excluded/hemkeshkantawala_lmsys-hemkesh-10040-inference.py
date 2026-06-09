!pip install --upgrade protobuf==3.20.3


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import warnings
warnings.filterwarnings('ignore')


print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Variable 'test' renamed to 'df_test_data'
df_test_data = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')

# Variable 'sample_submission' renamed to 'df_sample_submission'
df_sample_submission = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/sample_submission.csv')

print("Test Shape:", df_test_data.shape)
print("Sample Submission Shape:", df_sample_submission.shape)


df_test_data['text_concatenated'] = df_test_data['prompt'] + ' [SEP] ' + df_test_data['response_a'] + ' [SEP] ' + df_test_data['response_b']


SEQUENCE_LENGTH = 512

print("\n" + "="*80)
print("LOADING GRU MODELS AND TOKENIZER")
print("="*80)

with open('/kaggle/input/lmsys-hemkesh-10040-gru/tokenizer_gru.pkl', 'rb') as file_handler:
    text_tokenizer = pickle.load(file_handler)

test_sequences = text_tokenizer.texts_to_sequences(df_test_data['text_concatenated'])
test_padded_inputs = pad_sequences(test_sequences, maxlen=SEQUENCE_LENGTH, padding='post', truncating='post')

print("X_test shape:", test_padded_inputs.shape)


import tensorflow as tf
from tensorflow import keras
import json
import h5py
import tempfile
import shutil
import os
import traceback

class NonZeroMask(keras.layers.Layer):
    def __init__(self, convert_to_float=True, **kwargs):
        super().__init__(**kwargs)
        self.convert_to_float = convert_to_float
    
    def call(self, input_tensor):
        if isinstance(input_tensor, (list, tuple)):
            input_tensor = input_tensor[0]
        
        output_tensor = tf.math.not_equal(input_tensor, 0)
        if self.convert_to_float:
            return tf.cast(output_tensor, tf.float32)
        return output_tensor
    
    def get_config(self):
        config = super().get_config()
        config.update({"cast_to_float": self.convert_to_float})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def load_model_with_custom_config(model_file_path):
    
    with h5py.File(model_file_path, 'r') as file_handler:
        architecture_config = json.loads(file_handler.attrs['model_config'])
        
        for network_layer in architecture_config['config']['layers']:
            if network_layer['class_name'] == 'NotEqual': # Checking against original class name in saved file
                for graph_node in network_layer.get('inbound_nodes', []):
                    if len(graph_node['args']) > 1 and isinstance(graph_node['args'][1], int):
                        graph_node['args'] = [graph_node['args'][0]]
    
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file_obj:
        temp_file_path = temp_file_obj.name
    
    shutil.copy2(model_file_path, temp_file_path)
    
    with h5py.File(temp_file_path, 'r+') as file_handler:
        del file_handler.attrs['model_config']
        file_handler.attrs['model_config'] = json.dumps(architecture_config)
    
    # Note: 'NotEqual' key remains because saved model expects that specific string
    loaded_model = keras.models.load_model(
        temp_file_path,
        custom_objects={'NotEqual': NonZeroMask}, 
        compile=False
    )
    
    os.remove(temp_file_path)
    
    return loaded_model

k_fold_count = 5
ensemble_predictions_list = []

for current_fold_index in range(k_fold_count):
    input_model_path = f'/kaggle/input/lmsys-hemkesh-10040-gru/gru_model_fold_{current_fold_index}.h5'
    print(f"Loading fold {current_fold_index} from {input_model_path}")
    try:
        current_gru_model = load_model_with_custom_config(input_model_path)
        
        # Using previously renamed 'test_padded_inputs'
        fold_output_probs = current_gru_model.predict(test_padded_inputs, batch_size=32, verbose=0)
        ensemble_predictions_list.append(fold_output_probs)
        print(f"Fold {current_fold_index} predictions shape:", fold_output_probs.shape)
        
    except Exception as error_msg:
        print(f"Failed to load fold {current_fold_index}: {repr(error_msg)}")
        traceback.print_exc()

if ensemble_predictions_list:
    final_ensemble_probabilities = tf.reduce_mean(ensemble_predictions_list, axis=0).numpy()
    print(f"\nFinal averaged predictions shape: {final_ensemble_probabilities.shape}")
else:
    print("\nNo models loaded successfully!")


print("\n" + "="*80)
print("AVERAGING PREDICTIONS")
print("="*80)

final_ensemble_probabilities = np.mean(ensemble_predictions_list, axis=0)

print("Average predictions shape:", final_ensemble_probabilities.shape)


print("\n" + "="*80)
print("CREATING SUBMISSION")
print("="*80)

df_final_submission = df_sample_submission.copy()

df_final_submission['winner_model_a'] = final_ensemble_probabilities[:, 0]
df_final_submission['winner_model_b'] = final_ensemble_probabilities[:, 1]
df_final_submission['winner_tie'] = final_ensemble_probabilities[:, 2]

submission_file_path = '/kaggle/working/submission.csv'
df_final_submission.to_csv(submission_file_path, index=False)

print("Saved:", submission_file_path)
print("Shape:", df_final_submission.shape)


print("\n" + "="*80)
print("SUBMISSION PREVIEW")
print("="*80)
print(df_final_submission.head(20))


print("\n" + "="*80)
print("PREDICTION STATISTICS")
print("="*80)
print(f"Model A - Mean: {final_ensemble_probabilities[:, 0].mean():.4f}, Std: {final_ensemble_probabilities[:, 0].std():.4f}, Min: {final_ensemble_probabilities[:, 0].min():.4f}, Max: {final_ensemble_probabilities[:, 0].max():.4f}")
print(f"Model B - Mean: {final_ensemble_probabilities[:, 1].mean():.4f}, Std: {final_ensemble_probabilities[:, 1].std():.4f}, Min: {final_ensemble_probabilities[:, 1].min():.4f}, Max: {final_ensemble_probabilities[:, 1].max():.4f}")
print(f"Tie     - Mean: {final_ensemble_probabilities[:, 2].mean():.4f}, Std: {final_ensemble_probabilities[:, 2].std():.4f}, Min: {final_ensemble_probabilities[:, 2].min():.4f}, Max: {final_ensemble_probabilities[:, 2].max():.4f}")


print("\n" + "="*80)
print("VERIFYING PROBABILITIES SUM TO 1")
print("="*80)

probability_row_sums = final_ensemble_probabilities.sum(axis=1)

print("Min sum:", float(probability_row_sums.min()))
print("Max sum:", float(probability_row_sums.max()))
print("Mean sum:", float(probability_row_sums.mean()))


print("\n" + "="*80)
print("INFERENCE COMPLETED")
print("="*80)
print("Submission file saved:", submission_file_path)

