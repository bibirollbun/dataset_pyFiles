# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


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

test = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')
sample_submission = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/sample_submission.csv')

print("Test Shape:", test.shape)
print("Sample Submission Shape:", sample_submission.shape)


test['combined_text'] = test['prompt'] + ' [SEP] ' + test['response_a'] + ' [SEP] ' + test['response_b']


MAX_LEN = 512

print("\n" + "="*80)
print("LOADING GRU MODELS AND TOKENIZER")
print("="*80)

with open('/kaggle/input/gru-lmsys-dipti-10033/tokenizer_gru.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

X_test_seq = tokenizer.texts_to_sequences(test['combined_text'])
X_test = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

print("X_test shape:", X_test.shape)


import tensorflow as tf
from tensorflow import keras

class NotEqual(keras.layers.Layer):
    def __init__(self, cast_to_float=True, **kwargs):
        super().__init__(**kwargs)
        self.cast_to_float = cast_to_float
    
    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[0]
        
        out = tf.math.not_equal(inputs, 0)
        if self.cast_to_float:
            return tf.cast(out, tf.float32)
        return out
    
    def get_config(self):
        config = super().get_config()
        config.update({"cast_to_float": self.cast_to_float})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def custom_load_model(filepath):
    import json
    import h5py
    
    with h5py.File(filepath, 'r') as f:
        model_config = json.loads(f.attrs['model_config'])
        
        for layer in model_config['config']['layers']:
            if layer['class_name'] == 'NotEqual':
                for node in layer.get('inbound_nodes', []):
                    if len(node['args']) > 1 and isinstance(node['args'][1], int):
                        node['args'] = [node['args'][0]]
    
    import tempfile
    import shutil
    
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = tmp.name
    
    shutil.copy2(filepath, tmp_path)
    
    with h5py.File(tmp_path, 'r+') as f:
        del f.attrs['model_config']
        f.attrs['model_config'] = json.dumps(model_config)
    
    model = keras.models.load_model(
        tmp_path,
        custom_objects={'NotEqual': NotEqual},
        compile=False
    )
    
    import os
    os.remove(tmp_path)
    
    return model

n_folds = 5
fold_predictions = []

for fold in range(n_folds):
    model_path = f'/kaggle/input/gru-lmsys-dipti-10033/gru_model_fold_{fold}.h5'
    print(f"Loading fold {fold} from {model_path}")
    try:
        model = custom_load_model(model_path)
        predictions = model.predict(X_test, batch_size=32, verbose=0)
        fold_predictions.append(predictions)
        print(f"Fold {fold} predictions shape:", predictions.shape)
    except Exception as e:
        print(f"Failed to load fold {fold}: {repr(e)}")
        import traceback
        traceback.print_exc()

if fold_predictions:
    avg_predictions = tf.reduce_mean(fold_predictions, axis=0).numpy()
    print(f"\nFinal averaged predictions shape: {avg_predictions.shape}")
else:
    print("\nNo models loaded successfully!")



print("\n" + "="*80)
print("AVERAGING PREDICTIONS")
print("="*80)

avg_predictions = np.mean(fold_predictions, axis=0)
print("Average predictions shape:", avg_predictions.shape)


print("\n" + "="*80)
print("CREATING SUBMISSION")
print("="*80)

submission = sample_submission.copy()
submission['winner_model_a'] = avg_predictions[:, 0]
submission['winner_model_b'] = avg_predictions[:, 1]
submission['winner_tie'] = avg_predictions[:, 2]

output_path = '/kaggle/working/submission.csv'
submission.to_csv(output_path, index=False)
print("Saved:", output_path)
print("Shape:", submission.shape)



print("\n" + "="*80)
print("SUBMISSION PREVIEW")
print("="*80)
print(submission.head(20))


print("\n" + "="*80)
print("PREDICTION STATISTICS")
print("="*80)

print(f"Model A - Mean: {avg_predictions[:, 0].mean():.4f}, Std: {avg_predictions[:, 0].std():.4f}, Min: {avg_predictions[:, 0].min():.4f}, Max: {avg_predictions[:, 0].max():.4f}")
print(f"Model B - Mean: {avg_predictions[:, 1].mean():.4f}, Std: {avg_predictions[:, 1].std():.4f}, Min: {avg_predictions[:, 1].min():.4f}, Max: {avg_predictions[:, 1].max():.4f}")
print(f"Tie     - Mean: {avg_predictions[:, 2].mean():.4f}, Std: {avg_predictions[:, 2].std():.4f}, Min: {avg_predictions[:, 2].min():.4f}, Max: {avg_predictions[:, 2].max():.4f}")



print("\n" + "="*80)
print("VERIFYING PROBABILITIES SUM TO 1")
print("="*80)

prob_sums = avg_predictions.sum(axis=1)
print("Min sum:", float(prob_sums.min()))
print("Max sum:", float(prob_sums.max()))
print("Mean sum:", float(prob_sums.mean()))


print("\n" + "="*80)
print("INFERENCE COMPLETED")
print("="*80)
print("Submission file saved:", output_path)

