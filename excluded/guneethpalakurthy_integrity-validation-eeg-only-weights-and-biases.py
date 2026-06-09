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


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization,LSTM



df = pd.read_csv('/kaggle/input/eeg-brainwave-dataset-feeling-emotions/emotions.csv')


df.head()


# ðŸ“Œ Separate features & labels
x = df.iloc[:, :-1].values  # EEG features (all columns except last)
y = df.iloc[:, -1].values   # Target labels (emotion category)



# Encoding labels (Converting text labels to numbers)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

#Normalizing EEG feature values
scaler = StandardScaler()
x = scaler.fit_transform(x)



x = np.expand_dims(x, axis=2)  # Shape: (2132, 2548, 1)


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)


model1 = Sequential([
    Dense(512, activation='relu', input_shape=(x.shape[1],)),  # ðŸ”¹ Input layer: 2548 neurons
    BatchNormalization(),
    Dropout(0.3),

    Dense(256, activation='relu'),  # Hidden layer 1
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),  # Hidden layer 2
    Dropout(0.3),

    Dense(len(np.unique(y)), activation='softmax')  # Output layer for classification
])


model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model1.fit(xtrain, ytrain, epochs=30, batch_size=32, validation_data=(xtest, ytest))



test_loss, test_acc = model1.evaluate(xtest, ytest)
print(f"Test Accuracy: {test_acc:.4f}")


# Function to extract only Dense layer weights (ignore BatchNorm)
def extract_dense_weights(model_weights):
    return [layer for layer in model_weights if len(layer) == 2]  # Only Dense layers (weights & biases)



# Extract final weights after training
final_weights = [layer.get_weights() for layer in model1.layers]


import numpy as np

# 1. Create an exact copy of final_weights without modification
same_weights = [
    [np.copy(w) if isinstance(w, np.ndarray) else w for w in layer_weights] 
    if isinstance(layer_weights, (list, tuple)) else np.copy(layer_weights) 
    for layer_weights in final_weights
]

# 2. Modify only 1 or 2 neurons per layer
diff_weights = []
for layer_weights in final_weights:
    if isinstance(layer_weights, np.ndarray):  # If layer is a NumPy array (single weight matrix)
        modified_layer = np.copy(layer_weights)  # Deep copy
        
        if modified_layer.shape[0] > 0:  # Ensure the array is not empty
            num_neurons = np.random.choice([1, 2])  # Modify 1 or 2 neurons
            neuron_indices = np.random.choice(modified_layer.shape[0], size=num_neurons, replace=False)  # Select neurons

            for i in neuron_indices:
                modified_layer[i] += np.random.normal(0, 0.01, size=modified_layer[i].shape)  # Add noise
        
        diff_weights.append(modified_layer)

    elif isinstance(layer_weights, (list, tuple)) and layer_weights:  # Ensure it's non-empty
        modified_layer = [
            np.copy(w) if isinstance(w, np.ndarray) else w for w in layer_weights
        ]  # Deep copy each NumPy array
        
        if isinstance(modified_layer[0], np.ndarray) and modified_layer[0].shape[0] > 0:
            num_neurons = np.random.choice([1, 2])  # Modify 1 or 2 neurons
            neuron_indices = np.random.choice(modified_layer[0].shape[0], size=num_neurons, replace=False)  # Select neurons
            
            for i in neuron_indices:
                modified_layer[0][i] += np.random.normal(0, 0.01, size=modified_layer[0][i].shape)  # Modify first weight matrix
        
        diff_weights.append(modified_layer)

    else:
        diff_weights.append(layer_weights)  # Keep as-is if it's an unsupported type

# Verify at least one layer has changed
layerwise_comparison = [
    any(np.any(mod_w != orig_w) for mod_w, orig_w in zip(mod_layer, orig_layer))
    if isinstance(mod_layer, (list, tuple)) else np.any(mod_layer != orig_layer)
    for mod_layer, orig_layer in zip(diff_weights, final_weights)
]

print("At least one layer modified:", any(layerwise_comparison))  # Should print True



# Extract only Dense layer weights and biases
filtered_final_weights = extract_dense_weights(final_weights)
filtered_same_weights = extract_dense_weights(same_weights)
filtered_diff_weights = extract_dense_weights(diff_weights)


for i, layer_weights in enumerate(filtered_final_weights):
    print(f"\nLayer {i+1}:")
    
    w, b = layer_weights  # Each layer contains weights and biases
    
    print("Weights shape:", w.shape)
    print("Biases shape:", b.shape)


# Print the shape and values of weights and biases
for i, layer_weights in enumerate(filtered_final_weights):
    w, b = layer_weights  # Each layer contains weights and biases
    
    print(f"\nLayer {i+1}:")
    print("Weights shape:", w.shape)
    print("Weights values:\n", w)
    
    print("Biases shape:", b.shape)
    print("Biases values:\n", b)


# Compare element-wise for identical weights (should be all True)
same_comparison = [
    np.array_equal(w1, w2)  
    for layer1, layer2 in zip(filtered_final_weights, filtered_same_weights)  
    for w1, w2 in zip(layer1, layer2)
]

# Compare element-wise for modified weights (should be all False)
diff_comparison = [
    np.array_equal(w1, w2)  
    for layer1, layer2 in zip(filtered_final_weights, filtered_diff_weights)  
    for w1, w2 in zip(layer1, layer2)
]

print("Comparison with identical weights (should be all True):", same_comparison)
print("Comparison with modified weights (should be all False):", diff_comparison)

