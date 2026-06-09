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


# ğŸ“Œ Separate features & labels
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
    Dense(512, activation='relu', input_shape=(x.shape[1],)),  # ğŸ”¹ Input layer: 2548 neurons
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


# Extract final weights after training
final_weights = [layer.get_weights() for layer in model1.layers]


print("\n==== Model Weights and Biases Shapes ====")
for i, layer_weights in enumerate(final_weights):
    print(f"\nLayer {i+1}:")
    
    if len(layer_weights) == 2:  # Dense Layer (weights & biases)
        w, b = layer_weights
        print("Weights shape:", w.shape)
        print("Biases shape:", b.shape)

    elif len(layer_weights) == 4:  # BatchNormalization Layer (gamma, beta, mean, variance)
        print("BatchNormalization Shapes:")
        for j, param in enumerate(layer_weights):
            print(f"Param {j+1} shape:", param.shape)
    
    else:  # Layers with no trainable weights (like Dropout)
        print("No trainable weights.")



# Print all weights and biases
print("\n==== Model Weights and Biases ====")
for i, layer_weights in enumerate(final_weights):
    print(f"\nLayer {i+1}:")
    
    if len(layer_weights) == 2:  # Standard Dense Layer (weights & biases)
        w, b = layer_weights
        print("Weights:\n", w)
        print("Biases:\n", b)

    elif len(layer_weights) > 2:  # BatchNormalization Layer (gamma, beta, mean, variance)
        print("BatchNormalization Parameters:")
        for j, param in enumerate(layer_weights):
            print(f"Param {j+1}:\n", param)
    
    else:  # Layers with no trainable weights (like Dropout)
        print("No trainable weights.")



import numpy as np

# 1. Same as final weights
same_weights = [np.array(layer_weights, dtype=object).copy() for layer_weights in final_weights]

# 2. Different from final weights (slightly modified)
diff_weights = []
for layer_weights in final_weights:
    if isinstance(layer_weights, list) or isinstance(layer_weights, tuple):
        # If the layer has multiple parameters (e.g., BatchNorm: gamma, beta, mean, variance)
        modified_layer = [w + np.random.normal(0, 0.01, size=w.shape) if isinstance(w, np.ndarray) else w for w in layer_weights]
    else:
        # If the layer has only one set of weights
        modified_layer = layer_weights + np.random.normal(0, 0.01, size=layer_weights.shape)
    
    diff_weights.append(modified_layer)



import numpy as np

# Compare element-wise for same_weights
same_comparison = [np.array_equal(w1, w2) for layer1, layer2 in zip(final_weights, same_weights) for w1, w2 in zip(layer1, layer2)]

# Compare element-wise for diff_weights
diff_comparison = [np.array_equal(w1, w2) for layer1, layer2 in zip(final_weights, diff_weights) for w1, w2 in zip(layer1, layer2)]

print("Comparison with identical weights (should be all True):", same_comparison)
print("Comparison with modified weights (should be all False):", diff_comparison)


