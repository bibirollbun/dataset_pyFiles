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
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization,LSTM


df = pd.read_csv('/kaggle/input/eeg-brainwave-dataset-feeling-emotions/emotions.csv')


df.head()
df.shape


x = df.iloc[:, :-1].values 
y = df.iloc[:, -1].values


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

scalar = StandardScaler()
x = scalar.fit_transform(x)
x, y


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)


xtrain.shape


from sklearn.decomposition import PCA

def apply_pca(X, explained_variance_threshold=0.95):
    # Fit PCA without reducing dimensions to check variance explained
    pca = PCA()
    pca.fit(X)
    
    # Find the number of components that explain the desired variance
    total_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(total_variance >= explained_variance_threshold) + 1
    
    # Apply PCA with selected number of components
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(X)
    
    print(f"Selected {n_components} components (explains {explained_variance_threshold * 100}% variance)")
    
    pca_df = pd.DataFrame(transformed_data, columns=[f"pc{i+1}" for i in range(n_components)])
    return pca_df, transformed_data


eeg_data =  xtrain
pca_df, pca_data = apply_pca(eeg_data)
pca_df.head()
xtrain.shape


from sklearn.neighbors import NearestNeighbors

def find_optimal_eps(X, k=5):
    
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X)
    distances, _ = neighbors_fit.kneighbors(X)

    distances = np.sort(distances[:, k-1])

    diffs = np.diff(distances)  
    elbow_index = np.argmax(diffs)
    optimal_eps = distances[elbow_index]

    print(f"Optimal eps: {optimal_eps:.4f}")
    return optimal_eps


optimal_eps = find_optimal_eps(pca_data, k=9) 
print("x")


import math
optimal_min_samples = max(2 * pca_data.shape[1], int(math.sqrt(len(pca_data))))
print(f"Optimal min_samples: {optimal_min_samples}")


from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# Step 1: Apply PCA for Dimensionality Reduction
def apply_pca(X, n_components=3):
    pca = PCA(n_components)
    transformed_data = pca.fit_transform(X)
    pca_df = pd.DataFrame(transformed_data, columns=[f"pc{i+1}" for i in range(n_components)])
    return pca_df

def apply_dbscan(pca_df, eps=0.5, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pca_df)
    labels = db.labels_

    # Insert cluster labels into DataFrame
    pca_df["Cluster"] = labels

    # Identify core points
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    pca_df["corePoint"] = core_samples_mask

    return pca_df, labels

def select_samples(pca_df, b=5, n=5, c=5):
    selected_samples = []

    # Count total clusters (excluding noise, labeled as -1)
    num_clusters = len(set(pca_df["Cluster"])) - (1 if -1 in pca_df["Cluster"].values else 0)

    # Select Border Points
    border_count = 0
    for cluster in range(num_clusters):
        cluster_border_points = pca_df[(pca_df["Cluster"] == cluster) & (~pca_df["corePoint"])]
        if not cluster_border_points.empty:
            selected_samples.extend(cluster_border_points.index[:b])
            border_count += len(cluster_border_points.index[:b])

    # Select Noise Points
    noise_samples = pca_df[pca_df["Cluster"] == -1].index[:n]
    selected_samples.extend(noise_samples)

    core_count = [0] * num_clusters
    for cluster in range(num_clusters):
        cluster_core_points = pca_df[(pca_df["Cluster"] == cluster) & (pca_df["corePoint"])]
        if not cluster_core_points.empty:
            selected_samples.extend(cluster_core_points.index[:c])
            core_count[cluster] += len(cluster_core_points.index[:c])

    return selected_samples

def main():
    # Example EEG dataset (replace with actual EEG data)
    global xtrain
    eeg_data = xtrain  # 2132 samples, 2549 features

    # Step 1: Apply PCA
    pca_df = apply_pca(eeg_data)

    # Step 2: Apply DBSCAN
    pca_df, labels = apply_dbscan(pca_df, eps=0.5, min_samples=5)

    # Step 3: Select Representative EEG Samples
    selected_samples = select_samples(pca_df, b=5, n=5, c=5)

    print("Selected EEG Sample Indices:", selected_samples)
    
if __name__ == "__main__":
    main()


selected_samples = [166, 273, 275, 277, 387, 566, 642, 871, 1209, 1306, 56, 80, 1414, 1469, 567, 599, 1138, 1338, 0, 1, 2, 3, 4, 8, 14, 16, 22, 30, 85, 108, 168, 285, 827, 573, 738]

selected_samples = np.array(selected_samples).flatten()  # Ensure it's 1D
selected_data = x[selected_samples]  # Extract rows from dataset

print(f"Selected data shape: {selected_data.shape}")


from tensorflow.keras.models import Model

input_layer = Input(shape=(x.shape[1],))

help = Dense(512, activation='relu')(input_layer)
help = BatchNormalization()(help)
help = Dense(256, activation='relu')(help)
help = BatchNormalization()(help)
help = Dense(128, activation='relu')(help)

output_layer = Dense(len(np.unique(y)), activation='softmax')(help)

model1 = Model(inputs=input_layer, outputs=output_layer)

model1.summary()



from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

model1.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=SparseCategoricalCrossentropy(), 
    metrics=['accuracy']
)

history = model1.fit(
    xtrain, ytrain,
    epochs=20,
    batch_size=32,
    validation_data=(xtest, ytest),
    verbose=1
)

test_loss, test_acc = model1.evaluate(xtest, ytest)
print(f"Test Accuracy: {test_acc * 100:.2f}%")



def compare_model_weights(model1, model2):
    for layer1, layer2 in zip(model1.layers, model2.layers):
        weights1 = layer1.get_weights()
        weights2 = layer2.get_weights()
        
        for w1, w2 in zip(weights1, weights2):
            if not np.array_equal(w1, w2):
                print(f"❌ Weights are different in layer: {layer1.name}")
                return False
    print("✅ All weights are identical between both models.")
    return True




import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization

def neuron_coverage_with_bn(model, inputs, threshold=1):
    
    relevant_layers = [layer for layer in model.layers if isinstance(layer, (Dense, BatchNormalization))]
    activation_model = Model(inputs=model.input, outputs=[layer.output for layer in relevant_layers])

    activations = activation_model.predict(inputs, batch_size=len(inputs), verbose=0)

    total_neurons = sum(layer.shape[1] for layer in activations)
    activated_neurons = sum(np.any(layer > threshold, axis=0).sum() for layer in activations)
    
    coverage = activated_neurons / total_neurons if total_neurons > 0 else 0
    return coverage, activated_neurons, total_neurons


coverage, active_neurons, total_neurons = neuron_coverage_with_bn(model1, selected_data)
print(f"Neuron Coverage: {coverage * 100:.2f}% ({active_neurons}/{total_neurons} neurons activated)")



selected_data
selected_samples


def filter_correct_predictions_tf(model, selected_indexes):
    # Predict probabilities for selected samples
    x_selected = x[selected_indexes]
    predictions = model.predict(x_selected)
    
    # Convert probabilities to predicted class
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get the actual classes
    true_classes = y[selected_indexes]

    # Compare and keep only correct predictions
    correct_mask = predicted_classes == true_classes
    filtered_indexes = np.array(selected_indexes)[correct_mask]

    return filtered_indexes

updated_samples = filter_correct_predictions_tf(model1, selected_samples)


updated_samples.shape
updated_samples


import numpy as np

def get_random_sample_different_class(given_class):
    # Get indices where class ≠ given_class
    valid_indices = np.where(y != given_class)[0]
    
    # Randomly select one index
    selected_index = np.random.choice(valid_indices)

    # Return the sample and its label
    return x[selected_index], y[selected_index]


import tensorflow as tf

def generate_changed_class_data(model, selected_indices, steps = 1000):

    changed_x = []
    changed_y = []
    
    for i in range(len(selected_indices)):
    
        x_i = x[i]  
        y_i = y[i]

        xx, yy = get_random_sample_different_class(y_i)
        
        for itr in range(steps + 1):
            alpha = itr / steps
            morphed = (1 - alpha) * x_i + alpha * xx

            morphed_input = np.expand_dims(morphed, axis=0)  # Add batch dimension
            prediction = model.predict(morphed_input, verbose=0)
            predicted_class = np.argmax(prediction, axis=1)[0]

            if predicted_class == yy: 
                changed_x.append(morphed)
                changed_y.append(yy)
                print(f"{itr}")
                break;
                
        print(f"{y_i}, {yy}, {predicted_class}")

    return changed_x, changed_y

help_x, help_y = generate_changed_class_data(model1, updated_samples)


np.array(help_x).shape


from tensorflow.keras.models import clone_model

# Clone the architecture
valid_clone = clone_model(model1)

# Build the model with same input shape
valid_clone.build(input_shape=(None, x.shape[1]))

# Copy the trained weights
valid_clone.set_weights(model1.get_weights())


# Compile the clone before evaluation
valid_clone.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Now you can safely evaluate
test_loss, test_acc = valid_clone.evaluate(xtest, ytest)
print(f"Test Accuracy: {test_acc * 100:.2f}%")


compare_model_weights(model1, valid_clone)


import numpy as np

# Step 1: Get all weights from the cloned model
weights = model1.get_weights()

print(f"Original weight value: {weights[0][0][0]}")  

weights[0][0][0] += 0.1

invalid_clone = clone_model(model1)
invalid_clone.build(input_shape=(None, x.shape[1]))
invalid_clone.set_weights(weights)

print(f"Modified weight value: {invalid_clone.get_weights()[0][0][0]}")  # after changing


compare_model_weights(model1, invalid_clone)


sub_set_x, sub_set_y = x[updated_samples], y[updated_samples]
modified_sub_set_x, modified_sub_set_y = np.array(help_x), np.array(help_y)


def chk_model(model):
    prediction = model.predict(sub_set_x, verbose=0)
    predicted_classes = np.argmax(prediction, axis=1)

    if not np.array_equal(predicted_classes, sub_set_y):
        print("Model is compromised")
        return False

    prediction = model.predict(modified_sub_set_x, verbose=0)
    predicted_classes = np.argmax(prediction, axis=1)

    if not np.array_equal(predicted_classes, modified_sub_set_y):
        print("Model is compromised")
        return False
        
    return True



chk_model(valid_clone)


chk_model(invalid_clone)

