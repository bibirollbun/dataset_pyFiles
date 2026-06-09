import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, GlobalMaxPooling2D, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
from keras import layers
import json
import math
from PIL import Image
import random
import pandas as pd
from pathlib import Path
from matplotlib import colors

#besok submission tidak error
CMAP = colors.ListedColormap(
    ['#ffffff','#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
NORM = colors.Normalize(vmin=-1, vmax=10)

SUB_TARGET = '/kaggle/working/submission.json'

def olahtraining(dataset, keys):
    inputfinal = []
    outputfinal = []
    for kunci in keys and dataset:
        items = dataset[kunci]['train']
        for item in items:
            try:
                input_img = render(np.array(item['input']))
                output_img = render(np.array(item['output']))
                input_img = input_img.resize((9, 9))
                output_img = output_img.resize((9, 9))
                input_arr = np.array(input_img)
                output_arr = np.array(output_img)
                inputfinal.append(input_arr)
                outputfinal.append(output_arr)
            except Exception as e:
                print(f"Error processing sample in {kunci}: {e}")
    return np.array(inputfinal), np.array(outputfinal)

def olahtest(dataset, keys):
    outputfinal = []
    for kunci in keys:
        outputs = dataset[kunci]['test']
        for item in outputs:
            try:
                test_img = render(np.array(item['input']))
                test_img = test_img.resize((9, 9))
                test_arr = np.array(test_img)
                outputfinal.append(test_arr)
            except Exception as e:
                print(f"Error processing test sample in {kunci}: {e}")
    return np.array(outputfinal)

SCALE_FACTOR = 10  

def render(grid):
    grid_up = np.repeat(np.repeat(grid, SCALE_FACTOR, axis=0), SCALE_FACTOR, axis=1)
    max_val = grid.max()
    grid_up[::SCALE_FACTOR, :] = max_val + 1
    grid_up[SCALE_FACTOR - 1::SCALE_FACTOR, :] = max_val + 1
    grid_up[:, ::SCALE_FACTOR] = max_val + 1
    grid_up[:, SCALE_FACTOR - 1::SCALE_FACTOR] = max_val + 1
    norm_grid = grid_up / (max_val + 1)
    cmap = plt.get_cmap('viridis')
    colored = cmap(norm_grid)
    colored = (colored[:, :, :3] * 255).astype(np.uint8)  # ambil RGB, buang alpha
    image = Image.fromarray(colored)
    return image

with open('/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json', 'r') as file:
    testing_data = json.load(file)

test_case_keys = list(testing_data.keys())
trainingkeys = list(testing_data.keys())
selected_test_keys = random.sample(test_case_keys, 10)
kuncitraining = random.sample(trainingkeys, 10)
train_inputs, train_outputs = olahtraining(testing_data,kuncitraining)
test_inputs = olahtest(testing_data,selected_test_keys)

train_inputs = train_inputs / 255
train_outputs = train_outputs / 255
test_inputs = test_inputs / 255

input_shape = train_inputs.shape[1:] 
output_shape = train_outputs.shape[1:] 

def energi1():
    angka = 2
    tumbuhkan = np.exp(angka)
    final = math.ceil(tumbuhkan)
    return final

def energi2():
    angka = 5
    tumbuh = np.exp(angka)
    final = math.ceil(tumbuh)
    return final

def energi3():
    angka = 9#copyright metode helmi
    tumbuh = np.exp(angka)
    dobel = np.log(tumbuh)
    final = math.ceil(dobel)
    return final

def energi4():
    angka = 11#copyright
    tumbuh = np.exp(angka)
    dobel = np.sin(tumbuh)
    final = math.ceil(dobel)
    return final

def energi5():
    angka = 12
    tumbuh = np.exp(angka)
    dobel = np.log(tumbuh)
    final = math.ceil(dobel)
    return final

def energi6():
    angka = 50
    angka2 = 10
    tumbuh = np.exp(angka) * np.log(angka2)
    final = math.ceil(tumbuh)
    return final
    
def build_generator(input_shape):
    model = Sequential([
    Conv2D(2048, kernel_size=(9, 9), activation='sigmoid', input_shape=input_shape, padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    #using ekponensial
    Conv2D(energi2(), kernel_size=(9, 9), activation='sigmoid', padding='same'),
    Conv2D(energi2(), kernel_size=(9, 9), activation='sigmoid', padding='same'),
    Conv2D(energi2(), kernel_size=(9, 9), activation='sigmoid', padding='same'),
    Conv2D(energi1(), kernel_size=(6, 6), activation='sigmoid', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(3, 3), padding='same'),
    #using eksponensial
    Conv2D(energi2(), kernel_size=(9, 9), activation='sigmoid', padding='same'),
    Conv2D(energi2(), kernel_size=(9, 9), activation='sigmoid', padding='same'),
    Conv2D(energi2(), kernel_size=(9, 9), activation='sigmoid', padding='same'),
    Conv2D(energi2(), kernel_size=(9, 9), activation='sigmoid', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    Conv2D(energi3(), kernel_size=(9, 9), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    Flatten(),
    #using eksponensial
    Dense(energi4(), activation='relu'),
    Dropout(0.5),
    Dense(9 * 9 * 3, activation='sigmoid'),
    Reshape((9, 9, 3))
])
    return model

def reflektif(input_shape):
    model = Sequential([
    Conv2D(4096, kernel_size=(9, 9), activation='relu', input_shape=input_shape, padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    #using ekponensial
    Conv2D(energi1(), kernel_size=(6, 6), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(3, 3), padding='same'),
    #using eksponensial
    Conv2D(energi2(), kernel_size=(9, 9), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    Conv2D(energi2() + 10, kernel_size=(9, 9), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    Flatten(),
    ])
    return model

reflektif(input_shape)

def merenung(input_shape):
    model = Sequential([
    Conv2D(energi2() - 10, kernel_size=(9, 9), activation='sigmoid', input_shape=input_shape, padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    #using ekponensial
    Conv2D(energi1(), kernel_size=(6, 6), activation='sigmoid', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(3, 3), padding='same'),
    #using eksponensial
    Conv2D(energi2() - 10, kernel_size=(9, 9), activation='sigmoid', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    Conv2D(energi2() - 10, kernel_size=(9, 9), activation='sigmoid', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    Flatten(),
    ])
    return model
    
build_generator(input_shape)
merenung(input_shape)

# Model definition
model = Sequential([
    #using eksponensial
    Conv2D(2048, kernel_size=(9, 9), activation='relu', padding='same'),
    Conv2D(2048, kernel_size=(9, 9), activation='relu', padding='same'),
    Conv2D(2048, kernel_size=(9, 9), activation='relu', padding='same'),
    Conv2D(2048, kernel_size=(9, 9), activation='relu', padding='same'),
    Conv2D(256 * 2, kernel_size=(9, 9), activation='sigmoid', input_shape=input_shape, padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    #using ekponensial
    Conv2D(2048, kernel_size=(9, 9), activation='relu', padding='same'),
    Conv2D(512 * 2, kernel_size=(6, 6), activation='relu', padding='same'),
    Conv2D(2048, kernel_size=(9, 9), activation='relu', padding='same'),
    Conv2D(2048, kernel_size=(9, 9), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(3, 3), padding='same'),
    #using eksponensial
    Conv2D(energi2(), kernel_size=(9, 9), activation='sigmoid', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    Conv2D(2048, kernel_size=(9, 9), activation='relu', padding='same'),
    Conv2D(2048, kernel_size=(9, 9), activation='relu', padding='same'),
    Conv2D(2048, kernel_size=(9, 9), activation='relu', padding='same'),
    Conv2D(2048, kernel_size=(9, 9), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    Flatten(),
    #using eksponensial
    Dense(4096, activation='sigmoid'),
    Dropout(0.9),
    Dense(9 * 9 * 3, activation='sigmoid'),  # Output dalam rentang [0,1]
    Reshape((9, 9, 3))
])

reflektif(input_shape)
merenung(input_shape)

model.compile(optimizer=Adam(learning_rate=2.5), loss='CategoricalCrossentropy', metrics=['accuracy'])
model.fit(train_inputs, train_outputs, epochs=10, batch_size=10, validation_split=0.2)              # list of list of int

# Untuk 1 task_id → 2 prediksi per test case
def preprocess_input(input_grid):
    img = render(np.array(input_grid)).resize((9, 9))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 9, 9, 3)

def postprocess_output(output_tensor):
    output_array = output_tensor.squeeze()  # (9, 9, 3)
    output_gray = np.mean(output_array, axis=-1)  # (9, 9)
    output_scaled = np.rint(output_gray * 9).astype(int)  # Kembali ke 0-9
    return output_scaled.tolist()

def predict_output(task_id):
    task = testing_data[task_id]
    predictions = []

    for test_case in task["test"]:
        input_grid = test_case["input"]
        processed_input = preprocess_input(input_grid)

        output1 = model.predict(processed_input, verbose=0)
        output2 = model.predict(processed_input, verbose=0)

        pred1 = postprocess_output(output1)
        pred2 = postprocess_output(output2)

        predictions.append({
            "attempt_1": pred1,
            "attempt_2": pred2
        })

    return predictions

def generate_submission(output_path: str = 'submission.json') -> None:
    submission = {}
    for task_id in testing_data:
        submission[task_id] = predict_output(task_id)

    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=2)
    print(f"✅ Submission saved to {Path(output_path).absolute()}")

# Jalankan
generate_submission()
print("Submission file saved as 'submission.json' in /kaggle/working/ directory.")

def plotting():
    for idx, test_case_key in enumerate(selected_test_keys):
        predictions = model.predict(test_inputs)
        prediction = predictions[idx].reshape(output_shape)
        grayscale = prediction[1]  # Ambil channel 0, atau gunakan rata-rata saluran warna
        fig, ax = plt.subplots(figsize=(3, 3))
        img = ax.imshow(grayscale, cmap='gray')  # Ganti 'viridis' dengan 'jet', 'gray', dst
        plt.title("Prediction Final")
        plt.colorbar(img, ax=ax)  # Opsional: menampilkan skala warna
        plt.grid(False)
        plt.axis('off')
        plt.show()

print("Submission file saved as 'submission.json' in /kaggle/working/ directory.")
plotting()
# Konversi ke format TFLite
model.export("saved_model")  
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
tflite_model = converter.convert()
with open("/kaggle/working/models.tflite", "wb") as f:
    f.write(tflite_model)
print("/kaggle/working/models.tflite")

