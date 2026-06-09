import pandas as pd

df = pd.read_csv('/kaggle/input/detect-ai-vs-human-generated-images/test.csv')

print(df.info())

print(df.head())

print(df.isnull().sum())

print(df.nunique())


import pandas as pd

train_df = pd.read_csv('/kaggle/input/detect-ai-vs-human-generated-images/train.csv')

print(train_df.head())

print(train_df.columns)



import kagglehub

path = kagglehub.dataset_download("alessandrasala79/ai-vs-human-generated-dataset")

print("Path to dataset files:", path)


import pandas as pd

test_csv_path = "/kaggle/input/ai-vs-human-generated-dataset/test.csv"
test_df = pd.read_csv(test_csv_path)

dataset_path = "/kaggle/input/ai-vs-human-generated-dataset"
test_df["id"] = test_df["id"].apply(lambda x: x.replace("test_data/", "test_data_v2/"))  # Adjust paths

print(test_df.head())



import os

test_data_path = "/kaggle/input/ai-vs-human-generated-dataset/test_data_v2"


import cv2
import matplotlib.pyplot as plt

def show_test_images(df, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    for i in range(num_images):
        img_path = f"/kaggle/input/ai-vs-human-generated-dataset/{df.iloc[i]['id']}"  
        img = cv2.imread(img_path) 
        
        if img is None:
            print(f"Error loading: {img_path}")
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"Image {i+1}")
    
    plt.show()

show_test_images(test_df)


import pandas as pd
import os
import random

train_csv_path = "/kaggle/input/ai-vs-human-generated-dataset/train.csv"
train_df = pd.read_csv(train_csv_path)

dataset_path = "/kaggle/input/ai-vs-human-generated-dataset"

real_images = train_df[train_df['label'] == 0]['file_name'].tolist()
ai_images = train_df[train_df['label'] == 1]['file_name'].tolist()

random_real = random.sample(real_images, 5)
random_ai = random.sample(ai_images, 5)

selected_images = {"Real": random_real, "AI": random_ai}

selected_images


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift

def visualize_image_differences(ai_images, real_images, dataset_path):
    num_images = min(len(ai_images), len(real_images), 5)  
    fig, axes = plt.subplots(num_images, 4, figsize=(20, 5 * num_images))

    for i in range(num_images):
        for j, (label, img_list) in enumerate(zip(["AI Generated", "Real"], [ai_images, real_images])):
            img_path = os.path.join(dataset_path, img_list[i])
            img = cv2.imread(img_path)

            if img is None:
                print(f"Error loading image: {img_path}")
                continue

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            col_offset = j * 2  

            axes[i, col_offset].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i, col_offset].set_title(f"{label} Image {i+1}")
            axes[i, col_offset].axis("off")

            edges = cv2.Canny(img_gray, 100, 200)
            axes[i, col_offset+1].imshow(edges, cmap='gray')
            axes[i, col_offset+1].set_title(f"{label} Canny Edges")
            axes[i, col_offset+1].axis("off")

    plt.tight_layout()
    plt.show()

selected_ai_images = np.random.choice(ai_images, 5, replace=False).tolist()
selected_real_images = np.random.choice(real_images, 5, replace=False).tolist()

visualize_image_differences(selected_ai_images, selected_real_images, dataset_path)


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('/kaggle/input/ai-vs-human-generated-dataset/train.csv')

def compute_lbp(img_gray):
    lbp = np.zeros_like(img_gray)
    for x in range(1, img_gray.shape[0] - 1):
        for y in range(1, img_gray.shape[1] - 1):
            center = img_gray[x, y]
            binary_str = "".join(['1' if img_gray[x + dx, y + dy] > center else '0' 
                                   for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]])
            lbp[x, y] = int(binary_str, 2)
    return lbp

def visualize_lbp(ai_images, real_images, dataset_path):
    num_images = min(len(ai_images), len(real_images), 5)  
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))

    for i in range(num_images):
        for j, (label, img_list) in enumerate(zip(["AI Generated", "Real"], [ai_images, real_images])):
            img_path = os.path.join(dataset_path, os.path.basename(img_list[i]))
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Error loading image: {img_path}")
                continue

            lbp = compute_lbp(img)
            
            axes[i, j].imshow(lbp, cmap='gray')
            axes[i, j].set_title(f"{label} - LBP")
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()

dataset_path = '/kaggle/input/ai-vs-human-generated-dataset/train_data'
ai_images = df[df['label'] == 1]['file_name'].sample(5).tolist()
real_images = df[df['label'] == 0]['file_name'].sample(5).tolist()

visualize_lbp(ai_images, real_images, dataset_path)


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('/kaggle/input/ai-vs-human-generated-dataset/train.csv')

def compute_gabor(img_gray, ksize=5, sigma=1.0, theta=0, lambd=10.0, gamma=0.5, psi=0):
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    gabor_response = cv2.filter2D(img_gray, cv2.CV_8UC3, gabor_kernel)
    return gabor_response

def visualize_gabor(ai_images, real_images, dataset_path):
    num_images = min(len(ai_images), len(real_images), 5)  
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))

    for i in range(num_images):
        for j, (label, img_list) in enumerate(zip(["AI Generated", "Real"], [ai_images, real_images])):
            img_path = os.path.join(dataset_path, os.path.basename(img_list[i]))
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Error loading image: {img_path}")
                continue

            gabor = compute_gabor(img)
            
            axes[i, j].imshow(gabor, cmap='gray')
            axes[i, j].set_title(f"{label} - Gabor Filter")
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()

dataset_path = '/kaggle/input/ai-vs-human-generated-dataset/train_data'
ai_images = df[df['label'] == 1]['file_name'].sample(5).tolist()
real_images = df[df['label'] == 0]['file_name'].sample(5).tolist()

visualize_gabor(ai_images, real_images, dataset_path)


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt

df = pd.read_csv('/kaggle/input/ai-vs-human-generated-dataset/train.csv')

def compute_wavelet_noise(img_gray):
    coeffs2 = pywt.dwt2(img_gray, 'haar')
    _, (LH, HL, HH) = coeffs2  
    noise_residual = np.abs(HH)
    return noise_residual

def compute_dct_artifacts(img_gray):
    dct = cv2.dct(np.float32(img_gray))
    dct_log = np.log(np.abs(dct) + 1) 
    return dct_log

def compute_gabor(img_gray, ksize=5, sigma=1.0, theta=0, lambd=10.0, gamma=0.5, psi=0):
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    gabor_response = cv2.filter2D(img_gray, cv2.CV_8UC3, gabor_kernel)
    return gabor_response

def visualize_artifacts(ai_images, real_images, dataset_path):
    num_images = min(len(ai_images), len(real_images), 5)  
    fig, axes = plt.subplots(num_images, 6, figsize=(18, 5 * num_images))

    for i in range(num_images):
        for j, (label, img_list) in enumerate(zip(["AI Generated", "Real"], [ai_images, real_images])):
            img_path = os.path.join(dataset_path, os.path.basename(img_list[i]))
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Error loading image: {img_path}")
                continue

            noise_residual = compute_wavelet_noise(img)
            dct_artifacts = compute_dct_artifacts(img)
            gabor_response = compute_gabor(img)

            axes[i, j * 3].imshow(noise_residual, cmap='gray')
            axes[i, j * 3].set_title(f"{label} - Noise Residual")
            axes[i, j * 3].axis("off")

            axes[i, j * 3 + 1].imshow(dct_artifacts, cmap='gray')
            axes[i, j * 3 + 1].set_title(f"{label} - DCT Artifacts")
            axes[i, j * 3 + 1].axis("off")
            
            axes[i, j * 3 + 2].imshow(gabor_response, cmap='gray')
            axes[i, j * 3 + 2].set_title(f"{label} - Gabor Filter")
            axes[i, j * 3 + 2].axis("off")

    plt.tight_layout()
    plt.show()

dataset_path = '/kaggle/input/ai-vs-human-generated-dataset/train_data'
ai_images = df[df['label'] == 1]['file_name'].sample(5).tolist()
real_images = df[df['label'] == 0]['file_name'].sample(5).tolist()

visualize_artifacts(ai_images, real_images, dataset_path)


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift
import pandas as pd

df = pd.read_csv('/kaggle/input/ai-vs-human-generated-dataset/train.csv')

def visualize_image_differences(ai_images, real_images, dataset_path):
    num_images = min(len(ai_images), len(real_images), 5)  
    fig, axes = plt.subplots(num_images, 12, figsize=(30, 5 * num_images))

    for i in range(num_images):
        for j, (label, img_list) in enumerate(zip(["AI Generated", "Real"], [ai_images, real_images])):
            img_path = os.path.join(dataset_path, os.path.basename(img_list[i]))
            img = cv2.imread(img_path)

            if img is None:
                print(f"Error loading image: {img_path}")
                continue

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            col_offset = j * 6
            
            axes[i, col_offset].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i, col_offset].set_title(f"{label} - Original")
            axes[i, col_offset].axis("off")

            axes[i, col_offset + 1].imshow(img_gray, cmap='gray')
            axes[i, col_offset + 1].set_title("Grayscale")
            axes[i, col_offset + 1].axis("off")

            edges = cv2.Canny(img_gray, 100, 200)
            axes[i, col_offset + 2].imshow(edges, cmap='gray')
            axes[i, col_offset + 2].set_title("Canny Edges")
            axes[i, col_offset + 2].axis("off")

            fft_image = np.log(1 + np.abs(fftshift(fft2(img_gray))))
            axes[i, col_offset + 3].imshow(fft_image, cmap='gray')
            axes[i, col_offset + 3].set_title("FFT Spectrum")
            axes[i, col_offset + 3].axis("off")

            grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
            grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            axes[i, col_offset + 4].hist(gradient_magnitude.ravel(), bins=50, color='blue', alpha=0.7)
            axes[i, col_offset + 4].set_title("Gradient Histogram")

            lbp = np.zeros_like(img_gray)
            for x in range(1, img_gray.shape[0] - 1):
                for y in range(1, img_gray.shape[1] - 1):
                    center = img_gray[x, y]
                    binary_str = "".join(['1' if img_gray[x + dx, y + dy] > center else '0' 
                                           for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]])
                    lbp[x, y] = int(binary_str, 2)
            axes[i, col_offset + 5].imshow(lbp, cmap='gray')
            axes[i, col_offset + 5].set_title("Local Binary Pattern")
            axes[i, col_offset + 5].axis("off")

    plt.tight_layout()
    plt.show()

dataset_path = '/kaggle/input/ai-vs-human-generated-dataset/train_data'
ai_images = df[df['label'] == 1]['file_name'].sample(5).tolist()
real_images = df[df['label'] == 0]['file_name'].sample(5).tolist()

visualize_image_differences(ai_images, real_images, dataset_path)


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt

df = pd.read_csv('/kaggle/input/ai-vs-human-generated-dataset/train.csv')

def compute_wavelet_noise(img_gray):
    coeffs2 = pywt.dwt2(img_gray, 'haar')
    _, (LH, HL, HH) = coeffs2 
    noise_residual = np.abs(HH)
    return noise_residual

def compute_dct_artifacts(img_gray):
    dct = cv2.dct(np.float32(img_gray))
    dct_log = np.log(np.abs(dct) + 1)  
    return dct_log
    
def visualize_artifacts(ai_images, real_images, dataset_path):
    num_images = min(len(ai_images), len(real_images), 5)  
    fig, axes = plt.subplots(num_images, 4, figsize=(15, 5 * num_images))

    for i in range(num_images):
        for j, (label, img_list) in enumerate(zip(["AI Generated", "Real"], [ai_images, real_images])):
            img_path = os.path.join(dataset_path, os.path.basename(img_list[i]))
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Error loading image: {img_path}")
                continue

            noise_residual = compute_wavelet_noise(img)
            dct_artifacts = compute_dct_artifacts(img)

            axes[i, j * 2].imshow(noise_residual, cmap='gray')
            axes[i, j * 2].set_title(f"{label} - Noise Residual")
            axes[i, j * 2].axis("off")

            axes[i, j * 2 + 1].imshow(dct_artifacts, cmap='gray')
            axes[i, j * 2 + 1].set_title(f"{label} - DCT Artifacts")
            axes[i, j * 2 + 1].axis("off")

    plt.tight_layout()
    plt.show()

dataset_path = '/kaggle/input/ai-vs-human-generated-dataset/train_data'
ai_images = df[df['label'] == 1]['file_name'].sample(5).tolist()
real_images = df[df['label'] == 0]['file_name'].sample(5).tolist()

visualize_artifacts(ai_images, real_images, dataset_path)


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt

df = pd.read_csv('/kaggle/input/ai-vs-human-generated-dataset/train.csv')

def compute_wavelet_noise(img_gray):
    coeffs2 = pywt.dwt2(img_gray, 'haar')
    _, (LH, HL, HH) = coeffs2 
    noise_residual = np.abs(HH)
    return cv2.resize(noise_residual, (img_gray.shape[1], img_gray.shape[0]))

def compute_dct_artifacts(img_gray):
    dct = cv2.dct(np.float32(img_gray))
    dct_log = np.log(np.abs(dct) + 1)  
    return cv2.resize(dct_log, (img_gray.shape[1], img_gray.shape[0]))

def compute_gabor(img_gray, ksize=5, sigma=1.0, theta=0, lambd=10.0, gamma=0.5, psi=0):
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    gabor_response = cv2.filter2D(img_gray, cv2.CV_8UC3, gabor_kernel)
    return cv2.resize(gabor_response, (img_gray.shape[1], img_gray.shape[0]))

def visualize_feature_stack(ai_images, real_images, dataset_path):
    num_images = min(len(ai_images), len(real_images), 5)  
    fig, axes = plt.subplots(num_images, 4, figsize=(14, 5 * num_images))

    for i in range(num_images):
        for j, (label, img_list) in enumerate(zip(["AI Generated", "Real"], [ai_images, real_images])):
            img_path = os.path.join(dataset_path, os.path.basename(img_list[i]))
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Error loading image: {img_path}")
                continue

            noise_residual = compute_wavelet_noise(img)
            dct_artifacts = compute_dct_artifacts(img)
            gabor_response = compute_gabor(img)
 
            feature_stack = np.stack([img, noise_residual, dct_artifacts, gabor_response], axis=-1)
            feature_stack = np.mean(feature_stack, axis=-1) 
            
            axes[i, j * 2].imshow(img, cmap='gray')
            axes[i, j * 2].set_title(f"{label} - Original")
            axes[i, j * 2].axis("off")
            
            axes[i, j * 2 + 1].imshow(feature_stack, cmap='gray')
            axes[i, j * 2 + 1].set_title(f"{label} - Feature Stack")
            axes[i, j * 2 + 1].axis("off")

    plt.tight_layout()
    plt.show()

dataset_path = '/kaggle/input/ai-vs-human-generated-dataset/train_data'
ai_images = df[df['label'] == 1]['file_name'].sample(5).tolist()
real_images = df[df['label'] == 0]['file_name'].sample(5).tolist()

visualize_feature_stack(ai_images, real_images, dataset_path)


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt
from skimage.feature import graycomatrix, graycoprops

df = pd.read_csv('/kaggle/input/ai-vs-human-generated-dataset/train.csv')

def compute_wavelet_noise(img_gray):
    coeffs2 = pywt.dwt2(img_gray, 'haar')
    _, (LH, HL, HH) = coeffs2  
    noise_residual = np.abs(HH)
    return cv2.resize(noise_residual, (img_gray.shape[1], img_gray.shape[0]))

def compute_dct_artifacts(img_gray):
    dct = cv2.dct(np.float32(img_gray))
    dct_log = np.log(np.abs(dct) + 1) 
    return cv2.resize(dct_log, (img_gray.shape[1], img_gray.shape[0]))

def compute_gabor(img_gray, ksize=5, sigma=1.0, theta=0, lambd=10.0, gamma=0.5, psi=0):
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    gabor_response = cv2.filter2D(img_gray, cv2.CV_8UC3, gabor_kernel)
    return cv2.resize(gabor_response, (img_gray.shape[1], img_gray.shape[0]))

def compute_glcm_texture(img_gray):
    glcm = graycomatrix(img_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    texture_map = contrast + correlation + energy + homogeneity
    return np.full_like(img_gray, texture_map)

def visualize_feature_stack(ai_images, real_images, dataset_path):
    num_images = min(len(ai_images), len(real_images), 5)  
    fig, axes = plt.subplots(num_images, 6, figsize=(18, 5 * num_images))

    for i in range(num_images):
        for j, (label, img_list) in enumerate(zip(["AI Generated", "Real"], [ai_images, real_images])):
            img_path = os.path.join(dataset_path, os.path.basename(img_list[i]))
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Error loading image: {img_path}")
                continue

            noise_residual = compute_wavelet_noise(img)
            dct_artifacts = compute_dct_artifacts(img)
            gabor_response = compute_gabor(img)
            glcm_texture = compute_glcm_texture(img)
  
            axes[i, j * 3].imshow(img, cmap='gray')
            axes[i, j * 3].set_title(f"{label} - Original")
            axes[i, j * 3].axis("off")

            feature_stack = np.stack([noise_residual, dct_artifacts, gabor_response], axis=-1)
            feature_stack = np.mean(feature_stack, axis=-1) 
            axes[i, j * 3 + 1].imshow(feature_stack, cmap='gray')
            axes[i, j * 3 + 1].set_title(f"{label} - Feature Stack")
            axes[i, j * 3 + 1].axis("off")

            axes[i, j * 3 + 2].imshow(glcm_texture, cmap='gray')
            axes[i, j * 3 + 2].set_title(f"{label} - GLCM Texture")
            axes[i, j * 3 + 2].axis("off")

    plt.tight_layout()
    plt.show()

dataset_path = '/kaggle/input/ai-vs-human-generated-dataset/train_data'
ai_images = df[df['label'] == 1]['file_name'].sample(5).tolist()
real_images = df[df['label'] == 0]['file_name'].sample(5).tolist()

visualize_feature_stack(ai_images, real_images, dataset_path)


import os
import cv2
import numpy as np
import pandas as pd
import pywt
from tqdm import tqdm

df = pd.read_csv('/kaggle/input/ai-vs-human-generated-dataset/train.csv')

dataset_path = "/kaggle/input/ai-vs-human-generated-dataset/train_data"
output_dir = "/kaggle/working/preprocessed_dataset"
os.makedirs(output_dir, exist_ok=True)

def compute_wavelet_noise(img_gray):
    coeffs2 = pywt.dwt2(img_gray, 'haar')
    _, (LH, HL, HH) = coeffs2  
    return np.abs(HH)

def compute_dct_artifacts(img_gray):
    dct = cv2.dct(np.float32(img_gray))
    return np.log(np.abs(dct) + 1)

def compute_gabor(img_gray):
    gabor_kernel = cv2.getGaborKernel((5, 5), 1.0, 0, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    return cv2.filter2D(img_gray, cv2.CV_8UC3, gabor_kernel)

def process_and_save_images(image_list, dataset_path):
    for img_name in tqdm(image_list, desc="Processing Images", unit="image"):
        img_path = os.path.join(dataset_path, os.path.basename(img_name))  
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Error loading image: {img_path}")
            continue

        noise_residual = compute_wavelet_noise(img)
        dct_artifacts = compute_dct_artifacts(img)
        gabor_response = compute_gabor(img)

        height, width = img.shape
        noise_residual = cv2.resize(noise_residual, (width, height))
        dct_artifacts = cv2.resize(dct_artifacts, (width, height))
        gabor_response = cv2.resize(gabor_response, (width, height))

        noise_residual = cv2.normalize(noise_residual, None, 0, 255, cv2.NORM_MINMAX)
        dct_artifacts = cv2.normalize(dct_artifacts, None, 0, 255, cv2.NORM_MINMAX)
        gabor_response = cv2.normalize(gabor_response, None, 0, 255, cv2.NORM_MINMAX)

        feature_stack = np.dstack([noise_residual, dct_artifacts, gabor_response]).astype(np.uint8)

        save_path = os.path.join(output_dir, os.path.basename(img_name))  
        success = cv2.imwrite(save_path, feature_stack)

        if not success:
            print(f"Failed to save: {save_path}")

image_list = df['file_name'].tolist()
process_and_save_images(image_list, dataset_path)

print(f"✅ Preprocessing complete! Processed images saved in: {output_dir}")


import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

preprocessed_dir = "/kaggle/working/preprocessed_dataset"

image_files = [f for f in os.listdir(preprocessed_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

random_images = random.sample(image_files, min(5, len(image_files)))

fig, axes = plt.subplots(1, len(random_images), figsize=(15, 5))
for ax, img_name in zip(axes, random_images):
    img_path = os.path.join(preprocessed_dir, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) 

    if img is None:
        print(f"Error loading image: {img_path}")
        continue

    img_gray = np.mean(img, axis=-1).astype(np.uint8)

    ax.imshow(img_gray, cmap="gray")
    ax.set_title(img_name)
    ax.axis("off")

plt.tight_layout()
plt.show()



import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm

preprocessed_dir = "/kaggle/working/preprocessed_dataset"
csv_path = "/kaggle/input/detect-ai-vs-human-generated-images/train.csv"

df = pd.read_csv(csv_path)

df["file_name"] = df["file_name"].apply(lambda x: os.path.basename(x))

labels_dict = dict(zip(df["file_name"], df["label"]))

X, y = [], []
image_size = (64, 64) 

for img_name in os.listdir("/kaggle/working/preprocessed_dataset"):
    if img_name not in labels_dict:
        print(f" Missing label for: {img_name}")


for img_name in tqdm(os.listdir(preprocessed_dir), desc="Loading images"):
    
    if img_name in labels_dict: 
        img_path = os.path.join(preprocessed_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is not None:
            img = cv2.resize(img, image_size)
            img = img / 255.0  
            X.append(img)
            y.append(labels_dict[img_name]) 

X = np.array(X)
y = np.array(y)

if X.ndim == 3:
    X = np.expand_dims(X, axis=-1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),  
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=32)

model.save("/kaggle/working/ai_vs_real_classifier.h5")

print("Model training complete! Saved as 'ai_vs_real_classifier.h5'")


from tensorflow.keras.models import load_model

model = load_model("/kaggle/working/ai_vs_real_classifier.h5")


import cv2
import numpy as np
import pywt
import os

def compute_wavelet_noise(img_gray):
    coeffs2 = pywt.dwt2(img_gray, 'haar')
    _, (LH, HL, HH) = coeffs2  
    return cv2.resize(np.abs(HH), (img_gray.shape[1], img_gray.shape[0]))

def compute_dct_artifacts(img_gray):
    dct = cv2.dct(np.float32(img_gray))
    return cv2.resize(np.log(np.abs(dct) + 1), (img_gray.shape[1], img_gray.shape[0]))

def compute_gabor(img_gray):
    gabor_kernel = cv2.getGaborKernel((5, 5), 1.0, 0, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    return cv2.resize(cv2.filter2D(img_gray, cv2.CV_8UC3, gabor_kernel), (img_gray.shape[1], img_gray.shape[0]))

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(" Error: Image not found!")
        return None

    noise_residual = compute_wavelet_noise(img)
    dct_artifacts = compute_dct_artifacts(img)
    gabor_response = compute_gabor(img)

    noise_residual = cv2.normalize(noise_residual, None, 0, 255, cv2.NORM_MINMAX)
    dct_artifacts = cv2.normalize(dct_artifacts, None, 0, 255, cv2.NORM_MINMAX)
    gabor_response = cv2.normalize(gabor_response, None, 0, 255, cv2.NORM_MINMAX)

    feature_stack = np.dstack([noise_residual, dct_artifacts, gabor_response]).astype(np.uint8)

    feature_stack = cv2.resize(feature_stack, (64, 64))

    feature_stack = feature_stack / 255.0
    return feature_stack, img


import matplotlib.pyplot as plt

test_image_path = "/kaggle/input/ai-vs-human-generated-dataset/test_data_v2/0016e1d72d404fe68074cc87cb30aa37.jpg"

input_image, original_image = preprocess_image(test_image_path)

if input_image is not None:
    input_image = np.expand_dims(input_image, axis=0)

    prediction = model.predict(input_image)[0][0]

    label = "AI Generated" if prediction > 0.5 else "Real"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    print(f"Prediction: {label} (Confidence: {confidence:.2f})")

    plt.figure(figsize=(6, 6))
    plt.imshow(original_image, cmap="gray")
    plt.title(f"Prediction: {label}\nConfidence: {confidence:.2f}")
    plt.axis("off")
    plt.show()


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

test_dir = "/kaggle/input/ai-vs-human-generated-dataset/test_data_v2"
output_csv = "/kaggle/working/submission.csv"

model = tf.keras.models.load_model("/kaggle/working/ai_vs_real_classifier.h5")

image_size = (64, 64)

predictions = []

for img_name in tqdm(os.listdir(test_dir), desc="Processing images"):
    img_path = os.path.join(test_dir, img_name)

    input_image, original_image = preprocess_image(img_path)

    if input_image is not None:
        input_image = np.expand_dims(input_image, axis=0)
        prediction = model.predict(input_image, verbose=0)[0][0]
        label = "AI Generated" if prediction > 0.5 else "Real"
        predictions.append([img_name, label])

df_predictions = pd.DataFrame(predictions, columns=["image_name", "predicted_label"])
df_predictions.to_csv(output_csv, index=False)

print(f"Predictions saved to {output_csv}")


import pandas as pd

csv_path = "/kaggle/working/submission.csv"

df = pd.read_csv(csv_path)
print(df)

