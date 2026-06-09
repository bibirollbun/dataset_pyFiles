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


# ğŸ“¦ Library Setup and Data Exploration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import os

# For image processing and embeddings
import cv2
from PIL import Image
import io
import base64
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up visualization style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("ğŸ�‰ Libraries loaded successfully!")
print(f"ğŸ—‹ Available input directories:")

# Explore available datasets
for dirname, _, filenames in os.walk('/kaggle/input'):
    if filenames:  # Only show directories with files
        print(f"  ğŸ“� {dirname}")
        for filename in filenames[:5]:  # Show first 5 files
            file_path = os.path.join(dirname, filename)
            try:
                size_mb = os.path.getsize(file_path) / (1024*1024)
                print(f"    ğŸ“„ {filename} ({size_mb:.2f} MB)")
            except:
                print(f"    ğŸ“„ {filename}")
        if len(filenames) > 5:
            print(f"    ... and {len(filenames)-5} more files")
        print()


# ğŸ“‹ Load and Explore Fashion MNIST Dataset
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf
from collections import Counter
import json

# Define fashion labels (simulating e-commerce product categories)
fashion_labels = {
    0: 'T-shirt/top',
    1: 'Trouser', 
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt', 
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# Load Fashion MNIST data
print("ğŸ“¶ Loading Fashion MNIST dataset...")
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Basic dataset info
print(f"ğŸ“Š Dataset Information:")
print(f"  Training samples: {X_train.shape[0]:,}")
print(f"  Test samples: {X_test.shape[0]:,}") 
print(f"  Image shape: {X_train.shape[1:]}")
print(f"  Classes: {len(fashion_labels)}")

# Create product catalog DataFrame (simulating e-commerce data)
print("\nğŸ�¦ Creating synthetic e-commerce product catalog...")

# Sample product descriptions and features
product_descriptions = {
    0: "Comfortable cotton T-shirt, perfect for casual wear. Available in multiple colors with soft fabric blend.",
    1: "Classic fit trousers with wrinkle-resistant fabric. Ideal for work or casual outings.",
    2: "Cozy pullover sweater, made from premium wool blend. Perfect for cooler weather.", 
    3: "Elegant dress suitable for various occasions. Features modern design and comfortable fit.",
    4: "Stylish coat with weather protection. Premium materials and contemporary styling.",
    5: "Comfortable sandals with ergonomic sole design. Perfect for summer and beach activities.",
    6: "Professional shirt with crisp collar and tailored fit. Suitable for business attire.",
    7: "Athletic sneakers with advanced cushioning technology. Ideal for sports and daily activities.",
    8: "Versatile bag with multiple compartments. Durable materials and functional design.",
    9: "Fashionable ankle boots with premium leather. Combines style and comfort."
}

# Create synthetic pricing and ratings
np.random.seed(42)
prices = np.random.uniform(15, 200, len(fashion_labels))
ratings = np.random.uniform(3.5, 5.0, len(fashion_labels))
reviews_count = np.random.randint(50, 1000, len(fashion_labels))

print(f"\nğŸ“‹ Product Catalog Overview:")
for i, (label, desc) in enumerate(product_descriptions.items()):
    print(f"  {fashion_labels[i]}: ${prices[i]:.2f} | â˜…{ratings[i]:.1f} ({reviews_count[i]} reviews)")
    print(f"    '{desc[:80]}...'")
    print()

