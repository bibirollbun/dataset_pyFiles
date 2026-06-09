!pip install autogluon

import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageStat
import autogluon.multimodal as agmm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Paths to csv
train = pd.read_csv('/kaggle/input/cidaut-ai-fake-scene-classification-2024/train.csv')
test = pd.read_csv('/kaggle/input/cidaut-ai-fake-scene-classification-2024/sample_submission.csv')

# Paths to images
train_path = '/kaggle/input/cidaut-ai-fake-scene-classification-2024/Train/'
test_path = '/kaggle/input/cidaut-ai-fake-scene-classification-2024/Test/'

# Update paths in DataFrames
train['image'] = train['image'].apply(lambda x: os.path.join(train_path, x))
test['image'] = test['image'].apply(lambda x: os.path.join(test_path, x))

# Binary encode the labels
cls_to_idx = {'editada': 0, 'real': 1}
train['label'] = train['label'].map(cls_to_idx)


# Blurriness feature using Laplacian variance
def blurriness(im_file):
    im = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)
    return cv2.Laplacian(im, cv2.CV_64F).var()

# Noise levels feature using standard deviation of residuals
def noise_level(im_file):
    """Calculate noise level as the standard deviation of residuals."""
    im = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(im, (5, 5), 0)
    residual = im - blurred
    return np.std(residual)

# Brightness feature using cv2
def brightness(im_file):
    im = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)
    return np.mean(im)

# Entropy feature using cv2
def entropy(im_file):
    im = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)
    histogram = cv2.calcHist([im], [0], None, [256], [0, 256]).flatten()
    hist_sum = np.sum(histogram)
    histogram = histogram / hist_sum  # Normalize histogram
    histogram = histogram[histogram > 0]  # Avoid log(0)
    return -np.sum(histogram * np.log2(histogram))

# Feature extraction
def extract_features(df):
    blurs, noises, brightnesses, entropies = [], [], [], []
    for path in df['image']:
        blurs.append(blurriness(path))
        noises.append(noise_level(path))
        brightnesses.append(brightness(path))
        entropies.append(entropy(path))
    return blurs, noises, brightnesses, entropies


train['blurriness'], train['noise_level'], train['brightness'], train['entropy'] = extract_features(train)
test['blurriness'], test['noise_level'], test['brightness'], test['entropy'] = extract_features(test)

# Normalize features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
feature_cols = ['blurriness', 'noise_level', 'brightness', 'entropy']
train[feature_cols] = scaler.fit_transform(train[feature_cols])
test[feature_cols] = scaler.transform(test[feature_cols])



# AutoGluon Model
predictor = agmm.MultiModalPredictor(
    label='label',
    eval_metric='roc_auc',
    problem_type='binary',
    presets='best_quality'  # Optimized settings for high-quality results
)

# Train the model
predictor.fit(
    train_data=train
)


preds = predictor.predict(test.drop(columns='label'), as_pandas=False)


sub = pd.read_csv('/kaggle/input/cidaut-ai-fake-scene-classification-2024/sample_submission.csv')

sub['label'] = preds
sub.to_csv('submission.csv', index=False)
print(sub.head())

