!pip install autogluon


import pandas as pd
import numpy as np
import autogluon.multimodal as agmm
from PIL import Image, ImageStat
import cv2


train = pd.read_csv('/kaggle/input/cidaut-ai-fake-scene-classification-2024/train.csv')
test = pd.read_csv('/kaggle/input/cidaut-ai-fake-scene-classification-2024/sample_submission.csv')
sub = pd.read_csv('/kaggle/input/cidaut-ai-fake-scene-classification-2024/sample_submission.csv')

train_path = '/kaggle/input/cidaut-ai-fake-scene-classification-2024/Train/'
test_path = '/kaggle/input/cidaut-ai-fake-scene-classification-2024/Test/'


# label binary encoding
cls_to_idx = {'editada': 0, 'real': 1}

train['label'] = [cls_to_idx[x] for x in train['label'].values]

train['label'].head()


train['image'] = train_path + train['image']
test['image'] = test_path + test['image']

train['image'].iloc[0], test['image'].iloc[0]


import pandas as pd
import numpy as np
from PIL import Image, ImageStat
import cv2
from scipy.stats import entropy as sk_entropy

# --- Feature Functions ---

def brightness(im_file):
    im = Image.open(im_file).convert('L')
    stat = ImageStat.Stat(im)
    return stat.mean[0]

def blur(im_file):
    img = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.nan
    return cv2.Laplacian(img, cv2.CV_64F).var()

def contrast(im_file):
    im = Image.open(im_file).convert('L')
    stat = ImageStat.Stat(im)
    return stat.stddev[0]

def sharpness(im_file):
    img = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.nan
    return cv2.Laplacian(img, cv2.CV_64F).var()

def image_entropy(im_file):
    img = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.nan
    hist = cv2.calcHist([img], [0], None, [256], [0,256]).ravel()
    hist = hist / hist.sum()
    return sk_entropy(hist + 1e-10)

def colorfulness(im_file):
    img = cv2.imread(im_file)
    if img is None:
        return np.nan
    (B, G, R) = cv2.split(img.astype("float"))
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_root = np.sqrt(np.std(rg)**2 + np.std(yb)**2)
    mean_root = np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    return std_root + 0.3 * mean_root

def saturation(im_file):
    img = cv2.imread(im_file)
    if img is None:
        return np.nan
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv[..., 1].mean()

# --- General Feature Extraction Helper ---

def get_all_features(df, feature_fn):
    features = []
    for path in df['image'].values:
        features.append(feature_fn(path))
    return features

# --- Apply All Features to Train/Test ---

feature_functions = {
    'brightness': brightness,
    'blur': blur,
    'contrast': contrast,
    'sharpness': sharpness,
    'entropy': image_entropy,
    'colorfulness': colorfulness,
    'saturation': saturation
}

for name, func in feature_functions.items():
    train[name] = get_all_features(train, func)
    test[name] = get_all_features(test, func)



train_1 = train[train['label']==1]
train_0 = train[train['label']==0]


train_1['brightness'].plot(kind='hist', bins=100)


train_1['blur'].plot(kind='hist', bins=100)


train_0['brightness'].plot(kind='hist', bins=100)


train_0['blur'].plot(kind='hist', bins=100)


predictor = agmm.MultiModalPredictor(label='label', eval_metric='roc_auc')
predictor.fit(train, hyperparameters={'env.num_gpus': 0})



preds = predictor.predict(test.drop(columns='label'), as_pandas=False)


sub['label'] = preds
sub.to_csv('submission_extract_many_features.csv', index=False)


sub['label'].plot(kind='hist',bins=100)


