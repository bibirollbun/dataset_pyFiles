import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import cv2
import os
from tqdm import tqdm

# Set seeds
tf.random.set_seed(42)
np.random.seed(42)



class Config:
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 30
    
config = Config()


train_df = pd.read_csv('/kaggle/input/petfinder-pawpularity-score/train.csv')
test_df = pd.read_csv('/kaggle/input/petfinder-pawpularity-score/test.csv')

def get_image_path(image_id, folder='train'):
    return f"/kaggle/input/petfinder-pawpularity-score/{folder}/{image_id}.jpg"

train_df['image_path'] = train_df['Id'].apply(lambda x: get_image_path(x, 'train'))
test_df['image_path'] = test_df['Id'].apply(lambda x: get_image_path(x, 'test'))

METADATA_FEATURES = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 
                    'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 
                    'Info', 'Blur']

print(f"Train: {len(train_df)}, Test: {len(test_df)}")


train_df['pawpularity_bin'] = pd.cut(train_df['Pawpularity'], bins=10, labels=False)
train_idx, val_idx = train_test_split(
    train_df.index, 
    test_size=0.2, 
    random_state=42,
    stratify=train_df['pawpularity_bin']
)

train_data = train_df.iloc[train_idx].reset_index(drop=True)
val_data = train_df.iloc[val_idx].reset_index(drop=True)

print(f"Training: {len(train_data)}, Validation: {len(val_data)}")


def extract_simple_features(df):
    """Extract simple but effective features without pre-trained models"""
    features = []
    print("Extracting features...")
    
    for i in tqdm(range(len(df))):
        img_path = df['image_path'].iloc[i]
        
        try:
            # Read and resize image
            image = cv2.imread(img_path)
            if image is None:
                # If image can't be read, use zeros
                features.append(np.zeros(19))
                continue
                
            image = cv2.resize(image, (128, 128))  # Smaller for speed
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Color features (7 features)
            color_mean = np.mean(image, axis=(0, 1)) / 255.0  # 3 features
            color_std = np.std(image, axis=(0, 1)) / 255.0    # 3 features
            brightness = np.mean(image) / 255.0               # 1 feature
            
            # Texture features (12 features)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            texture_features = [
                np.mean(sobelx), np.std(sobelx),
                np.mean(sobely), np.std(sobely),
                np.mean(gray) / 255.0, np.std(gray) / 255.0
            ]
            
            # Combine all features (7 color + 6 texture = 13 features)
            all_features = np.concatenate([
                color_mean, color_std, [brightness], texture_features
            ])
            
            features.append(all_features)
            
        except Exception as e:
            # If any error, use default features
            features.append(np.zeros(19))
    
    return np.array(features)

# Extract features for all datasets
print("Extracting training features...")
train_image_features = extract_simple_features(train_data)
print("Extracting validation features...")
val_image_features = extract_simple_features(val_data)
print("Extracting test features...")
test_image_features = extract_simple_features(test_df)

print(f"Feature shapes - Train: {train_image_features.shape}, Val: {val_image_features.shape}")


X_train = np.concatenate([train_image_features, train_data[METADATA_FEATURES].values], axis=1)
X_val = np.concatenate([val_image_features, val_data[METADATA_FEATURES].values], axis=1)
X_test = np.concatenate([test_image_features, test_df[METADATA_FEATURES].values], axis=1)

y_train = train_data['Pawpularity'].values
y_val = val_data['Pawpularity'].values

print(f"Final feature shapes:")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


!pip install xgboost
from xgboost import XGBRegressor

print("Training XGBoost model...")
xgb_model = XGBRegressor(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)


from sklearn.metrics import mean_squared_error
test_predictions = xgb_model.predict(X_test)
test_predictions = np.clip(test_predictions, 1, 100)

print(f"Test predictions - Min: {test_predictions.min():.2f}, Max: {test_predictions.max():.2f}")

submission = pd.DataFrame({
    'Id': test_df['Id'],
    'Pawpularity': test_predictions
})

submission.to_csv('submission.csv', index=False)
print("Submission created successfully!")




