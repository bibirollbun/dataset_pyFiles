import os

deps_path = '/kaggle/input/segmentation-models-offline-deps/offline_sm_packages'
requirements_file_in_deps = os.path.join(deps_path, 'requirements.txt')
!pip install --no-index --find-links={deps_path} -r {requirements_file_in_deps} -q



os.environ["SM_FRAMEWORK"] = "tf.keras" 

import numpy as np
import pandas as pd
import cv2 
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import segmentation_models as sm 

TEST_IMAGE_DIR = '/kaggle/input/severstal-steel-defect-detection/test_images/'
NUM_CLASSES = 4
IMG_HEIGHT = 256
IMG_WIDTH = 256
TARGET_DIM = (IMG_HEIGHT, IMG_WIDTH)
N_TILES_PER_IMAGE = 6 
THRESHOLD = 0.5     

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def mask_to_rle(mask_img): 
    pixels = mask_img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

custom_objects = {
    'dice_loss': dice_loss,
    'dice_coefficient': dice_coefficient
}

model_path = '/kaggle/input/efficientnetb4-severstal-modelim/best_unet_model_with_backbone.keras' 
loaded_best_model = load_model(model_path, custom_objects=custom_objects)
print("En iyi model yeni notebook'ta başarıyla yüklendi.")


if 'loaded_best_model' in locals() or 'loaded_best_model' in globals():
    test_image_files = os.listdir(TEST_IMAGE_DIR)
    if not test_image_files:
        print("Test edilecek görüntü bulunamadı.")
    else:
        print(f"Test edilecek görüntü sayısı: {len(test_image_files)}")
        submission_data = []
        original_height = 256
        original_width = 1600

        for img_file in tqdm(test_image_files):
            img_path = os.path.join(TEST_IMAGE_DIR, img_file)
            img = cv2.imread(img_path) 

            tiles_for_prediction_batch = []

            for tile_idx in range(N_TILES_PER_IMAGE):
                start_col = tile_idx * TARGET_DIM[1] 
                end_col = start_col + TARGET_DIM[1]

                img_tile = img[:, start_col:end_col, :]
                img_tile_processed = img_tile.astype(np.float32) / 255.0
                tiles_for_prediction_batch.append(img_tile_processed)
            
            batch_of_tiles = np.array(tiles_for_prediction_batch)

            predicted_tiles_probs = loaded_best_model.predict(batch_of_tiles, batch_size=N_TILES_PER_IMAGE, verbose=0)

            
            full_pred_prob_mask = np.zeros((original_height, original_width, NUM_CLASSES), dtype=np.float32)
            for tile_idx in range(N_TILES_PER_IMAGE):
                start_col = tile_idx * TARGET_DIM[1]
                end_col = start_col + TARGET_DIM[1]
                full_pred_prob_mask[:, start_col:end_col, :] = predicted_tiles_probs[tile_idx]

            full_pred_binary_mask = (full_pred_prob_mask > THRESHOLD).astype(np.uint8)

            for class_id_idx in range(NUM_CLASSES):
                class_id_actual = class_id_idx + 1
                rle_encoded_pixels = mask_to_rle(full_pred_binary_mask[:, :, class_id_idx])

                if len(rle_encoded_pixels) == 0:
                    rle_encoded_pixels = ''

                submission_data.append({
                    'ImageId_ClassId': f"{img_file}_{class_id_actual}",
                    'EncodedPixels': rle_encoded_pixels
                })

        submission_df = pd.DataFrame(submission_data)
        print("\nÖrnek Submission Satırları:")
        print(submission_df.head())

        submission_df.to_csv('submission.csv', index=False)
        print("\nsubmission.csv dosyası başarıyla oluşturuldu!") 
else:
    print("Model yüklenemediği için submission dosyası oluşturulamadı.")




