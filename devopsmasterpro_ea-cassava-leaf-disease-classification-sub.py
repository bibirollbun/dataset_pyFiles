import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pathlib import Path

TEST_DIR = Path("/kaggle/input/cassava-leaf-disease-classification/test_images")
submission = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv")
submission.head()


from tensorflow.keras.models import load_model

model = load_model("/kaggle/input/ea-cassava-leaf-disease-classification-models/efficientnetb0_cassava.h5")
print("Model loaded successfully!")



from tensorflow.keras.applications.efficientnet import preprocess_input

test_images = []
image_ids = []

for img_path in TEST_DIR.iterdir():
    if img_path.suffix == ".jpg":
        image_ids.append(img_path.name)
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = preprocess_input(img)  # same as training
        test_images.append(img)

test_images = np.array(test_images)
print("Loaded test images:", test_images.shape)



from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from pathlib import Path
import pandas as pd
import numpy as np

# 1. Paths & image size (match training)
IMG_SIZE = (224, 224)
TEST_DIR = Path("/kaggle/input/cassava-leaf-disease-classification/test_images")

# 2. Build DataFrame of test images (sorted for stable order)
test_files = sorted(TEST_DIR.glob("*.jpg"))
test_df = pd.DataFrame({
    "image_id": [p.name for p in test_files],
    "filepath": [str(p) for p in test_files],
})

# 3. Test generator – use SAME preprocessing as training
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_gen = test_datagen.flow_from_dataframe(
    test_df,
    x_col="filepath",
    y_col=None,
    target_size=IMG_SIZE,
    class_mode=None,
    batch_size=32,
    shuffle=False
)

# 4. Predict
preds = model.predict(test_gen, verbose=1)
labels = np.argmax(preds, axis=1)



sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

# Make sure order matches (it should, since we sorted filenames)
print(sub.head())
print(test_df.head())

# Fill label column
sub["label"] = labels

# Save submission
sub.to_csv("submission.csv", index=False)
print("submission.csv saved!")
sub.head()




