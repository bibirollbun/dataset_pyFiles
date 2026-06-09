# DRW - Crypto Market Prediction Submission Template

import numpy as np
import pandas as pd

# ✅ Step 1: Define total number of prediction rows
NUM_ROWS = 538150  # As per competition requirement

# ✅ Step 2: Create the submission dataframe
submission = pd.DataFrame({
    'row_id': np.arange(NUM_ROWS),
    'label': np.random.normal(loc=0.0, scale=0.001, size=NUM_ROWS)  # Placeholder prediction values
})

# ✅ Step 3: Save to CSV (for direct submission)
submission.to_csv("submission.csv", index=False)

# ✅ Step 4 (Optional): Save to ZIP if you want to upload that format
import zipfile

with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write("submission.csv")

# Output confirmation
print("submission.csv created with shape:", submission.shape)



# DRW - Crypto Market Prediction - Submission Generator

import numpy as np
import pandas as pd
import os
import zipfile

# List all input files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# STEP 1: Set required number of rows from the competition
NUM_ROWS = 538150  # Make sure this is exact

# STEP 2: Generate placeholder predictions (replace this later with model output)
submission_df = pd.DataFrame({
    'row_id': np.arange(NUM_ROWS),
    'label': np.random.normal(loc=0.0, scale=0.001, size=NUM_ROWS)  # Simulated returns
})

# STEP 3: Save CSV for submission
submission_csv_path = '/kaggle/working/submission.csv'
submission_df.to_csv(submission_csv_path, index=False)
print(f"✅ submission.csv saved at {submission_csv_path} with shape {submission_df.shape}")

# OPTIONAL: Create a ZIP file (if preferred)
submission_zip_path = '/kaggle/working/submission.zip'
with zipfile.ZipFile(submission_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(submission_csv_path, arcname='submission.csv')
print(f"✅ submission.zip created at {submission_zip_path}")



submission_df.to_csv('/kaggle/working/submission.csv', index=False)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



pip install kaggle



{
  "id": "ishitabahamnia/DRW - Crypto Market Prediction Submission Template",
  "title": "My DRW Crypto Submission",
  "code_file": "DRW - Crypto Market Prediction Submission Template.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true"
}



import pandas as pd
import numpy as np

# Simulate training dataset
np.random.seed(42)
n_rows = 5000
df_train = pd.DataFrame({
    "ID": np.arange(1, n_rows + 1),
    "feature1": np.random.randn(n_rows),
    "feature2": np.random.rand(n_rows) * 100,
    "feature3": np.random.randint(0, 2, n_rows),
    "label": np.random.randint(0, 2, n_rows)
})

# Simulate test dataset
n_test_rows = 2000
df_test = pd.DataFrame({
    "ID": np.arange(10001, 10001 + n_test_rows),
    "feature1": np.random.randn(n_test_rows),
    "feature2": np.random.rand(n_test_rows) * 100,
    "feature3": np.random.randint(0, 2, n_test_rows)
})

# Save locally
df_train.to_csv("train.csv", index=False)
df_test.to_csv("test.csv", index=False)



"First LSTM model with precomputed features"




import tensorflow as tf
from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist
## use Fashion MNIST data set for example
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0

test_images = test_images / 255.0
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

