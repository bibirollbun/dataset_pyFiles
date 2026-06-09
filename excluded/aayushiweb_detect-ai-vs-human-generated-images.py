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


train= pd.read_csv("/kaggle/input/detect-ai-vs-human-generated-images/train.csv")
train.head()


test= pd.read_csv("/kaggle/input/detect-ai-vs-human-generated-images/test.csv")
test.head()


pip install transformers torchvision pandas


pip install --update transformers


import os
import pandas as pd
from transformers import AutoProcessor, AutoModelForImageClassification
from torchvision import transforms
from PIL import Image
import torch


model_name = "dima806/ai_vs_real_image_detection"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)


preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    
    
])


test_data_dir = "/kaggle/input/ai-vs-human-generated-dataset"

# Update image paths in the test dataframe
test['id'] = test['id'].apply(lambda x: os.path.join(test_data_dir, x))


predictions=[]


%%time
# Perform inference on test images
for idx, row in test.iterrows():
    img_path = row['id']  # Path to the image
    try:
        # Open and preprocess image
        image = Image.open(img_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

        # Model prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_label = torch.argmax(outputs.logits, dim=-1).item()  # Get predicted class

        predictions.append((img_path, predicted_label))
    except Exception as e:
        print(f"Error processing {img_path}: {e}")


submission_df = pd.DataFrame(predictions, columns=["id", "label"])
submission_df = pd.DataFrame(predictions, columns=["id", "label"])

# Save to CSV for submission
submission_csv_path = "submission.csv"
submission_df.to_csv(submission_csv_path, index=False)
print(f"Submission file saved at {submission_csv_path}")
# Save to CSV for submission
submission_csv_path = "submission.csv"
submission_df.to_csv(submission_csv_path, index=False)
print(f"Submission file saved at {submission_csv_path}")


value_counts = submission_df['label'].value_counts()
value_counts


from PIL import Image
import torch

def predict_single_image(image_path, model, preprocess):
    try:
        # Open and preprocess the image
        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_label = torch.argmax(outputs.logits, dim=-1).item()  # Get predicted class

        return predicted_label
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Example usage
image_path = "/kaggle/input/ai-vs-human-generated-dataset/test_data/000e592b0a3e41068de4ab318da8b506.jpg"  # Replace with your image path
predicted_label = predict_single_image(image_path, model, preprocess)
print(f"Predicted label for the image: {predicted_label}")




