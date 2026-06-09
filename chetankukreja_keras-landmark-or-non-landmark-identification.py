import os
import numpy as np
import pandas as pd 
from PIL import Image
from cv2 import resize
import matplotlib.pyplot as plt
print(os.listdir("../input"))


# VGG 16 Places 365 scripts in custom dataset
os.chdir("/kaggle/input/keras-vgg16-places365/")
from vgg16_places_365 import VGG16_Places365
os.chdir("/kaggle/working/")


# Get List of Images
image_samples = '../input/google-landmark-2019-samples/'
all_images = os.listdir(image_samples)

# Resize all images
all_images_resized = []
for filename in all_images:    
    im = np.array(Image.open(image_samples + filename).resize((224, 224), Image.LANCZOS))    
    all_images_resized.append(im)


# Plot image examples
fig = plt.figure(figsize = (16, 32))
for index, im in zip(range(1, len(all_images_resized)+1), all_images_resized):
    fig.add_subplot(10, 5, index)
    plt.title(filename)
    plt.imshow(im)   


# Placeholders for predictions
p0, p1, p2 = [], [], []

# Places365 Model
model = VGG16_Places365(weights='places')
topn = 5

# Loop through all images
for image in all_images_resized:
    
    # Predict Top N Image Classes
    image = np.expand_dims(image, 0)
    topn_preds = np.argsort(model.predict(image)[0])[::-1][0:topn]

    p0.append(topn_preds[0])
    p1.append(topn_preds[1])
    p2.append(topn_preds[2])

# Create dataframe for later usage
topn_df = pd.DataFrame()
topn_df['filename'] = np.array(all_images)
topn_df['p0'] = np.array(p0)
topn_df['p1'] = np.array(p1)
topn_df['p2'] = np.array(p2)
topn_df.to_csv('topn_class_numbers.csv', index = False)

# Summary
topn_df.head()


# Read Class number, class name and class indoor/outdoor marker
class_information = pd.read_csv('../input/keras-vgg16-places365/categories_places365_extended.csv')
class_information.head()

# Set Class Labels
for col in ['p0', 'p1', 'p2']:
    topn_df[col + '_label'] = topn_df[col].map(class_information.set_index('class')['label'])
    topn_df[col + '_landmark'] = topn_df[col].map(class_information.set_index('class')['io'].replace({1:'non-landmark', 2:'landmark'}))
topn_df.to_csv('topn_all_info.csv', index = False)

# Summary
topn_df.head()   



# Get 'landmark' images
n = 9
landmark_images =  topn_df[topn_df['p0_landmark'] == 'landmark']['filename']
landmark_indexes = landmark_images[:n].index.values

# Plot image examples
fig = plt.figure(figsize = (16, 16))
for index, im in zip(range(1, n+1), [ all_images_resized[i] for i in landmark_indexes]):
    fig.add_subplot(3, 3, index)
    plt.title(filename)
    plt.imshow(im)


# Get 'non-landmark' images
n = 9
landmark_images =  topn_df[topn_df['p0_landmark'] == 'non-landmark']['filename']
landmark_indexes = landmark_images[:n].index.values

# Plot image examples
fig = plt.figure(figsize = (16, 16))
for index, im in zip(range(1, n+1), [ all_images_resized[i] for i in landmark_indexes]):
    fig.add_subplot(3, 3, index)
    plt.title(filename)
    plt.imshow(im)


# VGG 16 Places 365 scripts in custom dataset
os.chdir("/kaggle/input/keras-vgg16-places365/")
from vgg16_places_365 import VGG16_Places365
os.chdir("/kaggle/working/")


import numpy as np
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from vgg16_places_365 import VGG16_Places365
import os 

class BinaryLandmarkDetector:
    def __init__(self, class_info_path='categories_places365_extended.csv'):
        """
        Initialize the detector with the VGG16 Places365 model
        and load landmark classification mapping
        """
        # Load the pre-trained model
        self.model = VGG16_Places365(weights='places')
        # VGG 16 Places 365 scripts in custom dataset
        os.chdir("/kaggle/input/keras-vgg16-places365/")
        # Load and prepare landmark mapping
        self.class_info = pd.read_csv(class_info_path)
        os.chdir("/kaggle/working/")
        self.landmark_mapping = self.class_info.set_index('class')['io'].map({
            1: 'non-landmark',
            2: 'landmark'
        }).to_dict()
        
    def load_image_from_url(self, url):
        """Load image from URL"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            # Convert to RGB if image is in a different mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image from URL: {e}")
            return None
            
    def load_image_from_path(self, image_path):
        """Load image from local path"""
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image from path: {e}")
            return None
    
    def preprocess_image(self, image):
        """Prepare image for model input"""
        image = image.resize((224, 224), Image.LANCZOS)
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, 0)
        return image_array
    
    def predict_url(self, url):
        """Predict landmark/non-landmark from URL"""
        image = self.load_image_from_url(url)
        if image is None:
            return None
        return self.predict_image(image)
    
    def predict_path(self, image_path):
        """Predict landmark/non-landmark from local path"""
        image = self.load_image_from_path(image_path)
        if image is None:
            return None
        return self.predict_image(image)
    
    def predict_image(self, image):
        """Core prediction function"""
        processed_image = self.preprocess_image(image)
        
        # Get model predictions
        predictions = self.model.predict(processed_image)[0]
        top_index = np.argmax(predictions)
        
        # Get top 3 classes and their probabilities
        top_3_indices = np.argsort(predictions)[::-1][:3]
        top_3_probs = predictions[top_3_indices]
        
        # Get landmark/non-landmark classification
        is_landmark = self.landmark_mapping[top_index]
        
        # Get scene categories for top 3 predictions
        top_3_scenes = [self.class_info.iloc[idx]['label'] for idx in top_3_indices]
        
        return {
            'classification': is_landmark,
            'confidence': float(predictions[top_index]),
            'top_3_scenes': list(zip(top_3_scenes, top_3_probs.tolist()))
        }


# Example usage
def main():
    # Initialize detector
    detector = BinaryLandmarkDetector()
    
    # Example with URL
    url = "https://www.outdooractive.com/api/staticmap?i=129107929&size=xlarge"
    result = detector.predict_url(url)
    
    if result:
        print("\nResults:")
        print(f"Classification: {result['classification']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nTop 3 scene categories:")
        for scene, prob in result['top_3_scenes']:
            print(f"- {scene}: {prob:.2%}")

if __name__ == "__main__":
    main()




