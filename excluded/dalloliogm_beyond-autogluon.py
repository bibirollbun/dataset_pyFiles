# Install necessary libraries
!pip install -q autogluon


import autogluon.core as ag
#from autogluon import ImagePredictor
import pandas as pd
import numpy as np
import os
from autogluon.multimodal import MultiModalPredictor
from PIL import Image



def is_interactive():
   return os.environ.get('KAGGLE_KERNEL_RUN_TYPE','') == "Interactive"
print("is interactive session?", is_interactive())
preset_quality = "medium_quality" if is_interactive() else "high_quality"

time_limit = 60 if is_interactive() else 3600



# Read the train.csv file that contains the labels
train_df = pd.read_csv('/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/train.csv')

print(train_df.head())


image_folder = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/ot/ot"
# Define the output directory where you will save the images
output_dir = '/kaggle/working/hyperspectral_images/'

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to load .npy files
def load_npy_image(image_name,
                  image_folder = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/ot/ot"):
    image_path = os.path.join(image_folder, image_name)
    image = np.load(image_path)
    return image

# Convert the .npy files to image files and save them
def save_image_from_npy(image_name, image_data,
                       output_dir = '/kaggle/working/hyperspectral_images/'
                       ):
    # Ensure image has correct dimensions (e.g., 128x128x125)
    if image_data.shape == (128, 128, 125):
        # Normalize the data to the range 0-255
        # Here, we'll just take the first band for simplicity
        image = image_data[:, :, 0]  # Extract the first band (change if you want to visualize different bands)
        image = np.clip(image, 0, 255)  # Clip values to valid image range
        image = image.astype(np.uint8)  # Convert to unsigned 8-bit integer type
        image_path = os.path.join(output_dir, image_name.replace('.npy', '.png'))
        # Convert numpy array to image using PIL
        pil_image = Image.fromarray(image)
        pil_image.save(image_path)
        #print(f"Saved {image_path}")
    else:
        print(f"Skipping {image_name} due to unexpected shape {image_data.shape}")

# Loop through the training data and save the images
for idx, row in train_df.iterrows():
    try:
        # Load the .npy image
        image = load_npy_image(row['id'])
        
        # Check the shape of the image and reshape if necessary
        save_image_from_npy(row['id'], image)
    
    except ValueError as e:
        print(f"Error loading {row['id']}: {e}")
        continue  # Skip the image and continue with the next one
        
# Map the image filenames to the new .png file paths
train_df['image_path'] = train_df['id'].apply(lambda x: os.path.join(output_dir, x.replace('.npy', '.png')))



import matplotlib.pyplot as plt

def plot_image(train_df, row_id):
    img = Image.open(train_df.iloc[row_id,2])
    plt.imshow(img)
    plt.title(f"{train_df.iloc[row_id,0]} - {train_df.iloc[row_id,1]}")
plot_image(train_df, 6)


import matplotlib.pyplot as plt
from PIL import Image

# Define a function to plot images in a grid
def plot_images_in_grid(train_df, row_ids, grid_size=(10, 7)):
    # Create a figure with a specified size
    fig, axes = plt.subplots(nrows=grid_size[0], ncols=grid_size[1], figsize=(20, 14))
    
    # Flatten the axes array for easier iteration
    axes = axes.flatten()
    
    # Loop through the row_ids and plot images
    for i, row_id in enumerate(row_ids):
        img = Image.open(train_df.iloc[row_id, 2])  # Load the image from the path
        axes[i].imshow(img)  # Display the image
        axes[i].set_title(f"{train_df.iloc[row_id, 0]} - {train_df.iloc[row_id, 1]}")
        axes[i].axis('off')  # Turn off the axis to keep the image clean
    
    # Adjust the layout to prevent overlapping titles and images
    plt.tight_layout()
    plt.show()

# Example usage:
# Choose the first 70 row IDs for plotting
row_ids = list(range(70))

# Call the function to plot the images
plot_images_in_grid(train_df, row_ids, grid_size=(10, 7))  # Adjust grid_size as needed



# Now the dataframe has 'image_path' and 'label' columns
# Preview the updated dataframe
print(train_df.head())

train_df.iloc[0,2]



!ls /kaggle/working/hyperspectral_images/sample697.png


from autogluon.multimodal import MultiModalPredictor
#from autogluon.vision import ImagePredictor

# Initialize AutoGluon MultiModalPredictor with label column name
predictor = MultiModalPredictor(label="label")

# Perform k-fold cross-validation (e.g., 5-fold)
cv_results = predictor.fit(
    train_data=train_df.drop(columns="id"),
    time_limit=time_limit,
    keep_only_best=True,
    problem_type="regression",
    save_space=True, # Sometimes autogluon uses all the disk space, and the submission file is not generated
    presets=preset_quality
)



cv_results.fit_summary()


test_df = pd.read_csv('/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/test.csv')

# Loop through the training data and save the images
for idx, row in test_df.iterrows():
    try:
        # Load the .npy image
        image = load_npy_image(row['id'])
        
        # Check the shape of the image and reshape if necessary
        save_image_from_npy(row['id'], image)
    
    except ValueError as e:
        print(f"Error loading {row['id']}: {e}")
        continue  # Skip the image and continue with the next one


test_df


      
# Map the image filenames to the new .png file paths
test_df['image_path'] = test_df['id'].apply(lambda x: os.path.join(output_dir, x.replace('.npy', '.png')))


test_df


!ls /kaggle/working/hyperspectral_images/sample1957.*


# Make predictions using the trained AutoGluon model
# Ensure we're passing the entire DataFrame, not just the image paths column
predictions = predictor.predict(test_df[['image_path']])

# Prepare the submission DataFrame
submission_df = pd.DataFrame({
    'ID': test_df['id'],  # The IDs from the test.csv
    'label': predictions   # The predicted labels from the model
})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

# Optionally, show the first few rows of the submission file
print(submission_df.head())



!ls 


import seaborn as sns
sns.histplot(data=train_df, x="label").set_title("distribution of labels in Train")



import seaborn as sns
sns.histplot(data=submission_df, x="label").set_title("distribution of predicted labels")


