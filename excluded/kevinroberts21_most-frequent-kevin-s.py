import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames[:10]:
        print(os.path.join(dirname, filename))



input_path = '/kaggle/input/cassava-leaf-disease-classification'
train_csv_filename = 'train.csv'

train_data = pd.read_csv(os.path.join(input_path, train_csv_filename))
train_data


plant_dict = {"0": "Cassava Bacterial Blight (CBB)", 
              "1": "Cassava Brown Streak Disease (CBSD)", 
              "2": "Cassava Green Mottle (CGM)", 
              "3": "Cassava Mosaic Disease (CMD)", 
              "4": "Healthy"}


first_plant_name = plant_dict[str(train_data.iloc[0, 1])]
print("The plant in row 50 is classified as: " + str(first_plant_name))


most_frequent = train_data.iloc[:, 1].mode()[0] # the most frequently classified plant in the train.csv
print("The most frequently occurring plant is: " + str(plant_dict[str(most_frequent)]))


test_images_folder = "test_images"
test_image_names = os.listdir(os.path.join(input_path, test_images_folder))

# Creating a submission Dataframe manually
submission = pd.DataFrame({
    "image_id": test_image_names,
    "label": most_frequent
})

# Save the submission file
submission.to_csv("most_frequent_submission.csv", index=False)







