import numpy as np 
import pandas as pd 
import glob
import json
import os
import seaborn as sns
import cv2
import matplotlib.pyplot as plt


INPUT_BASE_FILES = glob.glob('../input/herbarium-2022-fgvc9/*')

train_metadata_json = INPUT_BASE_FILES[0]
sample_submission_csv = INPUT_BASE_FILES[1]
test_metadata_json = INPUT_BASE_FILES[2]
train_images_folder = INPUT_BASE_FILES[3]
test_images_folder = INPUT_BASE_FILES[4]



with open(train_metadata_json) as json_file:
    train_metadata = json.load(json_file)
    
with open(test_metadata_json) as json_file:
    test_metadata = json.load(json_file)


print(train_metadata.keys()) # A dictionary


print(test_metadata[:2]) # A list


for k,v in train_metadata.items():
    print(f'| Key : {k}   >>  Total values  : {len(v)} ')


gen = train_metadata.get('genera')
genera_dict = {}
for i in gen:
    genera_dict[i.get('genus_id')] = i.get('genus')


print('Sample Values of each keys .. \n')

print('[+] Images ---\n')
print(train_metadata.get('images')[0])
print('\n')
print('[+] Annotations ---\n')
print(train_metadata.get('annotations')[0])
print('\n')
print('[+] Categories ---\n')
print(train_metadata.get('categories')[0])
print('\n')
print('[+] Genera --- \n ')
print(train_metadata.get('genera')[0])
print('\n')
print('[+] Distances --- \n ')
print(train_metadata.get('distances')[0])
print('\n')
print('[+] Institutions --- \n ')
print(train_metadata.get('institutions')[0])
print('\n')
print('[+] License --- \n ')
print(train_metadata.get('license')[0])


# Image information
file_names = []
image_ids = []
genus_ids = []
genus_names = []
category_ids = []
institution_ids = []
image_paths = []

for i,j in zip(train_metadata.get('images'),train_metadata.get('annotations')):
    
    image_id_im = i.get('image_id')
    image_id_anno = j.get('image_id')
    
    if image_id_im == image_id_anno:
        file_name = i.get('file_name')
        genus_id = j.get('genus_id')
        category_id = j.get('category_id')
        institution_id = j.get('institution_id')
        
        
        file_names.append(file_name)
        image_ids.append(image_id_anno)
        genus_ids.append(genus_id)
        genus_names.append(genera_dict.get(genus_id))
        category_ids.append(category_id)
        institution_ids.append(institution_id)
        image_paths.append(os.path.join(train_images_folder,file_name))


training_images_df = pd.DataFrame.from_dict({'FileNames' : file_names, 'ImageID' : image_ids, 'GenusID' : genus_ids,'GenusNames':genus_names,
                                             'CategoryID' : category_ids,'InstitutionID' : institution_ids,'ImagePath':image_paths})


training_images_df.sample(5)


print('Genus ID information')
id,count = np.unique(genus_ids,return_counts=True)
genus_count_df = pd.DataFrame.from_dict({'Genus ID' : id,'Count' : count}).sort_values(by=['Count'],ascending=False)
genus_count_df['Count'].hist(bins=100, figsize=(18, 6), grid=True)
plt.title('Histogram of Genus ID counts')
plt.show()


print('Category ID Information') 
id,count = np.unique(category_ids,return_counts=True)
category_id_df = pd.DataFrame.from_dict({'Category ID' : id,'Count' : count}).sort_values(by=['Count'],ascending=False)
category_id_df['Count'].hist(bins=100, figsize=(18, 6), grid=True)
plt.title('Histogram of Category ID counts')
plt.show()


print('Institution ID information')
id,count = np.unique(institution_ids,return_counts=True)
institution_id_df = pd.DataFrame.from_dict({'Institution ID' : id,'Count' : count}).sort_values(by=['Count'],ascending=False)
institution_id_df['Count'].hist(bins=100, figsize=(18, 6), grid=True)
plt.title('Histogram of Institution ID counts')
plt.show()


def visualize_data(df,show_by='Random',genus_name = None):
    
    if show_by == 'Genus':
        df = df[df['GenusNames']==genus_name]
            
    data = df.sample(10)
    
    image_paths = data['ImagePath'].to_list()
    genus_ids = data['GenusNames'].to_list()
    category_ids = data['CategoryID'].to_list()
    institution_ids = data['InstitutionID'].to_list()
    
    plt.figure(figsize=(13,13))
    
    for indx,im in enumerate(image_paths):
        plt.subplot(2,5,indx+1)
        image = cv2.imread(im)
        plt.imshow(image[:,:,::-1])
        plt.title(f'GeniusNames :{genus_ids[indx]},\nCategoryID : {category_ids[indx]},\nInstitutionID : {institution_ids[indx]}')
        plt.axis('off')
    plt.tight_layout()


# Visualize random 10 image
visualize_data(training_images_df,show_by='Random')


# Visualize random 10 image for a particular genus
visualize_data(training_images_df,show_by='Genus',genus_name='Asimina')




