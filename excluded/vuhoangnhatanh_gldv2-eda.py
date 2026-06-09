import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import cv2

from tqdm import tqdm_notebook as tqdm
import matplotlib.image as mpimg

import plotly.graph_objects as go


train_df = pd.read_csv('/kaggle/input/landmark-recognition-2021/train.csv')

test_list = glob.glob('/kaggle/input/landmark-recognition-2021/test/*/*/*/*')
train_list= glob.glob('/kaggle/input/landmark-recognition-2021/train/*/*/*/*')

landmark_info = pd.read_csv('/kaggle/input/mappings-gldv2/train_label_to_category.csv')


print(landmark_info[['landmark_id', 'category']].head())


landmark_info.to_csv('train_label_to_hierarchical_modified.csv', index=False)


landmark_info['landmark_id'] = landmark_info['landmark_id'] + 1


landmark_info_modified = pd.read_csv('/kaggle/working/train_label_to_hierarchical_modified.csv')


print("Train data shape -  rows:",train_df.shape[0]," columns:", train_df.shape[1])


train_df.head()


landmark_counts = train_df['landmark_id'].value_counts().head(50)
landmark_counts_least = train_df['landmark_id'].value_counts().tail(50)


plt.figure(figsize=(10, 6))
sns.histplot(train_df['landmark_id'].value_counts(), bins=50, kde=True)
plt.title('Distribution of Landmark Appearances')
plt.xlabel('Number of Appearances')
plt.ylabel('Count')
plt.show()


sns.set()
plt.title('Training set: number of images per class(line plot)')
landmarks_fold = pd.DataFrame(train_df['landmark_id'].value_counts())
landmarks_fold.reset_index(inplace=True)
landmarks_fold.columns = ['landmark_id','count']
ax = landmarks_fold['count'].plot(logy=True, grid=True)
locs, labels = plt.xticks()
plt.setp(labels, rotation=30)
ax.set(xlabel="Landmarks", ylabel="Number of images")


plt.figure(figsize = (8, 2))
plt.title('Landmark id density plot')
sns.kdeplot(train_df['landmark_id'], color="tomato", shade=True)
plt.show()


num_images = min(12, len(test_list))

plt.rcParams["axes.grid"] = False
f, axarr = plt.subplots(4, 3, figsize=(24, 22))

curr_row = 0
for i in range(num_images):  # Loop over the number of images, up to 12 or length of the list
    example = cv2.imread(test_list[i])
    example = example[:,:,::-1]  
    
    col = i % 3 
    axarr[curr_row, col].imshow(example)
    axarr[curr_row, col].axis('off')  # Hide axes for cleaner images
    
    if col == 2:  # Move to next row after 3 columns
        curr_row += 1

plt.show()


plt.rcParams["axes.grid"] = True
f, axarr = plt.subplots(6, 5, figsize=(24, 22))

curr_row = 0
for i in range(30):
    example = cv2.imread(train_list[i])
    example = example[:,:,::-1]
    
    col = i%6
    axarr[col, curr_row].imshow(example)
    if col == 5:
        curr_row += 1


missing_landmark_ids = train_df[~train_df['landmark_id'].isin(landmark_info['landmark_id'])]

# Count how many unique landmark_id values are missing
missing_landmark_count = missing_landmark_ids['landmark_id'].nunique()

print("Total number of missing landmark IDs:", missing_landmark_count)


temp = pd.DataFrame(train_df.landmark_id.value_counts().head(10))
temp.reset_index(inplace=True)
temp.columns = ['landmark_id', 'count']
temp


temp = pd.DataFrame(train_df.landmark_id.value_counts().head(50))
temp.reset_index(inplace=True)
temp.columns = ['landmark_id','count']
temp


temp = pd.DataFrame(train_df.landmark_id.value_counts().head(100))
temp.reset_index(inplace=True)
temp.columns = ['landmark_id', 'count']

# Define the chunk size
chunk_size = 50

# Print the data in chunks
for start in range(50, len(temp), chunk_size):
    end = min(start + chunk_size, len(temp))  # To ensure we don't go out of bounds
    print(f"Showing rows {start} to {end - 1}:")
    print(temp.iloc[start:end])
    print("\n" + "="*50 + "\n")  # Se


sns.set()
# plt.figure(figsize=(9, 8))
plt.title('Most frequent landmarks')
sns.set_color_codes("pastel")
sns.barplot(x="landmark_id", y="count", data=temp,
            label="Count")
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.show()


pd.set_option('display.max_colwidth', None) 
merged_data = pd.merge(temp, landmark_info_modified, on='landmark_id', how='left')


print(merged_data[['landmark_id', 'category']])


print(landmark_info['landmark_id'].nunique())
print(train_df['landmark_id'].nunique())


missing_landmark_ids = train_df[~train_df['landmark_id'].isin(landmark_info_1['landmark_id'])]

print("Missing Landmark IDs in Metadata:")
print(missing_landmark_ids[['id', 'landmark_id']])


landmark_id_to_search = 138982
result = landmark_info[landmark_info['landmark_id'] == landmark_id_to_search]

# Display the result
print(result)


landmark_id_to_search = 1924
count_images = train_df[train_df['landmark_id'] == landmark_id_to_search].shape[0]

# Display the count
print(f"Number of images with landmark_id {landmark_id_to_search}: {count_images}")


landmark_id_to_search = 138982
image_ids_with_landmark = train_df[train_df['landmark_id'] == landmark_id_to_search]['id'].tolist()
result = landmark_info[landmark_info['landmark_id'] == landmark_id_to_search]

# Display the result
print(result)

image_paths_to_display = [train_list[i] for i in range(len(train_list)) if train_df['id'][i] in image_ids_with_landmark]

# Set up the plot to display images
plt.rcParams["axes.grid"] = True
f, axarr = plt.subplots(6, 5, figsize=(24, 22))

curr_row = 0
num_images_to_display = min(30, len(image_paths_to_display))  # Display at most 30 images, if available

for i in range(num_images_to_display):
    # Read the image (make sure to read the correct path)
    example = cv2.imread(image_paths_to_display[i])
    example = cv2.cvtColor(example, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib
    
    # Calculate row and column position in the subplot grid
    col = i % 5  # 5 columns
    axarr[curr_row, col].imshow(example)
    axarr[curr_row, col].axis('off')  # Turn off axis for better visualization
    
    # Move to the next row after 5 images
    if col == 4:
        curr_row += 1

# Show the plot with the images
plt.show()


landmark_id_0_rows = train_df[train_df['landmark_id'] == '0']

# Display the rows with landmark_id = 0
print(landmark_id_0_rows)


temp = pd.DataFrame(train_df.landmark_id.value_counts().tail(10))
temp.reset_index(inplace=True)
temp.columns = ['landmark_id', 'count']
temp


INPUT_PATH = os.path.join('..', 'input')
DATASET_PATH = os.path.join(INPUT_PATH, 'landmark-recognition-2021')
TRAIN_IMAGE_PATH = os.path.join(DATASET_PATH, 'train')
TEST_IMAGE_PATH = os.path.join(DATASET_PATH, 'test')
TRAIN_CSV_PATH = os.path.join(DATASET_PATH, 'train.csv')
SUBMISSION_CSV_PATH = os.path.join(DATASET_PATH, 'sample_submission.csv')

train_df = pd.read_csv(TRAIN_CSV_PATH)
print(f"{'--'*20} \n SNIPPET OF TRAINING DATA: \n {train_df.head()} \n {'--'*20} \n Number of rows in train data: {train_df.shape[0]} \n {'--'*20}")

submission_df = pd.read_csv(SUBMISSION_CSV_PATH)
print(f"{'--'*20} \n SNIPPET OF TEST DATA: \n {submission_df.head()} \n {'--'*20} \n Number of rows in test data: {submission_df.shape[0]} \n {'--'*20}")

print(f"EXAMPLE FOR LANDMARK-LABEL MAPPING FOR [17660ef415d37059.jpg] \n FOLDER STRUCTURE: \n |---1 \n \t |---7 \n \t \t |---6 \n \t \t \t|---<17660ef415d37059.jpg>")
i=0
print(f"Image name: {train_df['id'].iloc[i]}")
print(f"First folder to look inside: {train_df['id'][i][0]}")
print(f"Second folder to look inside: {train_df['id'][i][1]}")
print(f"Second folder to look inside: {train_df['id'][i][2]}")


print(f"{'---'*20} \n Creating training data mapping \n {'---'*20}")
data_label_dict = {'image': [], 'target': []}
for i in tqdm(range(train_df.shape[0])):
    data_label_dict['image'].append(
        TRAIN_IMAGE_PATH + '/' +
        train_df['id'][i][0] + '/' + 
        train_df['id'][i][1]+ '/' +
        train_df['id'][i][2]+ '/' +
        train_df['id'][i] + ".jpg")
    data_label_dict['target'].append(
        train_df['landmark_id'][i])
#Convert to dataframe
train_pathlabel_df = pd.DataFrame(data_label_dict)
print(train_pathlabel_df.head())
    
print(f"{'---'*20} \n Creating test data mapping \n {'---'*20}")
data_label_dict = {'image': []}
for i in tqdm(range(submission_df.shape[0])):
    data_label_dict['image'].append(
        TEST_IMAGE_PATH + '/' +
        submission_df['id'][i][0] + '/' + 
        submission_df['id'][i][1]+ '/' +
        submission_df['id'][i][2]+ '/' +
        submission_df['id'][i] + ".jpg")

test_pathlabel_df = pd.DataFrame(data_label_dict)
print(test_pathlabel_df.head())


print(f"The data has {train_pathlabel_df['target'].nunique()} unique classes")

for tar in train_pathlabel_df['target'].unique()[:4]: 
    #Subset to just that target 
    label_df = train_pathlabel_df[train_pathlabel_df['target']==tar].reset_index()
    cols = 2
    rows = 2
    fig = plt.figure(figsize = (4*cols - 1, 4.5*rows - 1))
    for c in range(cols):
        for r in range(rows):
            ax = fig.add_subplot(rows, cols, c*rows + r + 1)
            img = mpimg.imread(label_df['image'][c+r])
            ax.imshow(img)#label_df[][c+r])
    fig.suptitle(f"Images corresponding to label [{tar}] with a total of {label_df.shape[0]} images available")
    plt.show()
    plt.close()


target_landmark_id = 138982

# Filter the DataFrame for the specific target landmark ID
label_df = train_pathlabel_df[train_pathlabel_df['target'] == target_landmark_id].reset_index()

# Set up grid for plotting images
cols = 2
rows = 2
fig = plt.figure(figsize = (4*cols - 1, 4.5*rows - 1))

# Loop through the images and plot
for c in range(cols):
    for r in range(rows):
        ax = fig.add_subplot(rows, cols, c*rows + r + 1)
        if c + r < len(label_df):  # Ensure we don't go out of bounds
            img = mpimg.imread(label_df['image'][c+r])
            ax.imshow(img)
            ax.axis('off')  # Hide axes for better visualization
        else:
            ax.axis('off')  # If there are fewer images, hide the empty subplot

# Title for the figure
fig.suptitle(f"Images corresponding to landmark_id [{target_landmark_id}] with a total of {label_df.shape[0]} images available")

# Show the images
plt.show()
plt.close()


import os

target_landmark_id = 138982

# Filter the DataFrame for the specific target landmark_id
label_df = train_df[train_df['landmark_id'] == target_landmark_id].reset_index()

# Set up grid for plotting images
cols = 2
rows = 2
fig = plt.figure(figsize=(4 * cols - 1, 4.5 * rows - 1))

# Loop through the images and plot
for c in range(cols):
    for r in range(rows):
        ax = fig.add_subplot(rows, cols, c * rows + r + 1)
        if c + r < len(label_df):  # Ensure we don't go out of bounds
            img_id = label_df['id'][c + r]  # Image ID from the DataFrame
            
            # Construct the image file path dynamically using the first three characters of the image ID
            first_char, second_char, third_char = img_id[:3]
            img_full_path = f'/kaggle/input/landmark-recognition-2021/train/{first_char}/{second_char}/{third_char}/{img_id}.jpg'
            
            # Get the image size in bytes and convert it to KB/MB
            try:
                img_size_bytes = os.path.getsize(img_full_path)
                img_size_kb = img_size_bytes / 1024  # Convert to KB
                img_size_mb = img_size_kb / 1024  # Convert to MB

                # Print the image size in the console
                print(f"Image {img_id} size: {img_size_kb:.2f} KB ({img_size_mb:.2f} MB)")

                # Read and display the image
                img = mpimg.imread(img_full_path)
                ax.imshow(img)
                ax.set_title(f"Size: {img_size_kb:.2f} KB")  # Show the size on the image
                ax.axis('off')  # Hide axes for better visualization
            except FileNotFoundError:
                ax.axis('off')  # If the image is not found, hide the subplot
                print(f"Image {img_id} not found at {img_full_path}")

        else:
            ax.axis('off')  # If there are fewer images, hide the empty subplot

# Title for the figure
fig.suptitle(f"Images corresponding to landmark_id [{target_landmark_id}] with a total of {label_df.shape[0]} images available")

# Show the images
plt.show()
plt.close()


target_landmark_id = 1924

# Filter the DataFrame for the specific target landmark_id
label_df = train_df[train_df['landmark_id'] == target_landmark_id].reset_index()

# Set up grid for plotting images
cols = 2
rows = 2
fig = plt.figure(figsize=(4 * cols - 1, 4.5 * rows - 1))

# Loop through the images and plot
for c in range(cols):
    for r in range(rows):
        ax = fig.add_subplot(rows, cols, c * rows + r + 1)
        if c + r < len(label_df):  # Ensure we don't go out of bounds
            img_id = label_df['id'][c + r]  # Image ID from the DataFrame
            
            # Construct the image file path dynamically using the first three characters of the image ID
            first_char, second_char, third_char = img_id[:3]
            img_full_path = f'/kaggle/input/landmark-recognition-2021/train/{first_char}/{second_char}/{third_char}/{img_id}.jpg'
            
            # Get the image size in bytes and convert it to KB/MB
            try:
                img_size_bytes = os.path.getsize(img_full_path)
                img_size_kb = img_size_bytes / 1024  # Convert to KB
                img_size_mb = img_size_kb / 1024  # Convert to MB

                # Print the image size in the console
                print(f"Image {img_id} size: {img_size_kb:.2f} KB ({img_size_mb:.2f} MB)")

                # Read and display the image
                img = mpimg.imread(img_full_path)
                ax.imshow(img)
                ax.set_title(f"Size: {img_size_kb:.2f} KB")  # Show the size on the image
                ax.axis('off')  # Hide axes for better visualization
            except FileNotFoundError:
                ax.axis('off')  # If the image is not found, hide the subplot
                print(f"Image {img_id} not found at {img_full_path}")

        else:
            ax.axis('off')  # If there are fewer images, hide the empty subplot

# Title for the figure
fig.suptitle(f"Images corresponding to landmark_id [{target_landmark_id}] with a total of {label_df.shape[0]} images available")

# Show the images
plt.show()
plt.close()


image_sizes = []

# Loop through the image IDs in the DataFrame and calculate sizes
for img_id in train_df['id']:
    # Construct the image file path dynamically using the first three characters of the image ID
    first_char, second_char, third_char = img_id[:3]
    img_full_path = f'/kaggle/input/landmark-recognition-2021/train/{first_char}/{second_char}/{third_char}/{img_id}.jpg'

    try:
        # Get the image size in bytes
        img_size_bytes = os.path.getsize(img_full_path)
        image_sizes.append(img_size_bytes / 1024)  # Convert size to KB
    except FileNotFoundError:
        print(f"Image {img_id} not found at {img_full_path}")
        image_sizes.append(0)  # If image not found, append 0 (you can handle this case as needed)

# Plot the histogram of image sizes
plt.figure(figsize=(10, 6))
plt.hist(image_sizes, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Image Sizes in the Train Set')
plt.xlabel('Image Size (KB)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


counts_df = pd.DataFrame(train_df[['landmark_id']].value_counts().reset_index())
counts_df.columns = ['Landmark', 'Count']
counts_df.sort_values('Count', ascending=False, inplace=True)
print(counts_df.head())

fig = go.Figure(data = [go.Bar(x = counts_df[:20].index,
                              y=counts_df[:20]['Count'],
                              text=counts_df[:20]['Count'],
                              textposition = 'outside')])
fig.update_layout(title="Image counts across top 20 landmarks",
                 xaxis_title = "Landmark id",
                 yaxis_title = "Count of images")
fig.update_xaxes(ticktext= counts_df[:20]['Landmark'],
                tickvals = counts_df[:20].index)
fig.show()



counts_df['Bin'] = np.where((counts_df['Count']<=10), "(0, 10]", "Rest")
counts_df['Bin'] = np.where((counts_df['Count']>10) & (counts_df['Count']<=20), "(10, 20]", counts_df['Bin'])
counts_df['Bin'] = np.where((counts_df['Count']>20) & (counts_df['Count']<=30), "(20, 30]", counts_df['Bin'])
counts_df['Bin'] = np.where((counts_df['Count']>30) & (counts_df['Count']<=50), "(30, 50]", counts_df['Bin'])
counts_df['Bin'] = np.where((counts_df['Count']>50) & (counts_df['Count']<=70), "(50, 70]", counts_df['Bin'])
counts_df['Bin'] = np.where((counts_df['Count']>70) & (counts_df['Count']<=100), "(70, 100]", counts_df['Bin'])
counts_df['Bin'] = np.where((counts_df['Count']>100) & (counts_df['Count']<=150), "(100, 150]", counts_df['Bin'])
# counts_df['Bin'] = np.where((counts_df['Count']>=20) & (counts_df['Count']<30), "Bin 3: 20-30", counts_df['Bin'])
bin_df = counts_df.groupby('Bin')['Count'].count().reset_index()
bin_df['Bin'] = bin_df['Bin'].astype('str')
print(bin_df)



fig = go.Figure(data = [go.Bar(x=bin_df['Bin'],
                               y=bin_df['Count'],
                               text = bin_df['Count'],
                              textposition = 'outside')])
fig.update_layout(title='Ímage counts across bins',
                  xaxis_title = "Bin/interval of image counts per landmark",
                 yaxis_title = "Count of images in bin")
# fig.update_xaxes('Bins/intervals of image counts')
fig.show()


test_df = pd.read_csv('/kaggle/input/mappings-gldv2/test.csv')

# Set up grid for plotting images
cols = 2
rows = 2
fig = plt.figure(figsize=(4 * cols - 1, 4.5 * rows - 1))

# Loop through the images and plot
for c in range(cols):
    for r in range(rows):
        ax = fig.add_subplot(rows, cols, c * rows + r + 1)
        if c + r < len(test_df):  # Ensure we don't go out of bounds
            img_id = test_df['id'][c + r]  # Image ID from the DataFrame
            
            # Construct the image file path dynamically using the first three characters of the image ID
            first_char, second_char, third_char = img_id[:3]
            img_full_path = f'/kaggle/input/landmark-recognition-2021/test/{first_char}/{second_char}/{third_char}/{img_id}.jpg'
            
            # Read and display the image
            try:
                img = mpimg.imread(img_full_path)
                ax.imshow(img)
                ax.set_title(f"ID: {img_id}")  # Show the image ID on the title
                ax.axis('off')  # Hide axes for better visualization
            except FileNotFoundError:
                ax.axis('off')  # If the image is not found, hide the subplot
                print(f"Image {img_id} not found at {img_full_path}")

        else:
            ax.axis('off')  # If there are fewer images, hide the empty subplot

# Title for the figure
fig.suptitle(f"Some images from the test set")

# Show the images
plt.show()
plt.close()


test_df = pd.read_csv('/kaggle/input/mappings-gldv2/test.csv')

# Set up grid for plotting images
cols = 2
rows = 2
fig = plt.figure(figsize=(4 * cols - 1, 4.5 * rows - 1))

# Loop through the images and plot
for c in range(cols):
    for r in range(rows):
        ax = fig.add_subplot(rows, cols, c * rows + r + 1)
        if c + r < len(test_df):  # Ensure we don't go out of bounds
            img_id = test_df['id'][c + r]  # Image ID from the DataFrame
            
            # Construct the image file path dynamically using the first three characters of the image ID
            first_char, second_char, third_char = img_id[:3]
            img_full_path = f'/kaggle/input/landmark-recognition-2021/test/{first_char}/{second_char}/{third_char}/{img_id}.jpg'
            
            # Print the path to debug the issue
            print(f"Trying to load image: {img_full_path}")
            
            # Read and display the image
            try:
                img = mpimg.imread(img_full_path)
                ax.imshow(img)
                ax.set_title(f"ID: {img_id}")  # Show the image ID on the title
                ax.axis('off')  # Hide axes for better visualization
            except FileNotFoundError:
                ax.axis('off')  # If the image is not found, hide the subplot
                print(f"Image {img_id} not found at {img_full_path}")

        else:
            ax.axis('off')  # If there are fewer images, hide the empty subplot

# Title for the figure
fig.suptitle(f"Some images from the test set")

# Show the images
plt.show()
plt.close()

