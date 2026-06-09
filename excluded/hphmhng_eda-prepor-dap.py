import os
for dirname, _, filenames in os.walk('/kaggle/input'):
        print(dirname)


import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objects as go
%matplotlib inline


def get_image_names(dataframe) : 
    image_names = dataframe["image_name"].values
    image_names = image_names + ".jpg"
    return image_names


def get_info(image_names) : 
    image_names = np.array(image_names)
    
    print("Length = ", len(image_names))
    print("Type = ", type(image_names))
    print("Shape = ", image_names.shape)
    
    return image_names


from scipy.stats import skew

def extract_information(image_names, directory) : 
    image_statistics = pd.DataFrame(index = np.arange(len(image_names)),
                                    columns = ["image_name", "path", "rows", "columns", "channels", 
                                              "image_mean", "image_standard_deviation", "image_skewness",
                                              "mean_red_value", "mean_green_value", "mean_blue_value"])
    i = 0 
    for name in tqdm(image_names) : 
        path = os.path.join(directory, name)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_statistics.iloc[i]["image_name"] = name
        image_statistics.iloc[i]["path"] = path
        image_statistics.iloc[i]["rows"] = image.shape[0]
        image_statistics.iloc[i]["columns"] = image.shape[1]
        image_statistics.iloc[i]["channels"] = image.shape[2]
        image_statistics.iloc[i]["image_mean"] = np.mean(image.flatten())
        image_statistics.iloc[i]["image_standard_deviation"] = np.std(image.flatten())
        image_statistics.iloc[i]["image_skewness"] = skew(image.flatten())
        image_statistics.iloc[i]["mean_red_value"] = np.mean(image[:,:,0])
        image_statistics.iloc[i]["mean_green_value"] = np.mean(image[:,:,1])
        image_statistics.iloc[i]["mean_blue_value"] = np.mean(image[:,:,2])
        
        i = i + 1
        del image
        
    return image_statistics


train_dir = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"
train = pd.DataFrame(pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv"))
train.head()


image_names = get_image_names(train)
image_names = get_info(image_names)


test_dir = "/kaggle/input/siim-isic-melanoma-classification/jpeg/test/"
test = pd.DataFrame(pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv"))
test.head()


image_names = get_image_names(test)
image_names = get_info(image_names)


train = pd.DataFrame(pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv"))
test = pd.DataFrame(pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv"))


train.shape, test.shape


train.head()


test.head()


train.info()


test.info()


len(train["patient_id"].unique()), len(test["patient_id"].unique())



print(train["target"].value_counts())


malignant = len(train[train["target"] == 1])
benign = len(train[train["target"] == 0])

labels = ["Malignant", "Benign"] 
size = [malignant, benign]

plt.figure(figsize = (8, 8))
plt.pie(size, labels = labels, shadow = True, startangle = 90, colors = ["r", "g"])
plt.title("Malignant VS Benign Cases")
plt.legend()


train_males = len(train[train["sex"] == "male"])
train_females  = len(train[train["sex"] == "female"])

test_males = len(test[test["sex"] == "male"])
test_females  = len(test[test["sex"] == "female"])

labels = ["Males", "Female"] 

size = [train_males, train_females]
explode = [0.1, 0.0]

plt.figure(figsize = (16, 16))
plt.subplot(1,2,1)
plt.pie(size, labels = labels, explode = explode, shadow = True, startangle = 90, colors = ["b", "g"])
plt.title("Male VS Female Training Set Count", fontsize = 18)
plt.legend()

print("Number of males in training set = ", train_males)
print("Number of females in training set= ", train_females)

size = [test_males, test_females]

plt.subplot(1,2,2)
plt.pie(size, labels = labels, explode = explode, shadow = True, startangle = 90, colors = ["b", "g"])
plt.title("Male VS Female Test Set Count", fontsize = 18)
plt.legend()

print("Number of males in testing set = ", test_males)
print("Number of females in testing set= ", test_females)


train_malignant  = train[train["target"] == 1]
train_malignant_males = len(train_malignant[train_malignant["sex"] == "male"])
train_malignant_females  = len(train_malignant[train_malignant["sex"] == "female"])

labels = ["Malignant Male Cases", "Malignant Female Cases"] 
size = [train_malignant_males, train_malignant_females]
explode = [0.1, 0.0]

plt.figure(figsize = (10, 10))
plt.pie(size, labels = labels, explode = explode, shadow = True, startangle = 90, colors = ["r", "c"])
plt.title("Malignant Male VS Female Cases", fontsize = 18)
plt.legend()
print("Malignant Male Cases = ", train_malignant_males)
print("Malignant Female Cases = ", train_malignant_females)


train_benign  = train[train["target"] == 0]

train_benign_males = len(train_benign[train_benign["sex"] == "male"])
train_benign_females  = len(train_benign[train_benign["sex"] == "female"]) 

labels = ["Benign Male Cases", "Benign Female Cases"] 
size = [train_benign_males, train_benign_females]
explode = [0.1, 0.0]

plt.figure(figsize = (10, 10))
plt.pie(size, labels = labels, explode = explode, shadow = True, startangle = 90, colors = ["g", "y"])
plt.title("Benign Male VS Benign Female Cases", fontsize = 18)
plt.legend()
print("Benign Male Cases = ", train_benign_males)
print("Benign Female Cases = ", train_benign_females)


cancer_versus_sex = train.groupby(["benign_malignant", "sex"]).size()
print(cancer_versus_sex)
print(type(cancer_versus_sex))


cancer_versus_sex = cancer_versus_sex.unstack(level = 1) / len(train) * 100
print(cancer_versus_sex)
print(type(cancer_versus_sex))


sns.set(style='whitegrid')
sns.set_context("paper", rc={"font.size":12,"axes.titlesize":20,"axes.labelsize":18})   

plt.figure(figsize = (10, 6))
sns.heatmap(cancer_versus_sex, annot=True, cmap="icefire", cbar=True)
plt.title("Cancer VS Sex Heatmap Analysis Normalized", fontsize = 18)
plt.tight_layout()


# train
train_torso = len(train[train["anatom_site_general_challenge"] == "torso"])
train_lower_extremity = len(train[train["anatom_site_general_challenge"] == "lower extremity"])
train_upper_extremity = len(train[train["anatom_site_general_challenge"] == "upper extremity"])
train_head_neck = len(train[train["anatom_site_general_challenge"] == "head/neck"])
train_palms_soles = len(train[train["anatom_site_general_challenge"] == "palms/soles"])
train_oral_genital = len(train[train["anatom_site_general_challenge"] == "oral/genital"])

# test
test_torso = len(test[test["anatom_site_general_challenge"] == "torso"])
test_lower_extremity = len(test[test["anatom_site_general_challenge"] == "lower extremity"])
test_upper_extremity = len(test[test["anatom_site_general_challenge"] == "upper extremity"])
test_head_neck = len(test[test["anatom_site_general_challenge"] == "head/neck"])
test_palms_soles = len(test[test["anatom_site_general_challenge"] == "palms/soles"])
test_oral_genital = len(test[test["anatom_site_general_challenge"] == "oral/genital"])

labels = ["Torso", "Lower Extremity", "Upper Extremity", "Head/Neck", "Palms/Soles", "Oral/Genital"] 

plt.figure(figsize = (16, 16))

plt.subplot(1,2,1)
size = [train_torso, train_lower_extremity, train_upper_extremity, train_head_neck, train_palms_soles, train_oral_genital]
explode = [0.05, 0.05, 0.05, 0.05, 0.05, 0.1]
plt.pie(size, labels = labels, explode = explode, shadow = True, startangle = 90)
plt.title("Anatomy Sites In Training Set", fontsize = 18)
plt.legend()

plt.subplot(1,2,2)
size = [test_torso, test_lower_extremity, test_upper_extremity, test_head_neck, test_palms_soles, test_oral_genital]
explode = [0.05, 0.05, 0.05, 0.05, 0.05, 0.1]
plt.pie(size, labels = labels, explode = explode, shadow = True, startangle = 90)
plt.title("Anatomy Sites In Testing Set", fontsize = 18)
plt.legend()

# Automatically adjust subplot parameters to give specified padding.
plt.tight_layout()


train_ages_benign = train.loc[train["target"] == 0, "age_approx"]
train_ages_malignant = train.loc[train["target"] == 1 , "age_approx"]

plt.figure(figsize = (10, 8))
sns.kdeplot(train_ages_benign, label = "Benign", shade = True, legend = True, cbar = True)
sns.kdeplot(train_ages_malignant, label = "Malignant", shade = True, legend = True, cbar = True)
plt.grid(True)
plt.xlabel("Age Of The Patients", fontsize = 18)
plt.ylabel("Probability Density", fontsize = 18)
plt.grid(which = "minor", axis = "both")
plt.title("Probabilistic Age Distribution In Training Set", fontsize = 18)


train_image_stats_01 = pd.DataFrame(pd.read_csv("/kaggle/input/compiled/melanoma_image_statistics_compiled_01"))
train_image_stats_02 = pd.DataFrame(pd.read_csv("/kaggle/input/compiled/melanoma_image_statistics_compiled_02"))
train_image_stats_03 = pd.DataFrame(pd.read_csv("/kaggle/input/compiled/melanoma_image_statistics_compiled_03"))
train_image_stats_04 = pd.DataFrame(pd.read_csv("/kaggle/input/compiled/melanoma_image_statistics_compiled_04"))
train_image_stats_05 = pd.DataFrame(pd.read_csv("/kaggle/input/compiled/melanoma_image_statistics_compiled_05"))
train_image_stats_06 = pd.DataFrame(pd.read_csv("/kaggle/input/compiled/melanoma_image_statistics_compiled_06"))

print(train_image_stats_01.shape)
print(train_image_stats_02.shape)
print(train_image_stats_03.shape)
print(train_image_stats_04.shape)
print(train_image_stats_05.shape)
print(train_image_stats_06.shape)


train_image_statistics = pd.concat([train_image_stats_01, train_image_stats_02, train_image_stats_03,
                                   train_image_stats_04, train_image_stats_05, train_image_stats_06],
                                  ignore_index = True)
train_image_statistics.shape


train_image_statistics.info()


test_image_stats_01 = pd.DataFrame(pd.read_csv("/kaggle/input/compiled-test/melanoma_image_statistics_compiled_test_01"))
test_image_stats_02 = pd.DataFrame(pd.read_csv("/kaggle/input/compiled-test/melanoma_image_statistics_compiled_test_02"))

print(test_image_stats_01.shape)
print(test_image_stats_02.shape)


test_image_statistics = pd.concat([test_image_stats_01, test_image_stats_02], ignore_index = True)

test_image_statistics.shape


test_image_statistics.info()


train_image_statistics.head()


test_image_statistics.head()


image_names = train_image_statistics["image_name"].values
random_images = [np.random.choice(image_names) for i in range(4)] # Generates a random sample from a given 1-D array
random_images 


train_dir = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"


plt.figure(figsize = (12, 8))
for i in range(4) : 
    plt.subplot(2, 2, i + 1) 
    image = cv2.imread(os.path.join(train_dir, random_images[i]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image, cmap = "gray")
    plt.grid(True)
plt.tight_layout()


benign_mean_red_value = []
benign_mean_green_value = []
benign_mean_blue_value = []

malignant_mean_red_value = []
malignant_mean_green_value = []
malignant_mean_blue_value = []

for image_name in tqdm(train_image_statistics["image_name"]) : 
    name = image_name[0:len(image_name)-4] 
    extracted_section = train[train["image_name"] == name]
    r = int(train_image_statistics[train_image_statistics["image_name"] == image_name]["mean_red_value"])
    g = int(train_image_statistics[train_image_statistics["image_name"] == image_name]["mean_green_value"])
    b = int(train_image_statistics[train_image_statistics["image_name"] == image_name]["mean_blue_value"])
    if int(extracted_section["target"]) == 0 : # benign
        benign_mean_red_value.append(r)
        benign_mean_green_value.append(g)
        benign_mean_blue_value.append(b)
    else:
        malignant_mean_red_value.append(r)
        malignant_mean_green_value.append(g)
        malignant_mean_blue_value.append(b)


range_of_spread = max(benign_mean_red_value) - min(benign_mean_red_value)

plt.figure(figsize = (12, 8))
plt.rc("font", weight = "bold")
sns.set_style("whitegrid")
fig = sns.distplot(benign_mean_red_value, hist = True, kde = True, label = "Mean Red Channel Intensities", color = "r")
fig.set(xlabel = "Mean red channel intensities observed in each image",
        ylabel = "Probability Density")
plt.title("Spread Of Red Channel In Benign Cases", fontsize = 18)
plt.legend()
print("The range of spread = {:.2f}".format(range_of_spread))


range_of_spread = max(benign_mean_green_value) - min(benign_mean_green_value)

plt.figure(figsize = (12, 8))
plt.rc("font", weight = "bold")
sns.set_style("whitegrid")
fig = sns.distplot(benign_mean_green_value, hist = True, kde = True, label = "Mean Green Channel Intensities", color = "g")
fig.set(xlabel = "Mean green channel intensities observed in each image",
        ylabel = "Probability Density") 
plt.title("Spread Of Green Channel In Benign Cases", fontsize = 18)
plt.legend()
print("The range of spread = {:.2f}".format(range_of_spread))


range_of_spread = max(benign_mean_blue_value) - min(benign_mean_blue_value)

plt.figure(figsize = (12, 8))
plt.rc("font", weight = "bold")
sns.set_style("whitegrid")
fig = sns.distplot(benign_mean_blue_value, hist = True, kde = True, label = "Mean Blue Channel Intensities", color = "b")
fig.set(xlabel = "Mean blue channel intensities observed in each image",
        ylabel = "Probability Density") 
plt.title("Spread Of Blue Channel In Benign Cases", fontsize = 18)
plt.legend()
print("The range of spread = {:.2f}".format(range_of_spread))


plt.figure(figsize = (12, 8))
plt.rc("font", weight = "bold")
sns.set_style("whitegrid")
fig = sns.distplot(benign_mean_blue_value, hist = False, kde = True, label = "Mean Blue Channel Intensities", color = "b")
fig = sns.distplot(benign_mean_red_value, hist = False, kde = True, label = "Mean Red Channel Intensities", color = "r")
fig = sns.distplot(benign_mean_green_value, hist = False, kde = True, label = "Mean Green Channel Intensities", color = "g")

fig.set(xlabel = "Mean channel intensities observed in each image",
        ylabel = "Probability Density") 
plt.title("Spread Of Channels In Benign Cases", fontsize = 18)
plt.legend()


del benign_mean_red_value
del benign_mean_green_value
del benign_mean_blue_value


import gc
gc.collect()


plt.figure(figsize = (12, 8))
plt.rc("font", weight = "bold")
sns.set_style("whitegrid")
fig = sns.distplot(malignant_mean_blue_value, hist = False, kde = True, label = "Mean Blue Channel Intensities", color = "b")
fig = sns.distplot(malignant_mean_red_value, hist = False, kde = True, label = "Mean Red Channel Intensities", color = "r")
fig = sns.distplot(malignant_mean_green_value, hist = False, kde = True, label = "Mean Green Channel Intensities", color = "g")

fig.set(xlabel = "Mean channel intensities observed in each image",
        ylabel = "Probability Density") 
plt.title("Spread Of Channels In Malignant Cases", fontsize = 18)
plt.legend()


gc.collect()


train.head()


missing = len(train[train["sex"].isna() == True])
available = len(train[train["sex"].isna() == False])

x = ["Availabe data", "Unavailable data"]
y = [np.log(available), np.log(missing)]

print("Count of missing data = ", missing)
print("Count of available data = ", available)

plt.figure(figsize = (12, 8))
plt.subplot(1,1,1)
plt.barh(x, y, color = "m")
plt.grid(True)
plt.title("Data On Patient's Sex")


train['sex'].fillna('male', inplace=True)


missing =  len(train[train["age_approx"].isna() == True]) 
available = len(train[train["age_approx"].isna() == False]) 

print("Missing age values = ", missing)
print("Available age data = ", available)

x = ["Availabe data", "Unavailable data"]
y = [np.log(available), np.log(missing)] 

plt.figure(figsize = (12, 8))
plt.subplot(1,1,1)
plt.barh(x, y, color = "y")
plt.grid(True)
plt.title("Data On Patient's Age")


# train
anatomy_sites = ["torso", "upper extremity", "lower extremity"]

relevant_dataframe_part = train[(train["sex"] == "male") &
                     (train["anatom_site_general_challenge"].isin(anatomy_sites)) &
                     (train["target"] == 0)]

median_value = relevant_dataframe_part["age_approx"].median()

print("Median value = ", median_value)


train["age_approx"].fillna(median_value, inplace = True)



train["anatom_site_general_challenge"].fillna("torso", inplace = True)
test["anatom_site_general_challenge"].fillna("torso", inplace = True)


train.info()


test.info()



train_image_stats_01 = pd.DataFrame(pd.read_csv("/kaggle/input/compiled/melanoma_image_statistics_compiled_01"))
train_image_stats_02 = pd.DataFrame(pd.read_csv("/kaggle/input/compiled/melanoma_image_statistics_compiled_02"))
train_image_stats_03 = pd.DataFrame(pd.read_csv("/kaggle/input/compiled/melanoma_image_statistics_compiled_03"))
train_image_stats_04 = pd.DataFrame(pd.read_csv("/kaggle/input/compiled/melanoma_image_statistics_compiled_04"))
train_image_stats_05 = pd.DataFrame(pd.read_csv("/kaggle/input/compiled/melanoma_image_statistics_compiled_05"))
train_image_stats_06 = pd.DataFrame(pd.read_csv("/kaggle/input/compiled/melanoma_image_statistics_compiled_06"))

print(train_image_stats_01.shape)
print(train_image_stats_02.shape)
print(train_image_stats_03.shape)
print(train_image_stats_04.shape)
print(train_image_stats_05.shape)
print(train_image_stats_06.shape)


train_image_statistics = pd.concat([train_image_stats_01, train_image_stats_02, train_image_stats_03,
                                   train_image_stats_04, train_image_stats_05, train_image_stats_06],
                                  ignore_index = True)
train_image_statistics.shape


train_image_statistics.head()


train_image_statistics.info()


train_dir = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"


image_names = train_image_statistics["image_name"].values
random_images = [np.random.choice(image_names) for i in range(4)] # Generates a random sample from a given 1-D array
random_images 


plt.figure(figsize = (12, 8))
for i in range(4) : 
    plt.subplot(2, 2, i + 1) 
    image = cv2.imread(os.path.join(train_dir, random_images[i]))
    # cv2 reads images in BGR format. Hence we convert it to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image, cmap = "gray")
    plt.grid(True)
# Automatically adjust subplot parameters to give specified padding.
plt.tight_layout()


def non_local_means_denoising(image) : 
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return denoised_image


sample_image = cv2.imread(os.path.join(train_dir, random_images[0]))
# cv2 reads images in BGR format. Hence we convert it to RGB
sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
denoised_image = non_local_means_denoising(sample_image)


plt.figure(figsize = (12, 8))
plt.subplot(1,2,1)
plt.imshow(sample_image, cmap = "gray")
plt.grid(False)
plt.title("Normal Image")

plt.subplot(1,2,2)  
plt.imshow(denoised_image, cmap = "gray")
plt.grid(False)
plt.title("Denoised image")    
# Automatically adjust subplot parameters to give specified padding.
plt.tight_layout() 


def hair_removal(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17,17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=2)

    dst = cv2.inpaint(image, thresh, 3, cv2.INPAINT_TELEA)

    return dst, thresh


hair_removed, hair_mask = hair_removal(sample_image)

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(sample_image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(hair_mask, cmap="gray")
plt.title("Detected Hair Mask")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(hair_removed)
plt.title("Hair Removed Image")
plt.axis("off")

plt.show()



def histogram_equalization(image) : 
    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
    y_channel = image_ycrcb[:,:,0] # apply local histogram processing on this channel
    cr_channel = image_ycrcb[:,:,1]
    cb_channel = image_ycrcb[:,:,2]
    
    # Local histogram equalization
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8,8))
    equalized = clahe.apply(y_channel)
    equalized_image = cv2.merge([equalized, cr_channel, cb_channel])
    equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_YCR_CB2RGB)
    return equalized_image


# 1. Denoise
denoised_image = non_local_means_denoising(sample_image)

# 2. Hair Removal
hair_removed, hair_mask = hair_removal(denoised_image)

# 3. Histogram Equalization (dùng ảnh đã xóa lông)
equalized_image = histogram_equalization(hair_removed)


plt.figure(figsize = (12, 8))
plt.subplot(1,4,1)
plt.imshow(sample_image, cmap = "gray")
plt.grid(False)
plt.title("Normal img", fontsize = 14)

plt.subplot(1,4,2)  
plt.imshow(denoised_image, cmap = "gray")
plt.grid(False)
plt.title("denoised img", fontsize = 14)

plt.subplot(1,4,3)  
plt.imshow(hair_removed, cmap = "gray")
plt.grid(False)
plt.title("img after hair removal", fontsize = 14)

plt.subplot(1,4,4)  
plt.imshow(equalized_image, cmap = "gray")
plt.grid(False)
plt.title("Histogram equalized img", fontsize = 14)
# Automatically adjust subplot parameters to give specified padding.
plt.tight_layout()


def segmentation(image, k, attempts) : 
    vectorized = np.float32(image.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    res , label , center = cv2.kmeans(vectorized, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    segmented_image = res.reshape((image.shape))
    return segmented_image


plt.figure(figsize = (12, 8))
plt.subplot(1,1,1)
plt.imshow(hair_removed, cmap = "gray")
plt.grid(False)
plt.title("hair remove Image")


plt.figure(figsize = (12, 8))
segmented_image = segmentation(hair_removed, 3, 10) # k = 3, attempt = 10
plt.subplot(1,3,1)
plt.imshow(segmented_image, cmap = "gray")
plt.grid(False)
plt.title("Segmented Img k = 3")

segmented_image = segmentation(hair_removed, 4, 10) # k = 4, attempt = 10
plt.subplot(1,3,2)
plt.imshow(segmented_image, cmap = "gray")
plt.grid(False)
plt.title("Segmented Img k = 4")

segmented_image = segmentation(hair_removed, 5, 10) # k = 5, attempt = 10
plt.subplot(1,3,3)
plt.imshow(segmented_image, cmap = "gray")
plt.grid(False)
plt.title("Segmented Img k = 5")


SHAPE = (224, 224, 3)


def resize(image, shape) : 
    image = cv2.resize(image, (shape[0], shape[1]))
    return image   


# Dùng cột benign_malignant làm nhãn
train["label"] = train["benign_malignant"].astype(str)

print(train["label"].value_counts())  # kiểm tra số lượng

from tensorflow.keras.preprocessing.image import ImageDataGenerator

augment_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2
)

# Thêm đuôi .jpg vào image_name
train["image_name"] = train["image_name"].astype(str) + ".jpg"

print(train["image_name"].head())  # kiểm tra lại tên file


# ✅ Train generator
train_generator = augment_datagen.flow_from_dataframe(
    dataframe=train,
    directory=train_dir,
    x_col="image_name",
    y_col="label",          # <--- dùng label mới
    target_size=(224,224),
    class_mode="binary",
    subset="training",
    batch_size=32,
    shuffle=True
)

# ✅ Validation generator
val_generator = augment_datagen.flow_from_dataframe(
    dataframe=train,
    directory=train_dir,
    x_col="image_name",
    y_col="label",          # <--- dùng label mới
    target_size=(224,224),
    class_mode="binary",
    subset="validation",
    batch_size=32,
    shuffle=False
)



# Đếm trước khi cân bằng
counts_before = train["label"].value_counts()

# ✅ Oversampling malignant cho cân bằng
benign_df = train[train["label"] == "benign"]
malignant_df = train[train["label"] == "malignant"]

# Nhân malignant lên cho gần bằng benign
malignant_oversampled = malignant_df.sample(len(benign_df), replace=True, random_state=42)

# Ghép lại
balanced_train = pd.concat([benign_df, malignant_oversampled], axis=0).reset_index(drop=True)

# Đếm sau khi cân bằng
counts_after = balanced_train["label"].value_counts()

# ✅ Vẽ biểu đồ so sánh
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Trước khi cân bằng
axes[0].bar(counts_before.index, counts_before.values, color=["skyblue", "salmon"])
axes[0].set_title("Before balanced")
axes[0].set_ylabel("Number of the img")

# Sau khi cân bằng
axes[1].bar(counts_after.index, counts_after.values, color=["skyblue", "salmon"])
axes[1].set_title("After balanced")

plt.suptitle("Comparing to Benign vs Malignant")
plt.show()




