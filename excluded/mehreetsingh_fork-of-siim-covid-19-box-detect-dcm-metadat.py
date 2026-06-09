# --- CSS STYLE ---
from IPython.core.display import HTML
def css_styling():
    styles = open("../input/2020-cost-of-living/alerts.css", "r").read()
    return HTML("<style>"+styles+"</style>")
css_styling()


# Libraries
import os
import re
import wandb
import tqdm
import warnings
import glob
import ast
import cv2
import math
import pandas as pd
import numpy as np
from IPython.display import display_html
import seaborn as sns
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from sklearn.cluster import KMeans
from skimage import morphology, measure


# Environment check
warnings.filterwarnings("ignore")
os.environ["WANDB_SILENT"] = "true"
CONFIG = {'competition': 'siim-fisabio-rsna', '_wandb_kernel': 'aot'}



# Secrets ğŸ¤«
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("mehreet_secret_wand")




# Custom colors
my_colors = ["#E97777", "#E9B077", "#E9E977", 
             "#77E977", "#7777E9", "#6F6BAC", "#B677E9"]
sns.palplot(sns.color_palette(my_colors))

# Set Style
sns.set_style("white")
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['axes.spines.left'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
plt.rcParams.update({'font.size': 22})

class color:
    BOLD = '\033[1m' + '\033[93m'
    END = '\033[0m'


! wandb login $secret_value_0


def show_values_on_bars(axs, h_v="v", space=0.4):
    '''Plots the value at the end of the a seaborn barplot.
    axs: the ax of the plot
    h_v: weather or not the barplot is vertical/ horizontal'''
    
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(round(_x, 5), round(_y, 5), format(round(value, 5), ','), ha="center") 
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax.text(_x, _y, format(value, ','), ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
        
        
def offset_png(x, y, path, ax, zoom, offset, border=2):
    '''For adding other .png images to the graph.
    source: https://stackoverflow.com/questions/61971090/how-can-i-add-images-to-bars-in-axes-matplotlib'''
    
    img = plt.imread(path)
    im = OffsetImage(img, zoom=zoom)
    im.image.axes = ax
    x_offset = offset
    ab = AnnotationBbox(im, (x, y), xybox=(x_offset, 0), frameon=False,
                        xycoords='data', boxcoords="offset points", pad=0)
    ax.add_artist(ab)
    
    
def get_image_metadata(study_id, df):
    '''Returns the label and bounding boxes (if any)
    for a speciffic study id.'''
    
    data = df[df["study_id"] == study_id]
    
    if data["Negative for Pneumonia"].values == 1:
        label = "negative_for_pneumonia"
    elif data["Typical Appearance"].values == 1:
        label = "typical"
    elif data["Indeterminate Appearance"].values == 1:
        label = "indeterminate"
    else:
        label = "atypical"
        
    bbox = list(data["boxes"].values)
    
    return label, bbox


def save_dataset_artifact(run_name, artifact_name, path):
    '''Saves dataset to W&B Artifactory.
    run_name: name of the experiment
    artifact_name: under what name should the dataset be stored
    path: path to the dataset'''
    
    run = wandb.init(project='siim-covid19', 
                     name=run_name, 
                     config=CONFIG, anonymous="allow")
    artifact = wandb.Artifact(name=artifact_name, 
                              type='dataset')
    artifact.add_file(path)

    wandb.log_artifact(artifact)
    wandb.finish()
    print("Artifact has been saved successfully.")
    
    
def create_wandb_plot(x_data=None, y_data=None, x_name=None, y_name=None, title=None, log=None, plot="line"):
    '''Create and save lineplot/barplot in W&B Environment.
    x_data & y_data: Pandas Series containing x & y data
    x_name & y_name: strings containing axis names
    title: title of the graph
    log: string containing name of log'''
    
    data = [[label, val] for (label, val) in zip(x_data, y_data)]
    table = wandb.Table(data=data, columns = [x_name, y_name])
    
    if plot == "line":
        wandb.log({log : wandb.plot.line(table, x_name, y_name, title=title)})
    elif plot == "bar":
        wandb.log({log : wandb.plot.bar(table, x_name, y_name, title=title)})
    elif plot == "scatter":
        wandb.log({log : wandb.plot.scatter(table, x_name, y_name, title=title)})
        
        
def create_wandb_hist(x_data=None, x_name=None, title=None, log=None):
    '''Create and save histogram in W&B Environment.
    x_data: Pandas Series containing x values
    x_name: strings containing axis name
    title: title of the graph
    log: string containing name of log'''
    
    data = [[x] for x in x_data]
    table = wandb.Table(data=data, columns=[x_name])
    wandb.log({log : wandb.plot.histogram(table, x_name, title=title)})
    
    
def return_coords(box):
    '''Returns coordinates from a bbox'''
    # Get the list of dictionaries
    box = ast.literal_eval(box)[0]
    # Get the exact x and y coordinates
    x1, y1, x2, y2 = box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]
    # Save coordinates
    return (int(x1), int(y1), int(x2), int(y2))


def fix_inverted_radiograms(data, img):
    '''Fixes inverted radiograms - with PhotometricInterpretation == "MONOCHROME1"
    data: the .dcm dataset
    img: the .dcm pixel_array'''
    
    if data.PhotometricInterpretation == "MONOCHROME1":
        img = np.amax(img) - img
    
    img = img - np.min(img)
    img = img / np.max(img)
    img = (img * 255).astype(np.uint8)
    
    return img


# Save data to W&B Dashboard
save_dataset_artifact(run_name='save-train1',
                      artifact_name='train_study_level', 
                      path="../input/siim-covid19-detection/train_study_level.csv")

save_dataset_artifact(run_name='save-train2',
                      artifact_name='train_image_level', 
                      path="../input/siim-covid19-detection/train_image_level.csv")


# Read in metadata
train_study = pd.read_csv("../input/siim-covid19-detection/train_study_level.csv")
train_image = pd.read_csv("../input/siim-covid19-detection/train_image_level.csv")

print(color.BOLD + "Train Study Shape:" + color.END, train_study.shape, "\n" +
      color.BOLD + "Train Image Shape:" + color.END, train_image.shape, "\n" +
      "\n" +
      "Note: There are {} missing values in train_image.".\
                              format(train_image["boxes"].isna().sum()), "\n" +
      "This happens for labels = 'none' - no checkboxes.", 3*"\n")


# Head of our 2 training metadata
df1_styler = train_study.head(3).style.set_table_attributes("style='display:inline'").\
                                set_caption('TRAIN STUDY')
df2_styler = train_image.head(3).style.set_table_attributes("style='display:inline'").\
                                set_caption('TRAIN IMAGE')


# Head of our 2 training metadata
df1_styler = train_study.head(3).style.set_table_attributes("style='display:inline'").\
                                set_caption('TRAIN STUDY')
df2_styler = train_image.head(3).style.set_table_attributes("style='display:inline'").\
                                set_caption('TRAIN IMAGE')




print(df1_styler)


#df1_styler + df2_styler


display_html(df1_styler._repr_html_() + df2_styler._repr_html_(), raw=True)


run = wandb.init(project='siim-covid19', name='metadata_eda', config=CONFIG, anonymous="allow")


print(train_study["id"].head(5))


# Process id
train_study["study_id"] = train_study["id"].apply(lambda x: x.split("_")[0])
print(train_study["study_id"].head(5))


# Data for plots
pneumonia = train_study["Negative for Pneumonia"]
typical = train_study["Typical Appearance"]
indeterminate = train_study["Indeterminate Appearance"]
atypical = train_study["Atypical Appearance"]


# Plotting
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(21,20))
axs = [ax1, ax2, ax3, ax4]
dfs = [pneumonia, typical, indeterminate, atypical]
titles = ["Pneumonia", "Typical", "Indeterminate", "Atypical"]

for ax, df, title in zip(axs, dfs, titles):
    sns.countplot(y=df, ax=ax, palette=my_colors[1:])
    ax.set_title(title, fontsize=25, weight='bold')
    show_values_on_bars(ax, h_v="h", space=0.4)
    ax.set_xticklabels([])
    ax.set_ylabel('')
    ax.set_xlabel('')
    
# Virus png
path='../input/siimfisabiorsna-covid-2021/PinClipart.com_virus-clip-art_742280.png'
offset_png(x=4378, y=1, path=path, ax=ax1, zoom=0.05, offset=-360, border=1)
offset_png(x=2855, y=1, path=path, ax=ax2, zoom=0.05, offset=-50, border=1)
offset_png(x=1049, y=1, path=path, ax=ax3, zoom=0.05, offset=-43, border=1)
offset_png(x=474, y=1, path=path, ax=ax4, zoom=0.05, offset=40, border=1)


# Save plots into W&B Dashboard
for title, df in zip(titles, dfs):
    create_wandb_plot(x_data=[0, 1], 
                      y_data=df.value_counts().values, 
                      x_name="Flag", y_name="Freq", title=title, 
                      log=title, plot="bar")


# Get data and transform frequencies to percentages
df = train_study.groupby(['Negative for Pneumonia', 'Typical Appearance',
       'Indeterminate Appearance', 'Atypical Appearance']).count().reset_index()

df["label"] = ['Atypical Appearance', 'Indeterminate Appearance',
               'Typical Appearance', 'Negative for Pneumonia']
df["perc"] = df["id"]/df["id"].sum()*100

# Plot
bar,ax = plt.subplots(figsize=(21,10))
ax = sns.barplot(x=df["label"], y=df["perc"], 
                 ci=None, palette=my_colors, orient='v')
ax.set_title("Label % in Total Observations", fontsize=25,
             weight = "bold")
ax.set_xlabel(" ")
ax.set_ylabel("Percentage")
for rect in ax.patches:
    ax.text (rect.get_x() + rect.get_width() / 2,rect.get_height(),
             "%.1f%%"% rect.get_height(), weight='bold')


create_wandb_plot(x_data=df["label"].values,
                  y_data=df["perc"].values,
                  x_name="Label", y_name="Percentage", 
                  title="Label % in Total Observations", 
                  log="perc", plot="bar")


wandb.finish()


run = wandb.init(project='siim-covid19', name='train_imgs_eda', config=CONFIG, anonymous="allow")


train_image.head(5)


# Process id
train_image["image_id"] = train_image["id"].apply(lambda x: x.split("_")[0])

# Data for plotting
df = train_image["label"].apply(lambda x: x.split(" ")[0]).\
                                    value_counts().reset_index()

# Plot
plt.figure(figsize=(21, 15))
ax = sns.barplot(data=df, y="index", x="label", palette=my_colors[3:])
show_values_on_bars(ax, h_v="h", space=0.4)
plt.title("How many images have a bounding box present?", 
          fontsize=30, weight='bold')
plt.xticks([])
plt.ylabel('')
plt.xlabel('');

# Virus png
path='../input/siimfisabiorsna-covid-2021/PinClipart.com_virus-clip-art_742280.png'
offset_png(x=4294, y=0, path=path, ax=ax, zoom=0.1, offset=-110, border=1)


create_wandb_plot(x_data=df["index"], 
                  y_data=df["label"].values, 
                  x_name="BBox", y_name="Freq", 
                  title="Images with bbox",
                  log="bbox", plot="bar")


train_image.head(5)


# Crate df
df = train_image["StudyInstanceUID"].value_counts().reset_index().\
                        sort_values("StudyInstanceUID", ascending=False)
print(color.BOLD + "Max number of images available per study:" + color.END, 
      df["StudyInstanceUID"].max(), "\n" +
      color.BOLD + "Min number of images available per study:" + color.END, 
      df["StudyInstanceUID"].min(), 2*"\n")



# Plot
plt.figure(figsize=(21, 14))
sns.distplot(a=df["StudyInstanceUID"], color=my_colors[6], 
             hist=False, kde_kws=dict(lw=7, ls="-"))
plt.title("How many images per study?", 
          fontsize=30, weight='bold')
plt.xticks([])
plt.ylabel('Study Frequency')
plt.xlabel('Image Ids');


wandb.log({"max_images_on_study" : df["StudyInstanceUID"].max()})
create_wandb_hist(x_data=df["StudyInstanceUID"], 
                  x_name="Image Freq", title="No. images per study", 
                  log="hist")


wandb.finish()


# Merge all info together
train = pd.merge(train_image, train_study, 
                 left_on="StudyInstanceUID", right_on="study_id")

train.drop(["id_x", "StudyInstanceUID", "id_y"], axis=1, inplace=True)


train.head(5)


train[train["study_id"] == "0fd2db233deb"]


run = wandb.init(project='siim-covid19', name='image_explore', config=CONFIG, anonymous="allow")


def show_dcm_info(study_ids, df):
    '''Show .dcm images along with description.'''
    wandb_logs = []
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(21,10))

    # Get .dcm paths
    dcm_paths = [glob.glob(f"../input/siim-covid19-detection/train/{study_id}/*/*")[0]
                 for study_id in study_ids]
    datasets = [pydicom.dcmread(path) for path in dcm_paths]
    images = [apply_voi_lut(dataset.pixel_array, dataset) for dataset in datasets]

    # Loop through the information
    for study_id, data, img, i in zip(study_ids, datasets, images, range(2*3)):
        # Fix inverted images
        img = fix_inverted_radiograms(data, img)

        # Below function available in functions section ;)
        label, bbox = get_image_metadata(study_id, df)
        
        # Check for bounding box and add if it's the case
        try: 
            # For no bbox, the list is [nan]
            no_box = math.isnan(bbox[0])
            pass
        except TypeError:
            # Retrieve the bounding box
            all_coords = []
            for box in bbox:
                all_coords.append(return_coords(box))

            for (x1, y1, x2, y2) in all_coords:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 80, 255), 15)
                cv2.putText(img, label, (x1, y1-14), 
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4)
                
        # Plot the image
        x = i // 3
        y = i % 3
        
        axes[x, y].imshow(img, cmap="rainbow")
        axes[x, y].set_title(f"Label: {label} \n Sex: {data.PatientSex} | Body Part: {data.BodyPartExamined}", 
                  fontsize=14, weight='bold')
        axes[x, y].axis('off');
        
        # Save to W&B
        wandb_logs.append(wandb.Image(img, 
                                      caption=f"Label: {label} \n Sex: {data.PatientSex} | Body Part: {data.BodyPartExamined}"))
          
    wandb.log({f"{label}": wandb_logs})


show_dcm_info(study_ids=["72044bb44d41", "5b65a69885b6", "6aa32e76f998",
                         "c9ffe6312921", "082cafb03942", "d3e83031ebea"], 
              df=train)


show_dcm_info(study_ids=["f807cd855d31", "8087e3bc0efe", "7249de10ed69",
                         "e300a4e86207", "4bac6c7da8b8", "f2d30ac37f7b"], 
              df=train)


show_dcm_info(study_ids=["b949689a9ef1", "fe7e6015560d", "feffa20fac13",
                         "747483509d0e", "c70369caef91", "1e1b4b1b53cb"], 
              df=train)


show_dcm_info(study_ids=["612ea5194007", "db14e640e037", "d4ab797396b4",
                         "6ae8a88c4b0c", "b3cf474bee3b", "0ba55e5422ab"], 
              df=train)


def wandb_bbox(image, bboxes, true_label, class_id_to_label):
    all_boxes = []
    for bbox in bboxes:
        box_data = {"position": {
                        "minX": bbox[0],
                        "minY": bbox[1],
                        "maxX": bbox[2],
                        "maxY": bbox[3]
                    },
                     "class_id" : int(true_label),
                     "box_caption": class_id_to_label[true_label],
                     "domain" : "pixel"}
        all_boxes.append(box_data)
    

    return wandb.Image(image, boxes={
        "ground_truth": {
            "box_data": all_boxes,
          "class_labels": class_id_to_label
        }
    })


def resize_img_and_coord(img, coord, resize):
    '''Resizes the image and its coordinates.
    img: the pixel.array image
    coord: the speciffic coordinates from return_coordinates() function
    resize: and integer specifying the desired size of new image'''
    
    # Resize the image
    w_old, h_old = img.shape

    # Resize the coordinates
    img = cv2.resize(img, (resize,resize))

    new_x1 = int(coord[0][0] / (w_old/resize))
    new_y1 = int(coord[0][1] / (h_old/resize))
    new_x2 = int(coord[0][2] / (w_old/resize))
    new_y2 = int(coord[0][3] / (h_old/resize))
    new_coord = [(new_x1, new_y1, new_x2, new_y2)]
    
    return img, new_coord


# Get a few example ids (image_id)
example_ids = ["000a312787f2", "0012ff7358bc", "001398f4ff4f",
               "001bd15d1891", "002e9b2128d0", "ffbeafe30b77",
               "0022227f5adf", "00a129830f4e", "01376c1ba556", "008ca392cff3"]

# Read in datas
study_ids = train[train["image_id"].isin(example_ids)]["study_id"].values
paths = [glob.glob(f"../input/siim-covid19-detection/train/{i}/*/*")[0]
         for i in study_ids]

# Retrieve resized information
images, coords, labels = [], [], []
for path, study_id in zip(paths, study_ids):
    try:
        # Read data file
        data = pydicom.dcmread(path)
        # Get image data
        img = apply_voi_lut(data.pixel_array, data)
        # Get image coordinates
        label, bbox = get_image_metadata(study_id=study_id, df=train)
        coord = [return_coords(box) for box in bbox]

        # Fix inverted radiograms + resize
        img = fix_inverted_radiograms(data, img)
        resized_img, resized_coord = resize_img_and_coord(img, coord, resize=200)

        images.append(resized_img)
        coords.append(resized_coord)
        labels.append(label)
    except RuntimeError:
        pass


# Map each label to a number
class_label_to_id = {'atypical': 0, 'indeterminate': 1, 'typical': 2}
# And each number to a label
class_id_to_label = {val: key for key, val in class_label_to_id.items()}

# Log each image
wandb_bbox_list = []

for image, coord, label in zip(images, coords, labels):
    wandb_bbox_list.append(wandb_bbox(image=image,
                                      bboxes=coord, 
                                      true_label=class_label_to_id[label],
                                      class_id_to_label=class_id_to_label))

# Save images to W&B Dashboard
wandb.log({"radiograph": wandb_bbox_list})

print(color.BOLD + "Finished! Your Images were uploaded in your W&B Dashboard!" + color.END)


wandb.finish()


def get_observation_data(path):
    """Get information from the .dcm files.
    path: complete path to the .dcm file"""

    image_data = pydicom.read_file(path)
    
    # Dictionary to store the information from the image
    observation_data = {
        "FileNumber" : path.split("/")[5],
        "Rows" : image_data.get("Rows"),
        "Columns" : image_data.get("Columns"),
        "PatientID" : image_data.get("PatientID"),
        "PatientName" : image_data.get("PatientName"),
        "PhotometricInterpretation" : image_data.get("PhotometricInterpretation"),
        "StudyInstanceUID" : image_data.get("StudyInstanceUID"),
        "SamplesPerPixel" : image_data.get("SamplesPerPixel"),
        "BitsAllocated" : image_data.get("BitsAllocated"),
        "BitsStored" : image_data.get("BitsStored"),
        "HighBit" : image_data.get("HighBit"),
        "PixelRepresentation" : image_data.get("PixelRepresentation"),
    }

    # String columns
    str_columns = ["ImageType", "Modality", "PatientSex", "BodyPartExamined"]
    for k in str_columns:
        observation_data[k] = str(image_data.get(k)) if k in image_data else None

    
    return observation_data


# # An example
# p = "../input/siim-covid19-detection/train/00792b5c8852/1f52bcb3143e/3fadf4b48db3.dcm"
# example = get_observation_data(p)
# print(example)

# # === GET ALL METADATA ===
# # Get all paths to .dcm files
# all_paths = glob.glob("../input/siim-covid19-detection/train/*/*/*")

# # === Get metadata ===
# exceptions = 0
# dicts = []

# for path in tqdm.tqdm(all_paths):
#     # Get .dcm metadata
#     ### TODO: add .dcm id
#     try:
#         d = get_observation_data(path)
#         dicts.append(d)
#     except Exception as e:
#         exceptions += 1
#         continue
        
# # === SAVE METADATA ===
# # Convert into df
# meta_train_data = pd.DataFrame(data=dicts, columns=example.keys())
# meta_train_data[""]
# # Export information
# meta_train_data.to_csv("meta_train.csv", index=False)

# print("Metadata processed & saved successfuly :)")


# Save data to W&B Dashboard
save_dataset_artifact(run_name='dave-dcm-meta',
                   artifact_name='dcm_metadata', 
                      path="../input/siimfisabiorsna-covid-2021/meta_train.csv")


# Import
dcm_meta = pd.read_csv("../input/siimfisabiorsna-covid-2021/meta_train.csv")
dcm_meta = pd.concat([dcm_meta, train], axis=1)

dcm_meta.head(2)
#dcm_meta.info()


#Show data from particular column
dcm_meta[dcm_meta["Negative for Pneumonia"].isin([0, 1])]


# Get the Data
labels = ['Negative for Pneumonia', 'Typical Appearance', 
          'Indeterminate Appearance', 'Atypical Appearance']
dt = dcm_meta.groupby("PatientSex")[labels].sum().reset_index()

# Plotting
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(21,20))
axs = [ax1, ax2, ax3, ax4]

for ax, title in zip(axs, labels):
    sns.barplot(data=dt, x=title, y="PatientSex", 
                ax=ax, palette=my_colors[0:])
    ax.set_title(title, fontsize=25, weight='bold')
    show_values_on_bars(ax, h_v="h", space=0.4)
    ax.set_xticklabels([])
    ax.set_ylabel('')
    ax.set_xlabel('')


# Get the Data
labels = ['Negative for Pneumonia', 'Typical Appearance', 
          'Indeterminate Appearance', 'Atypical Appearance']
dt = dcm_meta.groupby("BodyPartExamined")[labels].sum().reset_index()
dt = dt.sort_values("Negative for Pneumonia", ascending=False)

# Plotting
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(21,20))
axs = [ax1, ax2, ax3, ax4]

for ax, title in zip(axs, labels):
    sns.barplot(data=dt, x=title, y="BodyPartExamined", 
                ax=ax, palette=my_colors)
    ax.set_title(title, fontsize=25, weight='bold')
    show_values_on_bars(ax, h_v="h", space=0.4)
    ax.set_xticklabels([])
    ax.set_ylabel('')
    ax.set_xlabel('')
    fig.tight_layout()


# Get the Data
labels = ['Negative for Pneumonia', 'Typical Appearance', 
          'Indeterminate Appearance', 'Atypical Appearance']
dt = dcm_meta.groupby("PhotometricInterpretation")[labels].sum().reset_index()

# Plotting
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(21,20))
axs = [ax1, ax2, ax3, ax4]

for ax, title in zip(axs, labels):
    sns.barplot(data=dt, x=title, y="PhotometricInterpretation", 
                ax=ax, palette=my_colors[5:])
    ax.set_title(title, fontsize=25, weight='bold')
    show_values_on_bars(ax, h_v="h", space=0.4)
    ax.set_xticklabels([])
    ax.set_ylabel('')
    ax.set_xlabel('')
    fig.tight_layout()

