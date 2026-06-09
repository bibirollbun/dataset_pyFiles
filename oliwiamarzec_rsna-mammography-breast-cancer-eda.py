import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import pydicom as dcm
import seaborn as sns
import matplotlib.pyplot as plt


train_df = pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/train.csv")
test_df = pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/test.csv")


print("Train / test data shape: ", train_df.shape, test_df.shape)
print("Train images folders: ", len(os.listdir("/kaggle/input/rsna-breast-cancer-detection/train_images")))
print("Test images folders: ", len(os.listdir("/kaggle/input/rsna-breast-cancer-detection/test_images")))


train_df.info()


def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return np.transpose(pd.concat([total, percent], axis=1, keys=['Total', 'Percent']))


def unique_values(data):
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    uniques = []
    for col in data.columns:
        unique = data[col].nunique()
        uniques.append(unique)
    tt['Uniques'] = uniques
    return np.transpose(tt)


missing_data(train_df)


unique_values(train_df)


train_df.head(6)


test_df.info()


missing_data(test_df)


unique_values(test_df)


test_df.head()


file_list = []
train_path = "/kaggle/input/rsna-breast-cancer-detection/train_images"
folder_list = list(os.listdir(train_path))
for folder in tqdm(os.listdir(train_path)):
    file_list += [x.split(".dcm")[0] for x in os.listdir(os.path.join(train_path, folder))]
print(len(folder_list), len(file_list))


diff = list(set(folder_list) - set([str(x) for x in train_df.patient_id.unique()]))
print("Differences in patient/folder list: ",len(diff))


diff = list(set(file_list) - set([str(x) for x in train_df.image_id.unique()]))
print("Differences in patient/folder list: ",len(diff))


train_df.laterality.value_counts()


train_df.view.value_counts()


train_df["lv"] = train_df[["laterality", "view"]].apply(lambda x: "_".join(x), axis=1)


train_df["lv"].value_counts()


train_agg_df = train_df.groupby(["machine_id", "site_id"])["image_id"].count().reset_index()
train_agg_df.columns = ["machine_id", "site_id", "count"]
sns.barplot(data=train_agg_df, x="machine_id", y="count", hue="site_id")
plt.title("Number of images per machine id, grouped by site id")
plt.show()


train_agg_df = train_df.groupby(["patient_id"])["image_id"].count().reset_index()
train_agg_df.columns = ["patient_id", "images"]
train_agg_df.head(2)


train_agg_df.images.value_counts()


train_agg_df = train_df.groupby(["patient_id"])["lv"].nunique().reset_index()
train_agg_df.columns = ["patient_id", "lat_view"]


train_agg_df.lat_view.value_counts()


train_agg_df = train_df.groupby(["patient_id", "age"])["image_id"].count().reset_index()
train_agg_df.columns = ["patient", "age", "count"]
sns.histplot(train_agg_df.age, bins=20)
plt.title("Number of patients per age groups")
plt.show()


train_agg_df = train_df.loc[train_df.cancer==1].groupby(["patient_id", "age"])["image_id"].count().reset_index()
train_agg_df.columns = ["patient", "age", "count"]
sns.histplot(train_agg_df.age, bins=20)
plt.title("Number of patients with cancer per age groups")
plt.show()


train_agg_df = train_df.groupby(["patient_id", "cancer"])["image_id"].count().reset_index()
train_agg_df.columns = ["patient_id", "cancer", "count"]


print("Total cases: ", train_agg_df.shape[0])
print("Total patients: ", train_agg_df.patient_id.nunique())
print("Total cancer cases: ", train_agg_df.loc[train_agg_df.cancer==1].patient_id.nunique())
patients_with_cancer = train_agg_df.loc[train_agg_df.cancer==1].patient_id.unique()
pat_with_both = train_agg_df.loc[~train_agg_df.patient_id.isin(patients_with_cancer)].patient_id.nunique()
print("Total diagnosed without cancer: ", pat_with_both)


train_agg2_df = train_agg_df.groupby(["patient_id"])["cancer"].count().reset_index()
train_agg2_df.columns = ["patient_id", "diagnoses"]
train_agg2_df.diagnoses.value_counts()


train_agg_df = train_df.groupby(["patient_id", "difficult_negative_case"])["image_id"].count().reset_index()
train_agg_df.columns = ["patient_id", "difficult_negative_case", "count"]
print("Total cases: ", train_agg_df.shape[0])
print("Total patients: ", train_agg_df.patient_id.nunique())
print("Total difficult_negative_case cases: ", train_agg_df.loc[train_agg_df.difficult_negative_case==1].patient_id.nunique())
patients_with_dnc = train_agg_df.loc[train_agg_df.difficult_negative_case==1].patient_id.unique()
pat_with_both = train_agg_df.loc[~train_agg_df.patient_id.isin(patients_with_dnc)].patient_id.nunique()
print("Total diagnosed without difficult_negative_case: ", pat_with_both)


train_agg_df = train_df.groupby(["patient_id", "biopsy"])["image_id"].count().reset_index()
train_agg_df.columns = ["patient_id", "biopsy", "count"]
print("Total cases: ", train_agg_df.shape[0])
print("Total patients: ", train_agg_df.patient_id.nunique())
print("Total biopsy cases: ", train_agg_df.loc[train_agg_df.biopsy==1].patient_id.nunique())
patients_with_b = train_agg_df.loc[train_agg_df.biopsy==1].patient_id.unique()
pat_with_both = train_agg_df.loc[~train_agg_df.patient_id.isin(patients_with_b)].patient_id.nunique()
print("Total diagnosed without biopsy: ", pat_with_both)


train_agg_df = train_df.groupby(["implant"])["image_id"].count().reset_index()
train_agg_df.columns = ["implant", "count"]
sns.barplot(data=train_agg_df, x="implant", y="count")
plt.title("Number of images for patients with/without implant")
plt.show()


train_agg_df = train_df.groupby(["BIRADS"])["image_id"].count().reset_index()
train_agg_df.columns = ["BIRADS", "count"]
sns.barplot(data=train_agg_df, x="BIRADS", y="count")
plt.title("Number of images per BIRADS value")
plt.show()


train_agg_df = train_df.groupby(["invasive"])["image_id"].count().reset_index()
train_agg_df.columns = ["invasive", "count"]
sns.barplot(data=train_agg_df, x="invasive", y="count")
plt.title("Number of images for patients with/without invasive")
plt.show()


train_agg_df = train_df.groupby(["laterality", "view"])["image_id"].count().reset_index()
train_agg_df.columns = ["laterality", "view", "count"]
fig, ax = plt.subplots()
sns.barplot(data=train_agg_df, x="view", y="count", hue="laterality")
ax.set_yscale('log')
ax.set_ylabel('Number of images (log scale)')
plt.title("Number of images per view, grouped by laterality")
plt.show()


train_agg_df = train_df.groupby(["biopsy", "cancer"])["image_id"].count().reset_index()
train_agg_df.columns = ["biopsy", "cancer", "count"]
fig, ax = plt.subplots()
sns.barplot(data=train_agg_df, x="biopsy", y="count", hue="cancer")
ax.set_yscale('log')
ax.set_ylabel('Number of images (log scale)')
plt.title("Number of images per biopsy, grouped by cancer")
plt.show()


train_agg_df = train_df.groupby(["biopsy", "difficult_negative_case"])["image_id"].count().reset_index()
train_agg_df.columns = ["biopsy", "difficult_negative_case", "count"]
fig, ax = plt.subplots()
sns.barplot(data=train_agg_df, x="biopsy", y="count", hue="difficult_negative_case")
ax.set_yscale('log')
ax.set_ylabel('Number of images (log scale)')
plt.title("Number of images per biopsy, grouped by difficult_negative_case")
plt.show()


def extract_dicom_data(data_path, patient_id):
    images_path = os.path.join(data_path,patient_id)
    for image in os.listdir(images_path):
        image_id = image.split(".dcm")[0]
        image_path = os.path.join(images_path, image)
        data_row_img_data = dcm.read_file(image_path)
        print("=================================================")
        print(f"Patient: {patient_id} Image_id: {image_id}")
        print("=================================================")
        print(data_row_img_data)
        print("=================================================\n\n")


patient_id = '10006'
extract_dicom_data(train_path, patient_id)


def process_dicom_data(data_path, patient_id, dicom_features):
    images_path = os.path.join(data_path,str(patient_id))
    for image in os.listdir(images_path):
        try:
            image_id = image.split(".dcm")[0]
            image_path = os.path.join(images_path, image)
            data_row_img_data = dcm.read_file(image_path)
            rows = data_row_img_data.Rows
            columns = data_row_img_data.Columns
            content_date = data_row_img_data.ContentDate
            photometric_interpretation = data_row_img_data.PhotometricInterpretation
            dicom_features.append((image_id, rows, columns, content_date, photometric_interpretation))
        except Exception as ex:
            print(ex)
            continue


dicom_features = []
for patient_id in tqdm(train_df.patient_id.unique()):
    process_dicom_data(train_path, patient_id, dicom_features)


features_df = pd.DataFrame(dicom_features)
features_df.columns = ["image_id", "rows", "columns", "content_date", "photometric_interpretation"]
features_df["image_id"] = features_df["image_id"].apply(lambda x: int(x))
train_add_df = train_df.merge(features_df, on="image_id")


train_add_df.head()

