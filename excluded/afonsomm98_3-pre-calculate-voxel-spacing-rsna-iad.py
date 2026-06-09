import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

import pydicom

import numpy as np
import polars as pl

import pickle


BASE_PATH = "/kaggle/input/rsna-intracranial-aneurysm-detection"
SERIES_PATH = f"{BASE_PATH}/series"


train_df = pl.read_csv(f"{BASE_PATH}/train.csv")
train_df.head(2)


series_uid, modalities = train_df["SeriesInstanceUID"].to_numpy(), train_df["Modality"].to_numpy()


def make_domain_spacing_dict():
    return { modality: [] for modality in np.unique(modalities) }

make_domain_spacing_dict()


def get_serie_info(serie_uid, domain, store_filenames):
    filenames = os.listdir(f"{SERIES_PATH}/{serie_uid}")
    if len(filenames) == 1:
        return "3d", [serie_uid, domain, filenames if (store_filenames is not None) else None]
    else:
        return "2d", [serie_uid, domain, filenames if (store_filenames is not None) else None]

def divide_series_by_dicom_dim(series_uid, domains, store_filenames):

    series_with_3d_dicoms = []
    series_with_2d_dicoms = []
    
    series_uids = os.listdir(SERIES_PATH)
    n_series = len(series_uids)

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = []
        for i in range(n_series):
            serie_uid = series_uid[i]
            domain = domains[i] if (domains is not None) else None
            futures.append(executor.submit(get_serie_info, serie_uid, domain, store_filenames))

        n_jobs = len(futures)
        for future in tqdm(as_completed(futures), total=n_jobs):
            dim, serie_info = future.result()
            if dim == "3d":
                series_with_3d_dicoms.append(serie_info)
            else:
                series_with_2d_dicoms.append(serie_info)

    return series_with_3d_dicoms, series_with_2d_dicoms


series_with_3d_dicoms, series_with_2d_dicoms = divide_series_by_dicom_dim(series_uid, modalities, True)


series_with_3d_dicoms[0], series_with_2d_dicoms[0]


print(len(series_with_3d_dicoms), len(series_with_2d_dicoms))


def get_dicom3d_spacing(ds):

    pixel_spacing = None
    slice_thickness = None
    
    shared_functional_groups_sequence = getattr(ds, "SharedFunctionalGroupsSequence", None)
    if shared_functional_groups_sequence is not None:
        pixel_measures_sequence = getattr(shared_functional_groups_sequence[0], "PixelMeasuresSequence", None)
        if pixel_measures_sequence is not None:
            pixel_spacing = getattr(pixel_measures_sequence[0], "PixelSpacing", None)
            slice_thickness = getattr(pixel_measures_sequence[0], "SliceThickness", None)
    
    if pixel_spacing is None or slice_thickness is None:
        raise Exception("Missing either Pixel Spacing or Slice Thickness.")
    
    pixel_spacing = [float(axis_spacing) for axis_spacing in pixel_spacing]
    slice_thickness = float(slice_thickness)
    spacing = [*pixel_spacing, slice_thickness]
    
    return spacing


serie_uid_temp = series_with_3d_dicoms[0][0]
instance_filename_temp = series_with_3d_dicoms[0][2][0]
ds_temp = pydicom.dcmread(f"{SERIES_PATH}/{serie_uid_temp}/{instance_filename_temp}", stop_before_pixels=True)
print(get_dicom3d_spacing(ds_temp))


def get_dicom2d_spacing(ds):
    
    # Pixel Spacing (x, y)
    pixel_spacing = getattr(ds, "PixelSpacing", None)
    slice_thickness = getattr(ds, "SliceThickness", None)

    if pixel_spacing is None or slice_thickness is None:
        raise Exception("Missing either Pixel Spacing or Slice Thickness.")
    
    pixel_spacing = [float(axis_spacing) for axis_spacing in pixel_spacing]
    slice_thickness = float(slice_thickness)
    spacing = [*pixel_spacing, slice_thickness]

    return spacing


serie_uid_temp = series_with_2d_dicoms[0][0]
instance_filename_temp = series_with_2d_dicoms[0][2][0]
ds_temp = pydicom.dcmread(f"{SERIES_PATH}/{serie_uid_temp}/{instance_filename_temp}", stop_before_pixels=True)
print(get_dicom2d_spacing(ds_temp))


def get_serie_spacing(serie_uid, instances_filename_l, n_instances):

    if not instances_filename_l:
        instances_filename_l = os.listdir(f"{SERIES_PATH}/{serie_uid}")

    if not n_instances:
        n_instances = len(instances_filename_l)

    if n_instances == 1:
        ds = pydicom.dcmread(f"{SERIES_PATH}/{serie_uid}/{instances_filename_l[0]}", stop_before_pixels=True)
        spacing = np.array(get_dicom3d_spacing(ds))
    else:
        spacings = np.zeros((n_instances, 3))
        for i, instance_filename in enumerate(instances_filename_l):
            ds = pydicom.dcmread(f"{SERIES_PATH}/{serie_uid}/{instance_filename}", stop_before_pixels=True)
            spacings[i] = get_dicom2d_spacing(ds)
        spacings, counts = np.unique(spacings, return_counts=True, axis=0)
        spacing = spacings[np.argmax(counts)]

    return spacing


serie_idx = 1

# With pre-fetched filenames
print("---- Pre-fetched filenames ----\n")

serie_uid_temp = series_with_2d_dicoms[serie_idx][0]
instances_filename_l_temp = series_with_2d_dicoms[serie_idx][2]
spacing = get_serie_spacing(serie_uid_temp, instances_filename_l_temp, None)
print(f"2D DICOMS: {spacing}")

serie_uid_temp = series_with_3d_dicoms[serie_idx][0]
instances_filename_l_temp = series_with_3d_dicoms[serie_idx][2]
spacing = get_serie_spacing(serie_uid_temp, instances_filename_l_temp, 1)
print(f"3D DICOMS: {spacing}\n\n")

# Without pre-fetched filenames
print("---- No pre-fetched filenames ----\n")

serie_uid_temp = series_with_2d_dicoms[serie_idx][0]
spacing = get_serie_spacing(serie_uid_temp, None, None)
print(f"2D DICOMS: {spacing}")

serie_uid_temp = series_with_3d_dicoms[serie_idx][0]
spacing = get_serie_spacing(serie_uid_temp, None, None)
print(f"3D DICOMS: {spacing}")


def get_series_spacing(series_uid, instances_filename_l, n_instances):

    n_series = len(series_uid)
    
    if instances_filename_l is None:
        instances_filename_l = [None] * n_series
    if n_instances is None:
        n_instances = [None] * n_series

    assert len(instances_filename_l) == n_series and len(n_instances) == n_series, "Length of instances_filename_l or n_instances not the same as of series_uid."

    spacings = np.zeros((n_series, 3))
    
    for i in range(n_series):
        spacings[i] = get_serie_spacing(series_uid[i], instances_filename_l[i], n_instances[i])

    return spacings


series_uid_temp = [serie[0] for serie in series_with_3d_dicoms[0:10]]
instances_filename_l_temp = [serie[2] for serie in series_with_3d_dicoms[0:10]]
spacings = get_series_spacing(series_uid_temp, instances_filename_l_temp, None)
print(spacings)


def get_domain_spacings(series, has_filenames, batch_size):
    
    domain_spacing_dict = make_domain_spacing_dict()

    n_series = len(series)
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        job_to_idx_dict = {}
        for i in tqdm(range(0, n_series, batch_size)):
            ixs = slice(i, min(n_series, i+batch_size))
            series_i = series[ixs]
            
            series_uid = [serie[0] for serie in series_i]
            if has_filenames:
                instances_filename_l = [serie[2] for serie in series_i]
            else:
                instances_filename_l = None

            job = executor.submit(get_series_spacing, series_uid, instances_filename_l, None)
            job_to_idx_dict[job] = ixs

        n_jobs = len(job_to_idx_dict)
        for job in tqdm(as_completed(job_to_idx_dict), total=n_jobs):
            try:
                spacings = job.result()
    
                job_i = job_to_idx_dict[job]
                for i, serie in enumerate(series[job_i]):
                    domain = serie[1]
                    domain_spacing_dict[domain].append(spacings[i])
            except Exception as e:
                print(e)

    # Stack lists of numpy arrays
    for domain, spacings in domain_spacing_dict.items():
        if len(spacings) > 0:
            domain_spacing_dict[domain] = np.vstack(spacings)
        else:
            domain_spacing_dict[domain] = None
    
    return domain_spacing_dict


series = series_with_3d_dicoms + series_with_2d_dicoms
domain_spacing_dict_temp = get_domain_spacings(series, True, 32)


pickle.dump(domain_spacing_dict_temp, open("/kaggle/working/domain_spacing_dict.pkl", "wb"))
#domain_spacing_dict_temp = pickle.load(open("/kaggle/working/domain_spacing_dict.pkl", "rb"))


for domain, spacings in domain_spacing_dict_temp.items():
    print(f"{domain}: {np.median(spacings, axis=0)}")




