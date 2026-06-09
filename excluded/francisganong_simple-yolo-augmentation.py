import glob
import os

import numpy as np
import pandas as pd
from typing import List, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from PIL import Image, ImageDraw


#%env JOBLIB_TEMP_FOLDER=/tmp
# this may be necessary; because I was getting 'out of space' errors.


my_data_version = 14


class SliceAndTxt:
    def __init__(self, slice_base_dir: str, label_dir: Optional[str], slice_name: str):
        """
        Initialize the SliceAndTxt class.

        :param slice_base_dir: Directory containing the slice images.
        :param label_dir: Directory containing the labels (optional).
        :param slice_name: Name of the slice.
        """
        self.slice_base_dir = slice_base_dir
        self.label_dir = label_dir
        self.slice_name = slice_name
        self.img: Optional[Image.Image] = None
        # self.exif: Optional[dict] = None
        self.annotations_df: Optional[pd.DataFrame] = None
        self.output_tag = ''

    def read(self, do_print = False) -> None:
        """
        Read the slice image and its annotations (if label_dir is provided).
        """
        slice_file = os.path.join(self.slice_base_dir, self.slice_name) + '.jpg'

        if do_print:
            print (f'{slice_file=}')

        self.tomo_name, self.z,self.y,self.x = self.get_coords_from_filename()
        
        try:
            self.img = Image.open(slice_file)
            # self.exif = self.img.getexif()
        except FileNotFoundError:
            print(f"Error: File not found - {slice_file}")
        except Exception as e:
            print(f"Error reading image: {e}")
            
        self.read_annotations(do_print = do_print)
    
    def read_annotations(self, do_print = False) -> None:
        """
        Read the annotations for the slice (if label_dir is provided).
        """
        if not self.label_dir:
            return

        label_file = os.path.join(self.label_dir, self.slice_name) + '.txt'
        if do_print:
            print (f'{label_file=}')
        annotations = []
        img_width, img_height = self.img.size

        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                
                tomo, z,y,x = self.get_coords_from_filename()
                for line in f:
                    values = line.strip().split() # 0 x_center_norm y_center_norm, x_width_norm, y_width_norm
                    class_id = int(values[0])

                    _, x_center_norm, y_center_norm, width, height = values
                    x_center_norm = float(x_center_norm)
                    y_center_norm = float(y_center_norm)
                    x2 = x_center_norm * img_width
                    y2 = y_center_norm * img_height

                    assert abs(x-x2) < 0.0001, f'{x=}, {x2=}'
                    assert abs(y-y2) < 0.0001, f'{y=}, {y2=}'

                    annotations.append({
                        'class_id': class_id,
                        'width': width,
                        'height': height,
                        'x':x, 
                        'y':y, 
                        'z':z
                    })
            self.annotations_df = pd.DataFrame(annotations)
        else:
            self.annotations_df = pd.DataFrame()

    def get_coords_from_filename(self):

        name_coords = self.slice_name.split("tomo_")
        coords = name_coords[1].split('_')
        tomo = coords[0]
        z = int(coords[1][1:])
        y = int(coords[2][1:])
        x = int(coords[3][1:5])

        return tomo, z, y, x 

    def output_slice_name(self):
        output_name = f'tomo_{self.tomo_name}_z{self.z:04d}_y{self.y:04d}_x{self.x:04d}{self.output_tag}'
        return output_name
    

    def write(self, out_slice_base_dir, out_label_base_dir,  do_print: bool = False) -> None:
        """
        Write the slice image and its annotations (if label_dir is provided).

        :param new_name: New name for the slice image and annotations.
        :param do_print: Whether to print debug information.
        """
        output_name = f'{out_slice_base_dir}/{self.output_slice_name()}.jpg'

        os.makedirs(os.path.dirname(output_name), exist_ok = True)
        self.img.save(output_name)
        if do_print:
            print(f"Wrote image of size {self.img.size} to {output_name}")
        if self.label_dir:
            output_label_name = output_name.replace('image', 'label').replace('.jpg', '.txt')

            if do_print:
                print(f'{output_name=}\n{output_label_name=}')
            self.write_annotations(output_label_name,  do_print)

    def write_annotations(self, output_label_name,  do_print: bool = False) -> None:
        """
        Write the annotations for the slice (if label_dir is provided).

        :param do_print: Whether to print debug information.
        """
        if not self.label_dir:
            return

        img_width, img_height = self.img.size

        df = self.annotations_df.copy()
        df['class'] = 0
        df['x_center_norm'] = df['x'] / img_width
        df['y_center_norm'] = df['y'] / img_height

        out_df = df[['class', 'x_center_norm', 'y_center_norm', 'width', 'height']]
         
        os.makedirs(os.path.dirname(output_label_name), exist_ok = True)
        out_df.to_csv(output_label_name, sep=' ', header=False, index=False)
        if do_print:
            print(f"Wrote {out_df.shape[0]} lines to {output_label_name}")

    def print(self, print_details: bool = True) -> None:
        """
        Print the details of the slice and its annotations.

        :param print_details: Whether to print detailed information.
        """
        print(f"{self.slice_base_dir=}\n{self.label_dir=}\n{self.slice_name=}\n{self.output_tag=}")
    
        if print_details and self.annotations_df is not None:
            print(f"\n{self.annotations_df=}")
    
    def show(self) -> None:
        """
        Show the slice image with annotations (if label_dir is provided).
        """
        if  self.annotations_df is None:
            self.img.show()
            return

        num_samples = 1
        rows = int(np.ceil(num_samples / 2))
        cols = min(num_samples, 2)
        fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
        
        # Handle the case of a single subplot
        if num_samples == 1:
            axes = np.array([axes])
    
        # Flatten axes array for easy indexing
        axes = axes.flatten()

        overlay = Image.new('RGBA', self.img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        img_width, img_height = self.img.size

        for _, ann in self.annotations_df.iterrows():
            x = float(ann['x'])
            y = float(ann['y'])
            width = 24
            height = 24
            x1 = max(0, int(x - width / 2))
            y1 = max(0, int(y - height / 2))
            x2 = min(img_width, int(x + width / 2))
            y2 = min(img_height, int(y + height / 2))
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 200))

        annotated_img = Image.alpha_composite(self.img.convert('RGBA'), overlay).convert('RGB')
        axes[0].imshow(np.array(annotated_img))

        plt.show()

    def flip_horizontal(self) -> None:
        """
        Flip the slice image and annotations horizontally (if label_dir is provided).
        """
        img_width, _ = self.img.size
        self.img = self.img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.label_dir and self.annotations_df is not None:

            # self.annotations_df['x_center'] = img_width - self.annotations_df['x_center']
            # self.annotations_df['x_center_norm'] = 1 - self.annotations_df['x_center_norm']
            self.annotations_df['x'] = img_width - self.annotations_df['x']
            self.x = img_width - self.x
            
        self.output_tag += '-fh'

    def flip_vertical(self) -> None:
        """
        Flip the slice image and annotations vertically (if label_dir is provided).
        """
        _, img_height = self.img.size
        self.img = self.img.transpose(Image.FLIP_TOP_BOTTOM)
        if self.label_dir and self.annotations_df is not None:            
            # self.annotations_df['y_center'] = img_height - self.annotations_df['y_center']
            # self.annotations_df['y_center_norm'] = 1 - self.annotations_df['y_center_norm']
            self.annotations_df['y'] = img_height - self.annotations_df['y']
            self.y = img_height - self.y  # used for the filename
            
        self.output_tag += '-fv'


my_data_version = 14


!mkdir -p tmp/image tmp/label
base_dir = f'/kaggle/input/parse-data/yolo_dataset_{my_data_version}/images/train'
label_dir = f'/kaggle/input/parse-data/yolo_dataset_{my_data_version}/labels/train'
out_dir = 'tmp/image'
out_label_dir = 'tmp/label'

slice_name = os.listdir(base_dir)[40].replace('.jpg', '')


ls /kaggle/input/parse-data/


s1 = SliceAndTxt(base_dir, label_dir, slice_name)
s1.read()
s1.write(out_dir, out_label_dir)
s1.show()
s1.print()

print ('on to s2')
s2 = SliceAndTxt(out_dir, out_label_dir, slice_name)
s2.read()
s2.show()
s2.print()


# check that horizontal flip works ok
s3 = SliceAndTxt(base_dir, label_dir, slice_name)
s3.read()
s3.flip_horizontal()
s3.write(out_dir, out_label_dir)
fh_slice_name = s3.output_slice_name()
s3.show()
s3.print()


print ('on to s3b')

s3b = SliceAndTxt(out_dir, out_label_dir,  fh_slice_name)
s3b.read()

s3b.show()
s3b.print()



# check vertical flip

s4 = SliceAndTxt(base_dir, label_dir, slice_name)
s4.read()
s4.show()
s4,print()
s4.flip_vertical()
s4.write(out_dir, out_label_dir)
fh_slice_name = s4.output_slice_name()
s4.show()
s4.print()


s4b = SliceAndTxt(out_dir, out_label_dir,  fh_slice_name)
s4b.read()

s4b.show()
s4b.print()



# class for handling all the slices in a tomogram

class TomogramAndTxts:
    def __init__(self, base_dir: str, label_dir: Optional[str], tomogram_name: str, do_print = False):
        """
        Initialize the TomogramAndTxts class.

        :param base_dir: Directory containing the tomogram images.
        :param label_dir: Directory containing the labels (optional).
        :param tomogram_name: Name of the tomogram.
        """
        self.base_dir = base_dir
        self.label_dir = label_dir
        self.tomogram_name = tomogram_name
        self.slices: List[SliceAndTxt] = []

        # Find slice names
        glob_str = f'{self.base_dir}/tomo_{self.tomogram_name}_*.jpg'
        slice_full_names = glob.glob(glob_str)

        if do_print:
            print (f'{glob_str=}\n{slice_full_names=}')
        self.slice_names = [os.path.basename(slice_full_name).replace('.jpg', '') for slice_full_name in slice_full_names]

        # Initialize slices
        self.slices = [SliceAndTxt(self.base_dir, self.label_dir, slice_name,) for slice_name in self.slice_names]

        # Read slices
        for slice in self.slices:
            slice.read()

    def write(self, out_slice_base_dir, out_label_base_dir, do_print: bool = False) -> None:
        """
        Write the tomogram slices and their labels (if label_dir is provided).

        :param do_print: Whether to print debug information.
        """
        for slice in self.slices:
            slice.write(out_slice_base_dir, out_label_base_dir, do_print=do_print)
        if do_print:
            print(f"Wrote tomogram with {len(self.slices)} slices and {self.count_motors()} motors.")

    def flip_horizontal(self) -> None:
        """
        Flip the tomogram slices horizontally.
        """
        self._apply_to_slices("flip_horizontal")

    def flip_vertical(self) -> None:
        """
        Flip the tomogram slices vertically.
        """
        self._apply_to_slices("flip_vertical")

    def show(self) -> None:
        """
        Show the tomogram slices.
        """
        self._apply_to_slices("show")

    def print(self) -> None:
        """
        Print the details of the tomogram and its slices.
        """
        print(f"{self.base_dir=}\n{self.label_dir=}\n{self.tomogram_name=}\n")
        print(f"Total motors = {self.count_motors()}")

    def count_motors(self) -> int:
        """
        Count the total number of motors in the tomogram.  /this is quite wrong...

        :return: Total number of motors.
        """
        total_motors = 0
        for slice in self.slices:
            if slice.annotations_df is not None:
                total_motors += slice.annotations_df.shape[0]
        return total_motors

    def _apply_to_slices(self, method_name: str) -> None:
        """
        Apply a method to all slices in the tomogram.

        :param method_name: Name of the method to apply.
        """
        for slice in self.slices:
            method = getattr(slice, method_name, None)
            if callable(method):
                method()
            else:
                print(f"Method {method_name} not found in SliceAndTxt.")



data_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/"
def augmentation_vertical_and_horizontal (data_path = data_path, 
                                          image_path = f'/kaggle/input/parse-data/yolo_dataset_{my_data_version}/images/train', 
                                          label_path = f'/kaggle/input/parse-data/yolo_dataset_{my_data_version}/labels/train',
                                          output_image_path = f'/kaggle/working/yolo_dataset_{my_data_version}/images/train',
                                          output_label_path = f'/kaggle/working/yolo_dataset_{my_data_version}/labels/train',
                                          do_print = False,
                                          N_tomos = -1,
                                         ):

    start_time = time.time()

    labels_df = pd.read_csv(os.path.join(data_path, "train_labels.csv"))
    tomo_df = labels_df[labels_df['Number of motors'] > 0].copy()
    unique_tomos = tomo_df['tomo_id'].str.replace('tomo_', '').unique()
    
    for tomo in tqdm(unique_tomos if N_tomos == -1 else unique_tomos[:N_tomos]):

        tomo_v = TomogramAndTxts(image_path, label_path,  tomo, do_print = do_print )
        tomo_v.write(output_image_path, output_label_path, do_print = do_print)
        tomo_v.flip_vertical()
        tomo_v.write(output_image_path, output_label_path, do_print = do_print)

        if do_print:
            print (f'{tomo=}')
        t = TomogramAndTxts(image_path, label_path,  tomo, do_print = do_print)
        # print (f'{t.count_motors()=}')
        t.write(output_image_path, output_label_path, do_print = do_print)
        t.flip_horizontal()
        t.write(output_image_path, output_label_path, do_print = do_print)
        t.flip_vertical()
        t.write(output_image_path, output_label_path, do_print = do_print)

    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")


labels = pd.read_csv(f'{data_path}/train_labels.csv')




def write_labels(labels_df, output_path, labels_df_columns, suffix ):
    out_df = labels_df[labels_df_columns]
    output_filename = os.path.join(output_path, f'train_labels{suffix}.csv') 
    out_df.to_csv(output_filename, index=False)
    print(f'wrote {out_df.shape} to {output_filename}')
    
def labels_flip_vertical( df):
    df = df.copy()
    df['Motor axis 1'] = df['height'] - df['Motor axis 1']  
    df.loc[df['Number of motors'] == 0, 'Motor axis 1']  = -1
    assert min(df['Motor axis 1'].values) >= -1 
    return df

def labels_flip_horizontal( df):
    df = df.copy()
    df['Motor axis 2'] = df['width'] - df['Motor axis 2']
    df.loc[df['Number of motors'] == 0, 'Motor axis 2'] = -1
    assert min(df['Motor axis 2'].values) >= -1 
    return df

def get_sizes(tomo, image_path = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train/', do_print = False):

    if do_print:
        print(f'{image_path=}')

    img_files = glob.glob(f'{image_path}/*{tomo}*/slice*.jpg')
    if do_print:
        print(f'{img_files=}')

    if len(img_files) == 0:
        print(f'no img_files found for {image_path}/*{tomo}*')
        return None

    img_file = img_files[0]

    assert img_file.endswith('.jpg')

    name_coords = img_file.split("tomo_")
    coords = name_coords[1].split('_')
    tomo_id = f'tomo_{coords[0].split("/")[0]}'
    
    if do_print:
        print (f'{tomo=}, {tomo_id=},{img_file=}, {name_coords=}')
    

    img = Image.open(img_file)
    img_width, img_height = img.size
    out_df = pd.DataFrame({'tomo_id':tomo_id, 'height':img_height, 'width':img_width}, index=[0])#.drop_duplicates()

    if do_print:
        print (f'{tomo_id=}, {img_file=}, {name_coords=}, {img_width=}, {img_height=}, {out_df=}')
    return out_df    


import os
import pandas as pd

os.makedirs('/kaggle/working/labels', exist_ok = True)

data_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/"
def augmentation_vertical_and_horizontal_label_df (data_path = data_path, 
                                                   image_path = f'/kaggle/input/parse-data/yolo_dataset_{my_data_version}/images/train', 
                                                   output_path = '/kaggle/working/labels/',                                                   
                                                   do_print = False,
                                                   ):

    labels_df = pd.read_csv(os.path.join(data_path, "train_labels.csv"))
    labels_df_columns = labels_df.columns
    
    unique_tomos = labels_df['tomo_id'].drop_duplicates()

    if do_print:
        print(f'{unique_tomos=}')

    size_list = []
    
    for tomo in tqdm(unique_tomos):
        sizes = get_sizes(tomo)
        size_list.append(sizes)

    sizes_df = pd.concat(size_list)
    
    if do_print:
        print(f'{len(size_list)=}\n{sizes_df.describe()=}, {unique_tomos.shape=}')
        print(f'{sizes_df=}, {sizes_df["tomo_id"].head()=}, {labels_df["tomo_id"].head()=}')

    labels_df_w_sizes = labels_df.merge(sizes_df, on = 'tomo_id', validate = 'm:1')
    assert labels_df_w_sizes.shape[0] == labels_df.shape[0], f"{labels_df_w_sizes.shape=}, {labels_df.shape=}, {sizes_df.shape=}"

    write_labels(labels_df, output_path=output_path, labels_df_columns=labels_df_columns, suffix = '-plain')

    labels_df_horizontal = labels_flip_horizontal(labels_df_w_sizes.copy())
    write_labels(labels_df_horizontal, output_path=output_path, labels_df_columns=labels_df_columns, suffix = '-fh')

    labels_df_vertical = labels_flip_vertical(labels_df_w_sizes.copy())
    write_labels(labels_df_vertical, output_path=output_path, labels_df_columns=labels_df_columns, suffix = '-fv')

    labels_df_horizontal_and_vertical = labels_flip_horizontal(labels_df_vertical)
    write_labels(labels_df_horizontal_and_vertical, output_path=output_path, labels_df_columns=labels_df_columns, suffix = '-fh-fv')

augmentation_vertical_and_horizontal_label_df()


# augment the training data

augmentation_vertical_and_horizontal (image_path = f'/kaggle/input/parse-data/yolo_dataset_{my_data_version}/images/train', 
                                      label_path = f'/kaggle/input/parse-data/yolo_dataset_{my_data_version}/labels/train', 
                                      output_image_path = f'/kaggle/working/yolo_dataset_{my_data_version}/images/train',
                                      output_label_path = f'/kaggle/working/yolo_dataset_{my_data_version}/labels/train',
                                     )


#augment the validationg data
augmentation_vertical_and_horizontal (image_path = f'/kaggle/input/parse-data/yolo_dataset_{my_data_version}/images/val', 
                                      label_path = f'/kaggle/input/parse-data/yolo_dataset_{my_data_version}/labels/val', 
                                      output_image_path = f'/kaggle/working/yolo_dataset_{my_data_version}/images/val',
                                      output_label_path = f'/kaggle/working/yolo_dataset_{my_data_version}/labels/val',
                                     )


data_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/"
def show_augmentation_vertical_and_horizontal (data_path = data_path, 
                                          # image_path = f'/kaggle/input/parse-data/yolo_dataset_{my_data_version}/images/train', 
                                          # label_path = f'/kaggle/input/parse-data/yolo_dataset_{my_data_version}/labels/train',
                                          output_image_path = f'/kaggle/working/yolo_dataset_{my_data_version}/images/train',
                                          output_label_path = f'/kaggle/working/yolo_dataset_{my_data_version}/labels/train',
                                          do_print = True,
                                          N_tomos = 3):

    start_time = time.time()

    labels_df = pd.read_csv(os.path.join(data_path, "train_labels.csv"))
    tomo_df = labels_df[labels_df['Number of motors'] > 0].copy()
    unique_tomos = tomo_df['tomo_id'].str.replace('tomo_', '').unique()

    print(f'{len(unique_tomos)=}')
    
    for tomo in tqdm(unique_tomos if N_tomos == -1 else unique_tomos[:N_tomos]):
        for variant in ['_fh', '_plain']:
            if do_print:
                print (f'{tomo=}, {variant=}')
                
            t = TomogramAndTxts(image_path+variant, label_path+variant,  tomo, do_print = do_print)
            # print (f'{t.count_motors()=}')
            t.show()

    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")


import filecmp
import os

def compare_directories_recursive(dir1, dir2, ignore_file_set = set()):
    non_equals = 0
    for root1, dirs1, files1 in os.walk(dir1):
        root2 = root1.replace(dir1, dir2, 1)  # Map corresponding directory in dir2

        if not os.path.exists(root2):
            print(f"Directory missing: {root2}")
            return False

        # Compare directories
        dirs2 = set(os.listdir(root2)) - set(files1)  # Filter out files
        dirs1_set = set(dirs1)

        if dirs1_set != dirs2:
            print(f"Different subdirectories in: {root1} vs {root2}\n{dirs1_set=}\n{dirs2=}")
            return False

        # Compare files
        files2 = set(os.listdir(root2)) - set(dirs1)  # Filter out directories
        files1_set = set(files1)

        if (files1_set - ignore_file_set) != (files2 - ignore_file_set):
            print(f"Different files in: {root1} vs {root2}")
            print(f'{files1_set - files2=}')
            print(f'{files2     - files1_set=}')
            intersect = files1_set.intersection(files2)
            print(f'{list(intersect)[:5]=}\n{list(intersect)[-5:]=}')
            return False

        for file in files1:
            path1 = os.path.join(root1, file)
            path2 = os.path.join(root2, file)

            if not filecmp.cmp(path1, path2, shallow=False):
                print(f"File content differs: {path1} vs {path2}")
                non_equals += 1

    if non_equals == 0:
        print("Directories and all subdirectories are identical.")
        return True
    else:
        print(f'{non_equals} files are different')
        return False

# Example usage
dir1 = "path/to/directory1"
dir2 = "path/to/directory2"
compare_directories_recursive(dir1, dir2, set())


!cp -r /kaggle/working/yolo_dataset_{my_data_version} /tmp

!rm /tmp/yolo_dataset_{my_data_version}/*/*/*-fh*
!rm /tmp/yolo_dataset_{my_data_version}/*/*/*-fv*


rm -rf /kaggle/working/yolo_dataset_{my_data_version}/yolo_dataset.andrew_darling/yolo_dataset


dir1 = "/kaggle/input/parse-data/yolo_dataset/labels"
dir2 = f"/tmp/yolo_dataset_{my_data_version}/labels"
assert compare_directories_recursive(dir1, dir2)

