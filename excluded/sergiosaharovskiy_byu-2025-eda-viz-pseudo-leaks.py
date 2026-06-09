from IPython.display import display, HTML
from IPython.display import Image as Im

import os

from glob import glob
from tqdm.auto import tqdm
tqdm.pandas()


import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler

import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image


import warnings 
warnings.filterwarnings('ignore')

from colorama import Style, Fore

RC = {
    "axes.facecolor": "#F8F8F8",
    "figure.facecolor": "#F8F8F8",
    "axes.edgecolor": "#000000",
    "grid.color": "#EBEBE7" + "30",
    "font.family": "serif",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "grid.alpha": 0.4
}


PALETTE = ['#302c36', '#037d97', '#91013E', '#C09741',
           '#EC5B6D', '#90A6B1', '#6ca957', '#D8E3E2']


sns.set(rc=RC)

class ColorStyle:
    def __init__(self):
        self.red = Style.BRIGHT + Fore.RED
        self.blk = Style.BRIGHT + Fore.BLACK
        self.gld = Style.BRIGHT + Fore.YELLOW
        self.grn = Style.BRIGHT + Fore.GREEN
        self.cya = Style.BRIGHT + Fore.CYAN
        self.mgt = Style.BRIGHT + Fore.MAGENTA
        self.blu = Style.BRIGHT + Fore.BLUE
        self.cya_v2 = self._init_new_color(30, 255, 255)  # 153, 229, 255
        self.res = Style.RESET_ALL

    @staticmethod
    def _init_new_color(r, g, b, background=False):
        return f'\033[{"48" if background else "38"};2;{r};{g};{b}m'

    def color_from_rgb(self, r, g, b):
        return self._init_new_color(r, g, b)


cS = ColorStyle()


!apt-get install tree -qq


!tree -L 1 /kaggle/input/byu-locating-bacterial-flagellar-motors-2025
!tree -L 2 /kaggle/input/byu-locating-bacterial-flagellar-motors-2025 | head -n 10


def magnify(is_test: bool = False):
    base_color = '#1AAB70'
    if is_test:
        highlight_target_row = []
    else:
        highlight_target_row = [dict(selector='tr:last-child',
                                     props=[('background-color', f'{base_color}'+'20')])]

    return [dict(selector="th",
                 props=[("font-size", "11pt"),
                        ('background-color', f'{base_color}'),
                        ('color', 'white'),
                        ('font-weight', 'bold'),
                        ('border-bottom', '0.1px solid white'),
                        ('border-left', '0.1px solid white'),
                        ('text-align', 'right')]),

            dict(selector='th.blank.level0', 
                props=[('font-weight', 'bold'),
                       ('border-left', '1.7px solid white'),
                       ('background-color', 'white')]),

            dict(selector="td",
                 props=[('padding', "0.5em 1em"),
                        ('text-align', 'right')]),

            dict(selector="th:hover",
                 props=[("font-size", "14pt")]),

            dict(selector="tr:hover td:hover",
                 props=[('max-width', '250px'),
                        ('font-size', '14pt'),
                        ('color', f'{base_color}'),
                        ('font-weight', 'bold'),
                        ('background-color', 'white'),
                        ('border', f'1px dashed {base_color}')]),

             dict(selector="caption",
                  props=[(('caption-side', 'bottom'))])] + highlight_target_row

def stylize_min_max_count(pivot_table):
    """Waps the min_max_count pivot_table into the Styler.

        Args:
            df: |min_train| max_train |min_test |max_test |top10_counts_train |top_10_counts_train|

        Returns:
            s: the dataframe wrapped into Styler.
    """
    s = pivot_table
    # A formatting dictionary for controlling each column precision (.000 <-). 
    di_frmt = {(i if i.startswith('m') else i):
              ('{:.3f}' if i.startswith('m') else '{:}') for i in s.columns}

    s = s.style.set_table_styles(magnify(True))\
        .format(di_frmt)\
        .set_caption(f"The train and test datasets min, max, top10 values side by side (hover to magnify).")
    return s
  
    
def stylize_describe(df: pd.DataFrame, dataset_name: str = 'train', is_test: bool = False) -> Styler:
    """Applies .descibe() method to the df and wraps it into the Styler.
    
        Args:
            df: any dataframe (train/test/origin)
            dataset_name: default 'train'
            is_test: the bool parameter passed into magnify() function
                     in order to control the highlighting of the last row.
                     
        Returns:
            s: the dataframe wrapped into Styler.
    """
    s = df.describe().T
    # A formatting dictionary for controlling each column precision (.000 <-). 
    di_frmt = {(i if i == 'count' else i):
              ('{:.0f}' if i == 'count' else '{:.3f}') for i in s.columns}
    
    s = s.style.set_table_styles(magnify(is_test))\
        .format(di_frmt)\
        .set_caption(f"The {dataset_name} dataset descriptive statistics (hover to magnify).")
    return s

def stylize_simple(df: pd.DataFrame, caption: str) -> Styler:
    """Waps the min_max_count pivot_table into the Styler.

        Args:
            df: any dataframe (train/test/origin)

        Returns:
            s: the dataframe wrapped into Styler.
    """
    s = df
    s = s.style.set_table_styles(magnify(True)).set_caption(f"{caption}")
    return s


WORKDIR = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/'
TRAIN_DIR = WORKDIR + 'train'

train = pd.read_csv(WORKDIR + 'train_labels.csv')
sub = pd.read_csv(WORKDIR + 'sample_submission.csv')

display(stylize_simple(train.head(), '* train_labels.csv '))
display(stylize_simple(sub, '* sample_submission.csv '))


stylize_describe(train)


print(f'{cS.red}Number of Unique tomo_ids in train.csv:{cS.res} {len(train.tomo_id.unique())}')


# Group by `tomo_id` and perform aggregation
agg_train = train.groupby('tomo_id').agg(
    mean_voxel_spacing=('Voxel spacing', 'mean'),
    n_unique_array_shape_0=('Array shape (axis 0)', 'nunique'),
    n_unique_array_shape_1=('Array shape (axis 1)', 'nunique'),
    n_unique_array_shape_2=('Array shape (axis 2)', 'nunique'),
    mean_number_of_motors=('Number of motors', 'mean')
).reset_index()

# Check for consistency: Array shapes should be consistent within each `tomo_id`
agg_train['array_shapes_consistent'] = (
    (agg_train['n_unique_array_shape_0'] == 1) &
    (agg_train['n_unique_array_shape_1'] == 1) &
    (agg_train['n_unique_array_shape_2'] == 1)
)

# Validate motor axis consistency: Motor axis should be within the array coordinates
motor_axis_validity = (
    (train['Motor axis 0'] >= 0) &
    (train['Motor axis 0'] <= train['Array shape (axis 0)']) &
    (train['Motor axis 1'] >= 0) &
    (train['Motor axis 1'] <= train['Array shape (axis 1)']) &
    (train['Motor axis 2'] >= 0) &
    (train['Motor axis 2'] <= train['Array shape (axis 2)'])
)

train['motor_axis_valid'] = motor_axis_validity

# Group by `tomo_id` again to validate number of motors matches number of rows per `tomo_id`
motors_per_tomo = train.groupby('tomo_id').size().reset_index(name='num_rows')
motors_per_tomo = motors_per_tomo.merge(agg_train[['tomo_id', 'mean_number_of_motors']], on='tomo_id')

# Validate number of motors for each `tomo_id`
motors_per_tomo['num_motors_match_rows'] = motors_per_tomo['num_rows'] == motors_per_tomo['mean_number_of_motors']

# # Displaying the results
display(stylize_simple(agg_train.head(3), '* sanity check for the shapes consistency across the rows.'))
display(stylize_simple(motors_per_tomo.head(3), '* sanity check for the Number of motors consistency across the rows.'))

# Print validation results
print(f"{cS.red}* Aggregated Train Data{cS.res}")
print(f"Consistency of Array Shapes all: {agg_train['array_shapes_consistent'].all()}")
print(f"Motor Axis Validity all: {motor_axis_validity.all()}")
print(f"Number of Motors Match Rows all: {motors_per_tomo['num_motors_match_rows'].all()}")



stylize_simple(train[train['motor_axis_valid'].eq(False)].head(5), '* train with Number of motors equal to 0, Motor Axes are -1, -1, -1')


stylize_simple(motors_per_tomo[motors_per_tomo['num_motors_match_rows'].eq(False)].head(5), '* number of train rows not matched to number of motors, total 290')


stylize_simple(train[train.tomo_id.eq('tomo_003acc')], '* looking at the specific example where number of motors does not match number of rows')


set(train[train['motor_axis_valid'].eq(False)].tomo_id) ^ set(motors_per_tomo[motors_per_tomo['num_motors_match_rows'].eq(False)].tomo_id)


stylize_simple(train[train.tomo_id.isin(['tomo_2b3cdf', 'tomo_62eea8', 'tomo_c84b8e', 'tomo_e6f7f7'])], '* non zero examples with single row but two motors')


%%time
# Use glob to find all files inside the subdirectories
files = glob(f'{TRAIN_DIR}/**/*')

print(f"{cS.red}* len train files {len(files)}{cS.res}")


def split_path(path):
    from pathlib import Path
    p = Path(path)
    
    # Extract parent directory, base name, and stem
    tomo_id = p.parent.name
    base_name = p.name
    stem = p.stem
    ext = p.suffix
    
    # Extract the slice number from the stem by removing 'slice_' and converting to integer
    slice_num = int(stem.replace('slice_', '')) if 'slice_' in stem else None
    return tomo_id, base_name, slice_num, ext

# Inits train_paths_df
train_paths_df = pd.DataFrame(files, columns=['abs_path_jpeg'])

# Apply the split_path function to the 'abs_path_jpeg' column
train_paths_df[
    ['tomo_id', 'base_name', 'slice_num', 'ext']
] = train_paths_df['abs_path_jpeg'].progress_apply(lambda x: pd.Series(split_path(x)))

stylize_simple(train_paths_df.head(5), '* traversing TRAIN_DIR and collecting the file paths parsed in train_paths_df')


# Display the value counts for 'ext' column
ext_value_counts = train_paths_df.ext.value_counts().to_dict()
print(f"{cS.red}* Value Counts for ext: {cS.res}{ext_value_counts}")  

# Display the number of unique parent directories
len_uniq_tomo = len(train_paths_df.tomo_id.unique())
print(f"{cS.red}* Number of Unique tomo_ids: {cS.res}{len_uniq_tomo}")


num_slices_per_tomo = train_paths_df.groupby(by=['tomo_id'])['slice_num'].count().reset_index() 
num_motors_per_tomo = train.groupby(by=['tomo_id'])['Number of motors'].first().reset_index()
merged_slices_motors = num_slices_per_tomo.merge(num_motors_per_tomo, how='left', on='tomo_id')

# Create a count plot for 'Number of motors'
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=num_motors_per_tomo, x='Number of motors', palette=PALETTE)

# Despine top and right
sns.despine(top=True, right=True)

# Add annotations on top of the bars
for p in ax.patches:
    height = p.get_height()
    total = len(merged_slices_motors)
    percentage = (height / total) * 100
    ax.annotate(f'{height}\n({percentage:.1f}%)', 
                (p.get_x() + p.get_width() / 2., height + 5), 
                ha='center', va='center', 
                xytext=(0, 10), 
                textcoords='offset points')

plt.title('Count of Number of Motors', color='black')
plt.xlabel('Number of Motors')
plt.ylabel('Count')
plt.show()


fig = plt.figure(figsize=(15, 8))  
ax = sns.histplot(data=merged_slices_motors, x='slice_num', hue='Number of motors', palette=PALETTE, multiple='dodge', shrink=0.8)

# Add annotations on top of the bars
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(
            f'{int(height)}',                          # Text to display (height of the bar)
            (p.get_x() + p.get_width() / 2., height),  # Position of the annotation
            ha='center',    # Horizontal alignment
            va='bottom',    # Vertical alignment
            fontsize=8,     # Font size
            color='black',  # Text color
            fontweight='bold'
        )

# Set x-axis ticks with a step size of 50
ax.set_xticks(range(250, int(merged_slices_motors['slice_num'].max()) + 50, 50))
ax.set_ylim(0, 320)
sns.despine(top=True, right=True)
plt.title('Number of motors vs number of slices per tomo_id', color='black')
plt.show()


%%time
from multiprocessing import Pool, cpu_count

def process_group(tomo_id, group):
    volume = []
    for _, row in group.iterrows():
        img = Image.open(row['abs_path_jpeg'])
        img_array = np.array(img)
        volume.append(img_array)
    
    volume = np.stack(volume)
    z, y, x = volume.shape
    
    # Calculate volume size in megabytes
    volume_size_mb = (volume.nbytes) / (1024 * 1024)
    
    return {
        'tomo_id': tomo_id,
        'min': np.min(volume),
        'max': np.max(volume),
        'std': np.std(volume),
        'mean': np.mean(volume),
        'z_shape': z,
        'y_shape': y,
        'x_shape': x,
        'volume_size_mb': volume_size_mb  
    }

# Sort the dataframe by parent_dir and slice_num
train_paths_df = train_paths_df.sort_values(by=['tomo_id', 'slice_num'])

# Group by parent_dir
grouped = list(train_paths_df.groupby('tomo_id'))[:2]

# Use multiprocessing to process groups in parallel
if os.path.exists('/kaggle/input/2025-byu-locating-bacterial-motors-public-repo/volume_stats.csv'):
    results_df = pd.read_csv('/kaggle/input/2025-byu-locating-bacterial-motors-public-repo/volume_stats.csv')
else:
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.starmap(process_group, [(tomo_id, group) for tomo_id, group in grouped]), desc="Processing groups"))

    # Combine results into a dataframe and save
    results_df = pd.DataFrame(results)
    results_df.to_csv('volume_stats.csv', index=False)
    
stylize_simple(results_df.head(5), '* results_df combines tomograms as volumes and provides basic volume stats')


# Calculate aspect ratio (y_shape / x_shape)
results_df['aspect_ratio'] = results_df['y_shape'] / results_df['x_shape']
merged_df = results_df.merge(num_motors_per_tomo, on='tomo_id', how='left')

# Create a 3x2 grid of subplots
fig, axes = plt.subplots(3, 2, figsize=(15, 18))

# Plot mean distribution
sns.histplot(data=merged_df, x='mean', hue='Number of motors', palette=PALETTE, alpha=0.7, multiple='stack', kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Mean Distribution for Each Tomogram', color='black')
axes[0, 0].set_xlabel('Mean Intensity')
axes[0, 0].set_ylabel('Frequency')

# Plot std distribution
sns.histplot(data=merged_df, x='std', hue='Number of motors', palette=PALETTE, alpha=0.7, multiple='stack', kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Standard Deviation Distribution for Each Tomogram', color='black')
axes[0, 1].set_xlabel('Standard Deviation')
axes[0, 1].set_ylabel('Frequency')

# Plot y_shape distribution
sns.histplot(data=merged_df, x='y_shape', color=PALETTE[1], kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Y Shape Distribution for Each Tomogram', color='black')
axes[1, 0].set_xlabel('Y Shape')
axes[1, 0].set_ylabel('Frequency')

# Plot x_shape distribution
sns.histplot(data=merged_df, x='x_shape', color=PALETTE[1], kde=True, ax=axes[1, 1])
axes[1, 1].set_title('X Shape Distribution for Each Tomogram', color='black')
axes[1, 1].set_xlabel('X Shape')
axes[1, 1].set_ylabel('Frequency')

# Scatterplot of x_shape vs y_shape
sns.scatterplot(data=merged_df, x='x_shape', y='y_shape', hue='Number of motors', palette=PALETTE, alpha=0.7, ax=axes[2, 0])
axes[2, 0].set_title('X Shape vs Y Shape', color='black')
axes[2, 0].set_xlabel('X Shape')
axes[2, 0].set_ylabel('Y Shape')

# Histplot of aspect ratio
sns.histplot(data=merged_df, x='aspect_ratio', hue='Number of motors', palette=PALETTE, alpha=0.7, multiple='stack', kde=True, ax=axes[2, 1])
axes[2, 1].set_title('Aspect Ratio Distribution', color='black')
axes[2, 1].set_xlabel('Aspect Ratio (y_shape / x_shape)')
axes[2, 1].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(15, 6))
sns.histplot(data=merged_df, x='volume_size_mb', hue='Number of motors', palette=PALETTE)
sns.despine(top=True, right=True)
plt.title('Histogram of tomogram sizes in MB', color='black')
plt.show()


stylize_simple(merged_df[merged_df['aspect_ratio'].lt(0.95)], 'possible apect ratio (AR) leak - AR < 0.95 = 0 Number of motors')


!cp "/kaggle/input/2025-byu-locating-bacterial-motors-public-repo/Screencastfrom03-09-2025025314PM-ezgif.com-video-to-gif-converter.gif" .


# import imageio.v2 as imageio
# import vtkmodules.all as vtk
# from vtkmodules.util.numpy_support import numpy_to_vtk

# folder_path = WORKDIR + "/data/train/tomo_3e7407"

# slice_files = sorted([file for file in os.listdir(folder_path) if file.endswith('.jpg')])
# slices = [imageio.imread(os.path.join(folder_path, file)) for file in slice_files]

# volume = np.stack(slices, axis=0)
# volume = (1.0 - (volume - np.min(volume)) / (np.max(volume) - np.min(volume))) * 255
# volume = volume.astype(np.uint8)

# vtk_data = numpy_to_vtk(volume.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
# image_data = vtk.vtkImageData()
# image_data.SetDimensions(volume.shape[::-1])
# image_data.GetPointData().SetScalars(vtk_data)

# mapper = vtk.vtkSmartVolumeMapper()
# mapper.SetInputData(image_data)

# volume_property = vtk.vtkVolumeProperty()
# composite_function = vtk.vtkPiecewiseFunction()
# composite_function.AddPoint(0, 0.0)
# composite_function.AddPoint(255, 0.01)
# volume_property.SetScalarOpacity(composite_function)

# color = vtk.vtkColorTransferFunction()
# color.AddRGBPoint(0, 0.0, 0.0, 0.0)
# color.AddRGBPoint(255, 1.0, 1.0, 1.0)
# volume_property.SetColor(color)

# volume_actor = vtk.vtkVolume()
# volume_actor.SetMapper(mapper)
# volume_actor.SetProperty(volume_property)

# renderer = vtk.vtkRenderer()
# renderer.AddVolume(volume_actor)
# renderer.SetBackground(0, 0, 0)

# render_window = vtk.vtkRenderWindow()
# render_window.AddRenderer(renderer)
# render_window.SetSize(1280, 720)

# interactor = vtk.vtkRenderWindowInteractor()
# interactor.SetRenderWindow(render_window)

# interactor_style = vtk.vtkInteractorStyleTrackballCamera()
# interactor.SetInteractorStyle(interactor_style)

# #render_window.Render()


# interactor.Start()




