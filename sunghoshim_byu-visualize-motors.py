import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


TRAIN_DIR = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train'


df = pd.read_csv("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv")
print(df.shape)
df.head()


df_motors = df[df['Number of motors'] > 0].copy()
df_motors = df_motors.sort_values(['tomo_id', 'Motor axis 0']).reset_index(drop=True)
print(df_motors.shape)
df_motors.head()


print('min(cy - 0):', df_motors['Motor axis 1'].min())
print('min(cx - 0):', df_motors['Motor axis 2'].min())


print('min(yshape - cy):', (df_motors['Array shape (axis 1)'] - df_motors['Motor axis 1']).min())
print('min(xshape - cx):', (df_motors['Array shape (axis 2)'] - df_motors['Motor axis 2']).min())


def plot_motor(index, bbox_size=60, z_offsets=range(0, 1)):
    """Plot a motor slice with a bounding box and its cropped region.

    Parameters:
    - index (int): Index of the motor data from `df_motors`.
    - bbox_size (int, optional): Size of the bounding box (default is 60).
    - z_offsets (iterable, optional): Range of z-axis offsets to plot (default is range(0, 1)).

    Examples:
    - range(0, 1) → [0]
    - range(-3, 4) → [-3, -2, -1, 0, 1, 2, 3] (Plots 7 slices around z)
    """
    for z_offset in z_offsets:
        ser = df_motors.loc[index]
        tomo_id = ser['tomo_id']
        cz = int(ser['Motor axis 0'])
        z = cz + z_offset        
        cy = int(ser['Motor axis 1'])
        cx = int(ser['Motor axis 2'])
        zshape = int(ser['Array shape (axis 0)'])
        yshape = int(ser['Array shape (axis 1)'])
        xshape = int(ser['Array shape (axis 2)'])

        if z < 0 or z >= zshape:
            print(f'Warning! {tomo_id}. z({z}) is out of range')
            continue;

        slice_filename = f"{TRAIN_DIR}/{tomo_id}/slice_{z:04d}.jpg"

        half_w = bbox_size / 2
        half_h = bbox_size / 2

        if cx - half_w < 0:
            half_w = cx
        elif cx + half_w >= xshape:
            half_w = xshape - cx

        if cy - half_h < 0:
            half_h = cy
        elif cy + half_h >= yshape:
            half_h = yshape - cy
        
        x1 = cx - half_w
        y1 = cy - half_h
        x2 = cx + half_w        
        y2 = cy + half_h
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), )
        img = plt.imread(slice_filename)
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title(f'({index}) {tomo_id} ({cz}{z_offset:+d}, {cy}, {cx})')
        
        rect = patches.Rectangle(
            xy=(x1, y1), width=(x2 - x1), height=(y2 - y1), 
            linewidth=1, edgecolor='lightgreen', facecolor='none',
        )
        axes[0].add_patch(rect)

        cropped = img[int(y1):int(y2+1), int(x1):int(x2+1)]
        axes[1].imshow(
            cropped, cmap='bone',  # gray, bone, viridis, cividis, hot
            interpolation='bicubic'  # none, bilinear(default), bicubic
        )
        
        plt.show()


for i in range(20):
    plot_motor(i, bbox_size=60)


for box_size in [24, 48, 60, 100]:
    plot_motor(13, bbox_size=box_size)


plot_motor(14, bbox_size=60, z_offsets=range(-4, 5))

