import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import imageio


df0=pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
print(df0.columns.tolist())
df0['IDi']=df0['ID'].apply(lambda x:x.split('_')[0])
display(df0)
unique_names = df0['IDi'].unique()


unique_resnames = df0['resname'].unique()
print(unique_resnames)
colors = plt.cm.jet(np.linspace(0, 1, len(unique_resnames)))
color_map = dict(zip(unique_resnames, colors))


# Create directory for images
os.makedirs("frames", exist_ok=True)

for i, name in enumerate(unique_names):
    df = df0[df0['IDi'] == name]

    # Create 3D plot
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    for resname in unique_resnames:
        subset = df[df['resname'] == resname]
        ax.scatter(subset['x_1'], subset['y_1'], subset['z_1'], label=resname, color=color_map[resname], alpha=0.3)

    # Fix axis range
    ax.set_xlim(-200,200)
    ax.set_ylim(-200,200)
    ax.set_zlim(-200,200)

    # Labels & Legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(title=name)

    # Save figure
    filename = f"frames/frame_{i:03d}.png"
    plt.savefig(filename)
    plt.close(fig)



image_files = sorted([f"frames/{f}" for f in os.listdir("frames") if f.endswith(".png")])


gif_filename = "animation.gif"
with imageio.get_writer(gif_filename, mode='I', duration=1.0, loop=0) as writer:  
    for img_file in image_files:
        image = imageio.imread(img_file)
        writer.append_data(image)

print("GIF saved as", gif_filename)


from IPython.display import Image
Image(open('./animation.gif','rb').read())




