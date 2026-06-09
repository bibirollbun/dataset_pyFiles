import cv2
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  
from colorama import Fore, Back, Style
from prettytable import PrettyTable
from graphviz import Digraph
from IPython.display import display, Image

print(Back.YELLOW + Fore.BLACK + "âœ… OpenCV Version: " + cv2.__version__ + Style.RESET_ALL)
print(Back.YELLOW + Fore.BLACK + "âœ… NumPy Version: " + np.__version__ + Style.RESET_ALL)
print(Back.YELLOW + Fore.BLACK + "âœ… Pandas Version: " + pd.__version__ + Style.RESET_ALL)
print(Back.YELLOW + Fore.BLACK + "âœ… Matplotlib Version: " + matplotlib.__version__ + Style.RESET_ALL)
print(Back.YELLOW + Fore.BLACK + "âœ… mpl_toolkits.mplot3d: Part of Matplotlib (No separate version)" + Style.RESET_ALL)

# Additional Libraries
print(Back.YELLOW + Fore.BLACK + "âœ… PrettyTable Version: " + PrettyTable.__module__ + Style.RESET_ALL)
print(Back.YELLOW + Fore.BLACK + "âœ… Graphviz Version: " + Digraph.__module__ + Style.RESET_ALL)
print(Back.YELLOW + Fore.BLACK + "âœ… IPython Display Module: Available (No version attribute)" + Style.RESET_ALL)



train_labels = pd.read_csv("/kaggle/input/image-matching-challenge-2025/train_labels.csv")
print(Back.GREEN + Fore.BLACK + Style.BRIGHT + "âœ… Successfully imported and stored the dataset." + Style.RESET_ALL)


df_head = train_labels.head(4)
table = PrettyTable()
table.field_names = df_head.columns.tolist()
for row in df_head.itertuples(index=False):
    table.add_row(row)
print(table)


missing_values = train_labels.isnull().sum()
missing_table = PrettyTable()
missing_table.title = "ğŸ”� Missing Values in Dataset"  # Set the table title
missing_table.field_names = ["Column Name", "Missing Values"]
for col, missing in missing_values.items():
    missing_table.add_row([col, missing])
print(missing_table)


dot = Digraph()
dot.node('A', "train_labels['dataset']")
dot.node('B', "train_labels['scene']")
dot.node('C', "train_labels['dataset_scene']\n= train_labels['dataset'] + '/' + train_labels['scene']", shape='oval')
dot.edge('A', 'C')
dot.edge('B', 'C')
dot.render('dataset_scene_graph', format='png', cleanup=False)
dot
display(Image('dataset_scene_graph.png'))


train_labels["dataset_scene"] = train_labels["dataset"]+"/"+train_labels["scene"]


from graphviz import Source
graph_code = """
digraph dataset_scene_deduplication {
    graph [size="15,10"]; // Adjust size width=8, height=5
    rankdir=TB;
    node [shape=box];
    
    subgraph cluster_original {
        label="Original train_labels";
        columns [label="Columns: dataset_scene | image | rotation_matrix | translation_vector", shape=plaintext];
        dataset1 [label="dataset1/sceneA\nimage1, R1, T1"];
        dataset2 [label="dataset1/sceneA\nimage2, R2, T2"];
        dataset3 [label="dataset2/sceneB\nimage3, R3, T3"];
        dataset4 [label="dataset2/sceneB\nimage4, R4, T4"];
        columns -> dataset1 [style=invis];
    }
    
    subgraph cluster_unique {
        label="Unique dataset_scene Rows";
        unique_columns [label="Columns: dataset_scene | image | rotation_matrix | translation_vector", shape=plaintext];
        unique1 [label="dataset1/sceneA\nimage1, R1, T1"];
        unique2 [label="dataset2/sceneB\nimage3, R3, T3"];
        unique_columns -> unique1 [style=invis];
    }
    
    dataset1 -> unique1 [label="Keep First"];
    dataset2 -> unique1 [style=dashed, label="Dropped"];
    dataset3 -> unique2 [label="Keep First"];
    dataset4 -> unique2 [style=dashed, label="Dropped"];
}
"""
graph = Source(graph_code)
graph.render('dataset_scene_deduplication', format='png', cleanup=False)
graph
display(Image('dataset_scene_deduplication.png'))


unique_rows = train_labels.drop_duplicates(subset=['dataset_scene'], keep='first')[['dataset_scene', 'image', 'rotation_matrix', 'translation_vector']]
print(f"The unique combination of dataset and scene with first associated image (including outliers combinations): {len(unique_rows)}")


outlier_code= """
digraph dataset_scene_outliers {
    graph [size="30,20"]; // Adjust size width=8, height=5
    rankdir=LR;
    node [shape=box];
    
    subgraph cluster_dataset_scene {
        label="dataset_scene Column Values";
        ds1 [label="imc2023_haiper/fountain"];
        ds2 [label="imc2023_haiper/bike"];
        ds3 [label="imc2023_haiper/chairs"];
        ds4 [label="imc2023_heritage/outliers", color=red, style=filled, fillcolor=pink];
        ds5 [label="imc2023_heritage/dioscuri"];
        ds6 [label="imc2023_heritage/cyprus"];
        ds7 [label="imc2023_heritage/wall"];
        ds8 [label="imc2023_theather_imc2024_church/church"];
        ds9 [label="imc2023_theather_imc2024_church/kyiv-puppet-theater"];
        ds10 [label="imc2024_dioscuri_baalshamin/baalshamin"];
        ds11 [label="imc2024_dioscuri_baalshamin/outliers", color=red, style=filled, fillcolor=pink];
        ds12 [label="imc2024_dioscuri_baalshamin/dioscuri"];
        ds13 [label="imc2024_lizard_pond/lizard"];
        ds14 [label="imc2024_lizard_pond/outliers", color=red, style=filled, fillcolor=pink];
        ds15 [label="imc2024_lizard_pond/pond"];
        ds16 [label="ETs/outliers", color=red, style=filled, fillcolor=pink];
        ds17 [label="ETs/ET"];
        ds18 [label="ETs/another_ET"];
        ds19 [label="16 more values..."]
    }    
    subgraph cluster_outliers {
        label="Identified Outliers (4)";
        rankdir=LR;
        out1 [label="ETs/outliers", color=red, style=filled, fillcolor=pink];
        out2 [label="imc2023_heritage/outliers", color=red, style=filled, fillcolor=pink];
        out3 [label="imc2024_dioscuri_baalshamin/outliers", color=red, style=filled, fillcolor=pink];
        out4 [label="imc2024_lizard_pond/outliers", color=red, style=filled, fillcolor=pink];
    }
    ds4 -> out2 [style=dashed, color=red];
    ds11 -> out3 [style=dashed, color=red];
    ds14 -> out4 [style=dashed, color=red];
    ds16 -> out1 [style=dashed, color=red];
}
"""
# Render and display the graph
outlier_graph = Source(outlier_code)
outlier_graph.render('outlier', format='png', cleanup=False)
outlier_graph
from IPython.display import display, Image
display(Image('outlier.png'))


unique_rows.to_csv("Image_for_Visual.csv")


data = pd.read_csv("/kaggle/working/Image_for_Visual.csv")
data.columns


if 'dataset_scene' not in train_labels.columns:
    train_labels['dataset_scene'] = train_labels['dataset'] + "_" + train_labels['scene']
unique_rows = train_labels.drop_duplicates(subset=['dataset_scene'], keep='first')[['dataset_scene', 'image', 'rotation_matrix', 'translation_vector']]

base_path = "/kaggle/input/image-matching-challenge-2025/train"
image_paths = [f"{base_path}/{row['dataset_scene'].split('/')[0]}/{row['image']}" for _, row in unique_rows.iterrows()]

# Function to display images
def show_images(image_paths, title="Unique Scene Images", rows=6, cols=6):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(len(axes)):
        if i < len(image_paths) and os.path.exists(image_paths[i]):  # Ensure the image exists
            img = cv2.imread(image_paths[i])
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(img)
            else:
                axes[i].text(0.5, 0.5, "Image Not Loaded", ha='center', va='center', fontsize=12)
        else:
            axes[i].text(0.5, 0.5, "No Image", ha='center', va='center', fontsize=12)

        axes[i].axis("off")

    plt.suptitle(title, fontsize=16)
    plt.show()

show_images(image_paths[:34])


# Sharpening Kernel for Image Enhancement
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

# Function to display images and camera angles
def show_images_with_camera_angles(image_paths, train_labels, rows=34, cols=2):
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3))
    
    for i in range(rows):
        if i >= len(image_paths):
            break
        
        img_path = image_paths[i]
        dataset_scene = data.iloc[i]["dataset_scene"]
        
        # Load Image
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.filter2D(img, -1, sharpen_kernel)
                axes[i, 0].imshow(img)
                axes[i, 0].set_title(f"Scene: {dataset_scene}", fontsize=10)
            else:
                axes[i, 0].text(0.5, 0.5, "Image Not Loaded", ha='center', va='center', fontsize=12)
        else:
            axes[i, 0].text(0.5, 0.5, "No Image", ha='center', va='center', fontsize=12)
        
        axes[i, 0].axis("off")

        #Camera Pose
        rotation_matrix = np.array(data.iloc[i]["rotation_matrix"].split(";"), dtype=float).reshape(3, 3)
        translation_vector = np.array(data.iloc[i]["translation_vector"].split(";"), dtype=float)
        
        #Displaying Rotation Matrix and Translation Vector
        rm_text = f"Rotation Matrix:\n{rotation_matrix}"
        tv_text = f"Translation Vector:\n{translation_vector}"
        axes[i, 0].text(0.5, -0.1, rm_text, ha='center', va='center', fontsize=8, transform=axes[i, 0].transAxes)
        axes[i, 0].text(0.5, -0.3, tv_text, ha='center', va='center', fontsize=8, transform=axes[i, 0].transAxes)
        
        # Compute camera center: -Râ�»Â¹ * T
        camera_center = -np.linalg.inv(rotation_matrix) @ translation_vector
        
        # Plot Camera Position
        ax = fig.add_subplot(rows, cols, 2 * i + 2, projection='3d')
        ax.scatter(camera_center[0], camera_center[1], camera_center[2], c='red', marker='o', label="Camera Position")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        ax.set_title("Camera Position", fontsize=10)
    
    plt.tight_layout()
    plt.show()

show_images_with_camera_angles(image_paths[:34], train_labels)

