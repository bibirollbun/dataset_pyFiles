# CODE REQUIRED TO SHOW IMAGES
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

PTD = "/kaggle/input/amazon-site-discoveries-from-cbers" 

# 1. Define the images list once
images = [
    'CBERS4/CBERS4-discovery1.png', 'CBERS4/CBERS4-discovery2.png', 'CBERS4/CBERS4-discovery3.png', 
    'CBERS4/CBERS4-discovery4a.png','CBERS4/CBERS4-discovery4b.png','CBERS4/CBERS4-discovery4c.png','CBERS4/CBERS4-discovery4d.png',
    'CBERS4a/CBERS4a-discovery1.png', 'CBERS4a/CBERS4a-discovery2.png', 'CBERS4a/CBERS4a-discovery3.png', 'CBERS4a/CBERS4a-discovery4.png',
]
images_path = [os.path.join(PTD, img) for img in images]

def show_images_by_idx(idx_list, images_list=images, path=PTD):
    fig, axs = plt.subplots(1, len(idx_list), figsize=(5 * len(idx_list), 5))
    if len(idx_list) == 1:
        axs = [axs]
    for ax, idx in zip(axs, idx_list):
        images_path = os.path.join(path, images_list[idx])  # 0-based index
        img = mpimg.imread(images_path)
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


show_images_by_idx([0,1,2])


show_images_by_idx([3,4,5,6])


show_images_by_idx([7,8,9,10])

