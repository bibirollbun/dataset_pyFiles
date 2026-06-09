from PIL import Image
import matplotlib.pyplot as plt

# Load the image
img_path = '/kaggle/input/sugarloaf-mountain/Sugarloaf Mountain.png'
image = Image.open(img_path)

# Show it with title
plt.figure(figsize=(12, 12))
plt.imshow(image)
plt.axis('off')
plt.title("Sugarloaf Mountain – Sentinel-2 or LiDAR Visualization")


# Annotate the image with a label and arrow
# Adjust (x, y) coordinates based on image layout
plt.annotate('Peak of Sugarloaf Mountain',
             xy=(300, 300),        # Arrow tip
             xytext=(500, 200),    # Text position
             arrowprops=dict(facecolor='red', arrowstyle='->', lw=2),
             fontsize=14,
             color='red',
             bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))

plt.show()



