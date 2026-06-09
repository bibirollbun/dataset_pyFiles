import cv2
import numpy as np
import matplotlib.pyplot as plt

file_path = '/kaggle/input/amazon-geo-images-v5/E539B100-3F40-46F5-859A-9BDBC46925EA.png'


img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(img, 50, 150)


plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')
plt.axis('off')
plt.show()



img_color = cv2.imread('/kaggle/input/amazon-geo-images-v5/E539B100-3F40-46F5-859A-9BDBC46925EA.png')


lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)


if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title('Detected Lines (Hough Transform)')
plt.axis('off')
plt.show()


from PIL import Image
import matplotlib.pyplot as plt


img = Image.open('/kaggle/input/amazon-geo-images-v5/E539B100-3F40-46F5-859A-9BDBC46925EA.png')
plt.imshow(img)
plt.axis('off')
plt.show()


from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('/kaggle/input/amazon-geo-images-v5/9D41C972-085B-4E6C-9749-22C007C7CFBD.png')
plt.imshow(img)
plt.axis('off')
plt.show()

