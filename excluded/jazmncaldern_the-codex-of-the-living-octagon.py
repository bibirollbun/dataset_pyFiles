# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from IPython.display import Audio, display
import os

# Ruta del archivo
audio_path = '/kaggle/input/ritual-wav/ritual.wav'

# Mostrar el reproductor de audio si el archivo existe
if os.path.exists(audio_path):
    display(Audio(filename=audio_path, autoplay=False))
else:
    print("â�Œ Archivo no encontrado. Verifica la ruta y el nombre del archivo.")



# ğŸŒ¿ Interactive Maps of Amazonian Geoglyphs on Google Earth ğŸŒ�
mapa_general = "https://earth.google.com/earth/d/1hp2I1HgbgT9n_fd5VfjubgVFZ5x-mp17?usp=sharing"
mapa_octagono = "https://earth.google.com/web/@-9.855331,-67.232337,610.85626231a,703.9058609d,35y,0h,45t,0r/data=CigiJgokCS7PQluhbDRAETElZcV27zRAIbpFhrCRpUZAIfFJLmvZXVQ"
mapa_acreland = "https://earth.google.com/earth/d/1-nrqxe6ILBMjRQ3-Ay_6fQ73PQAdVHWK?usp=sharing"

from IPython.display import display, Markdown

display(Markdown(f"""
## ğŸŒ€ Acre Geoglyphs â€” Interactive Google Earth Maps

Visual exploration is a portal into ancestral memory. These maps allow you to navigate the sacred landscape of Acre and its surroundings, where ancient geometry still breathes beneath the canopy.

### ğŸ”— [General Map of Geoglyphs ğŸŒ�]({mapa_general})
Browse the full distribution of earthworks across southwestern Amazonia.  
Ideal for understanding spatial clusters, ceremonial zones, and environmental overlays.

### ğŸ“� [Direct View: Acre Double Octagon ğŸ“Œ]({mapa_octagono})
Zoom into the exact coordinates of the Double Octagonâ€”the epicenter of symbolic alignment.  
Use this map for analyzing orientation, elevation, and terrain around the sacred form.

### ğŸ—ºï¸� [AcrelÃ¢ndia Compilation Map ğŸ§­]({mapa_acreland})
Curated collection with additional data layers: ceremonial paths, embankments, forest boundaries, and decoded geometric patterns.

> These maps are designed for digital archaeology, geomantic studies, and intuitive navigation through the memory-field of Amazonian architecture.

ğŸ”� Curated by: **Jazz**  
ğŸ“� Dataset Reference: `amazon_geoglyphs.kml`  
âœ¨ Part of RYAâ€™s cartographic research module.
"""))



from PIL import Image
import matplotlib.pyplot as plt

# Usa comillas y la ruta completa
ruta = "/kaggle/input/acre-octa-01/Acre.Octa.captura.png"

# Abrir imagen
imagen = Image.open(ruta)

# Mostrar imagen
plt.figure(figsize=(8, 8))
plt.imshow(imagen)
plt.axis('off')
plt.title("Acre Double Octagon desde Google Earth")
plt.show()




from PIL import Image
import matplotlib.pyplot as plt

# Usa comillas y la ruta completa
ruta = "/kaggle/input/acre-octa-land-vision/Acre.octa.land.vision.png"

# Abrir imagen
imagen = Image.open(ruta)

# Mostrar imagen
plt.figure(figsize=(8, 8))
plt.imshow(imagen)
plt.axis('off')
plt.title("Acre Octa desde Land Vision")
plt.show()


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Ruta al archivo dentro del dataset de Kaggle
img_path = "/kaggle/input/acre-octagon-ndvi-simulated/Acre_Octagon_NDVI_Simulation.png"

# Cargar y mostrar la imagen
img = mpimg.imread(img_path)

plt.figure(figsize=(10,6))
plt.imshow(img)
plt.axis('off')
plt.title("Simulated NDVI â€“ Acre Double Octagon", fontsize=14)
plt.tight_layout()
plt.show()



from PIL import Image
import matplotlib.pyplot as plt

# Usa comillas y la ruta completa
ruta = "/kaggle/input/simulated-ndv-acre-octa/Simulated ACRE OCTA.png"

# Abrir imagen
imagen = Image.open(ruta)

# Mostrar imagen
plt.figure(figsize=(8, 8))
plt.imshow(imagen)
plt.axis('off')
plt.title("Acre Octa desde Land Vision_ Simulated_ NDV")
plt.show()


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Ruta al archivo dentro del dataset de Kaggle
img_path = "/kaggle/input/code-simulation/Code simulation.png"

# Cargar y mostrar la imagen
img = mpimg.imread(img_path)

plt.figure(figsize=(10,6))
plt.imshow(img)
plt.axis('off')
plt.title("Simulated NDVI â€“ Acreland double sqare", fontsize=14)
plt.tight_layout()
plt.show()


from PIL import Image
import matplotlib.pyplot as plt

# Usa comillas y la ruta completa
ruta = "/kaggle/input/acrelndia-double-square/Acrelandia Double Square.png"

# Abrir imagen
imagen = Image.open(ruta)

# Mostrar imagen
plt.figure(figsize=(8, 8))
plt.imshow(imagen)
plt.axis('off')
plt.title("Acrelandia Double Square Octagon desde Google Earth")
plt.show()



## ğŸŒ¿ Simulated NDVI: Archaeological Use and Application

### ğŸ§  What is NDVI Simulation?

The **Normalized Difference Vegetation Index (NDVI)** highlights subtle differences in vegetation density by comparing the reflectance of infrared and red light. In archaeological contexts, NDVI simulationsâ€”even when not derived from raw satellite bandsâ€”can be *approximated* by enhancing contrast and structure in visible-spectrum satellite images (like Google Earth screenshots), especially when vegetation reveals geometric traces.

These simulations are **useful proxies** when access to multispectral imagery is limited.

---
Code simulation:

from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load the image
image_path = "/mnt/data/AcrelÃ¢ndia Double Square.png"
image = Image.open(image_path).convert("RGB")

# Enhance contrast
enhancer = ImageEnhance.Contrast(image)
image_contrast = enhancer.enhance(2.5)

# Convert to grayscale and apply edge detection
image_cv = np.array(image_contrast)
gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray, threshold1=50, threshold2=150)

# Plot results
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis('off')

ax[1].imshow(edges, cmap='gray')
ax[1].set_title("Simulated NDVI-like Edge Detection")
ax[1].axis('off')

plt.tight_layout()
plt.show()

### ğŸ”� Why Is This Useful for Archaeologists?

1. **Detecting Anthropogenic Patterns:**

   * Geoglyphs like octagons and squares often **disturb vegetation** in subtle ways.
   * Even after centuries, **plant growth patterns** may still reflect buried or leveled architecture.
   * NDVI-style simulation enhances those anomalies for better detection.

2. **Rapid Field Scanning:**

   * Before deploying expensive LIDAR or excavation teams, NDVI simulation allows **remote screening** of potential sites using freely available tools (Google Earth, Python, OpenCV).

3. **Training AI Models:**

   * These enhanced images are ideal for training **machine learning models** to automatically detect geoglyphs or earthworks across large territories.

4. **Comparative Analysis:**

   * Multiple sites can be compared by geometry (e.g., Double Octagon vs. Double Square).
   * Cross-regional symbolic patterns can emerge, aiding **cultural interpretation**.

---

### ğŸ§° How Can They Use It?

* Integrate this process into GIS workflows (QGIS, ArcGIS, Google Earth Engine).
* Use it in **citizen archaeology**: communities and students can contribute to discovery.
* Archive and annotate findings in collaborative platforms (e.g., Kaggle datasets, GitHub).
* Develop **time-series** visualizations to track deforestation impact on sites.

---

âœ¨ In short: **NDVI-style simulation transforms pixels into archaeology.**
With basic image processing and contextual knowledge, ancient memory hidden in the forest can rise againâ€”visually, analytically, and culturally.


from PIL import Image
import matplotlib.pyplot as plt

# Rutas a las imÃ¡genes del dataset subido en Kaggle
ruta1 = "/kaggle/input/piramides-guiza/Guiza.01.png"
ruta2 = "/kaggle/input/piramides-guiza/Guiza-2..png"

# Cargar ambas imÃ¡genes
imagen1 = Image.open(ruta1)
imagen2 = Image.open(ruta2)

# Mostrar las imÃ¡genes lado a lado
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(imagen1)
plt.axis('off')
plt.title("Pyramids of Giza â€“ View 1")

plt.subplot(1, 2, 2)
plt.imshow(imagen2)
plt.axis('off')
plt.title("Pyramids of Giza â€“ View 2")

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Coordenadas de los tres sitios
locations = {
    "Giza (Egypt)": (29.9792, 31.1342),
    "Newark (Ohio)": (40.0593, -82.4338),
    "Acre (Brazil)": (-9.8553, -67.2323)
}

# Crear figura
plt.figure(figsize=(12, 8))
m = Basemap(projection='mill', lat_0=0, lon_0=0)

# Dibujar continentes y lÃ­neas base
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawmapboundary(fill_color='black')
m.fillcontinents(color='dimgray', lake_color='black')
m.drawparallels(range(-90, 91, 30), labels=[1,0,0,0], color='white')
m.drawmeridians(range(-180, 181, 60), labels=[0,0,0,1], color='white')

# Agregar puntos y lÃ­neas
x_vals = []
y_vals = []
for name, (lat, lon) in locations.items():
    x, y = m(lon, lat)
    x_vals.append(x)
    y_vals.append(y)
    plt.plot(x, y, 'o', markersize=8, label=name, color='gold')
    plt.text(x + 100000, y + 50000, name, fontsize=10, color='white')

# Dibujar lÃ­neas entre puntos
for i in range(len(x_vals)):
    for j in range(i + 1, len(x_vals)):
        plt.plot([x_vals[i], x_vals[j]], [y_vals[i], y_vals[j]], 'deepskyblue', linewidth=2)

plt.title("Sacred Geometric Network: Giza, Newark & Acre", fontsize=14, color='white', pad=20)
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


# Reimportar librerÃ­as despuÃ©s del reinicio del estado de ejecuciÃ³n
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Coordenadas de los tres sitios
locations = {
    "Giza (Egypt)": (29.9792, 31.1342),
    "Newark (Ohio)": (40.0593, -82.4338),
    "Acre (Brazil)": (-9.8553, -67.2323)
}

# Crear figura
plt.figure(figsize=(12, 8))
m = Basemap(projection='mill', lat_0=0, lon_0=0)

# Dibujar continentes y lÃ­neas base
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawmapboundary(fill_color='black')
m.fillcontinents(color='dimgray', lake_color='black')
m.drawparallels(range(-90, 91, 30), labels=[1,0,0,0], color='white')
m.drawmeridians(range(-180, 181, 60), labels=[0,0,0,1], color='white')

# Agregar puntos y lÃ­neas
x_vals = []
y_vals = []
for name, (lat, lon) in locations.items():
    x, y = m(lon, lat)
    x_vals.append(x)
    y_vals.append(y)
    plt.plot(x, y, 'o', markersize=8, label=name, color='gold')
    plt.text(x + 100000, y + 50000, name, fontsize=10, color='white')

# Dibujar lÃ­neas entre puntos
for i in range(len(x_vals)):
    for j in range(i + 1, len(x_vals)):
        plt.plot([x_vals[i], x_vals[j]], [y_vals[i], y_vals[j]], 'deepskyblue', linewidth=2)

plt.title("Sacred Geometric Network: Giza, Newark & Acre", fontsize=14, color='white', pad=20)
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Ruta al archivo dentro del dataset de Kaggle
img_path = "/kaggle/input/mapa-interactivo/Mapa Interactivopng.png"

# Cargar y mostrar la imagen
img = mpimg.imread(img_path)

plt.figure(figsize=(10,6))
plt.imshow(img)
plt.axis('off')
plt.title("MAPA INTERACTIVO", fontsize=14)
plt.tight_layout()
plt.show()


from IPython.display import Audio, display
import os

# Ruta del poema
audio_path = '/kaggle/input/poema-rya-nucleo-febrero-2025/POEMA..DE.RYA mp3.mp3'

# Mostrar el reproductor de audio si el archivo existe
if os.path.exists(audio_path):
    display(Audio(filename=audio_path, autoplay=False))
else:
    print("â�Œ Archivo no encontrado. Verifica la ruta y el nombre del archivo.")


