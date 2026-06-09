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


import geopandas as gpd
import folium
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon


# Crear GeoDataFrame con las zonas de anomalÃ­a simbÃ³lica
zonas = [
    {"name": "Z1", "lat": -4.3512, "lon": -55.9121, "desc": "cuenca hundida irregular"},
    {"name": "Z2", "lat": -7.2845, "lon": -54.1108, "desc": "patrÃ³n fractal con brillo cristalino"},
    {"name": "Z3", "lat": -6.0033, "lon": -53.8471, "desc": "formaciÃ³n circular no natural"},
    {"name": "Z4", "lat": -5.9900, "lon": -55.3000, "desc": "zona de sobreexposiciÃ³n geogrÃ¡fica"},
    {"name": "Z5", "lat": -6.7121, "lon": -52.4412, "desc": "geometrÃ­a triangular sagrada"}
]

df = pd.DataFrame(zonas)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
gdf.set_crs(epsg=4326, inplace=True)

gdf


# Crear mapa centrado en la zona media
mapa = folium.Map(location=[-6.0, -54.0], zoom_start=6)

# Agregar puntos
for _, row in gdf.iterrows():
    folium.Marker(
        location=[row["lat"], row["lon"]],
        popup=f"{row['name']}: {row['desc']}",
        icon=folium.Icon(color='red')
    ).add_to(mapa)

mapa


from IPython.display import Image, display

# Primera imagen con glitch visual
display(Image(filename="/kaggle/input/hide-code/page_3_processed.jpg"))


from IPython.display import Image, display

# Primera imagen con glitch visual
display(Image(filename="/kaggle/input/hide-code/page_5_processed.jpg"))


from IPython.display import Image, display

# Primera imagen con glitch visual
display(Image(filename="/kaggle/input/hide-code/page_6_processed.jpg"))


from IPython.display import Image, display

# Primera imagen con glitch visual
display(Image(filename="/kaggle/input/hide-code/page_8_processed.jpg"))


!pip install pymupdf  # solo si no estÃ¡ ya en el entorno

import fitz  # PyMuPDF

# Ruta al archivo PDF dentro de Kaggle
pdf_path = "/kaggle/input/hide-code/mss1483728.pdf"

# Abrimos y leemos
doc = fitz.open(pdf_path)

for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    text = page.get_text()
    print(f"--- PÃ¡gina {page_num + 1} ---")
    print(text)



!pip install pymupdf

import fitz  # PyMuPDF
from PIL import Image
from IPython.display import display



# Cambia la ruta al nombre correcto del archivo en tu dataset
pdf_path = "/kaggle/input/hide-code/mss1483728.pdf"

doc = fitz.open(pdf_path)

# Elegimos una pÃ¡gina (por ejemplo, la primera)
page = doc.load_page(0)

# Renderizamos como imagen
pix = page.get_pixmap()
img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

# Mostramos la imagen
display(img)



from IPython.display import Image, display

# Primera imagen con glitch visual
display(Image(filename="/kaggle/input/redlab/Captura de pantalla 2025-06-01 a la(s) 1.07.34p.m..png"))



from IPython.display import Image, display

# Primera imagen con glitch visual
display(Image(filename="/kaggle/input/redlab/Captura de pantalla 2025-06-01 a la(s) 1.20.34p.m..png"))


from IPython.display import Image, display

# Primera imagen con glitch visual
display(Image(filename="/kaggle/input/redlab/Captura de pantalla 2025-06-01 a la(s) 1.27.04p.m..png"))



from IPython.display import Image, display

# Primera imagen con glitch visual
display(Image(filename="/kaggle/input/redlab/Captura de pantalla 2025-06-01 a la(s) 3.29.50p.m..png"))



from IPython.display import Image, display

# Primera imagen con glitch visual
display(Image(filename="/kaggle/input/redlab/Captura de pantalla 2025-06-01 a la(s) 3.29.59p.m..png"))



from IPython.display import Image, display

# Primera imagen con glitch visual
display(Image(filename="/kaggle/input/redlab/Captura de pantalla 2025-06-02 a la(s) 12.02.29a.m..png"))


