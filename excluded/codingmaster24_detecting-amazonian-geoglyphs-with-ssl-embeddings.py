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


import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import numpy as np
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


!cp /kaggle/input/archaeoblog-amazon-geoglyphs/AmazoniaGeoglyphs.png .


img_path = "/kaggle/input/archaeoblog-amazon-geoglyphs/AmazoniaGeoglyphs.png"
img = Image.open(img_path)

plt.figure(figsize=(10, 6))
plt.imshow(img)
plt.title("Amazonia Geoglyphs - Full Map")
plt.axis("off")
plt.show()


geojson_path = "/kaggle/input/archaeoblog-amazon-geoglyphs/geoglyph_points.geojson"
gdf = gpd.read_file(geojson_path)

print("ğŸ—ºï¸� Loaded GeoJSON Points:", len(gdf))
gdf.head()


gdf.plot(figsize=(8, 6), color="red", markersize=10)
plt.title("Geoglyph Locations")
plt.show()


parquet_path = "/kaggle/input/major-tom-core-s2l1c-ssl4eo-amazonia-embeddings/Major-TOM-Core-S2L1C-SSL4EO-Amazonia-Filtered_Subset_0.parquet"
df = pd.read_parquet(parquet_path)

print("SSL4EO Embeddings Loaded:", df.shape)
df.head()


import pandas as pd

df = pd.read_parquet("/kaggle/input/major-tom-core-s2l1c-ssl4eo-amazonia-embeddings/Major-TOM-Core-S2L1C-SSL4EO-Amazonia-Filtered_Subset_0.parquet")
print("Columns in the embedding dataset:")
print(df.columns.tolist())


import os
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


geoglyphs = gpd.read_file("/kaggle/input/archaeoblog-amazon-geoglyphs/geoglyph_points.geojson")
print("Loaded geoglyphs:", len(geoglyphs))


df = pd.read_parquet("/kaggle/input/major-tom-core-s2l1c-ssl4eo-amazonia-embeddings/Major-TOM-Core-S2L1C-SSL4EO-Amazonia-Filtered_Subset_0.parquet")


df = df.rename(columns={"centre_lat": "lat", "centre_lon": "lon"})
gdf_embed = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")


geoglyph_buffer = geoglyphs.to_crs("EPSG:4326").buffer(0.05)  # Was 0.01
gdf_embed["is_geoglyph"] = gdf_embed.geometry.apply(
    lambda pt: any(geoglyph_buffer.contains(pt))
).astype(int)


import numpy as np


embedding_matrix = np.vstack(gdf_embed["embedding"].values)
embedding_df = pd.DataFrame(embedding_matrix, columns=[f"feat_{i}" for i in range(embedding_matrix.shape[1])])


X = embedding_df
y = gdf_embed["is_geoglyph"]


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Geoglyph", "Geoglyph"], yticklabels=["No Geoglyph", "Geoglyph"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


importances = model.feature_importances_
plt.figure(figsize=(12, 4))
plt.bar(range(len(importances)), importances)
plt.title("Feature Importances from XGBoost")
plt.xlabel("Embedding Feature Index")
plt.ylabel("Importance")
plt.show()


base = geoglyphs.plot(color='red', figsize=(8, 6), alpha=0.5)
gdf_embed.sample(500).plot(ax=base, color='blue', markersize=3)
plt.title("Red = Known Geoglyphs, Blue = Embedding Patches")
plt.show()

