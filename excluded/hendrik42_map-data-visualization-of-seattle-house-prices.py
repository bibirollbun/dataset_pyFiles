import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import binned_statistic_2d


data=pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')


cities = data['city'].unique()
city_to_int = {city: i for i, city in enumerate(cities)}
city_code = data['city'].map(city_to_int)
cmap = plt.get_cmap('tab20', len(cities))

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    data['longitude'],
    data['latitude'],
    c=city_code,
    cmap=cmap,
    s=1,
    alpha=0.7,
)

handles = [
    plt.Line2D([], [], marker='o', color='w', markerfacecolor=cmap(i), markersize=6, label=city)
    for i, city in enumerate(cities)
]

plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', title='City')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Housing Locations Colored by City')
plt.grid(True)
plt.tight_layout()
plt.show()


stat, x_edges, y_edges, binnumber = binned_statistic_2d(
    data['longitude'], data['latitude'], data['sale_price'], 
    statistic='median', bins=100
)
vmin = np.nanpercentile(stat, 5)
vmax = np.nanpercentile(stat, 95)

plt.imshow(
    stat.T, 
    origin='lower', 
    extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    aspect='auto',
    cmap='viridis',
    vmin=vmin,
    vmax=vmax
)

plt.colorbar(label='Median Sale Price')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Median Sale Price by Location')
plt.grid(True)
plt.tight_layout()
plt.show()

