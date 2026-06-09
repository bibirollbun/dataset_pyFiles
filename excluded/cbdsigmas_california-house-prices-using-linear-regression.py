import pandas as pd
import numpy as np

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s3e1/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s3e1/test.csv')

# Quick look
train.head()


from sklearn.cluster import MiniBatchKMeans

coords = train[['Latitude','Longitude']]
mbk = MiniBatchKMeans(n_clusters=50, batch_size=1000, random_state=42)
train['loc_cluster'] = mbk.fit_predict(coords)
test['loc_cluster']  = mbk.predict(test[['Latitude','Longitude']])

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(train['Longitude'], train['Latitude'], c=train['loc_cluster'], cmap='tab20', s=2)
plt.title("Geolocation Clusters")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.colorbar(label='Cluster')
plt.show()


def add_rot_coords(df, angles=(15,30,45,60,75)):
    for ang in angles:
        r = np.deg2rad(ang)
        df[f'rot{ang}_x'] = np.cos(r)*df['Longitude'] - np.sin(r)*df['Latitude']
        df[f'rot{ang}_y'] = np.sin(r)*df['Longitude'] + np.cos(r)*df['Latitude']
    return df

train = add_rot_coords(train)
test  = add_rot_coords(test)


from sklearn.preprocessing import SplineTransformer

spl_lat = SplineTransformer(n_knots=5, degree=3, include_bias=False)
spl_lon = SplineTransformer(n_knots=5, degree=3, include_bias=False)

lat_spl = spl_lat.fit_transform(train[['Latitude']])
lon_spl = spl_lon.fit_transform(train[['Longitude']])

for i in range(lat_spl.shape[1]):
    train[f'lat_spline_{i}'] = lat_spl[:, i]
    test[f'lat_spline_{i}']  = spl_lat.transform(test[['Latitude']])[:, i]
    train[f'lon_spline_{i}'] = lon_spl[:, i]
    test[f'lon_spline_{i}']  = spl_lon.transform(test[['Longitude']])[:, i]


for df in (train, test):
    df['inc_rooms']      = df['MedInc'] * df['AveRooms']
    df['age_rooms']      = df['HouseAge'] * df['AveRooms']
    df['pop_rooms']      = df['Population'] * df['AveRooms']
    df['dist_to_coast']  = np.abs(df['Longitude'] + 122.5)
    df['is_coastal']     = (df['dist_to_coast'] < 0.1).astype(int)
    CENTER = (37.0, -119.5)
    df['dist_center']    = np.hypot(df['Latitude']-CENTER[0],
                                     df['Longitude']-CENTER[1])
    df['lat_long']       = df['Latitude'] * df['Longitude']
    df['pop_density']    = df['Population'] / (df['AveRooms'] + 1)
    df['bedroom_ratio']  = df['AveBedrms'] / (df['AveRooms'] + 1e-6)
    df['rooms_per_occ']  = df['AveRooms'] / (df['AveOccup'] + 1)


import seaborn as sns

extra_cols = ['dist_to_coast', 'dist_center', 'pop_density',
              'bedroom_ratio', 'rooms_per_occ', 'inc_rooms',
              'age_rooms', 'pop_rooms']

train[extra_cols].hist(bins=30, figsize=(16, 10), edgecolor='black')
plt.suptitle("Distributions of Engineered Features", fontsize=16)
plt.show()


base_feats   = ['MedInc','HouseAge','AveRooms','AveBedrms',
                'Population','AveOccup','Latitude','Longitude']
rot_feats    = [c for c in train if c.startswith('rot')]
extras       = ['dist_to_coast','is_coastal','dist_center',
                'lat_long','pop_density','bedroom_ratio',
                'rooms_per_occ','inc_rooms','age_rooms','pop_rooms']
spline_feats = [c for c in train if 'spline_' in c]
loc_feats    = [c for c in train if c.startswith('loc_')]

features = base_feats + rot_feats + extras + spline_feats + loc_feats
X = train[features]; y = train['MedHouseVal']
X_test = test[features]


from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False,
                                    interaction_only=True))
    ]), base_feats + rot_feats + extras)
], remainder='passthrough')

model = ElasticNetCV(
    l1_ratio=[.1,.5,.9],
    n_alphas=20,
    cv=3,
    precompute='auto',
    max_iter=5000,
    n_jobs=-1,
    random_state=42
)

pipeline = Pipeline([('prep', preprocessor), ('reg', model)])


pipeline.fit(X, y)
preds = pipeline.predict(X_test)

submission = pd.DataFrame({'id': test['id'], 'MedHouseVal': preds})
submission.to_csv('submission_fast.csv', index=False)
print("Wrote submission_fast.csv")

