import warnings
warnings.filterwarnings('ignore')
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler
import itertools

def scale_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

train_loader = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
train = train_loader.copy()
train["lighting"] = (train["lighting"] == "night").astype(int)
train["weather"] = (train["weather"] == "clear").astype(int)
train["speed"] = (train["speed_limit"] >= 60).astype(int)
train["accidents"] = (train["num_reported_accidents"] > 2).astype(int)
train = train.drop(['id', 'accident_risk', 'road_type', 'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season'], axis=1)
features = train.columns.tolist()
train['accident_risk'] = train_loader['accident_risk']

train_len = len(train)
train['GP_01'] = np.cos(np.cos(np.cos(train['weather']))*np.cos(np.sin(train['speed'])))*np.cos(np.cos(train['lighting']))*np.sin(np.exp(train['curvature'] - np.cos(train['speed'])))
train_loader['GP_01'] = np.cos(np.cos(np.cos(train['weather']))*np.cos(np.sin(train['speed'])))*np.cos(np.cos(train['lighting']))*np.sin(np.exp(train['curvature'] - np.cos(train['speed'])))
train['GP_02'] = np.sin(np.sin(np.sin(np.sin(np.exp(train['curvature'] + train['lighting'] - np.exp(np.cos(train['speed'])) + np.sqrt(np.abs(-0.233446*train['weather'] - 0.233446*train['speed'] + np.sqrt(np.abs(-0.173967)))))))))
train_loader['GP_02'] = np.sin(np.sin(np.sin(np.sin(np.exp(train['curvature'] + train['lighting'] - np.exp(np.cos(train['speed'])) + np.sqrt(np.abs(-0.233446*train['weather'] - 0.233446*train['speed'] + np.sqrt(np.abs(-0.173967)))))))))
train['GP_03'] = np.sin(np.sin(np.sin(np.exp(train['curvature'] - np.exp(np.cos(train['speed'])) + np.sin(np.sin(train['lighting'])) + np.sqrt(np.abs(-0.309718*train['weather'] - 0.309718*train['speed'] + np.sqrt(np.abs(-0.365759))))))) + 0.005682)
train_loader['GP_03'] = np.sin(np.sin(np.sin(np.exp(train['curvature'] - np.exp(np.cos(train['speed'])) + np.sin(np.sin(train['lighting'])) + np.sqrt(np.abs(-0.309718*train['weather'] - 0.309718*train['speed'] + np.sqrt(np.abs(-0.365759))))))) + 0.005682)
train['GP_04'] = np.cos(np.cos(train['accidents'])*np.cos(np.exp(train['weather']))*np.cos(np.sin(train['speed'] - 0.10149)))*np.cos(np.cos(train['lighting']))*np.sin(np.exp(train['curvature'] - np.cos(train['speed'])))
train_loader['GP_04'] = np.cos(np.cos(train['accidents'])*np.cos(np.exp(train['weather']))*np.cos(np.sin(train['speed'] - 0.10149)))*np.cos(np.cos(train['lighting']))*np.sin(np.exp(train['curvature'] - np.cos(train['speed'])))
train['GP_05'] = np.where(np.abs(np.cos(np.cos(train['speed'])) + np.cos(np.cos(np.sin(np.sin(np.sin(np.sin(train['lighting'])))) + np.sin(np.sin(train['curvature']))/np.cos(np.cos(train['weather']) - 0.071271/np.cos(train['speed']))))) < 1.0e-6, 0, np.log(np.abs(np.cos(np.cos(train['speed'])) + np.cos(np.cos(np.sin(np.sin(np.sin(np.sin(train['lighting'])))) + np.sin(np.sin(train['curvature']))/np.cos(np.cos(train['weather']) - 0.071271/np.cos(train['speed'])))))))
train_loader['GP_05'] = np.where(np.abs(np.cos(np.cos(train['speed'])) + np.cos(np.cos(np.sin(np.sin(np.sin(np.sin(train['lighting'])))) + np.sin(np.sin(train['curvature']))/np.cos(np.cos(train['weather']) - 0.071271/np.cos(train['speed']))))) < 1.0e-6, 0, np.log(np.abs(np.cos(np.cos(train['speed'])) + np.cos(np.cos(np.sin(np.sin(np.sin(np.sin(train['lighting'])))) + np.sin(np.sin(train['curvature']))/np.cos(np.cos(train['weather']) - 0.071271/np.cos(train['speed'])))))))
train['GP_06'] = np.sqrt(np.abs(-0.250863*(train['curvature'] + np.exp(train['accidents'] - np.cos(train['lighting']))*np.sin(train['curvature'] + np.sin(np.exp(np.cos(train['speed']))))*np.sin(np.exp(train['weather'])) + np.sqrt(np.abs(train['lighting'])))*np.sin(np.sin(np.exp(train['weather'])) + np.sin(np.sin(np.sin(train['curvature']))))*np.sin(np.exp(np.cos(train['speed'])))))
train_loader['GP_06'] = np.sqrt(np.abs(-0.250863*(train['curvature'] + np.exp(train['accidents'] - np.cos(train['lighting']))*np.sin(train['curvature'] + np.sin(np.exp(np.cos(train['speed']))))*np.sin(np.exp(train['weather'])) + np.sqrt(np.abs(train['lighting'])))*np.sin(np.sin(np.exp(train['weather'])) + np.sin(np.sin(np.sin(train['curvature']))))*np.sin(np.exp(np.cos(train['speed'])))))
train[features], scaler = scale_data(train[features].values)
train['GP_07'] = (-0.86127*np.sqrt(np.abs(train['lighting']))*np.sqrt(np.abs((np.sin(train['curvature']) + 1.714692)*np.sqrt(np.abs(np.sin(train['speed']) + np.sqrt(np.abs(train['speed'] + train['accidents'])))))) - 9.0e-6)*np.where(np.abs(-0.69225) < 1.0e-6, 0, np.log(np.abs(-0.69225)))
train_loader['GP_07'] = (-0.86127*np.sqrt(np.abs(train['lighting']))*np.sqrt(np.abs((np.sin(train['curvature']) + 1.714692)*np.sqrt(np.abs(np.sin(train['speed']) + np.sqrt(np.abs(train['speed'] + train['accidents'])))))) - 9.0e-6)*np.where(np.abs(-0.69225) < 1.0e-6, 0, np.log(np.abs(-0.69225)))
train['GP_08'] = np.cos(np.sin(np.sin(train['lighting'] + train['speed']))*np.sqrt(np.abs(np.sqrt(np.abs(train['weather'])))))
train_loader['GP_08'] = np.cos(np.sin(np.sin(train['lighting'] + train['speed']))*np.sqrt(np.abs(np.sqrt(np.abs(train['weather'])))))
train['GP_09'] = np.exp(np.cos(2.335401)/np.sqrt(np.abs(0.390823923270321*train['lighting']*(np.sin(np.sin(train['speed'])) + np.sqrt(np.abs(2.143246)))/(train['weather']*np.exp(-0.513233*train['curvature'])))))
train_loader['GP_09'] = np.exp(np.cos(2.335401)/np.sqrt(np.abs(0.390823923270321*train['lighting']*(np.sin(np.sin(train['speed'])) + np.sqrt(np.abs(2.143246)))/(train['weather']*np.exp(-0.513233*train['curvature'])))))
train['GP_10'] = np.sin(np.sin(np.sqrt(np.abs(np.sin(train['lighting'])*np.sin(np.sin(train['speed']*np.sin(np.sin(np.sin(np.sin(train['lighting']*np.sin(train['speed']*np.sin(np.sin(np.sin(train['speed']*np.sin(np.exp(train['curvature'])*np.sin(np.sin(np.sin(np.sin(train['accidents'])))))))))))))/train['weather']))))))
train_loader['GP_10'] = np.sin(np.sin(np.sqrt(np.abs(np.sin(train['lighting'])*np.sin(np.sin(train['speed']*np.sin(np.sin(np.sin(np.sin(train['lighting']*np.sin(train['speed']*np.sin(np.sin(np.sin(train['speed']*np.sin(np.exp(train['curvature'])*np.sin(np.sin(np.sin(np.sin(train['accidents'])))))))))))))/train['weather']))))))
train['GP_11'] = np.sqrt(np.abs(np.sin(np.sin(np.sin(np.sin(np.sin(np.sin(0.254397*train['speed']*np.sqrt(np.abs(train['lighting']*train['speed']*np.exp(train['curvature'])*np.sqrt(np.abs(train['lighting']**3*np.sin(train['speed']**2*train['accidents']*np.cos(train['weather']))/train['weather']))))))))))))
train_loader['GP_11'] = np.sqrt(np.abs(np.sin(np.sin(np.sin(np.sin(np.sin(np.sin(0.254397*train['speed']*np.sqrt(np.abs(train['lighting']*train['speed']*np.exp(train['curvature'])*np.sqrt(np.abs(train['lighting']**3*np.sin(train['speed']**2*train['accidents']*np.cos(train['weather']))/train['weather']))))))))))))

df = train.copy()
# Calculate the correlation matrix
correlation_matrix = df.corr()
# Create a heatmap of the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('GP Correlation Matrix Heatmap')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('gp_correlation_1.png', dpi=150)
plt.show()

train_loader.to_csv('train_GP_1.csv', index=False)
print("\n Train dataset shape:", train_loader.shape)

test_loader = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
test = test_loader.copy()
test["lighting"] = (test["lighting"] == "night").astype(int)
test["weather"] = (test["weather"] == "clear").astype(int)
test["speed"] = (test["speed_limit"] >= 60).astype(int)
test["accidents"] = (test["num_reported_accidents"] > 2).astype(int)
test = test.drop(['id', 'road_type', 'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season'], axis=1)

test_len = len(test)
test_loader['GP_01'] = np.cos(np.cos(np.cos(test['weather']))*np.cos(np.sin(test['speed'])))*np.cos(np.cos(test['lighting']))*np.sin(np.exp(test['curvature'] - np.cos(test['speed'])))
test_loader['GP_02'] = np.sin(np.sin(np.sin(np.sin(np.exp(test['curvature'] + test['lighting'] - np.exp(np.cos(test['speed'])) + np.sqrt(np.abs(-0.233446*test['weather'] - 0.233446*test['speed'] + np.sqrt(np.abs(-0.173967)))))))))
test_loader['GP_03'] = np.sin(np.sin(np.sin(np.exp(test['curvature'] - np.exp(np.cos(test['speed'])) + np.sin(np.sin(test['lighting'])) + np.sqrt(np.abs(-0.309718*test['weather'] - 0.309718*test['speed'] + np.sqrt(np.abs(-0.365759))))))) + 0.005682)
test_loader['GP_04'] = np.cos(np.cos(test['accidents'])*np.cos(np.exp(test['weather']))*np.cos(np.sin(test['speed'] - 0.10149)))*np.cos(np.cos(test['lighting']))*np.sin(np.exp(test['curvature'] - np.cos(test['speed'])))
test_loader['GP_05'] = np.where(np.abs(np.cos(np.cos(test['speed'])) + np.cos(np.cos(np.sin(np.sin(np.sin(np.sin(test['lighting'])))) + np.sin(np.sin(test['curvature']))/np.cos(np.cos(test['weather']) - 0.071271/np.cos(test['speed']))))) < 1.0e-6, 0, np.log(np.abs(np.cos(np.cos(test['speed'])) + np.cos(np.cos(np.sin(np.sin(np.sin(np.sin(test['lighting'])))) + np.sin(np.sin(test['curvature']))/np.cos(np.cos(test['weather']) - 0.071271/np.cos(test['speed'])))))))
test_loader['GP_06'] = np.sqrt(np.abs(-0.250863*(test['curvature'] + np.exp(test['accidents'] - np.cos(test['lighting']))*np.sin(test['curvature'] + np.sin(np.exp(np.cos(test['speed']))))*np.sin(np.exp(test['weather'])) + np.sqrt(np.abs(test['lighting'])))*np.sin(np.sin(np.exp(test['weather'])) + np.sin(np.sin(np.sin(test['curvature']))))*np.sin(np.exp(np.cos(test['speed'])))))
test[features], _ = scale_data(test[features].values, scaler)
test_loader['GP_07'] = (-0.86127*np.sqrt(np.abs(test['lighting']))*np.sqrt(np.abs((np.sin(test['curvature']) + 1.714692)*np.sqrt(np.abs(np.sin(test['speed']) + np.sqrt(np.abs(test['speed'] + test['accidents'])))))) - 9.0e-6)*np.where(np.abs(-0.69225) < 1.0e-6, 0, np.log(np.abs(-0.69225)))
test_loader['GP_08'] = np.cos(np.sin(np.sin(test['lighting'] + test['speed']))*np.sqrt(np.abs(np.sqrt(np.abs(test['weather'])))))
test_loader['GP_09'] = np.exp(np.cos(2.335401)/np.sqrt(np.abs(0.390823923270321*test['lighting']*(np.sin(np.sin(test['speed'])) + np.sqrt(np.abs(2.143246)))/(test['weather']*np.exp(-0.513233*test['curvature'])))))
test_loader['GP_10'] = np.sin(np.sin(np.sqrt(np.abs(np.sin(test['lighting'])*np.sin(np.sin(test['speed']*np.sin(np.sin(np.sin(np.sin(test['lighting']*np.sin(test['speed']*np.sin(np.sin(np.sin(test['speed']*np.sin(np.exp(test['curvature'])*np.sin(np.sin(np.sin(np.sin(test['accidents'])))))))))))))/test['weather']))))))
test_loader['GP_11'] = np.sqrt(np.abs(np.sin(np.sin(np.sin(np.sin(np.sin(np.sin(0.254397*test['speed']*np.sqrt(np.abs(test['lighting']*test['speed']*np.exp(test['curvature'])*np.sqrt(np.abs(test['lighting']**3*np.sin(test['speed']**2*test['accidents']*np.cos(test['weather']))/test['weather']))))))))))))

test_loader.to_csv('test_GP_1.csv', index=False)
print(" Test dataset shape:", test_loader.shape)

original_loader = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')
tmp = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_10k.csv')
original_loader = pd.concat((original_loader, tmp), axis=0)
tmp = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_2k.csv')
original_loader = pd.concat((original_loader, tmp), axis=0)
original = original_loader.copy()
original["lighting"] = (original["lighting"] == "night").astype(int)
original["weather"] = (original["weather"] == "clear").astype(int)
original["speed"] = (original["speed_limit"] >= 60).astype(int)
original["accidents"] = (original["num_reported_accidents"] > 2).astype(int)
original = original.drop(['accident_risk', 'road_type', 'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season'], axis=1)
original['accident_risk'] = original_loader['accident_risk']

original_len = len(original)
original_loader['GP_01'] = np.cos(np.cos(np.cos(original['weather']))*np.cos(np.sin(original['speed'])))*np.cos(np.cos(original['lighting']))*np.sin(np.exp(original['curvature'] - np.cos(original['speed'])))
original_loader['GP_02'] = np.sin(np.sin(np.sin(np.sin(np.exp(original['curvature'] + original['lighting'] - np.exp(np.cos(original['speed'])) + np.sqrt(np.abs(-0.233446*original['weather'] - 0.233446*original['speed'] + np.sqrt(np.abs(-0.173967)))))))))
original_loader['GP_03'] = np.sin(np.sin(np.sin(np.exp(original['curvature'] - np.exp(np.cos(original['speed'])) + np.sin(np.sin(original['lighting'])) + np.sqrt(np.abs(-0.309718*original['weather'] - 0.309718*original['speed'] + np.sqrt(np.abs(-0.365759))))))) + 0.005682)
original_loader['GP_04'] = np.cos(np.cos(original['accidents'])*np.cos(np.exp(original['weather']))*np.cos(np.sin(original['speed'] - 0.10149)))*np.cos(np.cos(original['lighting']))*np.sin(np.exp(original['curvature'] - np.cos(original['speed'])))
original_loader['GP_05'] = np.where(np.abs(np.cos(np.cos(original['speed'])) + np.cos(np.cos(np.sin(np.sin(np.sin(np.sin(original['lighting'])))) + np.sin(np.sin(original['curvature']))/np.cos(np.cos(original['weather']) - 0.071271/np.cos(original['speed']))))) < 1.0e-6, 0, np.log(np.abs(np.cos(np.cos(original['speed'])) + np.cos(np.cos(np.sin(np.sin(np.sin(np.sin(original['lighting'])))) + np.sin(np.sin(original['curvature']))/np.cos(np.cos(original['weather']) - 0.071271/np.cos(original['speed'])))))))
original_loader['GP_06'] = np.sqrt(np.abs(-0.250863*(original['curvature'] + np.exp(original['accidents'] - np.cos(original['lighting']))*np.sin(original['curvature'] + np.sin(np.exp(np.cos(original['speed']))))*np.sin(np.exp(original['weather'])) + np.sqrt(np.abs(original['lighting'])))*np.sin(np.sin(np.exp(original['weather'])) + np.sin(np.sin(np.sin(original['curvature']))))*np.sin(np.exp(np.cos(original['speed'])))))
original[features], _ = scale_data(original[features].values, scaler)
original_loader['GP_07'] = (-0.86127*np.sqrt(np.abs(original['lighting']))*np.sqrt(np.abs((np.sin(original['curvature']) + 1.714692)*np.sqrt(np.abs(np.sin(original['speed']) + np.sqrt(np.abs(original['speed'] + original['accidents'])))))) - 9.0e-6)*np.where(np.abs(-0.69225) < 1.0e-6, 0, np.log(np.abs(-0.69225)))
original_loader['GP_08'] = np.cos(np.sin(np.sin(original['lighting'] + original['speed']))*np.sqrt(np.abs(np.sqrt(np.abs(original['weather'])))))
original_loader['GP_09'] = np.exp(np.cos(2.335401)/np.sqrt(np.abs(0.390823923270321*original['lighting']*(np.sin(np.sin(original['speed'])) + np.sqrt(np.abs(2.143246)))/(original['weather']*np.exp(-0.513233*original['curvature'])))))
original_loader['GP_10'] = np.sin(np.sin(np.sqrt(np.abs(np.sin(original['lighting'])*np.sin(np.sin(original['speed']*np.sin(np.sin(np.sin(np.sin(original['lighting']*np.sin(original['speed']*np.sin(np.sin(np.sin(original['speed']*np.sin(np.exp(original['curvature'])*np.sin(np.sin(np.sin(np.sin(original['accidents'])))))))))))))/original['weather']))))))
original_loader['GP_11'] = np.sqrt(np.abs(np.sin(np.sin(np.sin(np.sin(np.sin(np.sin(0.254397*original['speed']*np.sqrt(np.abs(original['lighting']*original['speed']*np.exp(original['curvature'])*np.sqrt(np.abs(original['lighting']**3*np.sin(original['speed']**2*original['accidents']*np.cos(original['weather']))/original['weather']))))))))))))

original_loader.insert(0, 'id', np.arange(original_loader.shape[0]) + 1000000)
original_loader.to_csv('original_GP_1.csv', index=False)
print(" Original dataset shape:", original_loader.shape)

for col in train_loader.columns[train_loader.columns.str.startswith('GP_')].tolist():
    train_loader[col] = train_loader[col].map(lambda x: x * (1 + np.random.uniform(-0.05, 0.05)))
    test_loader[col] = test_loader[col].map(lambda x: x * (1 + np.random.uniform(-0.05, 0.05)))
    original_loader[col] = original_loader[col].map(lambda x: x * (1 + np.random.uniform(-0.05, 0.05)))

train_loader.to_csv('train_GP_1_noise.csv', index=False)
test_loader.to_csv('test_GP_1_noise.csv', index=False)
original_loader.to_csv('original_GP_1_noise.csv', index=False)

