import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_regression
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
import matplotlib.pyplot as plt


def feature_distributions(df, features, plot_name='plot.png'):
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
    axs = axs.flatten()

    axis_counter = 0
    for _, feature in enumerate(features):
        _ = sns.kdeplot(df[feature],
                        fill=True,
                        color="r",
                        ax=axs[axis_counter])
        _ = axs[axis_counter].set_title("{}".format(feature), fontsize=15)
        _ = axs[axis_counter].set_ylabel("")
        _ = axs[axis_counter].set_xlabel("")
        axis_counter += 1

    plt.tight_layout()
    plt.savefig(plot_name, dpi=150)
    plt.show()


train_original = pd.read_csv('/kaggle/input/bpm-prediction-challenge/Train.csv')
features = train_original.columns.tolist()
feature_distributions(train_original, features, 'plot_original.png')
features.remove('BeatsPerMinute')
X = train_original[features]
y = train_original['BeatsPerMinute']
MI_orig = mutual_info_regression(X, y)
MI_df = pd.DataFrame({'feature': features, 'MI_orig': MI_orig})
print(MI_df)


X, y = make_regression(n_samples=698888,
                       n_features=9,
                       n_informative=8,
                       effective_rank=4,
                       random_state=1001,
                       noise=3,
                       bias=120.0)

X, y = pd.DataFrame(X), pd.DataFrame(y)
train_synth = pd.concat([X, y], axis=1)
train_synth.columns = [
    'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
    'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
    'TrackDurationMs', 'Energy', 'BeatsPerMinute'
]
train_synth.insert(0, 'id', np.arange(train_synth.shape[0]))
train_synth.to_csv('regression.csv', index=False)

features = [w for w in train_synth.columns.tolist() if not w in ['id']]
feature_distributions(train_synth, features, 'plot_synthetic.png')
features.remove('BeatsPerMinute')
X = train_synth[features]
y = train_synth['BeatsPerMinute']
MI_synth = mutual_info_regression(X, y)
MI_df['MI_synth'] = MI_synth
print(MI_df)


train_noise = train_synth.copy()
features = [
    w for w in train_noise.columns.tolist()
    if not w in ['id', 'BeatsPerMinute']
]
train_noise[features] = train_noise[features].applymap(
    lambda x: x * (1 + np.random.uniform(-0.5, 0.5)))
train_noise['BeatsPerMinute'] = train_noise['BeatsPerMinute'].map(
    lambda x: x * (1 + np.random.uniform(-0.05, 0.05)))

cols_scale = ['MoodScore']
scaler = MinMaxScaler(feature_range=(-1.5, 2.5))
train_noise[cols_scale] = np.clip(
    scaler.fit_transform(train_noise[cols_scale].values), 0, 1)
cols_scale = ['RhythmScore']
scaler = MinMaxScaler(feature_range=(-0.3, 1.5))
train_noise[cols_scale] = np.clip(
    scaler.fit_transform(train_noise[cols_scale].values), 0, 1)
cols_scale = ['AcousticQuality']
scaler = MinMaxScaler(feature_range=(-0.5, 1))
train_noise[cols_scale] = np.clip(
    scaler.fit_transform(train_noise[cols_scale].values), 0, 1)
cols_scale = ['Energy']
scaler = MinMaxScaler(feature_range=(0, 1))
train_noise[cols_scale] = scaler.fit_transform(train_noise[cols_scale].values)
cols_scale = ['AudioLoudness']
scaler = MinMaxScaler(feature_range=(-27.5, 7.5))
train_noise[cols_scale] = np.clip(
    scaler.fit_transform(train_noise[cols_scale].values), -27.5, -1.5)
cols_scale = ['TrackDurationMs']
scaler = MinMaxScaler(feature_range=(160000, 175000))
train_noise[cols_scale] = scaler.fit_transform(train_noise[cols_scale].values)
cols_scale = ['LivePerformanceLikelihood']
scaler = MinMaxScaler(feature_range=(-0.5, 0.6))
train_noise[cols_scale] = np.clip(
    scaler.fit_transform(train_noise[cols_scale].values), 0, 0.6)
cols_scale = ['VocalContent']
scaler = MinMaxScaler(feature_range=(-0.25, 0.25))
train_noise[cols_scale] = np.clip(
    scaler.fit_transform(train_noise[cols_scale].values), 0, 0.25)
cols_scale = ['InstrumentalScore']
scaler = MinMaxScaler(feature_range=(-0.5, 0.9))
train_noise[cols_scale] = np.clip(
    scaler.fit_transform(train_noise[cols_scale].values), 0, 0.9)
cols_scale = ['BeatsPerMinute']
scaler = MinMaxScaler(feature_range=(40, 210))
train_noise[cols_scale] = scaler.fit_transform(train_noise[cols_scale].values)

train_noise.to_csv('regression_noise.csv', index=False)
features = [w for w in train_noise.columns.tolist() if not w in ['id']]
feature_distributions(train_noise, features, 'plot_noise.png')
features.remove('BeatsPerMinute')
X = train_noise[features]
y = train_noise['BeatsPerMinute']
MI_noise = mutual_info_regression(X, y)
MI_df['MI_noise'] = MI_noise
print(MI_df)


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
features = [w for w in train.columns.tolist() if not w in ['id']]
feature_distributions(train, features, 'plot_train.png')
features.remove('BeatsPerMinute')
X = train[features]
y = train['BeatsPerMinute']
MI_train = mutual_info_regression(X, y)
MI_df['MI_train'] = MI_train
print(MI_df)

