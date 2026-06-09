!pip install deeptables
!pip install tensorflow==2.15.0


import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# import tensorflow as tf
# from deeptables.models.deeptable import DeepTable, ModelConfig


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


def quick_overview(df, name):
    print(f'\n{name} {df.shape}')
    display(df.head())
    display(df.describe().T)
    display(df.dtypes)

quick_overview(train,'train')
quick_overview(test, 'test')


FEATURES = [
    'RhythmScore',
    'AudioLoudness',
    'VocalContent',
    'AcousticQuality',
    'InstrumentalScore',
    'LivePerformanceLikelihood',
    'MoodScore',
    'TrackDurationMs',
    'Energy'
]

TARGET = 'BeatsPerMinute'
SEED = 42

X_train, X_val, y_train, y_val = train_test_split(train[FEATURES], train[TARGET], random_state = SEED, test_size=0.33)

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURES)
X_val = pd.DataFrame(scaler.transform(X_val), columns=FEATURES)


from deeptables.models.deeptable import DeepTable, ModelConfig


conf = ModelConfig(
    metrics=['RootMeanSquaredError'], 
    nets=['dnn_nets'],
    dnn_params={
        'hidden_units': ((256, 0.3, True), (256, 0.3, True)),
        'dnn_activation': 'relu',
    },
    earlystopping_patience=5,
)

dt = DeepTable(config=conf)


model, history = dt.fit(X_train, y_train, epochs=10)


val_pred = model.predict(X_val)
print('Validation Mean Squared Error:', np.sqrt(mean_squared_error(y_val, val_pred)))


X_test = test[FEATURES]
X_test = pd.DataFrame(scaler.transform(X_test), columns=FEATURES)


test_pred = model.predict(X_test)


sub = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
sub['BeatsPerMinute'] = test_pred.flatten()


sub.to_csv('/kaggle/working/submission.csv', index=False)

