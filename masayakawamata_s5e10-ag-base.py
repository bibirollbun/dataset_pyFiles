!pip install autogluon.tabular -qq


filename = 'base'


import warnings
warnings.simplefilter('ignore')


import pandas as pd, numpy as np

train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
print('Train Shape:', train.shape)
print('Test Shape:', test.shape)

train.head(3)


TARGET = 'accident_risk'
BASE = [col for col in train.columns if col not in ['id', TARGET]]
print(f'{len(BASE)} Base Features:{BASE}')


FEATURES = BASE
print(len(FEATURES), 'Features.')


X = train[FEATURES]
y = train[TARGET]


from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold_number, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    train.loc[val_idx, 'fold'] = fold_number
train.head()


from autogluon.tabular import TabularPredictor

label = TARGET
time_limit = 3600 * 11  # 11 hours
metric = 'rmse'

# Train the model
predictor = TabularPredictor(label=label, eval_metric=metric, groups='fold').fit(
    train, 
    time_limit=time_limit, 
    presets='extreme',
    num_cpus=4,
    num_gpus=0,
    # excluded_model_types=[''],
    # included_model_types=[''],
    num_stack_levels=1,
    dynamic_stacking=False,
)


predictor.leaderboard(silent=True).style.background_gradient(subset=['score_val'], cmap='RdYlGn')


oof_pred = predictor.predict_oof()
oof_to_save = pd.DataFrame({
        'id': train['id'],
        TARGET: oof_pred
    })

oof_filename = f'oof_ag_{filename}.csv'
oof_to_save.to_csv(oof_filename, index=False)
test_pred = predictor.predict(data=test)
test_to_save = pd.DataFrame({
    'id': test['id'],
    TARGET: test_pred
})

test_filename = f'test_ag_{filename}.csv'
test_to_save.to_csv(test_filename, index=False)

