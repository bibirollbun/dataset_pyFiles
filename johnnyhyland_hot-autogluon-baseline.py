!pip install autogluon.tabular -q


from autogluon.tabular import TabularDataset, TabularPredictor

train_data = TabularDataset('/kaggle/input/hill-of-towie-wind-turbine-power-prediction/training_dataset.parquet')


print(len(train_data))

train_data.dropna(inplace=True)

print(len(train_data))


not_in_sub = ['wtc_AcWindSp_mean;1', 'wtc_AcWindSp_min;1', 'wtc_AcWindSp_max;1', 'wtc_AcWindSp_stddev;1', 
              'wtc_ScYawPos_mean;1', 'wtc_ScYawPos_min;1', 'wtc_ScYawPos_max;1', 'wtc_ScYawPos_stddev;1', 
              'wtc_NacelPos_mean;1', 'wtc_NacelPos_min;1', 'wtc_NacelPos_max;1', 'wtc_GenRpm_mean;1', 
              'wtc_GenRpm_min;1', 'wtc_GenRpm_max;1', 'wtc_GenRpm_stddev;1', 'wtc_PitcPosA_mean;1', 
              'wtc_PitcPosA_min;1', 'wtc_PitcPosA_max;1', 'wtc_PitcPosA_stddev;1', 'wtc_PitcPosB_mean;1', 
              'wtc_PitcPosC_mean;1', 'wtc_PowerRef_endvalue;1', 'wtc_ScReToOp_timeon;1', 'wtc_ActPower_mean;1', 
              'wtc_ActPower_min;1', 'wtc_ActPower_max;1', 'wtc_ActPower_stddev;1', 'wtc_AmbieTmp_mean;1', 
              'ShutdownDuration;1']

train_data.drop(columns=not_in_sub, inplace=True)


label = 'target'

predictor = TabularPredictor(
    label=label, 
    eval_metric='mean_absolute_error',
).fit(
    train_data,
    presets='experimental',
    time_limit=3600*8,
    num_bag_folds=15,
    num_stack_levels=2,
)


test_data = TabularDataset('/kaggle/input/hill-of-towie-wind-turbine-power-prediction/submission_dataset.parquet')

y_pred = predictor.predict(test_data)
y_pred.head()


import pandas as pd

submission = pd.DataFrame({'id': test_data.index, 'prediction': y_pred})
submission.to_csv('submission.csv', index=False)
print("Predictions saved to submission.csv")




