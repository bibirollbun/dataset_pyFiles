import numpy as np
import pandas as pd


predictions = pd.DataFrame()

columns = ["object_id", "class_6", "class_15", "class_16", "class_42", "class_52", "class_53", "class_62", "class_64", "class_65", "class_67", "class_88", "class_90", "class_92", "class_95", "class_99"]

predictions = pd.DataFrame()
for column in columns:
    predictions[column] = 0
    
for i in range(1,12):
    batch = np.load(f"/kaggle/input/predictions5/predictions_batch{i}.npy") #xgbpredictions or predictions

    predictions_batch_DataFrame = pd.DataFrame()
    for column in columns:
        predictions_batch_DataFrame[column] = 0
        
    predictions_batch = []
    
    for item in batch:
        star_class = 'class_' + str(int(item[1]))
        predictions_batch.append({'object_id': int(item[0]), star_class: 1})

    predictions_batch = pd.DataFrame(predictions_batch)
    predictions_batch_DataFrame = pd.concat((predictions_batch_DataFrame, predictions_batch), ignore_index = True)
    
    predictions_batch_DataFrame.fillna(0, inplace=True)
    predictions = pd.concat((predictions, predictions_batch_DataFrame), ignore_index = True)

    #print(predictions.loc[10])
    print(f"done with batch {i}")
    #print(predictions_batch_DataFrame)

for item in predictions.columns:
    predictions[item] = pd.to_numeric(predictions[item], downcast='integer', errors='coerce')
    
print(predictions.loc[:10])
predictions.to_csv("submission.csv", index=False)


print(predictions)


train = pd.read_csv('/kaggle/working/submission.csv', index_col=None)
print(train.loc[:10])




