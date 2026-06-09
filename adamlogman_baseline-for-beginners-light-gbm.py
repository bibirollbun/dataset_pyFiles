import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold
import joblib
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score


train_=pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
train_d=pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
test_=pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
test_d=pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv')



train=train_.merge(train_d,on='subject',how='left')
test=test_.merge(test_d,on='subject',how='left')


train.gesture=train.gesture.replace(['Cheek - pinch skin', 'Forehead - pull hairline',
       'Write name on leg', 'Feel around in tray and pull out an object',
       'Neck - scratch', 'Neck - pinch skin', 'Eyelash - pull hair',
       'Eyebrow - pull hair', 'Forehead - scratch',
       'Above ear - pull hair', 'Wave hello', 'Write name in air',
       'Text on phone', 'Pull air toward your face',
       'Pinch knee/leg skin', 'Scratch knee/leg skin',
       'Drink from bottle/cup', 'Glasses on/off'],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])


X=train.drop(['gesture', 'phase', 'behavior', 'orientation', 'sequence_type', 'sequence_id','row_id','sequence_id','subject'],axis=1)
y_0=train['gesture'].copy()
X_train_0,X_test_0,y_train_0,y_test_0=train_test_split(X,y_0,test_size=0.2,random_state=21)



model_0=XGBClassifier(device= 'gpu',)
model_0.fit(X_train_0,y_train_0)
y_pred_0=model_0.predict(X_test_0)


f1_score(y_pred_0,y_test_0,average='micro')


model_0.fit(X,y_0)


def predict(sequence: pd.DataFrame, demographics: pd.DataFrame) -> str:
    # Convert from Polars to Pandas if needed
    if not isinstance(sequence, pd.DataFrame):
        sequence = sequence.to_pandas()
    if not isinstance(demographics, pd.DataFrame):
        demographics = demographics.to_pandas()
    
    # Merge with demographics
    sequence = sequence.merge(demographics, on="subject", how="left")
    
    # Store sequence_id if it exists
    sequence_id = None
    if 'sequence_id' in sequence.columns:
        sequence_id = sequence['sequence_id'].iloc[0]
    
    # Drop unnecessary columns but keep sequence_id for now
    drop_cols = ['row_id', 'subject','sequence_id']
    sequence = sequence.drop(columns=[col for col in drop_cols if col in sequence.columns])
    
    # Drop sequence_id before prediction if it exists
    if 'sequence_id' in sequence.columns:
        sequence = sequence.drop('sequence_id', axis=1)
    
    # Make prediction using our trained model
    prediction = model_0.predict(sequence)
    
    # Get the most common prediction (mode)
    mode_pred = int(pd.Series(prediction).mode()[0])
    
    # Map back to gesture labels
    gesture_mapping = {
        0: 'Cheek - pinch skin',
        1: 'Forehead - pull hairline',
        2: 'Write name on leg',
        3: 'Feel around in tray and pull out an object',
        4: 'Neck - scratch',
        5: 'Neck - pinch skin',
        6: 'Eyelash - pull hair',
        7: 'Eyebrow - pull hair',
        8: 'Forehead - scratch',
        9: 'Above ear - pull hair',
        10: 'Wave hello',
        11: 'Write name in air',
        12: 'Text on phone',
        13: 'Pull air toward your face',
        14: 'Pinch knee/leg skin',
        15: 'Scratch knee/leg skin',
        16: 'Drink from bottle/cup',
        17: 'Glasses on/off'
    }
    
    return gesture_mapping.get(mode_pred, 'Text on phone')


inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )

