! cp /kaggle/input/wsdmandrotdiflogregpipeline_v1/other/default/1/* /kaggle/working
import pandas as pd
import sys
sys.path.append('/kaggle/working/')

from wsdmandrotdiflogregpipeline_v1 import wsdmandrotdiflogregpipeline_v1


from datetime import datetime

print(f"[{datetime.now()}] Loading train and test data...")
pipeline = wsdmandrotdiflogregpipeline_v1()
train = pd.read_parquet('/kaggle/input/wsdm-cup-multilingual-chatbot-arena/train.parquet')
test = pd.read_parquet('/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet')
print(f"[{datetime.now()}] Data loaded.")

print(f"[{datetime.now()}] Starting model fitting...")
pipeline.fit(train)
print(f"[{datetime.now()}] Model fitting completed.")

print(f"[{datetime.now()}] Starting predictions...")
submission = pipeline.predict(test)
print(f"[{datetime.now()}] Predictions completed.")

print(f"[{datetime.now()}] Saving submission file...")
submission.to_csv('submission.csv', index=False)
print(f"[{datetime.now()}] Submission file saved as 'submission.csv'.")





