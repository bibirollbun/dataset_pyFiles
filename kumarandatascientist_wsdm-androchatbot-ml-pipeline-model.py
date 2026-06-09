# Move all necessary files
!cp -r /kaggle/input/andro_wsdm_chatbot_pipeline/other/default/1/* /kaggle/working/

import sys
sys.path.append('/kaggle/working/')

from  wsdmandrochatbotpipeline_v1 import wsdmandrochatbotpipeline_v1


import pandas as pd

# Load train and test data
train = pd.read_parquet('/kaggle/input/wsdm-cup-multilingual-chatbot-arena/train.parquet')
test = pd.read_parquet('/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet')

# Initialize the pipeline
pipeline = wsdmandrochatbotpipeline_v1()

# Fit the pipeline with the training data
pipeline.fit(train)

# Predict using the test data
submission = pipeline.predict(test)

# Save the predictions to a CSV file
submission.to_csv('submission.csv', index=False)





