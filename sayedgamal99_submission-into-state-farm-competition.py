%%capture cell
!pip install ultralytics tqdm


from kaggle_secrets import UserSecretsClient
from kaggle.api.kaggle_api_extended import KaggleApi
import os

secrets = UserSecretsClient()
kaggle_key = secrets.get_secret("KAGGLE_KEY")
kaggle_username = 'sayedgamal99'


os.environ["KAGGLE_USERNAME"] = kaggle_username
os.environ["KAGGLE_KEY"] = kaggle_key

api = KaggleApi()
api.authenticate()

print("✅ Kaggle API authenticated successfully!")


 !wget -O best.pt "https://storage.googleapis.com/kaggle-script-versions/220359309/output/runs/classify/train/weights/best.pt?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250205%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250205T053930Z&X-Goog-Expires=3600&X-Goog-SignedHeaders=host&X-Goog-Signature=95c57893c3a8a25003e1b4d677fedfd1b9f516241e3d9797f2d56d2e6696c70440004fb770520a4fc1383c14f9796d597d111f652128785b0bbe75bb8ddf64b3e64a55ed29f971f0068d8c41dba5536c20e36f21bd53157832dad35847513e360f4a7cf0a3e4685094e4c703b70d051d8197456eb6bae1d88dd71d83e64f5aae811d9dba09bcdadb6203efa77f3510787b67ba84282a4f6edd7f3e57e27476ff48c34ad9e9ae842146b2d9e9e444944a13df64bb95b8fc1c05fc824b933c27b37eff04d78386d33cc50da8ef05d49299447ed01d9ad5cad1f1369de9a02d72326dd0b2b3dc3ab8eb265076193dabc91105c5a7e144cc3ebbe07408c6050d9ab1"


import os
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import time

def create_submission(batch_size=32):
    # Load the trained model
    model = YOLO('/kaggle/working/best.pt')
    
    # Path to test images
    test_dir = '/kaggle/input/state-farm-distracted-driver-detection/imgs/test'
    
    # Get sorted list of test images for consistent order
    test_images = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])
    
    # Initialize results dictionary
    results_dict = {'img': test_images}
    for i in range(10):
        results_dict[f'c{i}'] = []
    
    # Get predictions using YOLO's built-in batch processing
    results = model.predict(
        source=test_dir,
        batch=batch_size,
        conf=0.1,
        save=False,
        stream=True,
        verbose=False  # This will show progress bar
    )
    
    # Process results
    for idx, result in enumerate(results):
        probs = result.probs.data.cpu().numpy()
        
        # If no probabilities, use uniform distribution
        if probs is None or len(probs) == 0:
            probs = np.ones(10) / 10
            
        # Add probabilities for each class
        for i in range(10):
            results_dict[f'c{i}'].append(float(probs[i]))
    
    # Create DataFrame
    submission_df = pd.DataFrame(results_dict)
    
    # Save submission file
    submission_path = 'submission.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to {submission_path}")



create_submission(batch_size=64)


message = "YOLO model submission1 with batch processing"
competition = "state-farm-distracted-driver-detection"
api.competition_submit("/kaggle/working/submission.csv", message, competition)
print(f"Successfully submitted to {competition}")




