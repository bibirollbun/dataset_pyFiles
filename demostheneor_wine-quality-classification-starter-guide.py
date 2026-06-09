# ğŸ“š Libraries

# Data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import mode

import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




# === a. Load the training data ===
url = "https://raw.githubusercontent.com/Demosthene-OR/Student-AI-and-Data-Management/main/Wine%20Classification%20Competition/"
url = "/kaggle/input/wine-itba2025/"

df = pd.read_csv(url+"train.csv", sep=',', index_col='id')  # Replace with your dataset
















# === Load the test dataset ===
test_df = pd.read_csv(url+"test.csv", sep=',', index_col='id')  

# .....


# Prepare submission (include 'id' as the first column)
submission = pd.DataFrame({
    "id": test_df.index,
    # "quality": test_preds    # replace test_preds by your prediction variable
})

# Save submission file
submission.to_csv("submission.csv", index=False)
# If running from colab, replace the previous line with the 2 following ones
    # from google.colab import files
    # files.download('submission.csv.')  # Change the file name according to the model you want to download
    
print("âœ… Submission file saved as submission.csv")

