import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from category_encoders import TargetEncoder
import xgboost as xgb
plt.style.use("seaborn-v0_8-darkgrid")
warnings.filterwarnings("ignore")
plt.rc("font",family="SimHei",size="15")  
# import csv
train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
datasert_df = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


datasert_df = (
    datasert_df
    .rename(columns={'Personality': 'match_p'})
    .drop_duplicates(['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                      'Going_outside', 'Drained_after_socializing', 
                      'Friends_circle_size', 'Post_frequency'])
)

merge_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
              'Going_outside', 'Drained_after_socializing', 
              'Friends_circle_size', 'Post_frequency']

train_df = train_df.merge(datasert_df, how='left', on=merge_cols)
test_df = test_df.merge(datasert_df, how='left', on=merge_cols)

train_df.info()


numeric_df = train_df.select_dtypes(include='number').drop(columns=['id'])
numeric_df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(),annot=True,cmap='coolwarm',fmt='.2f', vmin=-1, vmax=1)
plt.title('Heatmap')
plt.show()


train_ID = train_df['id']
test_ID = test_df['id']

#Now drop the  'id' colum since it's unnecessary for  the prediction process.
train_df.drop("id", axis = 1, inplace = True)
test_df.drop("id", axis = 1, inplace = True)

ntrain = train_df.shape[0] 
ntest = test_df.shape[0]
y_train = train_df['Personality'].map({'Extrovert': 1, 'Introvert': 0}).values # è®­ç»ƒé›†çš„Y

all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
all_data.drop(['Personality'], axis=1, inplace=True)


def fill_missing_by_quantile_group(df, group_source_col, target_col, quantiles=[0, 0.25, 0.5, 0.75, 1.0], labels=None):
    """
    Group the target_col based on quantiles of group_source_col, and fill missing values in target_col
    within each group using the group's median.
    
    Parameters:
        df (pd.DataFrame): Original dataset
        group_source_col (str): Column used for grouping (numerical)
        target_col (str): Target column to fill missing values
        quantiles (list): Quantile cut points for grouping (default is quartiles)
        labels (list): Labels for each group (default auto-generated as Q1, Q2, ...)
    
    Returns:
        pd.DataFrame: DataFrame with missing values filled (in-place modification)
    """
    #  Automatically generate group labels
    if labels is None:
        labels = [f'Q{i+1}' for i in range(len(quantiles)-1)]

    temp_bin_col = f'{group_source_col}_bin'

    # Step 1: Create grouping column
    df[temp_bin_col] = pd.qcut(df[group_source_col], q=quantiles, labels=labels)

    # Step 2: Fill missing values within each group using the group's median
    df[target_col] = df[target_col].fillna(df.groupby(temp_bin_col)[target_col].transform('median'))

    # Step 3: Remove the temporary grouping column
    df.drop(columns=[temp_bin_col], inplace=True)

    return df

all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Social_event_attendance',
    target_col='Time_spent_Alone'
)

all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Going_outside',
    target_col='Time_spent_Alone'
)

all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Friends_circle_size',
    target_col='Social_event_attendance'
)

all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Going_outside',
    target_col='Social_event_attendance'
)

all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Post_frequency',
    target_col='Social_event_attendance'
)


all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Social_event_attendance',
    target_col='Going_outside'
)

all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Post_frequency',
    target_col='Friends_circle_size'
)
all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Going_outside',
    target_col='Friends_circle_size'
)
all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Friends_circle_size',
    target_col='Post_frequency'
)
all_data.info()


all_data.fillna({
    'Stage_fear': 'UnKnow',
    'Drained_after_socializing': 'UnKnow'
}, inplace=True)
all_data.info()


all_data = pd.get_dummies(all_data, columns=['Stage_fear', 'Drained_after_socializing','match_p'], prefix=['Stage', 'Drained','match'])
all_data.info()


warnings.filterwarnings('ignore')



X_train = all_data[:ntrain]
X_test = all_data[ntrain:]
X=X_train
y=y_train


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim

# è¯»å�–æ•°æ�®ï¼ˆè¯·æ›¿æ�¢ä¸ºä½ çš„æ•°æ�®ï¼‰
# df = pd.read_csv("your_dataset.csv")

# ç¤ºä¾‹ï¼šå�‡è®¾æ ‡ç­¾åˆ—ä¸º 'Personality'ï¼Œ1 ä¸ºå¤–å�‘ï¼Œ0 ä¸ºå†…å�‘
X = X_train  # ç‰¹å¾�åˆ—
y = y_train                     # æ ‡ç­¾åˆ—

# æ•°å€¼æ ‡å‡†åŒ–
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# è½¬æ�¢ä¸ºå¼ é‡�
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (N, 1)

# æ•°æ�®åˆ’åˆ†
X_train, X_val, y_train, y_val = train_test_split(
    X_tensor, y_tensor, test_size=0.2, stratify=y_tensor, random_state=42
)

# åˆ›å»ºç¥�ç»�ç½‘ç»œæ¨¡å�‹
class PersonalityNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# æ¨¡å�‹åˆ�å§‹åŒ–
input_dim = X_train.shape[1]
model = PersonalityNN(input_dim)

# è®¾å¤‡è®¾ç½®
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
X_train, X_val = X_train.to(device), X_val.to(device)
y_train, y_val = y_train.to(device), y_val.to(device)

# æ�Ÿå¤±å‡½æ•°å’Œä¼˜åŒ–å™¨
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# è®­ç»ƒæ¨¡å�‹
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            val_preds = (model(X_val) >= 0.5).float()
            acc = accuracy_score(y_val.cpu(), val_preds.cpu())
            f1 = f1_score(y_val.cpu(), val_preds.cpu())
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")

# ğŸ‘‰ ç”¨äº�é¢„æµ‹æµ‹è¯•é›†

# æ ‡å‡†åŒ–æµ‹è¯•é›†
test_X_scaled = scaler.transform(X_test)

# è½¬ä¸ºå¼ é‡�
test_tensor = torch.tensor(test_X_scaled, dtype=torch.float32).to(device)

# é¢„æµ‹
model.eval()
test_probs = model(test_tensor).detach().cpu().numpy().flatten()
test_preds = (test_probs >= 0.5).astype(int)

# ç”Ÿæˆ� submissionï¼ˆå�‡è®¾æœ‰ test_IDï¼‰
submission = pd.DataFrame({
    "id": test_ID,
    "Personality": np.where(test_preds == 1, "Extrovert", "Introvert")
})
submission.to_csv("submission_nn.csv", index=False)
print("Submission saved.")

