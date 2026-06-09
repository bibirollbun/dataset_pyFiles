# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd

# æ•°æ�®è·¯å¾„
train_path = "/kaggle/input/playground-series-s5e5/train.csv"
test_path = "/kaggle/input/playground-series-s5e5/test.csv"
submission_path = "/kaggle/input/playground-series-s5e5/sample_submission.csv"

# è¯»å�–æ•°æ�®
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
submission_df = pd.read_csv(submission_path)

# æ•°æ�®é›†ä¿¡æ�¯æ¦‚è§ˆ
print("Train Dataset Info:")
print(train_df.info())
print("\nTest Dataset Info:")
print(test_df.info())


# å‰�äº”è¡Œæ•°æ�®é¢„è§ˆ
print("\nFirst 5 rows of Train Dataset:")
print(train_df.head())

print("\nFirst 5 rows of Test Dataset:")
print(test_df.head())

print("\nFirst 5 rows of Sample Submission:")
print(submission_df.head())


print("\nâœ… è®­ç»ƒæ•°æ�®å�„åˆ—çš„æœ€å¤§å€¼ï¼š")
print(train_df.max(numeric_only=True))

print("\nâœ… è®­ç»ƒæ•°æ�®å�„åˆ—çš„æœ€å°�å€¼ï¼š")
print(train_df.min(numeric_only=True))



!pip install -q ipyplot


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# è®¾ç½®é£�æ ¼
sns.set(style="whitegrid")

# 1. Calories åˆ†å¸ƒ
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(train_df["Calories"], bins=50, kde=True, color='blue')
plt.title("Calories Distribution")

plt.subplot(1, 2, 2)
sns.boxplot(x=train_df["Calories"], color='blue')
plt.title("Calories Boxplot")

plt.tight_layout()
plt.show()

# 2. æ•°å€¼å�‹ç‰¹å¾�åˆ†å¸ƒä¸�æ•£ç‚¹å›¾
numeric_features = ["Duration", "Heart_Rate", "Body_Temp", "Height", "Weight", "Age"]

# ç»˜åˆ¶æ•£ç‚¹å›¾ä¸�åˆ†å¸ƒå›¾
fig, axes = plt.subplots(3, 2, figsize=(14, 18))
axes = axes.flatten()

for i, feature in enumerate(numeric_features):
    sns.histplot(train_df[feature], bins=50, kde=True, ax=axes[i], color='green')
    axes[i].set_title(f"{feature} Distribution")
    
plt.tight_layout()
plt.show()

# âœ… 3. æ•°å€¼ç‰¹å¾�ä¸� Calories çš„å…³ç³» (æ•£ç‚¹å›¾ + ç›¸å…³ç³»æ•°ï¼ŒåŒºåˆ†æ€§åˆ«)
fig, axes = plt.subplots(3, 2, figsize=(14, 18))
axes = axes.flatten()

for i, feature in enumerate(numeric_features):
    sns.scatterplot(
        data=train_df, 
        x=feature, 
        y="Calories", 
        hue="Sex", 
        palette=["blue", "orange"], 
        alpha=0.5, 
        ax=axes[i]
    )
    # è®¡ç®— Spearman ç›¸å…³ç³»æ•°ï¼ˆä¸�åŒºåˆ†æ€§åˆ«ï¼‰
    spearman_corr = train_df[[feature, "Calories"]].corr(method="spearman").iloc[0, 1]
    axes[i].set_title(f"{feature} vs Calories (Spearman: {spearman_corr:.2f})")
    
plt.tight_layout()
plt.show()

# 4. Sex åˆ†å¸ƒå�Šå…¶å¯¹ Calories çš„å½±å“�
plt.figure(figsize=(12, 6))
sns.violinplot(x="Sex", y="Calories", data=train_df, palette="Set2")
plt.title("Sex vs Calories")
plt.show()

# 5. ç‰¹å¾�é—´ç›¸å…³æ€§çŸ©é˜µ
# ä»…é€‰æ‹©æ•°å€¼å�‹åˆ—è¿›è¡Œç›¸å…³æ€§çŸ©é˜µè®¡ç®—
numeric_columns = train_df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = train_df[numeric_columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix of Numeric Features")
plt.show()



# import matplotlib.pyplot as plt
# import seaborn as sns

# # è®¾ç½®æ ·å¼�
# sns.set(style="whitegrid")

# # ç­›é€‰ç›¸å…³æ€§è¾ƒé«˜çš„ç‰¹å¾�
# selected_columns = ["Duration", "Heart_Rate", "Body_Temp", "Calories"]
# corr_matrix = train_df[selected_columns].corr()

# # 1. å±€éƒ¨ç›¸å…³æ€§çƒ­åŠ›å›¾
# plt.figure(figsize=(8, 6))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", cbar=True)
# plt.title("Correlation Matrix (Duration, Heart_Rate, Body_Temp, Calories)")
# plt.show()

# # 2. æ•£ç‚¹å›¾ + æ‹Ÿå�ˆçº¿
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# for i, feature in enumerate(["Duration", "Heart_Rate", "Body_Temp"]):
#     sns.regplot(x=train_df[feature], y=train_df["Calories"], ax=axes[i], color='blue', scatter_kws={'alpha': 0.3})
#     axes[i].set_title(f"{feature} vs. Calories")

# plt.tight_layout()
# plt.show()

# # 3. è�”å�ˆåˆ†å¸ƒå›¾ (Pairplot)
# sns.pairplot(train_df[selected_columns], kind='reg', diag_kind='kde', plot_kws={'scatter_kws': {'alpha': 0.3}})
# plt.suptitle("Pairplot of Key Features and Calories", y=1.02)
# plt.show()



# # ç‰¹å¾�åˆ—
# feature_columns = ["Sex", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

# # æ‰¾å‡ºå­˜åœ¨å¤šä¸ª Calories å€¼çš„ç»„å�ˆ
# duplicate_calories = train_df.groupby(feature_columns)["Calories"].nunique()

# # è�·å�–éœ€è¦�åˆ é™¤çš„ç»„å�ˆ
# duplicate_combinations = duplicate_calories[duplicate_calories > 1].index.tolist()

# # åˆ é™¤è¿™äº›ç»„å�ˆå¯¹åº”çš„è¡Œ
# for combination in duplicate_combinations:
#     condition = (train_df[feature_columns] == combination).all(axis=1)
#     indices_to_drop = train_df[condition].index
#     train_df.drop(indices_to_drop, inplace=True)

# print(f"After removing conflicting rows, dataset shape: {train_df.shape}")



# ç‰¹å¾�ç»„å�ˆ
feature_columns = ["Sex", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

# æŒ‰ç‰¹å¾�ç»„å�ˆè¿›è¡Œåˆ†ç»„å¹¶æ£€æŸ¥å�¡è·¯é‡Œå€¼çš„å”¯ä¸€æ€§
duplicate_calories = train_df.groupby(feature_columns)["Calories"].nunique()

# ç­›é€‰å‡ºå­˜åœ¨å¤šä¸ªå�¡è·¯é‡Œå€¼çš„ç»„
duplicate_calories = duplicate_calories[duplicate_calories > 1]

# æŸ¥çœ‹å¼‚å¸¸ç�°è±¡
if not duplicate_calories.empty:
    print(f"Found {len(duplicate_calories)} feature combinations with different Calories values.")
    print(duplicate_calories)
else:
    print("No feature combinations with differing Calories values found.")


# æŸ¥çœ‹è¯¦ç»†å¼‚å¸¸ä¿¡æ�¯
feature_columns = ["Sex", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

# æŒ‰ç‰¹å¾�ç»„å�ˆè¿›è¡Œåˆ†ç»„å¹¶åˆ—å‡ºå�¡è·¯é‡Œå€¼å�Šå…¶å‡ºç�°é¢‘æ¬¡
calories_variation = train_df.groupby(feature_columns)["Calories"].agg(['min', 'max', 'nunique']).reset_index()

# ç­›é€‰å‡ºå­˜åœ¨å¤šä¸ªå�¡è·¯é‡Œå€¼çš„ç»„å�ˆ
calories_variation = calories_variation[calories_variation["nunique"] > 1]

# è®¡ç®—å�¡è·¯é‡Œå€¼å·®å¼‚èŒƒå›´
calories_variation["Difference"] = calories_variation["max"] - calories_variation["min"]

# æ�’åº�ä»¥ä¾¿æŸ¥çœ‹æœ€å¤§å·®å¼‚å€¼
calories_variation = calories_variation.sort_values(by="Difference", ascending=False)

# æ˜¾ç¤ºå‰� 10 ä¸ªå·®å¼‚æœ€å¤§çš„ç»„å�ˆ
print(calories_variation.head(10))



# ç‰¹å¾�ç»„å�ˆ
feature_columns = ["Sex", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

# æŒ‰ç‰¹å¾�ç»„å�ˆè¿›è¡Œåˆ†ç»„å¹¶æ£€æŸ¥å�¡è·¯é‡Œå€¼çš„å”¯ä¸€æ€§
duplicate_calories = train_df.groupby(feature_columns)["Calories"].nunique()

# ç­›é€‰å‡ºå­˜åœ¨å¤šä¸ªå�¡è·¯é‡Œå€¼çš„ç»„å�ˆ
duplicate_combinations = duplicate_calories[duplicate_calories > 1].reset_index()

# å°†é‡�å¤�ç»„å�ˆè½¬æ�¢ä¸º DataFrame ä»¥ä¾¿å¿«é€ŸæŸ¥æ‰¾
duplicate_combinations["is_duplicate"] = True

# å�ˆå¹¶æ•°æ�®é›†ï¼Œæ ‡è®°éœ€è¦�åˆ é™¤çš„è¡Œ
train_df = train_df.merge(duplicate_combinations, on=feature_columns, how='left')

# åˆ é™¤æ ‡è®°ä¸ºé‡�å¤�çš„è¡Œ
train_df = train_df[train_df["is_duplicate"].isnull()]

# åˆ é™¤è¾…åŠ©åˆ—å¹¶é‡�ç½®ç´¢å¼•
train_df.drop(columns=["is_duplicate"], inplace=True)
train_df.reset_index(drop=True, inplace=True)

# æ‰“å�°åˆ é™¤å��çš„æ•°æ�®é›†å½¢çŠ¶
print(f"After removing conflicting rows, dataset shape: {train_df.shape}")






import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
# è¯»å�–æ•°æ�®
train_df = pd.read_csv(train_path)

# ç‰¹å¾�å·¥ç¨‹
train_df['BMI'] = train_df['Weight'] / (train_df['Height'] / 100) ** 2
train_df['Duration_HeartRate'] = train_df['Duration'] * train_df['Heart_Rate']
train_df['BodyTemp_Duration'] = train_df['Body_Temp'] * train_df['Duration']
train_df['Age_HeartRate'] = train_df['Age'] * train_df['Heart_Rate']
train_df['Duration_Squared'] = train_df['Duration'] ** 2
train_df['Duration_Cubed'] = train_df['Duration'] ** 3

# ç”·æ€§ CB å…¬å¼�
train_df['CB_Male'] = train_df['Duration'] * (
    0.6309 * train_df['Heart_Rate'] +
    0.1988 * train_df['Weight'] +
    0.2017 * train_df['Age'] - 55.0969
) / 4.184

# å¥³æ€§ CB å…¬å¼�
train_df['CB_Female'] = train_df['Duration'] * (
    0.4472 * train_df['Heart_Rate'] -
    0.1263 * train_df['Weight'] +
    0.074 * train_df['Age'] - 20.4022
) / 4.184

# HR å� æœ€å¤§å¿ƒç�‡æ¯”ä¾‹
train_df["HR_pct_max"] = train_df["Heart_Rate"] / (220 - train_df["Age"])

# BMI åˆ†ç±»
def bmi_category(bmi):
    if bmi < 18.5 or bmi >= 30:
        return "Abnormal"
    elif bmi < 25:
        return "Normal"
    else:
        return "Overweight"

train_df["BMI_Category"] = train_df["BMI"].apply(bmi_category)

# å¹´é¾„åˆ†æ®µ
def age_group(age):
    if age < 35:
        return "adult"
    elif age < 60:
        return "middle_age"
    elif age < 80:
        return "elderly"
    else:
        return "old"

train_df["Age_Group"] = train_df["Age"].apply(age_group)


# å¯¹æ•°è½¬æ�¢ç›®æ ‡åˆ—
train_df['Log_Calories'] = np.log1p(train_df['Calories'])

categorical_cols = ['BMI_Category', 'Age_Group']
onehot_encoded = pd.get_dummies(train_df[categorical_cols], prefix=categorical_cols)

# å�ˆå¹¶åˆ°å�Ÿå§‹ train_df ä¸­ï¼Œå¹¶åˆ é™¤å�Ÿå§‹ç±»åˆ«åˆ—
train_df = pd.concat([train_df, onehot_encoded], axis=1)
train_df.drop(columns=categorical_cols, inplace=True)

onehot_feature_names = list(onehot_encoded.columns)

# ç‰¹å¾�åˆ—è¡¨ï¼ˆä¸�å†�åŒ…å�«æ€§åˆ«ä¿¡æ�¯ï¼‰
# features_male = [
#     'Duration_Squared', 'Duration_Cubed', 'Heart_Rate',
#     'Body_Temp', 'BMI', 'Duration_HeartRate', 'BodyTemp_Duration',
#     'Age_HeartRate', 'CB_Male', 'HR_pct_max'
# ] + onehot_feature_names  # â¬… æ·»åŠ  OneHot ç¼–ç �åˆ—
features_male = [
    'Duration_Squared', 'Duration_Cubed', 'Heart_Rate',
    'Body_Temp', 'BMI', 'Duration_HeartRate', 'BodyTemp_Duration',
    'Age_HeartRate', 'CB_Male', 'HR_pct_max'
]

features_female = ['Duration_Squared', 'Duration_Cubed', 'Heart_Rate',
                   'Body_Temp', 'BMI', 'Duration_HeartRate', 'BodyTemp_Duration',
                   'Age_HeartRate', 'CB_Female']

# æ•°æ�®æ‹†åˆ†
male_df = train_df[train_df['Sex'] == 'male'].drop(columns=['Sex'])
female_df = train_df[train_df['Sex'] == 'female'].drop(columns=['Sex'])

# X å’Œ y æ•°æ�®æ�„å»º
X_male = male_df[features_male]
y_male = male_df['Log_Calories']

X_female = female_df[features_female]
y_female = female_df['Log_Calories']

# æ‰“å�°æ£€æŸ¥
print("Male Dataset Shape:", X_male.shape, y_male.shape)
print("Female Dataset Shape:", X_female.shape, y_female.shape)



# ================== RMSLE è®¡ç®—å‡½æ•° ==================
def calculate_rmsle(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32).to(device)).squeeze().cpu().numpy()
    
    # è½¬æ�¢å›�å�Ÿå§‹å°ºåº¦
    y_original = np.expm1(y)
    preds_original = np.expm1(preds)

    # è®¡ç®— RMSLE
    rmsle = np.sqrt(mean_squared_error(np.log1p(y_original), np.log1p(preds_original)))
    return rmsle


from sklearn.model_selection import train_test_split
# Train-Test Split
X_train_male, X_val_male, y_train_male, y_val_male = train_test_split(X_male, y_male, test_size=0.2, random_state=42)
X_train_female, X_val_female, y_train_female, y_val_female = train_test_split(X_female, y_female, 
                                                                              test_size=0.2, random_state=42)

# æ ‡å‡†åŒ–
scaler_male = StandardScaler()
X_train_male = scaler_male.fit_transform(X_train_male)
X_val_male = scaler_male.transform(X_val_male)

scaler_female = StandardScaler()
X_train_female = scaler_female.fit_transform(X_train_female)
X_val_female = scaler_female.transform(X_val_female)


# #NN

# # å®šä¹‰ç¥�ç»�ç½‘ç»œæ¨¡å�‹
# class CalorieModel(nn.Module):
#     def __init__(self, input_dim):
#         super(CalorieModel, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 1)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # Model Initialization
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ================== ç”·æ€§æ¨¡å�‹è®­ç»ƒ ==================
# print("Training Male Model...")
# male_model = CalorieModel(input_dim=X_train_male.shape[1]).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(male_model.parameters(), lr=1e-3)
# num_epochs = 2000

# for epoch in range(num_epochs):
#     male_model.train()
#     optimizer.zero_grad()
#     outputs = male_model(torch.tensor(X_train_male, dtype=torch.float32).to(device))
#     loss = criterion(outputs.squeeze(), torch.tensor(y_train_male.values, dtype=torch.float32).to(device))
#     loss.backward()
#     optimizer.step()

#     # Validation
#     male_model.eval()
#     with torch.no_grad():
#         val_outputs = male_model(torch.tensor(X_val_male, dtype=torch.float32).to(device)).squeeze()
#         val_loss = criterion(val_outputs, torch.tensor(y_val_male.values, dtype=torch.float32).to(device))

#     # æ¯�50ä¸ªepochæ‰“å�°ä¸€æ¬¡ç»“æ�œ
#     if epoch % 100 == 0:
#         train_rmsle_male = calculate_rmsle(male_model, X_train_male, y_train_male)
#         val_rmsle_male = calculate_rmsle(male_model, X_val_male, y_val_male)

#         print(f"Epoch {epoch+1}/{num_epochs} - Male Model Loss: {loss.item():.4f} - Val Loss: {val_loss.item():.4f}")
#         print(f"Male Model RMSLE - Train: {train_rmsle_male:.4f}, Val: {val_rmsle_male:.4f}")
# # ================== å¥³æ€§æ¨¡å�‹è®­ç»ƒ ==================
# print("\nTraining Female Model...")
# female_model = CalorieModel(input_dim=X_train_female.shape[1]).to(device)
# optimizer = optim.Adam(female_model.parameters(), lr=1e-3)

# for epoch in range(num_epochs):
#     female_model.train()
#     optimizer.zero_grad()
#     outputs = female_model(torch.tensor(X_train_female, dtype=torch.float32).to(device))
#     loss = criterion(outputs.squeeze(), torch.tensor(y_train_female.values, dtype=torch.float32).to(device))
#     loss.backward()
#     optimizer.step()

#     # Validation
#     female_model.eval()
#     with torch.no_grad():
#         val_outputs = female_model(torch.tensor(X_val_female, dtype=torch.float32).to(device)).squeeze()
#         val_loss = criterion(val_outputs, torch.tensor(y_val_female.values, dtype=torch.float32).to(device))

#     # æ¯�50ä¸ªepochæ‰“å�°ä¸€æ¬¡ç»“æ�œ
#     if epoch % 100 == 0:
#         train_rmsle_female = calculate_rmsle(female_model, X_train_female, y_train_female)
#         val_rmsle_female = calculate_rmsle(female_model, X_val_female, y_val_female)

#         print(f"Epoch {epoch+1}/{num_epochs} - Female Model Loss: {loss.item():.4f} - Val Loss: {val_loss.item():.4f}")
#         print(f"Female Model RMSLE - Train: {train_rmsle_female:.4f}, Val: {val_rmsle_female:.4f}")

# # ================== æ¨¡å�‹è¯„ä¼° ==================
# def evaluate_model(model, X, y):
#     model.eval()
#     with torch.no_grad():
#         preds = model(torch.tensor(X, dtype=torch.float32).to(device)).squeeze().cpu().numpy()
#     y_original = np.expm1(y)
#     preds_original = np.expm1(preds)
#     rmse = np.sqrt(mean_squared_error(y_original, preds_original))
#     return rmse

# # ç”·æ€§æ¨¡å�‹è¯„ä¼°
# train_rmse_male = evaluate_model(male_model, X_train_male, y_train_male)
# val_rmse_male = evaluate_model(male_model, X_val_male, y_val_male)

# print(f"\nMale Model - Train RMSE: {train_rmse_male:.4f}, Val RMSE: {val_rmse_male:.4f}")

# # å¥³æ€§æ¨¡å�‹è¯„ä¼°
# train_rmse_female = evaluate_model(female_model, X_train_female, y_train_female)
# val_rmse_female = evaluate_model(female_model, X_val_female, y_val_female)

# print(f"Female Model - Train RMSE: {train_rmse_female:.4f}, Val RMSE: {val_rmse_female:.4f}")



# # ================== è®¡ç®—ç”·æ€§æ¨¡å�‹ RMSLE ==================
# train_rmsle_male = calculate_rmsle(male_model, X_train_male, y_train_male)
# val_rmsle_male = calculate_rmsle(male_model, X_val_male, y_val_male)

# print(f"Male Model - Train RMSLE: {train_rmsle_male:.4f}, Validation RMSLE: {val_rmsle_male:.4f}")

# # ================== è®¡ç®—å¥³æ€§æ¨¡å�‹ RMSLE ==================
# train_rmsle_female = calculate_rmsle(female_model, X_train_female, y_train_female)
# val_rmsle_female = calculate_rmsle(female_model, X_val_female, y_val_female)

# print(f"Female Model - Train RMSLE: {train_rmsle_female:.4f}, Validation RMSLE: {val_rmsle_female:.4f}")



# ================= XGBoost Model ==================
xgb_model_male = xgb.XGBRegressor(
        max_depth=9,
        colsample_bytree=0.7,
        subsample=0.9,
        n_estimators=3000,
        learning_rate=0.01,
        gamma=0.01,
        max_delta_step=2,
        early_stopping_rounds=200,
        random_state=42,
        eval_metric="rmse",
        enable_categorical=True,
        device = 'cuda'    
    )
xgb_model_male.fit(
    X_train_male, y_train_male,
    eval_set=[(X_val_male, y_val_male)],
    verbose=False
)


xgb_model_female = xgb.XGBRegressor(
        max_depth=9,
        colsample_bytree=0.7,
        subsample=0.9,
        n_estimators=3000,
        learning_rate=0.01,
        gamma=0.01,
        max_delta_step=2,
        early_stopping_rounds=200,
        random_state=42,
        eval_metric="rmse",
        enable_categorical=True,
        device = 'cuda'    
    )
xgb_model_female.fit(
    X_train_female, y_train_female,
    eval_set=[(X_val_female, y_val_female)],
    verbose=False
)


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


xgb_val_preds_male = xgb_model_male.predict(X_val_male)
xgb_val_preds_male = np.expm1(xgb_val_preds_male)

xgb_val_preds_female = xgb_model_female.predict(X_val_female)
xgb_val_preds_female = np.expm1(xgb_val_preds_female)

y_val_original_male = np.expm1(y_val_male)
y_val_original_female = np.expm1(y_val_female)

rmsle_xgb_male = rmsle(y_val_original_male, xgb_val_preds_male)
print(f"Validation RMSLE (XGBoost) Male: {rmsle_xgb_male:.4f}")

rmsle_xgb_female = rmsle(y_val_original_female, xgb_val_preds_female)
print(f"Validation RMSLE (XGBoost) feMale: {rmsle_xgb_female:.4f}")


import catboost as cb
# ================= CatBoost Model ==================
cat_model_male = cb.CatBoostRegressor(
        iterations= 2500,
        learning_rate= 0.02,
        depth= 10,
        loss_function= 'RMSE',
        l2_leaf_reg= 3,
        random_seed= 42,
        eval_metric= 'RMSE',
        early_stopping_rounds = 200,
        verbose= 1000,
        task_type= 'GPU'  )
cat_model_male.fit(
    X_train_male, y_train_male,
    eval_set=[(X_val_male, y_val_male)],
    verbose=False
)


cat_model_female = cb.CatBoostRegressor(
        iterations= 2500,
        learning_rate= 0.02,
        depth= 10,
        loss_function= 'RMSE',
        l2_leaf_reg= 3,
        random_seed= 42,
        eval_metric= 'RMSE',
        early_stopping_rounds = 200,
        verbose= 1000,
        task_type= 'GPU'  )
cat_model_female.fit(
    X_train_female, y_train_female,
    eval_set=[(X_val_female, y_val_female)],
    verbose=False
)


# CatBoost Predictions
cat_val_preds_male = cat_model_male.predict(X_val_male)
cat_val_preds_male = np.expm1(cat_val_preds_male)  # Inverse log1p transform

rmsle_cat_male = rmsle(y_val_original_male, cat_val_preds_male)
print(f"Validation RMSLE (CatBoost) Male: {rmsle_cat_male:.4f}")

# CatBoost Predictions
cat_val_preds_female = cat_model_female.predict(X_val_female)
cat_val_preds_female = np.expm1(cat_val_preds_female)  # Inverse log1p transform

rmsle_cat_female = rmsle(y_val_original_female, cat_val_preds_female)
print(f"Validation RMSLE (CatBoost) feMale: {rmsle_cat_female:.4f}")


# === è��å�ˆé¢„æµ‹ï¼ˆXGBoost + CatBoostï¼Œç®€å�•åŠ æ�ƒï¼‰
val_preds_male = 0.5 * xgb_val_preds_male + 0.5 * cat_val_preds_male
val_preds_female = 0.5 * xgb_val_preds_female + 0.5 * cat_val_preds_female

# === è®¡ç®—åˆ†æ€§åˆ« RMSLE
rmsle_male = rmsle(y_val_original_male, val_preds_male)
rmsle_female = rmsle(y_val_original_female, val_preds_female)

# === æ‹¼æ�¥æ€»ä½“éªŒè¯�é›†
all_true = np.concatenate([y_val_original_male, y_val_original_female])
all_pred = np.concatenate([val_preds_male, val_preds_female])
all_pred = np.clip(all_pred, 1, 314)

# === è®¡ç®—æ€»ä½“ RMSLE
rmsle_total = rmsle(all_true, all_pred)

# === è¾“å‡ºç»“æ�œ
print(f"âœ… Fused RMSLE Male: {rmsle_male:.4f}")
print(f"âœ… Fused RMSLE Female: {rmsle_female:.4f}")
print(f"ğŸ�¯ Fused RMSLE Overall: {rmsle_total:.4f}")



best_rmsle = float('inf')
best_w = 0

for w in np.linspace(0, 1, 101):
    val_preds_male = w * xgb_val_preds_male + (1 - w) * cat_val_preds_male
    val_preds_female = w * xgb_val_preds_female + (1 - w) * cat_val_preds_female
    
    all_true = np.concatenate([y_val_original_male, y_val_original_female])
    all_pred = np.concatenate([val_preds_male, val_preds_female])
    all_pred = np.clip(all_pred, 1, 314)
    
    rmsle_total = rmsle(all_true, all_pred)
    
    if rmsle_total < best_rmsle:
        best_rmsle = rmsle_total
        best_w = w

print(f"ğŸ�¯ Best Weight for XGBoost: {best_w:.2f} â€” RMSLE: {best_rmsle:.4f}")



print("Max predicted log-calorie (before expm1):", xgb_val_preds_male.max())
print("Min predicted log-calorie (before expm1):", xgb_val_preds_male.min())


# è¯»å�–æ•°æ�®
test_df = pd.read_csv(test_path)

# ç‰¹å¾�å·¥ç¨‹
test_df['BMI'] = test_df['Weight'] / (test_df['Height'] / 100) ** 2
test_df['Duration_HeartRate'] = test_df['Duration'] * test_df['Heart_Rate']
test_df['BodyTemp_Duration'] = test_df['Body_Temp'] * test_df['Duration']
test_df['Age_HeartRate'] = test_df['Age'] * test_df['Heart_Rate']
test_df['Duration_Squared'] = test_df['Duration'] ** 2
test_df['Duration_Cubed'] = test_df['Duration'] ** 3

# ç”·æ€§ CB å…¬å¼�
test_df['CB_Male'] = test_df['Duration'] * (
    0.6309 * test_df['Heart_Rate'] +
    0.1988 * test_df['Weight'] +
    0.2017 * test_df['Age'] - 55.0969
) / 4.184

# å¥³æ€§ CB å…¬å¼�
test_df['CB_Female'] = test_df['Duration'] * (
    0.4472 * test_df['Heart_Rate'] -
    0.1263 * test_df['Weight'] +
    0.074 * test_df['Age'] - 20.4022
) / 4.184

# HR å� æœ€å¤§å¿ƒç�‡æ¯”ä¾‹
test_df["HR_pct_max"] = test_df["Heart_Rate"] / (220 - test_df["Age"])

# BMI åˆ†ç±»
def bmi_category(bmi):
    if bmi < 18.5 or bmi >= 30:
        return "Abnormal"
    elif bmi < 25:
        return "Normal"
    else:
        return "Overweight"

test_df["BMI_Category"] = test_df["BMI"].apply(bmi_category)

# å¹´é¾„åˆ†æ®µ
def age_group(age):
    if age < 35:
        return "adult"
    elif age < 60:
        return "middle_age"
    elif age < 80:
        return "elderly"
    else:
        return "old"

test_df["Age_Group"] = test_df["Age"].apply(age_group)


# å¯¹æ•°è½¬æ�¢ç›®æ ‡åˆ—
# test_df['Log_Calories'] = np.log1p(test_df['Calories'])

categorical_cols = ['BMI_Category', 'Age_Group']
onehot_encoded = pd.get_dummies(test_df[categorical_cols], prefix=categorical_cols)

# å�ˆå¹¶åˆ°å�Ÿå§‹ train_df ä¸­ï¼Œå¹¶åˆ é™¤å�Ÿå§‹ç±»åˆ«åˆ—
test_df = pd.concat([test_df, onehot_encoded], axis=1)
test_df.drop(columns=categorical_cols, inplace=True)

onehot_feature_names = list(onehot_encoded.columns)

# ç‰¹å¾�åˆ—è¡¨ï¼ˆä¸�å†�åŒ…å�«æ€§åˆ«ä¿¡æ�¯ï¼‰
# features_male = [
#     'Duration_Squared', 'Duration_Cubed', 'Heart_Rate',
#     'Body_Temp', 'BMI', 'Duration_HeartRate', 'BodyTemp_Duration',
#     'Age_HeartRate', 'CB_Male', 'HR_pct_max'
# ] + onehot_feature_names  # â¬… æ·»åŠ  OneHot ç¼–ç �åˆ—
features_male = [
    'Duration_Squared', 'Duration_Cubed', 'Heart_Rate',
    'Body_Temp', 'BMI', 'Duration_HeartRate', 'BodyTemp_Duration',
    'Age_HeartRate', 'CB_Male', 'HR_pct_max'
]

features_female = ['Duration_Squared', 'Duration_Cubed', 'Heart_Rate',
                   'Body_Temp', 'BMI', 'Duration_HeartRate', 'BodyTemp_Duration',
                   'Age_HeartRate', 'CB_Female']

# æ•°æ�®æ‹†åˆ†
male_df_test = test_df[test_df['Sex'] == 'male'].drop(columns=['Sex'])
female_df_test = test_df[test_df['Sex'] == 'female'].drop(columns=['Sex'])

# X å’Œ y æ•°æ�®æ�„å»º
X_male_test = male_df_test[features_male]
# y_male = male_df['Log_Calories']

X_female_test = female_df_test[features_female]
# y_female = female_df['Log_Calories']

# æ‰“å�°æ£€æŸ¥
print("Male Dataset Shape:", X_male_test.shape)
print("Female Dataset Shape:", X_female_test.shape)


male_ids = test_df[test_df['Sex'] == 'male']['id'].values
female_ids = test_df[test_df['Sex'] == 'female']['id'].values



# é¢„æµ‹ï¼ˆå�‡è®¾æ¨¡å�‹å·²è®­ç»ƒï¼‰
xgb_preds_male = np.expm1(xgb_model_male.predict(X_male_test))
cat_preds_male = np.expm1(cat_model_male.predict(X_male_test))

xgb_preds_female = np.expm1(xgb_model_female.predict(X_female_test))
cat_preds_female = np.expm1(cat_model_female.predict(X_female_test))

# è��å�ˆ
final_preds_male = 0.65 * xgb_preds_male + 0.35 * cat_preds_male
final_preds_female = 0.65 * xgb_preds_female + 0.35 * cat_preds_female

# æ‹¼æ�¥ä¿�å­˜
submission = pd.DataFrame({
    "id": np.concatenate([male_ids, female_ids]),
    "Calories": np.concatenate([final_preds_male, final_preds_female])
}).sort_values("id")

submission.to_csv("submission.csv", index=False)





