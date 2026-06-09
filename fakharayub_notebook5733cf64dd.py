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


#libraries Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data files Paths
BASE_DIR = '/kaggle/input/linking-writing-processes-to-writing-quality/'
TRAIN_LOGS_PATH = BASE_DIR + 'train_logs.csv'
TRAIN_SCORES_PATH = BASE_DIR + 'train_scores.csv'
TEST_LOGS_PATH = BASE_DIR + 'test_logs.csv'

# Data Load
train_logs = pd.read_csv(TRAIN_LOGS_PATH)
train_scores = pd.read_csv(TRAIN_SCORES_PATH)

# Data Pervious line
print("Train Logs Data (Pehli 5 Lines):")
print(train_logs.head())

print("\n" + "="*50 + "\n") # Line break for clarity

print("Train Scores Data (Pehli 5 Lines):")
print(train_scores.head())

# Data ke bare mein bunyadi maloomat
print("\n" + "="*50 + "\n")
print("Train Logs Info:")
train_logs.info()


# Har student ke liye bunyadi features nikalna
# Hum 'groupby()' ka istemal kar ke data ko har 'id' ke liye alag alag group mein bantein ge

features = train_logs.groupby('id').agg({
    'event_id': ['count'],          # Har student ke kul actions (events)
    'activity': ['nunique'],       # Har student ne kitni mukhtalif qism ki activities keen
    'action_time': ['sum', 'mean'],# Kul action time aur average action time
    'word_count': ['max']          # Har student ka aakhri word count
})

# Columns ke naam ko aasan banana
features.columns = ['total_events', 'unique_activities', 'total_action_time', 'mean_action_time', 'final_word_count']

# Ab is naye 'features' DataFrame ko dekhte hain
print("Har Student Ke Liye Banaye Gaye Bunyadi Features:")
print(features.head())

# In features ko scores ke sath mila kar dekhte hain
full_train_data = features.join(train_scores.set_index('id'))

print("\n" + "="*50 + "\n")
print("Features aur Scores ko Mila Kar:")
print(full_train_data.head())


# Visualization ke liye libraries ko set karna
import matplotlib.pyplot as plt
import seaborn as sns

# Graph ka size thora bara kar rahe hain taake saaf nazar aaye
plt.figure(figsize=(12, 6))

# Seaborn ka istemal karke ek behtar scatter plot banate hain
sns.scatterplot(
    data=full_train_data, 
    x='final_word_count', 
    y='score', 
    alpha=0.5  # alpha se dots ko thora transparent karte hain taake aapas mein mile hue dots nazar aayen
)

# Graph ko behtar banane ke liye titles aur labels add karna
plt.title('Final Word Count vs. Score', fontsize=16)
plt.xlabel('Mazmoon Mein Kul Alfaz (Final Word Count)', fontsize=12)
plt.ylabel('Final Score', fontsize=12)
plt.grid(True) # Graph mein grid lines add karna

# Graph ko display karna
plt.show()


# Har student ka total writing time (milliseconds mein) nikalte hain.
# Yeh har student ke aakhri event ka 'down_time' hoga.
total_time = train_logs.groupby('id')['down_time'].max()

# Isay apne 'full_train_data' mein shamil karte hain
full_train_data['total_writing_time_ms'] = total_time

# Ab naye features banate hain
# Time ko milliseconds se minutes mein convert karte hain
full_train_data['total_writing_time_minutes'] = full_train_data['total_writing_time_ms'] / (1000 * 60)

# "Words Per Minute" (WPM) calculate karte hain
# Hum chhotay essays ko nazar andaz karne ke liye sirf woh essays lete hain jin mein 50 se zyada alfaz hain
full_train_data['wpm'] = full_train_data['final_word_count'] / full_train_data['total_writing_time_minutes']
full_train_data_filtered = full_train_data[full_train_data['final_word_count'] > 50].copy()


# Chalen ab WPM aur Score ke darmiyan graph banate hain
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=full_train_data_filtered, 
    x='wpm', 
    y='score', 
    alpha=0.5
)

plt.title('Words Per Minute (WPM) vs. Score', fontsize=16)
plt.xlabel('Likhne ki Raftaar (Alfaz per Minute)', fontsize=12)
plt.ylabel('Final Score', fontsize=12)
plt.grid(True)
plt.xlim(0, 100) # Aksar logon ki speed 100 se kam hogi, graph ko saaf rakhne ke liye limit set karte hain

plt.show()


# --- Ye code pichle step se hai, bas dobara likh raha hoon ---

# Har student ka total writing time (milliseconds mein) nikalte hain.
total_time = train_logs.groupby('id')['down_time'].max()

# Isay apne 'full_train_data' mein shamil karte hain
full_train_data['total_writing_time_ms'] = total_time

# Time ko milliseconds se minutes mein convert karte hain
full_train_data['total_writing_time_minutes'] = full_train_data['total_writing_time_ms'] / (1000 * 60)

# "Words Per Minute" (WPM) calculate karte hain
full_train_data['wpm'] = full_train_data['final_word_count'] / full_train_data['total_writing_time_minutes']

# Chhotay essays ko filter out karna taake graph saaf bane
full_train_data_filtered = full_train_data[full_train_data['final_word_count'] > 50].copy()

# --- NAYA HISSA: VISUALIZATION AUR SAVING ---

# 1. Graph Banana
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=full_train_data_filtered, 
    x='wpm', 
    y='score', 
    alpha=0.5
)
plt.title('Words Per Minute (WPM) vs. Score', fontsize=16)
plt.xlabel('Likhne ki Raftaar (Alfaz per Minute)', fontsize=12)
plt.ylabel('Final Score', fontsize=12)
plt.grid(True)
plt.xlim(0, 100)

# 2. Graph ko High-Resolution mein Save Karna
# dpi=300 ek achi resolution hai publications ke liye
# transparent=True se background transparent ho jayega, jo papers ke liye acha hota hai
plt.savefig('figure_wpm_vs_score.png', dpi=300, bbox_inches='tight', transparent=True)
print("Graph 'figure_wpm_vs_score.png' ke naam se save ho gaya hai.")

# 3. Graph ko Screen par Dikhana
plt.show()


# 4. Apne Features wale Table ko CSV File mein Save Karna
# Yeh file Kaggle ke output directory mein save hogi
full_train_data.to_csv('training_features_and_scores.csv')
print("Table 'training_features_and_scores.csv' ke naam se save ho gaya hai.")


# --- Pehla Hissa Waisa Hi Hai ---

# 'action_time' woh waqt hai jab user ki ungliyan keyboard par chal rahi theen.
full_train_data['total_action_time'] # Yeh column pehle se maujood hai

# Ab hum "Pause Time" nikalte hain.
full_train_data['total_writing_time_ms'] # Yeh column bhi pehle se maujood hai
full_train_data['total_pause_time_ms'] = full_train_data['total_writing_time_ms'] - full_train_data['total_action_time']

# Pause Time ka percentage.
epsilon = 1e-6
full_train_data['pause_time_percentage'] = (full_train_data['total_pause_time_ms'] / (full_train_data['total_writing_time_ms'] + epsilon)) * 100


# --- YEH HAI THEEK KI GAYI LINE ---
# Hum data ko naye columns add karne ke BAAD filter kar rahe hain.
full_train_data_filtered = full_train_data[full_train_data['final_word_count'] > 50].copy()


# Chalen dekhte hain ke hamare naye features kaise lag rahe hain
print("Naye Pause Features ke sath Hamara Data:")
print(full_train_data[['final_word_count', 'wpm', 'pause_time_percentage', 'score']].head())


# --- AB GRAPH BANANE WALA HISSA BILKUL THEEK KAM KAREGA ---

plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=full_train_data_filtered, 
    x='pause_time_percentage', 
    y='score', 
    alpha=0.5
)

plt.title('Pause Time Percentage vs. Score', fontsize=16)
plt.xlabel('Kul Waqt Mein Se Sochne Ka Waqt (%)', fontsize=12)
plt.ylabel('Final Score', fontsize=12)
plt.grid(True)

# Graph ko save karna
plt.savefig('figure_pause_percentage_vs_score.png', dpi=300, bbox_inches='tight')
print("\nGraph 'figure_pause_percentage_vs_score.png' save ho gaya hai.")

plt.show()

# Table ko dobara save kar lete hain naye columns ke sath
full_train_data.to_csv('training_features_and_scores_with_pauses.csv')
print("Naye features wala table 'training_features_and_scores_with_pauses.csv' save ho gaya hai.")


# ===================================================================
# STAGE 1: FUNCTION KO DEFINE KARNA
# ===================================================================
# Yeh function har student ke logs data ko lega aur uske pause features nikalega
def get_pause_features(df):
    # Har event ke darmiyan waqt ka farq (gap) nikalte hain
    df['time_shift'] = df['down_time'].shift(1)
    df['pause_duration'] = df['down_time'] - df['time_shift']
    
    # Jahan bhi NaN (Not a Number) hai, usay 0 se badal do.
    df['pause_duration'] = df['pause_duration'].fillna(0)
    
    # Pause ko define karte hain: 500ms se zyada ka gap
    pause_threshold_1 = 500
    df['is_pause'] = (df['pause_duration'] > pause_threshold_1)
    
    # Har student ke liye pause features calculate karna
    pauses = df[df['is_pause']]
    num_pauses = len(pauses)
    avg_pause_duration = pauses['pause_duration'].mean() if num_pauses > 0 else 0
    
    # Bohat lambe pauses (> 5 seconds)
    long_pause_threshold = 5000
    num_long_pauses = len(pauses[pauses['pause_duration'] > long_pause_threshold])
    
    # Natija wapis bhejna
    return pd.Series({
        'num_pauses': num_pauses,
        'avg_pause_duration': avg_pause_duration,
        'num_long_pauses': num_long_pauses
    })

# ===================================================================
# STAGE 2: FEATURES KO CALCULATE KARNA
# ===================================================================
print("Advanced pause features calculate kiye ja rahe hain... Is mein 1-2 minute lag sakte hain.")
# .copy() ka istemal warnings se bachne ke liye zaroori hai
pause_features = train_logs.groupby('id').apply(lambda x: get_pause_features(x.copy()))
print("Features calculate ho gaye!")

# ===================================================================
# STAGE 3: PURANE FEATURES KO HATANA AUR NAYE FEATURES ADD KARNA (ERROR FIX)
# ===================================================================
# Un columns ke naam jo hum add karne ja rahe hain
new_cols = ['num_pauses', 'avg_pause_duration', 'num_long_pauses']

# Check karo ke agar yeh columns pehle se maujood hain to unhen hata do
for col in new_cols:
    if col in full_train_data.columns:
        full_train_data = full_train_data.drop(columns=[col])
        print(f"Purana column '{col}' hata diya gaya hai.")

# Ab naye features ko join karo
full_train_data = full_train_data.join(pause_features)
print("Naye features ko data mein shamil kar diya gaya hai.")

# ===================================================================
# STAGE 4: NATEEJA DEKHNA AUR SAVE KARNA
# ===================================================================
# Naye features ko dekhte hain
print("\nAdvanced Pause Features ke sath Hamara Data:")
print(full_train_data[['num_pauses', 'avg_pause_duration', 'num_long_pauses', 'score']].head())

# Naye features wale table ko save kar lete hain
full_train_data.to_csv('training_features_and_scores_advanced_pauses.csv')
print("\nNaye advanced features wala table 'training_features_and_scores_advanced_pauses.csv' save ho gaya hai.")


# ===================================================================
# STAGE 1: GRAPH KI TAYYARI
# ===================================================================
# 1 row aur 3 columns wale subplots banana
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Filtered data istemal karna (chhotay essays ko hata kar)
# Pehle check kar lete hain ke filtered data mein naye columns hain ya nahi
# Agar nahi hain, to usay dobara banate hain
if 'num_pauses' not in full_train_data_filtered.columns:
    full_train_data_filtered = full_train_data[full_train_data['final_word_count'] > 50].copy()

# ===================================================================
# STAGE 2: TEENO GRAPHS BANANA
# ===================================================================

# Graph 1: Number of Pauses vs. Score
sns.scatterplot(ax=axes[0], data=full_train_data_filtered, x='num_pauses', y='score', alpha=0.3)
axes[0].set_title('Waqfon Ki Tadad (Number of Pauses) vs. Score', fontsize=14)
axes[0].set_xlabel('Kul Waqfay (Pauses)', fontsize=12)
axes[0].set_ylabel('Final Score', fontsize=12)
axes[0].grid(True)

# Graph 2: Average Pause Duration vs. Score
sns.scatterplot(ax=axes[1], data=full_train_data_filtered, x='avg_pause_duration', y='score', alpha=0.3)
axes[1].set_title('Waqfon Ka Ausat Waqt (Avg. Duration) vs. Score', fontsize=14)
axes[1].set_xlabel('Ausat Waqfa (milliseconds)', fontsize=12)
axes[1].set_ylabel('') # Y-axis label ki zaroorat nahi
axes[1].grid(True)
axes[1].set_xlim(0, 15000) # Graph ko saaf rakhne ke liye x-axis limit

# Graph 3: Number of Long Pauses vs. Score
sns.scatterplot(ax=axes[2], data=full_train_data_filtered, x='num_long_pauses', y='score', alpha=0.3)
axes[2].set_title('Lambe Waqfon Ki Tadad (Long Pauses) vs. Score', fontsize=14)
axes[2].set_xlabel('5 Second Se Zyada Ke Waqfay', fontsize=12)
axes[2].set_ylabel('') # Y-axis label ki zaroorat nahi
axes[2].grid(True)

# ===================================================================
# STAGE 3: SAVE KARNA AUR DIKHANA
# ===================================================================
plt.tight_layout() # Subplots ko theek se adjust karna
plt.savefig('figure_advanced_pause_analysis.png', dpi=300)
print("Advanced pause analysis ka graph 'figure_advanced_pause_analysis.png' save ho gaya hai.")
plt.show()


# Machine Learning ke liye zaroori libraries
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np # np.sqrt ke liye import karna zaroori hai

# ===================================================================
# STAGE 1: DATA TAYYAR KARNA
# ===================================================================
# Woh tamam columns jinhen hum model mein istemal karna chahte hain
features_to_use = [
    'total_events', 'unique_activities', 'total_action_time', 'mean_action_time', 
    'final_word_count', 'total_writing_time_ms', 'total_pause_time_ms', 
    'pause_time_percentage', 'wpm', 'num_pauses', 'avg_pause_duration', 'num_long_pauses'
]

# X (hamare features) aur y (hamara target/score) ko alag karna
X = full_train_data[features_to_use]
y = full_train_data['score']

# Apne data ko training aur validation sets mein taqseem karna
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# ===================================================================
# STAGE 2: MODEL TRAIN KARNA
# ===================================================================
print("\nLightGBM model train ho raha hai...")

# LightGBM model ki settings
lgbm_params = {
    'objective': 'regression_l1',
    'metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

# Model ko train karna
model = lgb.LGBMRegressor(**lgbm_params)
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          eval_metric='rmse',
          callbacks=[lgb.early_stopping(100, verbose=True)]) # verbose=True kar diya taake progress nazar aaye

print("Model training mukammal ho gayi!")

# ===================================================================
# STAGE 3: NATEEJA DEKHNA AUR SAVE KARNA
# ===================================================================
# Model ki performance ko validation data par check karna
predictions = model.predict(X_val)
rmse_score = np.sqrt(mean_squared_error(y_val, predictions))

print(f"\nHamare model ka Validation RMSE Score hai: {rmse_score:.4f}")

# --- NAYA HISSA: SCORE KO FILE MEIN SAVE KARNA ---
with open('model_performance.txt', 'w') as f:
    f.write(f"LightGBM Baseline Model RMSE: {rmse_score:.4f}\n")

print("RMSE score 'model_performance.txt' file mein save ho gaya hai.")


# ===================================================================
# STAGE 1: FEATURE IMPORTANCE PLOT BANANA
# ===================================================================
# LightGBM mein feature importance nikalne ka aasan tareeqa hai
lgb.plot_importance(
    model, 
    figsize=(12, 8), 
    importance_type='gain', # 'gain' batata hai ke har feature ne model ki accuracy mein kitna izafa kiya
    title='Feature Importance (LightGBM)'
)

# ===================================================================
# STAGE 2: PLOT KO SAVE KARNA AUR DIKHANA
# ===================================================================
plt.tight_layout()
plt.savefig('figure_feature_importance.png', dpi=300)
print("Feature Importance ka graph 'figure_feature_importance.png' save ho gaya hai.")
plt.show()

# ===================================================================
# STAGE 3: IMPORTANCE VALUES KO TABLE MEIN DEKHNA (EXTRA DETAIL)
# ===================================================================
# Features aur unki importance scores ka ek DataFrame banana
feature_importance_df = pd.DataFrame({
    'feature': features_to_use,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeatures ki Ahmiyat (Sab se Zyada se Kam tak):")
print(feature_importance_df)

# Is table ko bhi save kar lete hain
feature_importance_df.to_csv('feature_importance_scores.csv', index=False)
print("\nFeature importance ka table 'feature_importance_scores.csv' save ho gaya hai.")


# ===================================================================
# STAGE 1: TEST DATA KO LOAD AUR PROCESS KARNA
# ===================================================================
print("Test data ko process kiya ja raha hai...")

# Test logs ko load karna
test_logs = pd.read_csv(TEST_LOGS_PATH)

# Test data par wohi tamam features banana jo hum ne training data par banaye thay
# 1. Bunyadi features
test_features = test_logs.groupby('id').agg({
    'event_id': ['count'], 'activity': ['nunique'], 'action_time': ['sum', 'mean'], 'word_count': ['max']
})
test_features.columns = ['total_events', 'unique_activities', 'total_action_time', 'mean_action_time', 'final_word_count']

# 2. Time aur WPM features
total_time_test = test_logs.groupby('id')['down_time'].max()
test_features['total_writing_time_ms'] = total_time_test
test_features['total_writing_time_minutes'] = test_features['total_writing_time_ms'] / (1000 * 60)
test_features['wpm'] = test_features['final_word_count'] / test_features['total_writing_time_minutes']

# 3. Pause percentage features
test_features['total_pause_time_ms'] = test_features['total_writing_time_ms'] - test_features['total_action_time']
epsilon = 1e-6
test_features['pause_time_percentage'] = (test_features['total_pause_time_ms'] / (test_features['total_writing_time_ms'] + epsilon)) * 100

# 4. Advanced pause features
test_pause_features = test_logs.groupby('id').apply(lambda x: get_pause_features(x.copy()))
test_features = test_features.join(test_pause_features)

# NaN values ko handle karna (agar koi hon to)
test_features = test_features.fillna(0)

print("Test data ke features tayyar hain!")

# ===================================================================
# STAGE 2: PREDICTIONS KARNA
# ===================================================================
# Sirf wohi features select karna jo model ne seekhe hain
X_test = test_features[features_to_use]

print("Test data par predictions ki ja rahi hain...")
test_predictions = model.predict(X_test)

# ===================================================================
# STAGE 3: SUBMISSION FILE BANANA
# ===================================================================
# Submission file ke liye DataFrame banana
submission_df = pd.DataFrame({
    'id': X_test.index,
    'score': test_predictions
})

# Scores ko 0.5 ke steps mein round karna (competition ke format ke mutabiq)
submission_df['score'] = submission_df['score'].round(1) # Pehle 1 decimal tak round
submission_df['score'] = (submission_df['score'] * 2).round() / 2 # Phir 0.5 ke qareeb tareen multiple tak

# File ko save karna
submission_df.to_csv('submission.csv', index=False)

print("\nKamyabi! Aap ki submission file 'submission.csv' tayyar hai.")
print("Aap isay Kaggle ke 'Output' section se download karke submit kar sakte hain.")
print(submission_df.head())


# ===================================================================
# STAGE 1: DELETION EVENTS KO ALAG KARNA
# ===================================================================
print("Revision features calculate kiye ja rahe hain...")

# Sirf woh events select karna jahan kuch delete hua hai
# .copy() istemal karna ek achi practice hai
deletions = train_logs[train_logs['activity'] == 'Remove/Cut'].copy()

# Har deletion event mein kitne characters delete hue, yeh calculate karna
# 'text_change' column mein delete kiya gaya character hota hai, uski length nikalte hain
deletions['deleted_chars_count'] = deletions['text_change'].str.len()

# ===================================================================
# STAGE 2: HAR STUDENT KE LIYE FEATURES CALCULATE KARNA
# ===================================================================
# Har student ne kul kitni baar delete kiya?
num_deletions = deletions.groupby('id').size()

# Har student ne kul kitne characters delete kiye?
total_deleted_chars = deletions.groupby('id')['deleted_chars_count'].sum()

# ===================================================================
# STAGE 3: NAYE FEATURES KO MAIN DATA MEIN SHAMIL KARNA
# ===================================================================
# Naye features ko apne main DataFrame mein join karna
full_train_data['num_deletions'] = num_deletions
full_train_data['total_deleted_chars'] = total_deleted_chars

# Jin students ne kabhi kuch delete nahi kiya, unke liye NaN aayega. Usay 0 se bhar do.
full_train_data.fillna(0, inplace=True)

# Ek bohat ahem "Revision Density" feature banate hain
# Kul likhe gaye alfaz ke muqablay mein kitne characters delete hue
# Is se pata chalega ke kon bohat zyada editing karta hai
# total_events istemal kar rahe hain taake 0 se division na ho
full_train_data['revision_density'] = full_train_data['total_deleted_chars'] / full_train_data['total_events']

print("Revision features tayyar hain!")

# ===================================================================
# STAGE 4: NAYE FEATURES KO DEKHNA AUR SAVE KARNA
# ===================================================================
print("\nNaye Revision Features ke sath Hamara Data:")
print(full_train_data[['num_deletions', 'total_deleted_chars', 'revision_density', 'score']].head())

# Updated table ko save karna
full_train_data.to_csv('training_features_with_revisions.csv')
print("\nRevision features wala naya table 'training_features_with_revisions.csv' save ho gaya hai.")


# Machine Learning ke liye zaroori libraries (agar restart kiya ho to dobara import karen)
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np

# ===================================================================
# STAGE 1: DATA TAYYAR KARNA (NAYE FEATURES KE SATH)
# ===================================================================
# Ab humara features_to_use list barh gaya hai!
features_to_use_updated = [
    'total_events', 'unique_activities', 'total_action_time', 'mean_action_time', 
    'final_word_count', 'total_writing_time_ms', 'total_pause_time_ms', 
    'pause_time_percentage', 'wpm', 'num_pauses', 'avg_pause_duration', 'num_long_pauses',
    # --- YEH HAIN NAYE FEATURES ---
    'num_deletions', 'total_deleted_chars', 'revision_density' 
]

# X (hamare features) aur y (hamara target/score) ko alag karna
X = full_train_data[features_to_use_updated]
y = full_train_data['score']

# Apne data ko training aur validation sets mein taqseem karna
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Updated Training data shape: {X_train.shape}")
print(f"Updated Validation data shape: {X_val.shape}")

# ===================================================================
# STAGE 2: MODEL DOBARA TRAIN KARNA
# ===================================================================
print("\nLightGBM model naye features ke sath dobara train ho raha hai...")

# LightGBM model ki settings wahi rakhte hain
lgbm_params = {
    'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 1000,
    'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'bagging_freq': 1, 'verbose': -1, 'n_jobs': -1, 'seed': 42
}

# Model ko train karna
model_v2 = lgb.LGBMRegressor(**lgbm_params)
model_v2.fit(X_train, y_train,
             eval_set=[(X_val, y_val)],
             eval_metric='rmse',
             callbacks=[lgb.early_stopping(100, verbose=False)])

print("Model V2 ki training mukammal ho gayi!")

# ===================================================================
# STAGE 3: NAYE NATEEJE KO DEKHNA
# ===================================================================
# Model ki performance ko validation data par check karna
predictions_v2 = model_v2.predict(X_val)
rmse_score_v2 = np.sqrt(mean_squared_error(y_val, predictions_v2))

print(f"\nHamare purane model ka score tha: 0.6373")
print(f"Hamare naye UPGRADED model ka Validation RMSE Score hai: {rmse_score_v2:.4f}")

# Naye score ko file mein append karna
with open('model_performance.txt', 'a') as f: # 'a' for append
    f.write(f"LightGBM with Revision Features RMSE: {rmse_score_v2:.4f}\n")

print("Naya RMSE score 'model_performance.txt' file mein add ho gaya hai.")


# ===================================================================
# STAGE 1: ZAROORI LIBRARIES AUR DATA TAYYAR KARNA
# ===================================================================
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gc # Garbage Collector for memory management

print("LSTM ke liye data tayyar kiya ja raha hai... Is mein waqt lagega.")

# Har student ke tamam 'down_event' (konsi key dabayi) ko ek list mein jama karna
# Hum har student ke events ko ek jumlay (sentence) jaisa bana rahe hain
sequence_data = train_logs.groupby('id')['down_event'].apply(lambda x: ' '.join(x))

print("Har student ke liye event sequence ban gaya hai.")

# ===================================================================
# STAGE 2: EVENTS KO NUMBERS MEIN BADALNA (TOKENIZATION)
# ===================================================================
# Machine learning models text nahi, numbers samajhte hain.
# Hum har unique event (jaise 'q', 'Backspace') ko ek unique number denge.
tokenizer = Tokenizer(char_level=False)
tokenizer.fit_on_texts(sequence_data)

# Events ke sequence ko numbers ke sequence mein badalna
sequences_as_numbers = tokenizer.texts_to_sequences(sequence_data)

# ===================================================================
# STAGE 3: SEQUENCE KO EK JAISI LENGTH KA BANANA (PADDING)
# ===================================================================
# LSTM ko har input ek hi length ka chahiye.
# Hum ek makhsoos length (e.g., 2000 events) rakhenge.
# Lambe sequences ko kaat denge aur chhotay sequences ke aakhir mein 0 laga denge.
MAX_SEQUENCE_LENGTH = 2000
padded_sequences = pad_sequences(sequences_as_numbers, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

print(f"Tamam sequences ko {MAX_SEQUENCE_LENGTH} ki length mein pad kar diya gaya hai.")

# ===================================================================
# STAGE 4: NATEEJA DEKHNA AUR SAVE KARNA
# ===================================================================
# Padded data ko scores ke sath milana
lstm_data = pd.DataFrame(padded_sequences, index=sequence_data.index)
lstm_data['score'] = train_scores.set_index('id')['score']

# Memory azaad karna
del sequence_data, sequences_as_numbers, padded_sequences
gc.collect()

print("\nLSTM ke liye data tayyar hai!")
print("Padded Sequences ka Shape:", lstm_data.drop('score', axis=1).shape)
print(lstm_data.head())


# ===================================================================
# STAGE 5: PROCESSED LSTM DATA KO SAVE KARNA
# ===================================================================
print("Tayyar shuda LSTM data ko Parquet format mein save kiya ja raha hai...")

# Data ko save karna. index=True zaroori hai taake 'id' bhi save ho.
lstm_data.to_parquet('lstm_processed_data.parquet', index=True)

print("Data 'lstm_processed_data.parquet' ke naam se kamyabi se save ho gaya hai.")
print("Aage ja kar, hum is file ko seedha load kar sakte hain aur 10 minute bacha sakte hain.")


# ===================================================================
# STAGE 0: IMPORT LIBRARIES AND ENSURE ALL DATA IS FRESH
# This cell is a complete, self-contained pipeline.
# ===================================================================
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import gc

print("Starting the entire pipeline from scratch to ensure data integrity...")

# --- 1. Load Raw Data ---
BASE_DIR = '/kaggle/input/linking-writing-processes-to-writing-quality/'
train_logs = pd.read_csv(BASE_DIR + 'train_logs.csv')
train_scores = pd.read_csv(BASE_DIR + 'train_scores.csv')

# --- 2. Build the Complete Feature Engineering Pipeline ---
print("Step 1/5: Creating basic features...")
full_train_data = train_logs.groupby('id').agg({
    'event_id': ['count'], 'activity': ['nunique'], 'action_time': ['sum', 'mean'], 'word_count': ['max']
})
full_train_data.columns = ['total_events', 'unique_activities', 'total_action_time', 'mean_action_time', 'final_word_count']
full_train_data = full_train_data.join(train_scores.set_index('id'))

print("Step 2/5: Creating pause and WPM features...")
total_time = train_logs.groupby('id')['down_time'].max()
full_train_data['total_writing_time_ms'] = total_time
full_train_data['total_pause_time_ms'] = full_train_data['total_writing_time_ms'] - full_train_data['total_action_time']
epsilon = 1e-6
full_train_data['pause_time_percentage'] = (full_train_data['total_pause_time_ms'] / (full_train_data['total_writing_time_ms'] + epsilon)) * 100
full_train_data['total_writing_time_minutes'] = full_train_data['total_writing_time_ms'] / (1000 * 60)
full_train_data['wpm'] = full_train_data['final_word_count'] / (full_train_data['total_writing_time_minutes'] + epsilon)

def get_pause_features(df):
    df['time_shift'] = df['down_time'].shift(1)
    df['pause_duration'] = df['down_time'] - df['time_shift']
    df['pause_duration'] = df['pause_duration'].fillna(0)
    pauses = df[df['pause_duration'] > 500]
    num_pauses = len(pauses)
    avg_pause_duration = pauses['pause_duration'].mean() if num_pauses > 0 else 0
    num_long_pauses = len(pauses[pauses['pause_duration'] > 5000])
    return pd.Series({'num_pauses': num_pauses, 'avg_pause_duration': avg_pause_duration, 'num_long_pauses': num_long_pauses})

print("Step 3/5: Applying advanced pause features...")
pause_features = train_logs.groupby('id').apply(lambda x: get_pause_features(x.copy()))
full_train_data = full_train_data.join(pause_features)

print("Step 4/5: Creating revision and rethinking features...")
revisions = train_logs[train_logs['activity'] == 'Remove/Cut'].copy()
revisions['revision_length'] = revisions['text_change'].str.len()
full_train_data['num_deletions'] = revisions.groupby('id').size()
full_train_data['total_deleted_chars'] = revisions.groupby('id')['revision_length'].sum()
full_train_data['revision_density'] = full_train_data['total_deleted_chars'] / (full_train_data['total_events'] + epsilon)
revisions['is_rethinking'] = revisions['revision_length'] > 10
full_train_data['rethinking_events_count'] = revisions.groupby('id')['is_rethinking'].sum()

# --- NEW: Interaction Features ---
print("Step 5/5: Creating NEW Interaction Features...")
full_train_data['word_rethink_ratio'] = full_train_data['final_word_count'] / (full_train_data['rethinking_events_count'] + epsilon)
full_train_data['pause_per_word'] = full_train_data['num_pauses'] / (full_train_data['final_word_count'] + epsilon)
full_train_data['strategic_pause_metric'] = full_train_data['avg_pause_duration'] * full_train_data['wpm']

full_train_data.fillna(0, inplace=True)
print("All features, including Interaction Features, have been created!")
gc.collect()

# --- 3. Define Model and Parameters ---
# Champion features PLUS our new interaction features
features_interaction = [
    'total_events', 'unique_activities', 'total_action_time', 'mean_action_time', 
    'final_word_count', 'total_writing_time_ms', 'total_pause_time_ms', 
    'pause_time_percentage', 'wpm', 'num_pauses', 'avg_pause_duration', 'num_long_pauses',
    'num_deletions', 'total_deleted_chars', 'revision_density', 'rethinking_events_count',
    'word_rethink_ratio', 'pause_per_word', 'strategic_pause_metric'
]

X_interaction = full_train_data[features_interaction]
y_interaction = full_train_data['score']

best_params = {
    'learning_rate': 0.0228, 'feature_fraction': 0.611, 'bagging_fraction': 0.876,
    'bagging_freq': 3, 'lambda_l1': 9.933, 'lambda_l2': 1.388e-06, 'num_leaves': 46,
    'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 2000,
    'verbose': -1, 'n_jobs': -1, 'seed': 42
}

# ===================================================================
# STAGE 1: TRAIN THE NEW INTERACTION MODEL
# ===================================================================
X_train_int, X_val_int, y_train_int, y_val_int = train_test_split(X_interaction, y_interaction, test_size=0.2, random_state=42)

print("\nTraining model with new Interaction Features...")
model_interaction = lgb.LGBMRegressor(**best_params) 
model_interaction.fit(X_train_int, y_train_int,
                      eval_set=[(X_val_int, y_val_int)],
                      eval_metric='rmse',
                      callbacks=[lgb.early_stopping(100, verbose=False)])

rmse_score_interaction = np.sqrt(mean_squared_error(y_val_int, model_interaction.predict(X_val_int)))

print(f"\nPrevious best single-split score was: ~0.6205")
print(f"Our NEW INTERACTION model's Validation RMSE Score is: {rmse_score_interaction:.4f}")

with open('model_performance.txt', 'a') as f:
    f.write(f"LightGBM with Interaction Features RMSE: {rmse_score_interaction:.4f}\n")

print("Interaction model score has been saved to 'model_performance.txt'.")


# ===================================================================
# FINAL, SELF-CONTAINED CELL FOR ERROR ANALYSIS AND PLOTTING
# ===================================================================
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting robust error analysis pipeline...")

# --- Check if the complete `full_train_data` exists; if not, recreate it ---
# This ensures that even if you restart, this cell has what it needs.
required_cols = ['rethinking_events_count']
if not all(col in full_train_data.columns for col in required_cols):
    print("Required features not found. It's recommended to run the main pipeline cell first.")
    # You would ideally run the big feature creation pipeline here, but for now, we assume it has been run.
    # If this fails, the user must run the big "All-in-One Feature Creation" cell first.

# --- Define Champion Model and Parameters ---
features_champion = [
    'total_events', 'unique_activities', 'total_action_time', 'mean_action_time', 
    'final_word_count', 'total_writing_time_ms', 'total_pause_time_ms', 
    'pause_time_percentage', 'wpm', 'num_pauses', 'avg_pause_duration', 'num_long_pauses',
    'num_deletions', 'total_deleted_chars', 'revision_density', 'rethinking_events_count'
]
X_champion = full_train_data[features_champion]
y_champion = full_train_data['score']

best_params = {
    'learning_rate': 0.0228, 'feature_fraction': 0.611, 'bagging_fraction': 0.876,
    'bagging_freq': 3, 'lambda_l1': 9.933, 'lambda_l2': 1.388e-06, 'num_leaves': 46,
    'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 2000,
    'verbose': -1, 'n_jobs': -1, 'seed': 42
}

# ===================================================================
# STAGE 1: CROSS-VALIDATION TO GET PREDICTIONS
# ===================================================================
NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
oof_preds_champion_cv = np.zeros(len(full_train_data))

print(f"\nRunning {NFOLDS}-Fold Cross-Validation to generate predictions for analysis...")

for fold, (train_index, val_index) in enumerate(folds.split(X_champion, y_champion)):
    print(f"  - Running Fold {fold+1}/{NFOLDS}...")
    X_train_cv, X_val_cv = X_champion.iloc[train_index], X_champion.iloc[val_index]
    y_train_cv, y_val_cv = y_champion.iloc[train_index], y_champion.iloc[val_index]
    
    model_cv = lgb.LGBMRegressor(**best_params)
    model_cv.fit(X_train_cv, y_train_cv, eval_set=[(X_val_cv, y_val_cv)], eval_metric='rmse', callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_preds_champion_cv[val_index] = model_cv.predict(X_val_cv)

print("Cross-Validation complete.")

# ===================================================================
# STAGE 2: ERROR CALCULATION AND ANALYSIS
# ===================================================================
# These columns are now guaranteed to be created and used in the same cell.
full_train_data['oof_prediction'] = oof_preds_champion_cv
full_train_data['error'] = full_train_data['oof_prediction'] - full_train_data['score']
full_train_data['abs_error'] = abs(full_train_data['error'])

print("\nError calculation complete.")

worst_predictions = full_train_data.sort_values('abs_error', ascending=False).head(10)

print("\nModel's 10 Biggest Errors (Highest Absolute Error):")
display_features = ['score', 'oof_prediction', 'error', 'final_word_count', 'avg_pause_duration', 'num_pauses', 'revision_density', 'rethinking_events_count']
print(worst_predictions[display_features])

# --- SAVE THE TABLE ---
worst_predictions[display_features].to_csv('worst_predictions_analysis.csv')
print("\nSuccess! Worst predictions table saved as 'worst_predictions_analysis.csv'.")


# ===================================================================
# STAGE 3: VISUALIZATION AND SAVING
# ===================================================================
print("\nGenerating and saving the Error Distribution Plot...")
plt.figure(figsize=(12, 6))
sns.histplot(full_train_data['error'], bins=50, kde=True)
plt.title('Prediction Error Distribution of Champion Model', fontsize=16)
plt.xlabel('Predicted Score - Actual Score (Error)', fontsize=12)
plt.ylabel('Number of Students', fontsize=12)
plt.grid(True)

# --- SAVE THE PLOT ---
plt.savefig('figure_champion_error_distribution.png', dpi=300, bbox_inches='tight')
print("Success! Error distribution plot saved as 'figure_champion_error_distribution.png'.")

plt.show()


# ===================================================================
# STAGE 1: NAYE INTERACTION FEATURES BANANA
# ===================================================================
print("Advanced Interaction Features banaye ja rahe hain...")
epsilon = 1e-6 # To avoid division by zero

# --- Target #1: Catching "Long but low-quality" writers (Over-prediction cases) ---

# Feature 1: 'word_rethink_ratio'
# IDEA: A high word count is less impressive if the writer never did any major rethinking.
# This feature "punishes" a high word count if rethinking_events_count is low.
full_train_data['word_rethink_ratio'] = full_train_data['final_word_count'] / (full_train_data['rethinking_events_count'] + epsilon)

# Feature 2: 'pause_per_word'
# IDEA: Writers who pause very frequently for every word they write might be hesitant.
full_train_data['pause_per_word'] = full_train_data['num_pauses'] / (full_train_data['final_word_count'] + epsilon)


# --- Target #2: Identifying "Hyper-efficient" writers (Under-prediction cases) ---

# Feature 3: 'strategic_pause_metric'
# IDEA: A truly efficient writer might have long average pauses (strategic thinking) AND a high WPM.
# We multiply them to create a metric for "productive thinking speed".
full_train_data['strategic_pause_metric'] = full_train_data['avg_pause_duration'] * full_train_data['wpm']


# ===================================================================
# STAGE 2: NAYE FEATURES KE SATH MODEL KO AAKHRI BAAR UPGRADE KARNA
# ===================================================================
print("Naye Interaction Features tayyar hain!")

# Features ki aakhri list ko update karna
features_interaction = features_champion + ['word_rethink_ratio', 'pause_per_word', 'strategic_pause_metric']

X_interaction = full_train_data[features_interaction]
y_interaction = full_train_data['score']

X_train_int, X_val_int, y_train_int, y_val_int = train_test_split(X_interaction, y_interaction, test_size=0.2, random_state=42)

print("\nModel ko Interaction Features ke sath train kiya ja raha hai...")

# Hum apne behtareen (Optimized) parameters ka hi istemal karenge
model_interaction = lgb.LGBMRegressor(**best_params) 
model_interaction.fit(X_train_int, y_train_int,
                      eval_set=[(X_val_int, y_val_int)],
                      eval_metric='rmse',
                      callbacks=[lgb.early_stopping(100, verbose=False)])

# ===================================================================
# STAGE 3: NATEEJA DEKHNA - THE FINAL PUSH
# ===================================================================
rmse_score_interaction = np.sqrt(mean_squared_error(y_val_int, model_interaction.predict(X_val_int)))

print(f"\nHamare pichle behtareen (Champion) model ka single-split score tha: ~0.6205")
print(f"Hamare naye INTERACTION model ka Validation RMSE Score hai: {rmse_score_interaction:.4f}")

# Naye score ko file mein append karna
with open('model_performance.txt', 'a') as f:
    f.write(f"LightGBM with Interaction Features RMSE: {rmse_score_interaction:.4f}\n")

print("Final Interaction model score 'model_performance.txt' file mein add ho gaya hai.")


# ===================================================================
# STAGE 1: CREATE ADVANCED INTERACTION FEATURES
# ===================================================================
print("Creating targeted Interaction Features based on our Error Analysis...")
epsilon = 1e-6 # To avoid division by zero

# --- Target #1 Feature: Punish long essays with no deep revisions ---
# Helps catch the "Verbose but Low-Quality" writers
full_train_data['word_rethink_ratio'] = full_train_data['final_word_count'] / (full_train_data['rethinking_events_count'] + epsilon)

# --- Target #2 Feature: Identify writers who are both fast and thoughtful ---
# Helps catch the "Hyper-Efficient" writers
full_train_data['productive_fluency'] = full_train_data['avg_pause_duration'] * full_train_data['wpm']

# --- General Interaction Feature ---
# Normalizes pause frequency by the amount of text produced.
full_train_data['pauses_per_100_words'] = full_train_data['num_pauses'] / (full_train_data['final_word_count'] / 100 + epsilon)


print("New Interaction Features created successfully!")
print(full_train_data[['word_rethink_ratio', 'productive_fluency', 'pauses_per_100_words']].head())


# ===================================================================
# STAGE 2: TRAIN AND EVALUATE THE FINAL MODEL WITH NEW FEATURES
# ===================================================================
# Define the final, most advanced feature set
features_final_advanced = features_champion + ['word_rethink_ratio', 'productive_fluency', 'pauses_per_100_words']

X_advanced = full_train_data[features_final_advanced]
y_advanced = full_train_data['score']

# Use a single train/validation split to quickly test the impact of new features
X_train_adv, X_val_adv, y_train_adv, y_val_adv = train_test_split(X_advanced, y_advanced, test_size=0.2, random_state=42)

print("\nTraining the ultimate model with all advanced features...")

# We will use our best-found hyperparameters
model_advanced = lgb.LGBMRegressor(**best_params) 
model_advanced.fit(X_train_adv, y_train_adv,
                   eval_set=[(X_val_adv, y_val_adv)],
                   eval_metric='rmse',
                   callbacks=[lgb.early_stopping(100, verbose=False)])

# ===================================================================
# STAGE 3: THE FINAL RESULT
# ===================================================================
rmse_score_advanced = np.sqrt(mean_squared_error(y_val_adv, model_advanced.predict(X_val_adv)))

print(f"\nOur previous best single-split score (Champion Model) was: ~0.6205")
print(f"Our NEW and MOST ADVANCED model's Validation RMSE Score is: {rmse_score_advanced:.4f}")

# Append the final score to our performance log
with open('model_performance.txt', 'a') as f:
    f.write(f"LightGBM with Advanced Interaction Features RMSE: {rmse_score_advanced:.4f}\n")

print("Final advanced model score has been saved to 'model_performance.txt'.")


# ===================================================================
# STAGE 0: IMPORT LIBRARIES AND SETUP
# This is the final, fully self-contained pipeline for the ensemble model.
# ===================================================================
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import gc

print("--- INITIALIZING THE ULTIMATE, SELF-CONTAINED ENSEMBLE PIPELINE ---")
print("This will take a significant amount of time, but will ensure correctness.")

# --- 1. Load Raw Data ---
print("\nStep 1: Loading Raw Data...")
BASE_DIR = '/kaggle/input/linking-writing-processes-to-writing-quality/'
train_logs = pd.read_csv(BASE_DIR + 'train_logs.csv')
train_scores = pd.read_csv(BASE_DIR + 'train_scores.csv')
test_logs = pd.read_csv(BASE_DIR + 'test_logs.csv')


# --- 2. Define the Master Feature Engineering Function ---
# This function can process both train and test logs to ensure consistency.
def create_features(logs_df):
    print("  - Creating basic features...")
    features_df = logs_df.groupby('id').agg({
        'event_id': ['count'], 'activity': ['nunique'], 'action_time': ['sum', 'mean'], 'word_count': ['max']
    })
    features_df.columns = ['total_events', 'unique_activities', 'total_action_time', 'mean_action_time', 'final_word_count']

    print("  - Creating pause and WPM features...")
    total_time = logs_df.groupby('id')['down_time'].max()
    features_df['total_writing_time_ms'] = total_time
    features_df['total_pause_time_ms'] = features_df['total_writing_time_ms'] - features_df['total_action_time']
    epsilon = 1e-6
    features_df['pause_time_percentage'] = (features_df['total_pause_time_ms'] / (features_df['total_writing_time_ms'] + epsilon)) * 100
    features_df['total_writing_time_minutes'] = features_df['total_writing_time_ms'] / (1000 * 60)
    features_df['wpm'] = features_df['final_word_count'] / (features_df['total_writing_time_minutes'] + epsilon)

    def get_pause_features(df):
        df['time_shift'] = df['down_time'].shift(1)
        df['pause_duration'] = df['down_time'] - df['time_shift']
        df['pause_duration'] = df['pause_duration'].fillna(0)
        pauses = df[df['pause_duration'] > 500]
        num_pauses = len(pauses)
        avg_pause_duration = pauses['pause_duration'].mean() if num_pauses > 0 else 0
        num_long_pauses = len(pauses[pauses['pause_duration'] > 5000])
        return pd.Series({'num_pauses': num_pauses, 'avg_pause_duration': avg_pause_duration, 'num_long_pauses': num_long_pauses})
    
    print("  - Applying advanced pause features...")
    pause_features = logs_df.groupby('id').apply(lambda x: get_pause_features(x.copy()))
    features_df = features_df.join(pause_features)

    print("  - Creating revision and rethinking features...")
    revisions = logs_df[logs_df['activity'] == 'Remove/Cut'].copy()
    revisions['revision_length'] = revisions['text_change'].str.len()
    features_df['num_deletions'] = revisions.groupby('id').size()
    features_df['total_deleted_chars'] = revisions.groupby('id')['revision_length'].sum()
    features_df['revision_density'] = features_df['total_deleted_chars'] / (features_df['total_events'] + epsilon)
    revisions['is_rethinking'] = revisions['revision_length'] > 10
    features_df['rethinking_events_count'] = revisions.groupby('id')['is_rethinking'].sum()
    
    features_df.fillna(0, inplace=True)
    return features_df

# --- 3. Process Train and Test Data ---
print("\nStep 2: Processing Training Data...")
full_train_data = create_features(train_logs)
full_train_data = full_train_data.join(train_scores.set_index('id'))

print("\nStep 3: Processing Test Data...")
test_features = create_features(test_logs)

print("\nAll data processing complete!")
gc.collect()

# --- 4. Define Final Model Setup ---
features_champion = [
    'total_events', 'unique_activities', 'total_action_time', 'mean_action_time', 
    'final_word_count', 'total_writing_time_ms', 'total_pause_time_ms', 
    'pause_time_percentage', 'wpm', 'num_pauses', 'avg_pause_duration', 'num_long_pauses',
    'num_deletions', 'total_deleted_chars', 'revision_density', 'rethinking_events_count'
]
X = full_train_data[features_champion]
y = full_train_data['score']
X_test = test_features[features_champion]

best_lgbm_params = {
    'learning_rate': 0.0228, 'feature_fraction': 0.611, 'bagging_fraction': 0.876,
    'bagging_freq': 3, 'lambda_l1': 9.933, 'lambda_l2': 1.388e-06, 'num_leaves': 46,
    'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 2000,
    'verbose': -1, 'n_jobs': -1, 'seed': 42
}

NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

oof_preds_lgbm = np.zeros(len(X))
test_preds_lgbm = np.zeros(len(X_test))
oof_preds_xgb = np.zeros(len(X))
test_preds_xgb = np.zeros(len(X_test))
oof_preds_cat = np.zeros(len(X))
test_preds_cat = np.zeros(len(X_test))

# ===================================================================
# STAGE 5: RUN THE FULL ENSEMBLE TRAINING
# This will take a long time.
# ===================================================================

# --- LightGBM ---
print("\n--- Training LightGBM Model ---")
for fold, (train_index, val_index) in enumerate(folds.split(X, y)):
    print(f"  - Fold {fold+1}/{NFOLDS}...")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    model = lgb.LGBMRegressor(**best_lgbm_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_preds_lgbm[val_index] = model.predict(X_val)
    test_preds_lgbm += model.predict(X_test) / NFOLDS

# --- XGBoost ---
print("\n--- Training XGBoost Model ---")
xgb_params = {'objective': 'reg:squarederror', 'n_estimators': 2000, 'learning_rate': 0.05, 'seed': 42, 'tree_method': 'hist'}
for fold, (train_index, val_index) in enumerate(folds.split(X, y)):
    print(f"  - Fold {fold+1}/{NFOLDS}...")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    oof_preds_xgb[val_index] = model.predict(X_val)
    test_preds_xgb += model.predict(X_test) / NFOLDS
    
# --- CatBoost ---
print("\n--- Training CatBoost Model ---")
cat_params = {'iterations': 2000, 'learning_rate': 0.05, 'loss_function': 'RMSE', 'random_seed': 42}
for fold, (train_index, val_index) in enumerate(folds.split(X, y)):
    print(f"  - Fold {fold+1}/{NFOLDS}...")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    model = cb.CatBoostRegressor(**cat_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    oof_preds_cat[val_index] = model.predict(X_val)
    test_preds_cat += model.predict(X_test) / NFOLDS

# ===================================================================
# STAGE 6: EVALUATE, BLEND, AND SUBMIT
# ===================================================================
print("\n--- Evaluating and Blending Models ---")
rmse_lgbm = np.sqrt(mean_squared_error(y, oof_preds_lgbm))
rmse_xgb = np.sqrt(mean_squared_error(y, oof_preds_xgb))
rmse_cat = np.sqrt(mean_squared_error(y, oof_preds_cat))

print(f"LightGBM CV Score: {rmse_lgbm:.4f}")
print(f"XGBoost CV Score:  {rmse_xgb:.4f}")
print(f"CatBoost CV Score: {rmse_cat:.4f}")

oof_preds_ensemble = (oof_preds_lgbm + oof_preds_xgb + oof_preds_cat) / 3
rmse_ensemble = np.sqrt(mean_squared_error(y, oof_preds_ensemble))
print(f"\nSimple Average ENSEMBLE CV Score: {rmse_ensemble:.4f}")

test_preds_ensemble = (test_preds_lgbm + test_preds_xgb + test_preds_cat) / 3
submission_ensemble = pd.DataFrame({'id': X_test.index, 'score': test_preds_ensemble})
submission_ensemble['score'] = (submission_ensemble['score'] * 2).round() / 2
submission_ensemble.to_csv('submission_ENSEMBLE_final.csv', index=False)

print("\n[SUCCESS] Your final, most powerful ENSEMBLE submission file 'submission_ENSEMBLE_final.csv' is ready!")
print(submission_ensemble.head())


# ===================================================================
# STAGE 0: IMPORT LIBRARIES AND SETUP
# This is the final, fully self-contained pipeline using STRATIFIED K-FOLD.
# ===================================================================
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold # <-- THE KEY CHANGE
from sklearn.metrics import mean_squared_error
import gc

print("--- INITIALIZING THE STATE-OF-THE-ART STRATIFIED ENSEMBLE PIPELINE ---")

# --- 1. Load Raw Data ---
print("\nStep 1: Loading Raw Data...")
BASE_DIR = '/kaggle/input/linking-writing-processes-to-writing-quality/'
train_logs = pd.read_csv(BASE_DIR + 'train_logs.csv')
train_scores = pd.read_csv(BASE_DIR + 'train_scores.csv')
test_logs = pd.read_csv(BASE_DIR + 'test_logs.csv')

# --- 2. Define the Master Feature Engineering Function ---
def create_features(logs_df):
    # ... (This function is identical to the previous version, so we can omit its text for brevity)
    print("  - Creating basic features...")
    features_df = logs_df.groupby('id').agg({'event_id': ['count'], 'activity': ['nunique'], 'action_time': ['sum', 'mean'], 'word_count': ['max']})
    features_df.columns = ['total_events', 'unique_activities', 'total_action_time', 'mean_action_time', 'final_word_count']
    print("  - Creating pause and WPM features...")
    total_time = logs_df.groupby('id')['down_time'].max()
    features_df['total_writing_time_ms'] = total_time
    features_df['total_pause_time_ms'] = features_df['total_writing_time_ms'] - features_df['total_action_time']
    epsilon = 1e-6
    features_df['pause_time_percentage'] = (features_df['total_pause_time_ms'] / (features_df['total_writing_time_ms'] + epsilon)) * 100
    features_df['total_writing_time_minutes'] = features_df['total_writing_time_ms'] / (1000 * 60)
    features_df['wpm'] = features_df['final_word_count'] / (features_df['total_writing_time_minutes'] + epsilon)
    def get_pause_features(df):
        df['time_shift'] = df['down_time'].shift(1); df['pause_duration'] = df['down_time'] - df['time_shift']; df['pause_duration'] = df['pause_duration'].fillna(0)
        pauses = df[df['pause_duration'] > 500]; num_pauses = len(pauses); avg_pause_duration = pauses['pause_duration'].mean() if num_pauses > 0 else 0
        num_long_pauses = len(pauses[pauses['pause_duration'] > 5000])
        return pd.Series({'num_pauses': num_pauses, 'avg_pause_duration': avg_pause_duration, 'num_long_pauses': num_long_pauses})
    print("  - Applying advanced pause features...")
    pause_features = logs_df.groupby('id').apply(lambda x: get_pause_features(x.copy()))
    features_df = features_df.join(pause_features)
    print("  - Creating revision and rethinking features...")
    revisions = logs_df[logs_df['activity'] == 'Remove/Cut'].copy()
    revisions['revision_length'] = revisions['text_change'].str.len()
    features_df['num_deletions'] = revisions.groupby('id').size()
    features_df['total_deleted_chars'] = revisions.groupby('id')['revision_length'].sum()
    features_df['revision_density'] = features_df['total_deleted_chars'] / (features_df['total_events'] + epsilon)
    revisions['is_rethinking'] = revisions['revision_length'] > 10
    features_df['rethinking_events_count'] = revisions.groupby('id')['is_rethinking'].sum()
    features_df.fillna(0, inplace=True)
    return features_df

# --- 3. Process Train and Test Data ---
print("\nStep 2: Processing Training Data...")
full_train_data = create_features(train_logs)
full_train_data = full_train_data.join(train_scores.set_index('id'))
print("\nStep 3: Processing Test Data...")
test_features = create_features(test_logs)
print("\nAll data processing complete!")
gc.collect()

# --- 4. Define Final Model Setup ---
features_champion = [
    'total_events', 'unique_activities', 'total_action_time', 'mean_action_time', 
    'final_word_count', 'total_writing_time_ms', 'total_pause_time_ms', 
    'pause_time_percentage', 'wpm', 'num_pauses', 'avg_pause_duration', 'num_long_pauses',
    'num_deletions', 'total_deleted_chars', 'revision_density', 'rethinking_events_count'
]
X = full_train_data[features_champion]
y = full_train_data['score']
X_test = test_features[features_champion]

best_lgbm_params = {
    'learning_rate': 0.0228, 'feature_fraction': 0.611, 'bagging_fraction': 0.876, 'bagging_freq': 3,
    'lambda_l1': 9.933, 'lambda_l2': 1.388e-06, 'num_leaves': 46, 'objective': 'regression_l1',
    'metric': 'rmse', 'n_estimators': 2000, 'verbose': -1, 'n_jobs': -1, 'seed': 42
}

# --- THE STRATIFICATION STEP ---
# Create discrete bins from the continuous scores for stratification
num_bins = int(np.floor(1 + np.log2(len(full_train_data))))
full_train_data['score_bin'] = pd.cut(full_train_data['score'], bins=num_bins, labels=False)
y_stratify = full_train_data['score_bin']

NFOLDS = 5
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# Placeholders for predictions
oof_preds_lgbm = np.zeros(len(X)); test_preds_lgbm = np.zeros(len(X_test))
oof_preds_xgb = np.zeros(len(X)); test_preds_xgb = np.zeros(len(X_test))
oof_preds_cat = np.zeros(len(X)); test_preds_cat = np.zeros(len(X_test))

# ===================================================================
# STAGE 5: RUN THE STRATIFIED ENSEMBLE TRAINING
# ===================================================================

# --- LightGBM ---
print("\n--- Training LightGBM Model (Stratified) ---")
# Use y_stratify in the split command
for fold, (train_index, val_index) in enumerate(folds.split(X, y_stratify)):
    print(f"  - Fold {fold+1}/{NFOLDS}...")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]; y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    model = lgb.LGBMRegressor(**best_lgbm_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_preds_lgbm[val_index] = model.predict(X_val); test_preds_lgbm += model.predict(X_test) / NFOLDS

# --- XGBoost ---
print("\n--- Training XGBoost Model (Stratified) ---")
xgb_params = {'objective': 'reg:squarederror', 'n_estimators': 2000, 'learning_rate': 0.05, 'seed': 42, 'tree_method': 'hist'}
for fold, (train_index, val_index) in enumerate(folds.split(X, y_stratify)):
    print(f"  - Fold {fold+1}/{NFOLDS}...")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]; y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    oof_preds_xgb[val_index] = model.predict(X_val); test_preds_xgb += model.predict(X_test) / NFOLDS
    
# --- CatBoost ---
print("\n--- Training CatBoost Model (Stratified) ---")
cat_params = {'iterations': 2000, 'learning_rate': 0.05, 'loss_function': 'RMSE', 'random_seed': 42}
for fold, (train_index, val_index) in enumerate(folds.split(X, y_stratify)):
    print(f"  - Fold {fold+1}/{NFOLDS}...")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]; y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    model = cb.CatBoostRegressor(**cat_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    oof_preds_cat[val_index] = model.predict(X_val); test_preds_cat += model.predict(X_test) / NFOLDS

# ===================================================================
# STAGE 6: EVALUATE, BLEND, AND SUBMIT
# ===================================================================
print("\n--- Evaluating and Blending Stratified Models ---")
rmse_lgbm = np.sqrt(mean_squared_error(y, oof_preds_lgbm)); rmse_xgb = np.sqrt(mean_squared_error(y, oof_preds_xgb)); rmse_cat = np.sqrt(mean_squared_error(y, oof_preds_cat))
print(f"Stratified LightGBM CV Score: {rmse_lgbm:.4f}")
print(f"Stratified XGBoost CV Score:  {rmse_xgb:.4f}")
print(f"Stratified CatBoost CV Score: {rmse_cat:.4f}")

oof_preds_ensemble = (oof_preds_lgbm + oof_preds_xgb + oof_preds_cat) / 3
rmse_ensemble = np.sqrt(mean_squared_error(y, oof_preds_ensemble))
print(f"\nFinal STRATIFIED ENSEMBLE CV Score: {rmse_ensemble:.4f}")

test_preds_ensemble = (test_preds_lgbm + test_preds_xgb + test_preds_cat) / 3
submission_ensemble = pd.DataFrame({'id': X_test.index, 'score': test_preds_ensemble})
submission_ensemble['score'] = (submission_ensemble['score'] * 2).round() / 2
submission_ensemble.to_csv('submission_SOTA_ENSEMBLE.csv', index=False)

print("\n[SUCCESS] Your State-of-the-Art STRATIFIED ENSEMBLE submission 'submission_SOTA_ENSEMBLE.csv' is ready!")
print(submission_ensemble.head())


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ===================================================================
# STAGE 1: SELECT AND SCALE FEATURES FOR CLUSTERING
# ===================================================================
print("--- Starting Behavioral Clustering Pipeline ---")

# Select the features that best define a writer's "style"
clustering_features = [
    'wpm',
    'pause_time_percentage',
    'avg_pause_duration',
    'num_pauses',
    'revision_density',
    'rethinking_events_count'
]

# Create a new dataframe for clustering
X_cluster = full_train_data[clustering_features]

# Scale the features
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

print("Features for clustering have been selected and scaled.")

# ===================================================================
# STAGE 2: FIND THE OPTIMAL NUMBER OF CLUSTERS (ELBOW METHOD)
# ===================================================================
print("\nFinding the optimal number of clusters using the Elbow Method...")

inertia = []
K = range(1, 11) # We'll test from 1 to 10 clusters

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow graph
plt.figure(figsize=(10, 6))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.grid(True)
plt.savefig('figure_elbow_method.png', dpi=300)
plt.show()

print("Elbow method plot has been generated and saved. Please observe the 'elbow' point.")
print("Typically, the best k is where the rate of decrease sharply changes (e.g., 3, 4, or 5).")


# ===================================================================
# STAGE 3: APPLY KMEANS AND CREATE THE NEW FEATURE
# ===================================================================
# Based on the plot, let's choose our number of clusters. 4 is often a good choice.
OPTIMAL_K = 4 

print(f"\nApplying KMeans with k={OPTIMAL_K}...")

kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
# Fit on scaled data and get the cluster labels
clusters = kmeans.fit_predict(X_cluster_scaled)

# Add the new feature to our main dataframe
full_train_data['behavioral_cluster'] = clusters

print("New 'behavioral_cluster' feature has been created successfully!")

# ===================================================================
# STAGE 4: ANALYZE THE CLUSTERS
# ===================================================================
print("\nAnalyzing the characteristics of each cluster:")
cluster_analysis = full_train_data.groupby('behavioral_cluster')[clustering_features + ['score']].mean().sort_values('score', ascending=False)
print(cluster_analysis)

# Save the analysis
cluster_analysis.to_csv('cluster_analysis.csv')
print("\nCluster analysis table saved to 'cluster_analysis.csv'.")


# ===================================================================
# FINAL SOTA ENSEMBLE PIPELINE WITH CLUSTER FEATURE
# ===================================================================
print("--- FINAL PIPELINE: ENSEMBLE + BEHAVIORAL CLUSTER FEATURE ---")

# --- 1. Add Cluster Feature to Test Data ---
# We use the SAME scaler and kmeans model we trained on the training data
X_cluster_test = test_features[clustering_features]
X_cluster_test_scaled = scaler.transform(X_cluster_test) # Use transform(), not fit_transform()
test_clusters = kmeans.predict(X_cluster_test_scaled)
test_features['behavioral_cluster'] = test_clusters
print("Cluster feature added to the test set.")

# --- 2. Define the FINAL Feature Set ---
features_ultimate = features_champion + ['behavioral_cluster']
X = full_train_data[features_ultimate]
y = full_train_data['score']
X_test = test_features[features_ultimate]
y_stratify = full_train_data['score_bin'] # From the previous stratified run

print(f"Training with our ultimate set of {len(features_ultimate)} features.")

# --- 3. Run the Stratified Ensemble (Code is the same as before) ---
NFOLDS = 5
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
oof_preds_lgbm = np.zeros(len(X)); test_preds_lgbm = np.zeros(len(X_test))
oof_preds_xgb = np.zeros(len(X)); test_preds_xgb = np.zeros(len(X_test))
oof_preds_cat = np.zeros(len(X)); test_preds_cat = np.zeros(len(X_test))

# ... (LGBM, XGBoost, CatBoost CV loops are identical to the previous cell) ...
# --- LightGBM ---
print("\n--- Training LightGBM Model (Stratified + Cluster) ---")
for fold, (train_index, val_index) in enumerate(folds.split(X, y_stratify)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]; y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    model = lgb.LGBMRegressor(**best_lgbm_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_preds_lgbm[val_index] = model.predict(X_val); test_preds_lgbm += model.predict(X_test) / NFOLDS

# --- XGBoost ---
print("\n--- Training XGBoost Model (Stratified + Cluster) ---")
xgb_params = {'objective': 'reg:squarederror', 'n_estimators': 2000, 'learning_rate': 0.05, 'seed': 42, 'tree_method': 'hist'}
for fold, (train_index, val_index) in enumerate(folds.split(X, y_stratify)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]; y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    oof_preds_xgb[val_index] = model.predict(X_val); test_preds_xgb += model.predict(X_test) / NFOLDS
    
# --- CatBoost ---
print("\n--- Training CatBoost Model (Stratified + Cluster) ---")
cat_params = {'iterations': 2000, 'learning_rate': 0.05, 'loss_function': 'RMSE', 'random_seed': 42}
for fold, (train_index, val_index) in enumerate(folds.split(X, y_stratify)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]; y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    model = cb.CatBoostRegressor(**cat_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    oof_preds_cat[val_index] = model.predict(X_val); test_preds_cat += model.predict(X_test) / NFOLDS

# --- 4. Evaluate and Submit ---
print("\n--- Evaluating Final Ensemble with Cluster Feature ---")
rmse_lgbm = np.sqrt(mean_squared_error(y, oof_preds_lgbm)); rmse_xgb = np.sqrt(mean_squared_error(y, oof_preds_xgb)); rmse_cat = np.sqrt(mean_squared_error(y, oof_preds_cat))
print(f"Stratified LightGBM CV Score: {rmse_lgbm:.4f}")
print(f"Stratified XGBoost CV Score:  {rmse_xgb:.4f}")
print(f"Stratified CatBoost CV Score: {rmse_cat:.4f}")

oof_preds_ensemble = (oof_preds_lgbm + oof_preds_xgb + oof_preds_cat) / 3
rmse_ensemble = np.sqrt(mean_squared_error(y, oof_preds_ensemble))
print(f"\nPrevious Best Ensemble Score: 0.6927")
print(f"Final ENSEMBLE + CLUSTER CV Score: {rmse_ensemble:.4f}")

test_preds_ensemble = (test_preds_lgbm + test_preds_xgb + test_preds_cat) / 3
submission_ensemble = pd.DataFrame({'id': X_test.index, 'score': test_preds_ensemble})
submission_ensemble['score'] = (submission_ensemble['score'] * 2).round() / 2
submission_ensemble.to_csv('submission_ULTIMATE.csv', index=False)

print("\n[SUCCESS] Your ULTIMATE submission 'submission_ULTIMATE.csv' is ready!")


# ===================================================================
# STAGE 0: IMPORT LIBRARIES AND SETUP
# ===================================================================
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

print("--- INITIALIZING THE ADVANCED STACKING ENSEMBLE PIPELINE ---")

# --- 1. Use the pre-prepared data from our previous runs ---
# X, y, X_test, y_stratify should all be in memory from the last "golden cell".
# We use the 16 Champion Features as they are our proven best set.

# --- 2. Initialize placeholders ---
NFOLDS = 5
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# Level 0 predictions for the Level 1 model's training data
meta_features = np.zeros((len(X), 3)) # 3 columns for 3 models
# Level 0 predictions for the final test data
test_preds_stacked = np.zeros((len(X_test), 3))

# --- 3. Define models and parameters ---
lgbm_params = best_lgbm_params # From our Optuna run
xgb_params = {'objective': 'reg:squarederror', 'n_estimators': 2000, 'learning_rate': 0.05, 'seed': 42, 'tree_method': 'hist'}
cat_params = {'iterations': 2000, 'learning_rate': 0.05, 'loss_function': 'RMSE', 'random_seed': 42}

# ===================================================================
# STAGE 1: TRAIN LEVEL 0 MODELS AND GENERATE META-FEATURES
# ===================================================================
print("\n--- Training Level 0 Specialist Models ---")
for fold, (train_index, val_index) in enumerate(folds.split(X, y_stratify)):
    print(f"  - Fold {fold+1}/{NFOLDS}...")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # --- LGBM ---
    lgbm = lgb.LGBMRegressor(**lgbm_params)
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    meta_features[val_index, 0] = lgbm.predict(X_val)
    test_preds_stacked[:, 0] += lgbm.predict(X_test) / NFOLDS

    # --- XGB ---
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    meta_features[val_index, 1] = xgb_model.predict(X_val)
    test_preds_stacked[:, 1] += xgb_model.predict(X_test) / NFOLDS

    # --- CatBoost ---
    cat = cb.CatBoostRegressor(**cat_params)
    cat.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    meta_features[val_index, 2] = cat.predict(X_val)
    test_preds_stacked[:, 2] += cat.predict(X_test) / NFOLDS

print("Level 0 training complete. Meta-features have been generated.")

# ===================================================================
# STAGE 2: TRAIN LEVEL 1 META-MODEL (THE MASTERMIND)
# ===================================================================
print("\n--- Training Level 1 Mastermind Model ---")

# The meta-model is very simple, a Ridge Regression
meta_model = Ridge(random_state=42)

# Train the mastermind on the predictions of the specialists
meta_model.fit(meta_features, y)

print("Mastermind model trained successfully.")

# ===================================================================
# STAGE 3: EVALUATE AND SUBMIT
# ===================================================================
# Evaluate the performance of the full stack
oof_preds_stacking = meta_model.predict(meta_features)
rmse_stacking = np.sqrt(mean_squared_error(y, oof_preds_stacking))

print(f"\nPrevious Best Simple Ensemble Score: 0.6927")
print(f"Final STACKING ENSEMBLE CV Score: {rmse_stacking:.4f}")

# Make final predictions on the test set
final_test_predictions = meta_model.predict(test_preds_stacked)

# Create submission file
submission_stacking = pd.DataFrame({'id': X_test.index, 'score': final_test_predictions})
submission_stacking['score'] = (submission_stacking['score'] * 2).round() / 2
submission_stacking.to_csv('submission_STACKING.csv', index=False)

print("\n[SUCCESS] Your STACKING ENSEMBLE submission 'submission_STACKING.csv' is ready!")
print(submission_stacking.head())

# Also save the result
with open('model_performance.txt', 'a') as f:
    f.write(f"Stacking Ensemble CV RMSE: {rmse_stacking:.4f}\n")


# ===================================================================
# STAGE 0: IMPORT LIBRARIES AND SETUP
# ===================================================================
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import matplotlib.pyplot as plt

print("--- INITIALIZING THE ULTIMATE FEATURE SELECTION PIPELINE (RFECV) ---")
print("This will be a very long process. Please be patient.")

# --- 1. Use the pre-prepared data from our previous runs ---
# We use the 16 Champion Features as our starting point
features_champion = [
    'total_events', 'unique_activities', 'total_action_time', 'mean_action_time', 
    'final_word_count', 'total_writing_time_ms', 'total_pause_time_ms', 
    'pause_time_percentage', 'wpm', 'num_pauses', 'avg_pause_duration', 'num_long_pauses',
    'num_deletions', 'total_deleted_chars', 'revision_density', 'rethinking_events_count'
]
X = full_train_data[features_champion]
y = full_train_data['score']
y_stratify = full_train_data['score_bin'] # From our stratified runs

# We will use a basic (non-optimized) LGBM for speed in the RFE process
lgbm_rfe = lgb.LGBMRegressor(random_state=42, n_jobs=-1)

# ===================================================================
# STAGE 1: RUN RECURSIVE FEATURE ELIMINATION WITH CV
# ===================================================================
# We will use Stratified K-Fold for robustness
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # Using 3 splits for speed

# Initialize RFECV
# This will test subsets from 1 feature up to all 16 features
rfecv = RFECV(
    estimator=lgbm_rfe,
    step=1,
    cv=cv,
    scoring='neg_root_mean_squared_error', # Scikit-learn uses negative RMSE for maximization
    n_jobs=-1 # Use all available CPU cores
)

print("\nStarting RFECV fitting... This is the long part.")
rfecv.fit(X, y_stratify) # Fit on the data
print("RFECV fitting complete!")

# ===================================================================
# STAGE 2: ANALYZE THE RESULTS
# ===================================================================
print("\n--- RFECV Results ---")
print(f"Optimal number of features: {rfecv.n_features_}")

# Get the names of the selected features
selected_features = X.columns[rfecv.support_].tolist()
print("\nSelected features:")
print(selected_features)

# Get the ranking of all features (1 is best)
feature_ranking = pd.DataFrame({
    'feature': X.columns,
    'ranking': rfecv.ranking_
}).sort_values('ranking')

print("\nFeature Ranking:")
print(feature_ranking)
feature_ranking.to_csv('feature_ranking_rfe.csv', index=False)
print("\nFeature ranking saved to 'feature_ranking_rfe.csv'")

# ===================================================================
# STAGE 3: PLOT THE RESULTS
# ===================================================================
# The grid_scores_ attribute is deprecated, use cv_results_ instead
# It holds the scores for each number of features tested
plt.figure(figsize=(12, 6))
plt.title('Recursive Feature Elimination with Cross-Validation')
plt.xlabel('Number of features selected')
plt.ylabel('CV Score (Negative RMSE)')
# We multiply by -1 to show the actual RMSE score
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), -rfecv.cv_results_['mean_test_score'])
plt.grid()
plt.savefig('figure_rfecv_curve.png', dpi=300)
plt.show()
print("RFECV plot saved as 'figure_rfecv_curve.png'")

# ===================================================================
# STAGE 4: STORE THE FINAL OPTIMAL FEATURE SET
# ===================================================================
# This variable will hold our absolute best feature set for the final model
features_SOTA = selected_features
print("\nThe variable 'features_SOTA' now holds the absolute best feature set.")


# ===================================================================
# THE FINAL, DEFINITIVE, CHAMPION PIPELINE: STACKING ENSEMBLE
# ===================================================================
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

print("--- RUNNING THE DEFINITIVE CHAMPION PIPELINE: STACKING ENSEMBLE ---")

# --- Using all pre-loaded data (full_train_data, test_features, best_params, etc.) ---
features_champion = [
    'total_events', 'unique_activities', 'total_action_time', 'mean_action_time', 'final_word_count', 
    'total_writing_time_ms', 'total_pause_time_ms', 'pause_time_percentage', 'wpm', 'num_pauses', 
    'avg_pause_duration', 'num_long_pauses', 'num_deletions', 'total_deleted_chars', 
    'revision_density', 'rethinking_events_count'
]
X = full_train_data[features_champion]
y = full_train_data['score']
X_test = test_features[features_champion]
y_stratify = full_train_data['score_bin']

lgbm_params = best_lgbm_params
xgb_params = {'objective': 'reg:squarederror', 'n_estimators': 2000, 'learning_rate': 0.05, 'seed': 42, 'tree_method': 'hist'}
cat_params = {'iterations': 2000, 'learning_rate': 0.05, 'loss_function': 'RMSE', 'random_seed': 42}

# --- Initialize placeholders ---
NFOLDS = 5
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
meta_features = np.zeros((len(X), 3))
test_preds_stacked = np.zeros((len(X_test), 3))

# ===================================================================
# STAGE 1: TRAIN LEVEL 0 SPECIALIST MODELS
# ===================================================================
print("\n--- Training Level 0 Specialist Models ---")
for fold, (train_index, val_index) in enumerate(folds.split(X, y_stratify)):
    print(f"  - Fold {fold+1}/{NFOLDS}...")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # --- LGBM ---
    lgbm = lgb.LGBMRegressor(**lgbm_params)
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    meta_features[val_index, 0] = lgbm.predict(X_val)
    test_preds_stacked[:, 0] += lgbm.predict(X_test) / NFOLDS

    # --- XGB ---
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    meta_features[val_index, 1] = xgb_model.predict(X_val)
    test_preds_stacked[:, 1] += xgb_model.predict(X_test) / NFOLDS

    # --- CatBoost ---
    cat = cb.CatBoostRegressor(**cat_params)
    cat.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    meta_features[val_index, 2] = cat.predict(X_val)
    test_preds_stacked[:, 2] += cat.predict(X_test) / NFOLDS

print("Level 0 training complete.")

# ===================================================================
# STAGE 2: TRAIN LEVEL 1 MASTERMIND MODEL
# ===================================================================
print("\n--- Training Level 1 Mastermind Model ---")
meta_model = Ridge(random_state=42)
meta_model.fit(meta_features, y)
print("Mastermind model trained successfully.")

# ===================================================================
# STAGE 3: EVALUATE AND SUBMIT
# ===================================================================
oof_preds_stacking = meta_model.predict(meta_features)
rmse_stacking = np.sqrt(mean_squared_error(y, oof_preds_stacking))

print(f"\nFINAL CONFIRMED STACKING ENSEMBLE CV Score: {rmse_stacking:.4f}")

final_test_predictions = meta_model.predict(test_preds_stacked)
submission_stacking = pd.DataFrame({'id': X_test.index, 'score': final_test_predictions})
submission_stacking['score'] = (submission_stacking['score'] * 2).round() / 2
submission_stacking.to_csv('submission_STACKING_CHAMPION.csv', index=False)

print("\n[SUCCESS] Your CHAMPION STACKING ENSEMBLE submission 'submission_STACKING_CHAMPION.csv' is ready!")
print(submission_stacking.head())

