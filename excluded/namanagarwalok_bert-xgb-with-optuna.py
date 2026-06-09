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


train_path = '/kaggle/input/playground-series-s5e10/train.csv'
test_path = '/kaggle/input/playground-series-s5e10/test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df.head()


test_df.head()


train_df.isna().sum()


test_df.isna().sum()


#from sklearn.preprocessing import OneHotEncoder
#ohe  = OneHotEncoder()
#column transformer is used when we want to apply different preprocessing on seperate columns
#here we want to one hot encode road_type, lighting, weather, road_signs_present	
#public_road time_of_day	holiday and school season but not the others
#from sklearn.compose import make_column_transformer

#column_trans = make_column_transformer((ohe,['road_type', 'lighting', 'weather', 'road_signs_present',
                                      #     'public_road', 'time_of_day','holiday', 'school_season']),
                                     # remainder = 'passthrough')


#first seperating the train_df into X and Y
#X = train_df.drop(['accident_risk','id'],axis = 'columns')
#Y = train_df['accident_risk']


#train_ohe = column_trans.fit_transform(X)


#test_ids = test_df["id"]
#test_df = test_df.drop('id',axis = 'columns')
#test_ohe = column_trans.transform(test_df)


#train_ohe


#test_ohe


#from xgboost import XGBRegressor
#reg = XGBRegressor(n_estimators = 100, learning_rate = 0.1,max_depth = 5)
#reg.fit(train_ohe,Y)


#y_pred = reg.predict(test_ohe)


sample = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
sample.head()


#sub_df = pd.DataFrame({'id':test_ids,'accident_risk':y_pred})
#sub_df.to_csv('submission.csv',index = False)


#from sklearn.model_selection import train_test_split

#x_train,x_valid,y_train,y_valid = train_test_split(train_ohe,Y,test_size = 0.15,random_state = 69)


"""reg2 = XGBRegressor(n_estimators = 5000, learning_rate = 0.03,
                    max_depth = 6,subsamples = 0.8,colsample_bytree = 0.8, 
                    random_state = 69, tree_method = 'hist')
reg2.fit(x_train,y_train,
         eval_set = [(x_valid,y_valid)],
        early_stopping_rounds=100,  # stops if no improvement
        verbose=100)"""


#y_pred = reg2.predict(test_ohe)
#sub_df = pd.DataFrame({'id':test_ids,'accident_risk':y_pred})
#sub_df.to_csv('submission.csv',index = False)


'''import optuna
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def objective(trial):
    params = {
        'n_estimators': 5000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 69,
        'device': 'cuda',
        'tree_method': 'hist',
        'early_stopping_rounds': 100  
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for tr_idx, val_idx in kf.split(train_ohe):
        X_tr, y_tr = train_ohe[tr_idx], Y.iloc[tr_idx]
        X_val, y_val = train_ohe[val_idx], Y.iloc[val_idx]
        
        model = XGBRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        ) 
        
        preds = model.predict(X_val)
        score = mean_squared_error(y_val, preds, squared=False)
        scores.append(score)
    
    return np.mean(scores)

# Start the optimization (outside the function)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)
print("Best params:", study.best_params)
print("Best RMSE:", study.best_value)'''


'''best_params = study.best_params.copy()
best_params.update({
    'n_estimators': 5000,
    'random_state': 69,
    'tree_method': 'hist',
    'device': 'cuda',
    'early_stopping_rounds': 100  
})

reg3 = XGBRegressor(**best_params)
reg3.fit(
    x_train, y_train,
    eval_set=[(x_valid, y_valid)],
    verbose=100
)  

print(f"Best iteration: {reg3.best_iteration}")
print(f"Best score: {reg3.best_score}")'''


#y_pred = reg3.predict(test_ohe)
#sub_df = pd.DataFrame({'id':test_ids,'accident_risk':y_pred})
#sub_df.to_csv('submission.csv',index = False)


train_df.head()


'''I realised the features should be communicating with each other as they affect each other
for example: if its a holiday the it doesnt matter if its a school season or not 
similarly if the road is a highway then features like speed limit and weather etc should be considered more than the school season or public road.'''


from transformers import AutoTokenizer, AutoModel
import torch
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


categorical_cols = ['road_type','lighting'	,'weather'	,
                    'road_signs_present'	,'public_road'	,'time_of_day'	
                    ,'holiday'	,'school_season']

def create_natural_sentences(df):
    sentences = []
    
    for idx, row in df.iterrows():
        # Convert boolean/binary values to natural language
        road_signs = "with" if row['road_signs_present'] == True else "without"
        public_status = "public" if row['public_road'] == True else "private"
        holiday_status = "holiday" if row['holiday'] == True else "weekday"
        school_status = "during school season" if row['school_season'] == True else "outside school season"
        
        # Build natural sentence
        sentence = (
            f"A {row['road_type']} road with {row['lighting']} lighting "
            f"in {row['weather']} weather, {road_signs} road signs, "
            f"{public_status} road, during {row['time_of_day']} "
            f"on a {holiday_status} {school_status}"
        )
        
        sentences.append(sentence)
    
    return sentences


'''train_sentences = create_natural_sentences(train_df)
test_sentences = create_natural_sentences(test_df)

# Check example
print("Example sentences:")
print(train_sentences[0])
print(train_sentences[1])
print(train_sentences[2])'''


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

'''# ===== 1. INITIALIZE BERT =====

# ===== 2. CREATE BERT EMBEDDINGS =====
def get_bert_embeddings(sentences, batch_size=32):
    """Convert sentences to BERT embeddings"""
    all_embeddings = []
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        
        # Tokenize
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Move to device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
        
        if (i // batch_size) % 10 == 0:
            print(f"Processed {i + len(batch)}/{len(sentences)} sentences")
    
    return np.vstack(all_embeddings)

# Generate embeddings
print("\nGenerating BERT embeddings for train set...")
train_bert_embeddings = get_bert_embeddings(train_sentences, batch_size=32)

print("\nGenerating BERT embeddings for test set...")
test_bert_embeddings = get_bert_embeddings(test_sentences, batch_size=32)

print(f"\nTrain BERT embeddings shape: {train_bert_embeddings.shape}")
print(f"Test BERT embeddings shape: {test_bert_embeddings.shape}")
'''



#print("\nSaving BERT embeddings...")

# Save as numpy files (fast and efficient)
#np.save('train_bert_embeddings.npy', train_bert_embeddings)
#np.save('test_bert_embeddings.npy', test_bert_embeddings)

#print(" Embeddings saved!")
#print(f"   - train_bert_embeddings.npy: {train_bert_embeddings.shape}")
#print(f"   - test_bert_embeddings.npy: {test_bert_embeddings.shape}")


print("Loading saved BERT embeddings...")
train_bert_embeddings = np.load('/kaggle/input/bert-embeddings-predicting-road-accidents/train_bert_embeddings.npy')
test_bert_embeddings = np.load('/kaggle/input/bert-embeddings-predicting-road-accidents/test_bert_embeddings.npy')


# ===== 3. COMBINE WITH NUMERICAL FEATURES =====
# numerical column names (excluding 'id' and target)
#numerical_cols = ['num_lanes', 'curvature', 'speed_limit','num_reported_accidents'] 

# Get numerical features
#train_numerical = train_df[numerical_cols].values
#test_numerical = test_df[numerical_cols].values

# Combine BERT embeddings + numerical features
#X_train_full = np.hstack([train_bert_embeddings, train_numerical])
#X_test_final = np.hstack([test_bert_embeddings, test_numerical])

# ===== 4. PREPARE TARGET AND VALIDATION SPLIT =====
#target_col = 'accident_risk'  # CHANGE if different
#y_train_full = train_df[target_col].values

# Split train into train/validation for evaluation
#X_train, X_val, y_train, y_val = train_test_split(
 #   X_train_full, y_train_full, test_size=0.2, random_state=42
#)

#print(f"\nTraining set: {X_train.shape}")
#print(f"Validation set: {X_val.shape}")
#print(f"Test set: {X_test_final.shape}")


'''# ===== 5. TRAIN XGBOOST =====
print("\nTraining XGBoost model...")

xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=45,
    tree_method = 'hist',
    device = 'cuda',
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

# ===== 6. EVALUATE ON VALIDATION =====
train_pred = xgb_model.predict(X_train)
val_pred = xgb_model.predict(X_val)

print("\n===== VALIDATION RESULTS =====")
print(f"Train R²: {r2_score(y_train, train_pred):.4f}")
print(f"Validation R²: {r2_score(y_val, val_pred):.4f}")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, train_pred)):.4f}")
print(f"Validation RMSE: {np.sqrt(mean_squared_error(y_val, val_pred)):.4f}")

# ===== 7. RETRAIN ON FULL TRAINING DATA =====
print("\nRetraining on full training data...")
xgb_model.fit(X_train_full, y_train_full)'''


from sklearn.preprocessing import LabelEncoder
import optuna

# ===== 1. PREPARE CATEGORICAL FEATURES =====
categorical_cols = ['road_type', 'lighting', 'weather', 
                    'road_signs_present', 'public_road', 'time_of_day', 
                    'holiday', 'school_season']

# Label encode categorical features
print("Encoding categorical features...")
label_encoders = {}
train_categorical_encoded = np.zeros((len(train_df), len(categorical_cols)))
test_categorical_encoded = np.zeros((len(test_df), len(categorical_cols)))

for idx, col in enumerate(categorical_cols):
    le = LabelEncoder()
    # Fit on combined train+test to handle all categories
    le.fit(pd.concat([train_df[col], test_df[col]]).astype(str))
    
    train_categorical_encoded[:, idx] = le.transform(train_df[col].astype(str))
    test_categorical_encoded[:, idx] = le.transform(test_df[col].astype(str))
    
    label_encoders[col] = le

print(f"Categorical features shape: {train_categorical_encoded.shape}")

# ===== 3. COMBINE ALL FEATURES =====
numerical_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

train_numerical = train_df[numerical_cols].values
test_numerical = test_df[numerical_cols].values

# Stack: BERT embeddings + categorical encoded + numerical
X_train_full = np.hstack([
    train_bert_embeddings,
    train_categorical_encoded,
    train_numerical
])

X_test_final = np.hstack([
    test_bert_embeddings,
    test_categorical_encoded,
    test_numerical
])

print(f"\nFull feature dimensions:")
print(f"  BERT embeddings: {train_bert_embeddings.shape[1]}")
print(f"  Categorical: {train_categorical_encoded.shape[1]}")
print(f"  Numerical: {train_numerical.shape[1]}")
print(f"  Total: {X_train_full.shape[1]}")

# ===== 4. PREPARE TARGET =====
target_col = 'accident_risk'
y_train_full = train_df[target_col].values

# Split for hyperparameter tuning
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")

# ===== 5. HYPERPARAMETER TUNING WITH OPTUNA =====
def objective(trial):
    """Optuna objective function for XGBoost hyperparameter tuning"""
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'random_state': 45,
        'tree_method': 'hist',
        'device': 'cuda',
        'n_jobs': -1
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    
    val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    
    return rmse

print("\n===== STARTING HYPERPARAMETER TUNING =====")
print("This will take a few minutes...\n")

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\n===== BEST HYPERPARAMETERS =====")
print(study.best_params)
print(f"\nBest validation RMSE: {study.best_value:.4f}")

# ===== 6. TRAIN WITH BEST PARAMETERS =====
best_params = study.best_params
best_params.update({
    'random_state': 45,
    'tree_method': 'hist',
    'device': 'cuda',
    'n_jobs': -1
})

print("\n===== TRAINING FINAL MODEL =====")
final_model = xgb.XGBRegressor(**best_params)

# Train on validation split first to check
final_model.fit(X_train, y_train)

train_pred = final_model.predict(X_train)
val_pred = final_model.predict(X_val)

print("\n===== VALIDATION RESULTS =====")
print(f"Train R²: {r2_score(y_train, train_pred):.4f}")
print(f"Validation R²: {r2_score(y_val, val_pred):.4f}")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, train_pred)):.4f}")
print(f"Validation RMSE: {np.sqrt(mean_squared_error(y_val, val_pred)):.4f}")

# ===== 7. RETRAIN ON FULL DATA =====
print("\nRetraining on full training data...")
final_model.fit(X_train_full, y_train_full)

# ===== 8. PREDICT ON TEST =====
print("\nPredicting on test set...")
test_predictions = final_model.predict(X_test_final)

# ===== 9. CREATE SUBMISSION =====
submission = pd.DataFrame({
    'id': test_df['id'],
    'accident_risk': test_predictions
})

submission.to_csv('submission_bert_tuned.csv', index=False)
print("\n✅ Submission saved to 'submission_bert_tuned.csv'")
print(f"Submission shape: {submission.shape}")
print("\nFirst few predictions:")
print(submission.head())

