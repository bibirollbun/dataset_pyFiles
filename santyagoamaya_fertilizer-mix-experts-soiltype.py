import numpy as np 
import pandas as pd 
import os


train, test, submission = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv'), pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv'), pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv') 


labels = train['Fertilizer Name'].unique()
submission.head()


print(test.sample(5))
print(train.sample(5))


from sklearn.preprocessing import LabelEncoder
le_fertilizer = LabelEncoder().fit(labels)
le_crop = LabelEncoder()
train['Fertilizer Name'] =  le_fertilizer.fit_transform(train['Fertilizer Name'])
train['Crop Type'] = le_crop.fit_transform(train['Crop Type'])
test['Crop Type'] = le_crop.transform(test['Crop Type'])



print(train.sample(3))
print('decode = ', le_fertilizer.inverse_transform(train['Fertilizer Name']))


train_soil_npk_crop = train[train.columns]


print(train['Soil Type'].unique())


clayey_df = train_soil_npk_crop[train_soil_npk_crop['Soil Type'] == 'Clayey'].drop(columns='Soil Type')
sandy_df = train_soil_npk_crop[train_soil_npk_crop['Soil Type'] == 'Sandy'].drop(columns='Soil Type')
red_df = train_soil_npk_crop[train_soil_npk_crop['Soil Type'] == 'Red'].drop(columns='Soil Type')
loamy_df = train_soil_npk_crop[train_soil_npk_crop['Soil Type'] == 'Loamy'].drop(columns='Soil Type')
black_df = train_soil_npk_crop[train_soil_npk_crop['Soil Type'] == 'Black'].drop(columns='Soil Type')


clayey_df


test_soil_npk_crop = test[test.columns]
test_soil_npk_crop


test_clayey_df = test_soil_npk_crop[test_soil_npk_crop['Soil Type'] == 'Clayey'].drop(columns='Soil Type')
test_sandy_df = test_soil_npk_crop[test_soil_npk_crop['Soil Type'] == 'Sandy'].drop(columns='Soil Type')
test_red_df = test_soil_npk_crop[test_soil_npk_crop['Soil Type'] == 'Red'].drop(columns='Soil Type')
test_loamy_df = test_soil_npk_crop[test_soil_npk_crop['Soil Type'] == 'Loamy'].drop(columns='Soil Type')
test_black_df = test_soil_npk_crop[test_soil_npk_crop['Soil Type'] == 'Black'].drop(columns='Soil Type')


print('train ==>', len(clayey_df), len(sandy_df), len(red_df), len(loamy_df), len(black_df))
print('test ==>', len(test_clayey_df), len(test_sandy_df), len(test_red_df), len(test_loamy_df), len(test_black_df))


from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

def split_data(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_val, y_train, y_val

def apk(actual, predicted, k=3):
    """
    Compute the average precision at k.
    actual : int, the single true label.
    predicted : list of ints, predicted labels ranked by confidence.
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p == actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)  # precision at i+1
            break  # since only one true label, stop once found

    return score

def mapk(actuals, predicted_lists, k=3):
    """
    Compute mean average precision at k.
    actuals : list or array of true labels.
    predicted_lists : list of lists, each inner list is predicted labels for an instance.
    """
    return np.mean([apk(a, p, k) for a, p in zip(actuals, predicted_lists)])

def create_lgb_model(params=None):
    default_params = {
        'objective': 'multiclass',
        'num_class': None,
        'random_state': 42,
        'n_estimators': 100,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'verbosity': -1
    }
    if params:
        default_params.update(params)
    return lgb.LGBMClassifier(**default_params)

def create_xgb_model(params=None):
    default_params = {
        'objective': 'multi:softprob',
        'num_class': None,
        'random_state': 42,
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'use_label_encoder': False,
    }
    if params:
        default_params.update(params)
    return xgb.XGBClassifier(**default_params)

def create_catboost_model(params=None):
    default_params = {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 6,
        'random_seed': 42,
        'verbose': False,
        'loss_function': 'MultiClass'
    }
    if params:
        default_params.update(params)
    return CatBoostClassifier(**default_params)


# LightGBM Training Function with MAP@3 and Early Stopping via Callbacks
def train_lgb_model_with_map3(X, y, label_encoder, params=None):
    num_classes = len(label_encoder.classes_)
    X_train, X_val, y_train, y_val = split_data(X, y)
    if params is None:
        params = {}
    params.update({"num_class": num_classes})

    model = create_lgb_model(params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )

    y_proba = model.predict_proba(X_val)
    top3_pred_indices = np.argsort(y_proba, axis=1)[:, ::-1][:, :3]
    top3_pred_labels = [label_encoder.inverse_transform(row) for row in top3_pred_indices]

    score = mapk(y_val, top3_pred_indices, k=3)
    print(f"LightGBM Validation MAP@3: {score:.4f}")

    return model

def train_xgb_model_with_map3(X, y, label_encoder, params=None):
    num_classes = len(label_encoder.classes_)
    X_train, X_val, y_train, y_val = split_data(X, y)
    if params is None:
        params = {}
    params.update({"num_class": num_classes})

    model = create_xgb_model(params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=False
    )

    y_proba = model.predict_proba(X_val)
    top3_pred_indices = np.argsort(y_proba, axis=1)[:, ::-1][:, :3]
    top3_pred_labels = [label_encoder.inverse_transform(row) for row in top3_pred_indices]

    score = mapk(y_val, top3_pred_indices, k=3)
    print(f"XGBoost Validation MAP@3: {score:.4f}")

    return model

def train_catboost_model_with_map3(X, y, label_encoder, params=None):
    X_train, X_val, y_train, y_val = split_data(X, y)
    if params is None:
        params = {}
    model = create_catboost_model(params)

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=20,
        verbose=False
    )

    y_proba = model.predict_proba(X_val)
    top3_pred_indices = np.argsort(y_proba, axis=1)[:, ::-1][:, :3]
    top3_pred_labels = [label_encoder.inverse_transform(row) for row in top3_pred_indices]

    score = mapk(y_val, top3_pred_indices, k=3)
    print(f"CatBoost Validation MAP@3: {score:.4f}")

    return model
    


X_clayey_df = clayey_df.drop(columns=['Fertilizer Name'])
X_sandy_df = sandy_df.drop(columns=['Fertilizer Name'])
X_red_df = red_df.drop(columns=['Fertilizer Name'])
X_loamy_df = loamy_df.drop(columns=['Fertilizer Name'])
X_black_df = black_df.drop(columns=['Fertilizer Name'])
y_clayey_df = clayey_df['Fertilizer Name']
y_sandy_df = sandy_df['Fertilizer Name']
y_red_df = red_df['Fertilizer Name']
y_loamy_df = loamy_df['Fertilizer Name']
y_black_df = black_df['Fertilizer Name']
# Train models
lgb_model_clay = train_lgb_model_with_map3(X_clayey_df, y_clayey_df, le_fertilizer)
xgb_model_clay = train_xgb_model_with_map3(X_clayey_df, y_clayey_df, le_fertilizer)
cat_model_clay = train_catboost_model_with_map3(X_clayey_df, y_clayey_df, le_fertilizer)

lgb_model_sandy = train_lgb_model_with_map3(X_sandy_df, y_sandy_df, le_fertilizer)
xgb_model_sandy = train_xgb_model_with_map3(X_sandy_df, y_sandy_df, le_fertilizer)
cat_model_sandy = train_catboost_model_with_map3(X_sandy_df, y_sandy_df, le_fertilizer)

lgb_model_red = train_lgb_model_with_map3(X_red_df, y_red_df, le_fertilizer)
xgb_model_red = train_xgb_model_with_map3(X_red_df, y_red_df, le_fertilizer)
cat_model_red = train_catboost_model_with_map3(X_red_df, y_red_df, le_fertilizer)

lgb_model_loamy = train_lgb_model_with_map3(X_loamy_df, y_loamy_df, le_fertilizer)
xgb_model_loamy = train_xgb_model_with_map3(X_loamy_df, y_loamy_df, le_fertilizer)
cat_model_loamy = train_catboost_model_with_map3(X_loamy_df, y_loamy_df, le_fertilizer)

lgb_model_black = train_lgb_model_with_map3(X_black_df, y_black_df, le_fertilizer)
xgb_model_black = train_xgb_model_with_map3(X_black_df, y_black_df, le_fertilizer)
cat_model_black = train_catboost_model_with_map3(X_black_df, y_black_df, le_fertilizer)


# Function to get predictions from a model (returns top 3 predictions)
def get_top3_predictions(model, X_test):
    y_proba = model.predict_proba(X_test)
    top3_pred_indices = np.argsort(y_proba, axis=1)[:, ::-1][:, :3]
    return top3_pred_indices

# Store IDs for each soil type before making predictions
clayey_ids =  test_clayey_df['id'].copy()  # or if you have an 'id' column
sandy_ids = test_sandy_df['id'].copy()
red_ids = test_red_df['id'].copy()
loamy_ids = test_loamy_df['id'].copy()
black_ids = test_black_df['id'].copy()
# Step 1: Get predictions for each soil type using the respective models
# Clayey Soil Predictions
clayey_lgb_pred = get_top3_predictions(lgb_model_clay, test_clayey_df)
clayey_xgb_pred = get_top3_predictions(xgb_model_clay, test_clayey_df)
clayey_cat_pred = get_top3_predictions(cat_model_clay, test_clayey_df)

# Sandy Soil Predictions
sandy_lgb_pred = get_top3_predictions(lgb_model_sandy, test_sandy_df)
sandy_xgb_pred = get_top3_predictions(xgb_model_sandy, test_sandy_df)
sandy_cat_pred = get_top3_predictions(cat_model_sandy, test_sandy_df)

# Red Soil Predictions
red_lgb_pred = get_top3_predictions(lgb_model_red, test_red_df)
red_xgb_pred = get_top3_predictions(xgb_model_red, test_red_df)
red_cat_pred = get_top3_predictions(cat_model_red, test_red_df)

# Loamy Soil Predictions
loamy_lgb_pred = get_top3_predictions(lgb_model_loamy, test_loamy_df)
loamy_xgb_pred = get_top3_predictions(xgb_model_loamy, test_loamy_df)
loamy_cat_pred = get_top3_predictions(cat_model_loamy, test_loamy_df)

# Black Soil Predictions
black_lgb_pred = get_top3_predictions(lgb_model_black, test_black_df)
black_xgb_pred = get_top3_predictions(xgb_model_black, test_black_df)
black_cat_pred = get_top3_predictions(cat_model_black, test_black_df)




from collections import Counter
from scipy import stats

fertilizer_mapping = dict(zip(range(len(le_fertilizer.classes_)), le_fertilizer.classes_))

def ensemble_predictions_voting(pred_arrays, method='hard_voting'):
    """
    Ensemble multiple prediction arrays using different voting methods
    
    Args:
        pred_arrays: List of prediction arrays (each is n_samples x 3)
        method: 'hard_voting', 'soft_voting', or 'rank_voting'
    
    Returns:
        Final ensemble predictions (n_samples x 3)
    """
    n_samples = pred_arrays[0].shape[0]
    ensemble_pred = np.zeros((n_samples, 3), dtype=int)
    
    if method == 'hard_voting':
        # For each sample, count votes for each position
        for i in range(n_samples):
            # Get all predictions for this sample
            sample_preds = [pred[i] for pred in pred_arrays]
            
            # Count votes for each class across all models
            all_votes = []
            for pred in sample_preds:
                all_votes.extend(pred)
            
            # Get top 3 most voted classes
            vote_counts = Counter(all_votes)
            top_3 = [class_id for class_id, _ in vote_counts.most_common(3)]
            
            # Ensure we have exactly 3 predictions
            if len(top_3) < 3:
                # Add remaining classes that weren't predicted
                all_classes = set(range(max(all_votes) + 1))
                remaining = list(all_classes - set(top_3))
                top_3.extend(remaining[:3-len(top_3)])
            
            ensemble_pred[i] = top_3[:3]
    
    elif method == 'rank_voting':
        # Assign scores based on ranking position
        for i in range(n_samples):
            class_scores = {}
            
            for pred in pred_arrays:
                for rank, class_id in enumerate(pred[i]):
                    score = 3 - rank  # 1st place gets 3 points, 2nd gets 2, 3rd gets 1
                    if class_id in class_scores:
                        class_scores[class_id] += score
                    else:
                        class_scores[class_id] = score
            
            # Sort by score and take top 3
            sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
            ensemble_pred[i] = [class_id for class_id, _ in sorted_classes[:3]]
    
    return ensemble_pred

def ensemble_with_probabilities(models, X_test, method='average'):
    """
    Ensemble using probability averaging (if you have access to models)
    """
    all_probas = []
    
    for model in models:
        proba = model.predict_proba(X_test)
        all_probas.append(proba)
    
    if method == 'average':
        # Average probabilities
        ensemble_proba = np.mean(all_probas, axis=0)
    elif method == 'weighted_average':
        # You can add weights based on model performance
        weights = [0.3, 0.33, 0.33]  # Example weights for LGB, XGB, CatBoost
        ensemble_proba = np.average(all_probas, axis=0, weights=weights)
    
    # Get top 3 predictions
    top3_pred_indices = np.argsort(ensemble_proba, axis=1)[:, ::-1][:, :3]
    return top3_pred_indices


# Function to ensemble predictions for each soil type
def create_soil_ensemble(lgb_pred, xgb_pred, cat_pred, method='rank_voting'):
    """Create ensemble for a specific soil type"""
    pred_arrays = [lgb_pred, xgb_pred, cat_pred]
    return ensemble_predictions_voting(pred_arrays, method=method)

# Create ensembles for each soil type
clayey_ensemble = create_soil_ensemble(clayey_lgb_pred, clayey_xgb_pred, clayey_cat_pred)
sandy_ensemble = create_soil_ensemble(sandy_lgb_pred, sandy_xgb_pred, sandy_cat_pred)
red_ensemble = create_soil_ensemble(red_lgb_pred, red_xgb_pred, red_cat_pred)
loamy_ensemble = create_soil_ensemble(loamy_lgb_pred, loamy_xgb_pred, loamy_cat_pred)
black_ensemble = create_soil_ensemble(black_lgb_pred, black_xgb_pred, black_cat_pred)

print("Ensemble predictions created for all soil types!")

def decode_and_create_submission(ensemble_predictions, test_ids, label_encoder, soil_type_name):
    """
    Decode ensemble predictions and return formatted results
    
    Args:
        ensemble_predictions: numpy array of shape (n_samples, 3) with class indices
        test_ids: the saved IDs for this soil type (pandas Series or list)
        label_encoder: the label encoder used for this soil type
        soil_type_name: name of soil type for debugging
    
    Returns:
        DataFrame with id and decoded predictions
    """
    decoded_predictions = []
    
    for i, pred_indices in enumerate(ensemble_predictions):
        # Decode the class indices to fertilizer names
        fertilizer_names = label_encoder.inverse_transform(pred_indices)
        # Join with spaces as shown in your example
        fertilizer_string = ' '.join(fertilizer_names)
        
        # Get the corresponding ID from the saved IDs
        original_id = test_ids.iloc[i] if hasattr(test_ids, 'iloc') else test_ids[i]
        
        decoded_predictions.append({
            'id': original_id,
            'Fertilizer Name': fertilizer_string
        })
    
    return pd.DataFrame(decoded_predictions)

# Use the same fertilizer label encoder for all soil types
clayey_label_encoder = le_fertilizer
sandy_label_encoder = le_fertilizer
red_label_encoder = le_fertilizer
loamy_label_encoder = le_fertilizer
black_label_encoder = le_fertilizer

# Create submission dataframes for each soil type using the saved IDs
clayey_submission = decode_and_create_submission(
    clayey_ensemble, 
    clayey_ids,  
    clayey_label_encoder,  
    'clayey'
)

sandy_submission = decode_and_create_submission(
    sandy_ensemble,
    sandy_ids,  
    sandy_label_encoder,
    'sandy'
)

red_submission = decode_and_create_submission(
    red_ensemble,
    red_ids,  
    red_label_encoder,
    'red'
)

loamy_submission = decode_and_create_submission(
    loamy_ensemble,
    loamy_ids,  
    loamy_label_encoder,
    'loamy'
)

black_submission = decode_and_create_submission(
    black_ensemble,
    black_ids,  
    black_label_encoder,
    'black'
)

# Combine all submissions
final_submission = pd.concat([
    clayey_submission,
    sandy_submission,
    red_submission,
    loamy_submission,
    black_submission
], ignore_index=True)




# Sort by id to match expected format
final_submission.to_csv('submission.csv', index=False)
# Save to CSV
print(f"Final submission created with {len(final_submission)} rows")
print(final_submission.head())


print(len(final_submission))
print(len(submission))







