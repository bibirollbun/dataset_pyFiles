import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import KNNImputer
import xgboost as xgb
import lightgbm as lgb


import warnings
warnings.filterwarnings("ignore")
train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
train_df_personality = train_df[['Personality']].copy()
#original_df = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv')
#original_df_personality = original_df[['Personality']].copy()
#orignial_train_df_personality = pd.concat([train_df_personality, original_df_personality], ignore_index=True)


train_df


print("Train Shape :", train_df.shape)
print("Test Shape :",test_df.shape)
#print("Original Shape :",original_df.shape)


print(train_df.isna().sum())        


print(test_df.isna().sum())   


#print(original_df.isna().sum())        


train_df.info()


train_ID = train_df['id']
test_ID = test_df['id']

train_df.drop("id", axis = 1, inplace = True)
test_df.drop("id", axis = 1, inplace = True)
'''
expanded_train_df = pd.concat([train_df, original_df], ignore_index=True)

ntrain = expanded_train_df.shape[0] 
ntest = test_df.shape[0] 
y_train = expanded_train_df['Personality'].map({'Extrovert': 1, 'Introvert': 0}).values 

all_data = pd.concat((expanded_train_df, test_df)).reset_index(drop=True)
all_data.drop(['Personality'], axis=1, inplace=True)

print(f"Original training size: {train_df.shape[0]}")
print(f"Additional data size: {original_df.shape[0]}")
print(f"New training size: {ntrain}")
print(f"Test size: {ntest}")
all_data.info()
'''
ntrain = train_df.shape[0] 
ntest = test_df.shape[0] 
y_train = train_df['Personality'].map({'Extrovert': 1, 'Introvert': 0}).values 

all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
all_data.drop(['Personality'], axis=1, inplace=True)
all_data.info()


all_data['social_attend_bin'] = pd.qcut(
    all_data['Social_event_attendance'], 
    q=[0, 0.25, 0.5, 0.75, 1.0], 
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)


# 2. Define a function to fill missing values in Time\_spent\_Alone with the median within each group
def fill_by_group_median(df, group_col, target_col):
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

# 3.Perform group-wise filling of missing values
all_data['Time_spent_Alone'] = fill_by_group_median(
    all_data, group_col='social_attend_bin', target_col='Time_spent_Alone'
)


all_data.drop(columns=['social_attend_bin'], inplace=True)

all_data.info()


all_data['Going_outside_bin'] = pd.qcut(
    all_data['Going_outside'], 
    q=[0, 0.25, 0.5, 0.75, 1.0], 
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)


# 2. Define a function to fill missing values in Time\_spent\_Alone with the median within each group
def fill_by_group_median(df, group_col, target_col):
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

# 3. Perform group-wise filling of missing values
all_data['Time_spent_Alone'] = fill_by_group_median(
    all_data, group_col='Going_outside_bin', target_col='Time_spent_Alone'
)


all_data.drop(columns=['Going_outside_bin'], inplace=True)

all_data.info()


all_data['Going_outside_bin'] = pd.qcut(
    all_data['Going_outside'], 
    q=[0, 0.25, 0.5, 0.75, 1.0], 
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)


# 2. Define a function to fill missing values in Time\_spent\_Alone with the median within each group
def fill_by_group_median(df, group_col, target_col):
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

# 3. Perform group-wise filling of missing values
all_data['Social_event_attendance'] = fill_by_group_median(
    all_data, group_col='Going_outside_bin', target_col='Social_event_attendance'
)


all_data.drop(columns=['Going_outside_bin'], inplace=True)

all_data.info()


all_data['Friends_circle_bin'] = pd.qcut(
    all_data['Friends_circle_size'], 
    q=[0, 0.25, 0.5, 0.75, 1.0], 
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)


# 2. Define a function to fill missing values in Time\_spent\_Alone with the median within each group
def fill_by_group_median(df, group_col, target_col):
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

# 3. Perform group-wise filling of missing values
all_data['Social_event_attendance'] = fill_by_group_median(
    all_data, group_col='Friends_circle_bin', target_col='Social_event_attendance'
)


all_data.drop(columns=['Friends_circle_bin'], inplace=True)

all_data.info()


all_data['Post_frequency_bin'] = pd.qcut(
    all_data['Post_frequency'], 
    q=[0, 0.25, 0.5, 0.75, 1.0], 
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)


# 2. Define a function to fill missing values in Time\_spent\_Alone with the median within each group

def fill_by_group_median(df, group_col, target_col):
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

# 3. Perform group-wise filling of missing values
all_data['Social_event_attendance'] = fill_by_group_median(
    all_data, group_col='Post_frequency_bin', target_col='Social_event_attendance'
)

all_data.drop(columns=['Post_frequency_bin'], inplace=True)

all_data.info()


def fill_missing_by_quantile_group(df, group_source_col, target_col, quantiles=[0, 0.25, 0.5, 0.75, 1.0], labels=None):
    """
        Fill missing values in `target_col` by grouping based on the quantiles of `group_source_col`, and using the median of each group to impute missing values.
        
        **Parameters:**
        - `df` (`pd.DataFrame`): The original dataset
        - `group_source_col` (`str`): The numeric column used for grouping
        - `target_col` (`str`): The target column with missing values to be filled
        - `quantiles` (`list`): Quantile breakpoints for grouping (default is quartiles)
        - `labels` (`list`): Labels for each group (default is auto-generated as Q1/Q2/...)
        
        **Returns:**
        - `pd.DataFrame`: The DataFrame with missing values filled (modifies in place)

    """
    # Automatically Generate Group Labels
    if labels is None:
        labels = [f'Q{i+1}' for i in range(len(quantiles)-1)]

    temp_bin_col = f'{group_source_col}_bin'

    # Step 1: Create Grouping Column
    df[temp_bin_col] = pd.qcut(df[group_source_col], q=quantiles, labels=labels)

    # Step 2: Fill Missing Values Within Groups Using the Median
    df[target_col] = df[target_col].fillna(df.groupby(temp_bin_col)[target_col].transform('median'))

    # Step 3: Delete Temporary Columns
    df.drop(columns=[temp_bin_col], inplace=True)

    return df

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


train_df = all_data[:ntrain]
test_df = all_data[ntrain:]


def create_enhanced_features(df):
    df = df.copy()
    
    numeric_columns = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                      'Friends_circle_size', 'Post_frequency']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    df['social_score'] = (df['Social_event_attendance'] + df['Going_outside'] + 
                         df['Friends_circle_size'] + df['Post_frequency']) / 4
    df['introversion_score'] = df['Time_spent_Alone'] - df['Social_event_attendance']
    df['social_going_interaction'] = df['Social_event_attendance'] * df['Going_outside']
    df['time_vs_social'] = df['Time_spent_Alone'] / (df['Social_event_attendance'] + 1)
    df['social_intensity'] = df['Social_event_attendance'] + df['Going_outside']
    df['social_score_squared'] = df['social_score'] ** 2
    df['social_score_cubed'] = df['social_score'] ** 3
    df['top_features_combo'] = df['social_score'] * df['introversion_score']
    
    df['social_butterfly'] = ((df['Social_event_attendance'] > df['Social_event_attendance'].quantile(0.75)) & 
                                (df['Friends_circle_size'] > df['Friends_circle_size'].quantile(0.75))).astype(int)
    df['balanced_social'] = ((df['Time_spent_Alone'] > 2) & (df['Social_event_attendance'] > 2)).astype(int)
    df['online_preference'] = df['Post_frequency'] / (df['Going_outside'] + 1)
    
    df['alone_to_friends_ratio'] = df['Time_spent_Alone'] / (df['Friends_circle_size'] + 1)
    df['post_to_social_ratio'] = df['Post_frequency'] / (df['Social_event_attendance'] + 1)
    df['friends_to_going_ratio'] = df['Friends_circle_size'] / (df['Going_outside'] + 1)
    df['social_to_alone_ratio'] = df['Social_event_attendance'] / (df['Time_spent_Alone'] + 1)
    df['digital_vs_physical_ratio'] = df['Post_frequency'] / (df['Social_event_attendance'] + df['Going_outside'] + 1)
    
    df['high_alone_time'] = (df['Time_spent_Alone'] >= 6).fillna(False).astype(int)
    df['low_social_events'] = (df['Social_event_attendance'] <= 2).fillna(False).astype(int)
    df['small_friend_circle'] = (df['Friends_circle_size'] <= 5).fillna(False).astype(int)
    df['rare_going_out'] = (df['Going_outside'] <= 2).fillna(False).astype(int)
    df['low_posting'] = (df['Post_frequency'] <= 2).fillna(False).astype(int)
    df['high_posting'] = (df['Post_frequency'] >= 7).fillna(False).astype(int)
    
    df['classic_introvert'] = ((df['Time_spent_Alone'] >= 5) & 
                              (df['Social_event_attendance'] <= 3) & 
                              (df['Friends_circle_size'] <= 6)).fillna(False).astype(int)
    
    df['classic_extrovert'] = ((df['Time_spent_Alone'] <= 2) & 
                              (df['Social_event_attendance'] >= 6) & 
                              (df['Friends_circle_size'] >= 10)).fillna(False).astype(int)
    
    df['social_paradox'] = ((df['Time_spent_Alone'] >= 5) & 
                           (df['Social_event_attendance'] >= 6)).fillna(False).astype(int)
    
    df['digital_introvert'] = ((df['Post_frequency'] >= 6) & 
                              (df['Going_outside'] <= 3)).fillna(False).astype(int)
    
    df['introversion_score_squared'] = df['introversion_score'] ** 2
    df['time_vs_social_squared'] = df['time_vs_social'] ** 2
    df['social_intensity_squared'] = df['social_intensity'] ** 2
    
    df['alone_social_product'] = df['Time_spent_Alone'] * df['Social_event_attendance']
    df['friends_post_product'] = df['Friends_circle_size'] * df['Post_frequency']
    df['going_post_product'] = df['Going_outside'] * df['Post_frequency']
    
    df['social_level'] = pd.cut(df['Social_event_attendance'], 
                               bins=[0, 2, 5, 8, 10], 
                               labels=[0, 1, 2, 3], include_lowest=True)
    df['social_level'] = df['social_level'].fillna(1).astype(int)  # Fill NaN with middle category
    
    df['alone_level'] = pd.cut(df['Time_spent_Alone'], 
                              bins=[0, 2, 5, 8, 10], 
                              labels=[0, 1, 2, 3], include_lowest=True)
    df['alone_level'] = df['alone_level'].fillna(1).astype(int)  # Fill NaN with middle category
    
    engineered_features = [col for col in df.columns if col not in numeric_columns + ['Stage_fear', 'Drained_after_socializing', 'Introversion']]
    
    for col in engineered_features:
        if df[col].dtype in ['float64', 'float32']:
            df[col] = df[col].fillna(df[col].median())
        elif df[col].dtype in ['int64', 'int32']:
            df[col] = df[col].fillna(0)
    
    return df

train_df = create_enhanced_features(train_df)
test_df = create_enhanced_features(test_df)


base_numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                     'Friends_circle_size', 'Post_frequency']

existing_engineered = ['social_score', 'introversion_score', 'social_going_interaction', 
                      'time_vs_social', 'social_intensity', 'social_score_squared',
                      'social_score_cubed', 'top_features_combo', 'social_butterfly',
                      'balanced_social', 'online_preference']

new_engineered = ['alone_to_friends_ratio', 'post_to_social_ratio', 'friends_to_going_ratio',
                 'social_to_alone_ratio', 'digital_vs_physical_ratio', 'high_alone_time',
                 'low_social_events', 'small_friend_circle', 'rare_going_out', 'low_posting',
                 'high_posting', 'classic_introvert', 'classic_extrovert', 'social_paradox',
                 'digital_introvert', 'introversion_score_squared', 'time_vs_social_squared',
                 'social_intensity_squared', 'alone_social_product', 'friends_post_product',
                 'going_post_product', 'social_level', 'alone_level']

# Combine all numeric features
numeric_cols = base_numeric_cols + existing_engineered + new_engineered

binary_cols = ['Stage_fear', 'Drained_after_socializing']


'''
for col in binary_cols:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
    test_df[col] = test_df[col].fillna(train_df[col].mode()[0])

knn_imputer = KNNImputer(n_neighbors=5)
train_df[numeric_cols] = knn_imputer.fit_transform(train_df[numeric_cols])
test_df[numeric_cols] = knn_imputer.transform(test_df[numeric_cols])
'''


from sklearn.preprocessing import LabelEncoder

binary_encoders = {}
for col in binary_cols:
    le = LabelEncoder()
    train_df[col + '_encoded'] = le.fit_transform(train_df[col])
    test_df[col + '_encoded'] = le.transform(test_df[col])
    binary_encoders[col] = le
    
# Drop original binary columns
train_df.drop(columns=binary_cols, inplace=True)
test_df.drop(columns=binary_cols, inplace=True)

# Update feature list
feature_cols = numeric_cols + [col + '_encoded' for col in binary_cols]


train_df = pd.merge(train_df, train_df_personality, left_index=True, right_index=True)


train_df


from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])


le_target = LabelEncoder()
train_df['Personality'] = le_target.fit_transform(train_df['Personality'])


print(train_df.isna().sum())  
print('*'*50)
print(test_df.isna().sum())  


X = train_df.drop(['Personality'], axis=1)
y = train_df['Personality']

print("Using all features:", X.columns.tolist())
print("Number of features:", X.shape[1])


from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier

def optimize_feature_selection(X, y, feature_names):
    """
    Intelligent feature selection based on importance and performance
    """
    print("=== FEATURE SELECTION OPTIMIZATION ===")
    
    # Method 1: Keep only features with importance > threshold
    print("\n1. Filtering by Feature Importance...")
    
    # You can get feature importance from your ensemble
    # For now, let's use the importance values you provided
    high_importance_features = [
        'social_score_squared', 'social_score_cubed', 'social_score', 
        'top_features_combo', 'time_vs_social', 'time_vs_social_squared',
        'social_to_alone_ratio', 'social_going_interaction', 'introversion_score',
        'going_post_product', 'alone_to_friends_ratio', 'social_intensity',
        'Drained_after_socializing_encoded', 'social_intensity_squared',
        'Stage_fear_encoded', 'friends_post_product', 'Time_spent_Alone',
        'Post_frequency', 'Going_outside', 'Social_event_attendance'
    ]
    
    # Filter features that exist in your dataset
    available_high_importance = [f for f in high_importance_features if f in feature_names]
    print(f"High importance features available: {len(available_high_importance)}")
    
    # Method 2: Statistical feature selection
    print("\n2. Statistical Feature Selection...")
    selector = SelectKBest(f_classif, k=25)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"Statistically selected features: {len(selected_features)}")

    X_statistical = X[selected_features]
    
    # Method 3: Combine both approaches
    print("\n3. Combined Selection...")
    combined_features = list(set(available_high_importance + selected_features))
    print(f"Combined features: {len(combined_features)}")
    
    return X[combined_features], combined_features

# Method 4: Progressive feature selection
def progressive_feature_selection(X_train, y_train, X_val, y_val, base_features, all_features):
    """
    Add features one by one and keep only those that improve performance
    """
    print("\n=== PROGRESSIVE FEATURE SELECTION ===")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # Start with base features (your original high-performing features)
    current_features = base_features.copy()
    current_score = 0
    
    # Test base features
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_temp.fit(X_train[current_features], y_train)
    current_score = accuracy_score(y_val, rf_temp.predict(X_val[current_features]))
    print(f"Base features score: {current_score:.4f}")
    
    # Try adding each remaining feature
    remaining_features = [f for f in all_features if f not in current_features]
    
    for feature in remaining_features:
        test_features = current_features + [feature]
        
        # Test performance with this feature added
        rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_temp.fit(X_train[test_features], y_train)
        test_score = accuracy_score(y_val, rf_temp.predict(X_val[test_features]))
        
        # Keep feature if it improves performance
        if test_score > current_score + 0.001:  # Small threshold for improvement
            current_features.append(feature)
            current_score = test_score
            print(f"Added {feature}: {test_score:.4f}")
    
    print(f"Final progressive selection: {len(current_features)} features")
    return current_features

# IMPLEMENTATION - Add this to your notebook after feature creation

# Option 1: Simple importance-based selection
print("=== QUICK FEATURE OPTIMIZATION ===")

# Define your core high-performing features
core_features = [
    'social_score', 'social_score_squared', 'social_score_cubed',
    'top_features_combo', 'time_vs_social', 'time_vs_social_squared',
    'introversion_score', 'social_going_interaction', 'social_to_alone_ratio',
    'social_intensity', 'going_post_product', 'alone_to_friends_ratio',
    'Drained_after_socializing_encoded', 'Stage_fear_encoded',
    'Time_spent_Alone', 'Post_frequency', 'Going_outside', 'Social_event_attendance'
]

# Filter to only features that exist in your dataset
available_core_features = [f for f in core_features if f in feature_cols]
print(f"Using {len(available_core_features)} core features")

# Option 2: Use this if you want to be more aggressive
X_optimized, optimized_features = optimize_feature_selection(X, y, feature_cols)
print(f"Optimized feature count: {len(optimized_features)}")

# Option 3: Manual selection based on your importance analysis
manual_selection = [
    # Top 15 features from your importance analysis
    'social_score_squared', 'social_score_cubed', 'social_score', 
    'top_features_combo', 'time_vs_social', 'time_vs_social_squared',
    'social_to_alone_ratio', 'social_going_interaction', 'introversion_score',
    'going_post_product', 'alone_to_friends_ratio', 'social_intensity',
    'Drained_after_socializing_encoded', 'social_intensity_squared',
    'Stage_fear_encoded'
]

print(f"Manual selection: {len(manual_selection)} features")

# RECOMMENDED: Try each approach and compare
print("\n=== TESTING DIFFERENT FEATURE SETS ===")

# Test 1: Core features only
print("1. Testing core features...")
X_core = X[available_core_features]

# Test 2: Manual selection
print("2. Testing manual selection...")
X_manual = X[manual_selection]

# Test 3: All features (your current approach)
print("3. Testing all features...")
X_all = X[feature_cols]

print(f"Core features: {X_core.shape[1]} features")
print(f"Manual selection: {X_manual.shape[1]} features") 
print(f"All features: {X_all.shape[1]} features")


X = X_core


from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
def create_ensemble_model():
    
    rf = RandomForestClassifier(
        n_estimators=934,
        max_depth=4,
        min_samples_split=12,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=False,
        random_state=42,
        n_jobs=-1
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=160,
        learning_rate=0.02609055139581458,
        max_depth=3,
        min_samples_split=9,
        min_samples_leaf=4,
        subsample=0.8068071565510975,
        max_features ='log2',
        random_state=42
    )
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=351,
        learning_rate=0.022036372065623862,
        max_depth=8,
        min_child_weight=2,
        subsample=0.6175825740397892,
        colsample_bytree=0.6311270954678605,
        reg_alpha=0.27911678555917296,
        reg_lambda=0.731356525648762,
        gamma=0.3201746940450867,
        random_state=42
    )
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=191,
        learning_rate=0.04530128105537246,
        max_depth=3,
        num_leaves=224,
        min_child_samples=12,
        subsample=0.8661050589090472,
        colsample_bytree=0.7801667636972905,
        reg_alpha=0.9071045310780,
        reg_lambda=0.731356525648762,
        random_state=42,
        verbose=-1
    )
    
    # Voting classifier
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('xgb', xgb_model),
            ('lgb', lgb_model)
        ],
        voting='soft'
    )
    
    return ensemble

# IMPROVEMENT 7: Stratified Cross-Validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Create and train ensemble
ensemble_model = create_ensemble_model()
ensemble_model.fit(X_train, y_train)

# Evaluate
y_pred = ensemble_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Ensemble Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))


val_probs = ensemble_model.predict_proba(X_test)[:, 1]
best_threshold = 0.5
best_acc = 0

for threshold in np.arange(0.1, 0.99, 0.001):
    preds = (val_probs >= threshold).astype(int)
    acc = accuracy_score(y_test, preds)
    if acc > best_acc:
        best_acc = acc
        best_threshold = threshold

print(f"Best threshold: {best_threshold:.4f}")
print(f"Best accuracy: {best_acc:.4f}")


test_probs = ensemble_model.predict_proba(X_test)[:, 1]
optimized_preds = (test_probs >= best_threshold).astype(int)
print(f"Optimized threshold accuracy: {accuracy_score(y_test, optimized_preds):.4f}")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(ensemble_model, X, y, cv=skf, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")


test_X = test_df[available_core_features]  # .drop(['id'], axis=1)
test_probs = ensemble_model.predict_proba(test_X)[:, 1]
test_predictions = (test_probs >= best_threshold).astype(int)


test_predictions_original = le_target.inverse_transform(test_predictions)
test_df['Personality_pred_threshold'] = test_predictions_original
submission = pd.DataFrame({
    'id': test_ID,
    'Personality': test_predictions_original  
})
submission.to_csv('submission.csv', index=False)
print("submission with all features created!")


if hasattr(ensemble_model.estimators_[0], 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': ensemble_model.estimators_[0].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nMost Important Features:")
    print(feature_importance)


submission


personality_counts = submission['Personality'].value_counts()
print("Personality Distribution:")
print(personality_counts)




