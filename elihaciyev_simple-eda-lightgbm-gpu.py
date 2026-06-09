import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif


# Outputs the full path to all files in the file /kaggle/input
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
original = pd.read_csv('/kaggle/input/fertilizers-original-dataset/Fertilizer Prediction.csv')


# Data overlook
train.head()


# Chek for Null values
train.info()


original.head()


test.head()


# Standardize column names: lowercase and replace spaces with underscores
train.columns = train.columns.str.lower().str.replace(' ', '_')
test.columns = test.columns.str.lower().str.replace(' ', '_')
original.columns = original.columns.str.lower().str.replace(' ', '_')


X_train = pd.concat([train.drop('id', axis=1), original], axis=0).reset_index(drop=True)
X_train.head()


# categorical columns unique values

print(f"The columns:    {train.columns}")
print("-"*60)
print(f"Soil Type: {train['soil_type'].unique()}")
print("-"*60)
print(f"Crop type: {train['crop_type'].unique()}")
print("-"*60)
print(f"Fertilizers: {train['fertilizer_name'].unique()}")
print("-"*60)


def calculate_mutual_info(data, plot=True, discrete_features='auto', random_state=42):

    X = data.iloc[:, :-1].copy()
    y = data.iloc[:, -1].copy()

    label_enc = LabelEncoder()
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = label_enc.fit_transform(X[col].astype(str))

    if y.dtype == 'object' or y.dtype.name == 'category':
        y = label_enc.fit_transform(y.astype(str))

    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=random_state)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    result = pd.DataFrame(mi_series, columns=['Mutual Information'])

    # plot
    if plot:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=result['Mutual Information'], y=result.index, palette='viridis')
        plt.title("Mutual Information Scores")
        plt.xlabel("Score")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.show()

    return result
calculate_mutual_info(X_train)


# A function to visualize the distribution of a categorical feature. To avoid repeating code when analyzing different features, 
# we create a reusable function that plots two charts:
#  - a countplot to show the number of occurrences for each category
#  - a pie chart to visually represent category proportions
def plot_count_and_pie(df, column):
    
    plt.figure(figsize=(14, 6))
    
    # Countplot
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x=column, order=df[column].value_counts().index)
    plt.title(f'Countplot of {column}')
    plt.xticks(rotation=45)
    
    # Pie chart
    plt.subplot(1, 2, 2)
    counts = df[column].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Pie Chart of {column}')
    
    plt.tight_layout()
    plt.show()


plot_count_and_pie(X_train, 'soil_type')


plot_count_and_pie(X_train, 'crop_type')


def feature_engineering(data):
    data = data.copy()

    # Minimum of NPK by line
    min_npk = data[['nitrogen', 'phosphorous', 'potassium']].min(axis=1).replace(0, 1)  # Avoid division by 0

    # NPK Ratio (relative)
    data['N_ratio'] = data['nitrogen'] / min_npk
    data['P_ratio'] = data['phosphorous'] / min_npk
    data['K_ratio'] = data['potassium'] / min_npk

    # Soil Moisture Index (SMI)
    data['SMI'] = data['humidity'] / (data['temparature'] + 0.0000001)

    # Evapotranspiration Index (EvapoIndex)
    data['EvapoIndex'] = data['temparature'] * (1 - data['humidity'] / 100)

    return data


X_train = feature_engineering(X_train)
X_test = feature_engineering(test)
y = X_train.fertilizer_name


X_train.shape


X_train.head()


def preprocessing_data(X, y, X_test):
    # 1. Remove column 'id'
    for df in [X, X_test]:
        if 'id' in df.columns:
            df.drop(columns='id', inplace=True)

    # 2. Remove target column from X if it accidentally included
    target_name = y.name
    if target_name in X.columns:
        X = X.drop(columns=target_name)

    # 3. Label Encoding for categorical features
    for col in ['soil_type', 'crop_type']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        X_test[col] = le.transform(X_test[col])

    # 4. Scaling numerical features
    num_cols = X.select_dtypes(include=['number']).columns.tolist()

    scaler = StandardScaler()
    X_scaled = X.copy()
    X_test_scaled = X_test.copy()

    X_scaled[num_cols] = scaler.fit_transform(X[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    # 5. Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X_scaled, y_encoded, X_test_scaled, label_encoder


X_scaled, y_encoded, X_test_scaled, label_encoder = preprocessing_data(X_train, y, X_test)
X_scaled.head()


X = X_scaled
X_scaled.head()


X_test = X_test_scaled
X_test_scaled.head()


y = y_encoded
label_encoder.inverse_transform(y)


num_classes = len(label_encoder.classes_)

def map3_metric(y_true, y_pred_proba):
   
    top_3 = np.argsort(y_pred_proba, axis=1)[:, -3:][:, ::-1]
    y_true = y_true.reshape(-1, 1)
    return np.mean([1 / (np.where(top == true)[0][0] + 1) if true in top else 0
                    for top, true in zip(top_3, y_true)])


params = {
    "device": "gpu",
    "boosting_type": "gbdt",
    "objective": "multiclass",
    "num_class": num_classes,  
    "learning_rate": 0.03,
    "max_depth": 12,
    "feature_fraction": 0.467,
    "bagging_fraction": 0.86,
    "bagging_freq": 1,
    "min_split_gain": 0.26,
    "lambda_l1": 2.7,
    "lambda_l2": 1.4,
    "verbosity": -1,
    "seed": 13}

# Cross validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
models = []


for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_encoded)):
    print(f"Fold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100)
    ])

    y_val_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
    score = map3_metric(y_val, y_val_pred_proba)
    scores.append(score)
    models.append(model) # Save model for each fold
    
    print(f"MAP@3 Score: {score:.4f}")

print(f"\nAverage MAP@3 over 5 folds: {np.mean(scores):.4f}")



def create_submission(models, X_test, label_encoder, filename="submission.csv", top_n=3, separator=" "):
    # We average the probabilities across all models
    preds = np.zeros((X_test.shape[0], len(label_encoder.classes_)))
    for model in models:
        preds += model.predict(X_test, num_iteration=model.best_iteration)
    preds /= len(models)  

    
    top_n_idx = np.argsort(preds, axis=1)[:, -top_n:][:, ::-1]

    
    top_n_labels = label_encoder.inverse_transform(top_n_idx.flatten()).reshape(top_n_idx.shape)
    prediction_strings = [separator.join(row) for row in top_n_labels]

    
    ids = test['id']
    submission = pd.DataFrame({'id': ids, 'Fertilizer Name': prediction_strings})
    submission.to_csv(filename, index=False)
    print(f"âœ… Submission file '{filename}' created successfully.")

    return submission



submission = create_submission(models, X_test, label_encoder, filename="submission.csv", top_n=3, separator=" ")


def plot_feature_importance(models, top_n=20, importance_type='gain'):

    # We collect importances from all models
    all_importances = []

    for i, model in enumerate(models):
        imp_df = pd.DataFrame({
            'feature': model.feature_name(),
            'importance': model.feature_importance(importance_type=importance_type),
            'model': f'fold_{i + 1}'
        })
        all_importances.append(imp_df)

    # Let's unite
    full_importance_df = pd.concat(all_importances, axis=0)

    # Aggregate: average importance across all models
    mean_importance = (
        full_importance_df
        .groupby('feature')['importance']
        .mean()
        .reset_index()
        .sort_values(by='importance', ascending=False)
    )

    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=mean_importance.head(top_n),
        y='feature',
        x='importance',
        palette='coolwarm'
    )
    plt.title(f'Top {top_n} Feature Importances ({importance_type}) averaged over folds')
    plt.xlabel('Mean Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

    return mean_importance


plot_feature_importance(models)

