import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import re 

RANDOM_SEED = 42

try:
    train_df = pd.read_csv("/kaggle/input/sparta-2024-data-science-competition/train.csv")
    test_df = pd.read_csv("/kaggle/input/sparta-2024-data-science-competition/test.csv")
    sample_submission_df = pd.read_csv("/kaggle/input/sparta-2024-data-science-competition/sample_submission.csv")
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: Ensure train.csv, test.csv, and sample_submission.csv are in the /kaggle/input/sparta-2024-data-science-competition/ directory.")
    print("Please check your Kaggle dataset path.")
    exit() 
    

train_df.info(verbose=False, show_counts=True)

test_df.info(verbose=False, show_counts=True)

X = train_df.drop('price', axis=1)
y = train_df['price']
X_test = test_df.copy()

test_ids = X_test['id']


combined_df = pd.concat([X, X_test], ignore_index=True)

date_cols = ['host_since', 'first_review', 'last_review']
for col in date_cols:
    combined_df[col] = pd.to_datetime(combined_df[col], errors='coerce')

    current_date = pd.to_datetime('2023-11-01') 
    combined_df[f'{col}_days_ago'] = (current_date - combined_df[col]).dt.days
    
    combined_df = combined_df.drop(columns=[col])

boolean_map = {'t': 1, 'f': 0}
boolean_cols = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'has_availability']
for col in boolean_cols:
    combined_df[col] = combined_df[col].map(boolean_map).fillna(-1) 
    
text_to_drop = ['name', 'description', 'neighborhood_overview', 'host_about', 'host_name', 'host_url', 'host_neighbourhood']
combined_df = combined_df.drop(columns=text_to_drop, errors='ignore')

def extract_bathroom_number(text):
    if pd.isna(text):
        return np.nan
    match = re.search(r'(\d+\.?\d*)\s*(?:bath|baths)|Half-bath', str(text), re.IGNORECASE)
    if match:
        if match.group(1):
            return float(match.group(1))
        elif match.group(0).lower() == 'half-bath':
            return 0.5
    return np.nan

combined_df['bathrooms_numeric'] = combined_df['bathrooms_text'].apply(extract_bathroom_number)
combined_df = combined_df.drop(columns=['bathrooms_text']) 

def count_amenities(amenities_str):
    if pd.isna(amenities_str) or amenities_str == '[]':
        return 0
    return len(amenities_str.strip('[]" ').split('","')) if amenities_str else 0

combined_df['num_amenities'] = combined_df['amenities'].apply(count_amenities)
combined_df = combined_df.drop(columns=['amenities']) 

def count_verifications(verifications_str):
    if pd.isna(verifications_str) or verifications_str == '[]':
        return 0
    return len(verifications_str.strip('[]" ').split("', '")) if verifications_str else 0

combined_df['num_host_verifications'] = combined_df['host_verifications'].apply(count_verifications)
combined_df = combined_df.drop(columns=['host_verifications']) 

combined_df = combined_df.drop(columns=['id'], errors='ignore')

numerical_cols = combined_df.select_dtypes(include=np.number).columns.tolist()

categorical_cols = combined_df.select_dtypes(include='object').columns.tolist()

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough' 
)


X_processed = preprocessor.fit_transform(combined_df.iloc[:len(X)])
X_test_processed = preprocessor.transform(combined_df.iloc[len(X):])

if hasattr(X_processed, "todense"):
    X_processed = X_processed.todense()
if hasattr(X_test_processed, "todense"):
    X_test_processed = X_test_processed.todense()



model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000, 
    learning_rate=0.05,
    max_depth=7,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=RANDOM_SEED,
    n_jobs=-1, 
    tree_method='hist' 
)

model.fit(X_processed, y)


predictions = model.predict(X_test_processed)

predictions[predictions < 0] = 0


submission_df = pd.DataFrame({'id': test_ids, 'price': predictions})
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully.")


