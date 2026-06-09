# MOC Competition: Mental Health - Advanced EDA
# Big acknowledgments to Himan Manduja and their notebook, which I have modified.

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
import joblib
import warnings

# Suppress FutureWarning from pandas
warnings.simplefilter(action='ignore', category=FutureWarning)


# Set plot style
sns.set_style("darkgrid")

# 2. Load Data
try:
    df = pd.read_csv('/kaggle/input/moc-competition-mental-health/train.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: train.csv not found. Please make sure the dataset is in the correct directory.")
    exit()


# 3. Initial Data Inspection
df.head()



print("\n\nDataset Info:\n")
df.info()

print("\n\nDropping 'id' and 'Name' columns as they are not needed for modeling.")
df.drop(columns=['id', 'Name'], inplace=True)


print("\n--- Null Value Analysis ---")
null_values = df.isnull().sum()
null_percentage = (null_values / len(df)) * 100
null_df = pd.DataFrame({'Null Count': null_values, 'Null Percentage': null_percentage})
null_df = null_df[null_df['Null Count'] > 0].sort_values(by='Null Count', ascending=False)
print("Columns with null values:")
print(null_df)

# Visualize missing values heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap', fontsize=16)
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df.sort_values('Working Professional or Student').isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap, Sorted by Student -> Working Professional', fontsize=16)
plt.show()


plt.figure(figsize=(7, 5))
sns.countplot(x='Depression', data=df, palette='pastel')
plt.title('Distribution of Depression', fontsize=16)
plt.xlabel('Depression (0: No, 1: Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, bins=30, color='skyblue')
plt.title('Distribution of Age', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

# Categorical Features
categorical_features = ['Gender', 'Working Professional or Student', 'Sleep Duration', 'Dietary Habits', 
                        'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness', 'Degree']

for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(y=feature, data=df, order=df[feature].value_counts().index, palette='viridis')
    plt.title(f'Distribution of {feature}', fontsize=16)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel(feature, fontsize=12)
    plt.tight_layout()
    plt.show()


print("\n--- Bivariate Analysis: Feature vs. Depression ---")

# Categorical Features vs. Depression
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=feature, hue='Depression', data=df, palette='magma')
    plt.title(f'Depression by {feature}', fontsize=16)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Depression', labels=['No', 'Yes'])
    plt.tight_layout()
    plt.show()

# Numerical Features vs. Depression
numerical_features = ['Age', 'Financial Stress', 'Work/Study Hours']

for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Depression', y=feature, data=df, palette='coolwarm')
    plt.title(f'Depression by {feature}', fontsize=16)
    plt.xlabel('Depression (0: No, 1: Yes)', fontsize=12)
    plt.ylabel(feature, fontsize=12)
    plt.show()



df[['Depression', 'Working Professional or Student']].value_counts()/len(df)


# 7. Correlation Analysis
print("\n--- Correlation Analysis ---")
# Create a copy for encoding
df_corr = df.copy()

# Label encode object columns for correlation matrix
for col in df_corr.select_dtypes(include=['object']).columns:
    df_corr[col] = df_corr[col].astype('category').cat.codes

# Impute NaNs with median for correlation matrix calculation
for col in df_corr.columns:
    if df_corr[col].isnull().any():
        df_corr[col] = df_corr[col].fillna(df_corr[col].median())

#Create mask to look prettier
corr=df_corr.corr()
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

# Correlation Matrix
plt.figure(figsize=(18, 14))
sns.heatmap(corr, mask=mask, annot=True, cmap='bwr', fmt='.2f', linewidths=.5, vmin=-0.7, vmax=0.7)
plt.title('Correlation Matrix of All Features', fontsize=20)
plt.show()

print("\nEDA notebook complete. ðŸ“ˆ")


### "Working Professional or Student" deeper dive
# It has one of the highest correlations with depression, 
# so I want to examine any inconsistencies and improve the feature

WProfessional_null = df[df['Working Professional or Student']=='Working Professional']['Profession'].isnull().mean()
print(f"Working Professional null 'Profession': {WProfessional_null*100:.2f}%")

Student_null = df[df['Working Professional or Student']=='Student']['Profession'].isnull().mean()
print(f"Student null 'Profession': {Student_null*100:.2f}%")

print(df[df['Working Professional or Student']=='Student']['Profession'].value_counts())

print('\n\nThe students are likely indicating their degree, which is arguably not informative.')
print('Will update all students to have "student" profession and')
print('Working Professional nulls with "Not Applicable"')


print(df["Dietary Habits"].value_counts())
print(f"\nNull: {df['Dietary Habits'].isnull().sum()}")

print("\nWill only keep [Unhealthy, Moderate, and Healthy] and replace all others (including null) with mode")


print(df["Sleep Duration"].value_counts())
print(f"\nNull: {df['Sleep Duration'].isnull().sum()}")

print("\nWill only keep [<5, 5-6, 7-8, >8] and replace all others (including null) with mode")
print("Most of the non-standard answers are less than five hours, anyway, so this is an okay appx.")
print("\nDo the majority of people really sleep less than 5 hours a day??")


def preprocess_data(df, n_variables=10,
                    merge_student_professional=True,
                    sleep=True, diet=True, degree=True): #, encoders=None, is_training=False):
    """
    Preprocesses data for training or prediction.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        encoders (dict): A dictionary of fitted LabelEncoders. Required if is_training=False.
        is_training (bool): If True, fits new encoders. If False, uses provided encoders.
        
    Returns:
        pd.DataFrame: The preprocessed dataframe.
        dict: The fitted label encoders (only if is_training=True).
    """
    # print(f"Preprocessing data... Mode: {'Training' if is_training else 'Prediction'}")
    processed_df = df.copy()
    processed_df.drop(['CGPA', 'City'], axis=1, inplace=True)
    
    
    # --- Step 1: Handle Missing Values ---
    # Fill in the following with mode
    for col in ['Degree', 'Financial Stress', 'Dietary Habits', 'Sleep Duration']:
        if processed_df[col].isnull().any():
            mode_val = processed_df[col].mode()[0]
            processed_df[col].fillna(mode_val, inplace=True)

    
    #These categories are very similar, but reported from different tests
    if merge_student_professional:
        # Merge working professional / student features
        # study satisfaction -> job satisfaction
        # academic pressure -> work pressure
        # student==True -> Profession = Student
        mask = processed_df['Working Professional or Student']=='Student'
        processed_df.loc[mask, 'Work Pressure'] = processed_df.loc[mask, 'Academic Pressure']
        processed_df.loc[mask, 'Job Satisfaction'] = processed_df.loc[mask, 'Study Satisfaction']
        processed_df.loc[mask, 'Profession'] ='Student'
        # Drop the associated features, as well as "Working Professional or Student" to avoid co-linearity
        processed_df.drop(['Academic Pressure', 'Study Satisfaction', 'Working Professional or Student'], axis=1, inplace=True)
    else:
        for feature in ['Academic Pressure', 'Work Pressure', 'Job Satisfaction', 'Study Satisfaction']:
            processed_df[feature].fillna(0, inplace=True)
    processed_df['Profession'].fillna('Not Applicable', inplace=True)
    
    # --- Step 2: Encode Categorical Variables ---
    if degree:
        categorical_cols = ['Profession', 'Degree']
    else:
        print('drop degree')
        categorical_cols = ['Profession']
        processed_df.drop('Degree', axis=1, inplace=True)

    encoder = OneHotEncoder(sparse_output=False, min_frequency=0.02, dtype=bool)
    one_hot_encoded = encoder.fit_transform(processed_df[categorical_cols])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_cols))
    processed_df = pd.concat([processed_df, one_hot_df], axis=1)    
    processed_df = processed_df.drop(categorical_cols, axis=1)


    # --- Step 3: Encode Ordinal Variables ---
    di = {'Unhealthy': 0,
          'Moderate': 1,
          'Healthy': 2
          }
    processed_df['Dietary Habits'] = pd.to_numeric(processed_df['Dietary Habits'].replace(di), downcast='integer', errors='coerce')
    mask = processed_df['Dietary Habits']>2 # or processed_df['Dietary Habits']<0
    processed_df.loc[mask, 'Dietary Habits'] = 1
    processed_df['Dietary Habits'].fillna(1, inplace=True)


    di = {'Less than 5 hours': 0,
          '7-8 hours': 1,
          'More than 8 hours': 2
          }
    processed_df['Sleep Duration'] = pd.to_numeric(processed_df['Sleep Duration'].replace(di), downcast='integer', errors='coerce')
    mask = processed_df['Sleep Duration']>2
    processed_df.loc[mask, 'Sleep Duration'] = 0
    processed_df['Sleep Duration'].fillna(0, inplace=True)
    

   # Drop uninformative categories
    if not sleep:
        processed_df.drop(['Sleep Duration'], axis=1, inplace=True)
    if not diet:
        processed_df.drop(['Dietary Habits'], axis=1, inplace=True)
    
    # if is_training:
    encoders = {}
    categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns
    # categorical_cols = ['Gender',
    #                 #'Working Professional or Student',
    #                 'Have you ever had suicidal thoughts ?',
    #                 'Family History of Mental Illness'
    #                ]
    for col in categorical_cols:
        le = LabelEncoder() #(min_frequency=0.01, drop='first')
        processed_df[col] = le.fit_transform(processed_df[col])
        encoders[col] = { 'classes': le.classes_.tolist(), 'unknown': -1 }
        print(col)
    # print("Fitted and saved new label encoders.")

    #Binarize certain categories
    cols = ['Gender', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
    for col in cols:
        processed_df[col] = processed_df[col].astype(bool)

    #Center others
    numerical_cols = processed_df.select_dtypes(include=['float64']).columns
    for col in numerical_cols:
        processed_df[col] = StandardScaler().fit_transform(processed_df[col].values.reshape(-1, 1))
    
    return processed_df, encoders

    # else:
    #     if encoders is None:
    #         raise ValueError("Encoders must be provided for prediction mode.")
    #     for col in categorical_cols:
    #         le = LabelEncoder()
    #         le.classes_ = np.array(encoders[col]['classes'])
    #         # Handle unseen values in test data by mapping them to a special 'unknown' value
    #         processed_df[col] = processed_df[col].map(lambda s: s if s in le.classes_ else '<unknown>')
    #         # Add '<unknown>' to the classes if it's not there
    #         if '<unknown>' not in le.classes_:
    #              le.classes_ = np.append(le.classes_, '<unknown>')
    #         processed_df[col] = le.transform(processed_df[col])
    #     print("Applied saved label encoders.")
    #     return processed_df



processed_df, encoders = preprocess_data(df, merge_student_professional=True, degree=False,diet=True)

processed_df.head(10)
#processed_df.dtypes


sum(processed_df['Dietary Habits'] > 2)



def train_and_save_model(**preprocess_kwargs):
    """
    Loads training data, preprocesses it, evaluates the model, 
    then retrains on all data and saves the model and encoders.
    """
    print("--- Starting Model Training and Evaluation ---")
    # Load data
    df_train = pd.read_csv('/kaggle/input/moc-competition-mental-health/train.csv')
    
    # Preprocess training data
    X = df_train.drop(['id', 'Name', 'Depression'], axis=1)
    y = df_train['Depression']
    X_processed, encoders = preprocess_data(X, **preprocess_kwargs)

    # Save encoders to a JSON file
    # with open('label_encoders.json', 'w') as f:
    #     json.dump(encoders, f, indent=4)
    # print("Label encoders saved to 'label_encoders.json'")

    # --- Model Evaluation Step ---
    print("\n--- Evaluating Model Performance on a Validation Set ---")
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle class imbalance
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    
    # Initialize the XGBoost model
    xgb_classifier = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=5000,
        max_depth=20,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        booster='gbtree',
        # gamma=3,
        # reg_lambda=0.1,
        # reg_alpha=0.1,
        random_state=42
    )
    
    # Train on the training subset
    xgb_classifier.fit(X_train, y_train)
    
    # Predict on the validation set
    y_pred_val = xgb_classifier.predict(X_val)
    
    # Show reports
    print(f"Validation Accuracy: {accuracy_score(y_val, y_pred_val):.4f}")
    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_pred_val, target_names=['Not Depressed', 'Depressed']))
    
    # Show confusion matrix
    cm = confusion_matrix(y_val, y_pred_val)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Depressed', 'Depressed'], 
                yticklabels=['Not Depressed', 'Depressed'])
    plt.title('Validation Confusion Matrix', fontsize=16)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.show()

    # --- Final Model Training ---
    print("\n--- Retraining Model on Full Dataset for Saving ---")
    # Re-initialize and train on the entire dataset to build the most robust model
    
    # full_data_scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]
    # final_model = xgb.XGBClassifier(
    #     objective='binary:logistic',
    #     scale_pos_weight=full_data_scale_pos_weight,
    #     use_label_encoder=False,
    #     eval_metric='logloss',
    #     n_estimators=5000,
    #     max_depth=20,
    #     learning_rate=0.1,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     random_state=42
    # )
    # final_model.fit(X_processed, y)
    
    # # # Save the final trained model
    # # joblib.dump(final_model, 'xgboost_model.joblib')
    # # print("Final model saved to 'xgboost_model.joblib'")
    # print("--- Training Complete ---")

    return xgb_classifier.get_booster()





preprocess_kwargs = {"merge_student_professional":True, "degree":True, 'sleep':True, 'diet':True}

booster = train_and_save_model(**preprocess_kwargs)


xgb.plot_importance(booster, max_num_features=16, importance_type="weight")
xgb.plot_importance(booster, max_num_features=16, importance_type="cover")
xgb.plot_importance(booster, max_num_features=16, importance_type="gain")


from sklearn.linear_model import LogisticRegression

df_train = pd.read_csv('/kaggle/input/moc-competition-mental-health/train.csv')
print(df_train.columns)

y = df_train['Depression']


def plot_lr(X, clf, label):
    x = np.linspace(min(X), max(X), 1000)
    y = clf.predict_proba(x)
    plt.figure()
    plt.plot(x,y[:,1])
    plt.ylim(0,1)
    plt.legend()
    plt.xlabel(col)
    plt.ylabel('Probability of Depression')
    plt.show()

for col in ['Age', 'Work/Study Hours', 'Job Satisfaction']:
    print(f"\n\n{col}")
    X = df_train[col].fillna(0).values.reshape(-1, 1)
    clf = LogisticRegression(random_state=0).fit(X, y)
    # clf.predict(X)
    # clf.predict_proba(X)
    print(f"Score\n{clf.score(X, y)}")
    plot_lr(X, clf, col)





# --- PART 4: BATCH PREDICTION AND SUBMISSION FILE CREATION ---

def create_submission_file():
    """
    Loads the test data, predicts outcomes, and creates a submission.csv file.
    """
    print("\n--- Creating Submission File ---")
    # Load test data and encoders
    df_test = pd.read_csv('/kaggle/input/moc-competition-mental-health/test.csv')
    test_ids = df_test['id']
    X_test = df_test.drop(['id', 'Name'], axis=1)

    with open('label_encoders.json', 'r') as f:
        encoders = json.load(f)
        
    # Load model
    model = joblib.load('xgboost_model.joblib')
    
    # Preprocess test data
    X_test_processed = preprocess_data(X_test) #, encoders=encoders, is_training=False)
    
    # Ensure column order matches training data
    training_cols = model.get_booster().feature_names
    X_test_processed = X_test_processed[training_cols]

    # Make predictions
    predictions = model.predict(X_test_processed)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({'id': test_ids, 'Depression': predictions})
    
    # Save to CSV
    submission_df.to_csv('submission.csv', index=False)
    print("Submission file 'submission.csv' created successfully.")
    print("Top 5 rows of submission file:")
    print(submission_df.head())
    print("--- Submission Complete ---")
# --- MAIN EXECUTION ---


# --- PART 3: PREDICTOR CLASS FOR USER INPUT ---

class MentalHealthPredictor:
    def __init__(self, model_path='xgboost_model.joblib', encoders_path='label_encoders.json'):
        """
        Initializes the predictor by loading the model and encoders.
        """
        print("--- Initializing Predictor ---")
        try:
            self.model = joblib.load(model_path)
            with open(encoders_path, 'r') as f:
                self.encoders = json.load(f)
            print("Model and encoders loaded successfully.")
        except FileNotFoundError:
            print("Error: Model or encoder file not found. Please run the training script first.")
            self.model = None
            self.encoders = None
        print("--- Predictor Ready ---")

    def predict_single(self, user_data):
        """
        Predicts the mental health status for a single user's data.
        
        Args:
            user_data (dict): A dictionary containing user features.
            
        Returns:
            str: The prediction ('Depressed' or 'Not Depressed').
        """
        if not self.model:
            return "Predictor not initialized."
            
        # Convert dictionary to a DataFrame with a single row
        df_user = pd.DataFrame([user_data])
        
        # Preprocess the user data using saved encoders
        df_user_processed = preprocess_data(df_user, encoders=self.encoders, is_training=False)
        
        # Ensure column order matches the training data
        # This is a safety check in case the dictionary order is different
        training_cols = self.model.get_booster().feature_names
        df_user_processed = df_user_processed[training_cols]

        # Make prediction
        prediction = self.model.predict(df_user_processed)[0]
        
        return "Depressed" if prediction == 1 else "Not Depressed"





# Step 1: Train and save the model and encoders
train_and_save_model()


# Step 2: Create the submission file for the competition
create_submission_file()


# Step 3: Demonstrate the predictor class with a sample user
print("\n--- Demonstrating Single Prediction ---")
predictor = MentalHealthPredictor()

# Example user data (a student with high academic pressure)
sample_user = {
    'Gender': 'Female',
    'Age': 21.0,
    'City': 'Delhi',
    'Working Professional or Student': 'Student',
    'Profession': np.nan, # Student has no profession
    'Academic Pressure': 5.0,
    'Work Pressure': np.nan,
    'CGPA': 8.5,
    'Study Satisfaction': 2.0,
    'Job Satisfaction': np.nan,
    'Sleep Duration': 'Less than 5 hours',
    'Dietary Habits': 'Unhealthy',
    'Degree': 'B.Tech',
    'Have you ever had suicidal thoughts ?': 'Yes',
    'Work/Study Hours': 10.0,
    'Financial Stress': 4.0,
    'Family History of Mental Illness': 'Yes'
}

if predictor.model:
  prediction_result = predictor.predict_single(sample_user)
  print(f"\nPrediction for sample user: {prediction_result}")




