import pandas as pd


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv', index_col='id')


train.head()


train.shape


test.shape


train.isna().sum()


test.isna().sum()


# Select categorical columns
cat_cols = train.select_dtypes(include='object').columns
print('CATEGORICAL COLUMNS')
print(cat_cols, '\n')

# Select numerical columns
num_cols = train.select_dtypes(include='float').columns
print('NUMERICAL COLUMNS')
print(num_cols)


# def fill_missing_values(df, cat_cols, num_cols):

#     # Fill categorical columns
#     df[cat_cols] = df[cat_cols].fillna('Missing').astype('category')
    
#     # Fill numerical columns
#     for col in num_cols:
#         df[col] = df[col].fillna(df[col].median())
    
#     return df

# # Identify categorical and numerical columns
# cat_cols = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment', 
#             'Waterproof', 'Style', 'Color']
# num_cols = ['Weight Capacity (kg)']

# # Apply the function to both train and test datasets
# train = fill_missing_values(train, cat_cols, num_cols)
# test = fill_missing_values(test, cat_cols, num_cols)

# # Verify missing values have been handled
# print("Missing values in train dataset:\n", train.isnull().sum(), "\n")
# print("Missing values in test dataset:\n", test.isnull().sum())


!pip install ray==2.10.0


!pip install autogluon.tabular --no-cache-dir -q
!pip install -U ipywidgets


from autogluon.tabular import TabularPredictor
from autogluon.common import space
import warnings

warnings.simplefilter("ignore")

predictor = TabularPredictor(
    path = '/kaggle/working/Autogluon',
    problem_type='regression',
    eval_metric='mean_absolute_percentage_error',
    label='Price',
    verbosity=2
)

predictor.fit(
    train_data=train,
    time_limit=3600 * 1,
    presets='best_quality',
    excluded_model_types=['KNN', 'NN_TORCH', 'FASTAI', 'LINEAR_MODEL', 'RF'],
    ag_args_fit={'num_cpus': 4}
)


predictor.leaderboard(silent=True).style.background_gradient(subset=['score_val'], cmap='viridis')


predictor = TabularPredictor.load("/kaggle/working/Autogluon")


predictions = predictor.predict(test)
predictions = predictions.reset_index(drop=True)
print(predictions)


submission =  pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
submission.head()


# Create the submission DataFrame
submission_df = pd.DataFrame({
    'id': submission['id'],  # Use the 'id' column from the sample submission
    'Price': predictions      # Use the entire array of predictions
})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

# Display the first few rows of the submission DataFrame
print(submission_df.head())

