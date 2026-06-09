
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import h2o
import h2o
from h2o.automl import H2OAutoML


#Ignore warnings
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


# create already now an empty submission file 
submission = pd.DataFrame()

# add id column to submission file
# as we will drop 'id' column later on

submission['id'] = test['id']


train.info()


train.describe().T


# check for NA

train.isna().sum().sort_values(ascending=False)





from sklearn.impute import KNNImputer

def handle_na_knn(df):
    """
    fill all NAs with KNN value

        Args:
        df (pd.DataFrame): The input DataFrame.
        n_neighbors (int, optional): The number of neighbors to use for KNN. Defaults to 5.

    Returns:
        pd.DataFrame: The DataFrame with missing values imputed using KNN.
    """

    imputer = KNNImputer(n_neighbors = 10)

    # Define columns for imputation
    imputable_cols = ['Guest_Popularity_percentage', 'Episode_Length_minutes', 'Number_of_Ads']

    # Create a copy to avoid modifying the original DataFrame directly
    df_imputed = df.copy()

    # Fit and transform the numerical columns
    df_imputed[imputable_cols] = imputer.fit_transform(df[imputable_cols])

    return df_imputed




train = handle_na_knn(train)
test = handle_na_knn(test)


# code idea from @satya
# https://www.kaggle.com/code/satyaprakashshukl/predict-podcast-listening-time 


from sklearn.preprocessing import LabelEncoder

# define categorical columns
categorical_cols = train.select_dtypes(include=['object']).columns


categorical_cols


# label encode 
label_encoders = {} 
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col]) 
    label_encoders[col] = le  


train = train.astype(float)
test = test.astype(float)

print("âœ… Categorical columns to numerical")


train.head()


# ***********************************Feature Engineering    ********************************************
# from https://www.kaggle.com/code/satyaprakashshukl/predict-podcast-listening-time

def feature_engineering(df, is_train=True):
    

    df['Is_Weekend'] = df['Publication_Day'].apply(lambda x: 1 if x in [6, 7] else 0)
    df['Daypart'] = df['Publication_Time'].apply(lambda x: 
                                                 'Morning' if 6 <= x < 12 else 
                                                 'Afternoon' if 12 <= x < 18 else 
                                                 'Evening' if 18 <= x < 24 else 
                                                 'Night')  

    # re-map part of the day
    df['Daypart'] = df['Daypart'].map({'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3})   
    
    df['Host_Guest_Popularity_Ratio'] = df['Host_Popularity_percentage'] / (df['Guest_Popularity_percentage'] + 1e-5) 
    df['Ads_Per_Minute'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1e-5)
    df['Popularity_Score'] = (df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage']) / 2  
    df['Long_Episode'] = (df['Episode_Length_minutes'] > 75).astype(int)
    df['Highly_Popular_Host'] = (df['Host_Popularity_percentage'] > 75).astype(int)
    df['Highly_Popular_Guest'] = (df['Guest_Popularity_percentage'] > 75).astype(int)
    df['Host_Guest_Popularity_Diff'] = df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']
    df['Host_Guest_Popularity_Sum'] = df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage']
    df['Ad_Impact'] = df['Number_of_Ads'] * df['Episode_Length_minutes']

    # binning episode length
    df['Episode_Length_Bin'] = pd.cut(df['Episode_Length_minutes'],
                                      bins=[-1, 187500, 375000, 562500, np.inf],
                                      labels=[0, 1, 2, 3])  
    df['Episode_Length_Bin'] = df['Episode_Length_Bin'].astype(int)

    # most have very little ads.  Let's mark if ads numbers are very high
    df['High_Ad_Load'] = (df['Number_of_Ads'] > 2).astype(int)

    return df




train = feature_engineering(train, is_train=True)
test = feature_engineering(test, is_train=False)

print("âœ… Feature Engineering Complete!")


#initialize

h2o.init() 



# remove ID columns, as they are not relevant for modeling

train = train.drop(columns = ['id'])
test = test.drop(columns = ['id'])


# put train and test into H2OFrame

h2o_train = h2o.H2OFrame(train)
h2o_test = h2o.H2OFrame(test)


splits = h2o_train.split_frame(ratios=[0.75],seed=15)
train = splits[0]
test = splits[1]


y = "Listening_Time_minutes" 
x = h2o_train.columns 
x.remove(y)


# train AutoML

aml = H2OAutoML(
    max_runtime_secs= 3600,  # adjust - longer training would bring better results
    seed=15,
    sort_metric="RMSE"  # optimizing RMSE metric (on which the competition is scored)
)

aml.train(
    x=x,
    y=y, 
    training_frame=train
)


aml.explain(frame = splits[0])


# leaderboard

lb = aml.leaderboard
lb.head()


# get all model_ids
model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])


# Get the "All Models" Stacked Ensemble model
se = h2o.get_model([mid for mid in model_ids if "StackedEnsemble_AllModels" in mid][0])



# Get the Stacked Ensemble metalearner model
metalearner = h2o.get_model(se.metalearner()['name'])


metalearner.std_coef_plot()


pred = aml.predict(h2o_test).as_data_frame(use_multi_thread=True)

pred.head(5)







# add prediction column to submission file 

submission["Listening_Time_minutes"] = pred['predict']


h2o.shutdown(prompt = False)


# write to csv
submission.to_csv("submission.csv", index = False)


submission.head()

