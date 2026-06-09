import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from tqdm import tqdm


### Loading train data from a CSV file
train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")


train_df.head(10)


train_df.shape


train_df.describe()


### Suppress specific Seaborn FutureWarning
warnings.filterwarnings("ignore", 
                      message="use_inf_as_na option is deprecated", 
                      category=FutureWarning)

### Features for visualization
numeric_columns = ["Age", "Height", "Weight", "Duration", 
                  "Heart_Rate", "Body_Temp", "Calories"]

def plot_numeric_distributions(features, dataframe):
    ### Convert inf values to NaN to prevent warnings
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
    
    for feature in features:
        ### Create figure with two subplots
        fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
        
        ### First plot - distribution histogram
        sns.histplot(data=dataframe[feature], 
                    kde=True, 
                    bins=30, 
                    ax=axes[0])
        axes[0].set_title(f"Distribution of {feature}")
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel("Frequency")
        
        ### Second plot - boxplot
        sns.boxplot(x=dataframe[feature], 
                   ax=axes[1])
        axes[1].set_title(f"Boxplot of {feature}")
        
        ### Adjust layout
        plt.tight_layout()
        plt.show()
        
        ### Display statistics
        print(f"\nAnalysis for {feature}:")
        print(f"Skewness: {dataframe[feature].skew():.2f}")
        print(f"Missing values: {dataframe[feature].isna().sum()}")

### Execute visualization function
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    plot_numeric_distributions(numeric_columns, train_df.copy())


### Calculate gender distribution
gender_distribution = train_df["Sex"].value_counts()

### Display gender statistics
print("Gender Distribution Statistics:")
print("-----------------------------")
print(f"Male count: {gender_distribution.get('male', 0)}")
print(f"Female count: {gender_distribution.get('female', 0)}")
print(f"Total records: {gender_distribution.sum()}")
print(f"Male percentage: {gender_distribution.get('male', 0)/gender_distribution.sum()*100:.1f}%")
print(f"Female percentage: {gender_distribution.get('female', 0)/gender_distribution.sum()*100:.1f}%")
print(f"Number of unique genders: {gender_distribution.count()}")


### Let's build a scatterplot matrix
numeric_df = train_df.select_dtypes(include=['number'])  
numeric_cols = [col for col in numeric_df.columns if col != 'id']  

### Generate scatterplot matrix
sns.pairplot(numeric_df[numeric_cols], 
             corner=True, 
             plot_kws={'alpha': 0.5, 's': 20})
plt.suptitle('Analysis of mutual distribution of features', 
             y=1.01, 
             fontsize=20)
plt.show()



### Select only numeric columns and exclude id and BMI
numerical_features = [col for col in train_df.select_dtypes(include=['number']).columns 
                     if col not in ['id', 'BMI']]

### Calculate correlation matrix
correlation_matrix = train_df[numerical_features].corr()

### Create the plot
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, 
           annot=True,
           cmap="coolwarm",
           center=0,
           fmt=".2f",
           linewidths=.5,
           annot_kws={"size": 10},
           mask=np.isnan(correlation_matrix)) 

plt.title("Correlation Matrix (excluding ID and BMI)", pad=20, fontsize=14)
plt.xticks(rotation=45)  
plt.yticks(rotation=0) 
plt.tight_layout()      
plt.show()


### Loading test data from a CSV file
test_df=pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


### Calculate BMI for train and test data
train_df['BMI'] = train_df['Weight'] / ((train_df['Height'] / 100) ** 2)
test_df['BMI'] = test_df['Weight'] / ((test_df['Height'] / 100) ** 2)


### Select numeric columns excluding 'Calories'
num_cols = [col for col in numeric_df if col != "Calories"]

### Calculate initial skewness and sort by most skewed
skew_before = train_df[num_cols].skew().sort_values(ascending=False)

### Create copies of original data for transformation
train_tr = train_df.copy()
test_tr = test_df.copy()
transformers = {}  

### Apply transformations to each numeric column
for col in num_cols:
    if train_df[col].nunique() <= 1:
        continue
    
### Handle positive skew (> 0.5)
    if skew_before[col] > 0.5: 
        if (train_df[col] > 0).all():
            train_tr[col] = np.log1p(train_df[col])
            test_tr[col] = np.log1p(test_df[col])
        else:
            pt = PowerTransformer()
            train_tr[col] = pt.fit_transform(train_df[[col]]).flatten()
            test_tr[col] = pt.transform(test_df[[col]]).flatten()
            transformers[col] = pt 
            
### Handle negative skew (< -0.5)
    elif skew_before[col] < -0.5: 
        pt = PowerTransformer()
        train_tr[col] = pt.fit_transform(train_df[[col]]).flatten()
        test_tr[col] = pt.transform(test_df[[col]]).flatten()
        transformers[col] = pt

### Calculate skewness after transformations
skew_after = train_tr[num_cols].skew().sort_values(ascending=False)

### Create comparison dataframe showing before/after skewness
result = pd.DataFrame({
    'Skew before': skew_before,
    'Skew after': skew_after
}).sort_values(by='Skew before', ascending=False) 
print("Skewness Reduction Results:")
print(result)


### Make copies to avoid changing the original
train_clean = train_tr.copy()
test_clean = test_tr.copy()

### Select only numeric columns and exclude 'Calories'
numeric_features = [feature for feature in numeric_df.columns 
                   if feature != "Calories"]

### Removing outliers using the IQR method
for feature in numeric_features:
    q1 = train_clean[feature].quantile(0.25)
    q3 = train_clean[feature].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
### Filtered data
    train_clean = train_clean[
        (train_clean[feature] >= lower_bound) & 
        (train_clean[feature] <= upper_bound)
    ]


### Binary classification based on temperature for train_df and test_df
train_clean['Temp_Binary'] = np.where(train_clean['Body_Temp'] <= 39.5, 0, 1)
test_clean['Temp_Binary'] = np.where(test_clean['Body_Temp'] <= 39.5, 0, 1)

### Encode sex column for train_df and test_df
train_clean['Sex'] = train_clean['Sex'].map({'male': 1, 'female': 0})
test_clean['Sex'] = test_clean['Sex'].map({'male': 1, 'female': 0})

### Binary classification based on heart rate for train_df and test_df
train_clean['HeartRate_Binary'] = np.where(train_clean['Heart_Rate'] <= 99.5, 0, 1)
test_clean['HeartRate_Binary'] = np.where(test_clean['Heart_Rate'] <= 99.5, 0, 1)


### Prepare data
X = train_clean.drop(columns=['Calories', 'id'])
y = train_clean['Calories']

### Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=100)

### Initialize XGBoost model
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    colsample_bytree=0.3,
    learning_rate=0.1,
    max_depth=5,
    alpha=10,
    n_estimators=1000,
    random_state=100,
    eval_metric='mae',
    early_stopping_rounds=50,
    verbosity=1
)

### Train the model
model.fit(
    X_train, 
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

### Make predictions on test data
test_features = test_clean.drop(columns=['id'])
predictions = model.predict(test_features)

### Create submission DataFrame in required format
submission = pd.DataFrame({
    'id': test_df['id'],
    'Calories': predictions.clip(0)  # Ensure no negative values
})

### Save to CSV without index and with header
submission.to_csv('predictions.csv', index=False, float_format='%.2f')

### Display first 5 rows of the saved file
print("First 5 predictions saved to CSV:")
print(submission.head())

### Verification output
print(f"\nPredictions saved to 'predictions.csv'")
print("File format verification:")
print("Header:", ','.join(submission.columns))
print("Sample row:", f"{submission.iloc[0,0]},{submission.iloc[0,1]:.2f}")

