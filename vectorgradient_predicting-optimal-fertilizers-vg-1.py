# Name - Vidhan Prajapati
# Here is my notebook , so you can see my thought process
# The way your code is the way of thinking so keep it clean

# So Enjoyyyzzz


# So the problem is pretty simple , we just have to predict the type of fertilizer through given features

# IMPORTANT NOTE HERE  - Notice how per row you have to predict 3 fertilizer 
# This is because evaluation metric is MAP@3


import pandas as pd

sample = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

sample.head()


# Seeking the Data



train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

print('\nTrain shape : ' , train.shape)
print('\n\nTest shape : ',test.shape)
print('\n\nTrain columns : ',train.columns,'\n\n')

train.head()


# Sweet so data is pretty good and decent , nice number of column and rows ,
# Now lets check faults in data


train.info()
print('\n\n Missing Values in Train : \n',train.isnull().sum())


# No missing values that makes the things easier 
# Now lets see how the targets are distributed


import seaborn as sns
import matplotlib.pyplot as plt

#plt.figure(figsize=(14,6))
sns.countplot(data=train,y='Fertilizer Name',order=train['Fertilizer Name'].value_counts().index)
plt.show

plt.title('Target Class Distribution')


# Ok so Target class is pretty evenly distributed
# So lets get some summary


train.describe()


# There are no features with 0 variance so thats a relief
# Now lets check unique values , so we know what to encode or not


for col in train.columns:

    if train[col].dtype=='object' and col!= 'Fertilizer Name':

        print(f"{col} :  {train[col].nunique()} unique values \n")
        print(train[col].value_counts(),'\n')


# Now we will prepare or preprocess the data for the model
# First lets drop unnecessary columns which are not reuired for particular purpose


train = train.drop(columns=['id'])
test_ids = test['id']
test = test.drop(columns=['id'])


# Now lets do some label encoding
# Caution - Label encoding can introduce a fake order — like "Black < Loamy < Sandy" — which may confuse linear models.
# But tree-based models don’t assume order, so it’s safe and efficient for them.


from sklearn.preprocessing import LabelEncoder

cat_features  = ['Soil Type' , 'Crop Type']
label_encoders = {}

for col in cat_features:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le


# Just for your concern here is what it looks likke
# Generally we dont check it after each line ok

train.head()

# It will be same for test head


# Now lets encode the Target


target_encoder = LabelEncoder()
train['Fertilizer Name'] = target_encoder.fit_transform(train['Fertilizer Name'])

train.head()


# Alright Tree Based Models dont require  much pre preocessing


# Now we will Split Splat


from sklearn.model_selection import train_test_split

X = train.drop(columns=['Fertilizer Name'])
y = train['Fertilizer Name']

# Split Splat

X_train , X_val , y_train , y_val = train_test_split(
    X,y,
    test_size=0.2,
    stratify = y,
    random_state = 42 # Just complimentary but necessary
)

print('Train : ',X_train.shape)
print('Validation : ',X_val.shape)


# See we are almost done


# Ok so now lets use XG Boost as its Leaderboard favorite for structured/tabular data
# We could use LightBoost but it compromises result for performance , we dont need fast model rn
# Also if you use simple Random Forest its baseline of Tree Models , other would always be better than them
# and CatBoost is good for more categorical and here are only 2 so


# Ok lets go


number_of_classes = y_train.nunique()
print(number_of_classes)


import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np

xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=number_of_classes,
    eval_metric='mlogloss',
    use_label_encoder=False,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

xgb_model.fit(X_train,y_train)


# Basic Parameterrs no Hyperparameter Tuning


# Lets See How our model performed


from sklearn.metrics import accuracy_score

y_val_preds = xgb_model.predict(X_val)
acc = accuracy_score(y_val,y_val_preds)

print(f'Validation Accuracy : {acc:.4f}')


# Submission File


X_test = test.copy()

test_probs = xgb_model.predict_proba(X_test)

top_3_indices = np.argsort(test_probs,axis=1)[: ,-3:][:,::-1]

fertilizer_labels = target_encoder.inverse_transform(np.arange(number_of_classes))
top_3_labels = fertilizer_labels[top_3_indices]

top_3_strings = [" ".join(preds) for preds in top_3_labels]

submission_df = pd.DataFrame({
    'id' : test_ids,
    'Fertilizer Name':top_3_strings
})

submission_df.head()


submission_df.to_csv('submission.csv',index=False)

print('Submission File Created ')




