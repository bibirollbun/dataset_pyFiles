import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train_data=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_extra=pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test_data=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
sample_submission=pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


train_data.head()


test_data.head()


train_data.info()


#downcasting

def downcast_dataframe(df):
    # Downcast integer columns
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    
    # Downcast float columns
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    
    return df

train_data=downcast_dataframe(train_data)
train_extra=downcast_dataframe(train_extra)
test_data=downcast_dataframe(test_data)

print(train_data.memory_usage(deep=True))


num_cols=['Compartments','Weight Capacity (kg)']
train_data[num_cols].corr()


print(train_data['Brand'].value_counts())
print(train_data['Material'].value_counts())
print(train_data['Size'].value_counts())



train_data.describe().T


concat_df = pd.concat([train_data, train_extra], ignore_index=True)


concat_df


#lets plot histograms to see the distributions
fig,axes=plt.subplots(2,1,figsize=(10,8))
axes[0].hist(concat_df['Weight Capacity (kg)'].sample(10000))
axes[0].set_title('Weight Capacity')

axes[1].hist(concat_df['Compartments'].sample(10000))
axes[1].set_title('Compartments')
plt.tight_layout()
plt.show()


concat_df.isna().sum()


#encoding size into ordinal columns
concat_df['Size']=concat_df['Size'].map({'Small':0,'Medium':1,'Large':2}).fillna(-1)
test_data['Size']=test_data['Size'].map({'Small':0,'Medium':1,'Large':2}).fillna(-1)
#encoding waterproof, Laptop Compartment to binary encoding

concat_df['Waterproof']=concat_df['Waterproof'].map({'Yes':1,'No':0}).fillna(-1)
concat_df['Laptop Compartment']=concat_df['Laptop Compartment'].map({'Yes':1,'No':0}).fillna(-1)

test_data['Waterproof']=test_data['Waterproof'].map({'Yes':1,'No':0}).fillna(-1)
test_data['Laptop Compartment']=test_data['Laptop Compartment'].map({'Yes':1,'No':0}).fillna(-1)
categorical_cols=['Brand','Material','Style','Color']


concat_df.head()


concat_df[concat_df['Laptop Compartment']==1]


#size Vs Compartments, Laptop Comparment vs Size
fig,axes=plt.subplots(1,2,figsize=(10,7))
sns.barplot(data=concat_df.sample(10000),x='Size',y='Compartments',hue='Brand',ax=axes[0])
axes[0].set_title('Size Vs Compartments')
axes[0].legend(loc='upper right')

sns.histplot(data=concat_df.sample(10000),x='Size',hue='Laptop Compartment',ax=axes[1])
axes[1].set_title('Different Size and Laptop Compartments')
axes[1].legend(loc=0)

plt.tight_layout()
plt.show()


concat_df['Compartment_weight_ratio']=concat_df['Weight Capacity (kg)']/concat_df['Compartments'].replace(0,np.nan)
concat_df['Compartment_weight_ratio']=concat_df['Compartment_weight_ratio'].fillna(0)

test_data['Compartment_weight_ratio']=test_data['Weight Capacity (kg)']/test_data['Compartments'].replace(0,np.nan)
test_data['Compartment_weight_ratio']=test_data['Compartment_weight_ratio'].fillna(0)


#concat_df[(concat_df['Size']>0) & (concat_df['Style']=='Tote')]


#impute missing values

from sklearn.impute import SimpleImputer
numericals_to_impute=['Compartments','Weight Capacity (kg)']
categorical_to_impute=['Brand','Material','Style','Color']

imputer1=SimpleImputer(strategy='median').fit(concat_df[numericals_to_impute])
imputer2=SimpleImputer(strategy='most_frequent').fit(concat_df[categorical_to_impute])

concat_df[numericals_to_impute]=imputer1.transform(concat_df[numericals_to_impute])
concat_df[categorical_to_impute]=imputer2.transform(concat_df[categorical_to_impute])

test_data[numericals_to_impute]=imputer1.transform(test_data[numericals_to_impute])
test_data[categorical_to_impute]=imputer2.transform(test_data[categorical_to_impute])




concat_df.isna().sum()


#check for outliers

fig,axes=plt.subplots(1,2,figsize=(10,8))
sns.boxplot(data=concat_df,x='Weight Capacity (kg)',ax=axes[0])
axes[0].set_title('Weight Capacity (kg) BoxPlot')

sns.boxplot(data=concat_df,x='Compartments',ax=axes[1])
axes[1].set_title('Compartments Boxplot')
plt.tight_layout()
plt.show()


#IQR And outliers calculation
Q1=np.percentile(concat_df['Weight Capacity (kg)'],25)
Q3=np.percentile(concat_df['Weight Capacity (kg)'],75)

IQR=Q3-Q1

#lowerbound
lb=Q1-1.5*IQR

#uperbound
up=Q3+1.5*IQR

outliers=concat_df[(concat_df['Weight Capacity (kg)']<lb)| (concat_df['Weight Capacity (kg)']>up)]

print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
print(f"Lower Bound: {lb}, Upper Bound: {up}")
print("Outliers found:", outliers)


# Calculate mean price by "Style" and "Color"
mean_price = concat_df.groupby(['Material', 'Color'])['Price'].mean().unstack()

# Heatmap of mean price
plt.figure(figsize=(10, 6))
sns.heatmap(mean_price, cmap="coolwarm", annot=True, fmt='.2f', linewidths=0.5)
plt.title('Mean Price by Material and Color Combination')
plt.show()



brand_style=concat_df.groupby(["Brand","Style"])["Price"].mean().unstack()
plt.figure(figsize=(10,8))
sns.heatmap(brand_style,cmap='Blues',annot=True,fmt='.2f', linewidths=0.5)
plt.title("Mean Price by Brand and Style")
plt.show()


concat_df['Material_color']=concat_df['Material'] + '_' + concat_df['Color']
test_data['Material_color']=test_data['Material'] + '_' + test_data['Color']
concat_df['Brand_style']=concat_df["Brand"]+"_"+concat_df["Style"]
test_data['Brand_style']=test_data['Brand'] + '_' + test_data['Style']


concat_df.head()


concat_df["size_to_weight"]=concat_df["Weight Capacity (kg)"]*concat_df["Size"].replace(0,1)
test_data["size_to_weight"]=test_data["Weight Capacity (kg)"]*test_data["Size"].replace(0,1)


from sklearn.preprocessing import OneHotEncoder

# Columns to encode
to_encode_cols = ['Brand', 'Style', 'Material_color']

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(concat_df[to_encode_cols])

encoded_cols = list(encoder.get_feature_names_out(to_encode_cols))

concat_encoded = pd.DataFrame(encoder.transform(concat_df[to_encode_cols]), columns=encoded_cols, index=concat_df.index)

test_encoded = pd.DataFrame(encoder.transform(test_data[to_encode_cols]), columns=encoded_cols, index=test_data.index)

inputs_data = pd.concat([concat_encoded, concat_df[['size_to_weight', 'Laptop Compartment', 'Waterproof', 'Compartment_weight_ratio']]], axis=1)
test_inputs = pd.concat([test_encoded, test_data[['size_to_weight', 'Laptop Compartment', 'Waterproof', 'Compartment_weight_ratio']]], axis=1)

# Checking the shape of final inputs
print(inputs_data.shape, test_inputs.shape)



from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

# Create the XGBRegressor model
model = XGBRegressor(n_jobs=-1,n_estimators=250,learning_rate=0.05,max_depth=10,subsample=0.8)



from sklearn.model_selection import KFold,cross_val_score

kf=KFold(n_splits=5,random_state=42,shuffle=True)

scores=cross_val_score(model,inputs_data,concat_df["Price"],cv=kf)

plt.figure(figsize=(10,7))
sns.barplot(x=np.arange(1,len(scores)+1),y=scores,color="b")
plt.title("Cross Validation Accuracy")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1,len(scores)+1))
plt.axhline(y=scores.mean(),linestyle="--",label="Mean Accuracy Score")
plt.legend()
plt.show()


model.fit(inputs_data,concat_df["Price"])


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_inputs,valid_inputs=train_test_split(inputs_data,test_size=0.2,random_state=42)
train_targets,valid_targets=train_test_split(concat_df['Price'],test_size=0.2,random_state=42)
pred=model.predict(valid_inputs)
mse=mean_squared_error(valid_targets,pred)
print(np.sqrt(mse))


prediction=model.predict(test_inputs)


output=pd.DataFrame({
    'id':test_data['id'],
    'Price':prediction
})


output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")




