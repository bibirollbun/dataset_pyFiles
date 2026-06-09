import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train = pd.read_csv("/kaggle/input/Predict-the-Customer-Satisfaction-CSE-22/train_dataset.csv") #Training Data For the Model
test = pd.read_csv("/kaggle/input/Predict-the-Customer-Satisfaction-CSE-22/test_dataset.csv")  #Unseen Data for the predictions


train


test


train.describe()


desc = pd.DataFrame(index=train.columns.to_list())
desc['type'] = train.dtypes
desc['count'] = train.count()
desc['nunique'] = train.nunique()
desc['null'] = train.isnull().sum()
desc['min'] = train.min()
desc['max'] = train.max()
desc


train.info(memory_usage='deep')


train[['purchased_datetime', 'payment_datetime']]


date_columns=['Date_Registered', 'purchased_datetime', 'payment_datetime', 'released_date', 'estimated_delivery_date', 'received_date'   ]

for col in date_columns:    
    date_lengths = train[col].str.len()
    print(date_lengths.value_counts())


train.columns = train.columns.str.lower()
test.columns = test.columns.str.lower()


if(train['payment_datetime'].equals(train['purchased_datetime'])):
    train = train.drop('payment_datetime', axis=1)

if(test['payment_datetime'].equals(test['purchased_datetime'])):
    test = test.drop('payment_datetime', axis=1)
    


train = train.drop( columns = ['id','user_id', 'transaction_id', 'order_id', 'tracking_number' ], axis=1)
test = test.drop( columns = ['id','user_id', 'transaction_id', 'order_id', 'tracking_number' ], axis=1)


train.fillna(-1, inplace=True)
test.fillna(-1, inplace=True)


train['age'] = train['age'].astype('int8')
train['loyalty_points_redeemed'] = train['loyalty_points_redeemed'].astype('int8')
train['loyalty_tier'] = train['loyalty_tier'].astype('int8')
train['received_tier_discount_percentage'] = train['received_tier_discount_percentage'].astype('int8')
train['received_card_discount_percentage'] = train['received_card_discount_percentage'].astype('float32')
train['received_coupon_discount_percentage'] = train['received_coupon_discount_percentage'].astype('float32')
train['final_payment'] = train['final_payment'].astype('float32')
train['product_value'] = train['product_value'].astype('float64')


train['date_registered'] = pd.to_datetime(train['date_registered'])
train['purchased_datetime'] = pd.to_datetime(train['purchased_datetime'])
train['released_date'] = pd.to_datetime(train['released_date'])
train['estimated_delivery_date'] = pd.to_datetime(train['estimated_delivery_date'])
train['received_date'] = pd.to_datetime(train['received_date'])


test['age'] = test['age'].astype('int8')
test['loyalty_points_redeemed'] = test['loyalty_points_redeemed'].astype('int8')
test['loyalty_tier'] = test['loyalty_tier'].astype('int8')
test['received_tier_discount_percentage'] = test['received_tier_discount_percentage'].astype('int8')
test['received_card_discount_percentage'] = test['received_card_discount_percentage'].astype('float32')
test['received_coupon_discount_percentage'] = test['received_coupon_discount_percentage'].astype('float32')
test['final_payment'] = test['final_payment'].astype('float32')
test['product_value'] = test['product_value'].astype('float64')


test['date_registered'] = pd.to_datetime(test['date_registered'])
test['purchased_datetime'] = pd.to_datetime(test['purchased_datetime'])
test['released_date'] = pd.to_datetime(test['released_date'])
test['estimated_delivery_date'] = pd.to_datetime(test['estimated_delivery_date'])
test['received_date'] = pd.to_datetime(test['received_date'])


from sklearn.preprocessing import OneHotEncoder

def one_hot_encode_and_add(df, column):
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = one_hot_encoder.fit_transform(df[[column]])
    encoded_columns = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out([column]))
    encoded_columns.index = df.index
    df = pd.concat([df, encoded_columns], axis=1)
    df = df.drop(columns=[column])
    return df

columns_to_encode = ['gender', 'purchase_medium', 'shipping_method', 'is_current_loyalty_program_member']

for col in columns_to_encode:
    train = one_hot_encode_and_add(train, col)
    test = one_hot_encode_and_add(test, col)


#train['payment_method'].unique()
train['product_category'].unique()


train['payment_method'] = pd.Categorical(train['payment_method'], categories=['visa_c', 'amex', 'mastercard_c', 'coinsph', 'visa_d', 'gcash',
    'maya', 'cash', 'bank_transfer', 'shopeepay', 'otc', 'grabpay',
    'mastercard_d'], ordered=True)
train['payment_method'] = train['payment_method'].cat.codes

train['product_category'] = pd.Categorical(train['product_category'], categories=['office supplies', 'electronics', 'pet supplies', 'clothing',
    'books', 'appliances', 'groceries', 'home', 'health', 'music',
    'tools', 'automotive', 'toys', 'sports', 'video games', 'beauty',
    'movies', 'jewelry', 'garden', 'furniture'], ordered=True)
train['product_category'] = train['product_category'].cat.codes


# train['gender'] = pd.Categorical(train['gender'], categories=['O', 'F', 'M'], ordered=True)
# train['gender'] = train['gender'].cat.codes


# train['purchase_medium'] = pd.Categorical(train['purchase_medium'], categories=['online', 'in-store'], ordered=True)
# train['purchase_medium'] = train['purchase_medium'].cat.codes


# train['shipping_method'] = pd.Categorical(train['shipping_method'], categories=['standard', 'express'], ordered=True)
# train['shipping_method'] = train['shipping_method'].cat.codes


# train['is_current_loyalty_program_member'] = pd.Categorical(train['is_current_loyalty_program_member'], categories=['NO', 'YES'], ordered=True)
# train['is_current_loyalty_program_member'] = train['is_current_loyalty_program_member'].cat.codes



test['payment_method'] = pd.Categorical(test['payment_method'], categories=['visa_c', 'amex', 'mastercard_c', 'coinsph', 'visa_d', 'gcash',
    'maya', 'cash', 'bank_transfer', 'shopeepay', 'otc', 'grabpay',
    'mastercard_d'], ordered=True)
test['payment_method'] = test['payment_method'].cat.codes

test['product_category'] = pd.Categorical(test['product_category'], categories=['office supplies', 'electronics', 'pet supplies', 'clothing',
    'books', 'appliances', 'groceries', 'home', 'health', 'music',
    'tools', 'automotive', 'toys', 'sports', 'video games', 'beauty',
    'movies', 'jewelry', 'garden', 'furniture'], ordered=True)
test['product_category'] = test['product_category'].cat.codes


# test['gender'] = pd.Categorical(test['gender'], categories=['O', 'F', 'M'], ordered=True)
# test['gender'] = test['gender'].cat.codes


# test['purchase_medium'] = pd.Categorical(test['purchase_medium'], categories=['online', 'in-store'], ordered=True)
# test['purchase_medium'] = test['purchase_medium'].cat.codes


# test['shipping_method'] = pd.Categorical(test['shipping_method'], categories=['standard', 'express'], ordered=True)
# test['shipping_method'] = test['shipping_method'].cat.codes


# test['is_current_loyalty_program_member'] = pd.Categorical(test['is_current_loyalty_program_member'], categories=['NO', 'YES'], ordered=True)
# test['is_current_loyalty_program_member'] = test['is_current_loyalty_program_member'].cat.codes


train


train.describe()


train['received_card_discount_percentage'].hist()


train['received_coupon_discount_percentage'].hist()


train['age'].hist()


train[train['received_coupon_discount_percentage']>100].equals(train[train['received_card_discount_percentage']>100])


train[train['received_coupon_discount_percentage']>100].equals(train[train['product_value']>30000])


train[train['product_value']>30000].shape


train[train['received_coupon_discount_percentage']>100].shape


train[train['received_card_discount_percentage']>100].equals(train[train['age']==0])


train[train['age']==0].shape


train[train['received_card_discount_percentage']>100].shape


train = train[train['received_coupon_discount_percentage'] <= 100]
train = train[train['received_coupon_discount_percentage'] <= 100]
train = train[train['age']>0]
train = train[train['product_value']<30000]



train.info()


# # Apply Min-Max Scaling to scale values to 0-100 
# min_value = train['received_card_discount_percentage'].min()
# max_value = train['received_card_discount_percentage'].max() 
# train['scaled_card_discount_percentage'] = 100 * (train['received_card_discount_percentage'] - min_value) / (max_value - min_value)


# min_value = test['received_card_discount_percentage'].min()
# max_value = test['received_card_discount_percentage'].max() 
# test['scaled_card_discount_percentage'] = 100 * (test['received_card_discount_percentage'] - min_value) / (max_value - min_value)


# train['scaled_card_discount_percentage'].min()
# train['scaled_card_discount_percentage'].max()


# min_value = train['received_coupon_discount_percentage'].min()
# max_value = train['received_coupon_discount_percentage'].max() 
# train['scaled_coupon_discount_percentage'] = 100 * (train['received_coupon_discount_percentage'] - min_value) / (max_value - min_value)


# min_value = test['received_coupon_discount_percentage'].min()
# max_value = test['received_coupon_discount_percentage'].max() 
# test['scaled_coupon_discount_percentage'] = 100 * (test['received_coupon_discount_percentage'] - min_value) / (max_value - min_value)


# train['scaled_coupon_discount_percentage'].min()
# train['scaled_coupon_discount_percentage'].max()


# train = train.drop(columns=['received_coupon_discount_percentage','received_card_discount_percentage'] , axis=1)


# Extract components
train['reg_year'] = train['date_registered'].dt.year
train['reg_month'] = train['date_registered'].dt.month
train['reg_day'] = train['date_registered'].dt.day
train['reg_dayofweek'] = train['date_registered'].dt.dayofweek


train['released_year'] = train['released_date'].dt.year
train['released_month'] = train['released_date'].dt.month
train['released_day'] = train['released_date'].dt.day
train['released_dayofweek'] = train['released_date'].dt.dayofweek

train['purchased_year'] = train['purchased_datetime'].dt.year
train['purchased_month'] = train['purchased_datetime'].dt.month
train['purchased_day'] = train['purchased_datetime'].dt.day
train['purchased_dayofweek'] = train['purchased_datetime'].dt.dayofweek

train['est_deivery_year'] = train['estimated_delivery_date'].dt.year
train['est_delivery_month'] = train['estimated_delivery_date'].dt.month
train['est_deivery_day'] = train['estimated_delivery_date'].dt.day
train['est_delivery_dayofweek'] = train['estimated_delivery_date'].dt.dayofweek

train['received_year'] = train['received_date'].dt.year
train['received_month'] = train['received_date'].dt.month
train['received_day'] = train['received_date'].dt.day
train['received_dayofweek'] = train['received_date'].dt.dayofweek



test['reg_year'] = test['date_registered'].dt.year
test['reg_month'] = test['date_registered'].dt.month
test['reg_day'] = test['date_registered'].dt.day
test['reg_dayofweek'] = test['date_registered'].dt.dayofweek


test['released_year'] = test['released_date'].dt.year
test['released_month'] = test['released_date'].dt.month
test['released_day'] = test['released_date'].dt.day
test['released_dayofweek'] = test['released_date'].dt.dayofweek

test['purchased_year'] = test['purchased_datetime'].dt.year
test['purchased_month'] = test['purchased_datetime'].dt.month
test['purchased_day'] = test['purchased_datetime'].dt.day
test['purchased_dayofweek'] = test['purchased_datetime'].dt.dayofweek

test['est_deivery_year'] = test['estimated_delivery_date'].dt.year
test['est_delivery_month'] = test['estimated_delivery_date'].dt.month
test['est_deivery_day'] = test['estimated_delivery_date'].dt.day
test['est_delivery_dayofweek'] = test['estimated_delivery_date'].dt.dayofweek

test['received_year'] = test['received_date'].dt.year
test['received_month'] = test['received_date'].dt.month
test['received_day'] = test['received_date'].dt.day
test['received_dayofweek'] = test['received_date'].dt.dayofweek



train.info(memory_usage='deep')


import numpy as np

all_numeric_cols = train.select_dtypes(include=[np.number]).columns
all_numeric_cols


import seaborn as sns
import matplotlib.pyplot as plt

columns_to_plot = all_numeric_cols

num_cols = 5
num_rows = int(np.ceil(len(columns_to_plot) / num_cols))

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 4 * num_rows)) 

axes = axes.flatten()

for i, column in enumerate(columns_to_plot):
    sns.boxplot(y=train[column], ax=axes[i])
    axes[i].set_title(f"Boxplot - {column}")
    axes[i].grid(False)

plt.tight_layout()
plt.show()


sns.relplot(x='customer_experience', y='product_value', data=train)


train['product_value'].hist(bins=30)
plt.title('Histogram of Product Value')
plt.show()


sns.boxplot(x='product_value', data=train) 
plt.title('Box Plot of Product Value') 
plt.show()


#If not entries dropped before we could have scaled
# train['product_value_log'] = np.log1p(train['product_value'])
# test['product_value_log'] = np.log1p(test['product_value'])


# sns.boxplot(x='product_value_log', data=train) 
# plt.title('Box Plot of Log Product Value') 
# plt.show()


# sns.relplot(x='customer_experience', y='product_value_log', data=train)


# train = train.drop('product_value', axis=1)
# test = test.drop('product_value', axis=1)


train.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)


categorical_features = train.select_dtypes(include=['int8','int32', 'int64'] ).columns.to_list()
continuous_features = train.select_dtypes(include=['float32','float64']).columns.to_list()


categorical_features


continuous_features


# for i, col in enumerate(categorical_features): 
#     plt.subplot(12, 4, i+1) # Adjusted to 12 rows and 4 columns for better spacing
#     sns.countplot(data=train, x=col, hue='customer_experience') 
#     plt.title(col) # Adjust layout for better spacing 

# plt.subplots_adjust(hspace=0.5, wspace=0.3) # Adding spacing between plots
# plt.show()


features=[ 'loyalty_tier', 'received_tier_discount_percentage','received_card_discount_percentage','loyalty_points_redeemed']


plt.figure(figsize=(20, 24)) 
for i, col in enumerate(features): 
    plt.subplot(3, 2, i+1) # Adjusted to 8 rows and 3 columns for better spacing
    sns.countplot(data=train, x=col, hue='customer_experience') 
    plt.title(col) # Adjust layout for better spacing 

plt.tight_layout() 
plt.subplots_adjust(hspace=0.5, wspace=0.3) # Adding spacing between plots
plt.show()


sns.violinplot(data=train, x='customer_experience', y='age')
plt.show()


sns.violinplot(data=train, x='customer_experience', y='product_value')
plt.show()


sns.violinplot(data=train, x='customer_experience', y='final_payment')
plt.show()


# plt.figure(figsize=(16, 4))
# for i, col in enumerate(continuous_features):
#     plt.subplot(5, 3, i+1)
#     sns.violinplot(data=train, x='customer_experience', y=col)
# plt.tight_layout()
# plt.show()


sns.violinplot(data=train, x='customer_experience', y='received_dayofweek')
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

X = train.drop(columns=['customer_experience', 'received_date', 'estimated_delivery_date', 'purchased_datetime', 'released_date', 'date_registered'], axis=1)  
y = train['customer_experience']

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
val_X_scaled = scaler.transform(val_X)

train_X = pd.DataFrame(train_X_scaled, columns=train_X.columns, index=train_X.index)
val_X = pd.DataFrame(val_X_scaled, columns=val_X.columns, index=val_X.index)


from sklearn.metrics import classification_report, confusion_matrix

def predict(train_X, train_y, val_X, val_y, model):

    model.fit(train_X, train_y)
    pred_y = model.predict(val_X)
    print("\nClassification Report:\n", classification_report(val_y, pred_y))
    
    cm = confusion_matrix(val_y, pred_y)

    labels = ['Bad', 'Neutral','Good']
    
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

predict(train_X, train_y, val_X, val_y, rf_model)


from sklearn.tree import DecisionTreeClassifier

rf_model = DecisionTreeClassifier(max_depth=11, random_state=42, min_samples_leaf=4, min_samples_split=2)

predict(train_X, train_y, val_X, val_y, rf_model)


from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier( n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42 )

predict(train_X, train_y, val_X, val_y, gb_model)


train['customer_experience'] = train['customer_experience'].replace('bad',0)
train['customer_experience'] = train['customer_experience'].replace('neutral',1)
train['customer_experience'] = train['customer_experience'].replace('good',2)


cols = [col for col in train if col != 'customer_experience'] + ['customer_experience'] 
train = train[cols]
train


cor_mat = train.corr(method="pearson")

mask = np.triu(np.ones_like(cor_mat))

plt.figure(figsize=(25, 20))
sns.heatmap(cor_mat, annot=True, fmt=".2f", cmap="coolwarm", mask=mask)
plt.show()


from sklearn.feature_selection import mutual_info_classif
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(['float32','float64', 'int8','int32', 'int64']):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)  # Sort for better visualization
    width = np.arange(len(scores))  # Create indices for y-axis
    ticks = list(scores.index)  # Get feature names as ticks

    plt.figure(dpi=100, figsize=(10, 8))  
    plt.barh(width, scores, color='skyblue')  
    plt.yticks(width, ticks, fontsize=10)  # Feature names on y-axis
    plt.xlabel("Mutual Information Score", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.title("Mutual Information Scores", fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout() 
    plt.show()


X = train.drop(columns=['customer_experience', 'received_date', 'estimated_delivery_date', 'purchased_datetime', 'released_date', 'date_registered'], axis=1)  
y = train['customer_experience']

mi_scores = make_mi_scores(X, y)
mi_scores



# from sklearn.feature_selection import mutual_info_classif

# # Calculate mutual information
# mi_scores = mutual_info_classif(train[['reg_year', 'reg_month', 'reg_day', 'reg_dayofweek']], train['customer_experience'])
# mi_scores = pd.Series(mi_scores, index=['reg_year', 'reg_month', 'reg_day', 'reg_dayofweek'])

# print("Mutual Information scores with customer experience:")
# print(mi_scores)



# from sklearn.feature_selection import mutual_info_classif

# # Calculate mutual information
# mi_scores = mutual_info_classif(train[['released_year', 'released_month', 'released_day', 'released_dayofweek']], train['customer_experience'])
# mi_scores = pd.Series(mi_scores, index=['released_year', 'released_month', 'released_day', 'released_dayofweek'])

# print("Mutual Information scores with customer experience:")
# print(mi_scores)


# from sklearn.feature_selection import mutual_info_classif

# # Calculate mutual information
# mi_scores = mutual_info_classif(train[['received_year', 'received_month', 'received_day', 'received_dayofweek']], train['customer_experience'])
# mi_scores = pd.Series(mi_scores, index=['received_year', 'received_month', 'received_day', 'received_dayofweek'])

# print("Mutual Information scores with customer experience:")
# print(mi_scores)


train['days_to_deliver'] = (train['received_date'] - train['purchased_datetime']).dt.days
train['est_days_to_deliver'] = (train['estimated_delivery_date'] - train['purchased_datetime']).dt.days


test['days_to_deliver'] = (test['received_date'] - test['purchased_datetime']).dt.days
test['est_days_to_deliver'] = (test['estimated_delivery_date'] - test['purchased_datetime']).dt.days


train['diff_est_actual'] = train['est_days_to_deliver'] - train['days_to_deliver'] 
test['diff_est_actual'] = test['est_days_to_deliver'] - test['days_to_deliver'] 


train['is_delayed_delivery'] = train['diff_est_actual'].apply(lambda x: 0 if x >= 0 else 1)
test['is_delayed_delivery'] = test['diff_est_actual'].apply(lambda x: 0 if x >= 0 else 1)


train['prod_age'] = (train['received_date'] - train['released_date']).dt.days
test['prod_age'] = (test['received_date'] - test['released_date']).dt.days


train['payment_overproduct_ratio'] = (train['final_payment'] / train['product_value']).astype('float32')
test['payment_overproduct_ratio'] = (test['final_payment'] / test['product_value']).astype('float32')



train['total_discount_percentage'] = (
    train['received_tier_discount_percentage'] +
    train['received_card_discount_percentage'] +
    train['received_coupon_discount_percentage']
)

test['total_discount_percentage'] = (
    test['received_tier_discount_percentage'] +
    test['received_card_discount_percentage'] +
    test['received_coupon_discount_percentage']
)

train['effective_product_value'] = train['product_value'] * (1 - train['total_discount_percentage'] / 100)
test['effective_product_value'] = test['product_value'] * (1 - test['total_discount_percentage'] / 100)



train['cost_difference'] = train['product_value'] - train['final_payment']
test['cost_difference'] = test['product_value'] - test['final_payment']


X = train.drop(columns=['customer_experience', 'received_date', 'estimated_delivery_date', 'purchased_datetime', 'released_date', 'date_registered'], axis=1)  
y = train['customer_experience']

mi_scores = make_mi_scores(X, y)
mi_scores


sns.relplot(x='customer_experience', y='payment_method', data=train)


train['shipping_reliability'] = (train['shipping_method_express'] + train['shipping_method_standard'])*train['is_delayed_delivery']
test['shipping_reliability'] = (test['shipping_method_express'] + test['shipping_method_standard'])*test['is_delayed_delivery']


X = train.drop(columns=['customer_experience', 'received_date', 'estimated_delivery_date', 'purchased_datetime', 'released_date', 'date_registered'], axis=1)  
y = train['customer_experience']

mi_scores = make_mi_scores(X, y)
mi_scores


train['customer_tenure'] = (train['received_date'] - train['date_registered']).dt.days
test['customer_tenure'] = (test['received_date'] - test['date_registered']).dt.days

train['loyalty_engagement'] = train['customer_tenure'] * train['loyalty_tier']
test['loyalty_engagement'] = test['customer_tenure'] * test['loyalty_tier']



train['loyalty_engagement'] = train['customer_tenure'] * train['loyalty_tier']
test['loyalty_engagement'] = test['customer_tenure'] * test['loyalty_tier']


train['customer_lifetime_value'] = train['customer_tenure'] * train['final_payment']
test['customer_lifetime_value'] = test['customer_tenure'] * test['final_payment']


X = train.drop(columns=['customer_experience', 'received_date', 'estimated_delivery_date', 'purchased_datetime', 'released_date', 'date_registered'], axis=1)  
y = train['customer_experience']

mi_scores = make_mi_scores(X, y)
mi_scores


train.info()


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

def make_mi_scores_and_plot(X, y):
    X = X.copy()
    
    # Align X and y
    X, y = X.align(y, axis=0, join='inner')
    X = X.dropna()  # Ensure no NaNs remain in X or y
    y = y.loc[X.index]  # Align y with X

    # Ensure all features are numeric
    X = X.apply(pd.to_numeric, errors='coerce').dropna(axis=1)
    
    # Identify discrete features with a threshold
    discrete_features = [
        pd.api.types.is_integer_dtype(X[col]) and X[col].nunique() < 20
        for col in X.columns
    ]
    
    # Drop low-variance features
    low_variance_cols = X.columns[X.var() == 0]
    if not low_variance_cols.empty:
        print("Dropping low variance columns:", list(low_variance_cols))
        X = X.drop(columns=low_variance_cols)

    # Debugging Outputs
    print("Final X shape:", X.shape)
    print("First few rows of X:\n", X.head())
    
    # Calculate MI scores
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)

    # Print MI scores
    print("Mutual Information Scores:")
    print(mi_scores)

    # Plotting MI Scores
    def plot_mi_scores(scores):
        scores = scores.sort_values(ascending=True)  # Sort for better visualization
        width = np.arange(len(scores))  # Create indices for y-axis
        ticks = list(scores.index)  # Get feature names as ticks

        plt.figure(dpi=100, figsize=(10, 8))  # Adjust figure size
        plt.barh(width, scores, color='skyblue')  # Horizontal bar chart
        plt.yticks(width, ticks, fontsize=10)  # Feature names on y-axis
        plt.xlabel("Mutual Information Score", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.title("Mutual Information Scores", fontsize=14)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()  # Optimize spacing
        plt.show()

    # Plot the MI Scores
    plot_mi_scores(mi_scores)

    return mi_scores




X = train.drop(columns=['customer_experience'], axis=1)  
y = train['customer_experience']

mi_scores = make_mi_scores_and_plot(X, y)

# print(mi_scores)
# print(mi_scores.tail(20))  

# plt.figure(dpi=100, figsize=(8, 5))
# plot_mi_scores(mi_scores)
# plot_mi_scores(mi_scores.tail(20))  


features = [
    'received_dayofweek',
    'discount',
    'days_to_deliver',
    'est_days_to_deliver',
    'prod_age',
    'shipping_meth_xdelay',
    'is_delayed_delivery',
    'reg_year',
    'shipping_method',
    'diff_est_actual',
    'Received_coupon_discount_percentage',
    'purchase_medium',
    'Is_current_loyalty_program_member',
    'Received_card_discount_percentage',
    'received_year',
    'purchased_year',
    'est_deivery_year',
    'Product_value',
    'released_year',
    'loyalty_tier',
    'Received_tier_discount_percentage',
    'Gender',
    'est_delivery_dayofweek',
    'total_discount_percentage',
    'reg_month',
    'loyalty_points_redeemed',
    'reg_dayofweek',
    'age',
    'released_dayofweek',
    'purchased_dayofweek',
    'id',
    'payment_method',
    'released_month',
    'est_delivery_month',
    'final_payment'
]



best_features = [
    'received_dayofweek',
    'days_to_deliver',
    'est_days_to_deliver',
    'customer_tenure',
    'discount',
    'shipping_meth_xdelay',
    'is_delayed_delivery',
    'Is_current_loyalty_program_member',
    'loyalty_tier',
    'loyalty_benefit',
    'Received_card_discount_percentage',
    'age',
    'purchase_medium',
    'Gender'
]



from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

X = train.drop(columns=['customer_experience', 'received_date', 'estimated_delivery_date', 'purchased_datetime', 'released_date', 'date_registered'], axis=1)  
y = train['customer_experience']

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
val_X_scaled = scaler.transform(val_X)

train_X = pd.DataFrame(train_X_scaled, columns=train_X.columns, index=train_X.index)
val_X = pd.DataFrame(val_X_scaled, columns=val_X.columns, index=val_X.index)


train_X.info()


# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import GridSearchCV

# dt_model = DecisionTreeClassifier(random_state=42)

# param_grid = {
#     'max_depth': [3, 5, 10, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'criterion': ['gini', 'entropy']
# }

# grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)

# print("Best Parameters:", grid_search.best_params_)
# print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# best_model = grid_search.best_estimator_
# test_accuracy = best_model.score(X_test, y_test)
# print("Test Accuracy:", test_accuracy)


from sklearn.metrics import classification_report, confusion_matrix

def predict(train_X, train_y, val_X, val_y, model):

    model.fit(train_X, train_y)
    pred_y = model.predict(val_X)
    print("\nClassification Report:\n", classification_report(val_y, pred_y))
    
    cm = confusion_matrix(val_y, pred_y)

    labels = ['Bad', 'Neutral','Good']
    
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

predict(train_X, train_y, val_X, val_y, rf_model)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=100)

predict(train_X, train_y, val_X, val_y, model)


from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier( n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42 )

predict(train_X, train_y, val_X, val_y, gb_model)


from sklearn.tree import DecisionTreeClassifier

rf_model = DecisionTreeClassifier(max_depth=11, random_state=42, min_samples_leaf=4, min_samples_split=2)

predict(train_X, train_y, val_X, val_y, rf_model)


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation

fold = 1
for train_index, test_index in kf.split(X):
    X_train, val_X = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = predict(X_train, y_train, val_X, y_test, model)
    # print("\nClassification Report:\n", classification_report(y_test, y_pred))
    fold += 1


final_model = DecisionTreeClassifier(max_depth=10, random_state=42, min_samples_leaf=4, min_samples_split=2)
final_model.fit(X, y)
predictions = final_model.predict(test.drop(columns=['received_date', 'estimated_delivery_date', 'purchased_datetime', 'released_date', 'date_registered'], axis=1))
predictions


submission = pd.read_csv('/kaggle/input/Predict-the-Customer-Satisfaction-CSE-22/sample_submission.csv')
submission


submission['customer_experience'] = predictions
submission


submission['customer_experience'] = submission['customer_experience'].replace(0,'bad')
submission['customer_experience'] = submission['customer_experience'].replace(1,'neutral')
submission['customer_experience'] = submission['customer_experience'].replace(2,'good')


submission.to_csv("submission.csv", index=False)

