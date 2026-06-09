import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('/kaggle/input/Predict-the-Customer-Satisfaction-CSE-22/train_dataset.csv')
df


df.info()


df.describe()


def date_time(df):
    date_columns = ['Date_Registered', 'payment_datetime', 'purchased_datetime', 
                    'released_date', 'estimated_delivery_date', 'received_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df

df = date_time(df)


df.fillna({
    'Received_tier_discount_percentage': 0,
    'Received_card_discount_percentage': 0,
    'Received_coupon_discount_percentage': 0,
    'loyalty_tier': 0
}, inplace=True)


df.isna().sum()


df.info()


payment_map = {
    'visa_c': 'Card',
    'visa_d': 'Card',
    'mastercard_c': 'Card',
    'mastercard_d': 'Card',
    'amex': 'Card',

    'gcash': 'E-Wallet',
    'maya': 'E-Wallet',
    'grabpay': 'E-Wallet',
    'shopeepay': 'E-Wallet',
    'coinsph': 'E-Wallet',

    'cash': 'Cash',
    'otc': 'Over-the-Counter',

    'bank_transfer': 'Bank Transfer'
}


df['payment_method_grouped'] = df['payment_method'].map(payment_map)


df['delivery_delay'] = (df['received_date'] - df['estimated_delivery_date']).dt.days
def classify_delivery(delay):
    if delay == 0:
        return 'in time'
    elif delay < 0:
        return 'early'
    else:
        return 'lately'

df['delivery_status'] =df['delivery_delay'].apply(classify_delivery)





df.columns


    Q1 = df['Received_coupon_discount_percentage'].quantile(0.25)
    Q3 = df['Received_coupon_discount_percentage'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df['Received_coupon_discount_percentage'] < lower_bound) | (df['Received_coupon_discount_percentage'] > upper_bound)]


    Q1 = df['Received_card_discount_percentage'].quantile(0.25)
    Q3 = df['Received_card_discount_percentage'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df['Received_card_discount_percentage'] < lower_bound) | (df['Received_card_discount_percentage'] > upper_bound)]





import pandas as pd
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
df = remove_outliers_iqr(df, 'Received_card_discount_percentage')
df = remove_outliers_iqr(df, 'Received_coupon_discount_percentage')





outlier_count = df[df['Product_value'] > 10000].shape[0]
print("Outlier count:", outlier_count)


df.info()





df = df[df['Product_value'] <= 10000]
df = df[df['Received_card_discount_percentage'] <= 5]
df = df[df['Received_coupon_discount_percentage'] <= 5]


df.info()


df.drop(['user_id', 'transaction_id', 'order_id', 'tracking_number'], axis=1, inplace=True)


numeric_cols = df.select_dtypes(include=['int64','int32', 'float64']).columns


def process_dataframe(df):

    # Financial Features
    # Discount amount features
    df['total_discount_percentage'] = df['Received_tier_discount_percentage'] + df['Received_card_discount_percentage'] + df['Received_coupon_discount_percentage']
    df['Total_Discount_Amount'] = df['Product_value'] * ((df['total_discount_percentage']) / 100)
  
    df['discount_amount_ratio'] = df['Total_Discount_Amount'] / (df['Product_value'] + 1e-9)
    df['high_discount_order'] = (df['Total_Discount_Amount'] > df['Total_Discount_Amount'].median()).astype(int)
    
    # shipping cost calculation
    df['shipping_cost'] = df['final_payment'] - (df['Product_value'] - df['Total_Discount_Amount'])
    df['shipping_cost_ratio'] = df['shipping_cost'] / df['Product_value']
    
    # Price sensitivity metrics
    df['price_tier'] = pd.qcut(df['Product_value'], q=5, labels=[0,1,2,3,4])
    df['discount_types_used'] = (df[['Received_tier_discount_percentage', 
                                   'Received_card_discount_percentage', 
                                   'Received_coupon_discount_percentage']] > 0).sum(axis=1)
    
    # Loyalty Features
    df['loyalty_engagement_score'] = df['loyalty_points_redeemed'] / df['Product_value']
    
    # Temporal Features
    # Purchase timing
    df['purchase_hour'] = df['purchased_datetime'].dt.hour
    df['purchase_day_of_week'] = df['purchased_datetime'].dt.dayofweek
    df['is_weekend_purchase'] = df['purchase_day_of_week'].isin([5, 6]).astype(int)
    df['is_business_hours'] = ((df['purchase_hour'] >= 9) & (df['purchase_hour'] <= 17)).astype(int)
    df['purchase_month'] = df['purchased_datetime'].dt.month
    df['purchase_quarter'] = df['purchased_datetime'].dt.quarter
    
    # Delivery Features
    df['receive_day_of_week'] = df['received_date'].dt.dayofweek
    df['processing_days'] = (df['released_date'] - df['purchased_datetime']).dt.days
    df['delivery_days'] = (df['received_date'] - df['released_date']).dt.days
    df['total_order_days'] = (df['received_date'] - df['purchased_datetime']).dt.days
    df['delivery_delay'] = (df['received_date'] - df['estimated_delivery_date']).dt.days
    df['is_delayed'] = (df['delivery_delay'] > 0).astype(int)
    
    # Customer Features
    df['customer_tenure_days'] = (df['purchased_datetime'] - df['Date_Registered']).dt.days
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 25, 35, 50, 65], 
                            labels=[0, 1, 2, 3])
    

    
    return df



df = process_dataframe(df)


cols_to_encode = [
    'Gender', 'Is_current_loyalty_program_member',
    'product_category', 'payment_method', 'purchase_medium',
    'shipping_method',
    'payment_method_grouped', 'delivery_status','price_tier','age_group'
]
df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)








df.info()


[df_col for df_col in df.columns if "experience" in df_col.lower()]



experience_map = {
    'bad': 0,
    'neutral': 1,
    'good': 2
}

df['customer_experience'] = df['customer_experience'].map(experience_map)


X = df.drop(['customer_experience','Date_Registered', 'payment_datetime', 'purchased_datetime', 
                    'released_date', 'estimated_delivery_date', 'received_date'], axis=1)
y = df['customer_experience']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





print(X.columns)


print(X_train.shape)
print(y_train.shape)



from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
xgb_model = XGBClassifier(
    max_depth=6,                
    n_estimators=1000,          
    learning_rate=0.05,         
    subsample=0.8,              
    colsample_bytree=0.8,      
    gamma=1,                    
    reg_alpha=0.5,              
    reg_lambda=1,              
    eval_metric='mlogloss',    
    use_label_encoder=False,
    tree_method='hist',
    random_state=42
)

xgb_model.fit(X_train, y_train)








y_pred = xgb_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


test_df = pd.read_csv("/kaggle/input/Predict-the-Customer-Satisfaction-CSE-22/test_dataset.csv")




X_test = test_df.copy()


for col in X_train.columns:
    if col not in X_test.columns:
        X_test[col] = 0


X_test = X_test[X_train.columns]




y_test_pred = xgb_model.predict(X_test)



y_train = df['customer_experience']



label_map = {
    0: 'bad',
    1: 'neutral',
    2: 'good'
}
predicted_labels = [label_map[i] for i in y_test_pred]


submission = pd.DataFrame({
    'id': test_df['id'], 
    'customer_experience': predicted_labels
})
submission.to_csv('submission.csv', index=False)






