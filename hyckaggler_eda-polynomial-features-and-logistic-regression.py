import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold,cross_val_score
from sklearn.metrics import accuracy_score,recall_score,precision_score,roc_auc_score
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

rainfall_df = pd.read_csv(r"/kaggle/input/playground-series-s5e3/train.csv",index_col='id')


display(rainfall_df.head())


print(rainfall_df.shape)


print(rainfall_df.duplicated().any())


rainfall_df.isnull().any()


numerical_cols_not_day = rainfall_df.loc[:,~rainfall_df.columns.isin(['day','rainfall'])].columns
for i in numerical_cols_not_day:
    print(i+': '+str(rainfall_df[i].max()),str(rainfall_df[i].min()))


rainfall_df[['maxtemp','temparature','mintemp','dewpoint']].boxplot()
plt.xticks(rotation=45)
plt.show()

rainfall_df[['pressure']].boxplot()
plt.xticks(rotation=45)
plt.show()

rainfall_df[['humidity','cloud']].boxplot()
plt.xticks(rotation=45)
plt.show()

rainfall_df[['sunshine']].boxplot()
plt.xticks(rotation=45)
plt.show()

rainfall_df[['winddirection']].boxplot()
plt.xticks(rotation=45)
plt.show()

rainfall_df[['windspeed']].boxplot()
plt.xticks(rotation=45)
plt.show()


dewpoint_calc = rainfall_df['temparature'] - ((100 - rainfall_df['humidity'])/5)
dewpoint_test = dewpoint_calc.between(rainfall_df['dewpoint']-6,rainfall_df['dewpoint']+6) # Td = T - ((100 - RH)/5)
display(pd.DataFrame(dewpoint_calc[dewpoint_test==False]).merge(rainfall_df['dewpoint'][dewpoint_test==False],on='id'))


display(rainfall_df.loc[1801])
# I don't have much knowledge about dew point so I will assume that this difference between the dewpoint data point provided for this row and the one I
# calculated is acceptable and leave the data point in.


for i in numerical_cols_not_day:
    sns.histplot(data=rainfall_df, x=i, kde=True)
    plt.show()


rainfall_df.corr()


poly = PolynomialFeatures(2)
poly.set_output(transform="pandas")
rainfall_df_poly = poly.fit_transform(rainfall_df.loc[:,~rainfall_df.columns.isin(['rainfall'])])
display(rainfall_df_poly)


merged_poly_with_target = rainfall_df_poly.merge(rainfall_df[['rainfall']],left_index=True,right_index=True)
abs_corr_merged_poly_with_target_rainfall_only_sorted = abs(merged_poly_with_target.corr()['rainfall'].sort_values(ascending=False))
cols_to_select = abs_corr_merged_poly_with_target_rainfall_only_sorted[abs_corr_merged_poly_with_target_rainfall_only_sorted.between(0.4,1)].index
print(list(cols_to_select))


merged_poly_with_target = merged_poly_with_target.loc[:,merged_poly_with_target.columns.isin(cols_to_select)]


scaler = StandardScaler()
scaler.set_output(transform="pandas")
scaler.fit(merged_poly_with_target.loc[:,~merged_poly_with_target.columns.isin(['rainfall'])])
merged_poly_with_target = scaler.transform(merged_poly_with_target.loc[:,~merged_poly_with_target.columns.isin(['rainfall'])])
merged_poly_with_target = merged_poly_with_target.merge(rainfall_df[['rainfall']],left_index=True,right_index=True)


X_train, X_val, y_train, y_val = train_test_split(
    merged_poly_with_target.drop('rainfall',axis=1), merged_poly_with_target[['rainfall']], test_size=0.2, random_state=42)
clf = LogisticRegression(random_state=42,solver='liblinear').fit(X_train, y_train.values.ravel())
pred_train = clf.predict(X_train)
pred_val = clf.predict(X_val)

print(accuracy_score(y_train,pred_train), recall_score(y_train,pred_train), precision_score(y_train,pred_train),roc_auc_score(y_train,pred_train))
print(accuracy_score(y_val,pred_val), recall_score(y_val,pred_val), precision_score(y_val,pred_val),roc_auc_score(y_val,pred_val))


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(LogisticRegression(solver='liblinear'), merged_poly_with_target.drop('rainfall', axis=1), merged_poly_with_target['rainfall'], cv=cv, scoring="roc_auc")
print(np.mean(scores))


rainfall_df_test = pd.read_csv(r"/kaggle/input/playground-series-s5e3/test.csv",index_col='id')


rainfall_df_test['winddirection'][rainfall_df_test['winddirection'].isnull()==True]


rainfall_df_test['winddirection'].fillna(rainfall_df_test['winddirection'].mean(),inplace=True)


cols_to_select_list = list(cols_to_select)
cols_to_select_list.remove("rainfall")
poly_test = PolynomialFeatures(2)
poly_test.set_output(transform="pandas")
rainfall_df_poly_test = poly_test.fit_transform(rainfall_df_test)
rainfall_df_poly_test = rainfall_df_poly_test.loc[:,rainfall_df_poly_test.columns.isin(cols_to_select_list)]
rainfall_df_poly_test = scaler.transform(rainfall_df_poly_test)


test_pred = clf.predict_proba(rainfall_df_poly_test)[:,1]
rainfall_df_poly_test['rainfall'] = test_pred
submission_df = rainfall_df_poly_test[['rainfall']]
print(submission_df)


submission_df.to_csv(r'/kaggle/working/submission_file.csv', index=True)  

