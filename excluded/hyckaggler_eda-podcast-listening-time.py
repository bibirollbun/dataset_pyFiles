import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_float_dtype


lt_df_train = pd.read_csv(r"/kaggle/input/playground-series-s5e4/train.csv",index_col='id')
lt_df_test = pd.read_csv(r"/kaggle/input/playground-series-s5e4/test.csv",index_col='id')


display(lt_df_train.head())


# trim spaces at beginning and end of string
# Capitalize first letter of each word
# search for spaces between words - after trimming beginning and end
# Any other cleaning for categorical variables?


lt_df_train_obj = lt_df_train.select_dtypes('object')
lt_df_test_obj = lt_df_test.select_dtypes('object')

print("\nTrain synth data categorical variables before trim:\n" + str(lt_df_train[lt_df_train_obj.columns].nunique()))
print("\nTest synth categorical variables before trim:\n" + str(lt_df_test[lt_df_train_obj.columns].nunique()))
lt_df_train[lt_df_train_obj.columns] = lt_df_train_obj.transform(lambda x: x.str.strip())
lt_df_test[lt_df_test_obj.columns] = lt_df_test_obj.transform(lambda x: x.str.strip())
print("\nTrain synth categorical variables after trim:\n" + str(lt_df_train[lt_df_train_obj.columns].nunique()))
print("\nTest synth categorical variables after trim:\n" + str(lt_df_test[lt_df_train_obj.columns].nunique()))


print("\nTrain synth data categorical variables before capitalizing first letter of each word:\n" + str(lt_df_train[lt_df_train_obj.columns].nunique()))
print("\nTest synth categorical variables before capitalizing first letter of each word:\n" + str(lt_df_test[lt_df_train_obj.columns].nunique()))
lt_df_train[lt_df_train_obj.columns] = lt_df_train_obj.transform(lambda x: x.str.title())
lt_df_test[lt_df_test_obj.columns] = lt_df_test_obj.transform(lambda x: x.str.title())
print("\nTrain synth categorical variables after capitalizing first letter of each word:\n" + str(lt_df_train[lt_df_train_obj.columns].nunique()))
print("\nTest synth categorical variables after capitalizing first letter of each word:\n" + str(lt_df_test[lt_df_train_obj.columns].nunique()))


print(lt_df_train_obj.apply(lambda x: x.str.contains(' ', regex=False)).any())


print(lt_df_train.shape)
print(lt_df_train.dtypes)


print(lt_df_test.shape)
print(lt_df_test.dtypes)


print("Index: " + str(lt_df_train.index.duplicated().any()))
print("Row: " + str(lt_df_train.duplicated().any()))


print("Index: " + str(lt_df_test.index.duplicated().any()))
print("Row: " + str(lt_df_test.duplicated().any()))


for col in lt_df_train.loc[:,['Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage','Number_of_Ads']].columns:
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    sns.boxplot(data=lt_df_train,y=col,ax=ax[0])
    ax[0].set_title('Synthetic Train Data')
    sns.boxplot(data=lt_df_test,y=col,ax=ax[1])
    ax[1].set_title('Synthetic Test Data')
    plt.show()
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
sns.boxplot(data=lt_df_train,y='Listening_Time_minutes',ax=ax)
ax.set_title('Synthetic Train Data')
plt.show()


display(lt_df_train[lt_df_train['Episode_Length_minutes']>250])


display(lt_df_test[lt_df_test['Episode_Length_minutes']>250])


display(lt_df_train[lt_df_train['Number_of_Ads']>10])


display(lt_df_test[lt_df_test['Number_of_Ads']>10])


outlier_indexes_ep_length_syn_train = lt_df_train[lt_df_train['Episode_Length_minutes']>250].index
lt_df_train.loc[outlier_indexes_ep_length_syn_train,['Episode_Length_minutes']] = lt_df_train['Episode_Length_minutes'].drop(outlier_indexes_ep_length_syn_train).mean(skipna=True).round(1)
display(lt_df_train.loc[outlier_indexes_ep_length_syn_train])


outlier_indexes_ep_length_syn_test = lt_df_test[lt_df_test['Episode_Length_minutes']>250].index
lt_df_test.loc[outlier_indexes_ep_length_syn_test,['Episode_Length_minutes']] = lt_df_train['Episode_Length_minutes'].mean(skipna=True).round(1)
display(lt_df_test.loc[outlier_indexes_ep_length_syn_test])


outlier_indexes_num_ads_syn_train = lt_df_train[lt_df_train['Number_of_Ads']>10].index
lt_df_train.loc[outlier_indexes_num_ads_syn_train,['Number_of_Ads']] = lt_df_train['Number_of_Ads'].drop(outlier_indexes_num_ads_syn_train).mean(skipna=True).round(2)
display(lt_df_train.loc[outlier_indexes_num_ads_syn_train])


outlier_indexes_num_ads_syn_test = lt_df_test[lt_df_test['Number_of_Ads']>10].index
lt_df_test.loc[outlier_indexes_num_ads_syn_test,['Number_of_Ads']] = lt_df_train['Number_of_Ads'].mean(skipna=True).round(2)
display(lt_df_test.loc[outlier_indexes_num_ads_syn_test])


for col in lt_df_train.loc[:,['Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage','Number_of_Ads']].columns:
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    sns.histplot(data=lt_df_train,x=col,kde=True,ax=ax[0])
    ax[0].set_title('Synthetic Train Data')
    sns.histplot(data=lt_df_test,x=col,kde=True,ax=ax[1])
    ax[1].set_title('Synthetic Test Data')
    plt.show()
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
sns.histplot(data=lt_df_train,x='Listening_Time_minutes',kde=True,ax=ax)
ax.set_title('Synthetic Train Data')
plt.show()


lt_df_train.isnull().any()


lt_df_train['Episode_Length_minutes'] = lt_df_train.groupby(['Podcast_Name','Episode_Title'])['Episode_Length_minutes'].transform(lambda x: x.fillna(x.mean()))
display(lt_df_train[lt_df_train['Episode_Length_minutes'].isnull()].head())


lt_df_train['Guest_Popularity_percentage'] = lt_df_train.groupby(['Podcast_Name','Episode_Title'])['Guest_Popularity_percentage'].transform(lambda x: x.fillna(x.mean()))
display(lt_df_train[lt_df_train['Guest_Popularity_percentage'].isnull()].head())


lt_df_train['Number_of_Ads'] = lt_df_train.groupby(['Podcast_Name','Episode_Title'])['Number_of_Ads'].transform(lambda x: x.fillna(x.mean()))
display(lt_df_train[lt_df_train['Number_of_Ads'].isnull()].head())


lt_df_test.isnull().any()


lt_df_test['Episode_Length_minutes'] = lt_df_test.groupby(['Podcast_Name','Episode_Title'])['Episode_Length_minutes'].transform(lambda x: x.fillna(x.mean()))
display(lt_df_test[lt_df_test['Episode_Length_minutes'].isnull()].head())


lt_df_test['Guest_Popularity_percentage'] = lt_df_test.groupby(['Podcast_Name','Episode_Title'])['Guest_Popularity_percentage'].transform(lambda x: x.fillna(x.mean()))
display(lt_df_test[lt_df_test['Guest_Popularity_percentage'].isnull()].head())

