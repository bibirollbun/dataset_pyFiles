import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datasets
import os






import kagglehub

# Download latest version
path = kagglehub.dataset_download("kaggle/meta-kaggle")

print("Path to dataset files:", path)






#users_df = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv')
#users_df.head(3)




#users_df[users_df['Id']==1723677] #Chris Deotte 	


#users_df[users_df['Id']==75976] #CPMP 	


messages_df = pd.read_csv('/kaggle/input/meta-kaggle/ForumMessages.csv')


messages_df.head(3)


#messages_df['PostUserId'].value_counts().head(30)


selected_df1 = messages_df[messages_df['PostUserId']==75976].reset_index(drop=True).copy()
selected_df2 = messages_df[messages_df['PostUserId']==1723677].reset_index(drop=True).copy()


selected_df1.head(3)


def split_string(x):
    x = str(x)
    idx = x.find('. ')
    if idx < 0:
        return 0
    else:
        return [x[:idx+1], x[(idx+1):]]

def process_df(df):
    df['split_list'] = df['Message'].map(split_string)
    df = df[df['split_list'] != 0].reset_index(drop=True).copy()
    df['prompt'] = df['split_list'].map(lambda x: x[0])
    df['completion'] = df['split_list'].map(lambda x: x[1])
    df['l1'] = df['prompt'].map(lambda x: len(x))
    df['l2'] = df['completion'].map(lambda x: len(x))
    return df[(df['l1']>1) & (df['l2']>1)]

selected_df1 = process_df(selected_df1)
selected_df2 = process_df(selected_df2)



selected_df1.head(3)


selected_df2.head(3)




selected_df1[['prompt','completion']].to_csv('cpmpml.csv', index=False)

selected_df2[['prompt','completion']].to_csv('cdeotte.csv', index=False)

