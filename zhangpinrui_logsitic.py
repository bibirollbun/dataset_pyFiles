import pandas as pd


def data_process(filepath):
    df_train = pd.read_csv(filepath)
    # print(df.head())
    print(df_train.describe())
    # 独处
    df_train["Time_spent_Alone"] = df_train["Time_spent_Alone"].fillna(df_train["Time_spent_Alone"].mean())


    # 舞台时间用众数
    df_train['Stage_fear']=df_train['Stage_fear'].fillna(df_train['Stage_fear'].mode()[0])
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    df_train['Stage_fear'] = le.fit_transform(df_train['Stage_fear'])
    # 社交时间
    df_train['Social_event_attendance']=df_train['Social_event_attendance'].fillna(df_train['Social_event_attendance'].mean())
    # print(df_train['Stage_fear'].value_counts())
    # 外出时间
    df_train['Going_outside']=df_train['Going_outside'].fillna(df_train['Going_outside'].mean())
    # 社交后精疲力尽的
    df_train['Drained_after_socializing']=df_train['Drained_after_socializing'].fillna(df_train['Drained_after_socializing'].mode()[0])
    le = LabelEncoder()
    df_train['Drained_after_socializing'] = le.fit_transform(df_train['Drained_after_socializing'])
    # 好友圈大小
    df_train['Friends_circle_size']=df_train['Friends_circle_size'].fillna(df_train['Friends_circle_size'].mean())
    # 帖子频率
    df_train['Post_frequency']=df_train['Post_frequency'].fillna(df_train['Post_frequency'].mean())
    # print(df_train.isna().sum())

    df_train['Personality'] = df_train['Personality'].map({'Extrovert':0,"Introvert":1})
    df_train.to_csv("/kaggle/working/trained", index=False, encoding="utf-8")
    print(df_train.describe())
    print("✅ 已保存到/kaggle/working")
if __name__ == '__main__':
    data_process('/kaggle/input/playground-series-s5e7/train.csv')
    # print(df_train[''])


import pandas as pd


def data_process(filepath):
    df_train = pd.read_csv(filepath)
    # print(df.head())
    print(df_train.describe())
    # 独处
    df_train["Time_spent_Alone"] = df_train["Time_spent_Alone"].fillna(df_train["Time_spent_Alone"].mean())


    # 舞台时间用众数
    df_train['Stage_fear']=df_train['Stage_fear'].fillna(df_train['Stage_fear'].mode()[0])
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    df_train['Stage_fear'] = le.fit_transform(df_train['Stage_fear'])
    # 社交时间
    df_train['Social_event_attendance']=df_train['Social_event_attendance'].fillna(df_train['Social_event_attendance'].mean())
    # print(df_train['Stage_fear'].value_counts())
    # 外出时间
    df_train['Going_outside']=df_train['Going_outside'].fillna(df_train['Going_outside'].mean())
    # 社交后精疲力尽的
    df_train['Drained_after_socializing']=df_train['Drained_after_socializing'].fillna(df_train['Drained_after_socializing'].mode()[0])
    le = LabelEncoder()
    df_train['Drained_after_socializing'] = le.fit_transform(df_train['Drained_after_socializing'])
    # 好友圈大小
    df_train['Friends_circle_size']=df_train['Friends_circle_size'].fillna(df_train['Friends_circle_size'].mean())
    # 帖子频率
    df_train['Post_frequency']=df_train['Post_frequency'].fillna(df_train['Post_frequency'].mean())
    # print(df_train.isna().sum())

    # df_train['Personality'] = df_train['Personality'].map({'Extrovert':0,"Introvert":1})
    df_train.to_csv("/kaggle/working/tested", index=False, encoding="utf-8")
    print(df_train.describe())
    print("✅ 已保存到/kaggle/working")
if __name__ == '__main__':
    data_process('/kaggle/input/playground-series-s5e7/test.csv')
    # print(df_train[''])


from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("/kaggle/working/trained")
x_test = pd.read_csv("/kaggle/working/tested")
# print(df.head())
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
# print(X)
# print(y)
pro = StandardScaler()
x = pro.fit_transform(X)
x_test = pro.transform(x_test)
# 构建模型
model = LogisticRegression()
model.fit(x, y)
pred = model.predict(x_test)
df_test_raw = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
test_ids = df_test_raw['id']
y_prob = pd.Series(pred).map({0: 'Extrovert', 1: 'Introvert'})
df_submission = pd.DataFrame({
    'id': test_ids,
    'y': y_prob
})

# 保存为 CSV
df_submission.to_csv('/kaggle/working/submission.csv', index=False)


