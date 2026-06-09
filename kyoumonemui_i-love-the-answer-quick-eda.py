import pandas as pd
train_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
train_df.loc[10781].StudentExplanation


pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)


train_df


train_df.Category.value_counts()


train_df.groupby('QuestionId').apply(lambda x:len(x)).sort_values()


count_category_df = train_df.groupby('QuestionId').apply(lambda x: x.Category.value_counts()).reset_index().pivot(index='QuestionId', columns=['Category'], values=['count']).fillna(0)
count_category_df


count_category_ratio_df = count_category_df / count_category_df.sum(axis=1).values[:,None]
count_category_ratio_df


count_category_ratio_df.describe()


for id in train_df.QuestionId.unique():
    print(train_df[train_df.QuestionId==id].QuestionText.iloc[0])
    print("\n".join(train_df[train_df.QuestionId==id].MC_Answer.unique()),end='\n\n')
    print("\n\n".join(train_df[train_df.QuestionId==id].StudentExplanation.head(5)))
    display(train_df[train_df.QuestionId==id].head(5))




