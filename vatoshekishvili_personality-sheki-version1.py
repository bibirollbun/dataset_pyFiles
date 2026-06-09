import pandas as pd
import numpy as np


train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


#Train-Validation Data Split:
trainset = train_df.sample(frac=0.8, random_state=0)
valset = train_df.drop(trainset.index)


def encode_categorical_features(df, status='train', info_dict=None):
    df = df.copy()
    if status == 'train':
        info_dict = {}

    cat_cols = df.select_dtypes(include='object').columns

    for col in cat_cols:
        unique_vals = df[col].dropna().unique()

        # Binary categorical → Label Encoding
        if len(unique_vals) == 2:
            if status == 'train':
                mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                info_dict[col + '_map'] = mapping
            else:
                mapping = info_dict[col + '_map']

            df[col] = df[col].map(mapping)

        # Multi-category → One-Hot Encoding
        else:
            if status == 'train':
                dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
                info_dict[col + '_cols'] = dummies.columns.tolist()
            else:
                dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
                for c in info_dict[col + '_cols']:
                    if c not in dummies.columns:
                        dummies[c] = 0
                dummies = dummies[info_dict[col + '_cols']]

            df = pd.concat([df.drop(col, axis=1), dummies], axis=1)

    return df, info_dict



def prepare_personality_data(df, status='train', info_dict=None):
    df = df.copy()
    if status == 'train':
        info_dict = {}

    #Dropping ID columns bc it doesnot have anything to do with personality
    drop_cols = ['Id']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    if status == 'train':
        info_dict['drop_cols'] = drop_cols

    #Fill Missing Values(Median):
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        if status == 'train':
            median = df[col].median()
            info_dict[col+'_median'] = median
        else:
            median = info_dict[col+'_median']
        df[col] = df[col].fillna(median)

    #Outlier Handling:
    def cap_outliers(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR
        return series.clip(lower, upper)

    for col in num_cols:
        df[col] = cap_outliers(df[col])

   #Scaling:
    for col in num_cols:
        if status == 'train':
            mean = df[col].mean()
            std = df[col].std()
            info_dict[col+'_scale'] = (mean, std)
        else:
            mean, std = info_dict[col+'_scale']
        df[col] = (df[col] - mean) / std

    return df, info_dict



clean_trainset, info_dict = prepare_personality_data(trainset, status='train')

clean_valset, _ = prepare_personality_data(
    valset,
    status='val',
    info_dict=info_dict
)

clean_testset, _ = prepare_personality_data(
    test_df,
    status='test',
    info_dict=info_dict
)


