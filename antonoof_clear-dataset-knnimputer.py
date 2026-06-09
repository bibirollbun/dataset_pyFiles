import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


train.head()


train.columns


train.info()


categorical_features = ['Stage_fear', 'Drained_after_socializing']
features_with_NaN = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
       'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']


ordinal_encoder = OrdinalEncoder()

train[categorical_features] = ordinal_encoder.fit_transform(train[categorical_features])
test[categorical_features] = ordinal_encoder.transform(test[categorical_features])


imputer = KNNImputer(n_neighbors=1)

train[features_with_NaN] = imputer.fit_transform(train[features_with_NaN])
test[features_with_NaN] = imputer.transform(test[features_with_NaN])


train.head()


train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)

