import warnings
warnings.filterwarnings('ignore')
import pandas as pd


com_train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
com_test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

first_ds1_df = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv')
sec_ds1_df = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_dataset.csv')

first_ds2_df = pd.read_csv('/kaggle/input/personality-prediction-data-introvert-extrovert/personality_dataset.csv')
sec_ds2_df = pd.read_excel('/kaggle/input/personality-prediction-data-introvert-extrovert/personality_dataset.xlsx', sheet_name='Sheet1')


print(first_ds1_df.equals(sec_ds1_df))
print('-----')
print(first_ds2_df.equals(sec_ds2_df))


com_train_df.drop('id', axis=1, inplace=True)


com_df_cols = com_train_df.columns.tolist()
ds1_df1_cols = first_ds1_df.columns.tolist()
ds1_df2_cols = sec_ds1_df.columns.tolist()
ds2_df1_cols = first_ds2_df.columns.tolist()


if com_df_cols == ds1_df1_cols == ds1_df2_cols == ds2_df1_cols:
    print('Different datasets have the same column names and are ready to be merged.')
else:
    print('Datasets do not have the same columns; each dataset must be processed.')


merged_df = pd.concat([com_train_df, first_ds1_df, sec_ds1_df, first_ds2_df], ignore_index=True)


test_id = com_test_df['id']
com_test_df.drop('id', axis=1, inplace=True)


merged_df.info()
print('--------------------------------------------------------')
com_test_df.info()


for df_name, df in [('merged_df', merged_df), ('com_test_df', com_test_df)]:
    print(f"\nStarting to handle missing values. DataFrame: {df_name}")

    for column in df.columns:
        if df[column].isnull().any():

            # If the column is numerical
            if pd.api.types.is_numeric_dtype(df[column]):
                mean_value = df[column].mean()
                df[column] = df[column].fillna(mean_value)
                print(f"[{df_name}]The column '{column}' was compensated with the average: {mean_value}")

            # If the column is text (object or string)
            elif pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
                if not df[column].mode().empty:
                    mode_value = df[column].mode()[0]
                    df[column] = df[column].fillna(mode_value)
                    print(f"[{df_name}] The column '{column}' has been compensated with the mode: {mode_value}")
                else:
                    print(f"[{df_name}] column '{column}' Does not contain a value for the mode; no substitution has been made.")


print(merged_df.info())
print('--------------------------------------------------------')
print(com_test_df.info())


X_train = merged_df.drop(columns='Personality')
y_train = merged_df['Personality']

X_test = com_test_df


from sklearn.preprocessing import LabelEncoder

categorical_cols = ['Stage_fear', 'Drained_after_socializing']

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    encoders[col] = le


target_encoder = LabelEncoder()
y_train = target_encoder.fit_transform(y_train)


import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_tr, y_tr)

y_pred = model.predict(X_val)
print("Validation accuracy with XGBoost:", accuracy_score(y_val, y_pred))


y_test_pred = model.predict(X_test)

y_test_labels = target_encoder.inverse_transform(y_test_pred)

submission = pd.DataFrame({'id': test_id, 'Personality': y_test_labels})

submission.to_csv('submission.csv', index=False)




