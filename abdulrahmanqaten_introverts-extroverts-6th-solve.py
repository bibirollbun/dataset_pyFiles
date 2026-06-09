import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

sample = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


print(train.info())
print('--------------------------------------------------------')
print(test.info())


for df_name, df in [('train', train), ('test', test)]:
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



print(train.info())
print('--------------------------------------------------------')
print(test.info())


# Create a new feature that sums related social activity columns
train['Social_Activity_Score'] = train['Social_event_attendance'] + train['Going_outside'] + train['Post_frequency']
test['Social_Activity_Score'] = test['Social_event_attendance'] + test['Going_outside'] + test['Post_frequency']


cat_cols = ['Stage_fear', 'Drained_after_socializing']

# Combine train and test for fitting encoder
combined = pd.concat([train[cat_cols], test[cat_cols]], axis=0)

ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
ohe.fit(combined)

# Transform train set
train_ohe = pd.DataFrame(ohe.transform(train[cat_cols]), columns=ohe.get_feature_names_out(cat_cols), index=train.index)
train = pd.concat([train.drop(columns=cat_cols), train_ohe], axis=1)

# Transform test set
test_ohe = pd.DataFrame(ohe.transform(test[cat_cols]), columns=ohe.get_feature_names_out(cat_cols), index=test.index)
test = pd.concat([test.drop(columns=cat_cols), test_ohe], axis=1)


from sklearn.preprocessing import LabelEncoder

X = train.drop(columns=['id', 'Personality'])
le = LabelEncoder()
y = le.fit_transform(train['Personality'])


model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X, y)


scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy: {scores.mean():.4f}")


X_test = test.drop(columns=['id'])
test_preds = model.predict(X_test)


test_preds_labels = le.inverse_transform(test_preds)


submission = test[['id']].copy()
submission['Personality'] = test_preds_labels
submission.to_csv('submission.csv', index=False)




