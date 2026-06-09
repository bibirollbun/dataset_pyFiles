import warnings
warnings.filterwarnings('ignore')
import pandas as pd


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

sample = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


print(train.info())
print('--------------------------------------------------------')
print(test.info())


from sklearn.impute import SimpleImputer

# Columns to impute (based only on train columns)
numeric_cols = train.select_dtypes(include=['number']).drop(columns=['id']).columns.tolist()
categorical_cols = train.select_dtypes(include=['object', 'category', 'string']).drop(columns=['Personality']).columns.tolist()

# Imputers
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

for df_name, df in [('train', train), ('test', test)]:
    print(f"\nHandling missing values for: {df_name}")
    
    # Only use columns that exist in current DataFrame
    numeric_cols_in_df = [col for col in numeric_cols if col in df.columns]
    categorical_cols_in_df = [col for col in categorical_cols if col in df.columns]

    # Numeric imputation
    df[numeric_cols_in_df] = num_imputer.fit_transform(df[numeric_cols_in_df])
    print(f"[{df_name}] Numeric columns filled with mean.")

    # Categorical imputation
    df[categorical_cols_in_df] = cat_imputer.fit_transform(df[categorical_cols_in_df])
    print(f"[{df_name}] Categorical columns filled with mode.")


print(train.info())
print('--------------------------------------------------------')
print(test.info())


X_train = train.drop(columns=['id', 'Personality'])
y_train = train['Personality']

X_test = test.drop(columns=['id'])


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


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = GaussianNB()

model.fit(X_tr, y_tr)

y_pred = model.predict(X_val)

print("Validation accuracy with Naive Bayes:", accuracy_score(y_val, y_pred))


y_test_pred = model.predict(X_test)

y_test_labels = target_encoder.inverse_transform(y_test_pred)

submission = test[['id']].copy()
submission['Personality'] = y_test_labels

submission.to_csv('submission.csv', index=False)




