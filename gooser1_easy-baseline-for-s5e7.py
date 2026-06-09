import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_score, accuracy_score


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


X = train_df.copy().drop("Personality", axis=1)
y = train_df["Personality"].copy()



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


cat_atts = ["Stage_fear", "Stage_fear"]

X_train_cat = X_train[cat_atts]

# Missing: Handeling of cat nan values #

# one_hot = OneHotEncoder()

#X_train_cat_enc = one_hot.fit_transform(X_train_cat)


num_atts = ["Time_spent_Alone", "Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency"]

X_train_num = X_train[num_atts]

num_simple_imput = SimpleImputer(strategy="median")

min_max = MinMaxScaler()

num_pipeline = Pipeline([
    ("imput", SimpleImputer(strategy="median")),
    ("scale", MinMaxScaler()) # Change range for NN!
])


combined = ColumnTransformer([
    ("cat", OneHotEncoder(), cat_atts),
    ("num", num_pipeline, num_atts),
])


X_train_prepro = combined.fit_transform(X_train)
X_val_prepro = combined.fit_transform(X_val)


rfc = RandomForestClassifier(n_estimators=200, max_leaf_nodes=12, random_state=42)


rfc.fit(X_train_prepro, y_train)


y_train_pred = rfc.predict(X_train_prepro)
# Cross-validation missing


accuracy_score(y_train, y_train_pred)


y_val_pred = rfc.predict(X_val_prepro)


accuracy_score(y_val, y_val_pred)


X_test_prepro = combined.fit_transform(test_df)

y_test_pred = rfc.predict(X_test_prepro)


submission = pd.DataFrame(
    {
        'id': test_df['id'].copy(),
        'Personality': y_test_pred
    }
)
submission.to_csv("submission.csv", index = None)


submission




