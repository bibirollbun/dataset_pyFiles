import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
from plotly import express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


raw_data = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
original_data = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer_Prediction.csv')
raw_data.head()


raw_data.info()
original_data.info()


def get_numerical_summary(df):
    total = raw_data.shape[0]
    missing_columns = [col for col in raw_data.columns if raw_data[col].isnull().sum() > 0]
    missing_percent = {}
    for col in missing_columns:
        null_count = raw_data[col].isnull().sum()
        per = (null_count/total) * 100
        missing_percent[col] = per
        print("{} : {} ({}%)".format(col, null_count, round(per, 3)))
    return missing_percent

missing_percent = get_numerical_summary(raw_data)
missing_percent_original = get_numerical_summary(original_data)

msno.matrix(raw_data)
msno.matrix(original_data)


df_full = raw_data.copy()

df_full.rename(columns={'Soil Type':'Soil_Type', 'Crop Type':'Crop_Type', 'Fertilizer Name':'Fertilizer_Name'}, inplace=True)
original_data.rename(columns={'Soil Type':'Soil_Type', 'Crop Type':'Crop_Type', 'Fertilizer Name':'Fertilizer_Name'}, inplace=True)

num_columns = [col for col in df_full.columns if df_full[col].dtype != 'object' and col!='id']
cat_columns = [col for col in df_full.columns if df_full[col].dtype == 'object']

print ("numeric columns count:", len(num_columns))
print ("categorical columns count:", len(cat_columns))

print(df_full[num_columns].describe())
print(original_data[num_columns].describe())


for col in num_columns:
    fig = go.Figure(px.box(df_full, y = col, title = ('Box Plot of '+col)))
    fig.update_layout(title_x=0.5)
    fig.show()


for col in num_columns:
    fig = go.Figure(px.box(original_data, y = col, title = ('Box Plot of '+col)))
    fig.update_layout(title_x=0.5)
    fig.show()


for col in df_full[cat_columns]:
    print(col, df_full[col].unique())

for col in original_data[cat_columns]:
    print(col, original_data[col].unique())


label = LabelEncoder()
df_full['Fertilizer_Name'] = label.fit_transform(df_full['Fertilizer_Name'])
original_data['Fertilizer_Name'] = label.fit_transform(original_data['Fertilizer_Name'])

X = df_full.drop(columns=['id', 'Fertilizer_Name'])
y = df_full['Fertilizer_Name']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.015, random_state=24)

X_ori = original_data.drop(columns=['Fertilizer_Name'])
y_ori = original_data['Fertilizer_Name']

X_train = pd.concat([X_train, X_ori], ignore_index=True)
y_train = pd.concat([y_train, y_ori], ignore_index=True)


category_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']

for i in category_cols:
    cat_encoder = LabelEncoder()
    X_train[i] = cat_encoder.fit_transform(X_train[i])
    X_train[i] = X_train[i].astype("category")

    X_test[i] = cat_encoder.transform(X_test[i])
    X_test[i] = X_test[i].astype("category")

X_train.info()
X_test.info()


corr_var = [col for col in X_train.columns if (X_train[col].dtype != 'object') and (col not in ["Soil_Type", "Crop_Type"])]
corr = X_train[corr_var].corr()

# Correlation Heatmap
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)


rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
feature_importance_df


# Initialize XGBClassifier
xgb_classifier = XGBClassifier( max_depth=11,
                                colsample_bytree=0.6,
                                subsample=0.9,
                                n_estimators=3200,
                                learning_rate=0.05,
                                gamma=0.6,
                                max_delta_step=1,
                                reg_alpha=25,
                                reg_lambda=25,
                                early_stopping_rounds=200,
                                objective='multi:softprob',
                                eval_metric=['mlogloss', 'merror'],
                                random_state=24,
                                enable_categorical=True,
                                tree_method='hist'                                
                               )

# This is for timer
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        print('Starting time is ', start_time)
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('End time is ', start_time)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

start_time = timer(None) # timing starts from this point for "start_time" variable
xgb_classifier.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)
timer(start_time) # timing ends here for "start_time" variable


xgb_preds = xgb_classifier.predict(X_test)
print(accuracy_score(y_test, xgb_preds))
print(classification_report(y_test,xgb_preds))

train_preds = xgb_classifier.predict(X_train)
print(accuracy_score(y_train, train_preds))
print(classification_report(y_train,train_preds))


y_pred_probs = xgb_classifier.predict_proba(X_test)
top_3_preds = np.argsort(y_pred_probs, axis=1)[:, -3:][:, ::-1]  
actual = [[label] for label in y_test]

def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
map3_score = mapk(actual, top_3_preds)
print(f"✅ MAP@3 Score: {map3_score:.5f}")


evals_result = xgb_classifier.evals_result()

mlogloss = evals_result['validation_0']['mlogloss']
merror = evals_result['validation_0']['merror']
epochs = range(len(mlogloss))

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(epochs, mlogloss, label='mlogloss', color='steelblue')
ax1.set_xlabel('Boosting Round')
ax1.set_ylabel('mlogloss', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')

ax2 = ax1.twinx()
ax2.plot(epochs, merror, label='merror', color='darkorange')
ax2.set_ylabel('merror', color='darkorange')
ax2.tick_params(axis='y', labelcolor='darkorange')

plt.title('Validation mlogloss and merror (Dual Axis)')
fig.tight_layout()
plt.grid(True)
plt.show()


test_set=pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
test_set.rename(columns={'Soil Type':'Soil_Type', 'Crop Type':'Crop_Type', 'Fertilizer Name':'Fertilizer_Name'}, inplace=True)

for i in category_cols:
    test_set[i] = cat_encoder.fit_transform(test_set[i])
    test_set[i] = test_set[i].astype("category")

test_set_id = test_set.copy()
test_set.drop(columns=['id'], inplace=True)

test_probs = xgb_classifier.predict_proba(test_set)
top_3_preds = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]
top_3_labels = label.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)
submission_competition= pd.DataFrame({'id': test_set_id['id'],
                                      'Fertilizer Name': [' '.join(row) for row in top_3_labels]})
submission_competition


submission_competition.to_csv('submission.csv', index=False)




