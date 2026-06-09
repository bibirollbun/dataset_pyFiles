import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


df=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df.sample(5)


df.shape





le = LabelEncoder()
df['Personality_encoded']= le.fit_transform(df['Personality'])

df.head(5)


# Using .map() for 'Stage_fear'and 'Drained_after_socializing'

df['Stage_fear']=df['Stage_fear'].map({'Yes':1,'No':0})
df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes':1,'No':0})

df.sample(5)


df.isnull().sum()


cols = [
    'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
    'Going_outside', 'Drained_after_socializing',
    'Friends_circle_size', 'Post_frequency'
]

si = SimpleImputer(strategy = 'median')
df[cols]=si.fit_transform(df[cols])

df.isnull().sum()


features =[
    'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
    'Going_outside', 'Drained_after_socializing',
    'Friends_circle_size', 'Post_frequency'
]
X= df[features]
y= df['Personality_encoded']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

print("Shape of X_train: ",X_train.shape)
print("Shape of X_test: ",X_test.shape)


df['Personality'].value_counts().plot(kind='bar')



df[features].hist(figsize=(12,8))
# x-> column value, y->count


plt.figure(figsize=(10,8))
sns.heatmap(df[features+['Personality_encoded']].corr(),annot=True,cmap='coolwarm')
plt.show()


plt.figure(figsize=(12, len(features) * 2))

for i, col in enumerate(features):
    plt.subplot((len(features) + 1) // 2, 2, i + 1)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()


lgr = LogisticRegression(max_iter=1000)

lgr.fit(X_train,y_train)
y_pred = lgr.predict(X_test)

# test_pred = lgr.predict(test_df[features])
# test_pred_labels = le.inverse_transform(test_pred)

accuracy = accuracy_score(y_test,y_pred)*100
print(f"Accuracy of logistic regression :{accuracy:.4f}")


rf = RandomForestClassifier(random_state=42)
rf.fit(X_train,y_train)
y_pred_rf=rf.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf) * 100
print(f"Accuracy of Random Forest: {accuracy_rf:.4f}")


test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

test_df['Stage_fear'] = test_df['Stage_fear'].map({'Yes':1,'No':0})
test_df['Drained_after_socializing'] = test_df['Drained_after_socializing'].map({'Yes':1,'No':0})

test_df[cols] = si.transform(test_df[cols])

test_pred = lgr.predict(test_df[features])

test_pred_labels = le.inverse_transform(test_pred)

submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': test_pred_labels
})

submission.to_csv('submission.csv', index=False)




