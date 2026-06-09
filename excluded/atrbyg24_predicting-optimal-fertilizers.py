import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv',index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


train.head()


train.info()


train.describe()


cat_cols = ['Soil Type','Crop Type','Fertilizer Name']
num_cols = ['Temparature','Humidity','Moisture','Nitrogen','Potassium','Phosphorous']


for col in cat_cols:
    print(train[col].value_counts())


for col in num_cols:
    plt.figure()
    ax = sns.histplot(data = train,x=col,hue='Fertilizer Name',multiple='stack')
    plt.title(f'Histogram of {col}')
    sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.show()
    plt.clf()


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train['Fertilizer Name'].copy())
X = train.drop('Fertilizer Name', axis=1).copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=17)


def ratio_feature(X):
    return X[:,[0]] / (X[:,[1]] + 1e-6)

def ratio_name(function_transformer,feature_names_in):
    return ['ratio']

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(ratio_feature,feature_names_out=ratio_name),
        StandardScaler()
    )

def sum_feature(X):
    return X[:,[0]] + X[:,[1]] + X[:,[2]]

def sum_name(function_transformer,feature_names_in):
    return ['sum']

def sum_pipeline():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(sum_feature,feature_names_out=sum_name),
        StandardScaler()
    )

def average_feature(X):
    return (X[:,[0]] + X[:,[1]])/2

def average_name(function_transformer,feature_names_in):
    return ['avg']

def average_pipeline():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(average_feature,feature_names_out=average_name),
        StandardScaler()
    )

def difference_feature(X):
    return X[:,[0]] - X[:,[1]]

def difference_name(function_transformer,feature_names_in):
    return ['diff']

def difference_pipeline():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(difference_feature,feature_names_out=difference_name),
        StandardScaler()
    )

nominal_transformer = Pipeline(steps=[
    ('imputter',SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('NP_ratio',ratio_pipeline(),['Nitrogen','Phosphorous']),
    ('NK_ratio', ratio_pipeline(),['Nitrogen','Potassium']),
    ('PK_ratio', ratio_pipeline(),['Potassium','Phosphorous']),
    ('NPK_sum', sum_pipeline(),['Nitrogen','Potassium','Phosphorous']),
    ('Climate',average_pipeline(),['Temparature','Humidity']),
    ('Water',difference_pipeline(),['Humidity','Moisture']),
    ('nominal', nominal_transformer, ['Soil Type', 'Crop Type'])
], remainder='passthrough')

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier(objective='multi:softprob',
                            eval_metric='mlogloss',
                            use_label_encoder=False,
                            random_state=42))
])


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


probabilities = model.predict_proba(X_test)
print("\nPredicted probabilities on test set (first 5 samples):\n", probabilities[:5])


print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

classes = label_encoder.inverse_transform(model.named_steps['classifier'].classes_)

results = []
for i, prob_array in enumerate(probabilities[:5]): 
    sorted_indices = np.argsort(prob_array)[::-1]
    top3 = []
    for j in range(min(3, len(classes))):
        class_label = classes[sorted_indices[j]]
        top3.append(class_label)
    results.append(" ".join(top3))
print(results)


test_preds = model.predict_proba(test)
top_3_fertilizers = []
for i, pred in enumerate(test_preds):
    top_indices = np.argsort(pred)[::-1]  
    top3 = []
    for j in range(min(3, len(classes))):
        class_label = classes[sorted_indices[j]]
        top3.append(class_label)
    top_3_fertilizers.append(" ".join(top3))

submission = pd.DataFrame({'id': test['id'], 'Fertilizer Name': top_3_fertilizers})

submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")



submission.head(20)




