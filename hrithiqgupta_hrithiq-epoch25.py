import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

train_df = pd.read_csv('GalacticTravel_Train.csv')
test_df  = pd.read_csv('GalacticTravel_TestInput.csv')


train_df.info()
train_df.describe()


print(train_df.isnull().sum())


plt.figure(figsize=(10, 6))
train_df['travel_outcome'].value_counts().plot(kind='bar')
plt.title('Distribution of Travel Outcomes')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.show()


# correlation heatmap for numerical features
numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = train_df[numeric_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot(x='ticket_class', hue='travel_outcome', data=train_df)
plt.title('Travel Outcome by Ticket Class')
plt.show()

plt.figure(figsize=(10, 6))
train_df['route'].hist(bins=20)
plt.title('Distribution of Route Lengths')
plt.xlabel('Number of Stops')
plt.ylabel('Frequency')
plt.show()


numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns

fig, axes = plt.subplots(nrows=(len(numerical_cols) + 1) // 2, ncols=2, figsize=(20, 5 * ((len(numerical_cols) + 1) // 2)))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    sns.boxplot(x='travel_outcome', y=col, data=train_df, ax=axes[i])
    axes[i].set_title(f'{col} vs Travel Outcome')
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()



categorical_cols = train_df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    plt.figure(figsize=(12, 6))
    sns.countplot(x=col, hue='travel_outcome', data=train_df)
    plt.title(f'{col} vs Travel Outcome')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Travel Outcome', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


#removed death in space as there was one row of it(hindered with smote)
train_df.drop(train_df[train_df['travel_outcome'] == 'Death in Space'].index, inplace=True)

#dates's useless for pred
train_df = train_df.drop(columns=['arrival_date', 'purchase_date', 'departure_date'], errors='ignore')
test_df  = test_df.drop(columns=['arrival_date', 'purchase_date', 'departure_date'], errors='ignore')

#consistency of datatype 
bool_cols = train_df.select_dtypes(include=['bool']).columns
train_df[bool_cols] = train_df[bool_cols].astype(int)
test_df[bool_cols] = test_df[bool_cols].astype(int)

# external datasets
additional_costs     = pd.read_csv("AdditionalCosts.csv")
black_market_info    = pd.read_csv("BlackMarketInfo.csv")
delays               = pd.read_csv("Delays.csv")
dest_status          = pd.read_csv("DestStatus.csv")
duty_free            = pd.read_csv("DutyFree.csv")
entertainment        = pd.read_csv("Entertainment.csv")
foreign_acceptance   = pd.read_csv("ForeignAcceptance.csv")
kyber_conversion     = pd.read_csv("KyberConversion.csv")
past_incidences      = pd.read_csv("PastIncidences.csv")
republic             = pd.read_csv("Republic.csv")
stranded             = pd.read_csv("Stranded.csv")
unruliness           = pd.read_csv("Unruliness.csv")

# Encoding attributes
# encode emerging_smuggling_risk ("Low", "Medium", "High")
mapping = {"Low": 0, "Medium": 1, "High": 2}
black_market_info["emerging_smuggling_risk"] = black_market_info["emerging_smuggling_risk"].map(mapping)

# encode galactic_prestige_grade using categorical codes
dest_status["galactic_prestige_grade"] = dest_status["galactic_prestige_grade"].astype("category").cat.codes

# encode galactic_accomaodation using categorical codes
entertainment["galactic_accomaodation"] = entertainment["galactic_accomaodation"].astype("category").cat.codes

kyber_conversion["crystal_category"] = kyber_conversion["crystal_category"].astype("category").cat.codes

republic["enhanced_security_level"] = republic["enhanced_security_level"].map(mapping)

# adding prefix to avoid errors and collision
additional_costs     = additional_costs.add_prefix("ac_")
black_market_info    = black_market_info.add_prefix("bmi_")
delays               = delays.add_prefix("del_")
dest_status          = dest_status.add_prefix("ds_")
duty_free            = duty_free.add_prefix("df_")
entertainment        = entertainment.add_prefix("ent_")
foreign_acceptance   = foreign_acceptance.add_prefix("fa_")
kyber_conversion     = kyber_conversion.add_prefix("kc_")
past_incidences      = past_incidences.add_prefix("pi_")
republic             = republic.add_prefix("rep_")
stranded             = stranded.add_prefix("str_")
unruliness           = unruliness.add_prefix("unr_")

# merging with the main data.
train_df = pd.concat([
    train_df.reset_index(drop=True),
    additional_costs.reset_index(drop=True),
    black_market_info.reset_index(drop=True),
    delays.reset_index(drop=True),
    dest_status.reset_index(drop=True),
    duty_free.reset_index(drop=True),
    entertainment.reset_index(drop=True),
    foreign_acceptance.reset_index(drop=True),
    kyber_conversion.reset_index(drop=True),
    past_incidences.reset_index(drop=True),
    republic.reset_index(drop=True),
    stranded.reset_index(drop=True),
    unruliness.reset_index(drop=True)
], axis=1)

test_df = pd.concat([
    test_df.reset_index(drop=True),
    additional_costs.reset_index(drop=True),
    black_market_info.reset_index(drop=True),
    delays.reset_index(drop=True),
    dest_status.reset_index(drop=True),
    duty_free.reset_index(drop=True),
    entertainment.reset_index(drop=True),
    foreign_acceptance.reset_index(drop=True),
    kyber_conversion.reset_index(drop=True),
    past_incidences.reset_index(drop=True),
    republic.reset_index(drop=True),
    stranded.reset_index(drop=True),
    unruliness.reset_index(drop=True)
], axis=1)



# if "route" in train_df.columns:
#     train_df["route_first"] = train_df["route"].apply(lambda x: x.split(">")[0].strip() if isinstance(x, str) else "")
#     train_df["route_last"] = train_df["route"].apply(lambda x: x.split(">")[-1].strip() if isinstance(x, str) else "")
#     train_df["route_length"] = train_df["route"].apply(lambda x: len(x.split(">")) if isinstance(x, str) else 0)
#     train_df = train_df.drop(columns=["route"], errors='ignore')
# if "route" in test_df.columns:
#     test_df["route_first"] = test_df["route"].apply(lambda x: x.split(">")[0].strip() if isinstance(x, str) else "")
#     test_df["route_last"] = test_df["route"].apply(lambda x: x.split(">")[-1].strip() if isinstance(x, str) else "")
#     test_df["route_length"] = test_df["route"].apply(lambda x: len(x.split(">")) if isinstance(x, str) else 0)
#     test_df = test_df.drop(columns=["route"], errors='ignore')

# if "passenger_frustration_factor" in train_df.columns:
#     train_df["passenger_frustration_factor"] = train_df["passenger_frustration_factor"].abs()
# if "passenger_frustration_factor" in test_df.columns:
#     test_df["passenger_frustration_factor"] = test_df["passenger_frustration_factor"].abs()

#
# if "pi_passenger_frustration_factor" in train_df.columns:
#     train_df["pi_passenger_frustration_factor"] = train_df["pi_passenger_frustration_factor"].abs()
# if "pi_passenger_frustration_factor" in test_df.columns:
#     test_df["pi_passenger_frustration_factor"] = test_df["pi_passenger_frustration_factor"].abs()


train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

target = 'travel_outcome'
train_df[target] = train_df[target].fillna('Missing').astype(str)

#dropping attributes
drop_cols = ['travel_outcome', 'passenger_name', 'passenger_id', 'arrival_date']
X = train_df.drop(columns=drop_cols, errors='ignore').copy()
y = train_df[target].copy()
X_test = test_df.drop(columns=['ID'], errors='ignore').copy()

# from sklearn.impute import SimpleImputer
# cat_cols = X.select_dtypes(include=['object']).columns.tolist()
# num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# imputer_num = SimpleImputer(strategy='mean')
# if num_cols:
#     X[num_cols] = imputer_num.fit_transform(X[num_cols])
#     X_test[num_cols] = imputer_num.transform(X_test[num_cols])

# imputer_cat = SimpleImputer(strategy='most_frequent')
# if cat_cols:
#     X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])
#     X_test[cat_cols] = imputer_cat.transform(X_test[cat_cols])

# Label encoding
le = LabelEncoder()
y = le.fit_transform(y)

for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype(str)
for col in X_test.select_dtypes(include=['object']).columns:
    X_test[col] = X_test[col].astype(str)

#pipeline for scaling and encoding attributes 
cats = X.select_dtypes(include=['object']).columns.tolist()
nums = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), nums),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cats)
    ],
    remainder='passthrough'
)


# from imblearn.over_sampling import SMOTENC
# categorical_features_indices = [X.columns.get_loc(col) for col in cat_features]
# smote = SMOTENC(categorical_features=categorical_features_indices, random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)


model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, 
                                 eval_metric='logloss',max_depth=5,learning_rate=0.1,n_estimators=100))
])

# model_pipeline = Pipeline([
#     ('preprocessor', preprocessor),
#     ('classifier', XGBClassifier(
#         max_depth=5,
#         learning_rate=0.1,
#         n_estimators=150,
#         random_state=42,
#         colsample_bytree=0.8,
#         reg_lambda=1,
#         scale_pos_weight=1,
#         gamma=0.1,
#         subsample=1.0,
#         use_label_encoder=False,
#         eval_metric='rmse'
#     ))
# ])

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, random_state=42)
model.fit(X_train, y_train)
y_val_pred = model.predict(X_val)
print(accuracy_score(y_val, y_val_pred))

#Hyperparameter tuning(this was done my trial and error in the end I just removed it)
# param_grid = {
#     'classifier__max_depth': [5],
#     'classifier__learning_rate': [0.1],
#     'classifier__n_estimators': [100]
#     'classifier__subsample': [1.0, 1.2, 1.4],
#     'classifier__gamma': [0, 0.1, 0.2, 0.3],
#     'classifier__reg_alpha': [0, 0.1, 1, 10],
#     'classifier__reg_lambda': [0, 0.1, 1, 10],
#     'classifier__scale_pos_weight': [1, 3, 5]
# }
# cv = KFold(n_splits=3, shuffle=True, random_state=42)
# grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
# grid_search.fit(X, y)
# print(grid_search.best_params_)
# print(grid_search.best_score_)

# best_model = grid_search.best_estimator_
test_preds_num = model.predict(X_test)
# original labels
test_preds = le.inverse_transform(test_preds_num)

#The data merging ended up causing extra rows to be output so I just took the first 500 outputs 
output = pd.DataFrame({
    'ID': test_df['ID'][:500],
    'travel_outcome': test_preds[:500]
})

sub = 'submission_xg.csv'
output.to_csv(sub, index=False)


