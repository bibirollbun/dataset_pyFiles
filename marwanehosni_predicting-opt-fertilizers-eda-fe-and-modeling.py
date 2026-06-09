import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway 
import warnings

warnings.filterwarnings('ignore')


df = pd.read_csv('train.csv')
display(df.head(), df.tail(), df.shape)


from scipy.stats import skew, kurtosis

def better_summary(telco_df):
    summary_data = []

    for col in telco_df.columns:
        col_data = telco_df[col]
        col_summary = {
            "Column": col,
            "Data Type": col_data.dtype,
            "Non-Null Count": col_data.notnull().sum(),
            "Missing Count": col_data.isnull().sum(),
            "Missing %": col_data.isnull().mean() * 100,
            "Unique Count": col_data.nunique()
        }

        if pd.api.types.is_numeric_dtype(col_data):
            col_summary.update({
                "Min": col_data.min(),
                "Max": col_data.max(),
                "Mean": col_data.mean(),
                "Median": col_data.median(),
                "Std Dev": col_data.std(),
                "Skewness": skew(col_data.dropna()),
                "Kurtosis": kurtosis(col_data.dropna())
            })
        else:
            mode = col_data.mode().iloc[0] if not col_data.mode().empty else np.nan
            mode_freq = col_data.value_counts().iloc[0] if not col_data.value_counts().empty else np.nan
            col_summary.update({
                "Top (Mode)": mode,
                "Freq": mode_freq
            })

        summary_data.append(col_summary)

    return pd.DataFrame(summary_data).style.format(precision=2).background_gradient(cmap="Blues")

summary = better_summary(df)
summary


categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    print(df[col].value_counts())
    print("---" * 8)


for col in df.select_dtypes(include='number').columns:
    if df[col].nunique() > 2:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        extreme_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        if not extreme_outliers.empty:
            print(f"\nğŸ“Œ Extreme outliers in '{col}': {len(extreme_outliers)} rows")
            display(extreme_outliers[[col]].sort_values(by=col, ascending=False).head(10))

            plt.figure(figsize=(6, 1.5))
            sns.boxplot(data=df, x=col, whis=3, color='skyblue')
            plt.title(f"Boxplot of {col} (Extreme Outliers Shown)")
            plt.grid(True, axis='x')
            plt.tight_layout()
            plt.show()
        else: 
            print("No extreme outliers in the Dataset")


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.countplot(ax=axes[0], data=df, x='Fertilizer Name', color='skyblue')
axes[0].set_title('Count of Fertilizer Names')
axes[0].set_xlabel('Fertilizer Name')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

counts = df['Fertilizer Name'].value_counts()
axes[1].pie(counts, labels=counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
axes[1].set_title('Distribution of Fertilizer Names')
axes[1].axis('equal')

plt.tight_layout()
plt.show()


categorical_columns = df.select_dtypes(include=["object"]).columns
categorical_columns = [col for col in categorical_columns if col != 'Id']

for col in categorical_columns:
    print(f"\nğŸ“Š Fertilizer Name by {col}")
    pivot = pd.pivot_table(
        df,
        index=col,
        columns='Fertilizer Name',
        aggfunc='size',
        fill_value=0
    )
    display(pivot)

    pivot.plot(kind='bar', stacked=True, figsize=(6, 5))
    plt.title(f"Fertilizer Name by {col}")
    plt.ylabel("Count")
    plt.xlabel(col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Fertilizer Label'] = le.fit_transform(df['Fertilizer Name'])

label_to_name = dict(zip(le.transform(le.classes_), le.classes_))

def visualize_strong_correlations(df, target='Fertilizer Label', threshold=0.5):
    numerical_columns = df.select_dtypes(include='number').columns
    numerical_columns = [col for col in numerical_columns if col != 'Id' and col != target]

    correlations = df[numerical_columns + [target]].corr()[target].drop(target)
    strong_corrs = correlations[correlations.abs() >= threshold].sort_values(key=abs, ascending=False)

    if not strong_corrs.empty:
        print(f"ğŸ“ˆ Numerical columns with strong correlation (|r| â‰¥ {threshold}) with {target}:\n")
        for col, corr in strong_corrs.items():
            print(f" - {col}: correlation = {corr:.2f}")
            
            plt.figure(figsize=(6, 4))
            sns.scatterplot(data=df, x=col, y=target, alpha=0.5)
            plt.yticks(
                ticks=sorted(df[target].unique()),
                labels=[label_to_name[i] for i in sorted(df[target].unique())]
            )
            plt.title(f"{target} vs {col}\nCorrelation = {corr:.2f}")
            plt.tight_layout()
            plt.show()
    else:
        print(f"No numerical columns found with correlation above |{threshold}| with {target}.")

visualize_strong_correlations(df)



numerical_columns = df.select_dtypes(include='number').columns
corr = df[numerical_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr[['Fertilizer Label']].sort_values(by='Fertilizer Label', ascending=False), annot=True, cmap='coolwarm')
plt.title("Correlation with Fertilizer Label")
plt.show()


corr_num = df[numerical_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_num, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


for col in ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x='Fertilizer Name', y=col)
    plt.title(f'{col} by Fertilizer')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



from scipy.stats import chi2_contingency

categorical_features = ['Soil Type', 'Crop Type']
categorical_target = 'Fertilizer Name'

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    phi2 = chi2 / n
    v = np.sqrt(phi2 / min(k - 1, r - 1))
    return v

print("\n--- Chi-squared Test and Cramer's V Results (Categorical Feature vs. Categorical Target) ---")
for feature in categorical_features:
    contingency_table = pd.crosstab(df[feature], df[categorical_target])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    c_v = cramers_v(df[feature], df[categorical_target])

    print(f"Feature: {feature}")
    print(f"  Chi2 Statistic: {chi2:.2f}")
    print(f"  P-value: {p_value:.3f}")
    if p_value < 0.05:
        print(f"  Conclusion: Reject Null Hypothesis. There is a significant association between {feature} and '{categorical_target}'.")
    else:
        print(f"  Conclusion: Fail to Reject Null Hypothesis. No significant association between {feature} and '{categorical_target}'.")
    print(f"  Cramer's V: {c_v:.3f}")
    if c_v < 0.1:
        print("  Strength: Negligible/Very Weak")
    elif 0.1 <= c_v < 0.3:
        print("  Strength: Weak")
    elif 0.3 <= c_v < 0.5:
        print("  Strength: Moderate")
    else:
        print("  Strength: Strong")
    print("-" * 30)

    contingency_table.plot(kind='bar', stacked=True, figsize=(8, 5))
    plt.title(f'Distribution of {categorical_target} by {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title=categorical_target)
    plt.tight_layout()
    plt.show()



def apply_feature_engineering(df):
    df["temperature_humidity"] = df["Temparature"] * df["Humidity"]
    df["humidity_moisture"] = df["Humidity"] * df["Moisture"]
    df["temparature_moisture"] = df["Temparature"] * df["Moisture"]
    df["NP_ratio"] = df["Nitrogen"] / df["Phosphorous"]
    df["NK_ratio"] = df["Nitrogen"] / df["Potassium"]
    df["PK_ratio"] = df["Phosphorous"] / df["Potassium"]
    df["Total_NPK"] = df["Nitrogen"] + df["Phosphorous"] + df["Potassium"]
    df["N_Proportion"] = df["Nitrogen"] / df["Total_NPK"]
    df["P_Proportion"] = df["Phosphorous"] / df["Total_NPK"]
    df["K_Proportion"] = df["Potassium"] / df["Total_NPK"]
    df["Temparature_Squared"] = df["Temparature"] ** 2
    df["Humidity_Squared"] = df["Humidity"] ** 2
    df["Moisture_Squared"] = df["Moisture"] ** 2
    df["Nitrogen_Squared"] = df["Nitrogen"] ** 2
    df["Phosphorous_Squared"] = df["Phosphorous"] ** 2
    df["Potassium_Squared"] = df["Potassium"] ** 2
    return df

df = apply_feature_engineering(df)


numerical_columns = df.select_dtypes(include='number').columns

corr = df[numerical_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr[['Fertilizer Label']].sort_values(by='Fertilizer Label', ascending=False), annot=True, cmap='coolwarm')
plt.title("Correlation with Fertilizer Label")
plt.show()


df_test = pd.read_csv('test.csv')
df_test = apply_feature_engineering(df_test)
df_test.head()
df_test.columns


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

label_to_name = dict(zip(df['Fertilizer Label'], df['Fertilizer Name']))

original_features = [
    'Temparature', 'Humidity', 'Moisture',
    'Soil Type', 'Crop Type',
    'Nitrogen', 'Potassium', 'Phosphorous'
]
X = df[original_features]
y = df['Fertilizer Label']

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

def clean_data(X, y):
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)
    mask = ~X.isna().any(axis=1) & ~y.isna()
    return X.loc[mask], y.loc[mask]

X_train, y_train = clean_data(X_train, y_train)
X_val, y_val = clean_data(X_val, y_val)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', lgb.LGBMClassifier(objective='multiclass',
                                      num_class=len(label_to_name),
                                      random_state=42))
])

model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)
print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")

X_test = df_test[original_features].copy()
test_ids = df_test['id']

X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_test = X_test.applymap(lambda x: np.nan if isinstance(x, (int, float)) and abs(x) > 1e10 else x)

for col in X_test.select_dtypes(include=[np.number]).columns:
    X_test[col].fillna(X_test[col].median(), inplace=True)

for col in X_test.select_dtypes(include=['object']).columns:
    X_test[col].fillna('missing', inplace=True)

test_probabilities = model.predict_proba(X_test)

class_labels = model.named_steps['classifier'].classes_

predictions = []
for probs in test_probabilities:
    top3_idx = np.argsort(probs)[::-1][:3]
    top3_names = [label_to_name[label] for label in class_labels[top3_idx]]
    predictions.append(" ".join(top3_names))

submission_df = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': predictions
})

submission_df.to_csv('submission3.csv', index=False)
print("Submission saved.")
print(submission_df.head())

def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        if a in p[:k]:
            return 1.0 / (p.index(a) + 1)
        return 0.0
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

val_probabilities = model.predict_proba(X_val)
class_labels = model.named_steps['classifier'].classes_

val_predictions = []
for probs in val_probabilities:
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = class_labels[top3_idx].tolist()
    val_predictions.append(top3)

map3_score = mapk(y_val.tolist(), val_predictions, k=3)
print(f"Validation MAP@3: {map3_score:.4f}")



import optuna
from optuna.integration import LightGBMPruningCallback

def objective(trial):
    param = {
        'objective': 'multiclass',
        'num_class': len(label_to_name),
        'metric': 'multi_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42
    }

    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', lgb.LGBMClassifier(**param))
    ])
    
    clf.fit(X_train, y_train)
    
    val_probs = clf.predict_proba(X_val)
    class_labels = clf.named_steps['classifier'].classes_

    val_top3 = []
    for probs in val_probs:
        top3 = class_labels[np.argsort(probs)[::-1][:3]].tolist()
        val_top3.append(top3)

    score = mapk(y_val.tolist(), val_top3, k=3)
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)



best_params = study.best_params
best_params['objective'] = 'multiclass'
best_params['num_class'] = len(label_to_name)
best_params['random_state'] = 42

final_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', lgb.LGBMClassifier(**best_params))
])

final_model.fit(X_train, y_train)
val_probs = final_model.predict_proba(X_val)
val_top3 = []
for probs in val_probs:
    top3 = class_labels[np.argsort(probs)[::-1][:3]].tolist()
    val_top3.append(top3)

print(f"MAP@3 after tuning: {mapk(y_val.tolist(), val_top3, k=3):.4f}")


best_params = {'learning_rate': 0.1432200001835848,
 'num_leaves': 129,
 'max_depth': 8,
 'min_child_samples': 56,
 'subsample': 0.8013183385637461,
 'colsample_bytree': 0.5037042846618193,
 'reg_alpha': 2.1920510340189377e-06,
 'reg_lambda': 0.41410032035894406,
 'objective': 'multiclass',
 'num_class': 7,
 'random_state': 42}


df_fert = pd.read_csv('FertilizerPrediction.csv')
df_combined = pd.concat([df, df_fert], axis=0)
le = LabelEncoder()
df_combined['Fertilizer Label'] = le.fit_transform(df_combined['Fertilizer Name'])
df_combined = apply_feature_engineering(df_combined)
df_combined.drop(columns=[
    'id', 'temperature_humidity', 'humidity_moisture', 
    'temparature_moisture', 'PK_ratio',
    'N_Proportion', 'P_Proportion', 'K_Proportion',
    'Temparature_Squared', 'Humidity_Squared', 'Moisture_Squared',
    'Nitrogen_Squared', 'Phosphorous_Squared', 'Potassium_Squared'
], inplace=True)
df_combined.isna().sum()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import lightgbm as lgb

label_to_name = dict(zip(df_combined['Fertilizer Label'], df_combined['Fertilizer Name']))

original_features = [
    'Temparature', 'Humidity', 'Moisture',
    'Soil Type', 'Crop Type',
    'Nitrogen', 'Potassium', 'Phosphorous'
]
X = df_combined[original_features]
y = df_combined['Fertilizer Label']

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

def clean_data(X, y):
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)
    mask = ~X.isna().any(axis=1) & ~y.isna()
    return X.loc[mask], y.loc[mask]

X_train, y_train = clean_data(X_train, y_train)
X_val, y_val = clean_data(X_val, y_val)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', lgb.LGBMClassifier(**best_params))
])

model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)
print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")

X_test = df_test[original_features].copy()
test_ids = df_test['id']

X_test = X_test.replace([np.inf, -np.inf], np.nan)
print(X_test.isna().sum())
X_test = X_test.applymap(lambda x: np.nan if isinstance(x, (int, float)) and abs(x) > 1e10 else x)
print(X_test.isna().sum())

for col in X_test.select_dtypes(include=[np.number]).columns:
    X_test[col].fillna(X_test[col].median(), inplace=True)

for col in X_test.select_dtypes(include=['object']).columns:
    X_test[col].fillna('missing', inplace=True)

test_probabilities = model.predict_proba(X_test)

class_labels = model.named_steps['classifier'].classes_

predictions = []
for probs in test_probabilities:
    top3_idx = np.argsort(probs)[::-1][:3]
    top3_names = [label_to_name[label] for label in class_labels[top3_idx]]
    predictions.append(" ".join(top3_names))

submission_df = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': predictions
})

submission_df.to_csv('submission6.csv', index=False)
print("Submission saved.")
print(submission_df.head())

def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        if a in p[:k]:
            return 1.0 / (p.index(a) + 1)
        return 0.0
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


val_probabilities = model.predict_proba(X_val)
class_labels = model.named_steps['classifier'].classes_

val_predictions = []
for probs in val_probabilities:
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = class_labels[top3_idx].tolist()
    val_predictions.append(top3)

map3_score = mapk(y_val.tolist(), val_predictions, k=3)
print(f"Validation MAP@3: {map3_score:.4f}")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import xgboost as xgb

label_to_name = dict(zip(df_combined['Fertilizer Label'], df_combined['Fertilizer Name']))

original_features = [
    'Temparature', 'Humidity', 'Moisture',
    'Soil Type', 'Crop Type',
    'Nitrogen', 'Potassium', 'Phosphorous'
]
X = df_combined[original_features]
y = df_combined['Fertilizer Label']

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

def clean_data(X, y):
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)
    mask = ~X.isna().any(axis=1) & ~y.isna()
    return X.loc[mask], y.loc[mask]

X_train, y_train = clean_data(X_train, y_train)
X_val, y_val = clean_data(X_val, y_val)

if 'best_params' not in locals():
    best_params_xgb = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': len(np.unique(y_train)),
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'use_label_encoder': False,
        'random_state': 42
    }
else:
    best_params_xgb = best_params.copy()
    if 'n_estimators' in best_params_xgb:
        best_params_xgb['n_estimators'] = best_params_xgb['n_estimators']
    if 'objective' not in best_params_xgb:
        best_params_xgb['objective'] = 'multi:softprob'
    if 'eval_metric' not in best_params_xgb:
        best_params_xgb['eval_metric'] = 'mlogloss'
    if 'num_class' not in best_params_xgb:
        best_params_xgb['num_class'] = len(np.unique(y_train))
    if 'use_label_encoder' not in best_params_xgb:
        best_params_xgb['use_label_encoder'] = False
    if 'random_state' not in best_params_xgb:
        best_params_xgb['random_state'] = 42

model_xgb = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(**best_params_xgb))
])

model_xgb.fit(X_train, y_train)

y_val_pred = model_xgb.predict(X_val)
print(f"Validation Accuracy (XGBoost): {accuracy_score(y_val, y_val_pred):.4f}")

X_test = df_test[original_features].copy()
test_ids = df_test['id']

X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_test = X_test.applymap(lambda x: np.nan if isinstance(x, (int, float)) and abs(x) > 1e10 else x)

for col in X_test.select_dtypes(include=[np.number]).columns:
    X_test[col].fillna(X_test[col].median(), inplace=True)

for col in X_test.select_dtypes(include=['object']).columns:
    X_test[col].fillna('missing', inplace=True)

test_probabilities = model_xgb.predict_proba(X_test)

class_labels = model_xgb.named_steps['classifier'].classes_

predictions = []
for probs in test_probabilities:
    top3_idx = np.argsort(probs)[::-1][:3]
    top3_names = [label_to_name[label] for label in class_labels[top3_idx]]
    predictions.append(" ".join(top3_names))

submission_df = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': predictions
})

submission_df.to_csv('submission_xgboost.csv', index=False)
print("Submission saved.")
print(submission_df.head())

def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        if a in p[:k]:
            return 1.0 / (p.index(a) + 1)
        return 0.0
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

val_probabilities = model_xgb.predict_proba(X_val)
class_labels = model_xgb.named_steps['classifier'].classes_

val_predictions = []
for probs in val_probabilities:
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = class_labels[top3_idx].tolist()
    val_predictions.append(top3)

map3_score = mapk(y_val.tolist(), val_predictions, k=3)
print(f"Validation MAP@3 (XGBoost): {map3_score:.4f}")


probs_lgb = model.predict_proba(X_val)
probs_xgb = model_xgb.predict_proba(X_val)


combined_probs = (probs_lgb + probs_xgb) / 2

w_lgb = 0.6
w_xgb = 0.4
combined_probs = w_lgb * probs_lgb + w_xgb * probs_xgb



class_labels = model.named_steps['classifier'].classes_ 

combined_val_predictions = []
for probs in combined_probs:
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = class_labels[top3_idx].tolist()
    combined_val_predictions.append(top3)



map3_score_combined = mapk(y_val.tolist(), combined_val_predictions, k=3)
print(f"Validation MAP@3 (Combined): {map3_score_combined:.4f}")



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import numpy as np

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds_lgb = np.zeros((len(X_train), len(class_labels)))
oof_preds_xgb = np.zeros((len(X_train), len(class_labels)))

for train_idx, val_idx in skf.split(X_train, y_train):
    X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model_lgb_cv = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', lgb.LGBMClassifier(**best_params))
    ])
    model_lgb_cv.fit(X_tr, y_tr)
    oof_preds_lgb[val_idx, :] = model_lgb_cv.predict_proba(X_va)

    model_xgb_cv = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(**best_params_xgb))
    ])
    model_xgb_cv.fit(X_tr, y_tr)
    oof_preds_xgb[val_idx, :] = model_xgb_cv.predict_proba(X_va)

stacked_features = np.hstack([oof_preds_lgb, oof_preds_xgb])

meta_model = LogisticRegression(multi_class='multinomial', max_iter=1000)
meta_model.fit(stacked_features, y_train)

val_probs_lgb = model.predict_proba(X_val)
val_probs_xgb = model_xgb.predict_proba(X_val)

val_stacked = np.hstack([val_probs_lgb, val_probs_xgb])

final_probs = meta_model.predict_proba(val_stacked)

final_val_predictions = []
for probs in final_probs:
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = class_labels[top3_idx].tolist()
    final_val_predictions.append(top3)

print(f"Stacked Validation MAP@3: {mapk(y_val.tolist(), final_val_predictions, k=3):.4f}")




from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier



def get_oof_predictions(model_cls, best_params, X, y, preprocessor, n_splits=5, cat_features=None, is_catboost=False):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(X), len(np.unique(y))))
    test_preds = []  

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        if is_catboost:
            model = CatBoostClassifier(**best_params, random_seed=42, verbose=0)
            cat_idx = [X.columns.get_loc(c) for c in cat_features] if cat_features else []
            model.fit(X_train_fold, y_train_fold, cat_features=cat_idx)
            preds = model.predict_proba(X_val_fold)
        else:
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model_cls(**best_params))
            ])
            model.fit(X_train_fold, y_train_fold)
            preds = model.predict_proba(X_val_fold)

        oof_preds[val_idx] = preds

        print(f"Fold {fold+1} done")

    return oof_preds




best_params_lgb = best_params

best_params_xgb = best_params_xgb

best_params_cat = {
    'iterations': 500,
    'learning_rate': 0.05,
    'depth': 6,
    'loss_function': 'MultiClass',
    'verbose': False,
    'random_seed': 42
}



categorical_features = ['Soil Type', 'Crop Type'] 
X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)

print("Getting OOF preds for LightGBM...")
oof_lgb = get_oof_predictions(lgb.LGBMClassifier, best_params_lgb, X_train, y_train, preprocessor)

print("Getting OOF preds for XGBoost...")
oof_xgb = get_oof_predictions(xgb.XGBClassifier, best_params_xgb, X_train, y_train, preprocessor)

print("Getting OOF preds for CatBoost...")
best_params_cat_no_seed = {k: v for k, v in best_params_cat.items() if k not in ['random_seed', 'verbose']}
oof_cat = get_oof_predictions(CatBoostClassifier, best_params_cat_no_seed, X_train, y_train, preprocessor=None, cat_features=categorical_features, is_catboost=True)



from sklearn.linear_model import LogisticRegression

X_meta_train = np.hstack([oof_lgb, oof_xgb, oof_cat])

meta_model = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
meta_model.fit(X_meta_train, y_train)


print("Training base models on full training data...")

model_lgb_full = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', lgb.LGBMClassifier(**best_params_lgb))
])
model_lgb_full.fit(X_train, y_train)
val_pred_lgb = model_lgb_full.predict_proba(X_val)

model_xgb_full = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(**best_params_xgb))
])
model_xgb_full.fit(X_train, y_train)
val_pred_xgb = model_xgb_full.predict_proba(X_val)

cat_params = best_params_cat.copy()
cat_params.pop('random_seed', None)
cat_params.pop('verbose', None)
model_cat_full = CatBoostClassifier(**cat_params, random_seed=42, verbose=0)
model_cat_full.fit(X_train, y_train, cat_features=[X_train.columns.get_loc(c) for c in categorical_features])
val_pred_cat = model_cat_full.predict_proba(X_val)

X_meta_val = np.hstack([val_pred_lgb, val_pred_xgb, val_pred_cat])



final_val_pred = meta_model.predict_proba(X_meta_val)

final_val_preds_top3 = []
class_labels = model_lgb_full.named_steps['classifier'].classes_

for probs in final_val_pred:
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = class_labels[top3_idx].tolist()
    final_val_preds_top3.append(top3)

print(f"Stacked Ensemble Validation MAP@3: {mapk(y_val.tolist(), final_val_preds_top3, k=3):.4f}")



test_pred_lgb = model_lgb_full.predict_proba(X_test)
test_pred_xgb = model_xgb_full.predict_proba(X_test)
test_pred_cat = model_cat_full.predict_proba(X_test)

X_meta_test = np.hstack([test_pred_lgb, test_pred_xgb, test_pred_cat])

final_test_pred = meta_model.predict_proba(X_meta_test)

predictions = []
for probs in final_test_pred:
    top3_idx = np.argsort(probs)[::-1][:3]
    top3_names = [label_to_name[label] for label in class_labels[top3_idx]]
    predictions.append(" ".join(top3_names))

submission_df = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': predictions
})

submission_df.to_csv('submission_stacked.csv', index=False)
print("Stacked submission saved.")



