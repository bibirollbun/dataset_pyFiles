# import libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from tqdm.notebook import tqdm
import warnings
import random
import optuna

warnings.simplefilter("ignore")


# Static Variable.
train_filepath = "/kaggle/input/playground-series-s5e12/train.csv"
test_filepath = "/kaggle/input/playground-series-s5e12/test.csv"
random_state = 42
target = "diagnosed_diabetes"
test_size = 0.2


def style_column_headers(df):
    
    # Define the CSS styles for the header (<th> tag)
    header_styles = [
        {
            # Apply to ALL column headers
            'selector': 'th',
            'props': [
                ('background-color', '#E6F7FF'),  # Dark Green background
                ('color', '#004085'),              # White text color
                ('font-family', 'sans-serif'),   # Clean font
                ('font-size', '14px'),
                ('text-align', 'center'),
                ('border', '1px solid black')    # Add a border
            ]
        }
    ]
    
    # Apply the styles and render the DataFrame
    return df.style.set_table_styles(header_styles)


# Create a function to handle or the processing of the data.
def process(filepath:str, transform=False)->pd.DataFrame:
    "This reads the csv file and transform it where neccessary"
    # Read the csv filepath.
    df = pd.read_csv(filepath, index_col= "id")

    # control flow.
    if transform:
        transformation = {}
        # Starting with age 
        df["age_group"] = pd.cut(df["age"], 3, labels=["Youth", "Adult", "Elderly"]).astype("object")
        transformation["age_group"] = "Age is Grouped into 3 categories"

        # df["cholesterol_unkown"] = (df["cholesterol_total"] - (df["hdl_cholesterol"] + df["ldl_cholesterol"]))
        # transformation["cholesterol_unknown"] = "Unknown Cholesterol"

        df["bp_diff"] = (df["systolic_bp"] - df["diastolic_bp"])
        transformation["bp_diff"] = "Blood Pressure Difference"

        df["smoking_status"] = df["smoking_status"].replace({"Never": 0, "Former": 1, "Current": 2})
        transformation["smoking_status"] = "Encoding Smoking Status"

        df["CRI"] = df["cholesterol_total"] + df["triglycerides"] + df["systolic_bp"]
        transformation["CRI"] = "Cardiovascular Risk Index"

        df["bmi_group"] = pd.cut(df["bmi"], 3, labels=["Low", "Moderate", "High"]).astype("object")
        transformation["bmi_group"] = "BMI is Grouped into 3 categories"

        df["heart_group"] = pd.cut(df["heart_rate"], 3, labels=["Low", "Moderate", "High"]).astype("object")
        transformation["heart_group"] = "Heart rate is Grouped into 3 categories"

        df["history"] = df[['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']].sum(axis=1)
        transformation["history"] = "Past Event"

        df["physical_activity_minutes_per_week"] = np.log(df["physical_activity_minutes_per_week"])
        transformation["physical_activity_minutes_per_week"] = "Transform the Distribution"

        df["new"] = df.groupby(["smoking_status", "family_history_diabetes", "alcohol_consumption_per_week", "age"])["physical_activity_minutes_per_week"].transform("mean")
        transformation["new"] = "Grouping by the most important column"
        
        series_transformation = pd.DataFrame(transformation, index=["Description"]).T
        display(style_column_headers(series_transformation))
        print("=" * 70)

    # Get the information regarding the data.
    display(df.info(verbose=True))
    print("=" * 70)
    # Get the Statisical Distribution regarding the dataset.
    des = display(style_column_headers(df.describe().T))
    print("=" * 70)

    # Drop high cardinality categorical variable.
    df = df.drop(columns=["cholesterol_total", "waist_to_hip_ratio"])
    
    return df


df = process(train_filepath, transform=True)
test_df = process(test_filepath, transform=True)
df.head()


sns.heatmap(df.select_dtypes("number").corr());


abs(df.select_dtypes("number").corr()["diagnosed_diabetes"]).sort_values()


df.nunique().sort_values()


target_dis = df[target].value_counts(normalize=True)
target_dis
fig, ax = plt.subplots(figsize=(6, 4))
target_plot = ax.bar(height= target_dis.values, x=target_dis.index)
ax.set_xticks([0, 1])
ax.bar_label(target_plot, fmt='%.2f%%')
plt.xlabel(target)
plt.ylabel("Frequency")
plt.title("Target Distribution")
plt.tight_layout();


# Create a function to plot.
def plot(data: pd.DataFrame, column_name: str, target= target) -> plt.figure:
    """
    This Create a plot for the particular column.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    column_title_name = column_name.replace("_", " ").title()
        
    # Check the unique value for the column.
    if df[column_name].nunique() < 10:
        # Create a clustered barchart and a pie chart for the column.
        vis = df.groupby([target, column_name])[column_name].value_counts().reset_index()
        
        sns.barplot(x= vis[column_name], y=vis["count"], hue=vis[target], ax=ax1)
        ax1.set_ylabel("Frequency")
        ax1.set_xlabel(column_title_name)
        ax1.set_title(f"Distribution of {column_title_name} against the Diagnose Diabetes")
        
        column_profile = df[column_name].value_counts()
        ax2.pie(column_profile, labels=column_profile.index, autopct="%.2f%%", radius=1)
        ax2.set_title(f"Distribution of {column_title_name}")
        
        plt.suptitle(f"{column_title_name}")
        plt.tight_layout();
    else:
        # Create a boxplot for the column as well as the horizontal barchart for the column
        sns.boxplot(y= df[column_name], x=df[target], ax=ax1)
        ax1.set_ylabel("Range")
        ax1.set_xlabel("Diagnose Diabetes")
        ax1.set_title(f"Distribution of {column_title_name} against the Diagnose Diabetes")

        vis = df[column_name].value_counts().tail(10).sort_values()
        vis.plot(kind="barh", ax=ax2)
        ax2.set_ylabel(column_title_name)
        ax2.set_xlabel("Frequency")
        ax2.set_title(f"Top 10 {column_title_name} Distribution")

        plt.suptitle(f"{column_title_name}")
        plt.tight_layout();


columns = df.drop(columns=target).nunique().index.tolist()
choices = random.choices(columns, k=3)
for choice in choices:
    plot(df, choice)


# Create a function that takes in different model.
def models_score(models:list, df:pd.DataFrame, test_size=test_size, random_state=random_state, test_df=None, target=target):
    # split the data.
    X = df.drop(columns=target)
    y = df[target]
    
    # training and test set.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model_score = {}
    # Loop through each the models.
    for model in tqdm(models, desc="Processing"):
        # Create a pipeline.
        model_arch = make_pipeline(
            OrdinalEncoder(),
            StandardScaler(),
            model
        )
        print(f"Training {model}")
        # Train the model.
        model_arch.fit(X_train, y_train)
        
        # Get the last item in the model architecture.
        model_name = list(model_arch.named_steps.keys())[-1]
        # Get the prediction for both the train and test.
        y_train_pred = model_arch.predict_proba(X_train)[:, 1]
        y_test_pred = model_arch.predict_proba(X_test)[:, 1]
    
        roc_train = roc_auc_score(y_train, y_train_pred)
        roc_test = roc_auc_score(y_test, y_test_pred)
        print("Scoring Completed")
        model_score[model_name] = [roc_train, roc_test]

        if test_df is not None:
            test_pred = model_arch.predict_proba(test_df)[:, 1]
            sub_df = pd.DataFrame({"diagnosed_diabetes": test_pred}, index=test_df.index)
            sub_df.to_csv(f"{model_name}.csv")
            print(f"\nSubmission File for {model_name} Created.")
        
    # Create dataframe.
    df = pd.DataFrame(model_score, index=["Traing ROC_AUC Score", 
                                          "Test ROC_AUC Score"]).T
    df = style_column_headers(df)
        
    return df


best_lgb_params = {'n_estimators': 546,
 'learning_rate': 0.04302800218068587,
 'num_leaves': 27,
 'max_depth': 10,
 'min_child_samples': 94}
xgb_params = {'max_depth': 3,
 'learning_rate': 0.1959335824862102,
 'n_estimators': 775,
 'min_child_weight': 9,
 'gamma': 0.4396600592994249}
lgb_params = {'n_estimators': 1794, 'learning_rate': 0.018670185412964255, 'num_leaves': 20, 'max_depth': 15, 'min_child_samples': 86}
cat_params = {'iterations': 1304, 'depth': 4, 'learning_rate': 0.1261673651357195}
model = [CatBoostClassifier(**cat_params, random_state=random_state,verbose=0),
        # GradientBoostingClassifier(random_state=random_state),
        # AdaBoostClassifier(random_state=random_state),
        XGBClassifier(**xgb_params, random_state=random_state),
        LGBMClassifier(**lgb_params, random_state=random_state, verbose=-1)]
models_score(model, df, test_df = test_df)


clf1 = make_pipeline(
    OrdinalEncoder(),
    StandardScaler(),
    CatBoostClassifier(**cat_params, random_state=random_state,verbose=0)
)
clf2 = make_pipeline(
    OrdinalEncoder(),
    StandardScaler(),
    XGBClassifier(**xgb_params, random_state=random_state)
)
clf3 = make_pipeline(
    OrdinalEncoder(),
    StandardScaler(),
    LGBMClassifier(**lgb_params, random_state=random_state,verbose=-1)
)

estimators = [
    ("cat", clf1),
    ("xgb", clf2),
    ("lgb", clf3)
]
model = VotingClassifier(estimators, voting="soft", weights = [3, 1, 2])
X = df.drop(columns=target)
y = df[target]

# training and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
model.fit(X_train, y_train)
y_test_pred = model.predict_proba(X_test)[:, 1]
test_pred = model.predict_proba(test_df)[:, 1]
sub_df = pd.DataFrame({"diagnosed_diabetes": test_pred}, index=test_df.index)
sub_df.to_csv(f"voting.csv")


# from optuna.exceptions import TrialPruned
# def objective(trial):
    
#     # Optuna suggests a value for each hyperparameter in this trial
#     param = {
#         "iterations": trial.suggest_int("iterations", 500, 2000),
#         "depth": trial.suggest_int("depth", 4, 10),
#         "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.3, log=True),
#     }

#     # split the data.
#     X = df.drop(columns=target)
#     y = df[target]
    
#     # training and test set.
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
#     model_arch = make_pipeline(
#             OneHotEncoder(use_cat_names=True),
#             StandardScaler(),
#             CatBoostClassifier(**param, random_state=random_state, verbose = 0)
#     )
#     # Train the model.
#     model_arch.fit(X_train, y_train)
    
#     # Get the last item in the model architecture.
#     model_name = list(model_arch.named_steps.keys())[-1]
#     # Get the prediction for both the train and test.
#     y_test_pred = model_arch.predict_proba(X_test)[:, 1]
#     y_train_pred = model_arch.predict_proba(X_train)[:, 1]
#     roc_test = roc_auc_score(y_test, y_test_pred)
#     roc_train = roc_auc_score(y_train, y_train_pred)
#     if roc_train - roc_test >= 0.01:
#         raise TrialPruned(f"{roc_test}: Score below threshold — prune trial")
#     else:
#         roc_test = roc_auc_score(y_test, y_test_pred)
#         return roc_test


# study = optuna.create_study(
#     direction="maximize",
#     sampler=optuna.samplers.TPESampler(seed=42), # Use TPE Sampler for efficiency
#     study_name='Cat_Optuna_Tuning'
# )


# study.optimize(
#     objective, 
#     n_trials=100, 
#     show_progress_bar=True
# )


# best_params = study.best_params
# best_params


# optuna.visualization.plot_param_importances(study)


# optuna.visualization.plot_slice(study)

