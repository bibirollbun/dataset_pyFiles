from IPython.core.display import HTML

HTML("""
<style>

/* Only style Markdown cells, NOT outputs */
div.text_cell_render {
    padding: 20px !important;
    background: #eef6ff !important;
    border-left: 6px solid #005f99 !important;
    border-radius: 10px !important;
    margin-bottom: 20px !important;
    font-family: 'Inter', sans-serif !important;
    color: #000 !important;
}

/* Style markdown headings only */
div.text_cell_render h1,
div.text_cell_render h2,
div.text_cell_render h3 {
    color: #005f99 !important;
    font-weight: 600 !important;
}

</style>
""")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Display prefs
pd.set_option('display.max_columns', 160)
pd.set_option('display.width', 160)
plt.rcParams['figure.dpi'] = 300  # ensure high-res figures





loan_payback_data = pd.read_csv("/Users/ganeshjai/Documents/LoanPayback/.venv/data/playground-series-s5e11/train.csv")
loan_payback_data


import pandas as pd
from autoviz import AutoViz_Class

# if you want charts inside the notebook
%matplotlib inline  

AV = AutoViz_Class()



dft = AV.AutoViz(
    filename="",        # empty when using dfte
    sep=",",
    depVar="",          # no target
    dfte=loan_payback_data,
    header=0,
    verbose=1,
    lowess=False,
    chart_format="svg",  # or 'html', 'bokeh', 'server'
    max_rows_analyzed=150000,
    max_cols_analyzed=30,
    save_plot_dir=None   # or "AutoViz_Plots" or another folder
)



target_col = "loan_paid_back"

dft = AV.AutoViz(
    filename="",
    sep=",",
    depVar=target_col,
    dfte=loan_payback_data,
    header=0,
    verbose=1,
    lowess=False,
    chart_format="svg",     # 'html' will save interactive Bokeh dashboards
    max_rows_analyzed=150000,
    max_cols_analyzed=30,
    save_plot_dir="AutoViz_Plots"
)



loan_payback_data = pd.DataFrame(loan_payback_data)
loan_payback_data


loan_payback_data.head()


loan_payback_data.shape



loan_payback_data.columns



print("\nDTypes summary:")
print(loan_payback_data.dtypes.value_counts())



loan_payback_data.isnull().sum()


loan_payback_data.isnull().mean() * 100


loan_payback_data.duplicated().sum()


loan_payback_data[loan_payback_data.duplicated()]


loan_payback_data.describe()
#Summary for numerical columns 


loan_payback_data.describe(include='all')
#Includes categorical columns as well


loan_payback_data.nunique()
#Count unique values per column 


loan_payback_data.value_counts()


if 'id' in loan_payback_data.columns:
    loan_payback_data = loan_payback_data.drop(columns=['id'])



import pandas as pd
import plotly.express as px

# Get only numeric columns
numeric_df = loan_payback_data.select_dtypes(include=['float64', 'int64']).copy()

# Drop columns you don't want to visualize
for col in ["id", "loan_paid_back"]:
    if col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[col])

numeric_cols = numeric_df.columns.tolist()
numeric_cols



for col in numeric_cols:
    fig = px.histogram(
        numeric_df,
        x=col,
        nbins=40,
        title=f"{col} — Histogram",
        color_discrete_sequence=["#4C78A8"]
    )

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="#f9f9f9",
        paper_bgcolor="#f2f2f2",
        xaxis_title=col,
        yaxis_title="Count",
        bargap=0.05,
        font=dict(size=12)
    )

    fig.show()



loan_payback_data[numeric_cols].skew()



for col in numeric_cols:
    fig = px.box(
        numeric_df,
        x=col,                 # x for horizontal box
        title=f"{col} — Box Plot",
        points="outliers",     # show outlier points
        color_discrete_sequence=["#F58518"]
    )

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="#f9f9f9",
        paper_bgcolor="#f2f2f2",
        xaxis_title=col,
        yaxis_title="",
        font=dict(size=12)
    )

    fig.show()



import plotly.express as px

numeric_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score',
                'loan_amount', 'interest_rate']  # or however you defined them

for col in numeric_cols:
    fig = px.violin(
        loan_payback_data,                       #  use df, not numeric_df
        x="loan_paid_back",       # target column exists in df
        y=col,                    # current numeric feature
        color="loan_paid_back",   # color by target too
        box=True,                 # show inner boxplot
        points="outliers",
        title=f"{col} — Distribution by loan_paid_back",
        color_discrete_sequence=["#4C78A8", "#F58518"],
    )

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="#f9f9f9",
        paper_bgcolor="#f2f2f2",
        xaxis_title="loan_paid_back (0 = no, 1 = yes)",
        yaxis_title=col,
        font=dict(size=12),
    )

    fig.show()



colors = ["#6A0DAD", "#C060FF"]   # deep purple + soft lavender

fig = px.scatter(
    loan_payback_data,
    x="annual_income",
    y="loan_amount",
    color="loan_paid_back",
    title="Annual Income vs Loan Amount (colored by loan_paid_back)",
    color_discrete_sequence=colors
)

fig.update_layout(
    template="plotly_white",
    plot_bgcolor="#f9f9f9",
    paper_bgcolor="#f2f2f2"
)

fig.show()



fig = px.scatter(
    loan_payback_data,
    x="credit_score",
    y="interest_rate",
    color="loan_paid_back",
    title="Credit Score vs Interest Rate",
    color_discrete_sequence=["#008080", "#FF7F0E"])

fig.update_layout(
    template="plotly_white",
    plot_bgcolor="#f9f9f9",
    paper_bgcolor="#f2f2f2"
)

fig.show()



for col in numeric_cols:
    fig = px.box(
        loan_payback_data,
        y=col,      
        points="outliers",  # highlight outliers
        title=f"{col} — Outlier Visualization",
        color_discrete_sequence=["#008080"]
    )

    fig.update_layout(
        template="plotly_white",
        height=350
    )
    fig.show()



import plotly.express as px

counts_df = loan_payback_data["loan_paid_back"].value_counts().reset_index()
counts_df.columns = ["loan_paid_back", "count"]

fig = px.bar(
    counts_df,
    x="loan_paid_back",
    y="count",
    title="Distribution of Loan Repayment (0 vs 1)",
    color="loan_paid_back",
    color_discrete_sequence=["#008080", "#FF7F0E"]
)

fig.update_layout(
    template="plotly_white",
    xaxis_title="loan_paid_back (0 = No, 1 = Yes)",
    yaxis_title="Count",
    plot_bgcolor="#f9f9f9",
    paper_bgcolor="#f2f2f2"
)

fig.show()



fig = px.pie(
    loan_payback_data,
    names="loan_paid_back",
    title="Loan Repayment Ratio (0 vs 1)",
    color="loan_paid_back",
    color_discrete_sequence=["#008080", "#FF7F0E"]
)

fig.update_layout(template="plotly_white")
fig.show()



loan_payback_data


categorical_cols = loan_payback_data.select_dtypes(include=["object", "category"]).columns
categorical_cols



for col in categorical_cols:
    print(f"--- {col} ---")
    print(loan_payback_data[col].unique())
    print("\n")



for col in categorical_cols:
    print(f"--- {col} ---")
    print(loan_payback_data[col].value_counts())
    print("\n")



cat_summary = pd.DataFrame({
    "column": categorical_cols,
    "num_unique_values": [loan_payback_data[col].nunique() for col in categorical_cols]
})

cat_summary



import plotly.express as px

# 1. Get categorical columns
categorical_cols = loan_payback_data.select_dtypes(include=["object", "category"]).columns

for col in categorical_cols:
    # 2. Build summary dataframe: category + count
    counts = (
        loan_payback_data[col]
        .value_counts()
        .reset_index(name="count")
    )
    counts.columns = [col, "count"]

    # 3. Horizontal bar chart
    fig = px.bar(
        counts,
        x="count",       # count on x-axis
        y=col,           # categories on y-axis
        orientation="h", # horizontal bars
        title=f"Distribution of {col}",
        color=col,
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Count",
        yaxis_title=col,
        showlegend=False,
        plot_bgcolor="#f9f9f9",
        paper_bgcolor="#f2f2f2",
        margin=dict(l=60, r=40, t=60, b=40),
        font=dict(size=12)
    )

    fig.show()



import plotly.express as px

# Build summary: sub_grade + count
sub_counts = (
    loan_payback_data["grade_subgrade"]
    .value_counts()
    .reset_index(name="count")
)
sub_counts.columns = ["sub_grade", "count"]

# (Optional but recommended) Correctly sort sub_grade values
# e.g., A1, A2, A3 ... B1, B2, ...
sub_counts = sub_counts.sort_values(
    by="sub_grade",
    key=lambda x: x.str.extract(r'([A-Z])(\d)').apply(lambda r: (r[0], int(r[1])), axis=1)
)

# Vertical bar chart
fig = px.bar(
    sub_counts,
    x="sub_grade",
    y="count",
    title="Distribution of Sub Grades",
    color="sub_grade",
    color_discrete_sequence=px.colors.qualitative.Set2
)

fig.update_layout(
    template="plotly_white",
    xaxis_title="Sub Grade",
    yaxis_title="Count",
    showlegend=False,
    plot_bgcolor="#f9f9f9",
    paper_bgcolor="#f2f2f2",
    margin=dict(l=40, r=40, t=60, b=60),
    font=dict(size=12)
)

fig.show()



for col in categorical_cols:
    print(f"--- {col} ---")
    print(loan_payback_data[col].unique())
    print("\n")


X = loan_payback_data.drop("loan_paid_back", axis=1)
y = loan_payback_data["loan_paid_back"]



X


y


# Before training a model, we split our dataset into two parts:
# a training set (used to fit/learn the model) and a test set 
# (used only at the end to evaluate performance on unseen data).
#
# X = all the input features (everything except the target column)
# y = the target variable we want to predict (“loan_paid_back”).
#
# train_test_split() randomly divides X and y into four subsets:
#   - X_train: 80% of the feature data used to train the model
#   - X_test: 20% of the feature data used to evaluate the model
#   - y_train: target values corresponding to X_train
#   - y_test: target values corresponding to X_test
#
# test_size=0.2 means 20% of the data becomes the test set.
#
# stratify=y ensures that the proportion of classes (0 and 1 in 
# loan_paid_back) is preserved in both the training and test sets.
# This is very important for imbalanced datasets so the model 
# does not train or evaluate on distorted class ratios.
#
# random_state=42 makes the split reproducible — running this code again
# will generate the same split every time, which is essential for
# consistency and debugging.
#
# IMPORTANT: We do this split BEFORE performing any encoding, scaling,
# outlier removal, or transformations. This prevents data leakage,
# meaning the model cannot “peek” at information from the test set.
# All preprocessing steps must be fit ONLY on the training data and
# then applied to the test data later.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)



X_train


X_test


y_train


y_test


from sklearn.compose import make_column_selector

numeric_selector = make_column_selector(dtype_include=['int64','float64'])
categorical_selector = make_column_selector(dtype_include=['object','category'])

numeric_cols = numeric_selector(X)
categorical_cols = categorical_selector(X)

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)



# We import two preprocessing tools from scikit-learn:
# 1. OneHotEncoder – converts categorical features (strings) into numeric
#    binary columns (0/1). This is required because machine learning
#    models cannot use raw text categories directly.
# 2. StandardScaler – rescales numerical features so they have mean = 0
#    and standard deviation = 1. This helps many models train properly,
#    especially logistic regression, SVM, and neural networks.

# ColumnTransformer allows us to apply different preprocessing steps to
# different columns within a single unified transformer object.
# This is extremely useful because:
# - Numeric columns need scaling
# - Categorical columns need one-hot encoding
# - Other columns may remain unchanged or get different transforms
#
# ColumnTransformer handles all of that automatically and outputs one 
# combined transformed feature matrix (all numeric).

# Here we define the preprocessing configuration.
# The 'transformers' argument is a list of tuples, each describing:
# (name, transformer, columns_to_apply_it_to)
#
# 1. ("num", StandardScaler(), numeric_cols)
#    - Applies StandardScaler ONLY to the numeric_cols we defined earlier.
#    - These columns will be transformed into standardized values 
#      (z-scores), which helps optimize model performance.
#
# 2. ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
#    - Applies OneHotEncoder ONLY to columns listed in categorical_cols.
#    - drop="first" removes the first category in each column to prevent 
#      the “dummy variable trap” (perfect multicollinearity).
#      (Helpful for linear/logistic regression; optional for trees.)
#    - handle_unknown="ignore" prevents errors if the model sees a category
#      in the test set that was not present in training data. Instead, it
#      assigns all-zero encoding for that unseen category.
#
# After this step, the output contains:
# - Scaled numeric features
# - One-hot encoded categorical features
# - No text/categorical raw data
# - All features now numeric and model-ready


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
    ]
)



# The Pipeline class lets us link multiple processing steps together
# into a single, unified machine learning workflow.
#
# Instead of manually doing:
#       X_train_scaled = scaler.fit_transform(X_train)
#       X_train_encoded = encoder.fit_transform(X_train_scaled)
#       model.fit(X_train_encoded)
#
# we can combine ALL steps into one object. This has huge benefits:
#
# 1. Prevents data leakage – preprocessing is always fit on the
#    training set only, then automatically applied to the test set.
#
# 2. Guarantees correct order of operations – encoding → scaling → model.
#
# 3. Keeps code clean and production-ready – same structure used in
#    professional ML systems and Kaggle competition solutions.
#
# 4. Allows hyperparameter tuning with GridSearchCV or RandomizedSearchCV
#    across both preprocessing and model parameters.
#
# 5. Ensures reproducibility – the entire ML workflow is saved inside one object.
#

# Here we define the full model pipeline.
# The 'steps' argument is a list of (name, transformer_or_estimator) tuples.
#
# Step 1: ("preprocess", preprocess)
#   - 'preprocess' is the ColumnTransformer we created earlier.
#   - It handles:
#         * StandardScaler for numeric columns
#         * OneHotEncoder for categorical columns
#   - When we call model.fit(), this step FIRST fits all preprocessing
#     on X_train, and then transforms X_train.
#
# Step 2: ("clf", LogisticRegression(max_iter=1000))
#   - 'clf' stands for "classifier"
#   - LogisticRegression is the classification model we are training
#     to predict loan repayment.
#   - max_iter=1000 increases the number of allowed optimization iterations.
#     Logistic regression can fail to converge if max_iter is too low,
#     especially after one-hot encoding creates many features.
#     Setting max_iter=1000 ensures stable convergence.
#
# The pipeline executes in order:
#      X_train → preprocess → transformed X_train → LogisticRegression.fit()
# And at prediction time:
#      X_test → preprocess (using SAME fitted transformations) → predict()
#

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=1000))
])



model.fit(X_train, y_train)



from sklearn.metrics import classification_report, roc_auc_score

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))



import plotly.graph_objs as go
from sklearn.metrics import roc_curve, auc

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Create Plotly figure
fig = go.Figure()

# Add ROC curve
fig.add_trace(
    go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"ROC Curve (AUC = {roc_auc:.3f})",
        line=dict(width=3, color="#1f77b4")
    )
)

# Add diagonal baseline
fig.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Baseline",
        line=dict(width=2, dash="dash", color="gray")
    )
)

# Layout styling
fig.update_layout(
    title="ROC Curve – Logistic Regression",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    template="plotly_white",
    width=700,
    height=600,
    font=dict(size=14),
    plot_bgcolor="#f9f9f9",
    paper_bgcolor="#f2f2f2",
)

fig.show()



from sklearn.metrics import precision_recall_curve, average_precision_score
import plotly.graph_objs as go

precision, recall, thresholds = precision_recall_curve(y_test, y_prob, pos_label=1)
ap = average_precision_score(y_test, y_prob)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=recall,
        y=precision,
        mode="lines",
        name=f"PR Curve (AP = {ap:.3f})",
        line=dict(width=3, color="#008080")
    )
)

fig.update_layout(
    title="Precision–Recall Curve – Logistic Regression",
    xaxis_title="Recall",
    yaxis_title="Precision",
    template="plotly_white",
    width=700,
    height=600,
    font=dict(size=14),
    plot_bgcolor="#f9f9f9",
    paper_bgcolor="#f2f2f2",
)

fig.show()



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 5), dpi=300)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix – Logistic Regression")
plt.tight_layout()
plt.show()



X_train_transformed = model.named_steps["preprocess"].transform(X_train)

# If it's a sparse matrix, convert to dense NumPy array
if hasattr(X_train_transformed, "toarray"):
    X_train_transformed = X_train_transformed.toarray()



feature_names = model.named_steps["preprocess"].get_feature_names_out()
print(X_train_transformed.shape)   # should be (n_rows, n_features)
print(len(feature_names))          # should match n_features



import pandas as pd

X_train_df = pd.DataFrame(
    X_train_transformed,
    columns=feature_names,
    index=X_train.index
)

X_train_df



import joblib

# Save the entire pipeline to a file
joblib.dump(model, "loan_default_pipeline.joblib")



import joblib
import pandas as pd

# Load the saved pipeline
model = joblib.load("loan_default_pipeline.joblib")



# Load new data
test_df = pd.read_csv("/Users/ganeshjai/Documents/LoanPayback/.venv/data/playground-series-s5e11/test.csv")   # adjust path if needed

# If test.csv DOES NOT have the target column:
X_new = test_df

# If test.csv DOES have loan_paid_back and you only want to predict from features:
# X_new = test_df.drop("loan_paid_back", axis=1)

# Get predictions (0/1)
y_new_pred = model.predict(X_new)

# Get predicted probabilities for class 1 (repay)
y_new_prob = model.predict_proba(X_new)[:, 1]

# Attach to dataframe if you like
test_df["predicted_loan_paid_back"] = y_new_pred
test_df["prob_loan_paid_back"] = y_new_prob

# Save results
test_df.to_csv("test_with_predictions.csv", index=False)



import pandas as pd
import joblib

# Load model
model = joblib.load("loan_default_pipeline.joblib")

# Load test.csv
test_df = pd.read_csv("/Users/ganeshjai/Documents/LoanPayback/.venv/data/playground-series-s5e11/test.csv")

# Make predictions (0 or 1)
ensemble_pred = model.predict(test_df)

# (Optional) probabilities if needed
ensemble_prob = model.predict_proba(test_df)[:, 1]



sub = pd.DataFrame()
sub["id"] = test_df["id"]        # replace with correct ID column
sub["loan_paid_back"] = ensemble_pred



sub.to_csv("submission.csv", index=False)



sub


