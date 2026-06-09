# Importing Dependencies
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency, skew, mannwhitneyu
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, accuracy_score

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.subplots as sp
import warnings

warnings.filterwarnings('ignore')
from plotly.offline import init_notebook_mode

# Initialize notebook mode
init_notebook_mode(connected=True)

# Use a persistent renderer
pio.renderers.default = 'iframe'  # Embeds plot directly in output

# Use a persistent renderer
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Loading Data
df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


def plot_numeric_binned(df, col, x_title):
    # Create quantile bins
    temp_bins = pd.qcut(df[col], q=8, duplicates="drop")

    binned_df = (
        df.groupby(temp_bins)
          .agg(
              Diabetes_Rate=("diagnosed_diabetes", "mean"),
              Count=("diagnosed_diabetes", "size"),
              Min_Val=(col, "min"),
              Max_Val=(col, "max")
          )
          .reset_index(drop=True)
    )

    # Clean bin labels like "0â€“30"
    binned_df["Range_Label"] = (
        binned_df["Min_Val"].round(0).astype(int).astype(str)
        + "â€“" +
        binned_df["Max_Val"].round(0).astype(int).astype(str)
    )

    # Sort by min value to keep x ordered
    binned_df = binned_df.sort_values("Min_Val")

    fig = go.Figure()

    # Bar: number of people (navy)
    fig.add_trace(go.Bar(
        x=binned_df["Range_Label"],
        y=binned_df["Count"],
        name="Number of people",
        marker_color="#002455",
        opacity=0.8,
        yaxis="y",
        hovertemplate=x_title + ": %{x}<br>People: %{y}<extra></extra>"
    ))

    # Line: diabetes rate (red)
    fig.add_trace(go.Scatter(
        x=binned_df["Range_Label"],
        y=binned_df["Diabetes_Rate"],
        name="Diabetes rate",
        mode="lines+markers",
        line=dict(color="#DC0000", width=3),
        marker=dict(size=8, color="#DC0000"),
        yaxis="y2",
        hovertemplate=x_title + ": %{x}<br>Diabetes rate: %{y:.1%}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text=x_title + " vs Diabetes Rate",
            x=0.5,
            xanchor="center",
            font=dict(size=22)
        ),
        xaxis=dict(
            title=x_title + " (binned)",
            tickangle=0,
            showgrid=False
        ),
        yaxis=dict(
            title="Number of people",
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.3)"
        ),
        yaxis2=dict(
            title="Diabetes rate",
            overlaying="y",
            side="right",
            tickformat=".0%",
            showgrid=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=550,
        bargap=0.15,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    fig.show(renderer="iframe_connected")


def print_numeric_summary(df, cols, target="diagnosed_diabetes"):
    """Print mean values by diabetes status for numeric columns"""
    
    print("\n" + "="*80)
    print("  NUMERIC METRICS BY DIABETES STATUS")
    print("="*80)
    print(f"{'Metric':<40} {'No Diabetes':<20} {'Diagnosed':<20}")
    print("-"*80)
    
    for col in cols:
        no_diab = df[df[target] == 0.0][col].mean()
        diab = df[df[target] == 1.0][col].mean()
        print(f"{col:<40} {no_diab:<20.2f} {diab:<20.2f}")
    
    print("="*80 + "\n")


def print_category_rates(df, col, target="diagnosed_diabetes"):
    """Print diabetes rates by categorical column"""
    
    rates_df = (
        df.groupby(col)[target]
          .mean()
          .reset_index()
    )
    rates_df.columns = [col, "Diabetes_Rate"]
    rates_df = rates_df.sort_values("Diabetes_Rate", ascending=False)
    rates_df["Rate_Percent"] = (rates_df["Diabetes_Rate"] * 100).round(2).astype(str) + "%"
    
    print("\n" + "="*70)
    print(f"  DIABETES RATE BY {col.upper().replace('_', ' ')}")
    print("="*70)
    print(f"{'Category':<35} {'Rate':<15} {'Percentage':<15}")
    print("-"*70)
    for _, row in rates_df.iterrows():
        print(f"{str(row[col]):<35} {row['Diabetes_Rate']:<15.4f} {row['Rate_Percent']:<15}")
    print("="*70 + "\n")


plot_numeric_binned(df, "physical_activity_minutes_per_week",
                     "Weekly physical activity (minutes)")


plot_numeric_binned(df, "screen_time_hours_per_day",
                     "Screen time (hours/day)")


print_category_rates(df, "smoking_status")
print_category_rates(df,"employment_status")


print_numeric_summary(df, [
    "sleep_hours_per_day",
    "alcohol_consumption_per_week",
    "physical_activity_minutes_per_week",
    "screen_time_hours_per_day"
])


def plot_binary_category_rates(df, col, target="diagnosed_diabetes"):
    """Horizontal bar plot for binary categorical columns (Yes/No, True/False)"""
    
    # Aggregate rates
    rates_df = (
        df.groupby(col)[target]
          .mean()
          .reset_index()
    )
    rates_df.columns = [col, "Diabetes_Rate"]
    rates_df = rates_df.sort_values("Diabetes_Rate", ascending=True)
    
    # Map values to labels
    label_map = {0: "No", 1: "No", "No": "No", "Yes": "Yes", False: "No", True: "Yes"}
    rates_df["Label"] = rates_df[col].map(label_map)
    
    # Colors: blue for No, red for Yes/Diagnosed
    color_map = {0: "#001F3D", 1: "#DC0000", "No": "#001F3D", "Yes": "#DC0000", False: "#001F3D", True: "#DC0000"}
    bar_colors = rates_df[col].map(color_map)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=rates_df["Label"],
        x=rates_df["Diabetes_Rate"],
        orientation="h",
        marker=dict(color=bar_colors),
        hovertemplate="%{y}<br>Diabetes rate: %{x:.2%}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(
            text="Diabetes Rate by " + col.replace("_", " ").title(),
            x=0.5,
            xanchor="center",
            font=dict(size=20)
        ),
        xaxis=dict(
            title="Diabetes Rate",
            tickformat=".0%",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)"
        ),
        yaxis=dict(
            title=col.replace("_", " ").title()
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400
    )
    
    fig.show()


plot_binary_category_rates(df, "family_history_diabetes")


plot_binary_category_rates(df, "cardiovascular_history")


plot_binary_category_rates(df, "hypertension_history")


print_numeric_summary(df, [
    "bmi",
    "cholesterol_total",
    "diastolic_bp",
    "diet_score",
    "hdl_cholesterol",
    "heart_rate",
    "ldl_cholesterol",
    "systolic_bp",
    "triglycerides",
    "waist_to_hip_ratio"
])


plot_numeric_binned(df, "age",
                     "Age (years)")


def plot_demographic_counts(df, col):
    """Horizontal bar plot for demographic categorical columns - count distribution"""
    
    # Count distribution
    counts_df = (
        df[col]
          .value_counts()
          .reset_index()
    )
    counts_df.columns = [col, "Count"]
    counts_df = counts_df.sort_values("Count", ascending=True)
    
    # Custom color palette
    colors = ["#050E3C", "#DC0000","#002455", "#FF3838"]
    bar_colors = [colors[i % len(colors)] for i in range(len(counts_df))]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=counts_df[col],
        x=counts_df["Count"],
        orientation="h",
        marker=dict(color=bar_colors),
        hovertemplate="%{y}<br>Count: %{x}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(
            text="Distribution by " + col.replace("_", " ").title(),
            x=0.5,
            xanchor="center",
            font=dict(size=20)
        ),
        xaxis=dict(
            title="Count",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)"
        ),
        yaxis=dict(
            title=col.replace("_", " ").title()
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=450
    )
    
    fig.show()


plot_demographic_counts(df, "gender")


plot_demographic_counts(df, "ethnicity")


plot_demographic_counts(df, "education_level")


plot_demographic_counts(df, "income_level")


print_category_rates(df, "gender")
print_category_rates(df, "education_level")
print_category_rates(df, "ethnicity")
print_category_rates(df, "income_level")


cat_cols = df.select_dtypes(include='object').columns
extra_cat = ['cardiovascular_history', 'hypertension_history', 'family_history_diabetes']
cat_cols = list(set(cat_cols).union(extra_cat))

results = []

for col in cat_cols:
    ct = pd.crosstab(df[col], df["diagnosed_diabetes"])
    chi2, p, _, _ = chi2_contingency(ct)

    results.append({
        "feature": col,
        "chi2": chi2,
        "p_value": p
    })

chi_results = (
    pd.DataFrame(results)
      .sort_values("p_value")
      .reset_index(drop=True)
)

print("\n" + "="*80)
print("  CATEGORICAL FEATURES â€” CHI-SQUARE TEST")
print("="*80)
print(f"{'Feature':<30} {'Chi2':>12} {'p-value':>12} {'Significant?':>14}")
print("-"*80)

for _, row in chi_results.iterrows():
    signif = "Yes (pâ‰¤0.05)" if row["p_value"] <= 0.05 else "No"
    print(f"{row['feature']:<30} "
          f"{row['chi2']:>12.3f} "
          f"{row['p_value']:>12.4g} "
          f"{signif:>14}")

print("="*80 + "\n")


num_cols = (
    df.select_dtypes(include="number")
      .columns.difference(cat_cols + ["diagnosed_diabetes", "id"])
      .tolist()
)

def numeric_feature_tests(df, num_cols, target="diagnosed_diabetes"):
    results = []

    for col in num_cols:
        g0 = df[df[target] == 0.0][col].dropna()
        g1 = df[df[target] == 1.0][col].dropna()

        # Mannâ€“Whitney U test (non-parametric alternative to t-test)
        stat, p = mannwhitneyu(g0, g1, alternative="two-sided")

        results.append({
            "feature": col,
            "u_stat": stat,
            "p_value": p,
            "mean_no_diab": g0.mean(),
            "mean_diab": g1.mean()
        })

    res_df = (
        pd.DataFrame(results)
        .sort_values("p_value")
        .reset_index(drop=True)
    )

    print("\n" + "="*90)
    print("  NUMERIC FEATURES â€” MANNâ€“WHITNEY U TEST (Non-parametric)")
    print("="*90)
    print(f"{'Feature':<30} {'p-value':>12} {'Mean 0':>12} {'Mean 1':>12} {'Significant?':>14}")
    print("-"*90)

    for _, row in res_df.iterrows():
        signif = "Yes (pâ‰¤0.05)" if row["p_value"] <= 0.05 else "No"
        print(f"{row['feature']:<30} "
              f"{row['p_value']:>12.4g} "
              f"{row['mean_no_diab']:>12.2f} "
              f"{row['mean_diab']:>12.2f} "
              f"{signif:>14}")

    print("="*90 + "\n")
    return res_df

mw_results = numeric_feature_tests(df, num_cols)


df = df.drop(columns=["id", "smoking_status"])


vif_data = df[num_cols].copy()


# Add intercept for VIF computation
X = add_constant(vif_data)

vif_list = []
for i in range(1, X.shape[1]):  # skip intercept at index 0
    vif_list.append({
        "feature": X.columns[i],
        "VIF": variance_inflation_factor(X.values, i)
    })

vif_df = (
    pd.DataFrame(vif_list)
      .sort_values("VIF", ascending=False)
      .reset_index(drop=True)
)

# Threshold for "high" multicollinearity (common: 5 or 10)
threshold = 5.0
high_vif = vif_df[vif_df["VIF"] > threshold]

print("\n" + "="*70)
print("  HIGH MULTICOLLINEARITY FEATURES (VIF > {:.1f})".format(threshold))
print("="*70)
if high_vif.empty:
    print("No features with VIF above threshold.")
else:
    print(f"{'Feature':<30} {'VIF':>10}")
    print("-"*70)
    for _, row in high_vif.iterrows():
        print(f"{row['feature']:<30} {row['VIF']:>10.2f}")
print("="*70 + "\n")


# -----------------------------
# Ordinal columns with explicit order
# -----------------------------
ordinal_cols = {
    'income_level': ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High'],
    'education_level': ['No formal', 'Highschool','Graduate','Postgraduate']
}

for col, order in ordinal_cols.items():
    ord_enc = OrdinalEncoder(categories=[order])
    df[col] = ord_enc.fit_transform(df[[col]])

# -----------------------------
# Non-ordinal categorical columns â†’ Label Encoding
# -----------------------------
non_ordinal_cols = ['ethnicity', 'employment_status', 'gender']

for col in non_ordinal_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])


TARGET = "diagnosed_diabetes"
FEATURES = [col for col in df.columns if col != TARGET]


X = df[FEATURES]
y = df[TARGET]


best_params = {
    "learning_rate": 0.14141008604431932,
    "max_depth": 4,
    "min_child_weight": 6,
    "subsample": 0.6043858586325254,
    "colsample_bytree": 0.9076381150343775,
    "gamma": 2.9365900809348835,
    "reg_lambda": 7.8511309801919715,
    "reg_alpha": 4.365923016193024,
}


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_oof = np.zeros(len(X))
auc_scores = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    model = XGBClassifier(
        n_estimators=500,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="gpu_hist",
        random_state=42,
        **best_params
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
    
    y_pred_proba = model.predict_proba(X_valid)[:, 1]
    xgb_oof[valid_idx] = y_pred_proba
    
    auc = roc_auc_score(y_valid, y_pred_proba)
    auc_scores.append(auc)
    print(f"Fold {fold} ROC-AUC: {auc:.4f}")

print("\nMean ROC-AUC: {:.4f} Â± {:.4f}".format(
    np.mean(auc_scores), np.std(auc_scores)
))

# Global OOF metrics for plotting
xgb_cv = roc_auc_score(y, xgb_oof)
y_pred_xgb = (xgb_oof > 0.5).astype(int)
cm_xgb = confusion_matrix(y, y_pred_xgb)
fpr_xgb, tpr_xgb, _ = roc_curve(y, xgb_oof)


colors = ["#001F3D", "#888888", "#DC0000", "#FFFFFF"]

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Confusion Matrix", "ROC Curve"),
    specs=[[{"type": "heatmap"}, {"type": "scatter"}]]
)

# Confusion matrix heatmap
fig.add_trace(
    go.Heatmap(
        z=cm_xgb,
        x=["Pred 0", "Pred 1"],
        y=["Actual 0", "Actual 1"],
        colorscale=[[0, colors[0]], [1, colors[2]]],
        text=cm_xgb,
        texttemplate="%{text}",
        textfont=dict(size=14, color=colors[3]),
        showscale=False
    ),
    row=1,
    col=1
)

# ROC curve
fig.add_trace(
    go.Scatter(
        x=fpr_xgb,
        y=tpr_xgb,
        mode="lines",
        name=f"XGBoost (AUC={xgb_cv:.4f})",
        line=dict(color=colors[2], width=3)
    ),
    row=1,
    col=2
)

# Random baseline
fig.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(color=colors[1], width=2, dash="dash")
    ),
    row=1,
    col=2
)

fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)
fig.update_xaxes(title_text="Predicted label", row=1, col=1)
fig.update_yaxes(title_text="True label", row=1, col=1)

fig.update_layout(
    title=f"XGBoost | OOF AUC: {xgb_cv:.6f} | Acc: {accuracy_score(y, y_pred_xgb):.4f}",
    height=500,
    plot_bgcolor="#f6f5f5",
    paper_bgcolor="#f6f5f5",
    legend=dict(orientation="h", x=0.55, y=0.02)
)

fig.show()


test.drop('smoking_status', axis=1, inplace=True)


# -----------------------------
# Ordinal columns with explicit order
# -----------------------------
ordinal_cols = {
    'income_level': ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High'],
    'education_level': ['No formal', 'Highschool','Graduate','Postgraduate']
}

for col, order in ordinal_cols.items():
    ord_enc = OrdinalEncoder(categories=[order])
    test[col] = ord_enc.fit_transform(test[[col]])

# -----------------------------
# Non-ordinal categorical columns â†’ Label Encoding
# -----------------------------
non_ordinal_cols = ['ethnicity', 'employment_status', 'gender']

for col in non_ordinal_cols:
    le = LabelEncoder()
    test[col] = le.fit_transform(test[col])


# Features in same order as training
X_test = test[FEATURES]


# -----------------------------
# Predict probabilities on test set
# -----------------------------
test_pred_proba = model.predict_proba(X_test)[:, 1]

# -----------------------------
# Create submission file
# -----------------------------
submission = pd.DataFrame({
    "id": test["id"],                    # keep id as in test.csv
    "diagnosed_diabetes": test_pred_proba  # probability for positive class
})

submission.to_csv("submission.csv", index=False)

