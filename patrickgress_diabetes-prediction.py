from pandas import DataFrame, read_csv, CategoricalDtype
from numpy import maximum, minimum


LABELED: DataFrame = read_csv("/kaggle/input/playground-series-s5e12/train.csv")


DataFrame(LABELED.isnull().sum()).transpose()


LABELED.columns


LABELED.dtypes[LABELED.dtypes == object]


LABELED.gender = LABELED.gender.astype("category")
LABELED.ethnicity = LABELED.ethnicity.astype("category")
LABELED.employment_status = LABELED.employment_status.astype("category")


LABELED.income_level.unique()


income_level_type = CategoricalDtype(["Low", "Lower-Middle", "Upper-Middle", "Middle" "High"], ordered=True)
LABELED.income_level = LABELED.income_level.astype(income_level_type)



LABELED.smoking_status.unique()


smoking_status_type = CategoricalDtype(["Never", "Former", "Current"], ordered=True)
LABELED.smoking_status = LABELED.smoking_status.astype(smoking_status_type)


LABELED.education_level.unique()


education_level_type = CategoricalDtype(["No formal", "Highschool", "Graduate", "Postgraduate"], ordered=True)
LABELED.education_level = LABELED.education_level.astype(education_level_type)


LABELED["_systolic_bp"] = maximum(LABELED.systolic_bp, LABELED.diastolic_bp)
LABELED["_diastolic_bp"] = minimum(LABELED.systolic_bp, LABELED.diastolic_bp)



LABELED["_pulse_pressure"] = LABELED._systolic_bp - LABELED._diastolic_bp


(LABELED._pulse_pressure < 5.0).sum(), (LABELED._pulse_pressure > 100.0).sum()


DataFrame([LABELED._diastolic_bp.describe(), LABELED._systolic_bp.describe(), LABELED._pulse_pressure.describe()])


LABELED.triglycerides.describe()





DataFrame([LABELED.age.describe()])


LABELED.physical_activity_minutes_per_week.describe()


LABELED.sleep_hours_per_day.describe()


LABELED.diet_score.describe()


LABELED[["diet_score", "bmi"]].corr()


LABELED.bmi.describe()


LABELED.waist_to_hip_ratio.describe()


LABELED.screen_time_hours_per_day.describe()


from numpy import log
from seaborn import heatmap
from plotnine import ggplot, aes, geom_point, geom_histogram, facet_wrap, after_stat, geom_vline
from matplotlib.pyplot import savefig, figure
from collections.abc import Callable




def histo_summary(bins: int = 10):
    return ggplot(LABELED.select_dtypes(include="number").melt(["diagnosed_diabetes"]), aes(x="value", y=after_stat("density"), fill="factor(diagnosed_diabetes)"))\
    + geom_histogram(bins=bins, position="identity", alpha=0.4) + facet_wrap("variable", scales="free")


histo_summary().save("histograms.png", dpi=100, width=420, height=297, units="mm", limitsize=False)


ENG_STEPS: list[Callable[[DataFrame], None]] = []

def eng_step(func: Callable[[DataFrame], None]) -> Callable[[DataFrame], None]:
    if func not in ENG_STEPS:
        ENG_STEPS.append(func)
    return func


@eng_step
def pulse_pressure(df):
    df["_systolic_bp"] = maximum(df.systolic_bp, df.diastolic_bp)
    df["_diastolic_bp"] = minimum(df.systolic_bp, df.diastolic_bp)
    df["_pulse_pressure"] = df._systolic_bp - df._diastolic_bp


@eng_step
def scale_triglycerides(df):
    df["_log_triglycerides"] = log(df["triglycerides"])


@eng_step
def ldl_hdl_ratio(df):
    df["_ldl_hdl_ratio"] = df.ldl_cholesterol / df.hdl_cholesterol



@eng_step
def health_score(df):
    df["_health_score"] =  df.sleep_hours_per_day * df.diet_score / (df.heart_rate * df.waist_to_hip_ratio)


@eng_step
def triglycerides_risk(df):
    df["_elevated_triglycerides"] = maximum(df.triglycerides - 150, 0)


def _hdl_excess(row):
    return 40 if row["gender"] == "Female" else 50

@eng_step
def hdl_risk(df):
    df["_elevated_hdl"] = df.apply(lambda r: max(0, r["hdl_cholesterol"] - _hdl_excess(r)), axis=1)

@eng_step
def ldl_risk(df):
    df["_elevated_ldl"] = maximum(df.ldl_cholesterol - 100, 0)


figure(figsize=(20, 20)) 
heatmap(LABELED.select_dtypes(include="number").corr(), cmap="coolwarm", annot=True)
savefig("correlation.png", dpi=300)


def create_vs_plot(x: str, y: str):
    return ggplot(LABELED, aes(x=x, y=y, fill="diagnosed_diabetes")) + geom_point()

def create_histogram(x: str, bins: int = 10):
    df = LABELED[[x, "diagnosed_diabetes"]]
    return ggplot(df, aes(x=x, y=after_stat("density"), fill="factor(diagnosed_diabetes)"))\
    + geom_histogram(bins=bins, position="identity", alpha=0.4)


@eng_step
def obesity_level(df):
    df["_obesity_level"] = maximum(df.bmi - 25, 0)



LABELED.columns


@eng_step
def activity_score(df):
    df["_activity_score"] = (df.physical_activity_minutes_per_week / 7.0 - df.screen_time_hours_per_day + df.sleep_hours_per_day) / (df.waist_to_hip_ratio * df.diastolic_bp)


@eng_step
def tri_hdl_ratio(df):
    df["_triglicerides_hdl_ratio"] = df.triglycerides / df.hdl_cholesterol


from sklearn.model_selection import train_test_split


def apply_pipeline(df):
    for step in ENG_STEPS:
        step(df)
    return df


LABELED = apply_pipeline(LABELED)


histo_summary().save("histograms_en.png", dpi=100, width=420, height=297, units="mm", limitsize=False)


TRAIN, TEST = train_test_split(apply_pipeline(LABELED), stratify=LABELED.diagnosed_diabetes)


from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier, plot_tree, plot_importance, early_stopping, log_evaluation
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform, loguniform
from shap import TreeExplainer, summary_plot



SELECTED_FEATURES: list[str] = [
    "age", 
    "family_history_diabetes", 
    "physical_activity_minutes_per_week", 
    "_obesity_level",
    "_ldl_hdl_ratio",
    "_elevated_triglycerides",
    "_health_score",
    "_activity_score",
    "_triglicerides_hdl_ratio",
    "_elevated_ldl",
    "bmi"
]


LABELED.columns


model = LGBMClassifier(
    n_estimators=400, 
    learning_rate=0.05, 
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    min_child_weight=1e-3,
    subsample=0.8,  
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=42
)
model.fit(
    TRAIN[SELECTED_FEATURES], 
    TRAIN.diagnosed_diabetes, 
    eval_set=[(TEST[SELECTED_FEATURES], TEST.diagnosed_diabetes)], 
    eval_metric="auc", 
)


model.score(TEST[SELECTED_FEATURES], TEST.diagnosed_diabetes)


model.score(TRAIN[SELECTED_FEATURES], TRAIN.diagnosed_diabetes)


plot_importance(model, importance_type="gain")


explainer = TreeExplainer(model)
shap_values = explainer.shap_values(TEST[SELECTED_FEATURES])
plot = summary_plot(shap_values[1], TEST[SELECTED_FEATURES], plot_type="dot")
savefig("SHAP.png", dpi=300)
plot


PREDICT = apply_pipeline(read_csv("/kaggle/input/playground-series-s5e12/test.csv"))
prediction = model.predict(PREDICT[SELECTED_FEATURES])
PREDICT["diagnosed_diabetes"] = prediction.astype(int)


PREDICT[["id", "diagnosed_diabetes"]].to_csv("submission.csv", index=False)

