import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor

from IPython.display import display
import warnings
warnings.filterwarnings("ignore")


df_train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
test_ids = df_test['id']

# Remove IDs
df_train = df_train.drop("id", axis=1)
df_test = df_test.drop("id", axis=1)


df_train


print(df_train.isnull().sum())


df_train.shape



df_train.info()



df_train.describe()



print(df_train.duplicated().sum())



df_train = df_train.drop_duplicates()



print(df_train.duplicated().sum())



def plot_and_percent_subplots(cat_cols, y, df, ncols=2):
    nrows = len(cat_cols) * 2 // ncols + (len(cat_cols)*2 % ncols > 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 3*nrows))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
    
        ax_plot = axes[i*2]
        sns.barplot(data=df, x=col, y=y, palette="tab10", ax=ax_plot)
        ax_plot.set_title(f"{col} vs {y}")
        ax_plot.tick_params(axis='x', rotation=45)


        ax_table = axes[i*2+1]
        ax_table.axis("off")
        table = (
            df.groupby(col)[y]
              .mean()
              .sort_values(ascending=False)
              .mul(100)
              .round(2)
              .astype(str) + "%"
        )
        tbl = ax_table.table(
            cellText=[[val] for val in table.values],
            rowLabels=table.index,
            colLabels=[y],
            cellLoc="center",
            loc="center"
        )
        tbl.scale(1, 1.3)

    for j in range(len(cat_cols)*2, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# Plort categorical columns

cat_cols = [col for col in df_train.select_dtypes(exclude=np.number).columns if df_train[col].nunique() > 2]

plot_and_percent_subplots(cat_cols, "accident_risk", df_train)


def plot_distributions(df, num_cols=None, ncols=3):
    
    if num_cols is None:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

    nrows = int(np.ceil(len(num_cols)/ncols))
    plt.figure(figsize=(5*ncols, 3*nrows))

    for i, col in enumerate(num_cols, 1):
        plt.subplot(nrows, ncols, i)
        sns.histplot(data=df, x=col, kde=True, color="skyblue")
        plt.title(f"Distribution of {col}")

    plt.tight_layout()
    plt.show()


# Plot distributions

plot_distributions(df_train)


def create_features(df):
    
    df_processed = df.copy()

    # Convert Bool to Int
    bol_cols = df_processed.select_dtypes(include=bool).columns
    df_processed[bol_cols] = df_processed[bol_cols].astype(int)

    df_processed['curvature_speed'] = df_processed['curvature'] * df_processed['speed_limit']
    df_processed['lighting_weather_risk'] = ((df_processed['lighting'].isin(['dim', 'night'])) & (df_processed['weather'].isin(['foggy', 'rainy']))).astype(int)
    df_processed['speed_curvature_high_risk'] = ((df_processed['speed_limit'] > 50) & (df_processed['curvature'] > 0.5)).astype(int)
    df_processed['lighting_speed_interaction'] = df_processed['speed_limit'] * df_processed['lighting'].map({'daylight': 1, 'dim': 2, 'night': 3})
    df_processed['weather_curvature_risk'] = df_processed['curvature'] * df_processed['weather'].map({'clear': 1, 'rainy': 2, 'foggy': 3})
    df_processed['visibility_risk'] = (df_processed['lighting'].map({'daylight': 0.1, 'dim': 0.4, 'night': 0.7}) + df_processed['weather'].map({'clear': 0.1, 'rainy': 0.3, 'foggy': 0.5}))
    df_processed['high_speed_complex_road'] = ((df_processed['speed_limit'] > 60) & (df_processed['curvature'] > 0.3)).astype(int)
    df_processed['extreme_weather_curvature'] = ((df_processed['weather_curvature_risk'] > 1.5) & (df_processed['speed_curvature_high_risk'] == 1)).astype(int)
    df_processed['night_speed_curvature'] = (df_processed['lighting'] == 'night').astype(int) * df_processed['speed_limit'] * df_processed['curvature']
    df_processed['night_rainy_combo'] = ((df_processed['lighting'] == 'night') & (df_processed['weather'] == 'rainy')).astype(int)

    return df_processed


df_train = create_features(df_train)
df_test = create_features(df_test)


df_train.sample(3)



x = df_train.drop(["accident_risk"] , axis=1)
y = df_train["accident_risk"]


from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

cat_cols = x.select_dtypes(exclude=np.number).columns
num_cols = x.select_dtypes(include=np.number).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(feature_range=(0, 1)), num_cols),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=int, drop="first"), cat_cols)
    ]
)

x_preprocessed = preprocessor.fit_transform(x)
x_test_preprocessed = preprocessor.transform(df_test)


def cross_validate_model(model, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = np.sqrt(mean_squared_error(y_val, preds))
        scores.append(score)
    
    return np.mean(scores), np.std(scores)


xgb_params = {
    'max_depth': 9,
    'learning_rate': 0.0115340465557138,
    'n_estimators': 987,
    'subsample': 0.7700123220274983,
    'colsample_bytree': 0.6330336975986194,
    'reg_alpha': 0.837550570295205,
    'reg_lambda': 0.33966361687283775,
    'min_child_weight': 1,
    "random_state": 42,
    "tree_method": "hist",
    "device": "cuda",
    "n_jobs": -1
}

cat_params = {
    'depth': 7,
    'learning_rate': 0.046343808617582485,
    'iterations': 1704,
    'l2_leaf_reg': 12.953008902156173,
    'random_seed': 42,
    'task_type': 'GPU',
    "verbose": 0
}

xgb_model = XGBRegressor(**xgb_params)
cat_model = CatBoostRegressor(**cat_params)

final_model = VotingRegressor([
    ('xgb', xgb_model),
    ('cat', cat_model)
], 
    weights=[0.7, 0.3],
    n_jobs=-1
)

ensemble_score, ensemble_std = cross_validate_model(final_model, x_preprocessed, y)
print(f"Ensemble CV RMSE: {ensemble_score:.6f} Â± {ensemble_std:.6f}")
print(f"ðŸ“Š Estimated Kaggle Score: ~{ensemble_score:.5f}")


final_model.fit(x_preprocessed, y)
final_preds = final_model.predict(x_test_preprocessed)


submission = pd.DataFrame({
    "id": test_ids,
    "accident_risk": final_preds
})

submission.to_csv("submission.csv", index=False)


submission.sample(3)

