!pip install -U kaleido


import matplotlib.pyplot as plt
import seaborn as sns

import category_encoders as ce
import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
from catboost import CatBoostRegressor
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


pio.renderers.default = 'png'


df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df.head()


class PodcastEDA:
    def __init__(self, df: pd.DataFrame, fig_width: int = 800, fig_height: int = 600):
        self.df = df.copy()
        self.df.rename(columns={
            'Episode_Length_minutes': 'Episode_Length',
            'Host_Popularity_percentage': 'Host_Popularity',
            'Guest_Popularity_percentage': 'Guest_Popularity',
            'Listening_Time_minutes': 'Listening_Time'
        }, inplace=True)

        self.color_sequence = [
            '#415A77',
            '#E0E1DD'
        ]

        self.fig_width = fig_width
        self.fig_height = fig_height

    def summary_statistics(self):
        stats = self.df.describe().T
        print("Summary Statistics:\n", stats)
        print("\nNan values", self.df.isna().sum())

    def plot_numeric_distribution(self, nbins: int = 20):
        numeric_cols = [
            'Episode_Length',
            'Host_Popularity',
            'Guest_Popularity',
            'Number_of_Ads',
            'Listening_Time'
        ]
        for col in numeric_cols:
            if col not in self.df.columns:
                print(f"Warning: column '{col}' not found in dataframe.")
                continue
    
            # Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ¿ÑƒÑ�Ñ‚ÑƒÑ� Ñ„Ğ¸Ğ³ÑƒÑ€Ñƒ Ñ� 2 Ñ�Ñ‚Ñ€Ğ¾ĞºĞ°Ğ¼Ğ¸: Ğ³Ğ¸Ñ�Ñ‚Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¼Ğ° Ğ¸ boxplot
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(f'Distribution of {col}', f'Boxplot of {col}')
            )
    
            # Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ³Ğ¸Ñ�Ñ‚Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¼Ñƒ Ñ‡ĞµÑ€ĞµĞ· px
            fig_hist = px.histogram(
                self.df,
                x=col,
                nbins=nbins,
                color_discrete_sequence=self.color_sequence
            )
    
            # Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ boxplot Ñ‡ĞµÑ€ĞµĞ· px
            fig_box = px.box(
                self.df,
                y=col,
                color_discrete_sequence=self.color_sequence
            )
    
            # Ğ”Ğ¾Ğ±Ğ°Ğ²Ğ»Ñ�ĞµĞ¼ Ñ‚Ñ€Ğ°Ñ�Ñ‹ Ğ¸Ğ· fig_hist Ğ² Ğ¿ĞµÑ€Ğ²ÑƒÑ� Ñ�Ñ‚Ñ€Ğ¾ĞºÑƒ
            for trace in fig_hist.data:
                fig.add_trace(trace, row=1, col=1)
    
            # Ğ”Ğ¾Ğ±Ğ°Ğ²Ğ»Ñ�ĞµĞ¼ Ñ‚Ñ€Ğ°Ñ�Ñ‹ Ğ¸Ğ· fig_box Ğ²Ğ¾ Ğ²Ñ‚Ğ¾Ñ€ÑƒÑ� Ñ�Ñ‚Ñ€Ğ¾ĞºÑƒ
            for trace in fig_box.data:
                fig.add_trace(trace, row=2, col=1)
    
            # Ğ�Ğ±Ğ½Ğ¾Ğ²Ğ»Ñ�ĞµĞ¼ Ñ€Ğ°Ğ·Ğ¼ĞµÑ€Ñ‹ Ğ¸ Ğ¾Ñ„Ğ¾Ñ€Ğ¼Ğ»ĞµĞ½Ğ¸Ğµ
            fig.update_layout(
                height=self.fig_height,
                width=self.fig_width,
                showlegend=False,
                title_text=f'Distribution and Boxplot of {col}',
            )
    
            fig.show()

    def plot_categorical_counts(self):
        cat_cols = [
            'Podcast_Name',
            'Genre',
            'Publication_Day',
            'Publication_Time',
            'Episode_Sentiment'
        ]
        for col in cat_cols:
            if col not in self.df.columns:
                print(f"Warning: column '{col}' not found in dataframe.")
                continue

            fig = px.histogram(
                self.df,
                x=col,
                title=f'Count of {col}',
                color_discrete_sequence=self.color_sequence,
                width=self.fig_width,
                height=self.fig_height
            )
            fig.update_layout(xaxis={'categoryorder': 'total descending'})
            fig.show()

    def plot_correlation_heatmap(self):
        numeric_df = self.df.select_dtypes(include=['int64', 'float64'])
        if numeric_df.shape[1] < 2:
            print("Not enough numeric columns to compute correlation.")
            return

        corr = numeric_df.corr().round(2)
        heatmap = ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            colorscale=self.color_sequence,
            showscale=True
        )
        heatmap.update_layout(
            title='Correlation Heatmap',
            width=self.fig_width,
            height=self.fig_height
        )
        heatmap.show()

    def run_all(self, nbins: int = 20):
        self.summary_statistics()
        self.plot_numeric_distribution(nbins=nbins)
        self.plot_categorical_counts()
        self.plot_correlation_heatmap()

eda = PodcastEDA(df)
eda.run_all(nbins=30)


class PodcastPreprocessor:
    def __init__(self):
        self.ordinal_mappings = {
            'Episode_Sentiment': ['Negative', 'Neutral', 'Positive'],
            'Publication_Time': ['Night', 'Morning', 'Afternoon', 'Evening'],
            'Publication_Day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        }
        self.ordinal_encoders = {}
        self.target_encoders = {}
        self.fitted = False

    def fill_guest_popularity_with_gamma(self, df: pd.DataFrame) -> pd.DataFrame:
        df_filled = df.copy()
        
        if 'Guest_Popularity_percentage' not in df_filled.columns:
            print("Warning: 'Guest_Popularity_percentage' column not found.")
            return df_filled
        
        nan_mask = df_filled['Guest_Popularity_percentage'].isna()
        n_missing = nan_mask.sum()
        
        if n_missing == 0:
            return df_filled
    
        shape = 1.0    # alpha (Ñ‡ĞµĞ¼ Ğ¼ĞµĞ½ÑŒÑˆĞµ, Ñ‚ĞµĞ¼ Ñ‚Ñ�Ğ¶ĞµĞ»ĞµĞµ Ñ…Ğ²Ğ¾Ñ�Ñ‚)
        scale = 3.0
        
        gamma_samples = np.random.gamma(shape=shape, scale=scale, size=n_missing)
    
        gamma_samples = np.clip(gamma_samples, 0, 13)
        
        np.random.shuffle(gamma_samples)
        
        df_filled.loc[nan_mask, 'Guest_Popularity_percentage'] = gamma_samples
    
        return df_filled
    
    def remove_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()

    def remove_outliers_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df

    def fit(self, df: pd.DataFrame, target_col: str):
        df = df.copy()
        for col, categories in self.ordinal_mappings.items():
            if col in df.columns:
                oe = OrdinalEncoder(categories=[categories])
                oe.fit(df[[col]])
                self.ordinal_encoders[col] = oe

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        non_ordinal_cols = [col for col in categorical_cols if col not in self.ordinal_mappings]
        
        for col in non_ordinal_cols:
            te = ce.TargetEncoder(cols=[col])
            te.fit(df[col], df[target_col])
            self.target_encoders[col] = te
        
        self.fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise Exception("You must call 'fit' before 'transform'.")

        df = df.copy()
        df = self.fill_guest_popularity_with_gamma(df)
        for col, encoder in self.ordinal_encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[[col]])

        for col, encoder in self.target_encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])

        return df

    def fit_transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df = self.fill_guest_popularity_with_gamma(df)
        df = self.remove_missing(df)
        df = self.remove_outliers_iqr(df)
        self.fit(df, target_col)
        df = self.transform(df)
        return df

preprocessor = PodcastPreprocessor()
df = preprocessor.fit_transform(df, target_col='Listening_Time_minutes')
eda = PodcastEDA(df)
eda.run_all(nbins=30)


target_col = 'Listening_Time_minutes'

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 1000, 5000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'random_strength': trial.suggest_float('random_strength', 1.0, 20.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'early_stopping_rounds': 100,
        'verbose': 100,
        'random_seed': 42
    }

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)

    preds = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, preds, squared=False)
    return rmse

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50, timeout=3600)

# print("Ğ›ÑƒÑ‡ÑˆĞ¸Ğµ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹:", study.best_params)
# print("Ğ›ÑƒÑ‡ÑˆĞµĞµ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğµ RMSE:", study.best_value)

# optuna.visualization.plot_optimization_history(study).show()

# optuna.visualization.plot_param_importances(study).show()

params = {'iterations': 3251,
         'depth': 7,
         'learning_rate': 0.04324832077365222,
         'l2_leaf_reg': 3.841820196149257,
         'subsample': 0.9697625553027608,
         'colsample_bylevel': 0.5242465013905303,
         'min_data_in_leaf': 9,
         'random_strength': 10.309366397864505,
         'bagging_temperature': 0.8771047827180957,
         'loss_function': 'RMSE',
         'eval_metric': 'RMSE',
         'early_stopping_rounds': 100,
         'verbose': 100,
         'random_seed': 42}

model = CatBoostRegressor(**params)
model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)


test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
test = preprocessor.transform(test)

submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
submission['Listening_Time_minutes'] = model.predict(test)
submission.to_csv('submission.csv', index=False)
submission

