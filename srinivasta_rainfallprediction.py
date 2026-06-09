import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
import optuna
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns


# Load the training dataset
train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')

# Load the test dataset
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Convert 'rainfall' column to binary (0/1) if it's not already
train_data['rainfall'] = train_data['rainfall'].astype(int)

# Select features and target variable
features = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
X_train = train_data[features]
y_train = train_data['rainfall']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    C = trial.suggest_float("C", 1e-10, 1e10, log=True)
    solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"])
    max_iter = trial.suggest_int("max_iter", 100, 1000)

    # Create a pipeline with scaling and the model
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),  # Use MinMaxScaler
        ('model', LogisticRegression(C=C, solver=solver, max_iter=max_iter))
    ])

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Evaluate the model using AUC
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    return auc  # Optuna maximizes this value

# Create an Optuna study (maximize AUC)
study = optuna.create_study(direction="maximize")

# Optimize the hyperparameters
study.optimize(objective, n_trials=100)

# Get the best hyperparameters
best_params = study.best_params
print(f"Best hyperparameters: {best_params}")

# Train the model with the best hyperparameters, including scaling
best_pipeline = Pipeline([
    ('scaler', MinMaxScaler()),  # Use MinMaxScaler
    ('model', LogisticRegression(**best_params))
])
best_pipeline.fit(X_train, y_train)

# Select features for prediction and keep column names
X_test = test_data[features].copy()

# Impute missing values using the mean
imputer = SimpleImputer(strategy='mean')
X_test_imputed = imputer.fit_transform(X_test)

# Convert the imputed array back to a DataFrame with original column names
X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns, index=X_test.index)

# Make predictions on the imputed test data using the pipeline (get probabilities)
test_predictions_proba = best_pipeline.predict_proba(X_test_imputed)[:, 1]

# Create a submission DataFrame with probabilities
submission_df = pd.DataFrame({'id': test_data['id'], 'rainfall': test_predictions_proba})

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)



# Define the titles for the plots
titles = [
    "Pressure",
    "Maximum Temperature",
    "Temperature",
    "Minimum Temperature",
    "Temperature (dew point)",
    "Relative Humidity",
    "Cloud Cover",
    "Sunshine",
    "Wind speed",
    "Wind direction in degrees",
    "Rainfall",
]

# Define the feature keys corresponding to the titles
feature_keys = [
    "pressure",
    "maxtemp",
    "temparature",
    "mintemp",
    "dewpoint",
    "humidity",
    "cloud",
    "sunshine",
    "winddirection",
    "windspeed",
    "rainfall",
]

colors = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gold",
    "yellow",
    "gray",
    "olive",
    "cyan",
]

def show_raw_visualization(data):
    fig, axes = plt.subplots(
        nrows=6, ncols=2, figsize=(15, 24), dpi=80, facecolor="w", edgecolor="k"
    )
    axes = axes.flatten()

    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = colors[i % (len(colors))]
        t_data = data[[key, 'day']]  # Select the feature and 'day'
        t_data.set_index('day', inplace=True)  # Set 'day' as the index
        ax = t_data.plot(
            ax=axes[i],
            color=c,
            title="{} - {}".format(titles[i], key),
            rot=25,
        )
        ax.legend([titles[i]])
    plt.tight_layout()

# Load the data
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")  # Replace with your file path

# Assuming 'day' column is not present and index represents day
train['day'] = train.index  # Assign index values to 'day' column

# Visualize the data
show_raw_visualization(train)


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Assuming 'train' DataFrame is already loaded
numerical_features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']

# Create subplots
num_plots = len(numerical_features)
num_cols = 3
num_rows = (num_plots + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
axes = axes.flatten()  # Flatten the axes array for easier iteration

# Define a list of colors for each plot (you can customize these)
colors = ['skyblue', 'coral', 'lightgreen', 'plum', 'lightseagreen', 'salmon', 'wheat',  'gold', 'orchid']

for i, feature in enumerate(numerical_features):
    sns.histplot(train[feature], kde=True, ax=axes[i], color=colors[i % len(colors)])  # Assign color using modulo
    axes[i].set_title(f'Distribution of {feature}')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_axis_off()

plt.tight_layout()
plt.show()


display(submission_df)

