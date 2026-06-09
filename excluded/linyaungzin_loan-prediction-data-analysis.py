import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
train_df.drop(columns=['id'], inplace=True)
train_df.head()


train_df.describe(include='all')


train_df["loan_paid_back"].value_counts()


for column in train_df[["gender", "marital_status", "education_level", "employment_status", "loan_purpose", "grade_subgrade"]]:
    print(f"Column: {column}")
    print(f"Uniques: {train_df[column].unique()}")
    print()


norminal_columns = ["gender", "marital_status", "education_level", "employment_status", "loan_purpose"]

encoder = OneHotEncoder()
encoder.fit(train_df[norminal_columns])
encoded_norminal_cols_array = encoder.transform(train_df[norminal_columns]).toarray()
encoded_norminal_cols_df = pd.DataFrame(encoded_norminal_cols_array, columns=encoder.get_feature_names_out())
encoded_norminal_cols_df.head()


ordinal_encoder = OrdinalEncoder()
ordinal_encoder.fit(train_df[["grade_subgrade"]])
encoded_ordinal_columns_array = ordinal_encoder.transform(train_df[["grade_subgrade"]])
encoded_ordinal_columns_df = pd.DataFrame(encoded_ordinal_columns_array, columns=["grade_subgrade"])
encoded_ordinal_columns_df.head()


train_df.drop(columns=norminal_columns+["grade_subgrade"], inplace=True)
x_data = train_df.drop(columns=["loan_paid_back"])


scaler = StandardScaler()
scaler.fit(x_data)
standardized_x_data = scaler.transform(x_data)
standardized_x_df = pd.DataFrame(standardized_x_data, columns=scaler.get_feature_names_out())
standardized_x_df


encoded_standard_df = pd.concat([standardized_x_df, encoded_norminal_cols_df, encoded_ordinal_columns_df, train_df[["loan_paid_back"]]], axis=1)
encoded_standard_df.head()


sns.boxplot(
    data=encoded_standard_df[["annual_income", "debt_to_income_ratio", "credit_score", "loan_amount", "interest_rate"]],
    orient='h'
)
sns.despine(offset=10, trim=True)


# Compute correlation matrix
correlation_matrix = encoded_standard_df.corr(numeric_only=True)

# Generate mask for upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


encoded_standard_df.skew().to_dict()


from scipy.stats import yeojohnson

yeojohnson_df = encoded_standard_df.copy()
for k, v in yeojohnson_df.skew().to_dict().items():
    if k != 'loan_paid_back' and abs(round(v, 1)) > 0.0:
        print(f"Processing column {k} with skewness {v}")
        yeojohnson_df[k], _ = yeojohnson(yeojohnson_df[k])


yeojohnson_df.skew()


from sklearn.preprocessing import QuantileTransformer

quantile_transformers = {}

quantile_df = encoded_standard_df.copy()

for k, v in quantile_df.skew().to_dict().items():
    if k != 'loan_paid_back' and abs(round(v, 1)) > 0.0:
        q_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
        quantile_df[k] = q_transformer.fit_transform(quantile_df[k].values.reshape(-1, 1)).flatten()
        quantile_transformers[k] = q_transformer
        after_skew = quantile_df[k].skew()
        print(f"Processing column {k} with skewness {v} to {after_skew}")


quantile_df.skew()


# Compute correlation matrix
correlation_matrix = quantile_df.corr(numeric_only=True)

# Generate mask for upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

