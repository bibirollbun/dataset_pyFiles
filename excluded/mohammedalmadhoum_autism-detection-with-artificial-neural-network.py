import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
df = pd.read_csv('/kaggle/input/autismdiagnosis/Autism_Prediction/train.csv')


df.head()


# data shape
df.shape


# Class balance
plt.bar([1, 0], df['Class/ASD'].value_counts(), color=['#1f77b4', '#d62728'])
plt.xlabel('Class/ASD')
plt.ylabel('Count')
plt.title('Class Balance - Autism Screening')
plt.xticks([0, 1], ['ASD', 'Non-ASD'])
plt.show()


# See unique values for object/categorical columns
print("Categorical Columns Unique Values:")
print("=" * 50)
for col in df.select_dtypes(include=['object']).columns:
    print(f"{col}: {df[col].unique()}")
    print(f"   Value counts: {dict(df[col].value_counts().head())}")
    print()


# check duplicated
df.duplicated().sum()


# check Nan
df.isna().sum()


df['age'].describe()


df.info()


pd.unique(df['age_desc'])


# Discretization For Age
bins = [0, 18, 35, 55, 65]
age_label = ['Junior', 'Young Adult', 'Middle Age', 'Senior']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=age_label)

age_counts = df['age_group'].value_counts().sort_index()
plt.bar(age_label,age_counts, color='skyblue')
plt.show()


import plotly.express as px
fig = px.choropleth(df, 
                    locations='contry_of_res', 
                    locationmode='country names',
                    color_continuous_scale=['#1f77b4', '#ff7f0e'],
                    color='Class/ASD',
                    title='Distribution of autism cases by country')
fig.show()


import numpy as np

# Filter out unknown ethnicity values
df_filtered = df[df['ethnicity'] != '?']

# Create a cross-tabulation of ethnicity vs Class/ASD
cross_tab = pd.crosstab(df_filtered['ethnicity'], df_filtered['Class/ASD'])

# Sort by total counts for better visualization
cross_tab['total'] = cross_tab.sum(axis=1)
cross_tab = cross_tab.sort_values('total', ascending=False).drop('total', axis=1)

# Set up the plot
plt.figure(figsize=(12, 8))
bar_width = 0.35
x_pos = np.arange(len(cross_tab.index))

# Create bars for each class
bars0 = plt.bar(x_pos - bar_width/2, cross_tab[0], bar_width, 
                label='Non-ASD (0)', color='#ff7f0e', alpha=0.8)
bars1 = plt.bar(x_pos + bar_width/2, cross_tab[1], bar_width, 
                label='ASD (1)', color='#1f77b4', alpha=0.8)

# Customize the chart
plt.xlabel('Ethnicity', fontsize=12, fontweight='bold')
plt.ylabel('Number of Cases', fontsize=12, fontweight='bold')
plt.title('Distribution of Autism Cases by Ethnicity', fontsize=14, fontweight='bold')
plt.xticks(x_pos, cross_tab.index, rotation=45, ha='right')
plt.legend()

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # Only add label if height > 0
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')

add_value_labels(bars0)
add_value_labels(bars1)

# Add grid for better readability
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# Show the plot
plt.show()


df.drop(columns=['ID', 'age_group', 'relation', 'age_desc', 'used_app_before', 'contry_of_res'], inplace=True)


df.info()


from sklearn.preprocessing import LabelEncoder
categorical_columns = ['gender', 'ethnicity', 'jaundice', 'austim']

label_encoder = LabelEncoder()

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col].astype(str))




df.info()


df


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Example data
X = df.drop('Class/ASD', axis=1).values  # features
y = df['Class/ASD'].values               # binary target

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



X_train.shape


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Dense(8, input_dim = 16, kernel_initializer='normal', activation='relu'))
model.add(Dense(5, activation = "relu", kernel_initializer='normal'))
model.add(Dense(1, activation = 'sigmoid'))


# compiling model
model.compile(optimizer = Adam(learning_rate = 0.001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])


# Train the model
history =  model.fit(X_train, y_train, epochs = 20, batch_size = 10)


loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")





