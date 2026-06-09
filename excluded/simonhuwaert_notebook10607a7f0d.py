from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


state_data_file_path = 'd:\\Downloads\\forest-fire-prediction-epoch-hackathon\\new_Book1.csv'
state_data = pd.read_csv(state_data_file_path)

features = ['month', 'PRCP', 'EVAP', 'TMIN', 'TMAX', 'TAVG']
X = state_data[features]
#print(X)
y = state_data.total_fire_size
#print(y)

#train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(X, y)


#forest_preds = forest_model.predict(val_X)
#print(mean_absolute_error(val_y, forest_preds))

test_data_file_path = 'd:\\Downloads\\forest-fire-prediction-epoch-hackathon\\rerenew_Book1.csv'
test_data = pd.read_csv(test_data_file_path)

test_X = test_data[features]
comp_preds = forest_model.predict(test_X)
print(comp_preds)


csv_file = "d:\\Downloads\\forest-fire-prediction-epoch-hackathon\\rerenew_Book1.csv"  # Replace with your CSV file
new_column_data = comp_preds  # Replace with your list

df = pd.read_csv(csv_file)
df["fire_area"] = new_column_data  # Replace "your_column" with the column name
df.to_csv(csv_file, index=False)  # Overwrites the original file

print("Column replaced successfully.")

