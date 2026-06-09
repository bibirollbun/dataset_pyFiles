# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
        
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt
import seaborn as sns


train_sales=pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv')
calendar=pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv')
inventory=pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv')
test_sales=pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv')
test_weights=pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv')
solution=pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv')


import pandas as pd

def basic_eda(df, name):
    """Perform basic EDA for a given DataFrame."""
    print(f"\nğŸ”¹ Dataset: {name}")
    print("-" * 50)
    print(f" Shape: {df.shape}")
    print("\n Missing Values:")
    print(df.isnull().sum())
    print("\n Data Types:")
    print(df.dtypes)
    print("\n First 5 Rows:")
    print(df.head())
    print("\n" + "="*80)

# List of datasets with their names
datasets = {
    "train_sales": train_sales,
    "calendar": calendar,
    "inventory": inventory,
    "test_sales": test_sales,
    "test_weights": test_weights,
    
}

# Run EDA on all datasets
for name, df in datasets.items():
    basic_eda(df, name)


train_sales['sales'].fillna(train_sales['sales'].mean(), inplace=True)
train_sales['total_orders'].fillna(train_sales['total_orders'].mean(), inplace=True)



train_sales.isnull().sum()


## Check for Duplicates Values ##
train_sales.drop_duplicates()
test_sales = test_sales.drop_duplicates()
inventory = inventory.drop_duplicates()
test_weights=test_weights.drop_duplicates()


print (f"Date duplicates in calendar: {calendar.duplicated(subset=['date']).sum()}")



calendar_df = calendar.drop_duplicates(subset=['date'], keep='first')
print(f"After removing duplicates in calendar_df for date: {calendar_df.shape[0]} rows")


is_unique_5 = calendar_df['date'].is_unique
print(f"Are all 'date' values unique? {is_unique_5}")


# Creating Dataframe to fill in holiday_name blank values for year 2020

holiday_name_2020_Frankfurt_1_url = "https://www.timeanddate.com/calendar/?year=2020&country=8"
holiday_name_2020_Munich_1_url = "https://www.timeanddate.com/calendar/?year=2020&country=8"

holiday_name_2020_Frankfurt_1_table = pd.read_html(holiday_name_2020_Frankfurt_1_url)
holiday_name_2020_Munich_1_table = pd.read_html(holiday_name_2020_Munich_1_url)

# Define column headers
columns = ["date", "holiday_name", "Country", "Year"]


# For Frankfurt_1_2020
holiday_name_2020_Frankfurt_1_table = pd.concat([
    holiday_name_2020_Frankfurt_1_table[15],
    holiday_name_2020_Frankfurt_1_table[16],
    holiday_name_2020_Frankfurt_1_table[17]
], ignore_index=True)

# Define the original DataFrame with raw holiday data
holiday_name_2020_Frankfurt_1_table = pd.DataFrame({
    "date": [
        "1. Jan", "6. Jan", "14. Feb", "24. Feb", "25. Feb", "26. Feb", "8. MÃ¤r", "8. MÃ¤r",
        "5. Apr", "9. Apr", "10. Apr", "11. Apr", "12. Apr", "12. Apr", "13. Apr", "1. Mai",
        "8. Mai", "10. Mai", "21. Mai", "21. Mai", "31. Mai", "31. Mai", "1. Jun", "11. Jun",
        "15. Aug", "15. Aug", "20. Sep", "3. Okt", "31. Okt", "31. Okt", "1. Nov", "11. Nov",
        "15. Nov", "18. Nov", "22. Nov", "29. Nov", "6. Dez", "6. Dez", "13. Dez", "20. Dez",
        "25. Dez", "26. Dez"
    ],
    "holiday_name": [
        "New Year's Day", "Epiphany (BM, BY, ST)", "Valentine's Day", "Shrove Monday", "Carnival Tuesday",
        "Carnival/Ash Wednesday", "International Women's Day (Most Regions)", "International Women's Day (Berlin)",
        "Palm Sunday", "Maundy Thursday (All)", "Good Friday", "Holy Saturday (Many regions)", "Easter Sunday (Brandenburg)",
        "Eastern Sunday (Most regions)", "Easter Monday", "May Day", "End of World War 2 (Many regions)",
        "Mothers' Day", "Fathers' Day", "Ascension Day", "Whit Sunday (Brandenburg)",
        "Whit Sunday (Most regions)", "Whit Monday", "Corpus Christi (Many regions)",
        "Assumption of Mary (Bavaria, Saarland)", "Assumption of Mary (Saxony, Thuringia)",
        "German World Children's Day (Thuringia)", "Day of German Unity", "Reformation Day (Most Regions)",
        "Halloween", "All Saints' Day (Many Regions)", "St. Martin's Day", "National Day of Mourning",
        "Repentance Day (Saxony)", "Sunday of the Dead", "First Advent Sunday", "Second Advent Sunday",
        "Saint Nicholas Day", "Third Advent Sunday", "Fourth Advent Sunday", "Christmas Day", "Boxing Day"
    ]
})



# Convert date format to YYYY-MM-DD
month_map = {
    "Jan": "01", "Feb": "02", "MÃ¤r": "03", "Apr": "04", "Mai": "05", "Jun": "06",
    "Jul": "07", "Aug": "08", "Sep": "09", "Okt": "10", "Nov": "11", "Dez": "12"
}

# Format the date column into YYYY-MM-DD
holiday_name_2020_Frankfurt_1_table["date"] = holiday_name_2020_Frankfurt_1_table["date"].apply(
    lambda x: f"2020-{month_map[x.split('.')[1].strip()]}-{x.split('.')[0].zfill(2)}"
)

# Add Year and Country columns
holiday_name_2020_Frankfurt_1_table["Year"] = 2020
holiday_name_2020_Frankfurt_1_table["Country"] = "Germany (Frankfurt)"

# For Munich_1_2020
holiday_name_2020_Munich_1_table = pd.concat([
    holiday_name_2020_Munich_1_table[15],
    holiday_name_2020_Munich_1_table[16],
    holiday_name_2020_Munich_1_table[17]
], ignore_index=True)

holiday_name_2020_Munich_1_table = pd.DataFrame({
    "date": [
        "1. Jan", "6. Jan", "14. Feb", "24. Feb", "25. Feb", "26. Feb", "8. MÃ¤r", "8. MÃ¤r",
        "5. Apr", "9. Apr", "10. Apr", "11. Apr", "12. Apr", "12. Apr", "13. Apr", "1. Mai",
        "8. Mai", "10. Mai", "21. Mai", "21. Mai", "31. Mai", "31. Mai", "1. Jun", "11. Jun",
        "15. Aug", "15. Aug", "20. Sep", "3. Okt", "31. Okt", "31. Okt", "1. Nov", "11. Nov",
        "15. Nov", "18. Nov", "22. Nov", "29. Nov", "6. Dez", "6. Dez", "13. Dez", "20. Dez",
        "25. Dez", "26. Dez"
    ],
    "holiday_name": [
        "New Year's Day", "Epiphany (BM, BY, ST)", "Valentine's Day", "Shrove Monday", "Carnival Tuesday",
        "Carnival/Ash Wednesday", "International Women's Day (Most Regions)", "International Women's Day (Berlin)",
        "Palm Sunday", "Maundy Thursday (All)", "Good Friday", "Holy Saturday (Many regions)", "Easter Sunday (Brandenburg)",
        "Eastern Sunday (Most regions)", "Easter Monday", "May Day", "End of World War 2 (Many regions)",
        "Mothers' Day", "Fathers' Day", "Ascension Day", "Whit Sunday (Brandenburg)",
        "Whit Sunday (Most regions)", "Whit Monday", "Corpus Christi (Many regions)",
        "Assumption of Mary (Bavaria, Saarland)", "Assumption of Mary (Saxony, Thuringia)",
        "German World Children's Day (Thuringia)", "Day of German Unity", "Reformation Day (Most Regions)",
        "Halloween", "All Saints' Day (Many Regions)", "St. Martin's Day", "National Day of Mourning",
        "Repentance Day (Saxony)", "Sunday of the Dead", "First Advent Sunday", "Second Advent Sunday",
        "Saint Nicholas Day", "Third Advent Sunday", "Fourth Advent Sunday", "Christmas Day", "Boxing Day"
    ]
})


# Convert date format to YYYY-MM-DD
month_map = {
    "Jan": "01", "Feb": "02", "MÃ¤r": "03", "Apr": "04", "Mai": "05", "Jun": "06",
    "Jul": "07", "Aug": "08", "Sep": "09", "Okt": "10", "Nov": "11", "Dez": "12"
}

# Format the date column into YYYY-MM-DD
holiday_name_2020_Munich_1_table["date"] = holiday_name_2020_Munich_1_table["date"].apply(
    lambda x: f"2020-{month_map[x.split('.')[1].strip()]}-{x.split('.')[0].zfill(2)}"
)

# Add Year and Country columns
holiday_name_2020_Munich_1_table["Year"] = 2020
holiday_name_2020_Munich_1_table["Country"] = "Germany (Munich)"

# For Prague and Brno holidays in 2020
Prague_1_holiday_data = {
    "date": [
        "2020-01-01", "2020-01-01", "2020-02-14", "2020-03-08", "2020-04-10",
        "2020-04-13", "2020-05-01", "2020-05-08", "2020-05-10", "2020-06-01",
        "2020-06-21", "2020-07-05", "2020-07-06", "2020-09-28", "2020-10-28",
        "2020-11-17", "2020-12-24", "2020-12-25", "2020-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Restoration of the Czech Independence Day", "St. Valentine's Day", 
        "International Women's Day", "Good Friday", "Easter Monday", "Labor Day / May Day",
        "Victory in Europe Day", "Mother's Day", "Children's Day", "Father's Day",
        "Day of Saints Cyril and Methodius", "Jan Hus Day", "St. Wenceslas Day",
        "Independent Czechoslovak State Day", "Struggle for Freedom and Democracy Day",
        "Christmas Eve", "Christmas Day", "St. Stephen's Day"
    ],
    "Country": ["Czech Republic (Prague_1)"] * 19,  # Apply same value to all rows
    "Year": [2020] * 19
}

Prague_2_holiday_data = {
    "date": [
        "2020-01-01", "2020-01-01", "2020-02-14", "2020-03-08", "2020-04-10",
        "2020-04-13", "2020-05-01", "2020-05-08", "2020-05-10", "2020-06-01",
        "2020-06-21", "2020-07-05", "2020-07-06", "2020-09-28", "2020-10-28",
        "2020-11-17", "2020-12-24", "2020-12-25", "2020-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Restoration of the Czech Independence Day", "St. Valentine's Day", 
        "International Women's Day", "Good Friday", "Easter Monday", "Labor Day / May Day",
        "Victory in Europe Day", "Mother's Day", "Children's Day", "Father's Day",
        "Day of Saints Cyril and Methodius", "Jan Hus Day", "St. Wenceslas Day",
        "Independent Czechoslovak State Day", "Struggle for Freedom and Democracy Day",
        "Christmas Eve", "Christmas Day", "St. Stephen's Day"
    ],
    "Country": ["Czech Republic (Prague_2)"] * 19,  # Apply same value to all rows
    "Year": [2020] * 19
}

Prague_3_holiday_data = {
    "date": [
        "2020-01-01", "2020-01-01", "2020-02-14", "2020-03-08", "2020-04-10",
        "2020-04-13", "2020-05-01", "2020-05-08", "2020-05-10", "2020-06-01",
        "2020-06-21", "2020-07-05", "2020-07-06", "2020-09-28", "2020-10-28",
        "2020-11-17", "2020-12-24", "2020-12-25", "2020-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Restoration of the Czech Independence Day", "St. Valentine's Day", 
        "International Women's Day", "Good Friday", "Easter Monday", "Labor Day / May Day",
        "Victory in Europe Day", "Mother's Day", "Children's Day", "Father's Day",
        "Day of Saints Cyril and Methodius", "Jan Hus Day", "St. Wenceslas Day",
        "Independent Czechoslovak State Day", "Struggle for Freedom and Democracy Day",
        "Christmas Eve", "Christmas Day", "St. Stephen's Day"
    ],
    "Country": ["Czech Republic (Prague_3)"] * 19,  # Apply same value to all rows
    "Year": [2020] * 19
}

Brno_1_holiday_data = {
    "date": [
        "2020-01-01", "2020-01-01", "2020-02-14", "2020-03-08", "2020-04-10",
        "2020-04-13", "2020-05-01", "2020-05-08", "2020-05-10", "2020-06-01",
        "2020-06-21", "2020-07-05", "2020-07-06", "2020-09-28", "2020-10-28",
        "2020-11-17", "2020-12-24", "2020-12-25", "2020-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Restoration of the Czech Independence Day", "St. Valentine's Day", 
        "International Women's Day", "Good Friday", "Easter Monday", "Labor Day / May Day",
        "Victory in Europe Day", "Mother's Day", "Children's Day", "Father's Day",
        "Day of Saints Cyril and Methodius", "Jan Hus Day", "St. Wenceslas Day",
        "Independent Czechoslovak State Day", "Struggle for Freedom and Democracy Day",
        "Christmas Eve", "Christmas Day", "St. Stephen's Day"
    ],
    "Country": ["Czech Republic (Brno)"] * 19,  # Apply same value to all rows
    "Year": [2020] * 19
}

# Convert to DataFrame
holiday_name_2020_Prague_1_table = pd.DataFrame(Prague_1_holiday_data)
holiday_name_2020_Prague_2_table = pd.DataFrame(Prague_2_holiday_data)
holiday_name_2020_Prague_3_table = pd.DataFrame(Prague_3_holiday_data)
holiday_name_2020_Brno_1_table = pd.DataFrame(Brno_1_holiday_data)


# For Hungary holidays in 2020 (Budapest)
Budapest_1_holiday_data = {
    "date": [
        "2020-01-01", "2020-03-15", "2020-04-10", "2020-04-12", "2020-04-13",
        "2020-05-01", "2020-05-03", "2020-05-31", "2020-06-01", "2020-06-21",
        "2020-08-20", "2020-08-21", "2020-10-23", "2020-11-01", "2020-12-06",
        "2020-12-24", "2020-12-25", "2020-12-26", "2020-12-31"
    ],
    "holiday_name": [
        "New Year's Day", "1848 Revolution Memorial Day", "Good Friday", "Easter Sunday", "Easter Monday",
        "Labor Day / May Day", "Motherâ€™s Day", "Whit Sunday", "Whit Monday", "Fatherâ€™s Day",
        "Hungary National Day", "Hungary National Day Holiday", "1956 Revolution Memorial Day", "All Saints' Day",
        "Saint Nicholas Day", "Christmas Eve", "Christmas Day", "Second Day of Christmas", "New Year's Eve"
    ],
    "Country": ["Hungary (Budapest_1)"] * 19,  # Apply same value to all rows
    "Year": [2020] * 19
}

# Convert to DataFrame
holiday_name_2020_Budapest_1_table = pd.DataFrame(Budapest_1_holiday_data)

# One Data Frame for 2020
holiday_name_2020 = pd.concat([
    holiday_name_2020_Frankfurt_1_table,
    holiday_name_2020_Munich_1_table,
    holiday_name_2020_Prague_1_table,
    holiday_name_2020_Prague_2_table,
    holiday_name_2020_Prague_3_table,
    holiday_name_2020_Brno_1_table,
    holiday_name_2020_Budapest_1_table
], ignore_index=True)


holiday_name_2021_Frankfurt_1 = {
    "date": [
        "2021-01-01", "2021-01-06", "2021-02-14", "2021-02-15", "2021-02-16", "2021-02-17",
        "2021-03-08", "2021-03-08", "2021-03-28", "2021-04-01", "2021-04-02", "2021-04-03",
        "2021-04-04", "2021-04-04", "2021-04-05", "2021-05-01", "2021-05-08", "2021-05-09",
        "2021-05-13", "2021-05-13", "2021-05-23", "2021-05-23", "2021-05-24", "2021-06-03",
        "2021-08-15", "2021-08-15", "2021-09-20", "2021-10-03", "2021-10-31", "2021-10-31",
        "2021-11-01", "2021-11-11", "2021-11-14", "2021-11-17", "2021-11-21", "2021-11-28",
        "2021-12-05", "2021-12-06", "2021-12-12", "2021-12-19", "2021-12-25", "2021-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Epiphany (BW, BY, ST)", "Valentine's Day", "Shrove Monday",
        "Carnival Tuesday", "Carnival / Ash Wednesday", "International Women's Day (Most regions)",
        "International Women's Day (Berlin)", "Palm Sunday", "Maundy Thursday (All)", "Good Friday",
        "Holy Saturday (Many regions)", "Easter Sunday (Brandenburg)", "Easter Sunday (Most regions)",
        "Easter Monday", "May Day", "End of World War II (Many regions)", "Mothers' Day",
        "Fathers' Day", "Ascension Day", "Whit Sunday (Most regions)", "Whit Sunday (Brandenburg)",
        "Whit Monday", "Corpus Christi (Many regions)", "Assumption of Mary (Bavaria, Saarland)",
        "Assumption of Mary (Saxony, Thuringia)", "German World Children's Day (Thuringia)",
        "Day of German Unity", "Reformation Day (Most regions)", "Halloween", "All Saints' Day (Many regions)",
        "St. Martin's Day", "National Day of Mourning", "Repentance Day (Saxony)", "Sunday of the Dead",
        "First Advent Sunday", "Second Advent Sunday", "Saint Nicholas Day", "Third Advent Sunday",
        "Fourth Advent Sunday", "Christmas Day", "Boxing Day"
    ],
    "Country": ["Germany (Frankfurt_1)"] * 42,  # Apply the same value to all rows
    "Year": [2021] * 42
}

# Convert to DataFrame
holiday_name_2021_Frankfurt_1_table = pd.DataFrame(holiday_name_2021_Frankfurt_1)


holiday_name_2021_Munich_1 = {
    "date": [
        "2021-01-01", "2021-01-06", "2021-02-14", "2021-02-15", "2021-02-16", "2021-02-17",
        "2021-03-08", "2021-03-08", "2021-03-28", "2021-04-01", "2021-04-02", "2021-04-03",
        "2021-04-04", "2021-04-04", "2021-04-05", "2021-05-01", "2021-05-08", "2021-05-09",
        "2021-05-13", "2021-05-13", "2021-05-23", "2021-05-23", "2021-05-24", "2021-06-03",
        "2021-08-15", "2021-08-15", "2021-09-20", "2021-10-03", "2021-10-31", "2021-10-31",
        "2021-11-01", "2021-11-11", "2021-11-14", "2021-11-17", "2021-11-21", "2021-11-28",
        "2021-12-05", "2021-12-06", "2021-12-12", "2021-12-19", "2021-12-25", "2021-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Epiphany (BW, BY, ST)", "Valentine's Day", "Shrove Monday",
        "Carnival Tuesday", "Carnival / Ash Wednesday", "International Women's Day (Most regions)",
        "International Women's Day (Berlin)", "Palm Sunday", "Maundy Thursday (All)", "Good Friday",
        "Holy Saturday (Many regions)", "Easter Sunday (Brandenburg)", "Easter Sunday (Most regions)",
        "Easter Monday", "May Day", "End of World War II (Many regions)", "Mothers' Day",
        "Fathers' Day", "Ascension Day", "Whit Sunday (Most regions)", "Whit Sunday (Brandenburg)",
        "Whit Monday", "Corpus Christi (Many regions)", "Assumption of Mary (Bavaria, Saarland)",
        "Assumption of Mary (Saxony, Thuringia)", "German World Children's Day (Thuringia)",
        "Day of German Unity", "Reformation Day (Most regions)", "Halloween", "All Saints' Day (Many regions)",
        "St. Martin's Day", "National Day of Mourning", "Repentance Day (Saxony)", "Sunday of the Dead",
        "First Advent Sunday", "Second Advent Sunday", "Saint Nicholas Day", "Third Advent Sunday",
        "Fourth Advent Sunday", "Christmas Day", "Boxing Day"
    ],
    "Country": ["Czech Republic (Prague)"] * 42,  # Apply the same value to all rows
    "Year": [2021] * 42
}

# Convert to DataFrame
holiday_name_2021_Munich_1_table = pd.DataFrame(holiday_name_2021_Munich_1)


holiday_name_2021_Prague_1 = {
    "date": [
        "2021-01-01", "2021-01-01", "2021-02-14", "2021-03-08", "2021-04-02",
        "2021-04-05", "2021-05-01", "2021-05-08", "2021-05-09", "2021-06-01",
        "2021-06-20", "2021-07-05", "2021-07-06", "2021-09-28", "2021-10-28",
        "2021-11-17", "2021-12-24", "2021-12-25", "2021-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Restoration of the Czech Independence Day", "St. Valentine's Day",
        "International Women's Day", "Good Friday", "Easter Monday", "Labor Day / May Day",
        "Victory in Europe Day", "Mother's Day", "Children's Day", "Father's Day",
        "Day of Saints Cyril and Methodius", "Jan Hus Day", "St. Wenceslas Day",
        "Independent Czechoslovak State Day", "Struggle for Freedom and Democracy Day",
        "Christmas Eve", "Christmas Day", "St. Stephen's Day"
    ],
    "Country": ["Czech Republic (Prague_1)"] * 19,  # Apply the same value to all rows
    "Year": [2021] * 19
}

holiday_name_2021_Prague_2 = {
    "date": [
        "2021-01-01", "2021-01-01", "2021-02-14", "2021-03-08", "2021-04-02",
        "2021-04-05", "2021-05-01", "2021-05-08", "2021-05-09", "2021-06-01",
        "2021-06-20", "2021-07-05", "2021-07-06", "2021-09-28", "2021-10-28",
        "2021-11-17", "2021-12-24", "2021-12-25", "2021-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Restoration of the Czech Independence Day", "St. Valentine's Day",
        "International Women's Day", "Good Friday", "Easter Monday", "Labor Day / May Day",
        "Victory in Europe Day", "Mother's Day", "Children's Day", "Father's Day",
        "Day of Saints Cyril and Methodius", "Jan Hus Day", "St. Wenceslas Day",
        "Independent Czechoslovak State Day", "Struggle for Freedom and Democracy Day",
        "Christmas Eve", "Christmas Day", "St. Stephen's Day"
    ],
    "Country": ["Czech Republic (Prague_2)"] * 19,  # Apply the same value to all rows
    "Year": [2021] * 19
}

holiday_name_2021_Prague_3 = {
    "date": [
        "2021-01-01", "2021-01-01", "2021-02-14", "2021-03-08", "2021-04-02",
        "2021-04-05", "2021-05-01", "2021-05-08", "2021-05-09", "2021-06-01",
        "2021-06-20", "2021-07-05", "2021-07-06", "2021-09-28", "2021-10-28",
        "2021-11-17", "2021-12-24", "2021-12-25", "2021-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Restoration of the Czech Independence Day", "St. Valentine's Day",
        "International Women's Day", "Good Friday", "Easter Monday", "Labor Day / May Day",
        "Victory in Europe Day", "Mother's Day", "Children's Day", "Father's Day",
        "Day of Saints Cyril and Methodius", "Jan Hus Day", "St. Wenceslas Day",
        "Independent Czechoslovak State Day", "Struggle for Freedom and Democracy Day",
        "Christmas Eve", "Christmas Day", "St. Stephen's Day"
    ],
    "Country": ["Czech Republic (Prague_3)"] * 19,  # Apply the same value to all rows
    "Year": [2021] * 19
}

# Convert to DataFrame
holiday_name_2021_Prague_1_table = pd.DataFrame(holiday_name_2021_Prague_1)
holiday_name_2021_Prague_2_table = pd.DataFrame(holiday_name_2021_Prague_2)
holiday_name_2021_Prague_3_table = pd.DataFrame(holiday_name_2021_Prague_3)

holiday_name_2021_Brno_1 = {
    "date": [
        "2021-01-01", "2021-01-01", "2021-02-14", "2021-03-08", "2021-04-02",
        "2021-04-05", "2021-05-01", "2021-05-08", "2021-05-09", "2021-06-01",
        "2021-06-20", "2021-07-05", "2021-07-06", "2021-09-28", "2021-10-28",
        "2021-11-17", "2021-12-24", "2021-12-25", "2021-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Restoration of the Czech Independence Day", "St. Valentine's Day",
        "International Women's Day", "Good Friday", "Easter Monday", "Labor Day / May Day",
        "Victory in Europe Day", "Mother's Day", "Children's Day", "Father's Day",
        "Day of Saints Cyril and Methodius", "Jan Hus Day", "St. Wenceslas Day",
        "Independent Czechoslovak State Day", "Struggle for Freedom and Democracy Day",
        "Christmas Eve", "Christmas Day", "St. Stephen's Day"
    ],
    "Country": ["Czech Republic (Brno)"] * 19,  # Apply the same value to all rows
    "Year": [2021] * 19
}

# Convert to DataFrame
holiday_name_2021_Brno_1_table = pd.DataFrame(holiday_name_2021_Brno_1) 


holiday_name_2021_Budapest_1 = {
    "date": [
        "2021-01-01", "2021-03-15", "2021-04-02", "2021-04-04", "2021-04-05",
        "2021-05-01", "2021-05-02", "2021-05-23", "2021-05-24", "2021-06-20",
        "2021-08-20", "2021-10-23", "2021-11-01", "2021-12-06", "2021-12-24",
        "2021-12-25", "2021-12-26", "2021-12-31"
    ],
    "holiday_name": [
        "New Year's Day", "1848 Revolution Memorial Day", "Good Friday", "Easter Sunday",
        "Easter Monday", "Labor Day / May Day", "Motherâ€™s Day", "Whit Sunday",
        "Whit Monday", "Fatherâ€™s Day", "Hungary National Day", "1956 Revolution Memorial Day",
        "All Saints' Day", "Saint Nicholas Day", "Christmas Eve", "Christmas Day",
        "Second Day of Christmas", "New Year's Eve"
    ],
    "Country": ["Hungary (Budapest)"] * 18,  # Apply the same value to all rows
    "Year": [2021] * 18
}

# Convert to DataFrame
holiday_name_2021_Budapest_1_table = pd.DataFrame(holiday_name_2021_Budapest_1)


# One Data Frame for 2021
holiday_name_2021 = pd.concat([
    holiday_name_2021_Frankfurt_1_table,
    holiday_name_2021_Munich_1_table,
    holiday_name_2021_Prague_1_table,
    holiday_name_2021_Prague_2_table,
    holiday_name_2021_Prague_3_table,
    holiday_name_2021_Brno_1_table,
    holiday_name_2021_Budapest_1_table
], ignore_index=True)


# Creating Dataframe to fill in holiday_name blank values for year 2022

holiday_name_2022_Frankfurt_1 = {
    "date": [
        "2022-01-01", "2022-01-06", "2022-02-14", "2022-02-28", "2022-03-01", "2022-03-02",
        "2022-03-08", "2022-03-08", "2022-04-10", "2022-04-14", "2022-04-15", "2022-04-16",
        "2022-04-17", "2022-04-17", "2022-04-18", "2022-05-01", "2022-05-08", "2022-05-08",
        "2022-05-26", "2022-05-26", "2022-06-05", "2022-06-05", "2022-06-06", "2022-06-16",
        "2022-08-15", "2022-08-15", "2022-09-20", "2022-10-03", "2022-10-31", "2022-10-31",
        "2022-11-01", "2022-11-11", "2022-11-13", "2022-11-16", "2022-11-20", "2022-11-27",
        "2022-12-04", "2022-12-06", "2022-12-11", "2022-12-18", "2022-12-25", "2022-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Epiphany (BW, BY, ST)", "Valentine's Day", "Shrove Monday",
        "Carnival Tuesday", "Carnival / Ash Wednesday", "International Women's Day (Most regions)",
        "International Women's Day (Berlin)", "Palm Sunday", "Maundy Thursday (All)",
        "Good Friday", "Holy Saturday (Many regions)", "Easter Sunday (Brandenburg)",
        "Easter Sunday (Most regions)", "Easter Monday", "May Day", "Mothers' Day",
        "End of World War II (Many regions)", "Fathers' Day", "Ascension Day",
        "Whit Sunday (Most regions)", "Whit Sunday (Brandenburg)", "Whit Monday",
        "Corpus Christi (Many regions)", "Assumption of Mary (Bavaria, Saarland)",
        "Assumption of Mary (Saxony, Thuringia)", "German World Children's Day (Thuringia)",
        "Day of German Unity", "Reformation Day (Most regions)", "Halloween",
        "All Saints' Day (Many regions)", "St. Martin's Day", "National Day of Mourning",
        "Repentance Day (Saxony)", "Sunday of the Dead", "First Advent Sunday",
        "Second Advent Sunday", "Saint Nicholas Day", "Third Advent Sunday",
        "Fourth Advent Sunday", "Christmas Day", "Boxing Day"
    ],
    "Country": ["Germany (Frankfurt)"] * 42,  # Apply the same value to all rows
    "Year": [2022] * 42
}

# Convert to DataFrame
holiday_name_2022_Frankfurt_1_table = pd.DataFrame(holiday_name_2022_Frankfurt_1)

holiday_name_2022_Munich_1 = {
    "date": [
        "2022-01-01", "2022-01-06", "2022-02-14", "2022-02-28", "2022-03-01", "2022-03-02",
        "2022-03-08", "2022-03-08", "2022-04-10", "2022-04-14", "2022-04-15", "2022-04-16",
        "2022-04-17", "2022-04-17", "2022-04-18", "2022-05-01", "2022-05-08", "2022-05-08",
        "2022-05-26", "2022-05-26", "2022-06-05", "2022-06-05", "2022-06-06", "2022-06-16",
        "2022-08-15", "2022-08-15", "2022-09-20", "2022-10-03", "2022-10-31", "2022-10-31",
        "2022-11-01", "2022-11-11", "2022-11-13", "2022-11-16", "2022-11-20", "2022-11-27",
        "2022-12-04", "2022-12-06", "2022-12-11", "2022-12-18", "2022-12-25", "2022-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Epiphany (BW, BY, ST)", "Valentine's Day", "Shrove Monday",
        "Carnival Tuesday", "Carnival / Ash Wednesday", "International Women's Day (Most regions)",
        "International Women's Day (Berlin)", "Palm Sunday", "Maundy Thursday (All)",
        "Good Friday", "Holy Saturday (Many regions)", "Easter Sunday (Brandenburg)",
        "Easter Sunday (Most regions)", "Easter Monday", "May Day", "Mothers' Day",
        "End of World War II (Many regions)", "Fathers' Day", "Ascension Day",
        "Whit Sunday (Most regions)", "Whit Sunday (Brandenburg)", "Whit Monday",
        "Corpus Christi (Many regions)", "Assumption of Mary (Bavaria, Saarland)",
        "Assumption of Mary (Saxony, Thuringia)", "German World Children's Day (Thuringia)",
        "Day of German Unity", "Reformation Day (Most regions)", "Halloween",
        "All Saints' Day (Many regions)", "St. Martin's Day", "National Day of Mourning",
        "Repentance Day (Saxony)", "Sunday of the Dead", "First Advent Sunday",
        "Second Advent Sunday", "Saint Nicholas Day", "Third Advent Sunday",
        "Fourth Advent Sunday", "Christmas Day", "Boxing Day"
    ],
    "Country": ["Germany (Munich)"] * 42,  # Apply the same value to all rows
    "Year": [2022] * 42
}

# Convert to DataFrame
holiday_name_2022_Munich_1_table = pd.DataFrame(holiday_name_2022_Munich_1)

holiday_name_2022_Prague_1 = {
    "date": [
        "2022-01-01", "2022-01-01", "2022-02-14", "2022-03-08", "2022-04-15",
        "2022-04-18", "2022-05-01", "2022-05-08", "2022-05-08", "2022-06-01",
        "2022-06-19", "2022-07-05", "2022-07-06", "2022-09-28", "2022-10-28",
        "2022-11-17", "2022-12-24", "2022-12-25", "2022-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Restoration of the Czech Independence Day", "St. Valentine's Day",
        "International Women's Day", "Good Friday", "Easter Monday", "Labor Day / May Day",
        "Victory in Europe Day", "Mother's Day", "Children's Day", "Father's Day",
        "Day of Saints Cyril and Methodius", "Jan Hus Day", "St. Wenceslas Day",
        "Independent Czechoslovak State Day", "Struggle for Freedom and Democracy Day",
        "Christmas Eve", "Christmas Day", "St. Stephen's Day"
    ],
    "Country": ["Czech Republic (Prague_1)"] * 19,  # Apply the same value to all rows
    "Year": [2022] * 19
}

holiday_name_2022_Prague_2 = {
    "date": [
        "2022-01-01", "2022-01-01", "2022-02-14", "2022-03-08", "2022-04-15",
        "2022-04-18", "2022-05-01", "2022-05-08", "2022-05-08", "2022-06-01",
        "2022-06-19", "2022-07-05", "2022-07-06", "2022-09-28", "2022-10-28",
        "2022-11-17", "2022-12-24", "2022-12-25", "2022-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Restoration of the Czech Independence Day", "St. Valentine's Day",
        "International Women's Day", "Good Friday", "Easter Monday", "Labor Day / May Day",
        "Victory in Europe Day", "Mother's Day", "Children's Day", "Father's Day",
        "Day of Saints Cyril and Methodius", "Jan Hus Day", "St. Wenceslas Day",
        "Independent Czechoslovak State Day", "Struggle for Freedom and Democracy Day",
        "Christmas Eve", "Christmas Day", "St. Stephen's Day"
    ],
    "Country": ["Czech Republic (Prague_2)"] * 19,  # Apply the same value to all rows
    "Year": [2022] * 19
}

holiday_name_2022_Prague_3 = {
    "date": [
        "2022-01-01", "2022-01-01", "2022-02-14", "2022-03-08", "2022-04-15",
        "2022-04-18", "2022-05-01", "2022-05-08", "2022-05-08", "2022-06-01",
        "2022-06-19", "2022-07-05", "2022-07-06", "2022-09-28", "2022-10-28",
        "2022-11-17", "2022-12-24", "2022-12-25", "2022-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Restoration of the Czech Independence Day", "St. Valentine's Day",
        "International Women's Day", "Good Friday", "Easter Monday", "Labor Day / May Day",
        "Victory in Europe Day", "Mother's Day", "Children's Day", "Father's Day",
        "Day of Saints Cyril and Methodius", "Jan Hus Day", "St. Wenceslas Day",
        "Independent Czechoslovak State Day", "Struggle for Freedom and Democracy Day",
        "Christmas Eve", "Christmas Day", "St. Stephen's Day"
    ],
    "Country": ["Czech Republic (Prague_3)"] * 19,  # Apply the same value to all rows
    "Year": [2022] * 19
}

# Convert to DataFrame
holiday_name_2022_Prague_1_table = pd.DataFrame(holiday_name_2022_Prague_1)
holiday_name_2022_Prague_2_table = pd.DataFrame(holiday_name_2022_Prague_2)
holiday_name_2022_Prague_3_table = pd.DataFrame(holiday_name_2022_Prague_3)

holiday_name_2022_Brno_1 = {
    "date": [
        "2022-01-01", "2022-01-01", "2022-02-14", "2022-03-08", "2022-04-15",
        "2022-04-18", "2022-05-01", "2022-05-08", "2022-05-08", "2022-06-01",
        "2022-06-19", "2022-07-05", "2022-07-06", "2022-09-28", "2022-10-28",
        "2022-11-17", "2022-12-24", "2022-12-25", "2022-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Restoration of the Czech Independence Day", "St. Valentine's Day",
        "International Women's Day", "Good Friday", "Easter Monday", "Labor Day / May Day",
        "Victory in Europe Day", "Mother's Day", "Children's Day", "Father's Day",
        "Day of Saints Cyril and Methodius", "Jan Hus Day", "St. Wenceslas Day",
        "Independent Czechoslovak State Day", "Struggle for Freedom and Democracy Day",
        "Christmas Eve", "Christmas Day", "St. Stephen's Day"
    ],
    "Country": ["Czech Republic (Brno)"] * 19,  # Apply the same value to all rows
    "Year": [2022] * 19
}

# Convert to DataFrame
holiday_name_2022_Brno_1_table = pd.DataFrame(holiday_name_2022_Brno_1) 

holiday_name_2022_Budapest_1 = {
    "date": [
        "2022-01-01", "2022-03-14", "2022-03-15", "2022-04-15", "2022-04-17",
        "2022-04-18", "2022-05-01", "2022-05-01", "2022-06-05", "2022-06-06",
        "2022-06-19", "2022-08-20", "2022-10-23", "2022-10-31", "2022-11-01",
        "2022-12-06", "2022-12-24", "2022-12-25", "2022-12-26", "2022-12-31"
    ],
    "holiday_name": [
        "New Year's Day", "1848 Revolution Memorial Day (Extra holiday)", "1848 Revolution Memorial Day",
        "Good Friday", "Easter Sunday", "Easter Monday", "Labor Day / May Day", "Motherâ€™s Day",
        "Whit Sunday", "Whit Monday", "Fatherâ€™s Day", "Hungary National Day", "1956 Revolution Memorial Day",
        "All Saints' Day Holiday", "All Saints' Day", "Saint Nicholas Day", "Christmas Eve",
        "Christmas Day", "Second Day of Christmas", "New Year's Eve"
    ],
    "Country": ["Hungary (Budapest)"] * 20,  # Apply the same value to all rows
    "Year": [2022] * 20
}

# Convert to DataFrame
holiday_name_2022_Budapest_1_table = pd.DataFrame(holiday_name_2022_Budapest_1)

# One Data Frame for 2022
holiday_name_2022 = pd.concat([
    holiday_name_2022_Frankfurt_1_table,
    holiday_name_2022_Munich_1_table,
    holiday_name_2022_Prague_1_table,
    holiday_name_2022_Prague_2_table,
    holiday_name_2022_Prague_3_table,
    holiday_name_2022_Brno_1_table,
    holiday_name_2022_Budapest_1_table
], ignore_index=True)


# Creating Dataframe to fill in holiday_name blank values for year 2023

holiday_name_2023_Frankfurt_1 = {
    "date": [
        "2023-01-01", "2023-01-06", "2023-02-14", "2023-02-20", "2023-02-21",
        "2023-02-22", "2023-03-08", "2023-03-08", "2023-04-02", "2023-04-06",
        "2023-04-07", "2023-04-08", "2023-04-09", "2023-04-09", "2023-04-10",
        "2023-05-01", "2023-05-08", "2023-05-14", "2023-05-18", "2023-05-18",
        "2023-05-28", "2023-05-28", "2023-05-29", "2023-06-08", "2023-08-15",
        "2023-08-15", "2023-09-20", "2023-10-03", "2023-10-31", "2023-10-31",
        "2023-11-01", "2023-11-11", "2023-11-19", "2023-11-22", "2023-11-26",
        "2023-12-03", "2023-12-06", "2023-12-10", "2023-12-17", "2023-12-24",
        "2023-12-25", "2023-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Epiphany (BW, BY, ST)", "Valentine's Day", "Shrove Monday",
        "Carnival Tuesday", "Carnival / Ash Wednesday", "International Women's Day (Most regions)",
        "International Women's Day (Berlin, Mecklenburg-Western Pomerania)", "Palm Sunday",
        "Maundy Thursday (All)", "Good Friday", "Holy Saturday (Many regions)",
        "Easter Sunday (Most regions)", "Easter Sunday (Brandenburg)", "Easter Monday",
        "May Day", "End of World War II (Many regions)", "Mothers' Day", "Fathers' Day",
        "Ascension Day", "Whit Sunday (Brandenburg)", "Whit Sunday (Most regions)",
        "Whit Monday", "Corpus Christi (Many regions)", "Assumption of Mary (Bavaria, Saarland)",
        "Assumption of Mary (Saxony, Thuringia)", "German World Children's Day (Thuringia)",
        "Day of German Unity", "Reformation Day (Most regions)", "Halloween",
        "All Saints' Day (Many regions)", "St. Martin's Day", "National Day of Mourning",
        "Repentance Day (Saxony)", "Sunday of the Dead", "First Advent Sunday",
        "Saint Nicholas Day", "Second Advent Sunday", "Third Advent Sunday",
        "Fourth Advent Sunday", "Christmas Day", "Boxing Day"
    ],
    "Country": ["Germany (Frankfurt)"] * 42,  # Apply the same value to all rows
    "Year": [2023] * 42
}

# Convert to DataFrame
holiday_name_2023_Frankfurt_1_table = pd.DataFrame(holiday_name_2023_Frankfurt_1)

holiday_name_2023_Munich_1 = {
    "date": [
        "2023-01-01", "2023-01-06", "2023-02-14", "2023-02-20", "2023-02-21",
        "2023-02-22", "2023-03-08", "2023-03-08", "2023-04-02", "2023-04-06",
        "2023-04-07", "2023-04-08", "2023-04-09", "2023-04-09", "2023-04-10",
        "2023-05-01", "2023-05-08", "2023-05-14", "2023-05-18", "2023-05-18",
        "2023-05-28", "2023-05-28", "2023-05-29", "2023-06-08", "2023-08-15",
        "2023-08-15", "2023-09-20", "2023-10-03", "2023-10-31", "2023-10-31",
        "2023-11-01", "2023-11-11", "2023-11-19", "2023-11-22", "2023-11-26",
        "2023-12-03", "2023-12-06", "2023-12-10", "2023-12-17", "2023-12-24",
        "2023-12-25", "2023-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Epiphany (BW, BY, ST)", "Valentine's Day", "Shrove Monday",
        "Carnival Tuesday", "Carnival / Ash Wednesday", "International Women's Day (Most regions)",
        "International Women's Day (Berlin, Mecklenburg-Western Pomerania)", "Palm Sunday",
        "Maundy Thursday (All)", "Good Friday", "Holy Saturday (Many regions)",
        "Easter Sunday (Most regions)", "Easter Sunday (Brandenburg)", "Easter Monday",
        "May Day", "End of World War II (Many regions)", "Mothers' Day", "Fathers' Day",
        "Ascension Day", "Whit Sunday (Brandenburg)", "Whit Sunday (Most regions)",
        "Whit Monday", "Corpus Christi (Many regions)", "Assumption of Mary (Bavaria, Saarland)",
        "Assumption of Mary (Saxony, Thuringia)", "German World Children's Day (Thuringia)",
        "Day of German Unity", "Reformation Day (Most regions)", "Halloween",
        "All Saints' Day (Many regions)", "St. Martin's Day", "National Day of Mourning",
        "Repentance Day (Saxony)", "Sunday of the Dead", "First Advent Sunday",
        "Saint Nicholas Day", "Second Advent Sunday", "Third Advent Sunday",
        "Fourth Advent Sunday", "Christmas Day", "Boxing Day"
    ],
    "Country": ["Germany (Munich)"] * 42,  # Apply the same value to all rows
    "Year": [2023] * 42
}

# Convert to DataFrame
holiday_name_2023_Munich_1_table = pd.DataFrame(holiday_name_2023_Munich_1)

holiday_name_2023_Prague_1 = {
    "date": [
        "2023-01-01", "2023-01-01", "2023-02-14", "2023-03-08", "2023-04-07",
        "2023-04-10", "2023-05-01", "2023-05-08", "2023-05-14", "2023-06-01",
        "2023-06-18", "2023-07-05", "2023-07-06", "2023-09-28", "2023-10-28",
        "2023-11-17", "2023-12-24", "2023-12-25", "2023-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Restoration of the Czech Independence Day", "St. Valentine's Day",
        "International Women's Day", "Good Friday", "Easter Monday", "Labor Day / May Day",
        "Victory in Europe Day", "Mother's Day", "Children's Day", "Father's Day",
        "Day of Saints Cyril and Methodius", "Jan Hus Day", "St. Wenceslas Day",
        "Independent Czechoslovak State Day", "Struggle for Freedom and Democracy Day",
        "Christmas Eve", "Christmas Day", "St. Stephen's Day"
    ],
    "Country": ["Czech Republic (Prague_1)"] * 19,  # Apply the same value to all rows
    "Year": [2023] * 19
}

holiday_name_2023_Prague_2 = {
    "date": [
        "2023-01-01", "2023-01-01", "2023-02-14", "2023-03-08", "2023-04-07",
        "2023-04-10", "2023-05-01", "2023-05-08", "2023-05-14", "2023-06-01",
        "2023-06-18", "2023-07-05", "2023-07-06", "2023-09-28", "2023-10-28",
        "2023-11-17", "2023-12-24", "2023-12-25", "2023-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Restoration of the Czech Independence Day", "St. Valentine's Day",
        "International Women's Day", "Good Friday", "Easter Monday", "Labor Day / May Day",
        "Victory in Europe Day", "Mother's Day", "Children's Day", "Father's Day",
        "Day of Saints Cyril and Methodius", "Jan Hus Day", "St. Wenceslas Day",
        "Independent Czechoslovak State Day", "Struggle for Freedom and Democracy Day",
        "Christmas Eve", "Christmas Day", "St. Stephen's Day"
    ],
    "Country": ["Czech Republic (Prague_2)"] * 19,  # Apply the same value to all rows
    "Year": [2023] * 19
}

holiday_name_2023_Prague_3 = {
    "date": [
        "2023-01-01", "2023-01-01", "2023-02-14", "2023-03-08", "2023-04-07",
        "2023-04-10", "2023-05-01", "2023-05-08", "2023-05-14", "2023-06-01",
        "2023-06-18", "2023-07-05", "2023-07-06", "2023-09-28", "2023-10-28",
        "2023-11-17", "2023-12-24", "2023-12-25", "2023-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Restoration of the Czech Independence Day", "St. Valentine's Day",
        "International Women's Day", "Good Friday", "Easter Monday", "Labor Day / May Day",
        "Victory in Europe Day", "Mother's Day", "Children's Day", "Father's Day",
        "Day of Saints Cyril and Methodius", "Jan Hus Day", "St. Wenceslas Day",
        "Independent Czechoslovak State Day", "Struggle for Freedom and Democracy Day",
        "Christmas Eve", "Christmas Day", "St. Stephen's Day"
    ],
    "Country": ["Czech Republic (Prague_3)"] * 19,  # Apply the same value to all rows
    "Year": [2023] * 19
}

# Convert to DataFrame
holiday_name_2023_Prague_1_table = pd.DataFrame(holiday_name_2023_Prague_1)
holiday_name_2023_Prague_2_table = pd.DataFrame(holiday_name_2023_Prague_2)
holiday_name_2023_Prague_3_table = pd.DataFrame(holiday_name_2023_Prague_3)

holiday_name_2023_Brno_1 = {
    "date": [
        "2023-01-01", "2023-01-01", "2023-02-14", "2023-03-08", "2023-04-07",
        "2023-04-10", "2023-05-01", "2023-05-08", "2023-05-14", "2023-06-01",
        "2023-06-18", "2023-07-05", "2023-07-06", "2023-09-28", "2023-10-28",
        "2023-11-17", "2023-12-24", "2023-12-25", "2023-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Restoration of the Czech Independence Day", "St. Valentine's Day",
        "International Women's Day", "Good Friday", "Easter Monday", "Labor Day / May Day",
        "Victory in Europe Day", "Mother's Day", "Children's Day", "Father's Day",
        "Day of Saints Cyril and Methodius", "Jan Hus Day", "St. Wenceslas Day",
        "Independent Czechoslovak State Day", "Struggle for Freedom and Democracy Day",
        "Christmas Eve", "Christmas Day", "St. Stephen's Day"
    ],
    "Country": ["Czech Republic (Brno)"] * 19,  # Apply the same value to all rows
    "Year": [2023] * 19
}

# Convert to DataFrame
holiday_name_2023_Brno_1_table = pd.DataFrame(holiday_name_2023_Brno_1) 

holiday_name_2023_Budapest_1 = {
    "date": [
        "2023-01-01", "2023-03-15", "2023-04-07", "2023-04-09", "2023-04-10",
        "2023-05-01", "2023-05-07", "2023-05-28", "2023-05-29", "2023-06-18",
        "2023-08-20", "2023-10-23", "2023-11-01", "2023-12-06", "2023-12-24",
        "2023-12-25", "2023-12-26", "2023-12-31"
    ],
    "holiday_name": [
        "New Year's Day", "1848 Revolution Memorial Day", "Good Friday", "Easter Sunday",
        "Easter Monday", "Labor Day / May Day", "Motherâ€™s Day", "Whit Sunday", "Whit Monday",
        "Fatherâ€™s Day", "Hungary National Day", "1956 Revolution Memorial Day",
        "All Saints' Day", "Saint Nicholas Day", "Christmas Eve", "Christmas Day",
        "Second Day of Christmas", "New Year's Eve"
    ],
    "Country": ["Hungary (Budapest)"] * 18,  # Apply the same value to all rows
    "Year": [2023] * 18
}

# Convert to DataFrame
holiday_name_2023_Budapest_1_table = pd.DataFrame(holiday_name_2023_Budapest_1)


# One Data Frame for 2023
holiday_name_2023 = pd.concat([
    holiday_name_2023_Frankfurt_1_table,
    holiday_name_2023_Munich_1_table,
    holiday_name_2023_Prague_1_table,
    holiday_name_2023_Prague_2_table,
    holiday_name_2023_Prague_3_table,
    holiday_name_2023_Brno_1_table,
    holiday_name_2023_Budapest_1_table
], ignore_index=True)


# Creating Dataframe to fill in holiday_name blank values for year 2024

holiday_name_2024_Frankfurt_1 = {
    "date": [
        "2024-01-01", "2024-01-06", "2024-02-12", "2024-02-13", "2024-02-14",
        "2024-02-14", "2024-03-08", "2024-03-08", "2024-03-24", "2024-03-28",
        "2024-03-29", "2024-03-30", "2024-03-31", "2024-03-31", "2024-04-01",
        "2024-05-01", "2024-05-08", "2024-05-09", "2024-05-09", "2024-05-12",
        "2024-05-19", "2024-05-19", "2024-05-20", "2024-05-30", "2024-08-15",
        "2024-08-15", "2024-09-20", "2024-10-03", "2024-10-31", "2024-10-31",
        "2024-11-01", "2024-11-11", "2024-11-17", "2024-11-20", "2024-11-24",
        "2024-12-01", "2024-12-06", "2024-12-08", "2024-12-15", "2024-12-22",
        "2024-12-25", "2024-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Epiphany (BW, BY, ST)", "Shrove Monday", "Carnival Tuesday",
        "Carnival / Ash Wednesday", "Valentine's Day", "International Women's Day (Most regions)",
        "International Women's Day (Berlin, Mecklenburg-Western Pomerania)", "Palm Sunday",
        "Maundy Thursday (All)", "Good Friday", "Holy Saturday (Many regions)",
        "Easter Sunday (Brandenburg)", "Easter Sunday (Most regions)", "Easter Monday",
        "May Day", "End of World War II (Many regions)", "Fathers' Day", "Ascension Day",
        "Mothers' Day", "Whit Sunday (Brandenburg)", "Whit Sunday (Most regions)",
        "Whit Monday", "Corpus Christi (Many regions)", "Assumption of Mary (Bavaria, Saarland)",
        "Assumption of Mary (Saxony, Thuringia)", "German World Children's Day (Thuringia)",
        "Day of German Unity", "Reformation Day (Most regions)", "Halloween",
        "All Saints' Day (Many regions)", "St. Martin's Day", "National Day of Mourning",
        "Repentance Day (Saxony)", "Sunday of the Dead", "First Advent Sunday",
        "Saint Nicholas Day", "Second Advent Sunday", "Third Advent Sunday",
        "Fourth Advent Sunday", "Christmas Day", "Boxing Day"
    ],
    "Country": ["Germany (Frankfurt)"] * 42,  # Apply the same value to all rows
    "Year": [2024] * 42
}

# Convert to DataFrame
holiday_name_2024_Frankfurt_1_table = pd.DataFrame(holiday_name_2024_Frankfurt_1)

holiday_name_2024_Munich_1 = {
    "date": [
        "2024-01-01", "2024-01-06", "2024-02-12", "2024-02-13", "2024-02-14",
        "2024-02-14", "2024-03-08", "2024-03-08", "2024-03-24", "2024-03-28",
        "2024-03-29", "2024-03-30", "2024-03-31", "2024-03-31", "2024-04-01",
        "2024-05-01", "2024-05-08", "2024-05-09", "2024-05-09", "2024-05-12",
        "2024-05-19", "2024-05-19", "2024-05-20", "2024-05-30", "2024-08-15",
        "2024-08-15", "2024-09-20", "2024-10-03", "2024-10-31", "2024-10-31",
        "2024-11-01", "2024-11-11", "2024-11-17", "2024-11-20", "2024-11-24",
        "2024-12-01", "2024-12-06", "2024-12-08", "2024-12-15", "2024-12-22",
        "2024-12-25", "2024-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Epiphany (BW, BY, ST)", "Shrove Monday", "Carnival Tuesday",
        "Carnival / Ash Wednesday", "Valentine's Day", "International Women's Day (Most regions)",
        "International Women's Day (Berlin, Mecklenburg-Western Pomerania)", "Palm Sunday",
        "Maundy Thursday (All)", "Good Friday", "Holy Saturday (Many regions)",
        "Easter Sunday (Brandenburg)", "Easter Sunday (Most regions)", "Easter Monday",
        "May Day", "End of World War II (Many regions)", "Fathers' Day", "Ascension Day",
        "Mothers' Day", "Whit Sunday (Brandenburg)", "Whit Sunday (Most regions)",
        "Whit Monday", "Corpus Christi (Many regions)", "Assumption of Mary (Bavaria, Saarland)",
        "Assumption of Mary (Saxony, Thuringia)", "German World Children's Day (Thuringia)",
        "Day of German Unity", "Reformation Day (Most regions)", "Halloween",
        "All Saints' Day (Many regions)", "St. Martin's Day", "National Day of Mourning",
        "Repentance Day (Saxony)", "Sunday of the Dead", "First Advent Sunday",
        "Saint Nicholas Day", "Second Advent Sunday", "Third Advent Sunday",
        "Fourth Advent Sunday", "Christmas Day", "Boxing Day"
    ],
    "Country": ["Germany (Munich)"] * 42,  # Apply the same value to all rows
    "Year": [2024] * 42
}

# Convert to DataFrame
holiday_name_2024_Munich_1_table = pd.DataFrame(holiday_name_2024_Munich_1)

holiday_name_2024_Prague_1 = {
    "date": [
        "2024-01-01", "2024-01-01", "2024-02-14", "2024-03-08", "2024-03-29",
        "2024-04-01", "2024-05-01", "2024-05-08", "2024-05-12", "2024-06-01",
        "2024-06-16", "2024-07-05", "2024-07-06", "2024-09-28", "2024-10-28",
        "2024-11-17", "2024-12-24", "2024-12-25", "2024-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Restoration of the Czech Independence Day", "St. Valentine's Day",
        "International Women's Day", "Good Friday", "Easter Monday", "Labor Day / May Day",
        "Victory in Europe Day", "Mother's Day", "Children's Day", "Father's Day",
        "Day of Saints Cyril and Methodius", "Jan Hus Day", "St. Wenceslas Day",
        "Independent Czechoslovak State Day", "Struggle for Freedom and Democracy Day",
        "Christmas Eve", "Christmas Day", "St. Stephen's Day"
    ],
    "Country": ["Czech Republic (Prague_1)"] * 19,  # Apply the same value to all rows
    "Year": [2024] * 19
}

holiday_name_2024_Prague_2 = {
    "date": [
        "2024-01-01", "2024-01-01", "2024-02-14", "2024-03-08", "2024-03-29",
        "2024-04-01", "2024-05-01", "2024-05-08", "2024-05-12", "2024-06-01",
        "2024-06-16", "2024-07-05", "2024-07-06", "2024-09-28", "2024-10-28",
        "2024-11-17", "2024-12-24", "2024-12-25", "2024-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Restoration of the Czech Independence Day", "St. Valentine's Day",
        "International Women's Day", "Good Friday", "Easter Monday", "Labor Day / May Day",
        "Victory in Europe Day", "Mother's Day", "Children's Day", "Father's Day",
        "Day of Saints Cyril and Methodius", "Jan Hus Day", "St. Wenceslas Day",
        "Independent Czechoslovak State Day", "Struggle for Freedom and Democracy Day",
        "Christmas Eve", "Christmas Day", "St. Stephen's Day"
    ],
    "Country": ["Czech Republic (Prague_2)"] * 19,  # Apply the same value to all rows
    "Year": [2024] * 19
}

holiday_name_2024_Prague_3 = {
    "date": [
        "2024-01-01", "2024-01-01", "2024-02-14", "2024-03-08", "2024-03-29",
        "2024-04-01", "2024-05-01", "2024-05-08", "2024-05-12", "2024-06-01",
        "2024-06-16", "2024-07-05", "2024-07-06", "2024-09-28", "2024-10-28",
        "2024-11-17", "2024-12-24", "2024-12-25", "2024-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Restoration of the Czech Independence Day", "St. Valentine's Day",
        "International Women's Day", "Good Friday", "Easter Monday", "Labor Day / May Day",
        "Victory in Europe Day", "Mother's Day", "Children's Day", "Father's Day",
        "Day of Saints Cyril and Methodius", "Jan Hus Day", "St. Wenceslas Day",
        "Independent Czechoslovak State Day", "Struggle for Freedom and Democracy Day",
        "Christmas Eve", "Christmas Day", "St. Stephen's Day"
    ],
    "Country": ["Czech Republic (Prague_3)"] * 19,  # Apply the same value to all rows
    "Year": [2024] * 19
}

# Convert to DataFrame
holiday_name_2024_Prague_1_table = pd.DataFrame(holiday_name_2024_Prague_1)
holiday_name_2024_Prague_2_table = pd.DataFrame(holiday_name_2024_Prague_2)
holiday_name_2024_Prague_3_table = pd.DataFrame(holiday_name_2024_Prague_3)

holiday_name_2024_Brno_1 = {
    "date": [
        "2024-01-01", "2024-01-01", "2024-02-14", "2024-03-08", "2024-03-29",
        "2024-04-01", "2024-05-01", "2024-05-08", "2024-05-12", "2024-06-01",
        "2024-06-16", "2024-07-05", "2024-07-06", "2024-09-28", "2024-10-28",
        "2024-11-17", "2024-12-24", "2024-12-25", "2024-12-26"
    ],
    "holiday_name": [
        "New Year's Day", "Restoration of the Czech Independence Day", "St. Valentine's Day",
        "International Women's Day", "Good Friday", "Easter Monday", "Labor Day / May Day",
        "Victory in Europe Day", "Mother's Day", "Children's Day", "Father's Day",
        "Day of Saints Cyril and Methodius", "Jan Hus Day", "St. Wenceslas Day",
        "Independent Czechoslovak State Day", "Struggle for Freedom and Democracy Day",
        "Christmas Eve", "Christmas Day", "St. Stephen's Day"
    ],
    "Country": ["Czech Republic (Brno)"] * 19,  # Apply the same value to all rows
    "Year": [2024] * 19
}

# Convert to DataFrame
holiday_name_2024_Brno_1_table = pd.DataFrame(holiday_name_2024_Brno_1) 

holiday_name_2024_Budapest_1 = {
    "date": [
        "2024-01-01", "2024-03-15", "2024-03-29", "2024-03-31", "2024-04-01",
        "2024-05-01", "2024-05-05", "2024-05-19", "2024-05-20", "2024-06-16",
        "2024-08-19", "2024-08-20", "2024-10-23", "2024-11-01", "2024-12-06",
        "2024-12-24", "2024-12-25", "2024-12-26", "2024-12-27", "2024-12-31"
    ],
    "holiday_name": [
        "New Year's Day", "1848 Revolution Memorial Day", "Good Friday", "Easter Sunday",
        "Easter Monday", "Labor Day / May Day", "Motherâ€™s Day", "Whit Sunday", "Whit Monday",
        "Fatherâ€™s Day", "Hungary National Day Holiday", "Hungary National Day",
        "1956 Revolution Memorial Day", "All Saints' Day", "Saint Nicholas Day",
        "Christmas Eve", "Christmas Day", "Second Day of Christmas",
        "Christmas Holiday", "New Year's Eve"
    ],
    "Country": ["Hungary (Budapest)"] * 20,  # Apply the same value to all rows
    "Year": [2024] * 20
}

# Convert to DataFrame
holiday_name_2024_Budapest_1_table = pd.DataFrame(holiday_name_2024_Budapest_1)


# One Data Frame for 2024
holiday_name_2024 = pd.concat([
    holiday_name_2024_Frankfurt_1_table,
    holiday_name_2024_Munich_1_table,
    holiday_name_2024_Prague_1_table,
    holiday_name_2024_Prague_2_table,
    holiday_name_2024_Prague_3_table,
    holiday_name_2024_Brno_1_table,
    holiday_name_2024_Budapest_1_table
], ignore_index=True)





# List of all holiday DataFrames (make sure these variables are defined)
holiday_dfs = [holiday_name_2020, holiday_name_2021, holiday_name_2022, holiday_name_2023, holiday_name_2024]

# Concatenate all holiday data into one DataFrame
holiday_name_df = pd.concat(holiday_dfs, ignore_index=True)


# Merge calendar_df with holiday_name_df on 'date' to bring in holiday names
calendar_df = calendar_df.merge(holiday_name_df[['date', 'holiday_name']], on='date', how='left')

# Fill missing values in 'holiday_name' column in calendar_df
calendar_df['holiday_name'] = calendar_df['holiday_name_x'].fillna(calendar_df['holiday_name_y'])

# Drop the duplicate columns created from merging
calendar_df.drop(columns=['holiday_name_x', 'holiday_name_y'], inplace=True)


print (calendar_df.isnull().sum())



calendar_df['holiday_name'] = calendar_df['holiday_name'].fillna("N/A")



train_sales.isnull().sum()
test_sales.isnull().sum()
inventory.isnull().sum()
calendar_df.isnull().sum()



# Define impact scores for holidays in each region
holiday_impact_by_region = {
    "Brno_1": {
        "New Year's Day": 3, "Labour Day": 3, "Jan Hus Day": 2, "Christmas Eve": 3,
        "Restoration Day of the Czech State": 3, "Victory Day": 2, "Saint Wencelas Day": 2,
        "Good Friday": 2, "VE Day": 2, "Christmas Day": 3, "Easter Monday": 3,
        "Saint Cyril and Methodius Day": 2, "Struggle for Freedom and Democracy Day": 2
    },
    
    "Budapest_1": {
        "New Year's Day": 3, "Easter Monday": 3, "State Foundation Day": 3, "Christmas Day": 3,
        "Memorial Day of 1848 Revolution": 2, "Labour Day": 3, "1956 Memorial Day": 2,
        "Good Friday": 2, "All Saint's Day": 2, "Easter": 3, "Whit Monday": 3
    },
    
    "Munich_1": {
        "New Year's Day": 3, "Labour Day": 3, "Ascension Day": 2, "Assumption of Mary": 2, 
        "Epiphany": 2, "German Unity Day": 3, "Good Friday": 2, "Whit Monday": 3, 
        "All Saint's Day": 2, "Easter Monday": 3, "Corpus Christi": 2,
        "Christmas Day": 3
    },

    "Prague_1": {
        "New Year's Day": 3, "Labour Day": 3, "Jan Hus Day": 2, "Christmas Eve": 3,
        "Restoration Day of the Czech State": 3, "Victory Day": 2, "Saint Wencelas Day": 2,
        "Good Friday": 2, "VE Day": 2, "Christmas Day": 3, "Easter Monday": 3,
        "Saint Cyril and Methodius Day": 2, "Struggle for Freedom and Democracy Day": 2
    },

    "Prague_2": {
        "New Year's Day": 3, "Labour Day": 3, "Jan Hus Day": 2, "Christmas Eve": 3,
        "Restoration Day of the Czech State": 3, "Victory Day": 2, "Saint Wencelas Day": 2,
        "Good Friday": 2, "VE Day": 2, "Christmas Day": 3, "Easter Monday": 3,
        "Saint Cyril and Methodius Day": 2, "Struggle for Freedom and Democracy Day": 2
    },

    "Prague_3": {
        "New Year's Day": 3, "Labour Day": 3, "Jan Hus Day": 2, "Christmas Eve": 3,
        "Restoration Day of the Czech State": 3, "Victory Day": 2, "Saint Wencelas Day": 2,
        "Good Friday": 2, "VE Day": 2, "Christmas Day": 3, "Easter Monday": 3,
        "Saint Cyril and Methodius Day": 2, "Struggle for Freedom and Democracy Day": 2
    },

    "Frankfurt_1": {
        "New Year's Day": 3, "Good Friday": 2, "Ascension Day": 2, "Christmas Day": 3,
        "Whit Monday": 3, "Easter Monday": 3, "Labour Day": 3, "Corpus Christi": 2,
        "German Unity Day": 3
    }
}

## We will use a scoring system:

# High Impact (3) â†’ Public holidays where most businesses and schools are closed.
# Moderate Impact (2) â†’ Holidays where some businesses close but not all.
# Low Impact (1) â†’ Observances that may not impact business hours much.


# Function to get holiday impact score based on region
def get_holiday_impact(row):
    region = row["warehouse"]  # Assuming region info is stored in this column
    holiday = row["holiday_name"]
    
    if region in holiday_impact_by_region and holiday in holiday_impact_by_region[region]:
        return holiday_impact_by_region[region][holiday]
    else:
        return 0  # Default impact if holiday is not in the predefined list

# Apply function to calendar_df
calendar_df["holiday_impact"] = calendar_df.apply(get_holiday_impact, axis=1)

# Ensure the column is integer type
calendar_df["holiday_impact"] = calendar_df["holiday_impact"].astype(int)


# List of all discount columns
discount_cols = ['type_0_discount', 'type_1_discount',
       'type_2_discount', 'type_3_discount', 'type_4_discount',
       'type_5_discount', 'type_6_discount']
# Replace negative values with 0 (since negative means no discount)
train_sales[discount_cols] = train_sales[discount_cols].applymap(lambda x: max(0, x))

# Get the highest discount for each row
train_sales['final_discount'] = train_sales[discount_cols].max(axis=1)



# List of all discount columns
discount_cols = ['type_0_discount', 'type_1_discount',
       'type_2_discount', 'type_3_discount', 'type_4_discount',
       'type_5_discount', 'type_6_discount']
# Replace negative values with 0 (since negative means no discount)
test_sales[discount_cols] = test_sales[discount_cols].applymap(lambda x: max(0, x))

# Get the highest discount for each row
test_sales['final_discount'] = test_sales[discount_cols].max(axis=1)


# Convert the date column to datetime format
train_sales['date'] = pd.to_datetime(train_sales['date'])
test_sales['date'] = pd.to_datetime(test_sales['date'])



calendar_df['date'] = pd.to_datetime(calendar_df['date'])



calendar_df.info()


# Merge sales_train with calendar_df
sales_train_df = train_sales.merge(calendar_df, on='date', how='left')

# Merge sales_test with calendar_df
sales_test_df = test_sales.merge(calendar_df, on='date', how='left')


# Merge sales_train with inventory_df
sales_train_df = sales_train_df.merge(inventory, on='unique_id', how='left')

# Merge sales_test with inventory_df
sales_test_df = sales_test_df.merge(inventory, on='unique_id', how='left')


sales_test_df = sales_test_df.merge(test_weights, on='unique_id', how='left')



sales_train_df = sales_train_df.merge(test_weights, on='unique_id', how='left')



merged_sales_test_df =  sales_test_df
merged_sales_train_df = sales_train_df


# Convert date to datetime and extract components
for df in [sales_train_df, sales_test_df]:
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def univariate_analysis(df):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    print("ğŸ”¹ Dataset Overview:")
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
    print(df.info())
    
    # ğŸ”¹ Numerical Feature Analysis
    for col in numerical_cols:
        print(f"\nğŸ“Œ Analyzing Numerical Column: {col}")
        print(df[col].describe())

        # ğŸ”¸ Histogram & KDE Plot (Check Distribution)
        plt.figure(figsize=(12, 5))
        sns.histplot(df[col], bins=50, kde=True, color='blue')
        plt.title(f"Distribution of {col}")
        plt.show()
        
        # ğŸ”¸ Boxplot (Check Outliers)
        plt.figure(figsize=(10, 3))
        sns.boxplot(x=df[col], color='red')
        plt.title(f"Boxplot of {col}")
        plt.show()
        
        # ğŸ”¸ Outlier Detection using IQR Method
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"ğŸ”º Outliers in {col}: {len(outliers)} rows")

        # ğŸ”¸ Z-Score Outlier Detection (Optional)
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        z_outliers = df[z_scores > 3]
        print(f"âš ï¸� Extreme Outliers (Z-Score > 3) in {col}: {len(z_outliers)} rows")

    # ğŸ”¹ Categorical Feature Analysis
    for col in categorical_cols:
        print(f"\nğŸ“Œ Analyzing Categorical Column: {col}")
        print(df[col].value_counts())

        # ğŸ”¸ Bar Plot
        plt.figure(figsize=(12, 5))
        sns.countplot(y=df[col], order=df[col].value_counts().index, palette='viridis')
        plt.title(f"Frequency Distribution of {col}")
        plt.show()

# Run Univariate Analysis
univariate_analysis(merged_sales_train_df)



sales_train_df.columns


# Define target variable
target = 'sales'  # In train dataset only

# Separate features and target
X_train = sales_train_df.drop(columns=['availability','total_orders', 'sales',
       'sell_price_main', 'availability', 'type_0_discount', 'type_1_discount',
       'type_2_discount', 'type_3_discount', 'type_4_discount',
       'type_5_discount', 'type_6_discount','warehouse_x', 'warehouse_y','sales', 'date', 'unique_id', 'name'])
y_train = sales_train_df[target]


X_train.columns


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Identify categorical columns
categorical_cols = ['holiday_name', 'L1_category_name_en', 'L2_category_name_en', 
                    'L3_category_name_en', 'L4_category_name_en', 'warehouse']

# Apply Label Encoding to categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))  # Convert to string first to avoid NaNs
    label_encoders[col] = le  # Store encoder for later use if needed

# Verify conversion
print(X_train.dtypes)



from xgboost import XGBRegressor

# Initialize XGBoost model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

X_train_sample = X_train.sample(frac=0.5, random_state=42)  # Use 50% of the data
y_train_sample = y_train.loc[X_train_sample.index]  # Get corresponding target values

xgb_model.fit(X_train_sample, y_train_sample)  # Train on the reduced dataset

# Predict on train data
y_pred_xgb_sample = xgb_model.predict(X_train_sample)



X_test = sales_test_df.drop(columns=['total_orders',
       'sell_price_main',  'type_0_discount', 'type_1_discount',
       'type_2_discount', 'type_3_discount', 'type_4_discount',
       'type_5_discount', 'type_6_discount','warehouse_x', 'warehouse_y', 'date', 'unique_id', 'name'])



import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Identify categorical columns
categorical_cols = ['holiday_name', 'L1_category_name_en', 'L2_category_name_en', 
                    'L3_category_name_en', 'L4_category_name_en', 'warehouse']

# Apply Label Encoding to categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_test[col] = le.fit_transform(X_test[col].astype(str))  # Convert to string first to avoid NaNs
    label_encoders[col] = le  # Store encoder for later use if needed

# Verify conversion
print(X_test.dtypes)



# Check column data types
print(X_test.dtypes)

# Identify non-numeric columns
non_numeric_cols = X_test.select_dtypes(include=['object']).columns
print(f"Non-numeric columns: {list(non_numeric_cols)}")


sales_test_df['sales_hat']=xgb_model.predict(X_test)


# Ensure 'id' column is correctly formatted
sales_test_df['id'] = sales_test_df['unique_id'].astype(str) + "_" + sales_test_df['date'].astype(str)

# Keep only the required columns for submission
submission_df = sales_test_df[['id', 'sales_hat']]

submission_df = submission_df.drop_duplicates()

# Display the first few rows
print(submission_df)


submission_df.to_csv("submission.csv", index=False)


