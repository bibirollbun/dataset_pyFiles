import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered",
    category=RuntimeWarning
)

from typing import List

from IPython.display import display, Markdown


root = "/kaggle/input/china-real-estate-demand-prediction/train/"

tables = {
    # Auxilary Tables with index as commented
    "cityIndex": root + "city_indexes.csv", # [city_indicator_data_year]
    "cityIndexSearch": root + "city_search_index.csv", # [month, keyword, source]

    # Main Tables with, index columns [month, sector] 
    "landT": root + "land_transactions.csv",
    "landTNS": root + "land_transactions_nearby_sectors.csv",
    "newHouseT": root + "new_house_transactions.csv",
    "newHouseTNS": root + "new_house_transactions_nearby_sectors.csv",
    "preHouseT": root + "pre_owned_house_transactions.csv",
    "preHouseTNS": root + "pre_owned_house_transactions_nearby_sectors.csv",

    # Auxilary table with only general sector info, index [sector]
    "sectorPOI": root + "sector_POI.csv"
}

for key in tables.keys():
    tables[key] = pd.read_csv(tables[key])


for _, _, files in os.walk(root):
    pass
assert len(files) == len(tables.keys()), "Some files in the root folder have not been read"


tables["ss"] = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/sample_submission.csv")


MONTHS12STR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
assert len(MONTHS12STR) == 12, "Missing some months"

# Utility Functions
def month_STR2INT(month: str):
    return MONTHS12STR.index(month) + 1
def month_INT2STR(month: int):
    return MONTHS12STR[month-1]

MONTHS12INT = [month_STR2INT(i) for i in MONTHS12STR]


# Get new month, sector columns for Test CSV
test = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/test.csv")
yearMonths, sectors = [], []
years, months = [], []
for yearMonth, sector in test["id"].str.split('_'):
    yearMonth = yearMonth.replace(" ", "-") # same format as train: 2019 Jan -> 2019-Jan
    yearMonths.append(yearMonth)
    sectors.append(int(sector.split(' ')[-1]))

    year, month = yearMonth.split('-')
    years.append(int(year))
    months.append(month_STR2INT(str(month))) # Replace tri-letteral month with int (1-12)  
    

test["yearMonth"] = yearMonths
test["sector"] = sectors
test["year"] = years
test["month"] = months

test.head()


test["sector"].unique()


test["year"].unique()


SECTORS = test["sector"].unique() # 1-96

mainTables = ['landT', 'landTNS', 'newHouseT', 'newHouseTNS', 'preHouseT', 'preHouseTNS']
for tableName in mainTables:
    yearMonths, sectors = [], []
    years, months = [], []
    table = tables[tableName]

    # yearMonth: year: INT(4-digit) and month: TRI-letteral to INT(1-12) separated
    yearMonths = table["month"].to_list()
    for yearMonth in yearMonths:
        year, month = yearMonth.split('-')
        year = int(year)
        month = month_STR2INT(str(month))
        years.append(year)
        months.append(month)
    table["year"] = years
    table["yearMonth"] = table["month"]
    table["month"] = months

    # sector: STR to INT
    for sector in table["sector"].to_list():
        sector = int(sector.split(' ')[1])
        sectors.append(sector)
    table["sector"] = sectors


# Just an example
tables["newHouseT"].head()


# Train + Val years
mainTableYears = set()
for tableName in mainTables:
    table = tables[tableName]
    mainTableYears = mainTableYears.union(table["year"].unique())
mainTableYears


# Missing years for each main table
missingYears = dict()
for tableName in mainTables:
    missingYears[tableName] = []
    table = tables[tableName]
    years = table["year"].unique()
    for year in mainTableYears:
        if year not in years:
            missingYears[tableName].append(year)

missingYears


YEARS = mainTableYears

# Missing yearMonth for each main table 
missingYearMonth = dict() # key: tableName, val: list of tuple (missingYearMonth pair)
for tableName in mainTables:
    table = tables[tableName]
    missingYearMonth[tableName] = []
    years, months = table["year"], table["month"]
    yearMonths = [(y, m) for (y, m) in zip(years, months)]
    for year in YEARS:
        for month in MONTHS12INT:
            if (year, month) not in yearMonths:
                missingYearMonth[tableName].append((year, month))

# No data is present for this yearMonths
missingYearMonth


# The missing year month pairs are the same for every table -> Oops, these are yearMonths after the train period
months = set(missingYearMonth["landT"])
for tableName in missingYearMonth.keys():
    months = months.union(set(missingYearMonth[tableName]))

assert set(missingYearMonth["landT"]) == months, f"Missing yearMonth differs from table to table."

# So for train period entire for all sectors [year, month] is not missing but [yearMonth, month, sector] might be missing
# i.e., at least one sector info for any given [year, month] 


# Sectors missing any entry per main table
sectorsNoInfo = dict()
for tableName in mainTables:
    table = tables[tableName]
    sectorsNoInfo[tableName] = []
    sectors = table["sector"].unique()
    for sector in SECTORS:
        if sector not in sectors:
            sectorsNoInfo[tableName].append(sector)

sectorsNoInfo


# Missing Triplets
missingTripletsPerTable = dict()
for tableName in mainTables:
    allTriplets = [(s, y, m) for s in SECTORS for y in YEARS for m in MONTHS12INT] # Assume all triplets are missing
    missingTripletsPerTable[tableName] = []
    for idx in table.index:
        row = table.iloc[idx]
        s, y, m = row["sector"], row["year"], row["month"]
        allTriplets.remove((s, y, m)) # Remove existing triplets
    missingTripletsPerTable[tableName] = allTriplets

missingTripletsPerTable["newHouseT"][:10]


# Get all unique columns
allCols = set()
for tableName in mainTables:
    table = tables[tableName]
    cols = set(table.columns)
    allCols |= cols

# Find which columns exist in which tables
colShared = dict()
for col in allCols:
    colShared[col] = []
    for tableName in mainTables:
        table = tables[tableName]
        if col in table.columns:
            colShared[col].append(tableName)

# Remove columns that are unique to a table
moreThanOne = dict()
for col in colShared:
    if len(colShared[col]) > 1:
        moreThanOne[col] = colShared[col]

# We can see that no table has the same columns
moreThanOne


# This can be further developed to detect perfectly, linearly dependent columns to reduce the number of features
for tableName in mainTables: 
    table = tables[tableName]
    cols = table.columns.to_list()
    for col in ["month", "year", "sector", "yearMonth"]:
        cols.remove(col)

    display(Markdown(f"### {tableName}"))
    display(table[cols].corr())

    print("="*75)
    print("\n\n")


# OUTER JOIN -> entry from any table
train = pd.DataFrame(
    columns=["sector", "year", "month", "yearMonth"]
)
for tableName in mainTables:
    table = tables[tableName]
    train = pd.merge(
        left=train,
        right=table,
        on=["sector", "year", "month", "yearMonth"], 
        how="outer")


train.describe()


train.isna().sum()


train.to_csv("train.csv", index=False)

