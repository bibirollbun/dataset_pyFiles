import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
from graphviz import Digraph

import warnings
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered",
    category=RuntimeWarning
)



train_df = None
test_df = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/test.csv")

city_I = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/city_indexes.csv")
city_SI = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/city_search_index.csv")

land_T = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/land_transactions.csv")
land_TNS = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/land_transactions_nearby_sectors.csv")

new_house_T = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv")
new_house_TNS = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv")

pre_house_T = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions.csv")
pre_house_TNS = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions_nearby_sectors.csv")

sector_POI = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/sector_POI.csv")
ss = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/sample_submission.csv")

# Put all DataFrames in a dict
tables = {
    "test_df": test_df,
    "city_I": city_I,
    "city_SI": city_SI,
    "land_T": land_T,
    "land_TNS": land_TNS,
    "new_house_T": new_house_T,
    "new_house_TNS": new_house_TNS,
    "pre_house_T": pre_house_T,
    "pre_house_TNS": pre_house_TNS,
    "sector_POI": sector_POI,
    "ss": ss
}



assert set(ss["id"]) == set(test_df["id"]), "Mismatch in ids in ss and test_df"
assert set(ss.columns) == set(test_df.columns), "Mismatch in columns in ss and test_df"

ss.head() # test_df is the same as this one


# Premise: Transactions are a proper subset of Transactions Nearby Sectors

def is_t2_a_subset_of_t1(t1: pd.DataFrame, t2: pd.DataFrame):
    """
    Checks whether the [month, sector] of t1 is a superset of t2
    """
    for month in set(t1["month"].unique()).union(set(t2["month"].unique())):
        month_idx = t1["month"] == month
        t1_month = t1[month_idx]

        month_idx = t2["month"] == month
        t2_month = t2[month_idx]

        s1 = set(t1_month["sector"].unique())
        s2 = set(t2_month["sector"].unique())

        if (diff := s2-s1):
            return False
    return True
    
for t1, t2 in zip(
        ["land_T", "new_house_T", "pre_house_T"], 
        ["land_TNS", "new_house_TNS", "pre_house_TNS"]):
    if not is_t2_a_subset_of_t1(tables[t1], tables[t2]):
        print(f"This pair does not hold the premise: {t1}, {t2}")


t1, t2 = tables["pre_house_T"], tables["pre_house_TNS"]

all_months = set(t1["month"].unique()).union(set(t2["month"].unique()))
erroric_months = []
erroric_sectors = set()

for month in all_months:
    month_idx = t1["month"] == month
    t1_month = t1[month_idx]

    month_idx = t2["month"] == month
    t2_month = t2[month_idx]

    s1 = set(t1_month["sector"].unique())
    s2 = set(t2_month["sector"].unique())

    if (diff := s1-s2):
        # Uncomment this to see that it is the same sectors that are problematic for all months -> There might have been data discrepancy, i.e.
        # print(diff)
        erroric_sectors = erroric_sectors.union(diff)
        erroric_months.append(month)
        
print(f"Count of Erroric Months: {len(erroric_months)} out of {len(all_months)}") # No house was pre-owned in these sectors, WOW!
print(f"Erroric Sectors: {erroric_sectors}")

# These sectors could be government owned or special where owning is illegal


def missing_months(df: pd.DataFrame, months):
    """
    Params:
        months: list(2019-Jan etc.), start and end of df. returns which year_months are missing out of this list.
    """
    sectors = ["sector "+str(i) for i in range(1, 96+1)]
    missing_months_per_sector = {}
    for sector in sectors:
        idx = df["sector"] == sector
        data = df[idx]
        missing_months = set(months) - set(data["month"])
        missing_months_per_sector[sector] = missing_months
    return missing_months_per_sector
        
def get_months(min_year_month, max_year_month):
    """
    Returns the month range in the format Year[4]:Month[3:] (same format as new_house_T["month"]).
    The inputs must be the begining and end of the consecutive range. For example: 2019-Jan to 2024-Feb
    """
    months = []
    months_12 = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    assert len(months_12) == 12, "There are 12 months in a year"

    min_year = int(min_year_month.split("-")[0])
    max_year = int(max_year_month.split("-")[0])

    min_month = min_year_month.split("-")[1]
    max_month = max_year_month.split("-")[1]
    
    for year in range(min_year, max_year+1):
        for month in months_12:
            if months_12.index(month) < months_12.index(min_month) and year == min_year:
                continue
            if month == max_month and year == max_year:
                break
            year_month = str(year) + '-' + str(month)
            months.append(year_month)    
    return months


min_year_month = "2019-Jan"
max_year_month = "2025-May"
months = get_months(min_year_month, max_year_month)

missingMonths = missing_months(new_house_T, months)


groups = []
for sector in missingMonths:
    months = missingMonths[sector]
    if set(months) not in groups:
        groups.append(set(months))

# Too many various missing months from sector to sector
# groups


def insert_missing_rows(df, missingMonths, val=0):
    """
    Insert [val]*column_size rows for missing months in each sector
    
    Params:
        df: pd.DataFrame
            Must contain columns: ["sector", "month", ...]
        missingMonths: dict[str, list]
            Keys = sectors, values = list of missing months for that sector
        val: int
            Replace the missing month's entries with val
    """
    
    new_rows = []

    sectors = ["sector "+str(i) for i in range(1, 96+1)]
    for sector in sectors:
        if sector not in df["sector"].to_list():
            print(f"this sector is missing for all months: {sector}")
    for sector in sectors:
        for month in missingMonths.get(sector, []): 
            row = {col: val for col in df.columns}
            row["sector"] = sector
            row["month"] = month
            new_rows.append(row)

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    return df


tables.keys()


city_SI.head()


# Min Max year month
min_year_month = "2019-Jan"
max_year_month = "2025-May"

# Replace int columns
for df_name in ["new_house_T", "new_house_TNS", "pre_house_T", "pre_house_TNS", "land_T", "land_TNS"]:
    df = tables[df_name]
    
    months = get_months(min_year_month, max_year_month)
    missingMonths = missing_months(df, months)

    df = insert_missing_rows(df, missingMonths, 0)
    df = df.fillna(0) # The datasets have missing values as well.
    df.to_csv(df_name+".csv", index=False)


# df_string_columns


# sectors = new_house_T["sector"].unique()

# for sector in sectors:
#     sector_idx = new_house_T["sector"] == sector
#     sector_data = new_house_T[sector_idx]

#     Y = sector_data.drop(columns=["sector", "month"])
#     x_ticks = sector_data["month"]

#     fig, ax = plt.subplots(figsize=(12.8, 6))

#     for col in Y.columns:
#         ax.plot(x_ticks, Y[col], label=col)

#     ax.set_xticks(x_ticks)
#     ax.set_xticklabels(x_ticks, rotation=90, fontsize=5)
#     ax.set_ylabel("Values")
#     ax.set_ylim(0, 300000)
#     ax.legend()
#     ax.set_title(f"new_house_T_{sector}")

#     plt.tight_layout()
#     fig.savefig(f"new_house_T_{sector}".replace(" ", "_") + ".jpg",
#                 dpi=300, bbox_inches="tight")
#     plt.close(fig)  # close to free memory



# new_house_T["month"].unique() 


# dates = []
# for x in test_df["id"]:
#     month = x.split('_')[0]
#     dates.append(month)
# dates = set(dates)
# sorted(dates)

