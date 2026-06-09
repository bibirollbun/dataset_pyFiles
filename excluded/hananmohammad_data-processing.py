# Required libraries
import pandas as pd                # Used for handling data like tables, CSV files, etc.
import numpy as np                # Used for mathematical and numerical operations
import scipy.stats                # Used for doing statistics (like mean, mode, correlation)
from sklearn.preprocessing import LabelEncoder  # Used to change text categories into numbers (for machine learning)
import matplotlib.pyplot as plt   # Used to draw charts and graphs

# Show all columns when printing DataFrames
pd.set_option("display.max_columns", None)  # Makes sure all columns show up when we print a DataFrame

# To ignore unnecessary warning messages in output
import warnings
warnings.filterwarnings("ignore")  # This hides warning messages to keep the output clean

# Garbage collector - helps manage memory
import gc
gc.enable()  # Turns on automatic memory cleanup to free unused memory



# Function to check how many values are missing in each column


def count_missings(data):
    # Count total missing values in each column and sort them in descending order
    total = data.isnull().sum().sort_values(ascending = False)
    
    # Calculate the percentage of missing values in each column
    percent = (data.isnull().sum() / data.isnull().count() * 100).sort_values(ascending = False)
    
    # Combine total and percent into one table
    table = pd.concat([total, percent], axis = 1, keys = ["Total", "Percent"])
    
    # Keep only the columns that have missing values
    table = table[table["Total"] > 0]
    
    # Return the final table showing total and percentage of missing values
    return table



# Function to convert days into months (or years) and handle negative values

def convert_days(data, features, t = 12, rounding = True, replace = False):
    # Loop through each feature (column) that we want to convert
    for var in features:
        if replace == True:
            # If we want to replace the original column with the converted values
            if rounding == True:
                # If rounding is True, convert and round the result
                data[var] = round(-data[var]/t)
            else:
                # If not rounding, just convert without rounding
                data[var] = -data[var]/t
            
            # Set any negative converted values to None (missing)
            data[var][data[var] < 0] = None
        else:
            # If we don't want to replace, create a new column with converted values
            if rounding == True:
                data["CONVERTED_" + str(var)] = round(-data[var]/t)
            else:
                data["CONVERTED_" + str(var)] = -data[var]/t
            
            # Set any negative converted values in new column to None (missing)
            data["CONVERTED_" + str(var)][data["CONVERTED_" + str(var)] < 0] = None
    
    # Return the updated DataFrame
    return data



# Function to create logarithmic values for selected columns
def create_logs(data, features, replace = False):
    # Loop through each feature (column) we want to apply log to
    for var in features:
        if replace == True:
            # If replacing original column, apply log and keep it in the same column
            data[var] = np.log(data[var].abs() + 1)
        else:
            # If not replacing, create a new column with log values
            data["LOG_" + str(var)] = np.log(data[var].abs() + 1)      
    
    # Return the updated DataFrame
    return data



# Function to create flag columns for missing values (1 if missing, 0 if not)
def create_null_flags(data, features = None):
    # If no specific columns are given, use all columns
    if features == None:
        features = data.columns
    
    # Loop through each feature (column)
    for var in features:
        # Create a 1 (True) or 0 (False) for missing values and convert to number
        num_null = data[var].isnull() + 0
        
        # Only add a new flag column if there are actually missing values
        if num_null.sum() > 0:
            data["ISNULL_" + str(var)] = num_null
    
    # Return the updated DataFrame
    return data



# Function to handle categorical (text) columns using label or dummy encoding
def treat_factors(data, method = "label"):
    
    # Label encoding (converts text to numbers)
    if method == "label":
        # Find all columns with text (object type)
        factors = [f for f in data.columns if data[f].dtype == "object"]
        for var in factors:
            # Replace text values with numbers (e.g., 'Yes', 'No' → 0, 1)
            data[var], _ = pd.factorize(data[var])
        
    # Dummy encoding (creates separate columns for each category)
    if method == "dummy":
        # Convert text columns to dummy variables (one-hot encoding)
        data = pd.get_dummies(data, drop_first = True)
    
    # Return the updated dataset
    return data



# Function to compute accept/reject ratios based on previous applications
def compute_accept_reject_ratio(data, lags = [1, 3, 5]):
    
    # Get relevant columns from previous applications
    dec_prev = data[["SK_ID_CURR", "SK_ID_PREV", "DAYS_DECISION", "NAME_CONTRACT_STATUS"]]
    
    # Convert days to positive values
    dec_prev["DAYS_DECISION"] = -dec_prev["DAYS_DECISION"]
    
    # Sort by customer ID and decision day (most recent first)
    dec_prev = dec_prev.sort_values(by = ["SK_ID_CURR", "DAYS_DECISION"])
    
    # Convert contract status into separate columns (Approved, Refused, etc.)
    dec_prev = pd.get_dummies(dec_prev)
     
    # Loop through each lag value (e.g., 1, 3, 5 most recent applications)
    for t in lags:
        
        # --- Acceptance Ratios ---
        # Take the first 't' rows for each customer with approval info
        tmp = dec_prev[["SK_ID_CURR", "NAME_CONTRACT_STATUS_Approved"]].groupby(["SK_ID_CURR"]).head(1)
        
        # Calculate the mean approval rate
        tmp = tmp.groupby(["SK_ID_CURR"], as_index = False).mean()
        
        # Rename column to show it is the approval ratio
        tmp.columns = ["SK_ID_CURR", "APPROVE_RATIO_" + str(t)]
        
        # Merge back to main data
        data = data.merge(tmp, how = "left", on = "SK_ID_CURR")
        
        # --- Rejection Ratios ---
        # Take the first 't' rows for each customer with refusal info
        tmp = dec_prev[["SK_ID_CURR", "NAME_CONTRACT_STATUS_Refused"]].groupby(["SK_ID_CURR"]).head(1)
        
        # Calculate the mean rejection rate
        tmp = tmp.groupby(["SK_ID_CURR"], as_index = False).mean()
        
        # Rename column to show it is the rejection ratio
        tmp.columns = ["SK_ID_CURR", "REJECT_RATIO_" + str(t)]
        
        # Merge back to main data
        data = data.merge(tmp, how = "left", on = "SK_ID_CURR")
        
    # Return the final dataset with added ratios
    return data



# Function to aggregate numeric and categorical (factor) data by group
def aggregate_data(data, id_var, label = None):
    
    # Separate features
    
    # Display info message
    print("- Preparing the dataset...")

    # Find all categorical columns (object types)
    data_factors = [f for f in data.columns if data[f].dtype == "object"]
    
    # Split data into numeric and categorical subsets
    num_data = data[list(set(data.columns) - set(data_factors))]
    fac_data = data[[id_var] + data_factors]
    
    # Count how many categorical and numeric columns there are
    num_facs = fac_data.shape[1] - 1
    num_nums = num_data.shape[1] - 1
    print("- Extracted %.0f factors and %.0f numerics..." % (num_facs, num_nums))

    ##### Aggregation

    # Aggregate numeric features
    if (num_nums > 0):
        print("- Aggregating numeric features...")
        num_data = num_data.groupby(id_var).agg(["mean", "std", "min", "max"])
        # Rename columns (e.g., "AMT_income_mean")
        num_data.columns = ["_".join(col).strip() for col in num_data.columns.values]
        num_data = num_data.sort_index()

    # Aggregate categorical (factor) features
    if (num_facs > 0):
        print("- Aggregating factor features...")
        fac_data = fac_data.groupby(id_var).agg([
            ("mode", lambda x: scipy.stats.mode(x)[0][0]),    # Most frequent value
            ("unique", lambda x: x.nunique())                  # Count of unique values
        ])
        # Rename columns (e.g., "NAME_CONTRACT_MODE")
        fac_data.columns = ["_".join(col).strip() for col in fac_data.columns.values]
        fac_data = fac_data.sort_index()

    # Merger

    # Combine both numeric and factor data
    if ((num_facs > 0) & (num_nums > 0)):
        agg_data = pd.concat([num_data, fac_data], axis = 1)
    
    # If only factor data exists
    if ((num_facs > 0) & (num_nums == 0)):
        agg_data = fac_data
        
    # If only numeric data exists
    if ((num_facs == 0) & (num_nums > 0)):
        agg_data = num_data

    ##### Last steps

    # Add a label prefix to all column names if label is provided
    if label != None:
        agg_data.columns = [label + "_" + str(col) for col in agg_data.columns]
    
    # Optionally fill missing standard deviations with 0 (commented out for now)
    # stdevs = agg_data.filter(like = "_std").columns
    # for var in stdevs:
    #     agg_data[var].fillna(0, inplace = True)

    # Show final shape of the aggregated data
    print("- Final dimensions:", agg_data.shape)
    
    # Return the final dataset
    return agg_data



import os

# Step: List all files and folders available in the Kaggle input directory

# Get the list of all files and folders in the /kaggle/input/ directory
input_items = os.listdir("/kaggle/input/")

# Print message
print("Available files and folders in /kaggle/input/:")

# Show the list of files and folders
print(input_items)



# Set the base path where all the data files are located
base_path = "/kaggle/input/home-credit-default-risk/"

# Load datasets from CSV files
train = pd.read_csv(base_path + "application_train.csv")       # Main training dataset
test = pd.read_csv(base_path + "application_test.csv")         # Test dataset (without target variable)
buro = pd.read_csv(base_path + "bureau.csv")                   # Bureau credit history
bbal = pd.read_csv(base_path + "bureau_balance.csv")           # Monthly updates for bureau loans
prev = pd.read_csv(base_path + "previous_application.csv")     # Previous loan applications
inst = pd.read_csv(base_path + "installments_payments.csv")    # Installment payments for previous loans
poca = pd.read_csv(base_path + "POS_CASH_balance.csv")         # Point-of-sale and cash loan balance
card = pd.read_csv(base_path + "credit_card_balance.csv")      # Credit card balances

# Confirm data has been loaded
print("✅ Data Loaded Successfully!")



# Extract the target variable (what we want to predict) from the training data
y = train[["SK_ID_CURR", "TARGET"]]

# Remove the TARGET column from the training data (so it's not used as a feature)
del train["TARGET"]



# Combine the training and test datasets into one (for easier processing)
appl = pd.concat([train, test])

# Delete the original train and test variables to save memory
del train, test



# Feature engineering

# Create income-based ratios
appl["CREDIT_BY_INCOME"]      = appl["AMT_CREDIT"] / appl["AMT_INCOME_TOTAL"]       # Credit amount vs income
appl["ANNUITY_BY_INCOME"]     = appl["AMT_ANNUITY"] / appl["AMT_INCOME_TOTAL"]      # Annuity amount vs income
appl["GOODS_PRICE_BY_INCOME"] = appl["AMT_GOODS_PRICE"] / appl["AMT_INCOME_TOTAL"]  # Goods price vs income
appl["INCOME_PER_PERSON"]     = appl["AMT_INCOME_TOTAL"] / appl["CNT_FAM_MEMBERS"]  # Income per family member

# Career duration as a percentage of life
appl["PERCENT_WORKED"] = appl["DAYS_EMPLOYED"] / appl["DAYS_BIRTH"]
appl["PERCENT_WORKED"][appl["PERCENT_WORKED"] < 0] = None  # Remove negative ratios

# Estimate number of adults and children ratio
appl["CNT_ADULTS"] = appl["CNT_FAM_MEMBERS"] - appl["CNT_CHILDREN"]
appl['CHILDREN_RATIO'] = appl['CNT_CHILDREN'] / appl['CNT_FAM_MEMBERS']

# Estimate how long it would take to repay the loan
appl['ANNUITY LENGTH'] = appl['AMT_CREDIT'] / appl['AMT_ANNUITY']

# Average of external score sources
appl["EXT_SOURCE_MEAN"] = appl[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis = 1)

# Count how many external sources are available (not null)
appl["NUM_EXT_SOURCES"] = 3 - (appl["EXT_SOURCE_1"].isnull().astype(int) +
                               appl["EXT_SOURCE_2"].isnull().astype(int) +
                               appl["EXT_SOURCE_3"].isnull().astype(int))

# Count how many document flags are marked as 1
doc_vars = ["FLAG_DOCUMENT_2",  "FLAG_DOCUMENT_3",  "FLAG_DOCUMENT_4",  "FLAG_DOCUMENT_5",  "FLAG_DOCUMENT_6",
            "FLAG_DOCUMENT_7",  "FLAG_DOCUMENT_8",  "FLAG_DOCUMENT_9",  "FLAG_DOCUMENT_10", "FLAG_DOCUMENT_11",
            "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13", "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16",
            "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19", "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21"]
appl["NUM_DOCUMENTS"] = appl[doc_vars].sum(axis = 1)

# Identify if the application was made on a weekend or working day
appl["DAY_APPR_PROCESS_START"] = "Working day"
appl["DAY_APPR_PROCESS_START"][(appl["WEEKDAY_APPR_PROCESS_START"] == "SATURDAY") |
                               (appl["WEEKDAY_APPR_PROCESS_START"] == "SUNDAY")] = "Weekend"

# Apply log transformation to reduce skew in financial variables
log_vars = ["AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_GOODS_PRICE", "AMT_ANNUITY"]
appl = create_logs(appl, log_vars, replace = True)

# Convert days into months and replace negatives
day_vars = ["DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_EMPLOYED", "DAYS_LAST_PHONE_CHANGE"]
appl = convert_days(appl, day_vars, t = 30, rounding = True, replace = True)

# Create ratios based on age-related features
appl["OWN_CAR_AGE_RATIO"] = appl["OWN_CAR_AGE"] / appl["DAYS_BIRTH"]
appl["DAYS_ID_PUBLISHED_RATIO"] = appl["DAYS_ID_PUBLISH"] / appl["DAYS_BIRTH"]
appl["DAYS_REGISTRATION_RATIO"] = appl["DAYS_REGISTRATION"] / appl["DAYS_BIRTH"]
appl["DAYS_LAST_PHONE_CHANGE_RATIO"] = appl["DAYS_LAST_PHONE_CHANGE"] / appl["DAYS_BIRTH"]

# Feature removal

# Drop columns that are too sparse (many missing values) or not useful
drops = ['APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 
         'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI',
         'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI','YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI',
         'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'COMMONAREA_MODE','ELEVATORS_MODE', 'ENTRANCES_MODE', 
         'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 
         'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'TOTALAREA_MODE',  'YEARS_BEGINEXPLUATATION_MODE']
appl = appl.drop(columns = drops)



# Rename all feature columns by adding 'app_' prefix, except for 'SK_ID_CURR'
appl.columns = ["SK_ID_CURR"] + ["app_" + str(col) for col in appl.columns if col not in "SK_ID_CURR"]



# Show the first 5 rows of the appl DataFrame to check the data
appl.head()



# Count missing values in each column and show the top results
nas = count_missings(appl)
nas.head()



# Show the first 5 rows of the bbal DataFrame to check the data
bbal.head()



# Feature engineering

# Create a loan default score based on payment status

# Start with a new column for numeric version of STATUS
bbal["NUM_STATUS"] = 0

# Convert STATUS values to numbers (ignore 'X', which means no info)
bbal["NUM_STATUS"][bbal["STATUS"] == "X"] = None
bbal["NUM_STATUS"][bbal["STATUS"] == "1"] = 1
bbal["NUM_STATUS"][bbal["STATUS"] == "2"] = 2
bbal["NUM_STATUS"][bbal["STATUS"] == "3"] = 3
bbal["NUM_STATUS"][bbal["STATUS"] == "4"] = 4
bbal["NUM_STATUS"][bbal["STATUS"] == "5"] = 5

# Calculate loan score: higher status and more recent (closer to 0) = higher risk
bbal["LOAN_SCORE"] = bbal["NUM_STATUS"] / (abs(bbal["MONTHS_BALANCE"]) + 1)

# Group by bureau ID and sum the loan scores for each loan
loan_score = bbal.groupby("SK_ID_BUREAU", as_index = False).LOAN_SCORE.sum()

# Remove temporary columns
del bbal["NUM_STATUS"]
del bbal["LOAN_SCORE"]

# Convert STATUS column to dummy variables (one-hot encoding)
bbal = pd.get_dummies(bbal, columns = ["STATUS"], prefix = "STATUS")



# Count missing values in each column of the bbal DataFrame and show the top results
nas = count_missings(bbal)
nas.head()



# Aggregations

# Count how many months of data are available for each loan
cnt_mon = bbal[["SK_ID_BUREAU", "MONTHS_BALANCE"]].groupby("SK_ID_BUREAU").count()

# Remove MONTHS_BALANCE column since it's no longer needed
del bbal["MONTHS_BALANCE"]

# Aggregate the remaining STATUS dummy variables by taking the mean per loan
agg_bbal = bbal.groupby("SK_ID_BUREAU").mean()

# Add the total month count as a new column
agg_bbal["MONTH_COUNT"] = cnt_mon

# Add the previously calculated loan score
agg_bbal = agg_bbal.merge(loan_score, how = "left", on = "SK_ID_BUREAU")



# Count missing values in each column of the agg_bbal DataFrame and show the top results
nas = count_missings(agg_bbal)
nas.head()




# Show the first 5 rows of the agg_bbal DataFrame to check the data
agg_bbal.head()




# Delete the bbal DataFrame from memory to save space
del bbal



# Show the first 5 rows of the buro DataFrame to check the data
buro.head()



# Merge the aggregated bureau balance data into the bureau data using SK_ID_BUREAU as the key
buro = buro.merge(right = agg_bbal.reset_index(), how = "left", on = "SK_ID_BUREAU")



# Feature engineering

# Count the number of bureau loans per customer
cnt_buro = buro[["SK_ID_CURR", "SK_ID_BUREAU"]].groupby(["SK_ID_CURR"], as_index = False).count()
cnt_buro.columns = ["SK_ID_CURR", "CNT_BURO_LOANS"]  # Rename column
buro = buro.merge(cnt_buro, how = "left", on = "SK_ID_CURR")

# Create ratios using overdue and debt amounts to better understand financial risk
buro["AMT_SUM_OVERDUE_RATIO_1"] = buro["AMT_CREDIT_SUM_OVERDUE"] / buro["AMT_ANNUITY"]
buro["AMT_SUM_OVERDUE_RATIO_2"] = buro["AMT_CREDIT_SUM_OVERDUE"] / buro["AMT_CREDIT_SUM"]
buro["AMT_MAX_OVERDUE_RATIO_1"] = buro["AMT_CREDIT_MAX_OVERDUE"] / buro["AMT_ANNUITY"]
buro["AMT_MAX_OVERDUE_RATIO_2"] = buro["AMT_CREDIT_MAX_OVERDUE"] / buro["AMT_CREDIT_SUM"]
buro["AMT_SUM_DEBT_RATIO_1"]    = buro["AMT_CREDIT_SUM_DEBT"] / buro["AMT_CREDIT_SUM"]
buro["AMT_SUM_DEBT_RATIO_2"]    = buro["AMT_CREDIT_SUM_DEBT"] / buro["AMT_CREDIT_SUM_LIMIT"]

# Apply log transformation to reduce skew in financial columns
log_vars = ["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_OVERDUE", "AMT_ANNUITY"]
buro = create_logs(buro, log_vars, replace = True)

# Convert day columns (in days) to months or keep as-is based on input
day_vars = ["DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT", "DAYS_CREDIT_UPDATE"]
buro = convert_days(buro, day_vars, t = 1, rounding = False, replace = True)

# Calculate recency-weighted loan score (more recent = higher weight)
buro["WEIGHTED_LOAN_SCORE"] = buro["LOAN_SCORE"] / (buro["DAYS_CREDIT"] / 12)

# Calculate how dates differ to estimate duration or delays
buro["DAYS_END_DIFF_1"] = buro["DAYS_ENDDATE_FACT"]   - buro["DAYS_CREDIT_ENDDATE"]
buro["DAYS_END_DIFF_2"] = buro["DAYS_CREDIT_UPDATE"]  - buro["DAYS_CREDIT_ENDDATE"]
buro["DAYS_DURATION_1"] = buro["DAYS_CREDIT_ENDDATE"] - buro["DAYS_CREDIT"]
buro["DAYS_DURATION_2"] = buro["DAYS_ENDDATE_FACT"]   - buro["DAYS_CREDIT"]

# Count number of active loans per customer
cnt_buro = buro[["SK_ID_CURR", "CREDIT_ACTIVE"]]
cnt_buro.columns = ["SK_ID_CURR", "CNT_BURO_ACTIVE"]
cnt_buro = cnt_buro[cnt_buro["CNT_BURO_ACTIVE"] == "Active"]
cnt_buro = cnt_buro[["SK_ID_CURR", "CNT_BURO_ACTIVE"]].groupby(["SK_ID_CURR"], as_index = False).count()
buro = buro.merge(cnt_buro, how = "left", on = "SK_ID_CURR")
buro["CNT_BURO_ACTIVE"].fillna(0, inplace = True)  # Fill missing with 0

# Count number of closed loans per customer
cnt_buro = buro[["SK_ID_CURR", "CREDIT_ACTIVE"]]
cnt_buro.columns = ["SK_ID_CURR", "CNT_BURO_CLOSED"]
cnt_buro = cnt_buro[cnt_buro["CNT_BURO_CLOSED"] == "Closed"]
cnt_buro = cnt_buro[["SK_ID_CURR", "CNT_BURO_CLOSED"]].groupby(["SK_ID_CURR"], as_index = False).count()
buro = buro.merge(cnt_buro, how = "left", on = "SK_ID_CURR")
buro["CNT_BURO_CLOSED"].fillna(0, inplace = True)

# Count number of bad debt loans per customer
cnt_buro = buro[["SK_ID_CURR", "CREDIT_ACTIVE"]]
cnt_buro.columns = ["SK_ID_CURR", "CNT_BURO_BAD"]
cnt_buro = cnt_buro[cnt_buro["CNT_BURO_BAD"] == "Bad debt"]
cnt_buro = cnt_buro[["SK_ID_CURR", "CNT_BURO_BAD"]].groupby(["SK_ID_CURR"], as_index = False).count()
buro = buro.merge(cnt_buro, how = "left", on = "SK_ID_CURR")
buro["CNT_BURO_BAD"].fillna(0, inplace = True)



# Convert categorical (text) columns into dummy/one-hot encoded variables and drop the first category to avoid duplication
buro = pd.get_dummies(buro, drop_first = True)



# Count missing values in each column of the buro DataFrame
nas = count_missings(buro)

# Show the first few rows of the missing values summary
nas.head()



# Aggregations

# Count how many bureau loans each customer has
cnt_buro = buro[["SK_ID_CURR", "SK_ID_BUREAU"]].groupby("SK_ID_CURR").count()

# Remove the bureau ID column since it's no longer needed
del buro["SK_ID_BUREAU"]

# Aggregate the bureau data by customer ID
agg_buro = aggregate_data(buro, id_var = "SK_ID_CURR", label = "buro")

# Add the loan count to the aggregated bureau data
agg_buro["buro_BURO_COUNT"] = cnt_buro

# Clean up unnecessary statistics for specific features
omits = ["WEIGHTED_LOAN_SCORE"]
for var in omits:
    del agg_buro["buro_" + str(var) + "_std"]  # Remove standard deviation column
    del agg_buro["buro_" + str(var) + "_min"]  # Remove minimum value column
    del agg_buro["buro_" + str(var) + "_max"]  # Remove maximum value column



# Count missing values in each column of the agg_buro DataFrame and show the top results
nas = count_missings(agg_buro)
nas.head()



# Show the first 5 rows of the agg_buro DataFrame to check the data
agg_buro.head()



# Delete the buro DataFrame from memory to save space
del buro



# Show the first 5 rows of the inst DataFrame to check the data
inst.head()



# Feature engineering

# Calculate how many days payment was late or early (no negative values allowed)
inst['DPD'] = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']  # Days Past Due
inst['DBD'] = inst['DAYS_INSTALMENT'] - inst['DAYS_ENTRY_PAYMENT']  # Days Before Due

# Replace negative values with 0 (only keep positive delays/early payments)
inst['DPD'] = inst['DPD'].apply(lambda x: x if x > 0 else 0)
inst['DBD'] = inst['DBD'].apply(lambda x: x if x > 0 else 0)

# Calculate how much of the installment was paid (as a percentage)
inst['PAYMENT_PERC'] = inst['AMT_PAYMENT'] / inst['AMT_INSTALMENT']

# Calculate the difference between expected and actual payment
inst['PAYMENT_DIFF'] = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']

# Apply log transformation to payment columns to reduce skew
log_vars = ["AMT_INSTALMENT", "AMT_PAYMENT"]
inst = create_logs(inst, log_vars, replace = True)



# Convert categorical columns into dummy variables (one-hot encoding) and drop the first category
inst = pd.get_dummies(inst, drop_first = True)



# Count missing values in each column of the inst DataFrame and show the top results
nas = count_missings(inst)
nas.head()



# Aggregations

# Count how many instalments each loan (SK_ID_PREV) has
cnt_inst = inst[["SK_ID_PREV", "NUM_INSTALMENT_NUMBER"]].groupby("SK_ID_PREV").count()

# Remove NUM_INSTALMENT_NUMBER column since it's already counted
del inst["NUM_INSTALMENT_NUMBER"]

# Save SK_ID_CURR and SK_ID_PREV for later use
inst_id = inst[["SK_ID_CURR", "SK_ID_PREV"]]

# Remove SK_ID_CURR from the main data to avoid duplicate grouping
del inst["SK_ID_CURR"]

# First aggregation: group by previous loan ID (SK_ID_PREV)
agg_inst = aggregate_data(inst, id_var = "SK_ID_PREV")

# Add the number of instalments per loan to the aggregated data
agg_inst["inst_INST_COUNT"] = cnt_inst

# Add back SK_ID_CURR by merging with the ID mapping (one row per loan)
inst_id = inst_id.drop_duplicates()  # Drop duplicates before merge
agg_inst = inst_id.merge(right = agg_inst.reset_index(), how = "right", on = "SK_ID_PREV")

# Remove SK_ID_PREV since we will now group by customer
del agg_inst["SK_ID_PREV"]

# Second aggregation: group by customer ID (SK_ID_CURR)
agg_inst = aggregate_data(agg_inst, id_var = "SK_ID_CURR", label = "inst")



# Count missing values in each column of the agg_inst DataFrame and show the top results
nas = count_missings(agg_inst)
nas.head()



# Show the first 5 rows of the agg_inst DataFrame to check the data
agg_inst.head()



# Delete the inst DataFrame from memory to free up space
del inst



# Show the first 5 rows of the poca DataFrame to check the data
poca.head()



# Feature engineering

# Calculate the percentage of remaining installments compared to total installments
poca["INSTALLMENTS_PERCENT"] = poca["CNT_INSTALMENT_FUTURE"] / poca["CNT_INSTALMENT"]



# Convert categorical columns into dummy variables (one-hot encoding) and drop the first category
poca = pd.get_dummies(poca, drop_first = True)



# Count missing values in each column of the poca DataFrame and show the top results
nas = count_missings(poca)
nas.head()



# Aggregations

# Count how many months of data each previous loan has
cnt_mon = poca[["SK_ID_PREV", "MONTHS_BALANCE"]].groupby("SK_ID_PREV").count()

# Remove MONTHS_BALANCE column since we already counted it
del poca["MONTHS_BALANCE"]

# Save SK_ID_CURR and SK_ID_PREV for later
poca_id = poca[["SK_ID_CURR", "SK_ID_PREV"]]

# Remove SK_ID_CURR from the main data before aggregating
del poca["SK_ID_CURR"]

# First aggregation: group by previous loan ID (SK_ID_PREV)
agg_poca = aggregate_data(poca, id_var = "SK_ID_PREV")

# Add the number of months per loan to the aggregated data
agg_poca["poca_MON_COUNT"] = cnt_mon

# Merge SK_ID_CURR back in using the saved mapping
poca_id = poca_id.drop_duplicates()
agg_poca = poca_id.merge(right = agg_poca.reset_index(), how = "right", on = "SK_ID_PREV")

# Remove SK_ID_PREV now that we will group by customer
del agg_poca["SK_ID_PREV"]

# Second aggregation: group by customer ID (SK_ID_CURR)
agg_poca = aggregate_data(agg_poca, id_var = "SK_ID_CURR", label = "poca")



# Count missing values in each column of the agg_poca DataFrame and show the top results
nas = count_missings(agg_poca)
nas.head()



# check data
agg_poca.head()# Show the first 5 rows of the agg_poca DataFrame to check the data
agg_poca.head()



# Delete the poca DataFrame from memory to free up space
del poca



# Show the first 5 rows of the card DataFrame to check the data
card.head()



# Feature engineering

# Apply log transformation to selected financial columns to reduce skew
log_vars = ["AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_ATM_CURRENT", "AMT_DRAWINGS_CURRENT",
            "AMT_DRAWINGS_OTHER_CURRENT", "AMT_DRAWINGS_POS_CURRENT", "AMT_INST_MIN_REGULARITY",
            "AMT_PAYMENT_CURRENT", "AMT_PAYMENT_TOTAL_CURRENT", "AMT_RECEIVABLE_PRINCIPAL",
            "AMT_RECIVABLE", "AMT_TOTAL_RECEIVABLE"]

# Apply log to each of the above columns and replace original values
card = create_logs(card, log_vars, replace = True)



# Convert categorical columns into dummy variables (one-hot encoding) and drop the first category to avoid redundancy
card = pd.get_dummies(card, drop_first = True)



# Count missing values in each column of the card DataFrame and show the top results
nas = count_missings(card)
nas.head()



# Aggregations

# Count how many months of credit card data are available for each loan (SK_ID_PREV)
cnt_mon = card[["SK_ID_PREV", "MONTHS_BALANCE"]].groupby("SK_ID_PREV").count()

# Remove MONTHS_BALANCE column since it has been counted
del card["MONTHS_BALANCE"]

# Save SK_ID_CURR and SK_ID_PREV to merge back later
card_id = card[["SK_ID_CURR", "SK_ID_PREV"]]

# Remove SK_ID_CURR before aggregating to avoid duplicate entries
del card["SK_ID_CURR"]

# First aggregation: group by SK_ID_PREV (loan level)
agg_card = aggregate_data(card, id_var = "SK_ID_PREV")

# Add the number of months of records for each card
agg_card["card_MON_COUNT"] = cnt_mon

# Merge SK_ID_CURR back using saved ID mapping
card_id = card_id.drop_duplicates()
agg_card = card_id.merge(right = agg_card.reset_index(), how = "right", on = "SK_ID_PREV")

# Remove SK_ID_PREV since we now aggregate at the customer level
del agg_card["SK_ID_PREV"]

# Second aggregation: group by SK_ID_CURR (customer level)
agg_card = aggregate_data(agg_card, id_var = "SK_ID_CURR", label = "card")



# Count missing values in each column of the agg_card DataFrame and show the top results
nas = count_missings(agg_card)
nas.head()



# Show the first 5 rows of the agg_card DataFrame to check the data
agg_card.head()



# Delete the card DataFrame from memory to free up space
del card



# Show the first 5 rows of the prev DataFrame to check the data
prev.head()



# Feature engineering

# Create ratios to understand how much was granted vs requested
prev["AMT_GIVEN_RATIO_1"]  = prev["AMT_CREDIT"] / prev["AMT_APPLICATION"]         # Credit granted vs applied
prev["AMT_GIVEN_RATIO_2"]  = prev["AMT_GOODS_PRICE"] / prev["AMT_APPLICATION"]    # Price of goods vs applied
prev["DOWN_PAYMENT_RATIO"] = prev["AMT_DOWN_PAYMENT"] / prev["AMT_APPLICATION"]  # Down payment vs applied

# Apply log transformation to reduce skew in selected financial columns
log_vars = ["AMT_CREDIT", "AMT_ANNUITY", "AMT_APPLICATION", "AMT_DOWN_PAYMENT", "AMT_GOODS_PRICE"]
prev = create_logs(prev, log_vars, replace = True)

# Convert day-based columns into positive values (or leave as days, based on settings)
day_vars = ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
            "DAYS_LAST_DUE", "DAYS_TERMINATION", "DAYS_DECISION"]
prev = convert_days(prev, day_vars, t = 1, rounding = False, replace = True)

# Count how many previous applications each customer has
cnt_prev = prev[["SK_ID_CURR", "SK_ID_PREV"]].groupby(["SK_ID_CURR"], as_index = False).count()
cnt_prev.columns = ["SK_ID_CURR", "CNT_PREV_APPLICATIONS"]
prev = prev.merge(cnt_prev, how = "left", on = "SK_ID_CURR")

# Count how many of those were the last application per contract
cnt_prev = prev[["SK_ID_CURR", "FLAG_LAST_APPL_PER_CONTRACT"]]
cnt_prev.columns = ["SK_ID_CURR", "CNT_PREV_CONTRACTS"]
cnt_prev = cnt_prev[cnt_prev["CNT_PREV_CONTRACTS"] == "Y"]
cnt_prev = cnt_prev[["SK_ID_CURR", "CNT_PREV_CONTRACTS"]].groupby(["SK_ID_CURR"], as_index = False).count()
prev = prev.merge(cnt_prev, how = "left", on = "SK_ID_CURR")

# Create ratio between number of applications and contracts
prev["APPL_PER_CONTRACT_RATIO"] = prev["CNT_PREV_APPLICATIONS"] / prev["CNT_PREV_CONTRACTS"]

# Calculate approval/rejection ratios for most recent previous applications
prev = compute_accept_reject_ratio(prev, lags = [1, 3, 5])

# Calculate day differences between due dates and other key dates
prev["DAYS_DUE_DIFF_1"] = prev["DAYS_LAST_DUE_1ST_VERSION"] - prev["DAYS_FIRST_DUE"]
prev["DAYS_DUE_DIFF_2"] = prev["DAYS_LAST_DUE"] - prev["DAYS_FIRST_DUE"]
prev["DAYS_TERMINATION_DIFF_1"] = prev["DAYS_TERMINATION"] - prev["DAYS_FIRST_DRAWING"]
prev["DAYS_TERMINATION_DIFF_2"] = prev["DAYS_TERMINATION"] - prev["DAYS_FIRST_DUE"]
prev["DAYS_TERMINATION_DIFF_3"] = prev["DAYS_TERMINATION"] - prev["DAYS_LAST_DUE"]

# Classify application day as working day or weekend
prev["DAY_APPR_PROCESS_START"] = "Working day"
prev["DAY_APPR_PROCESS_START"][(prev["WEEKDAY_APPR_PROCESS_START"] == "SATURDAY") |
                               (prev["WEEKDAY_APPR_PROCESS_START"] == "SUNDAY")] = "Weekend"

##### Feature removal

# Drop unnecessary columns (client type and previous loan ID)
drops = ["NAME_CLIENT_TYPE", "SK_ID_PREV"]
prev = prev.drop(columns = drops)



# Convert categorical columns into dummy variables (one-hot encoding) and drop the first category to avoid redundancy
prev = pd.get_dummies(prev, drop_first = True)



# Count missing values in each column of the prev DataFrame and show the top results
nas = count_missings(prev)
nas.head()



# Aggregations

# Aggregate the previous applications data by customer ID (SK_ID_CURR)
agg_prev = aggregate_data(prev, id_var = "SK_ID_CURR", label = "prev")

# Clean up: remove unnecessary statistics for specific features
omits = ["APPROVE_RATIO_1", "APPROVE_RATIO_3", "APPROVE_RATIO_5",
         "REJECT_RATIO_1", "REJECT_RATIO_3",  "REJECT_RATIO_5",
         "FLAG_LAST_APPL_PER_CONTRACT_Y", "CNT_PREV_CONTRACTS", "CNT_PREV_APPLICATIONS",
         "APPL_PER_CONTRACT_RATIO"]

# For each of the omitted features, delete std, min, and max columns to reduce clutter
for var in omits:
    del agg_prev["prev_" + str(var) + "_std"]
    del agg_prev["prev_" + str(var) + "_min"]
    del agg_prev["prev_" + str(var) + "_max"]



# Count missing values in each column of the agg_prev DataFrame and show the top results
nas = count_missings(agg_prev)
nas.head()



# Show the first 5 rows of the agg_prev DataFrame to check the data
agg_prev.head()



# Delete the prev DataFrame from memory to free up space
del prev



# Reset index for all aggregated DataFrames to make 'SK_ID_CURR' a normal column again
agg_buro = agg_buro.reset_index()
agg_prev = agg_prev.reset_index()
agg_inst = agg_inst.reset_index()
agg_poca = agg_poca.reset_index()
agg_card = agg_card.reset_index()

# Print column names of each DataFrame to confirm 'SK_ID_CURR' is present
print("agg_buro columns:", agg_buro.columns)
print("agg_prev columns:", agg_prev.columns)
print("agg_inst columns:", agg_inst.columns)
print("agg_poca columns:", agg_poca.columns)
print("agg_card columns:", agg_card.columns)



# ✅ Drop index before merging to avoid duplicates

# Merge agg_buro data into appl DataFrame and print new shape
appl = appl.merge(agg_buro.reset_index(drop=True), how="left", on="SK_ID_CURR")
print(appl.shape)
del agg_buro  # Delete agg_buro to save memory

# Merge agg_prev data into appl DataFrame and print new shape
appl = appl.merge(agg_prev.reset_index(drop=True), how="left", on="SK_ID_CURR")
print(appl.shape)
del agg_prev  # Delete agg_prev to save memory

# Merge agg_inst data into appl DataFrame and print new shape
appl = appl.merge(agg_inst.reset_index(drop=True), how="left", on="SK_ID_CURR")
print(appl.shape)
del agg_inst  # Delete agg_inst to save memory

# Merge agg_poca data into appl DataFrame and print new shape
appl = appl.merge(agg_poca.reset_index(drop=True), how="left", on="SK_ID_CURR")
print(appl.shape)
del agg_poca  # Delete agg_poca to save memory

# Merge agg_card data into appl DataFrame and print new shape
appl = appl.merge(agg_card.reset_index(drop=True), how="left", on="SK_ID_CURR")
print(appl.shape)
del agg_card  # Delete agg_card to save memory



# Cross-table Feature Engineering

# Create new features by calculating the ratio of application values to previous and bureau values

# Credit ratio between application and previous loan annuities
appl["mix_AMT_PREV_ANNUITY_RATIO"] = appl["app_AMT_ANNUITY"] / appl["prev_AMT_ANNUITY_mean"]

# Credit ratio between application and previous loan credit amounts
appl["mix_AMT_PREV_CREDIT_RATIO"] = appl["app_AMT_CREDIT"] / appl["prev_AMT_CREDIT_mean"]

# Credit ratio between application and previous loan goods prices
appl["mix_AMT_PREV_GOODS_PRICE_RATIO"] = appl["app_AMT_GOODS_PRICE"] / appl["prev_AMT_GOODS_PRICE_mean"]

# Credit ratio between application and bureau loan annuities
appl["mix_AMT_BURO_ANNUITY_RATIO"] = appl["app_AMT_ANNUITY"] / appl["buro_AMT_ANNUITY_mean"]

# Credit ratio between application and bureau loan credit amounts
appl["mix_AMT_BURO_CREDIT_RATIO"] = appl["app_AMT_CREDIT"] / appl["buro_AMT_CREDIT_SUM_mean"]



# Convert categorical columns into dummy variables (one-hot encoding) and drop the first category to avoid redundancy
appl = pd.get_dummies(appl, drop_first = True)



# Count missing values in each column of the appl DataFrame and show the top results
nas = count_missings(appl)
nas.head()



# Split the appl DataFrame into train and test based on SK_ID_CURR
train = appl[appl["SK_ID_CURR"].isin(y["SK_ID_CURR"]) == True]  # Select rows where SK_ID_CURR is in the target (train set)
test  = appl[appl["SK_ID_CURR"].isin(y["SK_ID_CURR"]) == False] # Select rows where SK_ID_CURR is not in the target (test set)

# Delete appl DataFrame to free up memory as it's no longer needed
del appl



# Print the dimensions (rows and columns) of the train and test DataFrames
print(train.shape)
print(test.shape)



# Export train, test, and target (y) DataFrames to CSV files with 8 decimal places
train.to_csv("/kaggle/working/train_full_cor.csv", index=False, float_format="%.8f")
test.to_csv("/kaggle/working/test_full_cor.csv", index=False, float_format="%.8f")
y.to_csv("/kaggle/working/y_full_cor.csv", index=False, float_format="%.8f")



train.to_feather("/kaggle/working/train_full_cor.feather")
test.to_feather("/kaggle/working/test_full_cor.feather")
y.to_feather("/kaggle/working/y_full_cor.feather")









