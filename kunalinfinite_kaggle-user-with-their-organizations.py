import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import kagglehub

# Download latest version
path = kagglehub.dataset_download("kaggle/meta-kaggle")

print("Path to dataset files:", path)


Users=pd.read_csv("/kaggle/input/meta-kaggle/Users.csv")
UserOrganizations=pd.read_csv("/kaggle/input/meta-kaggle/UserOrganizations.csv")
Organizations=pd.read_csv("/kaggle/input/meta-kaggle/Organizations.csv")


merged = pd.merge(
    Users[['Id', 'PerformanceTier']],
    UserOrganizations,
    left_on='Id',
    right_on='UserId',
    how='inner'
)

# Merge with Organizations to get names
merged = pd.merge(
    merged,
    Organizations[['Id', 'Name']],
    left_on='OrganizationId',
    right_on='Id',
    suffixes=('_user', '_org')
)


# Define tier thresholds
tiers = {
    'Grandmaster': 5,
    'Master': 4,
    'Expert': 3
}

# Count users per tier per organization
results = {}
for tier_name, tier_value in tiers.items():
    tier_users = merged[merged['PerformanceTier'] == tier_value]
    org_counts = tier_users['Name'].value_counts().reset_index()
    org_counts.columns = ['Organization', f'{tier_name}_Count']
    results[tier_name] = org_counts.head(10)  # Top 10 orgs per tier


print("Top Organizations with Kaggle Grandmasters:")
print(results['Grandmaster'])


print("\nTop Organizations with Kaggle Masters:")
print(results['Master'])


print("\nTop Organizations with Kaggle Experts:")
print(results['Expert'])

