!pip install ydata_profiling


!pip install --upgrade numba pandas visions ydata_profiling


import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')


df.info()


from ydata_profiling import ProfileReport
cols = ['Auction', 'Make', 'Model', 'Trim', 'SubModel', 'Color', 'Transmission',
        'WheelType', 'Nationality', 'TopThreeAmericanName', 'Size',
        'PRIMEUNIT', 'AUCGUART', 'VNST']

profile = ProfileReport(df, title="kicked car", type_schema = {col: 'categorical' for col in cols})
# Save report to HTML
profile.to_file("dont_get_kicked_training.html")


df_good = df[df.IsBadBuy == 0]
df_bad = df[df.IsBadBuy == 1]
# Generate a profile 
profile_0=ProfileReport(df_good, title='good buy', type_schema = {col: 'categorical' for col in cols})
profile_1=ProfileReport(df_bad, title='bad buy', type_schema = {col: 'categorical' for col in cols})
compare=profile_0.compare(profile_1)
compare.to_file("comparison.html")


# drop varios conditions of price
#import pandas
#df_first=df.drop(columns=['RefId'])
#column_price=["MMRAcquisitionAuctionAveragePrice", "MMRAcquisitionAuctionCleanPrice", "MMRAcquisitionRetailAveragePrice", "MMRAcquisitonRetailCleanPrice", "MMRCurrentAuctionAveragePrice", "MMRCurrentAuctionCleanPrice", "MMRCurrentRetailAveragePrice", "MMRCurrentRetailCleanPrice", "VehBCost"]
#df_without_price=df_first.drop(columns=column_price)


