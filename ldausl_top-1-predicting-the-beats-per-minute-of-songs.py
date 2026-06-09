%%time 

import pandas as pd

sub1 = pd.read_csv(f"/kaggle/input/playgrounds5e9-randomnoise/submission.csv")
sub2 = pd.read_csv(f"/kaggle/input/0-00001-gain/submission.csv")

sub1["BeatsPerMinute"] = (
    sub1["BeatsPerMinute"].values * -0.10 + sub2["BeatsPerMinute"].values * 1.1
)


# import pandas as pd

# sub1.loc[0, 'BeatsPerMinute'] = sub1.loc[0, 'BeatsPerMinute'] + 1

# print(sub1.head())


import pandas as pd
sub1['BeatsPerMinute'] = sub1['BeatsPerMinute'] + 1



sub1.to_csv("submission.csv", index = None)


sub1

