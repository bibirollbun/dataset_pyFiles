# Set up Code

import pandas as pd

def summarize_positions( row ):

    #pos = row["position"]
    summary_hash = {
        'CB':  'Cornerbacks',
        'SS':  'Safeties',
        'FS':  'Safeties',
        'ILB': 'Linebackers',
        'OLB': 'Linebackers',
        'MLB': 'Linebackers',
        'LB':  'Linebackers'
    }
    try:
        pos_summary = summary_hash[ row[ "position" ] ]
    except:
        pos_summary = "Other"

    return pos_summary


motion_filename = "/kaggle/input/nfl-2022-week-4-motion-events/motion.20250105.192330.csv"

df_motion = pd.read_csv(motion_filename)

col_name = 'Defense Positions'

# group the positions
df_motion[col_name] = df_motion.apply(summarize_positions, axis=1)
df_motion = df_motion[ df_motion[ col_name ] != "Other" ]

df_motion[col_name].value_counts()


df_motion_cbs = df_motion[ df_motion[ "Defense Positions" ] == "Safeties" ]
df_motion_cbs[ 'motionDir' ].value_counts()


df_motion_cbs = df_motion[ df_motion[ "Defense Positions" ] == "Cornerbacks" ]
df_motion_cbs[ 'motionDir' ].value_counts()

