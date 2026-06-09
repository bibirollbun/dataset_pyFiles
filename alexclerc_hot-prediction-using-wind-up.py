# this notebook demonstrates feature engineeing with wind-up is possible
# wind-up requires more information than provided in the kaggle, so a large fraction of this notebook runtime is downloading more HoT data from Zenodo
# Overall process demonstrated in this notebook:
# - download complete HoT dataset
# - run wind-up preprocessing, which learns basic turbine attributes like the typical power curve
# - run wind-up detrending on all nearby turbines, which models the wind speed at T1 as a function of wind speed at a nearby turbine
# - predict power at T1 using wind-up calculations
# - finally give autogluon all the wind-up columns (and the raw columns from the kaggle) to make the submission file

# first clone the hill of towie open source analysis repo and checkout commit from Nov 24th 2025
%cd /kaggle/working
! rm -rf hill-of-towie-open-source-analysis
! git clone https://github.com/resgroup/hill-of-towie-open-source-analysis
%cd hill-of-towie-open-source-analysis
! git checkout 169158a512a3846f4b499b6e251137f8cef3e119


# install dependencies

! pip install -e .
! pip install ephem autogluon


# set an environment variable to make sure the data is saved in an accessible place

import os
os.environ['WINDUP_ANALYSIS_DIR'] = '/kaggle/working/windup'


# run a script that generates a submission file using the wind-up package

! python scripts/wedowind_challenge/solution_using_wind_up.py

