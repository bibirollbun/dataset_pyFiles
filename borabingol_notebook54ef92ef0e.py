import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns

import matplotlib.pyplot as plt
import os
import time
import numpy as np
import glob
import json
import collections
import torch
import torch.nn as nn

import pydicom as dicom
import matplotlib.patches as patches

from matplotlib import animation, rc
import pandas as pd

import pydicom as dicom # dicom
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pandas as pd
from io import StringIO
# read data
train_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/'
sub         = pd.read_csv(train_path + 'sample_submission.csv')
# CSV metni (elinizdeki veriyi buraya yapıştırın)
csv_text = """study_id,series_id,series_description,image_path,condition,row_id
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/12.dcm,left_neural_foraminal_narrowing,44036939_left_neural_foraminal_narrowing_l1_l2
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/18.dcm,left_neural_foraminal_narrowing,44036939_left_neural_foraminal_narrowing_l2_l3
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/9.dcm,left_neural_foraminal_narrowing,44036939_left_neural_foraminal_narrowing_l3_l4
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/22.dcm,left_neural_foraminal_narrowing,44036939_left_neural_foraminal_narrowing_l4_l5
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/25.dcm,left_neural_foraminal_narrowing,44036939_left_neural_foraminal_narrowing_l5_s1
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/14.dcm,left_neural_foraminal_narrowing,44036939_left_neural_foraminal_narrowing_l1_l2
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/11.dcm,left_neural_foraminal_narrowing,44036939_left_neural_foraminal_narrowing_l2_l3
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/24.dcm,left_neural_foraminal_narrowing,44036939_left_neural_foraminal_narrowing_l3_l4
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/23.dcm,left_neural_foraminal_narrowing,44036939_left_neural_foraminal_narrowing_l4_l5
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/10.dcm,left_neural_foraminal_narrowing,44036939_left_neural_foraminal_narrowing_l5_s1
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/17.dcm,left_neural_foraminal_narrowing,44036939_left_neural_foraminal_narrowing_l1_l2
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/1.dcm,left_neural_foraminal_narrowing,44036939_left_neural_foraminal_narrowing_l2_l3
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/15.dcm,left_neural_foraminal_narrowing,44036939_left_neural_foraminal_narrowing_l3_l4
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/2.dcm,left_neural_foraminal_narrowing,44036939_left_neural_foraminal_narrowing_l4_l5
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/8.dcm,left_neural_foraminal_narrowing,44036939_left_neural_foraminal_narrowing_l5_s1
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/7.dcm,left_neural_foraminal_narrowing,44036939_left_neural_foraminal_narrowing_l1_l2
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/21.dcm,left_neural_foraminal_narrowing,44036939_left_neural_foraminal_narrowing_l2_l3
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/5.dcm,left_neural_foraminal_narrowing,44036939_left_neural_foraminal_narrowing_l3_l4
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/4.dcm,left_neural_foraminal_narrowing,44036939_left_neural_foraminal_narrowing_l4_l5
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/19.dcm,left_neural_foraminal_narrowing,44036939_left_neural_foraminal_narrowing_l5_s1
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/6.dcm,left_neural_foraminal_narrowing,44036939_left_neural_foraminal_narrowing_l1_l2
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/16.dcm,left_neural_foraminal_narrowing,44036939_left_neural_foraminal_narrowing_l2_l3
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/20.dcm,left_neural_foraminal_narrowing,44036939_left_neural_foraminal_narrowing_l3_l4
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/3.dcm,left_neural_foraminal_narrowing,44036939_left_neural_foraminal_narrowing_l4_l5
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/13.dcm,left_neural_foraminal_narrowing,44036939_left_neural_foraminal_narrowing_l5_s1
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/12.dcm,right_neural_foraminal_narrowing,44036939_right_neural_foraminal_narrowing_l1_l2
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/18.dcm,right_neural_foraminal_narrowing,44036939_right_neural_foraminal_narrowing_l2_l3
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/9.dcm,right_neural_foraminal_narrowing,44036939_right_neural_foraminal_narrowing_l3_l4
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/22.dcm,right_neural_foraminal_narrowing,44036939_right_neural_foraminal_narrowing_l4_l5
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/25.dcm,right_neural_foraminal_narrowing,44036939_right_neural_foraminal_narrowing_l5_s1
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/14.dcm,right_neural_foraminal_narrowing,44036939_right_neural_foraminal_narrowing_l1_l2
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/11.dcm,right_neural_foraminal_narrowing,44036939_right_neural_foraminal_narrowing_l2_l3
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/24.dcm,right_neural_foraminal_narrowing,44036939_right_neural_foraminal_narrowing_l3_l4
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/23.dcm,right_neural_foraminal_narrowing,44036939_right_neural_foraminal_narrowing_l4_l5
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/10.dcm,right_neural_foraminal_narrowing,44036939_right_neural_foraminal_narrowing_l5_s1
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/17.dcm,right_neural_foraminal_narrowing,44036939_right_neural_foraminal_narrowing_l1_l2
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/1.dcm,right_neural_foraminal_narrowing,44036939_right_neural_foraminal_narrowing_l2_l3
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/15.dcm,right_neural_foraminal_narrowing,44036939_right_neural_foraminal_narrowing_l3_l4
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/2.dcm,right_neural_foraminal_narrowing,44036939_right_neural_foraminal_narrowing_l4_l5
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/8.dcm,right_neural_foraminal_narrowing,44036939_right_neural_foraminal_narrowing_l5_s1
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/7.dcm,right_neural_foraminal_narrowing,44036939_right_neural_foraminal_narrowing_l1_l2
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/21.dcm,right_neural_foraminal_narrowing,44036939_right_neural_foraminal_narrowing_l2_l3
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/5.dcm,right_neural_foraminal_narrowing,44036939_right_neural_foraminal_narrowing_l3_l4
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/4.dcm,right_neural_foraminal_narrowing,44036939_right_neural_foraminal_narrowing_l4_l5
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/19.dcm,right_neural_foraminal_narrowing,44036939_right_neural_foraminal_narrowing_l5_s1
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/6.dcm,right_neural_foraminal_narrowing,44036939_right_neural_foraminal_narrowing_l1_l2
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/16.dcm,right_neural_foraminal_narrowing,44036939_right_neural_foraminal_narrowing_l2_l3
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/20.dcm,right_neural_foraminal_narrowing,44036939_right_neural_foraminal_narrowing_l3_l4
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/3.dcm,right_neural_foraminal_narrowing,44036939_right_neural_foraminal_narrowing_l4_l5
44036939,2828203845,Sagittal T1,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/2828203845/13.dcm,right_neural_foraminal_narrowing,44036939_right_neural_foraminal_narrowing_l5_s1
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/12.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l1_l2
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/18.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l2_l3
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/9.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l3_l4
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/22.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l4_l5
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/25.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l5_s1
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/39.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l1_l2
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/45.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l2_l3
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/14.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l3_l4
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/11.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l4_l5
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/44.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l5_s1
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/24.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l1_l2
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/34.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l2_l3
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/29.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l3_l4
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/23.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l4_l5
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/41.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l5_s1
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/35.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l1_l2
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/10.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l2_l3
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/46.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l3_l4
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/28.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l4_l5
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/43.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l5_s1
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/37.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l1_l2
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/17.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l2_l3
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/30.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l3_l4
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/1.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l4_l5
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/15.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l5_s1
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/2.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l1_l2
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/36.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l2_l3
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/8.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l3_l4
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/7.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l4_l5
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/21.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l5_s1
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/33.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l1_l2
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/5.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l2_l3
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/4.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l3_l4
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/42.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l4_l5
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/47.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l5_s1
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/31.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l1_l2
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/38.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l2_l3
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/19.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l3_l4
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/27.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l4_l5
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/6.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l5_s1
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/16.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l1_l2
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/20.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l2_l3
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/40.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l3_l4
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/3.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l4_l5
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/32.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l5_s1
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/26.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l1_l2
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/13.dcm,left_subarticular_stenosis,44036939_left_subarticular_stenosis_l2_l3
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/12.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l3_l4
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/18.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l4_l5
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/9.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l5_s1
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/22.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l1_l2
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/25.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l2_l3
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/39.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l3_l4
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/45.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l4_l5
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/14.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l5_s1
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/11.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l1_l2
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/44.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l2_l3
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/24.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l3_l4
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/34.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l4_l5
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/29.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l5_s1
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/23.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l1_l2
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/41.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l2_l3
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/35.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l3_l4
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/10.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l4_l5
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/46.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l5_s1
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/28.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l1_l2
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/43.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l2_l3
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/37.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l3_l4
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/17.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l4_l5
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/30.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l5_s1
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/1.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l1_l2
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/15.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l2_l3
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/2.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l3_l4
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/36.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l4_l5
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/8.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l5_s1
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/7.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l1_l2
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/21.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l2_l3
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/33.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l3_l4
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/5.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l4_l5
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/4.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l5_s1
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/42.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l1_l2
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/47.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l2_l3
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/31.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l3_l4
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/38.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l4_l5
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/19.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l5_s1
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/27.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l1_l2
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/6.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l2_l3
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/16.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l3_l4
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/20.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l4_l5
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/40.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l5_s1
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/3.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l1_l2
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/32.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l2_l3
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/26.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l3_l4
44036939,3481971518,Axial T2,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3481971518/13.dcm,right_subarticular_stenosis,44036939_right_subarticular_stenosis_l4_l5
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/12.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l5_s1
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/18.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l1_l2
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/9.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l2_l3
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/22.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l3_l4
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/25.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l4_l5
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/14.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l5_s1
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/11.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l1_l2
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/24.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l2_l3
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/23.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l3_l4
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/10.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l4_l5
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/17.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l5_s1
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/1.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l1_l2
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/15.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l2_l3
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/2.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l3_l4
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/8.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l4_l5
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/7.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l5_s1
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/21.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l1_l2
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/5.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l2_l3
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/4.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l3_l4
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/19.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l4_l5
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/6.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l5_s1
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/16.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l1_l2
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/20.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l2_l3
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/3.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l3_l4
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/13.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l4_l5
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/12.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l5_s1
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/18.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l1_l2
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/9.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l2_l3
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/22.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l3_l4
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/25.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l4_l5
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/14.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l5_s1
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/11.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l1_l2
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/24.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l2_l3
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/23.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l3_l4
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/10.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l4_l5
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/17.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l5_s1
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/1.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l1_l2
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/15.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l2_l3
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/2.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l3_l4
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/8.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l4_l5
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/7.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l5_s1
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/21.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l1_l2
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/5.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l2_l3
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/4.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l3_l4
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/19.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l4_l5
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/6.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l5_s1
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/16.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l1_l2
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/20.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l2_l3
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/3.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l3_l4
44036939,3844393089,Sagittal T2/STIR,/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/44036939/3844393089/13.dcm,spinal_canal_stenosis,44036939_spinal_canal_stenosis_l4_l5
"""

# StringIO ile CSV metni bir dosya gibi okunabilir
data = StringIO(csv_text)

# Pandas ile DataFrame'e dönüştürme
df = pd.read_csv(data)




df


# Mapping of series_description to conditions
condition_mapping = {
    'Sagittal T1': {'left': 'left_neural_foraminal_narrowing', 'right': 'right_neural_foraminal_narrowing'},
    'Axial T2': {'left': 'left_subarticular_stenosis', 'right': 'right_subarticular_stenosis'},
    'Sagittal T2/STIR': 'spinal_canal_stenosis'
}


test_data = df
expanded_test_desc = df 


import os

# Define a function to check if a path exists
def check_exists(path):
    return os.path.exists(path)

# Define a function to check if a study ID directory exists
def check_study_id(row):
    study_id = row['study_id']
    path = f'{train_path}/train_images/{study_id}'
    return check_exists(path)

# Define a function to check if a series ID directory exists
def check_series_id(row):
    study_id = row['study_id']
    series_id = row['series_id']
    path = f'{train_path}/train_images/{study_id}/{series_id}'
    return check_exists(path)

# Define a function to check if an image file exists
def check_image_exists(row):
    image_path = row['image_path']
    return check_exists(image_path)


def load_dicom(path):
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
# Define a custom test dataset class
class TestDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = self.dataframe['image_path'][index]
        image = load_dicom(image_path)  # Define this function to load your DICOM images
        if self.transform:
            image = self.transform(image)
        return image

# Define the transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# Create a test dataset and dataloader
test_dataset = TestDataset(expanded_test_desc, transform)
testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


for image in testloader:
    print(image.shape)
    break


import torch
from torchvision import models
import torch.nn as nn

class CustomResNet50(nn.Module):
    def __init__(self, num_classes=3, pretrained_weights=None):
        super(CustomResNet50, self).__init__()
        
        # Kendi modelinizi yükleyin (önceden eğitilmiş ağırlıklar olmadan)
        self.model = models.resnet50(pretrained=False).to(device)
        
        # Eğer manuel ağırlık yolu verilmişse, bu ağırlıkları yükle
        if pretrained_weights:
            self.model.load_state_dict(torch.load(pretrained_weights, map_location=device))
        
        # Son katmanı num_classes sayısına göre değiştirme
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

    def unfreeze_middle_layers(self):
        """Orta katmanları çöz."""
        for name, param in self.model.named_parameters():
            if 'layer3' in name or 'layer4' in name:  
                param.requires_grad = True
            else:
                param.requires_grad = False

"""!!!!!!!!!!!!!!!!!!!!!!!!BURADA YOLLARI OGRENMEN GEREK SABAH DUZELTCEKSIN!!!!!!!!!!!!!!!!!!!!!!!!"""
import torch

# Function to get the model based on series_description and load pretrained weights
def get_model(series_description, weights_paths):
    model_name = series_description
    if model_name in weights_paths:
        # Create a model instance (CustomResNet50 in this case)
        model = CustomResNet50(num_classes=3).to(device)
        
        # Load pretrained weights
        model.load_state_dict(torch.load(weights_paths[model_name], map_location=device))
        model.eval()  # Set the model to eval mode
        return model
    return None


# Function to make predictions on the test data
def predict_test_data(testloader, expanded_test_desc, weights_paths):
    predictions = []
    normal_mild_probs = []
    moderate_probs = []
    severe_probs = []
    
    for idx, images in enumerate(tqdm(testloader)):
        images = images.to(device)
        series_description = expanded_test_desc.iloc[idx]['series_description']
        model = get_model(series_description, weights_paths)
        
        if model:
            with torch.no_grad():
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1).squeeze(0)
                normal_mild_probs.append(probs[0].item())
                moderate_probs.append(probs[1].item())
                severe_probs.append(probs[2].item())
                predictions.append(probs)
        else:
            normal_mild_probs.append(None)
            moderate_probs.append(None)
            severe_probs.append(None)
            predictions.append(None)
    
    return normal_mild_probs, moderate_probs, severe_probs, predictions
weights_paths = {
    'Sagittal T1': '/kaggle/input/fork-of-fixed-train-with-severe-focused-augmentati/best_model_Sagittal_T1.pth',
    'Axial T2': '/kaggle/input/fork-of-fixed-train-with-severe-focused-augmentati/best_model_Axial_T2.pth',
    'Sagittal T2/STIR': '/kaggle/input/fork-of-fixed-train-with-severe-focused-augmentati/best_model_Sagittal_T2_STIR.pth'
}



# Make predictions on the test data
normal_mild_probs, moderate_probs, severe_probs, test_predictions = predict_test_data(testloader, expanded_test_desc, weights_paths)


test_predictions[0]


# Add predictions and probabilities to the test DataFrame
expanded_test_desc['normal_mild'] = normal_mild_probs
expanded_test_desc['moderate'] = moderate_probs
expanded_test_desc['severe'] = severe_probs


submission = expanded_test_desc[["row_id","normal_mild","moderate","severe"]]


# Group by 'row_id' and sum the values
grouped_submission = submission.groupby('row_id').max().reset_index()

# Normalize the columns
grouped_submission[['normal_mild', 'moderate', 'severe']] = grouped_submission[['normal_mild', 'moderate', 'severe']].div(grouped_submission[['normal_mild', 'moderate', 'severe']].sum(axis=1), axis=0)

# Check the first 3 rows
grouped_submission


# Dosya yolunu belirleyin
sample_submission_path = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/sample_submission.csv"

# sample_submission.csv'yi DataFrame olarak okuyun
sample_submission = pd.read_csv(sample_submission_path)
# Satırları sıralama
grouped_submission = grouped_submission.set_index('row_id').reindex(sample_submission['row_id']).reset_index()

# Normalizasyon
grouped_submission[['normal_mild', 'moderate', 'severe']] = grouped_submission[['normal_mild', 'moderate', 'severe']].div(
    grouped_submission[['normal_mild', 'moderate', 'severe']].sum(axis=1), axis=0
)


len(grouped_submission)


sub[['normal_mild', 'moderate', 'severe']] = grouped_submission[['normal_mild', 'moderate', 'severe']]


import os

# Save the DataFrame to "submission.csv" in the desired directory
sub.to_csv("/kaggle/working/submission.csv", index=False)


sub.head(5)

