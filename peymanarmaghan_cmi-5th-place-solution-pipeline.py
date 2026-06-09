#============================================
# Data prepration
#============================================

!python /kaggle/input/data-prepration/cmi-prepare-data.py \
  --input_dir /kaggle/input/child-mind-institute-problematic-internet-use \
  --output_dir /kaggle/working 



#============================================
# Model Train
#============================================

!python /kaggle/input/data-prepration/cmi-train_code.py \
  --prep_script /kaggle/input/data-prepration/cmi-prepare-data.py \
  --prep_input_dir /kaggle/input/child-mind-institute-problematic-internet-use \
  --prep_output_dir /kaggle/working \
  --train_script_output_dir /kaggle/working


#============================================
# Generate Predictions
#============================================
!python /kaggle/input/data-prepration/cmi-prediction_code.py \
  --prep_script /kaggle/input/data-prepration/cmi-prepare-data.py \
  --prep_input_dir /kaggle/input/child-mind-institute-problematic-internet-use \
  --prep_output_dir /kaggle/working 





