# # IMPORTANT: SOME KAGGLE DATA SOURCES ARE PRIVATE
# # RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES.
!pip install kagglehub
import kagglehub



# Import dataset
state_farm_distracted_driver_detection_path = kagglehub.competition_download("state-farm-distracted-driver-detection")

print('Data source import complete.')

