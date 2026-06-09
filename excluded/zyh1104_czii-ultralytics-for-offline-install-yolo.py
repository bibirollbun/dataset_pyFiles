!pip download -d ./packages ultralytics
!tar cfvz archive.tar.gz ./packages


!tar xfvz archive.tar.gz
!pip install --no-index --find-links=./packages ultralytics
!rm -rf ./packages 


from ultralytics import YOLO


#!tar xfvz /kaggle/input/ultralytics-for-offline-install/archive.tar.gz
#!pip install --no-index --find-links=./packages ultralytics
#!rm -rf ./packages


#from ultralytics import YOLO

