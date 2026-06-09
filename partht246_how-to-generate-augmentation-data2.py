!python --version


!pip uninstall tensorflow -y
!pip uninstall numpy -y
!pip install pydub
!pip install natsort
!pip install spleeter-gpu==2.0.2
!pip install numpy==1.18.5


from spleeter.separator import Separator 


!bash /kaggle/input/400-500-script/script.sh


/kaggle/working/StemOutput400_500


import shutil

shutil.make_archive('/kaggle/working/StemOutput400_500', 'zip', '/kaggle/working/StemOutput400_500')

