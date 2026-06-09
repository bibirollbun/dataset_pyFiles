!python --version


!pip uninstall tensorflow -y
!pip uninstall numpy -y
!pip install pydub
!pip install natsort
!pip install spleeter-gpu==2.0.2
!pip install numpy==1.18.5


from spleeter.separator import Separator 


!bash /kaggle/input/script500-600/script.sh


import shutil

shutil.make_archive('/kaggle/working/StemOutput500600', 'zip', '/kaggle/working/StemOutput500600')


!zip stemoutput300400.zip /kaggle/working/StemOutput

