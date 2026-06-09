!rm -rf VRUID-AAAI-DAKiet
!git clone https://github.com/ffyyytt/VRUID-AAAI-DAKiet.git


!python /kaggle/working/VRUID-AAAI-DAKiet/predict.py -category table


!python /kaggle/working/VRUID-AAAI-DAKiet/predict.py -category figure


!python /kaggle/working/VRUID-AAAI-DAKiet/predict.py -category form


!python /kaggle/working/VRUID-AAAI-DAKiet/predict.py -category form_body


!python /kaggle/working/VRUID-AAAI-DAKiet/predict.py -category list


!python /kaggle/working/VRUID-AAAI-DAKiet/generate_submission.py

