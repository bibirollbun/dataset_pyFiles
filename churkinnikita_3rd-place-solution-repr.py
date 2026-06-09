!pip install -q /kaggle/input/cibmtr-competition/qhoptim-1.1.0-py3-none-any.whl
!pip install pytorch_tabular -q --no-index --find-links=/kaggle/input/cibmtr-competition
!pip install rtdl_num_embeddings -q --no-index --find-links=/kaggle/input/cibmtr-competition/rtdl_num_embeddings


%%writefile SETTINGS.json
{
  "TRAIN_DATA_PATH": "/kaggle/input/equity-post-HCT-survival-predictions/train.csv",
  "TEST_DATA_PATH": "/kaggle/input/equity-post-HCT-survival-predictions/test.csv",
  "SAMPLE_SUBMISSION_PATH": "/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv",
  "LGB_ZEROES_MODELS_DIR": "/kaggle/input/cibmtr-models-reproduce/lgb_zeroes",
  "LGB_ONES_MODELS_DIR": "/kaggle/input/cibmtr-models-reproduce/lgb_ones",
  "LGB_CLF_MODELS_DIR": "/kaggle/input/cibmtr-models-reproduce/lgb_clf",
  "XGB_ZEROES_MODELS_DIR": "/kaggle/input/cibmtr-models-reproduce/xgb_zeroes",
  "XGB_ONES_MODELS_DIR": "/kaggle/input/cibmtr-models-reproduce/xgb_ones",
  "XGB_CLF_MODELS_DIR": "/kaggle/input/cibmtr-models-reproduce/xgb_clf",
  "CATBOOST_ZEROES_MODELS_DIR": "/kaggle/input/cibmtr-models-reproduce/catboost_zeroes",
  "CATBOOST_ONES_MODELS_DIR": "/kaggle/input/cibmtr-models-reproduce/catboost_ones",
  "CATBOOST_CLF_MODELS_DIR": "/kaggle/input/cibmtr-models-reproduce/catboost_clf",
  "NN_MODELS_DIR": "/kaggle/input/cibmtr-models-reproduce/nn",
  "TEST_PREDICTIONS_DIR": "./test_predictions",
  "SUBMISSION_DIR": "./"
}


!python /kaggle/input/cibmtr-scripts-reproduce/predict_nn.py

!python /kaggle/input/cibmtr-scripts-reproduce/predict_catboost_ones.py
!python /kaggle/input/cibmtr-scripts-reproduce/predict_catboost_zeroes.py
!python /kaggle/input/cibmtr-scripts-reproduce/predict_catboost_clf.py

!python /kaggle/input/cibmtr-scripts-reproduce/predict_lgb_ones.py
!python /kaggle/input/cibmtr-scripts-reproduce/predict_lgb_zeroes.py
!python /kaggle/input/cibmtr-scripts-reproduce/predict_lgb_clf.py

!python /kaggle/input/cibmtr-scripts-reproduce/predict_xgb_ones.py
!python /kaggle/input/cibmtr-scripts-reproduce/predict_xgb_zeroes.py
!python /kaggle/input/cibmtr-scripts-reproduce/predict_xgb_clf.py

!python /kaggle/input/cibmtr-scripts-reproduce/blend.py


!head submission.csv

