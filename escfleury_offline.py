!cp -r /kaggle/input/offline-scripts/kaggle/* /kaggle/working/
%cd /kaggle/working/

!python inference.py \
  --sequences /kaggle/input/stanford-rna-3d-folding/test_sequences.csv \
  --weights checkpoints/best_train_model.pt \
  --output submission.csv \
  --idcol target_id


