# Copy all files from your uploaded dataset into the working directory
!cp -r /kaggle/input/rna-submission-bundle/kaggle/* /kaggle/working/
%cd /kaggle/working/

# Run inference using the provided test_sequences.csv from the competition
!python inference.py \
  --sequences /kaggle/input/stanford-rna-3d-folding/test_sequences.csv \
  --weights checkpoints/best_train_model.pt \
  --output submission.csv \
  --idcol target_id


