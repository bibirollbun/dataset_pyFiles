import numpy as np
import pickle
import os


!pip install --upgrade google-auth


from kaggle_datasets import KaggleDatasets


datasets_val = ['gwi-tfrecs-v-f32']


combined_path_val = [(KaggleDatasets().get_gcs_path(dataset)) 
                 for dataset in datasets_val]
pickle.dump(combined_path_val,open('combined_path_val.p', 'bw'))
combined_path_val


%%capture

tffiles_val = []
for idx, curr_tfrecords_path in enumerate(combined_path_val):
    print(idx)
    folders = !gsutil ls $curr_tfrecords_path
    for folder in folders:
        curr_tffiles = !gsutil ls $folder
        tffiles_indices = [int(x.split('/')[-1][:-9]) for x in curr_tffiles]
        curr_tffiles = [x for _, x in sorted(zip(tffiles_indices, curr_tffiles))]
        tffiles_val.append(curr_tffiles)


pickle.dump(tffiles_val, open('tffiles_val.p', 'bw'))


len(tffiles_val[0])*50


datasets = [f'gwi-tfrecs-test-ds-{x}' for x in range(1,3)]
combined_path = [(KaggleDatasets().get_gcs_path(dataset)) 
                 for dataset in datasets]
pickle.dump(combined_path,open('combined_path.p', 'bw'))
combined_path


%%capture

tffiles = []
for idx, curr_tfrecords_path in enumerate(combined_path):
    print(idx)
    folders = !gsutil ls $curr_tfrecords_path
    for folder in folders:
        folder = folder
        curr_tffiles = !gsutil ls $folder
        tffiles_indices = [int(x.split('/')[-1][:-9]) for x in curr_tffiles]
        curr_tffiles = [x for _, x in sorted(zip(tffiles_indices, curr_tffiles))]
        tffiles.append(curr_tffiles)


print(np.sum([len(x) for x in tffiles]))
print(50*np.sum([len(x) for x in tffiles]))
pickle.dump(tffiles, open('tffiles_test.p', 'bw'))


datasets = ['gwi-tfrecs-example']
combined_path = [(KaggleDatasets().get_gcs_path(dataset)) 
                 for dataset in datasets]
pickle.dump(combined_path,open('combined_path.p', 'bw'))
combined_path


%%capture

tffiles = []
for idx, curr_tfrecords_path in enumerate(combined_path):
    print(idx)
    folders = !gsutil ls $curr_tfrecords_path
    for folder in folders:
        folder = folder
        curr_tffiles = !gsutil ls $folder
        tffiles_indices = [int(x.split('/')[-1][:-9]) for x in curr_tffiles]
        curr_tffiles = [x for _, x in sorted(zip(tffiles_indices, curr_tffiles))]
        tffiles.append(curr_tffiles)


print(np.sum([len(x) for x in tffiles]))
print(50*np.sum([len(x) for x in tffiles]))
pickle.dump(tffiles, open('tffiles_example.p', 'bw'))

