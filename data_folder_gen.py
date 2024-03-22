
import os


root_dir = 'C:/Users/mvmigem/Documents/data/project_1/raw_data/'
n_sub = 40
sub_dir_gro = ['eeg','behav']

for i in range(1,n_sub+1):
    sub_path  = os.path.join(root_dir, f'sub_{i}')
    for j in sub_dir_gro:
        spec_sub_path = os.path.join(sub_path, j)
        sub_dir  = os.makedirs(spec_sub_path)
