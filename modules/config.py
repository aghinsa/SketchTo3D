'''
Config params
'''
import os

main_dir = os.path.join(os.getcwd(),'data_char')
name_list_path = os.path.join(os.getcwd(),'data_char/list.txt')
sketch_dir = os.path.join(os.getcwd(),'data_char/sketch')
dnfs_dir = os.path.join(os.getcwd(),'data_char/dnfs')

# Loss tuning
lambda_pixel=1
lambda_adv=0.01
# Train configs
training_iter = 2
batch_size = 2
learning_rate = .0001

# Data
prefetch_buffer_size = 64
num_parallel_batches = 8
