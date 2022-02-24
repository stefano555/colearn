import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds
#import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K

from pathlib import Path
from PIL import Image
import pandas as pd

path_dir: str = r"/home/stefanovergani/colearn_2021_04_16/data"

x_train_temp_0 = np.load(os.sep.join([path_dir, 'x_train_numu_500_0.npy']))
y_train_0 = np.load(os.sep.join([path_dir, 'y_train_numu_500_event_0.npy']))

x_train_temp_1 = np.load(os.sep.join([path_dir, 'x_train_numu_500_1.npy']))
y_train_1 = np.load(os.sep.join([path_dir, 'y_train_numu_500_event_1.npy']))

x_train_temp_2 = np.load(os.sep.join([path_dir, 'x_train_numu_500_2.npy']))
y_train_2 = np.load(os.sep.join([path_dir, 'y_train_numu_500_event_2.npy']))

x_train_numu_3 = np.load(os.sep.join([path_dir, 'x_train_numu_500_3.npy']))
y_train_numu_3 = np.load(os.sep.join([path_dir, 'y_train_numu_500_event_3.npy']))

x_train_numu_4 = np.load(os.sep.join([path_dir, 'x_train_numu_500_4.npy']))
y_train_numu_4 = np.load(os.sep.join([path_dir, 'y_train_numu_500_event_4.npy']))

x_train_numu_5 = np.load(os.sep.join([path_dir, 'x_train_numu_500_5.npy']))
y_train_numu_5 = np.load(os.sep.join([path_dir, 'y_train_numu_500_event_5.npy']))

x_train_numu_6 = np.load(os.sep.join([path_dir, 'x_train_numu_500_6.npy']))
y_train_numu_6 = np.load(os.sep.join([path_dir, 'y_train_numu_500_event_6.npy']))

x_train_numu_7 = np.load(os.sep.join([path_dir, 'x_train_numu_500_7.npy']))
y_train_numu_7 = np.load(os.sep.join([path_dir, 'y_train_numu_500_event_7.npy']))

x_train_numu_8 = np.load(os.sep.join([path_dir, 'x_train_numu_500_8.npy']))
y_train_numu_8 = np.load(os.sep.join([path_dir, 'y_train_numu_500_event_8.npy']))

x_train_numu_9 = np.load(os.sep.join([path_dir, 'x_train_numu_500_9.npy']))
y_train_numu_9 = np.load(os.sep.join([path_dir, 'y_train_numu_500_event_9.npy']))

x_train_temp_3 = np.load(os.sep.join([path_dir, 'x_train_nue_500_0.npy']))
y_train_3 = np.load(os.sep.join([path_dir, 'y_train_nue_500_event_0.npy']))

x_train_temp_4 = np.load(os.sep.join([path_dir, 'x_train_nue_500_1.npy']))
y_train_4 = np.load(os.sep.join([path_dir, 'y_train_nue_500_event_1.npy']))

x_train_temp_5 = np.load(os.sep.join([path_dir, 'x_train_nue_500_2.npy']))
y_train_5 = np.load(os.sep.join([path_dir, 'y_train_nue_500_event_2.npy']))

x_train_temp_6 = np.load(os.sep.join([path_dir, 'x_train_nue_500_3.npy']))
y_train_6 = np.load(os.sep.join([path_dir, 'y_train_nue_500_event_3.npy']))

x_train_nue_4 = np.load(os.sep.join([path_dir, 'x_train_nue_500_4.npy']))
y_train_nue_4 = np.load(os.sep.join([path_dir, 'y_train_nue_500_event_4.npy']))

x_train_nue_5 = np.load(os.sep.join([path_dir, 'x_train_nue_500_5.npy']))
y_train_nue_5 = np.load(os.sep.join([path_dir, 'y_train_nue_500_event_5.npy']))

x_train_nue_6 = np.load(os.sep.join([path_dir, 'x_train_nue_500_6.npy']))
y_train_nue_6 = np.load(os.sep.join([path_dir, 'y_train_nue_500_event_6.npy']))

x_train_nue_7 = np.load(os.sep.join([path_dir, 'x_train_nue_500_7.npy']))
y_train_nue_7 = np.load(os.sep.join([path_dir, 'y_train_nue_500_event_7.npy']))

x_train_nue_8 = np.load(os.sep.join([path_dir, 'x_train_nue_500_8.npy']))
y_train_nue_8 = np.load(os.sep.join([path_dir, 'y_train_nue_500_event_8.npy']))

x_train_nue_9 = np.load(os.sep.join([path_dir, 'x_train_nue_500_9.npy']))
y_train_nue_9 = np.load(os.sep.join([path_dir, 'y_train_nue_500_event_9.npy']))
                         
x_test_numu_0 = np.load(os.sep.join([path_dir, 'x_test_numu_500_0.npy']))
y_test_numu_0 = np.load(os.sep.join([path_dir, 'y_test_numu_500_event_0.npy']))

x_test_numu_1 = np.load(os.sep.join([path_dir, 'x_test_numu_500_1.npy']))
y_test_numu_1 = np.load(os.sep.join([path_dir, 'y_test_numu_500_event_1.npy']))

x_test_nue_0 = np.load(os.sep.join([path_dir, 'x_test_nue_500_0.npy']))
y_test_nue_0 = np.load(os.sep.join([path_dir, 'y_test_nue_500_event_0.npy']))

x_test_nue_1 = np.load(os.sep.join([path_dir, 'x_test_nue_500_1.npy']))
y_test_nue_1 = np.load(os.sep.join([path_dir, 'y_test_nue_500_event_1.npy']))
    

#concatenate all the files together and then delete from the memory unnecessary files
x_train_temp_a=np.concatenate((x_train_temp_0, x_train_temp_1))
x_train_temp_b=np.concatenate((x_train_temp_a, x_train_temp_2))
x_train_temp_0 = None
del x_train_temp_0
x_train_temp_1 = None
del x_train_temp_1
x_train_temp_2 = None
del x_train_temp_2
x_train_temp_a = None
del x_train_temp_a
x_train_temp_c=np.concatenate((x_train_temp_b, x_train_temp_3))
x_train_temp_b=None
del x_train_temp_b
x_train_temp_d=np.concatenate((x_train_temp_c, x_train_temp_4))
x_train_temp_c=None
del x_train_temp_c
x_train_temp_e=np.concatenate((x_train_temp_d, x_train_temp_5))
x_train_temp_3 = None
del x_train_temp_3
x_train_temp_4 = None
del x_train_temp_4
x_train_temp_5 = None
del x_train_temp_5
x_train_temp_d = None
del x_train_temp_d
x_train_temp_f=np.concatenate((x_train_temp_e, x_train_temp_6))
x_train_temp_e=None
del x_train_temp_e
x_train_temp_6 = None
del x_train_temp_6
x_train_temp_g=np.concatenate((x_train_temp_f, x_train_numu_3))
x_train_temp_f=None
del x_train_temp_f
x_train_numu_3 = None
del x_train_numu_3
x_train_temp_h=np.concatenate((x_train_temp_g, x_train_nue_4))
x_train_temp_g=None
del x_train_temp_g
x_train_nue_4 = None
del x_train_nue_4
x_train_temp_i=np.concatenate((x_train_temp_h, x_train_numu_4))
x_train_temp_h=None
del x_train_temp_h
x_train_numu_4 = None
del x_train_numu_4
x_train_temp_l=np.concatenate((x_train_temp_i, x_train_nue_5))
x_train_temp_i=None
del x_train_temp_i
x_train_nue_5 = None
del x_train_nue_5
x_train_temp_m=np.concatenate((x_train_temp_l, x_train_numu_5))
x_train_temp_l=None
del x_train_temp_l
x_train_numu_5 = None
del x_train_numu_5
x_train_temp_n=np.concatenate((x_train_temp_m, x_train_nue_6))
x_train_temp_m=None
del x_train_temp_m
x_train_nue_6 = None
del x_train_nue_6
x_train_temp_o=np.concatenate((x_train_temp_n, x_train_numu_6))
x_train_temp_n=None
del x_train_temp_n
x_train_numu_6 = None
del x_train_numu_6
x_train_temp_p=np.concatenate((x_train_temp_o, x_train_nue_7))
x_train_temp_o=None
del x_train_temp_o
x_train_nue_7 = None
del x_train_nue_7
x_train_temp_q=np.concatenate((x_train_temp_p, x_train_numu_7))
x_train_temp_p=None
del x_train_temp_p
x_train_numu_7 = None
del x_train_numu_7
x_train_temp_r=np.concatenate((x_train_temp_q, x_train_nue_8))
x_train_temp_q=None
del x_train_temp_q
x_train_nue_8 = None
del x_train_nue_8
x_train_temp_s=np.concatenate((x_train_temp_r, x_train_numu_8))
x_train_temp_r=None
del x_train_temp_r
x_train_numu_8 = None
del x_train_numu_8
x_train_temp_t=np.concatenate((x_train_temp_s, x_train_nue_9))
x_train_temp_s=None
del x_train_temp_s
x_train_nue_9 = None
del x_train_nue_9
x_train_temp=np.concatenate((x_train_temp_t, x_train_numu_9))
x_train_temp_t=None
del x_train_temp_t
x_train_numu_9 = None
del x_train_numu_9



x_test_temp_a=np.concatenate((x_test_numu_0, x_test_nue_0))
x_test_numu_0 = None
del x_test_numu_0
x_test_nue_0 = None
del x_test_nue_0
x_test_temp_b=np.concatenate((x_test_temp_a, x_test_nue_1))
x_test_temp_a = None
del x_test_temp_a
x_test_nue_1 = None
del x_test_nue_1
x_test_temp=np.concatenate((x_test_temp_b, x_test_numu_1))
x_test_temp_b = None
del x_test_temp_b
x_test_numu_1 = None
del x_test_numu_1

y_train_a=np.concatenate((y_train_0, y_train_1))
y_train_0 = None
del y_train_0
y_train_1 = None
del y_train_1
y_train_b=np.concatenate((y_train_a, y_train_2))
y_train_c=np.concatenate((y_train_b, y_train_3))
y_train_a=None
del y_train_a
y_train_b=None
del y_train_b
y_train_d=np.concatenate((y_train_c, y_train_4))
y_train_c=None
del y_train_c
y_train_e=np.concatenate((y_train_d, y_train_5))
y_train_2 = None
del y_train_2
y_train_3 = None
del y_train_3
y_train_4 = None
del y_train_4
y_train_5 = None
del y_train_5
y_train_d=None
del y_train_d
y_train_f=np.concatenate((y_train_e, y_train_6))
y_train_e=None
del y_train_e
y_train_6 = None
del y_train_6
y_train_g=np.concatenate((y_train_f, y_train_numu_3))
y_train_f=None
del y_train_f
y_train_numu_3 = None
del y_train_numu_3
y_train_h=np.concatenate((y_train_g, y_train_nue_4))
y_train_g=None
del y_train_g
y_train_nue_4 = None
del y_train_nue_4
y_train_i=np.concatenate((y_train_h, y_train_numu_4))
y_train_h=None
del y_train_h
y_train_numu_4 = None
del y_train_numu_4
y_train_l=np.concatenate((y_train_i, y_train_nue_5))
y_train_i=None
del y_train_i
y_train_nue_5 = None
del y_train_nue_5
y_train_m=np.concatenate((y_train_l, y_train_numu_5))
y_train_l=None
del y_train_l
y_train_numu_5 = None
del y_train_numu_5
y_train_n=np.concatenate((y_train_m, y_train_nue_6))
y_train_m=None
del y_train_m
y_train_nue_6 = None
del y_train_nue_6
y_train_o=np.concatenate((y_train_n, y_train_numu_6))
y_train_n=None
del y_train_n
y_train_numu_6 = None
del y_train_numu_6
y_train_p=np.concatenate((y_train_o, y_train_nue_7))
y_train_o=None
del y_train_o
y_train_nue_7 = None
del y_train_nue_7
y_train_q=np.concatenate((y_train_p, y_train_numu_7))
y_train_p=None
del y_train_p
y_train_numu_7 = None
del y_train_numu_7
y_train_r=np.concatenate((y_train_q, y_train_nue_8))
y_train_q=None
del y_train_q
y_train_nue_8 = None
del y_train_nue_8
y_train_s=np.concatenate((y_train_r, y_train_numu_8))
y_train_r=None
del y_train_r
y_train_numu_8 = None
del y_train_numu_8
y_train_t=np.concatenate((y_train_s, y_train_nue_9))
y_train_s=None
del y_train_s
y_train_nue_9 = None
del y_train_nue_9
y_train=np.concatenate((y_train_t, y_train_numu_9))
y_train_t=None
del y_train_t
y_train_numu_9 = None
del y_train_numu_9


y_test_a=np.concatenate((y_test_numu_0, y_test_nue_0))
y_test_numu_0 = None
del y_test_numu_0
y_test_nue_0 = None
del y_test_nue_0
y_test_b=np.concatenate((y_test_a, y_test_nue_1))
y_test_a = None
del y_test_a
y_test_nue_1 = None
del y_test_nue_1
y_test=np.concatenate((y_test_b, y_test_numu_1))
y_test_b = None
del y_test_b
y_test_numu_1 = None
del y_test_numu_1


print('x_train_temp.shape',x_train_temp.shape)
print('x_test_temp.shape',x_test_temp.shape)
print('y_train.shape',y_train.shape)
print('y_test.shape',y_test.shape)

from tensorflow.keras.utils import to_categorical
y_train_bin = to_categorical(y_train,3)
y_test_bin = to_categorical(y_test,3)


y_train=None
del y_train
y_test=None
del y_test

print('y_train_bin.shape',y_train_bin.shape)
print('y_test_bin.shape',y_test_bin.shape)


#passages are: x_train_temp -> x_train (min 0 max 1, float32) -> x_train_reshaped (channel first) -> x_train_tensor

#normalise the data
old_min = np.min(x_train_temp)
old_max = np.max(x_train_temp)
print("Before:", old_min, old_max)
x_train = x_train_temp.astype('float32')
x_test = x_test_temp.astype('float32')
x_train /=255
x_test /= 255
new_min = np.min(x_train)
new_max = np.max(x_train)
print("After:", new_min, new_max)

print("Test Data Shape after conversion to float")
    
    #read the dimensions from one example in the trainig set
img_rows, img_cols = x_train[0].shape[0], x_train[0].shape[1]
print(x_train[0].shape[0])
print(x_train[0].shape[1])

#read the dimensions from one example in the trainig set
img_rows, img_cols = x_train[0].shape[0], x_train[0].shape[1]

#read the dimensions from one example in the trainig set
img_rows, img_cols = x_train[0].shape[0], x_train[0].shape[1]
print(x_train[0].shape[0])
print(x_train[0].shape[1])

#read the dimensions from one example in the trainig set
img_rows, img_cols = x_train[0].shape[0], x_train[0].shape[1]

#Different NN libraries (e.g., TF) use different ordering of dimensions
#Here we set the "input shape" so that later the NN knows what shape to expect
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)
print("input_shape",input_shape)
print("x_train[0].shape[0]", x_train[0].shape[0])
print("x_train[0].shape[1]", x_train[0].shape[1])
print("x_train[0].shape[2]", x_train[0].shape[2])

x_train_temp = None
del x_train_temp
x_test_temp = None
del x_test_temp


np.save('x_train', x_train)
np.save('y_train_bin', y_train_bin)
np.save('x_test', x_test)
np.save('y_test_bin', y_test_bin)

