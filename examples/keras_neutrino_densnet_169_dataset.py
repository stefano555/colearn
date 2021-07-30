# ------------------------------------------------------------------------------
#
#   Copyright 2021 Fetch.AI Limited
#
#   Licensed under the Creative Commons Attribution-NonCommercial International
#   License, Version 4.0 (the "License"); you may not use this file except in
#   compliance with the License. You may obtain a copy of the License at
#
#       http://creativecommons.org/licenses/by-nc/4.0/legalcode
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
####check memory usage

from guppy import hpy

heap = hpy()

print("Heap Status At Starting : ")
heap_status1 = heap.heap()
print("Heap Size : ", heap_status1.size, " bytes\n")
print(heap_status1)


import os

import tensorflow as tf
import tensorflow_datasets as tfds

#print(tf.config.experimental.get_memory_growth('CPU'))
#print(tf.config.experimental.get_memory_info('CPU'))


from colearn.training import initial_result, collective_learning_round, set_equal_weights
from colearn.utils.plot import ColearnPlot
from colearn.utils.results import Results, print_results
from colearn_keras.keras_learner import KerasLearner
from colearn_keras.utils import normalize_img

import numpy as np

#import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K

from pathlib import Path
from PIL import Image
import pandas as pd

from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPool2D, GlobalAveragePooling2D, Cropping2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model, Sequential, model_from_json
from tensorflow.keras.optimizers import SGD, Adam


print(tf.__version__)

"""
MNIST training example using Keras

Used dataset:
- MNIST is set of 60 000 black and white hand written digits images of size 28x28x1 in 10 classes

What script does:
- Loads MNIST dataset from Keras
- Sets up a Keras learner
- Randomly splits dataset between multiple learners
- Does multiple rounds of learning process and displays plot with results
"""

n_learners = 3
vote_threshold = 0.5
vote_batches = 2

testing_mode = bool(os.getenv("COLEARN_EXAMPLES_TEST", ""))  # for testing
n_rounds = 20 if not testing_mode else 1
width = 28
height = 28
n_classes = 10
l_rate = 0.001
batch_size = 8

# Load data for each learner
#train_dataset, info = tfds.load('mnist', split='train', as_supervised=True, with_info=True)
#n_datapoints = info.splits['train'].num_examples

#train_datasets = [train_dataset.shard(num_shards=n_learners, index=i) for i in range(n_learners)]

#test_dataset = tfds.load('mnist', split='test', as_supervised=True)
#test_datasets = [test_dataset.shard(num_shards=n_learners, index=i) for i in range(n_learners)]


#for i in range(n_learners):
#    train_datasets[i] = train_datasets[i].map(
#        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#    train_datasets[i] = train_datasets[i].shuffle(n_datapoints // n_learners)
#    train_datasets[i] = train_datasets[i].batch(batch_size)

#    test_datasets[i] = test_datasets[i].map(
#        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#    test_datasets[i] = test_datasets[i].batch(batch_size)
########################################################################################
#import my neutrino dataset
#this is done in an incredible ugly way but I'll fix that later

####removing hardcoded paths

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

print(y_train_bin)

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

####I shuffle the dataset and divide into three subsets
###all subsets will have 5000 images, but I will reshuffle the set at each round to ensure each subset is different

from sklearn.utils import shuffle

n_pics_subset=1000

x_train_sub_list = []
y_train_sub_list = []
###for some absurde reason it works only with two subsets and breaks with 3, no matter how small the subsets...


x_train, y_train_bin = shuffle(x_train, y_train_bin)

for i in range(n_pics_subset):
	x_train_sub_list.append(x_train[i])
	y_train_sub_list.append(y_train_bin[i])
	
x_train_sub = np.array(x_train_sub_list)
y_train_sub = np.array(y_train_sub_list)

x_train_sub_list = None
del x_train_sub_list
y_train_sub_list = None
del y_train_sub_list



#print("\nHeap Status After Importing dataset : ")
#heap_status2 = heap.heap()
#print("Heap Size : ", heap_status2.size, " bytes\n")
#print(heap_status2)

###################################
#creating a keras dataset

train_dataset = tf.data.Dataset.from_tensor_slices((x_train_sub, y_train_sub))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test_bin))

#train_dataset = train_dataset.shuffle(10000, reshuffle_each_iteration=False)
test_dataset = test_dataset.shuffle(2000, reshuffle_each_iteration=False)


train_datasets = [train_dataset.shard(num_shards=n_learners, index=i) for i in range(n_learners)]
test_datasets = [test_dataset.shard(num_shards=n_learners, index=i) for i in range(n_learners)]

train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)


#####################################################finish importing an shaping neutrino dataset################
#from sklearn.utils import shuffle

#x_train, y_train = shuffle(x_train, y_train)

#number_pics = 3333

#x_test_a = []
#x_test_b = []
#x_test_c = []
#y_test_a = []
#y_test_b = []
#y_test_c = []

#for i in range(number_pics):
#	x_test_a.append(x_test[i])
#	y_test_a.append(y_test[i])
#	x_test_b.append(x_test[i+number_pics])
#	y_test_b.append(y_test[i+number_pics])	
#	x_test_c.append(x_test[i+(2*number_pics)])
#	y_test_c.append(y_test[i+(2*number_pics)])	
################################################################################

# Define model
#def get_model():
 #   input_img = tf.keras.Input(
#        shape=(width, height, 1), name="Input"
#    )
#    x = tf.keras.layers.Conv2D(
#        64, (3, 3), activation="relu", padding="same", name="Conv1_1"
#    )(input_img)
#    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
#    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
#    x = tf.keras.layers.Conv2D(
#        128, (3, 3), activation="relu", padding="same", name="Conv2_1"
#    )(x)
#    x = tf.keras.layers.BatchNormalization(name="bn4")(x)
#    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
#    x = tf.keras.layers.Flatten(name="flatten")(x)
#    x = tf.keras.layers.Dense(
#        n_classes, activation="softmax", name="fc1"
#    )(x)
#    model = tf.keras.Model(inputs=input_img, outputs=x)

 #   opt = tf.keras.optimizers.Adam(lr=l_rate)
 #   model.compile(
 #       loss="sparse_categorical_crossentropy",
 #       metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
 #       optimizer=opt)
 #   return model
####define model densenet-169 
# One-hot encoding

# Define model
def get_model():
	num_classes = 3
	densenet_model = DenseNet169(include_top=False, weights=None, classes=3, pooling='avg', input_shape=input_shape)
	x = densenet_model.output
	#x = GlobalAveragePooling2D()(x)
	x = Dropout(0.5)(x)
	predictions = Dense(num_classes, activation= 'softmax')(x)
	densenet_model = Model(inputs = densenet_model.input, outputs = predictions)
	adam = Adam(lr=0.000001)
	densenet_model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['categorical_accuracy']) 

	return densenet_model

all_learner_models = []
for i in range(n_learners):
    all_learner_models.append(KerasLearner(
        model=get_model(),
        train_loader = train_dataset,
        test_loader=test_dataset,
        criterion="categorical_accuracy",
        minimise_criterion=False,
        model_evaluate_kwargs={"steps": vote_batches},
        #model_fit_kwargs={"steps": vote_batches},
    ))

set_equal_weights(all_learner_models)


#print("\nHeap Status Right Before using collective learning : ")
#heap_status3 = heap.heap()
#print("Heap Size : ", heap_status3.size, " bytes\n")
#print(heap_status3)

# Train the model using Collective Learning
results = Results()
results.data.append(initial_result(all_learner_models))

plot = ColearnPlot(score_name=all_learner_models[0].criterion)

for round_index in range(n_rounds):
    results.data.append(
        collective_learning_round(all_learner_models,
                                  vote_threshold, round_index)
    )

    print_results(results)
    plot.plot_results_and_votes(results)

plot.block()

print("Colearn Example Finished!")
