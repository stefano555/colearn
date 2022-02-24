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

#from guppy import hpy

#heap = hpy()

#print("Heap Status At Starting : ")
#heap_status1 = heap.heap()
#print("Heap Size : ", heap_status1.size, " bytes\n")
#print(heap_status1)


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
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
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

n_learners = 2
vote_threshold = 0.5
vote_batches = 2

testing_mode = bool(os.getenv("COLEARN_EXAMPLES_TEST", ""))  # for testing
n_rounds = 15 if not testing_mode else 1
#width = 28
#height = 28
n_classes = 3
l_rate = 0.01
batch_size = 2

####import neutrino dataset

path_dir: str = r"/home/stefanovergani/colearn_2021_04_16/data/entire"

x_train = np.load(os.sep.join([path_dir, 'x_train.npy']))
y_train_bin = np.load(os.sep.join([path_dir, 'y_train_bin.npy']))
x_test = np.load(os.sep.join([path_dir, 'x_test.npy']))
y_test_bin = np.load(os.sep.join([path_dir, 'y_test_bin.npy']))

img_rows, img_cols = x_train[0].shape[0], x_train[0].shape[1]

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)


####I shuffle the dataset and divide into three subsets
###all subsets will have 5000 images, but I will reshuffle the set at each round to ensure each subset is different

from sklearn.utils import shuffle

n_pics_subset=10000
n_test_subset=2000

x_train_sub_list = []
y_train_sub_list = []
x_test_sub_list = []
y_test_sub_list = []

#pics_per_shard = n_pics_subset/n_learners

x_train, y_train_bin = shuffle(x_train, y_train_bin)
x_test, y_test_bin = shuffle(x_test, y_test_bin)

for i in range(n_pics_subset):
	x_train_sub_list.append(x_train[i])
	y_train_sub_list.append(y_train_bin[i])
	
for j in range(n_test_subset):
	x_test_sub_list.append(x_test[j])
	y_test_sub_list.append(y_test_bin[j])
	
x_train_sub = np.array(x_train_sub_list)
y_train_sub = np.array(y_train_sub_list)
x_test_sub = np.array(x_test_sub_list)
y_test_sub = np.array(y_test_sub_list)

x_train_sub_list = None
del x_train_sub_list
y_train_sub_list = None
del y_train_sub_list
x_test_sub_list = None
del x_test_sub_list
y_test_sub_list = None
del y_test_sub_list

x_test = None
del x_test
y_test_bin = None
del y_test_bin
x_train = None
del x_train
y_train_bin = None
del y_train_bin

###################################
#creating a keras dataset

#this first version uses all of the data but uses too much memory
#train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_bin))
#test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test_bin))
#this version takes only a subset of the original data
train_datasets = tf.data.Dataset.from_tensor_slices((x_train_sub, y_train_sub))
test_datasets = tf.data.Dataset.from_tensor_slices((x_test_sub, y_test_sub))

x_train_sub = None
del x_train_sub
y_train_sub = None
del y_train_sub
x_test_sub = None
del x_test_sub
y_test_sub = None
del y_test_sub

#train_dataset = train_dataset.shuffle(n_pics_subset, reshuffle_each_iteration=False)
#test_dataset = test_dataset.shuffle(n_test_subset, reshuffle_each_iteration=False)

train_datasets = [train_datasets.shard(num_shards=n_learners, index=i) for i in range(n_learners)]
test_datasets = [test_datasets.shard(num_shards=n_learners, index=i) for i in range(n_learners)]

for i in range(n_learners):

	train_datasets[i] = train_datasets[i].batch(batch_size)
	test_datasets[i] = test_datasets[i].batch(batch_size)



#train_dataset = train_dataset.batch(batch_size)
#test_dataset = test_dataset.batch(batch_size)


####define model densenet-169 
# One-hot encoding

# Define model
def get_model_densenet():
	num_classes = 3
	densenet_model = DenseNet169(include_top=False, weights=None, classes=3, pooling='avg', input_shape=input_shape)
	x = densenet_model.output
	#x = GlobalAveragePooling2D()(x)
	x = Dropout(0.5)(x)
	predictions = Dense(num_classes, activation= 'softmax')(x)
	densenet_model = Model(inputs = densenet_model.input, outputs = predictions)
	adam = Adam(lr=0.0001)#was 0.000001
	densenet_model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['categorical_accuracy']) 

	return densenet_model
	
def get_model_resnet():
	num_classes = 3
	res_model = ResNet50V2(include_top=False, weights=None, classes=3, pooling='avg', input_shape=input_shape)
	x = res_model.output
	#x = GlobalAveragePooling2D()(x)
	x = Dropout(0.8)(x)
	predictions = Dense(num_classes, activation= 'softmax')(x)
	res_model = Model(inputs = res_model.input, outputs = predictions)

	learning_rate = 0.1
	decay_rate = learning_rate / 100
	momentum = 0.9
	sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
	adam = Adam(lr=0.000001)
	res_model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['categorical_accuracy']) 

	return res_model
	
#import timeit

#start = timeit.timeit()

all_learner_models = []
for i in range(n_learners):
    all_learner_models.append(KerasLearner(
        model=get_model_densenet(),
        train_loader = train_datasets[i],
        test_loader=test_datasets[i],
        criterion="categorical_accuracy",
        minimise_criterion=False,#False
        model_evaluate_kwargs={"steps": vote_batches},
        #model_fit_kwargs={"steps": vote_batches},
    ))

set_equal_weights(all_learner_models)


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
    
#end = timeit.timeit()
#print('total time ',end - start)

plot.savefig()
plot.block()


print("Colearn Example Finished!")


