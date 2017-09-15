
# coding: utf-8

# In[16]:

from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# In[17]:

#shape argument is optional
def weight_variable(shape,name=None):
    initializer = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initializer,name=name)

def bias_variable(shape, name=None):
    initializer = tf.constant(0.1, shape=shape)
    return tf.Variable(initializer, name=name)

def offset_variable(shape, name=None):
    print(shape)
    initializer = tf.constant(0.1, shape=shape)
    return tf.Variable(initializer, name=name)

def conv2d(x, W, strides, name=None):
    return tf.nn.conv2d(x, filter=W, strides=strides, padding='VALID', data_format="NCHW", name=name)

def max_pool_2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 2],
                        strides=[1, 1, 2, 2], padding='SAME', data_format="NCHW", name=name)

def batch_normalization(x, offset=None, name = None):
    mean, variance = tf.nn.moments(x, axes=[0, 1, 2], shift=None, name=None, keep_dims=False) 
    #momnets -- used with convolutional filters with shape [batch, height, width, depth]
    return tf.nn.batch_normalization(x, mean, variance, offset=None, scale=None, variance_epsilon = 1e-6, name=name)

# In[18]:

# Read data from csv file.
# Path to all data:  /global/cscratch1/sd/muszyng/data_astronomy_catalogs/trainingData.csv
df = np.genfromtxt('trainingData.csv', delimiter=" ")


# In[19]:

print(df.shape)


# In[20]:

# Create array of labels.
labels = df[:,-1]
print(labels)


# In[21]:

# Create binary matrix of labels.
nb_classes = 4
mat_labels = np.zeros([nb_classes, len(labels)], dtype=int)
print(mat_labels)


# In[22]:

# Convert labels to binary matrix.
for i in range(0, len(labels)):
    l = int(labels[i])
    mat_labels[l, i] = 1
print(mat_labels)

# In[29]:

train_image_size = df.shape[1] #input size 
channel = 1
num_classes = 4
image_data =  tf.convert_to_tensor(df)
labels_one_hot = np.transpose(mat_labels, [1,0])

train_size = 2100000 #number of samples for training= total number of samples-test samples 
train_images = image_data[:train_size]
train_labels = labels_one_hot[:train_size]
test_images = image_data[train_size:]
test_labels = labels_one_hot[train_size:]
print(image_data.shape)


# In[31]:

#model begins
run_metadata = tf.RunMetadata()
inforstring = os.getenv('inforstring')
trace_file = open('timeline'+str(inforstring)+'.ctf.json', 'w')

tf.app.flags.DEFINE_boolean('trace_flag', True, """If set, it produces an a trace of the threads  executing work during the training phase.""")
tf.app.flags.DEFINE_integer('epoch', 10, """Epochs to train the model.""")

x_train = np.float32(train_images)
test_images = np.float32(test_images)
y_train = np.float32(train_labels)


# In[9]:

#init placeholders
x = tf.placeholder(tf.float32, shape = [None, train_image_size ]) #change this to the vector shape
y = tf.placeholder(tf.float32, shape = [None, num_classes])



#dense, initializing the variables
W_fc1 = weight_variable([64, 32],name="W_fc1")
b_fc1 = bias_variable([32],name="b_fc1")
W_fc2 = weight_variable([32, 4],name="W_fc2")
b_fc2 = bias_variable([4],name="b_fc1")

# In[12]:

x_input = tf.reshape(x, [-1, train_image_size])

#dense 1
h_fc1 = tf.nn.relu(tf.matmul(x_input, W_fc1) + b_fc1)
dropout = tf.layers.dropout(h_fc1, rate=0.5)
#dense 2
y_conv = tf.matmul(dropout, W_fc2) + b_fc2


#graph construction done

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# In[15]:

if (os.getenv('NUM_INTER_THREADS', None) is not None and os.getenv('NUM_INTRA_THREADS', None) is not None):
    print("Custom NERSC/Intel config:inter_op_parallelism_threads({}),""intra_op_parallelism_threads({})".format(os.environ['NUM_INTER_THREADS'],os.environ['NUM_INTRA_THREADS']))
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=int(os.environ['NUM_INTER_THREADS']),intra_op_parallelism_threads=int(os.environ['NUM_INTRA_THREADS'])))


#batch size = all data entries
FLAGS = tf.app.flags.FLAGS
epoch = FLAGS.epoch

batch_size = 100
max_step = epoch*train_size/batch_size
print("max steps: ", max_step)
with tf.Session() as sess:
    if FLAGS.trace_flag:
        sess.run(tf.global_variables_initializer(),  options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
    else:
        sess.run(tf.global_variables_initializer())
    for i in range(max_step):
        train_step.run(feed_dict={x: x_train[i*batch_size%train_size:i*batch_size%train_size+batch_size], y: y_train[i*batch_size%train_size:i*batch_size%train_size+batch_size,:]})
        if ((i+1)*batch_size) % train_size == 0:
            correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("Training Accuracy:")
            if FLAGS.trace_flag:
                print(sess.run(accuracy, feed_dict={x: x_train, y: y_train}, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata))
            else:
                print(sess.run(accuracy, feed_dict={x: x_train, y: y_train}))
        
    train_step.run(feed_dict={x: test_images[:,:], y: test_labels[:,:]})
        
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Test Accuracy: \n")
    if FLAGS.trace_flag:
        print(sess.run(accuracy, feed_dict={x: test_images, y: test_labels}, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata))
    else:
        print(sess.run(accuracy, feed_dict={x: test_images, y: test_labels}))
    
    print("Shapes:")
    predictor = tf.argmax(y_conv,1)
    y_pred = sess.run(predictor, feed_dict={x: test_images})
    print(y_pred)
    
    y_truth = labels[train_size:]
    print(y_truth)
    
    print(y_pred.shape)
    print(y_truth.shape)
    
    cmat = confusion_matrix(y_truth, y_pred)
    print(cmat)
    
    np.savetxt('truth_labels.txt', y_truth)
    np.savetxt('pred_labels.txt', y_pred)
    
if FLAGS.trace_flag:        
    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
    trace_file.write(trace.generate_chrome_trace_format())
    trace_file.close()
    
print ("printing weights, 64*32, 32*4: ", W_fc1, W_fc2)
