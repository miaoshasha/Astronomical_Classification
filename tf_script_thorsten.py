
# coding: utf-8

# In[16]:

from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as tfk
from tensorflow.python.client import timeline
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# In[17]:

#shape argument is optional
def weight_variable(shape,name=None):
    initializer = tfk.initializers.he_normal() #tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initializer(shape),name=name)

def bias_variable(shape, name=None):
    initializer = tf.constant(0.1, shape=shape)
    return tf.Variable(initializer, name=name)

def offset_variable(shape, name=None):
    initializer = tf.constant(0.1, shape=shape)
    return tf.Variable(initializer, name=name)

def conv2d(x, W, strides, name=None):
    return tf.nn.conv2d(x, filter=W, strides=strides, padding='VALID', data_format="NCHW", name=name)

def max_pool_2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2], padding='SAME', data_format="NCHW", name=name)

def batch_normalization(x, shape, offset=None, name = None):
    #mean, variance = tf.nn.moments(x, axes=[0, 1, 2], shift=None, name=None, keep_dims=False) 
    #momnets -- used with convolutional filters with shape [batch, height, width, depth]
    initializer = tfk.initializers.he_normal() #tf.truncated_normal(shape, stddev=0.1)
    if not name:
        mean = tf.Variable(tf.constant(0.0, shape=shape))
        variance = tf.Variable(tf.constant(1., shape=shape))
    else:
        mean = tf.Variable(tf.constant(0.0, shape=shape), name = name + '_mean')
        variance = tf.Variable(tf.constant(1., shape=shape), name = name + '_var')
    return tf.nn.batch_normalization(x, mean, variance, offset=None, scale=None, variance_epsilon = 1e-6, name=name)

# In[18]:

# Read data from csv file.
# Path to all data:  /global/cscratch1/sd/muszyng/data_astronomy_catalogs/trainingData.csv
df = np.genfromtxt('trainingData.csv', delimiter=" ")
#df = np.genfromtxt('debugData.csv', delimiter=" ")

#initial shuffle
perm = np.random.permutation(range(df.shape[0]))
df = df[perm]

# In[19]:

print(df.shape)


# In[20]:

# Create array of labels.
labels_sdss = df[:,0]
labels = df[:,-1]
data = df[:,1:-1]
num_classes = int(np.max(labels))+1
print("We have ",num_classes," classes to predict")
labels = np.reshape(labels,(labels.shape[0],1))
labels_sdss = np.reshape(labels_sdss,(labels_sdss.shape[0],1))
#print(labels)


# In[21]:

# Create binary matrix of labels.
#nb_classes = 4
#mat_labels = np.zeros([nb_classes, len(labels)], dtype=int)
#print(mat_labels)


# In[22]:

# Convert labels to binary matrix.
#for i in range(0, len(labels)):
#    l = int(labels[i])
#    mat_labels[l, i] = 1
#print(mat_labels)

# In[29]:

train_image_size = data.shape[1] #input size 
channel = 1
#num_classes = 3
#image_data =  tf.convert_to_tensor(df,dtype=tf.float32)
#labels_one_hot = np.transpose(mat_labels, [1,0])

#fractions:
train_fraction=0.8
validation_fraction=0.1
#training and validation sizes
train_size = int(data.shape[0]*train_fraction)
validation_size = int(data.shape[0]*validation_fraction)
print("Size of Training set:",train_size)
print("Size of Validation set:",validation_size)
print("Size of Test set:",data.shape[0]-(train_size+validation_size))
#split the set
#train
train_images = data[:train_size]
train_labels = labels[:train_size]
train_labels_sdss = labels_sdss[:train_size]
#validation
validation_images = data[train_size:train_size+validation_size]
validation_labels = labels[train_size:train_size+validation_size]
validation_labels_sdss = labels_sdss[train_size:train_size+validation_size]
#test
test_images = data[train_size+validation_size:]
test_labels = labels[train_size+validation_size:]
test_labels_sdss = labels_sdss[train_size+validation_size:]

#model begins
run_metadata = tf.RunMetadata()
inforstring = os.getenv('inforstring')
trace_file = open('timeline'+str(inforstring)+'.ctf.json', 'w')

tf.app.flags.DEFINE_boolean('trace_flag', True, """If set, it produces an a trace of the threads  executing work during the training phase.""")
tf.app.flags.DEFINE_integer('epoch', 10, """Epochs to train the model.""")

x_train = train_images
y_train = train_labels


# In[9]:

#init placeholders
x = tf.placeholder(tf.float32, shape = [None, train_image_size]) #change this to the vector shape
y = tf.placeholder(tf.int32, shape = [None,1])
y_sdss = tf.placeholder(tf.int32, shape = [None,1])
keep_prob = tf.placeholder(tf.float32)



#dense, initializing the variables
hidden_dim = 64
W_fc1 = weight_variable([data.shape[1], hidden_dim],name="W_fc1")
b_fc1 = bias_variable([hidden_dim],name="b_fc1")
W_fc2 = weight_variable([hidden_dim, hidden_dim],name="W_fc2")
b_fc2 = bias_variable([hidden_dim],name="b_fc2")
W_fc3 = weight_variable([hidden_dim, num_classes],name="W_fc3")
b_fc3 = bias_variable([num_classes],name="b_fc3")

# In[12]:

x_input = tf.reshape(x, [-1, train_image_size])

#dense 1
h_fc1 = tf.matmul(x_input, W_fc1)
bn_fc1 = batch_normalization(h_fc1,[hidden_dim]) + b_fc1
re_fc1 = tf.nn.relu(bn_fc1)
dr_fc1 = tf.nn.dropout(re_fc1, keep_prob=keep_prob)
#dense 2
#h_fc2 = tf.matmul(dr_fc1, W_fc2)
#bn_fc2 = batch_normalization(h_fc2,[hidden_dim]) + b_fc2
#re_fc2 = tf.nn.relu(bn_fc2)
#dr_fc2 = tf.nn.dropout(re_fc2, keep_prob=keep_prob)
#output
y_conv = tf.matmul(dr_fc1, W_fc3) + b_fc3


#graph construction done
predictor = tf.argmax(y_conv,1)
cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_conv))
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
accuracy = tf.metrics.accuracy(labels=y[:,0], predictions=predictor, name='accuracy')
sdss_accuracy = tf.metrics.accuracy(labels=y[:,0], predictions=y_sdss[:,0])
#correct_prediction = tf.equal(tf.cast(tf.argmax(predictor,1),tf.int32), y[:,0])
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#train step
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# In[15]:

if (os.getenv('NUM_INTER_THREADS', None) is not None and os.getenv('NUM_INTRA_THREADS', None) is not None):
    print("Custom NERSC/Intel config:inter_op_parallelism_threads({}),""intra_op_parallelism_threads({})".format(os.environ['NUM_INTER_THREADS'],os.environ['NUM_INTRA_THREADS']))
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=int(os.environ['NUM_INTER_THREADS']),intra_op_parallelism_threads=int(os.environ['NUM_INTRA_THREADS'])))


#batch size = all data entries
FLAGS = tf.app.flags.FLAGS
epoch = FLAGS.epoch

batch_size = 100
keep_probability = 0.5
max_step = int(epoch*train_size/float(batch_size))
print("max steps: ", max_step)

#initialize to zero
loss_ = 0.
batches_ = 0
epochs_ = 0

with tf.Session() as sess:
    if FLAGS.trace_flag:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()],  options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
    else:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    for i in range(max_step):
        #extract batches
        if ((i+1)*batch_size)%train_size > (i*batch_size)%train_size:
            x_batch = x_train[(i*batch_size)%train_size:((i+1)*batch_size)%train_size]
            y_batch = y_train[(i*batch_size)%train_size:((i+1)*batch_size)%train_size]
            
            #train step, loss and accuracy update
            _, tmp_loss = sess.run([train_step, cross_entropy], feed_dict={x: x_batch, y: y_batch, keep_prob: keep_probability})
            
            #average loss
            loss_ += tmp_loss
            batches_ += 1
        
        #compute accuracy
        else:
            epochs_ += 1
            print( "Training Loss epoch ",epochs_, ": ", loss_/float(batches_))
            if FLAGS.trace_flag:
                pred, _ = sess.run([predictor, accuracy[1]], feed_dict={x: validation_images, y: validation_labels, keep_prob: 1.}, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
                acc = sess.run(accuracy[0], options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
            else:
                pred, _ = sess.run([predictor, accuracy[1]], feed_dict={x: validation_images, y: validation_labels, keep_prob: 1.})
                acc = sess.run(accuracy[0])
            print( "Validation Accuracy epoch ",epochs_ , ": ", acc)
            
            #shuffle data
            perm = np.random.permutation(range(train_size))
            x_train = x_train[perm, :]
            y_train = y_train[perm, :]
            
            #reset stuff
            sess.run([tf.local_variables_initializer()])
            batches_ = 0
            loss_ = 0.
    
    print( "Final Validation accuracy ML: ", acc)
    sess.run(sdss_accuracy[1], feed_dict={y: validation_labels, y_sdss: validation_labels_sdss})
    acc_sdss = sess.run(sdss_accuracy[0])
    print( "Final Validation accuracy SDSS: ", acc_sdss)
    
    
    #wrap-up
    #print("Test Accuracy: \n")
    #print("Y-labels: ",validation_labels.shape)
    #if FLAGS.trace_flag:
    #    print(sess.run(accuracy[0], feed_dict={x: validation_images, y: validation_labels, keep_prob: 1.}, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata))
    #else:
    #    print(sess.run(accuracy[0], feed_dict={x: validation_images, y: validation_labels, keep_prob: 1.}))
    #
    #print("Shapes:")
    #y_pred = sess.run(predictor, feed_dict={x: validation_images, keep_prob: 1.})
    ##print(y_pred)
    #
    #y_truth = validation_labels[:,0].astype(np.int32,copy=True)
    ##print(y_truth)
    #
    ##print(y_pred.shape)
    ##print(y_truth.shape)
    #
    #cmat = confusion_matrix(y_truth, y_pred)
    #print(cmat)
    #
    #np.savetxt('truth_labels.txt', y_truth)
    #np.savetxt('pred_labels.txt', y_pred)
    
if FLAGS.trace_flag:        
    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
    trace_file.write(trace.generate_chrome_trace_format())
    trace_file.close()
    
print ("printing weights, 64*32, 32*4: ", W_fc1, W_fc2)
