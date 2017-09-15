# coding: utf-8

from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as tfk
from tensorflow.python.client import timeline
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt


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


# Read data from csv file.
# Path to all data:  /global/cscratch1/sd/muszyng/data_astronomy_catalogs/trainingData.csv
df = pd.read_csv('trainingData.csv', delimiter=" ", header=0) #np.genfromtxt('trainingData.csv', delimiter=" ")
#initial shuffle
df = df.sample(frac=1).reset_index(drop=True)

#do sme computations for obtaining sizes of test and train and validation
#fractions:
num_classes = int(np.max(df.iloc[:,-1]))+1
print("We have ",num_classes," classes to predict")
train_fraction=0.8
validation_fraction=0.1
#determine feature_list
features=[x for x in df.columns if x not in ['type', 'truth']]
num_features = len(features)
print("We have the following ", num_features, " features: ",features)
#determine frequency per group
groups = df.groupby("truth")
countdf = pd.DataFrame(groups.count())["type"].reset_index().rename(columns={'type':'frequency'})
df = df.merge(countdf, on="truth", how="left")


#extract the given fractions for test train and validation PER CLASS (in case we have uneven class distribution) and shuffle
groups = df.groupby("truth")
#training
train_df = pd.DataFrame(groups.apply(lambda x: x.iloc[:int(x['frequency'].iloc[0]*train_fraction),:])).reset_index(drop=True)
train_df = train_df.sample(frac=1).reset_index(drop=True)
#del train_df['frequency']
train_size = train_df.shape[0]
#validation
validation_df = pd.DataFrame(groups.apply(lambda x: x.iloc[int(x['frequency'].iloc[0]*train_fraction):int(x['frequency'].iloc[0]*train_fraction)+int(x['frequency'].iloc[0]*validation_fraction),:])).reset_index(drop=True)
validation_df = validation_df.sample(frac=1).reset_index(drop=True)
#del validation_df['frequency']
validation_size = validation_df.shape[0]
#test
test_df = pd.DataFrame(groups.apply(lambda x: x.iloc[int(x['frequency'].iloc[0]*train_fraction)+int(x['frequency'].iloc[0]*validation_fraction):,:])).reset_index(drop=True)
test_df = test_df.sample(frac=1).reset_index(drop=True)
#del test_df['frequency']
test_size = test_df.shape[0]


#some informational printing
print("Training set size: ", train_size)
print("Validation set size: ", validation_size)
print("Test set size: ", test_size)


#apply preprocessing and split in data, label, label_sdss
#standard caler: warning the dataset has many outliers. However, works better than robust scaler
scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
#robust scaler, might be better
#scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=[0.25,0.75], copy=True)
scaler.fit(train_df.loc[:,features])
#train
train_images = scaler.transform(train_df.loc[:,features].values)
#number of features
train_labels = np.expand_dims(train_df.loc[:,"truth"].values, axis=1).astype(np.int32)
train_labels_sdss = np.expand_dims(train_df.loc[:,"type"].values, axis=1).astype(np.int32)
#validation
validation_images = scaler.transform(validation_df.loc[:,features].values)
validation_labels = np.expand_dims(validation_df.loc[:,"truth"].values, axis=1).astype(np.int32)
validation_labels_sdss = np.expand_dims(validation_df.loc[:,"type"].values, axis=1).astype(np.int32)
#test
test_images = scaler.transform(test_df.loc[:,features].values)
test_labels = np.expand_dims(test_df.loc[:,"truth"].values, axis=1).astype(np.int32)
test_labels_sdss = np.expand_dims(test_df.loc[:,"type"].values, axis=1).astype(np.int32)


#start tensorflow model construction
run_metadata = tf.RunMetadata()
inforstring = os.getenv('inforstring')
trace_file = open('timeline'+str(inforstring)+'.ctf.json', 'w')

tf.app.flags.DEFINE_boolean('trace_flag', True, """If set, it produces an a trace of the threads  executing work during the training phase.""")
tf.app.flags.DEFINE_integer('epoch', 10, """Epochs to train the model.""")

x_train = train_images
y_train = train_labels


#init placeholders
x = tf.placeholder(tf.float32, shape = [None, num_features]) #change this to the vector shape
y = tf.placeholder(tf.int32, shape = [None,1])
y_sdss = tf.placeholder(tf.int32, shape = [None,1])
keep_prob = tf.placeholder(tf.float32)

#variables
hidden_dim = 64
W_fc1 = weight_variable([num_features, hidden_dim],name="W_fc1")
b_fc1 = bias_variable([hidden_dim],name="b_fc1")
W_fc2 = weight_variable([hidden_dim, hidden_dim],name="W_fc2")
b_fc2 = bias_variable([hidden_dim],name="b_fc2")
W_fc3 = weight_variable([hidden_dim, num_classes],name="W_fc3")
b_fc3 = bias_variable([num_classes],name="b_fc3")

#input layer
x_input = tf.reshape(x, [-1, num_features])
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
#prediction
predictor = tf.argmax(y_conv,1)
#cross entropy: use sparse version so that we do not need to one-hot-encode ourselves
cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_conv))
#accuracy, this is a streaming metrics, please read tensorflow.org carefully how those are used
accuracy = tf.metrics.accuracy(labels=y[:,0], predictions=predictor, name='accuracy')
#same for sdss, we do not need a predictor here
sdss_accuracy = tf.metrics.accuracy(labels=y[:,0], predictions=y_sdss[:,0])
#training step
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


#print environment
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
