#Achieves around .019 when trained with the 3'rd convolutional module having 64 layers
#when trained on around 175,000 patches in total.
#The 32 layer in module 3) is not that different, achieving around .022
#In both cases after 10 training epochs.
#Sadly not the 205,000 due to Mathematica HDF5 bug (uses 32-bit program I think).

import tensorflow as tf
import numpy as np
import h5py
import argparse

#import the data
def readData( fileName ):
    file    = h5py.File( fileName, 'r')
    images = file['/images']
    labels = file['/labels']
    return ( images, labels )

# Model definitions

def weight_variable(shape):
  if len(shape) == 4:
    initial = tf.random_uniform(shape, -np.sqrt( 12 / ( shape[0]*shape[1]*shape[2] + shape[3]) ), np.sqrt( 12 / ( shape[0]*shape[1]*shape[2] + shape[3] ) ) )
  else:
    initial = tf.random_uniform(shape, -np.sqrt( 12 / ( shape[0] + shape[1] ) ), np.sqrt( 12 / ( shape[0] + shape[1] ) ) )
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def model( l3 ):
  x_image = tf.reshape(x, [-1,32,32,1])

  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.tanh(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  W_conv2 = weight_variable([5, 5, 32, 32])
  b_conv2 = bias_variable([32])
  h_conv2 = tf.nn.tanh(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  W_conv3 = weight_variable([5, 5, 32, l3])
  b_conv3 = bias_variable([l3])
  h_conv3 = tf.nn.tanh(conv2d(h_pool2, W_conv3) + b_conv3)
  h_pool3 = max_pool_2x2(h_conv3)

  W_fc1 = tf.Variable( tf.truncated_normal( [ l3,1 ], stddev=0.01 ) )
  b_fc1 = bias_variable([1])
  h_pool3_flat = tf.reshape(h_pool3, [-1, l3])
  y_conv = tf.matmul(h_pool3_flat, W_fc1) + b_fc1

#  loss = tf.reduce_mean( tf.pow( ( y_conv - y_ ), 2 ) )
  cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_conv[:,0], y_))

  tf.image_summary( "W1", tf.transpose( W_conv1, [ 3, 0, 1, 2 ] ) )

  return cross_entropy

#Some helper functions
def partition(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

def batch_process( x, y ):
  return zip( partition( x,100 ), partition( y, 100 ) )

def batch_run( sess, loss, inputs, targets ):
  total_loss = 0
  for batch in batch_process( inputs, targets ):
    batch_loss = sess.run( loss, feed_dict = { x: batch[0], y_: batch[1] } )
    total_loss += batch_loss * len( batch[0] )
  return total_loss / len( targets )

#Parsing the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-datafile",
                    help="HDF5 file containing training and validation data")
parser.add_argument("-logdir",
                    help="logging directory")
parser.add_argument("-checkpointfile",
                    help="file to store saved model")
args = parser.parse_args()

( images, labels ) = readData( args.datafile)

#TensorFlow code
x = tf.placeholder(tf.float32, [None, 32, 32])
y_ = tf.placeholder( tf.float32, shape=[None] )

training_size = int( len( images )*.8 )
print( "Training Set size = ", training_size )

training_images = images[:training_size]
training_labels = labels[:training_size]
validation_images = images[training_size:]
validation_labels = labels[training_size:]

def train_model( l3 ):
  loss = model( l3 )
  train_step = tf.train.MomentumOptimizer( learning_rate=0.01, use_nesterov=True, momentum=0.9 ).minimize( loss )
  gpu_options = tf.GPUOptions( per_process_gpu_memory_fraction=0.4 )
  sess = tf.Session( config=tf.ConfigProto( gpu_options = gpu_options, device_count = {'GPU':-1} ) )
  sess.run( tf.global_variables_initializer() )

  for epoch in range(6):
    for batch in batch_process( training_images, training_labels ):
      sess.run(train_step, feed_dict={ x: batch[0], y_: batch[1] } )

    training_loss = batch_run( sess, loss, training_images, training_labels )
    validation_loss = batch_run( sess, loss, validation_images, validation_labels )

    print( "Level", l3, " Train=", training_loss, "Validation=", validation_loss, "Overfitting=", training_loss/validation_loss )

  return( validation_loss )

res = []
for l3 in range(1,64):
  res.append( train_model( l3 ) )

print( res )
