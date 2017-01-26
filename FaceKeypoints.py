#Ref:
#http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
#Gained some ideas and inspiration from above reference.
#Our results are less comparable as we are being less ambitious, and just predicting locations of left and right eye
#However we have got more data for these variables, so we can potentially train a more accurate model
#Data source is an HDF5 file which contains filtered samples where the left-eye, right-eye coords are present
#where data originally came from the Kaggle Face kepypoints competition

import tensorflow as tf
import numpy as np
import h5py
import argparse

#import the data
def readData( fileName ):
    file    = h5py.File( fileName, 'r')
    images = file['/images']
    features = file['/features']
    return ( images, features )

# Model definitions

#Most basic model of all, simple linear regression
#old, from previous run b = [ 0.35217056 -0.20918441 -0.35843661 -0.20155163]
#Validation =  .00534, overfitting 0.659
#training means of 1:6000 = {0.382405, -0.216675, -0.370396, -0.209863}
#Applying training mean to validation 6000:7034 gives loss 0.005211
def model1():
  x_flat = tf.reshape( x, [-1, 96*96 ] ) / 255
  yt = ( y_ - 48 ) / 48

  W1 = tf.Variable(tf.truncated_normal([96*96, 4], stddev=1/np.sqrt(96*96)))
  b = tf.Variable( [ 0., 0., 0., 0. ] )
  y = tf.matmul( x_flat, W1 ) + b

  loss = tf.reduce_mean( tf.pow( ( y - yt ), 2 ) )

  return loss

#Basic neural net model with single hidden layer
#Validation =  0.003921, overfitting .663074
def model2():
  x_flat = tf.reshape( x, [-1, 96*96 ] ) / 255
  yt = ( y_ - 48 ) / 48

  W1 = tf.Variable(tf.truncated_normal([96*96, 100], stddev=1/np.sqrt(96*96)))
  b1 = tf.Variable( np.array( [ 100 ] ) * .0, dtype=tf.float32 )
  h = tf.nn.tanh( tf.matmul( x_flat, W1 ) + b1 )

  W2 = tf.Variable(tf.truncated_normal([100, 4], stddev=1/np.sqrt(100)))
  b2 = tf.Variable( [ 0., 0., 0., 0. ] )
  y = tf.nn.tanh( tf.matmul( h, W2 ) + b2 )

  loss = tf.reduce_mean( tf.pow( ( y - yt ), 2 ) )

  return loss

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#More powerful convolutional model
#Validation = Train= 0.000327785217824 Validation= 0.000376207323065 Overfitting= 0.871288775438
def model3():
  x_image = (tf.reshape(x, [-1,96,96,1])-48)/48
  yt = (y_ - 48)/48

  W_conv1 = weight_variable([3, 3, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  W_conv2 = weight_variable([2, 2, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  W_conv3 = weight_variable([2, 2, 64, 128])
  b_conv3 = bias_variable([128])
  h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
  h_pool3 = max_pool_2x2(h_conv3)

  W_fc1 = weight_variable([11 * 11 * 128, 500])
  b_fc1 = bias_variable([500])
  h_pool3_flat = tf.reshape(h_pool3, [-1, 11*11*128])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

  W_fc2 = weight_variable([500,500])
  b_fc2 = bias_variable([500])
  h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

  W_fc3 = weight_variable([500, 4])
  b_fc3 = bias_variable([4])
  y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3

  loss = tf.reduce_mean( tf.pow( ( y_conv - yt ), 2 ) )

  tf.image_summary( "W1", tf.transpose( W_conv1, [ 3, 0, 1, 2 ] ) )

  return loss

#Some helper functions
def partition(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

def batch_process( x, y ):
  return zip( partition( x,100 ), partition( y, 100 ) )

def batch_run( inputs, targets ):
  total_loss = 0
  for batch in batch_process( inputs, targets ):
    batch_loss = sess.run( loss, feed_dict = { x: batch[0], y_: batch[1] } )
    total_loss += batch_loss * len( batch )
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

( images, features ) = readData( args.datafile)

#TensorFlow code
x = tf.placeholder(tf.float32, [None, 96, 96])
y_ = tf.placeholder( tf.float32, shape=[None,4 ] )

loss = model3()
train_step = tf.train.MomentumOptimizer( learning_rate=0.0001, use_nesterov=True, momentum=0.9 ).minimize( loss )

gpu_options = tf.GPUOptions( per_process_gpu_memory_fraction=0.4 )
sess = tf.Session( config=tf.ConfigProto( gpu_options = gpu_options, device_count = {'GPU':-1} ) )
sess.run( tf.global_variables_initializer() )

train_summary_writer = tf.train.SummaryWriter( args.logdir + '/train', sess.graph )
validation_summary_writer = tf.train.SummaryWriter( args.logdir + '/validation', sess.graph )
weights_summary_writer = tf.train.SummaryWriter( args.logdir + '/weights', sess.graph )

merged = tf.summary.merge_all()

saver = tf.train.Saver()

for epoch in range(100000):
  for batch in batch_process( images[:6000], features[:6000] ):
    sess.run(train_step, feed_dict={ x: batch[0], y_: batch[1] } )

  training_loss = batch_run( images[:6000], features[:6000] )
  validation_loss = batch_run( images[6000:], features[6000:] )

  train_summary_writer.add_summary( tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=training_loss)]) )
  validation_summary_writer.add_summary( tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=validation_loss)]) )
  weights_summary_writer.add_summary( sess.run( merged ) )
  print( "Train=", training_loss, "Validation=", validation_loss, "Overfitting=", training_loss/validation_loss )
  saver.save( sess, args.checkpointfile )
