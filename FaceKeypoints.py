#Ref:
#http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
#Gained some ideas and inspiration from above reference.
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

#Some helper functions
def partition(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

def batch_process( x, y ):
  return zip( partition( x,100 ), partition( y, 100 ) )


#Parsing the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-datafile",
                    help="HDF5 file containing training and validation data")
parser.add_argument("-logdir",
                    help="logging directory")
args = parser.parse_args()

( images, features ) = readData( args.datafile)

#TensorFlow code
x = tf.placeholder(tf.float32, [None, 96, 96])
y_ = tf.placeholder( tf.float32, shape=[None,4 ] )

loss = model2()
train_step = tf.train.MomentumOptimizer( learning_rate=0.001, use_nesterov=True, momentum=0.9 ).minimize( loss )

gpu_options = tf.GPUOptions( per_process_gpu_memory_fraction=0.4 )
sess = tf.Session( config=tf.ConfigProto( gpu_options = gpu_options, device_count = {'GPU':-1} ) )
sess.run( tf.global_variables_initializer() )

train_summary_writer = tf.train.SummaryWriter( args.logdir + '/train', sess.graph )
validation_summary_writer = tf.train.SummaryWriter( args.logdir + '/validation', sess.graph )

loss_summary = tf.summary.scalar( 'loss', loss )
merged = tf.summary.merge_all()

for epoch in range(100000):
  for batch in batch_process( images[:6000], features[:6000] ):
    sess.run(train_step, feed_dict={ x: batch[0], y_: batch[1] } )

  [ train_loss, summary_str ] = sess.run( [ loss, merged ], feed_dict= {x: images[:6000], y_: features[:6000] } )
  train_summary_writer.add_summary( summary_str)
  [ validation_loss, summary_str ] = sess.run( [ loss, merged ], feed_dict= {x: images[6000:], y_: features[6000:] } )
  validation_summary_writer.add_summary( summary_str )
  print( "Train=", train_loss, "Validation=", validation_loss, "Overfitting=", train_loss/validation_loss )
