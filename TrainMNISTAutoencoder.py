# Attempt at implementing autoencoder for MNIST
# Multiple variations of this have been tried, eg. the linear (PCA),
# sigmoidal, and denoising. None of them end up producing local filters

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

import argparse
import sys

import json

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

import argparse


# Import data
mnist = input_data.read_data_sets('tmp/tensorflow/mnist/input_data', one_hot=True)


# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=.1))

b1 = tf.Variable(tf.zeros([500]))

x1 = x * ( numpy.random.rand( 100, 784 ) < .7 )

h = tf.nn.sigmoid( tf.matmul(x1, W1) + b1 )

W2 = tf.transpose( W1 )
b2 = tf.Variable(tf.zeros([784]))
y = tf.matmul(h, W2) + b2

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 784])

cross_entropy_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(y, y_) )

train_step = tf.train.GradientDescentOptimizer(0.1).minimize( cross_entropy_loss )

cross_entropy_summary = tf.summary.scalar( 'cross_entropy', cross_entropy_loss )

tf_weights = [ tf.split( 0, 28, W1[:,0:100] ) ]
tf_weights_4d = tf.transpose( tf_weights, [ 3, 1, 2, 0 ] )

weights_images = tf.image_summary( 'weights', tf_weights_4d, 100 )

sess = tf.Session()
saver = tf.train.Saver()

parser = argparse.ArgumentParser()
parser.add_argument("-restore",
                    help="restore model from file")
parser.add_argument("-save",
                    help="save model to file")
parser.add_argument("-logdir",
                    help="logdir")

args = parser.parse_args()
if ( args.restore is None ):
    sess.run( tf.global_variables_initializer() )
else:
     saver.restore( sess, args.restore )   

summary_writer = tf.train.SummaryWriter( args.logdir, sess.graph )

saver = tf.train.Saver()

# Train
step = 0
for _ in range(100000):
  for _ in range(100):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_xs})

  cross_entropy_summary_str = sess.run( cross_entropy_summary, feed_dict= {x: batch_xs, y_: batch_xs} )
  summary_writer.add_summary( cross_entropy_summary_str, global_step=step)
  wtimgs = sess.run( weights_images, feed_dict= {x: batch_xs, y_: batch_xs} )
  summary_writer.add_summary( wtimgs, global_step=step)
  step = step + 1
  save_path = saver.save( sess, args.save )

  with open('weights.json', 'w') as outfile:
    json.dump(sess.run( W1 ).tolist(), outfile)
