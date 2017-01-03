#
# V Simple program for UK style electrical socket recognition
# Not scale invariant so requires socket to be about .5m from camera
# Requires pre trained neural net weight files in current dir in JSON format
# Usage: python SocketRecognition.py cam
# Usage: python SocketRecognition.py image.jpg            where image must be 240*320


import tensorflow as tf

import urllib
import cStringIO
from PIL import Image
from PIL import ImageTk
import Tkinter
import numpy as np
import json
import sys

import numpy as nd

stationNo = "4"

def processImage():
    if ( sys.argv[1] == "cam" ):
       file = cStringIO.StringIO( urllib.urlopen( "http://192.168.0." + stationNo + "/image.jpg").read() )
       img = Image.open( file )
       img = img.resize( ( 240, 320 ) )
    else:
       img = Image.open( sys.argv[1] )
    tkpi = ImageTk.PhotoImage(img)

    simg = img.convert( 'L' )

    c = nd.array( simg ) / 255
    c = nd.array( simg ) / 255.0

    f = nd.array( [ c ] )
    z = f.transpose( (1,2,0) )
    z1 = [ z ]

    g = sess.run( result, feed_dict={x_image: z1,} )
    o = g[0]

    d = ImageTk.PhotoImage( Image.fromarray( (o[:,:,0]) * 255. ) )
    label_image.configure( image=d)
    label_image.img = d

def eventLoop():
    processImage()
    if ( sys.argv[1] == "cam" ):
       root.after( 200, eventLoop )

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def buildGraph():

    x_image = tf.placeholder( tf.float32, shape=[ 1, 320, 240, 1 ] )

    with open('conv1weights.json', 'r') as infile:
        W_conv1 = tf.Variable( json.load( infile) )
    with open('conv1bias.json', 'r') as infile:
        b_conv1 = tf.Variable( json.load( infile ) )
    h_conv1 = tf.nn.tanh( b_conv1 + conv2d(x_image, W_conv1) )
    h_pool1 = max_pool_2x2(h_conv1)

    with open('conv2weights.json', 'r') as infile:
        W_conv2 = tf.Variable( json.load( infile) )
    with open('conv2bias.json', 'r') as infile:
        b_conv2 = tf.Variable( json.load( infile ) )
    h_conv2 = tf.nn.tanh( b_conv2 + conv2d(h_pool1, W_conv2) )
    h_pool2 = max_pool_2x2(h_conv2)

    with open('conv3weights.json', 'r') as infile:
        W_conv3 = tf.Variable( json.load( infile) )
    with open('conv3bias.json', 'r') as infile:
        b_conv3 = tf.Variable( json.load( infile ) )
    h_conv3 = tf.nn.tanh( b_conv3 + conv2d(h_pool2, W_conv3) )

    with open('conv4weights.json', 'r') as infile:
        W_conv4 = tf.Variable( json.load( infile) )
    h_conv4 = tf.nn.sigmoid( -.61 + conv2d(h_conv3, W_conv4) )

    return ( x_image, h_conv4 )

(x_image, result ) = buildGraph()

root = Tkinter.Tk()
root.geometry( '%dx%d' % (320,240) )
label_image = Tkinter.Label( root )
label_image.place( x=0, y=0, width=320, height=240 )

sess = tf.Session()
sess.run(tf.initialize_all_variables())

eventLoop()

root.mainloop()
