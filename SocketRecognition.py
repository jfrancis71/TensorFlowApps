#
# V Simple program for UK style electrical socket recognition
# Requires pre trained neural net weight files in current dir in JSON format
# Usage: python SocketRecognition.py cam
# Usage: python SocketRecognition.py image.jpg            where image must be 240*320


import tensorflow as tf

import urllib
import cStringIO
from PIL import Image
from PIL import ImageTk
from PIL import ImageDraw
import Tkinter
import numpy as np
import json
import time
import sys
import math

import numpy as nd

stationNo = "62"

def CNObjectLocalizationConvolve( pilImage, threshold=0.9 ):
    start= time.clock()
    npImage = nd.array( pilImage ) / 255.0
    tfImage = [ nd.array( [ npImage ] ).transpose( (1,2,0) ) ]

    x_image = tf.placeholder( tf.float32, shape=[ 1, pilImage.height, pilImage.width, 1 ] )
    print pilImage.height
    print pilImage.width
    result = buildGraph( x_image )
    sess.run(tf.initialize_all_variables())
    mstart = time.clock()
    tfOutput = sess.run( result, feed_dict={x_image: tfImage,} )
    print " Main: ", time.clock() - mstart
    extractPositions = np.transpose( np.nonzero( tfOutput[0][:,:,0] > threshold ) )
    print extractPositions
    origCoords = list( map( lambda (y,x): (x*4,y*4), extractPositions ) )
    print origCoords
    print "Time taken = ", time.clock() - start
    return origCoords

def CNObjectLocalization( pilImage, threshold=0.9 ):
    tkImage = ImageTk.PhotoImage( pilImage )
    label_image.img = tkImage
    label_image.pack(side = Tkinter.TOP, expand=True, fill=Tkinter.BOTH)
    label_image.create_image( 120,160, image=tkImage )

    for s in range( -1 + int( ( math.log( 32 ) - math.log( pilImage.width ) ) / math.log (.8 ) ) ):
        height = pilImage.height * .8**s
        width  = pilImage.width * .8**s
        print( "idx = ", s, " width = ", width )
        objs = CNObjectLocalizationConvolve( pilImage.resize( ( int(width), int(height) ) ), threshold )
#    objs = [ ( 1,1 ) ]
        for obj in objs:
            label_image.create_rectangle( 16 + obj[0]-16, 16 + obj[1]-16, 16 + obj[0]+16, 16 + obj[1]+16, outline='green', width=3 )


def processImage():
    if ( sys.argv[1] == "cam" ):
       file = cStringIO.StringIO( urllib.urlopen( "http://192.168.0." + stationNo + "/image.jpg").read() )
       img = Image.open( file )
       img = img.resize( ( 240, 320 ) )
    else:
       img = Image.open( sys.argv[1] )

    pilImage = img.convert( 'L' )

    ret = CNObjectLocalization( pilImage, 0.9 )

def eventLoop():
    processImage()
    if ( sys.argv[1] == "cam" ):
       root.after( 200, eventLoop )

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

with open('conv1weights.json', 'r') as infile:
    W_conv1 = tf.Variable( json.load( infile) )
with open('conv1bias.json', 'r') as infile:
    b_conv1 = tf.Variable( json.load( infile ) )

with open('conv2weights.json', 'r') as infile:
    W_conv2 = tf.Variable( json.load( infile) )
with open('conv2bias.json', 'r') as infile:
    b_conv2 = tf.Variable( json.load( infile ) )

with open('conv3weights.json', 'r') as infile:
    W_conv3 = tf.Variable( json.load( infile) )
with open('conv3bias.json', 'r') as infile:
    b_conv3 = tf.Variable( json.load( infile ) )

with open('conv4weights.json', 'r') as infile:
    W_conv4 = tf.Variable( json.load( infile) )

def buildGraph( x_image ):

    h_conv1 = tf.nn.tanh( b_conv1 + conv2d(x_image, W_conv1) )
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.tanh( b_conv2 + conv2d(h_pool1, W_conv2) )
    h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.tanh( b_conv3 + conv2d(h_pool2, W_conv3) )

    h_conv4 = tf.nn.sigmoid( -.61 + conv2d(h_conv3, W_conv4) )

    return ( h_conv4 )

root = Tkinter.Tk()
root.geometry( '%dx%d' % (240,320) )
label_image = Tkinter.Canvas( root )

sess = tf.Session()

eventLoop()

root.mainloop()
