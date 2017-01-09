#
# V Simple program for UK style electrical socket recognition
# Requires pre trained neural net weight files in current dir in JSON format
# Usage: python SocketRecognition.py cam
# Usage: python SocketRecognition.py image.jpg            where image must be 240*320


import tensorflow as tf

import urllib
import io
from PIL import Image
from PIL import ImageTk
from PIL import ImageDraw
import tkinter
import numpy as np
import json
import time
import sys
import math
import urllib.request as ur
import urllib.request
from io import BytesIO
import requests


import numpy as nd

stationNo = "2"

def buildImagePyramid( pilImage ):
    images = []
    for s in range( -1 + int( ( math.log( 32 ) - math.log( pilImage.width ) ) / math.log (.8 ) ) ):
        height = pilImage.height * .8**s
        width  = pilImage.width * .8**s
        print( "idx = ", s, " width = ", width )
        images.append( pilImage.resize( ( int(width), int(height) ) ) )
    return images

def CNObjectLocalizationConvolve( pilImage, model, threshold=0.9 ):
    start= time.clock()
    npImage = nd.array( pilImage ) / 255.0
    tfImage = [ nd.array( [ npImage ] ).transpose( (1,2,0) ) ]

    #x_image = tf.placeholder( tf.float32, shape=[ 1, pilImage.height, pilImage.width, 1 ] )
    print (pilImage.height)
    print (pilImage.width)
    #result = buildGraph( x_image )
    #sess.run(tf.global_variables_initializer())
    mstart = time.clock()
    tfOutput = sess.run( model[1], feed_dict={model[0]: tfImage,} )
    print (" Main: ", time.clock() - mstart)
    extractPositions = np.transpose( np.nonzero( tfOutput[0][:,:,0] > threshold ) )
    print (extractPositions)
    origCoords = list( map( lambda x: (x[1]*4,x[0]*4), extractPositions ) )
    print (origCoords)
    print ("Time taken = ", time.clock() - start)
    return origCoords

def CNObjectLocalization( pilImage, threshold=0.9 ):
    tkImage = ImageTk.PhotoImage( pilImage )
    label_image.img = tkImage
    label_image.pack(side = tkinter.TOP, expand=True, fill=tkinter.BOTH)
    label_image.create_image( 120,160, image=tkImage )

    images = buildImagePyramid( pilImage )
    for s in range( len( images ) ):
        objs = CNObjectLocalizationConvolve( images[s], global_graph[s], threshold )
        for obj in objs:
            label_image.create_rectangle( 16 + obj[0]-16, 16 + obj[1]-16, 16 + obj[0]+16, 16 + obj[1]+16, outline='green', width=3 )


def processImage():
    if ( sys.argv[1] == "cam" ):
       response = requests.get(  "http://192.168.0." + stationNo + "/image.jpg" )
       file = BytesIO( response.content )
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

def initialize():
    initImgs = buildImagePyramid( Image.new( 'L', (240,320), 0 ) )
    gl = []
    for s in range( len(initImgs) ):
        x_image = tf.placeholder( tf.float32, shape=[ 1, initImgs[s].height, initImgs[s].width, 1 ] )
        gr1 = buildGraph( x_image )
        gl.append( (x_image, gr1 ) )

    return gl

root = tkinter.Tk()
root.geometry( '%dx%d' % (240,320) )
label_image = tkinter.Canvas( root )

sess = tf.Session()

global_graph = initialize()

sess.run(tf.global_variables_initializer())

eventLoop()

root.mainloop()
