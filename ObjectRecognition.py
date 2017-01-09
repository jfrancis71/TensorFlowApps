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

def extractObjects( outputMap, threshold ):
    extractPositions = np.transpose( np.nonzero( outputMap[0][:,:,0] > threshold ) )
    origCoords = list( map( lambda x: (x[1]*4,x[0]*4), extractPositions ) )
    return origCoords

def CNObjectLocalization( pilImage, threshold=0.9 ):
    start= time.clock()
    tkImage = ImageTk.PhotoImage( pilImage )
    label_image.img = tkImage
    label_image.pack(side = tkinter.TOP, expand=True, fill=tkinter.BOTH)
    label_image.create_image( 120,160, image=tkImage )

    images = buildImagePyramid( pilImage )
    tfOutputs = []
    tfImages = []
    for s in range( len( images ) ):
        tfOutputs.append( global_graph[s][1] )
        npImage = nd.array( images[s] ) / 255.0
        tfImages.append( [ nd.array( [ npImage ] ).transpose( (1,2,0) ) ] )

    fd = { global_graph[s][0] : tfImages[s] for s in range( len( tfImages ) ) }

    outputPyramid = sess.run( tfOutputs, feed_dict = fd )

    for s in range( len( images ) ):
        objs = extractObjects( outputPyramid[s], threshold )
        scale = pilImage.width / images[s].width
        for obj in objs:
            label_image.create_rectangle( scale*(16 + obj[0]-16), scale*(16 + obj[1]-16), scale*(16 + obj[0]+16), scale*(16 + obj[1]+16), outline='green', width=3 )

    print( "Time lapsed=", time.clock() - start )

fileSource = sys.argv[2]


def processImage():
    if ( fileSource == "cam" ):
       response = requests.get(  "http://192.168.0." + stationNo + "/image.jpg" )
       file = BytesIO( response.content )
       img = Image.open( file )
       img = img.resize( ( 240, 320 ) )
    else:
       img = Image.open( fileSource )

    pilImage = img.convert( 'L' )

    ret = CNObjectLocalization( pilImage, 0.997 )

def eventLoop():
    processImage()
    if ( fileSource == "cam" ):
       root.after( 5, eventLoop )

def conv2d(x, w):
  return w[0] + tf.nn.conv2d(x, w[1], strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def CNReadConv2DWeights( fileName ):
    with open(fileName, 'r') as infile:
        j = json.load( infile )
        b = tf.Variable( j[0] )
        W = tf.Variable( j[1] )
    return (b, W)

modelDir = sys.argv[1]
conv1Weights = CNReadConv2DWeights( modelDir + '/' + 'conv1.json' )
conv2Weights = CNReadConv2DWeights( modelDir + '/' + 'conv2.json' )
conv3Weights = CNReadConv2DWeights( modelDir + '/' + 'conv3.json' )
conv4Weights = CNReadConv2DWeights( modelDir + '/' + 'conv4.json' )

def buildGraph( x_image ):

    h_conv1 = tf.nn.tanh( conv2d(x_image, conv1Weights ) )
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.tanh( conv2d(h_pool1, conv2Weights) )
    h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.tanh( conv2d(h_pool2, conv3Weights) )

    h_conv4 = tf.nn.sigmoid( conv2d(h_conv3, conv4Weights) )

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
