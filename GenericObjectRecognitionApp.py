#
# Requires pre trained neural net weight files in JSON format

import tensorflow as tf
import numpy as np

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
import argparse

import CNObjectRecognition

#Example command line:
#python3 GenericObjectRecognitionApp.py -model ~/Google\ Drive/Personal/Computer\ Science/WebMonitor/sockets/net.json -image cam -station 4 -threshold .99

def blend( x, c1, c2 ):
    return x*c2 + (1-x)*c1

def colorF( patch ):

    if ( genderNetFilename is None ):
        return 'green'

    tfImage = [ nd.array( [ patch ] ).transpose( (1,2,0) ) ]
    score = sess.run( genderResult, feed_dict = { x_genderImage : tfImage } )[0][0]
    print( "gender score: ", score )

    c1 = ( 255, 105, 180 )
    c2 = ( 0, 0, 255 )

    c1 = np.array( [ 255., 105., 180. ] )
    c2 = np.array( [ 0., 0., 255. ] )

    bld = blend( score, c1, c2 )

    col = "#%02x%02x%02x"%( (int)(bld[0]), (int)(bld[1]), (int)(bld[2]) )

    return col

def processImage():
    if ( fileSource == "cam" ):
       response = requests.get(  "http://192.168.0." + stationNo + "/image.jpg" )
       file = BytesIO( response.content )
       img = Image.open( file )
       img = img.resize( ( 240, 320 ) )
    else:
       img = Image.open( fileSource )

    pilImage = img.convert( 'L' )

    CNObjectRecognition.CNObjectLocalization( pilImage, label_image, sess, global_graph, colorF, threshold )

def eventLoop():
    processImage()
    if ( fileSource == "cam" ):
       root.after( 100, eventLoop )

def initialize_genderNet():
    genderNet = CNObjectRecognition.CNReadNN( genderNetFilename )

    gx_image = tf.placeholder( tf.float32, shape=[ 1, 32, 32, 1 ] )

    paddings = [ [0, 0], [2, 2], [2, 2], [0, 0] ]
    gh_pad1 = tf.pad(gx_image, paddings, "CONSTANT")
    gh_conv1 = tf.nn.tanh( CNObjectRecognition.conv2d( gh_pad1, CNObjectRecognition.CNConv2DWeights( genderNet[0] ) ) )
    gh_pool1 = tf.nn.max_pool(gh_conv1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    gh_pad2 = tf.pad( gh_pool1, paddings, "CONSTANT")
    gh_conv2 = tf.nn.tanh( CNObjectRecognition.conv2d( gh_pad2, CNObjectRecognition.CNConv2DWeights( genderNet[1] ) ) )
    #gh_pool2 = max_pool_2x2( gh_conv2 )
    gh_pool2 = tf.nn.max_pool(gh_conv2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    gh_pad3 = tf.pad( gh_pool2, paddings, "CONSTANT")
    gh_conv3 = tf.nn.tanh( CNObjectRecognition.conv2d( gh_pad3, CNObjectRecognition.CNConv2DWeights( genderNet[2] ) ) )
    gh_pool3 = tf.nn.max_pool(gh_conv3, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    h_reshape4 = tf.transpose( gh_pool3, perm=[ 0, 2, 3, 1 ] )

    h_flat4 = tf.reshape(h_reshape4, [-1,1024])


    gfc4 = tf.nn.sigmoid(tf.matmul(h_flat4, genderNet[3][1] ) + genderNet[3][0] )

    return( gx_image, gfc4, h_flat4 )

def initialize():
    initImgs = CNObjectRecognition.buildImagePyramid( Image.new( 'L', (240,320), 0 ) )
    gl = []
    for s in range( len(initImgs) ):
        x_image = tf.placeholder( tf.float32, shape=[ 1, initImgs[s].height, initImgs[s].width, 1 ] )
        gr1 = CNObjectRecognition.buildObjectRecognitionGraph( x_image, modelFilename )
        gl.append( (x_image, gr1 ) )

    return gl

parser = argparse.ArgumentParser()

parser.add_argument("-image",
                    help="jpg file containing image for recognition")
parser.add_argument("-model",
                    help="JSON file with model weights")
parser.add_argument("-threshold", type=float, default=.99,
                    help="threshold for output detection")
parser.add_argument("-station",
                    help="station number for webcam")
parser.add_argument("-gender",
                    help="filename with json model to attempt gender recognition" )
args = parser.parse_args()

print( "Setting threshold to ", args.threshold )

threshold = args.threshold
fileSource = args.image
modelFilename = args.model
stationNo = args.station
genderNetFilename = args.gender

if ( genderNetFilename ):
    ( x_genderImage, genderResult , save)  = initialize_genderNet()

root = tkinter.Tk()
root.geometry( '%dx%d' % (240,320) )
label_image = tkinter.Canvas( root )

sess = tf.Session()

global_graph = initialize()

sess.run(tf.global_variables_initializer())

eventLoop()

root.mainloop()
