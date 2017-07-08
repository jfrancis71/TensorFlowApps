#
# V Simple program for object recognition
# Requires pre trained neural net weight files in current dir in JSON format


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

#Public Interfaces

#Takes a TensorFlow image array and builds a TensorFlow graph to process
#that image using the model parameters specified in modelFilename.
def buildObjectRecognitionGraph( tfImage, modelFilename ):

    modelNet = CNReadNN( modelFilename )

    h_conv1 = tf.nn.tanh( conv2d( tfImage, CNConv2DWeights( modelNet[0] ) ) )
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.tanh( conv2d( h_pool1, CNConv2DWeights( modelNet[1] ) ) )
    h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.tanh( conv2d( h_pool2, CNConv2DWeights( modelNet[2] ) ) )

    h_conv4 = tf.nn.sigmoid( conv2d( h_conv3, CNConv2DWeights( modelNet[3] ) ) )

    return ( h_conv4 )

#Note the conceptual difference with the CognitoZoo Mathematica implementation.
#Here we are making a single TensorFlow call to process all images at different scale.
#whereas MXNet implementation makes one call per image scale.

#Takes a pilImage, builds an image pyramid and applies the tfGraphs to each level
#of the image pyramid and writes the graphics output into label_image, a TkImage object.
def CNObjectLocalization( pilImage, label_image, sess, tfGraphs, colorF, threshold=0.997 ):
    start= time.clock()
    tkImage = ImageTk.PhotoImage( pilImage )
    label_image.img = tkImage
    label_image.pack(side = tkinter.TOP, expand=True, fill=tkinter.BOTH)
    label_image.create_image( 120,160, image=tkImage )

    objs = CZMultiScaleDetectObjects( pilImage, sess, tfGraphs, colorF, threshold )

    for obj in objs:
        label_image.create_rectangle( obj[0], obj[1], obj[2], obj[3], outline=obj[4], width=3 )

    print( "Time lapsed=", time.clock() - start )

def CZMultiScaleDetectObjects( pilImage, sess, tfGraphs, colorF, threshold=0.997 ):

    images = buildImagePyramid( pilImage )

    npImages = []
    for s in range( len( images ) ):
        npImage = np.array( images[s] ) / 255.0
        npImages.append( npImage )

    tfOutputs = []
    tfImages = []
    for s in range( len( images ) ):
        tfOutputs.append( tfGraphs[s][1] )
        tfImages.append( [ np.array( [ npImages[s] ] ).transpose( (1,2,0) ) ] )

    fd = { tfGraphs[s][0] : tfImages[s] for s in range( len( tfImages ) ) }

    outputPyramid = sess.run( tfOutputs, feed_dict = fd )

    objRet = []
    for s in range( len( outputPyramid ) ):
        objs = extractObjects( outputPyramid[s], threshold )
        scale = pilImage.width / images[s].width
        for obj in objs:
            col = colorF( npImages[s][obj[1]:obj[1]+32,obj[0]:obj[0]+32] )
            objRet.append( ( scale*(16 + obj[0]-16), scale*(16 + obj[1]-16), scale*(16 + obj[0]+16), scale*(16 + obj[1]+16), col ) )

    return objRet


#Private Code

def extractObjects( outputMap, threshold ):
    extractPositions = np.transpose( np.nonzero( outputMap[0][:,:,0] > threshold ) )
    origCoords = list( map( lambda x: (x[1]*4,x[0]*4), extractPositions ) )
    return origCoords

def CNConv2DWeights( layer ):
    return ( layer[0], layer[1] )

def CNReadNN( fileName ):
    with open( fileName, 'r' ) as infile:
       j = json.load( infile )
    return j

#Takes an image as an argument and returns a pyramid of images
#ie list of images of decreasing sizes
def buildImagePyramid( pilImage ):
    images = []
    for s in range( -1 + int( ( math.log( 32 ) - math.log( pilImage.width ) ) / math.log (.8 ) ) ):
        height = pilImage.height * .8**s
        width  = pilImage.width * .8**s
        print( "idx = ", s, " width = ", width )
        images.append( pilImage.resize( ( int(width), int(height) ) ) )
    return images

#   Note the first part of w is the biases, the second is the weights
def conv2d(x, w):
  return w[0] + tf.nn.conv2d(x, w[1], strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
