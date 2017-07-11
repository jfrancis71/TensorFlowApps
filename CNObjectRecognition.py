#
# V Simple program for object recognition
# Requires pre trained neural net weight files in current dir in JSON format

#Example Usage:
#global_graphs = cnor.CZBuildObjectRecognitionGraphs( '/Users/julian/FaceNet.json', 240, 320 )
#output = cnor.CZHighlightImage( img, cnor.CZMultiScaleDetectObjects( img, global_graphs ) )


import tensorflow as tf
import numpy as np

from PIL import Image
from PIL import ImageDraw
import json
import time
import sys
import math

#Public Interfaces

#This returns a list of tuples, where first item in tuple is the image placeholder
#and the second item in the tuple is a tensorflow tensor of probabilities indicating
#presence of object
def CZBuildObjectRecognitionGraphs( modelFilename, width, height ):

    modelParameters = CNReadNN( modelFilename )

    initImgs = buildImagePyramid( Image.new( 'L', (width,height), 0 ) )
    gl = []
    for s in range( len(initImgs) ):
        x_image = tf.placeholder( tf.float32, shape=[ 1, initImgs[s].height, initImgs[s].width, 1 ] )
        gr1 = buildObjectRecognitionGraph( x_image, modelParameters )
        gl.append( (x_image, gr1 ) )

    return gl

CZFaceDetectSession = tf.Session()

#Note the conceptual difference with the CognitoZoo Mathematica implementation.
#Here we are making a single TensorFlow call to process all images at different scale.
#whereas MXNet implementation makes one call per image scale.
def CZMultiScaleDetectObjects( pilImage, tfGraphs, threshold=0.997 ):

    images = buildImagePyramid( pilImage.convert( 'L' ) )

    npImages = [ np.array( image ) / 255.0 for image in images ]

    tfOutputs = [ graph[1] for graph in tfGraphs ]

    fd = { tfGraphs[s][0] : [ np.array( [ npImages[s] ] ).transpose( 1,2,0 ) ] for s in range( len( images ) ) }

    outputPyramid = CZFaceDetectSession.run( tfOutputs, feed_dict = fd )

    objRet = []
    for s in range( len( outputPyramid ) ):
        objs = extractObjects( outputPyramid[s], threshold )
        scale = pilImage.width / images[s].width
        for obj in objs:
            objRet.append( ( scale*(16 + obj[0]-16), scale*(16 + obj[1]-16), scale*(16 + obj[0]+16), scale*(16 + obj[1]+16) ) )

    return objRet

def CZHighlightImage( pilImage, rectangles ):
    img = pilImage.copy()
    draw = ImageDraw.Draw( img )
    draw.rectangle
    for obj in rectangles:
        draw.rectangle( [ obj[0], obj[1], obj[2], obj[3] ], outline = 'green' )
        draw.rectangle( [ obj[0]-1, obj[1]-1, obj[2]+1, obj[3]+1 ], outline = 'green' )
        draw.rectangle( [ obj[0]-2, obj[1]-2, obj[2]+2, obj[3]+2 ], outline = 'green' )

    return img


#Private Code

#Takes a TensorFlow image array and builds a TensorFlow graph to process
#that image using the model parameters specified in modelFilename.
def buildObjectRecognitionGraph( tfImage, modelParameters ):

    h_conv1 = tf.nn.tanh( conv2d( tfImage, CNConv2DWeights( modelParameters[0] ) ) )
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.tanh( conv2d( h_pool1, CNConv2DWeights( modelParameters[1] ) ) )
    h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.tanh( conv2d( h_pool2, CNConv2DWeights( modelParameters[2] ) ) )

    h_conv4 = tf.nn.sigmoid( conv2d( h_conv3, CNConv2DWeights( modelParameters[3] ) ) )

    return ( h_conv4 )

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
