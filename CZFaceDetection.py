#
#   The weights for this model come from training in a neural network library called CognitoNet which is now retired.
#
#   That training session used images from the Face Scrub data set:
#   http: http://vintage.winklerbros.net/facescrub.html
#   H.-W. Ng, S. Winkler.
#   A data-driven approach to cleaning large face datasets.
#   Proc. IEEE International Conference on Image Processing (ICIP), Paris, France, Oct. 27-30, 2014.
#
#   Example use:
#      global_graphs = cnor.CZBuildFaceRecognitionGraphs( 240, 320 )
#      output = cnor.CZHighlightImage( img, cnor.CZMultiScaleDetectObjects( img, global_graphs ) )
#
#   You need to download the following two files and install them somewhere on youe search path.
#   FaceNet2Convolve.json from https://drive.google.com/file/d/0Bzhe0pgVZtNUMFhfcGJwRE9sRWc/view?usp=sharing
#   GenderNet.json from https://drive.google.com/file/d/0Bzhe0pgVZtNUaDY5ZzFiN2ZfTFU/view?usp=sharing
#


#Public Interfaces

def CZBuildFaceRecognitionGraphs( width, height ):
    return CZBuildObjectRecognitionGraphs( CZFaceNetParameters, width, height )

def CZDetectFaces( pilImage, tfGraphs, threshold=0.997 ):
    return CZDeleteOverlappingWindows( CZMultiScaleDetectObjects( pilImage, tfGraphs, threshold ) )

def CZHighlightImage( pilImage, rectangles ):
    img = pilImage.copy()
    draw = ImageDraw.Draw( img )
    draw.rectangle
    for obj in rectangles:
        draw.rectangle( [ obj[0][0], obj[0][1], obj[1][0], obj[1][1] ], outline = 'green' )
        draw.rectangle( [ obj[0][0]-1, obj[0][1]-1, obj[1][0]+1, obj[1][1]+1 ], outline = 'green' )
        draw.rectangle( [ obj[0][0]-2, obj[0][1]-2, obj[1][0]+2, obj[1][1]+2 ], outline = 'green' )

    return img

def CZGender( pilImage ):
    norm = pilImage.convert( 'L' ).resize( ( 32, 32 ) )
    tfImage = [ np.array( [ np.array( norm ) ] ).transpose( (1,2,0) ) / 255. ]
    return CZFaceDetectSession.run( CZGenderNet[1], feed_dict = { CZGenderNet[0] : tfImage } )[0][0]

def CZHighlightFaces( pilImage, tfGraphs, threshold = .997 ):
    objs = CZDetectFaces( pilImage, tfGraphs, threshold )
    img = pilImage.copy()

    draw = ImageDraw.Draw( img )
    draw.rectangle
    for obj in objs:
        crp = pilImage.crop( ( obj[0][0], obj[0][1], obj[1][0], obj[1][1] ) )
        gender = CZGender( crp )
        c1 = np.array( [ 255., 105., 180. ] )
        c2 = np.array( [ 0., 0., 255. ] )
        bld = blend( gender, c1, c2 )
        color = "#%02x%02x%02x"%( (int)(bld[0]), (int)(bld[1]), (int)(bld[2]) )
        draw.rectangle( [ obj[0][0], obj[0][1], obj[1][0], obj[1][1] ], outline = color )
        draw.rectangle( [ obj[0][0]-1, obj[0][1]-1, obj[1][0]+1, obj[1][1]+1 ], outline = color )
        draw.rectangle( [ obj[0][0]-2, obj[0][1]-2, obj[1][0]+2, obj[1][1]+2 ], outline = color )

    return img

#Private Implementation Code

import tensorflow as tf
import numpy as np

from PIL import Image
from PIL import ImageDraw
import json
import time
import sys
import math
import os

def CZReadNN( fileName ):
    with open( fileName, 'r' ) as infile:
       j = json.load( infile )
    return j

CZFaceNetParameters = CZReadNN( os.path.join(os.path.expanduser('~'), 'FaceNet.json' ) )

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

#Takes an image as an argument and returns a pyramid of images

#This returns a list of tuples, where first item in tuple is the image placeholder
#and the second item in the tuple is a tensorflow tensor of probabilities indicating
#presence of object
def CZBuildObjectRecognitionGraphs( parameters, width, height ):

    initImgs = buildImagePyramid( Image.new( 'L', (width,height), 0 ) )
    gl = []
    for s in range( len(initImgs) ):
        x_image = tf.placeholder( tf.float32, shape=[ 1, initImgs[s].height, initImgs[s].width, 1 ] )
        gr1 = buildObjectRecognitionGraph( x_image, parameters )
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

        extractPositions = np.transpose( np.nonzero( outputPyramid[s][0][:,:,0] > threshold ) )
        objs = list( map( lambda x: (outputPyramid[s][0][:,:,0][x[0],x[1]],x[1]*4,x[0]*4), extractPositions ) )

        scale = pilImage.width / images[s].width
        for obj in objs:
            objRet.append( ( obj[0], ( scale*(16 + obj[1]-16), scale*(16 + obj[2]-16) ), ( scale*(16 + obj[1]+16), scale*(16 + obj[2]+16) ) ) )

    return objRet

def CZIntersection( a, b ):
    xa=max(a[0][0],b[0][0])
    ya=max(a[0][1],b[0][1])
    xb=min(a[1][0],b[1][0])
    yb=min(a[1][1],b[1][1]),
    if ( xa>xb or ya>yb ):
        ans = 0
    else:
        ans = (xb-xa+1)*(yb-ya+1) 
    return ans

def CZArea( a ):
    return ( a[0][0]-a[1][0] ) * ( a[0][1]-a[1][1] )

def CZUnion( a, b ):
    return CZArea( a ) + CZArea( b ) - CZIntersection( a, b )

def CZIntersectionOverUnion(a, b):
    return CZIntersection( a, b ) / CZUnion( a, b  ) 

def CZDeleteOverlappingWindows( objects ):
    filtered = []
    for a in objects:
        idel = 0
        for b in objects:
            if ( CZIntersectionOverUnion( a[1:3], b[1:3] ) > .25 and a[0] < b[0] ):
                idel = 1
        if ( idel == 0 ):
            filtered.append( a[1:3] )

    return filtered


def extractObjects( outputMap, threshold ):
    extractPositions = np.transpose( np.nonzero( outputMap[0][:,:,0] > threshold ) )
    origCoords = list( map( lambda x: (x[1]*4,x[0]*4), extractPositions ) )
    return origCoords

def CNConv2DWeights( layer ):
    return ( layer[0], layer[1] )

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

CZGenderNetFilename = os.path.join(os.path.expanduser('~'), 'GenderNet.json' )

def CZBuildGenderRecognitionGraph():
    genderNetParameters = CZReadNN( CZGenderNetFilename )

    gx_image = tf.placeholder( tf.float32, shape=[ 1, 32, 32, 1 ] )

    paddings = [ [0, 0], [2, 2], [2, 2], [0, 0] ]
    gh_pad1 = tf.pad(gx_image, paddings, "CONSTANT")
    gh_conv1 = tf.nn.tanh( conv2d( gh_pad1, CNConv2DWeights( genderNetParameters[0] ) ) )
    gh_pool1 = tf.nn.max_pool(gh_conv1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    gh_pad2 = tf.pad( gh_pool1, paddings, "CONSTANT")
    gh_conv2 = tf.nn.tanh( conv2d( gh_pad2, CNConv2DWeights( genderNetParameters[1] ) ) )
    gh_pool2 = tf.nn.max_pool(gh_conv2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    gh_pad3 = tf.pad( gh_pool2, paddings, "CONSTANT")
    gh_conv3 = tf.nn.tanh( conv2d( gh_pad3, CNConv2DWeights( genderNetParameters[2] ) ) )
    gh_pool3 = tf.nn.max_pool(gh_conv3, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    h_reshape4 = tf.transpose( gh_pool3, perm=[ 0, 3, 1, 2 ] )

    h_flat4 = tf.reshape(h_reshape4, [-1,1024])

    gfc4 = tf.nn.sigmoid(tf.matmul(h_flat4, genderNetParameters[3][1] ) + genderNetParameters[3][0] )

    return( gx_image, gfc4 )

CZGenderNet = CZBuildGenderRecognitionGraph()

def blend( x, c1, c2 ):
    return x*c2 + (1-x)*c1

