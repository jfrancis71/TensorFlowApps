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
#      output = CZFaceDetection.CZHighlightImage( img, cnor.CZDetectFaces( img ) )
#
#   You need to download the following two files and install them somewhere on youe search path.
#   FaceNet2Convolve.json from https://drive.google.com/file/d/0Bzhe0pgVZtNUMFhfcGJwRE9sRWc/view?usp=sharing
#   GenderNet.json from https://drive.google.com/file/d/0Bzhe0pgVZtNUaDY5ZzFiN2ZfTFU/view?usp=sharing
#


#Public Interfaces

#Timing: Approx 0.85s for 240x320 on MacOSX CPU
#Works like FindFaces, ie returns { {{x1,y1},{x2,y2}},... }
#   On the Caltech 1999 face dataset, we achieve a recognition rate of around 92% with
#   an average of 14% of false positives/image.
#   The Caltech dataset has 450 images where most faces are quite close to camera,
#   where images are of size 896x592. Most of these images are of good quality, but some
#   are challenging, eg. cartoon, significant obscuring of face or poor lighting conditions.
#   Reference comparison, FindFaces achieves 99.6% recognition, but 56% average false positive rate/image
def CZDetectFaces( pilImage, threshold=0.997 ):
    return CZDeleteOverlappingWindows( CZMultiScaleDetectObjects( pilImage, CZFaceNet, threshold ) )

def CZHighlightImage( pilImage, rectangles ):
    img = pilImage.copy()
    draw = ImageDraw.Draw( img )
    draw.rectangle
    for obj in rectangles:
        draw.rectangle( [ obj[0][0], obj[0][1], obj[1][0], obj[1][1] ], outline = 'green' )
        draw.rectangle( [ obj[0][0]-1, obj[0][1]-1, obj[1][0]+1, obj[1][1]+1 ], outline = 'green' )
        draw.rectangle( [ obj[0][0]-2, obj[0][1]-2, obj[1][0]+2, obj[1][1]+2 ], outline = 'green' )

    return img

#returns a gender score ranging from 0 (most likely female) to 1 (most likely male
def CZGender( pilImage ):
    norm = pilImage.convert( 'L' ).resize( ( 32, 32 ) )
    tfImage = [ np.array( [ np.array( norm ) ] ).transpose( (1,2,0) ) / 255. ]
    return CZFaceDetectSession.run( CZGenderNet[1], feed_dict = { CZGenderNet[0] : tfImage } )[0][0]

#Draws bounding boxes around detected faces and attempts to determine likely gender
def CZHighlightFaces( pilImage, threshold = .997 ):
    objs = CZDetectFaces( pilImage, threshold )
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

from PIL import ImageDraw
import json
import math
import os

def CZReadNN( fileName ):
    with open( fileName, 'r' ) as infile:
       j = json.load( infile )
    return j

CZFaceNetParameters = CZReadNN( os.path.join(os.path.expanduser('~'), 'FaceNet.json' ) )

#Takes a TensorFlow image array and builds a TensorFlow graph to process
#that image using the model parameters specified in modelFilename.
def buildObjectRecognitionGraph():

    x_image = tf.placeholder( tf.float32, shape=[ 1, None, None, 1 ] )

    h_conv1 = tf.nn.tanh( conv2d( x_image, CZFaceNetParameters[0] ) )
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.tanh( conv2d( h_pool1, CZFaceNetParameters[1] ) )
    h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.tanh( conv2d( h_pool2, CZFaceNetParameters[2] ) )

    h_conv4 = tf.nn.sigmoid( conv2d( h_conv3, CZFaceNetParameters[3] ) )

    return ( x_image, h_conv4 )

#   Note the first part of w is the biases, the second is the weights
def conv2d(x, w):
  return w[0] + tf.nn.conv2d(x, w[1], strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

CZFaceNet = buildObjectRecognitionGraph()
CZFaceDetectSession = tf.Session()

# Conceptually it is a sliding window (32x32) object detector running at a single scale.
#   In practice it is implemented convolutionally ( for performance reasons ) so the net
#   should be fully convolutional, ie no fully connected layers.
#   The net output should be a 2D array of numbers indicating a metric for likelihood of object being present.
#   The net filter should accept an array of real numbers (ie this works on greyscale images). You can supply a
#   colour image as input to the function, but this is just converted to greyscale before being fed to the neural net
#   Note the geometric factor 4 in mapping from the output array to the input array, this is because we have
#   downsampled twice in the neural network, so there is a coupling from this algorithm to the architecture
#   of the neural net supplied.
#   The rational for using a greyscale neural net is that I am particularly fascinated by shape (much more
#   than colour), so wanted to look at performance driven by that factor alone. A commercial use might take
#   a different view.
def CZSingleScaleDetectObjects( pilImage, tfGraph, threshold=0.997 ):
    npImage = np.array( pilImage.convert( 'L' ) ) / 255.0

    image = [ np.array( [ npImage ] ).transpose( 1,2,0 ) ]

    outputMap = CZFaceDetectSession.run( tfGraph[1], feed_dict = { tfGraph[0] : image } )

    extractPositions = np.transpose( np.nonzero( outputMap[0][:,:,0] > threshold ) )
    objs = list( map( lambda x: (outputMap[0][:,:,0][x[0],x[1]],x[1]*4,x[0]*4), extractPositions ) )

    return objs

# Implements a sliding window object detector at multiple scales.
#   The function resamples the image at scales ranging from a minimum width of 32 up to 800 at 20% scale increments.
#   The maximum width of 800 was chosen for 2 reasons: to limit inference run time and to limit the number of likely
#   false positives / image, implying the detector's limit is to recognise faces larger than 32/800 (4%) of the image width.
#   Note that if for example you had high resolution images with faces in the far distance and wanted to detect those and were
#   willing to accept false positives within the image, you might reconsider that tradeoff.
#   However, the main use case was possibly high resolution images where faces are not too distant with objective of limiting
#   false positives across the image as a whole.
def CZMultiScaleDetectObjects( pilImage, tfGraph, threshold=0.997 ):

    objRet = []
    for s in range( -1 + int( ( math.log( 32 ) - math.log( pilImage.width ) ) / math.log (.8 ) ) ):
        height = pilImage.height * .8**s
        width  = pilImage.width * .8**s
        print( "idx = ", s, " width = ", width )
        image = pilImage.resize( ( int(width), int(height) ) )
        objs = CZSingleScaleDetectObjects( image, tfGraph, threshold )
        scale = pilImage.width / image.width
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

CZGenderNetFilename = os.path.join(os.path.expanduser('~'), 'GenderNet.json' )

def CZBuildGenderRecognitionGraph():
    genderNetParameters = CZReadNN( CZGenderNetFilename )

    gx_image = tf.placeholder( tf.float32, shape=[ 1, 32, 32, 1 ] )

    paddings = [ [0, 0], [2, 2], [2, 2], [0, 0] ]
    gh_pad1 = tf.pad(gx_image, paddings, "CONSTANT")
    gh_conv1 = tf.nn.tanh( conv2d( gh_pad1, genderNetParameters[0] ) )
    gh_pool1 = tf.nn.max_pool(gh_conv1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    gh_pad2 = tf.pad( gh_pool1, paddings, "CONSTANT")
    gh_conv2 = tf.nn.tanh( conv2d( gh_pad2, genderNetParameters[1] ) )
    gh_pool2 = tf.nn.max_pool(gh_conv2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    gh_pad3 = tf.pad( gh_pool2, paddings, "CONSTANT")
    gh_conv3 = tf.nn.tanh( conv2d( gh_pad3, genderNetParameters[2] ) )
    gh_pool3 = tf.nn.max_pool(gh_conv3, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    h_reshape4 = tf.transpose( gh_pool3, perm=[ 0, 3, 1, 2 ] )

    h_flat4 = tf.reshape(h_reshape4, [-1,1024])

    gfc4 = tf.nn.sigmoid(tf.matmul(h_flat4, genderNetParameters[3][1] ) + genderNetParameters[3][0] )

    return( gx_image, gfc4 )

CZGenderNet = CZBuildGenderRecognitionGraph()

def blend( x, c1, c2 ):
    return x*c2 + (1-x)*c1

