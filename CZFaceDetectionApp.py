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

import CZFaceDetection

#Example command line:
#python3 GenericObjectRecognitionApp.py -image cam -station 4 -threshold .99

face_graphs = CZFaceDetection.CZBuildFaceRecognitionGraphs( 240, 320 )

def processImage():
       img = Image.open( fileSource )
       face_graphs = CZFaceDetection.CZBuildFaceRecognitionGraphs( img.width, img.height )
       processedImage = CZFaceDetection.CZHighlightFaces( img, face_graphs, threshold )
       processedImage.show()

def eventLoop():
    response = requests.get(  "http://192.168.0." + stationNo + "/image.jpg" )
    file = BytesIO( response.content )
    img = Image.open( file )
    img = img.resize( ( 240, 320 ) )
    tkImage = ImageTk.PhotoImage( CZFaceDetection.CZHighlightFaces( img, face_graphs, threshold ) )
    label_image.img = tkImage
    label_image.pack(side = tkinter.TOP, expand=True, fill=tkinter.BOTH)
    label_image.create_image( 120,160, image=tkImage )

    if ( fileSource == "cam" ):
       root.after( 100, eventLoop )

parser = argparse.ArgumentParser()

parser.add_argument("-image",
                    help="jpg file containing image for recognition")
parser.add_argument("-threshold", type=float, default=.99,
                    help="threshold for output detection")
parser.add_argument("-station",
                    help="station number for webcam")
args = parser.parse_args()

print( "Setting threshold to ", args.threshold )

threshold = args.threshold
fileSource = args.image
stationNo = args.station

if ( fileSource == "cam" ):
   root = tkinter.Tk()
   root.geometry( '%dx%d' % (240,320) )
   label_image = tkinter.Canvas( root )
   eventLoop()
   root.mainloop()
else:
   processImage()
