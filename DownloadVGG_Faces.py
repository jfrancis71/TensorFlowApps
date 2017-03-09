import os
from PIL import Image
import urllib.request as ur
import urllib.request
from io import BytesIO
import requests
import csv
import h5py
import numpy as np
import argparse

def retrieve_patch( rec ):
    response = requests.get( rec[1], timeout=10 )
    file = BytesIO( response.content )
    img = Image.open( file )
    ptch = img.crop( ( float(rec[2]),float(rec[3]),float(rec[4]), float(rec[5])) ).resize( (32,32) ).convert('L')
    return np.asarray( ptch, dtype=np.uint8 )

def retrieve_celeb( filename ):
  csvfile = open( filename, 'r')
  reader = csv.reader(csvfile, delimiter=' ')
  pts = []
  for row in reader:
    print( "image = ", row[0] )
    if ( row[8] != '1' ):
      continue
    try:
      pt = retrieve_patch( row )
      pts.append( pt )
    except IOError as e:
      continue
  return pts

#Parsing the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-folder",
                    help="folder for the HDF5 file and subfolder files")
args = parser.parse_args()

content_list = os.listdir( os.path.join( args.folder, "files") )

celebs = []
for celeb in content_list[0:100]:
  print( "Celeb", celeb )
  pts = retrieve_celeb( os.path.join( args.folder, "files", celeb ) )
  celebs = celebs + pts

file = h5py.File( os.path.join( args.folder, "dataset.hdf5" ), 'w')
dset = file.create_dataset("/patches", data = celebs )
file.close()
