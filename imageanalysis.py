# using scikit-image to read images

import skimage.io
from skimage import img_as_float

def read_img(path, asfloat=False):
    img = skimage.io.imread(path)
    if not asfloat:
        return img
    img = img_as_float(img)
    return img

# my function for parsing metadata from ome-tiffs and imagej tiffs

import re
import tifffile
import xml.etree.ElementTree as ET


def metadata(path):
    """accepts the path to an ome-tif file, or imageJ tif file.
    Attempts to validates image dimensions, returns pixel size 
    and units
    following ome-tif xml schema:
    http://www.openmicroscopy.org/Schemas/OME/2016-06"""
    try:
        with tifffile.TiffFile(path) as tif:
            if tif.is_ome:
                raw_metadata = tif[0].image_description
                parse = ET.fromstring(raw_metadata)
                pixels = parse.find('.//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels')
                # ensure all pixel units are the same
                assert pixels.get('PhysicalSizeXUnit') == pixels.get('PhysicalSizeZUnit') == \
                pixels.get('PhysicalSizeYUnit')
                units = pixels.get('PhysicalSizeXUnit')
                assert pixels.get('PhysicalSizeY') == pixels.get('PhysicalSizeX')
                # save pixel size
                size = pixels.get('PhysicalSizeY')
                # Z can be easily implemented (pizels.get(PhysicalSizeZ))
                return float(size), units
            else:
                # hopefully it is imagej format
                raw_metadata = tif[0]
                # ensure xy pixels are the same size
                assert raw_metadata.x_resolution == raw_metadata.y_resolution
                # imageJ encodes as 'pixels per micron' so we should convert back
                size = 1/(raw_metadata.y_resolution[0]/raw_metadata.y_resolution[-1])
                check_units = raw_metadata.image_description.decode('utf-8')
                # regex to search for units. 
                regex_check = re.search('(?<=unit=)\w+',check_units)
                if regex_check.group(0) == 'micron':
                    # If micron, return Unicode micron
                    units = '\xb5m'
                    return float(size), units
                else:
                    return 'Could not determine pixel size. expected micron \
                    got >> {}'.format(regex_check.group(0))
    except AssertionError:
        print("Image dimensions or units do not match")
    except ValueError as e:
        print("Incompatible format >>> {}".format(e))
    except Exception as x:
        print("Error. >>> {}".format(x))

# example of metadata returned form an imageJ tif

import tifffile  
neun_path_example = '/Volumes/EXTENSION/RESTREPOLAB/images/neuronavigation/macklin_zeiss/2017-08-01/figures/MAX_2017-08-01_H001-017_img006.tif'
with tifffile.TiffFile(neun_path_example) as tif:
    images = tif.asarray()
    for page in tif:
        for tag in page.tags.values():
            t = tag.name, tag.value
            print(t)

# example of a scalebar

import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

scalebar = ScaleBar(pixelLength, units, location = 'lower right', 
                   fixed_value = 25, color = 'black', frameon = False)

# function for plotting an image with a scalebar

import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

def scale_plot(img, imageSize, scale, units, scalebar_length, color):
    plt.figure(figsize=imageSize)
    plt.imshow(img)
    plt.axis('off')
    scalebar = ScaleBar(scale, units, location = 'lower right', 
                        fixed_value = scalebar_length, color = color, frameon = False)
    plt.gca().add_artist(scalebar)

# create three channel image from 2 channel


import numpy as np

equal3 =np.dstack((auto_channel_equal,neun_channel_equal, 
                   np.zeros_like(neun_channel_equal)))

# plot 2 channels of an image with scalebar

import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar


f, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(20,20))
ax1.imshow(equal3[:,:,0], cmap="Greens_r") # note colormap
ax1.axis('off')
ax1.set_title('Autofluorescence',size=15)
ax2.imshow(equal3[:,:,1],cmap="Reds_r") # note colormap
ax2.set_title('NeuN',size=15)
ax2.axis('off')
scalebar = ScaleBar(neun_size, units, location = 'lower right', 
                        fixed_value = 300, color = 'white', frameon = False)
ax3.imshow(equal3)
plt.gca().add_artist(scalebar)
ax3.set_title('Merge', size=15)
ax3.axis('off')
plt.tight_layout()

# draw a line profile interactively


import matplotlib
matplotlib.use('TKAgg') # I don't have matplotlib installed as a framework so I need this..
from skimage import data
from skimage.viewer import ImageViewer
from skimage.viewer.plugins.lineprofile import LineProfile

def make_profile(image):
    """ 
    Takes a 2D image, gives an PyQt image
    viewer that you can make a ROI on. 
    returns line profile values
    """
    viewer = ImageViewer(image)
    viewer += LineProfile()
    _, line = zip(*viewer.show())
    return line

# compare image channels with SSIM

import skimage.measure

score, diff = skimage.measure.compare_ssim(image[0,:,:], image[1,:,:], full=True,
                                           gaussian_weights=True, sigma=1.5, 
                                           use_sample_covariance=False)
