# Emacs-plot venv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
if not 'img' in os.listdir("."):
    os.mkdir('img')

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

## example data
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np # for image example

np.random.seed(0)
example_image = np.random.rand(250,250,3)

fig = plt.figure(figsize=(10,10))
fig.suptitle('Experiment title here', fontsize=15)
one = fig.add_subplot(131)
plt.imshow(example_image[:,:,0])
one.set_title('Time one', fontsize=15)
plt.axis('off')
two = fig.add_subplot(132)
two.set_title('Time two', fontsize=15)
plt.imshow(example_image[:,:,1], cmap='gray')
plt.axis('off')
three = fig.add_subplot(133)
plt.imshow(example_image[:,:,2])
three.set_title('Time three', fontsize=15)
scalebar = ScaleBar(1,'um',location='lower right', fixed_value=25, color='black',frameon=True)
plt.gca().add_artist(scalebar)
plt.axis('off')
#plt.tight_layout()
plt.subplots_adjust(wspace=0.01)
plt.subplots_adjust(top=1.35)
plt.savefig('img/three_panel.png',bbox_inches='tight')

import skimage.measure
# make line profiles

np.random.seed(0)
example_image = np.random.rand(250,250,3)

start_coords = [50,50]
stop_coords = [150,150]

# make line profiles


start_y_line = skimage.measure.profile_line(example_image[:,:,0], start_coords, stop_coords)
middle_y_line = skimage.measure.profile_line(example_image[:,:,1], start_coords, stop_coords)
last_y_line = skimage.measure.profile_line(example_image[:,:,2], start_coords, stop_coords)
linescan_dist = (np.linalg.norm(np.array(start_coords) - np.array(stop_coords)))
line_axis = np.linspace(0,linescan_dist+1,len(start_y_line))

# column 1
fig = plt.figure(figsize=(10,8))
fig.suptitle('1040nm exposure', fontsize=15)
one = fig.add_subplot(231)
plt.imshow(example_image[:,:,0], cmap='gray')
plt.plot([start_coords[0],stop_coords[0]], [start_coords[1],stop_coords[1]],'r-', linewidth=4)
one.set_title('Start exposure', fontsize=15)
plt.axis('off')
onescan = fig.add_subplot(234)
plt.plot(line_axis, start_y_line,'-', color='black')
onescan.spines['right'].set_visible(False)
onescan.spines['top'].set_visible(False)
plt.ylabel('Fluorescence intensity (AU)', fontsize=15)
plt.xlabel(r'Distance ($\mu{}m$)', fontsize=15)


#column2
two = fig.add_subplot(232)
two.set_title('22 s', fontsize=15)
plt.imshow(example_image[:,:,1], cmap='gray')
plt.plot([start_coords[0],stop_coords[0]], [start_coords[1],stop_coords[1]],'r-', linewidth=4)
plt.axis('off')
middlescan = fig.add_subplot(235)
plt.plot(line_axis, middle_y_line,'-', color='black')
plt.axis('off')


#column3
three = fig.add_subplot(233)
plt.imshow(example_image[:,:,2], cmap='gray')
plt.plot([start_coords[0],stop_coords[0]], [start_coords[1],stop_coords[1]],'r-', linewidth=4)
three.set_title('45 s', fontsize=15)
plt.axis('off')
scalebar = ScaleBar(1,'um',location='lower right', fixed_value=25,color = 'black', frameon=True)
plt.gca().add_artist(scalebar)
lastscan = fig.add_subplot(236)
plt.plot(line_axis, last_y_line, '-', color='black')
plt.axis('off')
plt.subplots_adjust(wspace=0.01)
plt.subplots_adjust(top=.9)
fig.savefig('img/three_panel_with_scans.png', bbox_inches='tight')

np.random.seed(0)
example_image = np.random.rand(250,250,3)

roi_stim_coords_start = [50,50]
roi_stim_coords_end = [150,150]

# make line profiles


pre_exposure_y_line = skimage.measure.profile_line(example_image[:,:,0], start_coords, stop_coords)
post_exposure_y_line = skimage.measure.profile_line(example_image[:,:,1], start_coords, stop_coords)
linescan_dist = (np.linalg.norm(np.array(roi_stim_coords_start) - np.array(roi_stim_coords_end)))
stim_line_axis = np.linspace(0,linescan_dist+1,len(pre_exposure_y_line))

# TP 1
fig = plt.figure(figsize=(25,10))
one = fig.add_subplot(141)
plt.imshow(example_image[:,:,0], cmap='gray')
one.set_title('Pre-exposure',fontsize=20)
plt.plot([roi_stim_coords_start[0],roi_stim_coords_end[0]], 
         [roi_stim_coords_start[1],roi_stim_coords_end[1]], 'r', linewidth=4)
plt.axis('off')

# TP 2
two = fig.add_subplot(142)
two.set_title('During exposure',fontsize=20)
plt.imshow(example_image[:,:,1], cmap='gray')
plt.plot([roi_stim_coords_start[0],roi_stim_coords_end[0]], 
        [roi_stim_coords_start[1],roi_stim_coords_end[1]], 'r', linewidth=4)
plt.axis('off')

# TP 3 
three = fig.add_subplot(143)
plt.imshow(example_image[:,:,2], cmap='gray')
three.set_title('post-exposure',fontsize=20)
plt.axis('off')
scalebar = ScaleBar(1,'um',location='lower left', fixed_value=25,color = 'black', frameon=True)
plt.gca().add_artist(scalebar)
plt.plot([roi_stim_coords_start[0],roi_stim_coords_end[0]], 
         [roi_stim_coords_start[1],roi_stim_coords_end[1]], 'r', linewidth=4)

# linescans
four = fig.add_subplot(144)
plt.plot(stim_line_axis, pre_exposure_y_line, '--', color='blue',linewidth=2, label='pre-exposure')
plt.plot(stim_line_axis, post_exposure_y_line, '-', color='black',linewidth=2, label='post-exposure')
four.spines['right'].set_visible(False)
four.spines['top'].set_visible(False)
four.legend(loc='lower center', fontsize=15)
plt.ylabel('Fluorescence intensity (a.u.)',fontsize=15)
plt.xlabel(r'Distance ($\mu{}m$)',fontsize=15)
plt.subplots_adjust(wspace=None)
fig.savefig('img/in_a_row.png', bbox_inches='tight')

# create three channel image from 2 channel


import numpy as np

two_channel_image = np.random.rand(250,250,2)
print('Original image shape is {}'.format(two_channel_image.shape))

# make it three

now_three =np.dstack((two_channel_image[:,:,0], two_channel_image[:,:,1],
                   np.zeros_like(two_channel_image[:,:,0])))
print('Three channel image shape is {}'.format(now_three.shape))

two_channel_image = np.random.rand(250,250,2)
now_three =np.dstack((two_channel_image[:,:,0], two_channel_image[:,:,1], 
                      np.zeros_like(two_channel_image[:,:,0])))

# plot 2 channels of an image with scalebar
fig = plt.figure(figsize=(10,10))
one = fig.add_subplot(131)
plt.imshow(now_three[:,:,0], cmap="Greens_r") # note colormap
one.axis('off')
one.set_title('Channel 1',size=15)
two = fig.add_subplot(132)
plt.imshow(now_three[:,:,1] ,cmap="Reds_r") # note colormap
two.set_title('Channel 2',size=15)
two.axis('off')
scalebar = ScaleBar(1, units, location = 'lower right', 
                        fixed_value = 25, color = 'black', frameon = True)
three = fig.add_subplot(133)
plt.imshow(now_three)
plt.gca().add_artist(scalebar)
three.set_title('Merge', size=15)
three.axis('off')
plt.tight_layout()
fig.savefig('img/fake_channels.png', bbox_inches='tight')

## max project

import numpy as np


def max_project(image, start_slice = 0, stop_slice = None):
    """ takes ONE CHANNEL nd array image
        optional args = start_slice, stop_slice
        range to max project into
        returns new projection"""
    if stop_slice is None:
        stop_slice = image.shape[0]
    print(stop_slice)
    max_proj = [image[i,:,:] for i in range(start_slice,stop_slice)]
    return np.maximum.reduce(max_proj)

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
