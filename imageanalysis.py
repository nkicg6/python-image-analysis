import skimage.io
from skimage import img_as_float

def read_img(path, asfloat=True):
    img = skimage.io.imread(path)
    if not asfloat:
        return img
    img = img_as_float(img)
    return img


import tifffile


def metadata(path):
    """accepts the path to an ome-tif file, validates image
    dimensions, returns pixel size and units
    following ome-tif xml schema:
    http://www.openmicroscopy.org/Schemas/OME/2016-06"""
    try:
        with tifffile.TiffFile(path) as tif:
            raw_metadata = tif[0].image_description
            parse = parse = ET.fromstring(raw_metadata)
            pixels = parse.find('.//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels')
            # ensure all pixel units are the same
            assert pixels.get('PhysicalSizeXUnit') == pixels.get('PhysicalSizeZUnit') == \
            pixels.get('PhysicalSizeYUnit')
            # save units
            units = pixels.get('PhysicalSizeXUnit')
            # ensure x-size = y-size
            assert pixels.get('PhysicalSizeY') == pixels.get('PhysicalSizeX')
            # save pixel size
            size = pixels.get('PhysicalSizeY')
            # Z can be easily implemented (pizels.get(PhysicalSizeZ))
            return float(size), units
    except AssertionError:
        print("Image dimensions or units do not match")
    except ValueError as e:
        print("Incompatible format >>> {}".format(e))
    except Exception as x:
        print("Error. >>> {}".format(x))


scalebar = ScaleBar(pixelLength, units, location = 'lower right',
                   fixed_value = 25, color = 'black', frameon = False)


def scale_plot(img, imageSize, scale, units, color):
    plt.figure(figsize=imageSize)
    plt.imshow(img)
    plt.axis('off')
    scalebar = ScaleBar(scale, units, location = 'lower right',
                        fixed_value = 25, color = color, frameon = False)
    plt.gca().add_artist(scalebar)
