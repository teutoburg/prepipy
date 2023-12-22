# prepipy

[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
![dev version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fteutoburg%2Fprepipy%2Fmain%2Fpyproject.toml&query=%24.tool.poetry.version&label=dev%20version&color=teal)

[![PyPI - Version](https://img.shields.io/pypi/v/prepipy)](https://pypi.org/project/prepipy/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/prepipy)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

PREtty PIctures using PYthon

## Overview

This package provides the ability to stretch and combine astronomical images from multiple bands into (RGB) colour images.

Images can be created in two main modes:
- JPEG image containing only the image in the pixel scale of the input, including coordinate information readable by e.g. [Aladin](https://aladin.u-strasbg.fr/).
- *DEVELOPMENT* Matplotlib image containing one ore more RGB conmibations from different bands in a grid layout. World coordinate axes are plotted on the images if available in the original input files. An additional sup-header containing the source name can be included. This mode also supportes multiple different options, such as plotting grid lines on top of the image, marking the center point in the image, of marking additional points of interest within the image, specified by world coordinates. By default, these images are saved in the pdf format.

## What does prepipy *not* do?

- Create an astrometric solution for input images. This must be provided in the FITS headers of the input images. Use a tool like e.g. [SCAMP](https://www.astromatic.net/software/scamp/) for this task.
- Resample the individual images to the same pixel scale. Input images must match exactly in terms of pixel scale, orientation and size (number of pixels along each axis). Prepipy assumes the input images can simply be added on a pixel-by-pixel basis. Use a tool like e.g. [SWarp](https://www.astromatic.net/software/swarp/) for this task.

## Example image

*Add text about example image, bands etc.*

![Example colour image of a star-forming region](https://nemesis.univie.ac.at/wp-content/uploads/2023/02/83.52054-5.39047.jpeg "Example image created using prepipy, centered around coordinates 83.52054, -5.39047.")


## Basic Usage

### JPEG mode

Current way to use from command line: run the `rgbcombo` script with arguments as described in the help message.

### Matplotlib mode

Development feature, not part of the current release.

### Input data

The input images are expected to fullfill the following criteria:
- FITS format images with the images data appearing in the primary HDU.
- Pixel scale and position matching across all input images. No additional resampling/reprojection is performed.
- WCS information is present in the FITS files.

### Setting options

The package currently uses two YAML configuration files to specify various options. These are referred to as the (general) `config` file and the `bands` file containing meta-information about the bands used in the original data.
If these files are not placed in the working directory, the path to them needs to be specified using the `-c` and `-b` command line options.

## Available classes

### Band

Data class used to store information about a passband.

The recommended way to construct instances is via a YAML config file.
Use the `Band.from_yaml_file(filename)` constructor to do so.
Alternatively, a minimalistic instance can be created by just providing the band name: `Band('foo')`. This will create a useable instance with all other parameters containing default or placeholder values.

### Frame

A `Frame` is an individual image taken in a given `Band`. Instances can be created manually or (recommended) either from a `astropy.io.fits.ImageHDU` object, `astropy.io.fits.HDUList` object plus an index or directly from a FITS file. Use the `from_hdu(hdu_object, band)`, `from_hdul(hdu_list_object, band, hdu_index)` or `from_fits(filename, band, hdu_index)` contructors respectively.

Operations like clipping, normalisation and stretching are performed as methods of the `Frame` class. Individual frames can be saved as single-HDU FITS files (`Frame.save_fits(filename)`).

### Picture

A `Picture` is a collection of one or more `Frame` objects. Frames can be added from a FITS file via `Picture.add_frame_from_file(filename, band)`, where `band` can be an instance of `Band` or, in the minimalistic case, a string containing the band name only. Frames can also be added directly from a 2D numpy array: `Picture.add_frame(image_array, band, header)`, where `header` can be an instance of `astropy.io.fits.Header` or `None`. A third option is to add `Frame` objects manually to the `Picture.frames` list. This has the downside that the frames's band is not checked against the other frames' bands already present in the picture. Normally, only one frame per band is allowed in a picture.

It is also possible to construct a `Picture` object from a 3D array containing 2D images, or to construct multiple instances from a 4D array. Warning: These features are currently highly experimental, not tested and not well documented.

The `Picture` class also provides a number of convenience properties, including `bands`, `primary_frame`, `image`, `coords`, `center`, `center_coords`, `center_coords_str`, `image_size`, `pixel_scale` and `image_scale`.

#### Subclasses of Picture

`RGBPicture` - subclass of `Picture` for handling 3-channel color composite images. A color channel is just a `Frame` object that is included in the `RGBPicture.rgb_channels` list. This attribute is set by calling the `select_rgb_channels(combination)` method. It is possible (and intended) to provide more than three frames for a `RGBPicture`, if multiple color composite images from different band combinations are desired. In this case, `select_rgb_channels(combination)` is called multiple times. Some operations modify only the frames which are set as color channels, so when processing multiple combinations, deep copies of the frames are created before they are modified as color channels.

`JPEGPicture` - subclass of `RGBPicture` for saving the final image as a JPEG (aka jpg) file. Currently one contains one method, `save_pil(filename, quality)`, which uses the interface provided by the `Pillow` package (version 9.4+ is required) to save the image. **NOTE**: in most cases, this is the class you'll want to use when actually dealing with color composite images, unless you want to manually process the 3D array containing the 3-channel image data in some other way.

# Acknowledgement

> This package was initially developed in the course of the [NEMESIS](https://nemesis.univie.ac.at) project at the University of Vienna. The NEMESIS project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No. 101004141.
