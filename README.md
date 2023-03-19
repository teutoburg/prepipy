# prepipy

PREtty PIctures using PYthon

## Overview

This package provides the ability to stretch and combine astronomical images from multiple bands into (RGB) colour images.

Images can be created in two main modes:
- JPEG image containing only the image in the pixel scale of the input, including coordinate information readable by e.g. [Aladin](https://aladin.u-strasbg.fr/).
- Matplotlib image containing one ore more RGB conmibations from different bands in a grid layout. World coordinate axes are plotted on the images if available in the original input files. An additional sup-header containing the source name can be included. This mode also supportes multiple different options, such as plotting grid lines on top of the image, marking the center point in the image, of marking additional points of interest within the image, specified by world coordinates. By default, these images are saved in the pdf format.

![Example colour image of a star-forming region](https://nemesis.univie.ac.at/wp-content/uploads/2023/02/83.52054-5.39047.jpeg "Example image created using prepipy, centered around coordinates 83.52054, -5.39047.")


## Basic Usage

### JPEG mode

Current way to use from command line: run `rgbcombo.py` with arguments as described in the help message.

### Matplotlib mode

TBA

### Input data

The input images are expected to fullfill the following criteria:
- FITS format images with the images data appearing in the primary HDU.
- Pixel scale and position matching across all input images. No additional resampling/reprojection is performed.
- WCS information is present in the FITS files.

### Setting options

The package currently uses two YAML configuration files to specify various options. These are referred to as the (general) `config` file and the `bands` file containing meta-information about the bands used in the original data.
If these files are not placed in the working directory, the path to them needs to be specified using the `-c` and `-b` command line options.
