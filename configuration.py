#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module to hold Configurator classes.

Created on Sun Mar 12 17:48:17 2023

@author: teuto
"""

from dataclasses import dataclass
from typing import Union

@dataclass
class Configurator:
    general
    process
    figures
    use_bands: list[str]
    combinations: list[list[str]]


@dataclass
class GeneralConfigurator:
    filenames: ${band_name}_${image_name}.fits
    # e.g. $band_name/fits/$image_name.fits or ${image_name}_${band_name}.fits
    multiprocess: no      # no (default) or yes
    jpeg_quality: 95      # see pillow documentation for details


@dataclass
class ProcessConfigurator:
    grey_mode: normal     # normal, lessback or moreback
    rgb_adjust: no        # no (default) or yes
    alpha: 1.             # float (default=1.), typ. 1-2
    gamma_lum: 1.         # float (default=1.), typ. 1-3
    clip: 10.             # float (default=10.)
    nanmode: max          # max (default) or median
    skymode: median       # median (default), clipmedian, quantile or debug
    maxmode: quantile     # quantile (default), max or debug
    slice: NULL           # NULL (default) or [wx, wy]
    equal_offset: .1      # float (default=.1)
    equal_norm: yes       # yes (default) or no

@dataclass
class FiguresConfigurator:
    titlemode: debug      # debug or pub
    include_suptitle: yes # yes (default) or no
    figsize: [3, 5.6]     # [col, row]
    max_cols: 4           # int (default=4)
    centermark: no        # no (default) or yes
    gridlines: no         # no (default) or yes
    additional_roi:       # each as [ra, dec] in decimal deg format
