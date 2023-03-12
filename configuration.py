#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module to hold Configurator classes.

Created on Sun Mar 12 17:48:17 2023

@author: teuto
"""

from dataclasses import dataclass, field
from typing import Union


@dataclass
class GeneralConfigurator:
    filenames: str = "${band_name}_${image_name}.fits"
    # e.g. $band_name/fits/$image_name.fits or ${image_name}_${band_name}.fits
    multiprocess: bool = False  # no (default) or yes
    jpeg_quality: int = 95      # see pillow documentation for details


@dataclass
class ProcessConfigurator:
    grey_mode: str = "normal"  # normal, lessback or moreback
    rgb_adjust: bool = False   # no (default) or yes
    alpha: float = 1.          # float (default=1.), typ. 1-2
    gamma_lum: float = 1.      # float (default=1.), typ. 1-3
    clip: float = 10.          # float (default=10.)
    nanmode: str = "max"       # max (default) or median
    skymode: str = "median"    # median (default), clipmedian, quantile or debug
    maxmode: str = "quantile"  # quantile (default), max or debug
    slice: Union[list[int], None] = None  # NULL (default) or [wx, wy]
    equal_offset: float = .1   # float (default=.1)
    equal_norm: bool = True    # yes (default) or no

@dataclass
class FiguresConfigurator:
    titlemode: str = "debug"       # debug or pub
    include_suptitle: bool = True  # yes (default) or no
    figsize: list[float] = field(default_factory=lambda: [3, 5.6])
    # [col, row]
    max_cols: int = 4              # int (default=4)
    centermark: bool = False       # no (default) or yes
    gridlines: bool = False        # no (default) or yes
    additional_roi: Union[list[list[int]], None] = None
    # each as [ra, dec] in decimal deg format


@dataclass
class Configurator:
    general: GeneralConfigurator = field(default_factory=lambda: GeneralConfigurator())
    process: ProcessConfigurator = field(default_factory=lambda: ProcessConfigurator())
    figures: FiguresConfigurator = field(default_factory=lambda: FiguresConfigurator())
    use_bands: list[str] = field(default_factory=list)
    combinations: list[list[str]] = field(default_factory=list)
