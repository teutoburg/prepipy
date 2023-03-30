#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module to hold Configurator classes.

Created on Sun Mar 12 17:48:17 2023

@author: teuto
"""

__version__ = "0.2"

# TODO: somehow add comment dumping, maybe check ruyaml on GitHub?

from dataclasses import dataclass, field, asdict
from typing import Union, Optional

from ruamel.yaml import YAML, yaml_object, comments

__author__ = "Fabian Haberhauer"
__copyright__ = "Copyright 2023"

yaml = YAML()

@yaml_object(yaml)
@dataclass
class GeneralConfigurator:
    filenames: str = "${image_name}_${band_name}.fits"
    # e.g. $band_name/fits/$image_name.fits or ${image_name}_${band_name}.fits
    multiprocess: int = 0      # int (default=0), details see cmd args
    jpeg_quality: int = 95     # see pillow documentation for details
    description: bool = False  # False (default) or True
    fits_dump: bool = False    # False (default) or True
    partial: bool = False      # False (default) or True
    create_outfolders: bool = False  # False (default) or True
    hdu: int = 0               # FITS HDU where image data is stored


@yaml_object(yaml)
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
    slice: Optional[list[int]] = None  # NULL (default) or [wx, wy]
    equal_offset: float = .1   # float (default=.1)
    equal_norm: bool = True    # yes (default) or no
    mask_path: Optional[str] = None  # path (relative or absolute) to mask file


@yaml_object(yaml)
@dataclass
class FiguresConfigurator:
    titlemode: str = "debug"       # debug or pub
    include_suptitle: bool = True  # yes (default) or no
    figsize: list[float] = field(default_factory=lambda: [3, 5.6])
    # [col, row]
    max_cols: int = 4              # int (default=4)
    centermark: bool = False       # no (default) or yes
    gridlines: bool = False        # no (default) or yes
    additional_roi: Optional[list[list[float]]] = None
    # each as [ra, dec] in decimal deg format


@yaml_object(yaml)
@dataclass
class Configurator:
    general: GeneralConfigurator = field(default_factory=lambda: GeneralConfigurator())
    process: ProcessConfigurator = field(default_factory=lambda: ProcessConfigurator())
    figures: FiguresConfigurator = field(default_factory=lambda: FiguresConfigurator())
    use_bands: list[str] = field(default_factory=list)
    combinations: list[list[str]] = field(default_factory=list)


# BUG: this will swallow the !class tags
def add_comments(instance, commentfile):
    codict = yaml.load(commentfile)
    comap = comments.CommentedMap()
    indict = asdict(instance)
    for key in indict:
        ncomap = comments.CommentedMap()
        ncomap.update(indict[key])
        comap[key] = ncomap
        if not key in codict:
            continue
        for comkey in comap[key]:
            if comkey in codict[key]:
                comap[key].yaml_add_eol_comment(codict[key][comkey], comkey,
                                                column=26)
    return comap
