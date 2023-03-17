#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxillary module for handling masking of frames.

Created on Sat Mar 11 21:52:05 2023

@author: teuto
"""

__version__ = "0.2"

from typing import Iterator

import numpy as np
from ruamel.yaml import YAML

from astropy.coordinates import Angle, SkyCoord
from regions import PixCoord, SkyRegion, CircleSkyRegion, \
                    RectangleSkyRegion, PolygonSkyRegion

from framework import Frame

__author__ = "Fabian Haberhauer"
__copyright__ = "Copyright 2023"

yaml = YAML()

def _maskparse(mask_dict) -> Iterator[SkyRegion]:
    for name, mask in mask_dict.items():
        sky_points = SkyCoord(mask["coords"], unit="deg")
        size = Angle(**mask["size"])

        if mask["type"] == "circ":
            region = CircleSkyRegion(sky_points[0], size)
        elif mask["type"] == "rect":
            region = RectangleSkyRegion(sky_points[0], *size)
        elif mask["type"] == "poly":
            region = PolygonSkyRegion(sky_points)
        else:
            region = None

        if region is not None:
            region.meta["name"] = name
            region.meta["comment"] = mask["mode"]

        yield region


def _region_mask(region: SkyRegion, frame: Frame) -> np.ndarray:
    pixels = PixCoord(*np.indices(frame.image.shape))
    pixel_region = region.to_pixel(frame.coords)
    cont = pixel_region.contains(pixels).T
    return cont


def _merge_masks(regions: list[SkyRegion], frame: Frame) -> np.ndarray:
    fill = all(region.meta["comment"] == "exclude" for region in regions)
    mask = np.full_like(frame.image, fill_value=fill, dtype=bool)
    for region in regions:
        cont = _region_mask(region, frame)
        mask = mask & ~cont | cont * (region.meta["comment"] == "limit")
    return mask


def get_mask(fname: str, frame: Frame) -> np.ndarray:
    with open(fname, "r", encoding="utf-8") as ymlfile:
        mask_dict = None#yaml.load(ymlfile)
    mask_regions = list(_maskparse(mask_dict))
    return _merge_masks(mask_regions, frame)
