#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing some auxiliary functions.

Created on Sat Mar 11 22:03:21 2023

@author: teuto
"""

import logging
from pathlib import Path

from framework import RGBPicture, Frame

def _dump_frame(frame: Frame, dump_path: Path,
                extension: str = "dump") -> None:
    dump_name: str = frame.band.name
    if not (fname := dump_path/f"{dump_name}_{extension}.fits").exists():
        frame.save_fits(fname)
        logger.info("Done dumping %s image.", dump_name)
    else:
        logger.warning("%s FITS file for %s exists, not overwriting.",
                       extension.title(), dump_name)


def _dump_rgb_channels(picture: RGBPicture, dump_path: Path) -> None:
    logger.info("Dumping stretched FITS files for each channel.")
    for channel in picture.rgb_channels:
        _dump_frame(channel, dump_path, "stretched")
    logger.info("Done dumping all channels for this combination.")

logger = logging.getLogger(__name__)
