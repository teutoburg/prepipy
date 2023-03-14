#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing some auxiliary functions.

Created on Sat Mar 11 22:03:21 2023

@author: teuto
"""

import logging
from pathlib import Path
from dataclasses import replace, asdict, is_dataclass, fields

from ruamel.yaml import YAML

from framework import RGBPicture, Frame

yaml = YAML()

absolute_path = Path(__file__).resolve(strict=True).parent
DEFAULT_CONFIG_NAME = "config_single.yml"

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


def _recursive_replace(instance, **kwargs):
    for field in fields(instance):
        new_val = None
        if is_dataclass(field.type):
            new_val = replace(getattr(instance, field.name),
                              **{key: value for key, value in kwargs.items()
                                 if key in [subfield.name for subfield in
                                            fields(getattr(instance,
                                                           field.name))]})
        elif field.name in kwargs:
            new_val = kwargs[field.name]
        else:
            continue
        logger.debug("Replacing field %s with value %s.", field.name, new_val)
        setattr(instance, field.name, new_val)
    return instance


def _config_parser(default, config_path=None, cmd_args=None):
    if not (fallback_config_path := Path.cwd()/DEFAULT_CONFIG_NAME).exists():
        fallback_config_path = absolute_path/"config"/DEFAULT_CONFIG_NAME
    config_path = config_path or fallback_config_path
    logger.debug("Config path = %s", config_path)
    logger.debug(("Attempting to replace default config settings with values "
                  "from config file."))
    config_fromfile = _recursive_replace(default,
                                         **asdict(yaml.load(config_path)))
    logger.debug(("Attempting to replace default config settings with values "
                  "from command line arguments."))
    config = _recursive_replace(config_fromfile, **cmd_args)
    logger.debug("Final config file is: %s", config)
    return config


logger = logging.getLogger(__name__)
