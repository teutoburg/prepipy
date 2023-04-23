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
from string import Template
from typing import Iterable, Optional, ClassVar, Dict, Protocol, TypeVar, Any

from ruamel.yaml import YAML

from .framework import RGBPicture, Frame, Band
from .configuration import Configurator

yaml = YAML()

absolute_path = Path(__file__).resolve(strict=True).parent
DEFAULT_CONFIG_NAME = "config_single.yml"
DEFAULT_BANDS_NAME = "bands.yml"

class DataclassType(Protocol):
    __dataclass_fields__: ClassVar[Dict]

_D = TypeVar("_D", bound=DataclassType)


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


def _recursive_replace(instance: _D, **kwargs) -> _D:
    for field in fields(instance):
        new_val = None
        if is_dataclass(field.type):
            sfl = [subfield.name for subfield in fields(getattr(instance, field.name))]
            try:
                subkwargs = {key: value for key, value in kwargs[field.name].items() if key in sfl and value is not None}
            except KeyError:
                subkwargs = {key: value for key, value in kwargs.items() if key in sfl and value is not None}
            new_val = replace(getattr(instance, field.name), **subkwargs)
        elif field.name in kwargs:
            new_val = kwargs[field.name]
        else:
            continue
        logger.debug("Replacing field %s with value %s.", field.name, new_val)
        setattr(instance, field.name, new_val)
    return instance


def _config_parser(default: Configurator,
                   config_path: Optional[Path] = None,
                   cmd_args: Optional[dict[str, Any]] = None) -> Configurator:
    if not (fallback_config_path := Path.cwd()/DEFAULT_CONFIG_NAME).exists():
        fallback_config_path = absolute_path/"local"/DEFAULT_CONFIG_NAME
    config_path = config_path or fallback_config_path
    logger.debug("Config path = %s", config_path)
    logger.debug(("Attempting to replace default config settings with values "
                  "from config file."))
    try:
        config_fromfile = yaml.load(config_path)
        logger.debug("Loaded from config file: %s", config_fromfile)
        config = _recursive_replace(default, **asdict(config_fromfile))
    except FileNotFoundError:
        logger.error("Config file not found: %s", config_path)
        logger.warning("Proceeding with default config options...")
        config = default
    logger.debug(("Attempting to replace default config settings with values "
                  "from command line arguments."))
    if cmd_args is not None:
        config = _recursive_replace(config, **cmd_args)
    if not config.combinations:
        if cmd_args is not None:
            config.combinations = [cmd_args["rgb"]]
    logger.debug("Final config file is: %s", config)
    return config


def _fallback_bands(combos: list[list[str]]) -> Iterable[Band]:
    all_combos = {band for combo in combos for band in combo}
    bands = (Band(band) for band in all_combos)
    logger.warning(("Bands reconstructed from main config file do not "
                    "contain metadata. RGB combinations cannot be checked "
                    "for correct physical order."))
    return bands


def _bands_parser(config: Configurator,
                  bands_path: Optional[Path] = None) -> Iterable[Band]:
    use_bands = config.use_bands or None
    if not (fallback_bands_path := Path.cwd()/DEFAULT_BANDS_NAME).exists():
        fallback_bands_path = absolute_path/"local"/DEFAULT_BANDS_NAME
    bands_path = bands_path or fallback_bands_path
    try:
        bands: Iterable[Band] = Band.from_yaml_file(bands_path, use_bands)
    except FileNotFoundError:
        logger.error(("No bands config file found! Attempting to reconstruct "
                      "bands from main config file..."))
        bands = _fallback_bands(config.combinations)
    return bands


def create_description_file(picture: RGBPicture,
                            filename: Path,
                            template_path: Path,
                            stretchalgo: str = "prepipy") -> None:
    """
    Create html description file containing information about the picture.

    Description is created based on a template stored in `template_path`, which
    is expected to be a YAML file contain the following entries: title, coord,
    bands, chnls, salgo, footr. Information in the description includes: pixel
    scale, image size in angular units, picture center coordinates, bands used
    as colour channels including metadata for the bands (if available). Also
    included in the default template is the stretching algorithm used (can be
    set via `stretchalgo`) and a line about the ability to open the picture in
    Aladin, including a link.

    Parameters
    ----------
    picture : RGBPicture or subclass
        Any instance of RGBPicture or a subclass containing valid RGB channels.
    filename : Path
        Name and path of the file to which the html is written.
    template_path : Path
        Path to the template file.
    stretchalgo : str, optional
        Algorithm mentioned in the standard template. The default is "prepipy".

    Returns
    -------
    None.

    """
    outstr: str = ""
    colors: tuple[str, str, str] = ("Red", "Green", "Blue")
    ul_margin: str = "-20px"

    templates = yaml.load(template_path)

    coord_tmplt = Template(templates["coord"])
    bands_tmplt = Template(templates["bands"])
    chnls_tmplt = Template(templates["chnls"])
    salgo_tmplt = Template(templates["salgo"])

    center = picture.coords.pixel_to_world(*picture.center)
    center = center.to_string("hmsdms", precision=0)
    coord = coord_tmplt.substitute(center=center,
                                   pixel_scale=str(picture.pixel_scale),
                                   image_scale=str(picture.image_scale))

    if not all(channel.band.meta_set for channel in picture.rgb_channels):
        logger.warning(("Some metadata is missing for some bands. Description "
                        "file will likely contain placeholders."))
    chnls = "".join(chnls_tmplt.substitute(color=color,
                                           band_str=channel.band.verbose_str)
                       for color, channel in zip(colors, picture.rgb_channels))
    bands = bands_tmplt.substitute(channels=chnls, ul_margin=ul_margin)

    salgo = salgo_tmplt.substitute(stretchalgo=stretchalgo)

    outstr = templates["title"] + coord + bands + salgo + templates["footr"]
    filename.write_text(outstr, "utf-8")


logger = logging.getLogger(__name__)
