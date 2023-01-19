# -*- coding: utf-8 -*-
"""

"""
import sys
import gc
import logging
from logging.config import dictConfig
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from framework import JPEGPicture, Frame, Band

TQDM_FMT = "{l_bar}{bar:50}{r_bar}{bar:-50b}"


def _gma(i, g):
    return np.power(i, 1/g)


def _pretty_info_log(msg_key):
    msg_dir = {"single": "Start RGB processing for single Image...",
               "multiple": "Start RGB processing for multiple Images...",
               "done": "RGB processing done"}
    logger.info("****************************************")
    logger.info(msg_dir.get(msg_key, "Unknown log message."))
    logger.info("****************************************")


def create_picture(image_name, input_path, bands, n_bands, multi=False):
    new_pic = JPEGPicture(name=image_name)
    if multi:
        new_pic.add_fits_frames_mp(input_path, bands)
    else:
        for band in tqdm(bands, total=n_bands,
                         bar_format=TQDM_FMT):
            fname = f"{image_name}_{band.name}.fits"
            new_pic.add_frame_from_file(input_path/fname, band)
    logger.info("Picture %s fully loaded.", new_pic.name)
    return new_pic


def get_bands(fname, config):
    with open(fname, "r") as ymlfile:
        bands_config = yaml.load(ymlfile, yaml.SafeLoader)
    bands = Band.from_yaml_dict(bands_config, config["use_bands"])
    return bands


def create_rgb_image(input_path, output_path, image_name,
                     config, bands, channel_combos):
    pic = create_picture(image_name, input_path, bands,
                         len(config["use_bands"]),
                         config["process"]["multiprocess"])

    n_combos = len(channel_combos)
    for combo in tqdm(channel_combos, total=n_combos, bar_format=TQDM_FMT):
        cols = "".join(combo)
        logger.info("Processing image %s in %s.", pic.name, cols)
        pic.select_rgb_channels(combo, single=(n_combos == 1))

        grey_values = {"normal": .3, "lessback": .08, "moreback": .5}
        grey_mode = config["process"]["grey_mode"]
        pic.stretch_frames("stiff-d", only_rgb=True,
                           stretch_function=Frame.stiff_stretch,
                           stiff_mode="user3",
                           grey_level=grey_values[grey_mode],
                           skymode=config["process"]["skymode"])

        if config["process"]["rgb_adjust"]:
            pic.adjust_rgb(config["process"]["alpha"], _gma,
                           config["process"]["gamma_lum"])
            logger.info("RGB sat. adjusting after contrast and stretch.")

        if pic.is_bright:
            logger.info(("Image is bright, performing additional color space "
                         "stretching to equalize colors."))
            pic.equalize("median", offset=.1, norm=True)
        else:
            logger.warning(("No equalisation or normalisation performed on "
                            "image %s in %s!"),
                           pic.name, cols)

        if grey_mode != "normal":
            logger.info("Used grey mode \"%s\".", grey_mode)
            fname = f"{pic.name}_img_{cols}_{grey_mode}.JPEG"
        else:
            logger.info("Used normal grey mode.")
            fname = f"{pic.name}_img_{cols}.JPEG"
        pic.save_pil(output_path/fname)
        logger.info("Image %s in %s done.", pic.name, cols)
    logger.info("Image %s fully completed.", pic.name)


def setup_rgb_single(input_path, output_path, image_name,
                     config_name=None, bands_name=None):
    _pretty_info_log("single")

    if config_name is None:
        config_name = "./config_single.yml"
    with open(config_name, "r") as ymlfile:
        config = yaml.load(ymlfile, yaml.SafeLoader)

    if bands_name is None:
        bands_name = "./bands.yml"
    bands = get_bands(bands_name, config)
    channel_combos = config["combinations"]

    create_rgb_image(input_path, output_path, image_name, config, bands,
                     channel_combos)

    _pretty_info_log("done")


def setup_rgb_multiple(input_path, output_path, image_names,
                       config_name=None, bands_name=None,
                       create_outfolder=False):
    _pretty_info_log("multiple")

    if config_name is None:
        config_name = "./config.yml"
    with open(config_name, "r") as ymlfile:
        config = yaml.load(ymlfile, yaml.SafeLoader)

    if bands_name is None:
        bands_name = "./bands.yml"
    bands = get_bands(bands_name, config)
    channel_combos = config["combinations"]

    for image_name in image_names:
        if create_outfolder:
            imgpath = output_path/image_name
            imgpath.mkdir(parents=True, exist_ok=True)
        else:
            imgpath = output_path
        create_rgb_image(input_path, imgpath, image_name, config, bands,
                         channel_combos)

    _pretty_info_log("RGB processing done")


def _logging_configurator():
    main_logger = logging.getLogger("main")
    try:
        with open("./log/logging_config.yml", "r") as ymlfile:
            dictConfig(yaml.load(ymlfile, yaml.SafeLoader))
    except FileNotFoundError as err:
        logging.error(err)
        logging.basicConfig()
        logger.setLevel(logging.INFO)
    return main_logger


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger = _logging_configurator()
    root = Path("D:/Nemesis/data/HOPS")
    path = root/"HOPS_53"
    imgpath = root/"RGBs"
    # root = Path("D:/Nemesis/data")
    # path = root/"stamps/LARGE/Orion"
    # imgpath = root/"comps_large"
    root = Path("C:/Users/ghost/Desktop/nemesis/iras32")
    path = root
    imgpath = root

    # target = "Hand"
    target = "HOPS_53"
    target = "larger_IRAS-32"
    # https://note.nkmk.me/en/python-pillow-concat-images/

    setup_rgb_single(path, imgpath, target)

    # root = Path("D:/Nemesis/data/perseus")
    # path = root/"stamps/"

    # for i in range(1, 4):
    #     target = f"IC 348-{i}"
    #     setup_rgb(path, root/"RGBs", target)

    gc.collect()
    sys.exit(0)
