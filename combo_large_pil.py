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

from framework import Picture, Frame, Band

TQDM_FMT = "{l_bar}{bar:50}{r_bar}{bar:-50b}"


def _gma(i, g):
    return np.power(i, 1/g)


def create_picture(image_name, input_path, bands, n_bands, multi=False):
    new_pic = Picture(name=image_name)
    if multi:
        new_pic.add_fits_frames_mp(input_path, bands)
    else:
        for band in tqdm(bands, total=n_bands,
                         bar_format=TQDM_FMT):
            fname = f"{image_name}_{band.name}.fits"
            new_pic.add_frame_from_file(input_path/fname, band)
    logger.info("Picture %s fully loaded.", new_pic.name)
    return new_pic


def create_rgb_image(input_path, output_path, image_name):
    logger.info("****************************************")
    logger.info("Start RGB processing...")
    logger.info("****************************************")

    with open("./config.yml", "r") as ymlfile:
        config = yaml.load(ymlfile, yaml.SafeLoader)
    with open("./bands.yml", "r") as ymlfile:
        bands_config = yaml.load(ymlfile, yaml.SafeLoader)

    bands = Band.from_yaml_dict(bands_config, config["use_bands"])
    channel_combos = config["combinations"]
    n_combo = len(channel_combos)

    pic = create_picture(image_name, input_path, bands,
                         len(config["use_bands"]),
                         config["process"]["multiprocess"])

    for combo in tqdm(channel_combos, total=n_combo,
                      bar_format=TQDM_FMT):
        cols = "".join(combo)
        logger.info("Processing image %s in %s.", pic.name, cols)
        pic.select_rgb_channels(combo, single=(n_combo == 1))

        grey_values = {"normal": .3, "lessback": .08, "moreback": .5}
        grey_mode = config["process"]["grey_mode"]
        pic.stretch_frames("stiff-d", only_rgb=True,
                           stretch_function=Frame.stiff_stretch,
                           stiff_mode="user3",
                           grey_level=grey_values[grey_mode])

        if config["process"]["rgb_adjust"]:
            pic.adjust_rgb(config["process"]["alpha"], _gma,
                           config["process"]["gamma_lum"])
            logger.info("RGB sat. adjusting after contrast and stretch.")

        if pic.is_bright:
            logger.info("Image is bright, performing additional color " +
                        "space stretching to equalize colors.")
            pic.equalize("median", offset=.1, norm=True)

        if grey_mode != "normal":
            logger.info("Used grey mode \"%s\".", grey_mode)
            fname = f"{pic.name}_img_{cols}_{grey_mode}.JPEG"
        else:
            logger.info("Used normal grey mode.")
            fname = f"{pic.name}_img_{cols}.JPEG"
        pic.save_pil(output_path/fname)

        logger.info("Image %s in %s done.", pic.name, cols)
    logger.info("Image %s fully completed.", pic.name)
    del pic
    gc.collect()

    logger.info("****************************************")
    logger.info("RGB processing done")
    logger.info("****************************************\n")


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
    path = root/"HOPS_99"
    imgpath = root/"RGBs"
    # root = Path("D:/Nemesis/data")
    # path = root/"stamps/LARGE/Orion"
    # imgpath = root/"comps_large"

    # target = "Hand"
    # target = "ONC"
    # target = "V883_Ori"
    target = "HOPS_99"
    # https://note.nkmk.me/en/python-pillow-concat-images/

    create_rgb_image(path, imgpath, target)

    gc.collect()
    sys.exit(0)
