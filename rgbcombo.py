#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Debug module to run main functions from command line."""

__version__ = "0.2"

import sys
import gc
import argparse
import logging
from logging.config import dictConfig
from string import Template
from pathlib import Path
from shutil import get_terminal_size
from typing import Iterator
from time import perf_counter

from ruamel.yaml import YAML
import numpy as np
from tqdm import tqdm

from framework import RGBPicture, JPEGPicture, Band
from masking import get_mask
from auxiliaries import _dump_frame, _dump_rgb_channels, _config_parser, \
                        _bands_parser
from configuration import Configurator

__author__ = "Fabian Haberhauer"
__copyright__ = "Copyright 2023"

width, _ = get_terminal_size((50, 20))
width = int(.8 * width)
bar_width = max(width - 40, 10)
tqdm_fmt = f"{{l_bar}}{{bar:{bar_width}}}{{r_bar}}{{bar:-{bar_width}b}}"

absolute_path = Path(__file__).resolve(strict=True).parent
DEFAULT_BANDS_NAME = "bands.yml"

yaml = YAML()

class Error(Exception):
    """Base class for exeptions in this module."""


class FramesMisalignedError(Error):
    """Different shapes found in some frames of the same picture."""


def _gma(i, g):
    return np.power(i, 1/g)


def _pretty_info_log(msg_key, time=None, console_width=50) -> None:
    msg_dir = {"single": "Start RGB processing for single image...",
               "multiple": "Start RGB processing for multiple images...",
               "partial": "Start partial image processing...",
               "done": "Processing done",
               "aborted": "Critical error occured, process could not finish."}
    msg = msg_dir.get(msg_key, "Unknown log message.")
    logger.info(console_width * "*")
    logger.info("{:^{width}}".format(msg, width=console_width))
    if time is not None:
        msg = f"Elapsed time: {time:.2f} s"
        logger.info("{:^{width}}".format(msg, width=console_width))
    logger.info(console_width * "*")


def create_description_file(picture: RGBPicture,
                            filename: Path,
                            template_path: Path,
                            stretchalgo: str = "prepipy") -> None:
    outstr: str = ""
    colors: tuple[str, str, str] = ("Red", "Green", "Blue")
    ul_margin: str = "-20px"

    templates = yaml.load(template_path)

    coord = Template(templates["coord"])
    bands = Template(templates["bands"])
    chnls = Template(templates["chnls"])
    salgo = Template(templates["salgo"])

    center = picture.coords.pixel_to_world(*picture.center)
    center = center.to_string("hmsdms", precision=0)
    coord = coord.substitute(center=center,
                             pixel_scale=str(picture.pixel_scale),
                             image_scale=str(picture.image_scale))

    channels = "".join(chnls.substitute(color=color,
                                        band_str=channel.band.verbose_str)
                       for color, channel in zip(colors, picture.rgb_channels))
    bands = bands.substitute(channels=channels, ul_margin=ul_margin)

    salgo = salgo.substitute(stretchalgo=stretchalgo)

    outstr = templates["title"] + coord + bands + salgo + templates["footr"]
    with filename.open("w+") as file:
        file.write(outstr)


def create_picture(image_name: str,
                   input_path: Path,
                   fname_template: Template,
                   bands: Iterator[Band],  # or Iterable??
                   n_bands: int,
                   multi: int = 0) -> JPEGPicture:
    new_pic = JPEGPicture(name=image_name)
    if multi >= 2:
        logger.info("Using multiprocessing for preprocessing of frames...")
        new_pic.add_fits_frames_mp(input_path, fname_template, bands)
    else:
        for band in tqdm(bands, total=n_bands,
                         bar_format=tqdm_fmt):
            fname = fname_template.substitute(image_name=image_name,
                                              band_name=band.name)
            new_pic.add_frame_from_file(input_path/fname, band)
    logger.info("Picture %s fully loaded.", new_pic.name)
    return new_pic


def process_combination(pic,
                        combination, n_combos,
                        output_path,
                        generalconfig, processconfig):
    cols: str = "".join(combination)
    fname: str = f"{pic.name}_img_{cols}"

    logger.info("Processing image %s in %s.", pic.name, cols)
    pic.select_rgb_channels(combination, single=(n_combos == 1))

    if processconfig.mask_path is not None:
        try:
            mask = get_mask(processconfig.mask_path, pic.primary_frame)
            logger.info("Mask successfully loaded.")
        except FileNotFoundError:
            logger.error("Masking file not found, proceed using no mask.")
            mask = None
    else:
        logger.info("No masking file specified, using no mask.")
        mask = None

    # TODO: put these values in a separate config file in resources
    grey_values = {"normal": .3, "lessback": .08, "moreback": .5}
    grey_mode = processconfig.grey_mode

    if grey_mode != "normal":
        logger.info("Using grey mode \"%s\".", grey_mode)
        fname += f"_{grey_mode}"
    else:
        logger.info("Using normal grey mode.")

    pic.stretch_rgb_channels("stiff",
                             stiff_mode="prepipy2",
                             grey_level=grey_values[grey_mode],
                             skymode=processconfig.skymode,
                             mask=mask)

    if processconfig.rgb_adjust:
        pic.adjust_rgb(processconfig.alpha, _gma, processconfig.gamma_lum)
        logger.info("RGB sat. adjusting after contrast and stretch.")

    if pic.is_bright:
        logger.info(("Image is bright, performing additional color space "
                     "stretching to equalize colors."))
        pic.equalize("mean",
                     offset=processconfig.equal_offset,
                     norm=processconfig.equal_norm,
                     mask=mask)
    else:
        logger.warning(("No equalisation or normalisation performed on image "
                        "%s in %s!"), pic.name, cols)

    if generalconfig.fits_dump:
        _dump_rgb_channels(pic, output_path)

    savename = (output_path/fname).with_suffix(".jpeg")
    pic.save_pil(savename, generalconfig.jpeg_quality)

    if generalconfig.description:
        logger.info("Creating html description file.")
        html_template_path = absolute_path/"resources/html_templates.yml"
        create_description_file(pic, savename.with_suffix(".html"),
                                html_template_path)

    logger.info("Image %s in %s done.", pic.name, cols)
    return pic


def create_rgb_image(input_path: Path,
                     output_path: Path,
                     image_name: str,
                     config: Configurator,
                     bands: Iterator[Band]) -> RGBPicture:
    fname_template = Template(config.general.filenames)
    pic = create_picture(image_name, input_path, fname_template,
                         bands, len(config.use_bands),
                         config.general.multiprocess)

    if (n_shapes := len(set(frame.image.shape for frame in pic.frames))) > 1:
        if config.general.partial:
            logger.warning(("Found %d distinct shapes for %d frames. "
                            "Some frames are likely misaligned. Proceed "
                            "with caution!"), n_shapes, len(pic.frames))
        else:
            raise FramesMisalignedError((f"Found {n_shapes} distinct shapes "
                                         f"for {len(pic.frames)} frames."))

    if config.general.partial:
        logger.info("Partial processing selected, normalizing and dumping...")
        for frame in pic.frames:
            # TODO: multiprocess this if possible
            frame.clip(5, True)
            frame.normalize()
            _dump_frame(frame, output_path, "partial")
        logger.info("Dumping of partial frames complete, aborting process.")
        return pic

    for combo in tqdm(config.combinations,
                      total=(n_combos := len(config.combinations)),
                      bar_format=tqdm_fmt):
        pic = process_combination(pic, combo, n_combos, output_path,
                                  config.general, config.process)
    logger.info("Image %s fully completed.", pic.name)
    return pic


def setup_rgb_single(input_path, output_path, image_name, config,
                     bands_path=None) -> RGBPicture:
    start_time = perf_counter()
    if not config.general.partial:
        _pretty_info_log("single", console_width=width)
    else:
        _pretty_info_log("partial", console_width=width)

    bands = _bands_parser(config, bands_path)
    pic = create_rgb_image(input_path, output_path, image_name, config, bands)

    elapsed_time = perf_counter() - start_time
    _pretty_info_log("done", time=elapsed_time, console_width=width)
    return pic


def main() -> None:
    """Execute in script mode."""
    parser = argparse.ArgumentParser(prog="rgbcombo",
                                     description="""Combines RGB channels to
                                     color image including stretching.""")

    parser.add_argument("input_path",
                        type=Path,
                        help="""The file path to the input fits files
                        containing the images that shall be processed.""")
    parser.add_argument("image_name",
                        help="""Name stem of the images. Image names and paths
                        to data from individual bands can be defined in the
                        config file. Default structure is: <name>_<band>.fits,
                        e.g. V883_Ori_J.fits.""")
    parser.add_argument("-o", "--output-path",
                        help="""The file path where the combined images will
                        be saved to. If omitted, images are dumped back into
                        the input folder.""")
    parser.add_argument("-c", "--config-file",
                        type=Path,
                        help="""The name of the main config file to be used.
                        If omitted, the code will look for a file named
                        "config.yml" in the main package folder.""")
    parser.add_argument("-b", "--bands-file",
                        type=Path,
                        help="""The name of the band config file to be used.
                        If omitted, the code will look for a file named
                        "bands.yml" in the main package folder.""")
    parser.add_argument("-g", "--grey_mode",
                        default="normal",
                        choices=["normal", "lessback", "moreback"],
                        help="""Background grey level mode, default is
                        'normal'. If you see a monochromatic 'fog' in normal
                        mode, setting to 'lessback' may help.""")
    parser.add_argument("-m", "--multiprocess",
                        action="count",
                        default=0,
                        help="""Whether to use multiprocessing. Using this
                        option once will make use of multiprocessing to split
                        creation of multiple images between processes. Using it
                        twice will additionally use multiprocessing to pre-
                        process individual frames. This is only recommended for
                        very large images (> 10 Mpx). If the program is
                        executed for a single image, using this option once
                        will not have any effect.""")
    parser.add_argument("-d", "--description",
                        action="store_true",
                        help="""Whether to include a html description file
                        about the image. If set, a file with the same name as
                        each processes image will be saved in the output
                        directory.""")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-f", "--fits-dump",
                       action="store_true",
                       help="""Dump stretched single-band FITS files into the
                       specified output directory. This option cannot be used
                       in combination with the optionn -p.""")
    group.add_argument("-p", "--partial",
                       action="store_true",
                       help="""Perform only pre-processing and normalisation.
                       No stretching is performed, no RGB image is created, all
                       parameters associated with those processes will be
                       ignored. This option cannot be used in combination with
                       the option -f.""")
    parser.add_argument("--masking",
                        # type=Path,
                        dest="mask_path",
                        help="""The path to the YAML file containing masking
                        configuration, if any. May be absolute path or relative
                        to current working directory.""")
    parser.add_argument("--create-outfolders",
                        action="store_true",
                        help="""Whether to create a separate folder in the
                        output path for each picture, which may already exist.
                        Can only be used if -m option is set.""")
    # TODO: add verbosity count which affects logging level...
    args = parser.parse_args()

    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        logging.warning("No output path specified, dumping into input folder.")
        output_path = args.input_path

    config = _config_parser(Configurator(), config_path=args.config_file,
                            cmd_args=vars(args))

    try:
        setup_rgb_single(args.input_path, output_path, args.image_name, config,
                         args.bands_file)
    except Error as err:
        logger.critical("INTERNAL ERROR, ABORTING PROCESS", exc_info=err)
        _pretty_info_log("aborted", console_width=width)
    except Exception as err:
        logger.critical("UNEXPECTED ERROR, ABORTING PROCESS", exc_info=err)
        _pretty_info_log("aborted", console_width=width)


def _logging_configurator():
    main_logger = logging.getLogger("main")
    try:
        with (absolute_path/"log/logging_config.yml").open("r") as ymlfile:
            dictConfig(yaml.load(ymlfile))
    except FileNotFoundError as err:
        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s",
                                      "%H:%M:%S")
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        main_logger = logging.getLogger("main")
        main_logger.setLevel(logging.INFO)
        main_logger.addHandler(handler)
        main_logger.error(err)
    return main_logger


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger = _logging_configurator()
    assert logger.level == 20

    # root = Path("D:/Nemesis/data/HOPS")
    # path = root/"HOPS_99"
    # imgpath = root/"RGBs"
    # target = "HOPS_99"

    # root = Path("D:/Nemesis/data")
    # path = root/"stamps/LARGE/Orion"
    # imgpath = root/"comps_large"
    # target = "Hand"

    root = Path("C:/Users/ghost/Desktop/nemesis/outreach/regions")
    path = root/"input"
    imgpath = root/"JPEGS/new3"
    target = "outreach_4"

    # mypic = setup_rgb_single(path, imgpath, target, description=True)

    main()
    gc.collect()
    sys.exit(0)
