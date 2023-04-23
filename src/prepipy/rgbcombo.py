#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Debug module to run main functions from command line."""

__version__ = "0.6"

import sys
import gc
import argparse
import logging
from logging.config import dictConfig
from string import Template
from pathlib import Path
from shutil import get_terminal_size
from collections.abc import Iterable, Collection, Callable
from typing import Optional, TypeVar
from typing_extensions import ParamSpec
from time import perf_counter

from ruamel.yaml import YAML
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from colorama import Fore, Back, Style

from .framework import JPEGPicture, Band
from .framework import logger as framework_logger
from .masking import get_mask
from . import auxiliaries
from .configuration import Configurator, GeneralConfigurator, ProcessConfigurator

__author__ = "Fabian Haberhauer"
__copyright__ = "Copyright 2023"

_P = ParamSpec("_P")
_T = TypeVar("_T")

width, _ = get_terminal_size((50, 20))
width = int(.8 * width)
bar_width = max(width - 40, 10)
tqdm_fmt = f"{{l_bar}}{{bar:{bar_width}}}{{r_bar}}{{bar:-{bar_width}b}}"

absolute_path = Path(__file__).resolve(strict=True).parent

yaml = YAML()

class Error(Exception):
    """Base class for exeptions in this module."""


class FramesMisalignedError(Error):
    """Different shapes found in some frames of the same picture."""


class PixelScaleError(Error):
    """Different pixel scales found in some frames of the same picture."""


class BandNotFoundError(Error):
    """The frame for the selected band could not be found."""


class ColoredFormatter(logging.Formatter):
    """Deal with custom colored logging output to console."""

    format_msg = "%(message)s"
    format_level = "%(levelname)s: "

    FORMATS = {
        logging.DEBUG: Fore.BLUE + format_msg + Style.RESET_ALL,
        logging.INFO: Fore.GREEN + format_msg + Style.RESET_ALL,
        logging.WARNING: (Fore.CYAN + format_level + format_msg
                          + Style.RESET_ALL),
        logging.ERROR: (Fore.RED + Style.BRIGHT + format_level + format_msg
                        + Style.RESET_ALL),
        logging.CRITICAL: (Fore.YELLOW + Back.RED + Style.BRIGHT + format_level
                           + format_msg + Style.RESET_ALL)
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def _pretty_info_log(msg_key: str, time: Optional[float] = None,
                     console_width: int = 50) -> None:
    msg_dir = {"single": "Start RGB processing for single image...",
               "multiple": "Start RGB processing for multiple images...",
               "partial": "Start partial image processing...",
               "done": "Processing done",
               "aborted": "Critical error occured, process could not finish."}
    msg = msg_dir.get(msg_key, "Unknown log message.")
    msg = f"{msg:^{console_width}}"
    logger.log(25, console_width * "*")
    logger.log(25, msg)
    if time is not None:
        msg = f"Elapsed time: {time:.2f} s"
        msg = f"{msg:^{console_width}}"
        logger.log(25, msg)
    logger.log(25, console_width * "*")


def pretty_infos(function: Callable[_P, _T]) -> Callable[_P, _T]:
    """Decorator for logging start and end msg, including execution time."""
    def _wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        start_time = perf_counter()
        if not args[3].general.partial:
            _pretty_info_log("single", console_width=width)
        else:
            _pretty_info_log("partial", console_width=width)
        result = function(*args, **kwargs)
        elapsed_time = perf_counter() - start_time
        _pretty_info_log("done", time=elapsed_time, console_width=width)
        return result
    return _wrapper


def create_picture(image_name: str,
                   input_path: Path,
                   fname_template: Template,
                   bands: Collection[Band],
                   multi: int = 0,
                   hdu: int = 0) -> JPEGPicture:
    """Factory for JPEGPicture class."""
    new_pic = JPEGPicture(name=image_name)
    if multi >= 2:
        logger.info("Using multiprocessing for preprocessing of frames...")
        new_pic.add_fits_frames_mp(input_path, fname_template, bands)
    else:
        with logging_redirect_tqdm(loggers=all_loggers):
            failings = False
            for band in tqdm(bands, bar_format=tqdm_fmt):
                fname = fname_template.substitute(image_name=image_name,
                                                  band_name=band.name)
                try:
                    new_pic.add_frame_from_file(input_path/fname,
                                                band, hdu=hdu)
                except FileNotFoundError:
                    logger.error("No input file found for %s band", band.name)
                    failings = True
                    continue
    if not failings:
        logger.info("Picture %s successfully loaded.", new_pic.name)
    else:
        logger.warning(("Some frames could not be loaded for picture %s. "
                        "Any combinations containing those will be skipped."),
                       new_pic.name)
    return new_pic


def process_combination(pic: JPEGPicture,
                        combination: list[str],
                        single: bool,
                        output_path: Path,
                        generalconfig: GeneralConfigurator,
                        processconfig: ProcessConfigurator) -> JPEGPicture:
    """Process one RGB combination for a picture instance."""
    cols: str = "".join(combination)
    fname: str = f"{pic.name}_img_{cols}"

    logger.info("Processing image %s in %s.", pic.name, cols)
    try:
        pic.select_rgb_channels(combination, single=single)
    except KeyError as err:
        logger.error("No frame for band %s found.", err)
        raise BandNotFoundError(str(err))

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
    grey_values = dict(yaml.load(absolute_path/"resources/grey_values.yml"))

    if processconfig.grey_mode != "normal":
        logger.info("Using grey mode \"%s\".", processconfig.grey_mode)
        fname += f"_{processconfig.grey_mode}"
    else:
        logger.info("Using normal grey mode.")

    # BUG: gamma_lum from config is not passed to min_inten etc...
    pic.stretch_rgb_channels("stiff",
                             stiff_mode="prepipy2",
                             grey_level=grey_values[processconfig.grey_mode],
                             skymode=processconfig.skymode,
                             mask=mask)

    if processconfig.rgb_adjust:
        pic.adjust_rgb(processconfig.alpha, processconfig.gamma_lum)
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
        auxiliaries._dump_rgb_channels(pic, output_path)

    savename = (output_path/fname).with_suffix(".jpeg")
    pic.save_pil(savename, generalconfig.jpeg_quality)

    if generalconfig.description:
        logger.info("Creating html description file.")
        html_template_path = absolute_path/"resources/html_templates.yml"
        auxiliaries.create_description_file(pic, savename.with_suffix(".html"),
                                            html_template_path)

    logger.info("Image %s in %s done.", pic.name, cols)
    return pic


@pretty_infos
def create_rgb_image(input_path: Path,
                     output_path: Path,
                     image_name: str,
                     config: Configurator,
                     bands: Collection[Band]) -> JPEGPicture:
    """
    Create, process, stretch, combine and save RGB image(s).

    Parameters
    ----------
    input_path : Path
        Folder containing raw monochromatic input images.
    output_path : Path
        Folder in which to save the final image(s).
    image_name : str
        Common part of the input images file names, also output name stem.
    config : Configurator
        Configurator instance containing config options.
    bands : Iterable[Band]
        Iterable of Band instances.

    Raises
    ------
    FramesMisalignedError
        Raised if frames are of different shapes.

    Returns
    -------
    JPEGPicture
        Final instance of JPEGPicture, containing the last processed RGB
        combination as rgb_channels.

    """
    fname_template = Template(config.general.filenames)
    # BUG: if use_bands is None, len() throws an error
    pic = create_picture(image_name, input_path,
                         fname_template, bands,
                         config.general.multiprocess,
                         config.general.hdu)

    if (n_shapes := len(set(frame.image.shape for frame in pic.frames))) > 1:
        if config.general.partial:
            logger.warning(("Found %d distinct shapes for %d frames. "
                            "Some frames are likely misaligned. Proceed "
                            "with caution!"), n_shapes, len(pic.frames))
        else:
            raise FramesMisalignedError((f"Found {n_shapes} distinct shapes "
                                         f"for {len(pic.frames)} frames."))

    if (n_scales := len(set(frame.pixel_scale for frame in pic.frames))) > 1:
        if config.general.partial:
            logger.warning(("Found %d distinct pixel scales for %d frames. "
                            "Partial images can still be produced. Proceed "
                            "with caution!"), n_scales, len(pic.frames))
        else:
            raise PixelScaleError((f"Found {n_scales} distinct pixel scales "
                                   f"for {len(pic.frames)} frames."))

    if config.general.partial:
        logger.info("Partial processing selected, normalizing and dumping...")
        for frame in pic.frames:
            # TODO: multiprocess this if possible
            frame.clip(5, True)
            frame.normalize()
            auxiliaries._dump_frame(frame, output_path, "partial")
        logger.info("Dumping of partial frames complete, exiting process.")
        return pic

    with logging_redirect_tqdm(loggers=all_loggers):
        for combo in tqdm(config.combinations,
                          total=(n_combos := len(config.combinations)),
                          bar_format=tqdm_fmt):
            try:
                pic = process_combination(pic, combo, n_combos == 1,
                                          output_path,
                                          config.general, config.process)
            except BandNotFoundError as err:
                logger.warning("Skipping combination %s.", "".join(combo))
                continue
    logger.info("Image %s fully completed.", pic.name)
    return pic


def main() -> None:
    """Execute in script mode."""
    global logger
    logger = _logging_configurator()
    global all_loggers
    all_loggers = [logger, auxiliaries.logger, framework_logger]
    assert logger.level == 20
    
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
                        choices=["normal", "moreback", "lessback", "leastback"],
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
    parser.add_argument("--rgb",
                        action="append",
                        help="""A single RGB combination can be set by using
                        this option multiple times (usually 3), each with the
                        name of the band to be used (must match file name
                        parts). This option is only meant for 'bare-bone' use
                        if no full config file is used. If a valid config file
                        containing band information is found, this option will
                        be entirely ignored!""")
    parser.add_argument("--create-outfolders",
                        action="store_true",
                        help="""Whether to create a separate folder in the
                        output path for each picture, which may already exist.
                        Can only be used if -m option is set.""")
    # TODO: add verbosity count which affects logging level and time output...
    args = parser.parse_args()

    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        logging.warning("No output path specified, dumping into input folder.")
        output_path = args.input_path

    try:
        config = auxiliaries._config_parser(Configurator(),
                                            config_path=args.config_file,
                                            cmd_args=vars(args))
        bands = list(auxiliaries._bands_parser(config, args.bands_file))
        create_rgb_image(args.input_path, output_path, args.image_name, config,
                         bands)
    except Error as err:
        logger.critical("INTERNAL ERROR, ABORTING PROCESS", exc_info=err)
        _pretty_info_log("aborted", console_width=width)
    except Exception as err:
        logger.critical("UNEXPECTED ERROR, ABORTING PROCESS", exc_info=err)
        _pretty_info_log("aborted", console_width=width)


def _logging_configurator() -> logging.Logger:
    main_logger = logging.getLogger("main")
    (Path.cwd()/"log").mkdir(exist_ok=True)
    try:
        with (absolute_path/"config/logging_config.yml").open("r") as ymlfile:
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
    main()
    gc.collect()
    sys.exit(0)
