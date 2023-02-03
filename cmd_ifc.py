# -*- coding: utf-8 -*-
"""Debug module to run main functions from command line."""

import sys
import argparse
import logging
from logging.config import dictConfig
from pathlib import Path

import yaml

from combo_large_pil import setup_rgb_single, setup_rgb_multiple


def main():
    """Execute in script mode."""
    parser = argparse.ArgumentParser(prog="RGBCOMBO",
                                     description="""Combines RGB channels to
                                     color image including stretching.""")

    # https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument

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
                        help="""The name of the main config file to be used.
                        If omitted, the code will look for a file named
                        "config.yml" in the main package folder.""")
    parser.add_argument("-b", "--bands-file",
                        help="""The name of the band config file to be used.
                        If omitted, the code will look for a file named
                        "bands.yml" in the main package folder.""")
    parser.add_argument("-m",
                        dest="many",
                        action="store_true",
                        help="""Whether to process multiple images. If set,
                        image-name is interpreted as a list of names.""")
    parser.add_argument("--create-outfolders",
                        action="store_true",
                        help="""Whether to create a separate folder in the
                        output path for each picture, which may already exist.
                        Can only be used if -m option is set.""")
    args = parser.parse_args()

    if args.output_path is not None:
        output_path = Path(args.output_path)
    else:
        logging.warning("No output path specified, dumping into input folder.")
        output_path = args.input_path

    if args.many:
        setup_rgb_multiple(args.input_path, output_path, args.image_name,
                           args.config_file, args.bands_file,
                           args.create_outfolders)
    else:
        setup_rgb_single(args.input_path, output_path, args.image_name,
                         args.config_file, args.bands_file)


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
    _logging_configurator()
    main()
    sys.exit(0)
