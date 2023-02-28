#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Debug module to run main functions from command line."""

import sys
import gc
import argparse
import logging
from logging.config import dictConfig
from string import Template
from pathlib import Path
from shutil import get_terminal_size
from typing import Iterator, Union

import yaml
import numpy as np
from tqdm import tqdm

from astropy.coordinates import Angle, SkyCoord
from regions import PixCoord
from regions import SkyRegion, CircleSkyRegion, \
                    RectangleSkyRegion, PolygonSkyRegion

from framework import RGBPicture, JPEGPicture, Frame, Band

width, _ = get_terminal_size((50, 20))
width = int(.6 * width)
tqdm_fmt = f"{{l_bar}}{{bar:{width}}}{{r_bar}}{{bar:-{width}b}}"

DEFAULT_CONFIG_FNAME = "./config_single.yml"
DEFAULT_BANDS_FNAME = "./bands.yml"

print(Path.cwd())
print(Path(".").resolve())
print(Path("/").resolve())


def _gma(i, g):
    return np.power(i, 1/g)


def _pretty_info_log(msg_key, width=50) -> None:
    msg_dir = {"single": "Start RGB processing for single Image...",
               "multiple": "Start RGB processing for multiple Images...",
               "done": "RGB processing done"}
    msg = msg_dir.get(msg_key, "Unknown log message.")
    logger.info(width * "*")
    logger.info("{:^{width}}".format(msg, width=width))
    logger.info(width * "*")


def create_description_file(picture: RGBPicture, filename: Path,
                            stretchalgo: str = "STIFF") -> None:
    outstr: str = ""
    colors: tuple[str, str, str] = ("Red", "Green", "Blue")

    center = picture.coords.pixel_to_world(*picture.center)
    center = center.to_string("hmsdms", precision=0)
    outstr += f"<p>Image is centered around ICRS coordinates: {center}.</p>\n"
    outstr += f"<p>Pixel scale: {picture.pixel_scale!s} per pixel.</p>\n"
    outstr += f"<p>Image size: {picture.image_scale}.</p>\n"

    outstr += "<p>Colour composite image was created using the following"
    outstr += " bands as colour channels:</p>\n"
    outstr += "<ul>\n"
    for color, channel in zip(colors, picture.rgb_channels):
        outstr += f"<li>{color:<5s}: {channel.band.verbose_str}</li>\n"
    outstr += "</ul>\n"

    outstr += f"<p>Images were stretched using {stretchalgo} algorithm.</p>\n"
    # print(outstr)
    with open(filename, "w+") as file:
        file.write(outstr)


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


def _get_mask(fname: str, frame: Frame) -> np.ndarray:
    with open(fname, "r") as ymlfile:
        mask_dict = yaml.load(ymlfile, yaml.SafeLoader)
    mask_regions = list(_maskparse(mask_dict))
    return _merge_masks(mask_regions, frame)


def create_picture(image_name: str,
                   input_path,
                   fname_template,
                   bands,
                   n_bands: int,
                   multi: bool = False) -> JPEGPicture:
    new_pic = JPEGPicture(name=image_name)
    if multi:
        new_pic.add_fits_frames_mp(input_path, bands)
    else:
        for band in tqdm(bands, total=n_bands,
                         bar_format=tqdm_fmt):
            fname = fname_template.substitute(image_name=image_name,
                                              band_name=band.name)
            new_pic.add_frame_from_file(input_path/fname, band)
    logger.info("Picture %s fully loaded.", new_pic.name)
    return new_pic


def create_rgb_image(input_path: Path,
                     output_path: Path,
                     image_name: str,
                     config,
                     bands,
                     channel_combos,
                     dump_stretch) -> RGBPicture:
    fname: Union[Path, str]
    fname_template = Template(config["general"]["filenames"])
    pic = create_picture(image_name, input_path, fname_template,
                         bands, len(config["use_bands"]),
                         config["general"]["multiprocess"])

    n_combos = len(channel_combos)
    for combo in tqdm(channel_combos, total=n_combos, bar_format=tqdm_fmt):
        cols = "".join(combo)
        logger.info("Processing image %s in %s.", pic.name, cols)
        pic.select_rgb_channels(combo, single=(n_combos == 1))

        mask = _get_mask("masking.yml", pic.primary_frame)

        grey_values = {"normal": .3, "lessback": .08, "moreback": .5}
        grey_mode = config["process"]["grey_mode"]
        pic.stretch_rgb_channels("stiff",
                                 stiff_mode="prepipy2",
                                 grey_level=grey_values[grey_mode],
                                 skymode=config["process"]["skymode"],
                                 mask=mask)

        if config["process"]["rgb_adjust"]:
            pic.adjust_rgb(config["process"]["alpha"], _gma,
                           config["process"]["gamma_lum"])
            logger.info("RGB sat. adjusting after contrast and stretch.")

        if pic.is_bright:
            logger.info(("Image is bright, performing additional color space "
                         "stretching to equalize colors."))
            pic.equalize("mean",
                         offset=config["process"].get("equal_offset", .1),
                         norm=config["process"].get("equal_norm", True),
                         mask=mask)
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

        if dump_stretch:
            logger.info("Dumping stretched FITS files for each channel.")
            for channel in pic.rgb_channels:
                dump_name = channel.band.name
                fname = output_path/f"{dump_name}_stretched.fits"
                if not fname.exists():
                    channel.save_fits(fname)
                    logger.info("Done dumping %s image.", dump_name)
                else:
                    logger.warning(("Stretched FITS file for %s exists, not"
                                    " overwriting."), dump_name)
            logger.info("Done dumping all channels for this combination.")
        logger.info("Image %s in %s done.", pic.name, cols)
    logger.info("Image %s fully completed.", pic.name)
    return pic


def setup_rgb_single(input_path, output_path, image_name,
                     config_name=None, bands_name=None,
                     dump_stretch=False) -> RGBPicture:
    _pretty_info_log("single")

    config_name = config_name or DEFAULT_CONFIG_FNAME
    with open(config_name, "r") as ymlfile:
        config = yaml.load(ymlfile, yaml.SafeLoader)

    bands_name = bands_name or DEFAULT_BANDS_FNAME
    bands = Band.from_yaml_file(bands_name, config["use_bands"])
    channel_combos = config["combinations"]

    pic = create_rgb_image(input_path, output_path, image_name, config, bands,
                           channel_combos, dump_stretch)

    _pretty_info_log("done")
    return pic


def setup_rgb_multiple(input_path, output_path, image_names,
                       config_name=None, bands_name=None,
                       create_outfolder=False,
                       dump_stretch=False) -> Iterator[RGBPicture]:
    _pretty_info_log("multiple")

    config_name = config_name or DEFAULT_CONFIG_FNAME
    with open(config_name, "r") as ymlfile:
        config = yaml.load(ymlfile, yaml.SafeLoader)

    bands_name = bands_name or DEFAULT_BANDS_FNAME
    bands = Band.from_yaml_file(bands_name, config["use_bands"])
    channel_combos = config["combinations"]

    for image_name in image_names:
        if create_outfolder:
            imgpath = output_path/image_name
            imgpath.mkdir(parents=True, exist_ok=True)
        else:
            imgpath = output_path
        pic = create_rgb_image(input_path, imgpath, image_name, config, bands,
                               channel_combos, dump_stretch)
        yield pic
    _pretty_info_log("RGB processing done")


def main() -> None:
    """Execute in script mode."""
    parser = argparse.ArgumentParser(prog="rgbcombo",
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
    parser.add_argument("--dump-stretched-fits",
                        dest="dump_stretch",
                        action="store_true",
                        help="""Dump stretched single-band FITS files into the
                        specified output directory.""")
    args = parser.parse_args()

    if args.output_path is not None:
        output_path = Path(args.output_path)
    else:
        logging.warning("No output path specified, dumping into input folder.")
        output_path = args.input_path

    if args.many:
        setup_rgb_multiple(args.input_path, output_path, args.image_name,
                           args.config_file, args.bands_file,
                           args.create_outfolders, args.dump_stretch)
    else:
        setup_rgb_single(args.input_path, output_path, args.image_name,
                         args.config_file, args.bands_file, args.dump_stretch)


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
    imgpath = root/"JPEGS/new2"
    target = "outreach_1"

    # https://note.nkmk.me/en/python-pillow-concat-images/

    mypic = setup_rgb_single(path, imgpath, target, dump_stretch=False)

    # root = Path("D:/Nemesis/data/perseus")
    # path = root/"stamps/"

    # for i in range(1, 4):
    #     target = f"IC 348-{i}"
    #     setup_rgb(path, root/"RGBs", target)

    # main()
    gc.collect()
    sys.exit(0)
