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
width = int(.8 * width)
bar_width = max(width - 40, 10)
tqdm_fmt = f"{{l_bar}}{{bar:{bar_width}}}{{r_bar}}{{bar:-{bar_width}b}}"

absolute_path = Path(__file__).resolve(strict=True).parent
DEFAULT_CONFIG_NAME = "config_single.yml"
DEFAULT_BANDS_NAME = "bands.yml"


def _gma(i, g):
    return np.power(i, 1/g)


def _pretty_info_log(msg_key, console_width=50) -> None:
    msg_dir = {"single": "Start RGB processing for single Image...",
               "multiple": "Start RGB processing for multiple Images...",
               "done": "RGB processing done"}
    msg = msg_dir.get(msg_key, "Unknown log message.")
    logger.info(console_width * "*")
    logger.info("{:^{width}}".format(msg, width=console_width))
    logger.info(console_width * "*")


def create_description_file(picture: RGBPicture, filename: Path,
                            stretchalgo: str = "STIFF") -> None:
    outstr: str = ""
    colors: tuple[str, str, str] = ("Red", "Green", "Blue")
    ul_margin: str = "-20px"

    outstr += "<h5>Click on image to view fullsize.</h5>\n"

    center = picture.coords.pixel_to_world(*picture.center)
    center = center.to_string("hmsdms", precision=0)
    outstr += ("<p>Image is centered around ICRS coordinates: "
               f"<b>{center}</b>.<br>\n Pixel scale: {picture.pixel_scale!s} "
               f"per pixel.<br>\n Image size: {picture.image_scale}.</p>\n")

    outstr += ("<p>Colour composite image was created using the following "
               "bands as colour channels:</p>\n")
    outstr += f"<ul style=\"margin-top:{ul_margin};\">\n"
    for color, channel in zip(colors, picture.rgb_channels):
        outstr += (f"<li><span style=\"color:{color};font-weight:bold;\">"
                   f"{color}:</span> {channel.band.verbose_str}</li>\n")
    outstr += "</ul>\n"

    outstr += f"<p>Images were stretched using {stretchalgo} algorithm.</p>\n"
    outstr += "<hr>\n"
    outstr += ("<p style=\"font-size:small; font-style:italic\">The fullsize "
               "version of this image (click on image to view) contains "
               "machine-readable coordinate (WCS) information. The image can "
               "be downloaded and viewed in applications such as the "
               "<a href=\"https://aladin.u-strasbg.fr/AladinDesktop/\" "
               "style=\"text-decoration:underline; color:#00133f;\" "
               "target=\"_blank\">Aladin sky atlas</a> via \"drag-and-drop\" "
               "or by pasting the image's permalink into the command-line."
               "<p>\n")
    # print(outstr)
    with filename.open("w+") as file:
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
                   input_path: Path,
                   fname_template: Template,
                   bands: Iterator[Band],  # or Iterable??
                   n_bands: int,
                   multi: bool = False) -> JPEGPicture:
    new_pic = JPEGPicture(name=image_name)
    if multi:
        new_pic.add_fits_frames_mp(input_path, fname_template, bands)
    else:
        for band in tqdm(bands, total=n_bands,
                         bar_format=tqdm_fmt):
            fname = fname_template.substitute(image_name=image_name,
                                              band_name=band.name)
            new_pic.add_frame_from_file(input_path/fname, band)
    logger.info("Picture %s fully loaded.", new_pic.name)
    return new_pic


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


def create_rgb_image(input_path: Path,
                     output_path: Path,
                     image_name: str,
                     config: dict,
                     bands: Iterator[Band],
                     channel_combos: list,
                     dump_stretch: bool,
                     description: bool,
                     partial: bool,
                     multi: bool) -> RGBPicture:
    fname: Union[Path, str]
    fname_template = Template(config["general"]["filenames"])
    pic = create_picture(image_name, input_path, fname_template,
                         bands, len(config["use_bands"]),
                         multi)

    if partial:
        logger.info("Partial processing selected, normalizing and dumping...")
        for frame in pic.frames:
            # TODO: multiprocess this if possible
            frame.clip(5, True)
            frame.normalize()
            _dump_frame(frame, output_path, "partial")
        logger.info("Dumping of partial frames complete, aborting process.")
        return

    for combo in tqdm(channel_combos, total=(n_combos := len(channel_combos)),
                      bar_format=tqdm_fmt):
        cols: str = "".join(combo)
        fname: str = f"{pic.name}_img_{cols}"

        logger.info("Processing image %s in %s.", pic.name, cols)
        pic.select_rgb_channels(combo, single=(n_combos == 1))

        try:
            mask = _get_mask("masking.yml", pic.primary_frame)
        except FileNotFoundError:
            logger.warning("No masking file found in cwd, using no mask.")
            mask = None

        grey_values = {"normal": .3, "lessback": .08, "moreback": .5}
        grey_mode = config["process"]["grey_mode"]

        if grey_mode != "normal":
            logger.info("Using grey mode \"%s\".", grey_mode)
            fname += f"_{grey_mode}"
        else:
            logger.info("Using normal grey mode.")

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

        if dump_stretch:
            _dump_rgb_channels(pic, output_path)

        savename = (output_path/fname).with_suffix(".jpeg")
        pic.save_pil(savename)

        if description:
            create_description_file(pic, savename.with_suffix(".html"))

        logger.info("Image %s in %s done.", pic.name, cols)
    logger.info("Image %s fully completed.", pic.name)
    return pic


def setup_rgb_single(input_path, output_path, image_name,
                     config_name=None, bands_name=None,
                     dump_stretch=False, description=False,
                     partial=False, multi=False) -> RGBPicture:
    _pretty_info_log("single", width)
    cwd = Path.cwd()

    fallback_config_path = cwd/DEFAULT_CONFIG_NAME
    if not fallback_config_path.exists():
        fallback_config_path = absolute_path/DEFAULT_CONFIG_NAME
    config_name = config_name or fallback_config_path
    with config_name.open("r") as ymlfile:
        config = yaml.load(ymlfile, yaml.SafeLoader)

    fallback_bands_path = Path.cwd()/DEFAULT_BANDS_NAME
    if not fallback_bands_path.exists():
        fallback_bands_path = absolute_path/DEFAULT_BANDS_NAME
    bands_name = bands_name or fallback_bands_path
    bands = Band.from_yaml_file(bands_name, config["use_bands"])
    channel_combos = config["combinations"]

    pic = create_rgb_image(input_path, output_path, image_name, config, bands,
                           channel_combos, dump_stretch, description, partial,
                           multi)
    _pretty_info_log("done", width)
    return pic


# def setup_rgb_multiple(input_path, output_path, image_names,
#                        config_name=None, bands_name=None,
#                        create_outfolder=False,
#                        dump_stretch=False) -> Iterator[RGBPicture]:
#     _pretty_info_log("multiple", width)

#     config_name = config_name or DEFAULT_CONFIG_FNAME
#     with open(config_name, "r") as ymlfile:
#         config = yaml.load(ymlfile, yaml.SafeLoader)

#     bands_name = bands_name or DEFAULT_BANDS_FNAME
#     bands = Band.from_yaml_file(bands_name, config["use_bands"])
#     channel_combos = config["combinations"]

#     for image_name in image_names:
#         if create_outfolder:
#             imgpath = output_path/image_name
#             imgpath.mkdir(parents=True, exist_ok=True)
#         else:
#             imgpath = output_path
#         pic = create_rgb_image(input_path, imgpath, image_name, config, bands,
#                                channel_combos, dump_stretch)
#         yield pic
#     _pretty_info_log("RGB processing done", width)


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
    parser.add_argument("-m", "--multi",
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
    parser.add_argument("--create-outfolders",
                        action="store_true",
                        help="""Whether to create a separate folder in the
                        output path for each picture, which may already exist.
                        Can only be used if -m option is set.""")
    args = parser.parse_args()

    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        logging.warning("No output path specified, dumping into input folder.")
        output_path = args.input_path

    picture = setup_rgb_single(args.input_path, output_path, args.image_name,
                               args.config_file, args.bands_file,
                               args.fits_dump, args.description, args.partial,
                               args.multi)


def _logging_configurator():
    main_logger = logging.getLogger("main")
    try:
        with (absolute_path/"log/logging_config.yml").open("r") as ymlfile:
            dictConfig(yaml.load(ymlfile, yaml.SafeLoader))
    except FileNotFoundError as err:
        logging.error(err)
        logging.basicConfig()
        main_logger.setLevel(logging.INFO)
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
    imgpath = root/"JPEGS/new2"
    target = "outreach_1"

    # https://note.nkmk.me/en/python-pillow-concat-images/

    # mypic = setup_rgb_single(path, imgpath, target, dump_stretch=False)

    # root = Path("D:/Nemesis/data/perseus")
    # path = root/"stamps/"

    # for i in range(1, 4):
    #     target = f"IC 348-{i}"
    #     setup_rgb(path, root/"RGBs", target)

    main()
    gc.collect()
    sys.exit(0)
