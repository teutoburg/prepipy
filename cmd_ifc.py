import sys
import argparse
import logging
from logging.config import dictConfig
import yaml
from pathlib import Path

from combo_large_pil import setup_rgb_single


def main():
    parser = argparse.ArgumentParser(
        description="Comines RGB channels to color image including stretching."
        )

    parser.add_argument('-i', '--input-path', nargs=1,
                        required=True,
                        type=str,
                        help='''The file path to the input fits files
                        containing the images that shall be processed.''')
    parser.add_argument('-n', '--image_name', nargs=1,
                        required=True,
                        type=str,
                        help='''Name stem of the images. Image names must have
                        the following structure: <name>_<band>.fits, e.g.
                        V883_Ori_J.fits.''')
    parser.add_argument('-o', '--output-path', nargs=1,
                        type=str,
                        help='''The file path where the combined images will
                        be saved to. If omitted, images are dumped back into
                        the input folder.''')
    parser.add_argument('-c', '--config-file', nargs=1,
                        type=str,
                        help='''The name of the main config file to be used.
                        If omitted, the code will look for a file named
                        "config.yml" in the main package folder.''')
    parser.add_argument('-b', '--bands-file', nargs=1,
                        type=str,
                        help='''The name of the band config file to be used.
                        If omitted, the code will look for a file named
                        "bands.yml" in the main package folder.''')
    args = parser.parse_args()
    input_path = Path(args.input_path[0])
    if args.output_path is not None:
        output_path = Path(args.output_path[0])
    else:
        logging.warning("No output path specified, dumping into input folder.")
        output_path = input_path
    setup_rgb_single(input_path, output_path, args.image_name[0],
                     args.config_file[0], args.bands_file[0])


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
