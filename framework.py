# -*- coding: utf-8 -*-
"""
Framework for images etc.
"""

# from pathlib import Path
import logging
# from logging.config import dictConfig
from operator import itemgetter
import copy
import struct
from dataclasses import dataclass
from multiprocessing import Pool

# import yaml
# from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy import wcs
# from astropy.stats import median_absolute_deviation as mad
from astropy.stats import sigma_clipped_stats as scs

from PIL import Image

TQDM_FMT = "{l_bar}{bar:50}{r_bar}{bar:-50b}"
logger = logging.getLogger(__name__)


class Error(Exception):
    """Base class for exeptions in this module."""

    pass


class BandExistsError(Error):
    """A Frame in this band already exists in this picture."""

    pass


class FileTypeError(Error):
    """The given file has the wrong or an unknown type."""

    pass


@dataclass
class Band():
    name: str
    wavelength: float = None
    instrument: str = "unknown"
    telescope: str = "unknown"

    @classmethod
    def from_yaml_dict(cls, bands, use_bands=None):
        # parse shortened names to correct parameter names
        for band in bands.values():
            band["instrument"] = band.pop("inst")
            band["telescope"] = band.pop("tele")
            band["wavelength"] = band.pop("wave")

        # either use specified or all, anyway turn into tuple without names
        if use_bands is not None:
            bands = itemgetter(*use_bands)(bands)
        else:
            bands = tuple(band.values())

        # in case we ever need the names as well...
        # for name, band in bands.items():
        #     yield name, cls(**band)
        for band in bands:
            yield cls(**band)


class Frame():
    """n/a."""

    def __init__(self, image, band, header=None, **kwargs):
        self.image = image
        self._band = band
        self.header = header
        self.background = np.nanmedian(self.image)  # estimated background
        self.clip_and_nan(**kwargs)

        self.coords = wcs.WCS(self.header)

        # self.stars = list()
        # self.coord = None

    def __repr__(self):
        return str(self)  # DEBUG only

    def __str__(self):
        return f"{self.shape} frame in \"{self.band.name}\" band"

    @classmethod
    def from_fits(cls, filename, band):
        with fits.open(filename) as file:
            return cls(file[0].data, band, file[0].header)

    @property
    def band(self):
        """The pass-band in which the frame was taken. Read-only property."""
        return self._band

    @property
    def shape(self):
        return f"{self.image.shape[0]} x {self.image.shape[1]}"

    def camera_aperture(self, center, radius):
        """
        Set everything outside a defined radius around a defined center to
        zero, to remove vignetting effects.

        Parameters
        ----------
        center : tuple
            Center of the camera aperture. Not necessarily the actual center
            of the image.
        radius : float
            Radius of the camera aperture in pixel.

        Returns
        -------
        None.

        """
        y_axis, x_axis = np.indices(self.image.shape)
        dst = np.sqrt((x_axis-center[0])**2 + (y_axis-center[1])**2)
        out = np.where(dst > radius)
        self.image[out] = 0.

    def clip(self, n_sigma=3.):
        """
        Perform n sigma clipping on the image (only affects max values).

        This method will change self.image data. Background level is taken from
        the original estimation upon instantiation, sigma is evaluated each
        time, meaning this method can be used iteratively.

        Parameters
        ----------
        n_sigma : float, optional
            Number of sigmas to be used for clipping. The default is 3.0.

        Returns
        -------
        None.

        """
        upper_limit = self.background + n_sigma * np.nanstd(self.image)
        self.image = np.clip(self.image, None, upper_limit)

    def clip_and_nan(self, clip=10, nanmode="max"):
        r"""
        Perform upper sigma clipping and replace NANs.

        Parameters
        ----------
        clip : int, optional
            Number of sigmas to be used for clipping. If 0, no clipping is
            performed. The default is 10.
        nanmode : str, optional
            Which value to use for replacing NANs. Allowed values are
            \"median\" or \"max\" (clipped if clipping is performed).
            The default is "max".

        Raises
        ------
        ValueError
            Raised if clip is negative or `nanmode` is invalid.

        Returns
        -------
        None.

        """
        med = np.nanmean(self.image)
        if clip:
            if clip < 0:
                raise ValueError("clip must be positive integer or 0.")
            logger.debug("Clipping to %s sigma.", clip)
            v_max = med + clip * np.nanstd(self.image)
            self.image = np.clip(self.image, None, v_max)
        else:
            v_max = np.nanmax(self.image)

        if nanmode == "max":
            logger.debug("Replacing NANs with clipped max value.")
            self.image = np.nan_to_num(self.image, False, nan=v_max)
        elif nanmode == "median":
            logger.debug("Replacing NANs with median value.")
            self.image = np.nan_to_num(self.image, False, nan=med)
        else:
            raise ValueError("nanmode not understood")

    def phot_zero(self):
        """Calculate global photometrical zeropoint."""
        for star in self.stars:
            star.mk_phot_zeropoint(EXPTIME, EXTINCT, AIRMASS)

        zeropoints = [star.phzp.get(self._band, None) for star in self.stars]
        zeropoint = np.mean([zero.mag for zero in zeropoints if zero is not None])
        err_zero = np.max([zero.err_mag for zero in zeropoints if zero is not None])
        # self.phot_zero = CatFlux(self._band, zeropoint, err_zero)
        return CatFlux(self._band, zeropoint, err_zero)

    def add_cal_mag(self):
        """legacy, check!"""
        for star in self.stars:
            star.cal_mag(self.phot_zero(), EXPTIME, EXTINCT, AIRMASS)

    def _match_wcs_ref(self):
        for star in self.stars:
            if star.name != "n/a":
                yield (star.wcs_ra, star.center["x"][0],
                       star.wcs_de, star.center["y"][0])

    def mk_wcs(self):
        """create coordinate system"""
        ra_cat, ra_px, de_cat, de_px = zip(*self._match_wcs_ref())

        plt.scatter(ra_cat, ra_px)
        plt.scatter(de_cat, de_px)

        delt_ra, ref_ra, _, _, err_ra = sta.linregress(ra_px, ra_cat)
        delt_de, ref_de, _, _, err_de = sta.linregress(de_px, de_cat)

        print(f"RA {ref_ra:0.6f} +/- {err_ra:0.6f}")
        print(f"DE {ref_de:0.6f} +/- {err_de:0.6f}")

        self.coord = wcs.WCS(naxis=2)
        self.coord.wcs.crpix = [ra_px[2], de_px[2]]
        self.coord.wcs.cdelt = [delt_ra, delt_de]
        self.coord.wcs.crval = [ra_cat[2], de_cat[2]]
        self.coord.wcs.ctype = ["RA---AIR", "DEC--AIR"]

        return self.coord

    def display_all(self):
        """Display image and all known stars.
        Uses coordinate system if available, otherwise axes are in pixel."""
        if self.coord is not None:
            plt.subplot(projection=self.coord)
        plt.imshow(self.image, vmin=100., vmax=1000., origin="lower")
        plt.grid(color='white', ls='dashed')
        plt.xlabel("RA")
        plt.ylabel("DE")
        plt.tight_layout()
        for star in self.stars:
            star.display()

    def display_3d(self):
        xx, yy = np.mgrid[0:self.image.shape[0], 0:self.image.shape[1]]
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(xx, yy, self.image, rstride=1, cstride=1,
                        cmap="viridis", linewidth=0)
        ax.set_title(self.band.name)
        plt.show()

    def normalize(self, new_range=1., new_offset=0.):
        """Subtract minimum and devide by maximum."""
        data_max = np.nanmax(self.image)
        data_min = np.nanmin(self.image)
        data_range = data_max - data_min
        if round(data_min, 5) == 0. and round(data_max, 5) == 1.:
            return data_range, data_max

        self.image -= data_min
        self.image /= data_range

        try:
            assert np.nanmin(self.image) == 0.0
        except AssertionError:
            print(f"ERROR: {data_max, data_min}")
        assert np.nanmax(self.image) == 1.0

        self.image = self.image * new_range + new_offset

        self.data_range = data_range
        self.data_min = data_min
        self.data_max = data_max

        return data_range, data_max

    def clipped_stats(self):
        print("CLIPPING")
        data = self.image.flatten()
        mean, median, stddev = np.mean(data), np.median(data), np.std(data)
        logger.debug("unclipped:\tmean=%.4f\tmedian=%.4f\tstddev=%.4f",
                     mean, median, stddev)

        mean, median, stddev = scs(data,
                                   cenfunc='median', stdfunc='mad_std',
                                   sigma_upper=5, sigma_lower=3)
        logger.debug("clipped:\tmean=%.4f\tmedian=%.4f\tstddev=%.4f",
                     mean, median, stddev)
        self.clipped_median = median
        self.clipped_stddev = stddev
        return mean, median, stddev

    def min_inten(self, gamma_lum, grey_level=.3,
                  sky_mode="median", max_mode="quantile", **kwargs):
        if sky_mode == "quantile":
            i_sky = np.quantile(self.image, .8)
        elif sky_mode == "median":
            i_sky = np.nanmedian(self.image)
        elif sky_mode == "clipmedian":
            self.clipped_stats()
            i_sky = self.clipped_median
        elif sky_mode == "debug":
            i_sky = .01
        else:
            raise ValueError("sky_mode not understood")

        if max_mode == "quantile":
            i_max = np.quantile(self.image, .995)
        elif max_mode == "max":  # FIXME: if we normalize before, this will always be == 1.0
            i_max = np.nanmax(self.image)
        elif max_mode == "debug":
            i_max = .99998
        else:
            raise ValueError("max_mode not understood")

        p_grey_g = grey_level**gamma_lum
        logger.debug("p_grey_g=%.4f", p_grey_g)
        logger.debug("i_sky=%.4f", i_sky)
        i_min = max((i_sky - p_grey_g * i_max) / (1. - p_grey_g), 0.)
        logger.debug("i_min=%.4f\t\ti_max=%.4f", i_min, i_max)
        return i_min, i_max

    def stiff_d(self, stretch_function, gamma_lum=1.5, grey_level=.1, **kwargs):
        logger.info("stretching %s band", self.band.name)
        data_range, data_max = self.normalize()

        # gamma_lum = self.auto_gma()

        i_min, i_max = self.min_inten(gamma_lum, grey_level, **kwargs)
        self.sky_mask = self.image < i_min
        self.image = self.image.clip(i_min, i_max)

        new_img = stretch_function(self.image, **kwargs)
        # new_img = self.stiff_stretch(self.image, **kwargs)
        new_img *= data_range

        self.image = new_img
        logger.info("%s band done", self.band.name)

    @staticmethod
    def stiff_stretch(image, stiff_mode="power-law", **kwargs):
        def_kwargs = {"power-law": {"gamma": 2.2, "a": 1., "b": 0., "i_t": 0.},
                      "srgb": {"gamma": 2.4, "a": 12.92, "b": .055, "i_t": .00304},
                      "rec709": {"gamma": 2.22, "a": 4.5, "b": .099, "i_t": .018},
                      "user": {"gamma": 2.25, "a": 3., "b": .08, "i_t": .003},
                      "user2": {"gamma": 2.25, "a": 3., "b": .1, "i_t": .8},
                      "user3": {"gamma": 2.25, "a": 3., "b": .05, "i_t": .001},
                      "user4": {"gamma": 2.25, "a": 3., "b": .05, "i_t": .003}}
        if not stiff_mode in def_kwargs:
            raise KeyError(f"Mode must be either of {list(def_kwargs.keys())}.")

        kwargs = def_kwargs[stiff_mode] | kwargs

        # kwargs["gamma"] = self.auto_gma()
        assert kwargs['gamma'] == 2.25  # HACK: DEBUG ONLY

        b, i_t = kwargs["b"], kwargs["i_t"]
        image_s = kwargs["a"] * image * (image < i_t)
        image_s += (1+b) * image**(1/kwargs["gamma"]) - b * (image >= i_t)
        return image_s

    @staticmethod
    def autostretch_light(image, **kwargs):
        # logger.info("Begin autostretch for \"%s\" band", self.band.name)
        # maximum = self.normalize()
        # logger.info("maximum:\t%s", maximum)

        mu = np.nanmean(image)
        sigma = np.nanstd(image)
        logger.info("$\mu$:\t%s", mu)
        logger.info("$\sigma$:\t%s", sigma)

        gamma = np.exp((1 - (mu + sigma)) / 2)
        logger.info("$\gamma$:\t%s", gamma)

        k = np.power(image, gamma) + (1 - np.power(image, gamma)) * (mu**gamma)
        image_s = np.power(image, gamma) / k

        # image_s *= maximum
        return image_s

    def auto_gma(self):
        return np.exp((1 - (self.clipped_median + self.clipped_stddev)) / 2)


class Picture():
    """n/a."""

    def __init__(self, name=None):
        self.frames = list()
        self.stars = list()
        self.name = name

    @property
    def num_stars(self):
        """Number of fitted, identified and cross-matched stars.
        Read-only property."""
        return len(self.stars)

    @property
    def bands(self):
        """List of all bands in the frames. Read-only property."""
        return [frame.band for frame in self.frames]

    @property
    def image(self):
        """Combined image of all frames. Read-only property."""
        if not self.frames:
            raise ValueError("No frame loaded.")
        # HACK: actually combine all images!
        return self.frames[0].image

    @property
    def coords(self):
        """WCS coordinates of the first frame."""
        return self.frames[0].coords

    @property
    def image_size(self):
        """Number of pixels per frame. Read-only property."""
        return self.frames[0].image.size

    @property
    def cube(self):
        """Stack images from all frames into one 3D cube."""
        return np.stack([frame.image for frame in self.frames])

    def get_rgb_cube(self, mode="0-1", order="cxy"):
        """Stack images from RGB channels into one 3D cube, normalized to 1.

        mode can be `0-1` or `0-255`.
        order can be `cxy` or `xyc`.
        """
        rgb = np.stack([frame.image for frame in self.rgb_channels])
        # rgb[rgb<0.] = 0.
        rgb /= rgb.max()

        # invert alpha channel if RGB(A)
        if len(rgb) == 4:
            rgb[3] = 1 - rgb[3]
        if order == "xyc":
            rgb = np.moveaxis(rgb, 0, -1)
        if mode == "0-1":
            return rgb
        if mode == "0-255":
            rgb *= 255.
            rgb = rgb.astype(np.uint8)
            return rgb

    @property
    def is_bright(self):
        """Return True is any median of the RGB frames is >.2."""
        return any(np.median(c.image) > .2 for c in self.rgb_channels)

    def _check_band(self, band):
        if isinstance(band, str):
            band = Band(band)
        elif not isinstance(band, Band):
            raise TypeError("Invalid type for band.")

        if band in self.bands:
            err_msg = f"Picture already includes a Frame in \"{band.name}\" band."
            raise BandExistsError(err_msg)

        return band

    def add_frame(self, image, band, header=None):
        band = self._check_band(band)
        new_frame = Frame(image, band, header=None)
        self.frames.append(new_frame)
        return new_frame

    def add_frame_from_file(self, filename, band, framelist=None):
        """
        Add a new frame to the picture. File must be in FITS format.

        Parameters
        ----------
        filename : Path object
            Path of the file containing the image.
        band : Band or str
            Pass-band (filter) in which the image was taken.
        framelist: list or None
            List to append the frames to. If None (default), the internal list
            is used. Used only for multiprocessing, do not change manually.

        Returns
        -------
        new_frame : Frame
            The newly created frame.

        """
        band = self._check_band(band)
        logger.info("Loading frame for %s band", band.name)

        if filename.suffix == ".fits":
            new_frame = Frame.from_fits(filename, band)
        else:
            raise FileTypeError("Currently only FITS files are supported.")

        if framelist is None:
            self.frames.append(new_frame)
        else:
            framelist.append(new_frame)
        return new_frame

    def add_fits_frames_mp(self, input_path, bands):
        args = [(input_path/f"{self.name}_{band.name}.fits", band)
                for band in bands]
        with Pool(len(args)) as p:
            framelist = p.starmap(Frame.from_fits, args)
        self.frames = framelist

    @classmethod
    def from_cube(cls, cube, bands=None):
        if not cube.ndim == 3:
            raise TypeError("A \"cube\" must have exactly 3 (three) dimensions!")

        if bands is None:
            bands = len(cube) * [Band("unknown")]
            bands = [Band(f"unkown{i}") for i, _ in enumerate(cube)]
        elif all(isinstance(band, str) for band in bands):
            bands = [Band(band) for band in bands]
        elif not all(isinstance(band, Band) for band in bands):
            raise TypeError("Invalid data type in bands.")

        if not len(bands) == len(cube):
            # HACK: change this to zip(..., strict=True) below in Python 3.10+
            raise IndexError("Length of bands must equal number of images in cube.")

        new_picture = cls()

        for image, band in zip(cube, bands):
            new_picture.add_frame(image, band)

        return new_picture

    @classmethod
    def from_tesseract(cls, tesseract, bands=None):
        for cube in tesseract:
            yield cls.from_cube(cube, bands)

    @staticmethod
    def merge_tesseracts(tesseracts):
        return np.hstack(list(tesseracts))

    @staticmethod
    def combine_into_tesseract(pictures):
        return np.stack([picture.cube for picture in pictures])

    def stretch_frames(self, mode="auto-light", only_rgb=False, **kwargs):
        if only_rgb:
            frames = self.rgb_channels
        else:
            frames = self.frames
        for frame in frames:
            if mode == "auto-light":
                frame.autostretch_light(**kwargs)
            elif mode == "stiff-d":
                frame.stiff_d(**kwargs)
            else:
                raise ValueError("stretch mode not understood")

    def select_rgb_channels(self, bands, single=False):
        """
        Select existing frames to be used as channels for multi-colour image.
        Usually 3 channels are interpreted as RGB. Names of the frame bands
        given in `bands` must match the name of the respective frame's band.
        The order of bands is interprted as R, G and B in the case of 3 bands.

        Parameters
        ----------
        bands : list of str
            List of names for the bands to be used as colour channels.
        single : bool, optional
            If only a single RGB combination is used on the instance, set this
            option to ``True`` to save memory. This will result in alteration
            of the original frame images. The default is False.

        Raises
        ------
        ValueError
            Raised if `bands` contains duplicates or if `bands` contains more
            than 4 elements.
        UserWarning
            Raised if channels are not ordered by descending wavelength, if
            wavelength informatin is available for all channels.

        Returns
        -------
        rgb_channels : list of Frame objects
            Equivalent to the instance property.

        """
        if not len(bands) == len(set(bands)):
            raise ValueError("bands contains duplicates")
        if len(bands) > 4:
            raise ValueError("RGB accepts up to 4 channels.")

        frames_dict = dict(((f.band.name, f) for f in self.frames))
        copyfct = copy.copy if single else copy.deepcopy
        self.rgb_channels = list(map(copyfct, (itemgetter(*bands)(frames_dict))))

        if all(channel.band.wavelength is not None for channel in self.rgb_channels):
            if not all(redder.band.wavelength >= bluer.band.wavelength
                       for redder, bluer
                       in zip(self.rgb_channels, self.rgb_channels[1:])):
                raise UserWarning("Not all RGB channels are ordered by descending wavelength.")

        _chnames = [channel.band.name for channel in self.rgb_channels]
        logger.info("Successfully selected %i RGB channels: %s",
                    len(_chnames), ", ".join(map(str, _chnames)))
        return self.rgb_channels

    def weightssss(self, weights):
        if weights is not None:
            if isinstance(weights, str):
                if weights == "auto":
                    weights = [1/frame.clipped_median for frame in self.rgb_channels]
                else:
                    raise ValueError("weights mode not understood")

            for channel, weight in zip(self.rgb_channels, weights):
                channel.image *= weight

    def autoparam(self):
        gamma = 2.25
        gamma_lum = 1.5
        alpha = 1.4
        grey_level = .3

        self.ap = {"gma": False, "alph": False}

        if not any((np.array([channel.clipped_median for channel in self.rgb_channels]) / np.mean([channel.clipped_median for channel in self.rgb_channels])) > 2.):
            gamma_lum = 1.2
            self.ap["gma"] = True

        if all(channel.clipped_median > 200. for channel in self.rgb_channels) and all(channel.clipped_stddev > 50. for channel in self.rgb_channels):
            alpha = 1.2
            self.ap["alph"] = True
            print(self.name)

        return gamma, gamma_lum, alpha, grey_level

    def luminance(self):
        sum_image = sum(frame.image for frame in self.rgb_channels)
        sum_image /= len(self.rgb_channels)
        return sum_image

    def stretch_luminosity(self, stretch_fkt_lum, gamma_lum, lum, **kwargs):
        lum_stretched = stretch_fkt_lum(lum, gamma_lum, **kwargs)
        for channel in self.rgb_channels:
            channel.image /= lum
            channel.image *= lum_stretched

    def adjust_rgb(self, alpha, stretch_fkt_lum, gamma_lum, **kwargs):
        """
        Adjust colour saturation of 3-channel (R, G, B) image.

        This method will modify the image data in the frames defined to be used
        as RGB channels.

        Parameters
        ----------
        alpha : float
            Colour saturation parameter, typically 0-2.

        Returns
        -------
        None.

        Notes
        -------
        This method should also work for 2-channel images, but this is not tested.
        This method should also work for 4-channel images, but this is not tested.

        """
        # BUG: sometimes lost of zeros, maybe normalize before this?????
        logger.info("RGB adjusting using alpha=%s, gamma_lum=%s.",
                    alpha, gamma_lum)
        lum = self.luminance()
        n_channels = len(self.rgb_channels)
        assert n_channels <= 4
        alpha /= n_channels

        channels = [channel.image for channel in self.rgb_channels]
        channels_adj = []
        for i, channel in enumerate(channels):
            new = lum + alpha * (n_channels-1) * channel
            for j in range(1, n_channels):
                new -= alpha * channels[(i+j) % n_channels]
            zero_mask = new < 0.
            logger.debug("zero fraction: %.2f percent",
                         zero_mask.sum()/new.size*100)
            new[zero_mask] = 0.
            channels_adj.append(new)

        for channel, adjusted in zip(self.rgb_channels, channels_adj):
            channel.image = adjusted

        # gamma_lum = kwargs.get("gamma_lum", gamma)
        self.stretch_luminosity(stretch_fkt_lum, gamma_lum, lum, **kwargs)

    def equalize(self, mode="mean", offset=.5, supereq=False, norm=False):
        means = []
        for channel in self.rgb_channels:
            channel.image /= np.nanmax(channel.image)
            if mode == "median":
                channel.image -= np.nanmedian(channel.image)
            elif mode == "mean":
                channel.image -= np.nanmean(channel.image)
            means.append(np.nanmean(channel.image))
            channel.image += offset
            channel.image[channel.image < 0.] = 0.
            if norm:
                channel.normalize()
        if supereq:
            maxmean = max(means)
            for channel, m in zip(self.rgb_channels, means):
                eq = min(maxmean/m, 10.)
                channel.image *= eq

    def cmyk_to_rgb(c, m, y, k, cmyk_scale, rgb_scale=255):
        cmyk_scale = float(cmyk_scale)
        r = rgb_scale * (1. - c / cmyk_scale) * (1. - k / cmyk_scale)
        g = rgb_scale * (1. - m / cmyk_scale) * (1. - k / cmyk_scale)
        b = rgb_scale * (1. - y / cmyk_scale) * (1. - k / cmyk_scale)
        return r, g, b

    def combine_starlists(self):
        """Combine star lists of all frames to a master list for the picture."""
        def match(star_list_1, star_list_2):
            for star_1 in star_list_1:
                for star_2 in star_list_2:
                    if star_2 == star_1:
                        new_star = copy.deepcopy(star_1)
                        new_star.flux.update(star_2.flux)
                        new_star.ctmg.update(star_2.ctmg)
                        # print(new_star.ctmg)
                        # FIXME: add proper combining function, all fluxes and
                        #        all cat info!!!
                        # print(star_1.name, star_2.name)
                        yield new_star
        self.stars = list(match(self.frames[0].stars, self.frames[1].stars))

    def display_all(self, coord=None):
        """
        Display image and all known stars.

        Parameters
        ----------
        coord : wcs, optional
            Coordinate system to use for the image. If None, axes will be in
            pixels. The default is None.

        Returns
        -------
        None.

        """
        if coord is not None:
            plt.subplot(projection=coord)
        plt.imshow(self.image, vmin=100., vmax=1000., origin="lower")
        plt.grid(color='white', ls='dashed')
        plt.xlabel("RA")
        plt.ylabel("DE")
        plt.tight_layout()
        for star in self.stars:
            star.display()

    def color_mag_diagram(self, bands=None):
        """
        Produce and plot a color magnitude diagram.

        Parameters
        ----------
        bands : list of two strings, optional
            The bands to use when creating the CMD. The first value will be
            used for the y-axis. If None, the first two bands of the picture
            will be used. The default is None.

        Raises
        ------
        ValueError
            Will be raised if less than two bands are given.

        Returns
        ------
        None.

        """
        if bands is None:
            bands = self.bands

        if len(bands) < 2:
            raise ValueError("Need at least two bands to create CMD.")

        def mk_mags_cols():
            for star in self.stars:
                mag = star.flux[bands[0]].mag
                col = star.flux[bands[0]].mag - \
                      star.flux[bands[1]].mag
                yield (mag, col)

        mags, cols = zip(*mk_mags_cols())
        plt.scatter(cols, mags)
        plt.xlabel(f"{self.bands[0]} - {self.bands[1]}")
        plt.ylabel(self.bands[0])
        plt.gca().invert_yaxis()

    def to_txt(self, fname):
        """save star list to txt file"""
        with open(fname, "w") as file:
            sout = "star_identification\t"
            for band in self.bands:
                sout += f"m{band}[mag]\terr_m{band}\t"
            sout = sout[:-1] + "\r"
            file.write(sout)
            for star in self.stars:
                sout = str(star)
                for band in self.bands:
                    flux = star.flux.get(band, None)
                    if flux is not None:
                        length = int(abs(np.log10(flux.err_mag))) + 1
                        mag = round(flux.mag, length)
                        error = round(flux.err_mag, length)
                        sout += f"\t{mag}\t{error}"
                        # TODO: proper str formatting instead of round
                        # FIXME: change all this to new Flux syntax!
                    else:
                        sout += f"\t{np.nan}\t{np.nan}"
                sout += "\r"
                file.write(sout)

    def _update_header(self):
        hdr = self.frames[0].header
        # TODO: properly do this ^^
        hdr.update(AUTHOR="Fabian Haberhauer")

    @staticmethod
    def _make_jpeg_variable_segment(marker: int, payload: bytes) -> bytes:
        """Make a JPEG segment from the given payload."""
        return struct.pack('>HH', marker, 2 + len(payload)) + payload

    @staticmethod
    def _make_jpeg_comment_segment(comment: bytes) -> bytes:
        """Make a JPEG comment/COM segment."""
        return Picture._make_jpeg_variable_segment(0xFFFE, comment)

    @staticmethod
    def save_hdr(fname, hdr):
        # TODO: log all of this crape
        logger.debug("saving header:")
        logger.debug(hdr.tostring(sep="\n"))
        with Image.open(fname) as img:
            app = img.app["APP0"]

        with open(fname, mode='rb') as file:
            binary = file.read()

        pos = binary.find(app) + len(app)
        bout = binary[:pos]
        bout += Picture._make_jpeg_comment_segment(hdr.tostring().encode())
        bout += binary[pos:]

        with open(fname, mode='wb') as file:
            file.write(bout)

    def save_pil(self, fname):
        logger.info("Saving image as JPEG to %s", fname)
        rgb = self.get_rgb_cube(mode="0-255", order="xyc")
        # HACK: does this always produce correct orientation??
        rgb = np.flip(rgb, 0)
        hdr = self._update_header()

        Image.MAX_IMAGE_PIXELS = self.image_size + 1
        with Image.fromarray(rgb) as img:
            try:
                img.save(fname)
            except (KeyError, OSError):
                logger.warning("Cannot save RGBA as JPEG, converting to RGB.")
                img = img.convert('RGB')
                img.save(fname)
            # img.save(fname, comment=hdr.tostring())

        self.save_hdr(fname, hdr)
