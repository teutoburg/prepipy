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
import matplotlib as mpl
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy import wcs
# from astropy.stats import median_absolute_deviation as mad
from astropy.stats import sigma_clipped_stats as scs
from astropy.nddata import Cutout2D

from PIL import Image
from tqdm import tqdm

mpl.rcParams["font.family"] = ["Computer Modern", "serif"]

TQDM_FMT = "{l_bar}{bar:50}{r_bar}{bar:-50b}"
logger = logging.getLogger(__name__)


class Error(Exception):
    """Base class for exeptions in this module."""


class BandExistsError(Error):
    """A Frame in this band already exists in this picture."""


class FileTypeError(Error):
    """The given file has the wrong or an unknown type."""


@dataclass
class Band():
    """n/a."""

    name: str
    printname: str = None
    wavelength: float = None
    instrument: str = "unknown"
    telescope: str = "unknown"

    @classmethod
    def from_yaml_dict(cls, bands, use_bands=None):
        """Create incstance from YAML dictionary entry (default factory)."""
        # parse shortened names to correct parameter names
        for printname, band in bands.items():
            band["instrument"] = band.pop("inst")
            band["telescope"] = band.pop("tele")
            band["wavelength"] = band.pop("wave")
            band["printname"] = printname.replace("_", " ")

        # either use specified or all, anyway turn into tuple without names
        if use_bands is not None:
            bands = itemgetter(*use_bands)(bands)
        else:
            bands = tuple(bands.values())

        # in case we ever need the names as well...
        # for name, band in bands.items():
        #     yield name, cls(**band)
        for band in bands:
            yield cls(**band)


class Frame():
    """n/a."""

    def __init__(self, image, band, header=None, **kwargs):
        self.header = header
        self.coords = wcs.WCS(self.header)

        if "imgslice" in kwargs:
            cen = [sh//2 for sh in image.shape]
            cutout = Cutout2D(image, cen, kwargs["imgslice"], self.coords)
            self.image = cutout.data
            self.coords = cutout.wcs
        else:
            self.image = image

        self._band = band
        self.background = np.nanmedian(self.image)  # estimated background
        self.clip_and_nan(**kwargs)

        self.sky_mask = None

    def __repr__(self):
        return str(self)  # DEBUG only

    def __str__(self):
        return f"{self.shape} frame in \"{self.band.printname}\" band"

    @classmethod
    def from_fits(cls, filename, band, **kwargs):
        """Create instance from fits file."""
        with fits.open(filename) as file:
            return cls(file[0].data, band, file[0].header, **kwargs)

    @property
    def band(self):
        """Get pass-band in which the frame was taken. Read-only property."""
        return self._band

    @property
    def shape(self):
        """Get shape of image array as pretty string."""
        return f"{self.image.shape[0]} x {self.image.shape[1]}"

    def camera_aperture(self, center: tuple[int], radius: float) -> None:
        """
        Remove vignetting effects.

        Set everything outside a defined radius around a defined center to
        zero.

        Parameters
        ----------
        center : 2-tuple of ints
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

    def clip_and_nan(self, clip=10, nanmode="max", **kwargs):
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

    def display_3d(self):
        """Show frame as 3D plot (z=intensity)."""
        x_grid, y_grid = np.mgrid[0:self.image.shape[0], 0:self.image.shape[1]]
        fig = plt.figure()
        axis = fig.gca(projection="3d")
        axis.plot_surface(x_grid, y_grid, self.image, rstride=1, cstride=1,
                          cmap="viridis", linewidth=0)
        axis.set_title(self.band.name)
        plt.show()

    def normalize(self, new_range=1., new_offset=0.):
        """Subtract minimum and devide by maximum."""
        data_max = np.nanmax(self.image)
        data_min = np.nanmin(self.image)
        data_range = data_max - data_min

        # Check if data is already normalized for all practical purposes
        if np.isclose((data_min, data_max), (0., 1.), atol=1e-5).all():
            return data_range, data_max

        self.image -= data_min
        self.image /= data_range

        try:
            assert np.nanmin(self.image) == 0.0
            assert np.nanmax(self.image) == 1.0
        except AssertionError:
            logger.error("Normalisation error: %f, %f", data_max, data_min)

        self.image = self.image * new_range + new_offset

        return data_range, data_max

    def clipped_stats(self):
        """Calculate sigma-clipped image statistics."""
        data = self.image.flatten()
        mean, median, stddev = np.mean(data), np.median(data), np.std(data)
        logger.debug("%10s:  mean=%-8.4fmedian=%-8.4fstddev=%.4f", "unclipped",
                     mean, median, stddev)
        mean, median, stddev = scs(data,
                                   cenfunc="median", stdfunc="mad_std",
                                   sigma_upper=5, sigma_lower=3)
        logger.debug("%10s:  mean=%-8.4fmedian=%-8.4fstddev=%.4f", "clipped",
                     mean, median, stddev)
        return mean, median, stddev

    def _min_inten(self, gamma_lum, grey_level=.3,
                   sky_mode="median", max_mode="quantile", **kwargs):
        if sky_mode == "quantile":
            i_sky = np.quantile(self.image, .8)
        elif sky_mode == "median":
            i_sky = np.nanmedian(self.image)
        elif sky_mode == "clipmedian":
            _, clp_median, _ = self.clipped_stats()
            i_sky = clp_median
        elif sky_mode == "debug":
            i_sky = .01
        else:
            raise ValueError("sky_mode not understood")

        if max_mode == "quantile":
            i_max = np.quantile(self.image, .995)
        elif max_mode == "max":
            # FIXME: if we normalize before, this will always be == 1.0
            i_max = np.nanmax(self.image)
        elif max_mode == "debug":
            i_max = .99998
        else:
            raise ValueError("max_mode not understood")

        p_grey_g = grey_level**gamma_lum
        logger.debug("p_grey_g=%.4f", p_grey_g)
        logger.debug("i_sky=%.4f", i_sky)
        i_min = max((i_sky - p_grey_g * i_max) / (1. - p_grey_g), 0.)
        logger.debug("i_min=%-10.4fi_max=%.4f", i_min, i_max)
        return i_min, i_max

    def stiff_d(self, stretch_function, gamma_lum=1.5, grey_level=.1,
                **kwargs):
        """Stretch frame based on modified STIFF algorithm."""
        logger.info("stretching %s band", self.band.name)
        data_range, _ = self.normalize()

        # gamma_lum = self.auto_gma()

        i_min, i_max = self._min_inten(gamma_lum, grey_level, **kwargs)
        self.sky_mask = self.image < i_min
        self.image = self.image.clip(i_min, i_max)

        new_img = stretch_function(self.image, **kwargs)
        # new_img = self.stiff_stretch(self.image, **kwargs)
        new_img *= data_range

        self.image = new_img
        logger.info("%s band done", self.band.name)

    @staticmethod
    def stiff_stretch(image, stiff_mode="power-law", **kwargs):
        """Stretch frame based on STIFF algorithm."""
        def_kwargs = {"power-law": {"gamma": 2.2, "a": 1., "b": 0., "i_t": 0.},
                      "srgb":
                          {"gamma": 2.4, "a": 12.92, "b": .055, "i_t": .00304},
                      "rec709":
                          {"gamma": 2.22, "a": 4.5, "b": .099, "i_t": .018},
                      "user": {"gamma": 2.25, "a": 3., "b": .08, "i_t": .003},
                      "user2": {"gamma": 2.25, "a": 3., "b": .1, "i_t": .8},
                      "user3": {"gamma": 2.25, "a": 3., "b": .05, "i_t": .001},
                      "user4": {"gamma": 2.25, "a": 3., "b": .05, "i_t": .003}}
        if stiff_mode not in def_kwargs:
            raise KeyError(f"Mode must be one of {list(def_kwargs.keys())}.")

        kwargs = def_kwargs[stiff_mode] | kwargs

        # kwargs["gamma"] = self.auto_gma()
        # assert kwargs['gamma'] == 2.25  # HACK: DEBUG ONLY

        b_slope, i_t = kwargs["b"], kwargs["i_t"]
        image_s = kwargs["a"] * image * (image < i_t)
        image_s += (1 + b_slope) * image**(1/kwargs["gamma"])
        image_s -= b_slope * (image >= i_t)
        return image_s

    @staticmethod
    def autostretch_light(image, **kwargs):
        """Stretch frame based on autostretch algorithm."""
        # logger.info("Begin autostretch for \"%s\" band", self.band.name)
        # maximum = self.normalize()
        # logger.info("maximum:\t%s", maximum)

        mean = np.nanmean(image)
        sigma = np.nanstd(image)
        logger.info(r"$\mu$:\t%s", mean)
        logger.info(r"$\sigma$:\t%s", sigma)

        gamma = np.exp((1 - (mean + sigma)) / 2)
        logger.info(r"$\gamma$:\t%s", gamma)

        k = np.power(image, gamma) + ((1 - np.power(image, gamma))
                                      * (mean**gamma))
        image_s = np.power(image, gamma) / k

        # image_s *= maximum
        return image_s

    def auto_gma(self):
        """Find gamma based on exponential function. Highly experimental."""
        clp_mean, _, clp_stddev = self.clipped_stats()
        return np.exp((1 - (clp_mean + clp_stddev)) / 2)


class Picture():
    """n/a."""

    def __init__(self, name=None):
        self.frames = list()
        self.name = name

    @property
    def bands(self):
        """List of all bands in the frames. Read-only property."""
        return [frame.band for frame in self.frames]

    @property
    def primary_frame(self):
        """Get the first frame from the frames list. Read-only property."""
        if not self.frames:
            raise ValueError("No frame loaded.")
        return self.frames[0]

    @property
    def image(self):
        """Get combined image of all frames. Read-only property."""
        # HACK: actually combine all images!
        return self.primary_frame.image

    @property
    def coords(self):
        """WCS coordinates of the first frame. Read-only property."""
        return self.primary_frame.coords

    @property
    def center(self):
        """Pixel coordinates of center of first frame. Read-only property."""
        return [sh//2 for sh in self.image.shape]

    @property
    def center_coords(self):
        """WCS coordinates of the center of first frame. Read-only property."""
        return self.coords.pixel_to_world(*self.center)

    @property
    def center_coords_str(self):
        """Get string conversion of center coords. Read-only property."""
        cen = self.center_coords
        return cen.to_string("hmsdms", sep=" ", precision=2)

    @property
    def image_size(self):
        """Get number of pixels per frame. Read-only property."""
        return self.frames[0].image.size

    @property
    def cube(self):
        """Stack images from all frames in one 3D cube. Read-only property."""
        return np.stack([frame.image for frame in self.frames])

    def _check_band(self, band):
        if isinstance(band, str):
            band = Band(band)
        elif not isinstance(band, Band):
            raise TypeError("Invalid type for band.")

        if band in self.bands:
            raise BandExistsError(("Picture already includes a Frame in "
                                   "\"{band.name}\" band."))

        return band

    def add_frame(self, image, band, header=None, **kwargs):
        """Add new frame to Picture using image array and band information."""
        band = self._check_band(band)
        new_frame = Frame(image, band, header, **kwargs)
        self.frames.append(new_frame)
        return new_frame

    def add_frame_from_file(self, filename, band, framelist=None, **kwargs):
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
            new_frame = Frame.from_fits(filename, band, **kwargs)
        else:
            raise FileTypeError("Currently only FITS files are supported.")

        if framelist is None:
            self.frames.append(new_frame)
        else:
            framelist.append(new_frame)
        return new_frame

    def add_fits_frames_mp(self, input_path, bands):
        """Add frames from fits files for each band, using multiprocessing."""
        args = [(input_path/f"{self.name}_{band.name}.fits", band)
                for band in bands]
        with Pool(len(args)) as pool:
            framelist = pool.starmap(Frame.from_fits, args)
        self.frames = framelist

    @classmethod
    def from_cube(cls, cube, bands=None):
        """
        Create Picture instance from 3D array (data cube) and list of bands.

        Parameters
        ----------
        cube : array_like, shape (N, M, K)
            Data cube (3D array) containing image data. Shape is interpreted as
            N images of dimension M x K.
        bands : iterable of Band objects or str of length N, optional
            Any iterable containing information about the bands. Bands can be
            given as Band objects or simply str. Length of the iterable must
            match number of images in `cube` (N). If None, bands will be marked
            "unknown". The default is None.

        Raises
        ------
        TypeError
            Raised if data type of `bands` is invalid.
        IndexError
            Raised if ``len(bands) != len(cube)``.

        Returns
        -------
        new_picture : Picture
            New instance of Picture including the created frames.

        """
        if not cube.ndim == 3:
            raise TypeError("A \"cube\" must have exactly 3 dimensions!")

        if bands is None:
            bands = len(cube) * [Band("unknown")]
            bands = [Band(f"unkown{i}") for i, _ in enumerate(cube)]
        elif all(isinstance(band, str) for band in bands):
            bands = [Band(band) for band in bands]
        elif not all(isinstance(band, Band) for band in bands):
            raise TypeError("Invalid data type in bands.")

        if not len(bands) == len(cube):
            # HACK: change this to zip(..., strict=True) below in Python 3.10+
            raise IndexError(("Length of bands in list must equal the number "
                              "of images in the cube."))

        new_picture = cls()

        for image, band in zip(cube, bands):
            new_picture.add_frame(image, band)

        return new_picture

    @classmethod
    def from_tesseract(cls, tesseract, bands=None):
        """Generate individual pictures from 4D cube."""
        for cube in tesseract:
            yield cls.from_cube(cube, bands)

    @staticmethod
    def merge_tesseracts(tesseracts):
        """Merge multiple 4D image cubes into a single one."""
        return np.hstack(list(tesseracts))

    @staticmethod
    def combine_into_tesseract(pictures):
        """Combine multiple 3D picture cubes into one 4D cube."""
        return np.stack([picture.cube for picture in pictures])

    def stretch_frames(self, mode="auto-light", **kwargs):
        """Perform stretching on frames."""
        for frame in self.frames:
            if mode == "auto-light":
                frame.autostretch_light(**kwargs)
            elif mode == "stiff-d":
                frame.stiff_d(**kwargs)
            else:
                raise ValueError("stretch mode not understood")

    def _update_header(self):
        hdr = self.frames[0].header
        # TODO: properly do this ^^
        hdr.update(AUTHOR="Fabian Haberhauer")
        return hdr


class RGBPicture(Picture):
    """Picture subclass for ccombining frames to colour image (RGB)."""

    def __init__(self, name=None):
        super().__init__(name)

        self.params = None
        self.rgb_channels = None

    @property
    def is_bright(self):
        """Return True is any median of the RGB frames is >.2."""
        return any(np.median(c.image) > .2 for c in self.rgb_channels)

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

        if order == "cxy":
            pass
        elif order == "xyc":
            rgb = np.moveaxis(rgb, 0, -1)
        else:
            raise ValueError("order not understood")

        if mode == "0-1":
            pass
        elif mode == "0-255":
            rgb *= 255.
            rgb = rgb.astype(np.uint8)
        else:
            raise ValueError("mode not understood")

        return rgb

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
        self.rgb_channels = list(map(copyfct,
                                     (itemgetter(*bands)(frames_dict))))

        if all(channel.band.wavelength is not None
               for channel in self.rgb_channels):
            if not all(redder.band.wavelength >= bluer.band.wavelength
                       for redder, bluer
                       in zip(self.rgb_channels, self.rgb_channels[1:])):
                raise UserWarning(("Not all RGB channels are ordered by "
                                   "descending wavelength."))

        _chnames = [channel.band.name for channel in self.rgb_channels]
        logger.info("Successfully selected %i RGB channels: %s",
                    len(_chnames), ", ".join(map(str, _chnames)))
        return self.rgb_channels

    def norm_by_weights(self, weights):
        """Normalize channels by weights.

        Currently only supports "auto" mode, using clipped median as weight.
        """
        if weights is not None:
            if isinstance(weights, str):
                if weights == "auto":
                    clipped_stats = [chnl.clipped_stats()
                                     for chnl in self.rgb_channels]
                    _, clp_medians, _ = zip(*clipped_stats)
                    weights = [1/median for median in clp_medians]
                else:
                    raise ValueError("weights mode not understood")

            for channel, weight in zip(self.rgb_channels, weights):
                channel.image *= weight

    def stretch_frames(self, mode="auto-light", only_rgb=False, **kwargs):
        """Perform stretching on frames."""
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

    def autoparam(self):
        """Experimental automatic parameter estimation."""
        gamma = 2.25
        gamma_lum = 1.5
        alpha = 1.4
        grey_level = .3

        self.params = {"gma": False, "alph": False}

        clipped_stats = [chnl.clipped_stats() for chnl in self.rgb_channels]
        _, clp_medians, clp_stddevs = zip(*clipped_stats)

        if not any((np.array(clp_medians) / np.mean(clp_medians)) > 2.):
            gamma_lum = 1.2
            self.params["gma"] = True

        if (all(median > 200. for median in clp_medians)
            and
            all(stddev > 50. for stddev in clp_stddevs)):
            alpha = 1.2
            self.params["alph"] = True
            print(self.name)

        return gamma, gamma_lum, alpha, grey_level

    def luminance(self):
        """Calculate the luminance of the RGB image.

        The luminance is defined as the (pixel-wise) sum of all colour channels
        divided by the number of channels.
        """
        sum_image = sum(frame.image for frame in self.rgb_channels)
        sum_image /= len(self.rgb_channels)
        return sum_image

    def stretch_luminance(self, stretch_fkt_lum, gamma_lum, lum, **kwargs):
        """Perform luminance stretching.

        The luminance stretch function `stretch_fkt_lum` is expected to take
        positional arguments `lum` (image luminance) and `gamma_lum` (gamma
        factor used for stretching). Any additional kwargs will be passed to
        `stretch_fkt_lum`.
        """
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
        -----
        Method should also work for 2-channel images, but this is not tested.
        Method should also work for 4-channel images, but this is not tested.

        """
        # BUG: sometimes lost of zeros, maybe normalize before this?
        #      Is this still an issue??
        logger.info("RGB adjusting using alpha=%.3f, gamma_lum=%.3f.",
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
        # FIXME: should the lu stretch be done with the original luminance
        #        (as is currently) or with the adjusted one???
        self.stretch_luminance(stretch_fkt_lum, gamma_lum, lum, **kwargs)

    def equalize(self, mode="mean", offset=.5, norm=True, supereq=False):
        """
        Perform a collection of processes to enhance the RGB image.

        Parameters
        ----------
        mode : str, optional
            "median" or "mean". The default is "mean".
        offset : TYPE, optional
            To be added before clipping negative values. The default is 0.5.
        norm : bool, optional
            Whether to perform normalisation in each channel.
            The default is True.
        supereq : bool, optional
            Whether to perform additional crocc-channel equalisation. Currently
            highly experimental feature. The default is False.

        Returns
        -------
        None.

        """
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
            for channel, mean in zip(self.rgb_channels, means):
                equal = min(maxmean/mean, 10.)
                channel.image *= equal

    @staticmethod
    def cmyk_to_rgb(cmyk, cmyk_scale, rgb_scale=255):
        """Convert CMYK to RGB."""
        cmyk_scale = float(cmyk_scale)
        scale_factor = rgb_scale * (1. - cmyk[3] / cmyk_scale)
        rgb = ((1. - cmyk[0] / cmyk_scale) * scale_factor,
               (1. - cmyk[1] / cmyk_scale) * scale_factor,
               (1. - cmyk[2] / cmyk_scale) * scale_factor
               )
        # TODO: make this an array, incl. mult. w/ sc.fc. afterwards etc.
        return rgb


class JPEGPicture(RGBPicture):
    """RGBPicture subclass for single image in JPEG format using Pillow."""

    @staticmethod
    def _make_jpeg_variable_segment(marker: int, payload: bytes) -> bytes:
        """Make a JPEG segment from the given payload."""
        return struct.pack('>HH', marker, 2 + len(payload)) + payload

    @staticmethod
    def _make_jpeg_comment_segment(comment: bytes) -> bytes:
        """Make a JPEG comment/COM segment."""
        return JPEGPicture._make_jpeg_variable_segment(0xFFFE, comment)

    @staticmethod
    def save_hdr(fname, hdr):
        """Save header as JPEG comment. Redundant with pillow 9.4.x."""
        # TODO: log all of this crape
        logger.debug("saving header:")
        logger.debug(hdr.tostring(sep="\n"))
        with Image.open(fname) as img:
            app = img.app["APP0"]

        with open(fname, mode="rb") as file:
            binary = file.read()

        pos = binary.find(app) + len(app)
        bout = binary[:pos]
        bout += JPEGPicture._make_jpeg_comment_segment(hdr.tostring().encode())
        bout += binary[pos:]

        with open(fname, mode="wb") as file:
            file.write(bout)

    def save_pil(self, fname: str):
        """
        Save RGB image to specified file name using pillow.

        Parameters
        ----------
        fname : str
            Full file path and name.

        Returns
        -------
        None.

        """
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
                img = img.convert("RGB")
                img.save(fname)
            # img.save(fname, comment=hdr.tostring())

        self.save_hdr(fname, hdr)


class MPLPicture(RGBPicture):
    """RGBPicture subclass for collage of one or more RGB combinations.

    Additional information can be added to the collage, which is created using
    Matplotlib and saved in pdf format.
    """

    # padding = {1: 5, 2: 5, 4: 4}
    padding = {2: 3.5}

    @property
    def title(self):
        """Get string-formatted name of Picture."""
        return f"Source ID: {self.name}"

    def _add_histo(self, axis):
        cube = self.get_rgb_cube(order="cxy")
        axis.hist([img.flatten() for img in cube],
                  20, color=("r", "g", "b"))

    @staticmethod
    def _plot_coord_grid(axis):
        axis.grid(color="w", ls=":")

    def _plot_center_marker(self, axis):
        axis.plot(*self.center, "w+", ms=10)

    def _display_cube(self, axis, center=False, grid=False):
        # FIXME: What about origin="lower" ??
        axis.imshow(self.get_rgb_cube(order="xyc"),
                    aspect="equal", origin="lower")
        axis.set_xlabel("right ascension")
        axis.set_ylabel("declination", labelpad=0)
        axis.coords[0].set_ticklabel(exclude_overlapping=True)
        axis.coords[1].set_ticklabel(exclude_overlapping=True)
        if center:
            self._plot_center_marker(axis)
        if grid:
            self._plot_coord_grid(axis)

    def _display_cube_histo(self, axes, cube):
        axes[0].imshow(cube.T, origin="lower")
        # self._mk_coord_etc(axes[0])
        self._add_histo(axes[1])

    def _get_axes(self, nrows, ncols, figsize_mult):
        # 3, 5.6
        figsize = tuple(n * s for n, s in zip((ncols, nrows), figsize_mult))
        fig = plt.figure(figsize=figsize, dpi=300)
        # subfigs = fig.subfigures(nrows)
        # for subfig in subfigs[::2]:
        # for subfig in subfigs:
        #     subfig.subplots(1, ncols, subplot_kw={"projection": coord})
        # for subfig in subfigs[1::2]:
        #     subfig.subplots(1, ncols)
        axes = fig.subplots(nrows, ncols,
                            subplot_kw={"projection": self.coords})
        # axes = [subfig.axes for subfig in subfigs]
        # axes = list(map(list, zip(*axes)))
        return fig, axes.T

    # def _get_axes(self, nrows, ncols):
    #     fig = plt.figure(figsize=(ncols * 15, nrows * 15),
    #                      dpi=600, frameon=False)
    #     axes = fig.subplots(nrows, ncols,
    #                         subplot_kw={"projection": self.coords})
    #     return fig, axes

    def _create_title(self, axis, combo, mode="debug", equal=False):
        if mode == "debug":
            title = "R: {}, G: {}, B: {}".format(*combo)
            title += "\nequalize = "
        elif mode == "pub":
            channels = (f"{chnl.band.printname} ({chnl.band.wavelength} µm)"
                        for chnl in self.rgb_channels)
            title = "Red: {}\nGreen: {}\nBlue: {}".format(*channels)
        else:
            raise ValueError("Title mode not understood.")
        axis.set_title(title, pad=7, fontdict={"multialignment": "left"})

    def stuff(self, channel_combos, imgpath, grey_mode="normal",
              figurekwargs=None, **kwargs):
        """DEBUG ONLY."""
        default_figurekwargs = {"titlemode": "debug",
                                "include_suptitle": True,
                                "figsize": (3, 5.6)}
        if figurekwargs is not None:
            figurekwargs = default_figurekwargs | figurekwargs
        else:
            figurekwargs = default_figurekwargs
        grey_values = {"normal": .3, "lessback": .08, "moreback": .7}

        nrows, ncols = 1, len(channel_combos)
        fig, axes = self._get_axes(nrows, ncols, figurekwargs["figsize"])
        for combo, column in zip(tqdm(channel_combos), axes.flatten()):
            self.select_rgb_channels(combo)
            self.stretch_frames("stiff-d", only_rgb=True,
                                stretch_function=Frame.stiff_stretch,
                                stiff_mode="user3",
                                grey_level=grey_values[grey_mode], **kwargs)

            if self.is_bright:
                self.equalize("median",
                              offset=kwargs.get("equal_offset", .1),
                              norm=kwargs.get("equal_norm", True))
                equal = "True"
            else:
                equal = "False"

            self._create_title(column, combo, figurekwargs["titlemode"], equal)
            # _display_cube_histo(column[:2], pic.rgb_cube)
            # _display_cube(column[0], pic.rgb_cube)
            self._display_cube(column)
            # _display_cube_histo(column[2:], pic.rgb_cube)
            # _display_cube(column[1], pic.rgb_cube)

        if figurekwargs["include_suptitle"]:
            suptitle = self.title + "\n" + self.center_coords_str
            fig.suptitle(suptitle, fontsize="xx-large")

        fig.tight_layout(pad=self.padding[ncols])
        # fig.tight_layout()

        fig.savefig(imgpath/f"{self.name}.pdf")
        plt.close(fig)
        del fig
