#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Framework for images etc. when used with source extraction etc.

Not actively maintained!
"""

import logging
import copy

import numpy as np
import matplotlib.pyplot as plt

from astropy import wcs


from framework import Frame, Picture

logger = logging.getLogger(__name__)


class Error(Exception):
    """Base class for exeptions in this module."""

    pass


class SourcesFrame(Frame):
    """Subclass of Frame for use with source extraction."""

    EXPTIME = 0.
    EXTINCT = 0.
    AIRMASS = 0.

    def __init__(self, image, band, header=None, **kwargs):
        super().__init__(image, band, header, **kwargs)
        self.stars = list()
        self.coord = None

    def phot_zero(self):
        """Calculate global photometrical zeropoint."""
        for star in self.stars:
            star.mk_phot_zeropoint(self.EXPTIME, self.EXTINCT, self.AIRMASS)

        zeropoints = [star.phzp.get(self._band, None) for star in self.stars]
        zeropoint = np.mean([zero.mag for zero in zeropoints if zero is not None])
        err_zero = np.max([zero.err_mag for zero in zeropoints if zero is not None])
        # self.phot_zero = CatFlux(self._band, zeropoint, err_zero)
        return CatFlux(self._band, zeropoint, err_zero)

    def add_cal_mag(self):
        """legacy, check!"""
        for star in self.stars:
            star.cal_mag(self.phot_zero(), self.EXPTIME, self.EXTINCT,
                         self.AIRMASS)

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
        plt.grid(color="white", ls="dashed")
        plt.xlabel("RA")
        plt.ylabel("DE")
        plt.tight_layout()
        for star in self.stars:
            star.display()


class SourcesPicture(Picture):
    """n/a."""

    def __init__(self, name=None):
        super().__init__(name)
        self.stars = list()

    @property
    def num_stars(self):
        """Number of fitted, identified and cross-matched stars.
        Read-only property."""
        return len(self.stars)

    def combine_starlists(self):
        """Combine star lists of all frames to master list for the picture."""
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
        plt.grid(color="white", ls="dashed")
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
