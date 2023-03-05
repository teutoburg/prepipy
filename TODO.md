# TODO prepipy, in no particular order...

* **Big one**: Check if stiff stretching case selection is implemented correctly (probably not)...
* quantile, percentile, nan-
* Make autostretch useable again.
* If bands.yml not found &rarr; try to use those specified in config, if not &rarr; glob
* Command-line mode with \* in image name for multiple
* Maybe add Gaussian stretching if not too complicated.
* See if MPL features can be included in JPEG putput. See also [this](https://stackoverflow.com/questions/57316491/how-to-convert-matplotlib-figure-to-pil-image-object-without-saving-image) and [this](https://stackoverflow.com/questions/3938676/python-save-matplotlib-figure-on-an-pil-image-object) link.
* Maybe change stiff algo to ufunc casting, aka:

        image_s = np.add(image, slope, where=linear_part)
        np.power(image, 1/gamma, out=image_s, where=~linear_part)
        np.multiply(image_s, offset + 1, out=image_s, where=~linear_part)
        np.subtract(image_s, offset, out=image_s, where=~linear_part)

    but chech if this makes any difference in execution time and memory and if it produces the same result
* Possible future additions to command line args: masking/regions, ROI, MPL option(s), cutout (pixel), rgbcombo (if no config and bands)
* Also inlude these in config file and vice verse. Ideally, load config file, then overwrite all options set from command line, then pass config object, not individual kwargs.
* Check relative paths for config files using ./ when running from command-line, especially in Linux. image_name should be glob-able, aka nargs="?" and on Linux it should work out of the box (shell), on Windows need to glob manually, see also [this post](https://stackoverflow.com/a/71353522/8467078).
