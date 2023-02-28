# TODO prepipy, in no particular order...

* **Big one**: Check if stiff stretching case selection is implemented correctly (probably not)...
* quantile, percentile, nan-
* Make autostretch useable again.
* Simplify choice of stretch mode, passing function should not be necessary.
* If bands.yml not found &rarr; try to use those specified in config, if not &rarr; glob
* Command-line mode with \* in image name for multiple &rarr; maybe this can replace the current `-m` keyword, which could then become mask.
* Check relative paths for config files using ./ when running from command-line, especially in Linux.
* Multiprocessing is not using filename template...
* Maybe add Gaussian stretching if not too complicated.
* See if MPL features can be included in JPEG putput. See also [this](https://stackoverflow.com/questions/57316491/how-to-convert-matplotlib-figure-to-pil-image-object-without-saving-image) and [this](https://stackoverflow.com/questions/3938676/python-save-matplotlib-figure-on-an-pil-image-object) link.
* Maybe change stiff algo to ufunc casting, aka:

        image_s = np.add(image, slope, where=linear_part)
        np.power(image, 1/gamma, out=image_s, where=~linear_part)
        np.multiply(image_s, offset + 1, out=image_s, where=~linear_part)
        np.subtract(image_s, offset, out=image_s, where=~linear_part)

    but chech if this makes any difference in execution time and memory and if it produces the same result
