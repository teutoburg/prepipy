# TODO prepipy, in no particular order...

## Process

* quantile, percentile, nan-
* Make autostretch useable again.
* Maybe add Gaussian stretching if not too complicated.
* See if MPL features can be included in JPEG putput. See also [this](https://stackoverflow.com/questions/57316491/how-to-convert-matplotlib-figure-to-pil-image-object-without-saving-image) and [this](https://stackoverflow.com/questions/3938676/python-save-matplotlib-figure-on-an-pil-image-object) link.

## Numpy ufunc stuff

Maybe change stiff algo to ufunc casting, aka:

    image_s = np.add(image, slope, where=linear_part)
    np.power(image, 1/gamma, out=image_s, where=~linear_part)
    np.multiply(image_s, offset + 1, out=image_s, where=~linear_part)
    np.subtract(image_s, offset, out=image_s, where=~linear_part)

but chech if this makes any difference in execution time and memory and if it produces the same result

## Changes to configuration

### Possible future additions to command line args

* ROI coordinate list...
* MPL option(s), like in config file, maybe some kind of sub-parser (see also below)?
* Image cutouts, in pixel coordinates for now.
* Command-line mode with \* in image name for multiple. image_name should be glob-able, aka nargs="?" and on Linux it should work out of the box (shell), on Windows need to glob manually, see also [this post](https://stackoverflow.com/a/71353522/8467078).
* An option to create a template config, bands and masking file containing defaults in the CWD (and abort afterwards).
* Investigate [sub-parsers](https://docs.python.org/3/library/argparse.html#sub-commands), maybe in the future we would have `prepipy rgbcombo ...` or something like that.

## Multiprocessing

Bring back progress bar for MP loading. See [this](https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm), [this](https://stackoverflow.com/questions/41920124/multiprocessing-use-tqdm-to-display-a-progress-bar) and maybe [this](https://tqdm.github.io/docs/contrib.concurrent/)
Make MP a bit config-able, no of workers and chunksize...


## Combine images

See [this](https://note.nkmk.me/en/python-pillow-concat-images/) maybe...
