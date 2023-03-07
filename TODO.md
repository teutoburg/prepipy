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


## Possible future additions to command line args

* Masking/regions, maybe in command line additionally to config file if that makes sense, otherwise just an option to specify the masking YAML file.
* ROI coordinate list...
* MPL option(s), like in config file, maybe some kind of sub-parser?
* Image cutouts, in pixel coordinates for now.
* Specify RGB combination, if no config and bands files are given.
* Check relative paths for config files using ./ when running from command-line, especially in Linux.
* Command-line mode with \* in image name for multiple. image_name should be glob-able, aka nargs="?" and on Linux it should work out of the box (shell), on Windows need to glob manually, see also [this post](https://stackoverflow.com/a/71353522/8467078).

## Changes to configuration

Also inlude the command line arguments in the config file and vice verse.
1. Load config file
2. Overwrite all options from the config file which were set from the command line.
3. Pass config object, not individual kwargs.

Additionally: If bands.yml not found &rarr; try to use those specified in config, if not &rarr; glob

## Multiprocessing

Bring back progress bar for MP loading. See [this](https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm), [this](https://stackoverflow.com/questions/41920124/multiprocessing-use-tqdm-to-display-a-progress-bar) and maybe [this](https://tqdm.github.io/docs/contrib.concurrent/)
Make MP a bit config-able, no of workers and chunksize...
