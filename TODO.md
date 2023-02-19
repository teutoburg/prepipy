TODO prepipy, in no particular order...
* **Big one**: Check if stiff stretching case selection is implemented correctly (probably not)...
* quantile, percentile, nan-
* Check gamma lum stretch everywhere, confusion about X^gamma versus X^1/gamma
* Check if the implementation of is_bright is still useful with current normalisations.
* Make autostretch useable again.
* Simplify choice of stretch mode, passing function should not be necessary.
* For MPL mode, include:
    1. pixel or wcs frame choice
    2. histo choice
    3. before-after choice
    4. multiple sources in one figure, if feasable (low priority)
* Configuration class (where to put it...)
* If bands.yml not found &rarr; try to use those specified in config, if not &rarr; glob
* Check if clipping and normalisation are still performed multiple times. Can this be reduced?
* Command-line mode with \* in image name for multiple &rarr; maybe this can replace the current `-m` keyword, which could then become mask.
* Put parameters for different stiff modes in separate config file? Should not have to be read every time stretch is run though...
* Check relative paths for config files using ./ when running from command-line, especially in Linux.
* In the readme example image maybe put all three original channels on top.
* Multiprocessing is not using filename template...
