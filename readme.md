KoDI analysis
=============

Code for analyzing penumbral imaging, including PCIS, KoDI, SRTE, and XRIS aka PIX aka x-ray KoDI.

# Installation

This codebase has some PyPI dependencies.
I think they’re all in the requirements.txt, so you should be all set if you do
~~~~
 pip install -r requirements.txt
~~~~

It also requires Peter Heuer’s [cr39py](https://github.com/pheuer/CR39py) library, which is not on PyPI as of writing,
so for that just install it like
~~~~
 pip install git+https://github.com/pheuer/CR39py.git
~~~~

# File structure

The file structure goes like this:
- `src` – various Python scripts
    - `main` – Java files for some of the more time-consuming operations
- `input` – input files (including the files `shot_info.csv` and `LOS_info.csv`)
  - `scans` – CR-39 and image plate scan files (put your scans here!)
  - `tables` – stopping power and cross section tables
- `results` – outputs from the analysis (including the file `summary.csv` and varius logs)
  - `plots` – pretty pictures of the reconstructions (see your results here!)
  - `data` – HDF5 files of the reconstructions
- `tmp` – dump for temporary files used to communicate between Java and Python
- `out` – compiled Java files

# How to use

The typical workflow for doing 2D reconstructions looks like this:
1. Drop your `.cpsa` or `.h5` scan files into the `input/scans` directory.
   If you have `.hdf` files, you’ll need to convert them to `.h5` using NASA’s `h4toh5` tool.
2. Edit the `shot_info.csv` and `LOS_info.csv` files in the `input` directory to include the shots and lines of sight you want to analyze.
   The `shot_info.csv` file should have "shot", "standoff", "magnification", "aperture radius", "aperture spacing", and "aperture arrangement" columns,
   while `LOS_info.csv` should have "shot", "LOS", and "filtering" columns.
   You may instead put "standoff", "magnification", "aperture radius", "aperture spacing", and "aperture arrangement" in `LOS_info.csv` if it varies from LOS to LOS,
   for example if you have different magnifications on different lines of sight.
   - The "shot" column should contain the shot number (or any other identifying filename substring) of each implosion you have data for.
   - The "standoff" is the distance from the aperture(s) to TCC in centimeters.
   - The "magnification" is the ratio of the TCC-to-detector distance to the TCC-to-aperture distance
     Note that this is the *radiography magnification* and not the *pinhole magnification* (shown on the SRF) which is one less.
   - The "aperture radius" is the radius of each circular aperture in micrometers.
   - The "aperture spacing" is the distance from each aperture to its nearest neighbor in micrometers.
     When there is only one aperture, this number doesn’t matter and may be set to 0.
   - The "aperture arrangement" is the name of the shape of the aperture array.
     Supported options are "single" for a single aperture, "hex" for an equilateral hexagonal grid like KoDI uses,
     "srte" for the particular skew hexagonal grid that SR-TE has, and "square" for a square grid like DIXI has.
   - The "LOS" is the name of the line of sight on which the image was taken.
     Supported options include "TIM1", "TIM2", "TIM3", "TIM4", "TIM5", "TIM6", and "SRTE".
     When the TIM is "SRTE", the "aperture radius", "aperture spacing" and "aperture arrangement" are all overwritten with SR-TE’s fixed aperture array specifications
     (so you can put those three in "shot_info.csv" even when a shot uses both KoDI and SR-TE).
   - The "filtering" is a string specifying the makeup of the detector stack.
     It should be given as a list of thicknesses (in micrometers) and materials, from TCC outward.
     CR39 is denoted "[]" and image plates are denoted "|".
     Split filter layers are denoted with "/".
     For example, "15Ta/50Al [] 200Al | 200Al |" represents a common KoDI/XRIS setup with a split tantalum/aluminum filter on the front.
     The spaces are optional.
3. If you are analyzing TIM-based x-ray data, you must run `src/split_ip_scans.py`
   to convert the multiple-scan-files to individual image plate scans.
4. Run `python src/reconstruct_2d.py shot_number [--show] [--skip] [--proton] [--MCMC]` with the shots you want to analyze passed as an argument.
   This looks for the `.txt` files in the `input/scans/` directory and performs the reconstruction.
   - The first argument should be a comma-separated list of shot numbers.
     You can also specify specific lines of sight to analyze; for example, `95520tim4,95521srte`
   - The `--show` flag causes it to show each plot it generates and wait for the user to close it
     before moving on.  By default, it will only save them to the `results/plots` directory without showing them.
   - The `--skip` flag causes it to reload the previous reconstructions and simply update the plots,
     skipping the actual reconstruction algorithm.
   - The `--proton` flag tell it that you’re analyzing charged particles that don’t follow a knock-on deuteron spectrum.
     This is important so that it doesn’t try to group the signal into < 6 MeV and > 9 MeV parts by their diameters.
   - The `--MCMC` flag tells it to do uncertainty quantification.  It will take a lot longer and generate a few extra plots.
     If you're running on a machine with GPUs, I recommend trying it with the `--GPU` option as well, which might make it faster.
5. - The `--charging` flag tells it to use the aperture charging model to try to compensate for distortions in charged particle data 
     (the validity of this model is questionable, so it's off by default).
6. The only input you need to provide once it starts running is the data region (and that’s only if you use `--show`).
   When it asks you to "select" a "region", simply click on the plot to draw a polygon enclosing the good signal region,
   and ideally excluding any scratches or fiducials.
   You may right-click at any time during this process to un-place a point.
6. It automatically outputs a bunch of plots, HDF5 files, and log messages.

For 3D reconstruction, run `src/reconstruct_3d.py shot_number (deuteron|xray) [--skip]`,
and it will automaticly run on the reconstructed 2d images.
Set the twoth argument to `deuteron` if it's deuteron images and `xray` if it's x-ray images.
The shot number argument on this one does not support commas.

For electron temperature inference, run `python src/calculate_temperature.py shot_numbers lines_of_sight [--show]`.
- `shot_numbers` is a comma-separated list of shot numbers.
- `lines_of_sight` is a comma-separated list of lines of sight; for example, `TIM2,TIM4,TIM5,SRTE`.
- The `--show` flag causes it to show each plot and wait for the user before moving on.
It should automatically find and load your previous 2D reconstructions for each shot and each line of sight.
It's okay if some of the data is missing (like if one TIM failed on one of the shots).
The plot limits aren't as smart, so you may want to adjust those manually.

All output files follow the naming convention `results/subdirectory/shotnumber/[tim-]quantity[-index]-domain[-operation].file_extension`.
The shot number is an integer prepended by `synth` if based on synthetic data.
The quantity is one of:
- `morphology` for combined mass density and neutron source
- `deuteron` for deuteron emission/detection
- `xray` for x-ray emission/detection

the index is three digits representing:
1. the index of the detector (0 is the IP or CR-39 closest to TCC, and so on)
2. the index of the filtering region (0 is the region with the weakest filtering, and so on)
3. the index of the diameter cut (0 is the largest-diameter and lowest-energy cut, and so on)

the domain is one of:
- `distribution` for 3D reconstructed (or synthetic) quantities
- `penumbra` for 2D measured penumbral images
- `source` for 2D reconstructed (or synthetic) emission images

and the operation is up to one of:
- `section` for 2D cross-sections of 3D quantities
- `lineout` for 1D cartesian lineouts of multidimensional quantities
- `profile` for 1D polar lineouts of multidimensional quantities
- `residual` for comparisons of measured and reconstructed images

Finally, there's a meager test suite ('suite' isn't even really accurate since it's just one) that you can run with `python -m pytest src/test.py`.
