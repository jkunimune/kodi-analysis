KODI analysis
=============

Code for analyzing penumbral imaging, and in particular knock-on deuteron & x-ray imaging.

# File structure

The file structure goes like this:
- `src` – various Python scripts
    - `main` – Java files for some of the more time-consuming operations
- `input` – input files (including the files `shots.csv` and `tim_info.txt`)
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
2. If you are analyzing x-ray data, you must run `src/split_ip_scans.py`
   to convert the multiple-scan-files to individual image plate scans.
3. Edit the `shot_list.csv` file in the `input` directory to include the shots you want to analyze.
   It should have "shot", "standoff", "magnification", "aperture radius", and "aperture spacing" columns.
4. Edit the `tim_info.txt` file in the `input` directory to include the lines of sight fielded and the filtering used on each one.
   The filtering should be specified as stacks of thicknesses (in microns) and materials, from TCC out,
   with "[]" standing for pieces of CR39 and "|" standing for image plates.
   For example, "2: 15Ta [] 200Al | 200Al |" for a typical KoDI/XRIS setup on TIM2.
5. Run `src/reconstruct_2d.py shot_number [--show] [--skip] [--proton]` with the shots you want to analyze passed as an argument.
   This looks for the `.txt` files in the `input/scans/` directory and performs the reconstruction.
   1. The first argument should be a comma-separated list of shot numbers.
      You can also specify specific TIMs to analyze; for example, `95520tim4,95521tim2`
   2. The `--show` flag causes it to show each plot it generates and wait for the user to close it
      before moving on.  By default, it will only save them to the `results/plots` directory without showing them.
   3. The `--skip` flag causes it to reload the previous reconstructions and simply update the plots,
      skipping the actual reconstruction algorithm.
   4. The `--proton` flag tell it that you’re analyzing charged particles that don’t follow a knock-on deuteron spectrum.
      This is important so that it doesn’t try to group the signal into < 6 MeV and > 9 MeV parts by their diameters.
6. The only input you need to provide once it starts running is the data region (and that’s only if you use `--show`).
   When it asks you to "select" a "region", simply click on the plot to draw a polygon enclosing the good signal region,
   and ideally excluding any scratches or fiducials.
   You may right-click at any time during this process to un-place a point.
7. It automatically outputs a bunch of plots, HDF5 files, and log messages.

For 3D reconstruction, run `src/reconstruct_3d.py shot_number [--skip]`,
and it will automaticly run on the reconstructed 2d images.
The shot number argument on this one does not support commas.
I'll get to that later maybe probably

All output files follow the naming convention `results/subdirectory/shotnumber[-tim]-quantity-coordinates[-operation].file_extension`.
The shot number is an integer prepended by `synth` if based on synthetic data.
The quantity is one of:
- `morphology` for combined mass density and neutron source
- `deuteron` for combined deuteron sources
- `deuteron-[i]` for deuteron sources on the i-th detector
- `xray-[i]` for x-ray sources on the i-th detector

the coordinates are one of:
- `distribution` for 3D reconstructed (or synthetic) quantities
- `penumbra` for 2D measured penumbral images
- `source` for 2D reconstructed (or synthetic) emission images

and the operation is up to one of:
- `section` for 2D cross-sections of 3D quantities
- `lineout` for 1D cartesian lineouts of multidimensional quantities
- `profile` for 1D polar lineouts of multidimensional quantities
- `residual` for comparisons of measured and reconstructed images

# Dependencies

This codebase has some PyPI dependencies you can figure out on your own.
It also requires Peter Heuer’s [cr39py](https://github.com/pheuer/CR39py) library, which is not on PyPI as of writing,
so for that just install it like
~~~~
 pip install git+https://github.com/pheuer/CR39py.git
~~~~
