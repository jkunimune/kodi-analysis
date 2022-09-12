KODI analysis
=============

Code for analyzing penumbral imaging, and in particular knock-on deuteron & x-ray imaging.

The file structure goes like this:
- `src` – various Python scripts
    - `main` – Java files for some of the more time-consuming operations
- `data` – input files (including the file `shots.csv`)
  - `scans` – CR-39 and image plate scan files
  - `tables` – stopping power and cross section tables
- `results` – outputs from the analysis (including logs and tables and stuff)
  - `plots` – pretty pictures of the reconstructions
  - `data` – HDF files of the reconstructions
- `tmp` – dump for temporary files used to communicate between Java and Python

The typical workflow for doing 2D reconstructions looks like this:
1. Get `.cpsa` files from the LLE or MIT etch/scan lab.
2. Open them in AnalyzeCR39 and export the track data to a `.txt` file. (in the future I will have this automatically read cpsa)
3. Create a `shot_list.csv` file in the root kodi-analysis directory that lists the shots and lines of sight to analyze.
   It should have "Shot number", "TIM", "L1", "Magnification", "Aperture Radius",
   "Aperture Separation", "Rotation", "Filtering", and "Etch time" columns.
4. Run `src/reconstruct_2d.py shot_number [--skip]` with the shots you want to analyze passed as an argument.
   This looks for the `.txt` files in the `scans/` directory and performs the reconstruction.
   1. The first argument should be a comma-separated list of shot numbers.
      You can also specify specific TIMs to analyze; for example, "95520tim4,95521tim2"
   2. You can also add "--skip" as an argument to tell it to reload the previous reconstructions and simply update the plots.
5. It automatically outputs a bunch of plots, HDF5 files, and log messages.

For 3D reconstruction, run `src/reconstruct_3d.py shot_number [--skip]`,
and it will automaticly run on the reconstructed 2d images.
The shot number argument on this one does not support commas.
I'll get to that later maybe probably

All output files follow the naming convention `results/subdirectory/shotnumber[-tim]-quantity-coordinates[-operation].file_extension`.
The shot number is an integer prepended by `synth` if based on synthetic data.
The quantity is one of:
- `morphology` for combined mass density and neutron source
- `deuteron` for combined deuteron sources
- `deuteron[i]` for deuteron sources in the i-th energy bin
- `xray[i]` for x-ray sources on the i-th detector

the coordinates are one of:
- `distribution` for 3D reconstructed (or synthetic) quantities
- `penumbra` for 2D measured penumbral images
- `source` for 2D reconstructed (or synthetic) emission images

and the operation is up to one of:
- `section` for 2D cross-sections of 3D quantities
- `lineout` for 1D cartesian lineouts of multidimensional quantities
- `profile` for 1D polar lineouts of multidimensional quantities
- `residual` for comparisons of measured and reconstructed images
