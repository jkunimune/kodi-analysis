KODI analysis
=============

The typical workflow for doing 2D reconstructions looks like this:
1. Get `.cpsa` files from the LLE or MIT etch/scan lab.
2. Open them in AnalyzeCR39 and export the track data to a `.txt` file.
3. Create a `shot_list.csv` file in the root kodi-analysis directory that lists the shots and lines of sight to analyze.
   It should have "Shot number", "TIM", "L1", "Magnification", "Aperture Radius",
   "Aperture Separation", "Rotation", "Filtering", and "Etch time" columns.
4. Run `src/main.py`, which reads `shot_list.csv`, looks for the `.txt` files in the `scans/` directory,
   and calls functions from `src/reconstruct_2d.py` to perform the reconstruction.
5. It automatically outputs a bunch of plots, HDF5 files, and log messages.

For 3D reconstruction, just run `src/reconstruct_3d.py`,
which will call the java program in the `src/main/` folder on synthetic data.
