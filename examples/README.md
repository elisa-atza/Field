# Worked example: field `S`, 2021

This example has one field and two flight dates: 2021-06-28 and 2021-07-03. The reference date is 2021-07-03. The other date gets its polygons from the reference date.

This folder is the default `data_root`, so you do not need to set anything to run on it.

The two images are about 750 MB each, which is too large for git. Download them from this repository's Releases page and put them in `S_2021_crops/`. All other files are already here.

Other dates and fields are available in the 4TU datasets. See the [main README](../README.md).

## Layout

```
S_2021_crops/
  Crop_<date>.npy                    input images, from the Releases page

S_2021/
  S_Color_bounds.csv                 hue bounds for each date (Step8)
  Markers/
    Markers_S_<date>.npy             markers clicked in Step2
    Sorted_markers_S_<date>.npy      the same markers, in matching order (Step6)
  Quadrants/
    S_2021_07_03_<q>_tops.npy        ridge ends selected in Step3,
    S_2021_07_03_<q>_bots.npy          for quadrants 1-4
  Grids/
    grid_<q>_2021_07_03.npy          grids detected in Step4
    Correct_grid_<q>.npy             grids accepted in Step5
  Plattegronds/
    S_named_plattegrond.npy          trial layout: variety and batch of each lot
    S_lotnum_plattegrond.npy         trial layout: lot numbers
  Polygons/
    S_2021_07_03_polygons.npy        reference-date polygons (Step7_1)
    S_2021_06_28_polygons.npy        polygons moved to the other date (Step7_2)
```

`canopy` adds these folders when you run it:

```
  Raw_canopy_data/
    Raw_canopy_ridge_<date>_S.csv    canopy of each ridge, one file for each date
  Canopy data/
    <YYYYMMDD>_S_2021.csv            one row for each ridge, one column for each date
```

## Running it

```sh
python cli.py run --status    # show which stages are done
python cli.py canopy          # measure both dates and write the table
```

`canopy` is automatic. You can also repeat each interactive stage by itself:

```sh
python cli.py markers --date 2021_06_28       # right-click the markers, middle-click to save
python cli.py boundaries                      # run once for each quadrant 1-4
python cli.py correct-grid                    # run once for each grid; middle-click even with no corrections
python cli.py sort-markers --date 2021_06_28  # confirm that the markers match
```

`plattegronds` needs the trial layout spreadsheet, which is not included. Its output files are included. `build-grid` uses far more CPU and memory than the other stages.
