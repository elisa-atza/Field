# Field: canopy measurement from drone images

This repository contains a classical computer-vision pipeline. It takes RGB drone images of a potato trial field and measures the canopy of each ridge on each flight date. The result is one table, with one row for each ridge and one column for each date.

The pipeline was first written in 2019 and was last used on the 2021 season.

## Start here

[`showcase/plant_pipeline_showcase.ipynb`](showcase/plant_pipeline_showcase.ipynb) shows how the pipeline works on real images, with all figures saved. You can read it without running anything.

## What is in this repository

- **The pipeline:** `cli.py`, the `Step*.py` modules, `field_cv.py` and `field_paths.py`. This is general process code.
- **The showcase:** `showcase/` illustrates the pipeline on two flight dates.
- **Example data:** `examples/` holds the small files for those two dates. The two images are on this repository's Releases page, because they are too large for git. See [`examples/README.md`](examples/README.md).

Other fields and images are available in the 4TU dataset:
Atza, Elisa; Budko, Neil (2025): Data underlying the publication: Predicting potato plant vigor from the seed tuber properties. Version 3. 4TU.ResearchData. dataset. [https://doi.org/10.4121/3a97fa0c-8c7d-451a-b8fe-d521f1cec55e.v3](https://doi.org/10.4121/3a97fa0c-8c7d-451a-b8fe-d521f1cec55e.v3)

You can use the code on those images or on your own, with some changes. See [Using other data](#using-other-data).

## How the pipeline works

1. Crop each image to the trial area.
2. Find the red markers in the field.
3. Select the ends of the ridges on one reference date.
4. Find the gaps between plants along each ridge. These gaps are the boundaries between lots.
5. Correct the result by hand where necessary.
6. Put the markers of each date in the same order.
7. Build one polygon for each lot, and move the polygons onto the other dates with the markers.
8. Select the plant colors for each date.
9. Count the canopy pixels in each ridge and convert the count to cm².

A ridge has 6 plants, and single plants cannot be separated in the images. The pipeline therefore divides the total canopy of a ridge by 6, which gives the average canopy per plant. Spatial trends across the field were handled later with the R package SpATS [https://CRAN.R-project.org/package=SpATS](https://CRAN.R-project.org/package=SpATS), outside this pipeline.

## Setup

The pinned package versions need Python 3.10.

```sh
python3.10 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
```

GDAL needs the system library at a matching version. The `crop` command and `.tif` images need it:

```sh
sudo apt install gdal-bin libgdal-dev
pip install GDAL==$(gdal-config --version)
```

To run only the showcase, use `showcase/requirements.txt` instead. It does not need GDAL.

## Running it

`cli.py` runs the pipeline, and each stage is one command. The year, fields, dates and folders are all set in [`pipeline_config.toml`](pipeline_config.toml).

```sh
python cli.py --list            # list every command
python cli.py markers --help    # show the options of one command
python cli.py run --status      # show which stages are done
python cli.py run               # do the remaining stages, in order
```

Stages pass their results to each other as files on disk, so you can run any stage again by itself. `run` skips each stage whose output already exists, so you can stop it and continue later. To redo a stage, delete its output file or run its own command. `run` does not crop, so run `crop` first if your images need it.

| Command | Step | What it does | Interactive |
|---|---|---|---|
| `crop` | `Step1` | Crops a full image to the trial area | Type the crop bounds |
| `markers` | `Step2` | Finds the red markers | Right-click the markers to keep |
| `boundaries` | `Step3` | Sets the ridge ends for each quadrant | Right-click the corners |
| `build-grid` | `Step4` | Finds the gaps along each ridge and fits them to a regular grid | No |
| `check-grids` | `Step5` | Shows all grids on the reference image | Look only |
| `correct-grid` | `Step5` | Moves individual grid points | Click points |
| `sort-markers` | `Step6` | Puts the markers of two dates in matching order | Confirm or swap |
| `plattegronds` | `Step7_0` | Reads the trial layout spreadsheet | No |
| `make-polygons` | `Step7_1` | Builds one 13-point polygon for each lot | No |
| `transform-polygons` | `Step7_2` | Moves the polygons from the reference date to another date | No |
| `timelapse` | `Step7_3` | Makes GIFs across dates, for documentation only | No |
| `color-bounds` | `Step8` | Selects the hue bounds for each date | Partly |
| `canopy` | `Step9` | Measures each ridge and writes the tables | No |
| `check-canopy` | `Step9` | Draws the canopy of each ridge next to the image | Look only |

`build-grid` starts one process for each ridge over the whole image, so it uses far more CPU and memory than the other stages.

The interactive stages need an interactive matplotlib backend. Right-click adds a point, shift + left-click removes one, and middle-click finishes.

## Using other data

Set `data_root` in `pipeline_config.toml` to your own folder, or set it without editing the file:

```sh
export FIELD_DATA_ROOT=/path/to/your/data
```

For each date in the config, the pipeline uses the image whose file name contains that date. All names of output files are set in `field_paths.py`.

The code is built for this trial layout, so other layouts need changes:

- `Step4` expects 30 lots per ridge and 4 ridges per lot.
- `Step7_1` and `Step9` expect 720 lots and 2,880 ridges.
- `Step9` expects 6 plants per ridge.
- `field_cv.py` holds the ridge spacing for each field. This value converts pixels to cm².
- `Step7_1` holds the table that connects each quadrant to the trial layout.

Fields are named by initial: `M`, `V` and `S`. Only `S` is included here.

## Known issues

This is research code, published as a record of the method and not as a library.

- **The code runs one year at a time.**
- **The vegetation indices are calculated on 8-bit values.** A value such as r² wraps around when it is above 255. The thresholds were set with this behavior, so it was kept.
- **The canopy mask drops foliage in shadow.** Dark leaves can fail the color test, so canopy can be too low on dates with hard shadows.
- **Some code is unused or repeated.** `Step4` passes lateral boundaries that it never reads, the layout helpers are repeated in `Step7_3`, and some imports are used only by commented-out code. Those comments are kept so that the blocks can be turned on again.
