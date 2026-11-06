"""Locations of the image and result folders used by every pipeline step.

All of it comes from pipeline_config.toml at the repository root (or the file
FIELD_CONFIG names) -- edit that file, not this one. The images are too large for this repository.

Folder layout for a field F (names configurable in the TOML):
    <images_dir>/                   flight images, one per date (.tif or .npy)
    <results_dir>/Markers/          Step2, Step6
    <results_dir>/Quadrants/        Step3
    <results_dir>/Grids/            Step4, Step5
    <results_dir>/Plattegronds/     Step7_0
    <results_dir>/Polygons/         Step7_1, Step7_2
    <results_dir>/Raw_canopy_data/ and <results_dir>/Canopy data/   Step9
"""

import os
from datetime import datetime

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

# FIELD_CONFIG points at a different config file, e.g. to keep one per setup
CONFIG_FILE = os.environ.get(
    'FIELD_CONFIG', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline_config.toml'))

with open(CONFIG_FILE, 'rb') as _f:
    CONFIG = tomllib.load(_f)

YEAR = str(CONFIG['year'])
SEGMENTATION = CONFIG.get('segmentation', 'fixed')
if SEGMENTATION not in ('fixed', 'per_date'):
    raise ValueError(f"segmentation in {CONFIG_FILE} must be 'fixed' or 'per_date', not {SEGMENTATION!r}")
# a relative data_root is taken from the config file's folder, not from
# wherever the command is run
DATA_ROOT = os.environ.get('FIELD_DATA_ROOT') or os.path.join(
    os.path.dirname(os.path.abspath(CONFIG_FILE)), CONFIG['data_root'])
FIELDS = tuple(CONFIG['fields'])


def _resolve(template, field=None):
    path = template.format(field=field, year=YEAR)
    return os.path.join(DATA_ROOT, path)


def field_config(field):
    if field not in CONFIG['fields']:
        raise KeyError(f'field {field!r} is not configured in {CONFIG_FILE} '
                       f'(configured: {", ".join(FIELDS)})')
    return CONFIG['fields'][field]


def dates(field):
    return list(field_config(field)['dates'])


def reference_date(field):
    return field_config(field)['reference_date']


def images_dir(field):
    return _resolve(CONFIG['images_dir'], field)


def results_dir(field, *sub):
    """<results_dir>/<sub...> for a field; creates the folder if missing."""
    path = os.path.join(_resolve(CONFIG['results_dir'], field), *sub)
    os.makedirs(path, exist_ok=True)
    return path


def plattegronden_xlsx():
    return _resolve(CONFIG['plattegronden_xlsx'])


def image_path(field, date):
    """The image in the field's images_dir whose filename contains `date`."""
    folder = images_dir(field)
    matches = sorted(x for x in os.listdir(folder)
                     if date in x and x.lower().endswith(('.tif', '.tiff', '.npy')))
    if not matches:
        raise FileNotFoundError(f'no image for date {date!r} in {folder}')
    if len(matches) > 1:
        raise FileExistsError(f'more than one image for date {date!r} in {folder}: {matches}')
    return os.path.join(folder, matches[0])


def markers_path(field, date):
    """Markers clicked in Step2."""
    return os.path.join(results_dir(field, 'Markers'), f'Markers_{field}_{date}.npy')


def sorted_markers_path(field, date):
    """Markers put in reference order by Step6."""
    return os.path.join(results_dir(field, 'Markers'), f'Sorted_markers_{field}_{date}.npy')


def polygons_path(field, date):
    """Per-lot polygons for one date: built by Step7_1 on the reference date,
    warped onto the other dates by Step7_2."""
    return os.path.join(results_dir(field, 'Polygons'), f'{field}_{date}_polygons.npy')


def plattegrond_paths(field):
    """(named, lot-number) plattegrond files written by Step7_0."""
    folder = results_dir(field, 'Plattegronds')
    return (os.path.join(folder, f'{field}_named_plattegrond.npy'),
            os.path.join(folder, f'{field}_lotnum_plattegrond.npy'))


def quadrant_path(field, quadrant, kind):
    """Ridge end points Step3 selected on the reference date; kind is
    'tops', 'bots' or 'mids'."""
    return os.path.join(results_dir(field, 'Quadrants'),
                        f'{field}_{reference_date(field)}_{quadrant}_{kind}.npy')


def grid_path(field, index):
    """Grid Step4 detected for one quadrant (index '1', or '1_1'/'1_2' for a
    split quadrant)."""
    return os.path.join(results_dir(field, 'Grids'), f'grid_{index}_{reference_date(field)}.npy')


def corrected_grid_path(field, index):
    """The same grid as accepted in Step5; Step7_1 reads only these."""
    return os.path.join(results_dir(field, 'Grids'), f'Correct_grid_{index}.npy')


def raw_canopy_path(field, date):
    """Per-ridge canopy table for one date, written by Step9."""
    return os.path.join(results_dir(field, 'Raw_canopy_data'), f'Raw_canopy_ridge_{date}_{field}.csv')


def color_bounds_path(field):
    """Per-date hue bounds chosen by Step8."""
    return os.path.join(results_dir(field), f'{field}_Color_bounds.csv')


def canopy_table_path(field, day=None):
    """Step9's one-column-per-date table, dated by the day it was written."""
    day = day or datetime.today().strftime('%Y%m%d')
    return os.path.join(results_dir(field, 'Canopy data'), f'{day}_{field}_{YEAR}.csv')


def latest_canopy_table(field):
    """Most recently dated table, or None if there is none."""
    folder = results_dir(field, 'Canopy data')
    ending = f'_{field}_{YEAR}.csv'
    tables = sorted(x for x in os.listdir(folder) if x.endswith(ending) and x[:8].isdigit())
    return os.path.join(folder, tables[-1]) if tables else None
