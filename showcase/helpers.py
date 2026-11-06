"""Plotting and loading helpers for the showcase notebook.

Import this module first: it selects the showcase configuration and limits the
numerical libraries to one thread before they load. The measurement logic stays
in the pipeline's ``Step*.py`` modules; this file only prepares inputs and draws.
"""

import contextlib
import io
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

SHOWCASE_DIR = Path(__file__).resolve().parent
REPO_DIR = SHOWCASE_DIR.parent
EXAMPLES_DIR = REPO_DIR / "examples"
SOURCE_RESULTS = EXAMPLES_DIR / "S_2021"

os.environ["FIELD_CONFIG"] = str(SHOWCASE_DIR / "showcase_config.toml")
# One thread per library keeps the run light on a shared laptop.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"
sys.path.insert(0, str(REPO_DIR))

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from field_cv import compute_ridge_lateral_bdry, extractHueSat, poly_cut  # noqa: E402
from field_paths import (  # noqa: E402
    color_bounds_path, corrected_grid_path, image_path, markers_path,
    plattegrond_paths, polygons_path, quadrant_path,
)
from Step4_Build_grid import (  # noqa: E402
    filterGaps_with_avg_periodic, get_avg_periodic, getGaps,
)
from Step7_1_Create_polygons_from_grids import (  # noqa: E402
    load_platts_associations, produce_polygons,
)
from Step7_2_Transform_polygons import aff_transform  # noqa: E402
from Step8_Select_day_spec_unbiased_color_bounds import (  # noqa: E402
    otsu_segmentation, segmentation_image_color_index,
)
from Step9_Save_canopy_to_file import get_ridge_from_poly  # noqa: E402

cv2.setNumThreads(1)

FIELD = "S"
REFERENCE_DATE = "2021_07_03"
OTHER_DATE = "2021_06_28"
DATES = (OTHER_DATE, REFERENCE_DATE)
QUADRANTS = ("1", "2", "3", "4")

# Bands of Step4.main's "all non-green set to 0" filter, [hue low, hue high,
# saturation low, saturation high], and its three morphology passes as
# (kernel shape, iterations). Step4 keeps them inline, in main().
GRID_HUE_SAT_BOUNDS = [35, 150, 70, 150]
GRID_DILATE_1 = ((4, 4), 5)
GRID_ERODE = ((4, 3), 4)
GRID_DILATE_2 = ((2, 2), 2)
# Pixels kept around a ridge when cropping for the single-ridge demonstration.
RIDGE_WINDOW_MARGIN_PX = 300

Polygons = Dict[int, np.ndarray]


def apply_figure_style() -> None:
    """Set the figure style shared by every plot in the notebook."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "image.cmap": "viridis",
    })


def prepare_workspace() -> None:
    """Copy the saved manual-step inputs into showcase/outputs/, keeping only
    this notebook's dates in the colour-bounds table."""
    for date in DATES:
        name = os.path.basename(markers_path(FIELD, date))
        shutil.copy(SOURCE_RESULTS / "Markers" / name, markers_path(FIELD, date))
    for quadrant in QUADRANTS:
        for kind in ("tops", "bots"):
            target = quadrant_path(FIELD, quadrant, kind)
            shutil.copy(SOURCE_RESULTS / "Quadrants" / os.path.basename(target), target)
        target = corrected_grid_path(FIELD, quadrant)
        shutil.copy(SOURCE_RESULTS / "Grids" / os.path.basename(target), target)
    for target in plattegrond_paths(FIELD):
        shutil.copy(SOURCE_RESULTS / "Plattegronds" / os.path.basename(target), target)

    bounds = pd.read_csv(SOURCE_RESULTS / f"{FIELD}_Color_bounds.csv", index_col=0)
    bounds[bounds["Date"].isin(DATES)].to_csv(color_bounds_path(FIELD))


def open_image(date: str) -> np.ndarray:
    """Memory-map one date's image: only the pixels actually used are read."""
    return np.load(image_path(FIELD, date), mmap_mode="r")


def hue_bounds(date: str) -> Tuple[float, float]:
    """Lower and upper hue bound stored for a date (OpenCV hue units, 0-179)."""
    table = pd.read_csv(color_bounds_path(FIELD))
    row = table[table["Date"] == date].iloc[0]
    return float(row["Lower_hue"]), float(row["Upper_hue"])


def load_polygons(date: str) -> Polygons:
    """Load the per-lot polygons of a date."""
    return np.load(polygons_path(FIELD, date), allow_pickle=True).item()


def build_reference_polygons() -> Polygons:
    """Build the reference-date polygons from the accepted grids, as Step7_1 does.

    Step7_1.main also renders a full-resolution check image; only its polygon
    loop is repeated here, around the same two pipeline functions.
    """
    associations, _, _ = load_platts_associations(FIELD)
    polygons: Polygons = {}
    for quadrant in QUADRANTS:
        grid = np.fliplr(np.load(corrected_grid_path(FIELD, quadrant)))
        top_boundary = compute_ridge_lateral_bdry(grid[0, :, :])
        bottom_boundary = compute_ridge_lateral_bdry(grid[-1, :, :])
        with contextlib.redirect_stdout(io.StringIO()):  # it prints array shapes
            polygons = produce_polygons(polygons, grid, top_boundary, bottom_boundary,
                                        associations[FIELD][quadrant])
    np.save(polygons_path(FIELD, REFERENCE_DATE), polygons)
    return polygons


def similarity_parameters(reference_pts: np.ndarray,
                          target_pts: np.ndarray) -> Dict[str, float]:
    """Read translation, rotation and scale off the fit made by ``aff_transform``,
    by applying it to the origin and a unit vector."""
    probe = {0: np.array([[0.0, 0.0], [1.0, 0.0]])}
    origin, unit = aff_transform(probe, reference_pts, target_pts)[0]
    a, b = unit - origin
    return {
        "translation x (px)": float(origin[0]),
        "translation y (px)": float(origin[1]),
        "rotation (degrees)": float(np.degrees(np.arctan2(b, a))),
        "scale": float(np.hypot(a, b)),
    }


def window_around(points: np.ndarray, margin: int) -> Tuple[int, int, int, int]:
    """Bounding box ``(x0, y0, x1, y1)`` of points, widened by margin, clipped at 0."""
    x0, y0 = np.maximum(points.min(axis=0).astype(int) - margin, 0)
    x1, y1 = points.max(axis=0).astype(int) + margin
    return int(x0), int(y0), int(x1), int(y1)


def lot_crop(image: np.ndarray, polygon: np.ndarray,
             margin: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """Crop a lot with a margin; returns the crop and the polygon in crop coordinates."""
    x0, y0, x1, y1 = window_around(polygon, margin)
    return np.asarray(image[y0:y1, x0:x1]), polygon - np.array([x0, y0])


def segmentation_stages(image: np.ndarray, polygon: np.ndarray,
                        date: str) -> Dict[str, np.ndarray]:
    """Segment a lot exactly as Step 9 does, one ridge at a time, and paste the
    ridges back together. Each mask is normalised on its own ridge, so a mask
    made on the whole lot would differ from the measured one."""
    crop, outline = lot_crop(image, polygon)
    x0, y0 = (np.asarray(polygon) - outline).astype(int)[0]
    low, high = hue_bounds(date)
    after_hue, after_mask = np.zeros_like(crop), np.zeros_like(crop)
    points = np.array(polygon, dtype=np.int32).reshape(-1, 2)
    for ridge in get_ridge_from_poly(points):
        ridge_points = np.array(ridge, dtype=np.int32).reshape(-1, 2)
        ridge_crop, left, top = poly_cut(image, ridge_points)
        masked, _ = segmentation_image_color_index(ridge_crop, date, FIELD)
        for target, source in ((after_hue, otsu_segmentation(ridge_crop, low, high)),
                               (after_mask, masked)):
            height, width = source.shape[:2]
            window = target[top - y0:top - y0 + height, left - x0:left - x0 + width]
            kept = np.any(source != 0, axis=-1)
            window[kept] = source[kept]
    return {"original": crop, "after hue band": after_hue, "canopy mask": after_mask}


def show_image(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    """Draw an image panel without axes; panels are photographs, not plots."""
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")


def ridge_demo(image: np.ndarray, ridge: int, quadrant: str = "1") -> Dict[str, np.ndarray]:
    """Run Step4's gap detection for one ridge on a window of the image.

    Step4 runs every ridge in its own process over the whole image; one ridge
    in one process is enough to show the method. Returns the colour and binary
    windows and the detected, filtered and accepted lot boundaries in window
    coordinates.
    """
    tops = np.load(quadrant_path(FIELD, quadrant, "tops"))
    bottoms = np.load(quadrant_path(FIELD, quadrant, "bots"))
    tops = tops[~np.all(tops == 0, axis=1)].astype(float)
    bottoms = bottoms[~np.all(bottoms == 0, axis=1)].astype(float)

    ends = np.vstack((tops[ridge - 1], bottoms[ridge - 1]))
    x0, y0, x1, y1 = window_around(ends, RIDGE_WINDOW_MARGIN_PX)
    binary = extractHueSat(np.asarray(image[y0:y1, x0:x1]), GRID_HUE_SAT_BOUNDS)
    binary[binary > 0] = 255
    for operation, (shape, iterations) in ((cv2.dilate, GRID_DILATE_1),
                                           (cv2.erode, GRID_ERODE),
                                           (cv2.dilate, GRID_DILATE_2)):
        binary = operation(binary, np.ones(shape, np.uint8), iterations=iterations)

    offset = np.array([x0, y0], dtype=float)
    figure = plt.figure()  # getGaps draws its ridge line on the current figure
    gaps = getGaps(binary, tops - offset, bottoms - offset, ridge)
    plt.close(figure)
    periodic = get_avg_periodic(tops - offset, bottoms - offset, FIELD)
    filtered = filterGaps_with_avg_periodic(gaps, periodic, ridge)
    accepted = np.load(corrected_grid_path(FIELD, quadrant))[:, ridge - 1] - offset
    return {"rgb": np.asarray(image[y0:y1, x0:x1]), "binary": binary,
            "detected": gaps[:, :2], "filtered": filtered, "accepted": accepted}


def accepted_gaps(quadrant: str = "1") -> np.ndarray:
    """All lot boundaries accepted in Step 5 for a quadrant, as (x, y) points."""
    return np.load(corrected_grid_path(FIELD, quadrant)).reshape(-1, 2).astype(float)


def pipeline_diagram(steps: List[Tuple[str, str, str]]) -> plt.Figure:
    """Draw the pipeline as a column of boxes; each step is (name, output, kind),
    where kind is "manual", "auto" or "input" and sets the box colour."""
    colours = {"manual": "#f4d9b0", "auto": "#c9e4d6", "input": "#e3e3e3"}
    figure, ax = plt.subplots(figsize=(9, 0.62 * len(steps)))
    for row, (name, output, kind) in enumerate(steps):
        y = len(steps) - 1 - row
        ax.text(0.27, y, name, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4", fc=colours[kind], ec="#555555"))
        ax.text(0.53, y, output, ha="left", va="center", color="#333333")
        if row < len(steps) - 1:
            ax.annotate("", xy=(0.27, y - 0.62), xytext=(0.27, y - 0.38),
                        arrowprops=dict(arrowstyle="->", color="#555555"))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, len(steps) - 0.3)
    ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colours[k], ec="#555555")
               for k in ("input", "manual", "auto")]
    ax.legend(handles, ["input", "interactive (click or confirm)", "automatic"],
              loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False)
    return figure
