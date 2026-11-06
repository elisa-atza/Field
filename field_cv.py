"""Helpers shared by the pipeline steps.

Each Step module used to carry its own copy of these.

Deliberately *not* collected here: the plattegrond/layout helpers
(`name_from_plot_num`, `get_section` and the `Association_M/S/V` tables).
Those are specific to each field's layout and live with the steps that use them.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


# --- Reading images ---------------------------------------------------------

def open_tif(path):
    '''Reads the first three bands of a GeoTIFF into an RGB uint8 array.
    Returns the image and the GDAL geotransform.
    '''
    from osgeo import gdal

    file = gdal.Open(path)
    gt = file.GetGeoTransform()
    R = file.GetRasterBand(1).ReadAsArray()
    G = file.GetRasterBand(2).ReadAsArray()
    B = file.GetRasterBand(3).ReadAsArray()
    tif = np.zeros([B.shape[0], B.shape[1], 3], np.uint8)
    tif[:, :, 0] = R
    tif[:, :, 1] = G
    tif[:, :, 2] = B
    return tif, gt


def load_image(path):
    '''Loads an orthophoto, dispatching on extension (.tif/.tiff vs .npy).'''
    if path.lower().endswith(('.tif', '.tiff')):
        image, _ = open_tif(path)
        return image
    return np.load(path, allow_pickle=True)


# --- Colour -----------------------------------------------------------------

def extractHueSat(imRGB, list_bds):
    '''This function takes in an RGB image, and a list with four elements:
    [hl, hh, sl, sh], where hl and sl are the lower bounds for the hue and
    saturation channels respectively, and hh, sh are the higher bounds for the
    respective channels.
    The function returns a filtered RGB image with the color of all pixels
    outside the hue (hl,hh) and saturation (sl,sh) ranges set to black (0,0,0)
    '''
    d1h, d2h, d1s, d2s = list_bds
    # image converted to HSV format
    im1hsv = cv2.cvtColor(imRGB, cv2.COLOR_RGB2HSV)
    # setting pixels outside ranges to (0,0,0) in the RGB image
    im1flt = np.copy(imRGB)
    im1flt[np.logical_or(im1hsv[:, :, 0] <= d1h, im1hsv[:, :, 0] >= d2h)] = (0, 0, 0)
    im1flt[np.logical_or(im1hsv[:, :, 1] <= d1s, im1hsv[:, :, 1] >= d2s)] = (0, 0, 0)
    return im1flt


# --- Planting geometry ------------------------------------------------------

# Nominal ridge spacing in cm. M was planted at 74 cm, the other two
# fields at 75 cm; this is what converts a pixel count into an area.
RIDGE_SPACING_CM = {'M': 74, 'V': 75, 'S': 75}

# Which coordinate to sort along: horizontal fields sort on y, vertical on x.
FIELD_DIRECTION = {'M': 1, 'S': 1, 'V': 0}


def orthogonal_projection(point, points_of_line):
    '''Projects point on the line passing by points_of_line
    don't forget to bring the line to the origin
    first and to bring it back afterwards'''
    x_1, y_1 = points_of_line[0] - points_of_line[1]
    x_2, y_2 = points_of_line[1] - points_of_line[1]
    point = point - points_of_line[1]
    direc_vec = np.array((x_1 - x_2, y_1 - y_2))
    proj = (np.dot(point, direc_vec) / np.dot(direc_vec, direc_vec)) * direc_vec
    return proj + points_of_line[1]


def compute_ridge_lateral_bdry(coords):
    '''Computes coordinates of the ridge's sides, we have selected the middle points'''
    numR = len(coords)
    bnds = np.zeros((numR + 1, 2))
    dx = coords[1, 0] - coords[0, 0]
    dy = coords[1, 1] - coords[0, 1]
    bnds[:, 0] = np.linspace(coords[0, 0] - dx / 2, coords[numR - 1, 0] + dx / 2,
                             numR + 1)
    bnds[:, 1] = np.linspace(coords[0, 1] - dy / 2, coords[numR - 1, 1] + dy / 2,
                             numR + 1)
    return bnds


def diagonal_sorting(coord_array, field):
    '''Orders points along the field's long axis by projecting them onto the
    line through the two extreme points and sorting by that projection.'''
    rav = coord_array.reshape((-1, 2))
    sort_coord = FIELD_DIRECTION[field]
    # we pick the min and max of the x or y coordinate depending on the
    # orientation of the field
    min_y = rav[np.argmin(rav[:, sort_coord])]
    max_y = rav[np.argmax(rav[:, sort_coord])]
    projections = np.zeros_like(rav)
    for row in range(rav.shape[0]):
        projections[row, :] = orthogonal_projection(rav[row, :], [min_y, max_y])
    # sort points in projection according to the sorting given by the first column(y)
    sorted_rav = rav[projections[:, sort_coord].argsort()]
    return sorted_rav


def poly_cut(img, pts):
    '''Crops img to the bounding box of the polygon pts and blacks out
    everything outside the polygon. Returns the crop and its top-left corner.'''
    points = np.array(pts, dtype=np.int32)
    points.reshape((-1, 2))
    points[points < 0] = 0
    rect = cv2.boundingRect(points)
    cropped = img[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    points = points - np.array([rect[0], rect[1]])
    height = cropped.shape[0]
    width = cropped.shape[1]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, np.int32([points]), (1))
    res = cv2.bitwise_and(cropped, cropped, mask=mask)
    return res, rect[0], rect[1]


def find_px_to_cm2_coeff(polygons, field):
    '''Area in cm2 of a single pixel, derived from the mean across-ridge
    spacing of the plot polygons and the known planting distance.

    Takes the already-loaded polygon dict: the two previous copies of this
    function each called their own get_right_polygons(), which behaved
    differently for 2020 and 2021.
    '''
    firr = np.array([])
    secc = np.array([])
    for value in polygons.values():
        first = np.linalg.norm(value[1:4] - value[2:5], axis=1)
        second = np.linalg.norm(value[-6:-4] - value[-5:-3], axis=1)
        firr = np.append(firr, np.mean(first))
        secc = np.append(secc, np.mean(second))
    pixel_measurement = np.mean((np.mean(firr), np.mean(secc)))
    pixels_for_1cm = pixel_measurement / RIDGE_SPACING_CM[field]
    cm_in_1px = 1 / pixels_for_1cm
    area_1_pixel = cm_in_1px ** 2
    return area_1_pixel


# --- Figure windows ---------------------------------------------------------

def wait_for_close(message='Close the figure window to continue.'):
    '''Blocks until every open figure is closed, then forgets them.

    The steps call plt.ion() and draw into figures. Inside an IDE console the
    figures outlive the call; from `python cli.py ...` the process would exit
    and take them along, so steps that need the user call this instead. With
    a non-interactive backend this returns immediately.
    '''
    print(message)
    plt.show(block=True)
    plt.close('all')  # no-op after the user closed them; tidies up otherwise
