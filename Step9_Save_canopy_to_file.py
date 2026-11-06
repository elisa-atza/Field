import functools
import numpy as np
import pandas as pd
from field_paths import (SEGMENTATION, canopy_table_path, dates, image_path,
                         plattegrond_paths, polygons_path, raw_canopy_path)
from field_cv import extractHueSat, find_px_to_cm2_coeff, load_image, poly_cut, wait_for_close
import cv2
from Step8_Select_day_spec_unbiased_color_bounds import segmentation_image_color_index
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import random

# Plants per ridge section. Single plants can't be isolated in the image, so a
# ridge's total canopy is divided by this to give the average canopy per plant.
PLANTS_PER_RIDGE = 6


def centroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length


def get_right_polygons(date_var, field):
    polygons=np.load(polygons_path(field, date_var), allow_pickle=True).item()
    return polygons


def midpoints_arrays(arr1):
    return (arr1[:-1,:]+arr1[1:,:])/2


@functools.lru_cache(maxsize=None)
def load_plattegrond(field):
    """Loads the (named, lot-number) plattegrond pair for a field.
    Cached: produce_raw_file_canopy() asks for a name 2880 times per run.
    """
    named_path, lotnum_path = plattegrond_paths(field)
    named = np.load(named_path, allow_pickle=True)
    lotnum = np.load(lotnum_path)
    return named, lotnum


def name_from_plot_num(plot_num, field):
    named, platt = load_plattegrond(field)
    name = named[platt == plot_num][0]
    # plattegrond names are VARIETY-BATCH in M 2020, VARIETY_BATCH in the 2021 layouts
    separator = '-' if '-' in name else '_'
    variety = name.split(separator)[0]
    batch = name.split(separator)[1]
    return variety, batch
    

def get_ridge_from_poly(coords):
    if coords.shape[0] != 13:
        raise ValueError(
            f'expected a 13-point plot polygon (4 ridges), got {coords.shape[0]} points')
    up_lateral_coords = midpoints_arrays(coords[1:5,:])
    down_lateral_coords = midpoints_arrays(np.flipud(coords[7:11,:]))
    ridge1 = np.vstack((coords[0:2,:], up_lateral_coords[0,:] , down_lateral_coords[0,:], coords[10:,:]))
    ridge2 = np.vstack((up_lateral_coords[0,:] , coords[2,:], up_lateral_coords[1,:], down_lateral_coords[1,:], coords[9,:], down_lateral_coords[0,:]))
    ridge3 = np.vstack((up_lateral_coords[1,:], coords[3,:], up_lateral_coords[2,:], down_lateral_coords[2,:], coords[8,:], down_lateral_coords[1,:]))
    ridge4 = np.vstack((up_lateral_coords[2,:], coords[4:8,:], down_lateral_coords[2,:]))
    return ridge1, ridge2,ridge3, ridge4


def fill_df_field(field, field_dates):
    columns = ['Plot number', 'Variety', 'Batch', 'Ridge' ]
    files={}
    for date in field_dates:
        columns.append(f'{date}')
        files[date] = pd.read_csv(raw_canopy_path(field, date))
    
    rows = []
    
    for index in range(0,2880):
        row = {}
        row['Plot number'] = int(index / 4) + 1
        k=random.choice(list(files.keys()))
        
        row['Batch'] =files[k].iloc[index]['Batch']
        row['Variety']=files[k].iloc[index]['Variety']
        
        ridge_n = (index%4) +1
        row['Ridge'] =ridge_n
        for key, value in files.items():
           
            val= value[(value['Lot number']==int(index/4)+1) & (value['Ridge']==ridge_n)]['Average canopy'].values[0]
            row[f'{key}']=val
        rows.append(row)
    
    table = pd.DataFrame(rows, columns=columns)
    table.to_csv(canopy_table_path(field), index=False)
    

def plot_field_canopy_as_field(field, date):
    data_frame_un=pd.read_csv(raw_canopy_path(field, date))
    image=load_image(image_path(field, date))
    max_canopy_un=data_frame_un['Average canopy'].max()

    polygons=get_right_polygons(date, field)
    
    plt.ion()
    fig, ax = plt.subplots(1,2, sharey=True, sharex=True)
    cm = plt.get_cmap("viridis")
    ax[0].invert_yaxis()
    
    colorsOrig = []
    
    for ii in range(1, 721):
        ridge1, ridge2, ridge3, ridge4 = get_ridge_from_poly(polygons[ii])
        this_lot_data_un=data_frame_un[data_frame_un['Lot number']==ii]

        for ix, ridge in enumerate([ridge1, ridge2, ridge3, ridge4 ]):
            ridge=np.array(ridge, dtype=np.int32).reshape(-1, 2)
            canopy=this_lot_data_un[this_lot_data_un['Ridge']==ix+1]['Average canopy'].values[0]
            colorsOrig.append(canopy/ max_canopy_un) # / maxGr
            
            ax[0].fill(*zip(*ridge), c=cm(canopy/max_canopy_un), alpha=0.8)

    fig.suptitle(f'{field} {date}')
    ax[0].set_title('Canopy')
    ax[1].imshow(image)
    ax[1].set_title('RGB image')
        
        
    # colour bar spanning the plotted values
    pOrig = ScalarMappable(cmap=cm)
    pOrig.set_array(np.array(colorsOrig))
        
    fig.subplots_adjust(right=0.9)  #0.8
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) #[0.85, 0.15, 0.05, 0.7]
    fig.colorbar(pOrig, cax=cbar_ax)
    plt.draw()
    plt.pause(0.5)

        

def check_canopy(field, date):
    """Canopy per ridge drawn in place, next to the image:
    misplaced polygons or wrongly attributed lots stand out as a pattern that
    doesn't match the field."""
    plot_field_canopy_as_field(field, date)
    wait_for_close(f'Check {field} {date}: canopy vs image. Close the window to continue.')


def count_soil_canopy_pixels(binary_imageRGB):
    '''This functions considers an RGB image, previously binarized, and counts the pixels relative to canopy and soil'''
    #make sure the image array is 2-dimensional
    if binary_imageRGB.ndim == 3:
        image_2d = cv2.cvtColor(binary_imageRGB, cv2.COLOR_RGB2GRAY)
    else:
        image_2d = binary_imageRGB.copy()
    image_2d[image_2d < 70] = 0
    tot_num_pixels = image_2d.shape[0] * image_2d.shape[1]
    canopy_pixels = int(np.count_nonzero(image_2d))
    soil_pixels = tot_num_pixels - canopy_pixels
    return soil_pixels, canopy_pixels

def count_canopy_pixels(im_bin):
    """Number of pixels left in a masked RGB image; each pixel counts once,
    however many of its channels are non-zero."""
    return int(np.count_nonzero(np.any(im_bin != 0, axis=-1)))


def produce_raw_file_canopy(field, date):
    columns = ['Lot number','Variety','Batch', 'Ridge', 'Average canopy', 'Ridge centroid' ]
    rows = []
    image=load_image(image_path(field, date))
    print(field, date)
    polygons=get_right_polygons(date, field)
    # constant for the whole run: derived from the polygon geometry
    areainpix=find_px_to_cm2_coeff(polygons, field)
    hue_saturation_ranges=[40, 68,86, 162 ]
    for plot_num in range(1,721):

        poly=np.array([polygons[plot_num]], dtype=np.int32)
        poly = poly.reshape((-1, 2))
        ridge1, ridge2, ridge3, ridge4 = get_ridge_from_poly(poly)
        for ix,element in enumerate([ridge1, ridge2, ridge3, ridge4]):
            # cut to polygon
            ridge_coords = np.array(element, dtype=np.int32)
            ridge_coords = ridge_coords.reshape(-1,2)
            im_ridge, r0, r1= poly_cut(image, ridge_coords)

            if SEGMENTATION == 'per_date':
                # bounds chosen for this date by Step8, plus its colour-index mask
                im_bin,_= segmentation_image_color_index(im_ridge, date, field)
                im_bin=np.nan_to_num(im_bin, nan=0)
            else:
                '''Convert crop to binary image'''
                blacksoil_imageRGB = extractHueSat(im_ridge, hue_saturation_ranges)

                '''Count pixels relative to the canopy and pixels relative to the soil'''
                soil, canopy = count_soil_canopy_pixels(blacksoil_imageRGB)

            ridge_avgs = {}
            ridge_avgs['Lot number'] = plot_num
            variety, batch=name_from_plot_num(plot_num,field)

            ridge_avgs['Batch'] = batch
            ridge_avgs['Variety']=variety
            ridge_avgs['Ridge']=ix+1
            
            x, y = centroidnp(ridge_coords)
            ridge_avgs['Ridge centroid']=np.array([x, y])
            
            # areainpix=0.36188
            if SEGMENTATION == 'per_date':
                avg_canopy_area=(count_canopy_pixels(im_bin)*areainpix)/PLANTS_PER_RIDGE
            else:
                avg_canopy_area = (canopy * areainpix) / PLANTS_PER_RIDGE
            ridge_avgs['Average canopy']=avg_canopy_area
            rows.append(ridge_avgs)
            
    stats = pd.DataFrame(rows, columns=columns)
    stats.to_csv(raw_canopy_path(field, date), index=False)

def main(field, date=None):
    # one date if given, otherwise every date configured for the field; the
    # per-date tables are then joined into one column per date
    field_dates = [date] if date else dates(field)
    for date in field_dates:
        produce_raw_file_canopy(field, date)  # produces a day specific canopy file
    fill_df_field(field, field_dates)


# This module is a pipeline stage: its main() is called from cli.py, not
# run directly.

#






