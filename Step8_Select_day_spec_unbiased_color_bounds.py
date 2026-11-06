import cv2
import numpy as np
import os
import pandas as pd
import random
import matplotlib.pyplot as plt
from field_paths import color_bounds_path, dates, image_path, polygons_path
import scipy.signal as sg
from field_cv import load_image, poly_cut, wait_for_close


def compute_Otsu(h, b):
    # Calculate centers of bins
    bin_mids = (b[:-1] + b[1:]) / 2.
    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(h)
    weight2 = np.cumsum(h[::-1])[::-1]
    # Get the class means mu0(t)
    mean1 = np.cumsum(h * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((h * bin_mids)[::-1]) / weight2[::-1])[::-1]
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)
    threshold = bin_mids[:-1][index_of_max_val]
    return threshold


def hist_analsis(im):
    imHSV = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    imHSV[:, :, 2] = cv2.equalizeHist(imHSV[:, :, 2])
    imHSV[:, :, 1] = cv2.equalizeHist(imHSV[:, :, 1])
    # Set total number of bins in the histogram
    bins_num = 90
    # Get the image histogram
    hist, bin_edges = np.histogram(imHSV[:, :, 0], bins=bins_num, density=True)
    hist = hist[1:]
    bin_edges = bin_edges[1:]
    threshold = compute_Otsu(hist, bin_edges)
    
    # ext_hue = sg.argrelextrema(hist, np.less)
    # please note that pk_hue will likely contain more than one peak or could not have any peak
    pk_hue, _ = sg.find_peaks(hist, height=np.mean(hist), prominence=np.mean(hist))
    if len(pk_hue) == 2 and not pk_hue[0] < threshold < pk_hue[1]:
        hist1 = hist[:pk_hue[1] + 25]
        bin1 = bin_edges[:pk_hue[1] + 26]
        threshold = compute_Otsu(hist1, bin1)
    
    elif len(pk_hue) != 0 and not pk_hue[0] < threshold:
        
        hist1 = hist[:pk_hue[0] + 15]
        bin1 = bin_edges[:pk_hue[0] + 16]
        threshold = compute_Otsu(hist1, bin1)
    
    return bin_edges, hist, pk_hue, threshold


def get_right_polygons(date_var, field):
    polygons = np.load(polygons_path(field, date_var), allow_pickle=True).item()
    return polygons


def otsu_segmentation(image, value_low, value_high=None):
    # image converted to HSV format
    im1hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    im1flt = np.copy(image)
    
    im1flt[im1hsv[:, :, 0] <= value_low] = (0, 0, 0)
    if value_high:
        im1flt[im1hsv[:, :, 0] >= value_high] = (0, 0, 0)
    
    return im1flt


def show_plots_with_given_otsus(otsus_low, random_plots, field_dates_ordered, field, otsus_high=[]):
    for ix, date_var in enumerate(field_dates_ordered):
        polygons = get_right_polygons(date_var, field)
        otsu_val = otsus_low[ix]
        img = load_image(image_path(field, date_var))
        plt.ion()
        fig1, axs = plt.subplots(nrows=2, ncols=5, sharey=True, sharex=True, figsize=(23.38, 11.69), dpi=100)
        for i, value in enumerate(random_plots):
            im1, r0, r1 = poly_cut(img, polygons[value])
            if otsus_high:
                otsu_high = otsus_high[ix]
                im = otsu_segmentation(im1, otsu_val, otsu_high)
            else:
                im = otsu_segmentation(im1, otsu_val)
            
            # im[im!=0]=255
            axs[int(i / 5), i % 5].imshow(im1)
            axs[int(i / 5) + 1, i % 5].imshow(im)
            axs[int(i / 5), i % 5].set_title(str(value))
        
        fig1.suptitle(f'{field} {date_var}')
        plt.draw()
        plt.pause(0.5)


def get_color_index(image, index):
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    
    if index == 'BI':
        BI = np.sqrt((r ** 2 + g ** 2) / 2)  # Brightness index
        result = BI
    elif index == 'SCI':
        # SCI= (r-g)/(r+g) # Soil Colour Index
        # result=SCI
        SCI = (2 * g) - (r + b)  # excessgreen
        result = SCI
    elif index == 'EXG':
        EXG = (2 * g) - (r + b)  # excessgreen
        result = EXG
    elif index == 'GLI':
        GLI = (2 * g - r - b) / (2 * g + r + b)  # Green leaf index Vis Louhaichi et al. (2001) or VDVI
        result = GLI
    elif index == 'HI':
        HI = (2 * r - g - b) / (g - b)  # Primary colours Hue Index
        result = HI
    elif index == 'NGRDI':
        NGRDI = (g - r) / (g + r)  # or synthetic NDVI
        result = NGRDI
    elif index == 'RI':
        RI = r ** 2 / (b * g ** 3)  # Redness Index
        result = RI
    elif index == 'SI':
        SI = (r - b) / (r + b)  # Spectral Slope Saturation Index
        result = SI
    elif index == 'VARI':
        VARI = (g - r) / (g + r - b)
        result = VARI
    elif index == 'VVI':
        VVI = (1 - (r - 30) / (r + 30)) * (1 - (g - 50) / (g + 50)) * (1 - (b - 1) / (b + 1))
        result = VVI
    elif index == 'GLAI':
        GLAI = (25 * (g - r) / (g + r - b) + 1.25)
        result = GLAI
    elif index == 'GRVI':
        GRVI = (g - r) / (g + r)
        result = GRVI
    elif index == 'CI':
        CI = (r - b) / r
        result = CI
    elif index == 'GCC':
        GCC = g / (r + g + b)  # green chromatic coordinate
        result = GCC
    else:
        raise ValueError(f'unknown colour index {index!r}')
    result[np.isnan(result)] = 0
    result = np.nan_to_num(result, posinf=255, neginf=-255)
    max_index = np.max(result)
    result = result / max_index

    return result


def get_otsu_hist_analysis(field_dates, field, plot=False, otsus_list=[]):
    peaks = {}
    otsus = {}
    hists = {}
    bins = {}
    for ix, date_var in enumerate(field_dates):
 
        img = load_image(image_path(field, date_var))
        if len(otsus_list) > 1:
            print('segmented')
            img = otsu_segmentation(img, otsus_list[ix])
        # polygons=get_right_polygons(date_var)
        # img1, r1, r2=poly_cut(img, polygons[random_plot[0]])
        bin_edges, hist, pk_hue, threshold = hist_analsis(img)
        peaks[date_var] = pk_hue
        otsus[date_var] = threshold
        hists[date_var] = hist
        bins[date_var] = bin_edges
        
        print('how many peaks?:', len(pk_hue))
        if plot:
            plt.ion()
            plt.figure()
            plt.bar(bin_edges[:-1], hist, width=1)
            plt.scatter(bin_edges[pk_hue], hist[pk_hue], marker='v', color='tab:orange')
            plt.axvline(x=threshold, color='tab:orange')
            plt.title(f'{date_var} peaks: {len(pk_hue)}')
            plt.draw()
            plt.pause(0.5)
        if threshold < bin_edges[pk_hue[0]]:
            if plot:
                wait_for_close('Read the histogram, then close the figure window.')
            otsus[date_var] = int(input(f'Choose the appropriate second boundary for this date {date_var}:'))
    return peaks, otsus, hists, bins


def patch_peaks(field_dates, hists, peaks, bins, otsus):
    first_pks = [hists[x][peaks[x][0]] for x in field_dates]
    second_pks = [hists[x][peaks[x][1]] if len(peaks[x]) > 1 else 0 for x in field_dates]
    first_bins = [bins[x][peaks[x][0]] for x in field_dates]
    second_bins = [bins[x][peaks[x][1]] if len(peaks[x]) > 1 else 0 for x in field_dates]
    ots = [otsus[x] for x in field_dates]
    sandwich_otsu = [x if first_bins[i] < x < second_bins[i] else 0 for i, x in enumerate(ots)]
    sand = [x for x in sandwich_otsu if x != 0]
    emergency_otsu = np.mean(np.array(sand))
    print(first_bins, second_bins, emergency_otsu)
    sandwich_otsu = [x if x != 0 else emergency_otsu for x in sandwich_otsu]
    return first_pks, second_pks, first_bins, second_bins, sandwich_otsu


def get_color_bounds(field):
    data_bounds = pd.read_csv(color_bounds_path(field))
    return data_bounds


def color_index_mask(im_o):
    '''Canopy mask from the fused vegetation index 0.2*BI + 0.2*HI + 0.6*ExG,
    normalised, thresholded at 0.45 and median-filtered; also returns the
    three indices.'''
    mask = np.zeros_like(im_o[:, :, 0])
    coeffs = {'BI': 0.2, 'HI': 0.2, 'EXG': 0.6}
    indici = []
    for i, index in enumerate(['BI', 'HI', 'EXG']):
        indexed = get_color_index(im_o, index)
        indici.append(indexed)
        mask = mask + coeffs[index] * indexed
    mask = mask / np.max(mask)
    mask = np.dstack((mask, mask, mask))
    mask[mask > 0.45] = 1
    mask[mask != 1] = 0
    
    mask = mask.astype(dtype='uint8')
    mask = cv2.medianBlur(mask, 3)
    return mask, indici


def segmentation_image_color_index(image, date, field):
    color_bounds = get_color_bounds(field)
    im_o = otsu_segmentation(image, color_bounds[color_bounds['Date'] == date]['Lower_hue'].values[0],
                             color_bounds[color_bounds['Date'] == date]['Upper_hue'].values[0])
    
    mask, indici = color_index_mask(im_o)
    imf = image.copy()
    imf[mask == 0] = 0
    imf = np.nan_to_num(imf, nan=0)
    return imf, indici


def main(field):
    
    field_dates = dates(field)
    
    if os.path.exists(color_bounds_path(field)):
        data_bounds = pd.read_csv(color_bounds_path(field))
    else:
        data_bounds = pd.DataFrame(columns=['Date', 'Lower_hue', 'Upper_hue'])
    
    '''Find rudimentary color bounds by histogram analysis'''
    # file_list=os.listdir(rgb_paths[year][field])
    random_plots = random.sample(range(1, 721), 5)
    
    peaks, otsus, hists, bins = get_otsu_hist_analysis(field_dates, field, plot=False)
    
    first_pks, second_pks, first_bins, second_bins, sandwich_otsu = patch_peaks(field_dates, hists, peaks, bins, otsus)
    
    peaks, s_otsus, hists, bins = get_otsu_hist_analysis(field_dates, field, plot=True, otsus_list=sandwich_otsu)
    
    second_otsus = s_otsus
    
    show_plots_with_given_otsus(sandwich_otsu, random_plots, field_dates, field, list(second_otsus.values()))
    wait_for_close('Check the segmentations, then close the figure windows.')
    data_bounds['Date'] = field_dates
    data_bounds['Lower_hue'] = sandwich_otsu
    data_bounds['Upper_hue'] = second_otsus
    
    data_bounds.to_csv(color_bounds_path(field))


# This module is a pipeline stage: its main() is called from cli.py, not
# run directly.
