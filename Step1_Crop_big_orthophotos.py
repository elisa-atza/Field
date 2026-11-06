import gc
from osgeo import gdal
import os
import numpy as np
import matplotlib.pyplot as plt
from field_cv import load_image, wait_for_close
from field_paths import images_dir


def openChannel(path, ch, result_path, date):
    # S2021 specific
    file = gdal.Open(path)
    CH=file.GetRasterBand(ch).ReadAsArray()
    print('read channel as array')
    tif=np.zeros([CH.shape[0],CH.shape[1]], np.uint8)
    tif[:,:]= CH
    del (tif)
    del(file)
    gc.collect()




def main(path, field, date):
    # the crop lands in the field's images folder, named so that
    # field_paths.image_path(field, date) finds it
    result_path=images_dir(field)
    os.makedirs(result_path, exist_ok=True)

    '''Crop full image'''

    tif=load_image(path)

    plt.ion()
    plt.figure(1)
    plt.imshow(tif)
    wait_for_close('Read off the crop bounds, then close the figure window.')
    left, right, top, bottom=[int(x) for x in input('Insert cropping coordinates, comma separated, in the order: left, right, top, bottom. ').split(',')]
    
    plt.ion()
    plt.figure(2)
    plt.imshow(tif[top:bottom, left:right])
    wait_for_close('Check the crop, then close the figure window.')
    good=input('Is this the crop you wanted?: (Yes/No)')
    if good.upper()=='YES':
        np.save(os.path.join(result_path, f'Crop_{date}'),
                tif[top:bottom, left:right])




