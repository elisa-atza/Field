import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from field_paths import FIELDS, dates, image_path, latest_canopy_table, plattegrond_paths, polygons_path, results_dir
import gif
from Step8_Select_day_spec_unbiased_color_bounds import *
from field_cv import extractHueSat, load_image

def _canopy_table(field):
    return pd.read_csv(latest_canopy_table(field))


def name_from_plot_num(plot_num, field):
    df=_canopy_table(field)
    sub=df[df['Plot number']==plot_num]
    variety=sub['Variety'].unique()
    batch=sub['Batch'].unique()
    return variety, batch


def get_plotnums_from_varbatch(field, var, batch):
    df=_canopy_table(field)
    variety=df[df['Variety']==var]
    batchnum=variety[variety['Batch']==batch]
    num_polygon=batchnum['Plot number'].unique()
    return num_polygon # list of 4 polygons containing the batch


def get_section(plattegrond, field, section):
    Association_M={'1_1':plattegrond[:30, :3],'1_2':plattegrond[30:, :3],
                   '2_1':plattegrond[:30, 3:6],'2_2':plattegrond[30:, 3:6],
                   '3_1':plattegrond[:30, 6:9],'3_2':plattegrond[30:, 6:9],
                   '4_1':plattegrond[:30, 9:],'4_2':plattegrond[30:, 9:]}
    Association_S={'1':plattegrond[:30, :], '2':plattegrond[30:60, :],
                   '3':plattegrond[60:90, :],
                   '4':plattegrond[90:,:]}
    Association_V={'1':plattegrond[:30, :6], '2':plattegrond[:30, 6:],
                   '3':plattegrond[30:,:6],
                   '4':plattegrond[30:,6:]}
    associations={'M':Association_M, 'S':Association_S, 'V':Association_V}
    return associations[field][section]
    
    
def find_extrema(polygons, num_portion):
    xs=np.array([])
    ys=np.array([])
    for lot in num_portion:
        xs=np.append(xs, polygons[lot][:,0])
        ys=np.append(ys, polygons[lot][:,1])
        
    xmin=int(np.min(xs))
    xmax=int(np.max(xs))
    ymin=int(np.min(ys))
    ymax=int(np.max(ys))
    return xmin, ymin, xmax, ymax


def produce_block_picture(date_var, varname,block_ix, field, num_portion, result_folder, blackout=False):
    polygons=np.load(polygons_path(field, date_var), allow_pickle=True).item()
    # file_name=f'{field}_Color_bounds.csv'
    # data_bounds=pd.read_csv(os.path.join(results_2021_fields[field], file_name))
    x_min,y_min,x_max,y_max=find_extrema(polygons, num_portion)
    
    image=load_image(image_path(field, date_var))
    crop=image[y_min-500:y_max+500,x_min-500:x_max+500]
    # if blackout:
        # bounds=data_bounds[data_bounds['Date']==date_var].iloc[:,1:].values[0]
        # print(bounds)
        # bounds=[int(x) for x in bounds]
        # crop=extractHueSat(crop,bounds)
    
    plt.ion()
    plt.figure(figsize=(8.27, 11.69), dpi=100)
    plt.imshow(crop)
    for lot in num_portion:
        val=polygons[lot]
        val=val-np.array([x_min-500,y_min-500])
        plt.plot(*zip(*val), color='tab:blue')
    plt.title(f'{varname}_Block{block_ix}_{field}_{date_var}')
    plt.savefig(f'{result_folder}/{varname}_Block{block_ix}_{field}_{date_var}')

@gif.frame
def produce_block_gif(date_var, varname,block_ix, field, num_portion, result_folder, segmentation=True):
    polygons=np.load(polygons_path(field, date_var), allow_pickle=True).item()
    image=load_image(image_path(field, date_var))
    # file_name=f'{field}_Color_bounds.csv'
    # data_bounds=pd.read_csv(os.path.join(results_2021_fields[field], file_name))
    x_min,y_min,x_max,y_max=find_extrema(polygons, num_portion) # needs list of polygon numbers
    
    print(y_max, y_min, x_max, x_min)
    crop=image[max(0,y_min-500):y_max+500,max(0,x_min-500):x_max+500]
    if segmentation:
        segmented,_= segmentation_image_color_index(crop, date_var, field)
        product=segmented
    else:
        product=crop
    print(product.shape)
    plt.ion()
    plt.figure(figsize=(8.27, 11.69), dpi=100)
    plt.imshow(product)
    for lot in num_portion:
        val=polygons[lot]
        val=val-np.array([x_min-500,y_min-500])
        plt.plot(*zip(*val), color='tab:blue')
    plt.title(f'{varname}_Block{block_ix}_{field}_{date_var}')
    # plt.savefig(f'{result_folder}/{varname}_Block{block_ix}_{field}_{date_var}')
    

'''Produce and save gifs for all varieties, all blocks, all fields'''


def main(segmentation=False):
    platts={}
    named={}
    for field in FIELDS:
        named_path, lotnum_path = plattegrond_paths(field)
        platts[field]=np.load(lotnum_path, allow_pickle=True)
        named[field]=np.load(named_path, allow_pickle=True)

    for field in FIELDS:
        field_dates=dates(field)
    
        for block_ix in ['1','2','3','4']:   # ['1_1','1_2','2_1','2_2','3_1','3_2','4_1','4_2']
             for varname in ['SAGITTA'  , 'CHALLENGER', 'SERESTA' ,'COLOMBA','FESTIEN','INNOVATOR']:
                frames=[]
                for date in field_dates:
                    date_var= date
    
                    result_folder=results_dir(field, 'Pictures')
 
                    named_platt=get_section(named[field], field, block_ix)
                    num_platt=get_section(platts[field], field, block_ix)
                    named_platt=named_platt.astype('str')
                  
                    num_portion=num_platt[np.char.startswith(named_platt,varname)]
                    if num_portion.shape[0]==0:
                        continue
                    else:
                        frame=produce_block_gif(date_var, varname, block_ix, field, num_portion, result_folder, segmentation=False)
                        frames.append(frame)
                if len(frames)>0:
                    if segmentation:
                        gif.save(frames, f'{result_folder}/Segmentation_{varname}_block_{block_ix}.gif', duration=2*len(field_dates), unit="s", between="startend")
                    else:
                        gif.save(frames, f'{result_folder}/Alignment_{varname}_block_{block_ix}.gif', duration=2*len(field_dates), unit="s", between="startend")



# This module is a pipeline stage: its main() is called from cli.py, not
# run directly.
