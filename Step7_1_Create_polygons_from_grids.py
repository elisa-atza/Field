import numpy as np
import matplotlib.pyplot as plt
import os
from field_paths import YEAR, image_path, plattegrond_paths, polygons_path, reference_date, results_dir
from field_cv import compute_ridge_lateral_bdry, load_image, orthogonal_projection
    
def produce_polygons(polygons,grid, lateral_bdry_top, lat_bdry_bot, plattengrond_cut):

    num_of_columns = int(grid.shape[1]/4)
    num_of_plots = 30
    print('Plots:',num_of_plots, 'Columns:',num_of_columns)
    print(plattengrond_cut.shape)
    for i in range(num_of_columns):
        for j in range(num_of_plots):
            top_row = grid[j, i * 4:i * 4 + 4]
            bot_row = grid[j + 1, i * 4:i * 4 + 4]
            left_proj_top = orthogonal_projection(grid[j, i * 4], [lateral_bdry_top[i * 4], lat_bdry_bot[i * 4]])
            left_proj_bot = orthogonal_projection(grid[j + 1, i * 4], [lateral_bdry_top[i * 4], lat_bdry_bot[i * 4]])
            right_proj_top = orthogonal_projection(grid[j, i * 4 + 3],
                                                   [lateral_bdry_top[i * 4 + 4], lat_bdry_bot[i * 4 + 4]])
            right_proj_bot = orthogonal_projection(grid[j + 1, i * 4 + 3],
                                                   [lateral_bdry_top[i * 4 + 4], lat_bdry_bot[i * 4 + 4]])
            ply = np.vstack((left_proj_top, top_row, right_proj_top, right_proj_bot, np.flipud(bot_row), left_proj_bot, left_proj_top))
            polygons[plattengrond_cut[j, i]] = ply
            # polygons[plattengrond_cut[i,j]] = ply
    return polygons


def load_platts_associations(field):
    # Only the requested field is loaded: a run has one field's plattegronds on disk.
    named_path, lotnum_path = plattegrond_paths(field)
    if field == 'S':
        S_platt=np.load(lotnum_path)
        S_named=np.load(named_path, allow_pickle=True)
        Association_S={'1':S_platt[:30, :], '2':S_platt[30:60, :],
                       '3':S_platt[60:90, :],
                       '4':S_platt[90:,:]}
        return {'S':Association_S}, {'S':S_platt}, {'S':S_named}
    if field == 'V':
        V_platt=np.load(lotnum_path)
        V_named=np.load(named_path, allow_pickle=True)
        Association_V={'1':V_platt[:30, :6], '2':V_platt[:30, 6:],
                       '3':V_platt[30:,:6],
                       '4':V_platt[30:,6:]}
        return {'V':Association_V}, {'V':V_platt}, {'V':V_named}
    if field != 'M':
        raise ValueError(f'no quadrant association defined for field {field!r}')
    M_platt=np.load(lotnum_path)
    M_platt=np.transpose(M_platt)
    M_named=np.load(named_path, allow_pickle=True)
    M_named=np.transpose(M_named)

    Association_M={'4_1':M_platt[:30, :3],'4_2':M_platt[30:, :3],
                   '3_1':M_platt[:30, 3:6],'3_2':M_platt[30:, 3:6],
                   '2_1':M_platt[:30, 6:9],'2_2':M_platt[30:, 6:9],
                   '1_1':M_platt[:30, 9:],'1_2':M_platt[30:, 9:]} #for the 2020 case
    associations={'M':Association_M}
    platts={'M':M_platt}
    named={'M':M_named}
    return associations, platts, named


def main(field):
    # polygons are built on the reference date; Step7_2 warps them onto the others
    date_var=reference_date(field)
    associations,platts,named=load_platts_associations(field)

    '''In this step we create the polygons -- for this we need to create late_bdry and orthogonal projection'''
    polygons={}
    #this assumes that all grids have been renamed after correction -- it is suggested to always create correction files,
    # even if empty, to make sure that all grids have been inspected
    grid_folder=results_dir(field, 'Grids')
    good_grids=[x for x in os.listdir(grid_folder) if x.startswith('Correct')]

    for grid in good_grids:
        index=grid[13:-4]
        print(index)
        grid=np.load(os.path.join(grid_folder, grid))
        if field in ['M','S', 'V']:
            grid=np.fliplr(grid)  ##NECESSARY FOR S AND M (HORIZONTAL FIELDS) -
            # - this might be different if the plattegrond is created differently
        lateral_bdry_top=compute_ridge_lateral_bdry(grid[0,:,:])
        lat_bdry_bot=compute_ridge_lateral_bdry(grid[-1,:,:])
        plattengrond_cut=associations[field][index]
        #the following function returns a dictionary of the shape
        # {... , 36: [(x_1,y_1),(x_2,y_2),...,(x_12,y_12),(x_1,y_1)], ..., 720:[...]}
        polygons=produce_polygons(polygons,grid, lateral_bdry_top, lat_bdry_bot, plattengrond_cut)


    np.save(polygons_path(field, date_var), polygons)

    '''After the produced polygons have been saved, an image is produced, where the polygons have a variety specific
    color, and some batch names are printed in order to verify the correct association of batch to location'''
    

    var_color={'CHALLENGER':'tab:blue','COLOMBA':'tab:orange','INNOVATOR':'tab:red',
               'SAGITTA':'tab:purple','SERESTA':'tab:brown','FESTIEN':'tab:green'}
    im0 = load_image(image_path(field, date_var))
    plt.ion()
    plt.figure(figsize=(23.38, 11.69), dpi=100)
    plt.imshow(im0)
    for plot_num in range(1,721):
        row, col =np.where(platts[field]==plot_num)
        # print(named[field][row, col][0])
        try:
            var, batch =named[field][row, col][0].split('_')
        except ValueError:
            var, batch = named[field][row, col][0].split('-')
        plt.plot(*zip(*polygons[plot_num]), color=var_color[var])
        
        if int(plot_num) in range(1, 721, 19):
            plt.annotate(f'{var[:3]}_{batch}', xy=(polygons[plot_num][0][0], polygons[plot_num][0][1]), xycoords='data'
                         ,rotation=30, bbox=dict(facecolor='white', edgecolor='white', pad=0.5, alpha=.7))

        if int(plot_num) in range(1, 721, 30):
            plt.annotate(f'{plot_num}', xy=(polygons[plot_num][0][0], polygons[plot_num][0][1]), xycoords='data'
                         ,rotation=30, bbox=dict(facecolor='white', edgecolor='white', pad=0.5, alpha=.7))
    
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in var_color.values()]
    plt.legend(markers, var_color.keys(), numpoints=1, bbox_to_anchor=(1,1), loc="upper left")
    plt.title(f'{field}_{YEAR}_Assignment')
    plt.savefig(os.path.join(results_dir(field), f'Assignment_{field}_{date_var}'))
    plt.close()


# This module is a pipeline stage: its main() is called from cli.py, not
# run directly.
