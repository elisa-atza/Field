import numpy as np
import matplotlib.pyplot as plt
import os
from field_cv import load_image, orthogonal_projection, wait_for_close
from field_paths import corrected_grid_path, grid_path, image_path, reference_date, results_dir

def find_point_to_correct(corr_point, original_grid):
    original_grid=original_grid.reshape((31,-1,2))
    projections_norms = np.zeros(original_grid.shape[1])
    # print(projections_norms.shape)
    for col in range(original_grid.shape[1]): #for each column in the grid we project the correction
        pt_proj = orthogonal_projection(corr_point, [original_grid[0, col, :], original_grid[-1, col, :]])
        projections_norms[col] = np.linalg.norm(pt_proj - corr_point)

    index_col = np.argmin(projections_norms)
    row_diff = original_grid[:, index_col, :] - corr_point
    row_diff = np.linalg.norm(row_diff, axis=1)

    index_row = np.argmin(row_diff)
    return index_row, index_col



def replace_grid_points(arr_corrections, grid):
    grid_corr = np.copy(grid)
    for elt in arr_corrections:
        index_row,index_col=find_point_to_correct(elt, grid)
        grid_corr[index_row, index_col,:] = elt

    return grid_corr


def show_all_grids(field):
    """CHECK: plot all grids to inspect for mistakes; in the next step the
    mistakes can be corrected grid by grid."""
    date = reference_date(field)
    grid_folder = results_dir(field, 'Grids')
    plt.ion()
    plt.figure()
    plt.imshow(load_image(image_path(field, date)))
    for file in sorted(os.listdir(grid_folder)):
        grid=np.load(os.path.join(grid_folder,file))
        grid=np.delete(grid, np.where(~grid.any(axis=1))[0], axis=0)
        grid=np.reshape(grid, (-1,2))
        plt.scatter(*zip(*grid), label=file[:-4], s=4)
    plt.legend()
    plt.title(f'{field} {date} grids')
    wait_for_close('Note which grids need corrections, then close the window.')


def main(field, index=None):
    date = reference_date(field)
    image = load_image(image_path(field, date))
    script_result_folder = results_dir(field, 'Correction arrays')
    i = str(index) if index else input('grid index (e.g. 1 or 1_2): ')
    print('grid', i)
    grid=np.load(grid_path(field, i))
    # grid= grid[~np.all(grid==0, axis=1)]
    grid=np.delete(grid, np.where(~grid.any(axis=1))[0], axis=0)


    curve = []

    plt.ion()

    fig = plt.figure()
    plt.imshow(image)
    plt.scatter(*zip(*grid.reshape((-1,2))), color='b',marker='h')
    plt.title(date)

    # #-------------------------corner selection-------------------------------------#

    def select_correction_points(event):
        """Callback function for the figure.
        Zoom to the required area, single right-click on the image adds one point
        to the polygon. Zoom on the top part of the ridge block, select a point at the
        leftmost ridge just above the green, then do the same at the rightmost ridge,
        then zoom at the bottom part of the ridge block, select left, right as above.
        Double click on the image to finish. Corner point coordinates are in the
        "curve" array, top and bottom ridge coordinates are in the "tops" and "bots"
        arrays.
        """
        nonlocal curve, grid  # the other names are only read, not rebound
        if event.inaxes is not None:
            if event.button == 3:  # and len(curve)<NumR
                xc = int(event.xdata)
                yc = int(event.ydata)
                print('x =', xc, ' y =', yc)
                plt.plot(xc, yc, 'r.')
                plt.show()
                curve.append([xc, yc])
                grid=grid.reshape((31,-1,2))
                index_r,index_c=find_point_to_correct(np.array([xc,yc]), grid)
                plt.plot(grid[index_r,index_c,0], grid[index_r,index_c,1], 'y.')
            elif event.button == 1 and event.key== "shift":  # if you misclicked you cn undo the last point with middle click
                del curve[-1]
                print('undone')
            elif event.button == 2 :
                print('all correction points selected')
                curve = np.array(curve)
                fig.canvas.mpl_disconnect(cid)
                np.save(f'{script_result_folder}/{field}_correction_grid_{i}', curve)
                corr_grid=replace_grid_points(curve, grid)
                np.save(corrected_grid_path(field, i), corr_grid)
                plt.close(fig)

                return
        else:
            print('Clicked ouside axes bounds but inside plot window')

    cid = fig.canvas.mpl_connect('button_press_event', select_correction_points)
    wait_for_close('Right-click where a grid point should be, shift+left-click to undo, '
                   'middle-click to save Correct_grid_' + str(i) + ' (do this even with no corrections).')



# This module is a pipeline stage: its main() is called from cli.py, not
# run directly.
