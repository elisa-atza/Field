import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, Rbf
from field_cv import load_image
from field_paths import image_path, polygons_path, reference_date, results_dir, sorted_markers_path
import os


def aff_transform(polygons, ref_pts, pts_to_warp):
    
    M = np.zeros((2*ref_pts.shape[0], 4))
    for row in range(0,ref_pts.shape[0]):
        M[2*row:(2*row)+2, :2] = np.identity(2)
        M[2*row:(2*row )+ 2, 2:] = np.array([ref_pts[row, 0], -ref_pts[row, 1], ref_pts[row, 1], ref_pts[row, 0]]).reshape((2,2))
    #
    pts_to_be_warped = pts_to_warp.reshape((-1,1))
    X,_,_,_ =np.linalg.lstsq(M, pts_to_be_warped, rcond=None)
    Rot = np.array([X[2],-X[3], X[3], X[2]]).reshape((2,2))
    new_grids = {}
    
    for key, value in polygons.items():
        new_grids[key] = np.zeros_like(value)
        for i in range(value.shape[0]):
            # for j in range(value.shape[1]):
            pt_grid = value[i,:].reshape((2,1))
            new_grids[key][i,:]=(X[:2] + np.matmul(Rot, pt_grid)).reshape((2,))
    return new_grids


def interpolate_polys(polygons, reference_pts, pts_to_warp, inter_method):
    xi=np.array([])
    yi=np.array([])
    for key, value in polygons.items():
        xi=np.append(xi, value[:,0])
        yi=np.append(yi,value[:,1])
    
    vec12 =pts_to_warp- reference_pts #transformation vectors at reference points
    x = np.copy(reference_pts[:,0])
    y = np.copy(reference_pts[:,1])

    zx = np.copy(vec12[:,0])
    zy = np.copy(vec12[:,1])
    Fx = griddata((x,y),zx,(xi,yi),method=inter_method, fill_value=0)
    Fy = griddata((x,y),zy,(xi,yi),method=inter_method, fill_value=0)
    plt.ion()
    plt.figure()
    plt.quiver(Fx,Fy, 10000,10000)
    plt.draw()
    # plt.pause(0.5)
    new_grids = {}
    
    for key, value in polygons.items():
        new_grids[key] = np.zeros_like(value)
        for i in range(value.shape[0]):
            pt_grid = value[i,:].reshape((2,1))
            new_grids[key][i,:]=np.array((pt_grid[0]+(Fx[np.where(xi==pt_grid[0])[0][0]]),pt_grid[1]+
                                          (Fy[ np.where(yi==pt_grid[1])[0][0]]))).reshape((2,))
    return new_grids


def interpolate_polys_thin_plate(polygons, reference_pts, pts_to_warp):
    xi=np.array([])
    yi=np.array([])
    for key, value in polygons.items():
        xi=np.append(xi, value[:,0])
        yi=np.append(yi,value[:,1])
    
    vec12 =pts_to_warp- reference_pts #transformation vectors at reference points
    x = np.copy(reference_pts[:,0])
    y = np.copy(reference_pts[:,1])
    
    zx = np.copy(vec12[:,0])
    zy = np.copy(vec12[:,1])
    rbfx=Rbf(x,y,zx, function='thin_plate')
    rbfy=Rbf(x,y,zy, function='thin_plate')
    Fx=rbfx(xi,yi)
    Fy=rbfy(xi,yi)

    new_grids = {}
    
    for key, value in polygons.items():
        new_grids[key] = np.zeros_like(value)
        for i in range(value.shape[0]):
            pt_grid = value[i,:].reshape((2,1))
            new_grids[key][i,:]=np.array((pt_grid[0]+(Fx[np.where(xi==pt_grid[0])[0][0]]),pt_grid[1]+
                                          (Fy[ np.where(yi==pt_grid[1])[0][0]]))).reshape((2,))
    return new_grids


def main(field, date_var, interpolation_method='nearest', transformation_method='affine'):
    # The reference date's polygons (Step7_1) are warped onto date_var using the
    # two dates' markers, as put in corresponding order by Step6.
    ref_date=reference_date(field)
    if date_var == ref_date:
        raise ValueError(f'{date_var} is the reference date; its polygons come from make-polygons')
    reference_points=np.load(sorted_markers_path(field, ref_date))
    pts_to_be_warped=np.load(sorted_markers_path(field, date_var))
    polygons=np.load(polygons_path(field, ref_date), allow_pickle=True).item()
    if transformation_method=='interpolation':
        new_grids=interpolate_polys_thin_plate(polygons, reference_points, pts_to_be_warped)

    elif transformation_method=='affine':
        
        new_grids=aff_transform(polygons,reference_points, pts_to_be_warped)

    np.save(polygons_path(field, date_var), new_grids)

    # overlay of the warped polygons on the target image, saved for checking
    im0=load_image(image_path(field, date_var))
    plt.ion()
    plt.figure()
    plt.imshow(im0)
    for i, value in new_grids.items():
        plt.plot(*zip(*value), 'r')
    plt.title(date_var)
    plt.savefig(os.path.join(results_dir(field), f'Alignment_{field}_{date_var}'))
    plt.close()
    
    return new_grids



# This module is a pipeline stage: its main() is called from cli.py, not
# run directly.
