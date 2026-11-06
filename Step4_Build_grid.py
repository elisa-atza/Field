import numpy as np
import scipy.ndimage as nd
import cv2
import os
import multiprocessing as mp
import time
import itertools
import matplotlib.pyplot as plt
from field_cv import compute_ridge_lateral_bdry, diagonal_sorting, extractHueSat, load_image
from field_paths import field_config, grid_path, image_path, quadrant_path, reference_date, results_dir


def build_save_grid(image, tops, bots, mid_t, mid_b, i,date,folder,field, LpR):
    # for top, bot in zip(list_of_tops, list_of_bots):
        numR = len(tops)
        print(numR, 'len tops')


        avg_periodic = get_avg_periodic(tops, bots, field)
        grid = np.zeros((LpR+2, len(tops), 2))
        for ii in range(0, numR):
            print('processing ridge', ii+1)
            # detecting all gaps in a ridge
            gaps = getGaps(image,tops, bots,ii+1)

            gapsf=filterGaps_with_avg_periodic( gaps,avg_periodic, ii+1)

            print('filtered')
            grid[:LpR+1, ii] = gapsf
        np.save(grid_path(field, i), grid)


def par_build_save_grid(image, tops, bots, mid_t, mid_b, i,date,folder,field, LpR):

    numR = len(tops)
    print(numR, 'len tops', mid_t.shape, 'shape midt')

    numProbs=numR
    numCPU = numR#int(numR/2) # int(mp.cpu_count()) #
    numProbsPerCPU = int(np.ceil(numProbs / numCPU))
    tic = time.time()
    # split the ridge indices into blocks of numProbsPerCPU, one per worker
    indices = list(range(numProbs))
    indList = [indices[k:k + numProbsPerCPU] for k in range(0, numProbs, numProbsPerCPU)]
    # running the loop in parallel

    pool = mp.Pool(numCPU)
    XList = pool.starmap(par_inner_function, zip(indList, itertools.repeat(tops),itertools.repeat(bots), itertools.repeat(mid_t), itertools.repeat(mid_b),itertools.repeat(LpR),itertools.repeat(field),itertools.repeat(image)))  # run on all workers, inList goes in, Xlist comes out
    pool.close()
    toc = time.time()
    timepar = toc - tic
    grid=np.concatenate(XList,axis=1)
    print(grid.shape,'shape of joined grid')
    print('parallel loop, time:', timepar/60)
    np.save(grid_path(field, i), grid)



def par_inner_function(ind, tops, bots, mid_t, mid_b, LpR, field, image):
    grid = np.zeros((LpR + 2, len(ind), 2))
    avg_periodic=get_avg_periodic(tops,bots, field)
    k=0
    for ii in ind:
        print('processing ridge', ii+1)
        # detecting all gaps in a ridge
        # getGaps uses the ridge index ii to identify where to look in the arrays midt and midb
        gaps = getGaps(image,tops, bots,ii+1)
        print('gotten')
        gapsf = filterGaps_with_avg_periodic( gaps,avg_periodic, ii+1)
        print('filtered')
        # change column indexing to local k, ii is too big
        grid[:LpR+1, k] = gapsf  #gapsf
        k+=1
    return grid


def extractLine(img,x0,x1,y0,y1,numPts):
    '''This function reads pixel values along the line segment (x0,y0)->(x1,y1)
    across the RGB image img and interpolates them over numPts equidistant
    points. Returns (3 x numPts)!!! array of pixel values
    '''
    ## Read pixels along a line with "numPts" points...
    x, y = np.linspace(x0, x1, numPts), np.linspace(y0, y1, numPts)
    # Extract the values along the line, using cubic interpolation
    z0 = nd.map_coordinates(img[:,:,0], np.vstack((y,x)))
    z1 = nd.map_coordinates(img[:,:,1], np.vstack((y,x)))
    z2 = nd.map_coordinates(img[:,:,2], np.vstack((y,x)))
    z = np.vstack((z0,z1,z2))
    return z


def get_avg_periodic(tops,bots,field):
    field_direction = {'M': 0, 'S': 0, 'V': 1}
    periodic_gaps=np.zeros((31,len(tops),2))
    for ridge in range(len(tops)):
        gapsp = np.zeros((30+1,2))
        gapsp[:,1] = np.linspace(tops[ridge,1],bots[ridge,1],30+1)
        gapsp[:,0] = np.linspace(tops[ridge,0],bots[ridge,0],30+1)
        periodic_gaps[:,ridge,:]=gapsp
    
    avg_periodic_gaps=periodic_gaps.copy()
    for column in range(int(len(tops)/4)):
        for row in range(31):
            avg_periodic_gaps[row,column*4:column*4+4,field_direction[field]]=periodic_gaps[row,column*4:column*4+4,field_direction[field]].mean()
    return avg_periodic_gaps


def getGaps(img,tops,bots,ridge):
    '''This function takes in a binary image, tops, bots, and the ridge number,
    detects all zero gaps, and returns coordinates of gaps' midpoints.

    The signature used to be (img, mid_ridge_l, mid_ridge_r, tops, bots, ridge),
    but both call sites passed tops and bots into mid_ridge_l/mid_ridge_r and the
    last two arrays were never read. Renamed to match what is actually passed.
    '''
    numPts = 2500 # number of points along a ridge
    # x- and y-shift between ridges

    dx0 = tops[(int((ridge-1)/4)*4)+1,0]-tops[(int((ridge-1)/4)*4),0]
    dx1 = bots[(int((ridge-1)/4)*4)+1,0]-bots[(int((ridge-1)/4)*4),0]
    dy0 = tops[(int((ridge-1)/4)*4)+1,1]-tops[(int((ridge-1)/4)*4),1]
    dy1 = bots[(int((ridge-1)/4)*4)+1,1]-bots[(int((ridge-1)/4)*4),1]
    # # main ridge
    x0 = tops[ridge-1,0]
    y0 = tops[ridge-1,1]
    x1 =bots[ridge-1,0]
    y1 = bots[ridge-1,1]
    plt.plot((x0, x1),(y0,y1), 'ro-')

    # print ('cords main ridge', x0, y0, x1, y1)
    ridgeMain = extractLine(img,x0,x1,y0,y1,numPts)
   
    # left shifted line
    x0left = x0-dx0/8  # 8
    y0left = y0-dy0/8  # 8
    x1left = x1-dx1/8  # 8
    y1left = y1-dy1/8  # 8
    # print('ccords left line', x0left, y0left, x1left, y1left)
    # right shifted line
    leftLine = extractLine(img,x0left,x1left,y0left,y1left,numPts)
   
    x0right = x0+dx0/8  # 8
    y0right = y0+dy0/8  # 8
    x1right = x1+dx1/8  # 8
    y1right = y1+dy1/8  # 8
    rightLine = extractLine(img,x0right,x1right,y0right,y1right,numPts)
    ridgeMain=ridgeMain/255
    leftLine=leftLine/255
    rightLine=rightLine/255
    # looking at coinciding zeros in three lines
    ridgeLine = np.mean(leftLine, axis=0)+np.mean(ridgeMain, axis = 0)+np.mean(rightLine, axis=0)

    divs = []
    startGap = 0
    endGap = 0
    numPts = len(ridgeLine)

    for ii in range(numPts-1):
        if (ridgeLine[ii] == 0) and (ridgeLine[ii+1] > 0):
            endGap = ii

        elif (ridgeLine[ii] > 0) and (ridgeLine[ii-1] == 0):
            if startGap == 0:
                kk = endGap-2 # index for first gap
            else:
                kk = (startGap+endGap)/2 # mid-point index
            x = x0 + (kk/(numPts-1))*(x1-x0)
            y = y0 + (kk/(numPts-1))*(y1-y0)
            divs.append(np.array([x,y, np.abs((endGap-startGap))]))

        elif ridgeLine[ii] > 0:
            startGap = ii+1
    kk = startGap+2 # index for last gap
    x = x0 + (kk/(numPts-1))*(x1-x0)
    y = y0 + (kk/(numPts-1))*(y1-y0)
    divs.append(np.array([x,y, np.abs((endGap-startGap))]))
    divs = np.array(divs)
    return divs


def filterGaps(image, gaps,numL):
    '''This function filters the array produced by getGaps() by choosing or
    inserting the points closest to the periodic (equispaced) grid along the
    ridge with a given number (numL) of lots.
    '''
    gapsp = np.zeros((numL+1,2))
    gapsp[:,0] = np.linspace(gaps[0,0],gaps[-1,0],numL+1)
    gapsp[:,1] = np.linspace(gaps[0,1],gaps[-1,1],numL+1)
    numG = len(gaps)
    numGp = len(gapsp)
    gapsc = np.zeros((numGp,2))
    pdist = np.sqrt((gapsp[1,0]-gapsp[0,0])**2
                   +(gapsp[1,1]-gapsp[0,1])**2)
    for ii in range(numGp):
        xp,yp = gapsp[ii,:]
        dist = np.zeros(numG)
        for jj in range(numG):
            x,y = gaps[jj,:-1]
            dist[jj] = np.sqrt((xp-x)**2+(yp-y)**2)
        mindist = np.min(dist)
        kk = np.argmin(dist)
        if mindist <= 0.35*pdist: # keep closest point, pdist is the periodic distance

            gapsc[ii,:] = gaps[kk,:-1]
        elif mindist > 0.35*pdist: # insert point from periodic grid
            x1, y1 = gaps[kk, :-1]

            if np.all(image[int(yp),int(xp),:]!=(0,0,0)) and np.all(image[int(y1),int(x1),:]==(0,0,0)):
                print('gaps kept')
                #keep the gaps
                gapsc[ii, :] = gaps[np.argmin(dist),:-1]
            elif np.all(image[int(yp),int(xp),:]!=(0,0,0)) and np.all(image[int(y1),int(x1),:]!=(0,0,0)):
                #keep the periodic
                print('periodic kept')
                gapsc[ii, :] = gapsp[ii, :]
            else:
                gapsc[ii,:] = gapsp[ii,:]

    return gapsc, gapsp


def filterGaps_with_avg_periodic(gaps,avg_periodic, ridge):
    '''This function filters the array produced by getGaps() by choosing or
    inserting the points closest to the periodic (equispaced) grid along the
    ridge with a given number (numL) of lots.
    '''

    ridge=ridge-1
    numG = len(gaps)
    numGp = avg_periodic.shape[0]
    gapsc = np.zeros((numGp,2))
    pdist = np.sqrt((avg_periodic[1,ridge, 0]-avg_periodic[0,ridge, 0])**2
                    +(avg_periodic[1,ridge, 1]-avg_periodic[0,ridge, 1])**2)
    distance_coeff=0.35
    for ii in range(numGp):
        xp,yp = avg_periodic[ii,ridge,:]
        dist = np.zeros(numG)
        for jj in range(numG):
            x,y = gaps[jj,:-1]
            dist[jj] = np.sqrt((xp-x)**2+(yp-y)**2)
        mindist = np.min(dist)
        kk = np.argmin(dist)
        if mindist <=distance_coeff*pdist: # keep closest point, pdist is the periodic distance
            
            gapsc[ii,:] = gaps[kk,:-1]
            # print(gaps[kk,-1])
        elif mindist > distance_coeff*pdist: # insert point from periodic grid

            gapsc[ii,:] =  avg_periodic[ii,ridge,:]

    
    return gapsc



def main(field):
    # grids are built on the reference date, from the Quadrants files Step3
    # wrote for that date
    LpR=field_config(field)['lots_per_ridge']  #Number of lots per ridge IN THE QUADRANT
    folder=results_dir(field)
    date=reference_date(field)
    image=load_image(image_path(field, date))
    script_result_folder = os.path.join(folder, 'Grids')
    os.makedirs(script_result_folder, exist_ok=True)
    imbw = extractHueSat(image, [35, 150, 70, 150])  # all non-green set to 0
    imbw[imbw > 0] = 255
    
    # Morphological  operations on teh binary image to get rid of stray white pixels,
    # we can be relatively rough, in order to clear the gaps
    kernel = np.ones((4,4), np.uint8)
    imgd = cv2.dilate(imbw, kernel, iterations=5)
    kernel=np.ones((4,3), np.uint8)
    img=cv2.erode(imgd, kernel,iterations=4)
    kernel = np.ones((2,2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=2)
    # plt.ion()
    # plt.figure()
    # plt.imshow(img)

    for i in range(1,5):

        tops = np.load(quadrant_path(field, i, 'tops'))
        bots = np.load(quadrant_path(field, i, 'bots'))
        tops= tops[~np.all(tops==0, axis=1)]
        bots= bots[~np.all(bots==0, axis=1)]
        # tops=diagonal_sorting(tops, field)
        # bots=diagonal_sorting(bots, field)
        left_lat = compute_ridge_lateral_bdry(tops)
        right_lat = compute_ridge_lateral_bdry(bots)
    
        if os.path.isfile(quadrant_path(field, i, 'mids')):
            mids=np.load(quadrant_path(field, i, 'mids'))
            mids = mids[~np.all(mids == 0, axis=1)]
            # mids = diagonal_sorting(mids, field)
            mid_lat= compute_ridge_lateral_bdry(mids)
            par_build_save_grid(img, tops, mids, left_lat, mid_lat,  f'{i}_1', date, script_result_folder, field,LpR)
            par_build_save_grid(img, mids, bots, mid_lat, right_lat, f'{i}_2', date, script_result_folder,field, 30)
        else:
            par_build_save_grid(img, tops, bots,  left_lat, right_lat, i, date, script_result_folder,field, LpR)


# This module is a pipeline stage: its main() is called from cli.py, not
# run directly.
