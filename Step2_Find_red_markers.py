import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from field_cv import load_image, wait_for_close
from field_paths import image_path, markers_path, results_dir


def get_markers(img_in):
    #convert image to hsv
    im1hsv = cv2.cvtColor(img_in, cv2.COLOR_RGB2HSV)
    #set hsv bounds for red
    # d1h, d2h, d1s, d2s, d1v, d2v = 120, 180, 70, 255, 100, 255  # d1v was 180
    markers = np.copy(img_in)
    
    # '''from internet, takes into account that it's a color wheel'''
    lower1 = np.array([0, 100, 100])
    upper1 = np.array([7, 255, 255])

    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160,100,100])
    upper2 = np.array([179,255,255])

    lower_mask = cv2.inRange(im1hsv, lower1, upper1)
    upper_mask = cv2.inRange(im1hsv, lower2, upper2)

    full_mask = lower_mask + upper_mask

    result = cv2.bitwise_and(markers, markers, mask=full_mask)
    kernel = np.ones((2,2), np.uint8)
    img_erosion = cv2.erode(result, kernel, iterations=1)

    # find connected components and centroids of "red" stuff
    markers = cv2.cvtColor(img_erosion, cv2.COLOR_RGB2GRAY)
    num,lab, stat, cent = cv2.connectedComponentsWithStats(markers,connectivity=8)
    # peaks = np.array(np.nonzero(markers[:,:,0])).transpose()
    cent = np.fliplr(cent)
    return markers, cent



def find_red_markers_and_select(image, result_folder_base):

    script_result_folder=os.path.join(result_folder_base, 'Markers')
    os.makedirs(script_result_folder, exist_ok=True)
    _,marks=get_markers(image)
    return script_result_folder, marks


def main(field, date):
    image=load_image(image_path(field, date))
    folder=results_dir(field)
    points=[]
    script_result_folder, marks=find_red_markers_and_select(image, folder)

    plt.ion()
    fig1, ax2=plt.subplots(nrows=1, ncols=1, sharey=True, sharex=True, num=1)
    ax2.imshow(image)
    ax2.plot(marks[:,1],marks[:,0],'.',color='r')
    ax2.set_title(str(len(marks)))
    def select_points(event):

        """Callback function for the figure.
        Zoom to the required area, single right-click on the image adds one point
        to the polygon. Zoom on the top part of the ridge block, select a point at the
        leftmost ridge just above the green, then do the same at the rightmost ridge,
        then zoom at the bottom part of the ridge block, select left, right as above.
        Double click on the image to finish. Corner point coordinates are in the
        "curve" array, top and bottom ridge coordinates are in the "tops" and "bots"
        arrays.
        """
        # names from main(); only `points` is mutated here
        if event.inaxes is not None:
            if event.button == 3:
                xc = int(event.xdata)
                yc = int(event.ydata)
                print('x =', xc, ' y =',yc)
                plt.plot(xc,yc,'b.')
                plt.show()
                points.append([xc,yc])
            elif event.button == 1 and event.key== "shift":  # if you misclicked you cn undo the last point with middle click
                del points[-1]
                print('undone')
            elif  event.button == 2:  #  and event.dblclick
               print('you double_clicked')
               np.save(markers_path(field, date), points)
               print('saved', markers_path(field, date))
               fig1.canvas.mpl_disconnect(cid)
               plt.close(fig1)
               return
        else:
            print('Clicked outside axes bounds but inside plot window')

    cid = fig1.canvas.mpl_connect('button_press_event', select_points)
    wait_for_close('Right-click markers, shift+left-click to undo, middle-click to save and finish.')



# This module is a pipeline stage: its main() is called from cli.py, not
# run directly.



