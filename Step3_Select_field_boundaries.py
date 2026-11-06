import numpy as np
import matplotlib.pyplot as plt
from field_cv import load_image, wait_for_close
from field_paths import image_path, quadrant_path, reference_date


def main(field, quadrant=None):
    # quadrants are only ever selected on the reference date; the other
    # dates get their polygons warped from it (Step7_2)
    date=reference_date(field)
    image=load_image(image_path(field, date))


    # # Fix the number of ridges, in our case different quadrants have different numbers of ridges

    if quadrant:
        quadrant = str(quadrant)
    else:
        quadrant = input('quadrant number (1-4, as build-grid reads them):')
    print('quadrant', quadrant)
    NumR = 4  # set here the expected number of ridges between two selected points
    curve = []
    tops = np.zeros((24, 2))
    bots = np.zeros((24, 2))
    plt.ion()

    fig = plt.figure()
    plt.imshow(image)
    plt.title(date)

    # #-------------------------corner selection-------------------------------------#


    def select_poly(event):
        """Callback function for the figure.
        Zoom to the required area, single right-click on the image adds one point
        to the polygon. Zoom on the top part of the ridge block, select a point at the
        leftmost ridge just above the green, then do the same at the rightmost ridge,
        then zoom at the bottom part of the ridge block, select left, right as above.
        Double click on the image to finish. Corner point coordinates are in the
        "curve" array, top and bottom ridge coordinates are in the "tops" and "bots"
        arrays.
        """
        nonlocal curve  # the other names are only read, not rebound
        if event.inaxes is not None:
            if event.button == 3 : #and len(curve)<NumR
                xc = int(event.xdata)
                yc = int(event.ydata)
                print('x =', xc, ' y =',yc)
                plt.plot(xc,yc,'r.')
                plt.show()
                curve.append([xc,yc])
            elif event.button==2:  # len(curve) == NumR
                print('all corners selected, disconnecting...')
                curve = np.array(curve)
                print('curve shape', curve.shape)
                if len(curve)<12:
                    print('you are selecting mid points')
                    j2=0
                    for inx in range(0, len(curve), 2):
                        bots[j2:j2+4,0] = np.linspace(curve[inx,0],curve[inx+1,0],NumR)
                        bots[j2:j2+4,1] = np.linspace(curve[inx,1],curve[inx+1,1],NumR)
                        j2+=4
                    plt.plot(bots[:,0],bots[:,1],'b.')
                    fig.canvas.mpl_disconnect(cid)
                    np.save(quadrant_path(field, quadrant, 'mids'), bots)
                    plt.close(fig)
                else:

                    j=0
                    for inx in range(0, int(len(curve)/2),2):
                        tops[j:j+4,0] = np.linspace(curve[inx,0],curve[inx+1,0],NumR)
                        tops[j:j+4,1] = np.linspace(curve[inx,1],curve[inx+1,1],NumR)
                        j+=4
                    j2=0
                    for inx in range(int((len(curve)/2)), len(curve),2):
                        bots[j2:j2+4,0] = np.linspace(curve[inx,0],curve[inx+1,0],NumR)
                        bots[j2:j2+4,1] = np.linspace(curve[inx,1],curve[inx+1,1],NumR)
                        j2+=4
                    plt.plot(tops[:,0],tops[:,1],'b.')
                    plt.plot(bots[:,0],bots[:,1],'b.')
                    # finding the end-points of ridge lines by uniform interpolation

                    fig.canvas.mpl_disconnect(cid)
                    np.save(quadrant_path(field, quadrant, 'bots'), bots)
                    np.save(quadrant_path(field, quadrant, 'tops'), tops)
                    plt.close(fig)
                    # print('Do not forget to save ridge coordinates (tops, bots)!')
                return
        else:
            print('Clicked ouside axes bounds but inside plot window')

    cid = fig.canvas.mpl_connect('button_press_event', select_poly)
    wait_for_close('Right-click ridge corners, middle-click to save and finish.')


# This module is a pipeline stage: its main() is called from cli.py, not
# run directly.
