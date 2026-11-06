import numpy as np
import matplotlib.pyplot as plt
from field_paths import image_path, markers_path, reference_date, sorted_markers_path
from field_cv import load_image, wait_for_close


def sort_markers (marks):
    '''Choose marker 0'''
    sorted_markers=sorted(marks , key=lambda k: [k[1], k[0]])
    marker_0=sorted_markers[0]
    row_marker_0=np.where(marks==marker_0)[0][0]
    
    other_markers=np.r_[marks[0:row_marker_0],marks[row_marker_0+1:]]
    
    '''Build dictionary of distances'''
    distances=np.zeros((other_markers.shape[0]))
    i=0
    for elt in other_markers:
        distances[i]=np.linalg.norm(elt-marker_0)
        i+=1
    distances=distances/np.max(distances)
    other_markers = other_markers[distances.argsort()]
    
    sort_marks=np.append(marker_0, other_markers)
    sort_marks=sort_marks.reshape((-1,2))
    distances=sorted(distances)
    distances=np.round(distances,3)
    return sort_marks, distances


def show_correspondence(ref_im, sort_standard, target_im, sort_nonref):
    """Both dates side by side, markers numbered in their sorted order."""
    plt.ion()
    fig, ax=plt.subplots(1,2)
    ax[0].imshow(ref_im)
    for ix in range(len(sort_standard)):
        ax[0].plot(sort_standard[ix,0],sort_standard[ix,1],'.',color='r')
        ax[0].annotate(str(int(ix)), xy=(sort_standard[ix,0],sort_standard[ix,1]), xycoords='data')

    ax[1].imshow(target_im)
    for ix in range(len(sort_nonref)):
        ax[1].plot(sort_nonref[ix,0],sort_nonref[ix,1],'.',color='r')
        ax[1].annotate(str(int(ix)), xy=(sort_nonref[ix,0],sort_nonref[ix,1]), xycoords='data')
    wait_for_close('Compare the marker numbering, then close the figure window.')


def main(field, date_var):
    # Markers clicked in Step2 on the target date are put in the same order as
    # the reference date's, so Step7_2 can fit a transform between them.
    ref_date = reference_date(field)
    if date_var == ref_date:
        raise ValueError(f'{date_var} is the reference date; sort a different date against it')
    non_ref_markers_path = markers_path(field, date_var)
    ref_markers_path = markers_path(field, ref_date)
    sorted_markers_out = sorted_markers_path(field, date_var)

    '''Load markers file'''

    non_ref_markers = np.load(non_ref_markers_path)
    ref_markers = np.load(ref_markers_path)


    if len(non_ref_markers)>len(ref_markers):
        # non_ref_markers=non_ref_markers[-(len(ref_markers)-1):]
        non_ref_markers=non_ref_markers[-len(ref_markers):]

    sort_standard, distance_standard =sort_markers(ref_markers)
    sort_nonref, distance_nonref=sort_markers(non_ref_markers)
    ref_im = load_image(image_path(field, ref_date))
    target_im = load_image(image_path(field, date_var))

    show_correspondence(ref_im, sort_standard, target_im, sort_nonref)
    correspondence_check=input('Is the correspondence between markers correct? (Yes/No):')
    if correspondence_check.upper()=='YES':
        np.save(sorted_markers_path(field, ref_date), sort_standard)
        np.save(sorted_markers_out, sort_nonref)
    else:
        switch_or_scramble=input('Is it sufficient to switch the place of two markers? (Yes/No):')
        if switch_or_scramble.upper()=='YES':
            indices=input('Give the indices that need to be swapped separated by a comma (i.e. ix1,ix2):')
            list_indices=indices.split(',')
            pt_ix_1=sort_nonref[int(list_indices[0])].copy()
            pt_ix_2 = sort_nonref[int(list_indices[1])].copy()
            sort_nonref[int(list_indices[0])]=pt_ix_2
            sort_nonref[int(list_indices[1])]=pt_ix_1
            plt.close('all')
            show_correspondence(ref_im, sort_standard, target_im, sort_nonref)
            correspondence_check = input('Is the correspondence between markers correct? (Yes/No):')
            if correspondence_check.upper() == 'YES':
                np.save(sorted_markers_path(field, ref_date), sort_standard)
                np.save(sorted_markers_out, sort_nonref)



    return sort_standard, sort_nonref


    



# This module is a pipeline stage: its main() is called from cli.py, not
# run directly.




