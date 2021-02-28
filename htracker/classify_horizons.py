import numpy as np
import pandas as pd
import scipy

from pylops.basicoperators import *
from pylops.signalprocessing import *
from pylops.utils.wavelets import *
from pylops.avo.poststack import *
from pylops.optimization.sparsity import *


def get_horizon_id(h_df,
                   line_id,
                   segpd_classes,
                   lb_lenclasscompute=50,
                   lb_lenclasswindow=10,
                   lb_idmethod='combined'
                   ):
    """Clean image and compute total variation

    Parameters
    ----------
    h_df : dataframe
        Selected points that make up a horizon
    line_id : dataframe
        Selected points that make up the horizon
    segpd_classes : np array
        Segmented classes
    lb_lenclasscompute : int
        Number of points used to determine line class
    lb_lenclasswindow : int
        Number of pixels of window above and below from which to sample values for
        line class labelling
    lb_idmethod : str
        Nethod for computing horizon identifier, choose from 'above', 'below', 'combined'

    Returns
    -------
    cl_tv : np array
        TV image for desired class
    cl_classimage : np.array
        cleaned, masked class image used as input to TV computation

    """

    # CHOOSE POINTS FOR COMPUTE CLASS STATS
    if len(h_df) < lb_lenclasscompute:
        tmp_df = h_df
    else:
        tmp_df = h_df.iloc[np.random.randint(len(h_df), size=lb_lenclasscompute).tolist()]

    # FIND CLASS VALUES ABOVE AND BELOW COMPUTE POINTS
    pointsaround = lb_lenclasswindow
    cl_ab = []
    cl_bl = []

    for i, r in tmp_df.iterrows():
        cl_ab.append(segpd_classes[int(r.y) - pointsaround:int(r.y), int(r.x)])
        cl_bl.append(segpd_classes[int(r.y):int(r.y) + pointsaround, int(r.x)])

    cl_above_ls = [item for sublist in cl_ab for item in sublist]
    cl_below_ls = [item for sublist in cl_bl for item in sublist]

    # CREATING HORIZON ID BASED ON CHOSEN ID METHOD
    if lb_idmethod=='above':
        # Take mode
        cl_above_mode, _ = scipy.stats.mode(np.array(cl_above_ls))
        # Safety check that mode never bigger than 1 class
        if len(cl_above_mode) > 1:
            raise ValueError("too many values from mode for above class")
        else:
            cl_above = cl_above_mode[0]
        h_id = "a%i" % (cl_above)
    elif lb_idmethod=='below':
        # Take mode
        cl_below_mode, _ = scipy.stats.mode(np.array(cl_below_ls))
        # Safety check that mode never bigger than 1 class
        if len(cl_below_mode) > 1:
            raise ValueError("too many values from mode for below class")
        else:
            cl_below = cl_below_mode[0]
        h_id = "b%i" % (cl_below)
    elif lb_idmethod=='combined':
        # Take modes
        cl_above_mode, _ = scipy.stats.mode(np.array(cl_above_ls))
        cl_below_mode, _ = scipy.stats.mode(np.array(cl_below_ls))

        # Safety check that mode never bigger than 1 class
        if len(cl_above_mode) > 1:
            raise ValueError("too many values from mode for above class")
        else:
            cl_above = cl_above_mode[0]
        if len(cl_below_mode) > 1:
            raise ValueError("too many values from mode for below class")
        else:
            cl_below = cl_below_mode[0]
        h_id = "a%ib%i" % (cl_above, cl_below)

    # ADD HORIZON ID TO HORIZON DATAFRAME
    h_df['line_id'] = line_id
    h_df['horizon_id'] = h_id

    return h_df


def get_classified_horizons(h_df_list,
                            segpd_classes,
                            lb_lenclasscompute=50,
                            lb_lenclasswindow=5,
                            lb_idmethod='combined'
                            ):
    cl_hdf_list = []

    for h in range(len(h_df_list)):
        cl_hdf_list.append(get_horizon_id(h_df_list[h], h,
                                          segpd_classes,
                                          lb_lenclasscompute,
                                          lb_lenclasswindow,
                                          lb_idmethod
                                          ))
    if len(cl_hdf_list) == 0:
        cl_hdf = None
    else:
        cl_hdf = pd.concat(cl_hdf_list, ignore_index=True)

    return cl_hdf


def join_horizons(classified_df, xjointhresh=100, yjointhresh=10):

    classified_df['segment_id'] = -1

    for u_id in classified_df['horizon_id'].unique():
        # identify edges of each line
        tmp_df = classified_df[classified_df['horizon_id'] == u_id]
        line_ids = tmp_df['line_id'].unique()
        lines = [tmp_df[tmp_df['line_id'] == i_line] for i_line in line_ids]
        left_edges = np.array(
            [list(l.iloc[l['x'].argmin()].loc[['x', 'y']]) for l in lines])
        right_edges = np.array(
            [list(l.iloc[l['x'].argmax()].loc[['x', 'y']]) for l in lines])
        line_order = np.argsort(left_edges[:, 0])
        nlines = len(lines)

        # reorder lines in monotonically increasing order based on left edge
        line_ids = line_ids[line_order]
        left_edges = left_edges[line_order]
        right_edges = right_edges[line_order]
        line_order = np.arange(nlines)

        # label lines in segment groups
        avail_lines = np.full(nlines, True)
        isegment = 0
        while np.sum(avail_lines) > 1:
            icurrent = line_order[avail_lines][0]
            classified_df['segment_id'][(classified_df['horizon_id'] == u_id)
                                        & (classified_df['line_id'] == line_ids[icurrent])] = isegment
            for inext in range(icurrent + 1, icurrent + np.sum(avail_lines[icurrent:])):
                # join if closer than threshold in a and y
                if (avail_lines[inext]) and \
                        (np.abs(right_edges[icurrent, 0] - left_edges[inext, 0]) < xjointhresh) and \
                        (np.abs(right_edges[icurrent, 1] - left_edges[inext, 1]) < yjointhresh):
                    avail_lines[icurrent] = False
                    icurrent = inext
                    classified_df['segment_id'][(classified_df['horizon_id'] == u_id) &
                                                (classified_df['line_id'] == line_ids[icurrent])] = isegment
                else:
                    avail_lines[icurrent] = False
            isegment += 1
        classified_df['segment_id'][(classified_df['horizon_id'] == u_id) &
                                    (classified_df['segment_id'] == -1)] = isegment

    return classified_df
