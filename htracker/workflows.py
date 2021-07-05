import numpy as np
import pandas as pd
import datetime

from scipy.signal import filtfilt
from pylops.basicoperators import *
from pylops.signalprocessing import *
from pylops.utils.wavelets import *
from pylops.avo.poststack import *
from pylops.optimization.sparsity import *

from .get_horizons import *
from .classify_horizons import *


def interpolate_horizon(cl_hlist):
    interp_cl_hlist = []

    for h in cl_hlist:
        # interp
        xo = h['x'].values
        yo = h['y'].values

        xnew = np.arange(min(xo), max(xo))
        ynew = np.interp(xnew, xo, yo)

        # interpolate probability
        pnew = np.interp(xnew, xo, h['prob'].values)

        # Put it all back into DF
        iDF = pd.DataFrame({'x': xnew, 'y': ynew, 'prob': pnew})

        interp_cl_hlist.append(iDF)
    return interp_cl_hlist


def detect_and_classify(segpd_classes,
                        cl_index,
                        spurioseventsize=250,
                        incl_erosion=False,
                        erosion_size=3,
                        tvlowerthresh=0.4,
                        nn_lim=3,
                        line_minlen=15,
                        lb_lenclasscompute=10,
                        lb_lenclasswindow=5,
                        lb_idmethod='combined',
                        xjointhresh=100,
                        yjointhresh=10,
                        hor_minlen=25,
                        verbose=False
                        ):
    # make tv
    s = datetime.datetime.now()
    cl_tv, _ = get_class_tv(segpd_classes,
                            class_index=cl_index,
                            spurioseventsize=spurioseventsize,
                            incl_erosion=incl_erosion,
                            erosion_size=erosion_size)
    if verbose:
        print('Compute TV Completed: %s sec' % str(datetime.datetime.now() - s))

    # get horizon points
    s = datetime.datetime.now()
    horizon_points_raw = get_horizon_points(cl_tv, tvlowerthresh=tvlowerthresh)
    if verbose:
        print('Get horizon points Completed: %s sec' % str(datetime.datetime.now() - s))

    # get separated horizons
    s = datetime.datetime.now()
    cl_hlist = get_seperated_lines(horizon_points_raw, cl_tv,
                                   nn_lim=nn_lim, line_minlen=line_minlen)
    if verbose:
        print('Get separated horizons Completed: %s sec' % str(
            datetime.datetime.now() - s))

    # interpolate in X all horizons
    s = datetime.datetime.now()
    interp_cl_hlist = interpolate_horizon(cl_hlist)
    if verbose:
        print('Interpolate horizons Completed: %s sec' % str(
            datetime.datetime.now() - s))

    # classify horizons detected based on classes above and below
    s = datetime.datetime.now()
    classified_df = get_classified_horizons(interp_cl_hlist, segpd_classes,
                                            lb_lenclasscompute=lb_lenclasscompute,
                                            lb_lenclasswindow=lb_lenclasswindow,
                                            lb_idmethod=lb_idmethod
                                            )
    if classified_df is None:
        return None, None, cl_tv

    if verbose:
        print('Label horizons Completed: %s sec' % str(
            datetime.datetime.now() - s))

    # join horizons based on class and vicinity
    s = datetime.datetime.now()
    classified_df = join_horizons(classified_df,
                                  xjointhresh=xjointhresh,
                                  yjointhresh=yjointhresh)
    if verbose:
        print('Join horizons Completed: %s sec' % str(
            datetime.datetime.now() - s))

    # For each unique horizon id, extract points & probability sum
    s = datetime.datetime.now()
    cl_joined_hlist = []
    for u_id in classified_df['horizon_id'].unique():
        tmp_df = classified_df[classified_df['horizon_id'] == u_id]

        for s_id in tmp_df['segment_id'].unique():
            s_tmp_df = tmp_df[tmp_df['segment_id'] == s_id]

            # Interpolate second time
            xo = s_tmp_df['x'].values
            yo = s_tmp_df['y'].values

            xnew = np.arange(0, cl_tv.shape[1])
            ynew = np.interp(xnew, xo, yo)

            empty_xs = np.setdiff1d(xnew, xo).astype(int).tolist()
            ynew[empty_xs] = np.nan

            if np.sum(~np.isnan(ynew)) > hor_minlen:
                # SAVE INTO DICT
                cl_joined_hlist.append({'h_id': u_id,
                                        's_id': s_id,
                                        'cl_id': cl_index,
                                        'hs_id': u_id+'_'+str(s_id),
                                        'prob_sum': np.sum(s_tmp_df['prob']),
                                        'x': s_tmp_df['x'],
                                        'y': s_tmp_df['y'],
                                        'regx': xnew,
                                        'regy': ynew,
                                        'chosen': True,
                                        })
    if verbose:
        print('Regredding horizons Completed: %s sec' % str(
            datetime.datetime.now() - s))

    return cl_joined_hlist, classified_df, cl_tv


def join_and_select(all_cl_hlist):
    unique_horizons = set([h['h_id'] for h in all_cl_hlist])

    for uh in unique_horizons:
        # print(uh)
        uh_index = [i for i, h in enumerate(all_cl_hlist) if h['h_id'] == uh]
        probs = np.array([all_cl_hlist[i]['prob_sum'] for i in uh_index])
        uh_selected = uh_index[np.argmax(probs)]
        # print(uh_selected)
        all_cl_hlist[uh_selected]['chosen'] = True

    return all_cl_hlist


def choose_horizons(all_cl_hlist, difflim=0.1):
    all_h_df = pd.DataFrame(all_cl_hlist)

    # find unique horizons
    horizon_labels = all_h_df['h_id'].unique()

    for hid in horizon_labels:  # horizon_labels:
        hdf = all_h_df[all_h_df["h_id"] == hid]
        hdf['lenx'] = [len(r['x']) for i, r in hdf.iterrows()]

        hdf.sort_values('lenx', ascending=False, inplace=True)

        while len(hdf) > 0:
            h_baseline = hdf.iloc[0]
            hdf.drop(hdf.iloc[0].name, inplace=True)

            drop_list = []
            for i, r in hdf.iterrows():

                common_x = np.intersect1d(h_baseline['x'], r['x']).astype('int')

                baseline_y = h_baseline['regy'][common_x]
                ref_hor_y = r['regy'][common_x]

                diff = np.mean(abs(baseline_y - ref_hor_y))

                if diff < difflim: # and len(baseline_y) > 50:
                    # Set to false in full horizon dataframe
                    all_h_df['chosen'].iloc[r.name] = False

                    # Add to list to be dropped after for loop
                    drop_list.append(r.name)

            # Drop now
            hdf.drop(drop_list, inplace=True)
    return all_h_df


def multiclass_horizons(ncl, segpd_classes,
                        spurioseventsize=250,
                        incl_erosion=False,
                        erosion_size=3,
                        tvlowerthresh=0.4,
                        nn_lim=3,
                        line_minlen=15,
                        lb_lenclasscompute=10,
                        lb_lenclasswindow=5,
                        lb_idmethod='combined',
                        xjointhresh=100,
                        yjointhresh=10,
                        hor_minlen=25,
                        difflim=0.1,
                        verbose=False,
                        ):
    """Horizon detection for multiple classes

    Parameters
    ----------
    ncl : int
        Number of classes
    segpd_classes : np array
        Segmented classes
    spurioseventsize : int
        size in pixels for small shape cleaning
    erode : bool
        whether to apply skimage erosion cleaning
    erosion_size : int
        size for skimage erosion cleaning
    tvlowerthresh : int
        TV threshold for horizon points
    nn_lim : int
        maximum distance limit on which to allow joining of points
    line_minlen : int
        minimum length of horizon line
    lb_lenclasscompute : int
        Number of points used to determine line class
    lb_lenclasswindow : int
        Number of pixels of window above and below from which to sample
        values for line class labelling
    lb_idmethod : str
        Method for computing horizon identifier, choose from 'above',
        'below', 'combined'
    xjointhresh : int
        Maximum separation in samples in x direction for two lines to be
        combined in a common segment
    yjointhresh : int
        Maximum separation in samples in z direction for two lines to be
        combined in a common segment
    hor_minlen : int
        Minimum lenght in samples for regridded horizon to be kept as final
    difflim : float
        Minimum allowed average absolute difference between two horizons with
        same label (when smaller, the shorter of the two horizons is discarded)

    Returns
    -------
    horizon_df : dataframe
        Set of interpreted horizons

    """
    all_cl_hlist_tmp = []
    for cl_index in range(ncl):
        if verbose:
            print('Class: %i' % cl_index)
            s = datetime.datetime.now()
            print('Start: %s' % str(s))
        cl_joined_hlist, classified_df, cl_tv = \
            detect_and_classify(segpd_classes,
                                cl_index,
                                spurioseventsize = spurioseventsize,
                                incl_erosion = incl_erosion,
                                erosion_size = erosion_size,
                                tvlowerthresh = tvlowerthresh,
                                nn_lim=nn_lim,
                                line_minlen=line_minlen,
                                lb_lenclasscompute=lb_lenclasscompute,
                                lb_lenclasswindow=lb_lenclasswindow,
                                lb_idmethod=lb_idmethod,
                                xjointhresh=xjointhresh,
                                yjointhresh=yjointhresh,
                                hor_minlen=hor_minlen,
                                verbose=verbose
                                )
        if cl_joined_hlist is not None:
            all_cl_hlist_tmp.append(cl_joined_hlist)
        if verbose:
            e = datetime.datetime.now()
            print('End: %s' % str(e))
            print('Duration: %s' % str(e - s))
            print(' ')

    if verbose:
        print('Combine and select horizons')
        s = datetime.datetime.now()
        print('Start: %s' % str(s))
    all_cl_hlist = [item for sublist in all_cl_hlist_tmp for item in sublist]
    horizon_df = choose_horizons(all_cl_hlist, difflim=difflim)

    return horizon_df


def uncertainty_horizons(segpd, horizon_list, hors_names,
                         tv_thresh=0.1, nwin=4, nsmooth=5):
    """Estimate horizon uncertainties based on segmentation probabilities

    Parameters
    ----------
    segpd : np array
        Segmentation probabilities
    horizon_list : pd.DataFrame
        Horizon list
    hors_names : list
        Horizons names
    tv_thresh : float
        TV thresholding
    nwin : int
        Size of vertical window used for averaging
    nsmooth : int
        Smoothing filter lenght

    Returns
    -------
    hors_unc : dict
        Horizons uncertainties
    tv_tot_raw : np.ndarray
        Total TV of probabilities

    """
    # compute overall TV norm
    nt0, nx, ncl = segpd.shape
    tv_tot_raw = np.zeros((nt0, nx))
    for icl in range(ncl):
        cl_tv = tv(segpd[..., icl])
        tv_tot_raw += cl_tv

    # create binarized tv map
    tv_tot = tv_tot_raw.copy()
    tv_tot[tv_tot_raw < tv_thresh] = 0.
    tv_tot[tv_tot_raw > tv_thresh] = 1.

    hors_unc = {}
    for ihor, horkey in enumerate(hors_names):
        horsel = [h for i, h in horizon_list.iterrows() if
                  h['chosen'] and h['h_id'] == horkey]
        horsel_uncs = []
        for hsel in horsel:
            horsel_unc = np.zeros(len(hsel['regy']))
            # average tv mask over time window
            valmask = ~np.isnan(hsel['regy'])
            horband = range(-nwin, nwin+1)
            for i in horband:
                iy = hsel['regy'][valmask].astype(int) + i
                iy[iy >= nt0] = nt0 - 1
                iy[iy < 0] = 0
                horsel_unc[valmask] += tv_tot[iy, hsel['regx'][valmask].astype(int)]
            horsel_unc /= len(horband)

            # fix to 1 where sum of TV is zero (likely to have interpreted outside of area)
            horsel_unc[horsel_unc == 0] = 1.

            # apply smoothing filter
            horsel_unc = filtfilt(np.ones(nsmooth) / nsmooth, 1, horsel_unc)
            horsel_uncs.append(horsel_unc)
        hors_unc[horkey] = horsel_uncs

    return hors_unc, tv_tot_raw