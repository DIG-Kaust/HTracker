import numpy as np
import pandas as pd

from scipy import ndimage as ndi
from skimage.morphology import erosion, dilation
from skimage.morphology import disk

from pylops.basicoperators import *
from pylops.signalprocessing import *
from pylops.utils.wavelets import *
from pylops.avo.poststack import *
from pylops.optimization.sparsity import *


def tv(x):
    """Compute Total Variation on image

    Parameters
    ----------
    x : np array
        2D array

    Returns
    -------
    f : np array
        Total Variation of input

    """
    Gop = Gradient(dims=x.shape, edge=True)
    y = Gop * x.ravel()
    f = np.sqrt(np.sum(y.reshape(2, x.size) ** 2, axis=0))
    return f.reshape(x.shape)


def small_shape_removal(image, spurioseventsize):
    """Remove small closed shapes in an image
    (see https://scikit-image.org/docs/0.6/user_guide/tutorial_segmentation.html)
    """
    label_objects, nb_labels = ndi.label(image)
    sizes = np.bincount(label_objects.ravel())
    clean_image = sizes > spurioseventsize
    clean_image[0] = 0
    return clean_image[label_objects].astype("int")


def clean_spurios(classimage,
                  spurioseventsize=100,
                  erode=False,
                  erosion_size=3,
                  ):
    """Clean spurious details from 2D image

    Parameters
    ----------
    classimage : np array
        2D array
    spurioseventsize : int
        size in pixels for small shape cleaning
    erode : bool
        whether to apply skimage erosion cleaning
    erosion_size : int
        size for skimage erosion cleaning

    Returns
    -------
    clean_image : np array
        cleaned image

    """

    # SKIMAGE EROSION CLEANING
    if erode:
        selem = disk(erosion_size)
        classimage = erosion(classimage, selem)
        classimage = dilation(classimage, selem)

    # SCIPY NDIMAGE SMALL SHAPES CLEANING
    clean_image = small_shape_removal(classimage, spurioseventsize)
    clean_image = 1 - small_shape_removal(1 - clean_image, spurioseventsize)
    return clean_image


def get_class_tv(segpd_classes,
                 class_index,
                 spurioseventsize=250,
                 incl_erosion = False,
                 erosion_size = 3,
                 ):
    """Clean image and compute total variation

    Parameters
    ----------
    segpd_classes : np array
        Segmented classes
    class_index : int
        Index of class for TV
    spurioseventsize : int
        Threshold for scipy ndimage cleaning
    incl_erosion : bool
        Whether to apply skimage erosion cleaning
    erosion_size : int
        Size for skimage erosion cleaning

    Returns
    -------
    cl_tv : np array
        TV image for desired class
    cl_classimage : np.array
        cleaned, masked class image used as input to TV computation

    """

    # Make into a class mask
    classimage = np.zeros_like(segpd_classes)
    classimage[np.where(segpd_classes==class_index)] = 1

    # Clean spurious events
    cl_classimage = clean_spurios(classimage,
                                  spurioseventsize,
                                  erode=incl_erosion,
                                  erosion_size=erosion_size
                                  )

    # Compute TV
    cl_tv = tv(cl_classimage)

    return cl_tv, cl_classimage


def get_horizon_points(cl_tv, tvlowerthresh=0.4):
    """Identify points where TV is greater than threshold

    Parameters
    ----------
    cl_tv : np array
        TV image for desired class
    tvlowerthresh : int
        TV threshold for horizon points

    Returns
    -------
    horizon_points : np array
        (x,y) of horizon points
    """
    horizon_points = np.argwhere(cl_tv>tvlowerthresh)
    return horizon_points


def nearest_point(xp, yp, df, lim=3):
    """ Find the next point along

    Parameters
    ----------
    xp : int
        Horizon point X
    yp : int
        Horizon point Y
    df : dataframe
        Dataframe of possible next horizon points
    lim : int
        Maximum step away from point for next horizon point

    Returns
    -------
    next_point : dict
        Nearest point that becomes next point
    u_df : dataframe
        Updated dataframe where the nearest points have been dropped
    """

    # Find all possibilities for next points (ie. all points within the limit)
    np_df = pd.DataFrame(df.query('%i+%i >= x > %i  & %i-%i <= y <= %i+%i' % (
        xp, lim,
        xp,
        yp, lim,
        yp, lim)))

    # If no nearest points return None
    if len(np_df) == 0:
        return None, None

    # Find closest points (points with minimum x)
    minx = int(min(np_df['x'].values))
    np_df_minx = np_df[np_df['x'] == minx]

    # Find which of closest points have the highest TV value
    maxprob = max(np_df_minx['prob'].values)
    poss_points = np_df_minx.loc[np_df_minx['prob'] == maxprob]

    # If multiple have same TV value choose the one closest to the point
    inext_point = np.argmin(np.abs(poss_points['y'] - yp))
    next_point_df = pd.DataFrame(np_df_minx.loc[np_df_minx['prob'] == maxprob].iloc[inext_point])
    next_point = next_point_df.T.astype({"x": int, "y": int, "prob": float})

    # Drop all points with the minimum x from the dataframe
    u_df = pd.concat([df, np_df_minx, np_df_minx]).drop_duplicates(keep=False)

    return next_point, u_df


def l2r_connect(sp, df, nx, nn_lim=3):
    """ Going from left to right connect nearby points

    Parameters
    ----------
    sp : dict
        starting point from which to begin connection
    df : dataframe
        containing all possible points
    nx : int
        maximum x integer
    nn_lim : int
        maximum distance limit on which to allow joining of points

    Returns
    -------
    h_df : dataframe
        selected points that make up the horizon
    u_df : dataframe
        updated df with all possible joins dropped

    """
    # Starting point for join and initial x
    point = sp
    ix = sp['x']

    # Drop starting point from the dataframe so it only contains possible next points
    df = df[~((df.x == sp['x']) & (df.y == sp['y']))]

    # Begin horizon list of points
    horizon_values = []
    horizon_values.append({'x': sp['x'],
                           'y': sp['y'],
                           'prob': sp['prob'],
                           })

    # Set that it is possible to find a next point
    np_poss = True

    # Look for next point
    while np_poss and ix < nx:
        xp = point['x']
        yp = point['y']

        nextpt, u_df = nearest_point(xp, yp, df, lim=nn_lim)

        if u_df is not None:
            horizon_values.append({'x': nextpt['x'].values[0],
                                   'y': nextpt['y'].values[0],
                                   'prob': nextpt['prob'].values[0],
                                   })

            # Update new starting values
            point = nextpt
            df = u_df.copy()
        else:
            np_poss = False
            u_df = df.copy()
        ix += 1

    # Convert horizon points into a DF
    h_df = pd.DataFrame(horizon_values)

    return h_df, u_df


def get_seperated_lines(horizon_points_raw,
                        cl_tv,
                        nn_lim=3,
                        line_minlen=15,
                        verbose=False):
    """Create list of lines by connecting points in a marching method
    (left-to-right)

    Parameters
    ----------
    horizon_points_raw : np array
        (x,y) of horizon points
    cl_tv : np array
        TV image for desired class
    nn_lim : int
        maximum distance limit to search for next point
    line_minlen : int
        minimum length of horizon line
    verbose : bool
        print output or not

    Returns
    -------
    horizon_list : list
        list of dictionaries describing the horizon lines

    """
    # Put horizon_points_raw into Dataframe sorted from L-R,T-B
    res = []
    for points in horizon_points_raw:
        res.append({'x': points[1],
                    'y': points[0],
                    'prob': cl_tv[points[0], points[1]],
                    })
    df = pd.DataFrame(res)
    df.sort_values(['x', 'y'], ascending=[True, True], inplace=True)

    # Initialise first point for horizon lines
    start_point = df.iloc[0]
    df_tmp = df.copy()
    horizon_list = []

    # Connect neighbouring points to create horizon lines
    while len(df_tmp) > line_minlen:
        if verbose:
            print("Start point:")
            print(start_point)
            print("Possible points:", df_tmp.shape)

        # Find and join neighbouring points and get updated df which doesnt include those points
        h_df, u_df = l2r_connect(start_point, df_tmp, cl_tv.shape[1], nn_lim=nn_lim)

        # Retain horizon line if above the minimum length
        if h_df is not None and len(h_df) > line_minlen:
            horizon_list.append(h_df)

        # update params for next line
        df_tmp = u_df.copy()
        if len(df_tmp) > line_minlen:
            u_df.sort_values(['x', 'y'], ascending=[True, True], inplace=True)
            start_point = u_df.iloc[0]

        if verbose:
            print("Remaining points: %i" % len(df_tmp))

    return horizon_list
