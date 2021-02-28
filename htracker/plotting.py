import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap


def final_plot(seismic, segpd_classes, horizon_list, scatter=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.imshow(seismic, vmin=-0.5, vmax=0.5, aspect="auto", cmap='seismic')
    for i,h in horizon_list.iterrows():
        if h['chosen']:
            if scatter:
                ax1.scatter(h['x'], h['y'], c='k', s=1)
            else:
                ax1.plot(h['regx'], h['regy'], c='k')

    ax2.imshow(segpd_classes, aspect="auto")
    for i,h in horizon_list.iterrows():
        if h['chosen']:
            if scatter:
                ax2.scatter(h['x'], h['y'], c='w', s=1)
            else:
                ax2.plot(h['regx'], h['regy'])

    ax1.set_title('Seismic w horizons')
    ax2.set_title('Segpd w horizons')


def inputs_plot(model, seismic, segpd_classes):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(model, aspect="auto")

    ax2.imshow(seismic, vmin=-0.5, vmax=0.5, aspect="auto", cmap='seismic')

    ax3.imshow(segpd_classes, aspect="auto")

    ax1.set_title('Model')
    ax2.set_title('Seismic')
    ax3.set_title('Segmented Inversion')

    for ax in (ax1, ax2, ax3):
        ax.axis('tight')


def tv_plot(segpd_classes, cl_tv, cl_classimage, class_index, suptitle=''):

    classimage = np.zeros_like(segpd_classes)
    classimage[np.where(segpd_classes==class_index)] = 1

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 3))

    im1 = ax1.imshow(segpd_classes)
    plt.colorbar(im1, ax=ax1)
    ax1.set_title('Segpd classes')

    im2 = ax2.imshow(classimage, cmap='gray_r')
    plt.colorbar(im2, ax=ax2)
    ax2.set_title('class image - pre cleaning')


    im3 = ax3.imshow(cl_classimage, cmap='gray_r')
    plt.colorbar(im3, ax=ax3)
    ax3.set_title('class image - post cleaning')

    im4 = ax4.imshow(cl_tv, cmap='gray_r')
    plt.colorbar(im4, ax=ax4)
    ax4.set_title('class TV')

    for ax in (ax1, ax2, ax3, ax4):
        ax.axis('tight')

    fig.suptitle(suptitle, y=1.05)


def cleaning_dif_plot(segpd_classes, cl_classimage, class_index, suptitle=''):
    classimage = np.zeros_like(segpd_classes)
    classimage[np.where(segpd_classes == class_index)] = 1

    dif_sum = classimage + cl_classimage
    cmap_dif = LinearSegmentedColormap.from_list('name', ['white', 'red', '#d3d3d3'], 3)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))

    im1 = ax1.imshow(classimage, cmap='gray_r')
    ax1.set_title('Pre cleaning')

    im2 = ax2.imshow(cl_classimage, cmap='gray_r')
    ax2.set_title('Post cleaning')

    im3 = ax3.imshow(dif_sum, cmap=cmap_dif)
    ax3.set_title('Difference')

    for ax in (ax1, ax2, ax3):
        ax.axis('tight')

    fig.suptitle(suptitle, y=1.05)


def raw_horizons_plot(seismic, segpd_classes, cl_tv, horizon_points_raw, suptitle=''):
    horizon_points_raw.tolist()
    hx = [h[1] for h in horizon_points_raw.tolist()]
    hy = [h[0] for h in horizon_points_raw.tolist()]

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))

    ax0.imshow(cl_tv, aspect="auto", cmap='gray_r')

    ax1.imshow(seismic, vmin=-0.5, vmax=0.5, aspect="auto", cmap='seismic')
    ax1.scatter(hx, hy, c='k', s=1)

    ax2.imshow(segpd_classes, aspect="auto")
    ax2.scatter(hx, hy, c='k', s=1)

    for ax in (ax1, ax2, ax0):
        ax.axis('tight')

    ax1.set_title('Seismic w horizons')
    ax2.set_title('Segmented Inversion w horizons')
    ax0.set_title('TV')

    fig.suptitle(suptitle, y=1.05)
