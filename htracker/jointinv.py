import matplotlib.pyplot as plt

from pylops.avo.poststack import *
from pylops.optimization.sparsity import *
from pylops.basicoperators import VStack as VStacklop

from pyproximal.proximal import *
from pyproximal.optimization.primal import *
from pyproximal.optimization.primaldual import *
from pyproximal.optimization.bregman import *
from pyproximal.optimization.segmentation import *


def callback(x, xhist):
    xhist.append(x)

def RRE(x, xinv):
    return np.linalg.norm(x - xinv) / np.linalg.norm(x)


def PSNR(x, xinv):
    return 10 * np.log10(len(xinv) * np.max(xinv) / np.linalg.norm(x - xinv))

def joint_inversion_segmentation(d, mback, cl, Op, alpha, beta, delta, tau, mu,
                                 niter=4, l2niter=20, pdniter=100,
                                 segmentniter=10, bisectniter=30, tolstop=0.,
                                 mtrue=None, plotflag=True, show=False):
    r"""Joint inversion-segmentation with Primal-Dual solver

    Parameters
    ----------
    d : :obj:`np.ndarray`
        Data (must have 2 or 3 dimensions with depth/time along first axis)
    mback : :obj:`np.ndarray`
        Background model (must have 2 or 3 dimensions with depth/time along
        first axis)
    cl : :obj:`numpy.ndarray`
        Classes
    Op : :obj:`pylops.avo.poststack.PoststackLinearModelling`
        Modelling operator
    alpha : :obj:`float`
        Scaling factor of the TV regularization of the model
    alpha : :obj:`float`
        Scaling factor of the TV regularization of the model
    beta : :obj:`float`
        Scaling factor of the TV regularization of the segmentation
    delta : :obj:`float`
        Positive scalar weight of the segmentation misfit term
    tau : :obj:`float`
        Stepsize of subgradient of :math:`f`
    mu : :obj:`float`
        Stepsize of subgradient of :math:`g^*`
    niter : :obj:`int`, optional
        Number of iterations of joint scheme
    l2niter : :obj:`int`, optional
        Number of iterations of l2 proximal
    pdniter : :obj:`int`, optional
        Number of iterations of Primal-Dual solver
    segmentniter : :obj:`int`, optional
        Number of iterations of Segmentation solve
    bisectniter : :obj:`int`, optional
        Number of iterations of bisection used in the simplex proximal
    tolstop : :obj:`float`, optional
        Stopping criterion based on the segmentation update
    mtrue : :obj:`np.ndarray`, optional
        True model (must have 2 or 3 dimensions with depth/time along
        first axis). When available use to compute metrics
    plotflag : :obj:`bool`, optional
        Display intermediate steps
    show : :obj:`bool`, optional
        Print solvers iterations log

    Returns
    -------
    minv : :obj:`numpy.ndarray`
        Inverted model.
    v : :obj:`numpy.ndarray`
        Classes probabilities.
    vcl : :obj:`numpy.ndarray`
        Estimated classes.
    rre : :obj:`list`
        RRE metric through iterations (only if ``mtrue`` is provided)
    psnr : :obj:`list`
        PSNR metric through iterations (only if ``mtrue`` is provided)
    minv_hist : :obj:`numpy.ndarray`
        History of inverted model through iterations
    v_hist : :obj:`numpy.ndarray`
        History of classes probabilities through iterations

    """
    print('Working with alpha=%f,  beta=%f,  delta=%f' % (alpha, beta, delta))
    mshape = mback.shape
    msize = mback.size
    ncl = len(cl)

    # TV regularization term
    Dop = Gradient(dims=mshape, edge=True, dtype=Op.dtype, kind='forward')
    l1 = L21(ndim=2, sigma=alpha)

    p = np.zeros(msize)
    q = np.zeros(ncl * msize)
    v = np.zeros(ncl * msize)
    minv = mback.copy().ravel()
    minv_hist = []
    v_hist = []

    rre = psnr = None
    if mtrue is not None:
        rre = np.zeros(niter)
        psnr = np.zeros(niter)

    if plotflag:
        fig, axs = plt.subplots(2, niter, figsize=(4 * niter, 10))

    for iiter in range(niter):
        print('Iteration %d...' % iiter)
        minvold = minv.copy()
        vold = v.copy()

        #############
        # Inversion #
        #############
        if iiter == 0:
            # define misfit term
            l2 = L2(Op=Op, b=d.ravel(), niter=l2niter, warm=True)

            # solve
            minv = PrimalDual(l2, l1, Dop, x0=mback.ravel(),
                              tau=tau, mu=mu, theta=1., niter=pdniter,
                              show=show)
            minv = np.real(minv)
            #dinv = Op * minv

            # Update p
            l2_grad = L2(Op=Op, b=d.ravel())
            dp = (1./alpha) * l2_grad.grad(minv)
            p -= np.real(dp)
        else:
            # define misfit term
            v = v.reshape((msize, ncl))

            L1op = VStacklop([Op] + [Diagonal(np.sqrt(2.*delta)*np.sqrt(v[:, icl])) for icl in range(ncl)])
            d1 = np.hstack([d.ravel(), (np.sqrt(2.*delta)*(np.sqrt(v) * cl[np.newaxis, :]).T).ravel()])
            l2 = L2(Op=L1op, b=d1, niter=l2niter, warm=True, q=p, alpha=-alpha)

            # solve
            minv = PrimalDual(l2, l1, Dop, x0=mback.ravel(),
                              tau=tau, mu=mu, theta=1., niter=pdniter,
                              show=show)
            minv = np.real(minv)
            #dinv = Op * minv

            # Update p
            l2_grad = L2(Op=L1op, b=d1)
            dp = (1./alpha) * l2_grad.grad(minv)
            #dp = (1./alpha) * (Lop.H * (Lop * minv.ravel() - dn.ravel()) -
            #                    2* delta * np.sum([v[:, icl] * (minv.ravel() - cl[icl])
            #                    for icl in range(ncl)]))
            p -= np.real(dp)

        minv_hist.append(minv)

        if plotflag:
            axs[0, iiter].imshow(np.real(minv).reshape(mshape), 'gray')
            axs[0, iiter].axis('tight')

        ################
        # Segmentation #
        ################
        v, vcl = Segment(minv, cl, 2 * delta, 2 * beta, z=-beta*q,
                         niter=segmentniter, callback=None, show=show,
                         kwargs_simplex=dict(engine='numba',
                                             maxiter=bisectniter, call=False))
                                             #xtol=1e-3, ftol=1e-3))
        v_hist.append(v)

        # Update q
        dq = (delta/beta) * ((minv.ravel() - cl[:, np.newaxis]) ** 2).ravel()
        q -= dq

        if plotflag:
            axs[1, iiter].imshow(vcl.reshape(mshape), 'gray')
            axs[1, iiter].axis('tight')

        # Monitor cost functions
        print('f=', L2(Op=Op, b=d.ravel())(minv))
        print('||v-v_old||_2=', np.linalg.norm(v.ravel() - vold.ravel()))
        print('||m-m_old||_2=', np.linalg.norm(minv.ravel() - minvold.ravel()))

        # Monitor quality of reconstruction
        if mtrue is not None:
            rre[iiter] = RRE(mtrue.ravel(), minv.ravel())
            psnr[iiter] = PSNR(mtrue.ravel(), minv.ravel())
            print('RRE=', rre[iiter])
            print('PSNR=', psnr[iiter])

        # Check stopping criterion
        if np.linalg.norm(v.ravel()-vold.ravel()) < tolstop:
            break

    return minv, v, vcl, rre, psnr, minv_hist, v_hist