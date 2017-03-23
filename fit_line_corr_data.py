# fit a line to correlated data
#   based on Hogg chapter and DFM emcee website

from __future__ import (absolute_import, print_function, division)

import numpy as np
import scipy.optimize as op

def lnlike(theta, x, y, yerr):
    """
    ln of likelihood
    
    Starting simple, this just has uncs on y values
    """
    m, b = theta
    model = m * x + b
    inv_sigma2 = 1.0/yerr**2
    print(m, b, -0.5*(np.sum((y-model)**2*inv_sigma2)))
    return -0.5*(np.sum((y-model)**2*inv_sigma2))

def lnlike_xy_cov(params, x, y, sig_x, sig_y, sig_xy):
    """
    ln of likelihood
    
    from Hogg section 7
    """
    theta, b_perp = params
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    delta_i = -sin_theta*x + cos_theta*y - b_perp
    sigma_i_sqr = (sin_theta*sin_theta*sig_x*sig_x
                   - 2.*sin_theta*cos_theta*sig_xy
                   + cos_theta*cos_theta*sig_y*sig_y)
    print(theta, b_perp, -0.5*np.sum(np.square(delta_i)/sigma_i_sqr))
    return -0.5*np.sum(np.square(delta_i)/sigma_i_sqr)

def line_fit_yunc_only(m_true, b_true, x, y, yerr):
    """
    Use scipy optimize to find the maximum likelihood
    """
    nll = lambda *args: -lnlike(*args)
    result = op.minimize(nll, [m_true, b_true], args=(x, y, yerr))
    #m_ml, b_ml = result["x"]

    return result["x"]

def line_fit_xy_cov(theta, b_perp, x, y, xerr, yerr, xyerr):
    """
    Use scipy optimize to find the maximum likelihood
    """
    nll = lambda *args: -lnlike_xy_cov(*args)
    result = op.minimize(nll, [theta, b_perp], args=(x, y, xerr, yerr, xyerr))
    #m_ml, b_ml = result["x"]

    return result["x"]

