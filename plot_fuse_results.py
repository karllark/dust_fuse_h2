#!/usr/bin/env python

from __future__ import (absolute_import, print_function, division)

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import rc

from get_data import (get_merged_table, get_bohlin78)
from fit_line_corr_data import (line_fit_yunc_only, line_fit_xy_cov)

def set_params(lw=1.5, universal_color='#262626', fontsize=16):
    '''Configure some matplotlib rcParams.

    Parameters
    ----------
    lw : scalar
        Linewidth of plot and axis lines. Default is 1.5.
    universal_color : str, matplotlib named color, rgb tuple
        Color of text and axis spines. Default is #262626, off-black
    fontsize : scalar
        Font size in points. Default is 12
    '''
    rc('font', size=fontsize)
    rc('lines', linewidth=lw, markeredgewidth=lw*0.5)
    rc('patch', linewidth=lw, edgecolor='#FAFAFA')
    rc('axes', linewidth=lw, edgecolor=universal_color,
       labelcolor=universal_color, 
       axisbelow=True)
    rc('image', origin='lower') # fits images
    rc('xtick.major', width=lw*0.75)
    rc('xtick.minor', width=lw*0.5)
    rc('xtick', color=universal_color)
    rc('ytick.major', width=lw*0.75)
    rc('ytick.minor', width=lw*0.5)
    rc('ytick', color=universal_color)
    rc('grid', linewidth=lw)
    rc('legend', loc='best', numpoints=1, scatterpoints=1, handlelength=1.5,
        fontsize=fontsize, columnspacing=1, handletextpad=0.75)

def initialize_parser():
    '''For running from command line, initialize argparse with common args
    '''
    ftypes = ['png', 'jpg', 'jpeg', 'pdf', 'ps', 'eps', 'rgba',
              'svg', 'tiff', 'tif', 'pgf', 'svgz', 'raw']
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--savefig', action='store', 
                        default=False, choices=ftypes,
                        help='Save figure to a file')
    return parser

def format_colname(name):
    """
    Convert the column name to a better formatted name
    """
    colnames = ['AV','RV', 'EBV', 
                'CAV1','CAV2','CAV3','CAV4',
                'C1','C2','C3','C4',
                'x_o','gamma', 'bump_area',
                'fh2',
                'nhtot', 'nh2', 'nhi', 
                'NH_AV', 'NH_EBV']
    plotnames = ['$A(V)$','$R(V)$', '$E(B-V)$', 
                 '$C^{A(V)}_1$','$C^{A(V)}_2$','$C^{A(V)}_3$','$C^{A(V)}_4$',
                 '$C_1$','$C_2$','$C_3$','$C_4$',
                 '$x_o$', '$\gamma$', '$\pi C^{A(V)}_3 / 2 \gamma$',
                 '$f(H_2)$', 
                 '$N(H)$', '$N(H_2)$', '$N(HI)$', 
                 '$N(H)/A(V)$', '$N(H)/E(B-V)$']
    dic_pairs = dict(zip(colnames, plotnames))

    out_name = name
    if name[:3] == 'log':
        out_name = '$\log (' + name[3:].upper() + ')$'
    elif name in dic_pairs.keys():
        out_name = dic_pairs[name]

    return out_name

def get_unc(param, data):
    """
    Returns the unc column if it is in the table
    """
    if param+'_unc' in data.colnames:
        return data[param+'_unc'].data
    else:
        return None

def get_corr(xparam, yparam, x, y, xerr, yerr,
             cterm=None, cterm_unc=None):
    """
    Return the correlation coefficient between pairs of parameters
    """
    if ((xparam == 'AV' and yparam == "NH_AV") or
        (xparam == 'EBV' and yparam == "NH_EBV")):
        yfac = yerr/y
        xfac = xerr/x
        corr = -1.0*xfac/yfac
    elif (xparam == 'RV' and yparam == "NH_AV"
        and cterm is not None and cterm_unc is not None):
        avfac = cterm_unc/cterm
        yfac = yerr/y
        corr = -1.0*avfac/yfac
    elif xparam == 'AV' and yparam == "RV":
        yfac = yerr/y
        xfac = xerr/x
        corr = xfac/yfac
    elif (((xparam == 'RV') or (xparam == 'AV')) and
          ((yparam[0:3] == "CAV") or (yparam == 'bump_area')) and
          cterm is not None and cterm_unc is not None):
        avfac = cterm_unc/cterm
        yfac = yerr/y
        corr = -1.0*avfac/yfac
    elif (((xparam == 'RV') or (xparam == 'EBV')) and
          (yparam[0:1] == "C") and
          cterm is not None and cterm_unc is not None):
        ebvfac = cterm_unc/cterm
        yfac = yerr/y
        corr = ebvfac/yfac
    else:
        corr = np.full(len(x), 0.0)

    return corr

def plot_errorbar_corr(ax, x, y, xerr, yerr, corr,
                       pellipse=False, pebars=True,
                       pcol='b', alpha=0.25):
    """
    Plot x, y errorbars that are correlated

    Parameters
    ----------
    x: x values
    xerr: x values uncertainties
    y: y values
    yerr: y values uncertainties
    corr: correlation coefficient of xerr and yerr

    pellipse: True to plot the ellipses
    pebars: True to plot the error bars

    alpha: transparancy
    pcol: plot color
    """
    
    # make a theta vector
    theta = 2.*np.pi*np.linspace(0.0,1.0,num=100)

    # loop over the points and plot the error ellipse
    for i in range(len(x)):
        # get the rotation angle in degrees
        #rot_angle = np.sign(corr)*np.arctan(corr)
        rot_angle = corr[i]*45.0*np.pi/180.0
        #print('corr = ', corr)
        #print('new angle = ', rot_angle*180./np.pi)

        # plot an ellipse that illustrates the covariance
        theta = 2.*np.pi*np.linspace(0.0,1.0,num=100)
        theta2 = 2.*np.pi*np.linspace(0.0,1.0,num=5)
        
        a = 1.0/np.cos(rot_angle)
        b = a*(1.0-np.absolute(corr[i]))
        if b == 0.0:  # case where corr = 1.0
            b = 0.01
        #print(a,b)

        r = a*b/np.sqrt(np.square(b*np.cos(theta)) +
                        np.square(a*np.sin(theta)))
        ex1 = r*np.cos(theta)
        ey1 = r*np.sin(theta)

        ex = ex1*np.cos(rot_angle) - ey1*np.sin(rot_angle)
        ey = ex1*np.sin(rot_angle) + ey1*np.cos(rot_angle)

        ex_range = max(ex) - min(ex)
        ey_range = max(ey) - min(ey)

        ex *= xerr[i]/(0.5*ex_range)
        ey *= yerr[i]/(0.5*ey_range)

        ex += x[i]
        ey += y[i]

        if pellipse:
            ax.plot(ex,ey, pcol+'-', alpha=alpha)

        # now plot the rotated axes of the ellipse
        r = a*b/np.sqrt(np.square(b*np.cos(theta2)) +
                        np.square(a*np.sin(theta2)))
        ex1 = r*np.cos(theta2)
        ey1 = r*np.sin(theta2)

        ex = ex1*np.cos(rot_angle) - ey1*np.sin(rot_angle)
        ey = ex1*np.sin(rot_angle) + ey1*np.cos(rot_angle)

        ex *= xerr[i]/(0.5*ex_range)
        ey *= yerr[i]/(0.5*ey_range)

        ex += x[i]
        ey += y[i]
        if pebars:
            ax.plot([ex[0],ex[2]],[ey[0],ey[2]], pcol+'-', alpha=alpha)
            ax.plot([ex[1],ex[3]],[ey[1],ey[3]], pcol+'-', alpha=alpha)

        # not working for non-zero angles
        #   possibly something to do with the large difference in the
        #   axes units
        #e = Ellipse(xy=(x[i], y[i]), 
        #            width=2.0*xerr[i], 
        #            height=2.0*yerr[i], 
        #            angle=10.0)
        #ax.add_patch(e)
        #e.set_clip_box(ax.bbox)
        #e.set_alpha(alpha)
        #e.set_facecolor((0,0,1))

def plot_results(data, xparam, yparam,
                 pxrange=None, pyrange=None,
                 data_comp=None,
                 data_bohlin=None,
                 figsize=None):
    """
    Plot the fuse results with specificed x and y axes

    Parameters
    ----------
    data: astropy.table
       Table of the data to plot

    xparam: str
       name of column to plot as the x variable

    yparam: str
       name of column to plot as the y variable

    pxrange: float[2]
       min/max x range to plot

    pyrange: float[2]
       min/max y range to plot

    data_comp: astropy.table
       Table of the data to plot for the comparision stars

    figsize : float[2]
       x,y size of plot

    """
    # set the plotting defaults
    set_params(lw=2)

    fig, ax = plt.subplots(figsize=figsize)

    if data_bohlin is not None:
        if ((xparam in data_bohlin.colnames) 
            and (yparam in data_bohlin.colnames)):
            xcol = data_bohlin[xparam]
            xcol_unc = get_unc(xparam, data_bohlin)
            ycol = data_bohlin[yparam]
            ycol_unc = get_unc(yparam, data_bohlin)
            ax.errorbar(xcol, ycol, xerr=xcol_unc, yerr=ycol_unc, 
                        fmt='ro', label='Bohlin (1978)', alpha=0.25)
    if data_comp is not None:
        xcol = data_comp[xparam]
        xcol_unc = get_unc(xparam, data_comp)
        ycol = data_comp[yparam]
        ycol_unc = get_unc(yparam, data_comp)
        #ax.errorbar(xcol, ycol, xerr=xcol_unc, yerr=ycol_unc, 
        #            fmt='go', label='FUSE Comparisons', alpha=0.25)
        ax.errorbar(xcol, ycol, fmt='go')
        # plot the error bars as ellipses illustrating the covariance
        corrs = get_corr(xparam, yparam, xcol, ycol, xcol_unc, ycol_unc,
                         cterm=data_comp['AV'].data,
                         cterm_unc=data_comp['AV_unc'].data)
        plot_errorbar_corr(ax, xcol, ycol, xcol_unc, ycol_unc, corrs,
                           alpha=0.25, pcol='b')

    xcol = data[xparam].data
    xcol_unc = get_unc(xparam, data)
    ycol = data[yparam].data
    ycol_unc = get_unc(yparam, data)
    #ax.errorbar(xcol, ycol, xerr=xcol_unc, yerr=ycol_unc, 
    #            fmt='bo', label='FUSE Reddened', alpha=0.25)
    ax.errorbar(xcol, ycol, fmt='bo')

    # plot the error bars as ellipses illustrating the covariance
    if yparam[0:3] == 'CAV':
        cparam = 'AV'
    elif yparam[0:1] == 'C':
        cparam = 'EBV'
    else:
        cparam = 'AV'

    corrs = get_corr(xparam, yparam, xcol, ycol, xcol_unc, ycol_unc,
                     cterm=data[cparam].data,
                     cterm_unc=data[cparam+'_unc'].data)
    plot_errorbar_corr(ax, xcol, ycol, xcol_unc, ycol_unc, corrs,
                       alpha=0.25, pcol='b')

    ax.set_xlabel(format_colname(xparam))
    ax.set_ylabel(format_colname(yparam))

    # remove "bad" point
    #indxs, = np.where(xcol < 3.0)
    #print(len(indxs), len(xcol))
    #xcol = xcol[indxs]
    #xcol_unc = xcol_unc[indxs]
    #ycol = ycol[indxs]
    #ycol_unc = ycol_unc[indxs]

    # remove "bad" points
    #indxs, = np.where(np.logical_or(2.0 >= xcol, xcol >= 2.5))
    #print(len(indxs), len(xcol))
    #xcol = xcol[indxs]
    #xcol_unc = xcol_unc[indxs]
    #ycol = ycol[indxs]
    #ycol_unc = ycol_unc[indxs]

    # fit a line
    params = np.polyfit(xcol, ycol, 1, w=1.0/ycol_unc)
    params = np.array([params[1],params[0]])
    print('linear fit params [y-intercept, slope]')
    print(params)
    xlim = ax.get_xlim()
    x_mod = np.linspace(xlim[0],xlim[1])
    y_mod = params[0] + x_mod*params[1]
    ax.plot(x_mod,y_mod, 'r-')

    # fit a line via own routines
    #   divide to get numbers in reasonable range
    #   otherwise does not work (likely a tolerance somewhere)
    div_val = 1e21
    params /= div_val
    params2 = line_fit_yunc_only(params[0], params[1], xcol,
                                 ycol/div_val, ycol_unc/div_val)
    params2 *= div_val
    y_mod = params2[0] + x_mod*params2[1]
    ax.plot(x_mod,y_mod, 'g-')
    print(params2)
    
    # now with covariances
    sin_theta = params[1]/(1.0 + np.square(params[1]))
    theta = np.arcsin(sin_theta)
    b_perp = params[0]*np.cos(theta)

    print('in: ', theta*180./np.pi, b_perp)
    params3 = line_fit_xy_cov(theta, b_perp, xcol, ycol/div_val,
                              xcol_unc, ycol_unc/div_val,
                              np.full((len(xcol)), 0.0))
    print('out: ', params3[1]*180./np.pi, params3[0])

    b = params3[0]/np.cos(params3[1])*div_val
    m = np.sqrt((1.0/np.cos(params3[1])) - 1.0)*div_val
    print(b, m)
    y_mod = b + x_mod*m
    ax.plot(x_mod,y_mod, 'c-')
    
    # plot stuff
    if pxrange is not None:
        ax.set_xlim(pxrange)
    if pyrange is not None:
        ax.set_ylim(pyrange)
    
    #ax.legend()
    fig.tight_layout()
    
    return fig

if __name__ == '__main__':

    # get the data table
    data = get_merged_table()
    colnames = data.colnames

    parser = initialize_parser()
    parser.add_argument('--xparam', action='store', default='AV', 
                        choices=colnames, 
                        help='Choose column type to plot')
    parser.add_argument('--yparam', action='store', default='lognhtot', 
                        choices=colnames, 
                        help='Choose column type to plot')
    parser.add_argument('--xrange', action='store', default=[0.0, 0.0], 
                        choices=colnames, 
                        help='Choose column type to plot')
    parser.add_argument('--comps', action='store_true',
                        help='plot the comparision sightlines')
    parser.add_argument('--bohlin', action='store_true',
                        help='plot the Bohlin78 sightlines')
    args = parser.parse_args()

    # get extra data if desired
    if args.comps:
        data_comp = get_merged_table(comp=True)
    else:
        data_comp = None
    if args.bohlin:
        data_bohlin78 = get_bohlin78()
    else:
        data_bohlin78 = None

    # make the requested plot
    fig = plot_results(data, args.xparam, args.yparam,
                       data_comp=data_comp,
                       data_bohlin=data_bohlin78)

    # save the plot
    basename = 'fuse_results_' + args.xparam + '_' + args.yparam
    if args.savefig:
        fig.savefig('{}.{}'.format(basename, args.savefig))
    else:
        plt.show()
