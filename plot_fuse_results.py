#!/usr/bin/env python

from __future__ import (absolute_import, print_function, division)

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from get_data import (get_merged_table, get_bohlin78)

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
    out_name = name
    if name[:3] == 'log':
        out_name = '$\log (' + name[3:].upper() + ')$'
    elif name in ['A(V)','R(V)']:
        out_name = '$' + name + '$'

    return out_name

def get_unc(param, data):
    """
    Returns the unc column if it is in the table
    """
    if param+'_unc' in data.colnames:
        return data[param+'_unc']
    else:
        return None

def plot_results(data, xparam, yparam,
                 data_comp=None,
                 data_bohlin=None,
                 fig=None, ax=None):
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

    data_comp: astropy.table
       Table of the data to plot for the comparision stars

    fig : matplotlib figure object, optional
        Figure to use for plot

    ax : matplotlib axes object, optional
        Subplot of figure to use
    """
    if fig is None:
        fig = plt.gcf() 
    if ax is None:
        ax = plt.gca()

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
        ax.errorbar(xcol, ycol, xerr=xcol_unc, yerr=ycol_unc, 
                    fmt='go', label='FUSE Comparisons', alpha=0.25)
        ax.errorbar(xcol, ycol, fmt='go')

    xcol = data[xparam]
    xcol_unc = get_unc(xparam, data)
    ycol = data[yparam]
    ycol_unc = get_unc(yparam, data)
    ax.errorbar(xcol, ycol, xerr=xcol_unc, yerr=ycol_unc, 
                fmt='bo', label='FUSE Reddened', alpha=0.25)
    ax.errorbar(xcol, ycol, fmt='bo')
    ax.set_xlabel(format_colname(xparam))
    ax.set_ylabel(format_colname(yparam))

    # fit a line
    params = np.polyfit(xcol, ycol, 1)#, w=1.0/ycol_unc)
    print('linear fit params')
    print(params)
    xlim = ax.get_xlim()
    x_mod = np.linspace(xlim[0],xlim[1])
    y_mod = params[1] + x_mod*params[0]
    ax.plot(x_mod,y_mod, 'r-')

    print(min(y_mod), max(y_mod))

    #ax.legend()
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

    # set the plotting defaults
    set_params(lw=2)

    # make the requested plot
    fig = plot_results(data, args.xparam, args.yparam,
                       data_comp=data_comp,
                       data_bohlin=data_bohlin78)
    fig.tight_layout()
    
    # save the plot
    basename = 'fuse_results_' + args.xparam + '_' + args.yparam
    if args.savefig:
        fig.savefig('{}.{}'.format(basename, args.savefig))
    else:
        plt.show()
