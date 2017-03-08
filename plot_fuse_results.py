#!/usr/bin/env python

from __future__ import (absolute_import, print_function, division)

import argparse
import matplotlib.pyplot as plt
from matplotlib import rc

from get_data import get_merged_table

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

def plot_results(data, xparam, yparam,
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

    fig : matplotlib figure object, optional
        Figure to use for plot

    ax : matplotlib axes object, optional
        Subplot of figure to use
    """

    xcol = data[xparam]
    if xparam+'_unc' in data.colnames:
        xcol_unc = data[xparam+'_unc']
    else:
        xcol_unc = None
    ycol = data[yparam]
    if yparam+'_unc' in data.colnames:
        ycol_unc = data[yparam+'_unc']
    else:
        ycol_unc = None
    
    if fig is None:
        fig = plt.gcf() 
    if ax is None:
        ax = plt.gca()
    ax.errorbar(xcol, ycol, xerr=xcol_unc, yerr=ycol_unc, fmt='o')
    ax.set_xlabel(xparam)
    ax.set_ylabel(yparam)
    return fig

if __name__ == '__main__':

    # get the data table
    data = get_merged_table()
    colnames = data.colnames

    parser = initialize_parser()
    parser.add_argument('--xparam', action='store', default='A(V)', 
                        choices=colnames, 
                        help='Choose column type to plot')
    parser.add_argument('--yparam', action='store', default='lognhtot', 
                        choices=colnames, 
                        help='Choose column type to plot')
    args = parser.parse_args()

    fig = plot_results(data, args.xparam, args.yparam)
    fig.tight_layout()

    basename = 'fuse_results_' + args.xparam + '_' + args.yparam
    if args.savefig:
        fig.savefig('{}.{}'.format(basename, args.savefig))
    else:
        plt.show()
