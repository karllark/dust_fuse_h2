#!/usr/bin/env python

from __future__ import (absolute_import, print_function, division)

import matplotlib.pyplot as plt
from matplotlib import rc

from get_data import get_fuse_h1_h2

def plot_results(data, xparam, yparam):
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
    """

    xcol = data[xparam]
    #xcol_unc = data[xparam+'_unc']
    ycol = data[yparam]
    #ycol_unc = data[yparam+'unc']
    
    fig = plt.gcf() 
    ax = plt.gca()
    ax.scatter(xcol, ycol)

    return fig

if __name__ == '__main__':

    data = get_fuse_h1_h2()
    fig = plot_results(data, 'lognhtot', 'lognh')
    fig.tight_layout()
    plt.show()
