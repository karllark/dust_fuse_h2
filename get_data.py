#!/usr/bin/env python

from __future__ import (absolute_import, print_function, division)

from astropy.table import Table

def get_fuse_h1_h2():
    """
    Read in the FUSE H1 and H2 column data.

    Returns
    -------
    data : astropy.table object
       Table of the data (h1, h2, htot, etc.)
    """
    
    data = Table.read('data/fuse_h1_h2.dat',
                      format='ascii.commented_header')

    return data

if __name__ == '__main__':

    h1h2_data = get_fuse_h1_h2()
    print(h1h2_data.colnames)
