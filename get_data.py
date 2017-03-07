#!/usr/bin/env python

from __future__ import (absolute_import, print_function, division)

import numpy as np

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

    print(data.colnames)

if __name__ == '__main__':

    get_fuse_h1_h2()
