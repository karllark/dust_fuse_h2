#!/usr/bin/env python

from __future__ import (absolute_import, print_function, division)

import numpy as np
from astropy.table import Table, join

def get_fuse_h1_h2():
    """
    Read in the FUSE H1 and H2 column data.

    Returns
    -------
    data : astropy.table object
       Table of the data (h1, h2, htot, etc.)
    """
    
    data = Table.read('data/fuse_h1_h2.dat',
                      format='ascii.commented_header',
                      header_start=-1)

    # rename column to have the same name as other tables
    data.rename_column('name','Name')

    # remove the ebv column as this is superceded by another table
    #    this ebv value is from the literature and is for each star
    #    the correct E(B-V) value is for the calculated extinction curve
    data.remove_column('ebv')

    return data

def get_fuse_ext_details():
    """
    Read in the FUSE extinction details [A(V), R(V), etc.]

    Returns
    -------
    data : astropy.table object
       Table of the data [A(V), R(V), etc.]
    """
    
    data = Table.read('data/fuse_ext_details.dat',
                      format='ascii.commented_header',
                      header_start=-1)

    # create the combined uncertainties
    keys = ['A(V)','E(B-V)','R(V)']
    for key in keys:
        data[key+'_unc'] = np.sqrt(np.square(data[key+'_runc']) + 
                                   np.square(data[key+'_sunc']))

    return data

def get_fuse_ext_fm90():
    """
    Read in the FUSE extinction details [A(V), R(V), etc.]

    Returns
    -------
    data : astropy.table object
       Table of the data [A(V), R(V), etc.]
    """
    
    data = Table.read('data/fuse_ext_fm90.dat',
                      format='ascii.commented_header',
                      header_start=-1)

    # create the combined uncertainties
    keys = ['CAV1','CAV2','CAV3','CAV4','x_o','gamma']
    for key in keys:
        data[key+'_unc'] = np.sqrt(np.square(data[key+'_runc']) + 
                                   np.square(data[key+'_sunc']))

    return data

def get_merged_table():

    # get the three tables to merge
    h1h2_data = get_fuse_h1_h2()
    ext_detail_data = get_fuse_ext_details()
    ext_fm90_data = get_fuse_ext_fm90()

    merged_table1 = join(h1h2_data, ext_detail_data, keys='Name')
    merged_table = join(merged_table1, ext_fm90_data, keys='Name')

    return(merged_table)

if __name__ == '__main__':

    merged_table = get_merged_table()
    print(merged_table.colnames)
    print(merged_table)
