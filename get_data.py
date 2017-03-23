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

    # recalculate the lognhtot columns
    #  updated lognhi columns and generally making sure basic math is correct
    nhi = np.power(10.0,data['lognhi'])
    nh2 = np.power(10.0,data['lognh2'])
    nhtot = nhi + 2.0*nh2
    nhi_unc = 0.5*(np.power(10.0,data['lognhi'] + data['lognhi_unc']) 
                   - np.power(10.0,data['lognhi'] - data['lognhi_unc']))
    nh2_unc = 0.5*(np.power(10.0,data['lognh2'] + data['lognh2_unc']) 
                   - np.power(10.0,data['lognh2'] - data['lognh2_unc']))
    nhtot_unc = np.sqrt(np.square(nhi_unc) + np.square(2.0*nh2_unc))

    # save the new total
    data['lognhtot'] = np.log10(nhtot)
    data['lognhtot_unc'] = 0.5*(np.log10(nhtot + nhtot_unc) 
                                - np.log10(nhtot - nhtot_unc))
    # save the linear versions
    data['nhi'] = nhi
    data['nhi_unc'] = nhi_unc
    data['nh2'] = nh2
    data['nh2_unc'] = nh2_unc
    data['nhtot'] = nhtot
    data['nhtot_unc'] = nhtot_unc

    # recalculate the f_H2
    data['fh2'] = 2.*data['nh2']/data['nhtot']
    data['fh2_unc'] = data['fh2']*np.sqrt(np.square(nh2_unc/nh2)
                                          + np.square(nhtot_unc/nhtot))

    data['logfh2'] = np.log10(data['fh2'])
                                  
    # H volume density
    #   needs updating when we have new distances
    data['nh'] = np.power(10.0,data['lognh'])

    return data

def get_fuse_ext_details(filename):
    """
    Read in the FUSE extinction details [A(V), R(V), etc.]

    Parameters
    ----------
    filename: str
       name of file with the data

    Returns
    -------
    data : astropy.table object
       Table of the data [A(V), R(V), etc.]
    """
    
    data = Table.read(filename,
                      format='ascii.commented_header',
                      header_start=-1)

    # create the combined uncertainties
    keys = ['AV','EBV','RV']
    for key in keys:
        if (key in data.colnames) and (not key+'_unc' in data.colnames):
            data[key+'_unc'] = np.sqrt(np.square(data[key+'_runc']) + 
                                       np.square(data[key+'_sunc']))

    # make EBV column if it does not exist        
    if 'EBV' not in data.colnames:
        data['EBV'] = data['AV']/data['RV']
        data['EBV_unc'] = data['EBV'] \
            *np.sqrt(np.square(data['AV_unc']/data['AV'])
                     + np.square(data['RV_unc']/data['RV']))

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

def get_bohlin78():
    """
    Read in the Bohlin et al. (1978) Copernicus data

    Returns
    -------
    data : astropy.table object
       Table of the data [EBV, etc.]
    """
    
    data = Table.read('data/bohlin78_copernicus.dat',
                      format='ascii.commented_header',
                      header_start=-1)

    # remove sightlines with non-physical EBV values
    indxs, = np.where(data['EBV'] > 0.0)
    data = data[indxs]

    # get the units correct
    data['nhi'] = 1e20*data['nhi']
    data['nhtot'] = 1e20*data['nhtot']

    # convert the uncertainties from % to total
    data['nhi_unc'] = data['nhi']*data['nhi_unc']*0.01

    # make log units for consitency with FUSE work
    data['lognhi'] = np.log10(data['nhi'])
    data['lognhi_unc'] = 0.5*(np.log10(data['nhi'] + data['nhi_unc'])
                              - np.log10(data['nhi'] - data['nhi_unc']))
 
    data['lognhtot'] = np.log10(data['nhtot'])

    # create the fh2 column
    data['fh2'] = 2.*data['nh2']/data['nhtot']
    
    # make a AV column assuming RV=3.1
    data['RV'] = np.full((len(data)),3.1)
    data['AV'] = data['RV']*data['EBV']
    
    # now the NH/AV and NH/EBV
    data['NH_AV'] = data['nhtot']/data['AV']
    data['NH_EBV'] = data['nhtot']/data['EBV']

    return data
   
def get_merged_table(comp=False):
    """
    Read in the different files and merge them

    Parameters
    ----------
    comp : boolean, optional
       get the comparision data
    """

    # get the three tables to merge
    h1h2_data = get_fuse_h1_h2()
    if comp:
        filename = 'data/fuse_comp_details_fm90.dat'
    else:
        filename = 'data/fuse_ext_details.dat'
    ext_detail_data = get_fuse_ext_details(filename)

    # merge the tables together
    merged_table = join(h1h2_data, ext_detail_data, keys='Name')

    if not comp:
        ext_fm90_data = get_fuse_ext_fm90()
        merged_table1 = join(merged_table, ext_fm90_data, keys='Name')
        merged_table = merged_table1

        # make the "regular" FM90 parameters
        #  normalized by E(B-V) instead of A(V)
        merged_table['C1'] = (merged_table['CAV1'] - 1.0)*merged_table['RV']
        merged_table['C1_unc'] = merged_table['C1'] \
            * np.sqrt(np.square(merged_table['CAV1_unc']/merged_table['CAV1'])
                      + np.square(merged_table['EBV_unc']/merged_table['EBV']))
        merged_table['C2'] = merged_table['CAV2']*merged_table['RV']
        merged_table['C2_unc'] = merged_table['C2'] \
            * np.sqrt(np.square(merged_table['CAV2_unc']/merged_table['CAV2'])
                      + np.square(merged_table['EBV_unc']/merged_table['EBV']))
        merged_table['C3'] = merged_table['CAV3']*merged_table['RV']
        merged_table['C3_unc'] = merged_table['C3'] \
            * np.sqrt(np.square(merged_table['CAV3_unc']/merged_table['CAV3'])
                      + np.square(merged_table['EBV_unc']/merged_table['EBV']))
        merged_table['C4'] = merged_table['CAV4']*merged_table['RV']
        merged_table['C4_unc'] = merged_table['C4'] \
            * np.sqrt(np.square(merged_table['CAV4_unc']/merged_table['CAV4'])
                      + np.square(merged_table['EBV_unc']/merged_table['EBV']))

    # generate the N(H)/A(V) columns
    merged_table['NH_AV'] = merged_table['nhtot']/merged_table['AV']
    merged_table['NH_AV_unc'] = merged_table['NH_AV'] \
        * np.sqrt(np.square(merged_table['nhtot_unc']/merged_table['nhtot'])
                  + np.square(merged_table['AV_unc']/merged_table['AV']))

    # generate the N(H)/E(B-V) columns
    merged_table['NH_EBV'] = merged_table['nhtot']/merged_table['EBV']
    merged_table['NH_EBV_unc'] = merged_table['NH_EBV'] \
        * np.sqrt(np.square(merged_table['nhtot_unc']/merged_table['nhtot'])
                  + np.square(merged_table['EBV_unc']/merged_table['EBV']))

    # make the 2175 A bump area
    if ('CAV3' in merged_table.colnames) and ('gamma' in merged_table.colnames):
        #C3 = (merged_table['CAV3'] - 1.0)*merged_table['RV']
        #indxs = np.where(C3 == 0.0)
        #print(C3[indxs], merged_table['CAV3'][indxs])
        #C3_unc = C3*np.sqrt(np.square(merged_table['CAV3_unc']
        #                              / merged_table['CAV3'])
        #                    + np.square(merged_table['RV_unc']
        #                                / merged_table['RV']))
        C3 = merged_table['CAV3']
        C3_unc = merged_table['CAV3_unc']
        merged_table['bump_area'] = (np.pi*C3
                                     / (2.0*merged_table['gamma']))
        bump_area_unc = np.sqrt(np.square(C3_unc/C3)
                                + np.square(merged_table['gamma_unc']
                                            / merged_table['gamma']))
        merged_table['bump_area_unc'] = merged_table['bump_area']*bump_area_unc
        
    return(merged_table)

if __name__ == '__main__':

    merged_table = get_merged_table()
    print(merged_table.colnames)
    print(merged_table)
