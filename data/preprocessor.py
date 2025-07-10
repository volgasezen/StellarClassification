# This file is the origin of "dataset_3.fits". It can be recreated by aligning
# processed/labeled_spectra.npy and final_cata. Note however, that they were
# subsetted by the array processed/filt.npy. This is a filter over concatenated
# arrays and merged set. The goal is to eliminate examples with too many empty
# spaces.

# %%
import numpy as np
import pandas as pd
import scipy as sp

import seaborn as sns
import matplotlib.pyplot as plt
from astropy.io import fits

from utils.data_utils import spectra_processing

# %%

proc = spectra_processing()
print(f'Processing: {proc.names}')


sdss_ids = np.load('data/source/MaSTAR/sdss_ids.npy')
elod_ids = np.load('data/source/ELODIE Full/elodie_ids.npy')
elodie_v0 = np.load('data/source/ELODIE Full/elodie.npy')
soph_ids = np.load('data/source/SOPHIE/sophie_ids3.npy')

# CFLIB
cflib_v0 = np.load('data/source/CFLIB/cflib.npy')
cflib_ids = cflib_v0['ID']

# XSL
xsl = np.load('data/source/XSL/xsl.npy')
xsl_ids = xsl['NAME']

#SDSS

# hdul = fits.open('data/source/MaSTAR/mastar-combspec-v3_1_1-v1_7_7-lsfpercent99.5.fits.gz')
# sdss_v0 = hdul[1].data
# cols = hdul[1].columns
# hdul.close()
# sdss_ids = sdss_v0['MANGAID']

# %%
merged = pd.read_csv('data/catalogue/merged.csv')

sdss = np.load('data/processed/sdss_nomed.npy')
cflib = np.load('data/processed/cflib_nomed2.npy')
elodie = np.load('data/processed/elodie_nomed.npy')
sophie = np.load('data/processed/sophie_nomed2.npy')
xsl = np.load('data/processed/xsl_nomed.npy')

assert sdss.shape[1] == cflib.shape[1] == elodie.shape[1] == sophie.shape[1] == xsl.shape[1]

factor = int(elodie.shape[1]/2305)

# %% 
# some observations correspond to multiple stars. will drop them.

id_per_ids = merged.groupby('ID')['IDS'].nunique()
double_ids = id_per_ids[id_per_ids != 1]

# merged.query('ID in @double_ids.index').groupby('ID')['SP_TYPE'].apply(lambda x: all(x == x[0] for x in x))
check_same = lambda x: all(x == x[0] for x in x)
assert ~any(merged.query('ID in @double_ids.index').groupby('ID')['SP_TYPE'].apply(check_same))

merged.query('ID not in @double_ids.index', inplace=True)
merged.reset_index(drop=True, inplace=True)

duplicate = merged.duplicated()

merged = merged[~duplicate]

# %%

sdss_cata_ids = merged.query('dataset=="SDSS"').ID.apply(str.strip).values
sdss_keep = [str.strip(x) in sdss_cata_ids for x in sdss_ids]

# sdss0 = sdss_v0['FLUX'][:,proc.limits[1][0]]
# sdss0 = np.where(sdss0==0, np.nan, sdss0)
# sdss0 = np.where(~np.isfinite(sdss0), np.nan, sdss0)

# s_zeros = np.sum(np.isnan(sdss0),axis=1)
# s_zeros # maximum of 34 na's inside the limit. no further action

print(sdss[sdss_keep].shape) # 22194 missing / 1736


# %%

cflib_cata_ids = merged.query('dataset=="CFLIB"').ID.apply(str.strip).values
cflib_keep = [x in cflib_cata_ids for x in cflib_ids]

zeros = np.sum(cflib_v0['FLUX'][:,proc.limits[0][0]] < 1.1e-4, axis=1)
cfl_idx = zeros<200*factor # 1170 filtering based on the common wavelength range

print(cflib[cflib_keep].shape) # 1267 (6 probably didn't have labels)
print(cflib[cflib_keep & cfl_idx].shape) # 1167

# %%

nr_obs = dict(merged.dataset.value_counts())

# %%
elod_cata_ids = merged.query('dataset=="ELODIE"').ID.apply(str.strip).values
elod_keep = [x in elod_cata_ids for x in elod_ids]

e_zeros = np.sum(np.isnan(elodie_v0['FLUX']), axis=1)
# ç = range(0, 10000, 100)
# plt.plot(ç, [sum(e_zeros < x)/nr_obs.get('ELODIE') for x in ç])
elo_idx = e_zeros<1000 # 6455 / total: 7283

print(elodie[elod_keep].shape) # 7047 (6 probably didn't have labels)
print(elodie[elod_keep & elo_idx].shape) # 6285

# %%

# soph_nan = np.unpackbits(np.load('data/SOPHIE/sophie_nan_z.npy'))[:-1].reshape(14141,-1)

# soph_zeros = np.sum(soph_nan[:,proc.limits[2][0]], axis=1) # No nans detected

# ç = range(0, 1000, 10)
# plt.plot(ç, [sum(soph_zeros < x)/len(sophie) for x in ç])

soph_cata_ids = merged.query('dataset=="SOPHIE"').ID.apply(str.strip).values
soph_keep = [x in soph_cata_ids for x in soph_ids]

sophie[soph_keep].shape # 7652

# %%
xsl_cata_ids = merged.query('dataset=="XSL"').ID.apply(str.strip).values
xsl_keep = [x in xsl_cata_ids for x in xsl_ids]

sum(xsl_keep) # 680

# %%

keep = np.concatenate([
    sdss_keep,
    cflib_keep,
    elod_keep,
    soph_keep,
    xsl_keep,
])

np.save('data/processed/keep.npy', keep)


# %%

sdss_med = proc.normalizer(sdss[sdss_keep], 'median')
cflib_med = proc.normalizer(cflib[cflib_keep], 'median')
elodie_med = proc.normalizer(elodie[elod_keep], 'median')
sophie_med = proc.normalizer(sophie[soph_keep], 'median')
xsl_med = proc.normalizer(xsl[xsl_keep], 'median')

# %%
sdss_med = sdss[sdss_keep] / sp.ndimage.median_filter(sdss[sdss_keep], 100, mode='nearest', axes=1)
print('1/5 SDSS median filtering is done...')
cflib_med = cflib[cflib_keep] / sp.ndimage.median_filter(cflib[cflib_keep], 100, mode='nearest', axes=1)
print('2/5 CFLIB median filtering is done...')
elodie_med = elodie[elod_keep] / sp.ndimage.median_filter(elodie[elod_keep], 100, mode='nearest', axes=1)
print('3/5 ELODIE median filtering is done...')
sophie_med = sophie[soph_keep] / sp.ndimage.median_filter(sophie[soph_keep], 100, mode='nearest', axes=1)
print('4/5 SOPHIE median filtering is done.')
xsl_med = xsl[xsl_keep] / sp.ndimage.median_filter(xsl[xsl_keep], 100, mode='nearest', axes=1)
print('5/5 XSL median filtering is done.')
# %%
labeled_spectra = np.vstack([
    sdss_med, cflib_med, 
    elodie_med, sophie_med, 
    xsl_med
])

labeled_spectra.shape
# %%
np.save('data/processed/labeled_spectra.npy', labeled_spectra)
# labeled_spectra = np.load('data/processed2/labeled_spectra.npy')


# %%
# DOING THIS BECAUSE CFLIB IDS AND SOPHIE IDS
# COINCIDE. ID+DATASET at the very least is unique now.

ids = np.concatenate([
    [str.strip(x)+'SDSS' for x in sdss_ids[sdss_keep]],
    [x+'CFLIB' for x in cflib_ids[cflib_keep]], 
    [x+'ELODIE' for x in elod_ids[elod_keep]],
    [x+'SOPHIE' for x in soph_ids[soph_keep]],
    [x+'XSL' for x in xsl_ids[xsl_keep]]
    ])

merged.ID = merged.ID.apply(str.strip)
merged.loc[:,'cata_merge'] = merged.ID + merged.dataset
merged = merged.set_index('cata_merge').loc[ids].reset_index(drop=True)

merged.to_csv('data/processed/final_cata3.csv',index=False)

# %%
# DOING THIS TO EXPULGE SPECTRA
# WITH LARGE # OF EMPTY SPACES

filt = np.concatenate([
    np.repeat(True, sum(sdss_keep)),
    cfl_idx[cflib_keep],
    elo_idx[elod_keep],
    np.repeat(True,sum(soph_keep)),
    np.repeat(True,sum(xsl_keep)),
])

np.save('data/processed/filt.npy', filt)

# %%
# Packaging everything up
# in a fits table

merged.fillna('', inplace=True)
merged = merged[filt]

fits_columns = [fits.Column(name='FLUX', array=labeled_spectra[filt], 
                            format='2305D')] # 2305E for float32

for col in merged.columns:

    fmt = max([len(x) for x in merged[col]])
    if col == 'lum':
        fmt += 1

    fits_col = fits.Column(name=col.upper(), array=merged[col], 
                           format=f'{fmt}A')
    fits_columns.append(fits_col)

t = fits.BinTableHDU.from_columns(fits_columns)

t.writeto('data/dataset_3.fits', overwrite=True)
