# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.io import fits

from utils.data_utils import spectra_processing, class_splitter

# %%

hdul = fits.open('data/dataset.fits')
dataset = hdul[1].data
hdul.close()
sp_original = spectra_processing()

# %%
filt = np.load('data/processed/filt.npy')
flux_numpy = dataset.FLUX.newbyteorder().byteswap()

elod_meta = pd.read_csv('data/source/ELODIE Full/elodie_meta.csv')
soph_meta = pd.read_csv('data/source/SOPHIE/sophie_meta.csv')
soph_meta.loc[:,'ID'] = soph_meta.ID.apply(str)

final_cata = pd.read_csv('data/processed/final_cata3.csv')[filt]
extras = pd.read_csv('data/processed/final_cata3_extras.csv')
extras2 = pd.read_csv('data/catalogues/simbad_tap_sptype2.csv')

# %%

final_cata.loc[:, 'OTYPES'] = extras.OTYPES
final_cata = final_cata.merge(elod_meta.loc[:, ['ID', 'SN']], on='ID', how='left')
final_cata = final_cata.merge(soph_meta.loc[:, ['ID', 'SN19']], on='ID', how='left')

binary = ['SB*', 'EB*', 'HXB', 'LXB']
 
# '**' left out because not all double stars mean binary systems.
# binary systems interact with each other and their lines can shift
# periodically. If a star has less than 50 km/s radial velocity,
# with the current 1.4 A resolution, we wouldn't detect any change
# in the observed spectra. Past this, pixels at the red end will 
# get affected the most. (v/c = Δλ/λ)
# (for a resolution of 0.4 A this limit drops to about 18 km/s.)

# leaving double stars in is still a limitation of the dataset.

find_bin = lambda y: any(x in y for x in binary)

bin1 = final_cata.OTYPES.apply(find_bin)
bin2 = final_cata.SP_TYPE.str.contains('\+') 
# 290 have plus sign in SP_TYPE but no specific binary
# 396 are a binary

binaries = np.array(bin1 | bin2)

# %%

rvz_cols = [col for col in extras2.columns if col.startswith('rvz')]
rvz = extras2.drop_duplicates('TYPED_ID').loc[:,['TYPED_ID']+rvz_cols]

final_cata = final_cata.merge(rvz, on='TYPED_ID', how='left')

# %%
# our models should be able to tolerate shifts of up to 2 pixels (assumption)
rvz_limit = np.abs(final_cata.rvz_redshift)*6800 > 1.2*2 # 0.4 for 7000

pd.Series(rvz_limit).value_counts()

# %%

from scipy.stats import chi2_contingency

chi2_contingency(pd.crosstab(rvz_limit, final_cata.rvz_qual != 'E'))
# needs_fix still exists a lot for good quality rvz measurements.

# %%
inter = np.max(flux_numpy, axis=1)
keep = inter < np.quantile(inter, q=0.98) #was 95% before

print('\033[1mDistribution of maxima for the subset \033[0m')
print(pd.Series(inter[keep]).describe(), end='\n\n')

print('\033[1mDistribution of maxima for the outliers \033[0m')
print(pd.Series(inter[~keep]).describe())

# %%

def spec_in_main(data, main):
  # Convert allowed_letters to a set for efficient membership checking
  main_set = set(main)

  # Create a boolean array using NumPy's vectorized operations
  boolean_array = np.array([spec in main_set for spec in data])

  return boolean_array

sp = spec_in_main(final_cata.spec, list('OBAFKGM'))
sp_q = (final_cata.SP_QUAL != 'E').values
lum = final_cata.lum.isna().values

ultimate_filter = sp & sp_q & ~lum & ~rvz_limit & keep & ~binaries
print(f'Number of observations in subset: {sum(ultimate_filter)}')

# %%
lum = pd.Series(dataset.LUM).replace({'':np.NaN}).dropna()

lums = '0|I|II|III|IV|V|VI'.split('|')

def lums_conv(x):
    if (x[-1] == '-') | (x[-1] == '/'):
        x = x[:-1]
    if x == 'Iab/b':
        x = x[0]
    if len(x.split('-')) == 1:
        x = x.split('/')
    else:
        x = x.split('-')
    out = [lum_conv(y) for y in x]
    return np.mean(out)

def lum_conv(x):
    if x.endswith('a'):
        x = lums.index(x[:-1])
        x -= .25
    elif (x.endswith('b')) & (not x.endswith('a', 0,-1)):
        x = lums.index(x[:-1])
        x += .25
    else:
        x = x.split('ab')[0]
        x = lums.index(x)
    return x

lum = lum.apply(lums_conv)
lum[lum < 0.5] = 0.55
lum

# %%
lums2 = lums[1:]

final_lum2 = lum.apply(lambda x: lums2[int(np.round(x))-1])

main = list('OBAFGKM')

spec = pd.Series(dataset.SPEC)[dataset.LUM != '']

cross = pd.crosstab(spec,final_lum2).loc[main,lums2]

import seaborn as sns
sns.heatmap(cross,annot=True, fmt='.0f', cmap='mako', 
            square=True, cbar=False)
plt.yticks(rotation=0)
plt.ylabel('Temperature (Most to least)')
plt.xlabel('Luminosity (Most to least)')
plt.title('Distribution of Stars in Merged Set');


# %%

lum_num = pd.Series(dataset.LUM)
lum_num.update(lum)
lum_num = lum_num.replace({'':np.NaN}).values

# %%
def sub_conv(x):
    if x == x:
        if len(x.split('-')) == 1:
            x = x.split('/')
        else:
            x = x.split('-')
        if x[-1] == '':
            x = x[0]

        out = [float(y) for y in x]
        return np.mean(out)
    else:
        return x

sub_num = final_cata['sub'].apply(sub_conv).values
sub_num
# %%
def spec_conv(row):
    sp = np.NaN
    if row == row:
        try:
            sp = main.index(row)
        except:
            pass
    return sp

spec_num = final_cata.spec.apply(spec_conv).values

# %%

spec_col = fits.Column(name='SPEC_NUM', array=spec_num, format='1E')
sub_col = fits.Column(name='SUB_NUM', array=sub_num, format='1E')
lum_col = fits.Column(name='LUM_NUM', array=lum_num, format='1E')

fits_columns = [*dataset.columns] + [spec_col, sub_col, lum_col]

t = fits.BinTableHDU.from_columns(fits_columns)
t = fits.BinTableHDU(data=t.data[ultimate_filter])
t.writeto('data/dataset3_subset2.fits', overwrite=True)