# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.io import fits

from data_utils import spectra_processing, class_splitter

# %%

hdul = fits.open('data/dataset_3.fits')
dataset = hdul[1].data
hdul.close()
sp_original = spectra_processing()

hdul = fits.open('data/dataset_7000_3.fits')
dataset_7000 = hdul[1].data
hdul.close()
sp_7000 = spectra_processing(exclude=['SDSS'])

# %%
i = 5090 #5060
# new_i = np.where(dataset_7000.IDS == dataset.IDS[i])[0][2]

import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=sp_original.sx_l, y=dataset.FLUX[i+1736], name='Low Res. (2305)'))
fig.add_trace(go.Scatter(x=sp_7000.sx_l, y=dataset_7000.FLUX[i], name='Higher Res. (7000)'))
fig.update_layout(
    title=f'Spectrum of {dataset.TYPED_ID[i]} ({dataset.SP_TYPE[i]})',
    xaxis_title='Angstrom')

fig.show()

# %%
filt = np.load('data/processed/filt.npy')
flux_numpy = dataset.FLUX.newbyteorder().byteswap()

elod_meta = pd.read_csv('data/source/ELODIE Full/elodie_meta.csv')
soph_meta = pd.read_csv('data/source/SOPHIE/sophie_meta.csv')
soph_meta.loc[:,'ID'] = soph_meta.ID.apply(str)

labeled_spectra = np.load('data/processed/labeled_spectra.npy')[filt]

spectra_nomed = np.load('data/processed/spectra_nomed2.npy')
final_cata = pd.read_csv('data/processed/final_cata3.csv')[filt]
extras = pd.read_csv('data/processed/final_cata3_extras.csv')
extras2 = pd.read_csv('data/Catalogues/simbad_tap_sptype2.csv')

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

plt.plot(flux_numpy[~keep][0])
# plt.yscale('log')

# %%
# ELODIE SN ex on same star

elod_sn_ex = final_cata.query('TYPED_ID == "HD044478"')
elod_sn_flux = flux_numpy[elod_sn_ex.index]

plt.figure(dpi=500)
plt.title('Same observations of star HD 44478')
plt.plot(elod_sn_flux[15]+1, linewidth=0.5, 
         label=f'SN: {elod_sn_ex.SN.values[15]}')
plt.plot(elod_sn_flux[14], linewidth=0.5, 
         label=f'SN: {elod_sn_ex.SN.values[14]}', alpha=0.6)
plt.legend();

# %%
plt.figure(dpi=500)
sn_ranges = np.arange(20, 500, 60)
elod_sn_ex.loc[:, 'FLUX'] = list(elod_sn_flux.reshape(len(elod_sn_flux),1,-1))
elod_sn_ex.loc[:, 'SN_CUT'] = pd.cut(elod_sn_ex.SN, bins=sn_ranges)
mean_flux = elod_sn_ex.groupby('SN_CUT', observed=False).FLUX.mean()

plt.plot(np.vstack(mean_flux).T+np.arange(len(sn_ranges)-1),
        label=mean_flux.index, linewidth=0.5)
plt.legend(loc=8, bbox_to_anchor=(0.5,-0.25), ncols=3, frameon=False);

plt.title('Fluxes of HD044478 with different SN values'
          '\n(Heights changed for visuals)');

# %%
plt.figure(dpi=500, figsize=(5,15))
elod_sn_ex.loc[:, 'FLUX'] = list(elod_sn_flux.reshape(len(elod_sn_flux),1,-1))
mean_flux = elod_sn_ex.groupby('SN_CUT', observed=False).FLUX.mean()

sorted_flux = np.vstack(elod_sn_ex.sort_values('SN').FLUX.values)

plt.plot(sorted_flux.T+np.arange(30),
        label=elod_sn_ex.ID, linewidth=0.5, )
# plt.legend(loc=8, bbox_to_anchor=(0.5,-0.25), ncols=3, frameon=False);

plt.title('Fluxes of HD044478'
          '\n(Heights changed for visuals)');


# %%
# MATCHBOX inc.
# Profile:
# Early galaxy (300 Myr old) looking for:
# SN & SN19 > 10
# HAS TO KNOW THEIR WEIGHT CLASS
# NON-BINARY (lol) (can be optical double)
# NORMAL (No C, S, OB etc.)
# stars to host

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
import scipy as sp

def information_criteria(row):
    splits = np.array_split(row, 4)
    width = [*range(5,20,5)]
    crit = []
    for split in splits:
        peaks = sp.signal.find_peaks_cwt(split, width)
        hist = np.histogram(np.diff(peaks))[0]
        ent = sp.stats.entropy(hist)

        std = np.std(np.diff(peaks))
        m = np.mean(np.diff(peaks))
        crit.append(1/ent+std/m)
    return crit

# %%
from tqdm import auto

info = []

for row in auto.tqdm(flux_numpy):
    info.append(information_criteria(row))

# %%
x = periodicity(sorted_flux[0])
y = periodicity(sorted_flux[-12])

print(x)
print(y)

# %%

import scipy as sp
i = 0
peaks = sp.signal.find_peaks_cwt(sorted_flux[i], widths=[5,10,15])
plt.figure(dpi=500)
plt.plot(sorted_flux[i], linewidth=0.5)
plt.plot(peaks, sorted_flux[i][peaks], "x")
plt.plot(np.zeros_like(sorted_flux[i]), "--", color="gray")

plt.show()
# %%
plt.plot(np.diff(peaks))

# %%

import statsmodels.api as sm

y = np.log(np.diff(peaks))
x = np.arange(len(y))
x = sm.add_constant(x)

model = sm.OLS(y, x)
results = model.fit()
results.rsquared

# %%

make_pred = lambda x: results.params[0] + x*results.params[1]

plt.plot(y)
plt.plot(np.sum)

# %%
peaks2 = sp.signal.find_peaks_cwt(sorted_flux[-12], widths=[5,10,15])
plt.figure(dpi=500)
plt.plot(sorted_flux[-12], linewidth=0.5)
plt.plot(peaks, sorted_flux[-12][peaks2], "x")
plt.plot(np.zeros_like(sorted_flux[-12]), "--", color="gray")

# %%
import scipy as sp
x1 = np.array([0]*2305)
x1[1152] = 1

fft_x1 = sp.fft.fft(x1)

plot_fft = lambda x: plt.scatter(x.real,x.imag,alpha=0.3)

plot_fft(fft_x1)

# %%

import scipy as sp
x2 = np.linspace(-1152,1152,1000)
x2 = np.sinc(x2)

fft_x2 = sp.fft.fft(x2)

plot_fft = lambda x: plt.scatter(x.real,x.imag,alpha=0.3)

plot_fft(fft_x2)

# %%

import scipy as sp
x3 = np.linspace(-1152,1152,1000)
x3 = np.sin(x3)

fft_x2 = sp.fft.fft(x3)

plot_fft = lambda x: plt.scatter(x.real,x.imag,alpha=0.3)

plot_fft(fft_x2)




# %%

from sklearn.metrics import mean_squared_error

import scipy as sp

def frequency_deviation(flux):
    '''Measures how much the frequency distribution of 
    a signal deviates from 0 for both real and imaginary
    portions. This doesn't mean anything on its own, but
    will be used to identify anomalies.'''
    fft = sp.fft.fft(flux)
    err_real = mean_squared_error(fft.real[1:], [0]*len(flux[1:]))
    err_imag = mean_squared_error(fft.imag, [0]*len(flux))
    return err_real + err_imag

frequency_deviation(dataset.FLUX[99])

# %%
frequency_deviations = [frequency_deviation(x) for x in dataset.FLUX]

plt.hist(frequency_deviations,bins=30)

# %%
id1 = final_cata.reset_index().query('ID == "828690"').index[0]
id2 = final_cata.reset_index().query('ID == "25-82"').index[0]
id3 = final_cata.reset_index().query('ID == "X0487"').index[0]

what = sp.fft.fft(dataset.FLUX[id1])
what2 = sp.fft.fft(dataset.FLUX[id2])
what3 = sp.fft.fft(dataset.FLUX[id3])

# plt.scatter(what.real, what.imag, alpha=0.2, label=f'{final_cata.iloc[id1].TYPED_ID} Error: {frequency_deviations[id1]:.2f}')
# plt.scatter(what2.real, what2.imag, alpha=0.2, label=f'{final_cata.iloc[id2].TYPED_ID} Error: {frequency_deviations[id2]:.2f}')
plt.scatter(what3.real, what3.imag, alpha=0.2, label=f'{final_cata.iloc[id3].TYPED_ID} Error: {frequency_deviations[id3]:.2f}')
plt.legend()


# %%
# %%
import tqdm

# fd = np.array(frequency_deviations)[keep]
# np.quantile(fd, q=[0.01, 0.99])
# idx = (fd > 40) & (fd < 120)

idx = np.array(info) < 1.2

for i,row in enumerate(tqdm.tqdm(dataset[keep & idx])):
    flux = row[0].newbyteorder().byteswap()
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(flux ,linewidth=0.5)
    ax.set_title(f'Star name: {row[2]}\n Class: {row[4]} \n Info: {np.array(info)[idx][i]}')
    ax.set_ylabel('Flux (arbitrary)')
    ax.set_xlabel('Wavelength (nm)')
    fig.savefig(f'data/low_info/{row[1]}.jpg')
    plt.close(fig)


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

# fix pesky lum stuff
# lum = lum.apply(lambda x: x[0] if x=='Iab/b' else x)
# lum = lum.apply(lambda x: x[:-1] if (x[-1] == '-') | (x[-1] == '/') else x)

# convert to numbers
# lum = lum.str.split('-|/').apply(lambda x: np.mean([lum_conv(y) for y in x]))
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
# %%
