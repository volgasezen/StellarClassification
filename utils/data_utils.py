import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.interpolate import CubicSpline
from suppnet.suppnet.NN_utility import get_suppnet, density_norm
from tqdm import tqdm
import os, sys

# HiddenPrints written by Alexander C 
# from https://stackoverflow.com/a/45669280

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def class_splitter(data, *args):
    '''Parses star class strings into four distinct 
    categories: (class, subclass, luminosity, misc)

    Returns:
        If input is a dataframe:
            The dataframe with the class categories appended
        If it is just the SpType column:
            A dataframe with class categories
    '''
    pat = r'^sd:?([A-Z][A-DFHJ-UWYZ]?)(\d(?:\/|\.|\-)?\d?)?(.*)?'
    rep = r'\1\2VI+\3'

    pat2 = r'^d:?([A-Z][A-DFHJ-UWYZ]?)(\d(?:\/|\.|\-)?\d?)?(.*)?'
    rep2 = r'\1\2V+\3'

    query = '([A-Z][A-DFHJ-UWYZ]?)' \
            '(\d(?:\/|\.|\-)?\d?)?(?:\+|:)?' \
            '((?:III|II|IV|I|0|VI|V)(?:ab|a|b)?(?:\/|\-)?' \
            '(?:III|II|IV|I|0|VI|V)?(?:ab|a|b)?)?(.*)?'

    if isinstance(data, pd.DataFrame):
        spcol = args[0]
        df = data[spcol]

        df = df.str.replace(pat, rep, regex=True)
        df = df.str.replace(pat2, rep2, regex=True)
        df = df.str.extract(query).replace('', np.NaN)

        df.columns = ['spec', 'sub', 'lum', 'misc']
        data = data.merge(df, right_index=True, left_index=True)

        return data

    elif isinstance(data, pd.Series):
        df = data.str.replace(pat, rep, regex=True)
        df = df.str.replace(pat2, rep2, regex=True)
        df = df.str.extract(query).replace('', np.NaN)

        df.columns = ['spec', 'sub', 'lum', 'misc']

        return df

    else:
        raise TypeError('Input must be a pandas object.')

class spectra_processing:
    def __init__(self, exclude: list = ''):
        
        x = np.load('wave_grids.npz')
        self.names = [*x]

        if exclude:
            for ex in exclude:
                self.names.pop(self.names.index(ex))

        self.xs = [x[n] for n in self.names]

        start = max([x[0] for x in self.xs])
        end = min([x[-1] for x in self.xs])

        self.r_xs = [np.mean(np.diff(x)) for x in self.xs]

        self.limits = [[(x > start) & (x < end)] for x in self.xs]
        xs_l = [x[(x > start) & (x < end)] for x in self.xs]
        sx = xs_l[np.argmax(self.r_xs)]
        self.sx_l = np.linspace(sx[0], sx[-1], len(sx))

        sigma = [self.sigma_gen(max(self.r_xs),r) for r in self.r_xs]
        self.sigmas = dict(zip(self.names, sigma))

    def density_minmax(self, x, thresh = 0.9):
        '''Scaling based on a given percentile 
        instead of max (default=90th percentile)

        Returns:
            Scaled version of the row
        '''
        q = np.quantile(x, q = thresh, axis=1)
        x_std = (x.T - x.min(axis=1))/(q - x.min(axis=1))
        return x_std.T

    def sigma_gen(self, x, y):
        # Returns a sigma when x, y are fwhm 
        f = 2*np.sqrt(2*np.log(2))
        return np.sqrt(np.abs(1/x**2-1/y**2))/f

    def na_start(self, row):
        '''For a single row, repeat the first 
        non-na observation to fill in the first 
        few na observations

        Returns:
            Row with first few na's
            filled in
        '''
        nans = np.isnan(row)
        if sum(nans) != 0:
            ix = np.where(np.diff(nans.astype('int'))==-1)[0]
            row[nans] = row[ix+1]
        return row

    def rem_na(self, x: np.array):
        '''Fill in na values with linear interpolation.

        Returns:
            Entire flux array ridden of na's
        '''
        x = pd.DataFrame(x).interpolate(method='linear', axis=1).values
        x = np.apply_along_axis(self.na_start, 1, x)
        return x

    def func_maker(self, name: str):
        '''Returns custom functions for
        resampling depending on which series
        the fluxes belong to

        Arguments:
            name -- Must be one of below:
            SDSS, CFLIB, ELODIE, SOPHIE, XLS

        Returns:
            Cubic spline interpolator
            Gaussian convolver
        '''
        def cubic_spline(flux: np.array):
            x = self.xs[self.names.index(name)]
            xa = CubicSpline(x, flux, axis=1)

            return xa(self.sx_l)

        def gaussian(flux: np.array):
            s = self.sigmas[name]
            return gaussian_filter1d(flux, s, 
                                     mode='nearest', axis=1)

        return cubic_spline, gaussian

    def normalizer(self, y: np.array, norm: str):
        if norm == 'median':
            y = self.density_minmax(y)

            flux_arr = y-median_filter(
                y, size=100*int(len(self.sx_l)/2305), 
                mode='nearest', axes=1
            )
            print('Spectra normalized by median filter response.')
        elif norm == 'suppnet':
            nn = get_suppnet(thresh=0.9)
            for i, flux in enumerate(tqdm(y)):
                y_norm = nn.normalize(self.sx_l, flux)
                y_rescale = nn.normalizer.normalize(flux)
                y_rescale = np.expand_dims(y_rescale, axis=0)
                if i == 0:
                    flux_arr = np.append(y_rescale, np.array(y_norm), axis=0)
                else:
                    flux_arr = np.append(
                        flux_arr, 
                        np.append(y_rescale, np.array(y_norm), axis=0),
                        axis=0
                    )
            print('Spectra normalized by SUPPNet predictions.')
        else:
            flux_arr = y
            print('Spectra was not normalized.')
        
        return flux_arr


    def preprocess(self, y: np.array, name: str, norm: str):
        '''Main preprocessor of spectra. Interpolates
        na values, convolves them with a defined kernel,
        resamples the wavelength grid to a common one,
        and does normalization if wanted.

        Arguments:
            x: Flux array (2d)
            name: Must be one of
            SDSS, CFLIB, ELODIE, SOPHIE, XSL
            norm: Can be one of below:
                med (normalize by median response)
                suppnet (normalize by suppnet prediction)
                '' (do not normalize)

        Returns:
            Preprocessed flux array.
        '''
        if y.dtype.names:
            raise TypeError('Provided array has many fields. '\
                            'Please only provide the field with '\
                            'FLUX values.')

        print(f'Starting {name} preprocessing!')
        match name:
            case 'CFLIB':
                y = np.where(y<1.1e-4, np.nan, y)
            case 'SDSS':
                y = np.where(y==0, np.nan, y)
                y = np.where(~np.isfinite(y), np.nan, y)

        y = self.rem_na(y)
        print(f'NAs removed...')

        cubic, gauss = self.func_maker(name)

        if name != self.names[np.argmax(self.r_xs)]:
            y = gauss(y)
            print(f'Spectra convolved...')
        else:
            print(f'Spectra not convolved ({max(self.r_xs):.2f} Å)...')

        y = cubic(y)

        res = self.r_xs[self.names.index(name)]
        print(f'Spectra resampled ({res:.2f} Å -> {max(self.r_xs):.2f} Å)...')

        y = self.normalizer(y, norm)

        return y

    def batch_processor(self, y: np.array, col: str, name: str, normed: bool, norm: str):
        '''Does preprocessing in a batched 
        fashion. Same args as preprocessor
        except for col: column name of fluxes 

        Returns:
            Preprocessed flux array.
        '''
        batch_size = len(y)

        if batch_size <= 1:
            raise TypeError('Provided array is not batched.'\
                            ' Please use the preprocess' \
                            ' function instead!')

        if normed:
            for i in tqdm(range(batch_size)):
                with HiddenPrints():
                    y[i] = self.normalizer(y[i], norm)
            return np.vstack(y)

        else:
            for i in tqdm(range(batch_size)):
                with HiddenPrints():
                    y[i] = self.preprocess(y[i][col], name, norm)
            return np.vstack(y)
            