# Stellar Classification Dataset

This dataset was created to merge 5 spectral series onto the same wavelength grid. Their labels were fetched from SIMBAD by querying the object identifier of each spectrum. 

The full set in file `dataset.fits` contains 17520 spectra from 12206 stars while the subset in file `dataset_subset.fits` contains 8622 spectra from 5674 stars. 

Details of the constituent series such as spectral resolution, wavelength range and number of observations are visible in this table:

| Telescope | Location | Series | Spectral<br>Resolution | Wavelength<br>Coverage | Number of <br>Observations | Subset |
|---|---|---|---|---|---|---|
| Sloan Foundation 2.5m | Apache Point Observatory, <br>New Mexico, USA | SDSS | 1.47 Å | 3621-10353 Å | 1736 | 699 |
| Coudé Feed 0.9m | Kitt Peak National <br>Observatory, Arizona, USA | CFLIB | 0.4 Å | 3465-9469 Å | 1167 | 805 |
| 1.93m | Observatoire de <br>Haute-Provence, France | ELODIE | 0.05 Å | 4000-6800 Å | 6285 | 3263 |
| " | " | SOPHIE | 0.01 Å | 3872-6944 Å | 7652 | 3551 |
| VLT Unit 2 8.2m | Atacama Desert, Chile | XSL | 0.17 Å | 3500-9939 Å | 680 | 304 |
|  | Unified Series |  | 1.21 Å | 4000-6800 Å | 17520 | 8622 |

The data was subseted to remove observations with:
* Rare spectral types outside O-M
* No luminosity class / low quality label (E)
* More than 1% of spectra missing
* Extreme emission features (normalized values bigger than 4)
* Extreme radial velocities (if the resulting doppler shift is >2 pixels)
* Blended features from nearby stars (Double stars and binary systems)

Resulting subset had this class distribution in the 2d grid:

![Class distribution of subset.](img/dataset_dist.png "Dataset Distribution")

## Acknowledgements of Data Sources
* Sloan Digital Sky Survey (SDSS-IV) MaNGA Stellar Library (MaStar)
[Web Page](https://www.sdss4.org/dr17/mastar/) - [Abstract](https://ui.adsabs.harvard.edu/abs/2019ApJ...883..175Y/abstract) - [Full Text](https://ui.adsabs.harvard.edu/link_gateway/2019ApJ...883..175Y/PUB_PDF)

MaStar is a collection of 23893 spectra. Each spectra had Manga ID's but not object identifiers. Known object identifiers were matched with Manga ID's via the accompanying [cross-match catalogue](https://www.sdss4.org/dr17/mastar/mastar-crossmatch/). In the end 1736 spectra was selected for the full dataset.

* The Indo-US Library of Coudé Feed Stellar Spectra (CFLIB)
[Web Page](https://noirlab.edu/science/observing-noirlab/observing-kitt-peak/telescope-and-instrument-documentation/cflib) - [Abstract](https://iopscience.iop.org/article/10.1086/386343) - [Full Text](https://iopscience.iop.org/article/10.1086/386343/pdf)

CFLIB is a spectral library with 1273 spectra, unique for each star. 885 of them are complete with no holes. For the wavelength range of our interest, and leaving a 1% margin that could be interpolated, 1167 spectra were chosen. (805 in the subset.)

* ELODIE/SOPHIE Archives
    * [ELODIE Archive Web Page](http://atlas.obs-hp.fr/elodie/)

    * [ELODIE library 3.1](https://perso.astrophy.u-bordeaux.fr/~csoubiran/elodie_library.html) - Older, curated version of ELODIE Archive
    [Abstract](https://ui.adsabs.harvard.edu/abs/2007astro.ph..3658P/abstract) - [Full Text (Preprint)](https://ui.adsabs.harvard.edu/link_gateway/2007astro.ph..3658P/EPRINT_PDF)

    This library contains 1962 spectra from 1388 stars. The full ELODIE archive contains 35535 spectra, of which 6285 were selected to represent a nearly-uniform temperature class distribution. Thus the ELODIE library itself was not utilized directly, though the linked paper highlights instrument and data reducing details.

    * [SOPHIE Archive Web Page](http://atlas.obs-hp.fr/sophie/)
    SOPHIE is the upgraded version of the ELODIE spectrograph. The SOPHIE archive contains more than 70000 spectra, of which 7652 were selected in a similar fashion to the ELODIE library.

* The X-Shooter Spectral Library
[Web Page](http://xsl.u-strasbg.fr/) - [Abstract](https://ui.adsabs.harvard.edu/abs/2022A%26A...660A..34V/abstract) - [Full Text](https://ui.adsabs.harvard.edu/link_gateway/2022A%26A...660A..34V/PUB_PDF)

XSL hosts 830 high-resolution spectra spanning wavelengths from NUV into NIR. Although built for a different purpose, 304 of their spectra were merged into this dataset.

* SIMBAD Astronomical Database - CDS (Strasbourg)
[SIMBAD Web Page](https://simbad.u-strasbg.fr/) - [Abstract](https://ui.adsabs.harvard.edu/abs/2000A%26AS..143....9W/abstract) - [Full Text](https://ui.adsabs.harvard.edu/link_gateway/2000A%26AS..143....9W/PUB_PDF)

SIMBAD database was used to query object identifiers and collect the following:
* Spectral class - Spectral class quality (A-E)
* Object type (Binary, Double)
* Radial velocity - Radial velocity uncertainty (km/s)

For each query, bibliography references are also retrieved, giving credence to the labels.