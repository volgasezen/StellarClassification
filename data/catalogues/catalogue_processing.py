# %%
import os
os.chdir('/home/oban/Desktop/Volga/stellar-classification/')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from astropy.io import fits
from astroquery.simbad import Simbad
import re

from data_utils import class_splitter

# utils function class_splitter now handles in-between cases
# but they need to be processed by taking means or choosing

Simbad.add_votable_fields('sptype', 'ids', 'typed_id')
Simbad.TIMEOUT = 120

# %% 
# SDSS 
# process crossmatch table from mastar

hdul = fits.open('data/MaSTAR/mastarall-gaiaedr3-extcorr-simbad-ps1-v3_1_1-v1_7_7-v1.fits')

data = hdul[1].data
cols = hdul[1].columns

hdul.close()

df = pd.DataFrame(data.tolist(), columns=cols.names)
subset = df.loc[:,['MANGAID', 'SIMBAD_MAIN_ID', 'OTYPE_S', 'SP_TYPE', 'SP_QUAL', 'SP_BIBCODE',
       'MK_DS', 'MK_SPECTRAL_TYPE', 'MK_BIBCODE']]

subset = subset.map(str.strip).replace('', np.NaN)
subset = subset[subset['SIMBAD_MAIN_ID'].notna()]

ind = subset.SP_TYPE.isna()
subset.loc[ind, 'SP_TYPE'] = subset[ind]['MK_SPECTRAL_TYPE']

# %% query simbad ID's from mastar catalogue for additional labels

result_table = Simbad.query_objects(subset['SIMBAD_MAIN_ID'])
res = result_table.to_pandas()

sdss_sp = res.iloc[:,[-2,-3,-6,-5,-4]].query('SP_TYPE != ""')
# %%
# MAIN_ID in SIMBAD is not exactly the same as SIMBAD_MAIN_ID
# in the subset dataframe. but querying works, so we will ignore
# when in doubt, check names in IDS

len(set(res['MAIN_ID']) & set(subset['SIMBAD_MAIN_ID']))

sdss = []
for row in res['IDS'].apply(lambda x: x.split('|')):
    sdss.extend(row)

len(set(subset['SIMBAD_MAIN_ID']) & set(sdss))
# %%
# simbad labels are overall better, and have quality measures

sdss_sp = class_splitter(sdss_sp, 'SP_TYPE')
sdss_sp.reset_index(drop=True, inplace=True)
# sdss_sp['ID'] = subset.query('SIMBAD_MAIN_ID in @sdss_sp.TYPED_ID').MANGAID.reset_index(drop=True)

# sdss_sp = sdss_sp.iloc[:,np.roll(np.arange(9),1)]

# %% 
# CFLIB
# import catalogue 

hdul = fits.open('data/CFLIB/cflibdb.fits')

data = hdul[1].data
cols = hdul[1].columns

hdul.close()

cflib = pd.DataFrame(data.tolist(), columns=cols.names)

# %%
# clean cflib ID's by appending HD prefix if first char is digit

id_cleaner = lambda x: f'HD{x.zfill(6)}' if re.match('^\d',x) else x

cflib['ID'] = cflib.TITLE.apply(id_cleaner).apply(str.strip)
result_table2 = Simbad.query_objects(cflib.ID)
res2 = result_table2.to_pandas()

# %%
# keep old ids to refer back to the files
# and get labels

# res2['ID'] = cflib.TITLE
cflib_sp = res2.iloc[:,[-2,-3,-6,-5,-4]].query('SP_TYPE != ""')
cflib_sp = class_splitter(cflib_sp,'SP_TYPE')
# %%
# actually, won't exclude weird classes.
# weird is cool

# weird = sdss_sp.spec.value_counts()[10:].index.tolist()
# weirdc = cflib_sp.spec.value_counts()[8:].index.tolist()
# 
# sdss_final = sdss_sp.query('spec not in @weird')
# cflib_final = cflib_sp.query('spec not in @weirdc')

# %%
# SOPHIE library

path = "data/SOPHIE/sophies_1709895388.txt"

sophie = open(path, 'r')
sophie = [*sophie]

skip = np.where([x.startswith('#') for x in sophie])[0]

soph = pd.read_table(path, sep='\t', skiprows=skip)

soph.columns = ['objname','j2000','S','E','seq','slen','date','mode',
                'fiber_b','exptime','sn26','view_spec','view_head',
                'get_spec','get_e2ds','customize','search_ccf']

# %%
# some objects aren't in SIMBAD. no redundancy

soph.objname = soph.objname.apply(str.strip)

with open('data/SOPHIE/sophie_no_simbad.txt', 'r') as outfile:
    banned = outfile.read().split('\n')

sophs = set(soph.objname) - set(banned)

# %%
# lots of objects to be queried
# it is divided to 4 parts

sophs = list(sophs)
sophs1, sophs2, sophs3, sophs4 = np.array_split(np.array(sophs), 4)

result_table = Simbad.query_objects(sophs1)
result_table2 = Simbad.query_objects(sophs2)
result_table3 = Simbad.query_objects(sophs3)
result_table4 = Simbad.query_objects(sophs4)

res = result_table.to_pandas()
res2 = result_table2.to_pandas()
res3 = result_table3.to_pandas()
res4 = result_table4.to_pandas()

soph_simbad = pd.concat([res,res2,res3,res4],axis=0)

# %%

soph_simbad = pd.read_csv('data/SIMBAD Queries/soph_simbad_2.csv')
# %%
# usual stuff, split classes etc.

soph_sp = soph_simbad.iloc[:,[-2,-3,-6,-5,-4]].query('~SP_TYPE.isnull()')
soph_sp = class_splitter(soph_sp.reset_index(drop=True), 'SP_TYPE')
# soph_sp = soph_sp.rename(columns={'TYPED_ID':'ID'})

# %%
# soph_simbad.to_csv('data/SIMBAD Queries/soph_simbad2.csv')
# soph_sp.to_csv('data\SOPHIE\sophie_sp_2.csv',index=False)

# 
# soph_sp = pd.read_csv('data\SOPHIE\sophie_sp_2.csv')
# soph_sp_old = pd.read_csv('data/SIMBAD Queries/soph_simbad.csv')
# 
# 
# soph_sp_old = soph_sp_old.query('~SP_TYPE.isnull()')
# soph_sp_old = soph_sp_old.rename(columns={'TYPED_ID':'ID'})
# 
# soph_sp.merge(soph_sp_old, on='ID', how='left')
# 
# 
# soph_sp_compare = soph_sp.merge(soph_sp_old, on='ID', how='left')
# soph_sp_compare.query('SP_TYPE_x != SP_TYPE_y')[['SP_TYPE_x', 'SP_TYPE_y']]

# %%
# ELODIE trial

path = 'data/ELODIE Full/e500_1710941926.txt'

elodie = open(path, 'r')
elodie = [*elodie]

skip = np.where([x.startswith('#') for x in elodie])[0]

# %%

elod = pd.read_table(path, sep='\t', skiprows=skip, header=None)
# %%

elod.columns = ['objname','j2000','S','O','dataset','imanum',
                'imatyp','exptime','sn','view_spec','view_head',
                'get_spec','get_e2ds','customize','search_ccf']
# %%

elod.head()

# %%
elod.objname = elod.objname.apply(str.strip)
elods = set(elod.objname)-set(sophs)-set(banned)
elods = list(elods)
print(f'ELODIE Objects: {len(elods)}, SOPHIE Objects: {len(sophs)}')

# %%

import re

def unified_ids(x):
    X = re.sub('(.*)(HD\s?\d*)(.*)', r'\2', x)

    if X.startswith('J'):
        X = '2MASS ' + X
    
    return X

unified_ids = np.vectorize(unified_ids)

elods1, elods2, elods3, elods4 = np.array_split(unified_ids(elods), 4)

# %% 
# Doesn't work. The loop below works but
# it takes ~17 minutes to finish

# Simbad.TIMEOUT = 240
result_table1 = Simbad.query_objects(elods1)
res1_1 = result_table1.to_pandas()

result_table2 = Simbad.query_objects(elods2)
res2_1 = result_table2.to_pandas()

result_table3 = Simbad.query_objects(elods3)
res3_1 = result_table3.to_pandas()

result_table4 = Simbad.query_objects(elods4)
res4_1 = result_table4.to_pandas()

elod_simbad = pd.concat([res1_1,res2_1,res3_1,res4_1],axis=0)

# %%
import time
from tqdm import auto

bad = []
good = pd.DataFrame()

Simbad.TIMEOUT = 5

for i in auto.tqdm(unified_ids(elods)):
    try:
        red = Simbad.query_objects([i])
        good = pd.concat([good,red.to_pandas()])
    except:
        bad.append(i)
        pass

# %%

wut = Simbad.query_objects(bad)

elod_final = pd.concat([good,wut.to_pandas()])

# elod_final.to_csv('data\ELODIE Full\elodie_simbad_2.csv',index=False)
# good.to_csv('data/SIMBAD Queries/elod_simbad_2.csv')
# %%

elod_final = pd.read_csv('data/SIMBAD Queries/elod_simbad_2.csv')

elod_sp = elod_final.iloc[:,[-2,-3,-6,-5,-4]].query('~SP_TYPE.isnull()')
elod_sp = class_splitter(elod_sp.reset_index(drop=True), 'SP_TYPE')

# elod_sp = elod_sp.rename(columns={'TYPED_ID':'ID'})
# both = set(elod_sp.TYPED_ID)&set(soph_sp.TYPED_ID)
# elod_sp = elod_sp.query('ID not in @both')

# %%
# XSL

xsl_cata = pd.read_csv('data/XSL/xsl_labels.csv')
xsl_cata = xsl_cata.iloc[:,1:-1]
# xsl_cata = xsl_cata.iloc[:,np.roll(np.arange(0,9),1)]
xsl_cata.query('~SP_TYPE.isnull()',inplace=True)
# xsl_cata.rename({'OBSID':'ID'}, axis=1, inplace=True)

# %%
from itertools import permutations

catalogues = [cflib_sp, sdss_sp, elod_sp, soph_sp, xsl_cata]
cata_names = ['CFLIB','SDSS','ELODIE','SOPHIE','XSL']

commons = np.empty(25)

for i,var in enumerate(permutations(range(5), r=2)):
    if i == 0:
        j = 1
    elif i % 5 == 0:
        j+=1
    id_1 = catalogues[var[0]].TYPED_ID
    id_2 = catalogues[var[1]].TYPED_ID
    commons[j] = len(set(id_1) & set(id_2))
    j+=1

import seaborn as sns

sns.heatmap(
    commons.reshape(5,5), annot=True, fmt='.0f',
    xticklabels=cata_names, yticklabels=cata_names
)


# %% 
# merge everything!!

merge = pd.concat(catalogues, ignore_index=True)

#merge.loc[:,'SIMBAD_MAIN_ID'] = merge.SIMBAD_MAIN_ID.fillna(ids2)

datasets =  ['CFLIB']*len(cflib_sp) + \
            ['SDSS']*len(sdss_sp) + \
            ['ELODIE']*len(elod_sp) + \
            ['SOPHIE']*len(soph_sp) + \
            ['XSL']*len(xsl_cata)

merge['dataset'] = datasets

# %%

merge.to_csv('data/Catalogues/merged2.csv', index=False)

# %%

merged2 = pd.read_csv('data/Catalogues/merged2.csv')


xsl_cata

pd.concat([merged2, xsl_cata.query('~SP_TYPE.isnull()')],ignore_index=True)
#%%

merge.to_csv("merged2.csv",index=False)

# %%

test = pd.read_csv('merged.csv')

test.info()

# %%

merge['SP_QUAL'] = merge.SP_QUAL.replace('',np.NaN)

plt.title('Label Quality Distribution (E is least, A is most)');
merge.SP_QUAL.value_counts().sort_values().plot(kind='barh',color='#606060');

# %%
import missingno as msno
msno.bar(merge.iloc[:,4:-1])

# %%
merge['SP_QUAL'] = merge.SP_QUAL.replace({'B':1,'C':2,'D':3,'E':4})

# %%

merge.lum.isna()

# %%
# from sklearn.linear import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

main = list('OBAFGKM')
merge_obaf = merge.query('spec in @main')

# def 

# merge_obaf['sub'].apply(lambda x: )
# %%
missing = merge_obaf.iloc[:,[4,5,6,7,9]]
missing['spec'] = [main.index(x) for x in merge_obaf.spec]

missing['lum_na'] = missing.lum.isna()

train, test = train_test_split(merge,test_size=0.2,random_state=1337)



# %%

msno.bar(merge.query('SP_TYPE_QUAL != "E"').iloc[:,:-1])

# %%

msno.heatmap(merge.iloc[:,:-1])

# %%
merge.query('SP_TYPE_QUAL != "E"').lum.value_counts().plot(kind='barh')
# %%

lums = 'Ia|Ib|II|III|IV|V|VI'.split('|')


import seaborn as sns
sns.heatmap(pd.crosstab(merge_obaf.spec,merge_obaf.lum), 
            annot=True, fmt='.0f')


# %%
from statistics import mode
from collections import Counter

def percent_e(list):
    data = Counter(list)
    return data

pd.crosstab(merge_obaf.spec,merge_obaf.lum, merge_obaf.SP_QUAL, aggfunc=percent_e)
# %%

import os
import tqdm

list_ = os.listdir('data/CFLIB/IRAF/')

arr = np.ndarray((1273,), dtype=[('FLUX', '<f8', 15011), ('NAME', '<U10')])

#counter = 0
#for i, file in tqdm.tqdm(enumerate(list_)):
#    hdul = fits.open(f'CFLIB/IRAF/{file}')
#    data = hdul[0].data
#    if file[:-5] in :
#        arr[i-counter] = data, file[:-5]
#    else:
#        counter += 1

for i, file in tqdm.tqdm(enumerate(list_)):
    hdul = fits.open(f'data/CFLIB/IRAF/{file}')
    data = hdul[0].data
    arr[i] = data, file[:-5]

# np.save("cflib.npy", arr)





# %%
compare.drop(weirdc_ids, inplace=True)
ind_ = compare.sub_x.astype(float) > 9
compare.loc[ind_,'sub_x'] = compare[ind_].sub_x.apply(lambda x: x[0])

compare[ind_]
# %%
# ids = []

# only replace lum from picktype if spec is same, 
# and sub difference is at most 3

# for i,row in compare.iterrows():
    # if (row.spec_x == row.spec_y) & np.isclose(float(row.sub_x), float(row.sub_y), atol=3):
        # ids.append(i)

# ind = compare.loc[ids,:].lum_y.isna()
# horrible indexing. take the index of the series above where lum_y is na
# and at that index replace lum_y with lum_x
# compare.loc[ind[ind].index.values, 'lum_y'] = compare.loc[ind[ind].index.values, 'lum_x']

# %%
empd = ~compare.misc_y.isna()
empi = compare[empd].misc_y.str.match('EMP')
compare.drop(empi[empi].index, inplace=True)
# compare.misc_y.apply(lambda x: x if math.isnan(x) else re.match('EMP', x))
# %%
compare = compare.iloc[:,4:]
compare.index.name = 'ID'
compare.columns = ['spec','sub','lum','misc']
compare.index = compare.index.to_series().apply(str.strip)
# compare.index = compare.index.to_series().apply(lambda x: re.sub('(HD)(0+)?', '', x)).values
# compare.index = compare.index.to_series().apply(lambda x: re.sub('^(\d)', 'HD', x)).values
# %%
compare.to_csv('cflib.csv')

# %%
sp_sdss_c = sp_sdss.drop(weird_ids.index)
sp_sdss_c = sp_sdss_c.set_index('MANGAID').drop(['SP_TYPE'],axis=1,)
sp_sdss_c.index.name = 'ID'
sp_sdss_c
# %%
cols = list(sp_sdss_c.columns)
cols = cols[1:] + [cols[0]]
sp_sdss_c = sp_sdss_c[cols]

sp_sdss_c.to_csv('sdss.csv')




# %%


# cflib_old = class_splitter(cflib,'PICKTYPE')
# cflib_new = class_splitter(cflib,'SP_TYPE_SIMBAD')

# compare = cflib_old.merge(cflib_new,left_index=True,right_index=True)

# %% 
# reloads class_splitter WHAT SORCERY IS THIS
# import importlib
# import data_utils
# 
# importlib.reload(data_utils)
# 
# from data_utils import *


# lol comparing old labels from picktype, not very reliable

# sp_cflib = class_splitter(cflib, 'SPTYPE')
# spick_cflib = class_splitter(cflib, 'PICKTYPE')

# spick has weird sub, need to average the two digits
# spick_cflib[~spick_cflib['sub'].isna()][spick_cflib['sub'][~spick_cflib['sub'].isna()].astype(int) > 9]



# either sp_type from catalogue or simbad has to exist
# to make it into the final subset

# index = res.SP_TYPE.notna() | subset.SP_TYPE.notna()
# 
# subset.loc[:,'SP_TYPE_SIMBAD'] = res.SP_TYPE
# subset.loc[:,'SP_TYPE_QUAL'] = res.SP_QUAL
# 
# final = subset.iloc[:,[0,1,3,-1,-2]][index]

# import missingno as msno
# na = final.SP_TYPE.isna() | final.SP_TYPE_SIMBAD.isna()

# msno.matrix(final[na])

# %% merging labels

# cases where no modification is needed

# sp_noissue = final[final.SP_TYPE == final.SP_TYPE_SIMBAD].drop('SP_TYPE_SIMBAD',axis=1)

# cases where simbad labels complement original catalogue
# by filling in the gaps 

# sp_filled = final[na].fillna({'SP_TYPE':final[na]['SP_TYPE_SIMBAD']})
# sp_filled.drop('SP_TYPE_SIMBAD',axis=1,inplace=True)

# cases where simbad sp type exists for both 
# but does not match the catalogue
# for nearly all cases taking the simbad sp type yields 
# more amount of info (which is more recent)
# %%
# sp_problem = final[(final.SP_TYPE != final.SP_TYPE_SIMBAD) & (~na)]

# sp_split = class_splitter(sp_problem,'SP_TYPE')
# sps_split = class_splitter(sp_problem,'SP_TYPE_SIMBAD')
# 
# sp_na = sp_split.isna().sum(axis=1)
# sps_na = sps_split.isna().sum(axis=1)
# 
# sp_final = pd.Series(name='SP_TYPE', index=sp_problem.index, 
                    # dtype="object")
# 
# ids = []
# for idx, row in sp_problem.iterrows():
    # if sps_na[idx] <= sp_na[idx]:
        # sp_final[idx] = row.SP_TYPE_SIMBAD
    # else:
        # ids.append(idx)
        # sp_final[idx] = row.SP_TYPE
# 
# final.loc[ids]
# sp_problem = sp_problem.iloc[:,:2].merge(sp_final, 
                                # right_index=True, left_index=True)
# 
# merging and processing labels
# 
# sp_sdss = pd.concat([sp_noissue,sp_filled,sp_problem])
# 
# sp_sdss.sort_index(inplace=True)
# 
# sp_sdss = sp_sdss.merge(class_splitter(sp_sdss,'SP_TYPE'),
                        # right_index=True, left_index=True)
# 
# main = list('OBAFGKM')
# obaf = sp_sdss.query('spec in @main')
# 
# obaf