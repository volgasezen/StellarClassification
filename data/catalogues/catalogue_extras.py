# %%
import os
os.chdir('/home/oban/Desktop/Volga/stellar-classification')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from astropy.io import fits
from astroquery.simbad import Simbad
import re

from data_utils import *

# %%
# using builtin functions for retrieval

Simbad.add_votable_fields('typed_id', 'otype(V)', 'otype(S)', 'otypes',
                          'fluxdata(U)', 'fluxdata(B)', 'fluxdata(V)', 
                          'fluxdata(R)', 'fluxdata(I)', 'fluxdata(G)', 
                          'fluxdata(J)', 'fluxdata(H)', 'fluxdata(K)', 
                          'fluxdata(u)', 'fluxdata(g)', 'fluxdata(r)', 
                          'fluxdata(i)', 'fluxdata(z)')

batch_size = 16
catalogue = pd.read_csv('data/processed/final_cata3.csv')
ids = catalogue.TYPED_ID.drop_duplicates().values
batches = np.array_split(ids, len(ids)//batch_size)

from tqdm import auto

good = pd.DataFrame()
bad = []

Simbad.TIMEOUT = 5

for batch in auto.tqdm(batches):
    try:
        red = Simbad.query_objects(batch)
        good = pd.concat([good,red.to_pandas()])
        Simbad.clear_cache()
    except:
        pass
        print('Previous batch failed.')
        bad.append(batch)
# %%
hmm = pd.DataFrame()

if len(bad) != 0:
    Simbad.TIMEOUT = 15

    for batch in auto.tqdm(bad):
        red = Simbad.query_objects(batch)
        hmm = pd.concat([hmm,red.to_pandas()])
        Simbad.clear_cache()
# %%
# final_extras = pd.concat([good,hmm])
final_extras = good
final_extras.reset_index(drop=True, inplace=True)
# final_extras.query('TYPED_ID == "BD+37  4734S"').index

# %%


# final_extras.drop(final_extras.query('TYPED_ID == "BD+37  4734S"').index, inplace=True)
# final_extras = pd.concat([final_extras,Simbad.query_object('BD+37 4734B').to_pandas()])
# %%
extras = final_extras.drop_duplicates('TYPED_ID').set_index('TYPED_ID').loc[catalogue.TYPED_ID]
extras.reset_index(inplace=True)
# extras.loc[:,'TYPED_ID'] = extras.TYPED_ID.replace({'BD+37 4734B':'BD+374734S'})
# %%
import missingno as msno

msno.matrix(extras.replace('',np.nan))
# %%
extras.to_csv('data/processed/final_cata3_extras.csv', index=False)
# %%
msno.bar(extras.replace('',np.nan))
# %%
# using TAP service via custom query

# query = "SELECT id, sptype, bibcode, otype, label, description, "\
#         "comment, is_candidate, rvz_type, rvz_radvel, rvz_redshift, "\
#         "rvz_err, rvz_nature, rvz_qual, rvz_bibcode "\
#         "FROM basic JOIN ident ON oid = ident.oidref "\
#         "JOIN otypedef using otype "\
#         "FULL OUTER JOIN mesSpT ON oid = mesSpT.oidref "\
#         "JOIN alltypes ON oid = alltypes.oidref WHERE id IN"

query = "SELECT id, mesSpT.*, comment, is_candidate, "\
        "rvz_type, rvz_radvel, rvz_redshift, "\
        "rvz_err, rvz_nature, rvz_qual, rvz_bibcode "\
        "FROM basic JOIN ident ON oid = ident.oidref "\
        "JOIN otypedef using(otype) "\
        "FULL OUTER JOIN mesSpT ON oid = mesSpT.oidref "\
        "WHERE id IN"

batch_size = 1
original = pd.read_csv('data/processed/final_cata3.csv')
catalogue = original.drop_duplicates(subset='TYPED_ID')
ids = catalogue.TYPED_ID.drop_duplicates().values
# batches = np.array_split(ids, len(ids)//batch_size)

from tqdm import auto

better = pd.DataFrame()
bad = []

Simbad.ROW_LIMIT = -1
Simbad.TIMEOUT = 5

for i in auto.tqdm(ids):
    try:
        red = Simbad.query_tap(f"{query} ('{i}')")
        res = red.to_pandas()
        ix = [i]*len(res)
        res.loc[:, 'TYPED_ID'] = ix
        better = pd.concat([better,res])
        Simbad.clear_cache()
    except:
        pass
        bad.append(batch)

# %%

# weird_ids = catalogue.IDS.value_counts().reset_index().query('count != 1').IDS
# weird_id = set(catalogue.TYPED_ID) - set(original.drop_duplicates(subset='IDS').TYPED_ID)

# hmm = catalogue.query('IDS in @weird_ids')
hey = catalogue.merge(catalogue.drop_duplicates('IDS').loc[:, ['TYPED_ID', 'IDS']], on='IDS')
hey.query('TYPED_ID_x != TYPED_ID_y')


# %%

from polyleven import levenshtein

typed_id = hey.TYPED_ID_y.drop_duplicates().values
merged_ids = '|'.join(hey.IDS.drop_duplicates()).split('|')
better_ids = better.id.drop_duplicates().values

lengths = np.cumsum(hey.IDS.drop_duplicates().str.split('|').apply(len))

matching_ids = []

for ix in auto.tqdm(better_ids):
    loc = np.argmin([levenshtein(ix, j, 10) for j in merged_ids])
    conv_loc = np.where(np.diff(loc > lengths))[0] + 1
    if conv_loc.size == 0:
        conv_loc = 0
    matching_ids.append(typed_id[conv_loc][0])

# %%
# matching_ids[np.where(np.array([len(x) for x in matching_ids]) == 0)[0][0]] = typed_id[0]

# len(np.hstack(matching_ids))

# len(set(matching_ids) - set(ids))

# %%

pd.DataFrame(better_ids, matching_ids)


# %%

huh = []
cata = catalogue.drop_duplicates(subset='TYPED_ID')

for _, row in auto.tqdm(cata.iterrows()):
    i = row['TYPED_ID']
    i_s = row['IDS'].split('|')
    extra_i = better.id.drop_duplicates().values
    huh.append([np.where(j == extra_i)[0] for j in i_s if j in extra_i])
    huh.append(np.argmin([levenshtein(x, y, 3) for y in better_ids]))
huh
# %%
better.groupby('id')['id'].value_counts().loc[better.id]

# %%

better.to_csv('data/processed/final_cata_extras2_needsfix.csv')
# %%
