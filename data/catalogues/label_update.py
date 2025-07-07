# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import auto
from astroquery.simbad import Simbad

from data_utils import preprocess, batch_processor, class_splitter
# %%

final_cata = pd.read_csv('data/processed/final_cata2.csv')
final_cata_historic = pd.read_csv('data/Catalogues/simbad_tap_sptype.csv')

Simbad.add_votable_fields('sptype', 'ids', 'typed_id')
Simbad.TIMEOUT = 5
# %%

batch_size = 128
ids = final_cata.TYPED_ID.drop_duplicates().values
batches = np.array_split(ids, len(ids)//batch_size)

better = pd.DataFrame()
bad = []

for batch in auto.tqdm(batches):
    try:
        red = Simbad.query_objects(batch)
        better = pd.concat([better,red.to_pandas()])
        Simbad.clear_cache()
    except:
        print('Failed to get anything...')
        pass
        bad.append(batch)
# %%
better = better.replace({'':np.nan})

# %%
import re

fix_cflib_ids = lambda x: f'HD{x}' if re.match('^\d',x) else x
cflib_ids = final_cata.query('dataset == 0').ID.apply(fix_cflib_ids)
final_cata.loc[cflib_ids.index, 'ID'] = cflib_ids

# %%
hey = final_cata.reset_index().merge(better.iloc[:,range(11,16)], on='TYPED_ID', how='left')
x = hey.loc[:,'index'].value_counts()
xx = x[x != 1]
xx.index
# %%
label_update = hey.query('index not in @xx.index')

# %%

# label_update.query('SP_TYPE_x != SP_TYPE_y')
label_update.query('~SP_TYPE_y.isnull() & SP_TYPE_x != SP_TYPE_y',engine='python')
# %%
# inter = better.TYPED_ID.value_counts()
# multiple = inter[inter != 1]

# better.query('TYPED_ID in @multiple.index')

# %%
better.to_csv('data/Catalogues/updated_labels.csv',index=False)
# %%
