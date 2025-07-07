import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import auto
from astroquery.simbad import Simbad

query = "SELECT id, mesSpT.*, comment, is_candidate, "\
        "rvz_type, rvz_radvel, rvz_redshift, "\
        "rvz_err, rvz_nature, rvz_qual, rvz_bibcode "\
        "FROM basic JOIN ident ON oid = ident.oidref "\
        "JOIN otypedef using(otype) "\
        "FULL OUTER JOIN mesSpT ON oid = mesSpT.oidref "\
        "WHERE id IN"

batch_size = 1
original = pd.read_csv('data/processed/final_cata2.csv')
catalogue = original.drop_duplicates(subset='TYPED_ID')
ids = catalogue.TYPED_ID.drop_duplicates().values
# batches = np.array_split(ids, len(ids)//batch_size)

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
        bad.append(i)

better.to_csv('simbad_tap_sptype.csv')
np.save('fails_tap.npy', np.array(bad))