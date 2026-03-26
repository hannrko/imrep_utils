import pandas as pd
import re

def cdr3_mat_extr(fpath):
    # assume no unproductive counts
    raw = pd.read_table(fpath)
    # get cdr3 matrix by summing counts
    cdr3_mat = raw.groupby("CDR3").aggregate({"Count": "sum"})
    return cdr3_mat.index, ["CDR3"], ["aaSeq"], cdr3_mat.values

# pass dict as arg
def lab_extr(names, lab_dict):
    # take name, extract first part, get label stored in dict
    neat_names = [re.split('_|-',name)[0] for name in names]
    lkey_func = lambda neat_name: ''.join([i for i in neat_name if not i.isdigit()]).strip()
    labs = [lab_dict.get(lkey_func(nn)) for nn in neat_names]
    return pd.Series(index=names, data=labs).to_frame(name="MyLabel")
