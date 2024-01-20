import pandas as pd
import re

def cdr3_mat_extr(fpath):
    # assume no unproductive counts
    raw = pd.read_table(fpath)
    # get cdr3 matrix by summing counts
    cdr3_mat = raw.groupby("CDR3").aggregate({"Count": "sum"})
    return cdr3_mat.index, ["CDR3"], ["aaSeq"], cdr3_mat.values

# pass dict as arg
def lab_extr(name, lab_dict):
    # take name, extract first part, get label stored in dict
    neat_name = re.split('_|-',name)[0]
    lkey = ''.join([i for i in neat_name if not i.isdigit()]).strip()
    return lab_dict.get(lkey)
