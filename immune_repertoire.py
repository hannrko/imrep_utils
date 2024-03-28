import numpy as np
import random
from . import diversity

class ImmuneRepertoire:
    # defines a single immune repertoire sample and any preprocessing operations that can be applied as class functions
    def __init__(self, fpath, name, extract_func):
        # this has flexibility to extract any sequence attribute from raw data
        # as long as appropriate extract_func is specified
        self.seq_info, self.seq_info_names, self.seq_info_types, self.seq_counts = extract_func(fpath)
        self.name = name
        self.count_seqs()

    def count_seqs(self):
        self.n_seq = len(self.seq_info)
        self.count = int(sum(self.seq_counts))
        if self.count == 0:
            self.props = self.seq_counts
        else:
            self.props = self.seq_counts/self.count

    def downsample(self, d, overwrite=True):
        # sample sequences without replacement
        # indices of unique sequences
        scodes = list(range(self.count))
        # randomise them
        random.shuffle(scodes)
        # only take up to a threshold
        sam_scodes = scodes[:d]
        # get bin boundaries
        sbins = np.concatenate(([0], np.cumsum(self.seq_counts)))
        # right false as default gives correct behaviour
        # which bin do each of our samples fall into?
        # bins labels start from 1 because we defined a lower bin limit that is
        # smaller or equal to all possible scodes
        sinds = np.digitize(sam_scodes, sbins)
        # count up sampled bin categories
        sindsu, down_seq_counts = np.unique(sinds, return_counts=True)
        down_seq_inds = sindsu - 1
        # make a new table of downsampled sequence by accessing the original sequences
        # update all class variables
        down_seq_info = self.seq_info[down_seq_inds]
        # optionally overwrite cdr3 matrix
        if overwrite:
            self.seq_info = down_seq_info
            self.seq_counts = down_seq_counts
            self.count_seqs()
        return down_seq_info, down_seq_counts

    def calc_div(self, div_names, q_vals=None):
        if isinstance(div_names, str):
            div_names = [div_names]
        div = {}
        for dvn in div_names:
            if dvn.lower() == "hill":
                div.update(self._calc_hill_div(dvn, q_vals))
            else:
                div.update(self._calc_single_div(dvn))
        return div

    def _calc_div(self, div_name, div_func_args=None):
        return diversity.diversity(div_name, self.props, args=div_func_args)

    def _calc_single_div(self, div_name):
        div = self._calc_div(div_name)
        return dict([(div_name, div)])

    def _calc_hill_div(self, div_name, q_vals):
        div = self._calc_div(div_name, (q_vals,))
        long_div_names = [div_name + str(q) for q in q_vals]
        return dict(zip(long_div_names, div))

    def get_as_pandas(self):
        import pandas as pd
        # construct index or multiindex
        # can't use automatic multiindex as sample may be empty
        if len(self.seq_info_names) == 1:
            seq_info_t = pd.Index(self.seq_info, name=self.seq_info_names[0])
        else:
            seq_info_t = pd.MultiIndex.from_tuples(self.seq_info, names=self.seq_info_names)
        return pd.Series(index=seq_info_t, data=self.seq_counts)

    def calc_vdj_usage(self, vdj_names):
        seq_tab = self.get_as_pandas()
        # then calculate clones with unique combination of segments
        return seq_tab.groupby(by=vdj_names).sum()
