import numpy as np
import aa_seq_utils as aasu
import utils

class KmerRepertoire:
    def __init__(self, k, seqs, counts=None, trim=None, kmer_kwargs=None):
        # no need to count duplicates
        # k should be an integer
        # counts assumed 1 if not specified
        # trim should be an integer if specified
        # kmer_kwargs are extra values supplied to kmer function
        # including: ...
        self.k = k
        if counts is None:
            counts = np.ones(len(seqs))
        self.seq_rep = [aasu.ImSeq(s, c) for s, c in zip(seqs, counts)]
        if trim is not None:
            self.seq_rep = self.trim(trim)
        self.kmers = self.get_kmers(kmer_kwargs)

    def trim(self, nr):
        # trim nr (number of residues) from each end of sequence
        return [seq.trim(nr) for seq in self.seq_rep]

    def get_kmers(self, kmer_kwargs=None):
        if kmer_kwargs is None:
            kmer_kwargs = {}
        kmer_rep = {}
        # leave option open to get multiple kmer lengths?
        k = self.k
        # generator to avoid two for loops
        kd_gen = (seq.to_kmers(k, **kmer_kwargs) for seq in self.seq_rep)
        kmer_rep = utils.dicts_to_dict(kd_gen)
        return kmer_rep

    def get_as_pandas(self):
        import pandas as pd
        return pd.Series(self.kmers)