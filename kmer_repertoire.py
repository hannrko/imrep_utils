import numpy as np
import aa_seq_utils as aasu

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
        # how to add these dicts together?
        for seq in self.seq_rep:
            kd = seq.to_kmers(k, **kmer_kwargs)
            for key, val in kd.items():
                if key in kmer_rep.keys():
                    kmer_rep[key] = kmer_rep[key] + val
                else:
                    kmer_rep[key] = val
        return kmer_rep
