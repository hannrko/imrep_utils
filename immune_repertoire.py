import numpy as np
import pandas as pd
import random

class ImmuneRepertoire:
    # defines a single immune repertoire sample and any preprocessing operations that can be applied as class functions
    def __init__(self, fpath, name, extract_func):
        # this has flexibility to extract any sequence attribute from raw data
        # as long as appropriate extract_func is specified
        self.seqtab = extract_func(fpath)
        self.name = name
        self.nseq = len(self.seqtab)

    def get_count(self):
        # get total sequences in repertoire
        self.count = int(self.seqtab.sum())
        return self.count

    def get_proportions(self):
        # get proportions of sequences that make up repertoire
        self.props = self.seqtab/self.count
        return self.props

    def downsample(self, d, overwrite=True):
        # sample sequences without replacement
        # indices of unique sequences
        scodes = list(range(int(self.seqtab.sum())))
        # randomise them
        random.shuffle(scodes)
        # only take up to a threshold (is this correct?)
        sam_scodes = scodes[:d]
        # get bin boundaries
        sbins = np.concatenate(([0], np.cumsum(self.seqtab.values)))
        # right false as default gives correct behaviour
        # which bin do each of our samples fall into?
        # bins labels start from 1 because we defined a lower bin limit that is
        # smaller or equal to all possible scodes
        sinds = np.digitize(sam_scodes, sbins)
        # count up sampled bin categories
        sindsu, dcounts = np.unique(sinds, return_counts=True)
        # make a new table of downsampled sequence by accessing the original sequences
        self.down = pd.Series(index=self.seqtab.index[sindsu-1], data=dcounts)
        # optionally overwrite cdr3 matrix
        if overwrite:
            self.seqtab = self.down
            # overwrite count too
            self.get_count()
        return self.down

    # kmer function
    def kmerize(self, k, p):
        # split sequences into short overlapping segments
        # only appropriate where we specify extract_func that gives a single amino acid or nucleotide sequence index
        all_kmers = {}
        short_kmers = []
        for s in self.seqtab.index:
            c = self.seqtab.loc[s]
            # kmerise each sequence
            kmers = self.split_into_kmers(s, c, k, p)
            if kmers is None:
                short_kmers.extend(s)
            else:
                # iterate over resulting kmers
                for kmer, count in kmers.items():
                    # if a kmer is already in the dictonary
                    if kmer in all_kmers:
                        # add counts to the existing entry
                        all_kmers[kmer] = all_kmers[kmer] + count
                    # if entry doesn't exist yet, create it
                    else:
                        all_kmers[kmer] = count
        self.kmers = pd.Series(all_kmers)
        self.short_kmers = short_kmers
        self.kmer_k = k
        if p is not None:
            self.positions = p


    @staticmethod
    def get_aa_pos_fracs(l_seq, n_pos):
        # find position of a letter in a sequence
        # initialise aa_pos
        aa_pos = np.zeros((0, l_seq))
        prev = np.zeros(l_seq)
        fracs = np.linspace(0, 1, l_seq+1)
        for i in np.arange(n_pos)+1:
            pointer = np.array([(fracs - i * 1/n_pos).clip(min=0)[j:j+2] for j in range(l_seq)])
            # indices that pointer has passed
            ind_sec = pointer == 0
            # if start or end fractions of each index are passed, add entry
            res = np.where(np.logical_or(ind_sec[:, 0], ind_sec[:, 1]), 1 - pointer.sum(axis=1)*l_seq, 0) - prev
            aa_pos = np.append(aa_pos, res.reshape(1, -1), axis=0)
            prev = prev + res
        return aa_pos

    def split_into_kmers(self, seq, counts, k, positional=None, position_cond_coeff=0):
        # prepare empty dictionary which will hold kmers as keys, kmer counts as values
        kmer_dict = {}
        # if the sequences is shorter than or equal to desired kmer length, return None
        if len(seq) < k:
            return None
        # the number of k-mers we'll end up with
        n_kmers = len(seq)-k + 1
        # split into kmers with moving window
        kmers = [seq[i:i+k] for i in range(n_kmers)]
        # if no positional labels supplied, just return kmers
        if not positional:
            kmer_dict.update(zip(kmers, np.tile(counts, n_kmers)))
            return kmer_dict
        # but if we do have positional labels
        else:
            # positional annotation condition: check if cdr3 is long enough to justify splitting into kmers
            if len(seq) <= position_cond_coeff*k:
                return None
            else:
                n_sec = len(positional)
                # calculate how far along CDR3 each AA is
                aa_pos = self.get_aa_pos_fracs(len(seq), n_sec)
                # for each kmer, take positional fractions of each AA
                kmer_pos_by_aa = np.array([aa_pos[:, i:i+k] for i in range(n_kmers)])
                # count up these fractions to determine fraction of kmer as whole
                kmer_counts = counts*(np.sum(kmer_pos_by_aa, axis=2)/k).flatten()
                # round kmer counts so that all exact halves are preserved
                rounded_kmer_counts = np.round(2*kmer_counts)/2
                # for each kmer, position, and corresponding kmer counts
                for kmer, p, c in zip(np.repeat(kmers, n_sec), np.tile(positional, n_kmers), rounded_kmer_counts):
                    # if count isn't zero for this positional kmer
                    if c.any():
                        # make new kmer name
                        kmer_name = kmer + p
                        # add kmer to dictionary
                        # in case positional labels are duplicates,
                        # check if entries already exist
                        if kmer_name in kmer_dict:
                            kmer_dict[kmer_name] += c
                        else:
                            kmer_dict[kmer_name] = c
            return kmer_dict