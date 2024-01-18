import numpy as np

class AAImSeq:
    def __init__(self, seq, count=1):
        self.aa_seq = seq
        self.count = count
        self.l_seq = len(self.aa_seq)

    def update_l_seq(self):
        self.l_seq = len(self.aa_seq)

    def trim(self, t):
        self.aa_seq = self.aa_seq[t:(len(self.aa_seq) - t)]
        self.update_l_seq()

    def kmerize(self, k):
        n_kmers = len(self.aa_seq)-k + 1
        kmers = [self.aa_seq[i:i + k] for i in range(n_kmers)]
        return kmers

    def get_aa_pos_fracs(self, l_seq, n_pos):
        # find position of a letter in a sequence
        # initialise aa_pos
        aa_pos = np.zeros((0, l_seq))
        prev = np.zeros(l_seq)
        fracs = np.linspace(0, 1, l_seq + 1)
        for i in np.arange(n_pos) + 1:
            pointer = np.array([(fracs - i * 1 / n_pos).clip(min=0)[j:j + 2] for j in range(l_seq)])
            # indices that pointer has passed
            ind_sec = pointer == 0
            # if start or end fractions of each index are passed, add entry
            res = np.where(np.logical_or(ind_sec[:, 0], ind_sec[:, 1]), 1 - pointer.sum(axis=1) * l_seq, 0) - prev
            aa_pos = np.append(aa_pos, res.reshape(1, -1), axis=0)
            prev = prev + res
        return aa_pos

    def get_kmer_pos_alloc(self, kmers, k, p=None):
        if type(p) is int:
            # p must be specified as int or list
            # could add default start, middle end as p=True
            pos = np.arange(p).astype(str)
            n_pos = p
        else:
            pos = p
            n_pos = len(p)
        aa_pos = self.get_aa_pos_fracs(self.l_seq, n_pos)
        # for each kmer, take positional fractions of each AA
        kmer_pos_by_aa = np.array([aa_pos[:, i:i+k] for i in range(len(kmers))])
        kmer_pos = np.sum(kmer_pos_by_aa, axis=2)/k
        return pos, kmer_pos

    # merge these two functions
    def foersetal_kmer_position_rounding(self, x):
        # halves are preserved
        return np.round(2*x)/2

    def count_informed_kmer_position(self, kmers, k, p, round_func=None):
        # ROUNDING
        if round_func is None:
            # if we don't want to round, use function that just passes value
            round_func = lambda x: x
        # determine counts of kmers in each position, round to integer
        # returns kmers and counts
        pos, kmer_pos = self.get_kmer_pos_alloc(kmers, k, p)
        kmer_count_grid = round_func(self.count*kmer_pos)
        obs_pkmer_counts = np.argwhere(kmer_count_grid > 0)
        # use indices to form kmers and assign counts
        # coming back with string counts...
        p_kmers = [kmers[i] + pos[j] for i, j in obs_pkmer_counts]
        p_kmer_counts = kmer_count_grid[*obs_pkmer_counts.T]
        return p_kmers, p_kmer_counts

    def simple_kmer_position(self, kmers, k, p):
        # determine single position label for each kmer instance
        # returns only a list of positional kmers
        pos, kmer_pos = self.get_kmer_pos_alloc(kmers, k, p)
        kmer_pos_ind = np.argmax(kmer_pos, axis=1)
        p_kmers = [kmer + pos[kmer_pos_ind[i]] for i, kmer in enumerate(kmers)]
        return p_kmers

    def to_kmers(self, k, p=None, p_use_counts=False, p_counts_round_func=None, ignore_dup_kmers=False):
        # kmerise
        kmers = self.kmerize(k)
        counts_flag = False
        # if positional, use positional method
        if p is not None:
            if p_use_counts:
                kmers, kmer_counts = self.count_informed_kmer_position(kmers, k, p, round_func=p_counts_round_func)
                counts_flag = True
            else:
                kmers = self.simple_kmer_position(kmers, k, p)
        if ignore_dup_kmers:
            kmers = np.unique(kmers)
        if not counts_flag:
            # if we haven't already calculated counts, do it now
            kmer_counts = np.repeat(self.count, len(kmers))
        # then use dict addition
        kmer_dict = {}
        # dict addition should be imported from somewhere else
        for i in range(len(kmers)):
            km = kmers[i]
            c = kmer_counts[i]
            if km in kmer_dict.keys():
                kmer_dict[km] = kmer_dict[km] + c
            else:
                kmer_dict[km] = c
        return kmer_dict
