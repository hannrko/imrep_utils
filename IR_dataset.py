import os
import re
import json
import pandas as pd
import numpy as np
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


class IRDataset:
    # initialises immune repertoire dataset by getting paths to sample files and loading other necessary metadata
    # defines generator functions with series of steps to apply to repertoires
    # assembles dataset matrix by specifying generator function to execute in gen2matrix
    def __init__(self, ddir, lfunc, dfunc, nfunc=None, largs=(), nargs=(), rs=None):
        # ddir is the directory that stores repertoire files and nothing else
        # lfunc exracts labels from file names
        # dfunc extracts relevant information from repertoire files
        # nfunc is optional and replaces filenames with sample names
        # largs and nargs can supply additional variables to lfunc and nfunc respectively
        # rs sets random seed which can be used in downsampling
        self.ddir = ddir
        # get list of files to read
        # just use all files in ddir  and sort them in ascending numerical order
        # we expect filenames to start with an identifier which contains letter(s) and a number
        # this should be separated from the rest of the filename with - or _, or be the entire filename
        files_in_ddir = [d for d in os.listdir(ddir) if os.path.isfile(os.path.join(ddir, d))]
        # sort by letter part of identifier and numerical part, letter part may refer to label or sample or otherwise
        self.fnames = sorted(files_in_ddir, key=self.extr_sam_info)
        # assemble sample paths from files
        self.fpaths = [os.path.join(ddir, d) for d in self.fnames]
        self.nsam = len(self.fpaths)
        self.lfunc = lfunc
        self.dfunc = dfunc
        # we haven't calculated counts yet
        self.count_flag = False
        # set random seed if specified
        self.rs = rs
        if self.rs:
            random.seed(self.rs)
        self.dropped = []
        self.drpd_labs = {}
        self.drpd_counts = {}
        # finally get all labels
        self.labs = pd.Series(index=self.fnames, data=[self.lfunc(fn, *largs) for fn in self.fnames]).replace('nan', np.NaN)
        # if a name extractor is specified, get the names, otherwise remove file extensions
        if nfunc:
            self.snames = np.array([nfunc(fn, *nargs) for fn in self.fnames])
        else:
            self.snames = np.array([fn.split(".")[0] for fn in self.fnames])
        self.labs.index = self.snames
        # drop samples with undefined labels
        self.drop(self.labs[self.labs.isna()].index)
        self.prepro_func_dict = {"ds_kmers": self.ds_kmers, "raw_kmers": self.raw_kmers,
                                 "ds_clones": self.ds_clones, "raw_clones": self.raw_clones}
        self.pos = {0: None, 1: ["start", "middle", "end"]}


    @staticmethod
    def extr_sam_info(fn):
        # fn is a filename
        # get sample identifier
        snm = re.split('_|-|\\.', fn)[0]
        # extract numerical part
        num = int(re.findall(r'\d+', snm)[0])
        # extract letter part
        mlab = re.findall(r'[A-Za-z]+', snm)[0]
        return mlab, num

    def get_counts(self):
        # dict from generator that executes get_count method for all repertoires
        self.counts = dict(((name, ImmuneRepertoire(fp, name, self.dfunc).get_count())
                            for fp, name in zip(self.fpaths, self.snames)))
        self.count_flag = True
        return self.counts

    def drop(self, sam_names):
        # remove samples in list sam_names from dataset
        # get indices of samples to drop
        idx_del = [np.argwhere(self.snames == sn)[0] for sn in sam_names]
        # add removed samples to dropped samples list
        self.dropped.extend(sam_names)
        # then overwrite sample names, file names and file paths lists
        self.fnames = np.delete(self.fnames, idx_del)
        self.fpaths = np.delete(self.fpaths, idx_del)
        self.snames = np.delete(self.snames, idx_del)
        # save labels of dropped samples
        self.drpd_labs.update(self.labs[sam_names].to_dict())
        # overwrite labels
        self.labs = self.labs[self.snames]
        # if we've calculated counts maybe we should delete relevant entries?
        if self.count_flag:
            # but we should have a dropped counts attribute to make sure we understand why they were dropped
            self.drpd_counts.update(dict((sn, self.counts[sn]) for sn in sam_names))
            for sn in sam_names:
                self.counts.pop(sn)

    def prep_dwnsmpl(self, d=None):
        # prepare downsampling
        # use threshold to determine which samples should be dropped due to insufficient counts
        if not self.count_flag:
            self.get_counts()
        if d is None:
            # if no threshold defined, set it as the minimum counts
            d = np.amin(list(self.counts.values()))
        # samples less deep than threshold are dropped
        drp_ind = np.array(list(self.counts.values())) < d
        ds_drp = np.array(self.snames)[drp_ind]
        self.drop(ds_drp)
        return d

    # generator function to get downsampled kmers
    # means that the sequence of functions to downsample and kmerise are applied
    # to each repertoire individually
    def ds_kmers(self, k, p=0, d=None):
        # downsample sequences, convert to kmers
        # first prep for downsampling
        # first do with just minimum value but need to add option
        d = self.prep_dwnsmpl(d)
        # now get generator for ImmuneRepertoire objects with downsampling and kmerisation applied
        # requires generator function
        for fp, name in zip(self.fpaths, self.snames):
            ir = ImmuneRepertoire(fp, name, self.dfunc)
            ir.downsample(d)
            # use the positionality dict defined in class
            ir.kmerize(k, self.pos[p])
            yield ir.kmers
            
    def raw_kmers(self, k, p=0):
        # NOTE: dfunc must give a pandas series
        # now get generator for ImmuneRepertoire objects with kmerisation applied
        # requires generator function
        for fp, name in zip(self.fpaths, self.snames):
            ir = ImmuneRepertoire(fp, name, self.dfunc)
            ir.kmerize(k, self.pos[p])
            yield ir.kmers
    
    # generator function to downsample each repertoire
    # useful for compaing repertoires by their summaries
    def ds_clones(self, d=None):
        # downsample sequences, convert to kmers
        # first prep for downsampling
        # first do with just minimum value but need to add option
        d = self.prep_dwnsmpl(d)
        # now get generator for ImmuneRepertoire objects with downsampling and kmerisation applied
        # requires generator function
        for fp, name in zip(self.fpaths, self.snames):
            ir = ImmuneRepertoire(fp, name, self.dfunc)
            ir.downsample(d)
            yield ir.down

    # generator to extract raw clones as defined by extraction function
    # useful to implement quality control after detailed analysis
    def raw_clones(self):
        # don't preprocess repertoires
        # this is helpful for exploratory repertoire plotting (small datasets only)
        # get repertoires, extracting relevant information with dfunc
        for fp, name in zip(self.fpaths, self.snames):
            ir = ImmuneRepertoire(fp, name, self.dfunc)
            # needs to change to reflect sequences generally
            yield ir.seqtab

    def prepro(self, prepro_func_key, kwargs, export=True, json_dir=None, lab_spec=None):
        gen_func = self.prepro_func_dict[prepro_func_key]
        prepro = self.gen2matrix(gen_func, kwargs)
        if export:
            if lab_spec is None:
                lab_spec = ""
            else:
                lab_spec = lab_spec + "_"
            # do we also need naming funcs?
            kwargs_name = "_".join([key + str(val) for key, val in kwargs.items()])
            if self.rs is not None:
                kwargs_name = "rs" + str(self.rs) + "_" + kwargs_name
            prepro_name = f"{lab_spec}{prepro_func_key}_{kwargs_name}"
            # save matrix to location, store location
            prepro_dir = os.path.join(self.ddir, "preprocessed")
            if not os.path.isdir(prepro_dir):
                os.makedirs(prepro_dir)
            prepro_fname = prepro_name + ".csv"
            prepro_path = os.path.join(prepro_dir, prepro_fname)
            prepro.to_csv(prepro_path)
            json_fname = f"{prepro_name}.json"
            if json_dir is None:
                json_dir = ""
            json_path = os.path.join(json_dir, json_fname)
            self.json_export(json_path, prepro_path, prepro_fname, prepro_name)
        return prepro

    def gen2matrix(self, gf, kwargs):
        # produce matrix containing resulting data
        gen = gf(**kwargs)
        # assemble the matrix
        outmat = pd.concat(gen, axis=1)
        outmat.columns = self.snames
        prepro = outmat.fillna(0)
        return prepro


    def json_export(self, svpath, prepro_path, prepro_fname, prepro_name):
        # store all information we need about the preprocessing as a json file
        # make everything python-built-in types
        sv_dict = dict(labs=self.labs.to_dict(), prepro_path=prepro_path,
                       prepro_name=prepro_name, prepro_fname = prepro_fname,
                       dropped=self.dropped, dropped_labs=self.drpd_labs)
        if self.count_flag:
            sv_dict["raw_counts"] = self.counts
            sv_dict["dropped_counts"] = self.drpd_counts
        with open(svpath, 'w', encoding='utf-8') as f:
            json.dump(sv_dict, f, ensure_ascii=False, indent=4)

