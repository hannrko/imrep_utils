import os
import re
import json
import pandas as pd
import numpy as np
import random
from . import immune_repertoire as imrep
from . import kmer_repertoire as krep

def get_files(dir):
    return [os.path.join(dir, d) for d in os.listdir(dir) if os.path.isfile(os.path.join(dir, d))]

def extr_sam_info(fp):
    # fp is a filepath
    fn = os.path.split(fp)[1]
    # get sample identifier
    snm = re.split('_|-|\\.', fn)[0]
    # extract numerical part
    num = int(re.findall(r'\d+', snm)[0])
    # extract letter part
    mlab = re.findall(r'[A-Za-z]+', snm)[0]
    return mlab, num

def remove_file_ext(fn):
    return fn.split(".")[0]


class IRDataset:
    # initialises immune repertoire dataset by getting paths to sample files and loading other necessary metadata
    # defines generator functions with series of steps to apply to repertoires
    # assembles dataset matrix by specifying generator function to execute in gen2matrix
    def __init__(self, ddir, lfunc, dfunc, nfunc=None, largs=(), nargs=(), primary_lab=None, rs=None):
        # ddir is the directory that stores repertoire files and nothing else
        # lfunc exracts labels from file names
        # dfunc extracts relevant information from repertoire files
        # nfunc is optional and replaces filenames with sample names
        # largs and nargs can supply additional variables to lfunc and nfunc respectively
        # rs sets random seed which can be used in downsampling
        if isinstance(ddir, list):
            self.ds_paths = np.concatenate(list(map(get_files, ddir)))
            self.ddir = ddir[0]
        else:
            self.ds_paths = get_files(ddir)
            self.ddir = ddir
        # get list of files to read
        # just use all files in ddir  and sort them in ascending numerical order
        # we expect filenames to start with an identifier which contains letter(s) and a number
        # this should be separated from the rest of the filename with - or _, or be the entire filename
        # sort by letter part of identifier and numerical part, letter part may refer to label or sample or otherwise
        self.ds_paths = sorted(self.ds_paths, key=extr_sam_info)
        # assemble sample paths from files
        self.fnames = [os.path.split(dsp)[1] for dsp in self.ds_paths]
        self.nsam = len(self.ds_paths)
        self.lfunc = lfunc
        self.dfunc = dfunc
        # we haven't calculated counts yet
        self.counts = None
        self.count_flag = False
        # set random seed if specified
        self.rs = rs
        if self.rs:
            random.seed(self.rs)
        self.log = {}
        self._init_log()
        # finally get all labels
        self.labs = self.make_labels(lfunc, largs)
        # if a name extractor is specified, get the names, otherwise remove file extensions
        if nfunc is None:
            nfunc = remove_file_ext
        self.snames = self.make_names(nfunc, nargs)
        self.labs["Names"] = self.snames
        # drop samples with undefined labels
        self.primary_lab = primary_lab
        if self.primary_lab is not None:
            sn_missing = self.labs["Names"].values[self.labs[self.primary_lab].isna()]
            self.drop(sn_missing)
        self.prepro_func_dict = {"ds_kmers": self.ds_kmers, "raw_kmers": self.raw_kmers,
                                 "ds_clones": self.ds_clones, "raw_clones": self.raw_clones,
                                 "ds_diversity": self.ds_diversity, "ds_vdj": self.ds_vdj}

    def _init_log(self):
        # initialise empty data structures in case we drop multiple times
        self.log["dropped"] = []
        self.log["dropped_plabs"] = {}
        self.log["dropped_counts"] = {}

    def make_labels(self, lfunc, largs=None):
        if largs is None:
            largs = ()
        labels = lfunc(self.fnames, *largs)
        return labels

    def make_names(self, nfunc, nargs=None):
        if nargs is None:
            nargs = ()
        snames = np.array([nfunc(fn, *nargs) for fn in self.fnames])
        return snames

    def get_counts(self):
        # dict from generator that executes get_count method for all repertoires
        self.counts = dict(((name, imrep.ImmuneRepertoire(fp, name, self.dfunc).count)
                            for fp, name in zip(self.ds_paths, self.snames)))
        self.count_flag = True
        return self.counts

    def drop(self, sam_names):
        # remove samples in list sam_names from dataset
        # get indices of samples to drop
        idx_del = [np.argwhere(self.snames == sn)[0] for sn in sam_names]
        # add removed samples to dropped samples list
        self.log["dropped"].extend(sam_names)
        # then overwrite sample names, file names and file paths lists
        self.fnames = np.delete(self.fnames, idx_del)
        self.ds_paths = np.delete(self.ds_paths, idx_del)
        self.snames = np.delete(self.snames, idx_del)
        # save labels of dropped samples
        if self.primary_lab is not None:
            pln_to_drop = self.labs.index[[ln in sam_names for ln in self.labs["Names"]]]
            self.log["dropped_plabs"].update(self.labs[self.primary_lab].loc[pln_to_drop].to_dict())
        # overwrite labels
        # do we definitely want to do this? means saving labels every time...
        self.labs = self.labs[[ln in self.snames for ln in self.labs["Names"]]]
        # if we've calculated counts we should delete relevant entries
        if self.count_flag:
            # we should also have a dropped counts attribute to make sure we understand why they were dropped
            self.log["dropped_counts"].update(dict((sn, self.counts[sn]) for sn in sam_names))
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
    def ds_kmers(self, k, d=None, trim=None, p=None, p_use_counts=False, p_counts_round_func=None,
                 ignore_dup_kmers=False):
        # downsample sequences, convert to kmers
        # set up kmer_kwargs
        # first prep for downsampling
        # first do with just minimum value but need to add option
        d = self.prep_dwnsmpl(d)
        # now get generator for ImmuneRepertoire objects with downsampling and kmerisation applied
        # requires generator function
        for fp, name in zip(self.ds_paths, self.snames):
            ir = imrep.ImmuneRepertoire(fp, name, self.dfunc)
            ir.downsample(d)
            # use the positionality dict defined in class
            # need to select cdr3
            kr = krep.KmerRepertoire(k, seqs=ir.seq_info, counts=ir.seq_counts, trim=trim, p=p, p_use_counts=p_use_counts,
                                     p_counts_round_func=p_counts_round_func, ignore_dup_kmers=ignore_dup_kmers)
            kmers = kr.get_as_pandas()
            yield kmers
            
    def raw_kmers(self, k, trim=None, p=None, p_use_counts=False, p_counts_round_func=None,
                 ignore_dup_kmers=False):
        # NOTE: dfunc must give a pandas series
        # now get generator for ImmuneRepertoire objects with kmerisation applied
        # requires generator function
        for fp, name in zip(self.ds_paths, self.snames):
            ir = imrep.ImmuneRepertoire(fp, name, self.dfunc)
            #kmer_kwargs = {"p": self.pos[p]}
            kr = krep.KmerRepertoire(k, seqs=ir.seq_info, counts=ir.seq_counts, trim=trim, p=p, p_use_counts=p_use_counts,
                                     p_counts_round_func=p_counts_round_func, ignore_dup_kmers=ignore_dup_kmers)
            kmers = kr.get_as_pandas()
            yield kmers
    
    # generator function to downsample each repertoire
    # useful for compaing repertoires by their summaries
    def ds_clones(self, d=None):
        # downsample sequences, convert to kmers
        # first prep for downsampling
        # first do with just minimum value but need to add option
        d = self.prep_dwnsmpl(d)
        # now get generator for ImmuneRepertoire objects with downsampling and kmerisation applied
        # requires generator function
        for fp, name in zip(self.ds_paths, self.snames):
            ir = imrep.ImmuneRepertoire(fp, name, self.dfunc)
            ir.downsample(d)
            clones = ir.get_as_pandas()
            yield clones

    # generator to extract raw clones as defined by extraction function
    # useful to implement quality control after detailed analysis
    def raw_clones(self):
        # don't preprocess repertoires
        # this is helpful for exploratory repertoire plotting (small datasets only)
        # get repertoires, extracting relevant information with dfunc
        for fp, name in zip(self.ds_paths, self.snames):
            ir = imrep.ImmuneRepertoire(fp, name, self.dfunc)
            clones = ir.get_as_pandas()
            # needs to change to reflect sequences generally
            yield clones

    def ds_diversity(self, d=None, div_names=["richness", "shannon", "simpson"], q_vals=None):
        # downsample sequences, convert to kmers
        # first prep for downsampling
        # first do with just minimum value but need to add option
        d = self.prep_dwnsmpl(d)
        # now get generator for ImmuneRepertoire objects with downsampling and kmerisation applied
        # requires generator function
        for fp, name in zip(self.ds_paths, self.snames):
            ir = imrep.ImmuneRepertoire(fp, name, self.dfunc)
            ir.downsample(d)
            div = ir.calc_div(div_names, q_vals=q_vals)
            yield pd.Series(div)

    def ds_vdj(self, seg_names, d=None):
        # downsample sequences, convert to kmers
        # first prep for downsampling
        # first do with just minimum value but need to add option
        d = self.prep_dwnsmpl(d)
        # now get generator for ImmuneRepertoire objects with downsampling and kmerisation applied
        # requires generator function
        for fp, name in zip(self.ds_paths, self.snames):
            ir = imrep.ImmuneRepertoire(fp, name, self.dfunc)
            ir.downsample(d)
            # if multiple gene segment types
            # then calculate clones with unique combination
            vdj = ir.calc_vdj_usage(seg_names)
            yield vdj

    def prepro(self, prepro_func_key, kwargs, export=True, json_dir=None, ds_name=None, del_path=None):
        gen_func = self.prepro_func_dict[prepro_func_key]
        self.log["prepro_func"] = prepro_func_key
        # kwargs can be logged by generator function and extracted later
        prepro = self.gen2matrix(gen_func, kwargs)
        if export:
            # do we also need naming funcs?
            # do this within the preprocessing functions
            kwargs_name = "_".join([key + str(val) for key, val in kwargs.items()])
            if self.rs is not None:
                kwargs_name = "rs" + str(self.rs) + "_" + kwargs_name
            if ds_name is None:
                ds_name = os.path.split(self.ddir)[1]
            prepro_name = f"{ds_name}_{prepro_func_key}_{kwargs_name}"
            # save matrix to location, store location
            ddir_path, ddir_name = os.path.split(self.ddir)
            prepro_dir = os.path.join(ddir_path, ddir_name + "_preprocessed")
            if not os.path.isdir(prepro_dir):
                os.makedirs(prepro_dir)
            prepro_fname = prepro_name + ".csv"
            prepro_path = os.path.join(prepro_dir, prepro_fname)
            prepro.to_csv(prepro_path)
            self.log["prepro_path"] = self.del_path(prepro_path, del_path)
            self.log["prepro_name"] = prepro_name
            self.log["prepro_fname"] = prepro_fname
            # labels
            lab_dir = os.path.join(ddir_path, "metadata")
            if not os.path.isdir(lab_dir):
                os.makedirs(lab_dir)
            pl = "" if self.primary_lab is None else "_" + self.primary_lab
            lab_name = ds_name + pl + "_labels.csv"
            lab_path = os.path.join(lab_dir, lab_name)
            self.labs.to_csv(lab_path)
            self.log["lab_path"] = self.del_path(lab_path, del_path)
            self.log["lab_name"] = lab_name
            # log
            json_fname = f"{prepro_name}.json"
            if json_dir is None:
                json_dir = os.path.join(ddir_path, ddir_name + "_logs")
            json_path = os.path.join(json_dir, json_fname)
            self.json_export(json_path)
        return json_fname

    def del_path(self, full_path, del_path):
        if del_path is None:
            del_path = ""
        return os.path.relpath(full_path, del_path)

    def gen2matrix(self, gf, kwargs):
        # produce matrix containing resulting data
        gen = gf(**kwargs)
        # assemble the matrix
        outmat = pd.concat(gen, axis=1)
        outmat.columns = self.snames
        prepro = outmat.fillna(0)
        return prepro

    def json_export(self, svpath):
        # store all information we need about the preprocessing as a json file
        with open(svpath, 'w', encoding='utf-8') as f:
            json.dump(self.log, f, ensure_ascii=False, indent=4)
