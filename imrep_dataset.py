import os
import re
import json
import pandas as pd
import numpy as np
import random
import immune_repertoire as imrep
import kmer_repertoire as krep

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
                                 "ds_clones": self.ds_clones, "raw_clones": self.raw_clones,
                                 "ds_diversity": self.ds_diversity, "ds_vdj": self.ds_vdj}
        #self.pos = {0: None, 1: ["start", "middle", "end"]}

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
        self.dropped.extend(sam_names)
        # then overwrite sample names, file names and file paths lists
        self.fnames = np.delete(self.fnames, idx_del)
        self.ds_paths = np.delete(self.ds_paths, idx_del)
        self.snames = np.delete(self.snames, idx_del)
        # save labels of dropped samples
        self.drpd_labs.update(self.labs[sam_names].to_dict())
        # overwrite labels
        self.labs = self.labs[self.snames]
        # if we've calculated counts we should delete relevant entries
        if self.count_flag:
            # we should also have a dropped counts attribute to make sure we understand why they were dropped
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

    def prepro(self, prepro_func_key, kwargs, export=True, json_dir=None, ds_name=None, lab_spec=None):
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
            if ds_name is None:
                ds_name = os.path.split(self.ddir)[1]
            prepro_name = f"{ds_name}_{lab_spec}{prepro_func_key}_{kwargs_name}"
            # save matrix to location, store location
            #prepro_dir = os.path.join(self.ddir, "preprocessed")
            ddir_path, ddir_name = os.path.split(self.ddir)
            prepro_dir = os.path.join(ddir_path, ddir_name + "_preprocessed")
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
        return json_fname

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
