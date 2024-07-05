import pandas as pd
import numpy as np
#from immune_repertoire import ImmuneRepertoire
#import kmer_repertoire as kr
import imrep_dataset as ird

def cdr3_extr(dpath):
    fmat = pd.read_table(dpath,index_col=0)
    # there are duplicate amino acid seqs
    if "seq_reads" in fmat.columns:
        ccol = "seq_reads"
    else:
        ccol = "templates"
    seqs = fmat.groupby("amino_acid").aggregate({ccol:"sum"})
    return seqs.index, ["CDR3"], ["aaSeq"], np.squeeze(seqs.values)#seqs.rename(columns={ccol:"counts"},index={"amino_acid":"aa_cdr3"})

def vdjcdr3_extr(dpath):
    fmat = pd.read_table(dpath,index_col=0)
    # there are duplicate amino acid seqs
    if "seq_reads" in fmat.columns:
        ccol = "seq_reads"
    else:
        ccol = "templates"
    seqs = fmat.groupby(["amino_acid", "v_resolved", "d_resolved", "j_resolved"]).aggregate({ccol:"sum"})
    return seqs.index, ["CDR3", "V", "D", "J"], ["aaSeq", "geneSeg", "geneSeg", "geneSeg"], np.squeeze(seqs.values)

def strip_allele(seg_str):
    return seg_str.split("*")[0]

def status_from_pn(emcmv_sam):
    if emcmv_sam.loc['Cytomegalovirus +'] and not emcmv_sam.loc['Cytomegalovirus -']:
        cmv_status = 1
    elif emcmv_sam.loc['Cytomegalovirus -'] and not emcmv_sam.loc['Cytomegalovirus +']:
        cmv_status = 0
    else:
        cmv_status = np.NAN#float('NaN')
    return cmv_status

def lab_extr(fn_list, tab_path):
    tab = pd.read_table(tab_path,index_col=0)
    cmv = np.array([status_from_pn(tab.loc[fn.split(".")[0]]) for fn in fn_list])
    labs = pd.DataFrame(index=fn_list, data=cmv, columns=["CMV"])
    # take filename and extract sample same
    # look up sample name in table to get label
    return labs

dpath = "emcmv_subset"
emcmv_ird = ird.IRDataset(dpath, dfunc=vdjcdr3_extr, lfunc=lab_extr, largs=("emcmv_subset_metadata/cohort1_sum.tsv",), rs=1, primary_lab="CMV")
#emcmv_ird.prepro("ds_kmers", dict(k=4, p=None, ignore_dup_kmers=True), export=True, json_dir="C:/Users/Hannrk/imrep_utils/tests/emcmv_subset_metadata")

#emcmv_ird.prepro("ds_clones", dict(), export=True, json_dir="emcmv_subset_metadata")

emcmv_ird.prepro("ds_clones", {"d": 800}, export=True, json_dir="emcmv_subset_metadata")


