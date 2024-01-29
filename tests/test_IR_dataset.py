import pandas as pd
import numpy as np
#from immune_repertoire import ImmuneRepertoire
#import kmer_repertoire as kr
import IR_dataset as ird

def cdr3_extr(dpath):
    fmat = pd.read_table(dpath,index_col=0)
    # there are duplicate amino acid seqs
    if "seq_reads" in fmat.columns:
        ccol = "seq_reads"
    else:
        ccol = "templates"
    seqs = fmat.groupby("amino_acid").aggregate({ccol:"sum"})
    print(seqs.values)
    return seqs.index, ["CDR3"], ["aaSeq"], np.squeeze(seqs.values)#seqs.rename(columns={ccol:"counts"},index={"amino_acid":"aa_cdr3"})

def strip_allele(seg_str):
    return seg_str.split("*")[0]

def status_from_pn(emcmv_sam):
    if emcmv_sam.loc['Cytomegalovirus +'] and not emcmv_sam.loc['Cytomegalovirus -']:
        cmv_status = 1
    elif emcmv_sam.loc['Cytomegalovirus -'] and not emcmv_sam.loc['Cytomegalovirus +']:
        cmv_status = 0
    else:
        cmv_status = float('NaN')
    return cmv_status

def lab_extr(fn, tab_path):
    tab = pd.read_table(tab_path,index_col=0)
    # take filename and extract sample same
    # look up sample name in table to get label
    return status_from_pn(tab.loc[fn.split(".")[0]])

dpath = "emcmv_subset"
emcmv_ird = ird.IRDataset(dpath, dfunc=cdr3_extr, lfunc=lab_extr, largs=("emcmv_subset_metadata/cohort1_sum.tsv",), rs=1)
#emcmv_ird.prepro("ds_kmers", dict(k=4, p=None, ignore_dup_kmers=True), export=True, json_dir="C:/Users/Hannrk/imrep_utils/tests/emcmv_subset_metadata")

#emcmv_ird.prepro("ds_clones", dict(), export=True, json_dir="emcmv_subset_metadata")

emcmv_ird.prepro("raw_clones", dict(), export=True, json_dir="emcmv_subset_metadata")


