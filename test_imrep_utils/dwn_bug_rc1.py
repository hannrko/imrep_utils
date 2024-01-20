# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:37:07 2023

@author: hannrk
"""

import os
cwd = os.getcwd()
import sys
parent_dir = os.path.abspath(os.path.join(cwd, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from immune_repertoire import ImmuneRepertoire
import kmer_repertoire as kr
import pandas as pd
import numpy as np
import random
from IR_dataset import IRDataset
import imrep_extract.demo_extr as de


def test_extr_unique(dpath):
    raw = pd.read_table(dpath,index_col=0, header=0)
    print(raw)
    useqs = np.unique(raw.index)
    return useqs, ["CDR3"], ["aaSeq"], np.ones(len(useqs))#pd.DataFrame(index=useqs,data=np.ones((len(useqs),1)),columns=["counts"])

def test_extr(dpath):
    raw = pd.read_table(dpath)
    return raw["CDR3"].values, ["CDR3"], ["aaSeq"], raw["count"].values#pd.Series(index=raw["CDR3"].values, data=raw["count"].values)

def test_extr_vj(dpath):
    raw = pd.read_table(dpath)
    return raw[["CDR3", "Vgene", "Jgene"]].values, ["CDR3", "V", "J"], ["aaSeq", "geneSeg"], raw["count"].values

def dummy_lab_extr(labinput):
    return 0
    
# Very basic test
dpath = "dummy_reads.txt"
#random.seed(1)
test = ImmuneRepertoire(dpath,"test",de.cdr3_mat_extr)
print(test.downsample(20))
#print(test.get_as_pandas())
#print(test.calc_vdj_usage(["V", "J"]))
#test_kmers = kr.KmerRepertoire(3, test.seq_info)
#print(test_kmers.kmers)
#print(test.calc_div(["shannon", "simpson", "richness", "hill"], q_vals=np.arange(5)))
#print(test.seqtab)
#test.kmerize(3,["a","b","c"])
#print(test.kmers)
# we get the same result over and over when random seed is 1.

# now what happens when we use IRDataset?
dummyird = IRDataset(["multiple_dummy_reads", "other_dummy_reads"], dummy_lab_extr, test_extr, rs=1)
print(dummyird.prepro("raw_kmers", dict( k=5, p=1), export=True, json_dir=None))
#dummy_3mers.to_csv("dskmers_rs1.csv")