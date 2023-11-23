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
from IR_dataset import ImmuneRepertoire
import pandas as pd
import numpy as np
import random
from IR_dataset import IRDataset


def test_extr_unique(dpath):
    raw = pd.read_table(dpath,index_col=0)
    useqs = np.unique(raw.index)
    return pd.DataFrame(index=useqs,data=np.ones((len(useqs),1)),columns=["counts"])

def test_extr(dpath):
    raw = pd.read_table(dpath,index_col=0)
    return raw

def dummy_lab_extr(labinput):
    return 0
    
# Very basic test
dpath = "dummy_reads.txt"
#random.seed(1)
test = ImmuneRepertoire(dpath,"test",test_extr)
test.downsample(24)
print(test.seqtab)
test.kmerize(3,["a","b","c"])
print(test.kmers)
# we get the same result over and over when random seed is 1.

# now what happens when we use IRDataset?
dummyird = IRDataset("multiple_dummy_reads", dummy_lab_extr, test_extr, rs=1)
dummy_3mers = dummyird.gen2matrix(dummyird.ds_kmers, dict(k=3, thresh=10), export=True, json_path=None)
#dummy_3mers.to_csv("dskmers_rs1.csv")