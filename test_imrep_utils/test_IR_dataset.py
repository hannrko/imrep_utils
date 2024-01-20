import pandas as pd
import numpy as np
from immune_repertoire import ImmuneRepertoire
import kmer_repertoire as kr

s1 = ["CASSRPSRPLSMYF", "CASSRPSRPF", "CASSHHHHHHHRRRRRRYW", "CASSRPSRPLSF", "CASSRRRRRRPLSMYF"]
s2 = [["CASSRPSRPLSMYF", "CASSRPSRPF", "CASSHHHHHHHRRRRRRYW", "CASSRPSRPLSF", "CASSRRRRRRPLSMYF"],
     ["V1", "V1", "V2", "V3", "V1"]]
sn1 = ["CDR3"]
sn2 = ["CDR3", "Vgene"]

#s_ind = pd.MultiIndex.from_arrays(s, )
ser = pd.Series(index=s2, data=np.ones(5))
ser.index.names = sn2
print(ser)

