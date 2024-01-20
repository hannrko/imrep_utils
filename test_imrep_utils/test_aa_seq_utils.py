import aa_seq_utils as aasu
import kmer_repertoire as kr
import utils
import time

#ts = aasu.ImSeq("CASSRPSRPLSMYF", count=8)
#ts.trim(2)
#print(ts.to_kmers(3,  p=3, p_use_counts=True, p_counts_round_func=ts.foersetal_kmer_position_rounding))

s = ["CASSRPSRPLSMYF", "CASSRPSRPF", "CASSHHHHHHHRRRRRRYW", "CASSRPSRPLSF", "CASSRRRRRRPLSMYF"]
start = time.time()
test_kmers = kr.KmerRepertoire(3, s, kmer_kwargs={"ignore_dup_kmers": True})
end = time.time()
print(end-start)
# 0.0009968280792236328
# 0.0
print(test_kmers.kmers)

d1 = {"A": 1, "B": 1, "C": 7}
d2 = {"A": 1, "B": 1, "C": 1, "D": 3}

print(utils.dicts_to_dict([d1, d2]))
print(utils.lists_to_dict(["A", "B", "C", "B", "C"], [1,1,1,2,4]))