# Immune Repertoire Utilities

## Description
Python code that enables preprocessing of immune repertoire datasets for supervised machine learning and statistical approaches, plus exploratory analysis.

## Prerequisites
* Python 3
* NumPy
* Pandas
 
## Modules
|Module|Class(es)|
|......|.........|
|aa_seq_utils.py|ImSeq|
|diversity.py|None, functions that calculate diversity|
|immune_repertoire.py|ImmuneRepertoire|
|imrep_dataset.py|IRDataset|
|imrep_plots.py|DatasetPlotter, DiversityDatasetPlotter, VDJDatasetPlotter, CloneDatasetPlotter|
|kmer_repertoire.py|KmerRepertoire|
|utils.py|None, utility functions|

### ImmuneRepertoire
Loads a single immune repertoire sample using a user-defined extraction function. Given a sample that contains n rows, extraction should return, in order:
* An array of sequence information with shape n*1 (also accepts shape n) or n*m. Each of n rows could include a single nucleotide sequence or amino acid sequence, but V, D and J segment information could be included too. 
* List of names of single or each m sequence data. For example, ["CDR3"] where m=1. 
* List of types of single or each m sequence data. For example, ["aaSeq"] when extracting amino acid sequence with names ["CDR3"]. 
* List or array of number of times each sequence/row entry is counted in sample.
See imrep_extract/demo_extr.py.

ImmuneRepertoire defines sample-wise preprocessing operations, indluding:
* count_seqs- count the total sequences and total unique sequences in a sample. Calculate the proportion of the total repertoire that each sequence makes up.
* downsample- randomly sample, without replacement, d sequences from repertoire. Optionally, overwrite the repertoire with this downsampled version.
* calc_div- calculate diversity as per functions in diversity.py, including hill diversity. q_vals are only used in the case of hill diversity.
* get_as_pandas- convert the repertoire to a pandas dataframe. 
* calc_vdj_usage- calculate how many sequences in repertoire use different V, D, or J gene segments.





