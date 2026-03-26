# Immune Repertoire Utilities

## Description

Python code for preprocessing of immune repertoire datasets for supervised machine learning and statistical approaches, plus exploratory analysis.

## Prerequisites

* Python 3
* Numpy
* Pandas
 
## Modules

| Module               | Class(es)                                                                       |
| -------------------- | ------------------------------------------------------------------------------- |
| aa_seq_utils.py      | ImSeq                                                                           |
| diversity.py         | None, functions that calculate diversity                                        |
| immune_repertoire.py | ImmuneRepertoire                                                                |
| imrep_dataset.py     | IRDataset                                                                       |
| imrep_plots.py       | DatasetPlotter, DiversityDatasetPlotter, VDJDatasetPlotter, CloneDatasetPlotter |
| kmer_repertoire.py   | KmerRepertoire                                                                  |
| utils.py             | None, utility functions                                                         |

### IRDataset

Main class to preprocess a dataset. 

IRDataset class loads information for dataset ready to read and preprocess it, and has args:
- `ddir`: string path or list of paths. Directory or directories to load dataset from. Files can be in any readable format if paired with a suitable dfunc. Hidden files starting '.' are ignored.
- `dfunc`: a user-written function that extracts relevant information from a file to pass to ImmuneRepertoire. For example, it may read a table with .txt extension, extract CDR3 amino acid column and corresponding counts. imrep_extract/demo_extr.py provides an example of the output format required from dfunc.
- `lfunc`: a user-written function to extract a pandas DataFrame of sample labels as columns from the list of file names. Main input is file name, and other arguments can be passed as `largs` if required
- `nfunc`: can optionally be used to name samples using file name, nargs can optionally be supplied as additional arguments
- `primary_lab`: optional, can exclude samples from dataset if missing a value for the corresponding column in `lfunc` output
- `rs`: set random seed for any preprocessing operations involving random sampling- downsampling
- `sort_flag`: if True (default), sort sample-wise information to produce ordered output matrices based on file alphabetical and numerical order.

prepro() can execute different preprocessing functions on the dataset.
A prepro_func_key is used to define the series of preprocessing steps, as below. kwargs should be a dictionary containing the arguments listed below for each prepro_key.

* `raw_kmers` - just convert the dataset into kmers. This is only appropriate when a single amino acid sequence is extracted through dfunc.
  - `k`- length of kmer
  - `trim`- how many positions to trim from each sequence end before converting to kmers
  - `p`- used to create positional kmers. None for non-positional, or list of annotations e.g. ["start", "middle", "end"]
  - `p_use_counts`- False, recreate counting of positional kmers as in Foers et al. J. Pathol. 2021
  - `p_counts_round_func` - default None, allow different rounding functions in above 
  - `ignore_dup_kmers`: default False, if True only count duplicate kmers from same sequence once
* `ds_kmers` - downsample the dataset to a user-defined threshold, or to the smallest sample size in the dataset, and convert to kmers.  Again, this is only appropriate when a single amino acid sequence is extracted through dfunc.
  - `d` is the number of sequences to downsample to. if None (default), set it automatically using sample with fewest sequences
  - other args as for `raw_kmers`
* `raw_clones` - extract raw information as extracted by ImmuneRepertoire. Suitable for mixed sequence information such as CDR3 amino acid sequence with V and J gene segments.
* `ds_clones` - extract information from ImmuneRepertoire as above, but downsample the dataset to a user-defined threshold, or to the smallest sample size in the dataset.
  - `d` defined above
* `ds_diversity` - downsample the dataset to a user-defined threshold, or to the smallest sample size in the dataset, and calculate any of richness, shannon, simpson, and hill diversity in parallel. Downsampling must be applied because diversity calculations can be biased by variable depth.
  - `div_names`: list of names of diversity indices to calculate, default ["richness", "shannon", "inv_simpson"], can include "richness", "shannon", "inv_simpson", "hill"
  - `q_vals`: default None, list of ints used to set q for Hill diversity
* `ds_vdj` - downsample the dataset to a user-defined threshold, or to the smallest sample size in the dataset, and calculate V, D or J segment usage if any are extracted by ImmuneReperoire.
  - `seq_names`: names of V, D or J segments to calculate usage of depending on column names of gene segments supplied in dfunc output
Using `export=True` we can export the preprocessed data as .csv to a directory related to ddir. 
We can also export a .json file containing labels, and a log of information about the preprocessing and export.

### ImmuneRepertoire

Loads a single immune repertoire sample using a user-defined extraction function. Given a sample that contains n rows, extraction should return, in order:
* An array of sequence information with shape n*1 (also accepts shape n) or n*m. Each of n rows could include a single nucleotide sequence or amino acid sequence, but V, D and J segment information could be included too. 
* List of names of single or each m sequence data. For example, ["CDR3"] where m=1. 
* List of types of single or each m sequence data. For example, ["aaSeq"] when extracting amino acid sequence with names ["CDR3"]. 
* List or array of number of times each sequence/row entry is counted in sample.
See imrep_extract/demo_extr.py.

ImmuneRepertoire defines sample-wise preprocessing operations, including:
* count_seqs- count the total sequences and total unique sequences in a sample. Calculate the proportion of the total repertoire that each sequence makes up.
* downsample- randomly sample, without replacement, d sequences from repertoire. Optionally, overwrite the repertoire with this downsampled version.
* calc_div- calculate diversity as per functions in diversity.py, including hill diversity. q_vals are only used in the case of hill diversity.
* get_as_pandas- convert the repertoire to a pandas dataframe. 
* calc_vdj_usage- calculate how many sequences in repertoire use different V, D, or J gene segments.

### ImSeq

Loads a single amino acid sequence, with respecive count that has a default of 1. Allows trimming, and decomposing sequences into kmers (short overlapping segments).
* trim- trim t amino acids from the start and end of a sequence. Overwrite the sequence and update the sequence length.
* kmerize- use a moving window operation with length k and step of 1 to split sequence into kmers.
* get_aa_pos_fracs- get position of each amino acid within a sequence
* get_kmer_pos_alloc- determine positon of kmer in sequence using it's member amino acid positions
* count_informed_kmer_position- since kmers may not be wholly in one position, allocate some of the total kmer counts to each relevant position. Apply rounding to final counts using python or numpy functions, round to nearest half as in [Foers et al. 2021](https://pathsocjournals.onlinelibrary.wiley.com/doi/10.1002/path.5592), or don't round at all.  
* simple_kmer_position- decide position of kmer based on which position it's mostly in, assign all counts to this position.
* to_kmers- convert the amino acid sequence to non-positional or positional kmers using simple or count-informed methods. If not retreiving positional count-informed kmers, can ignore multiple instances of same kmer in the amino acid sequence.

### KmerRepertoire

Loads a list of sequences and respective counts, and use ImSeq to convert to a repertoire of kmers that may come from trimmed sequences. This repertoire may be converted to a pandas series.


## Exploratory analysis
Dataset files obtained using IRDataset can be used by different relevant plotting classes below.

### DatasetPlotter
Base class that defines saving and other plotting utilities. sam_info should be a dataframe containing labels. Multiple label definitions can be stored for flexibility.

set_colour allows colour to be set according to a label definition in sam_info, using dictionary of colours.

reorder_samples changes the order of samples in a dataset, either using a defined new order or a function.

### DiversityDatasetPlotter
Plot diversity, as obtained via ds_diversity option, with a barplot, boxplot, or lines, the latter of which is appropriate for visualising Hill diversity. Boxplots can be annotated to indicate statistical significance.

### VDJDatasetPlotter
Plot V, D or J segment usage using a heatmap or boxplot. Both can be annotated to indicate statistical significance.

### CloneDatasetPlotter
Using counts, plot abundance of top clones as lines, and count distribution of a single sample as a histogram. Sequence information of length one or more can be used to show the top clones in a dataset using a heatmap.

