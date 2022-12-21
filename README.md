# Immune Repertoire Utilities
Python code that enables preprocessing of immune repertoire datasets for downstream machine learning. Also, immune repertoire dataset plotting is included to allow for exploratory data analysis.
IR_dataset.py is concerned with preprocessing and contains two classes: ImmuneRepertoire and IRDataset. IR_plots.py contains IRPlots, which contains plotting methods. The saved json output of IRDataset can be directly input to IRPlots.
## ImmuneRepertoire
This class loads a single sample and defines sample-wise preprocessing operations. An extraction function must be passed to take only the relevant information from the sample file.
## IRDataset
This class stores a series of instructions to initialise and preprocess a set of immune repertoire samples. It can finally return a matrix of dimensions (number of sequences or kmers)*(number of samples) with labels for each sample. A label extraction function must be specified.
Only two preprocessing generators are available so far: raw_clones which only retrieves relevant information from samples, and ds_kmers, which downsamples sequences to a specified threshold or minimum depth in dataset, and splits sequences into short overlapping segments of length k which may be annotated positionally. 
## IRPlots
This class loads the json output from IRDataset and gets the proprocessed data stored with the original data. seq_inds should be set by the user depending on which columns of the preprocessed data contain sequence information rather than counts. colour_func allows the user to set colours which coould be done per sample or per label, since it takes label series as input which has sample name index.
We can set class variables to determine whether plots should be saved or just opened, and whether legends should be used for some plots. A function can also be specified to modify the settings of matplotlib.
seg_heatmap and seg_boxplots can only be used if there are gene segment columns in the preprocessed data.
