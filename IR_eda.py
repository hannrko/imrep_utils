# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 07:54:00 2022

@author: hannrk
"""

import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
import seaborn as sns

class IRAnalysis:

    def __init__(self,data_dir,label_dict,input_type='MiXCR',legends=False,stage='align'):

        # save input type
        self.input = input_type
        self.legends=legends

        # get paths of all sample files
        self.sam_paths = [os.path.join(data_dir,d) for d in sorted(os.listdir(data_dir),key=self.extract_sam_info)]

        # get names of samples
        # could use extract
        self.sam_names = np.array(list(map(self.get_sam_name,self.sam_paths)))
        self.n_sam = len(self.sam_names)

        xdict = {'MiXCR':self.mixcr_extract,'IMGT':self.imgt_extract,'PGM':self.pgm_extract}
        # use input type to get name of extraction function
        if self.input in xdict.keys():
            xfunc = xdict[self.input]
        else:
            print('recognised input types are: MiXCR, IMGT, PGM')
        # extract cdr3 dicts, v d and j usage dict
        info_dicts = list(map(xfunc,self.sam_paths,np.repeat(stage,self.n_sam)))
        cdr3_dicts = [row[0] for row in info_dicts]
        v_dicts = [row[1] for row in info_dicts]
        d_dicts = [row[2] for row in info_dicts]
        j_dicts = [row[3] for row in info_dicts]
        # assemble pandas dataframes for CDR3s and vdj usage
        self.cdr3_mat = pd.DataFrame(cdr3_dicts,index=self.sam_names).fillna(0).T
        self.v_usage = pd.DataFrame(v_dicts,index=self.sam_names).fillna(0).T
        self.d_usage = pd.DataFrame(d_dicts,index=self.sam_names).fillna(0).T
        self.j_usage = pd.DataFrame(j_dicts,index=self.sam_names).fillna(0).T
        # extract sequences from cdr3 matrix
        self.seqs = self.cdr3_mat.index
        # get labels of samples
        # this enforces a structure of file names being indicative of label
        self.lab_dict = label_dict
        self.labels = np.array([self.lab_dict[self.extract_sam_info(s)[0]] for s in self.sam_names])
        # set colour scheme for plots
        self.class_n = np.unique(self.labels,return_counts=True)[1]
        # for now just set colours by class: negative class uses bright colours and positive class uses musted oclours
        # this is appropriate for a larger neagtive class
        self.set_colours(plt.cm.hsv(np.linspace(0,1,self.class_n[0])), plt.cm.cubehelix(np.linspace(0,1,self.class_n[1] + 10))[:-10])
        self.get_counts()
        self.prop = self.get_proportions(self.cdr3_mat)

    @staticmethod
    def Hill_div(counts,q_vals):
        # make sure q_vals is 1D array
        q_vals = np.array(q_vals).reshape(-1)
        div = []
        # remove zeros
        counts = counts[counts>0]
        for q in q_vals:
            if q == 1:
                ind = np.exp(-np.sum(counts*np.log(counts)))
            else:
                ind = np.sum(counts**q)**(1/(1-q))
            div = np.append(div,ind)
        return np.squeeze(div)

    @staticmethod
    def imgt_extract(filepath,stage):
        # load table
        sample = pd.read_table(filepath,dtype=str)
        # take only productive aa CDR3s
        sample = sample[~sample['CDR3'].isna()]
        # first take only sequences with productive v regions
        prodv = sample[sample['V-DOMAIN Functionality'] == 'productive']
        # then take ony CDR3s without special characters
        prodcdr3 = prodv[prodv['CDR3'].str.isalpha()]
        # count up identical CDR3s
        cdr3_counts = zip(*np.unique(prodcdr3['CDR3'].values.astype(str),return_counts=True))

        # takes first segment listed, no d in my exmaple data so ignore
        vsegs = dict(zip(*np.unique([v.split('or')[0].strip(',') for v in prodcdr3['V-Region']],return_counts=True)))
        dsegs = dict()
        jsegs = dict(zip(*np.unique([j.split('or')[0].strip(',') for j in prodcdr3['J-segment']],return_counts=True)))

        return dict(cdr3_counts), vsegs, dsegs, jsegs

    @staticmethod
    def mixcr_extract(filepath,stage):
        # load table
        sample = pd.read_table(filepath,dtype=str)
        # first remove rows with no CDR3
        sample = sample[~sample['aaSeqCDR3'].isna()]
        # and get all amino acid sequence columns in table
        aacols = sample.columns[['aaSeq' in heading for heading in sample.columns]]
        # concatenate strings in amino acid seq columns, replacing nan with '' beforehand
        fullseq = sample[aacols].fillna('').sum(axis=1)
        # determine which sequences are productive
        is_prod = fullseq.str.isalpha()
        prodcdr3 = sample[is_prod]

        # get vdj usage dicts
        vsegs = dict(zip(*np.unique([v.split(',')[np.argmax(list(map(int,re.findall(r"\(([A-Za-z0-9_]+)\)", v))))].split('(')[0] for v in prodcdr3['allVHitsWithScore'].fillna('NA(100)')],return_counts=True)))
        dsegs = dict(zip(*np.unique([d.split(',')[np.argmax(list(map(int,re.findall(r"\(([A-Za-z0-9_]+)\)", d))))].split('(')[0] for d in prodcdr3['allDHitsWithScore'].fillna('NA(100)')],return_counts=True)))
        jsegs = dict(zip(*np.unique([j.split(',')[np.argmax(list(map(int,re.findall(r"\(([A-Za-z0-9_]+)\)", j))))].split('(')[0] for j in prodcdr3['allJHitsWithScore'].fillna('NA(100)')],return_counts=True)))

        if stage == 'clone':
            uprodcdr3 = prodcdr3.groupby('aaSeqCDR3').aggregate({'cloneCount':'sum'})
            # return dictionary of CDR3s and counts
            return dict(zip(uprodcdr3.index,uprodcdr3['cloneCount'])), vsegs, dsegs, jsegs
        elif stage == 'align':
            return dict(zip(*np.unique(prodcdr3['aaSeqCDR3'].values.astype(str),return_counts=True))), vsegs, dsegs, jsegs


    @staticmethod
    def pgm_extract(filepath,stage):
        # load sample, don't make first column index because might be cdr3
        # don't add str as datatype, code breaks
        sample = pd.read_table(filepath)
        # take CDR3s that consist of alphabetical characters only
        # made decision to disregard any sequences with functionality comments
        prodcdr3 = sample[sample['CDR3'].str.isalpha()]
        # need to add together identical cdr3s to ensure that dictionary doesn't overwrite counts
        uprodcdr3 = prodcdr3.groupby('CDR3').aggregate({'Count':'sum'})

        vsegs = dict(zip(*np.unique([v for v in prodcdr3['SegmentV']],return_counts=True)))
        dsegs = dict()
        jsegs = dict(zip(*np.unique([j for j in prodcdr3['SegmentJ']],return_counts=True)))
        # return dictionary of CDR3s and counts
        return dict(zip(uprodcdr3.index,uprodcdr3['Count'])), vsegs, dsegs, jsegs

    @staticmethod
    def get_sam_name(file_dir):
        return re.split('_|-|\\.',os.path.split(file_dir)[-1])[0]

    @staticmethod
    def extract_sam_info(fn):
        # get sam name
        snm = re.split('_|-|\\.',fn)[0]
        num = int(re.findall(r'\d+', snm)[0])
        lab = re.findall(r'[A-Za-z]+',snm)[0]
        return lab, num

    @staticmethod
    def forget_seqs(mat):
        return mat[~(mat.sum(axis=1) == 0)]

    @staticmethod
    def get_proportions(mat):
        prop = mat/(mat.sum())
        return prop

    @staticmethod
    def get_AA_pos_fracs(l_seq,n_pos):
        # initlaise aa_pos
        aa_pos = np.zeros((0,l_seq))
        prev = np.zeros(l_seq)
        fracs = np.linspace(0,1,l_seq+1)
        for i in np.arange(n_pos)+1:
            pointer = np.array([(fracs -i*1/n_pos).clip(min=0)[j:j+2] for j in range(l_seq)])
            # indices that pointer has passed
            ind_sec = pointer == 0
            # if start or end fractions of each index are passed, add entry
            res = np.where(np.logical_or(ind_sec[:,0],ind_sec[:,1]), 1 - pointer.sum(axis=1)*l_seq, 0) - prev
            aa_pos = np.append(aa_pos,res.reshape(1,-1),axis=0)
            prev = prev + res
        return aa_pos

    @staticmethod
    def split_into_kmers(seq,counts,k,positional=None,position_cond_coeff = 0):
        # prepare empty dictionary which will hold kmers as keys, kmer counts as values
        kmer_dict = {}
        # if the sequences is shorter than or equal to desired kmer length, return None
        if len(seq) < k:
            return None
        # the number of k-mers we'll end up with
        n_kmers = len(seq)-k + 1
        # split into kmers with moving window
        kmers = [seq[i:i+k] for i in range(n_kmers)]
        # if no positional labels supplied, just return kmers
        if not positional:
            kmer_dict.update(zip(kmers,np.tile(counts,(n_kmers,1))))
            return kmer_dict
        # but if we do have positional labels
        else:
            # positional annotation condition: check if cdr3 is long enough to justify splitting into kmers
            if len(seq) <= position_cond_coeff*k:
                return None
            else:
                n_sec = len(positional)
                # calculate how far along CDR3 each AA is
                aa_pos = IRAnalysis.get_AA_pos_fracs(len(seq), n_sec)
                # for each kmer, take positional fractions of each AA
                kmer_pos_by_aa = np.array([aa_pos[:,i:i+k] for i in range(n_kmers)])
                # count up these fractions to determine fraction of kmer as whole
                kmer_counts = np.outer(np.sum(kmer_pos_by_aa,axis=2)/k,counts)
                # round kmer counts so that all exact halves are preserved
                rounded_kmer_counts = np.round(2*kmer_counts)/2
                # for each kmer, position, and corresponding kmer counts
                for kmer, p, c in zip(np.repeat(kmers,n_sec), np.tile(positional,n_kmers), rounded_kmer_counts):
                    #if count isn't zero for this positional kmer
                    if c.any():
                        # make new kmer name
                        kmer_name = kmer + p
                        # add kmer to dictionary
                        # in case positional labels are duplicates,
                        # check if entries already exist
                        if kmer_name in kmer_dict:
                            kmer_dict[kmer_name] += c
                        else:
                            kmer_dict[kmer_name]= c
            return kmer_dict

    def set_colours(self, clist1, clist2):
        # set a distinct colour for each sample
        unsorted_colours = np.append(clist1,clist2,axis=0)
        colour_dict = dict(zip(np.append(self.sam_names[self.labels == np.amin(list(self.lab_dict.values()))],self.sam_names[self.labels == np.amax(list(self.lab_dict.values()))]),unsorted_colours))
        self.colours = np.array([colour_dict[s] for s in self.sam_names])

    # important function
    def del_sample(self,to_del):
        # make sure we have a 1D array
        to_del = np.array(to_del).reshape(-1)
        # identify indices of samples to delete
        #del_ind = (np.array([np.where(self.sam_names==d) for d in to_del])).reshape(-1)
        # delete samples
        self.cdr3_mat = self.cdr3_mat.drop(columns = to_del)
        self.cdr3_mat = self.forget_seqs(self.cdr3_mat)
        self.overwrite(to_del=to_del)

    # does this function use replacement? it didn't but I altered implementation
    # so only samples that require downsampling are processed
    def downsample(self,thresh = None,rs = None,overwrite=False):
        if thresh == None:
            self.d_thresh = self.cdr3_mat.sum().min()
        else:
            self.d_thresh = thresh
        if rs is not None:
            np.random.seed(rs)
        # samples remaining are more or just as deep as threshold
        rem_ind = self.cdr3_mat.sum() >= self.d_thresh
        print(rem_ind, self.cdr3_mat.sum(),self.d_thresh)
        # copy the cdr3 matrix to prevent it being overwritten
        self.down = self.cdr3_mat.loc[:,rem_ind].copy()
        # iterate though columns
        for c in self.down.columns:
            # only downsample columns that actually need downsampling
            if self.down[c].sum()>self.d_thresh:
                # expand sequences index so that they are repeated according to counts
                exp_seqs= np.repeat(self.down[c].index,self.down[c].values)
                # sample sequences without replacement
                exp_sam = np.random.choice(exp_seqs,self.d_thresh,p = np.ones(len(exp_seqs))/len(exp_seqs),replace=False)
                # count up the downsampled sequences
                dseqs, dcounts = np.unique(exp_sam,return_counts=True)
                # and assemble back into pandas series
                a = pd.Series(index=self.down.index,data=np.zeros(len(self.down.index)))
                for ds,dc in zip(dseqs,dcounts):
                    a[ds] = dc
                # save in the relevant column
                self.down.loc[:,c] = a
        # need to remove zero rows
        self.down = self.forget_seqs(self.down)

        # could write an overwrite function
        if overwrite:
            # overwrite CDR3 matrix
            self.cdr3_mat = self.down
            self.overwrite(np.delete(self.sam_names,rem_ind))
        return self.down

    def overwrite(self,to_del):
        # identify indices where ntries no longer exist?
        # assume CDR3 matrix already overwritten
        # make sure we have a 1D array
        to_del = np.array(to_del).reshape(-1)
        # identify indices of samples to delete
        del_ind = (np.array([np.where(self.sam_names==d) for d in to_del])).reshape(-1).astype(int)
        self.sam_names = np.delete(self.sam_names,del_ind)
        self.n_sam = len(self.sam_names)
        self.labels = np.delete(self.labels,del_ind)
        self.sam_paths = np.delete(self.sam_paths,del_ind)
        self.class_n = np.unique(self.labels,return_counts=True)[1]
        # for now just set colours by class: negative class uses bright colours and positive class uses musted oclours
        # this is appropriate for a larger neagtive class
        self.colours = np.delete(self.colours,del_ind,axis=0)
        self.prop = self.get_proportions(self.cdr3_mat)
        self.counts = self.cdr3_mat.sum()
        self.seqs = self.cdr3_mat.index
        # if diversity has been calculated, delete it
        if hasattr(self,'div'):
            del self.div
        self.v_usage = self.v_usage.drop(columns = to_del)
        self.v_usage = self.forget_seqs(self.v_usage)
        self.d_usage = self.d_usage.drop(columns = to_del)
        self.d_usage = self.forget_seqs(self.d_usage)
        self.j_usage = self.j_usage.drop(columns = to_del)
        self.j_usage = self.forget_seqs(self.j_usage)


    def get_counts(self):
        self.counts = self.cdr3_mat.sum()
        return self.counts

    def to_kmers(self, k, positional=None, position_cond_coeff = 0,return_short_kmers=False):
        # initialise empty dictionary
        kmer_mat = {}
        short_kmers = []
        # iterate over sequences and counts
        for s, c in zip(self.seqs,self.cdr3_mat.values):
            # use split_into_kmers function with or without positional labels
            kmers = self.split_into_kmers(s, c, k, positional, position_cond_coeff = position_cond_coeff)
            if kmers is None:
                short_kmers.extend(s)
            else:
                # iterate over resulting kmers
                for kmer, count in kmers.items():
                    # if a kmer is already in the dictonary
                    if kmer in kmer_mat:
                        # add counts to the existing entry
                        kmer_mat[kmer] = kmer_mat[kmer] + count
                    # if entry doesn't exist yet, create it
                    else:
                        kmer_mat[kmer] = count
        # change format to dataframe, where columns are sample names, index is kmers
        self.kmer_mat = pd.DataFrame.from_dict(kmer_mat, orient='index', columns=self.sam_names)
        self.store_kmers(k,positional)
        if return_short_kmers:
            return self.kmer_mat, short_kmers
        else:
            # return kmer dataframe
            return self.kmer_mat

    # store kmer matrices in dict?
    def store_kmers(self,k,positional):
        # check if all_kmers exists
        try:
            self.all_kmers
        except AttributeError:
            self.all_kmers = {}
        if positional:
            pre=''
        else:
            pre='non-'
        kmer_name = pre + 'positional ' + str(k) + 'mers'
        self.all_kmers[kmer_name] = self.kmer_mat

    def plot_counts(self,inp='cdr3',figsize = (20,10),title=None,ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize= figsize)
        else:
            fig=None
        if inp =='cdr3':
            y = self.counts
        elif inp == 'kmer':
            y = self.kmer_mat.sum()
        ax.bar(self.sam_names,y,color = self.colours)
        ax.set_ylabel('Sequence counts')
        ax.set_xlabel('Sample')
        plt.xticks(rotation=90)
        plt.margins(x=0)
        #plt.legend()
        if title is not None:
            ax.set_title(title)
        if fig is not None:
            plt.show()
        else:
            return ax

    def plot_seq_counts(self,seq, inp='cdr3',figsize = (20,10),title=None,ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize= figsize)
        else:
            fig = None
        if inp =='cdr3':
            y = self.cdr3_mat.loc[seq]
        elif inp == 'kmer':
            y = self.kmer_mat.loc[seq]
        ax.bar(self.sam_names,y,color = self.colours)
        ax.set_ylabel('Sequence counts')
        ax.set_xlabel('Sample')
        plt.xticks(rotation=90)
        plt.margins(x=0)
        #plt.legend()
        if title is not None:
            ax.set_title(title)
        if fig is not None:
            plt.show()
        else:
            return ax

    def plot_total_count_hist(self,bins=20,figsize = (20,10),inc_class = None,title=None,ax=None):
        # if not returning plot to another figure, make our own figure
        if ax is None:
            fig, ax = plt.subplots(figsize= figsize)
        # if no classes specified for inclusion, include all
        if inc_class is None:
            inc_ind = np.arange(self.n_sam)
        else:
            inc_lab = np.unique([self.lab_dict[k] for k in inc_class])
            inc_ind =  np.concatenate([np.argwhere(self.labels == l).reshape(-1) for l in inc_lab])
        ax.hist(self.counts[inc_ind],color='k',bins=bins)
        ax.set_xlabel('Counts in sample')
        ax.set_ylabel('Frequency')
        plt.xticks(rotation=90)
        #plt.legend()
        if title is not None:
            ax.set_title(title)
        if fig is not None:
            plt.show()
        else:
            return ax

    def plot_total_count_kde(self,figsize = (20,10),inc_class = None,ax=None):
        # if not returning plot to another figure, make our own figure
        if ax is None:
            fig, ax = plt.subplots(figsize= figsize)
        # if no classes specified for inclusion, include all
        if inc_class is None:
            inc_ind = np.arange(self.n_sam)
        else:
            inc_lab = np.unique([self.lab_dict[k] for k in inc_class])
            inc_ind =  np.concatenate([np.argwhere(self.labels == l).reshape(-1) for l in inc_lab])

        kde = KernelDensity(kernel='epanechnikov', bandwidth=5000).fit(np.array(self.counts[inc_ind]).reshape(-1,1))
        x = np.linspace(0,np.amax(self.counts[inc_ind]),10000)
        logprob = kde.score_samples(x.reshape(-1,1))

        plt.fill_between(x, np.exp(logprob),color='k' ,alpha=0.5)
        ax.set_xlabel('Counts in sample')
        ax.set_ylabel('Frequency')
        plt.xticks(rotation=90)
        if fig is not None:
            plt.show()
        else:
            return ax

    def get_diversity(self,q_vals):
        self.div =  self.prop.apply(self.Hill_div,axis=0,args=(q_vals,))
        if isinstance(self.div, pd.DataFrame):
            self.div.index = q_vals
        return self.div.to_numpy().T

    def plot_diversity_bar(self,q_val,norm_by_counts=True,figsize= (20,20),title=None,ax=None):
        div = self.get_diversity(q_val)
        # if axes not already existing and passed, make a single axis figure
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        # normalise by counts by default
        if norm_by_counts:
            norm = self.counts
            ylabel = 'Diversity per sequencing depth'
        # but provide null normalisation
        else:
            norm = np.ones_like(self.counts)
            ylabel = 'Unadjusted diversity'
        ax.bar(self.sam_names,div/norm,color = self.colours)
        ax.set_xlabel('Sample')
        plt.xticks(rotation=90)
        ax.margins(x=0)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        if fig is not None:
            if self.legends:
                plt.legend()
            plt.show()
        else:
            return ax

    def plot_diversity_lines(self,q_vals,norm_by_counts=True,hili = [],figsize= (20,20),inc_class = None,title=None,ax=None):
        #calculate the diversity for q values supplied
        div = self.get_diversity(q_vals)
        # if axes not already existing and passed, make a single axis figure
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        # if no class to include is specified, include all data
        if inc_class is None:
            inc_ind = np.arange(self.n_sam)
        else:
            # otherwise get indices of all data to include based on label codes
            inc_lab = np.unique([self.lab_dict[k] for k in inc_class])
            inc_ind =  np.concatenate([np.argwhere(self.labels == l).reshape(-1) for l in inc_lab])
        # normalise by counts by default
        if norm_by_counts:
            norm = self.counts
            ylabel = 'Diversity per sequencing depth'
        # but provide null normalisation
        else:
            norm = np.ones_like(self.counts)
            ylabel = 'Unadjusted diversity'
        # plot multiple diversity lines on the same axes
        for i in range(len(inc_ind)):
            ax.plot(q_vals,(div[inc_ind][i])/norm[inc_ind][i],color = self.colours[inc_ind][i],label=np.where(self.legends,self.sam_names[inc_ind][i],'_nolegend_'), linewidth=1)
        # replot specified sample diversities over the top and thicker
        for s in hili:
            j = int(np.argwhere(np.array(self.sam_names[inc_ind]) == s))
            ax.plot(q_vals,div[inc_ind][j]/norm[j],color = self.colours[inc_ind][j],label=s, linewidth=3)
        ax.set_xlabel('q indicating diversity index')
        ax.set_ylabel(ylabel)
        ax.set_yscale('log')
        ax.set_xticks(q_vals)
        ax.margins(x=0)
        if title is not None:
            ax.set_title(title)
        if fig is not None:
            if self.legends:
                plt.legend()
            elif len(hili) > 0:
                plt.legend()
            plt.show()
        else:
            return ax

    def plot_vdj_usage(self,seg,figsize=(40,20)):
        if seg == 'V':
            data = self.v_usage
            cmap='Blues'
        elif seg == 'D':
            data = self.d_usage
            cmap='Reds'
        elif seg == 'J':
            data = self.j_usage
            cmap='Greens'
        plt.figure(figsize=figsize)
        sns.heatmap(data,cmap=cmap,linewidths=.5)
        plt.show()

    def plot_vdjseg_by_class(self,seg,gene,classes,title=None,ax=None,figsize=(10,8)):
        if seg == 'V':
            data = self.v_usage
        elif seg == 'D':
            data = self.d_usage
        elif seg == 'J':
            data = self.j_usage
        usage = self.get_proportions(data).loc[gene]
        labels = [self.lab_dict[c] for c in classes]
        by_class = [usage[self.labels == lab] for lab in labels]
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig=None
        ax.boxplot(by_class,labels = classes)
        ax.set_ylabel('Proportion of repertoire')
        if title is not None:
            ax.set_title(title)
        if fig is not None:
            plt.show()
        else:
            return ax

    # plot spectrum of sequences in repertoire
    def plot_seq_spec(self,sam,c,ax):
        ax.plot(self.cdr3_mat[sam].to_numpy(),color=c)
        ax.set_xlabel('Sequence ID')
        ax.set_ylabel('Counts')
        return ax

    def plot_count_hist(self,sam,c,b,ax):
        ax.hist(self.cdr3_mat[sam],label=sam,color = c,log=True, bins = b)#int(data.values.max()/5))
        ax.set_xlabel('Counts')
        ax.set_ylabel('Frequency')
        return ax

    def get_feature_pairs(self,f1,f2):
        f12 = np.array(np.meshgrid(f1,f2)).T.reshape(-1,2)
        f12 = f12[f12[:,0]!=f12[:,1]]
        f12 = f12[~np.array([(np.flip(f) == f12[:i]).all(axis=1).any() for i,f in enumerate(f12)])]
        return f12

    def set_pca(self,mat):
        self.all_pca = PCA()
        self.allpc = self.all_pca.fit_transform(mat.values.T)

    def plot_pca(self,f1,f2,inp='cdr3',ex = {},title=None,axs = None):
        f12 = self.get_feature_pairs(f1,f2)
        num_plots = len(f12)
        if axs is None:
            fig, axs = plt.subplots(num_plots,1,figsize = (10,10*num_plots))
        if inp == 'cdr3':
            y = self.prop
        elif inp=='kmer':
            y = self.get_proportions(self.kmer_mat)
        self.set_pca(y)
        try:
            all_ax = axs.ravel()
        except:
            all_ax = np.array([axs])
        for i, f in enumerate(f12):
            all_ax[i] = self.plot_pc_pair(f[0],f[1],all_ax[i],ex)
        if title is not None:
            fig.suptitle(title)
        if fig is not None:
            plt.show()
        else:
            return all_ax

    def plot_pc_pair(self,f1,f2,ax,ex):
        plot_key = str(f1+ 1) + ' ' + str(f2 + 1)
        if plot_key in ex.keys():
            ex_sam = ex[plot_key]
            # this is slightly dodgy but works
            ex_ind = [self.cdr3_mat.columns.get_loc(ex_col) for ex_col in ex_sam]
        else:
            ex_ind = []

        inc_data = np.delete(self.allpc,ex_ind,axis=0)
        ax.scatter(inc_data[:,f1],inc_data[:,f2],c = np.delete(self.labels,ex_ind))

        inc_names = np.delete(self.sam_names,ex_ind)
        texts = [ax.text(inc_data[:,f1][i],inc_data[:,f2][i], lab,size='large') for i,lab in enumerate(inc_names)]
        ax.set_xlabel('Principal component ' + str(f1 + 1))
        ax.set_ylabel('Principal component ' + str(f2 + 1))
        return ax

    # plot 100 most common sequences in repertoire
    def plot_top_seqs(self,n,include='all',title=None):
        # if all elements equal to first
        if self.cdr3_mat.sum().eq(self.cdr3_mat.sum()[0]).all():
            y_label =  'Downsampled counts'
        else:
            print('Warning: CDR3 matrix is not downsampled so plot is biased towards deeper samples')
            y_label = 'Non-downsampled counts'
        if include == 'all':
            include = self.sam_names
        inc_ind = (np.array([np.where(self.sam_names==i) for i in include])).reshape(-1)
        top_seqs = self.cdr3_mat[include].sum(axis=1).sort_values(ascending=False)[:n].index
        top_counts = self.cdr3_mat[include].loc[top_seqs]
        fig, ax = plt.subplots(figsize=(20,10))
        b = top_counts[self.sam_names[inc_ind][0]]
        ax.bar(top_seqs,b,label = self.sam_names[inc_ind][0],color = self.colours[inc_ind][0])
        for i, c in enumerate(self.sam_names[inc_ind][1:]):
            t = top_counts[c]
            ax.bar(top_seqs, t,bottom=b,label=c,color = self.colours[inc_ind][i+1])
            b = b + t
        plt.xticks(rotation=90)
        plt.xlabel('Top CDR3 sequences')
        plt.ylabel(y_label)
        plt.margins(x=0,y=0.01)
        if self.legends:
            plt.legend()
        if title is not None:
            plt.title(title)
        plt.show()
