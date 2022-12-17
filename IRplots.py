import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class IRPlots:
    def __init__(self,json_fn,colour_func,seq_inds):
        # read in json file containing info
        with open(json_fn, "rb") as f:
            ppinfo = json.load(f)
        self.labels = pd.Series(ppinfo["labs"])
        self.colours = colour_func(self.labels)
        self.data = pd.read_csv(ppinfo["prepro_path"], index_col=seq_inds)
        # set totc flag
        self.totc_flag = False
        # if we don't have proportions already, get proportions
        if (self.data.sum() == 1).all():
            self.props = self.data
        else:
            self.props = self.data/self.data.sum()


    # Counts plot
    def plot_totc(self, title=None):
        self.totc = self.data.sum()
        plt.figure()
        plt.bar(self.totc.index, self.totc.values, color=self.colours.values)
        plt.ylabel('Total productive sequences')
        plt.xlabel('Sample')
        plt.xticks(rotation=90)
        plt.margins(x=0)
        if title:
            plt.title(title)
        plt.show()

    # total counts histogram
    def totc_hist(self,bins, title=None):
        plt.figure()
        plt.hist(self.totc,color='k',bins=bins)
        plt.xlabel('Total productive sequences')
        plt.ylabel('Number of samples')
        plt.xticks(rotation=90)
        if title:
            plt.title(title)
        plt.show()


    # Abundance plot
    def plot_abund(self, top_n, title=None):
        plt.figure()
        for c in self.data.columns:
            plt.plot(sorted(self.data[c].values, reverse=True)[:top_n], color=self.colours[c])
        plt.ylabel('Clonal frequency')
        plt.xlabel('Sample')
        plt.xticks(rotation=90)
        plt.margins(x=0)
        if title:
            plt.title(title)
        plt.show()

    # Diversity plot
    def plot_div(self, divfunc, title=None):
        # calculate named diversity measures
        div = self.props.apply(divfunc)
        plt.figure()
        plt.bar(div.index, div.values, color=self.colours.values)
        plt.ylabel('Diversity')
        plt.xlabel('Sample')
        plt.xticks(rotation=90)
        plt.margins(x=0)
        if title:
            plt.title(title)
        plt.show()

    @staticmethod
    def richness(counts):
        return sum(counts > 0)

    @staticmethod
    def shannon(props):
        props = props[props > 0]
        return -sum(props*np.log(props))

    @staticmethod
    def simpson(props):
        props = props[props > 0]
        return sum(props**2)

    @staticmethod
    def Hill_div(props, q_vals):
        # make sure q_vals is 1D array
        q_vals = np.array(q_vals).reshape(-1)
        div = []
        # remove zeros
        props = props[props > 0]
        for q in q_vals:
            if q == 1:
                ind = np.exp(-np.sum(props * np.log(props)))
            else:
                ind = np.sum(props ** q) ** (1 / (1 - q))
            div = np.append(div, ind)
        return np.squeeze(div)

    def plot_dprofile(self, q_vals):
        # calculate hill profile
        self.hill = self.props.apply(self.Hill_div, args=(q_vals,))
        plt.figure()
        for c in self.data.columns:
            plt.plot(q_vals, self.hill[c], self.colours[c])
        plt.show()


    # VJ heatmap
    def seg_heatmap(self,col_name):
        seg_counts = self.props.groupby(by=col_name).sum()
        plt.imshow(seg_counts, cmap='hot', interpolation='nearest')
        plt.show()

    # VJ boxplots
    def seg_boxplots(self,seg_name=None,col_name=None):
        # one parameter must be passed
        # either plot for specific segment or all segments in v dj column
        seg_counts = self.props.groupby(by=col_name).sum()
        all_labs = np.unique(self.labels)
        print(all_labs)
        print(seg_counts.loc[seg_counts.index[0]])
        if seg_name:
            segs = [seg_name]
        else:
            segs = seg_counts.index
        for s in segs:
            plt.figure()
            plt.boxplot([seg_counts.loc[s][self.labels == lab] for lab in all_labs],labels=all_labs)
            plt.show()

    # top seqs
    def plot_top(self, n_top, spec_lab=None, title=None):
        # this works with proportions
        if spec_lab is not None:
            tp_data = self.props[self.props.columns[self.labels == spec_lab]]
        else:
            tp_data = self.props
        sum = tp_data.sum(axis=1)
        sum.index = range(len(tp_data))
        tp_sorted = tp_data.iloc[sum.sort_values(ascending=False).index].iloc[:n_top]
        b = pd.Series(index=tp_sorted.index,data =np.zeros(len(tp_sorted)))
        seq_ids = ["-".join([i[1],i[0],i[2]]) for i in tp_sorted.index]
        for i, c in enumerate(tp_sorted.columns):
            t = tp_sorted[c].values
            plt.bar(seq_ids, t, bottom=b, label=c, color=self.colours[c])
            b = b + t
        plt.xticks(rotation=90)
        plt.xlabel("Top " + str(n_top) + " sequences")
        plt.ylabel("Proportion of repertoire")
        plt.margins(x=0)
        plt.legend()
        if title is not None:
            plt.title(title)
        plt.show()



