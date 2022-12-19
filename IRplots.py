import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

class IRPlots:
    def __init__(self,json_fn,colour_func,seq_inds,neat_figs=False,save=None):
        # read in json file containing info
        with open(json_fn, "rb") as f:
            ppinfo = json.load(f)
        self.labels = pd.Series(ppinfo["labs"])
        self.colours = colour_func(self.labels)
        self.seq_inds = seq_inds
        self.data = pd.read_csv(ppinfo["prepro_path"], index_col=self.seq_inds)

        # plotting style for
        if neat_figs:
            mpl.rc("font",**{"family":"sans-serif"})
            mpl.rc('xtick', labelsize='small')
            mpl.rc('ytick', labelsize='small')
            mpl.rcParams['axes.spines.right'] = False
            mpl.rcParams['axes.spines.top'] = False
        if save:
            self.sv_path = save

        # set totc flag
        self.totc_flag = False
        # if we don't have proportions already, get proportions
        if (self.data.sum() == 1).all():
            self.props = self.data
        else:
            self.props = self.data/self.data.sum()


    # Counts plot
    def plot_totc(self, title=None, fig_kwargs={}):
        self.totc = self.data.sum()
        plt.figure(**fig_kwargs)
        plt.bar(self.totc.index, self.totc.values, color=self.colours.values)
        plt.ylabel('Total productive sequences')
        #plt.xlabel('Sample')
        plt.xticks(rotation=90)
        plt.margins(x=0)
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.show()

    # total counts histogram
    def totc_hist(self, bins, colour = 'k', by_class=None, title=None, fig_kwargs={}):
        self.totc = self.data.sum()
        if by_class is not None:
            selc = self.totc[self.labels == by_class]
        else:
            selc = self.totc
        plt.figure(**fig_kwargs)
        plt.hist(selc,color=colour,bins=bins)
        plt.xlabel('Total productive sequences')
        plt.ylabel('Number of samples')
        plt.xticks(rotation=90)
        if title:
            plt.title(title)
        plt.show()


    # Abundance plot
    def plot_abund(self, top_n, title=None, fig_kwargs={}):
        plt.figure(**fig_kwargs)
        for c in self.data.columns:
            plt.plot(np.arange(top_n)+1,sorted(self.data[c].values, reverse=True)[:top_n], color=self.colours[c], label=c)
        plt.ylabel('Clone frequency')
        plt.xlabel('Clone rank')
        plt.margins(x=0)
        if title:
            plt.title(title)
        plt.legend()
        plt.show()

    # Diversity plot
    def plot_div(self, divfunc, title=None, fig_kwargs={}):
        # calculate named diversity measures
        div = self.props.apply(divfunc)
        plt.figure(**fig_kwargs)
        plt.bar(div.index, div.values, color=self.colours.values)
        plt.ylabel('Diversity')
        #plt.xlabel('Sample')
        plt.xticks(rotation=90)
        plt.margins(x=0)
        if title:
            plt.title(title)
        plt.tight_layout()
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

    def plot_dprofile(self, q_vals, title=None, fig_kwargs={}):
        # calculate hill profile
        self.hill = self.props.apply(self.Hill_div, args=(q_vals,))
        plt.figure(**fig_kwargs)
        for c in self.hill.columns:
            plt.plot(q_vals, self.hill[c], color=self.colours[c], label=c)
        plt.xticks(q_vals)
        plt.margins(x=0)
        plt.xlabel("q")
        plt.legend()
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.show()


    # VJ heatmap
    def seg_heatmap(self,col_name,fig_kwargs={}):
        plt.figure(**fig_kwargs)
        seg_counts = self.props.groupby(by=col_name).sum().T
        hm = sns.heatmap(seg_counts, vmin=0, vmax=1, cmap='binary', linewidth=0.5, square=True, linecolor=(0,0,0), cbar=False, xticklabels=True)
        for ytl, c in zip(hm.axes.get_yticklabels(), self.colours):
            ytl.set_color(c)
        for _, spine in hm.spines.items():
            spine.set_visible(True)
        hm.tick_params(left=False, bottom=False)
        hm.set(xlabel=None)
        plt.show()

    # VJ boxplots
    def seg_boxplots(self,class_names, colours, seg_name=None,col_name=None):
        # one parameter must be passed
        # either plot for specific segment or all segments in v dj column
        seg_counts = self.props.groupby(by=col_name).sum()
        all_labs = np.unique(self.labels)
        if seg_name:
            segs = [seg_name]
        else:
            segs = seg_counts.index
        for s in segs:
            plt.figure()
            plt.title(s)
            fps = dict(marker='x', linestyle='none', markeredgecolor='k')
            mps = dict(color='k')
            bp = plt.boxplot([seg_counts.loc[s][self.labels == lab] for lab in all_labs],
                        vert=True, patch_artist=True, labels=class_names,flierprops=fps, medianprops=mps)
            for patch, color in zip(bp['boxes'], colours):
                patch.set_facecolor(color)
            plt.show()

    # top seqs
    def plot_top(self, n_top, spec_lab=None, title=None, fig_kwargs={}):
        # this works with proportions
        if spec_lab is not None:
            tp_data = self.props[self.props.columns[self.labels == spec_lab]]
        else:
            tp_data = self.props
        sum = tp_data.sum(axis=1)
        sum.index = range(len(tp_data))
        tp_sorted = tp_data.iloc[sum.sort_values(ascending=False).index].iloc[:n_top]
        b = pd.Series(index=tp_sorted.index,data =np.zeros(len(tp_sorted)))
        # need a more general version of this
        seq_ids = ["-".join([i[j] for j in self.seq_inds]) for i in tp_sorted.index]
        plt.figure(**fig_kwargs)
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
        plt.tight_layout()
        plt.show()