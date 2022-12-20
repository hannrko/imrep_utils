import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import os

class IRPlots:
    def __init__(self,json_fn,colour_func,seq_inds,glob_mpl_func=None,lgnd_flag=True,save=None):
        # read in json file containing info
        with open(json_fn, "rb") as f:
            ppinfo = json.load(f)
        # load in labels, extract colours for samples, save sequence indices, load data matrix
        self.labels = pd.Series(ppinfo["labs"])
        self.colours = colour_func(self.labels)
        self.seq_inds = seq_inds
        self.data = pd.read_csv(ppinfo["prepro_path"], index_col=self.seq_inds)

        # change plotting style with function
        if glob_mpl_func:
            glob_mpl_func()

        # if we want to save figures in a folder set save as path to that folder
        if save:
            self.sv_flag = True
            self.sv_path = save
        else:
            self.sv_flag=False
        # set legend flag
        self.lgnd_flag = lgnd_flag
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
        fig, ax = plt.subplots(1,1,**fig_kwargs)
        ax.bar(self.totc.index, self.totc.values, color=self.colours.values)
        ax.set_ylabel('Total productive sequences')
        plt.xticks(rotation=90)
        plt.margins(x=0)
        plt.tight_layout()
        if self.sv_flag:
            # convert title to filename
            if title:
                fn = title.replace(" ","_") + ".png"
            else:
                fn = "totc_prod_bar.png"
            # save figure in directory specified
            fig.savefig(os.path.join(self.sv_path,fn))
        else:
            if title:
                plt.title(title)
            plt.show()

    # total counts histogram
    def totc_hist(self, bins, colour = 'k', by_class=None, title=None, fig_kwargs={}):
        self.totc = self.data.sum()
        if by_class is not None:
            selc = self.totc[self.labels == by_class]
        else:
            selc = self.totc
        fig, ax = plt.subplots(1,1,**fig_kwargs)
        ax.hist(selc,color=colour,bins=bins)
        ax.set_xlabel('Total productive sequences')
        ax.set_ylabel('Number of samples')
        plt.xticks(rotation=90)
        if self.sv_flag:
            # convert title to filename
            if title:
                fn = title.replace(" ","_") + ".png"
            else:
                fn = "totc_prod_hist.png"
            # save figure in directory specified
            fig.savefig(os.path.join(self.sv_path,fn))
        else:
            if title:
                plt.title(title)
            plt.show()


    # Abundance plot
    def plot_abund(self, top_n, title=None, fig_kwargs={}):
        fig, ax = plt.subplots(1,1,**fig_kwargs)
        for c in self.data.columns:
            ax.plot(np.arange(top_n)+1,sorted(self.data[c].values, reverse=True)[:top_n], color=self.colours[c], label=c)
        ax.set_ylabel('Clone frequency')
        ax.set_xlabel('Clone rank')
        plt.margins(x=0)
        if self.lgnd_flag:
            plt.legend()
        if self.sv_flag:
            # convert title to filename
            if title:
                fn = title.replace(" ", "_") + ".png"
            else:
                fn = "top_clone_abund.png"
            # save figure in directory specified
            fig.savefig(os.path.join(self.sv_path, fn))
        else:
            if title:
                plt.title(title)
            plt.show()

    # Diversity plot
    def plot_div(self, divfunc, title=None, fig_kwargs={}):
        # calculate named diversity measures
        div = self.props.apply(divfunc)
        fig, ax = plt.subplots(1,1,**fig_kwargs)
        ax.bar(div.index, div.values, color=self.colours.values)
        ax.set_ylabel('Diversity')
        plt.xticks(rotation=90)
        plt.margins(x=0)
        plt.tight_layout()
        if self.sv_flag:
            # convert title to filename
            if title:
                fn = title.replace(" ", "_") + ".png"
            else:
                fn = "div_bar.png"
            # save figure in directory specified
            fig.savefig(os.path.join(self.sv_path, fn))
        else:
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

    def plot_dprofile(self, q_vals, title=None, fig_kwargs={}):
        # calculate hill profile
        self.hill = self.props.apply(self.Hill_div, args=(q_vals,))
        fig, ax = plt.subplots(1,1,**fig_kwargs)
        for c in self.hill.columns:
            ax.plot(q_vals, self.hill[c], color=self.colours[c], label=c)
        ax.set_xticks(q_vals)
        plt.margins(x=0)
        ax.set_xlabel("q")
        if self.lgnd_flag:
            plt.legend()
        plt.tight_layout()
        if self.sv_flag:
            # convert title to filename
            if title:
                fn = title.replace(" ", "_") + ".png"
            else:
                fn = "div_profile.png"
            # save figure in directory specified
            fig.savefig(os.path.join(self.sv_path, fn))
        else:
            if title:
                plt.title(title)
            plt.show()


    # VJ heatmap
    def seg_heatmap(self,col_name,title=None,fig_kwargs={}):
        fig, ax = plt.subplots(1,1,**fig_kwargs)
        seg_counts = self.props.groupby(by=col_name).sum().T
        ax = sns.heatmap(seg_counts, vmin=0, vmax=1, cmap='binary', linewidth=0.5, square=True, linecolor=(0,0,0),
                         cbar=False, xticklabels=True, yticklabels=True)
        for ytl, c in zip(ax.axes.get_yticklabels(), self.colours):
            ytl.set_color(c)
        for _, spine in ax.spines.items():
            spine.set_visible(True)
        ax.tick_params(left=False, bottom=False)
        ax.set(xlabel=None)
        if self.sv_flag:
            # convert title to filename
            if title:
                fn = title.replace(" ", "_") + ".png"
            else:
                fn = col_name + "_heatmap.png"
            # save figure in directory specified
            fig.savefig(os.path.join(self.sv_path, fn))
        else:
            if title:
                plt.title(title)
            plt.show()

    # VJ boxplots
    def seg_boxplots(self,class_names, colours, col_name, seg_name=None):
        # colname must be passed, segname optional
        seg_counts = self.props.groupby(by=col_name).sum()
        all_labs = np.unique(self.labels)
        if seg_name:
            segs = [seg_name]
        else:
            segs = seg_counts.index
        for s in segs:
            fig, ax = plt.subplots(1,1)
            plt.title(s)
            fps = dict(marker='x', linestyle='none', markeredgecolor='k')
            mps = dict(color='k')
            bp = ax.boxplot([seg_counts.loc[s][self.labels == lab] for lab in all_labs],
                        vert=True, patch_artist=True, labels=[class_names[l] for l in all_labs],flierprops=fps, medianprops=mps)
            for patch, color in zip(bp['boxes'], [colours[l] for l in all_labs]):
                patch.set_facecolor(color)
            if self.sv_flag:
                # convert title to filename
                # segment names can have slashes, change to unicode
                fn = s.replace("/",u'\u2215') + ".png"
                # save figure in directory specified
                fig.savefig(os.path.join(self.sv_path, col_name+"_boxplots", fn))
            else:
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
        fig, ax = plt.subplots(1,1,**fig_kwargs)
        for i, c in enumerate(tp_sorted.columns):
            t = tp_sorted[c].values
            plt.bar(seq_ids, t, bottom=b, label=c, color=self.colours[c])
            b = b + t
        plt.xticks(rotation=90)
        ax.set_xlabel("Top " + str(n_top) + " sequences")
        ax.set_ylabel("Proportion of repertoire")
        plt.margins(x=0)
        if self.lgnd_flag:
            plt.legend()
        plt.tight_layout()
        if self.sv_flag:
            # convert title to filename
            if title:
                fn = title.replace(" ", "_") + ".png"
            else:
                fn = "top_seqs.png"
            # save figure in directory specified
            fig.savefig(os.path.join(self.sv_path, fn))
        else:
            if title:
                plt.title(title)
            plt.show()
