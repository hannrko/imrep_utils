import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class IRPlots:
    def __init__(self,json_fn,colour_func,seq_inds,glob_mpl_func=None,lgnd=True,save=None):
        # read in json file containing dataset info saved using IRDataset class
        with open(json_fn, "rb") as f:
            ppinfo = json.load(f)
        # load in labels, extract colours for samples, save sequence indices, load data matrix
        self.labels = pd.Series(ppinfo["labs"])
        self.colours = colour_func(self.labels)
        self.seq_inds = seq_inds
        # data is a pandas dataframe
        # depending on what operations were applied in IRDataset, index may consist of
        # multiple columns, e.g. CDR3 sequence, and optionally V, D, and J segments
        # some plots will not work without gene segment columns in the index
        self.data = pd.read_csv(ppinfo["prepro_path"], index_col=self.seq_inds)
        # change plotting style with function if one is specified
        if glob_mpl_func:
            glob_mpl_func()
        # if we want to save figures save should be path to folder
        if save:
            self.sv_flag = True
            self.sv_path = save
        # if no path specified set save flag to false
        else:
            self.sv_flag=False
        # set legend flag
        self.lgnd_flag = lgnd
        # if we don't have proportions already, get proportions
        if (self.data.sum() == 1).all():
            self.props = self.data
        else:
            self.props = self.data/self.data.sum()
        # define variables we calculate later as None
        self.totc = None
        self.hill = None

    def reord_sams(self,new_ord=None,ord_func=None):
        # either new order for samples must be defined, or function applied to labels to get new order
        if not new_ord and not ord_func:
            print("either new_order or ord_func must be defined")
        elif ord_func:
            new_ord = ord_func(self.labels)
        # if neither conditions above satisfied then new_ord is defined at input
        # re-index everything defined in init
        self.labels = self.labels.loc[new_ord]
        self.colours = self.colours.loc[new_ord]
        self.data = self.data[new_ord]
        self.props = self.props[new_ord]
        # check totc and hill
        if self.totc:
            self.totc = self.totc.loc[new_ord]
        if self.hill:
            self.hill = self.hill[new_ord]

    def plot_totc(self, title=None, fig_kwargs={}):
        # plot bar chart of total counts or depth for all samples in dataset
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
        plt.close()

    def totc_hist(self, bins, colour = 'k', by_class=None, title=None, fig_kwargs={}):
        # plot histogram of total counts or depth, can be stratified by label
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
        plt.close()

    def plot_abund(self, top_n, title=None, fig_kwargs={}):
        # plot lines showing top n clonal frequencies for all samples in dataset
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
        plt.close()

    @staticmethod
    def richness(props):
        # count species that exist in sample
        return sum(props > 0)

    @staticmethod
    def shannon(props):
        # first remove any sequences with zero share of repertoire
        props = props[props > 0]
        # calculate shannon diversity using formula
        return -sum(props*np.log(props))

    @staticmethod
    def simpson(props):
        # first remove any sequences with zero share of repertoire
        props = props[props > 0]
        # calculate Simpson diversity using formula
        return sum(props**2)

    def plot_div(self, divfunc, title=None, fig_kwargs={}):
        # plot bar of diversity measures for all samples in dataset
        # calculate diversity measure using static class method or an external function specified by user
        # current class options are Richness, Shannon, Simpson
        # proportions are used to calculate diversity
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
        plt.close()

    @staticmethod
    def Hill_div(props, q_vals):
        # calculate Hill diversity profiles using list of q values
        # make sure q_vals is 1D array
        q_vals = np.array(q_vals).reshape(-1)
        # initialise empty list
        div = []
        # remove sequences with zero share of repertoire
        props = props[props > 0]
        for q in q_vals:
            if q == 1:
                # calculate perplexity separately due to log
                ind = np.exp(-np.sum(props * np.log(props)))
            else:
                # general formula works for remaining q values
                ind = np.sum(props ** q) ** (1 / (1 - q))
            div = np.append(div, ind)
        return np.squeeze(div)

    def plot_dprofile(self, q_vals, title=None, fig_kwargs={}):
        # plot lines showing diversity profile for all samples
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
        plt.close()

    def seg_heatmap(self,col_name,cmap="binary",vmax=1,title=None,fig_kwargs={}):
        # for V, D, or J segments, calculate usage as proportion of repertoires and plot heatmap
        fig, ax = plt.subplots(1,1,**fig_kwargs)
        seg_counts = self.props.groupby(by=col_name).sum().T
        # white is zero share of repertoire, black is full share of repertoire
        ax = sns.heatmap(seg_counts, vmin=0, vmax=vmax, cmap=cmap, linewidth=0.5, square=True, linecolor=(0,0,0),
                         cbar=False, xticklabels=True, yticklabels=True)
        # colour the sample names on y axis
        for ytl, c in zip(ax.axes.get_yticklabels(), self.colours):
            ytl.set_color(c)
        # ensure the heatmap outline shows
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
        plt.close()

    def seg_boxplots(self,class_names, colours, col_name, seg_name=None):
        # colname, either of the V D or J segments, must be passed
        # segname optional, can plot single segment usage or all segment usage in V D or J
        seg_counts = self.props.groupby(by=col_name).sum()
        all_labs = np.unique(self.labels)
        if seg_name:
            segs = [seg_name]
        else:
            segs = seg_counts.index
        for s in segs:
            fig, ax = plt.subplots(1,1)
            plt.title(s)
            # set filers to black crosses
            fps = dict(marker='x', linestyle='none', markeredgecolor='k')
            # make the median line black
            mps = dict(color='k')
            bp = ax.boxplot([seg_counts.loc[s][self.labels == lab] for lab in all_labs], vert=True, patch_artist=True,
                            labels=[class_names[l] for l in all_labs],flierprops=fps, medianprops=mps)
            # fill boxplots with colour specific to label
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
            plt.close()

    def plot_top(self, n_top, spec_lab=None, title=None, fig_kwargs={}):
        # plot a stacked bar chart displaying the top n sequences within the entire dataset
        # this uses proportions instead of raw counts
        if spec_lab is not None:
            tp_data = self.props[self.props.columns[self.labels == spec_lab]]
        else:
            tp_data = self.props
        # use sum to sort sequences because we might have multiple index columns
        sum = tp_data.sum(axis=1)
        sum.index = range(len(tp_data))
        tp_sorted = tp_data.iloc[sum.sort_values(ascending=False).index].iloc[:n_top]
        # initialise the bottom of the bars
        b = pd.Series(index=tp_sorted.index,data =np.zeros(len(tp_sorted)))
        # if we have multiple index columns, concatenate their strings to get sequence identifier labels
        seq_ids = ["-".join([i[j] for j in self.seq_inds]) for i in tp_sorted.index]
        fig, ax = plt.subplots(1,1,**fig_kwargs)
        for i, c in enumerate(tp_sorted.columns):
            t = tp_sorted[c].values
            plt.bar(seq_ids, t, bottom=b, label=c, color=self.colours[c])
            # update bottom of bars
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
        plt.close()
