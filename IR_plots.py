import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def hide_dupe_name(names):
    dup_mask = names.duplicated()
    names[dup_mask] = "_" + names
    return names
    
class IRPlots:
    def __init__(self, json_fn, colour_func, name_func, seq_inds, glob_mpl_func=None, lgnd=True, save=None):
        # read in json file containing dataset info saved using IRDataset class
        with open(json_fn, "rb") as f:
            ppinfo = json.load(f)
        # load in labels, extract colours for samples, save sequence indices, load data matrix
        self.labels = pd.Series(ppinfo["labs"])
        self.all_labs = np.unique(self.labels)
        self.colours = colour_func(self.labels)
        rep_sams = list((self.colours[~self.colours.duplicated()]).index)
        self.all_colours = self.colours.loc[rep_sams]
        self.names = name_func(self.labels)
        self.all_names = self.names.loc[rep_sams]        
        #self.neat_names = hide_dupe_name(self.names)
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
            self.sv_flag = False
        # set legend flag
        self.lgnd_flag = lgnd
        self.plt_resln = 1200
        # if we don't have proportions already, get proportions
        if (self.data.sum() == 1).all():
            self.props = self.data
        else:
            self.props = self.data/self.data.sum()
        # make dict of diversity indices available to be calculated with names and functions
        # current class options are Richness, Shannon, Simpson
        self.div_dict = {'richness': self.richness, 'Shannon': self.shannon, 'Simpson': self.simpson}
        # define variables we calculate later as None, or empty dict in the case of seg_usage
        self.totc = None
        self.hill = None
        self.diversity = None
        self.seg_usage = dict()

    def reord_sams(self, new_ord=None, ord_func=None):
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
        # check totc, hill and diversity
        if self.totc is not None:
            self.totc = self.totc.loc[new_ord]
        if self.hill is not None:
            self.hill = self.hill[new_ord]
        if self.diversity is not None:
            self.diversity = self.diversity[new_ord]
        if len(self.seg_usage.keys()) > 0:
            # if we have saved usage, go through dict and update dataframes with new column order
            [self.seg_usage.update({key: val[new_ord]}) for key, val in self.seg_usage.items()]

    def totc_bar(self, title=None, fig_kwargs=None):
        # plot bar chart of total counts or depth for all samples in dataset
        self.totc = self.data.sum()
        if fig_kwargs is None:
            fig_kwargs = {}
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
        ax.bar(self.totc.index, self.totc.values, color=self.colours.values)
        ax.set_ylabel('Total productive sequences')
        plt.xticks(rotation=90)
        plt.margins(x=0)
        plt.tight_layout()
        if self.lgnd_flag:
            f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
            handles = [f("s", self.all_colours[i]) for i in range(len(self.all_colours))]
            plt.legend(handles, self.all_names, bbox_to_anchor=(1.02, 1), borderaxespad=0, loc=1, framealpha=1, frameon=False)
            #plt.legend(list(self.neat_names.values))
        if self.sv_flag:
            # convert title to filename
            if title:
                fn = title.replace(" ", "_") + ".png"
            else:
                fn = "totc_prod_bar.png"
            # save figure in directory specified
            fig.savefig(os.path.join(self.sv_path, fn), dpi=self.plt_resln)
        else:
            if title:
                plt.title(title)
            plt.show()
        plt.close()

    def totc_hist(self, bins, colour='k', by_class=None, title=None, fig_kwargs=None):
        # plot histogram of total counts or depth, can be stratified by label
        self.totc = self.data.sum()
        if by_class is not None:
            selc = self.totc[self.labels == by_class]
        else:
            selc = self.totc
        if fig_kwargs is None:
            fig_kwargs = {}
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
        ax.hist(selc, color=colour, bins=bins)
        ax.set_xlabel('Total productive sequences')
        ax.set_ylabel('Number of samples')
        plt.xticks(rotation=90)
        if self.sv_flag:
            # convert title to filename
            if title:
                fn = title.replace(" ", "_") + ".png"
            else:
                fn = "totc_prod_hist.png"
            # save figure in directory specified
            fig.savefig(os.path.join(self.sv_path, fn), dpi=self.plt_resln)
        else:
            if title:
                plt.title(title)
            plt.show()
        plt.close()

    def abund_lines(self, top_n, title=None, fig_kwargs=None):
        # plot lines showing top n clonal frequencies for all samples in dataset
        if fig_kwargs is None:
            fig_kwargs = {}
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
        for c in self.data.columns:
            ax.plot(np.arange(top_n)+1, sorted(self.data[c].values, reverse=True)[:top_n], color=self.colours[c],
                    label=c)
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
            fig.savefig(os.path.join(self.sv_path, fn), dpi=self.plt_resln)
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

    @staticmethod
    def hill_div(props, q_vals):
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

    def calc_div(self, indices):
        if self.diversity is None:
            diversity = dict()
        else:
            # convert diversity class variable to dict
            diversity = self.diversity.T.to_dict()
        for ind in indices:
            # proportions are used to calculate diversity
            diversity[ind] = self.props.apply(self.div_dict[ind])
        self.diversity = pd.DataFrame(diversity).T

    def calc_hill(self, q_vals):
        # calculate hill diversity with given q values
        self.hill = self.props.apply(self.hill_div, args=(q_vals,))

    def div_bar(self, div_name, title=None, fig_kwargs=None):
        # plot bar of diversity measures for all samples in dataset
        if self.diversity is not None:
            # calculate diversity measure and store it for later if it's not already calculated
            if div_name not in self.diversity.index:
                self.calc_div([div_name])
        else:
            self.calc_div([div_name])
        if fig_kwargs is None:
            fig_kwargs = {}
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
        ax.bar(self.diversity.columns, self.diversity.loc[div_name].values, color=self.colours.values)
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
            fig.savefig(os.path.join(self.sv_path, fn), dpi=self.plt_resln)
        else:
            if title:
                plt.title(title)
            plt.show()
        plt.close()

    def _boxplot_by_class(self, quantity, class_names, colours, annot, ax):
        # to plot boxplot separated by binary classes
        # set fliers to black crosses
        fps = dict(marker='x', linestyle='none', markeredgecolor='k')
        # make median line black
        mps = dict(color='k')
        # split quantity by label, smallest first
        quant_by_lab = [quantity[self.labels == lab] for lab in self.all_labs]
        # get list of class names in same order
        cn_ord = [class_names[al] for al in self.all_labs]
        # plot boxplot that spans vertically and enable patch artist
        bp = ax.boxplot(quant_by_lab, vert=True, patch_artist=True, labels=cn_ord, flierprops=fps, medianprops=mps)
        # fill each boxplot with label-specific colour
        for patch, color in zip(bp['boxes'], [colours[al] for al in self.all_labs]):
            patch.set_facecolor(color)
        if annot:
            # inc is the increment we use to set bracket spacing for annotation
            inc = quantity.max()/100
            # define height of bracket as above max value of data
            y = quantity.max() + 2*inc
            # set height of bracket as increment
            h = inc
            # set horizontal edges of bracket as x location of each class, trace path in bracket shape using height h
            ax.plot([1, 1, 2, 2], [y, y + h, y + h, y], lw=1.5, c='k')
            # annotate in centre of bracket
            ax.text(1.5, y + h, annot, ha='center', va='bottom', color='k')
        return ax

    def div_boxplot(self, div_name, class_names, colours, title, annot="", fig_kwargs=None):
        # produce boxplots separated by class for any named diversity index
        if self.diversity is not None:
            # calculate diversity measure and store it for later if it's not already calculated
            if div_name not in self.diversity.index:
                self.calc_div([div_name])
        else:
            self.calc_div([div_name])
        if fig_kwargs is None:
            fig_kwargs = {}
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
        ax = self._boxplot_by_class(self.diversity.loc[div_name], class_names, colours, annot, ax)
        plt.title(title)
        if self.sv_flag:
            # convert title to filename
            # segment names can have slashes, change to unicode
            fn = "Boxplot_" + title.replace(" ", "_") + ".png"
            # save figure in directory specified
            fig.savefig(os.path.join(self.sv_path, fn), dpi=self.plt_resln)
        else:
            plt.show()
        plt.close()

    def dprofile_lines(self, q_vals, title=None, fig_kwargs=None):
        # plot lines showing diversity profile for all samples
        # if we've already calculated what we need, reuse hill values
        if self.hill is not None:
            reuse = all([q in self.hill.index for q in q_vals])
            if not reuse:
                # otherwise calculate hill profile
                self.calc_hill(q_vals)
        else:
            # calculate hill profile if it hasn't been calculated yet for any q values
            self.calc_hill(q_vals)
        if fig_kwargs is None:
            fig_kwargs = {}
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
        for c in self.hill.columns:
            ax.plot(q_vals, self.hill[c].loc[q_vals], color=self.colours[c], label=c)
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
            fig.savefig(os.path.join(self.sv_path, fn), dpi=self.plt_resln)
        else:
            if title:
                plt.title(title)
            plt.show()
        plt.close()

    def calc_segs(self, col_name):
        # use one of the data index columns to calculate V, D or J segment usage
        # can only be used if segment information is in data
        # add together the number of entries with same segment for each sample
        seg_usage = self.props.groupby(by=col_name).sum()
        # save usage in dict, col_name should indicate V, D or J
        self.seg_usage[col_name] = seg_usage

    def seg_heatmap(self, col_name, cmap="binary", vmax=1, disp_cbar=True, annots=None, title=None, fig_kwargs=None):
        # for V, D, or J segments, calculate usage as proportion of repertoires and plot heatmap
        if fig_kwargs is None:
            fig_kwargs = {}
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
        # check if we already calculated segment usage
        if col_name not in self.seg_usage.keys():
            # calculate segment proportions
            self.calc_segs(col_name)
        seg_counts = self.seg_usage[col_name].T
        # add annotations to segment names if we want to indicate significance etc.
        if annots is not None:
            seg_counts.columns = [seg + a for a, seg in zip(annots, seg_counts.columns)]
        # colormap indicates proportional usage of each segment#
        # defaults produce black if all sequences in a repertoire use a single segment for V, D or J
        # and produce white for other segments that are completely unused
        ax = sns.heatmap(seg_counts, vmin=0, vmax=vmax, cmap=cmap, linewidth=0.5, square=True, linecolor=(0, 0, 0),
                         cbar=disp_cbar, cbar_kws={"shrink": 0.5}, xticklabels=True, yticklabels=True)
        # colour the sample names on y-axis
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
            fig.savefig(os.path.join(self.sv_path, fn), bbox_inches='tight', dpi=self.plt_resln)
        else:
            if title:
                plt.title(title)
            plt.show()
        plt.close()

    def seg_boxplots(self, class_names, colours, col_name, annots=None, seg_name=None, fig_kwargs=None):
        # colname, either of the V D or J segments, must be passed
        # segname optional, can plot single segment usage or all segment usage in V D or J
        if col_name not in self.seg_usage.keys():
            # calculate segment proportions
            self.calc_segs(col_name)
        seg_counts = self.seg_usage[col_name]
        if seg_name:
            segs = [seg_name]
        else:
            segs = seg_counts.index
        if fig_kwargs is None:
            fig_kwargs = {}
        for s in segs:
            fig, ax = plt.subplots(1, 1, **fig_kwargs)
            plt.title(s)
            if annots is not None:
                annot = annots[s]
            else:
                annot = ""
            ax = self._boxplot_by_class(seg_counts.loc[s], class_names, colours, annot, ax)
            if self.sv_flag:
                # convert title to filename
                # segment names can have slashes, change to unicode
                fn = s.replace("/", u'\u2215') + ".png"
                # save figure in directory specified
                seg_path = os.path.join(self.sv_path, col_name+"_boxplots")
                # check exists
                if not os.path.isdir(seg_path):
                    os.makedirs(seg_path)
                fig.savefig(os.path.join(seg_path, fn), dpi=self.plt_resln)
            else:
                plt.show()
            plt.close()

    def top_seq_bar(self, n_top, spec_lab=None, title=None, fig_kwargs=None):
        # plot a stacked bar chart displaying the top n sequences within the entire dataset
        # this uses proportions instead of raw counts
        if spec_lab is not None:
            tp_data = self.props[self.props.columns[self.labels == spec_lab]]
        else:
            tp_data = self.props
        # use sum to sort sequences because we might have multiple index columns
        seqsum = tp_data.sum(axis=1)
        seqsum.index = range(len(tp_data))
        tp_sorted = tp_data.iloc[seqsum.sort_values(ascending=False).index].iloc[:n_top]
        # initialise the bottom of the bars
        b = pd.Series(index=tp_sorted.index, data=np.zeros(len(tp_sorted)))
        # if we have multiple index columns, concatenate their strings to get sequence identifier labels
        if isinstance(self.seq_inds,int):
            seq_ids = tp_sorted.index
        else:
            seq_ids = ["-".join([i[j] for j in self.seq_inds]) for i in tp_sorted.index]
        if fig_kwargs is None:
            fig_kwargs = {}
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
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
            fig.savefig(os.path.join(self.sv_path, fn), dpi=self.plt_resln)
        else:
            if title:
                plt.title(title)
            plt.show()
        plt.close()
