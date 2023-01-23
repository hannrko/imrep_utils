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
        self.all_labs = np.unique(self.labels)
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
        # make dict of diversity indices available to be calculated with names and functions
        # current class options are Richness, Shannon, Simpson
        self.div_dict = {'richness':self.richness,'Shannon':self.shannon,'Simpson':self.simpson}
        # define variables we calculate later as None
        self.totc = None
        self.hill = None
        self.diversity = None
        self.seg_props = dict()

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
        # check totc, hill and diversity
        if self.totc is not None:
            self.totc = self.totc.loc[new_ord]
        if self.hill is not None:
            self.hill = self.hill[new_ord]
        if self.diversity is not None:
            self.diversity = self.diversity[new_ord]
        if len(self.seg_props.keys()) > 0:
            # if we have saved usage
            [self.seg_props.update({key: val[new_ord]}) for key, val in self.seg_props.items()]

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

    def calc_hill(self,indices):
        self.hill = self.props.apply(self.Hill_div, args=(indices,))

    def plot_div(self, div_name, title=None, fig_kwargs={}):
        # plot bar of diversity measures for all samples in dataset
        if self.diversity is not None:
            # calculate diversity measure and store it for later if it's not already calculated
            if div_name not in self.diversity.index:
                self.calc_div([div_name])
        else:
            self.calc_div([div_name])
        fig, ax = plt.subplots(1,1,**fig_kwargs)
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
            fig.savefig(os.path.join(self.sv_path, fn))
        else:
            if title:
                plt.title(title)
            plt.show()
        plt.close()

    def _boxplot_by_class(self,quantity,class_names,colours,annot,ax):
        # to plot boxplot separated by binary classes
        # set fliers to black crosses
        fps = dict(marker='x', linestyle='none', markeredgecolor='k')
        # make median line black
        mps = dict(color='k')
        # split quantity by label, smallest first
        quant_by_lab = [quantity[self.labels == lab] for lab in self.all_labs]
        # get list of class names in same order
        cn_ord = [class_names[l] for l in self.all_labs]
        # plot boxpot that spans vertically and enable patch artist
        bp = ax.boxplot(quant_by_lab, vert=True, patch_artist=True, labels=cn_ord, flierprops=fps, medianprops=mps)
        # fill boxplots with label-specific colour
        for patch, color in zip(bp['boxes'], [colours[l] for l in self.all_labs]):
            patch.set_facecolor(color)
        if annot:
            # inc is the increment we use to set bracket spacing for annotation
            inc = quantity.max()/100
            # define height of bracket as above max value of data
            y = quantity.max() + 2*inc
            # set height of bracket as increment
            h = inc
            # set horizontal edges of bracket as x location of each class, and trace path in bracket shape using height h
            ax.plot([1, 1, 2, 2], [y, y + h, y + h, y], lw=1.5, c='k')
            # anotate in centre of bracket
            ax.text(1.5, y + h, annot, ha='center', va='bottom', color='k')
        return ax

    def div_boxplot(self, div_name, class_names, colours, title, star="", fig_kwargs={}):
        if self.diversity is not None:
            # calculate diversity measure and store it for later if it's not already calculated
            if div_name not in self.diversity.index:
                self.calc_div([div_name])
        else:
            self.calc_div([div_name])
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
        ax = self._boxplot_by_class(self.diversity.loc[div_name],class_names,colours,star,ax)
        plt.title(title)
        if self.sv_flag:
            # convert title to filename
            # segment names can have slashes, change to unicode
            fn = "Boxplot_" + title.replace(" ", "_") + ".png"
            # save figure in directory specified
            fig.savefig(os.path.join(self.sv_path, fn))
        else:
            plt.show()
        plt.close()

    def plot_dprofile(self, q_vals, title=None, fig_kwargs={}):
        # plot lines showing diversity profile for all samples
        # calculate hill profile
        # if we've already calculated what we need, reuse hill values
        if self.hill is not None:
            reuse = all([q in self.hill.index for q in q_vals])
            if not reuse:
                self.calc_hill(q_vals)
        else:
            self.calc_hill(q_vals)
        fig, ax = plt.subplots(1,1,**fig_kwargs)
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
            fig.savefig(os.path.join(self.sv_path, fn))
        else:
            if title:
                plt.title(title)
            plt.show()
        plt.close()

    def calc_segs(self,col_name):
        seg_props = self.props.groupby(by=col_name).sum()
        self.seg_props[col_name] = seg_props

    def seg_heatmap(self,col_name,cmap="binary",vmax=1,disp_cbar=True,stars=None,title=None,fig_kwargs={}):
        # for V, D, or J segments, calculate usage as proportion of repertoires and plot heatmap
        fig, ax = plt.subplots(1,1,**fig_kwargs)
        # check if we already calculated segment usage
        if col_name not in self.seg_props.keys():
            # calculate segment proportions
            self.calc_segs(col_name)
        seg_counts = self.seg_props[col_name].T
        # add stars to indicate significant difference only if true passed for segment
        if stars is not None:
            seg_counts.columns = [seg + s for s,seg in zip(stars,seg_counts.columns)]
        # white is zero share of repertoire, black is full share of repertoire
        ax = sns.heatmap(seg_counts, vmin=0, vmax=vmax, cmap=cmap, linewidth=0.5, square=True, linecolor=(0,0,0),
                         cbar=disp_cbar,cbar_kws={"shrink": 0.5}, xticklabels=True, yticklabels=True)
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

    def seg_boxplots(self,class_names, colours, col_name, annots=None, seg_name=None):
        # colname, either of the V D or J segments, must be passed
        # segname optional, can plot single segment usage or all segment usage in V D or J
        if col_name not in self.seg_props.keys():
            # calculate segment proportions
            self.calc_segs(col_name)
        seg_counts = self.seg_props[col_name]
        if seg_name:
            segs = [seg_name]
        else:
            segs = seg_counts.index
        for s in segs:
            fig, ax = plt.subplots(1,1)
            plt.title(s)
            if annots is not None:
                annot = annots[s]
            else:
                annot=""
            ax = self._boxplot_by_class(seg_counts.loc[s], class_names, colours, annot, ax)
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
