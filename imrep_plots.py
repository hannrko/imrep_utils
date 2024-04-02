import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl

class DatasetPlotter:
    def __init__(self, data, sam_info, resolution=1200, plt_format="png", save=None, glob_mpl_func=None):
        if glob_mpl_func:
            glob_mpl_func()
        # if we want to save figures save should be path to folder
        if save:
            self.sv_flag = True
            self.sv_path = save
            if not os.path.exists(save):
                os.mkdir(save)
        # if no path specified set save flag to false
        else:
            self.sv_flag = False
            self.sv_path = None
        # sam_info is a metadata table, choose column to use for colour
        self.sam_info = sam_info
        # initialise colour df
        self.sam_colour = pd.DataFrame(index=self.sam_info.index, columns=[])
        self.sam_colour_dicts = {}
        self.data = data
        # choose resolution
        self.resolution = resolution
        # choose output format
        self.plt_format = plt_format

    def _empty_kwargs(self, kwargs):
        if kwargs is None:
            return {}
        else:
            return kwargs

    def _title_to_fn(self, title):
        return title.replace(" ", "_") + "." + self.plt_format

    def _handle_output(self, fig, default_title, title=None):
        # decide on title
        if title is None:
            save_title = default_title
        else:
            save_title = title
        if self.sv_flag:
            fn = self._title_to_fn(save_title)
            fig.savefig(os.path.join(self.sv_path, fn), dpi=self.resolution)
        else:
            if title:
                fig.title(title)
            plt.show()
        plt.close()

    def _annot_box(self, annot, x, y, ax):
        u_x = np.unique(x)
        h = y.max()/100
        yb = y.max() + 2*h
        ax.plot([u_x[0], u_x[0], u_x[1], u_x[1]], [yb, yb + h, yb + h, yb], lw=1, c='k')
        # annotate in centre of bracket
        ax.text((u_x[0] + u_x[1])/2, yb + h, annot, ha='center', va='bottom', color='k')
        return ax

    def set_colour(self, sam_info_col, colour_dict=None):
        si = self.sam_info[sam_info_col]
        # if no colour specified
        if colour_dict is None:
            si_u = np.unique(si.values)
            # use default colourmap
            c_u = plt.cm.tab10(range(len(si_u)))
            colour_dict = dict(zip(si_u, c_u))
        self.sam_colour_dicts[sam_info_col] = colour_dict
        colours = [colour_dict[si_i] for si_i in si]
        self.sam_colour[sam_info_col] = colours

    def reorder_samples(self, new_ord=None, ord_func=None):
        if new_ord is None and ord_func is None:
            print("either new_order or ord_func must be defined")
        elif ord_func:
            new_ord = ord_func(self.sam_info)
        self.sam_info = self.sam_info.loc[new_ord]
        self.data = self.data[new_ord]
        self.sam_colour = self.sam_colour.loc[new_ord]

class DiversityDatasetPlotter(DatasetPlotter):
    def bar(self, div_name, colour_name, title=None, fig_kwargs=None):
        fig_kwargs = self._empty_kwargs(fig_kwargs)
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
        div_data = self.data.loc[div_name]
        ax.bar(div_data.index, div_data.values, color=self.sam_colour[colour_name])
        ax.set_ylabel('Diversity')
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=90)
        ax.margins(x=0)
        fig.set_tight_layout(True)
        df_title = div_name + " bar"
        self._handle_output(fig, df_title, title)
        # could split this up into single div plot, then multiple

    def box(self, div_name, colour_name, title=None, annot=None, fig_kwargs=None):
        # use seaborn, tidy data
        # need to combine sam info and data
        info_data = pd.concat([self.data.T, self.sam_info], axis=1, join="inner")
        fig_kwargs = self._empty_kwargs(fig_kwargs)
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
        ax = sns.boxplot(data=info_data, x=colour_name, y=div_name, hue=colour_name,
                         palette=self.sam_colour_dicts[colour_name], ax=ax)
        if annot is not None:
            ax = self._annot_box(annot, info_data[colour_name], info_data[div_name], ax)
        fig.set_tight_layout(True)
        df_title = div_name + " box"
        self._handle_output(fig, df_title, title)

    def profile_lines(self, div_names, colour_name, legend_flag=False, title=None, fig_kwargs=None):
        # add something about whether this is hill divrsity or not- onyl recommended for hill
        # with hill we can also plot q values
        fig_kwargs = self._empty_kwargs(fig_kwargs)
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
        for sam in self.data.columns:
            line = self.data[sam].loc[div_names]
            ax.plot(line.values, color=self.sam_colour[colour_name].loc[sam], label=sam)
        fig.set_tight_layout(True)
        df_title = "diversity lines"
        if legend_flag:
            ax.legend()
        self._handle_output(fig, df_title, title)

class VDJDatasetPlotter(DatasetPlotter):
    def __init__(self, data, sam_info, norm=True, resolution=1200, plt_format="png", save=None, glob_mpl_func=None):
        super().__init__(data, sam_info, resolution, plt_format, save, glob_mpl_func)
        if norm:
            self.data = self.data/self.data.sum()#, axis="index")#/np.array(list(vdj_data.sum(axis=1).values))
    def heatmap(self, colour_name, annots=None, cmap="binary", vmax=None, disp_cbar=True, title=None, fig_kwargs=None):
        fig_kwargs = self._empty_kwargs(fig_kwargs)
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
        vdj = self.data
        if annots is not None:
            vdj.index = [seg + a for a, seg in zip(annots, vdj.index)]
        ax = sns.heatmap(vdj, vmin=0, vmax=vmax, cmap=cmap, linewidth=0.5, square=True, linecolor=(0, 0, 0),
                         cbar=disp_cbar, cbar_kws={"shrink": 0.5}, xticklabels=True, yticklabels=True)
        for xtl, c in zip(ax.axes.get_xticklabels(), self.sam_colour[colour_name]):
            xtl.set_color(c)
        for _, spine in ax.spines.items():
            spine.set_visible(True)
        ax.tick_params(left=False, bottom=False)
        ax.set(xlabel=None)
        df_title = "vdj heatmap"
        self._handle_output(fig, df_title, title)

    def box(self, colour_name, annots=None, title=None, fig_kwargs=None):
        vdj = self.data.T.melt(ignore_index=False, var_name="seg", value_name="count")
        vdj = vdj.reset_index(names="sample")
        ri_sam_info = self.sam_info.reset_index(names="sample")
        info_data = vdj.merge(ri_sam_info, left_on="sample", right_on="sample")
        fig_kwargs = self._empty_kwargs(fig_kwargs)
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
        ax = sns.boxplot(data=info_data, x="seg", y="count", hue=colour_name,
                         palette=self.sam_colour_dicts[colour_name], ax=ax)
        if annots is not None:
            for i, annot in enumerate(annots):
                seg_data = info_data[info_data["seg"] == info_data["seg"].iloc[i]]
                ax = self._annot_box(annot, i+seg_data[colour_name]-1/2, seg_data["count"], ax)
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=90)
        fig.set_tight_layout(True)
        df_title = "vdj box"
        self._handle_output(fig, df_title, title)

class CloneDatasetPlotter(DatasetPlotter):
    def hist(self, sam, colour_name, bins=10, title=None, fig_kwargs=None):
        fig_kwargs = self._empty_kwargs(fig_kwargs)
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
        ax.hist(self.data[sam], bins, color=self.sam_colour[colour_name].loc[sam], label=sam)
        ax.set_xlabel("clone count")
        ax.set_yscale("log")
        ax.set_ylabel("frequency")
        fig.set_tight_layout(True)
        df_title = sam + " count hist"
        self._handle_output(fig, df_title, title)

    def heatmap(self, n_top_clones, colour_name, cmap="binary", disp_cbar=True, title=None, fig_kwargs=None):
        fig_kwargs = self._empty_kwargs(fig_kwargs)
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
        seq_tot = self.data.sum(axis=1)
        top_clones = seq_tot.sort_values(ascending=False).index[:n_top_clones]
        ax = sns.heatmap(self.data.loc[top_clones], vmin=0, cmap=cmap, linewidth=0.5, square=True, linecolor=(0, 0, 0),
                    cbar=disp_cbar, cbar_kws={"shrink": 0.5}, xticklabels=True, yticklabels=True, norm=mpl.colors.LogNorm())
        for xtl, c in zip(ax.axes.get_xticklabels(), self.sam_colour[colour_name]):
            xtl.set_color(c)
        for _, spine in ax.spines.items():
            spine.set_visible(True)
        ax.tick_params(left=False, bottom=False)
        ax.set(xlabel=None)
        fig.set_tight_layout(True)
        df_title = "top count heatmap"
        self._handle_output(fig, df_title, title)

    def lines(self, n_top_clones, colour_name, title=None, lgnd_flag=False, fig_kwargs=None):
        fig_kwargs = self._empty_kwargs(fig_kwargs)
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
        for sam in self.data.columns:
            x = np.arange(n_top_clones) + 1
            y = sorted(self.data[sam].values, reverse=True)[:n_top_clones]
            ax.plot(x, y, color=self.sam_colour[colour_name].loc[sam], label=sam)
        ax.set_ylabel("Clone frequency")
        ax.set_yscale("log")
        ax.set_xlabel("Clone rank within sample")
        ax.margins(x=0)
        if lgnd_flag:
            plt.legend()
        fig.set_tight_layout(True)
        df_title = "abundance lines"
        self._handle_output(fig, df_title, title)
