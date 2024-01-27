import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns

class DatasetPlotter:
    def __init__(self, data, sam_info, resolution=1200, plt_format="png", save=None, glob_mpl_func=None):
        if glob_mpl_func:
            glob_mpl_func()
        # if we want to save figures save should be path to folder
        if save:
            self.sv_flag = True
            self.sv_path = save
        # if no path specified set save flag to false
        else:
            self.sv_flag = False
            self.sv_path = None
        self.data = data
        # sam_info is a metadata table, choose column to use for colour
        self.sam_info = sam_info
        # initialise colour df
        self.sam_colour = pd.DataFrame(index=self.sam_info.index, columns=[])
        self.sam_colour_dicts = {}
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

    # reorder samples

class DatasetDiversityPlotter(DatasetPlotter):
    def bar(self, div_name, colour_name, title=None, fig_kwargs=None):
        fig_kwargs = self._empty_kwargs(fig_kwargs)
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
        div_data = self.data[div_name]
        ax.bar(div_data.index, div_data.values, color=self.sam_colour[colour_name])
        ax.set_ylabel('Diversity')
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=90)
        ax.margins(x=0)
        fig.set_tight_layout(True)
        df_title = div_name + " bar"
        self._handle_output(fig, df_title, title)
        # could split this up into single div plot, then multiple

    def box(self, div_name, colour_name, title=None, fig_kwargs=None):
        # use seaborn, tidy data
        # need to combine sam info and data
        info_data = pd.concat([self.data, self.sam_info], axis=1, join="inner")
        fig_kwargs = self._empty_kwargs(fig_kwargs)
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
        ax = sns.boxplot(data=info_data, x=colour_name, y=div_name, hue=colour_name, palette=self.sam_colour_dicts[colour_name], ax=ax)
        fig.set_tight_layout(True)
        df_title = div_name + " box"
        self._handle_output(fig, df_title, title)

    def profile_lines(self, div_names, colour_name, legend_flag=False, title=None, fig_kwargs=None):
        # add something about whether this is hill divrsity or not- onyl recommended for hill
        # with hill we can also plot q values
        fig_kwargs = self._empty_kwargs(fig_kwargs)
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
        for sam in self.data.index:
            line = self.data[div_names].loc[sam]
            ax.plot(line.values, color=self.sam_colour[colour_name].loc[sam], label=sam)
        fig.set_tight_layout(True)
        df_title = "diversity lines"
        if legend_flag:
            ax.legend()
        self._handle_output(fig, df_title, title)
