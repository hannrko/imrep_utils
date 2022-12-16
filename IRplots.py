import json
import pandas as pd
import matplotlib.pyplot as plt

class IRPlots:
    def __init__(self,json_fn,colour_func,seq_inds):
        # read in json file containing info
        with open(json_fn, "rb") as f:
            ppinfo = json.load(f)
        self.labels = ppinfo["labs"]
        self.colours = colour_func(self.labels)
        self.data = pd.read_csv(ppinfo["prepro_path"], index_col=seq_inds)

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
    def plot_div(self):
        # calculate named diversity measures

    def plot_dprofile(self):
        # calculate hill profile


    # VJ heatmap
    def seg_heatmap(self,col_name):


    # VJ boxplots
    def seg_boxplots(self,seg_name=None,col_name=None):
        # one parameter must be passed
        # either plot for specific segment or all segments in v dj column

    # top seqs
    def plot_top(self):

