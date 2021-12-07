import argparse
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MST = "mst"
ETA = "eta"
PARAM = "param"
BIN = "bin"
TOUR = "tour"

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

def get_data(filename):
    df = pd.read_csv(filename, comment='#')
    df = df.round(4)
    cols = list(df)

    if not MST in cols:
        print("No MST values found!")
        exit(0)

    algs = [c for c in cols if not c == MST and not c == ETA and not c == PARAM and not c == TOUR]
    others = [c for c in cols if c == MST or c == ETA or c == PARAM]

    df = pd.concat([df[algs].div(df[MST], axis=0), df[others]], axis=1)

    return df, algs


def plot(df, algs, blocking=False):
    max_param = df[PARAM].max()
    min_param = df[PARAM].min()

    ax = None

    if not blocking:
        ax = df[["hDFS", PARAM]].groupby([PARAM]).mean().plot(ax=ax, label="hDFS", legend=True, color='tab:blue')
    ax = df[["NN", PARAM]].groupby([PARAM]).mean().plot(ax=ax, label="NN", legend=True, color='tab:brown')
    if not blocking:
        ax = df[["DFS", PARAM]].groupby([PARAM]).mean().plot(ax=ax, label="DFS", linestyle="None", marker="o", markersize="5", legend=True, color='tab:green')
    
    if blocking:
        ax = df[["Blocking", PARAM]].groupby([PARAM]).mean().plot(ax=ax, style="-o", label="Blocking", markersize="4", legend=True, color='tab:green')
        ax = df[["Robust_Blocking (1.0)", PARAM]].groupby([PARAM]).mean().plot(ax=ax, linestyle="-", marker="o", markersize="4", legend=True, color='tab:red')
        ax = df[["Robust_Blocking (20.0)", PARAM]].groupby([PARAM]).mean().plot(ax=ax, linestyle="None", marker="o", markersize="4", legend=True, color='tab:pink')
    else:
        ax = df[["pTour (eta/opt = 1)", PARAM]].groupby([PARAM]).mean().plot(ax=ax, style="-s", markersize="4", legend=True, color='tab:cyan')
        ax = df[["Robust_pTour (1.0)", PARAM]].groupby([PARAM]).mean().plot(ax=ax, style="--s", markersize="4", color='tab:orange')
        ax = df[["Robust_pTour (20.0)", PARAM]].groupby([PARAM]).mean().plot(ax=ax, style="--s", markersize="4", legend=True, color='tab:olive')  
        ax = df[["Robust_hDFS_rounded (1.0)", PARAM]].groupby([PARAM]).mean().plot(ax=ax, linestyle="None", marker="o", markersize="4", legend=True, color='tab:red')
        ax = df[["Robust_hDFS_rounded (20.0)", PARAM]].groupby([PARAM]).mean().plot(ax=ax, linestyle="-", marker="o", markersize="4", legend=True, color='tab:pink')

    

    plt.xlabel('Rosenkrantz graph size parameter i')
    #plt.plot((min_param, max_param), (1, 1), 'black')
    plt.ylabel('Empirical competitive ratio')
    ax.legend(["hDFS",  "NN", "DFS", "$\overline{R}$(hDFS, 1.0)", "$\overline{R}$(FP, 1.0)", "$\overline{R}$(hDFS, 20.0)", "$\overline{R}$(FP, 20.0)", "FP (rel. error = 5)"], ncol=3,loc='upper left')

    if blocking:
        ax.legend(["NN", "Blocking", "$\overline{R}$(Blocking, 1.0)",  "$\overline{R}$(Blocking, 20.0)"], ncol=2,loc='upper left')
    else:
        ax.legend(["hDFS", "NN", "DFS", "FP (rel. error = 5)", "$\overline{R}$(FP, 1.0)", "$\overline{R}$(FP, 20.0)", "$\overline{R}$(hDFS, 1.0)",  "$\overline{R}$(hDFS, 20.0)"], ncol=3,loc='upper left')

    plt.ylim(top=4.1)
    plt.tight_layout()

    fig = plt.gcf()
    fig.set_dpi(290)
    fig.set_size_inches(3,2)  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parsed_args = parser.parse_args(sys.argv[1:])
    if os.path.exists(parsed_args.file):
        data, algs = get_data(parsed_args.file)
        cols = list(data)
        plot(data, algs)
        plot(data, algs, blocking=True)
        plt.show()