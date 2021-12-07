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

    return df


def plot_param(df):
    max_param = df[PARAM].max()
    min_param = df[PARAM].min()

    ax = None
    ax = df[["hDFS", PARAM]].groupby([PARAM]).mean().plot(ax=ax, label="hDFS", legend=True, color='tab:blue')
    ax = df[["Blocking", PARAM]].groupby([PARAM]).mean().plot(ax=ax, label="Blocking", legend=True, color='tab:green')
    ax = df[["NN", PARAM]].groupby([PARAM]).mean().plot(ax=ax, label="NN", legend=True, color='tab:brown')
    ax = df[["DFS", PARAM]].groupby([PARAM]).mean().plot(ax=ax, style=":", label="DFS", legend=True, color='tab:orange')

    ax = df[["Robust_hDFS", PARAM]].groupby([PARAM]).mean().plot(ax=ax, style="--o", markersize="4", label="Robust_hDFS", legend=True, color='tab:red')
    ax = df[["Robust_hDFS_rounded", PARAM]].groupby([PARAM]).mean().plot(ax=ax, style="--o", markersize="4", label="Robust_hDFS_rounded", legend=True, color='tab:purple')
    ax = df[["Robust_Blocking", PARAM]].groupby([PARAM]).mean().plot(ax=ax, style="--s", markersize="4", label="Robust_Blocking", legend=True, color='tab:cyan')
  
    
  
    
    plt.xlabel('Lambda')
    plt.plot((min_param, max_param), (1, 1), 'black')
    plt.ylabel('Empirical competitive ratio')
    plt.legend(ncol=2, loc="upper left")
    plt.ylim(top=4)
    plt.tight_layout()

    fig = plt.gcf()
    fig.set_dpi(400)
    fig.set_size_inches(3,2)  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parsed_args = parser.parse_args(sys.argv[1:])
    if os.path.exists(parsed_args.file):
        data = get_data(parsed_args.file)
        cols = list(data)
        plot_param(data)
        plt.show()