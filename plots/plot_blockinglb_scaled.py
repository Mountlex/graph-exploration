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


def plot(df, algs):
    max_param = df[PARAM].max()
    min_param = df[PARAM].min()

    ax = None

    ax = df[["hDFS", PARAM]].groupby([PARAM]).mean().plot(ax=ax, label="hDFS", legend=True, color='tab:blue')
    ax = df[["Blocking", PARAM]].groupby([PARAM]).mean().plot(ax=ax, label="Blocking", legend=True, color='tab:green')
    ax = df[["NN", PARAM]].groupby([PARAM]).mean().plot(ax=ax, label="NN", legend=True, color='tab:brown')
    #ax = df[["DFS", PARAM]].groupby([PARAM]).mean().plot(ax=ax, label="DFS", legend=True, color='tab:orange')

    for alg in algs:     
        if "Robust_Blocking (1.0)" in alg:
            ax = df[[alg, PARAM]].groupby([PARAM]).mean().plot(ax=ax, linestyle="--", marker="o", markersize="4", label=alg, legend=True, color='tab:red')
        elif "Robust_pTour (1.0)" in alg:
            ax = df[[alg, PARAM]].groupby([PARAM]).mean().plot(ax=ax, style="--s", markersize="4", label=alg, legend=True, color='tab:orange')
        elif "Robust_Blocking " in alg:
            ax = df[[alg, PARAM]].groupby([PARAM]).mean().plot(ax=ax, linestyle="None", marker="o", markersize="5", label=alg, legend=True, color='tab:pink')
        elif "Robust_pTour " in alg:
            ax = df[[alg, PARAM]].groupby([PARAM]).mean().plot(ax=ax, style="--s", markersize="4", label=alg, legend=True, color='tab:olive')
        elif "pTour" in alg:
            ax = df[[alg, PARAM]].groupby([PARAM]).mean().plot(ax=ax, style="-s", markersize="4", label=alg, legend=True, color='tab:cyan')
    
    plt.xlabel('Graph size parameter')
    plt.plot((min_param, max_param), (1, 1), 'black')
    plt.ylabel('Empirical competitive ratio')
    plt.legend()
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
        data, algs = get_data(parsed_args.file)
        cols = list(data)
        plot(data, algs)
        plt.show()