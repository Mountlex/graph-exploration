import argparse
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

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


def plot_eta(df, algs, bin_size, axis = None, legend_pos = "upper left"):
    df[BIN] = np.ceil(
        df[ETA] / (df[MST] * float(bin_size))) * float(bin_size)
   
    df = df[df[BIN] <= 30]

    max_bin = df[BIN].max()
    plt.rc('axes', prop_cycle=(cycler('color', colors)))
    
    ax = None
    for alg in algs:
        if PARAM in df and "Robust" in alg:
            cr_by_param = df[[alg,BIN,PARAM]].groupby([BIN,PARAM]).mean().unstack(PARAM)
            for label, l in cr_by_param:
                if l in [0.5, 0.75]:
                    ax=cr_by_param[(label, l)].plot(ax=ax, style='--D', markersize=3, label=f"{alg} (λ = {l:1.2f})", legend=True)
        elif "pTour" in alg: 
            cr_by_param = df[[alg,BIN]].groupby([BIN]).mean()
            ax=cr_by_param.plot(ax=ax, style='--o', markersize=5, label=alg, legend=True)
        else:
            cr_by_param = df[[alg,BIN]].groupby([BIN]).mean()
            ax=cr_by_param.plot(ax=ax,label=alg, legend=True)

    plt.plot((0, max_bin), (1, 1), 'black')
    plt.xlabel('Relative prediction error')
    plt.ylabel('Empirical competitive ratio')
    plt.legend(["Blocking", "hDFS", "NN", "DFS", "FP", "$\overline{R}$(FP, 0.5)", "$\overline{R}$(FP, 0.75)", "$R$(FP, 0.5)", "$R$(FP, 0.75)"], ncol=2, loc=legend_pos)
    if axis:
        plt.axis(axis)
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_dpi(280)
    fig.set_size_inches(3,2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parsed_args = parser.parse_args(sys.argv[1:])
    if os.path.exists(parsed_args.file):
        data, algs = get_data(parsed_args.file)
        plot_eta(data, algs, 5)
        plot_eta(data, algs, 1, [-0.2, 7, 1.2, 3.2], "lower right")
        plt.show()

   
    
    