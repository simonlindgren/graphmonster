#!/usr/bin/env python3

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
from numpy import loadtxt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", default = 1)
parser.add_argument("-a", "--aph", default = 0.6) # alpha
args = parser.parse_args()

def main():
    print("\n[-=GRAPHMONSTER=-]")
    read_files()
    colourise()
    visualise(int(args.size),float(args.aph))
    print("")
    print("Done!")

def read_files():
    print("- reading files")
    global G,model,embeddings_2d,data,keepcomms,degree
    G = nx.read_gpickle("gm-graph.pkl")
    model = pickle.load(open("gm-n2v.pkl", 'rb'))
    embeddings_2d = loadtxt("gm-2d.csv", delimiter=',')
    data = pd.read_csv("gm.csv")
    with open('keepcomms.pkl', 'rb') as f:
        keepcomms = pickle.load(f)
    with open('degrees.pkl', 'rb') as f:
        degree = pickle.load(f)
   
def colourise():
    print("\n- colourise function")
    nice_colours = ['#100c08','#00ff00','#FF0000','#ff8c00','#ff69b4','#7fffd4','#9400d3','#9400d3','#ffb6c1','#ffd700','#000000','#aaffc3','#800000','#bcf60c','#808080','#ffe119']
    boring_colour = '#c0c0c0' # silver
    
    if len(keepcomms) < 17:
        clu_colours = dict(zip(keepcomms,nice_colours[:len(keepcomms)]))
    else:
        taillength = len(keepcomms) - 16
        tail = []
        for i in range(taillength):
            tail.append(boring_colour)
        colz = nice_colours + tail
        clu_colours = dict(zip(keepcomms,colz))
    
    colourdict = dict(zip(data.node,data.colour))
    nodes = [n for n in model.wv.vocab]
    global colours
    colours = []
    for n in nodes:
        colours.append(colourdict.get(int(n)))

def visualise(size,aph):
    commlabels_df = pd.read_csv("commlabels.txt", sep=";")
    global full_df
    full_df = pd.merge(commlabels_df, data, on="community")
    
    print("\n- visualise function")
    print("----- Saving graph as pdf and svg")

    # Plot figure
    figure = plt.figure(figsize=(16, 12))
    ax = figure.add_subplot(111)
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
               s=[i*size for i in degree],
               alpha=aph,
               c=colours)
    
    # Legend
    legend_labels = [mpatches.Patch(color=colour, label=community_label) for community_label,colour in dict(zip(full_df.community_label,full_df.colour)).items()]
    ax.legend(handles=legend_labels)
    
    # Export
    figure.savefig("gm.pdf", bbox_inches='tight')
    figure.savefig("gm.svg")
    
if __name__ == '__main__':
    main()
