#!/usr/bin/env python3

import networkx as nx
from node2vec import Node2Vec
import numpy as np
from numpy import savetxt
from sklearn.manifold import TSNE
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import infomap
import pickle
from numpy import loadtxt

sns.set_style('whitegrid')

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--keep", default = 20)

args = parser.parse_args()

logo='''

   ____ ____      _    ____  _   _ __  __  ___  _   _ ____ _____ _____ ____  
  / ___|  _ \    / \  |  _ \| | | |  \/  |/ _ \| \ | / ___|_   _| ____|  _ \ 
 | |  _| |_) |  / _ \ | |_) | |_| | |\/| | | | |  \| \___ \ | | |  _| | |_) |
 | |_| |  _ <  / ___ \|  __/|  _  | |  | | |_| | |\  |___) || | | |___|  _ < 
  \____|_| \_\/_/   \_\_|   |_| |_|_|  |_|\___/|_| \_|____/ |_| |_____|_| \_|
   
   Written by Simon Lindgren <simon.lindgren@umu.se>
                                                                                    
'''

print(logo)

def main():
    read_files()
    communityrip(G,args.keep)
    colourise(args.keep)
    visualise()
    print("")
    print("Done!")

def read_files():
    global G,model,embeddings_2d,data,keepcomms,degrees
    G = nx.read_gpickle("gm-graph.pkl")
    model = pickle.load(open("gm-n2v.pkl", 'rb'))
    embeddings_2d = loadtxt("gm-tsne.csv", delimiter=',')
    data = pd.read_csv("gm.csv")
    with open('keepcomms.pkl', 'rb') as f:
        keepcomms = pickle.load(f)
 
    print(data.head())
def communityrip(G,keep):
    print("\n- communityrip function")
    communities = []
    for n,d in G.nodes(data=True):
        communities.append(d['community'])
    sizeranked_comms = list(pd.Series(communities).value_counts().index)
    global keepcomms
    keepcomms = sizeranked_comms[:int(keep)]
    removenodes = []
    for n,d in G.nodes(data=True):
        if not d['community'] in keepcomms:
            removenodes.append(n)
    print("----- Keeping " + str(keep) + " communities")
    before = len(G.nodes)
    G.remove_nodes_from(removenodes)
    after = len(G.nodes)
    percentage = round(100*(before-after)/before)
    print("----- "+ str(after) + " of " + str(before) + " nodes kept (" + str(percentage) + "% removed)")
    
def colourise(keep):
    print("\n- colourise function")
    nice_colours = ['#100c08','#00ff00','#FF0000','#ff8c00','#ff69b4','#7fffd4','#9400d3','#9400d3','#ffb6c1','#ffd700']
    boring_colour = '#c0c0c0' # silver
    
    if len(keepcomms) < 11:
        clu_colours = dict(zip(keepcomms,nice_colours[:len(keepcomms)]))
    else:
        taillength = len(keepcomms) - 10
        tail = []
        for i in range(taillength):
            tail.append(boring_colour)
        colz = nice_colours + tail
        clu_colours = dict(zip(keepcomms,colz))
    
    # df of nodes, names and clusters
    nodz = []
    names = []
    clus = []
    for n,d in G.nodes(data=True):
        nodz.append(n)
        names.append(d['name'])
        clus.append(d['community'])
        
    dfA = pd.DataFrame(zip(nodz,names,clus), columns = ['node','name','community'])
    

    
    # A df of cluster colours
    dfB = pd.DataFrame(clu_colours, index = [0]).T.reset_index()
    dfB.columns=["community", "colour"]
    
    
    # Join them
    global dfC
    dfC = pd.merge(dfA,dfB, on="community")
    
    print(dfC.head())
    
    colourdict = dict(zip(dfC.node,dfC.colour))

    #nodes = [n for n in G.nodes()] # get the order right here!
    nodes = [n for n in model.wv.vocab]
    global colours
    colours = []
    
    for n in nodes:
        colours.append(colourdict.get(int(n)))

def visualise():
    commlabels_df = pd.read_csv("commlabels.txt", sep=";")
    global full_df
    full_df = pd.merge(commlabels_df, dfC, on="community")
    
    print("\n- visualise function")
    print("----- Saving graph as pdf and svg")
    # Size by degree
    degree = [G.degree[n] for n in G.nodes()]
    # Save the degree list to disk
    with open('degrees.pkl', 'wb') as f:
        pickle.dump(degree, f)

    # Plot figure
    figure = plt.figure(figsize=(16, 12))
    ax = figure.add_subplot(111)
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=degree*50, alpha=0.2, c=colours)
    
    # Legend
    legend_labels = [mpatches.Patch(color=colour, label=community_label) for community_label,colour in dict(zip(full_df.community_label,full_df.colour)).items()]
    ax.legend(handles=legend_labels)
  
    figure.savefig("gm.pdf", bbox_inches='tight')
    figure.savefig("gm.svg")
    
if __name__ == '__main__':
    main()
