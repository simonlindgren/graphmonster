#!/usr/bin/env python3

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import numpy as np
from numpy import loadtxt
import seaborn as sns
import argparse
import gensim

sns.set_style('white')

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", default = 60)
parser.add_argument("-a", "--aph", default = 0.6) # alpha
parser.add_argument("--svg", default=False, action="store_true")
parser.add_argument("-d", "--dark", default=False, action="store_true")
parser.add_argument("-l", "--labels", nargs='+', help="Node names to label (space-separated)")
parser.add_argument("--all-labels", default=False, action="store_true", help="Label all nodes")
parser.add_argument("--fontsize", default=10, type=int, help="Font size for labels")

args = parser.parse_args()

#read the node2vec model
global model
model = gensim.models.Word2Vec.load("gm-n2v.model")

# read the data_df csv
global data_df
data_df = pd.read_csv("gm.csv")

# read embeddings
global embeddings_2d
embeddings_2d = loadtxt('gm-2d.csv', delimiter=',')
    

def main():
    print("\n[-=GRAPHMONSTER=-]")
    visualise(float(args.size),float(args.aph),args.labels,args.all_labels,args.fontsize)
    print("")
    print("Done!")
   
def visualise(size,aph,labels,all_labels,fontsize):
    # set up labels
    commlabels_df = pd.read_csv("commlabels.txt", sep=";")
    global full_df
    full_df = pd.merge(commlabels_df, data_df, on="community")
    
    # set up colours
    nodes = [n for n in list(model.wv.key_to_index.keys())]

    colourdict = dict(zip(data_df.node,data_df.colour))
    colours = [colourdict.get(int(n)) for n in nodes]
    
    # log degree to avoid extreme node sizes      
    log_degree = np.log(data_df.degree)
                    
    print("\n- visualise function")
    print("----- Saving graph image")

    # Plot figure
    figure = plt.figure(figsize=(16, 12))
    if args.dark is True:
        plt.style.use('dark_background')
    ax = figure.add_subplot(111)
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
               s=[i*(float(size)) for i in log_degree],
               alpha=aph,
               c=colours)
    
    # Add labels to nodes
    nodes = [n for n in list(model.wv.key_to_index.keys())]
    
    if all_labels:
        print(f"----- Adding labels to all {len(full_df)} nodes")
        for idx, row in full_df.iterrows():
            node_id = row['node']
            label_name = row['name']
            if str(node_id) in nodes:
                node_idx = nodes.index(str(node_id))
                x, y = embeddings_2d[node_idx, 0], embeddings_2d[node_idx, 1]
                ax.annotate(label_name, 
                           xy=(x, y), 
                           xytext=(5, 5), 
                           textcoords='offset points',
                           fontsize=fontsize,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'),
                           ha='left')
    elif labels:
        print(f"----- Adding labels to {len(labels)} nodes")
        for label_name in labels:
            # Find the node with this name in the dataframe
            matching_rows = full_df[full_df['name'] == label_name]
            if not matching_rows.empty:
                node_id = matching_rows.iloc[0]['node']
                # Find the index of this node in the nodes list
                if str(node_id) in nodes:
                    node_idx = nodes.index(str(node_id))
                    x, y = embeddings_2d[node_idx, 0], embeddings_2d[node_idx, 1]
                    ax.annotate(label_name, 
                               xy=(x, y), 
                               xytext=(5, 5), 
                               textcoords='offset points',
                               fontsize=fontsize,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'),
                               ha='left')
                else:
                    print(f"----- Warning: Node '{label_name}' (ID: {node_id}) not found in embeddings")
            else:
                print(f"----- Warning: Node name '{label_name}' not found in data")
    
    
    # Legend
    legend_labels = [mpatches.Patch(color=colour, label=community_label) for community_label,colour in dict(zip(full_df.community_label,full_df.colour)).items()]
    ax.legend(handles=legend_labels)
    
    # Export
    figure.savefig("gm.png")
    if args.svg is True:
        print("----- Creating svg file")
        figure.savefig("gm.svg")
    
if __name__ == '__main__':
    main()
