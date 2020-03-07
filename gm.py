#!/usr/bin/env python3

import networkx as nx
import infomap
import pandas as pd
from node2vec import Node2Vec
import umap
import numpy as np
from numpy import savetxt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# silence NumbaPerformanceWarning
import warnings
warnings.filterwarnings('ignore')


import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", default = "edgelist.txt")
parser.add_argument("-k", "--keep", default = 16)
parser.add_argument("-l", "--length", default = 16)
parser.add_argument("-n", "--num", default=10)
parser.add_argument("-w", "--win", default = 10)
parser.add_argument("-p", "--pparam", default=1)
parser.add_argument("-q", "--qparam", default=1)
parser.add_argument("-i", "--iters", default=600)
parser.add_argument("-x", "--perp", default = 20)
parser.add_argument("--nneigh", default = 20)
parser.add_argument("-m", "--mind", default = 1)
parser.add_argument("--mtrc", default = "euclidean")
parser.add_argument("--tsne", default=False, action="store_true")


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
    graphcrunch(args.file)
    infomap_clu(G)
    communityrip(G,args.keep)
    node2vec(args.length,args.num,args.pparam,args.qparam,args.win)
    if args.tsne is True:
        t_sne(args.perp,args.iters)
    else:
        umap_reduction(int(args.nneigh),float(args.mind),args.mtrc)
    visualise()
    print("")
    print("Done!")

def graphcrunch(file):
    global G
    print("- graphcrunch function")
    print("----- Creating weighted graph from edgelist")
    G = nx.Graph()
    with open(file,"r") as edgelist:
        for e in edgelist.readlines():
            try:
                s = e.split()[0]
                t = e.split()[1]
            except:
                print("----- Your edgelist has the wrong format!")
                sys.exit()
            if G.has_edge(s,t):
                G[s][t]['weight'] += 1
            else:
                G.add_edge(s,t,weight = 1)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    print("----- Removing edges by threshold")
    threshold = 2
    while len(G.edges()) > 2000000:
        removeedges = []
        for s,t,data in G.edges(data=True):
            if data['weight'] < threshold:
                removeedges.append((s,t))
        G.remove_edges_from(removeedges)
        threshold += 1
    
    print("----- Deleting unconnected components")
    giant_component_size = len(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    for component in list(nx.connected_components(G)):
        if len(component)<giant_component_size:
            for node in component:
                G.remove_node(node)
   
    print("----- Renaming nodes")
    # replace names with integer labels and set old label as 'name' attribute
    G = nx.convert_node_labels_to_integers(G,label_attribute="name")
    
def infomap_clu(G):
    print("\n- infomap_clu function")
    """
    Partition network with the Infomap algorithm.
    Annotates nodes with 'community' id and return number of communities found.
    """
    infomapX = infomap.Infomap("--two-level --silent")

    print("----- Building Infomap network")
    for e in G.edges():
        infomapX.network.addLink(*e)

    print("----- Finding communities")
    infomapX.run();

    communities = {}
    for node in infomapX.iterLeafNodes():
        communities[node.physicalId] = node.moduleIndex()

    nx.set_node_attributes(G, values=communities, name='community')

    # Save graph to disk
    nx.write_gpickle(G, "gm-graph.pkl")
    
    
def communityrip(G,keep):
    # Which are the biggest communities?
    communitylist = []
    for n,d in G.nodes(data = True):
        communitylist.append(d['community'])
    sizeranked_comms = list(pd.Series(communitylist).value_counts().index)

    # Keep a number of communities
    global keepcomms
    keepcomms = sizeranked_comms[:int(keep)]
    
    # Reduce the graph
    removenodes = []
    for n,d in G.nodes(data=True):
        if d['community'] not in keepcomms:
            removenodes.append(n)
    G.remove_nodes_from(removenodes)
    
    # Graph is all set, so we make a dataframe
    nodes = []
    names = []
    comms = []
    degrees = []

    for n,d in G.nodes(data=True):
        nodes.append(n)
        names.append(d['name'])
        comms.append(d['community'])
        degrees.append(G.degree[n])

    global data_df
    data_df = pd.DataFrame()

    data_df['node'] = nodes
    data_df['name'] = names
    data_df['community'] = comms
    data_df['degree'] = degrees
    
    # Add a colours column to the dataframe
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

    colour_df = pd.DataFrame(clu_colours, index = [0]).T.reset_index()
    colour_df.columns=["community", "colour"]

    data_df = pd.merge(data_df,colour_df, on="community")  
    
    # Save dataframe
    data_df.to_csv("gm.csv")
    
       
def node2vec(walk,num,pparam,qparam,win):
    print("\n- node2vec function")
    print("----- Generating walks")
    node2vec = Node2Vec(G, dimensions=20, walk_length=int(walk), num_walks=int(num), workers=1, p=float(pparam), q=float(qparam), quiet=True)
    print("----- Learning embeddings")
    global model
    model = node2vec.fit(window=int(win), min_count=1)    
    # save the model to disk       
    pickle.dump(model, open("gm-n2v.pkl", 'wb'))     
    
def t_sne(perp,iters):
    print("\n- t-sne function")
    print("----- Reducing to 2-dimensional space (t-SNE) ...")
    global nodes
    nodes = [n for n in model.wv.vocab]
    embeddings = np.array([model.wv[x] for x in nodes])
    tsne = TSNE(n_components=2, n_iter=int(iters), perplexity=int(perp))
    global embeddings_2d
    embeddings_2d = tsne.fit_transform(embeddings)
    savetxt('gm-2d.csv', embeddings_2d, delimiter=',')
       
def umap_reduction(nneigh,mind,mtrc):
    print("\n- umap function")
    print("----- Reducing to 2-dimensional space (umap) ...")
    global nodes
    nodes = [n for n in model.wv.vocab]
    embeddings = np.array([model.wv[x] for x in nodes])
    global embeddings_2d
    umap_r = umap.UMAP(n_neighbors=nneigh,min_dist=mind,metric=mtrc)
    embeddings_2d = umap_r.fit_transform(embeddings)
    savetxt('gm-2d.csv', embeddings_2d, delimiter=',')
        
def visualise():
    # Prepare for labelling
    with open("commlabels.txt", "w") as labelfile:
        labelfile.write("community;community_label\n")
        for c,comm in enumerate(keepcomms):
            labelfile.write(str(comm) + ";label" + str(c) + "\n")
    
    # Visualise
    print("\n- visualise function")
    print("----- Saving graph image")

    # colours must be in the same order as in the model
    # this was set as 'nodes' above
    colourdict = dict(zip(data_df.node,data_df.colour))
    colours = [colourdict.get(int(n)) for n in nodes]
    
    # log degree to avoid extreme node sizes      
    log_degree = np.log(data_df.degree)
    
    # Plot figure
    figure = plt.figure(figsize=(16, 12))
    ax = figure.add_subplot(111)
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
               s=[i*60 for i in log_degree], 
                  alpha=0.6, 
                  c=colours)

    figure.savefig("gm.png")
    
if __name__ == '__main__':
    main()