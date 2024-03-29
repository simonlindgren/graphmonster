#!/usr/bin/env python3

import networkx as nx
import infomap
import pandas as pd
from node2vec import Node2Vec

from sklearn.manifold import TSNE
import sys

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
                print("----- Skipped the row that said: " + str(e))
                pass
                #sys.exit()
            if G.has_edge(s,t):
                G[s][t]['weight'] += 1
            else:
                G.add_edge(s,t,weight = 1)
    G.remove_edges_from(nx.selfloop_edges(G))
   
    numnodes = len(G.nodes())
    numedges = len(G.edges())
    print("----- There are " + str(numnodes) + " nodes and " + str(numedges) + " edges in the graph.")
    
    print("----- Removing edges with a weight < 10")
    threshold = 10
    
    removeedges = []
    for s,t,data in G.edges(data=True):
      if data['weight'] < threshold:
         removeedges.append((s,t))
    G.remove_edges_from(removeedges)
    
    print("----- Removing nodes with a degree < average degree. ")
    
    N, K = G.order(), G.size()
    av_degree = float(K) / N
      
    print("----- Average degree is " + str(av_degree))
    remove = []
    for i in G.degree():
        if i[1] < av_degree:
            remove.append(i[0])
    G.remove_nodes_from(remove)
    
    leftnodes = len(G.nodes())
    leftedges = len(G.edges())
    if leftedges > 0:
        print("----- " + str(leftnodes) + " nodes and " + str(leftedges) + " edges still in the graph. Continuing ...")
    else:
        print("----- No edges left. Consider commenting away the node removal step in gm.py. Stopping.")
        sys.exit()
    
    
    print("----- Deleting unconnected components")
    giant_component_size = len(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    for component in list(nx.connected_components(G)):
        if len(component) < giant_component_size:
            for node in component:
                G.remove_node(node)
   
    print("----- Renaming nodes")
    # replace names with integer labels and set old label as 'name' attribute
    G = nx.convert_node_labels_to_integers(G,label_attribute="name")
 
def infomap_clu(G):
    print("\n- infomap_clu function")
    """
    Partition network with the Infomap algorithm.
    Annotates nodes with 'community' id and returns number of communities found.
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
    
    with open('keepcomms.pkl', 'wb') as f:
         pickle.dump(keepcomms, f)
          
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

    # pickle the degrees list
    with open('degrees.pkl', 'wb') as f:
        pickle.dump(degrees, f)

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
    nice_colours = ["#2f4f4f","#7f0000","#191970","#006400","#bdb76b","#ff0000","#ffa500","#ffff00","#0000cd","#00ff00","#00fa9a","#00ffff","#ff00ff","#1e90ff","#ff69b4","#e6e6fa"]
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

    data_df = pd.merge(data_df,colour_df, on="community").reset_index(drop=True)
    
    # Save dataframe
    data_df.to_csv("gm.csv")
 
def node2vec(walk,num,pparam,qparam,win):
    print("\n- node2vec function")
    print("----- Generating walks")
    print("----- This step takes significant time for large graphs ...")
    node2vec = Node2Vec(G, dimensions=20, walk_length=int(walk), num_walks=int(num), workers=1, p=float(pparam), q=float(qparam), quiet=True)
    print("----- Learning embeddings")
    global model
    model = node2vec.fit(window=int(win), min_count=1)    
    # save the model to disk using gensim
    model.save("gm-n2v.model")

    print("----- Saving embeddings")
    
if __name__ == '__main__':
    main()
