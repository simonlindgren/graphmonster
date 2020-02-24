#!/usr/bin/env python3

import networkx as nx
from node2vec import Node2Vec
import numpy as np
from sklearn.manifold import TSNE
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import matplotlib.patches as mpatches
import community
import random
import tweepy

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", default = "edgelist.txt")
parser.add_argument("-k", "--keep", default = 8)
parser.add_argument("-l", "--length", default = 20)
parser.add_argument("-n", "--num", default=10)
parser.add_argument("-w", "--win", default = 10)
parser.add_argument("-p", "--pparam", default=1)
parser.add_argument("-q", "--qparam", default=1)
parser.add_argument("-x", "--perp", default = 40)
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
    louvain(args.keep)
    node2vec(args.length,args.num,args.pparam,args.qparam,args.win)
    t_sne(args.perp)
    colourise()
    label()
    twittergrab()
    manualbreak()
    visualise()
    print("")
    print("Done!")

def graphcrunch(file):
    print("- graphcrunch function")
    print("----- Creating weighted graph from edgelist")
    global G
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

    while len(G.edges()) > 1000000:
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

def louvain(keep):
    print("\n- louvain function")
    print("----- Detecting communites")
    louvain_dict = community.best_partition(G)
    global communities
    communities = [louvain_dict.get(n) for n in G.nodes()]
    commrank = list(pd.Series(communities).value_counts().index)
    global keepcomms
    try:
        keepcomms = commrank[:int(keep)]
        print("----- Removing small communities")
    except: 
        keepcomms = commrank
    removenodes=[]
    for node,comm in louvain_dict.items():
        if not comm in keepcomms:
            removenodes.append(node)
    G.remove_nodes_from(removenodes)   
    communities = [louvain_dict.get(n) for n in G.nodes()]

def node2vec(walk,num,pparam,qparam,win):
    print("\n- node2vec function")
    print("----- Generating walks ...")
    node2vec = Node2Vec(G, dimensions=20, walk_length=int(walk), num_walks=int(num), workers=15, p=float(pparam), q=float(qparam), quiet=True)
    print("----- Learning embeddings ...")
    global model
    model = node2vec.fit(window=int(win), min_count=1)
    
def t_sne(perp):
    print("\n- t_sne function")
    print("----- Reducing dimensional space (t-SNE) ...")
    nodes = [n for n in model.wv.vocab]
    embeddings = np.array([model.wv[x] for x in nodes])
    tsne = TSNE(n_components=2, perplexity=int(perp))
    global embeddings_2d
    embeddings_2d = tsne.fit_transform(embeddings)
       
    
def colourise():
    print("\n- colourise function")
    print("----- Translating community list to colour list")
    
    mplcolours = [i for i,v in mplcolors.cnames.items()]
    numcolours = len(set(communities))
    
    colrz = random.sample(mplcolours,numcolours)
    
    df_a = pd.DataFrame(zip(G.nodes(),communities), columns=['node','community'])
    df_b = pd.DataFrame(zip(set(communities),colrz), columns=['community','colour'])
    global df
    df = pd.merge(df_a,df_b, on="community")
    global colours
    colours = df.colour
    
def label():
    # a dict of comm labels and colour
    with open("commlabels.txt", "w") as labelfile:
        labelfile.write("community;community_label\n")
        for comm in keepcomms:
            labelfile.write(str(comm) + ";\n")
    
def twittergrab():
    # add a degree column to the df
    degrees = []
    for n in G.nodes():
        degrees.append(G.degree(n))
    df['degree'] = degrees
    
    # call up twitter api 
    from credentials import consumer_key, consumer_secret, access_token_secret, access_token
    
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    
    # get top 10 users by degree in each community
    print("api call...")
    with open("community-identification.txt", "w") as outfile:
        for kc in keepcomms:
            comm_df = df[df['community'] == kc].sort_values(by="degree", ascending=False)
            topnames = list(comm_df['node'][:10])  
            users = api.lookup_users(topnames)
            
            
            degree_dict = dict(zip(df.node,df.degree))
            
            outfile.write("Community " + str(kc) + "\n" + "="*40)
            for c,u in enumerate(users):
                outfile.write("\n" + str(c+1) + " -- degree:" + str(degree_dict[topnames[c]])+"\n")
                outfile.write("user: " + u.name + "\n")
                outfile.write("screen_name: " + u.screen_name + "\n")
                outfile.write("description: " + u.description + "\n\n" + "--\n")

def manualbreak():
    x = input("Make manual edits, press enter to continue:")
    commlabels = pd.read_csv("commlabels.txt", sep=";")
    global df
    df = pd.merge(df,commlabels,on="community")
                
             
def visualise():
    print("\n- visualise function")
    print("----- Saving graph as pdf and svg")
    # Size by degree
    degree = [G.degree[n] for n in G.nodes()]

    # Plot figure
    figure = plt.figure(figsize=(16, 12))
    ax = figure.add_subplot(111)
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=degree*5, alpha=0.6, c=colours)
    
    # Legend
    legend_labels = [mpatches.Patch(color=colour, label=community_label) for community_label,colour in dict(zip(df.community_label,df.colour)).items()]
    ax.legend(handles=legend_labels)
    
    
    

    figure.savefig("gm.pdf", bbox_inches='tight')
    figure.savefig("gm.svg")


if __name__ == '__main__':
    main()
