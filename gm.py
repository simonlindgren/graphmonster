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
import tweepy
import seaborn as sns
import infomap

sns.set_style('whitegrid')

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", default = "edgelist.txt")
parser.add_argument("-k", "--keep", default = 20)
parser.add_argument("-l", "--length", default = 16)
parser.add_argument("-n", "--num", default=10)
parser.add_argument("-w", "--win", default = 10)
parser.add_argument("-p", "--pparam", default=1)
parser.add_argument("-q", "--qparam", default=1)
parser.add_argument("-i", "--iters", default=10000)
parser.add_argument("-x", "--perp", default = 20)
parser.add_argument("--tw", default=False, action="store_true")

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
    t_sne(args.perp,args.iters)
    colourise(args.keep)
    if args.tw == True:
        twittergrab(G)
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
        infomapX.network().addLink(*e)

    print("----- Finding communities")
    infomapX.run();

    communities = {}
    for node in infomapX.iterLeafNodes():
        communities[node.physicalId] = node.moduleIndex()

    nx.set_node_attributes(G, values=communities, name='community')
    return G

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
    return G

def node2vec(walk,num,pparam,qparam,win):
    print("\n- node2vec function")
    print("----- Generating walks")
    node2vec = Node2Vec(G, dimensions=20, walk_length=int(walk), num_walks=int(num), workers=15, p=float(pparam), q=float(qparam), quiet=True)
    print("----- Learning embeddings")
    global model
    model = node2vec.fit(window=int(win), min_count=1)
    
def t_sne(perp,iters):
    print("\n- t_sne function")
    print("----- Reducing to 2-dimensional space (t-SNE) ...")
    nodes = [n for n in model.wv.vocab]
    embeddings = np.array([model.wv[x] for x in nodes])
    tsne = TSNE(n_components=2, n_iter=int(iters), perplexity=int(perp))
    global embeddings_2d
    embeddings_2d = tsne.fit_transform(embeddings)
    savetxt('embeddings_2d.csv', embeddings_2d, delimiter=',')
       
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
        
    df1 = pd.DataFrame(zip(nodz,names,clus), columns = ['node','name','community'])
    
    
    # A df of cluster colours
    df2 = pd.DataFrame(clu_colours, index = [0]).T.reset_index()
    df2.columns=["community", "colour"]
    
    
    # Join them
    global df3
    df3 = pd.merge(df1,df2, on="community")
    df3['degree'] = G.degree()
    
    colourdict = dict(zip(df3.node,df3.colour))

    #nodes = [n for n in G.nodes()] # get the order right here!
    nodes = [n for n in model.wv.vocab]
    global colours
    colours = []
    
    for n in nodes:
        colours.append(colourdict.get(int(n)))

def twittergrab(G):
    
    print("\n- twittergrab function")
   
    # a dict of comm labels and colour
    with open("commlabels.txt", "w") as labelfile:
        labelfile.write("community;community_label\n")
        for c,comm in enumerate(keepcomms):
            labelfile.write(str(comm) + ";label" + str(c) + "\n")
        
    from credentials import consumer_key, consumer_secret, access_token_secret, access_token

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    
     # get top 10 users by degree in each community
    print("----- Making api call ...")
    with open("community-identification.txt", "w") as outfile:
        for kc in keepcomms:
            comm_df = df3[df3['community'] == kc].sort_values(by="degree", ascending=False)
            topnames = list(comm_df['name'][:10])  
            users = api.lookup_users(topnames)

            degree_dict = dict(zip(df3.name,df3.degree))

            outfile.write("Community " + str(kc) + "\n" + "="*40)
            for c,u in enumerate(users):
                outfile.write("\n" + str(c+1) + " -- degree:" + str(degree_dict[topnames[c]])+"\n")
                outfile.write("user: " + u.name + "\n")
                outfile.write("screen_name: " + u.screen_name + "\n")
                outfile.write("description: " + u.description + "\n\n" + "--\n")
                
    x = input("\nMake manual edits, press enter to continue:")
    
                
def visualise():
    commlabels_df = pd.read_csv("commlabels.txt", sep=";")
    global full_df
    full_df = pd.merge(commlabels_df, df3, on="community")
    full_df = full_df.sort_values(by="degree", ascending=False)
    full_df.to_csv("gm.csv")

    
    print("\n- visualise function")
    print("----- Saving graph as pdf and svg")
    # Size by degree
    degree = [G.degree[n] for n in G.nodes()]

    # Plot figure
    figure = plt.figure(figsize=(16, 12))
    ax = figure.add_subplot(111)
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=degree*5, alpha=0.2, c=colours)

    
    if args.tw == True:
        # Legend
        legend_labels = [mpatches.Patch(color=colour, label=community_label) for community_label,colour in dict(zip(full_df.community_label,full_df.colour)).items()]
        ax.legend(handles=legend_labels)

    figure.savefig("gm.pdf", bbox_inches='tight')
    figure.savefig("gm.svg")
    
if __name__ == '__main__':
    main()
