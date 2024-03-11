import umap
import numpy as np
from numpy import savetxt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import gensim
import pickle
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tsne", default=False, action="store_true")
parser.add_argument("-i", "--iters", default=300)
parser.add_argument("-x", "--perp", default = 5)

parser.add_argument("-n", "--nneigh", default = 4)
parser.add_argument("-m", "--mind", default = 0.5)
parser.add_argument("--mtrc", default = "euclidean")

args = parser.parse_args()


#read the node2vec model
global model
model = gensim.models.Word2Vec.load("gm-n2v.model")

# read the keepcooms pkl
global keepcomms
with open('keepcomms.pkl', 'rb') as f:
    keepcomms = pickle.load(f)

# read the data_df csv
global data_df
data_df = pd.read_csv("gm.csv")

def main():
    print("\n[-=GRAPHMONSTER=-]")
    if args.tsne is True:
        t_sne(args.perp,args.iters)
    else:
        umap_reduction(int(args.nneigh),float(args.mind),args.mtrc)
    visualise()
    print("")
    print("Done!")


def t_sne(perp,iters):
    print("\n- t-sne function")
    print("----- Reducing to 2-dimensional space (t-SNE) ...")
    global nodes
    nodes = [n for n in model.wv.key_to_index]
    embeddings = np.array([model.wv[x] for x in nodes])
    
    tsne = TSNE(n_components=2, n_iter=int(iters), perplexity=int(perp), init = 'pca')
    global embeddings_2d
    embeddings_2d = tsne.fit_transform(embeddings)
    savetxt('gm-2d.csv', embeddings_2d, delimiter=',')

def umap_reduction(nneigh,mind,mtrc):
    print("\n- umap function")
    print("----- Reducing to 2-dimensional space (umap) ...")
    global nodes
    nodes = [n for n in model.wv.key_to_index]
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
    
    # save embeddings_2d to disk
    savetxt('gm-2d.csv', embeddings_2d, delimiter=',')

    figure.savefig("gm.png")

if __name__ == '__main__':
    main()