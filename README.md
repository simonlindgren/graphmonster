# GRAPHMONSTER

Graphmonster will take an edgelist of twitter-ids, create a graph, and remove noise from it. It then uses Node2Vec to find node embeddings, and T-SNE for reducing dimensionality. Nodes are coloured by community (Louvain algorithm) and sized by degree. Before the graph is visualised as a 2D image, the user manually enters cluster labels in `commlabels.txt` by inspecting `community-identification.txt`.

### Requirements

A valid set of Twitter api credentials must be inserted into `credentials.py`.

### Usage

```
python gm.py -f <parameters>
```

### Parameters

`-f`, `--file`, name of edgelist file with one space-separated edge per line (e.g. `0 2` or `pig owl`), default = edgelist.txt

_Louvain_

`-k`, `--keep`, max number of communities to keep, default = 8.

_Node2Vec_

`-l`, `--length`, walk length, default = 5. 

`-n`, `--num`, number of walks per node, default = 10.

`-w`, `--win`, window for model, default = 10.

`-p`, `--pparam`, p-parameter (return), default = 1.

`-q`, `--qparam`, q-parameter (inout), default = 1.

_t-sne_

`-x`, `--perp`, perplexity for t-sne, recommended 1-50, default = 40.
