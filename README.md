# GRAPHMONSTER

Graphmonster will take an edgelist, create a graph, and remove noise from it. It then uses Node2Vec to find node embeddings, and t-sne for reducing dimensionality. Nodes are coloured by community (Infomap algorithm) and sized by degree.

### Usage

```
python gm.py <parameters>
```
Files for the visualisation(`gm.svg` and `gm.pdf`) and data on the most prominent nodes (`gm.csv`) will be created.

### Parameters

`-f`, `--file`, name of edgelist file with one space-separated edge per line (e.g. `0 2` or `pig owl`), default = edgelist.txt

`--tw`, set this flag to activate the twittergrab function.

#### Twittergrab
Activate this function only if your edgelist consists of Twitter user id numbers. Before the graph is visualised as a 2D image, the user manually enters cluster labels in `commlabels.txt` by inspecting `community-identification.txt`, which consists of user info that is retrieved from the Twitter API.


_Clustering_

`-k`, `--keep`, max number of communities to keep, default = 20.

_Node2Vec_

`-l`, `--length`, walk length, default = 16. 

`-n`, `--num`, number of walks per node, default = 10.

`-w`, `--win`, window for model, default = 10.

`-p`, `--pparam`, p-parameter (return), default = 1.

`-q`, `--qparam`, q-parameter (inout), default = 1.

_t-sne_

`-x`, `--perp`, perplexity for t-sne, recommended 5-50, default = 20.
`-i`, `--iters`, number of iterations, default 10000 (min 250).

### Prerequisites

Run the following command to install package dependencies:

```
pip install -r requirements.txt
```

A valid set of Twitter api credentials must be provided in `credentials.py` if using the twittergrab function.
