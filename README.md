# GRAPHMONSTER

Graphmonster will take an edgelist, create a graph, and remove noise from it. It then uses Node2Vec to find node embeddings, and t-sne for reducing dimensionality. In the resulting visualised graph, nodes are coloured by community (Infomap algorithm) and sized by degree.

### Usage

```
python gm.py <parameters>
```
Files for the visualisation(`gm.svg` and `gm.pdf`) and data on the most prominent nodes (`gm.csv`) will be created.

### Parameters

`-f`, `--file`, name of edgelist file with one space-separated edge per line (e.g. `0 2` or `pig owl`), default = edgelist.txt

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


#### Twittergrab function
If your edgelist consists of Twitter user id numbers, you can use the twittergrab function. After running graphmonster (which will output a file named `gm.csv`), run:

```
python gm_tw.py
```

Twittergrabber will read the csv, call up the Twitter API, and get profile data on the top ten users (by degree) in each cluster. This information will be saved in `community-identification.txt`. Inspect that file and enter community labels manually by editing `commlabels.txt`. Set the number of communities to look up with the `-n` flag (default = 10), and use `-f` to use any other filename than the default (`gm.csv`).

A valid set of Twitter api credentials must be provided in `credentials.py`.


### Prerequisites

Run the following command to install package dependencies:

```
pip install -r requirements.txt
```
